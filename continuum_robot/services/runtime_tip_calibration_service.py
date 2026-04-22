"""Dedicated 0A runtime tip calibration workflow service."""

from __future__ import annotations

from dataclasses import asdict
import copy
import json
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from continuum_robot.config.schemas import RegistrationWorkflowConfig
from continuum_robot.registration.legacy_compat import (
    AuroraPoseSample,
    average_tool_samples_as_transform,
)
from continuum_robot.registration.rigid_solver import RigidRegistrationSolver
from continuum_robot.registration.runtime_tip_calibration import (
    RuntimeTipTruthGeometry,
    load_runtime_tip_truth_geometry,
    solve_runtime_tip_calibration,
)
from continuum_robot.registration.runtime_tip_repository import (
    RuntimeTipCalibrationRecord,
    RuntimeTipCalibrationRepository,
)
from continuum_robot.services.models import (
    HEALTH_DEGRADED,
    HEALTH_FAILED,
    HEALTH_HEALTHY,
    RuntimeTipCalibrationSnapshot,
    ServiceHealthSnapshot,
)
from continuum_robot.services.registration_service import RegistrationService
from continuum_robot.services.tracking_service import TrackingService
from continuum_robot.tracking.transforms import make_transform_A_B
from continuum_robot.utils.time_utils import utc_now_iso


class RuntimeTipCalibrationService:
    """Own the advanced hat-based 0A runtime tip calibration workflow."""

    SESSION_MODE_FULL_HAT = "full_hat"
    SESSION_MODE_QUICK_4_POINT = "quick_4_point"

    def __init__(
        self,
        *,
        tracking_service: TrackingService,
        registration_service: RegistrationService,
        repository: RuntimeTipCalibrationRepository,
        solver: RigidRegistrationSolver,
        registration_config: RegistrationWorkflowConfig,
        config_path: Path,
        config_source: str,
        sleep_fn=time.sleep,
    ) -> None:
        self.tracking_service = tracking_service
        self.registration_service = registration_service
        self.repository = repository
        self.solver = solver
        self.config = registration_config
        self.config_path = config_path
        self.config_source = config_source
        self.sleep_fn = sleep_fn
        self.project_root = Path(__file__).resolve().parents[2]

        self._lock = threading.Lock()
        self._truth_geometry = self._load_truth_geometry()
        self._quick_labels = self._choose_quick_labels(self._truth_geometry.truth_points_in_tip_by_label)
        self._pending_record: RuntimeTipCalibrationRecord | None = None
        self._raw_measurement_tool_samples_by_label: dict[str, list[dict[str, Any]]] = {}
        self._raw_coil_samples: list[dict[str, Any]] = []

        self._state = RuntimeTipCalibrationSnapshot(
            health=ServiceHealthSnapshot(
                name="runtime_tip_calibration_service",
                health=HEALTH_DEGRADED,
                state="idle",
                status="Runtime tip calibration idle",
                current_config_source=config_source,
            ),
            measurement_tool_id=registration_config.capture_tool_id,
            coil_tool_id=registration_config.coil_tool_id,
            session_mode=self.SESSION_MODE_FULL_HAT,
            session_mode_summary="Full accepted hat calibration",
            selected_labels=list(self._truth_geometry.labels),
            labels=list(self._truth_geometry.labels),
            captures_per_landmark=int(registration_config.runtime_tip_captures_per_landmark),
            current_label=self._truth_geometry.labels[0] if self._truth_geometry.labels else None,
            truth_points_in_tip_by_label=copy.deepcopy(self._truth_geometry.truth_points_in_tip_by_label),
            raw_points_by_label={label: [] for label in self._truth_geometry.labels},
            captured_counts={label: 0 for label in self._truth_geometry.labels},
            coil_sample_count=int(registration_config.runtime_tip_coil_sample_count),
            hat_geometry_path=str(self._truth_geometry.points_file),
            hat_geometry_status="ready",
        )
        self._sync_dependency_status_locked()
        self.load_latest_accepted()
        with self._lock:
            self._recompute_health_locked()

    def begin_session(
        self,
        *,
        captures_per_landmark: int | None = None,
        session_mode: str | None = None,
    ) -> RuntimeTipCalibrationSnapshot:
        """Start a new 0A runtime tip calibration capture session."""
        measurement_status = self.registration_service.get_measurement_point_status(refresh=True)
        if not measurement_status["ready"]:
            raise RuntimeError(str(measurement_status["message"]))

        with self._lock:
            normalized_mode = self._normalize_session_mode(session_mode)
            active_labels = self._labels_for_session_mode(normalized_mode)
            requested_captures = max(
                1,
                int(
                    captures_per_landmark
                    if captures_per_landmark is not None
                    else (
                        1
                        if normalized_mode == self.SESSION_MODE_QUICK_4_POINT
                        else self.config.runtime_tip_captures_per_landmark
                    )
                ),
            )
            self._pending_record = None
            self._raw_measurement_tool_samples_by_label = {
                label: [] for label in active_labels
            }
            self._raw_coil_samples = []
            self._state.active = True
            self._state.pending_accept = False
            self._state.session_mode = normalized_mode
            self._state.session_mode_summary = self._session_mode_summary(normalized_mode)
            self._state.selected_labels = list(active_labels)
            self._state.labels = list(active_labels)
            self._state.captures_per_landmark = requested_captures
            self._state.current_label_index = 0
            self._state.current_label = active_labels[0] if active_labels else None
            self._state.truth_points_in_tip_by_label = {
                label: copy.deepcopy(self._truth_geometry.truth_points_in_tip_by_label[label])
                for label in active_labels
            }
            self._state.raw_points_by_label = {label: [] for label in active_labels}
            self._state.captured_counts = {label: 0 for label in active_labels}
            self._state.averaged_points_by_label = {}
            self._state.fit_residuals_by_label = {}
            self._state.fit_rmse_mm = None
            self._state.max_residual_mm = None
            self._state.translation_spread_summary_mm = {}
            self._state.rotation_spread_summary_deg = {}
            self._state.coil_sample_count = int(self.config.runtime_tip_coil_sample_count)
            self._state.coil_samples_captured = 0
            self._state.averaged_coil_translation_mm = None
            self._state.averaged_coil_quaternion_wxyz = None
            self._state.T_tip_aurora = None
            self._state.T_aurora_tip = None
            self._state.T_aurora_coil_avg = None
            self._state.T_coil_tip = None
            self._state.validation_metrics = {}
            self._state.pending_record = None
            self._state.measurement_point_status = str(measurement_status["message"])
            self._state.status_message = (
                "Runtime tip calibration started. "
                + (
                    "Capture the configured four-point quick subset, then collect stationary 0A samples."
                    if normalized_mode == self.SESSION_MODE_QUICK_4_POINT
                    else "Capture the configured hat points with the calibrated 0B pen probe."
                )
            )
            self._state.health.last_error = None
            self._state.health.last_successful_update_utc = utc_now_iso()
            self._state.health.state = "capturing_hat_points"
            self._sync_dependency_status_locked()
            self._recompute_health_locked()
            return copy.deepcopy(self._state)

    def capture_sample(self, label: str | None = None) -> list[float]:
        """Capture one 0B-measured hat point for the current or named label."""
        with self._lock:
            if not self._state.active:
                raise RuntimeError("Runtime tip calibration session has not been started")
            target_label = label or self._state.current_label
            if target_label is None:
                raise RuntimeError("No active hat point is selected")
            if target_label not in self._state.raw_points_by_label:
                raise ValueError(f"Unknown runtime tip calibration label: {target_label}")

        capture = self.registration_service.sample_measurement_point_capture()
        measurement_tool = capture["measurement_tool_snapshot"]
        point_xyz_mm = list(capture["point_xyz_mm"])
        measurement_status = dict(capture["measurement_point_status"])

        with self._lock:
            self._state.raw_points_by_label[target_label].append(point_xyz_mm)
            self._raw_measurement_tool_samples_by_label[target_label].append(
                self._tool_snapshot_to_dict(measurement_tool)
            )
            self._state.captured_counts[target_label] = len(self._state.raw_points_by_label[target_label])
            self._state.measurement_point_status = str(measurement_status.get("message", "ready"))
            self._state.status_message = (
                f"Captured hat point {target_label} sample {self._state.captured_counts[target_label]}."
            )
            self._state.health.last_error = None
            self._state.health.last_successful_update_utc = utc_now_iso()
            self._state.health.state = "capturing_hat_points"
            self._sync_dependency_status_locked()
            self._recompute_health_locked()
            return point_xyz_mm

    def complete_landmark(self) -> str | None:
        """Mark the current hat point complete and advance to the next point."""
        with self._lock:
            if not self._state.active:
                raise RuntimeError("Runtime tip calibration session has not been started")
            current_label = self._state.current_label
            if current_label is None:
                raise RuntimeError("All hat points are already complete")
            count = self._state.captured_counts.get(current_label, 0)
            if count < self._state.captures_per_landmark:
                raise RuntimeError(
                    f"Hat point {current_label} has {count} capture(s); requires {self._state.captures_per_landmark}"
                )
            self._state.current_label_index += 1
            if self._state.current_label_index >= len(self._state.labels):
                self._state.current_label = None
                self._state.health.state = "ready_for_coil_samples"
                self._state.status_message = (
                    "All hat points are complete. Collect stationary 0A samples next."
                )
            else:
                self._state.current_label = self._state.labels[self._state.current_label_index]
                self._state.health.state = "capturing_hat_points"
                self._state.status_message = f"{current_label} complete. Continue with {self._state.current_label}."
            self._sync_dependency_status_locked()
            self._recompute_health_locked()
            return self._state.current_label

    def collect_coil_samples(
        self,
        *,
        sample_count: int | None = None,
        sample_interval_s: float | None = None,
    ) -> RuntimeTipCalibrationSnapshot:
        """Collect stationary 0A coil poses for averaging."""
        with self._lock:
            if not self._state.active:
                raise RuntimeError("Runtime tip calibration session has not been started")
            if self._state.current_label is not None:
                raise RuntimeError("Complete all hat points before collecting 0A samples.")

        requested_count = max(1, int(sample_count or self.config.runtime_tip_coil_sample_count))
        interval_s = max(0.0, float(sample_interval_s if sample_interval_s is not None else self.config.runtime_tip_coil_sample_interval_s))
        raw_samples: list[dict[str, Any]] = []
        pose_samples: list[AuroraPoseSample] = []
        for index in range(requested_count):
            tool = self._require_tool_snapshot(self.config.coil_tool_id)
            raw = self._tool_snapshot_to_dict(tool)
            raw_samples.append(raw)
            pose_samples.append(self._tool_dict_to_pose_sample(raw))
            if index + 1 < requested_count and interval_s > 0.0:
                self.sleep_fn(interval_s)

        T_aurora_coil_avg, averaged_quaternion, averaged_translation, averaging_details = (
            self._average_coil_samples(pose_samples)
        )
        with self._lock:
            self._raw_coil_samples = raw_samples
            self._state.coil_sample_count = requested_count
            self._state.coil_samples_captured = requested_count
            self._state.averaged_coil_quaternion_wxyz = averaged_quaternion.astype(float).tolist()
            self._state.averaged_coil_translation_mm = averaged_translation.astype(float).tolist()
            self._state.T_aurora_coil_avg = T_aurora_coil_avg.tolist()
            self._state.translation_spread_summary_mm = dict(
                averaging_details.get("translation_spread_summary_mm", {})
            )
            self._state.rotation_spread_summary_deg = dict(
                averaging_details.get("rotation_spread_summary_deg", {})
            )
            self._state.status_message = f"Collected {requested_count} stationary 0A sample(s)."
            self._state.health.last_error = None
            self._state.health.last_successful_update_utc = utc_now_iso()
            self._state.health.state = "ready_to_solve"
            self._sync_dependency_status_locked()
            self._recompute_health_locked()
            return copy.deepcopy(self._state)

    def solve_calibration(self) -> dict[str, Any]:
        """Solve the hat-based runtime tip calibration and keep it pending until acceptance."""
        with self._lock:
            if not self._state.active:
                raise RuntimeError("Runtime tip calibration session has not been started")
            incomplete = [
                label
                for label in self._state.labels
                if self._state.captured_counts.get(label, 0) < self._state.captures_per_landmark
            ]
            if incomplete:
                raise RuntimeError(
                    "Complete all hat points before solving: " + ", ".join(incomplete)
                )
            if not self._raw_coil_samples:
                raise RuntimeError("Collect stationary 0A samples before solving the runtime tip calibration.")
            raw_points_by_label = copy.deepcopy(self._state.raw_points_by_label)
            truth_points_in_tip_by_label = copy.deepcopy(self._state.truth_points_in_tip_by_label)
            raw_coil_samples = list(self._raw_coil_samples)
            session_mode = str(self._state.session_mode or self.SESSION_MODE_FULL_HAT)

        result = solve_runtime_tip_calibration(
            solver=self.solver,
            labels=list(raw_points_by_label),
            raw_points_by_label=raw_points_by_label,
            truth_points_in_tip_by_label=truth_points_in_tip_by_label,
            coil_samples=[self._tool_dict_to_pose_sample(sample) for sample in raw_coil_samples],
            quaternion_average_method=self.config.quaternion_average_method,
        )
        if self.config.runtime_tip_max_hat_rmse_mm is not None and result.fit_rmse_mm > float(
            self.config.runtime_tip_max_hat_rmse_mm
        ):
            raise RuntimeError(
                f"Hat fit RMSE {result.fit_rmse_mm:.3f} mm exceeds configured limit "
                f"{float(self.config.runtime_tip_max_hat_rmse_mm):.3f} mm."
            )

        record = RuntimeTipCalibrationRecord(
            timestamp_utc=utc_now_iso(),
            calibration_kind=(
                "runtime_tip_calibration_quick_4_point"
                if session_mode == self.SESSION_MODE_QUICK_4_POINT
                else "runtime_tip_calibration_hat"
            ),
            measurement_tool_id=self.config.capture_tool_id,
            coil_tool_id=self.config.coil_tool_id,
            setup_id=self.config.runtime_tip_setup_id,
            truth_points_in_sw_by_label={
                label: copy.deepcopy(self._truth_geometry.truth_points_in_sw_by_label[label])
                for label in raw_points_by_label
            },
            truth_points_in_tip_by_label={
                label: copy.deepcopy(self._truth_geometry.truth_points_in_tip_by_label[label])
                for label in raw_points_by_label
            },
            raw_captured_hat_points_aurora_xyz_by_label=raw_points_by_label,
            averaged_hat_points_aurora_xyz_by_label=copy.deepcopy(result.averaged_points_by_label),
            raw_coil_samples=raw_coil_samples,
            residuals_tip_xyz_mm_by_label=copy.deepcopy(result.residuals_tip_xyz_mm_by_label),
            fit_rmse_mm=float(result.fit_rmse_mm),
            T_tip_aurora=result.T_tip_aurora.tolist(),
            T_aurora_tip=result.T_aurora_tip.tolist(),
            T_aurora_coil_avg=result.T_aurora_coil_avg.tolist(),
            T_coil_tip=result.T_coil_tip.tolist(),
            hat_geometry={
                "points_file": str(self._truth_geometry.points_file),
                "transform_file": str(self._truth_geometry.transform_file),
                "point_count": len(self._truth_geometry.labels),
            },
            validation_metrics=copy.deepcopy(result.validation_metrics),
            config_used={
                "registration_config_path": str(self.config_path),
                "measurement_tool_id": self.config.capture_tool_id,
                "coil_tool_id": self.config.coil_tool_id,
                "quaternion_average_method": self.config.quaternion_average_method,
                "captures_per_hat_point": self.config.runtime_tip_captures_per_landmark,
                "coil_sample_count": len(raw_coil_samples),
                "coil_sample_interval_s": self.config.runtime_tip_coil_sample_interval_s,
                "runtime_tip_truth_points_file": str(self._truth_geometry.points_file),
                "runtime_tip_T_sw_2_tip_file": str(self._truth_geometry.transform_file),
                "runtime_tip_setup_id": self.config.runtime_tip_setup_id,
                "session_mode": session_mode,
                "selected_labels": list(raw_points_by_label),
            },
        )

        with self._lock:
            self._pending_record = record
            self._state.averaged_points_by_label = copy.deepcopy(result.averaged_points_by_label)
            self._state.fit_residuals_by_label = copy.deepcopy(result.residuals_tip_xyz_mm_by_label)
            self._state.fit_rmse_mm = float(result.fit_rmse_mm)
            self._state.max_residual_mm = float(result.max_residual_mm)
            self._state.translation_spread_summary_mm = copy.deepcopy(result.coil_translation_spread_mm)
            self._state.rotation_spread_summary_deg = copy.deepcopy(result.coil_rotation_spread_deg)
            self._state.averaged_coil_quaternion_wxyz = result.averaged_coil_quaternion_wxyz.astype(float).tolist()
            self._state.averaged_coil_translation_mm = result.averaged_coil_translation_mm.astype(float).tolist()
            self._state.T_tip_aurora = result.T_tip_aurora.tolist()
            self._state.T_aurora_tip = result.T_aurora_tip.tolist()
            self._state.T_aurora_coil_avg = result.T_aurora_coil_avg.tolist()
            self._state.T_coil_tip = result.T_coil_tip.tolist()
            self._state.validation_metrics = copy.deepcopy(result.validation_metrics)
            self._state.pending_accept = True
            self._state.pending_record = asdict(record)
            self._state.status_message = (
                "Runtime tip calibration solved. Review the residuals and 0A spread metrics, then save the "
                + (
                    "quick four-point override."
                    if session_mode == self.SESSION_MODE_QUICK_4_POINT
                    else "accepted artifact."
                )
            )
            self._state.health.last_error = None
            self._state.health.last_successful_update_utc = utc_now_iso()
            self._state.health.state = "solved"
            self._sync_dependency_status_locked()
            self._recompute_health_locked()
            return copy.deepcopy(self._state.pending_record)

    def accept_calibration(self) -> Path:
        """Persist the pending runtime tip calibration as the latest accepted artifact."""
        with self._lock:
            if self._pending_record is None:
                raise RuntimeError("No solved runtime tip calibration is pending acceptance")
            record = self._pending_record

        is_quick = str(record.calibration_kind).strip().lower() == "runtime_tip_calibration_quick_4_point"
        output_path = self.repository.save_record(
            record,
            mark_as_latest=not is_quick,
            alias_path=(self.repository.quick_latest_path if is_quick else None),
        )
        self.tracking_service.refresh_registration()
        runtime_summary = self.get_runtime_tip_trust_summary(
            expected_timestamp_utc=record.timestamp_utc
        )
        with self._lock:
            self._state.accepted_output_path = str(output_path)
            self._state.latest_accepted_path = str(
                self.repository.quick_latest_path if is_quick else self.repository.latest_path
            )
            self._state.pending_accept = False
            self._state.active = False
            self._pending_record = None
            self._state.pending_record = None
            self._state.runtime_chain_state = str(runtime_summary.get("state", self._state.runtime_chain_state))
            self._state.runtime_chain_message = str(runtime_summary.get("message", self._state.runtime_chain_message))
            self._state.status_message = (
                f"Saved {'quick four-point runtime tip override' if is_quick else 'runtime tip calibration'} "
                f"to {output_path.name}."
            )
            self._state.health.last_error = None
            self._state.health.last_successful_update_utc = utc_now_iso()
            self._state.health.state = "accepted_override" if is_quick else "accepted"
            self._sync_dependency_status_locked()
            self._recompute_health_locked()
        return output_path

    def load_latest_accepted(self) -> dict[str, Any] | None:
        """Load the latest accepted full runtime tip calibration artifact if present."""
        return self.load_latest_saved(session_mode=self.SESSION_MODE_FULL_HAT)

    def load_latest_saved(self, *, session_mode: str | None = None) -> dict[str, Any] | None:
        """Load the latest saved runtime tip artifact for the requested session mode."""
        normalized_mode = self._normalize_session_mode(session_mode)
        payload = (
            self.repository.load_latest_quick_payload()
            if normalized_mode == self.SESSION_MODE_QUICK_4_POINT
            else self.repository.load_latest_payload()
        )
        with self._lock:
            latest_path = (
                self.repository.quick_latest_path
                if normalized_mode == self.SESSION_MODE_QUICK_4_POINT
                else self.repository.latest_path
            )
            self._state.latest_accepted_path = str(latest_path) if latest_path.exists() else None
        if payload is None:
            with self._lock:
                self._state.session_mode = normalized_mode
                self._state.session_mode_summary = self._session_mode_summary(normalized_mode)
                self._sync_dependency_status_locked()
                self._recompute_health_locked()
            return None

        with self._lock:
            labels = [str(label) for label in payload.get("truth_points_in_tip_by_label", {}).keys()] or list(self._truth_geometry.labels)
            calibration_kind = str(payload.get("calibration_kind", "runtime_tip_calibration_hat") or "runtime_tip_calibration_hat")
            session_mode = (
                self.SESSION_MODE_QUICK_4_POINT
                if calibration_kind == "runtime_tip_calibration_quick_4_point"
                else self.SESSION_MODE_FULL_HAT
            )
            self._state.session_mode = session_mode
            self._state.session_mode_summary = self._session_mode_summary(session_mode)
            self._state.selected_labels = list(labels)
            self._state.labels = labels
            self._state.truth_points_in_tip_by_label = dict(payload.get("truth_points_in_tip_by_label", {}) or {})
            self._state.raw_points_by_label = dict(payload.get("raw_captured_hat_points_aurora_xyz_by_label", {}) or {})
            self._state.captured_counts = {
                label: len(samples)
                for label, samples in self._state.raw_points_by_label.items()
            }
            self._state.averaged_points_by_label = dict(payload.get("averaged_hat_points_aurora_xyz_by_label", {}) or {})
            self._state.fit_residuals_by_label = dict(payload.get("residuals_tip_xyz_mm_by_label", {}) or {})
            self._state.fit_rmse_mm = float(payload["fit_rmse_mm"]) if payload.get("fit_rmse_mm") is not None else None
            self._state.max_residual_mm = float(
                (payload.get("validation_metrics", {}) or {}).get("max_hat_residual_mm")
            ) if (payload.get("validation_metrics", {}) or {}).get("max_hat_residual_mm") is not None else None
            self._state.validation_metrics = dict(payload.get("validation_metrics", {}) or {})
            self._state.T_tip_aurora = copy.deepcopy(payload.get("T_tip_aurora"))
            self._state.T_aurora_tip = copy.deepcopy(payload.get("T_aurora_tip"))
            self._state.T_aurora_coil_avg = copy.deepcopy(payload.get("T_aurora_coil_avg"))
            self._state.T_coil_tip = copy.deepcopy(payload.get("T_coil_tip"))
            self._state.coil_sample_count = len(list(payload.get("raw_coil_samples", []) or []))
            self._state.coil_samples_captured = self._state.coil_sample_count
            self._state.translation_spread_summary_mm = dict(
                (payload.get("validation_metrics", {}) or {}).get("coil_translation_spread_mm", {}) or {}
            )
            self._state.rotation_spread_summary_deg = dict(
                (payload.get("validation_metrics", {}) or {}).get("coil_rotation_spread_deg", {}) or {}
            )
            latest_path = (
                self.repository.quick_latest_path
                if session_mode == self.SESSION_MODE_QUICK_4_POINT
                else self.repository.latest_path
            )
            self._state.accepted_output_path = str(latest_path)
            self._state.pending_accept = False
            self._state.active = False
            self._state.current_label_index = 0
            self._state.current_label = None
            self._state.pending_record = None
            self._state.status_message = (
                "Loaded the latest quick four-point runtime tip override."
                if session_mode == self.SESSION_MODE_QUICK_4_POINT
                else "Loaded the latest accepted runtime tip calibration."
            )
            self._state.health.last_error = None
            self._state.health.last_successful_update_utc = str(payload.get("timestamp_utc") or utc_now_iso())
            self._state.health.state = "accepted"
            self._sync_dependency_status_locked()
            self._recompute_health_locked()
        return payload

    def get_snapshot(self) -> RuntimeTipCalibrationSnapshot:
        """Return a deep copy of the current runtime tip calibration state."""
        with self._lock:
            self._sync_dependency_status_locked()
            return copy.deepcopy(self._state)

    def get_runtime_tip_trust_summary(self, *, expected_timestamp_utc: str | None = None) -> dict[str, Any]:
        """Return operator-facing runtime-chain status for the current tip calibration artifact."""
        tracking_snapshot = self.tracking_service.get_snapshot()
        latest_payload = self.repository.load_latest_payload()
        latest_timestamp = (
            str(latest_payload.get("timestamp_utc"))
            if isinstance(latest_payload, dict) and latest_payload.get("timestamp_utc") is not None
            else None
        )
        expected = expected_timestamp_utc or latest_timestamp
        runtime_tip_state = getattr(tracking_snapshot, "runtime_tip_calibration_state", "missing_runtime_tip_calibration")
        timestamp_matches = expected is None or getattr(tracking_snapshot, "stored_runtime_tip_timestamp_utc", None) == expected
        role_match = (
            getattr(tracking_snapshot, "stored_runtime_tip_measurement_tool_id", None) == self.config.capture_tool_id
            and getattr(tracking_snapshot, "stored_runtime_tip_coil_tool_id", None) == self.config.coil_tool_id
        )
        if runtime_tip_state == "invalid_runtime_tip_calibration":
            state = "invalid_runtime_tip_calibration"
            message = "Tracking rejected the runtime tip calibration artifact."
        elif runtime_tip_state in {"missing_runtime_tip_calibration", "missing_quick_4_point_runtime_tip"}:
            state = "missing_runtime_tip_calibration"
            message = "The selected runtime tip mode does not currently have a usable saved artifact."
        elif runtime_tip_state == "quick_4_point_loaded":
            state = "quick_4_point_override"
            message = "Tracking is using the saved quick four-point runtime tip override."
        elif runtime_tip_state == "coil_as_tip":
            state = "coil_as_tip_override"
            message = "Tracking is using the explicit coil-as-tip override."
        elif runtime_tip_state == "identity_tip_fallback":
            state = "identity_tip_fallback"
            message = "Tracking is using the identity 0A-to-tip fallback instead of a saved runtime tip calibration."
        elif not timestamp_matches:
            state = "stale_runtime_tip_calibration_loaded"
            message = "Tracking loaded a runtime tip calibration, but it does not match the latest accepted timestamp."
        elif not role_match:
            state = "runtime_tip_role_mismatch"
            message = (
                f"Tracking loaded runtime tip roles {tracking_snapshot.stored_runtime_tip_measurement_tool_id}/"
                f"{tracking_snapshot.stored_runtime_tip_coil_tool_id} instead of "
                f"{self.config.capture_tool_id}/{self.config.coil_tool_id}."
            )
        elif tracking_snapshot.tip_pose_status == "ok":
            state = "ok"
            message = "Tracking is using the accepted runtime tip calibration in the live tip-pose chain."
        else:
            state = "waiting_for_live_tip_pose"
            message = (
                f"Tracking loaded the runtime tip calibration; live tip pose is waiting on '{tracking_snapshot.tip_pose_status}'."
            )
        return {
            "state": state,
            "message": message,
            "runtime_tip_calibration_state": runtime_tip_state,
            "runtime_tip_mode": getattr(tracking_snapshot, "runtime_tip_mode", None),
            "runtime_tip_trust_level": getattr(tracking_snapshot, "runtime_tip_trust_level", None),
            "stored_runtime_tip_timestamp_utc": getattr(tracking_snapshot, "stored_runtime_tip_timestamp_utc", None),
            "timestamp_matches_latest": timestamp_matches,
            "role_match": role_match,
            "runtime_tip_identity_fallback": bool(getattr(tracking_snapshot, "runtime_tip_identity_fallback", False)),
        }

    def _load_truth_geometry(self) -> RuntimeTipTruthGeometry:
        points_file = self._resolve_config_path(self.config.runtime_tip_truth_points_file)
        transform_file = self._resolve_config_path(self.config.runtime_tip_T_sw_2_tip_file)
        return load_runtime_tip_truth_geometry(points_file, transform_file)

    def _resolve_config_path(self, raw_path: str | None) -> Path:
        if raw_path in (None, ""):
            raise RuntimeError("Runtime tip calibration requires a configured path")
        path = Path(str(raw_path))
        if path.is_absolute():
            return path
        return self.project_root / path

    def _require_tool_snapshot(self, tool_id: str):
        snapshot = self.tracking_service.get_snapshot()
        if snapshot.packets_received_count <= 0:
            raise RuntimeError("Tracker has not produced any frames yet.")
        if snapshot.tracker_data_stale:
            age = snapshot.tracker_data_age_s
            if age is None:
                raise RuntimeError("Tracker data is stale. Refresh tracking before sampling 0A.")
            raise RuntimeError(f"Tracker data is stale ({age:.3f} s). Refresh tracking before sampling 0A.")
        tool = self.tracking_service.get_latest_tool(tool_id)
        if tool is None:
            raise RuntimeError(f"Tool {tool_id} has no runtime snapshot")
        if tool.tracking_state != "tracked":
            raise RuntimeError(f"Tool {tool_id} is not currently tracked ({tool.status})")
        if tool.T_aurora_tool is None or tool.quaternion_wxyz is None or tool.translation_mm is None:
            raise RuntimeError(f"Tool {tool_id} does not have a valid rigid transform")
        return tool

    @staticmethod
    def _tool_snapshot_to_dict(tool) -> dict[str, Any]:
        return {
            "tool_id": tool.tool_id,
            "tracking_state": tool.tracking_state,
            "valid": tool.valid,
            "validity_known": tool.validity_known,
            "status": tool.status,
            "quaternion_wxyz": list(tool.quaternion_wxyz) if tool.quaternion_wxyz is not None else None,
            "translation_mm": list(tool.translation_mm) if tool.translation_mm is not None else None,
            "quality": tool.quality,
            "source_row": tool.frame_number,
            "source_token": tool.last_update_utc,
        }

    @staticmethod
    def _tool_dict_to_pose_sample(raw: dict[str, Any]) -> AuroraPoseSample:
        quat = raw.get("quaternion_wxyz")
        translation = raw.get("translation_mm")
        if quat is None or translation is None:
            raise RuntimeError("Runtime tip calibration sample is missing quaternion/translation data")
        return AuroraPoseSample(
            tool_id=str(raw.get("tool_id", "")),
            quaternion_wxyz=tuple(float(value) for value in quat),
            translation_mm=tuple(float(value) for value in translation),
            quality=float(raw["quality"]) if raw.get("quality") is not None else None,
            source_row=int(raw["source_row"]) if raw.get("source_row") is not None else None,
            source_token=str(raw.get("source_token", "")),
        )

    def _average_coil_samples(
        self,
        samples: list[AuroraPoseSample],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        T_aurora_coil_avg, averaged_quaternion, averaged_translation, averaging_details = (
            average_tool_samples_as_transform(
                samples,
                method=self.config.quaternion_average_method,
            )
        )
        return (
            np.asarray(T_aurora_coil_avg, dtype=float),
            np.asarray(averaged_quaternion, dtype=float),
            np.asarray(averaged_translation, dtype=float),
            {
                "translation_spread_summary_mm": self._summarize_translation_spread(
                    samples,
                    np.asarray(averaged_translation, dtype=float),
                ),
                "rotation_spread_summary_deg": self._summarize_rotation_spread(
                    samples,
                    np.asarray(T_aurora_coil_avg, dtype=float),
                ),
                "quaternion_average_details": dict(averaging_details),
            },
        )

    @staticmethod
    def _summarize_translation_spread(samples: list[AuroraPoseSample], averaged_translation: np.ndarray) -> dict[str, Any]:
        deltas = [
            float(np.linalg.norm(np.asarray(sample.translation_mm, dtype=float) - averaged_translation))
            for sample in samples
        ]
        return RuntimeTipCalibrationService._summarize_scalar_values(deltas)

    @staticmethod
    def _summarize_rotation_spread(samples: list[AuroraPoseSample], T_aurora_coil_avg: np.ndarray) -> dict[str, Any]:
        R_avg = np.asarray(T_aurora_coil_avg[0:3, 0:3], dtype=float)
        deltas_deg: list[float] = []
        for sample in samples:
            T_aurora_coil = make_transform_A_B(sample.quaternion_wxyz, sample.translation_mm)
            delta = np.linalg.inv(R_avg) @ T_aurora_coil[0:3, 0:3]
            value = float((np.trace(delta) - 1.0) / 2.0)
            value = min(1.0, max(-1.0, value))
            deltas_deg.append(float(np.degrees(np.arccos(value))))
        return RuntimeTipCalibrationService._summarize_scalar_values(deltas_deg)

    @staticmethod
    def _summarize_scalar_values(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        arr = np.asarray(values, dtype=float)
        return {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    def _sync_dependency_status_locked(self) -> None:
        measurement_status = self.registration_service.get_measurement_point_status(refresh=True)
        self._state.measurement_point_status = str(measurement_status.get("message", "unknown"))
        tracking_snapshot = self.tracking_service.get_snapshot()
        active_runtime_tip_mode = str(
            getattr(tracking_snapshot, "runtime_tip_mode", TrackingService.RUNTIME_TIP_MODE_LATEST_ACCEPTED)
            or TrackingService.RUNTIME_TIP_MODE_LATEST_ACCEPTED
        )
        active_runtime_tip_trust = str(
            getattr(tracking_snapshot, "runtime_tip_trust_level", "missing") or "missing"
        )
        active_runtime_tip_message = str(
            getattr(tracking_snapshot, "runtime_tip_mode_message", "No runtime tip mode selected.")
            or "No runtime tip mode selected."
        )
        self._state.active_runtime_tip_mode = active_runtime_tip_mode
        self._state.active_runtime_tip_trust_level = active_runtime_tip_trust
        self._state.active_runtime_tip_mode_message = active_runtime_tip_message
        self._state.active_runtime_tip_guidance = self._runtime_tip_mode_guidance(
            mode=active_runtime_tip_mode,
        )
        runtime_summary = self.get_runtime_tip_trust_summary()
        self._state.runtime_chain_state = str(runtime_summary.get("state", self._state.runtime_chain_state))
        self._state.runtime_chain_message = str(
            runtime_summary.get("message", self._state.runtime_chain_message)
        )

    def _recompute_health_locked(self) -> None:
        faults: list[str] = []
        if not self.repository.latest_path.parent.exists():
            faults.append("runtime_tip_artifact_dir_missing")
        if self._state.health.last_error:
            faults.append("last_error")
        if self._state.active:
            incomplete = [
                label
                for label in self._state.labels
                if self._state.captured_counts.get(label, 0) < self._state.captures_per_landmark
            ]
            if incomplete:
                faults.append("hat_point_captures_incomplete")
        if self._state.pending_accept and self.config.runtime_tip_max_hat_rmse_mm is not None:
            if self._state.fit_rmse_mm is not None and self._state.fit_rmse_mm > float(self.config.runtime_tip_max_hat_rmse_mm):
                faults.append("hat_fit_above_limit")

        if self._state.health.last_error:
            health = HEALTH_FAILED
        elif faults:
            health = HEALTH_DEGRADED
        else:
            health = HEALTH_HEALTHY
        self._state.health.health = health
        self._state.health.current_config_source = self.config_source
        self._state.health.details = {
            "faults": list(faults),
            "measurement_tool_id": self._state.measurement_tool_id,
            "coil_tool_id": self._state.coil_tool_id,
            "captures_per_landmark": self._state.captures_per_landmark,
            "coil_sample_count": self._state.coil_sample_count,
            "fit_rmse_mm": self._state.fit_rmse_mm,
            "runtime_chain_state": self._state.runtime_chain_state,
            "runtime_chain_message": self._state.runtime_chain_message,
        }
        if self._state.health.state == "accepted":
            self._state.health.status = "Runtime tip calibration accepted"
        elif self._state.health.state == "accepted_override":
            self._state.health.status = "Runtime tip quick override saved"
        elif self._state.pending_accept:
            self._state.health.status = "Runtime tip calibration solved and pending acceptance"
        elif self._state.active:
            self._state.health.status = "Runtime tip calibration capture in progress"
        else:
            self._state.health.status = "Runtime tip calibration idle"

    @classmethod
    def _normalize_session_mode(cls, session_mode: str | None) -> str:
        value = str(session_mode or cls.SESSION_MODE_FULL_HAT).strip().lower()
        if value in {"quick_4_point", "quick4", "quick"}:
            return cls.SESSION_MODE_QUICK_4_POINT
        return cls.SESSION_MODE_FULL_HAT

    def _labels_for_session_mode(self, session_mode: str) -> list[str]:
        if session_mode == self.SESSION_MODE_QUICK_4_POINT:
            return list(self._quick_labels)
        return list(self._truth_geometry.labels)

    @classmethod
    def _session_mode_summary(cls, session_mode: str) -> str:
        if session_mode == cls.SESSION_MODE_QUICK_4_POINT:
            return "Quick 4-point runtime tip override"
        return "Full accepted hat calibration"

    @classmethod
    def _runtime_tip_mode_guidance(cls, *, mode: str) -> str:
        if mode == TrackingService.RUNTIME_TIP_MODE_COIL_AS_TIP:
            return (
                "Direct 0A / no-transform mode is active. The live tip is the 0A coil pose itself; "
                "you only need this calibration dialog if you want to save a new full or quick runtime-tip artifact."
            )
        if mode == TrackingService.RUNTIME_TIP_MODE_QUICK_4_POINT:
            return (
                "Quick 4-point live mode is selected. Use Session mode = Quick 4-Point Override below "
                "to capture or update the saved quick runtime-tip artifact."
            )
        return (
            "Latest accepted live mode is selected. Use Session mode = Full Accepted Hat Calibration below "
            "to capture or update the trusted full runtime-tip artifact."
        )

    @staticmethod
    def _choose_quick_labels(truth_points_in_tip_by_label: dict[str, list[float]]) -> list[str]:
        labels = [str(label) for label in truth_points_in_tip_by_label]
        if len(labels) <= 4:
            return list(labels)
        points = {
            str(label): np.asarray(point, dtype=float)
            for label, point in truth_points_in_tip_by_label.items()
        }
        best_labels = labels[:4]
        best_score = (-1.0, -1.0)
        label_count = len(labels)
        for index_a in range(label_count - 3):
            for index_b in range(index_a + 1, label_count - 2):
                for index_c in range(index_b + 1, label_count - 1):
                    for index_d in range(index_c + 1, label_count):
                        subset = [labels[index_a], labels[index_b], labels[index_c], labels[index_d]]
                        subset_points = [points[label] for label in subset]
                        pairwise = [
                            float(np.linalg.norm(first - second))
                            for first_index, first in enumerate(subset_points[:-1])
                            for second in subset_points[first_index + 1 :]
                        ]
                        min_pairwise = min(pairwise) if pairwise else 0.0
                        centered = np.stack(subset_points, axis=0) - np.mean(np.stack(subset_points, axis=0), axis=0)
                        singular_values = np.linalg.svd(centered, compute_uv=False)
                        conditioning = float(singular_values[-1]) if singular_values.size > 0 else 0.0
                        score = (min_pairwise, conditioning)
                        if score > best_score:
                            best_score = score
                            best_labels = subset
        return list(best_labels)
