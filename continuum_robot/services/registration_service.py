"""Reusable registration service built on top of the tracking service."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
from pathlib import Path
import re
import threading

import numpy as np
import yaml

from continuum_robot.registration.legacy_compat import (
    AuroraPoseSample,
    RegistrationAssetPaths,
    load_registration_assets,
    solve_registration_from_tool_samples,
)
from continuum_robot.registration.repository import RegistrationRecord, RegistrationRepository
from continuum_robot.registration.rigid_solver import RigidRegistrationSolver
from continuum_robot.registration.validation import (
    build_registration_history_summary,
    compute_fre_mm,
    compute_geometry_diagnostics,
    compute_residual_norms_mm,
    summarize_residual_norms_mm,
)
from continuum_robot.services.models import (
    HEALTH_DEGRADED,
    HEALTH_FAILED,
    HEALTH_HEALTHY,
    RegistrationSnapshot,
    ServiceHealthSnapshot,
)
from continuum_robot.services.tracking_service import TrackingService
from continuum_robot.tracking.transforms import assert_rigid_transform_matrix, compose_T_A_C
from continuum_robot.utils.time_utils import utc_now_iso


@dataclass
class RegistrationConfig:
    """Registration workflow settings loaded from YAML."""

    labels: list[str]
    captures_per_landmark: int
    nominal_landmarks_robot_xyz_mm: dict[str, list[float]]
    measurement_tool_id: str
    coil_tool_id: str
    capture_tool_tip_transform: list[list[float]] | None
    model_points_file: str | None
    tip_points_file: str | None
    T_sw_2_model_file: str | None
    T_sw_2_tip_file: str | None
    penprobe_file: str | None
    quaternion_average_method: str
    model_tre_reference_radius_mm: float
    tip_tre_reference_radius_mm: float
    max_fre_mm: float | None


class RegistrationService:
    """Own registration session state, solving, acceptance, and persistence."""

    def __init__(
        self,
        tracking_service: TrackingService,
        repository: RegistrationRepository,
        solver: RigidRegistrationSolver,
        *,
        config_path: Path,
        config_source: str,
    ) -> None:
        self.tracking_service = tracking_service
        self.repository = repository
        self.solver = solver
        self.config_path = config_path
        self.config_source = config_source

        self._lock = threading.Lock()
        self._config = self._load_config(config_path)
        self._assets = self._load_assets_if_configured(config_path, self._config)
        self._simple_measurement_point_transform = np.eye(4, dtype=float)
        self._simple_measurement_point_source = "coil_origin"
        self._simple_measurement_point_ready = True
        self._simple_measurement_point_error: str | None = None
        self._simple_measurement_point_path: str | None = None
        self._simple_measurement_point_mtime_ns: int | None = None
        self._pending_record: RegistrationRecord | None = None

        initial_labels = (
            [*self._assets.model_labels, *self._assets.tip_labels] if self._assets is not None else list(self._config.labels)
        )
        self._state = RegistrationSnapshot(
            health=ServiceHealthSnapshot(
                name="registration_service",
                health=HEALTH_DEGRADED,
                state="idle",
                status="Registration service ready",
                current_config_source=config_source,
            ),
            active=False,
            capture_tool_id=self._config.measurement_tool_id,
            labels=initial_labels,
            captures_per_landmark=self._config.captures_per_landmark,
            config_path=str(config_path),
            nominal_landmarks_robot_xyz_mm=copy.deepcopy(self._config.nominal_landmarks_robot_xyz_mm),
        )
        if self._assets is not None:
            self._state.truth_points_in_sw_by_label = self._truth_points_in_sw_by_label()
            self._state.group_by_label = self._group_by_label()
        self.refresh_measurement_point_geometry()
        self.load_latest_accepted()
        with self._lock:
            self._recompute_health_locked()

    def begin_session(
        self,
        *,
        labels: list[str] | None = None,
        captures_per_landmark: int | None = None,
        nominal_landmarks_robot_xyz_mm: dict[str, list[float]] | None = None,
    ) -> RegistrationSnapshot:
        """Start a new registration session."""
        measurement_status = self.get_measurement_point_status(refresh=True)
        if not measurement_status["ready"]:
            raise RuntimeError(str(measurement_status["message"]))
        use_legacy_assets = self._assets is not None and nominal_landmarks_robot_xyz_mm is None
        if use_legacy_assets:
            labels = list(labels or [*self._assets.model_labels, *self._assets.tip_labels])
            nominal = {}
            truth_points_in_sw_by_label = self._truth_points_in_sw_by_label()
            group_by_label = self._group_by_label()
        else:
            labels = list(labels or self._config.labels)
            nominal = copy.deepcopy(nominal_landmarks_robot_xyz_mm or self._config.nominal_landmarks_robot_xyz_mm)
            truth_points_in_sw_by_label = {}
            group_by_label = {}
        captures_per_landmark = int(captures_per_landmark or self._config.captures_per_landmark)

        with self._lock:
            self._pending_record = None
            self._state.active = True
            self._state.capture_tool_id = self._config.measurement_tool_id
            self._state.labels = labels
            self._state.captures_per_landmark = captures_per_landmark
            self._state.current_landmark_index = 0
            self._state.current_label = labels[0] if labels else None
            self._state.raw_points_by_label = {label: [] for label in labels}
            self._state.raw_measurement_tool_samples_by_label = {label: [] for label in labels}
            self._state.raw_coil_samples_by_label = {label: [] for label in labels}
            self._state.averaged_points_by_label = {}
            self._state.captured_counts = {label: 0 for label in labels}
            self._state.residuals_by_label = {}
            self._state.fre_mm = None
            self._state.T_robot_aurora = None
            self._state.last_sample_xyz_mm = None
            self._state.pending_accept = False
            self._state.nominal_landmarks_robot_xyz_mm = nominal
            self._state.truth_points_in_sw_by_label = truth_points_in_sw_by_label
            self._state.group_by_label = group_by_label
            self._state.validation_metrics = {}
            self._state.latest_validation_summary = {}
            self._state.pending_record = None
            self._state.health.last_error = None
            self._state.health.last_successful_update_utc = utc_now_iso()
            self._state.health.state = "capturing"
            self._recompute_health_locked()
            return copy.deepcopy(self._state)

    def capture_sample(self, label: str | None = None) -> list[float]:
        """Capture one registration sample from the configured tools."""
        with self._lock:
            if not self._state.active:
                raise RuntimeError("Registration session has not been started")
            target_label = label or self._state.current_label
            if target_label is None:
                raise RuntimeError("No active landmark is selected")
            if target_label not in self._state.raw_points_by_label:
                raise ValueError(f"Unknown registration label: {target_label}")

        capture = self.sample_measurement_point_capture()
        measurement_tool = capture["measurement_tool_snapshot"]
        sample = capture["point_xyz_mm"]

        with self._lock:
            self._state.raw_points_by_label[target_label].append(sample)
            self._state.raw_measurement_tool_samples_by_label[target_label].append(
                self._tool_snapshot_to_dict(measurement_tool)
            )
            if self._assets is not None:
                coil_tool = self._require_tool_snapshot(self._config.coil_tool_id)
                self._state.raw_coil_samples_by_label[target_label].append(self._tool_snapshot_to_dict(coil_tool))
            self._state.captured_counts[target_label] = len(self._state.raw_points_by_label[target_label])
            self._state.last_sample_xyz_mm = sample
            self._state.health.last_error = None
            self._state.health.last_successful_update_utc = utc_now_iso()
            self._state.health.state = "capturing"
            self._recompute_health_locked()
        return sample

    def sample_measurement_point_capture(self) -> dict[str, object]:
        """Capture the current 0B-measured point without mutating session state."""
        measurement_status = self.get_measurement_point_status(refresh=True)
        if not measurement_status["ready"]:
            raise RuntimeError(str(measurement_status["message"]))
        measurement_tool = self._require_tool_snapshot(self._config.measurement_tool_id)
        point_xyz_mm = self._measurement_point_from_tool_snapshot(measurement_tool)
        return {
            "point_xyz_mm": list(point_xyz_mm),
            "measurement_tool_snapshot": measurement_tool,
            "measurement_point_status": measurement_status,
        }

    def peek_current_measurement_point(self) -> dict[str, object]:
        """Return the current tracked measurement-point pose for GUI display."""
        measurement_status = self.get_measurement_point_status(refresh=True)
        if not measurement_status["ready"]:
            return {
                "available": False,
                "status": str(measurement_status["message"]),
                "tool_id": self._config.measurement_tool_id,
                "point_xyz_mm": None,
                "frame_number": None,
            }
        try:
            tool = self._require_tool_snapshot(self._config.measurement_tool_id)
            point_xyz = self._measurement_point_from_tool_snapshot(tool)
            return {
                "available": True,
                "status": "tracked",
                "tool_id": self._config.measurement_tool_id,
                "point_xyz_mm": point_xyz,
                "frame_number": tool.frame_number,
            }
        except Exception as exc:
            return {
                "available": False,
                "status": str(exc),
                "tool_id": self._config.measurement_tool_id,
                "point_xyz_mm": None,
                "frame_number": None,
            }

    def complete_landmark(self) -> str | None:
        """Advance to the next landmark once the current one has enough captures."""
        with self._lock:
            if not self._state.active or self._state.current_label is None:
                raise RuntimeError("No active registration landmark to complete")

            current_label = self._state.current_label
            count = len(self._state.raw_points_by_label[current_label])
            if count < self._state.captures_per_landmark:
                raise RuntimeError(
                    f"Landmark {current_label} has {count} capture(s); "
                    f"requires {self._state.captures_per_landmark}"
                )

            self._state.current_landmark_index += 1
            if self._state.current_landmark_index >= len(self._state.labels):
                self._state.current_label = None
                self._state.health.state = "ready_to_solve"
            else:
                self._state.current_label = self._state.labels[self._state.current_landmark_index]
                self._state.health.state = "capturing"
            self._recompute_health_locked()
            return self._state.current_label

    def solve_registration(self) -> dict:
        """Solve registration and keep the result pending until acceptance."""
        if self._assets is not None and self._state.truth_points_in_sw_by_label:
            return self._solve_legacy_compatible_registration()
        return self._solve_simple_registration()

    def accept_registration(self) -> Path:
        """Persist the pending registration as the latest accepted registration."""
        with self._lock:
            if self._pending_record is None:
                raise RuntimeError("No solved registration is pending acceptance")
            record = self._pending_record

        output_path = self.repository.save_record(record)
        validation_summary = self._build_repeated_validation_summary(
            current_record_path=output_path,
            current_record=record,
        )
        self.tracking_service.refresh_registration()
        validation_summary["runtime_application"] = self._build_runtime_application_summary(
            expected_timestamp_utc=record.timestamp_utc
        )
        self.repository.save_validation_summary(validation_summary, timestamp_utc=record.timestamp_utc)

        with self._lock:
            self._state.accepted_output_path = str(output_path)
            self._state.latest_accepted_path = str(self.repository.root_dir / "latest_registration.json")
            self._state.pending_accept = False
            self._state.active = False
            self._pending_record = None
            self._state.latest_validation_summary = copy.deepcopy(validation_summary)
            self._state.pending_record = None
            self._state.health.last_error = None
            self._state.health.last_successful_update_utc = utc_now_iso()
            self._state.health.state = "accepted"
            self._recompute_health_locked()
        return output_path

    def retry_session(self) -> RegistrationSnapshot:
        """Clear the current session and start over using the configured defaults."""
        return self.begin_session()

    def load_latest_accepted(self) -> dict | None:
        """Load metadata for the latest accepted registration if present."""
        latest_path = self.repository.root_dir / "latest_registration.json"
        with self._lock:
            self._state.latest_accepted_path = str(latest_path) if latest_path.exists() else None
            if not latest_path.exists():
                self._state.latest_validation_summary = {}
        if not latest_path.exists():
            return None
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
        with self._lock:
            labels = [str(label) for label in payload.get("landmark_labels", self._state.labels)]
            raw_points = {
                str(label): [list(sample) for sample in samples]
                for label, samples in dict(payload.get("raw_captured_landmarks_robot_xyz", {}) or {}).items()
            }
            averaged = {
                str(label): list(point)
                for label, point in dict(payload.get("averaged_landmarks_robot_xyz", {}) or {}).items()
            }
            residuals = {
                str(label): list(point)
                for label, point in dict(payload.get("residuals_robot_xyz_mm", {}) or {}).items()
            }
            self._state.accepted_output_path = str(latest_path)
            self._state.health.last_successful_update_utc = payload.get("timestamp_utc")
            self._state.active = False
            self._state.pending_accept = False
            self._state.current_landmark_index = 0
            self._state.current_label = None
            self._state.labels = labels
            self._state.raw_points_by_label = raw_points
            self._state.averaged_points_by_label = averaged
            self._state.captured_counts = {
                label: len(samples)
                for label, samples in raw_points.items()
            }
            self._state.residuals_by_label = residuals
            self._state.fre_mm = float(payload["fre_mm"]) if payload.get("fre_mm") is not None else None
            self._state.T_robot_aurora = (
                list(payload.get("T_robot_aurora"))
                if isinstance(payload.get("T_robot_aurora"), list)
                else (list(payload.get("T_aurora_2_model")) if isinstance(payload.get("T_aurora_2_model"), list) else None)
            )
            self._state.truth_points_in_sw_by_label = dict(payload.get("truth_points_in_sw_by_label", {}) or {})
            self._state.group_by_label = dict(payload.get("group_by_label", {}) or {})
            self._state.raw_measurement_tool_samples_by_label = dict(
                payload.get("raw_measurement_tool_samples_by_label", {}) or {}
            )
            self._state.raw_coil_samples_by_label = dict(payload.get("raw_coil_samples_by_label", {}) or {})
            self._state.validation_metrics = dict(payload.get("validation_metrics", {}) or {})
            self._state.latest_validation_summary = self.repository.load_latest_validation_summary() or {}
            self._state.pending_record = None
            self._recompute_health_locked()
        return payload

    def get_snapshot(self) -> RegistrationSnapshot:
        """Return a deep copy of the current registration state."""
        with self._lock:
            return copy.deepcopy(self._state)

    def refresh_measurement_point_geometry(self) -> dict[str, object]:
        """Reload the simple pen-tip geometry when the configured tip file changes."""
        if self._assets is not None:
            self._simple_measurement_point_ready = True
            self._simple_measurement_point_error = None
            self._simple_measurement_point_path = None
            self._simple_measurement_point_mtime_ns = None
            with self._lock:
                self._recompute_health_locked()
            return self.get_measurement_point_status(refresh=False)

        transform, source, ready, error, path, mtime_ns = self._load_simple_measurement_point_transform(
            self.config_path,
            self._config,
            previous_mtime_ns=self._simple_measurement_point_mtime_ns,
            previous_transform=self._simple_measurement_point_transform,
            previous_source=self._simple_measurement_point_source,
        )
        self._simple_measurement_point_transform = transform
        self._simple_measurement_point_source = source
        self._simple_measurement_point_ready = ready
        self._simple_measurement_point_error = error
        self._simple_measurement_point_path = str(path) if path is not None else None
        self._simple_measurement_point_mtime_ns = mtime_ns
        with self._lock:
            self._recompute_health_locked()
        return self.get_measurement_point_status(refresh=False)

    def get_measurement_point_status(self, *, refresh: bool = True) -> dict[str, object]:
        """Return the current pen-tip geometry source used for registration capture."""
        if refresh:
            self.refresh_measurement_point_geometry()
        transform = self._measurement_point_transform_matrix()
        tip_vector_mm = [float(value) for value in transform[0:3, 3]]
        transform_matrix = transform.tolist()
        file_path: str | None = None
        file_mtime_ns: int | None = None
        file_sha256: str | None = None
        if self._assets is not None:
            if self._config.penprobe_file:
                asset_path = self._resolve_config_path(self.config_path, self._config.penprobe_file)
                file_path = str(asset_path)
                if asset_path.exists():
                    file_mtime_ns = int(asset_path.stat().st_mtime_ns)
                    file_sha256 = self._sha256_file(asset_path)
            return {
                "ready": True,
                "source": "legacy_registration_assets",
                "path": file_path,
                "message": "Legacy asset geometry provides the pen-tip transform.",
                "tip_vector_mm": tip_vector_mm,
                "transform_matrix": transform_matrix,
                "file_mtime_ns": file_mtime_ns,
                "file_sha256": file_sha256,
                "offset_applied_before_solving": True,
            }
        if self._simple_measurement_point_ready:
            if self._simple_measurement_point_path:
                message = f"Pen-probe tip file loaded from {self._simple_measurement_point_path}."
            elif self._simple_measurement_point_source == "config.capture_tool_tip_transform":
                message = "Explicit 4x4 pen-tip transform loaded from config."
            else:
                message = "Registration is using coil origin because no explicit tip offset is configured."
        else:
            message = self._simple_measurement_point_error or "Pen-probe tip geometry is not ready."
        if self._simple_measurement_point_path:
            path = Path(self._simple_measurement_point_path)
            file_path = str(path)
            if path.exists():
                file_mtime_ns = int(path.stat().st_mtime_ns)
                file_sha256 = self._sha256_file(path)
        return {
            "ready": bool(self._simple_measurement_point_ready),
            "source": self._simple_measurement_point_source,
            "path": file_path,
            "message": message,
            "tip_vector_mm": tip_vector_mm,
            "transform_matrix": transform_matrix,
            "file_mtime_ns": file_mtime_ns,
            "file_sha256": file_sha256,
            "offset_applied_before_solving": True,
        }

    def _solve_legacy_compatible_registration(self) -> dict:
        with self._lock:
            if not self._state.active:
                raise RuntimeError("Registration session has not been started")
            labels = list(self._state.labels)
            grouped_measurement = copy.deepcopy(self._state.raw_measurement_tool_samples_by_label)
            grouped_coil = copy.deepcopy(self._state.raw_coil_samples_by_label)
            captures_per_landmark = self._state.captures_per_landmark

        for label in labels:
            measurement_count = len(grouped_measurement.get(label, []))
            coil_count = len(grouped_coil.get(label, []))
            if measurement_count != captures_per_landmark:
                raise RuntimeError(
                    f"Label {label} has {measurement_count} measurement-tool pose(s); expected {captures_per_landmark}"
                )
            if coil_count != captures_per_landmark:
                raise RuntimeError(f"Label {label} has {coil_count} coil-tool pose(s); expected {captures_per_landmark}")

        measurement_samples = self._flatten_pose_samples(grouped_measurement, labels)
        coil_samples = self._flatten_pose_samples(grouped_coil, labels)
        result = solve_registration_from_tool_samples(
            assets=self._assets,
            measurement_tool_samples=measurement_samples,
            coil_tool_samples=coil_samples,
            repetitions=captures_per_landmark,
            measurement_tool_id=self._config.measurement_tool_id,
            coil_tool_id=self._config.coil_tool_id,
            solver=self.solver,
            quaternion_average_method=self._config.quaternion_average_method,
            model_tre_reference_radius_mm=self._config.model_tre_reference_radius_mm,
            tip_tre_reference_radius_mm=self._config.tip_tre_reference_radius_mm,
        )
        self._validate_fre_limits(result.validation_metrics)
        record = self._record_from_legacy_result(result)

        with self._lock:
            self._pending_record = record
            self._state.raw_points_by_label = result.raw_points_by_label
            self._state.averaged_points_by_label = result.averaged_points_by_label
            self._state.residuals_by_label = result.residuals_by_label
            self._state.fre_mm = float(result.validation_metrics["overall_fre_mm"])
            self._state.validation_metrics = copy.deepcopy(record.validation_metrics)
            self._state.pending_accept = True
            self._state.pending_record = asdict(record)
            self._state.health.last_error = None
            self._state.health.last_successful_update_utc = utc_now_iso()
            self._state.health.state = "solved"
            self._recompute_health_locked()
            return copy.deepcopy(self._state.pending_record)

    def _solve_simple_registration(self) -> dict:
        with self._lock:
            if not self._state.active:
                raise RuntimeError("Registration session has not been started")
            labels = list(self._state.labels)
            captures = copy.deepcopy(self._state.raw_points_by_label)
            nominal = copy.deepcopy(self._state.nominal_landmarks_robot_xyz_mm)
            captures_per_landmark = self._state.captures_per_landmark

        for label in labels:
            points = captures.get(label, [])
            if len(points) < captures_per_landmark:
                raise RuntimeError(
                    f"Landmark {label} has {len(points)} capture(s); requires {captures_per_landmark}"
                )
            if label not in nominal:
                raise RuntimeError(f"Missing nominal landmark for label {label}")

        measurement_status = self.get_measurement_point_status(refresh=True)
        capture_tip_provenance = self._build_capture_tip_provenance(measurement_status)

        averaged = {
            label: np.asarray(points, dtype=float).mean(axis=0).tolist()
            for label, points in captures.items()
        }
        measured = np.asarray([averaged[label] for label in labels], dtype=float)
        truth = np.asarray([nominal[label] for label in labels], dtype=float)
        truth_rank = int(np.linalg.matrix_rank(truth - truth.mean(axis=0)))
        if truth_rank < 2:
            raise RuntimeError(
                "Selected landmarks are geometrically degenerate. Choose four points that span the platform more widely."
            )
        try:
            T_robot_aurora = self.solver.solve_T_robot_aurora(measured, truth)
        except Exception as exc:
            raise RuntimeError(f"Rigid registration solve failed: {exc}") from exc
        transformed = self.solver.apply_transform(T_robot_aurora, measured)
        residuals = truth - transformed
        residuals_by_label = {
            label: residuals[idx, :].tolist()
            for idx, label in enumerate(labels)
        }
        residual_norms_by_label = compute_residual_norms_mm(residuals_by_label)
        fre_mm = float(compute_fre_mm(list(residuals_by_label.values())))
        residual_summary = summarize_residual_norms_mm(residual_norms_by_label)
        max_residual_mm = float(residual_summary["max_residual_mm"] or 0.0)
        measured_geometry = compute_geometry_diagnostics(measured.tolist())
        truth_geometry = compute_geometry_diagnostics(truth.tolist())
        if self._config.max_fre_mm is not None and fre_mm > self._config.max_fre_mm:
            raise RuntimeError(f"Registration FRE {fre_mm:.3f} mm exceeds limit {self._config.max_fre_mm:.3f} mm")
        T_coil_tip, live_pose_tip_transform = self._build_simple_live_pose_tip_transform()
        validation_metrics = {
            "overall_fre_mm": fre_mm,
            "overall_rmse_mm": fre_mm,
            "max_residual_mm": max_residual_mm,
            "residual_norms_mm_by_label": residual_norms_by_label,
            "residual_vectors_mm_by_label": residuals_by_label,
            "registration_mode": "simple",
            "landmark_count": len(labels),
            "captures_per_landmark": captures_per_landmark,
            "capture_counts_by_label": {
                label: len(points)
                for label, points in captures.items()
            },
            "measured_landmark_frame": "aurora",
            "truth_landmark_frame": "robot",
            "residual_frame": "robot",
            "truth_geometry": truth_geometry,
            "measured_geometry": measured_geometry,
            "residual_summary": residual_summary,
            "worst_landmark_label": residual_summary["worst_landmark_label"],
            "worst_landmark_residual_mm": residual_summary["worst_landmark_residual_mm"],
            "configured_max_fre_mm": self._config.max_fre_mm,
        }

        record = RegistrationRecord(
            timestamp_utc=utc_now_iso(),
            landmark_labels=labels,
            raw_captured_landmarks_robot_xyz=captures,
            averaged_landmarks_robot_xyz=averaged,
            residuals_robot_xyz_mm=residuals_by_label,
            fre_mm=fre_mm,
            T_robot_aurora=T_robot_aurora.tolist(),
            T_coil_tip=T_coil_tip.tolist(),
            measurement_tool_id=self._config.measurement_tool_id,
            coil_tool_id=self._config.coil_tool_id,
            raw_measurement_tool_samples_by_label=copy.deepcopy(self._state.raw_measurement_tool_samples_by_label),
            raw_coil_samples_by_label=copy.deepcopy(self._state.raw_coil_samples_by_label),
            validation_metrics=validation_metrics,
            capture_tip_provenance=capture_tip_provenance,
            live_pose_tip_transform=live_pose_tip_transform,
            config_used={
                "registration_config_path": str(self.config_path),
                "capture_tool_id": self._config.measurement_tool_id,
                "measurement_tool_id": self._config.measurement_tool_id,
                "coil_tool_id": self._config.coil_tool_id,
                "registration_mode": "simple",
                "measurement_point_source": self._simple_measurement_point_source,
                "penprobe_file": self._config.penprobe_file,
                "capture_tip_offset_applied_before_solving": True,
                "capture_tip_file_path": capture_tip_provenance.get("path"),
                "capture_tip_file_sha256": capture_tip_provenance.get("file_sha256"),
                "capture_tip_vector_mm": capture_tip_provenance.get("tip_vector_mm"),
                "tip_calibration_source": live_pose_tip_transform.get("source"),
            },
        )

        with self._lock:
            self._pending_record = record
            self._state.averaged_points_by_label = averaged
            self._state.residuals_by_label = residuals_by_label
            self._state.fre_mm = fre_mm
            self._state.T_robot_aurora = T_robot_aurora.tolist()
            self._state.validation_metrics = copy.deepcopy(validation_metrics)
            self._state.pending_accept = True
            self._state.pending_record = asdict(record)
            self._state.health.last_error = None
            self._state.health.last_successful_update_utc = utc_now_iso()
            self._state.health.state = "solved"
            self._recompute_health_locked()
            return copy.deepcopy(self._state.pending_record)

    def _record_from_legacy_result(self, result) -> RegistrationRecord:
        validation_metrics = copy.deepcopy(result.validation_metrics)
        residual_norms_by_label = compute_residual_norms_mm(result.residuals_by_label)
        residual_summary = summarize_residual_norms_mm(residual_norms_by_label)
        validation_metrics.setdefault("residual_norms_mm_by_label", residual_norms_by_label)
        validation_metrics.setdefault("residual_vectors_mm_by_label", result.residuals_by_label)
        validation_metrics.setdefault("max_residual_mm", residual_summary["max_residual_mm"])
        validation_metrics.setdefault("residual_summary", residual_summary)
        validation_metrics.setdefault("worst_landmark_label", residual_summary["worst_landmark_label"])
        validation_metrics.setdefault("worst_landmark_residual_mm", residual_summary["worst_landmark_residual_mm"])
        validation_metrics.setdefault("configured_max_fre_mm", self._config.max_fre_mm)
        legacy_capture_tip_provenance = self._build_capture_tip_provenance(self.get_measurement_point_status(refresh=True))
        live_pose_tip_transform = {
            "coil_tool_id": result.coil_tool_id,
            "source": "legacy_registration_result",
            "assumption": (
                "Legacy-compatible registration solved the live 0A-to-tip transform directly. "
                "The saved T_coil_tip comes from the registration result."
            ),
            "T_coil_tip": result.T_coil_tip.tolist(),
        }
        return RegistrationRecord(
            timestamp_utc=utc_now_iso(),
            landmark_labels=result.ordered_labels,
            raw_captured_landmarks_robot_xyz=result.raw_points_by_label,
            averaged_landmarks_robot_xyz=result.averaged_points_by_label,
            residuals_robot_xyz_mm=result.residuals_by_label,
            fre_mm=float(result.validation_metrics["overall_fre_mm"]),
            T_robot_aurora=result.T_aurora_2_model.tolist(),
            T_coil_tip=result.T_coil_tip.tolist(),
            T_aurora_2_tip=result.T_aurora_2_tip.tolist(),
            measurement_tool_id=result.measurement_tool_id,
            coil_tool_id=result.coil_tool_id,
            raw_measurement_tool_samples_by_label=result.raw_measurement_tool_samples_by_label,
            raw_coil_samples_by_label=result.raw_coil_samples_by_label,
            truth_points_in_sw_by_label=result.truth_points_in_sw_by_label,
            group_by_label=result.group_by_label,
            validation_metrics=validation_metrics,
            capture_tip_provenance=legacy_capture_tip_provenance,
            live_pose_tip_transform=live_pose_tip_transform,
            config_used={
                "registration_config_path": str(self.config_path),
                "capture_tool_id": result.measurement_tool_id,
                "measurement_tool_id": result.measurement_tool_id,
                "coil_tool_id": result.coil_tool_id,
                "captures_per_landmark": result.repetitions,
                "quaternion_average_method": self._config.quaternion_average_method,
                "registration_mode": "legacy_compatible",
                "model_points_file": self._config.model_points_file,
                "tip_points_file": self._config.tip_points_file,
                "T_sw_2_model_file": self._config.T_sw_2_model_file,
                "T_sw_2_tip_file": self._config.T_sw_2_tip_file,
                "penprobe_file": self._config.penprobe_file,
                "capture_tip_offset_applied_before_solving": True,
                "capture_tip_file_path": legacy_capture_tip_provenance.get("path"),
                "capture_tip_file_sha256": legacy_capture_tip_provenance.get("file_sha256"),
                "capture_tip_vector_mm": legacy_capture_tip_provenance.get("tip_vector_mm"),
                "tip_calibration_source": live_pose_tip_transform.get("source"),
            },
        )

    def _measurement_point_from_tool_snapshot(self, tool) -> list[float]:
        measurement_status = self.get_measurement_point_status(refresh=True)
        if not measurement_status["ready"]:
            raise RuntimeError(str(measurement_status["message"]))
        if tool.quaternion_wxyz is None or tool.translation_mm is None:
            raise RuntimeError(f"Tool {tool.tool_id} is missing quaternion/translation data")
        T_aurora_tool = np.asarray(tool.T_aurora_tool, dtype=float) if tool.T_aurora_tool is not None else None
        if T_aurora_tool is None or T_aurora_tool.shape != (4, 4):
            raise RuntimeError(f"Tool {tool.tool_id} is missing a valid T_aurora_tool transform")
        if self._assets is not None:
            T_measurement_point = self._assets.T_measurement_point
        else:
            T_measurement_point = self._simple_measurement_point_transform
        T_aurora_point = compose_T_A_C(T_aurora_tool, T_measurement_point)
        return [float(v) for v in T_aurora_point[0:3, 3]]

    def _require_tool_snapshot(self, tool_id: str):
        snapshot = self.tracking_service.get_snapshot()
        if snapshot.packets_received_count <= 0:
            raise RuntimeError("Tracker has not produced any frames yet.")
        if snapshot.tracker_data_stale:
            age = snapshot.tracker_data_age_s
            if age is None:
                raise RuntimeError("Tracker data is stale. Refresh tracking before capture.")
            raise RuntimeError(f"Tracker data is stale ({age:.3f} s). Refresh tracking before capture.")
        tool = self.tracking_service.get_latest_tool(tool_id)
        if tool is None:
            raise RuntimeError(f"Tool {tool_id} has no runtime snapshot")
        if tool.tracking_state != "tracked":
            raise RuntimeError(f"Tool {tool_id} is not currently tracked ({tool.status})")
        if tool.T_aurora_tool is None:
            raise RuntimeError(f"Tool {tool_id} does not have a valid rigid transform")
        return tool

    @staticmethod
    def _tool_snapshot_to_dict(tool) -> dict[str, object]:
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
    def _flatten_pose_samples(grouped_samples: dict[str, list[dict]], ordered_labels: list[str]) -> list[AuroraPoseSample]:
        output: list[AuroraPoseSample] = []
        for label in ordered_labels:
            for raw in grouped_samples.get(label, []):
                if raw.get("quaternion_wxyz") is None or raw.get("translation_mm") is None:
                    raise RuntimeError(f"Label {label} contains a tool sample without quaternion/translation data")
                output.append(
                    AuroraPoseSample(
                        tool_id=str(raw["tool_id"]),
                        quaternion_wxyz=tuple(float(v) for v in raw["quaternion_wxyz"]),
                        translation_mm=tuple(float(v) for v in raw["translation_mm"]),
                        quality=float(raw["quality"]) if raw.get("quality") is not None else None,
                        source_row=int(raw["source_row"]) if raw.get("source_row") is not None else None,
                        source_token=str(raw.get("source_token", "")),
                    )
                )
        return output

    def _group_by_label(self) -> dict[str, str]:
        if self._assets is None:
            return {}
        return {label: "model" for label in self._assets.model_labels} | {
            label: "tip" for label in self._assets.tip_labels
        }

    def _truth_points_in_sw_by_label(self) -> dict[str, list[float]]:
        if self._assets is None:
            return {}
        output = {
            label: self._assets.model_truth_in_sw[:, idx].astype(float).tolist()
            for idx, label in enumerate(self._assets.model_labels)
        }
        output.update(
            {
                label: self._assets.tip_truth_in_sw[:, idx].astype(float).tolist()
                for idx, label in enumerate(self._assets.tip_labels)
            }
        )
        return output

    def _validate_fre_limits(self, validation_metrics: dict[str, object]) -> None:
        if self._config.max_fre_mm is None:
            return
        offending = {
            key: float(validation_metrics[key])
            for key in ("model_fre_mm", "tip_fre_mm", "overall_fre_mm")
            if key in validation_metrics and validation_metrics[key] is not None and float(validation_metrics[key]) > self._config.max_fre_mm
        }
        if offending:
            rendered = ", ".join(f"{key}={value:.3f}" for key, value in offending.items())
            raise RuntimeError(f"Registration FRE exceeds limit {self._config.max_fre_mm:.3f} mm: {rendered}")

    @staticmethod
    def _load_config(path: Path) -> RegistrationConfig:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
        payload = payload or {}
        candidate_nominal, enabled_candidate_labels = RegistrationService._candidate_landmarks_from_payload(payload)
        labels = payload.get("landmark_labels")
        if labels is None:
            labels = enabled_candidate_labels[:4] if enabled_candidate_labels else ["L1", "L2", "L3", "L4"]
        labels = list(labels)
        captures = int(payload.get("captures_per_landmark", 5))
        nominal = dict(candidate_nominal)
        nominal.update(dict(payload.get("nominal_landmarks_robot_xyz_mm", {})))
        validation = payload.get("validation", {})
        max_fre = validation.get("max_fre_mm")
        return RegistrationConfig(
            labels=labels,
            captures_per_landmark=captures,
            nominal_landmarks_robot_xyz_mm=nominal,
            measurement_tool_id=str(payload.get("capture_tool_id", "0B")),
            coil_tool_id=str(payload.get("coil_tool_id", "0A")),
            capture_tool_tip_transform=payload.get("capture_tool_tip_transform"),
            model_points_file=payload.get("model_points_file"),
            tip_points_file=payload.get("tip_points_file"),
            T_sw_2_model_file=payload.get("T_sw_2_model_file"),
            T_sw_2_tip_file=payload.get("T_sw_2_tip_file"),
            penprobe_file=payload.get("penprobe_file"),
            quaternion_average_method=str(payload.get("quaternion_average_method", "sign_aligned_mean")),
            model_tre_reference_radius_mm=float(payload.get("model_tre_reference_radius_mm", 5.0)),
            tip_tre_reference_radius_mm=float(payload.get("tip_tre_reference_radius_mm", 3.0)),
            max_fre_mm=float(max_fre) if max_fre is not None else None,
        )

    @staticmethod
    def _candidate_landmarks_from_payload(payload: dict) -> tuple[dict[str, list[float]], list[str]]:
        nominal: dict[str, list[float]] = {}
        enabled_labels: list[str] = []
        for index, entry in enumerate(payload.get("candidate_landmarks", []) or []):
            if not isinstance(entry, dict):
                raise ValueError(f"candidate_landmarks[{index}] must be a mapping")
            label = str(entry.get("id") or entry.get("label") or entry.get("name") or "").strip()
            xyz = entry.get("xyz_mm") or entry.get("coordinates_mm") or entry.get("xyz")
            if not label:
                raise ValueError(f"candidate_landmarks[{index}] is missing id")
            if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
                raise ValueError(f"candidate_landmarks[{index}] must define xyz_mm with exactly 3 values")
            nominal[label] = [float(value) for value in xyz]
            if bool(entry.get("enabled", True)):
                enabled_labels.append(label)
        return nominal, enabled_labels

    @staticmethod
    def _load_assets_if_configured(config_path: Path, config: RegistrationConfig):
        if not (
            config.model_points_file
            and config.tip_points_file
            and config.T_sw_2_model_file
            and config.T_sw_2_tip_file
            and config.penprobe_file
        ):
            return None
        project_root = RegistrationService._config_project_root(config_path)

        def _resolve(raw: str) -> Path:
            path = Path(raw)
            return path if path.is_absolute() else project_root / path

        return load_registration_assets(
            RegistrationAssetPaths(
                model_points_file=_resolve(config.model_points_file),
                tip_points_file=_resolve(config.tip_points_file),
                T_sw_2_model_file=_resolve(config.T_sw_2_model_file),
                T_sw_2_tip_file=_resolve(config.T_sw_2_tip_file),
                penprobe_file=_resolve(config.penprobe_file),
            ),
            measurement_point_transform=config.capture_tool_tip_transform,
        )

    @classmethod
    def _load_simple_measurement_point_transform(
        cls,
        config_path: Path,
        config: RegistrationConfig,
        *,
        previous_mtime_ns: int | None = None,
        previous_transform: np.ndarray | None = None,
        previous_source: str | None = None,
    ) -> tuple[np.ndarray, str, bool, str | None, Path | None, int | None]:
        if config.capture_tool_tip_transform is not None:
            return (
                cls._coerce_transform(config.capture_tool_tip_transform),
                "config.capture_tool_tip_transform",
                True,
                None,
                None,
                None,
            )
        if config.penprobe_file:
            penprobe_path = cls._resolve_config_path(config_path, config.penprobe_file)
            if not penprobe_path.exists():
                return (
                    np.eye(4, dtype=float),
                    str(penprobe_path),
                    False,
                    f"Pen-probe tip file is missing: {penprobe_path}. Run pivot calibration and save the tip file first.",
                    penprobe_path,
                    None,
                )
            mtime_ns = penprobe_path.stat().st_mtime_ns
            if (
                previous_mtime_ns is not None
                and previous_mtime_ns == mtime_ns
                and previous_transform is not None
                and previous_source is not None
            ):
                return previous_transform, previous_source, True, None, penprobe_path, mtime_ns
            try:
                vector = cls._load_vector3(penprobe_path)
            except Exception as exc:
                return (
                    np.eye(4, dtype=float),
                    str(penprobe_path),
                    False,
                    f"Pen-probe tip file is invalid: {penprobe_path}. Details: {exc}",
                    penprobe_path,
                    mtime_ns,
                )
            matrix = np.eye(4, dtype=float)
            matrix[0:3, 3] = vector
            return matrix, str(penprobe_path), True, None, penprobe_path, mtime_ns
        return np.eye(4, dtype=float), "coil_origin", True, None, None, None

    @staticmethod
    def _resolve_config_path(config_path: Path, raw_path: str) -> Path:
        path = Path(raw_path)
        return path if path.is_absolute() else RegistrationService._config_project_root(config_path) / path

    @staticmethod
    def _config_project_root(config_path: Path) -> Path:
        resolved = config_path.resolve()
        for parent in resolved.parents:
            if parent.name == "config":
                return parent.parent
        return resolved.parent

    @staticmethod
    def _load_vector3(path: Path) -> np.ndarray:
        raw = path.read_text(encoding="utf-8").strip()
        values = [float(token) for token in re.split(r"[\s,]+", raw) if token]
        if len(values) != 3:
            raise ValueError(f"Expected exactly 3 values in {path}")
        return np.asarray(values, dtype=float)

    @staticmethod
    def _coerce_transform(transform: list[list[float]] | None) -> np.ndarray:
        if transform is None:
            return np.eye(4)
        matrix = np.asarray(transform, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError("capture_tool_tip_transform must be 4x4")
        assert_rigid_transform_matrix(matrix, "capture_tool_tip_transform")
        return matrix

    def get_latest_registration_artifact_summary(self) -> dict[str, object]:
        """Return provenance details from the latest accepted registration artifact."""
        latest_path = self.repository.root_dir / "latest_registration.json"
        if not latest_path.exists():
            return {}
        try:
            payload = json.loads(latest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"path": str(latest_path), "error": str(exc)}
        return {
            "path": str(latest_path),
            "timestamp_utc": payload.get("timestamp_utc"),
            "fre_mm": payload.get("fre_mm"),
            "capture_tip_provenance": dict(payload.get("capture_tip_provenance", {}) or {}),
            "live_pose_tip_transform": dict(payload.get("live_pose_tip_transform", {}) or {}),
            "validation_metrics": dict(payload.get("validation_metrics", {}) or {}),
            "config_used": dict(payload.get("config_used", {}) or {}),
            "T_coil_tip": copy.deepcopy(payload.get("T_coil_tip")),
        }

    def get_registration_trust_summary(self) -> dict[str, object]:
        """Return operator-facing trust and transform-chain summary for the latest accepted registration."""
        latest_path = self.repository.root_dir / "latest_registration.json"
        if not latest_path.exists():
            return {
                "accepted_exists": False,
                "trust_state": "missing",
                "trust_message": "No accepted registration saved.",
                "live_chain_state": "missing_registration",
                "live_chain_message": "Live robot-frame pose is unavailable until an accepted registration is saved.",
                "comparison_message": "Repeat the registration workflow across runs to build comparison data.",
                "detail_lines": [
                    "Registration-dependent experiments should wait until an accepted registration exists.",
                ],
            }

        try:
            payload = json.loads(latest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "accepted_exists": True,
                "trust_state": "invalid",
                "trust_message": f"Accepted registration exists but could not be read: {exc}",
                "live_chain_state": "invalid_registration",
                "live_chain_message": "Tracking cannot trust the registration artifact until the file is readable again.",
                "comparison_message": "Repeated-run comparison is unavailable while the accepted artifact is unreadable.",
                "detail_lines": [f"Artifact error: {exc}"],
            }

        validation_metrics = dict(payload.get("validation_metrics", {}) or {})
        history_summary = self.repository.load_latest_validation_summary() or {}
        tracking_summary = self._build_runtime_application_summary(
            expected_timestamp_utc=str(payload.get("timestamp_utc")) if payload.get("timestamp_utc") is not None else None
        )
        fre_mm = validation_metrics.get("overall_fre_mm")
        max_residual_mm = validation_metrics.get("max_residual_mm")
        residual_summary = dict(validation_metrics.get("residual_summary", {}) or {})
        worst_label = residual_summary.get("worst_landmark_label") or validation_metrics.get("worst_landmark_label")
        worst_residual_mm = residual_summary.get("worst_landmark_residual_mm") or validation_metrics.get(
            "worst_landmark_residual_mm"
        )
        fre_limit_mm = validation_metrics.get("configured_max_fre_mm", self._config.max_fre_mm)
        trust_state = "trusted"
        trust_reasons: list[str] = []
        if fre_mm is None:
            trust_state = "warning"
            trust_reasons.append("FRE is missing from the accepted artifact.")
        elif fre_limit_mm is not None and float(fre_mm) > float(fre_limit_mm):
            trust_state = "warning"
            trust_reasons.append(
                f"FRE {float(fre_mm):.3f} mm exceeds the configured limit {float(fre_limit_mm):.3f} mm."
            )
        if (
            worst_residual_mm is not None
            and fre_limit_mm is not None
            and float(worst_residual_mm) > float(fre_limit_mm)
        ):
            trust_state = "warning"
            label_text = f"{worst_label} " if worst_label else ""
            trust_reasons.append(
                f"Worst landmark residual {label_text}{float(worst_residual_mm):.3f} mm exceeds the configured limit."
            )

        if not trust_reasons:
            fre_text = f"FRE {float(fre_mm):.3f} mm" if fre_mm is not None else "FRE unavailable"
            if worst_residual_mm is not None and worst_label:
                trust_reasons.append(
                    f"{fre_text}; worst landmark {worst_label} = {float(worst_residual_mm):.3f} mm."
                )
            else:
                trust_reasons.append(f"{fre_text}.")

        comparison_message = self._comparison_message(history_summary)
        detail_lines = self._build_trust_detail_lines(
            payload=payload,
            validation_metrics=validation_metrics,
            history_summary=history_summary,
            tracking_summary=tracking_summary,
        )
        return {
            "accepted_exists": True,
            "path": str(latest_path),
            "timestamp_utc": payload.get("timestamp_utc"),
            "fre_mm": float(fre_mm) if fre_mm is not None else None,
            "max_residual_mm": float(max_residual_mm) if max_residual_mm is not None else None,
            "worst_landmark_label": worst_label,
            "worst_landmark_residual_mm": float(worst_residual_mm) if worst_residual_mm is not None else None,
            "trust_state": trust_state,
            "trust_message": " ".join(str(reason) for reason in trust_reasons),
            "live_chain_state": tracking_summary["state"],
            "live_chain_message": tracking_summary["message"],
            "comparison_message": comparison_message,
            "history_summary": history_summary,
            "runtime_application": tracking_summary,
            "detail_lines": detail_lines,
        }

    def _build_repeated_validation_summary(
        self,
        *,
        current_record_path: Path,
        current_record: RegistrationRecord,
        comparison_limit: int = 5,
    ) -> dict[str, object]:
        current_payload = self.repository.load_payload(current_record_path)
        recent_paths = [
            path
            for path in self.repository.list_saved_records(limit=max(1, comparison_limit + 1))
            if path != current_record_path
        ][: max(0, int(comparison_limit))]
        previous_runs = [
            (str(path), self.repository.load_payload(path))
            for path in recent_paths
        ]
        summary = build_registration_history_summary(
            current_payload=current_payload,
            current_path=str(current_record_path),
            previous_runs=previous_runs,
        )
        summary["generated_at_utc"] = utc_now_iso()
        summary["validation_kind"] = "repeated_registration_comparison"
        summary["current_record_timestamp_utc"] = current_record.timestamp_utc
        summary["registration_mode"] = (
            str(current_record.validation_metrics.get("registration_mode"))
            if isinstance(current_record.validation_metrics, dict)
            else None
        )
        summary["current_fre_mm"] = (
            float(current_record.validation_metrics.get("overall_fre_mm"))
            if isinstance(current_record.validation_metrics, dict)
            and current_record.validation_metrics.get("overall_fre_mm") is not None
            else float(current_record.fre_mm)
        )
        return summary

    def _build_runtime_application_summary(self, *, expected_timestamp_utc: str | None = None) -> dict[str, object]:
        snapshot = self.tracking_service.get_snapshot()
        loaded_latest = bool(
            snapshot.registration_state == "loaded"
            and snapshot.registration_path == str(self.repository.root_dir / "latest_registration.json")
        )
        timestamp_matches = (
            expected_timestamp_utc is None
            or snapshot.stored_registration_timestamp_utc == expected_timestamp_utc
        )
        role_match = (
            snapshot.stored_registration_measurement_tool_id == self._config.measurement_tool_id
            and snapshot.stored_registration_coil_tool_id == self._config.coil_tool_id
            and snapshot.stored_runtime_tip_measurement_tool_id in (None, self._config.measurement_tool_id)
            and snapshot.stored_runtime_tip_coil_tool_id in (None, self._config.coil_tool_id)
        )
        if not loaded_latest:
            state = "registration_not_loaded"
            message = (
                f"Tracking registration state is {snapshot.registration_state}; "
                "live robot-frame pose is not currently using the accepted registration."
            )
        elif snapshot.runtime_tip_calibration_state == "invalid_runtime_tip_calibration":
            state = "invalid_runtime_tip_calibration"
            message = "Tracking rejected the runtime 0A hat-based tip calibration artifact."
        elif snapshot.runtime_tip_calibration_state == "missing_runtime_tip_calibration":
            state = "missing_runtime_tip_calibration"
            message = (
                "Tracking loaded the accepted base registration, but no separate runtime 0A tip calibration artifact "
                "is available yet."
            )
        elif snapshot.runtime_tip_calibration_state == "identity_tip_fallback":
            state = "identity_tip_fallback"
            message = (
                "Tracking is still using the identity 0A-to-tip fallback. "
                "Run the separate runtime tip calibration before trusting robot-frame tip pose."
            )
        elif not timestamp_matches:
            state = "stale_registration_loaded"
            message = (
                "Tracking loaded a registration artifact, but its timestamp does not match the latest accepted save."
            )
        elif not role_match:
            state = "role_mismatch"
            message = (
                f"Tracking loaded the registration, but stored roles are measurement={snapshot.stored_registration_measurement_tool_id} "
                f"and coil={snapshot.stored_registration_coil_tool_id} instead of "
                f"{self._config.measurement_tool_id}/{self._config.coil_tool_id}."
            )
        elif snapshot.tip_pose_status == "ok":
            state = "ok"
            message = "Tracking is using this accepted registration and live robot-frame pose is available."
        else:
            state = "registration_loaded_waiting_for_live_pose"
            message = (
                f"Tracking loaded this accepted registration; live pose is waiting on runtime status "
                f"'{snapshot.tip_pose_status}'."
            )
        return {
            "state": state,
            "message": message,
            "registration_state": snapshot.registration_state,
            "tip_pose_status": snapshot.tip_pose_status,
            "registration_path": snapshot.registration_path,
            "loaded_latest_registration": loaded_latest,
            "timestamp_matches_latest": timestamp_matches,
            "role_match": role_match,
            "stored_registration_timestamp_utc": snapshot.stored_registration_timestamp_utc,
            "stored_registration_fre_mm": snapshot.stored_registration_fre_mm,
            "stored_registration_measurement_tool_id": snapshot.stored_registration_measurement_tool_id,
            "stored_registration_coil_tool_id": snapshot.stored_registration_coil_tool_id,
            "runtime_tip_calibration_state": snapshot.runtime_tip_calibration_state,
            "runtime_tip_calibration_path": snapshot.runtime_tip_calibration_path,
            "stored_runtime_tip_timestamp_utc": snapshot.stored_runtime_tip_timestamp_utc,
            "stored_runtime_tip_measurement_tool_id": snapshot.stored_runtime_tip_measurement_tool_id,
            "stored_runtime_tip_coil_tool_id": snapshot.stored_runtime_tip_coil_tool_id,
            "runtime_tip_identity_fallback": snapshot.runtime_tip_identity_fallback,
        }

    @staticmethod
    def _comparison_message(history_summary: dict[str, object]) -> str:
        comparison_count = int(history_summary.get("comparison_count") or 0)
        if comparison_count <= 0:
            return "This is the first saved registration; repeat the workflow to build comparison data."
        translation_summary = dict(history_summary.get("translation_delta_summary_mm", {}) or {})
        rotation_summary = dict(history_summary.get("rotation_delta_summary_deg", {}) or {})
        translation_mean = translation_summary.get("mean")
        translation_max = translation_summary.get("max")
        rotation_mean = rotation_summary.get("mean")
        rotation_max = rotation_summary.get("max")
        return (
            f"Compared against {comparison_count} prior registration run(s): "
            f"translation delta mean={float(translation_mean or 0.0):.3f} mm, "
            f"max={float(translation_max or 0.0):.3f} mm; rotation delta mean="
            f"{float(rotation_mean or 0.0):.3f} deg, max={float(rotation_max or 0.0):.3f} deg."
        )

    @staticmethod
    def _build_trust_detail_lines(
        *,
        payload: dict[str, object],
        validation_metrics: dict[str, object],
        history_summary: dict[str, object],
        tracking_summary: dict[str, object],
    ) -> list[str]:
        lines = [
            f"Accepted registration file: {payload.get('timestamp_utc', 'unknown timestamp')}",
        ]
        fre = validation_metrics.get("overall_fre_mm")
        max_residual = validation_metrics.get("max_residual_mm")
        if fre is not None:
            lines.append(f"FRE / RMSE: {float(fre):.3f} mm")
        if max_residual is not None:
            lines.append(f"Max landmark residual: {float(max_residual):.3f} mm")
        residual_summary = dict(validation_metrics.get("residual_summary", {}) or {})
        worst_label = residual_summary.get("worst_landmark_label") or validation_metrics.get("worst_landmark_label")
        worst_residual = residual_summary.get("worst_landmark_residual_mm") or validation_metrics.get(
            "worst_landmark_residual_mm"
        )
        if worst_label and worst_residual is not None:
            lines.append(f"Worst landmark: {worst_label} = {float(worst_residual):.3f} mm")
        truth_geometry = dict(validation_metrics.get("truth_geometry", {}) or {})
        if truth_geometry:
            rank = truth_geometry.get("geometry_rank")
            min_pairwise = truth_geometry.get("min_pairwise_distance_mm")
            condition_number = truth_geometry.get("condition_number")
            if rank is not None:
                lines.append(f"Landmark geometry rank: {int(rank)}")
            if min_pairwise is not None:
                lines.append(f"Minimum landmark spacing: {float(min_pairwise):.3f} mm")
            if condition_number is not None:
                lines.append(f"Landmark conditioning: {float(condition_number):.3f}")
        lines.append(str(tracking_summary.get("message") or ""))
        runtime_tip_timestamp = tracking_summary.get("stored_runtime_tip_timestamp_utc")
        if runtime_tip_timestamp is not None:
            lines.append(f"Loaded runtime tip calibration: {runtime_tip_timestamp}")
        elif tracking_summary.get("runtime_tip_calibration_state"):
            lines.append(
                "Runtime tip calibration state: "
                f"{tracking_summary.get('runtime_tip_calibration_state')}"
            )
        if history_summary:
            lines.append(RegistrationService._comparison_message(history_summary))
            worst_history_label = history_summary.get("worst_landmark_by_mean_residual")
            worst_history_mean = history_summary.get("worst_landmark_mean_residual_mm")
            if worst_history_label and worst_history_mean is not None:
                lines.append(
                    f"Across recent runs, the worst average landmark was {worst_history_label} at {float(worst_history_mean):.3f} mm."
                )
        return [line for line in lines if line]

    def _measurement_point_transform_matrix(self) -> np.ndarray:
        if self._assets is not None:
            return np.asarray(self._assets.T_measurement_point, dtype=float)
        return np.asarray(self._simple_measurement_point_transform, dtype=float)

    def _build_capture_tip_provenance(self, measurement_status: dict[str, object]) -> dict[str, object]:
        matrix = np.asarray(measurement_status.get("transform_matrix") or self._measurement_point_transform_matrix().tolist(), dtype=float)
        return {
            "measurement_tool_id": self._config.measurement_tool_id,
            "source": measurement_status.get("source"),
            "path": measurement_status.get("path"),
            "file_mtime_ns": measurement_status.get("file_mtime_ns"),
            "file_sha256": measurement_status.get("file_sha256"),
            "tip_vector_mm": list(measurement_status.get("tip_vector_mm") or matrix[0:3, 3].astype(float).tolist()),
            "T_measurement_tool_tip": matrix.tolist(),
            "offset_applied_before_solving": bool(measurement_status.get("offset_applied_before_solving", True)),
            "applied_stage": "capture_before_point_averaging_and_registration_solve",
        }

    def _build_simple_live_pose_tip_transform(self) -> tuple[np.ndarray, dict[str, object]]:
        matrix = np.eye(4, dtype=float)
        payload = {
            "coil_tool_id": self._config.coil_tool_id,
            "source": "identity_assumption_simple_registration",
            "assumption": (
                "Simple 4-point registration calibrates T_robot_aurora from 0B pen-probe captures. "
                "It does not solve a separate 0A-to-tip offset, so T_coil_tip remains identity by design "
                "until the separate runtime 0A hat-based tip calibration is performed."
            ),
            "T_coil_tip": matrix.tolist(),
        }
        return matrix, payload

    @staticmethod
    def _sha256_file(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _recompute_health_locked(self) -> None:
        faults: list[str] = []
        if not self._simple_measurement_point_ready:
            faults.append("measurement_point_geometry_unavailable")
        if self._state.active:
            current_label = self._state.current_label
            if current_label is not None and self._state.captured_counts.get(current_label, 0) < self._state.captures_per_landmark:
                faults.append("captures_incomplete")
        if self._state.health.last_error:
            faults.append("last_error")
        if self._state.pending_accept:
            if self._config.max_fre_mm is not None and self._state.fre_mm is not None and self._state.fre_mm > self._config.max_fre_mm:
                faults.append("fre_above_limit")

        if self._state.health.state == "accepted":
            health = HEALTH_HEALTHY
        elif self._state.health.last_error:
            health = HEALTH_FAILED
        elif faults:
            health = HEALTH_DEGRADED
        else:
            health = HEALTH_HEALTHY

        self._state.health.health = health
        self._state.health.current_config_source = self.config_source
        self._state.health.details = {
            "labels": list(self._state.labels),
            "captures_per_landmark": self._state.captures_per_landmark,
            "current_label": self._state.current_label,
            "pending_accept": self._state.pending_accept,
            "fre_mm": self._state.fre_mm,
            "max_residual_mm": self._state.validation_metrics.get("max_residual_mm"),
            "worst_landmark_label": self._state.validation_metrics.get("worst_landmark_label"),
            "latest_accepted_path": self._state.latest_accepted_path,
            "latest_validation_comparison_count": (
                self._state.latest_validation_summary.get("comparison_count")
                if isinstance(self._state.latest_validation_summary, dict)
                else None
            ),
            "registration_mode": "legacy_compatible" if self._assets is not None else "simple",
            "measurement_point_source": self._simple_measurement_point_source,
            "measurement_point_ready": self._simple_measurement_point_ready,
            "measurement_point_path": self._simple_measurement_point_path,
            "measurement_point_error": self._simple_measurement_point_error,
        }

        if self._state.health.state == "accepted":
            if self._state.fre_mm is not None:
                max_residual = self._state.validation_metrics.get("max_residual_mm")
                if max_residual is not None:
                    self._state.health.status = (
                        f"Registration accepted with FRE {self._state.fre_mm:.3f} mm "
                        f"and max residual {float(max_residual):.3f} mm"
                    )
                else:
                    self._state.health.status = f"Registration accepted with FRE {self._state.fre_mm:.3f} mm"
            else:
                self._state.health.status = "Registration accepted and persisted"
        elif self._state.health.state == "ready_to_solve":
            self._state.health.status = "Registration captures complete; ready to solve"
        elif self._state.health.state == "solved":
            if "fre_above_limit" in faults:
                self._state.health.status = f"Registration solved with FRE {self._state.fre_mm:.3f} mm above limit"
            else:
                self._state.health.status = f"Registration solved with FRE {self._state.fre_mm:.3f} mm"
        elif self._state.active:
            self._state.health.status = (
                f"Registration session capturing measurement tool {self._config.measurement_tool_id} "
                f"and coil tool {self._config.coil_tool_id}"
            )
        elif not self._simple_measurement_point_ready:
            self._state.health.status = self._simple_measurement_point_error or "Registration tip geometry is not ready"
        else:
            self._state.health.status = "Registration service ready"
