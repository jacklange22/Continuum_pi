"""Reusable registration service built on top of the tracking service."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import json
from pathlib import Path
import threading

import numpy as np
import yaml

from continuum_robot.registration.repository import RegistrationRecord, RegistrationRepository
from continuum_robot.registration.rigid_solver import RigidRegistrationSolver
from continuum_robot.registration.validation import compute_fre_mm
from continuum_robot.services.models import (
    HEALTH_DEGRADED,
    HEALTH_FAILED,
    HEALTH_HEALTHY,
    RegistrationSnapshot,
    ServiceHealthSnapshot,
)
from continuum_robot.services.tracking_service import TrackingService
from continuum_robot.utils.time_utils import utc_now_iso


@dataclass
class RegistrationConfig:
    """Registration workflow settings loaded from YAML."""

    labels: list[str]
    captures_per_landmark: int
    nominal_landmarks_robot_xyz_mm: dict[str, list[float]]
    max_fre_mm: float | None


class RegistrationService:
    """Own registration session state, solving, acceptance, and persistence."""

    CAPTURE_TOOL_ID = "0B"

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
        self._pending_record: RegistrationRecord | None = None

        self._state = RegistrationSnapshot(
            health=ServiceHealthSnapshot(
                name="registration_service",
                health=HEALTH_DEGRADED,
                state="idle",
                status="Registration service ready",
                current_config_source=config_source,
            ),
            active=False,
            capture_tool_id=self.CAPTURE_TOOL_ID,
            labels=list(self._config.labels),
            captures_per_landmark=self._config.captures_per_landmark,
            config_path=str(config_path),
            nominal_landmarks_robot_xyz_mm=copy.deepcopy(self._config.nominal_landmarks_robot_xyz_mm),
        )
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
        labels = list(labels or self._config.labels)
        captures_per_landmark = int(captures_per_landmark or self._config.captures_per_landmark)
        nominal = copy.deepcopy(nominal_landmarks_robot_xyz_mm or self._config.nominal_landmarks_robot_xyz_mm)

        with self._lock:
            self._pending_record = None
            self._state.active = True
            self._state.labels = labels
            self._state.captures_per_landmark = captures_per_landmark
            self._state.current_landmark_index = 0
            self._state.current_label = labels[0] if labels else None
            self._state.raw_points_by_label = {label: [] for label in labels}
            self._state.averaged_points_by_label = {}
            self._state.captured_counts = {label: 0 for label in labels}
            self._state.residuals_by_label = {}
            self._state.fre_mm = None
            self._state.last_sample_xyz_mm = None
            self._state.pending_accept = False
            self._state.nominal_landmarks_robot_xyz_mm = nominal
            self._state.pending_record = None
            self._state.health.last_error = None
            self._state.health.last_successful_update_utc = utc_now_iso()
            self._state.health.state = "capturing"
            self._recompute_health_locked()
            return copy.deepcopy(self._state)

    def capture_sample(self, label: str | None = None) -> list[float]:
        """Capture one sample from tool 0B for the requested or current landmark."""
        with self._lock:
            if not self._state.active:
                raise RuntimeError("Registration session has not been started")
            target_label = label or self._state.current_label
            if target_label is None:
                raise RuntimeError("No active landmark is selected")
            if target_label not in self._state.raw_points_by_label:
                raise ValueError(f"Unknown registration label: {target_label}")

        tool = self.tracking_service.get_latest_tool(self.CAPTURE_TOOL_ID)
        if tool is None or not tool.present or not tool.valid or tool.translation_mm is None:
            raise RuntimeError("Tool 0B is not currently tracked")

        sample = [float(tool.translation_mm[0]), float(tool.translation_mm[1]), float(tool.translation_mm[2])]
        with self._lock:
            self._state.raw_points_by_label[target_label].append(sample)
            self._state.captured_counts[target_label] = len(self._state.raw_points_by_label[target_label])
            self._state.last_sample_xyz_mm = sample
            self._state.health.last_error = None
            self._state.health.last_successful_update_utc = utc_now_iso()
            self._state.health.state = "capturing"
            self._recompute_health_locked()
        return sample

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
        """Solve rigid registration and keep the result pending until acceptance."""
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

        averaged = {
            label: np.asarray(points, dtype=float).mean(axis=0).tolist()
            for label, points in captures.items()
        }
        measured = np.asarray([averaged[label] for label in labels], dtype=float)
        truth = np.asarray([nominal[label] for label in labels], dtype=float)
        T_robot_aurora = self.solver.solve_T_robot_aurora(measured, truth)
        transformed = (T_robot_aurora[0:3, 0:3] @ measured.T).T + T_robot_aurora[0:3, 3]
        residuals = truth - transformed
        residuals_by_label = {
            label: residuals[idx, :].tolist()
            for idx, label in enumerate(labels)
        }
        fre_mm = float(compute_fre_mm(list(residuals_by_label.values())))
        T_coil_tip, tip_calibration_source = self._load_tip_calibration()

        record = RegistrationRecord(
            timestamp_utc=utc_now_iso(),
            landmark_labels=labels,
            raw_captured_landmarks_robot_xyz=captures,
            averaged_landmarks_robot_xyz=averaged,
            residuals_robot_xyz_mm=residuals_by_label,
            fre_mm=fre_mm,
            T_robot_aurora=T_robot_aurora.tolist(),
            T_coil_tip=T_coil_tip.tolist(),
            config_used={
                "registration_config_path": str(self.config_path),
                "capture_tool_id": self.CAPTURE_TOOL_ID,
                "max_fre_mm": self._config.max_fre_mm,
                "tip_calibration_source": tip_calibration_source,
            },
        )

        with self._lock:
            self._pending_record = record
            self._state.averaged_points_by_label = averaged
            self._state.residuals_by_label = residuals_by_label
            self._state.fre_mm = fre_mm
            self._state.pending_accept = True
            self._state.pending_record = asdict(record)
            self._state.health.last_error = None
            self._state.health.last_successful_update_utc = utc_now_iso()
            self._state.health.state = "solved"
            self._recompute_health_locked()
            return copy.deepcopy(self._state.pending_record)

    def accept_registration(self) -> Path:
        """Persist the pending registration as the latest accepted registration."""
        with self._lock:
            if self._pending_record is None:
                raise RuntimeError("No solved registration is pending acceptance")
            record = self._pending_record

        output_path = self.repository.save_record(record)
        self.tracking_service.refresh_registration()

        with self._lock:
            self._state.accepted_output_path = str(output_path)
            self._state.latest_accepted_path = str(self.repository.root_dir / "latest_registration.json")
            self._state.pending_accept = False
            self._state.active = False
            self._pending_record = None
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
            return None
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
        with self._lock:
            self._state.accepted_output_path = str(latest_path)
            self._state.health.last_successful_update_utc = payload.get("timestamp_utc")
            self._recompute_health_locked()
        return payload

    def get_snapshot(self) -> RegistrationSnapshot:
        """Return a deep copy of the current registration state."""
        with self._lock:
            return copy.deepcopy(self._state)

    @staticmethod
    def _load_config(path: Path) -> RegistrationConfig:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
        payload = payload or {}
        labels = list(payload.get("landmark_labels", ["L1", "L2", "L3", "L4"]))
        captures = int(payload.get("captures_per_landmark", 5))
        nominal = dict(payload.get("nominal_landmarks_robot_xyz_mm", {}))
        validation = payload.get("validation", {})
        max_fre = validation.get("max_fre_mm")
        return RegistrationConfig(
            labels=labels,
            captures_per_landmark=captures,
            nominal_landmarks_robot_xyz_mm=nominal,
            max_fre_mm=float(max_fre) if max_fre is not None else None,
        )

    def _load_tip_calibration(self) -> tuple[np.ndarray, str]:
        latest_path = self.repository.root_dir / "latest_registration.json"
        if not latest_path.exists():
            return np.eye(4), "default_identity"

        try:
            payload = json.loads(latest_path.read_text(encoding="utf-8"))
            config_used = payload.get("config_used", {}) if isinstance(payload.get("config_used"), dict) else {}
            if "T_coil_tip" in payload:
                return np.asarray(payload["T_coil_tip"], dtype=float), str(
                    config_used.get("tip_calibration_source", "previous_registration")
                )
            if "T_tip_2_coil" in payload:
                return np.linalg.inv(np.asarray(payload["T_tip_2_coil"], dtype=float)), str(
                    config_used.get("tip_calibration_source", "legacy_registration")
                )
        except Exception:
            return np.eye(4), "default_identity"

        return np.eye(4), "default_identity"

    def _recompute_health_locked(self) -> None:
        faults: list[str] = []
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
            "latest_accepted_path": self._state.latest_accepted_path,
        }

        if self._state.health.state == "accepted":
            self._state.health.status = "Registration accepted and persisted"
        elif self._state.health.state == "ready_to_solve":
            self._state.health.status = "Registration captures complete; ready to solve"
        elif self._state.health.state == "solved":
            if "fre_above_limit" in faults:
                self._state.health.status = f"Registration solved with FRE {self._state.fre_mm:.3f} mm above limit"
            else:
                self._state.health.status = f"Registration solved with FRE {self._state.fre_mm:.3f} mm"
        elif self._state.active:
            self._state.health.status = "Registration session capturing tool 0B landmarks"
        else:
            self._state.health.status = "Registration service ready"
