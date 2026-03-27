"""Live registration workflow backed by tracker_bridge transforms."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from continuum_robot.registration.capture_session import RegistrationSession
from continuum_robot.registration.repository import RegistrationRecord, RegistrationRepository
from continuum_robot.registration.rigid_solver import RigidRegistrationSolver
from continuum_robot.registration.validation import compute_fre_mm
from continuum_robot.tracking.tracker_service_manager import TrackerServiceManager


@dataclass
class RegistrationResult:
    """Result payload after completing one registration run."""

    record: RegistrationRecord
    output_path: Path


class LiveRegistrationService:
    """Collects repeated landmark samples from tracker stream and solves registration."""

    def __init__(
        self,
        tracker_manager: TrackerServiceManager,
        repository: RegistrationRepository,
        solver: RigidRegistrationSolver,
    ) -> None:
        self.tracker_manager = tracker_manager
        self.repository = repository
        self.solver = solver
        self.session: RegistrationSession | None = None
        self.nominal_landmarks_robot_xyz_mm: dict[str, list[float]] = {}
        self.capture_tool_id = "0A"

    def begin_session(
        self,
        labels: list[str],
        captures_per_landmark: int,
        nominal_landmarks_robot_xyz_mm: dict[str, list[float]],
        capture_tool_id: str = "0A",
    ) -> RegistrationSession:
        self.session = RegistrationSession(
            labels=labels,
            captures_per_landmark=captures_per_landmark,
            raw_points_by_label={label: [] for label in labels},
        )
        self.nominal_landmarks_robot_xyz_mm = nominal_landmarks_robot_xyz_mm
        self.capture_tool_id = capture_tool_id
        return self.session

    def capture_current_sample(self, label: str) -> list[float]:
        """Capture current tracker sample for one landmark label."""
        if self.session is None:
            raise RuntimeError("Registration session has not been started")
        if label not in self.session.raw_points_by_label:
            raise ValueError(f"Unknown registration label: {label}")

        tool = self.tracker_manager.get_latest_tool(self.capture_tool_id)
        if tool is None:
            raise RuntimeError(f"No tracker sample available for tool {self.capture_tool_id}")
        if not tool.valid:
            raise RuntimeError(f"Latest sample for {self.capture_tool_id} is not valid ({tool.status})")

        sample = [float(tool.translation_mm[0]), float(tool.translation_mm[1]), float(tool.translation_mm[2])]
        self.session.raw_points_by_label[label].append(sample)
        return sample

    def complete_registration(self, config_used: dict, max_fre_mm: float | None = None) -> RegistrationResult:
        """Solve and persist registration using current session samples."""
        if self.session is None:
            raise RuntimeError("Registration session has not been started")

        labels = self.session.labels
        captures = self.session.raw_points_by_label

        for label in labels:
            points = captures.get(label, [])
            if len(points) < self.session.captures_per_landmark:
                raise RuntimeError(
                    f"Label {label} has {len(points)} capture(s); requires {self.session.captures_per_landmark}"
                )
            if label not in self.nominal_landmarks_robot_xyz_mm:
                raise RuntimeError(f"Missing nominal landmark for label {label}")

        averaged_measured_aurora: dict[str, list[float]] = {}
        for label in labels:
            points = np.asarray(captures[label], dtype=float)
            averaged_measured_aurora[label] = points.mean(axis=0).tolist()

        measured = np.asarray([averaged_measured_aurora[label] for label in labels], dtype=float)
        truth = np.asarray([self.nominal_landmarks_robot_xyz_mm[label] for label in labels], dtype=float)

        T_robot_aurora = self.solver.solve_T_robot_aurora(measured, truth)
        R = T_robot_aurora[0:3, 0:3]
        t = T_robot_aurora[0:3, 3]
        transformed = (R @ measured.T).T + t
        residuals = truth - transformed

        residuals_by_label = {
            label: residuals[idx, :].tolist()
            for idx, label in enumerate(labels)
        }
        fre_mm = compute_fre_mm(list(residuals_by_label.values()))
        if max_fre_mm is not None and fre_mm > max_fre_mm:
            raise RuntimeError(f"Registration FRE {fre_mm:.3f} mm exceeds limit {max_fre_mm:.3f} mm")

        T_coil_tip = self._load_existing_T_coil_tip()

        record = RegistrationRecord(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            landmark_labels=labels,
            raw_captured_landmarks_robot_xyz=captures,
            averaged_landmarks_robot_xyz=averaged_measured_aurora,
            residuals_robot_xyz_mm=residuals_by_label,
            fre_mm=float(fre_mm),
            T_robot_aurora=T_robot_aurora.tolist(),
            T_coil_tip=T_coil_tip.tolist(),
            config_used=config_used,
        )

        output_path = self.repository.save_record(record)
        return RegistrationResult(record=record, output_path=output_path)

    def _load_existing_T_coil_tip(self) -> np.ndarray:
        latest = self.repository.root_dir / "latest_registration.json"
        if not latest.exists():
            return np.eye(4)

        try:
            payload = json.loads(latest.read_text(encoding="utf-8"))
            if "T_coil_tip" in payload:
                return np.asarray(payload["T_coil_tip"], dtype=float)
            if "T_tip_2_coil" in payload:
                return np.linalg.inv(np.asarray(payload["T_tip_2_coil"], dtype=float))
        except Exception:
            return np.eye(4)

        return np.eye(4)
