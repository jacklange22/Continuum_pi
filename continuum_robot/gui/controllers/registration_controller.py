"""Registration tab controller backed by the shared registration service."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from continuum_robot.config.schemas import RegistrationWorkflowConfig


@dataclass
class RegistrationViewState:
    """UI-facing registration workflow state."""

    active: bool = False
    capture_tool_id: str = "0B"
    coil_tool_id: str = "0A"
    landmark_labels: list[str] = field(default_factory=list)
    current_label: str | None = None
    captures_per_landmark: int = 0
    captured_counts: dict[str, int] = field(default_factory=dict)
    raw_samples_by_label: dict[str, list[list[float]]] = field(default_factory=dict)
    latest_sample_by_label: dict[str, list[float]] = field(default_factory=dict)
    truth_points_in_sw_by_label: dict[str, list[float]] = field(default_factory=dict)
    last_error: str | None = None
    last_result_path: str | None = None
    pending_accept: bool = False
    overwrite_target_path: str | None = None
    overwrite_required: bool = False
    fre_mm: float | None = None
    residuals_by_label: dict[str, list[float]] = field(default_factory=dict)
    capture_geometry_status: str = "coil origin"
    current_tracked_xyz_mm: list[float] | None = None
    current_tracked_frame_id: int | None = None
    current_tracking_status: str = "waiting_for_tracking"
    completed_labels: list[str] = field(default_factory=list)
    accepted_registration_valid: bool = False
    status_message: str = "Registration idle."


@dataclass
class RegistrationActionResult:
    """Small controller return payload after one registration save."""

    output_path: Path
    payload: dict


class RegistrationController:
    """Owns guided 4-point capture, solve, and save actions."""

    def __init__(
        self,
        registration_service,
        registration_config: RegistrationWorkflowConfig,
    ) -> None:
        self.registration_service = registration_service
        self.config = registration_config
        initial = self.registration_service.get_snapshot()
        self.state = RegistrationViewState(
            capture_tool_id=initial.capture_tool_id,
            coil_tool_id=registration_config.coil_tool_id,
            landmark_labels=list(initial.labels),
            captures_per_landmark=initial.captures_per_landmark,
            captured_counts={label: 0 for label in initial.labels},
            current_label=initial.current_label,
            capture_geometry_status=self._capture_geometry_status(registration_config),
        )
        self._apply_snapshot(initial)
        self.refresh()

    def refresh(self) -> RegistrationViewState:
        snapshot = self.registration_service.get_snapshot()
        self._apply_snapshot(snapshot)
        live_point = self.registration_service.peek_current_measurement_point()
        self.state.current_tracked_xyz_mm = live_point.get("point_xyz_mm")
        self.state.current_tracked_frame_id = live_point.get("frame_number")
        self.state.current_tracking_status = str(live_point.get("status", "unknown"))
        return self.state

    def begin_session(self, capture_tool_id: str | None = None) -> None:
        if capture_tool_id is not None and capture_tool_id != self.config.capture_tool_id:
            raise RuntimeError(
                f"Registration controller is configured for capture tool {self.config.capture_tool_id}; "
                f"override {capture_tool_id} is not supported from the GUI controller"
            )
        snapshot = self.registration_service.begin_session()
        self._apply_snapshot(snapshot)
        self.state.status_message = (
            "4-point registration started. Capture one or more samples for the current point, "
            "then mark the point complete."
        )
        self.refresh()

    def capture_current_label_sample(self) -> list[float]:
        if self.state.current_label is None:
            raise RuntimeError("No current registration point is selected")
        return self.capture_label_sample(self.state.current_label)

    def capture_label_sample(self, label: str) -> list[float]:
        try:
            sample = self.registration_service.capture_sample(label)
            snapshot = self.registration_service.get_snapshot()
            self._apply_snapshot(snapshot)
            self.state.last_error = None
            count = snapshot.captured_counts.get(label, 0)
            self.state.status_message = (
                f"Captured sample {count} for {label}. "
                "Capture more if needed, then mark the point complete."
            )
            self.refresh()
            return sample
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Capture failed: {exc}"
            raise

    def complete_current_label(self) -> None:
        try:
            previous = self.state.current_label
            next_label = self.registration_service.complete_landmark()
            self._apply_snapshot(self.registration_service.get_snapshot())
            if next_label is None:
                self.state.status_message = (
                    f"{previous} marked complete. All four points are ready. Solve the registration next."
                )
            else:
                self.state.status_message = f"{previous} marked complete. Continue with {next_label}."
            self.refresh()
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Could not complete point: {exc}"
            raise

    def solve_session(self) -> dict:
        try:
            payload = self.registration_service.solve_registration()
            self._apply_snapshot(self.registration_service.get_snapshot())
            samples_used = self.total_samples_captured()
            self.state.status_message = (
                f"Registration solved. RMSE/FRE = {self.state.fre_mm:.3f} mm using {samples_used} captured samples. "
                "Review the result, then save the accepted registration."
            )
            self.refresh()
            return payload
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Registration solve failed: {exc}"
            raise

    def save_registration(self, *, confirm_overwrite: bool = False) -> RegistrationActionResult:
        overwrite_target = self._overwrite_target_path()
        if overwrite_target.exists() and not confirm_overwrite:
            self.refresh()
            self.state.status_message = (
                f"Saving will overwrite {overwrite_target.name}. Confirm overwrite to continue."
            )
            raise RuntimeError("Registration save requires explicit overwrite confirmation.")
        try:
            payload = self.registration_service.get_snapshot().pending_record or {}
            output_path = self.registration_service.accept_registration()
            snapshot = self.registration_service.get_snapshot()
            self._apply_snapshot(snapshot)
            self.state.status_message = f"Registration saved to {output_path.name}."
            self.refresh()
            return RegistrationActionResult(output_path=output_path, payload=payload)
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Registration save failed: {exc}"
            raise

    def retry_session(self) -> None:
        snapshot = self.registration_service.retry_session()
        self._apply_snapshot(snapshot)
        self.state.status_message = "Registration restarted. Begin capturing L1 again."
        self.refresh()

    def load_latest_result(self) -> None:
        payload = self.registration_service.load_latest_accepted()
        if payload is not None:
            self._apply_snapshot(self.registration_service.get_snapshot())
            self.state.status_message = "Loaded the latest accepted registration."
        self.refresh()

    def is_ready_to_complete_current(self) -> bool:
        if self.state.current_label is None:
            return False
        return self.state.captured_counts.get(self.state.current_label, 0) >= self.state.captures_per_landmark

    def is_ready_to_solve(self) -> bool:
        return all(
            self.state.captured_counts.get(label, 0) >= self.state.captures_per_landmark
            for label in self.state.landmark_labels
        )

    def total_samples_captured(self) -> int:
        return sum(len(samples) for samples in self.state.raw_samples_by_label.values())

    def _overwrite_target_path(self) -> Path:
        return self.registration_service.repository.root_dir / "latest_registration.json"

    def _apply_snapshot(self, snapshot) -> None:
        self.state.active = snapshot.active
        self.state.capture_tool_id = snapshot.capture_tool_id
        self.state.coil_tool_id = self.config.coil_tool_id
        self.state.landmark_labels = list(snapshot.labels)
        self.state.current_label = snapshot.current_label
        self.state.captures_per_landmark = snapshot.captures_per_landmark
        self.state.captured_counts = dict(snapshot.captured_counts)
        self.state.raw_samples_by_label = {
            label: [list(sample) for sample in samples]
            for label, samples in snapshot.raw_points_by_label.items()
        }
        self.state.latest_sample_by_label = {
            label: list(samples[-1])
            for label, samples in self.state.raw_samples_by_label.items()
            if samples
        }
        self.state.truth_points_in_sw_by_label = dict(snapshot.nominal_landmarks_robot_xyz_mm)
        self.state.fre_mm = snapshot.fre_mm
        self.state.residuals_by_label = dict(snapshot.residuals_by_label)
        self.state.pending_accept = bool(snapshot.pending_accept)
        self.state.last_result_path = snapshot.accepted_output_path or snapshot.latest_accepted_path
        self.state.last_error = snapshot.health.last_error
        self.state.accepted_registration_valid = bool(snapshot.latest_accepted_path)
        overwrite_target = self._overwrite_target_path()
        self.state.overwrite_target_path = str(overwrite_target)
        self.state.overwrite_required = bool(snapshot.pending_accept and overwrite_target.exists())
        self.state.completed_labels = self._completed_labels(snapshot)

    @staticmethod
    def _completed_labels(snapshot) -> list[str]:
        completed: list[str] = []
        labels = list(snapshot.labels)
        for index, label in enumerate(labels):
            count = snapshot.captured_counts.get(label, 0)
            if count < snapshot.captures_per_landmark:
                continue
            if snapshot.current_label is None or index < snapshot.current_landmark_index:
                completed.append(label)
        return completed

    @staticmethod
    def _capture_geometry_status(config: RegistrationWorkflowConfig) -> str:
        if config.capture_tool_tip_transform is not None:
            return "Explicit 4x4 pen-tip transform"
        if config.penprobe_file:
            return "Pen-probe tip file"
        return "Coil origin / no explicit tip offset"
