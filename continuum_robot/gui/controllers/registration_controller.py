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
    latest_sample_by_label: dict[str, list[float]] = field(default_factory=dict)
    truth_points_in_sw_by_label: dict[str, list[float]] = field(default_factory=dict)
    group_by_label: dict[str, str] = field(default_factory=dict)
    last_error: str | None = None
    last_result_path: str | None = None
    fre_mm: float | None = None
    residuals_by_label: dict[str, list[float]] = field(default_factory=dict)
    capture_geometry_status: str = "coil origin"
    status_message: str = "Registration idle."


@dataclass
class RegistrationActionResult:
    """Small controller return payload after one registration save."""

    output_path: Path
    payload: dict


class RegistrationController:
    """Owns guided landmark capture and registration actions."""

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

    def begin_session(self, capture_tool_id: str | None = None) -> None:
        if capture_tool_id is not None and capture_tool_id != self.config.capture_tool_id:
            raise RuntimeError(
                f"Registration controller is configured for capture tool {self.config.capture_tool_id}; "
                f"override {capture_tool_id} is not supported from the GUI controller"
            )
        snapshot = self.registration_service.begin_session()
        self._apply_snapshot(snapshot)
        self.state.latest_sample_by_label = {}
        self.state.status_message = (
            "Registration session started. "
            f"Capture geometry: {self.state.capture_geometry_status}."
        )

    def capture_current_label_sample(self) -> list[float]:
        if self.state.current_label is None:
            raise RuntimeError("No current landmark label is selected")
        return self.capture_label_sample(self.state.current_label)

    def capture_label_sample(self, label: str) -> list[float]:
        try:
            sample = self.registration_service.capture_sample(label)
            snapshot = self.registration_service.get_snapshot()
            if snapshot.captured_counts.get(label, 0) >= snapshot.captures_per_landmark and snapshot.current_label == label:
                self.registration_service.complete_landmark()
                snapshot = self.registration_service.get_snapshot()
            self._apply_snapshot(snapshot)
            self.state.last_error = None
            self.state.latest_sample_by_label[label] = sample
            self.state.status_message = f"Captured sample for {label}."
            return sample
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Capture failed: {exc}"
            raise

    def finish_session(self) -> RegistrationActionResult:
        try:
            payload = self.registration_service.solve_registration()
            output_path = self.registration_service.accept_registration()
            snapshot = self.registration_service.get_snapshot()
            self._apply_snapshot(snapshot)
            self.state.status_message = f"Registration saved to {output_path.name}."
            return RegistrationActionResult(output_path=output_path, payload=payload)
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Registration failed: {exc}"
            raise

    def retry_session(self) -> None:
        snapshot = self.registration_service.retry_session()
        self._apply_snapshot(snapshot)
        self.state.latest_sample_by_label = {}
        self.state.status_message = "Registration restarted."

    def load_latest_result(self) -> None:
        payload = self.registration_service.load_latest_accepted()
        if payload is not None:
            self._apply_snapshot(self.registration_service.get_snapshot())

    def is_ready_to_finish(self) -> bool:
        return all(
            self.state.captured_counts.get(label, 0) >= self.state.captures_per_landmark
            for label in self.state.landmark_labels
        )

    def _apply_snapshot(self, snapshot) -> None:
        self.state.active = snapshot.active
        self.state.capture_tool_id = snapshot.capture_tool_id
        self.state.coil_tool_id = self.config.coil_tool_id
        self.state.landmark_labels = list(snapshot.labels)
        self.state.current_label = snapshot.current_label
        self.state.captures_per_landmark = snapshot.captures_per_landmark
        self.state.captured_counts = dict(snapshot.captured_counts)
        self.state.truth_points_in_sw_by_label = dict(snapshot.truth_points_in_sw_by_label)
        self.state.group_by_label = dict(snapshot.group_by_label)
        self.state.fre_mm = snapshot.fre_mm
        self.state.residuals_by_label = dict(snapshot.residuals_by_label)
        self.state.last_result_path = snapshot.accepted_output_path or snapshot.latest_accepted_path
        self.state.last_error = snapshot.health.last_error

    @staticmethod
    def _capture_geometry_status(config: RegistrationWorkflowConfig) -> str:
        if config.capture_tool_tip_transform is None:
            if config.penprobe_file:
                return "protected penprobe file"
            return "coil origin / no explicit tip offset"
        return "explicit 4x4 tip transform applied"
