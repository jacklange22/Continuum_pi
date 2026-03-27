"""Registration tab controller for live tracker-backed workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

from continuum_robot.config.schemas import RegistrationWorkflowConfig
from continuum_robot.registration.live_registration_service import (
    LiveRegistrationService,
    RegistrationResult,
)


@dataclass
class RegistrationViewState:
    """UI-facing registration workflow state."""

    active: bool = False
    capture_tool_id: str = "0B"
    landmark_labels: list[str] = field(default_factory=list)
    current_label: str | None = None
    captures_per_landmark: int = 0
    captured_counts: dict[str, int] = field(default_factory=dict)
    latest_sample_by_label: dict[str, list[float]] = field(default_factory=dict)
    last_error: str | None = None
    last_result_path: str | None = None
    fre_mm: float | None = None
    residuals_by_label: dict[str, list[float]] = field(default_factory=dict)
    capture_geometry_status: str = "coil origin"
    status_message: str = "Registration idle."


class RegistrationController:
    """Owns guided landmark capture and registration actions."""

    def __init__(
        self,
        live_registration: LiveRegistrationService,
        registration_config: RegistrationWorkflowConfig,
    ) -> None:
        self.live_registration = live_registration
        self.config = registration_config
        self.state = RegistrationViewState(
            capture_tool_id=registration_config.capture_tool_id,
            landmark_labels=list(registration_config.landmark_labels),
            captures_per_landmark=registration_config.captures_per_landmark,
            captured_counts={label: 0 for label in registration_config.landmark_labels},
            current_label=registration_config.landmark_labels[0] if registration_config.landmark_labels else None,
            capture_geometry_status=self._capture_geometry_status(registration_config),
        )

    def begin_session(self, capture_tool_id: str | None = None) -> None:
        self.live_registration.begin_session(
            labels=self.config.landmark_labels,
            captures_per_landmark=self.config.captures_per_landmark,
            nominal_landmarks_robot_xyz_mm=self.config.nominal_landmarks_robot_xyz_mm,
            capture_tool_id=capture_tool_id or self.config.capture_tool_id,
            capture_tool_tip_transform=self.config.capture_tool_tip_transform,
        )
        self.state.active = True
        self.state.last_error = None
        self.state.fre_mm = None
        self.state.residuals_by_label = {}
        self.state.latest_sample_by_label = {}
        self.state.captured_counts = {label: 0 for label in self.config.landmark_labels}
        self.state.current_label = self.config.landmark_labels[0] if self.config.landmark_labels else None
        self.state.capture_tool_id = capture_tool_id or self.config.capture_tool_id
        self.state.capture_geometry_status = self._capture_geometry_status(self.config)
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
            sample = self.live_registration.capture_current_sample(label)
            self.state.last_error = None
            self.state.latest_sample_by_label[label] = sample
            self.state.captured_counts[label] = self.state.captured_counts.get(label, 0) + 1
            self.state.current_label = self._next_incomplete_label()
            self.state.status_message = f"Captured sample for {label}."
            return sample
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Capture failed: {exc}"
            raise

    def finish_session(self) -> RegistrationResult:
        try:
            result = self.live_registration.complete_registration(
                config_used={
                    "capture_tool_id": self.live_registration.capture_tool_id,
                    "landmark_labels": self.config.landmark_labels,
                    "captures_per_landmark": self.config.captures_per_landmark,
                    "capture_tool_tip_transform_configured": self.config.capture_tool_tip_transform is not None,
                },
                max_fre_mm=self.config.max_fre_mm,
            )
            self.state.active = False
            self.state.last_result_path = str(result.output_path)
            self.state.last_error = None
            self.state.fre_mm = result.record.fre_mm
            self.state.residuals_by_label = result.record.residuals_robot_xyz_mm
            self.state.status_message = f"Registration saved to {result.output_path.name}."
            return result
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Registration failed: {exc}"
            raise

    def retry_session(self) -> None:
        self.begin_session(self.state.capture_tool_id)
        self.state.status_message = "Registration restarted."

    def load_latest_result(self) -> None:
        latest = self.live_registration.repository.root_dir / "latest_registration.json"
        if latest.exists():
            self.state.last_result_path = str(latest)

    def is_ready_to_finish(self) -> bool:
        return all(
            self.state.captured_counts.get(label, 0) >= self.state.captures_per_landmark
            for label in self.state.landmark_labels
        )

    def _next_incomplete_label(self) -> str | None:
        for label in self.state.landmark_labels:
            if self.state.captured_counts.get(label, 0) < self.state.captures_per_landmark:
                return label
        return None

    @staticmethod
    def _capture_geometry_status(config: RegistrationWorkflowConfig) -> str:
        if config.capture_tool_tip_transform is None:
            return "coil origin / no explicit tip offset"
        return "tip transform applied"
