"""Registration tab controller for live tracker-backed workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml

from continuum_robot.registration.live_registration_service import (
    LiveRegistrationService,
    RegistrationResult,
)


@dataclass
class RegistrationConfig:
    """Registration workflow settings loaded from YAML."""

    landmark_labels: list[str]
    captures_per_landmark: int
    nominal_landmarks_robot_xyz_mm: dict[str, list[float]]
    max_fre_mm: float | None


@dataclass
class RegistrationViewState:
    """UI-facing registration workflow state."""

    active: bool = False
    last_error: str | None = None
    last_result_path: str | None = None


class RegistrationController:
    """Owns guided landmark capture and registration actions."""

    def __init__(
        self,
        live_registration: LiveRegistrationService,
        registration_config_path: Path,
    ) -> None:
        self.live_registration = live_registration
        self.registration_config_path = registration_config_path
        self.state = RegistrationViewState()
        self.config = self._load_registration_config(registration_config_path)

    @staticmethod
    def _load_registration_config(path: Path) -> RegistrationConfig:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        labels = payload.get("landmark_labels", [])
        captures = int(payload.get("captures_per_landmark", 1))
        nominal = payload.get("nominal_landmarks_robot_xyz_mm", {})
        validation = payload.get("validation", {})
        max_fre = validation.get("max_fre_mm")
        return RegistrationConfig(
            landmark_labels=list(labels),
            captures_per_landmark=captures,
            nominal_landmarks_robot_xyz_mm=dict(nominal),
            max_fre_mm=float(max_fre) if max_fre is not None else None,
        )

    def begin_session(self, capture_tool_id: str = "0A") -> None:
        self.live_registration.begin_session(
            labels=self.config.landmark_labels,
            captures_per_landmark=self.config.captures_per_landmark,
            nominal_landmarks_robot_xyz_mm=self.config.nominal_landmarks_robot_xyz_mm,
            capture_tool_id=capture_tool_id,
        )
        self.state.active = True
        self.state.last_error = None

    def capture_label_sample(self, label: str) -> list[float]:
        try:
            sample = self.live_registration.capture_current_sample(label)
            self.state.last_error = None
            return sample
        except Exception as exc:
            self.state.last_error = str(exc)
            raise

    def finish_session(self) -> RegistrationResult:
        try:
            result = self.live_registration.complete_registration(
                config_used={
                    "registration_yaml": str(self.registration_config_path),
                    "capture_tool_id": self.live_registration.capture_tool_id,
                },
                max_fre_mm=self.config.max_fre_mm,
            )
            self.state.active = False
            self.state.last_result_path = str(result.output_path)
            self.state.last_error = None
            return result
        except Exception as exc:
            self.state.last_error = str(exc)
            raise
