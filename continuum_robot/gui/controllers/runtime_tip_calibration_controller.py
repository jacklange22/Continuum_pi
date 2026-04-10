"""Controller for the advanced 0A runtime tip calibration dialog."""

from __future__ import annotations

from pathlib import Path

from continuum_robot.services.models import RuntimeTipCalibrationSnapshot


class RuntimeTipCalibrationController:
    """Wrap the runtime tip calibration service for the advanced operator workflow."""

    def __init__(self, runtime_tip_calibration_service) -> None:
        self.runtime_tip_calibration_service = runtime_tip_calibration_service
        self.state = self.runtime_tip_calibration_service.get_snapshot()

    def refresh(self) -> RuntimeTipCalibrationSnapshot:
        self.state = self.runtime_tip_calibration_service.get_snapshot()
        return self.state

    def begin_session(self, *, captures_per_landmark: int | None = None) -> None:
        self.state = self.runtime_tip_calibration_service.begin_session(
            captures_per_landmark=captures_per_landmark
        )

    def capture_current_label_sample(self) -> list[float]:
        point = self.runtime_tip_calibration_service.capture_sample(self.state.current_label)
        self.state = self.runtime_tip_calibration_service.get_snapshot()
        return point

    def complete_current_label(self) -> None:
        self.runtime_tip_calibration_service.complete_landmark()
        self.state = self.runtime_tip_calibration_service.get_snapshot()

    def collect_coil_samples(self, *, sample_count: int, sample_interval_s: float) -> None:
        self.state = self.runtime_tip_calibration_service.collect_coil_samples(
            sample_count=sample_count,
            sample_interval_s=sample_interval_s,
        )

    def solve_calibration(self) -> dict:
        payload = self.runtime_tip_calibration_service.solve_calibration()
        self.state = self.runtime_tip_calibration_service.get_snapshot()
        return payload

    def save_calibration(self) -> Path:
        path = self.runtime_tip_calibration_service.accept_calibration()
        self.state = self.runtime_tip_calibration_service.get_snapshot()
        return path

    def load_latest_result(self) -> None:
        self.runtime_tip_calibration_service.load_latest_accepted()
        self.state = self.runtime_tip_calibration_service.get_snapshot()

    def retry_session(self, *, captures_per_landmark: int | None = None) -> None:
        self.begin_session(captures_per_landmark=captures_per_landmark)

    def can_begin_session(self) -> bool:
        return not self.state.active and not self.state.pending_accept

    def can_capture(self) -> bool:
        return self.state.active and self.state.current_label is not None

    def can_complete_current(self) -> bool:
        if self.state.current_label is None:
            return False
        return self.state.captured_counts.get(self.state.current_label, 0) >= self.state.captures_per_landmark

    def can_collect_coil_samples(self) -> bool:
        return self.state.active and self.state.current_label is None

    def can_solve(self) -> bool:
        return (
            self.state.active
            and self.state.current_label is None
            and self.state.coil_samples_captured > 0
        )

    def can_save(self) -> bool:
        return bool(self.state.pending_accept)
