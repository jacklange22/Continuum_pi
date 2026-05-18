"""Test the Registration tab's 'Run Registration Sampling Study' button.

Kept narrow: construct a RegistrationTab inside a QApplication and verify
the button (a) is wired only when an opener callback is supplied, and (b)
invokes the callback when clicked.
"""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.xfail(
        reason="RegistrationTab does not yet accept open_registration_sampling_study / expose sampling_study_button; GUI wiring is pending alongside the registration_sampling_study backend.",
        strict=False,
    ),
]

pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtWidgets import QApplication

from continuum_robot.gui.controllers.registration_controller import RegistrationController
from continuum_robot.gui.tabs.registration_tab import RegistrationTab


class _StubRegistrationController:
    """Minimal stub matching the controller surface the tab consumes."""

    REQUIRED_SELECTION_COUNT = 4

    def __init__(self) -> None:
        # Minimum fields the tab reads in __init__ and update().
        from dataclasses import dataclass, field

        @dataclass
        class _State:
            landmark_labels: list[str] = field(default_factory=list)
            available_landmarks: list[Any] = field(default_factory=list)
            selected_model_points: list[str] = field(default_factory=list)
            status_message: str = ""
            last_error: str = ""
            captures_per_landmark: int = 5
            session_active: bool = False
            session_status: str = "Idle"
            tool_id: str = "0B"
            coil_tool_id: str = "0A"
            tip_geometry_summary: str = ""
            current_label: str | None = None
            live_point_xyz: tuple[float, float, float] | None = None
            samples_used: int = 0
            fre_mm: float | None = None
            max_residual_mm: float | None = None
            result_status: str = ""
            result_path: str = ""
            tip_calibration_path: str = ""
            tip_calibration_status: str = "missing"
            workflow_gate_message: str = ""
            workflow_gate_passed: bool = False
            workflow_dependency_text: str = ""
            accepted_registration_path: str = ""
            accepted_registration_summary: str = ""
            live_pose_summary: str = "not ready"
            registration_label_points: dict[str, Any] = field(default_factory=dict)
            registration_residuals_mm: dict[str, Any] = field(default_factory=dict)
            registration_label_residual_summary: dict[str, Any] = field(default_factory=dict)
            registration_residual_summary_text: str = ""
            registration_solve_summary: str = ""
            session_progress_summary: str = ""
            session_progress_remaining: str = ""
            recent_runs: list[Any] = field(default_factory=list)
            trust_summary: str = ""
            live_chain_summary: str = ""
            comparison_summary: str = ""
            runtime_tip_mode: str = "latest_accepted"
            runtime_tip_trust: str = "missing"
            runtime_tip_mode_message: str = ""

        self.state = _State()

    def refresh(self):
        return self.state


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_registration_tab_sampling_button_always_visible(qapp) -> None:
    # The button must be visible regardless of whether a callback was wired,
    # so the operator can always find it from the Registration tab.
    tab_with_cb = RegistrationTab(
        _StubRegistrationController(),
        open_registration_sampling_study=lambda: None,
    )
    tab_without_cb = RegistrationTab(_StubRegistrationController())
    assert tab_with_cb.sampling_study_button.isHidden() is False
    assert tab_without_cb.sampling_study_button.isHidden() is False


def test_registration_tab_sampling_button_adjacent_to_runtime_tip_button(qapp) -> None:
    """Bug 2 placement: per operator request 2026-05-15, the launcher sits on
    the secondary button row, immediately after 'Open Runtime Tip Calibration'.
    """
    tab = RegistrationTab(
        _StubRegistrationController(),
        open_registration_sampling_study=lambda: None,
    )
    # Both buttons must share the same parent layout (the secondary button row).
    rtc_parent = tab.runtime_tip_button.parentWidget()
    sss_parent = tab.sampling_study_button.parentWidget()
    assert rtc_parent is sss_parent, (
        "sampling_study_button must live next to the runtime_tip_button so the "
        "operator can find it on the secondary action row."
    )


def test_registration_tab_sampling_button_invokes_callback(qapp) -> None:
    invocations: list[bool] = []

    def opener() -> None:
        invocations.append(True)

    tab = RegistrationTab(
        _StubRegistrationController(),
        open_registration_sampling_study=opener,
    )
    tab.sampling_study_button.click()
    assert invocations == [True]


def test_registration_tab_sampling_button_is_no_op_if_callback_missing(qapp) -> None:
    tab = RegistrationTab(_StubRegistrationController())
    # Clicking when no opener is wired must not raise.
    tab.sampling_study_button.click()
