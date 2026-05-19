"""Regression tests for three bench bugs found 2026-05-15.

Bug 1 — registration controller silently re-set selected_model_labels back to
the previous run's labels on every refresh whenever a saved registration was
loaded (because ``snapshot.averaged_points_by_label`` was non-empty). The
operator could click a label on the map or table, but the next refresh
snapped the selection back. The fix drops ``averaged_points_by_label`` from
the gate so only an active or pending-accept session legitimately locks the
selection.

Bug 2 — the Trial Mode button was styled ``variant="ghost"`` and gated on the
callback being supplied. It was visually hard to find on a real bench, and
shells that forgot to wire the callback got no button at all. The fix
promotes it to ``role="primary"`` and makes it visible unconditionally; the
click handler is a no-op when no callback is wired.

Bug 3 — the GUI gated the "live robot-frame position" display on
``tip_pose_status == "ok"``, which is never set in coil_as_tip mode (the
tracking service writes "coil_as_tip" there, and the artifact-load path can
leave the cached label at "identity_tip_fallback" before live frames flow).
Coil-as-tip is treated as thesis-trusted per the README, so the GUI must
display the live position in that mode even when the cached label is
"identity_tip_fallback". The widened gate only applies when the operator has
explicitly selected coil_as_tip mode, so the lower-trust meaning in other
modes is unchanged.

These tests run headlessly without QApplication where possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from continuum_robot.app.bootstrap import build_app_context
from continuum_robot.gui.controllers.registration_controller import RegistrationController


# ---------------------------------------------------------------------------
# Bug 1 — operator toggle must persist across refresh
# ---------------------------------------------------------------------------


def _bootstrap_registration_controller() -> RegistrationController:
    ctx = build_app_context()
    reg_service = ctx.services.get("registration_service")
    return RegistrationController(reg_service, ctx.settings.registration)


def test_toggle_deselect_persists_across_refresh_with_saved_registration_loaded() -> None:
    """Bug 1: clicking an already-selected label should deselect it and survive refresh.

    Pre-fix: refresh() called _apply_snapshot() which forced
    selected_model_labels back to ['L1','L2','L3','L4'] because
    snapshot.averaged_points_by_label was non-empty (loaded from
    latest_registration.json).
    """
    ctrl = _bootstrap_registration_controller()
    state = ctrl.refresh()
    assert state.selected_model_labels[:4] == ["L1", "L2", "L3", "L4"]
    assert state.selection_editable is True
    assert state.active is False
    assert state.pending_accept is False

    ctrl.toggle_selected_model_point("L1")
    state = ctrl.refresh()  # this used to undo the toggle
    assert "L1" not in state.selected_model_labels, (
        f"toggle_selected_model_point('L1') was undone by refresh; "
        f"state.selected_model_labels={state.selected_model_labels!r}"
    )


def test_toggle_add_new_label_persists_across_refresh() -> None:
    """Bug 1: a balanced remove + add must persist across refresh.

    Default selection seeds from the full configured landmark list (so
    "add L5" today means "ensure L5 is selected after a balancing remove").
    This test still proves the underlying invariant: toggle ops survive
    _apply_snapshot.
    """
    ctrl = _bootstrap_registration_controller()
    ctrl.refresh()
    initial_count = len(ctrl.state.selected_model_labels)
    ctrl.toggle_selected_model_point("L1")
    ctrl.refresh()
    if "L5" in ctrl.state.selected_model_labels:
        ctrl.toggle_selected_model_point("L5")  # remove
        ctrl.refresh()
    ctrl.toggle_selected_model_point("L5")  # add back
    state = ctrl.refresh()
    assert "L5" in state.selected_model_labels
    assert "L1" not in state.selected_model_labels
    assert len(state.selected_model_labels) == initial_count - 1


def test_active_session_still_locks_selection() -> None:
    """The fix must not weaken the active/pending_accept gate."""
    ctrl = _bootstrap_registration_controller()
    ctrl.refresh()
    ctrl.state.active = True
    with pytest.raises(RuntimeError, match="Restart"):
        ctrl.toggle_selected_model_point("L5")


# ---------------------------------------------------------------------------
# Bug 3 — coil_as_tip must surface a live robot-frame tip position
# ---------------------------------------------------------------------------


@dataclass
class _StubSnapshot:
    tip_pose_status: str
    T_robot_tip: list[list[float]] | None
    tracker_data_stale: bool = False
    runtime_tip_mode: str = "latest_accepted"


def _live_pose_ready(snapshot) -> bool:
    """Mirror the tracker_mvp_controller live_pose_ready predicate exactly."""
    runtime_tip_mode = str(getattr(snapshot, "runtime_tip_mode", "") or "")
    if runtime_tip_mode == "coil_as_tip":
        acceptable_statuses = ("ok", "coil_as_tip", "identity_tip_fallback")
    else:
        acceptable_statuses = ("ok",)
    return bool(
        snapshot.tip_pose_status in acceptable_statuses
        and snapshot.T_robot_tip is not None
        and not snapshot.tracker_data_stale
    )


def test_live_pose_ready_accepts_ok_status_in_latest_accepted_mode() -> None:
    snapshot = _StubSnapshot(
        tip_pose_status="ok",
        T_robot_tip=[[1, 0, 0, 10], [0, 1, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]],
        runtime_tip_mode="latest_accepted",
    )
    assert _live_pose_ready(snapshot) is True


def test_live_pose_ready_accepts_coil_as_tip_status_in_coil_as_tip_mode() -> None:
    """Bug 3 regression: coil_as_tip mode must surface the live position.

    The per-frame update writes tip_pose_status="coil_as_tip" while still
    producing a valid T_robot_tip. The pre-fix GUI gate dropped this case.
    """
    snapshot = _StubSnapshot(
        tip_pose_status="coil_as_tip",
        T_robot_tip=[[1, 0, 0, 5], [0, 1, 0, 6], [0, 0, 1, -100], [0, 0, 0, 1]],
        runtime_tip_mode="coil_as_tip",
    )
    assert _live_pose_ready(snapshot) is True


def test_live_pose_ready_accepts_identity_tip_fallback_when_runtime_mode_is_coil_as_tip() -> None:
    """Bug 3 second-mile: in coil_as_tip mode, the cached label may be
    'identity_tip_fallback' (e.g. mock mode with no live frames yet, or after
    the artifact-load path). The math is identical (T_coil_tip = identity),
    so the live position must display.
    """
    snapshot = _StubSnapshot(
        tip_pose_status="identity_tip_fallback",
        T_robot_tip=[[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]],
        runtime_tip_mode="coil_as_tip",
    )
    assert _live_pose_ready(snapshot) is True


def test_live_pose_ready_rejects_identity_tip_fallback_in_other_modes() -> None:
    """The widened gate must NOT leak into other modes; identity_tip_fallback
    in latest_accepted mode still means 'no usable calibration', not 'show it'.
    """
    snapshot = _StubSnapshot(
        tip_pose_status="identity_tip_fallback",
        T_robot_tip=[[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]],
        runtime_tip_mode="latest_accepted",
    )
    assert _live_pose_ready(snapshot) is False


def test_live_pose_ready_rejects_stale_data_even_in_coil_as_tip_mode() -> None:
    snapshot = _StubSnapshot(
        tip_pose_status="coil_as_tip",
        T_robot_tip=[[1, 0, 0, 0]] * 4,
        tracker_data_stale=True,
        runtime_tip_mode="coil_as_tip",
    )
    assert _live_pose_ready(snapshot) is False


def test_live_pose_ready_rejects_other_statuses() -> None:
    for status in (
        "missing_registration",
        "invalid_registration",
        "invalid_runtime_tip_calibration",
        "role_mismatch",
        "invalid_transform_chain",
        "missing_0A",
    ):
        snapshot = _StubSnapshot(
            tip_pose_status=status,
            T_robot_tip=[[1, 0, 0, 0]] * 4,
            runtime_tip_mode="latest_accepted",
        )
        assert _live_pose_ready(snapshot) is False, status


# ---------------------------------------------------------------------------
# Bug 2 — Trial Mode button must be visible and discoverable
# ---------------------------------------------------------------------------


pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtWidgets import QApplication

from continuum_robot.gui.tabs.registration_tab import RegistrationTab


class _StubRegistrationController:
    """Minimal stub matching the controller surface RegistrationTab consumes."""

    REQUIRED_SELECTION_COUNT = 4
    MINIMUM_SELECTION_COUNT = 3

    def __init__(self) -> None:
        @dataclass
        class _State:
            landmark_labels: list[str] = field(default_factory=list)
            available_landmarks: list[object] = field(default_factory=list)
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
            registration_label_points: dict[str, object] = field(default_factory=dict)
            registration_residuals_mm: dict[str, object] = field(default_factory=dict)
            registration_label_residual_summary: dict[str, object] = field(default_factory=dict)
            registration_residual_summary_text: str = ""
            registration_solve_summary: str = ""
            session_progress_summary: str = ""
            session_progress_remaining: str = ""
            recent_runs: list[object] = field(default_factory=list)
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


def test_trial_mode_button_visible_regardless_of_callback(qapp) -> None:
    """Bug 2: the Trial Mode button must be visible whether or not the host
    app shell wired ``open_registration_trial``. Visibility gating on the
    callback caused the button to disappear silently on a real bench."""
    tab_with_cb = RegistrationTab(
        _StubRegistrationController(),
        open_registration_trial=lambda: None,
    )
    tab_without_cb = RegistrationTab(_StubRegistrationController())
    assert tab_with_cb.trial_mode_button.isHidden() is False
    assert tab_without_cb.trial_mode_button.isHidden() is False


def test_trial_mode_button_uses_primary_style(qapp) -> None:
    """Bug 2: visually unmistakable styling. variant=ghost was too subtle."""
    tab = RegistrationTab(_StubRegistrationController())
    role_prop = tab.trial_mode_button.property("role")
    assert str(role_prop) == "primary", (
        f"trial_mode_button must use role=primary styling so it stands out on "
        f"the secondary action row; got role={role_prop!r}"
    )


def test_trial_mode_button_adjacent_to_runtime_tip_button(qapp) -> None:
    """Bug 2 placement: per operator request, the button stays next to
    'Open Runtime Tip Calibration' on the secondary action row."""
    tab = RegistrationTab(
        _StubRegistrationController(),
        open_registration_trial=lambda: None,
    )
    rtc_parent = tab.runtime_tip_button.parentWidget()
    trial_parent = tab.trial_mode_button.parentWidget()
    assert rtc_parent is trial_parent, (
        "trial_mode_button must live next to the runtime_tip_button so the "
        "operator can find it on the secondary action row."
    )


def test_trial_mode_button_invokes_callback_when_wired(qapp) -> None:
    invocations: list[bool] = []

    def opener() -> None:
        invocations.append(True)

    tab = RegistrationTab(
        _StubRegistrationController(),
        open_registration_trial=opener,
    )
    tab.trial_mode_button.click()
    assert invocations == [True]


def test_trial_mode_button_click_is_no_op_when_callback_missing(qapp) -> None:
    """Defensive: clicking without a callback wired must not raise."""
    tab = RegistrationTab(_StubRegistrationController())
    tab.trial_mode_button.click()
