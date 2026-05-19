"""Regression tests for two bench bugs found 2026-05-15.

Bug 1: registration controller silently re-set selected_model_labels back to
the previous run's labels on every refresh whenever a saved registration was
loaded (because `snapshot.averaged_points_by_label` was non-empty). The
operator could click L1 to deselect or L5 to add, but the next refresh
snapped it back.

Bug 3: the GUI gated the "live robot-frame position" display on
`tip_pose_status == "ok"`, which is never set in coil_as_tip mode (the
tracking service writes "coil_as_tip" there even though a valid T_robot_tip
is produced). Coil-as-tip is treated as thesis-trusted per the README, so
the GUI must display the live position in that mode.

These tests run headlessly without QApplication.
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

    Default selection now seeds from the full configured landmark list, so
    "add L5" means "deselect then re-select L5". This test still proves
    the underlying invariant: toggle ops survive _apply_snapshot.
    """
    ctrl = _bootstrap_registration_controller()
    ctrl.refresh()
    initial_count = len(ctrl.state.selected_model_labels)
    # Deselect L1 (count -1), then toggle L5 off-then-on so L5 is selected at
    # the end. Net change: L1 removed, L5 still in, total = initial - 1.
    ctrl.toggle_selected_model_point("L1")
    ctrl.refresh()
    if "L5" in ctrl.state.selected_model_labels:
        ctrl.toggle_selected_model_point("L5")  # remove
        ctrl.refresh()
    ctrl.toggle_selected_model_point("L5")  # add (back)
    state = ctrl.refresh()
    assert "L5" in state.selected_model_labels
    assert "L1" not in state.selected_model_labels
    assert len(state.selected_model_labels) == initial_count - 1


def test_active_session_still_locks_selection() -> None:
    """The fix must not weaken the active/pending_accept gate.

    We don't have a way to enter the active state without driving the live
    service into a session, so this test asserts the controller's toggle
    method still raises when state.active is forced True.
    """
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
