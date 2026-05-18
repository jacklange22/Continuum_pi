"""Tests for the capture-phase landmark map in Registration Trial Mode.

The landmark map widget has two render modes:
- selection (driven by ``set_landmarks``, used in the registration tab)
- trial-capture (driven by ``set_trial_capture_state``, used in the trial dialog)

These tests pin the new trial-capture mode end-to-end:

1. The widget accepts the new state setter and renders without crashing.
2. The widget's internal state correctly reflects active / completed /
   pending / non-trial categories so the paint logic chooses the right
   visual treatment.
3. The Registration Trial dialog embeds the map only when XYZ coordinates
   are supplied, falls back gracefully otherwise, and pushes the controller's
   per-tick state into the map so the highlighted landmark always tracks
   ``state.current_label``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication


from continuum_robot.gui.widgets.registration_landmark_map_widget import (
    RegistrationLandmarkMapWidget,
)
from continuum_robot.gui.widgets.registration_trial_dialog import (
    RegistrationTrialDialog,
)


# ---------------------------------------------------------------------------
# Test infrastructure (QApplication + stub trial controller)
# ---------------------------------------------------------------------------


def _app() -> QApplication:
    """Return the singleton QApplication so widget tests can render headlessly."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@dataclass
class _StubControllerState:
    """Shape-compatible with TrialCaptureState; only the fields the dialog reads."""

    is_active: bool = False
    is_complete: bool = False
    captures_per_landmark: int = 5
    landmark_labels: list[str] = field(default_factory=list)
    current_label: str | None = None
    captured_counts_by_label: dict[str, int] = field(default_factory=dict)
    last_error: str | None = None
    status_message: str = ""
    last_run_output_dir: Path | None = None
    last_trial_report_md: Path | None = None
    last_trial_report_json: Path | None = None
    last_run_summary: dict[str, Any] = field(default_factory=dict)


class _StubTrialController:
    """Trial-controller stub that lets the test set state directly.

    Mirrors the public surface the dialog reads -- the dialog never mutates
    controller state during this test, it only renders what the controller
    reports, so a passive shape stub is enough.
    """

    def __init__(self) -> None:
        self.state = _StubControllerState()

    def start(self, *args, **kwargs) -> _StubControllerState:  # pragma: no cover - unused here
        return self.state

    def target_total(self) -> int:
        return int(self.state.captures_per_landmark) * len(self.state.landmark_labels)

    def total_captured(self) -> int:
        return sum(int(value) for value in self.state.captured_counts_by_label.values())

    def remaining_for_current_label(self) -> int:  # pragma: no cover - unused here
        return 0


def _twelve_candidate_layout() -> tuple[list[str], dict[str, list[float]], dict[str, str]]:
    """Produce a 12-point candidate set with realistic XYZ at a single z plane.

    Mirrors the shape used in production registration.yaml: L1..L12, friendly
    display labels, all at z = -5 mm.
    """
    layout = [
        ("L1", "North Outer", [0.0, 35.0, -5.0]),
        ("L2", "West Outer", [-35.0, 0.0, -5.0]),
        ("L3", "South Outer", [0.0, -35.0, -5.0]),
        ("L4", "East Outer", [35.0, 0.0, -5.0]),
        ("L5", "North Inner", [0.0, 25.0, -5.0]),
        ("L6", "West Inner", [-25.0, 0.0, -5.0]),
        ("L7", "South Inner", [0.0, -25.0, -5.0]),
        ("L8", "East Inner", [25.0, 0.0, -5.0]),
        ("L9", "NE Inner", [10.0, 25.0, -5.0]),
        ("L10", "WN Inner", [-25.0, 10.0, -5.0]),
        ("L11", "SW Inner", [-10.0, -25.0, -5.0]),
        ("L12", "ES Inner", [25.0, -10.0, -5.0]),
    ]
    labels = [item[0] for item in layout]
    display_labels = {item[0]: item[1] for item in layout}
    points_by_label = {item[0]: item[2] for item in layout}
    return labels, points_by_label, display_labels


# ---------------------------------------------------------------------------
# Widget-level tests for the new trial-capture rendering mode
# ---------------------------------------------------------------------------


class TestLandmarkMapTrialCaptureMode:
    def test_set_trial_capture_state_populates_internal_state(self) -> None:
        _app()
        widget = RegistrationLandmarkMapWidget()
        labels, points, display = _twelve_candidate_layout()
        fixed_numeric = {label: index + 1 for index, label in enumerate(labels)}
        trial_labels = labels[:5]  # only first 5 selected for this trial
        widget.set_trial_capture_state(
            points_by_label=points,
            display_labels=display,
            fixed_numeric_labels=fixed_numeric,
            trial_labels=trial_labels,
            active_label="L3",
            completed_labels=["L1", "L2"],
        )
        # Internal state for the paint pass.
        assert widget._render_mode == widget._MODE_TRIAL_CAPTURE
        assert widget._active_label == "L3"
        assert widget._completed_labels == {"L1", "L2"}
        assert widget._trial_labels == {"L1", "L2", "L3", "L4", "L5"}
        assert widget._fixed_numeric_labels == fixed_numeric
        # XYZ accepts 3-element coordinates and stores them verbatim.
        assert widget._points_by_label["L3"][2] == -5.0
        # Selection-mode state was reset so paint cannot accidentally use it.
        assert widget._enabled_by_label == {}
        assert widget._selected_order == {}

    def test_point_center_for_label_works_in_trial_capture_mode(self) -> None:
        _app()
        widget = RegistrationLandmarkMapWidget()
        widget.resize(640, 480)
        labels, points, display = _twelve_candidate_layout()
        widget.set_trial_capture_state(
            points_by_label=points,
            display_labels=display,
            fixed_numeric_labels={label: index + 1 for index, label in enumerate(labels)},
            trial_labels=labels[:4],
            active_label="L1",
            completed_labels=[],
        )
        center = widget.point_center_for_label("L1")
        assert isinstance(center, QPoint)
        # Two different points should project to different screen positions.
        other = widget.point_center_for_label("L3")
        assert other is not None
        assert center != other

    def test_trial_capture_mode_ignores_clicks(self) -> None:
        """Trial-capture mode is read-only -- mousePressEvent must not emit pointToggled."""
        _app()
        widget = RegistrationLandmarkMapWidget()
        widget.resize(640, 480)
        widget.show()
        labels, points, display = _twelve_candidate_layout()
        widget.set_trial_capture_state(
            points_by_label=points,
            display_labels=display,
            fixed_numeric_labels={label: index + 1 for index, label in enumerate(labels)},
            trial_labels=labels[:4],
            active_label="L1",
            completed_labels=[],
        )
        seen: list[str] = []
        widget.pointToggled.connect(seen.append)
        # Synthesize a click at a known landmark center; should NOT emit.
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest

        center = widget.point_center_for_label("L1")
        assert center is not None
        QTest.mouseClick(widget, Qt.LeftButton, Qt.NoModifier, center)
        assert seen == []

    def test_switching_back_to_selection_mode_clears_trial_state(self) -> None:
        _app()
        widget = RegistrationLandmarkMapWidget()
        labels, points, display = _twelve_candidate_layout()
        widget.set_trial_capture_state(
            points_by_label=points,
            display_labels=display,
            fixed_numeric_labels={label: index + 1 for index, label in enumerate(labels)},
            trial_labels=labels,
            active_label="L1",
            completed_labels=["L2"],
        )
        widget.set_landmarks(
            points_by_label=points,
            display_labels=display,
            enabled_by_label={label: True for label in labels},
            selected_labels=["L1", "L3"],
        )
        assert widget._render_mode == widget._MODE_SELECTION
        assert widget._active_label is None
        assert widget._completed_labels == set()
        assert widget._trial_labels == set()
        assert widget._fixed_numeric_labels == {}

    def test_paint_event_does_not_crash_in_trial_capture_mode(self) -> None:
        """Render once headlessly to ensure paint logic doesn't blow up on edge cases."""
        _app()
        widget = RegistrationLandmarkMapWidget()
        widget.resize(640, 480)
        widget.show()
        labels, points, display = _twelve_candidate_layout()
        # Cover every visual state: one active, one completed, some pending,
        # and some not in the trial at all (the faint reference dots).
        widget.set_trial_capture_state(
            points_by_label=points,
            display_labels=display,
            fixed_numeric_labels={label: index + 1 for index, label in enumerate(labels)},
            trial_labels=["L1", "L2", "L3", "L4", "L5", "L6"],
            active_label="L3",
            completed_labels=["L1", "L2"],
        )
        # Force a paint cycle so paintEvent runs.
        widget.repaint()
        QApplication.processEvents()


# ---------------------------------------------------------------------------
# Dialog-level integration tests
# ---------------------------------------------------------------------------


class TestRegistrationTrialDialogCaptureMap:
    def test_dialog_without_xyz_omits_map(self, tmp_path: Path) -> None:
        _app()
        dialog = RegistrationTrialDialog(
            _StubTrialController(),
            candidate_labels=["L1", "L2", "L3", "L4"],
        )
        try:
            # No XYZ provided -> no map widget instantiated.
            assert dialog.capture_map is None
        finally:
            dialog.deleteLater()

    def test_dialog_with_xyz_creates_map_in_capture_phase(self, tmp_path: Path) -> None:
        _app()
        labels, points, display = _twelve_candidate_layout()
        dialog = RegistrationTrialDialog(
            _StubTrialController(),
            candidate_labels=labels,
            candidate_points_by_label=points,
            candidate_display_labels=display,
        )
        try:
            assert dialog.capture_map is not None
            # Fixed numeric labels follow the candidate-list order, so L1->1, L12->12.
            assert dialog._fixed_numeric_labels == {label: index + 1 for index, label in enumerate(labels)}
            # The dialog stores XYZ as plain floats it can hand back to the widget.
            assert dialog._candidate_points_by_label["L1"][:2] == [0.0, 35.0]
        finally:
            dialog.deleteLater()

    def test_refresh_pushes_active_and_completed_into_map(self, tmp_path: Path) -> None:
        _app()
        labels, points, display = _twelve_candidate_layout()
        controller = _StubTrialController()
        controller.state = _StubControllerState(
            is_active=True,
            is_complete=False,
            captures_per_landmark=10,
            landmark_labels=["L1", "L2", "L3", "L4", "L5"],
            current_label="L3",
            captured_counts_by_label={"L1": 10, "L2": 10, "L3": 4, "L4": 0, "L5": 0},
            status_message="capturing L3",
        )
        dialog = RegistrationTrialDialog(
            controller,
            candidate_labels=labels,
            candidate_points_by_label=points,
            candidate_display_labels=display,
        )
        try:
            dialog._refresh_capture_view()
            map_widget = dialog.capture_map
            assert map_widget is not None
            assert map_widget._render_mode == map_widget._MODE_TRIAL_CAPTURE
            assert map_widget._active_label == "L3"
            # L1, L2 reached their quota -> completed; L3 active; L4, L5 pending.
            assert map_widget._completed_labels == {"L1", "L2"}
            assert map_widget._trial_labels == {"L1", "L2", "L3", "L4", "L5"}
            # L6..L12 are in the candidate set but not part of this trial; they
            # should appear in points_by_label so the map draws them as faint
            # reference dots, but not in trial_labels.
            assert "L7" in map_widget._points_by_label
            assert "L7" not in map_widget._trial_labels
        finally:
            dialog.deleteLater()

    def test_refresh_clears_active_when_trial_complete(self, tmp_path: Path) -> None:
        _app()
        labels, points, display = _twelve_candidate_layout()
        controller = _StubTrialController()
        controller.state = _StubControllerState(
            is_active=True,
            is_complete=True,
            captures_per_landmark=5,
            landmark_labels=["L1", "L2", "L3"],
            current_label=None,
            captured_counts_by_label={"L1": 5, "L2": 5, "L3": 5},
            status_message="done",
        )
        dialog = RegistrationTrialDialog(
            controller,
            candidate_labels=labels,
            candidate_points_by_label=points,
            candidate_display_labels=display,
        )
        try:
            dialog._refresh_capture_view()
            map_widget = dialog.capture_map
            assert map_widget is not None
            # No active landmark once complete.
            assert map_widget._active_label is None
            # Everything captured -> all in completed set.
            assert map_widget._completed_labels == {"L1", "L2", "L3"}
        finally:
            dialog.deleteLater()

    def test_refresh_handles_zero_quota_safely(self, tmp_path: Path) -> None:
        """``captures_per_landmark=0`` is degenerate but must not raise or
        falsely flag every landmark as completed."""
        _app()
        labels, points, display = _twelve_candidate_layout()
        controller = _StubTrialController()
        controller.state = _StubControllerState(
            is_active=True,
            captures_per_landmark=0,
            landmark_labels=["L1", "L2"],
            current_label="L1",
            captured_counts_by_label={"L1": 0, "L2": 0},
        )
        dialog = RegistrationTrialDialog(
            controller,
            candidate_labels=labels,
            candidate_points_by_label=points,
            candidate_display_labels=display,
        )
        try:
            dialog._refresh_capture_view()
            map_widget = dialog.capture_map
            assert map_widget is not None
            # With target=0, the "count >= target > 0" condition skips marking
            # any landmark as completed (avoids the absurd "all done immediately").
            assert map_widget._completed_labels == set()
            assert map_widget._active_label == "L1"
        finally:
            dialog.deleteLater()
