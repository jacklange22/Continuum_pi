from __future__ import annotations
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QImage, QWheelEvent
from PySide6.QtWidgets import QApplication

from continuum_robot.gui.theme import COLORS
from continuum_robot.gui.widgets.grid_accuracy_preview_widget import GridAccuracyPreviewWidget
from continuum_robot.gui.widgets.lightweight_scene_3d_widget import SceneViewState
from continuum_robot.gui.widgets.registration_plot_widget import (
    compute_registration_preview_alignment,
)
from continuum_robot.gui.widgets.tool_plot_widget import ToolPlotWidget, build_tracking_scene_model
from continuum_robot.registration.rigid_solver import RigidRegistrationSolver


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _tracking_state(
    *,
    registration_state: str = "missing_registration",
    runtime_tip_mode: str = "latest_accepted",
    runtime_tip_trust_level: str = "missing",
    runtime_tip_calibration_state: str = "missing_runtime_tip_calibration",
    runtime_tip_mode_message: str = "missing",
    runtime_tip_identity_fallback: bool = False,
    tip_pose_status: str = "missing_registration",
    T_robot_aurora=None,
    T_robot_tip_matrix=None,
    tools=None,
    latest_frame_number: int | None = 1,
):
    tools = tools or {}
    return SimpleNamespace(
        registration_state=registration_state,
        runtime_tip_mode=runtime_tip_mode,
        runtime_tip_trust_level=runtime_tip_trust_level,
        runtime_tip_calibration_state=runtime_tip_calibration_state,
        runtime_tip_mode_message=runtime_tip_mode_message,
        runtime_tip_identity_fallback=runtime_tip_identity_fallback,
        tip_pose_status=tip_pose_status,
        T_robot_aurora=T_robot_aurora,
        T_robot_tip_matrix=T_robot_tip_matrix,
        tools=tools,
        latest_frame_number=latest_frame_number,
    )


def test_registration_preview_one_point_uses_translation_only_alignment() -> None:
    result = compute_registration_preview_alignment(
        truth_points_by_label={"L1": (10.0, 20.0, 30.0)},
        captured_points_by_label={"L1": (1.0, 2.0, 3.0)},
        ordered_labels=["L1"],
    )

    assert result.mode == "translation_preview"
    assert result.underconstrained is True
    assert np.allclose(result.transform[0:3, 0:3], np.eye(3))
    assert np.allclose(result.transform[0:3, 3], np.array([9.0, 18.0, 27.0]))
    assert np.allclose(result.aligned_points_by_label["L1"], np.array([10.0, 20.0, 30.0]))


def test_registration_preview_two_points_enters_underconstrained_segment_mode() -> None:
    result = compute_registration_preview_alignment(
        truth_points_by_label={
            "L1": (0.0, 0.0, 0.0),
            "L2": (0.0, 10.0, 0.0),
        },
        captured_points_by_label={
            "L1": (5.0, 5.0, 0.0),
            "L2": (15.0, 5.0, 0.0),
        },
        ordered_labels=["L1", "L2"],
    )

    assert result.mode == "segment_preview"
    assert result.underconstrained is True
    assert "underconstrained" in result.message.lower()
    assert np.allclose(result.aligned_points_by_label["L1"], np.array([0.0, 0.0, 0.0]), atol=1e-6)
    assert np.allclose(result.aligned_points_by_label["L2"], np.array([0.0, 10.0, 0.0]), atol=1e-6)


def test_registration_preview_three_points_uses_rigid_fit_preview_path() -> None:
    truth = {
        "L1": (0.0, 0.0, 0.0),
        "L2": (10.0, 0.0, 0.0),
        "L3": (0.0, 8.0, 0.0),
    }
    captured = {
        "L1": (2.0, 3.0, 1.0),
        "L2": (12.0, 3.0, 1.0),
        "L3": (2.0, 11.0, 1.0),
    }

    result = compute_registration_preview_alignment(
        truth_points_by_label=truth,
        captured_points_by_label=captured,
        ordered_labels=["L1", "L2", "L3"],
    )

    assert result.mode == "rigid_fit_preview"
    assert result.underconstrained is False
    assert max(result.residual_norms_by_label.values()) < 1e-6


def test_registration_preview_uses_actual_solved_transform_when_available() -> None:
    truth = {
        "L1": (0.0, 0.0, 0.0),
        "L2": (10.0, 0.0, 0.0),
        "L3": (0.0, 10.0, 0.0),
    }
    captured = {
        "L1": (2.0, 3.0, 0.0),
        "L2": (12.0, 3.0, 0.0),
        "L3": (2.0, 13.0, 0.0),
    }
    transform = np.eye(4)
    transform[0:3, 3] = np.array([-2.0, -3.0, 0.0])

    result = compute_registration_preview_alignment(
        truth_points_by_label=truth,
        captured_points_by_label=captured,
        ordered_labels=["L1", "L2", "L3"],
        solved_transform=transform,
        solved_residuals_by_label={"L1": (0.0, 0.0, 0.0), "L2": (0.0, 0.0, 0.0), "L3": (0.0, 0.0, 0.0)},
    )

    assert result.mode == "solved_fit"
    assert result.underconstrained is False
    assert "actual rigid registration transform" in result.message.lower()
    assert np.allclose(result.aligned_points_by_label["L2"], np.array([10.0, 0.0, 0.0]))


def test_registration_preview_does_not_mutate_solver_inputs_or_outputs() -> None:
    truth = {
        "L1": (0.0, 0.0, 0.0),
        "L2": (10.0, 0.0, 0.0),
        "L3": (0.0, 10.0, 0.0),
    }
    captured = {
        "L1": (1.0, 2.0, 0.0),
        "L2": (11.0, 2.0, 0.0),
        "L3": (1.0, 12.0, 0.0),
    }
    truth_before = {key: tuple(value) for key, value in truth.items()}
    captured_before = {key: tuple(value) for key, value in captured.items()}
    solver = RigidRegistrationSolver()
    baseline = solver.solve_T_robot_aurora(
        np.asarray(list(captured.values()), dtype=float),
        np.asarray(list(truth.values()), dtype=float),
    )

    _ = compute_registration_preview_alignment(
        truth_points_by_label=truth,
        captured_points_by_label=captured,
        ordered_labels=["L1", "L2", "L3"],
    )

    after = solver.solve_T_robot_aurora(
        np.asarray(list(captured.values()), dtype=float),
        np.asarray(list(truth.values()), dtype=float),
    )

    assert truth == truth_before
    assert captured == captured_before
    assert np.allclose(after, baseline)


def test_tracking_scene_only_shows_tip_and_robot_frame_when_registration_chain_is_available() -> None:
    missing_scene, _, missing_frame = build_tracking_scene_model(
        live_state=_tracking_state(
            tools={
                "0A": {"translation_mm": (1.0, 2.0, 3.0), "quaternion_wxyz": (1.0, 0.0, 0.0, 0.0), "tracking_state": "tracked", "status": "tracked", "frame_number": 1},
                "0B": {"translation_mm": (4.0, 5.0, 6.0), "quaternion_wxyz": (1.0, 0.0, 0.0, 0.0), "tracking_state": "tracked", "status": "tracked", "frame_number": 1},
            }
        ),
    )

    assert missing_frame == "aurora"
    assert not any(point.key == "tip" for point in missing_scene.points)
    assert any("Registration: missing registration" == line for line in missing_scene.overlay_lines)

    robot_scene, _, robot_frame = build_tracking_scene_model(
        live_state=_tracking_state(
            registration_state="loaded",
            runtime_tip_mode="latest_accepted",
            runtime_tip_trust_level="trusted",
            runtime_tip_calibration_state="loaded",
            runtime_tip_mode_message="Latest accepted runtime tip artifact is active.",
            tip_pose_status="ok",
            T_robot_aurora=np.eye(4).tolist(),
            T_robot_tip_matrix=np.array(
                [
                    [1.0, 0.0, 0.0, 7.0],
                    [0.0, 1.0, 0.0, 8.0],
                    [0.0, 0.0, 1.0, 9.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ).tolist(),
            tools={
                "0A": {"translation_mm": (1.0, 2.0, 3.0), "quaternion_wxyz": (1.0, 0.0, 0.0, 0.0), "tracking_state": "tracked", "status": "tracked", "frame_number": 2},
            },
        ),
    )

    assert robot_frame == "robot"
    assert any(point.key == "tip" for point in robot_scene.points)
    assert any("Runtime tip mode: latest accepted | trusted" == line for line in robot_scene.overlay_lines)
    assert any("Runtime tip source: Latest accepted runtime tip artifact is active." == line for line in robot_scene.overlay_lines)
    assert not any(line.startswith("Tip glyph") for line in robot_scene.overlay_lines)
    assert not any(line.startswith("Tip pose") for line in robot_scene.overlay_lines)
    assert any(axis.label == "Robot" for axis in robot_scene.axes)


def test_tracking_scene_reports_identity_fallback_without_tip_overlay() -> None:
    scene, _, frame = build_tracking_scene_model(
        live_state=_tracking_state(
            registration_state="loaded",
            runtime_tip_mode_message="Tracking is using the identity runtime tip fallback.",
            runtime_tip_identity_fallback=True,
            tip_pose_status="identity_tip_fallback",
            T_robot_aurora=np.eye(4).tolist(),
            T_robot_tip_matrix=None,
            tools={
                "0A": {
                    "translation_mm": (1.0, 2.0, 3.0),
                    "quaternion_wxyz": (1.0, 0.0, 0.0, 0.0),
                    "tracking_state": "tracked",
                    "status": "tracked",
                    "frame_number": 3,
                }
            },
        ),
    )

    assert frame == "robot"
    assert not any(point.key == "tip" for point in scene.points)
    assert any("Runtime tip source: Tracking is using the identity runtime tip fallback." == line for line in scene.overlay_lines)
    assert not any(line.startswith("Tip pose") for line in scene.overlay_lines)
    assert not any(line.startswith("Tip glyph") for line in scene.overlay_lines)


def test_tracking_scene_reports_explicit_coil_as_tip_mode_with_visible_tip_glyph() -> None:
    scene, _, frame = build_tracking_scene_model(
        live_state=_tracking_state(
            registration_state="loaded",
            runtime_tip_mode="coil_as_tip",
            runtime_tip_trust_level="thesis_trusted",
            runtime_tip_calibration_state="coil_as_tip",
            runtime_tip_mode_message="Coil-as-tip override is active. The 0A coil pose is shown directly as the tip.",
            tip_pose_status="coil_as_tip",
            T_robot_aurora=np.eye(4).tolist(),
            T_robot_tip_matrix=np.array(
                [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 2.0],
                    [0.0, 0.0, 1.0, 3.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ).tolist(),
            tools={
                "0A": {
                    "translation_mm": (1.0, 2.0, 3.0),
                    "quaternion_wxyz": (1.0, 0.0, 0.0, 0.0),
                    "tracking_state": "tracked",
                    "status": "tracked",
                    "frame_number": 4,
                }
            },
        ),
    )

    assert frame == "robot"
    assert any(point.key == "tip" for point in scene.points)
    assert any("Runtime tip mode: coil as tip | thesis trusted" == line for line in scene.overlay_lines)
    assert not any(line.startswith("Tip pose") for line in scene.overlay_lines)
    assert not any(line.startswith("Tip glyph") for line in scene.overlay_lines)


def test_tool_plot_widget_preserves_camera_state_across_benign_refresh() -> None:
    _app()
    widget = ToolPlotWidget()
    widget.resize(900, 700)
    widget.show()
    widget.set_tracking_state(
        _tracking_state(
            registration_state="loaded",
            runtime_tip_calibration_state="loaded",
            tip_pose_status="ok",
            T_robot_aurora=np.eye(4).tolist(),
            T_robot_tip_matrix=np.eye(4).tolist(),
            tools={
                "0A": {"translation_mm": (1.0, 2.0, 3.0), "quaternion_wxyz": (1.0, 0.0, 0.0, 0.0), "tracking_state": "tracked", "status": "tracked", "frame_number": 1},
            },
        )
    )
    custom_view = SceneViewState(yaw_deg=10.0, pitch_deg=-15.0, zoom=3.0, pan_x_px=12.0, pan_y_px=-18.0, target_xyz=(1.0, 1.0, 1.0))
    widget.set_view_state(custom_view)

    widget.set_tracking_state(
        _tracking_state(
            registration_state="loaded",
            runtime_tip_calibration_state="loaded",
            tip_pose_status="ok",
            T_robot_aurora=np.eye(4).tolist(),
            T_robot_tip_matrix=np.eye(4).tolist(),
            tools={
                "0A": {"translation_mm": (1.5, 2.5, 3.5), "quaternion_wxyz": (1.0, 0.0, 0.0, 0.0), "tracking_state": "tracked", "status": "tracked", "frame_number": 2},
            },
        )
    )

    assert widget.view_state() == custom_view


def test_grid_accuracy_preview_widget_renders_under_current_theme() -> None:
    _app()
    widget = GridAccuracyPreviewWidget()
    widget.resize(720, 360)
    widget.set_preview(
        truth_catalog=[
            {"label": "P01", "target_index": 0, "truth_point_mm": [0.0, 0.0, 0.0]},
            {"label": "P02", "target_index": 1, "truth_point_mm": [25.4, 0.0, 0.0]},
            {"label": "P03", "target_index": 2, "truth_point_mm": [0.0, 25.4, 0.0]},
        ],
        metrics={
            "alignment_ready": True,
            "alignment_ready_reason": "At least three labeled point centroids are available for the rigid alignment solve.",
            "point_count_complete": 3,
            "point_count_partial": 0,
            "point_count_not_started": 0,
            "per_point_metrics": {
                "P01": {"raw_sample_count": 3, "outlier_count": 0, "aligned_centroid_truth_mm": [0.1, 0.0, 0.0], "residual_mm": 0.1},
                "P02": {"raw_sample_count": 3, "outlier_count": 0, "aligned_centroid_truth_mm": [25.5, 0.0, 0.0], "residual_mm": 0.1},
                "P03": {"raw_sample_count": 3, "outlier_count": 0, "aligned_centroid_truth_mm": [0.0, 25.2, 0.0], "residual_mm": 0.2},
            },
        },
        selected_target_index=1,
        expected_samples=3,
    )

    assert COLORS.accent == COLORS.button_primary_border
    image = QImage(widget.size(), QImage.Format_ARGB32)
    image.fill(0)
    widget.render(image)

    assert image.width() == 720
    assert image.height() == 360


def test_grid_accuracy_preview_widget_partial_state_renders_safely() -> None:
    _app()
    widget = GridAccuracyPreviewWidget()
    widget.resize(720, 360)
    widget.set_preview(
        truth_catalog=[
            {"label": "P01", "target_index": 0, "truth_point_mm": [0.0, 0.0, 0.0]},
            {"label": "P02", "target_index": 1, "truth_point_mm": [25.4, 0.0, 0.0]},
            {"label": "P03", "target_index": 2, "truth_point_mm": [0.0, 25.4, 0.0]},
            {"label": "P04", "target_index": 3, "truth_point_mm": [25.4, 25.4, 0.0]},
        ],
        metrics={
            "alignment_ready": False,
            "alignment_ready_reason": "Capture 1 more labeled point(s) before aligned RMS residuals are available.",
            "point_count_complete": 2,
            "point_count_partial": 1,
            "point_count_not_started": 1,
            "per_point_metrics": {
                "P01": {"raw_sample_count": 3, "outlier_count": 0},
                "P02": {"raw_sample_count": 2, "outlier_count": 0},
                "P03": {"raw_sample_count": 1, "outlier_count": 0},
            },
        },
        selected_target_index=2,
        expected_samples=3,
    )

    image = QImage(widget.size(), QImage.Format_ARGB32)
    image.fill(0)
    widget.render(image)

    assert image.width() == 720
    assert image.height() == 360


def test_grid_accuracy_preview_widget_paint_failure_does_not_poison_future_renders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _app()
    widget = GridAccuracyPreviewWidget()
    widget.resize(720, 360)
    widget.set_preview(
        truth_catalog=[{"label": "P01", "target_index": 0, "truth_point_mm": [0.0, 0.0, 0.0]}],
        metrics={
            "alignment_ready": False,
            "alignment_ready_reason": "Capture 2 more labeled point(s) before aligned RMS residuals are available.",
            "point_count_complete": 0,
            "point_count_partial": 0,
            "point_count_not_started": 1,
            "per_point_metrics": {},
        },
        selected_target_index=0,
        expected_samples=3,
    )

    original = widget._paint_contents

    def _boom(_painter):
        raise RuntimeError("paint boom")

    monkeypatch.setattr(widget, "_paint_contents", _boom)
    failed_image = QImage(widget.size(), QImage.Format_ARGB32)
    failed_image.fill(0)
    widget.render(failed_image)

    monkeypatch.setattr(widget, "_paint_contents", original)
    recovered_image = QImage(widget.size(), QImage.Format_ARGB32)
    recovered_image.fill(0)
    widget.render(recovered_image)

    assert failed_image.width() == 720
    assert recovered_image.width() == 720


def test_tool_plot_widget_ignores_wheel_zoom_without_control_modifier() -> None:
    _app()
    widget = ToolPlotWidget()
    view = SceneViewState(yaw_deg=0.0, pitch_deg=0.0, zoom=2.0, pan_x_px=0.0, pan_y_px=0.0, target_xyz=(0.0, 0.0, 0.0))
    widget.set_view_state(view)
    event = QWheelEvent(
        QPointF(20.0, 20.0),
        QPointF(20.0, 20.0),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.ScrollUpdate,
        False,
    )

    widget.wheelEvent(event)

    assert widget.view_state() == view
    assert event.isAccepted() is False


def test_tool_plot_widget_allows_control_wheel_zoom() -> None:
    _app()
    widget = ToolPlotWidget()
    widget.set_view_state(SceneViewState(yaw_deg=0.0, pitch_deg=0.0, zoom=2.0, pan_x_px=0.0, pan_y_px=0.0, target_xyz=(0.0, 0.0, 0.0)))
    event = QWheelEvent(
        QPointF(20.0, 20.0),
        QPointF(20.0, 20.0),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.NoButton,
        Qt.ControlModifier,
        Qt.ScrollUpdate,
        False,
    )

    widget.wheelEvent(event)

    assert widget.view_state().zoom > 2.0
    assert event.isAccepted() is True


def test_tool_plot_widget_handles_missing_registration_and_runtime_tip_without_crashing() -> None:
    _app()
    widget = ToolPlotWidget()
    widget.set_tracking_state(
        _tracking_state(
            registration_state="missing_registration",
            runtime_tip_calibration_state="missing_runtime_tip_calibration",
            tip_pose_status="missing_registration",
            tools={},
            latest_frame_number=None,
        )
    )

    assert widget.scene().frame_label == "aurora"
    assert any("Registration: missing registration" == line for line in widget.scene().overlay_lines)
