"""Lightweight live spatial monitor for Aurora tools and runtime tip pose."""

from __future__ import annotations

from collections import deque
from typing import Mapping, Sequence

import numpy as np

from continuum_robot.gui.theme import COLORS
from continuum_robot.gui.widgets.lightweight_scene_3d_widget import (
    LightweightScene3DWidget,
    SceneAxes3D,
    SceneLine3D,
    SceneModel3D,
    ScenePoint3D,
    ScenePolyline3D,
)
from continuum_robot.tracking.transforms import make_transform_A_B


class ToolPlotWidget(LightweightScene3DWidget):
    """Draw tracker tools and tip pose in one lightweight interactive scene."""

    MAX_TRAIL_POINTS = 18

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._points: dict[str, tuple[float, float, float]] = {}
        self._trail_points_by_key: dict[str, deque[tuple[float, float, float]]] = {}
        self._last_frame_by_key: dict[str, int | None] = {}
        self._display_frame_label = "aurora"
        self.setMinimumHeight(260)

    def set_points(self, points: dict[str, tuple[float, float] | tuple[float, float, float]]) -> None:
        normalized: dict[str, tuple[float, float, float]] = {}
        for label, coords in points.items():
            if len(coords) >= 3:
                normalized[str(label)] = (float(coords[0]), float(coords[1]), float(coords[2]))
            else:
                normalized[str(label)] = (float(coords[0]), float(coords[1]), 0.0)
        self._points = normalized
        scene = SceneModel3D(
            frame_label="aurora",
            overlay_lines=("Legacy compatibility point preview.",),
            axes=(SceneAxes3D(key="frame", origin_xyz=(0.0, 0.0, 0.0), label="Aurora"),),
            points=tuple(
                ScenePoint3D(
                    key=label,
                    xyz=coords,
                    color_hex=_tool_color(label),
                    label=label,
                    radius_px=6.0,
                    shape="circle" if label != "tip" else "diamond",
                )
                for label, coords in normalized.items()
            ),
        )
        self.set_scene(scene)

    def set_tracking_state(self, live_state) -> None:
        scene, points_by_key, frame_label = build_tracking_scene_model(
            live_state=live_state,
            trail_points_by_key=self._trail_points_by_key,
            last_frame_by_key=self._last_frame_by_key,
        )
        if frame_label != self._display_frame_label:
            self._display_frame_label = frame_label
            self._trail_points_by_key = {
                key: deque([point], maxlen=self.MAX_TRAIL_POINTS)
                for key, point in points_by_key.items()
            }
            self._last_frame_by_key = {
                key: live_state.tools.get(key, {}).get("frame_number") if key in live_state.tools else None
                for key in points_by_key
            }
        else:
            self._update_trails(points_by_key=points_by_key, live_state=live_state)
        self._points = dict(points_by_key)
        self.set_scene(scene)

    def _update_trails(self, *, points_by_key: dict[str, tuple[float, float, float]], live_state) -> None:
        visible_keys = set(points_by_key)
        for key in list(self._trail_points_by_key):
            if key not in visible_keys:
                self._trail_points_by_key.pop(key, None)
                self._last_frame_by_key.pop(key, None)
        for key, point in points_by_key.items():
            frame_number = None
            if key == "tip":
                frame_number = live_state.latest_frame_number
            else:
                frame_number = live_state.tools.get(key, {}).get("frame_number")
            if self._last_frame_by_key.get(key) == frame_number and key in self._trail_points_by_key:
                continue
            trail = self._trail_points_by_key.setdefault(key, deque(maxlen=self.MAX_TRAIL_POINTS))
            if not trail or np.linalg.norm(np.asarray(trail[-1], dtype=float) - np.asarray(point, dtype=float)) > 0.5:
                trail.append(point)
            self._last_frame_by_key[key] = frame_number


def build_tracking_scene_model(
    *,
    live_state,
    trail_points_by_key: Mapping[str, Sequence[tuple[float, float, float]]] | None = None,
    last_frame_by_key: Mapping[str, int | None] | None = None,
) -> tuple[SceneModel3D, dict[str, tuple[float, float, float]], str]:
    del last_frame_by_key
    T_robot_aurora = _as_transform(getattr(live_state, "T_robot_aurora", None))
    T_robot_tip = _as_transform(getattr(live_state, "T_robot_tip_matrix", None))
    registration_loaded = str(getattr(live_state, "registration_state", "missing_registration")) == "loaded" and T_robot_aurora is not None
    frame_label = "robot" if registration_loaded else "aurora"
    frame_axes_label = "Robot" if registration_loaded else "Aurora"

    points: list[ScenePoint3D] = []
    lines: list[SceneLine3D] = []
    axes: list[SceneAxes3D] = [SceneAxes3D(key="frame", origin_xyz=(0.0, 0.0, 0.0), label=frame_axes_label)]
    polylines: list[ScenePolyline3D] = []
    overlay_lines = [
        f"0A raw pose: {_tool_overlay_state(live_state.tools.get('0A', {}))}",
        f"0B raw pose: {_tool_overlay_state(live_state.tools.get('0B', {}))}",
        f"Registration: {str(getattr(live_state, 'registration_state', 'missing_registration')).replace('_', ' ')}",
        (
            "Runtime tip mode: "
            f"{str(getattr(live_state, 'runtime_tip_mode', 'latest_accepted')).replace('_', ' ')} | "
            f"{str(getattr(live_state, 'runtime_tip_trust_level', 'missing')).replace('_', ' ')}"
        ),
        (
            "Runtime tip source: "
            f"{str(getattr(live_state, 'runtime_tip_mode_message', 'missing')).strip() or 'missing'}"
        ),
        f"Tip pose: {str(getattr(live_state, 'tip_pose_status', 'missing_registration')).replace('_', ' ')}",
        "Tip glyph: shown" if registration_loaded and T_robot_tip is not None else "Tip glyph: hidden",
    ]

    if registration_loaded:
        lines.extend(_grid_lines(size_mm=80.0, step_mm=20.0))

    display_points_by_key: dict[str, tuple[float, float, float]] = {}
    for tool_id in ("0A", "0B"):
        tool = live_state.tools.get(tool_id, {})
        translation = tool.get("translation_mm")
        quaternion = tool.get("quaternion_wxyz")
        if translation is None:
            continue
        T_display_tool = _tool_transform_in_display_frame(
            translation=translation,
            quaternion=quaternion,
            T_robot_aurora=T_robot_aurora,
            registration_loaded=registration_loaded,
        )
        if T_display_tool is None:
            continue
        display_xyz = tuple(float(T_display_tool[row, 3]) for row in range(3))
        display_points_by_key[tool_id] = display_xyz
        points.append(
            ScenePoint3D(
                key=tool_id,
                xyz=display_xyz,
                color_hex=_tool_color(tool_id),
                label=tool_id,
                radius_px=6.0,
                outline_hex=COLORS.text_primary,
                shape="circle",
            )
        )
        axes.append(
            SceneAxes3D(
                key=f"{tool_id}_axes",
                origin_xyz=display_xyz,
                rotation_rows=_rotation_rows(T_display_tool[0:3, 0:3]),
                axis_length_mm=12.0,
                label=tool_id,
            )
        )

    if registration_loaded and T_robot_tip is not None:
        tip_xyz = tuple(float(T_robot_tip[row, 3]) for row in range(3))
        display_points_by_key["tip"] = tip_xyz
        points.append(
            ScenePoint3D(
                key="tip",
                xyz=tip_xyz,
                color_hex=COLORS.scene_tip,
                label="tip",
                radius_px=6.5,
                outline_hex=COLORS.text_primary,
                shape="diamond",
            )
        )
        axes.append(
            SceneAxes3D(
                key="tip_axes",
                origin_xyz=tip_xyz,
                rotation_rows=_rotation_rows(T_robot_tip[0:3, 0:3]),
                axis_length_mm=10.0,
                label="Tip",
            )
        )

    for key, trail_points in dict(trail_points_by_key or {}).items():
        if key not in display_points_by_key:
            continue
        trail = []
        for point in trail_points:
            if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) < 3:
                continue
            trail.append(tuple(float(value) for value in point[0:3]))
        current = display_points_by_key[key]
        if not trail or trail[-1] != current:
            trail = [*trail, current]
        if len(trail) >= 2:
            polylines.append(
                ScenePolyline3D(
                    key=f"{key}_trail",
                    points_xyz=tuple(trail[-ToolPlotWidget.MAX_TRAIL_POINTS :]),
                    color_hex=_trail_color(key),
                    width_px=1.2,
                )
            )

    scene = SceneModel3D(
        frame_label=frame_label,
        overlay_lines=tuple(overlay_lines),
        points=tuple(points),
        lines=tuple(lines),
        polylines=tuple(polylines),
        axes=tuple(axes),
    )
    return scene, display_points_by_key, frame_label


def _tool_transform_in_display_frame(
    *,
    translation,
    quaternion,
    T_robot_aurora: np.ndarray | None,
    registration_loaded: bool,
) -> np.ndarray | None:
    if translation is None or quaternion is None:
        return None
    T_aurora_tool = make_transform_A_B(tuple(float(value) for value in quaternion), tuple(float(value) for value in translation))
    if registration_loaded and T_robot_aurora is not None:
        return T_robot_aurora @ T_aurora_tool
    return T_aurora_tool


def _tool_overlay_state(tool: Mapping[str, object]) -> str:
    translation = tool.get("translation_mm")
    if translation is None:
        return "missing"
    tracking_state = str(tool.get("tracking_state", "unknown"))
    status = str(tool.get("status", "unknown"))
    return tracking_state if tracking_state == status else f"{tracking_state} ({status})"


def _tool_color(label: str) -> str:
    return {
        "0A": COLORS.scene_live_0a,
        "0B": COLORS.scene_live_0b,
        "tip": COLORS.scene_tip,
    }.get(label, COLORS.surface_border)


def _trail_color(label: str) -> str:
    return {
        "0A": COLORS.scene_live_0a,
        "0B": COLORS.scene_live_0b,
        "tip": COLORS.scene_tip,
    }.get(label, COLORS.text_secondary)


def _rotation_rows(rotation: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    return tuple(tuple(float(value) for value in row.tolist()) for row in np.asarray(rotation, dtype=float))


def _grid_lines(*, size_mm: float, step_mm: float) -> list[SceneLine3D]:
    lines: list[SceneLine3D] = []
    half = float(size_mm) / 2.0
    step = max(5.0, float(step_mm))
    coordinate = -half
    while coordinate <= half + 1e-6:
        lines.append(
            SceneLine3D(
                start_xyz=(-half, coordinate, 0.0),
                end_xyz=(half, coordinate, 0.0),
                color_hex=COLORS.scene_grid,
                width_px=1.0,
            )
        )
        lines.append(
            SceneLine3D(
                start_xyz=(coordinate, -half, 0.0),
                end_xyz=(coordinate, half, 0.0),
                color_hex=COLORS.scene_grid,
                width_px=1.0,
            )
        )
        coordinate += step
    return lines


def _as_transform(matrix) -> np.ndarray | None:
    if matrix is None:
        return None
    arr = np.asarray(matrix, dtype=float)
    if arr.shape != (4, 4):
        return None
    return arr
