"""Lightweight interactive 3D scene widget for operator-facing trust graphics."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import QSizePolicy, QWidget

from continuum_robot.gui.theme import COLORS, qcolor


Vec3 = tuple[float, float, float]
Mat3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True)
class ScenePoint3D:
    key: str
    xyz: Vec3
    color_hex: str
    label: str = ""
    radius_px: float = 5.0
    outline_hex: str | None = None
    shape: str = "circle"


@dataclass(frozen=True)
class SceneLine3D:
    start_xyz: Vec3
    end_xyz: Vec3
    color_hex: str
    width_px: float = 1.5
    dashed: bool = False


@dataclass(frozen=True)
class ScenePolyline3D:
    key: str
    points_xyz: tuple[Vec3, ...]
    color_hex: str
    width_px: float = 1.2
    dashed: bool = False


@dataclass(frozen=True)
class SceneAxes3D:
    key: str
    origin_xyz: Vec3
    rotation_rows: Mat3 = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    axis_length_mm: float = 14.0
    label: str = ""


@dataclass(frozen=True)
class SceneAnnotation3D:
    xyz: Vec3
    text: str
    color_hex: str = COLORS.text_primary


@dataclass(frozen=True)
class SceneModel3D:
    frame_label: str = ""
    overlay_lines: tuple[str, ...] = ()
    points: tuple[ScenePoint3D, ...] = ()
    lines: tuple[SceneLine3D, ...] = ()
    polylines: tuple[ScenePolyline3D, ...] = ()
    axes: tuple[SceneAxes3D, ...] = ()
    annotations: tuple[SceneAnnotation3D, ...] = ()


@dataclass(frozen=True)
class SceneViewState:
    yaw_deg: float = -42.0
    pitch_deg: float = -24.0
    zoom: float = 1.0
    pan_x_px: float = 0.0
    pan_y_px: float = 0.0
    target_xyz: Vec3 = (0.0, 0.0, 0.0)


class LightweightScene3DWidget(QWidget):
    """Fast orthographic 3D projection widget with preserved camera state."""

    MIN_ZOOM = 0.08
    MAX_ZOOM = 40.0
    ZOOM_MODIFIER = Qt.ControlModifier

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = SceneModel3D()
        self._view_state = SceneViewState()
        self._auto_fit_pending = True
        self._last_mouse_pos = None
        self._drag_mode: str | None = None
        self.setMinimumHeight(240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

    def set_scene(self, scene: SceneModel3D) -> None:
        if scene == self._scene:
            return
        self._scene = scene
        if self._auto_fit_pending:
            self.reset_view(fit_scene=True)
        else:
            self.update()

    def scene(self) -> SceneModel3D:
        return self._scene

    def view_state(self) -> SceneViewState:
        return self._view_state

    def set_view_state(self, state: SceneViewState) -> None:
        self._view_state = state
        self._auto_fit_pending = False
        self.update()

    def reset_view(self, *, fit_scene: bool = True) -> None:
        state = SceneViewState()
        if fit_scene:
            target, zoom = self._fit_target_and_zoom(state)
            state = SceneViewState(
                yaw_deg=state.yaw_deg,
                pitch_deg=state.pitch_deg,
                zoom=zoom,
                pan_x_px=0.0,
                pan_y_px=0.0,
                target_xyz=target,
            )
        self._view_state = state
        self._auto_fit_pending = False
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # pragma: no cover - exercised in GUI
        self._last_mouse_pos = event.position()
        if event.button() == Qt.LeftButton:
            self._drag_mode = "orbit"
        elif event.button() in {Qt.MiddleButton, Qt.RightButton}:
            self._drag_mode = "pan"
        else:
            self._drag_mode = None
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # pragma: no cover - exercised in GUI
        if self._last_mouse_pos is None or self._drag_mode is None:
            super().mouseMoveEvent(event)
            return
        delta = event.position() - self._last_mouse_pos
        state = self._view_state
        if self._drag_mode == "orbit":
            self._view_state = SceneViewState(
                yaw_deg=state.yaw_deg + float(delta.x()) * 0.45,
                pitch_deg=max(-85.0, min(85.0, state.pitch_deg + float(delta.y()) * 0.35)),
                zoom=state.zoom,
                pan_x_px=state.pan_x_px,
                pan_y_px=state.pan_y_px,
                target_xyz=state.target_xyz,
            )
        elif self._drag_mode == "pan":
            self._view_state = SceneViewState(
                yaw_deg=state.yaw_deg,
                pitch_deg=state.pitch_deg,
                zoom=state.zoom,
                pan_x_px=state.pan_x_px + float(delta.x()),
                pan_y_px=state.pan_y_px + float(delta.y()),
                target_xyz=state.target_xyz,
            )
        self._last_mouse_pos = event.position()
        self._auto_fit_pending = False
        self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # pragma: no cover - exercised in GUI
        self._drag_mode = None
        self._last_mouse_pos = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # pragma: no cover - exercised in GUI
        if not bool(event.modifiers() & self.ZOOM_MODIFIER):
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        factor = 1.1 if delta > 0 else 1.0 / 1.1
        state = self._view_state
        self._view_state = SceneViewState(
            yaw_deg=state.yaw_deg,
            pitch_deg=state.pitch_deg,
            zoom=max(self.MIN_ZOOM, min(self.MAX_ZOOM, state.zoom * factor)),
            pan_x_px=state.pan_x_px,
            pan_y_px=state.pan_y_px,
            target_xyz=state.target_xyz,
        )
        self._auto_fit_pending = False
        self.update()
        event.accept()

    def paintEvent(self, event) -> None:  # pragma: no cover - covered indirectly via widget tests
        _ = event
        painter = QPainter(self)
        painter.fillRect(self.rect(), qcolor(COLORS.surface_bg))
        painter.setRenderHint(QPainter.Antialiasing, True)

        viewport = QRectF(18.0, 16.0, max(40.0, self.width() - 36.0), max(40.0, self.height() - 32.0))
        center = QPointF(
            viewport.center().x() + self._view_state.pan_x_px,
            viewport.center().y() + self._view_state.pan_y_px,
        )
        rotation = _view_rotation_matrix(self._view_state.yaw_deg, self._view_state.pitch_deg)
        target = np.asarray(self._view_state.target_xyz, dtype=float)
        zoom = float(self._view_state.zoom)

        drawables = self._project_scene(rotation=rotation, target=target, center=center, zoom=zoom)
        for drawable in drawables["lines"]:
            pen = QPen(QColor(drawable["color"]), drawable["width"])
            if drawable["dashed"]:
                pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(drawable["start"], drawable["end"])
        for drawable in drawables["points"]:
            painter.setPen(
                QPen(QColor(drawable["outline"]), 1.2)
                if drawable["outline"] is not None
                else Qt.NoPen
            )
            painter.setBrush(QColor(drawable["color"]))
            radius = drawable["radius"]
            point = drawable["point"]
            if drawable["shape"] == "square":
                painter.drawRect(point.x() - radius, point.y() - radius, radius * 2.0, radius * 2.0)
            elif drawable["shape"] == "diamond":
                diamond = [
                    QPointF(point.x(), point.y() - radius),
                    QPointF(point.x() + radius, point.y()),
                    QPointF(point.x(), point.y() + radius),
                    QPointF(point.x() - radius, point.y()),
                ]
                painter.drawPolygon(diamond)
            else:
                painter.drawEllipse(point, radius, radius)
            if drawable["label"]:
                painter.setPen(QPen(qcolor(COLORS.text_primary)))
                painter.drawText(point.x() + radius + 4.0, point.y() - radius - 3.0, drawable["label"])
        for annotation in drawables["annotations"]:
            painter.setPen(QPen(QColor(annotation["color"])))
            painter.drawText(annotation["point"], annotation["text"])

        self._draw_overlay(painter, viewport)

    def _project_scene(self, *, rotation: np.ndarray, target: np.ndarray, center: QPointF, zoom: float) -> dict[str, list[dict]]:
        lines: list[dict] = []
        points: list[dict] = []
        annotations: list[dict] = []

        def _project_xyz(xyz: Vec3) -> tuple[QPointF, float]:
            vec = np.asarray(xyz, dtype=float) - target
            rotated = rotation @ vec
            return (
                QPointF(center.x() + float(rotated[0]) * zoom, center.y() - float(rotated[1]) * zoom),
                float(rotated[2]),
            )

        for axis in self._scene.axes:
            origin = np.asarray(axis.origin_xyz, dtype=float)
            basis = np.asarray(axis.rotation_rows, dtype=float)
            colors = (COLORS.scene_axis_x, COLORS.scene_axis_y, COLORS.scene_axis_z)
            labels = ("X", "Y", "Z")
            for index in range(3):
                end = origin + basis[:, index] * float(axis.axis_length_mm)
                start_point, start_depth = _project_xyz(tuple(origin.tolist()))
                end_point, end_depth = _project_xyz(tuple(end.tolist()))
                lines.append(
                    {
                        "start": start_point,
                        "end": end_point,
                        "depth": (start_depth + end_depth) / 2.0,
                        "color": colors[index],
                        "width": 1.6,
                        "dashed": False,
                    }
                )
                if axis.label and index == 2:
                    annotations.append(
                        {
                            "point": QPointF(end_point.x() + 4.0, end_point.y() - 4.0),
                            "text": axis.label,
                            "color": COLORS.text_primary,
                        }
                    )
                elif not axis.label:
                    annotations.append(
                        {
                            "point": QPointF(end_point.x() + 3.0, end_point.y() - 3.0),
                            "text": labels[index],
                            "color": colors[index],
                        }
                    )

        for polyline in self._scene.polylines:
            if len(polyline.points_xyz) < 2:
                continue
            for start_xyz, end_xyz in zip(polyline.points_xyz[:-1], polyline.points_xyz[1:]):
                start_point, start_depth = _project_xyz(start_xyz)
                end_point, end_depth = _project_xyz(end_xyz)
                lines.append(
                    {
                        "start": start_point,
                        "end": end_point,
                        "depth": (start_depth + end_depth) / 2.0,
                        "color": polyline.color_hex,
                        "width": polyline.width_px,
                        "dashed": polyline.dashed,
                    }
                )

        for line in self._scene.lines:
            start_point, start_depth = _project_xyz(line.start_xyz)
            end_point, end_depth = _project_xyz(line.end_xyz)
            lines.append(
                {
                    "start": start_point,
                    "end": end_point,
                    "depth": (start_depth + end_depth) / 2.0,
                    "color": line.color_hex,
                    "width": line.width_px,
                    "dashed": line.dashed,
                }
            )

        for point in self._scene.points:
            projected_point, depth = _project_xyz(point.xyz)
            points.append(
                {
                    "point": projected_point,
                    "depth": depth,
                    "color": point.color_hex,
                    "label": point.label,
                    "radius": point.radius_px,
                    "outline": point.outline_hex,
                    "shape": point.shape,
                }
            )

        for annotation in self._scene.annotations:
            projected_point, _ = _project_xyz(annotation.xyz)
            annotations.append(
                {
                    "point": QPointF(projected_point.x() + 5.0, projected_point.y() - 5.0),
                    "text": annotation.text,
                    "color": annotation.color_hex,
                }
            )

        lines.sort(key=lambda item: item["depth"])
        points.sort(key=lambda item: item["depth"])
        return {"lines": lines, "points": points, "annotations": annotations}

    def _draw_overlay(self, painter: QPainter, viewport: QRectF) -> None:
        lines = [line for line in self._scene.overlay_lines if str(line).strip()]
        if not self._scene.frame_label and not lines:
            return
        overlay_lines = []
        if self._scene.frame_label:
            overlay_lines.append(f"Frame: {self._scene.frame_label}")
        overlay_lines.extend(lines)
        box_height = 18.0 + (len(overlay_lines) * 18.0)
        box_width = min(max(220.0, viewport.width() * 0.42), viewport.width() - 12.0)
        box = QRectF(viewport.left() + 8.0, viewport.top() + 8.0, box_width, box_height)
        painter.setPen(Qt.NoPen)
        painter.setBrush(qcolor(COLORS.overlay_bg, alpha=220))
        painter.drawRoundedRect(box, 10.0, 10.0)
        painter.setPen(QPen(qcolor(COLORS.text_primary)))
        y = box.top() + 18.0
        for line in overlay_lines:
            painter.drawText(QRectF(box.left() + 10.0, y - 12.0, box.width() - 20.0, 16.0), Qt.TextWordWrap, line)
            y += 18.0

    def _fit_target_and_zoom(self, base_state: SceneViewState) -> tuple[Vec3, float]:
        points = list(_scene_points(self._scene))
        if not points:
            return base_state.target_xyz, 1.0
        points_arr = np.asarray(points, dtype=float)
        target = tuple(float(value) for value in points_arr.mean(axis=0).tolist())
        rotation = _view_rotation_matrix(base_state.yaw_deg, base_state.pitch_deg)
        centered = (rotation @ (points_arr - np.asarray(target, dtype=float)).T).T
        min_xy = centered[:, 0:2].min(axis=0)
        max_xy = centered[:, 0:2].max(axis=0)
        span = np.maximum(max_xy - min_xy, np.asarray([20.0, 20.0], dtype=float))
        viewport_width = max(120.0, float(self.width()) - 80.0)
        viewport_height = max(120.0, float(self.height()) - 80.0)
        zoom = min(viewport_width / span[0], viewport_height / span[1])
        zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, float(zoom)))
        return target, zoom


def _scene_points(scene: SceneModel3D) -> Iterable[Vec3]:
    for point in scene.points:
        yield point.xyz
    for line in scene.lines:
        yield line.start_xyz
        yield line.end_xyz
    for polyline in scene.polylines:
        for point in polyline.points_xyz:
            yield point
    for axis in scene.axes:
        origin = np.asarray(axis.origin_xyz, dtype=float)
        basis = np.asarray(axis.rotation_rows, dtype=float)
        yield axis.origin_xyz
        for index in range(3):
            endpoint = origin + basis[:, index] * float(axis.axis_length_mm)
            yield tuple(float(value) for value in endpoint.tolist())
    for annotation in scene.annotations:
        yield annotation.xyz


def _view_rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    cz = math.cos(yaw)
    sz = math.sin(yaw)
    cx = math.cos(pitch)
    sx = math.sin(pitch)
    rz = np.asarray(
        [
            [cz, -sz, 0.0],
            [sz, cz, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    rx = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, cx, -sx],
            [0.0, sx, cx],
        ],
        dtype=float,
    )
    return rx @ rz
