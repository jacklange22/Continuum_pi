"""Experiment visualization widget with safe native-3D fallback behavior."""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QVector3D
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from continuum_robot.gui.theme import COLORS, qcolor

try:
    from PySide6.QtDataVisualization import Q3DScatter, QAbstract3DSeries, QScatter3DSeries, QScatterDataItem, QScatterDataProxy

    _QT_3D_IMPORT_OK = True
except Exception:  # pragma: no cover - import fallback for minimal environments
    _QT_3D_IMPORT_OK = False

from continuum_robot.gui.experiment_visualization import ScatterSeries3D


ENV_VISUALIZATION_MODE = "CONTINUUM_VISUALIZATION_MODE"
ENV_SAFE_EFFECTS = "CONTINUUM_VISUALIZATION_SAFE_EFFECTS"

VIS_MODE_AUTO = "auto"
VIS_MODE_NATIVE_3D = "3d"
VIS_MODE_PROJECTION = "2d"
VIS_MODE_PLACEHOLDER = "placeholder"

BACKEND_NATIVE_3D = "native_3d"
BACKEND_PROJECTION = "projection"
BACKEND_PLACEHOLDER = "placeholder"

_HEADLESS_QPA = {"offscreen", "minimal"}


def resolve_visualization_backend(
    *,
    requested_mode: str = VIS_MODE_AUTO,
    safe_effects: bool = True,
    platform_name: str | None = None,
    qpa_platform: str | None = None,
    qt_3d_import_ok: bool = _QT_3D_IMPORT_OK,
) -> tuple[str, str]:
    """Return the safest visualization backend for the current runtime."""
    env_mode = os.environ.get(ENV_VISUALIZATION_MODE, "").strip().lower()
    env_safe = os.environ.get(ENV_SAFE_EFFECTS, "").strip().lower()
    if env_mode:
        requested_mode = env_mode
    if env_safe in {"0", "false", "no"}:
        safe_effects = False
    elif env_safe in {"1", "true", "yes"}:
        safe_effects = True

    mode = str(requested_mode or VIS_MODE_AUTO).strip().lower()
    platform_name = str(platform_name or sys.platform).strip().lower()
    qpa = str(qpa_platform or os.environ.get("QT_QPA_PLATFORM", "")).strip().lower()

    if qpa in _HEADLESS_QPA:
        return BACKEND_PLACEHOLDER, "Headless Qt platform detected. Using a stable non-OpenGL placeholder."
    if mode == VIS_MODE_PLACEHOLDER:
        return BACKEND_PLACEHOLDER, "Visualization placeholder mode is forced by configuration."
    if mode == VIS_MODE_PROJECTION:
        return BACKEND_PROJECTION, "Projection mode is forced by configuration."
    if not qt_3d_import_ok:
        return BACKEND_PROJECTION, "QtDataVisualization is unavailable. Using the projection viewer instead."
    if platform_name.startswith("darwin"):
        return BACKEND_PROJECTION, "Native Qt 3D is disabled on macOS because the QtDataVisualization stack is unstable."
    if mode not in {VIS_MODE_AUTO, VIS_MODE_NATIVE_3D}:
        return BACKEND_PROJECTION, f"Unknown visualization mode '{mode}'. Falling back to the projection viewer."
    return BACKEND_NATIVE_3D, (
        "Native 3D visualization is active with conservative QtDataVisualization settings."
        if safe_effects
        else "Native 3D visualization is active."
    )


class Experiment3DWidget(QWidget):
    """Qt-compatible experiment view that prefers safe 3D and degrades gracefully."""

    def __init__(self, *, requested_mode: str = VIS_MODE_AUTO, safe_effects: bool = True, parent=None) -> None:
        super().__init__(parent)
        self._series: list[ScatterSeries3D] = []
        self._show_labels = False
        self._show_axes = True
        self._requested_mode = requested_mode
        self._safe_effects = bool(safe_effects)
        self._backend_mode, self._backend_message = resolve_visualization_backend(
            requested_mode=requested_mode,
            safe_effects=safe_effects,
        )
        self._graph = None
        self._container = None
        self._projection = None
        self._placeholder = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.mode_label = QLabel(_backend_display_name(self._backend_mode))
        self.mode_label.setWordWrap(True)
        self.mode_label.setToolTip(self._backend_message)
        self.mode_label.setStyleSheet(
            f"padding: 4px 10px; border-radius: 999px; background: {COLORS.status_bg}; "
            f"color: {COLORS.text_secondary}; font-weight: 600;"
        )
        layout.addWidget(self.mode_label)

        if self._backend_mode == BACKEND_NATIVE_3D:
            self._build_native_3d(layout)
        elif self._backend_mode == BACKEND_PROJECTION:
            self._build_projection(layout)
        else:
            self._build_placeholder(layout)

        self.legend_label = QLabel("No samples loaded.")
        self.legend_label.setWordWrap(True)
        self.legend_label.setStyleSheet(f"color: {COLORS.text_muted}; padding: 0 2px;")
        layout.addWidget(self.legend_label)

    @property
    def backend_mode(self) -> str:
        return self._backend_mode

    def set_view_options(self, *, show_axes: bool, show_labels: bool) -> None:
        self._show_axes = bool(show_axes)
        self._show_labels = bool(show_labels)
        if self._graph is not None:
            for axis in (self._graph.axisX(), self._graph.axisY(), self._graph.axisZ()):
                if axis is not None:
                    axis.setTitleVisible(bool(show_axes))
            for series in self._graph.seriesList():
                series.setItemLabelVisible(bool(show_labels))
        if self._projection is not None:
            self._projection.set_view_options(show_axes=show_axes, show_labels=show_labels)

    def set_series(self, series_models: list[ScatterSeries3D]) -> None:
        self._series = list(series_models)
        self._update_legend()
        if self._graph is not None:
            self._render_native_series()
            return
        if self._projection is not None:
            self._projection.set_series(self._series)
            return
        if self._placeholder is not None:
            total_points = sum(len(model.points_xyz) for model in self._series)
            self._placeholder.setText(
                "Visualization is not available in this session.\n"
                f"Loaded points: {total_points}."
            )

    def save_screenshot(self, path: str) -> bool:
        target = str(path).strip()
        if not target:
            return False
        surface = self._container or self._projection or self
        return surface.grab().save(target)

    def _build_native_3d(self, layout: QVBoxLayout) -> None:
        # Keep the native 3D setup intentionally minimal. The QtDataVisualization
        # bindings are fragile on some platforms, and calls like setShadowQuality()
        # have caused native crashes in development.
        self._graph = Q3DScatter()
        self._container = QWidget.createWindowContainer(self._graph)
        self._container.setMinimumHeight(320)
        self._container.setFocusPolicy(Qt.StrongFocus)
        for axis, title in (
            (self._graph.axisX(), "X (mm)"),
            (self._graph.axisY(), "Y (mm)"),
            (self._graph.axisZ(), "Z (mm)"),
        ):
            if axis is not None:
                axis.setTitle(title)
                axis.setTitleVisible(True)
        layout.addWidget(self._container)

    def _build_projection(self, layout: QVBoxLayout) -> None:
        self._projection = _ProjectionCanvas()
        self._projection.setMinimumHeight(320)
        layout.addWidget(self._projection)

    def _build_placeholder(self, layout: QVBoxLayout) -> None:
        self._placeholder = QLabel("Visualization is unavailable in this environment.")
        self._placeholder.setMinimumHeight(320)
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(
            f"border: 1px dashed {COLORS.surface_border}; border-radius: 14px; "
            f"background: {COLORS.surface_bg}; color: {COLORS.text_muted}; padding: 24px;"
        )
        layout.addWidget(self._placeholder)

    def _render_native_series(self) -> None:
        for series in list(self._graph.seriesList()):
            self._graph.removeSeries(series)
        all_points: list[tuple[float, float, float]] = []
        for model in self._series:
            if not model.points_xyz:
                continue
            proxy = QScatterDataProxy()
            items = [QScatterDataItem(QVector3D(float(x), float(y), float(z))) for x, y, z in model.points_xyz]
            proxy.resetArray(items)
            series = QScatter3DSeries(proxy)
            series.setBaseColor(QColor(model.color_hex))
            series.setItemSize(float(model.point_size))
            series.setItemLabelFormat(f"{model.name}\nX=@xLabel\nY=@yLabel\nZ=@zLabel")
            series.setItemLabelVisible(bool(self._show_labels))
            series.setMesh(_mesh_from_name(model.mesh))
            self._graph.addSeries(series)
            all_points.extend(model.points_xyz)
        self._update_axes(all_points)

    def _update_axes(self, points_xyz: list[tuple[float, float, float]]) -> None:
        if self._graph is None or not points_xyz:
            return
        xs = [point[0] for point in points_xyz]
        ys = [point[1] for point in points_xyz]
        zs = [point[2] for point in points_xyz]
        for axis, values in ((self._graph.axisX(), xs), (self._graph.axisY(), ys), (self._graph.axisZ(), zs)):
            if axis is None:
                continue
            low = min(values)
            high = max(values)
            pad = max(1.0, (high - low) * 0.15)
            axis.setRange(float(low - pad), float(high + pad))

    def _update_legend(self) -> None:
        if not self._series:
            self.legend_label.setText("No samples loaded.")
            return
        chips = [
            f'<span style="color:{model.color_hex}; font-weight:600;">&#9632;</span> {model.name}'
            for model in self._series
            if model.points_xyz
        ]
        if not chips:
            self.legend_label.setText("No visible sample groups are available yet.")
            return
        self.legend_label.setText("Series: " + " &nbsp;&nbsp; ".join(chips))


class _ProjectionCanvas(QWidget):
    """Stable 2D projection fallback for experiment point clouds."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._series: list[ScatterSeries3D] = []
        self._show_axes = True
        self._show_labels = False
        self.setMinimumHeight(360)

    def set_series(self, series_models: list[ScatterSeries3D]) -> None:
        self._series = list(series_models)
        self.update()

    def set_view_options(self, *, show_axes: bool, show_labels: bool) -> None:
        self._show_axes = bool(show_axes)
        self._show_labels = bool(show_labels)
        self.update()

    def paintEvent(self, event) -> None:  # pragma: no cover - exercised through GUI tests only
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), qcolor(COLORS.surface_bg))

        left_rect = QRectF(14.0, 12.0, max(40.0, (self.width() - 42.0) / 2.0), self.height() - 24.0)
        right_rect = QRectF(left_rect.right() + 12.0, 12.0, left_rect.width(), left_rect.height())
        self._draw_panel(painter, left_rect, "Top View (X/Y)", axis_a=0, axis_b=1)
        self._draw_panel(painter, right_rect, "Side View (X/Z)", axis_a=0, axis_b=2)

    def _draw_panel(self, painter: QPainter, rect: QRectF, title: str, *, axis_a: int, axis_b: int) -> None:
        painter.setPen(QPen(qcolor(COLORS.surface_border), 1))
        painter.setBrush(qcolor(COLORS.surface_alt_bg))
        painter.drawRoundedRect(rect, 12.0, 12.0)
        painter.setPen(QPen(qcolor(COLORS.text_primary), 1))
        painter.drawText(rect.adjusted(14.0, 10.0, -8.0, -8.0), f"{title}")

        all_points = [
            point
            for model in self._series
            for point in model.points_xyz
        ]
        inner = rect.adjusted(16.0, 28.0, -16.0, -16.0)
        if not all_points:
            painter.setPen(QPen(qcolor(COLORS.text_muted), 1))
            painter.drawText(inner, Qt.AlignCenter, "No sample points to project yet.")
            return

        a_values = [float(point[axis_a]) for point in all_points]
        b_values = [float(point[axis_b]) for point in all_points]
        min_a, max_a = min(a_values), max(a_values)
        min_b, max_b = min(b_values), max(b_values)
        pad_a = max(1.0, (max_a - min_a) * 0.1)
        pad_b = max(1.0, (max_b - min_b) * 0.1)
        min_a -= pad_a
        max_a += pad_a
        min_b -= pad_b
        max_b += pad_b

        if self._show_axes:
            painter.setPen(QPen(qcolor(COLORS.chart_grid), 1))
            mid_x = inner.left() + inner.width() / 2.0
            mid_y = inner.top() + inner.height() / 2.0
            painter.drawLine(QPointF(inner.left(), mid_y), QPointF(inner.right(), mid_y))
            painter.drawLine(QPointF(mid_x, inner.top()), QPointF(mid_x, inner.bottom()))

        for model in self._series:
            if not model.points_xyz:
                continue
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(model.color_hex))
            for point in model.points_xyz:
                px = _scale(inner.left(), inner.right(), float(point[axis_a]), min_a, max_a)
                py = _scale(inner.bottom(), inner.top(), float(point[axis_b]), min_b, max_b)
                radius = max(3.0, float(model.point_size) * 30.0)
                painter.drawEllipse(QPointF(px, py), radius, radius)

        if self._show_labels:
            painter.setPen(QPen(qcolor(COLORS.text_secondary), 1))
            painter.drawText(inner.adjusted(4.0, 4.0, -4.0, -4.0), Qt.AlignBottom | Qt.AlignRight, f"{axis_a}/{axis_b}")


def _backend_display_name(mode: str) -> str:
    return {
        BACKEND_NATIVE_3D: "Interactive 3D",
        BACKEND_PROJECTION: "Projection View",
        BACKEND_PLACEHOLDER: "Visualization Placeholder",
    }.get(mode, "Visualization")


def _scale(low_px: float, high_px: float, value: float, low_value: float, high_value: float) -> float:
    if high_value <= low_value:
        return (low_px + high_px) / 2.0
    fraction = (value - low_value) / (high_value - low_value)
    return low_px + (high_px - low_px) * fraction


def _mesh_from_name(name: str):
    mesh_name = str(name or "sphere").strip().lower()
    if mesh_name == "cube":
        return QAbstract3DSeries.MeshCube
    return QAbstract3DSeries.MeshSphere
