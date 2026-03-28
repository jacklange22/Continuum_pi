"""Interactive 3D experiment visualization widget."""

from __future__ import annotations

import os

from PySide6.QtGui import QColor, QVector3D
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

try:
    from PySide6.QtDataVisualization import (
        Q3DScatter,
        QAbstract3DGraph,
        QAbstract3DSeries,
        Q3DTheme,
        QScatter3DSeries,
        QScatterDataItem,
        QScatterDataProxy,
        QValue3DAxis,
    )

    _QT_3D_AVAILABLE = os.environ.get("QT_QPA_PLATFORM", "").strip().lower() not in {"offscreen", "minimal"}
except Exception:  # pragma: no cover - import fallback for minimal environments
    _QT_3D_AVAILABLE = False

from continuum_robot.gui.experiment_visualization import ScatterSeries3D


class Experiment3DWidget(QWidget):
    """Qt-compatible interactive 3D scatter view for experiment samples."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._series: list[ScatterSeries3D] = []
        self._show_labels = False
        self._show_axes = True

        layout = QVBoxLayout(self)
        if not _QT_3D_AVAILABLE:
            self._placeholder = QLabel("3D visualization is unavailable in this environment.")
            layout.addWidget(self._placeholder)
            return

        self._graph = Q3DScatter()
        self._graph.setShadowQuality(QAbstract3DGraph.ShadowQualityNone)
        self._graph.activeTheme().setType(Q3DTheme.ThemeQt)
        self._graph.axisX().setTitle("X (mm)")
        self._graph.axisY().setTitle("Y (mm)")
        self._graph.axisZ().setTitle("Z (mm)")
        self._graph.axisX().setTitleVisible(True)
        self._graph.axisY().setTitleVisible(True)
        self._graph.axisZ().setTitleVisible(True)
        self._container = QWidget.createWindowContainer(self._graph)
        self._container.setMinimumHeight(320)
        layout.addWidget(self._container)

    def set_view_options(self, *, show_axes: bool, show_labels: bool) -> None:
        self._show_axes = bool(show_axes)
        self._show_labels = bool(show_labels)
        if not _QT_3D_AVAILABLE:
            return
        for axis in (self._graph.axisX(), self._graph.axisY(), self._graph.axisZ()):
            axis.setTitleVisible(bool(show_axes))
        for series in self._graph.seriesList():
            series.setItemLabelVisible(bool(show_labels))

    def set_series(self, series_models: list[ScatterSeries3D]) -> None:
        self._series = list(series_models)
        if not _QT_3D_AVAILABLE:
            return
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
        if not points_xyz:
            return
        xs = [point[0] for point in points_xyz]
        ys = [point[1] for point in points_xyz]
        zs = [point[2] for point in points_xyz]
        for axis, values in ((self._graph.axisX(), xs), (self._graph.axisY(), ys), (self._graph.axisZ(), zs)):
            low = min(values)
            high = max(values)
            pad = max(1.0, (high - low) * 0.15)
            axis.setRange(float(low - pad), float(high + pad))


def _mesh_from_name(name: str):
    mesh_name = str(name or "sphere").strip().lower()
    if mesh_name == "cube":
        return QAbstract3DSeries.MeshCube
    return QAbstract3DSeries.MeshSphere
