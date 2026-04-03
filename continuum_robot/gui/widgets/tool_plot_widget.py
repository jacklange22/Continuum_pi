"""Simple live isometric tool plot for operator feedback."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget


class ToolPlotWidget(QWidget):
    """Draw tracker tool and tip positions in a lightweight isometric 3D view."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._points: dict[str, tuple[float, float, float]] = {}
        self.setMinimumHeight(240)

    def set_points(self, points: dict[str, tuple[float, float] | tuple[float, float, float]]) -> None:
        normalized: dict[str, tuple[float, float, float]] = {}
        for label, coords in points.items():
            if len(coords) >= 3:
                normalized[str(label)] = (float(coords[0]), float(coords[1]), float(coords[2]))
            else:
                normalized[str(label)] = (float(coords[0]), float(coords[1]), 0.0)
        self._points = normalized
        self.update()

    def paintEvent(self, event) -> None:
        _ = event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#f7f7f5"))
        painter.setRenderHint(QPainter.Antialiasing, True)

        viewport = QRectF(24.0, 20.0, max(40.0, self.width() - 48.0), max(40.0, self.height() - 40.0))
        projected_points = {
            label: self._project_point(*coords)
            for label, coords in self._points.items()
        }
        bounds = self._projected_bounds(projected_points.values())
        scale = self._projection_scale(bounds, viewport)
        center = QPointF(viewport.center().x(), viewport.center().y())
        self._draw_projection_axes(painter, center, scale)
        self._draw_projection_bounds(painter, center, scale, self._points)
        colors = {
            "0A": QColor("#0d6e6e"),
            "0B": QColor("#d97706"),
            "tip": QColor("#b91c1c"),
        }
        for label, (px_world, py_world) in projected_points.items():
            px = center.x() + px_world * scale
            py = center.y() - py_world * scale
            painter.setPen(Qt.NoPen)
            painter.setBrush(colors.get(label, QColor("#374151")))
            painter.drawEllipse(QPointF(px, py), 6.0, 6.0)
            painter.setPen(QPen(QColor("#1f2937")))
            painter.drawText(px + 8.0, py - 8.0, label)
        painter.setPen(QPen(QColor("#64748b")))
        painter.drawText(18.0, self.height() - 14.0, "Isometric view (mm)")

    @staticmethod
    def _project_point(x_mm: float, y_mm: float, z_mm: float) -> tuple[float, float]:
        screen_x = (float(x_mm) - float(y_mm)) * 0.8660254
        screen_y = float(z_mm) + (float(x_mm) + float(y_mm)) * 0.35
        return screen_x, screen_y

    @classmethod
    def _projected_bounds(cls, points: list[tuple[float, float]] | tuple[tuple[float, float], ...]) -> tuple[float, float, float, float]:
        projected = list(points)
        if not projected:
            return (-40.0, 40.0, -40.0, 40.0)
        min_x = min(point[0] for point in projected)
        max_x = max(point[0] for point in projected)
        min_y = min(point[1] for point in projected)
        max_y = max(point[1] for point in projected)
        if max_x - min_x < 40.0:
            center_x = (min_x + max_x) / 2.0
            min_x = center_x - 20.0
            max_x = center_x + 20.0
        if max_y - min_y < 40.0:
            center_y = (min_y + max_y) / 2.0
            min_y = center_y - 20.0
            max_y = center_y + 20.0
        return (min_x - 10.0, max_x + 10.0, min_y - 10.0, max_y + 10.0)

    @staticmethod
    def _projection_scale(bounds: tuple[float, float, float, float], viewport: QRectF) -> float:
        min_x, max_x, min_y, max_y = bounds
        span_x = max(1.0, max_x - min_x)
        span_y = max(1.0, max_y - min_y)
        return min(viewport.width() / span_x, viewport.height() / span_y)

    @classmethod
    def _draw_projection_axes(cls, painter: QPainter, center: QPointF, scale: float) -> None:
        origin = center
        axes = {
            "X": ((0.0, 0.0, 0.0), (35.0, 0.0, 0.0), QColor("#2563eb")),
            "Y": ((0.0, 0.0, 0.0), (0.0, 35.0, 0.0), QColor("#10b981")),
            "Z": ((0.0, 0.0, 0.0), (0.0, 0.0, 35.0), QColor("#dc2626")),
        }
        for label, (start, end, color) in axes.items():
            start_x, start_y = cls._project_point(*start)
            end_x, end_y = cls._project_point(*end)
            painter.setPen(QPen(color, 1.5))
            painter.drawLine(
                QPointF(origin.x() + start_x * scale, origin.y() - start_y * scale),
                QPointF(origin.x() + end_x * scale, origin.y() - end_y * scale),
            )
            painter.drawText(
                origin.x() + end_x * scale + 4.0,
                origin.y() - end_y * scale - 4.0,
                label,
            )

    @classmethod
    def _draw_projection_bounds(
        cls,
        painter: QPainter,
        center: QPointF,
        scale: float,
        points: dict[str, tuple[float, float, float]],
    ) -> None:
        if points:
            xs = [coords[0] for coords in points.values()]
            ys = [coords[1] for coords in points.values()]
            zs = [coords[2] for coords in points.values()]
            min_x, max_x = min(xs) - 10.0, max(xs) + 10.0
            min_y, max_y = min(ys) - 10.0, max(ys) + 10.0
            min_z, max_z = min(zs) - 10.0, max(zs) + 10.0
        else:
            min_x = min_y = min_z = -45.0
            max_x = max_y = max_z = 45.0
        corners_3d = [
            (min_x, min_y, min_z),
            (max_x, min_y, min_z),
            (max_x, max_y, min_z),
            (min_x, max_y, min_z),
            (min_x, min_y, max_z),
            (max_x, min_y, max_z),
            (max_x, max_y, max_z),
            (min_x, max_y, max_z),
        ]
        projected = [cls._project_point(*corner) for corner in corners_3d]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        painter.setPen(QPen(QColor("#d1d5db"), 1))
        for start, end in edges:
            sx, sy = projected[start]
            ex, ey = projected[end]
            painter.drawLine(
                QPointF(center.x() + sx * scale, center.y() - sy * scale),
                QPointF(center.x() + ex * scale, center.y() - ey * scale),
            )
