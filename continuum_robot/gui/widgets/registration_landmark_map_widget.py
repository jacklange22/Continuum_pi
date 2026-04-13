"""Simple 2D top-view selector for registration landmarks."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from continuum_robot.gui.theme import COLORS, qcolor


class RegistrationLandmarkMapWidget(QWidget):
    """Draw candidate registration landmarks in XY and emit click selections."""

    pointToggled = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._points_by_label: dict[str, tuple[float, float, float]] = {}
        self._display_labels: dict[str, str] = {}
        self._enabled_by_label: dict[str, bool] = {}
        self._selected_order: dict[str, int] = {}
        self.setMinimumHeight(220)
        self.setMouseTracking(True)

    def set_landmarks(
        self,
        *,
        points_by_label: dict[str, list[float]],
        display_labels: dict[str, str],
        enabled_by_label: dict[str, bool],
        selected_labels: list[str],
    ) -> None:
        self._points_by_label = {
            str(label): (float(point[0]), float(point[1]), float(point[2]) if len(point) > 2 else 0.0)
            for label, point in points_by_label.items()
            if len(point) >= 2
        }
        self._display_labels = {str(label): str(text) for label, text in display_labels.items()}
        self._enabled_by_label = {str(label): bool(enabled) for label, enabled in enabled_by_label.items()}
        self._selected_order = {
            str(label): index + 1
            for index, label in enumerate(selected_labels)
        }
        self.update()

    def point_center_for_label(self, label: str) -> QPoint | None:
        point = self._points_by_label.get(label)
        if point is None:
            return None
        projected = self._project_xy(point[0], point[1])
        return QPoint(int(round(projected.x())), int(round(projected.y())))

    def mousePressEvent(self, event) -> None:  # pragma: no cover - exercised via GUI tests
        if event.button() != Qt.LeftButton:
            return
        label = self._label_at(event.position())
        if label is not None:
            self.pointToggled.emit(label)

    def paintEvent(self, event) -> None:  # pragma: no cover - exercised via GUI tests
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), qcolor(COLORS.surface_bg))

        plot_rect = self._plot_rect()
        painter.setPen(QPen(qcolor(COLORS.surface_border), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(plot_rect, 14.0, 14.0)

        if not self._points_by_label:
            painter.setPen(qcolor(COLORS.text_muted))
            painter.drawText(self.rect(), Qt.AlignCenter, "No registration landmarks configured.")
            return

        self._draw_axes(painter, plot_rect)
        for label, (x_mm, y_mm, z_mm) in self._points_by_label.items():
            projected = self._project_xy(x_mm, y_mm)
            enabled = self._enabled_by_label.get(label, True)
            selected_order = self._selected_order.get(label)
            radius = 11.0 if selected_order is not None else 8.0

            if not enabled:
                fill = qcolor(COLORS.surface_alt_bg)
                border = qcolor(COLORS.text_subtle)
                text = qcolor(COLORS.text_muted)
            elif selected_order is not None:
                fill = qcolor(COLORS.selection_bg)
                border = qcolor(COLORS.button_primary_border)
                text = qcolor(COLORS.selection_fg)
            else:
                fill = qcolor(COLORS.surface_alt_bg)
                border = qcolor(COLORS.scene_truth)
                text = qcolor(COLORS.text_primary)

            painter.setPen(QPen(border, 2))
            painter.setBrush(fill)
            painter.drawEllipse(projected, radius, radius)

            label_font = QFont(painter.font())
            label_font.setPointSize(9)
            label_font.setBold(selected_order is not None)
            painter.setFont(label_font)
            if selected_order is not None:
                painter.setPen(text)
                painter.drawText(
                    QRectF(projected.x() - radius, projected.y() - radius, radius * 2.0, radius * 2.0),
                    Qt.AlignCenter,
                    str(selected_order),
                )

            painter.setPen(qcolor(COLORS.text_primary if enabled else COLORS.text_subtle))
            caption = self._display_labels.get(label, label)
            painter.drawText(
                QPointF(projected.x() + radius + 6.0, projected.y() - 6.0),
                f"{caption} ({z_mm:.0f})",
            )

    def _draw_axes(self, painter: QPainter, plot_rect: QRectF) -> None:
        center = QPointF(plot_rect.center().x(), plot_rect.center().y())
        painter.setPen(QPen(qcolor(COLORS.chart_grid), 1, Qt.DashLine))
        painter.drawLine(plot_rect.left(), center.y(), plot_rect.right(), center.y())
        painter.drawLine(center.x(), plot_rect.top(), center.x(), plot_rect.bottom())
        painter.setPen(qcolor(COLORS.text_muted))
        painter.drawText(QPointF(plot_rect.left() + 12.0, plot_rect.top() + 18.0), "Top view (X/Y)")
        painter.drawText(QPointF(plot_rect.right() - 36.0, center.y() - 8.0), "+X")
        painter.drawText(QPointF(center.x() + 6.0, plot_rect.top() + 18.0), "+Y")

    def _label_at(self, point: QPointF) -> str | None:
        for label, (x_mm, y_mm, _z_mm) in self._points_by_label.items():
            if not self._enabled_by_label.get(label, True):
                continue
            projected = self._project_xy(x_mm, y_mm)
            if (projected - point).manhattanLength() <= 16.0:
                return label
        return None

    def _project_xy(self, x_mm: float, y_mm: float) -> QPointF:
        plot_rect = self._plot_rect()
        xs = [point[0] for point in self._points_by_label.values()]
        ys = [point[1] for point in self._points_by_label.values()]
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)
        margin = 26.0
        usable_width = max(plot_rect.width() - margin * 2.0, 1.0)
        usable_height = max(plot_rect.height() - margin * 2.0, 1.0)
        scale = min(usable_width / span_x, usable_height / span_y)
        left = plot_rect.left() + (plot_rect.width() - span_x * scale) / 2.0
        bottom = plot_rect.bottom() - (plot_rect.height() - span_y * scale) / 2.0
        px = left + (x_mm - min_x) * scale
        py = bottom - (y_mm - min_y) * scale
        return QPointF(px, py)

    def _plot_rect(self) -> QRectF:
        return QRectF(10.0, 10.0, max(self.width() - 20.0, 1.0), max(self.height() - 20.0, 1.0))
