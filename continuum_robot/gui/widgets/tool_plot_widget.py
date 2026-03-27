"""Simple live XY tool plot for operator feedback."""

from __future__ import annotations

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget


class ToolPlotWidget(QWidget):
    """Draw tracker tool and tip positions on a simple XY plane."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._points: dict[str, tuple[float, float]] = {}
        self.setMinimumHeight(180)

    def set_points(self, points: dict[str, tuple[float, float]]) -> None:
        self._points = dict(points)
        self.update()

    def paintEvent(self, event) -> None:
        _ = event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#f7f7f5"))
        painter.setRenderHint(QPainter.Antialiasing, True)

        center = QPointF(self.width() / 2.0, self.height() / 2.0)
        painter.setPen(QPen(QColor("#c8c4bd"), 1))
        painter.drawLine(0, center.y(), self.width(), center.y())
        painter.drawLine(center.x(), 0, center.x(), self.height())

        scale = min(self.width(), self.height()) / 80.0
        colors = {
            "0A": QColor("#0d6e6e"),
            "0B": QColor("#d97706"),
            "tip": QColor("#b91c1c"),
        }
        for label, (x, y) in self._points.items():
            px = center.x() + x * scale
            py = center.y() - y * scale
            painter.setPen(Qt.NoPen)
            painter.setBrush(colors.get(label, QColor("#374151")))
            painter.drawEllipse(QPointF(px, py), 6.0, 6.0)
            painter.setPen(QPen(QColor("#1f2937")))
            painter.drawText(px + 8.0, py - 8.0, label)
