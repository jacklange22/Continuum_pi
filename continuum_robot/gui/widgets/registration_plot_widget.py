"""Simple registration landmark/capture plot."""

from __future__ import annotations

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget


class RegistrationPlotWidget(QWidget):
    """Draw nominal landmarks and captured samples in XY."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._nominal: dict[str, tuple[float, float]] = {}
        self._captured: dict[str, list[tuple[float, float]]] = {}
        self.setMinimumHeight(180)

    def set_data(
        self,
        nominal: dict[str, tuple[float, float]],
        captured: dict[str, list[tuple[float, float]]],
    ) -> None:
        self._nominal = dict(nominal)
        self._captured = {key: list(value) for key, value in captured.items()}
        self.update()

    def paintEvent(self, event) -> None:
        _ = event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#f8fafc"))
        painter.setRenderHint(QPainter.Antialiasing, True)

        center = QPointF(self.width() / 2.0, self.height() / 2.0)
        painter.setPen(QPen(QColor("#d1d5db"), 1))
        painter.drawLine(0, center.y(), self.width(), center.y())
        painter.drawLine(center.x(), 0, center.x(), self.height())

        scale = min(self.width(), self.height()) / 80.0
        for label, (x, y) in self._nominal.items():
            px = center.x() + x * scale
            py = center.y() - y * scale
            painter.setBrush(QColor("#2563eb"))
            painter.setPen(Qt.NoPen)
            painter.drawRect(px - 5, py - 5, 10, 10)
            painter.setPen(QPen(QColor("#1f2937")))
            painter.drawText(px + 6.0, py - 6.0, label)

        painter.setBrush(QColor("#dc2626"))
        painter.setPen(Qt.NoPen)
        for points in self._captured.values():
            for x, y in points:
                px = center.x() + x * scale
                py = center.y() - y * scale
                painter.drawEllipse(QPointF(px, py), 4.0, 4.0)
