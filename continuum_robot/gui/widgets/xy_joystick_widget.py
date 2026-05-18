"""Round drag-pad for issuing bounded XY tendon displacement commands."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from continuum_robot.gui.theme import COLORS


class XyJoystickWidget(QWidget):
    """Round drag-pad that maps a 2D position to a bounded XY vector in cm.

    The puck stays where placed on mouse release; a caller-provided "Center"
    action snaps it back. While dragging, the widget emits `position_changed`
    with the live (x_cm, y_cm) inside the configured circular envelope. It
    does not talk to the servo bus directly — the parent throttles and sends.
    """

    position_changed = Signal(float, float)

    def __init__(self, *, radius_cm: float = 1.0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._radius_cm = max(0.05, float(radius_cm))
        self._target_xy_cm: tuple[float, float] = (0.0, 0.0)
        self._dragging = False
        self._enabled_for_motion = True
        self.setMinimumSize(180, 180)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setCursor(Qt.OpenHandCursor)

    # ----- public API ------------------------------------------------------

    def radius_cm(self) -> float:
        return self._radius_cm

    def set_radius_cm(self, radius_cm: float) -> None:
        new_radius = max(0.05, float(radius_cm))
        if new_radius == self._radius_cm:
            return
        self._radius_cm = new_radius
        # Re-clamp the puck inside the new envelope so the displayed position
        # never sits outside the visible circle.
        x_cm, y_cm = self._target_xy_cm
        r = math.hypot(x_cm, y_cm)
        if r > self._radius_cm > 0:
            scale = self._radius_cm / r
            self._target_xy_cm = (x_cm * scale, y_cm * scale)
            self.position_changed.emit(*self._target_xy_cm)
        self.update()

    def position_cm(self) -> tuple[float, float]:
        return self._target_xy_cm

    def center(self) -> None:
        if self._target_xy_cm == (0.0, 0.0):
            return
        self._target_xy_cm = (0.0, 0.0)
        self.update()
        self.position_changed.emit(0.0, 0.0)

    def set_motion_enabled(self, enabled: bool) -> None:
        if enabled == self._enabled_for_motion:
            return
        self._enabled_for_motion = bool(enabled)
        self.setCursor(Qt.OpenHandCursor if enabled else Qt.ForbiddenCursor)
        self.update()

    # ----- painting --------------------------------------------------------

    def sizeHint(self):  # type: ignore[override]
        from PySide6.QtCore import QSize
        return QSize(220, 220)

    def paintEvent(self, event) -> None:  # noqa: ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        size = float(min(self.width(), self.height()))
        cx = self.width() / 2.0
        cy = self.height() / 2.0
        outer_r = size / 2.0 - 6.0
        envelope_r = outer_r

        # Backdrop disc
        backdrop_color = QColor(COLORS.input_bg)
        painter.setBrush(QBrush(backdrop_color))
        painter.setPen(QPen(QColor(COLORS.surface_border), 1.0))
        painter.drawEllipse(QPointF(cx, cy), outer_r, outer_r)

        # Envelope ring (matches `radius_cm`, drawn at the full disc radius)
        envelope_pen = QPen(QColor(COLORS.button_primary_border), 1.5)
        envelope_pen.setStyle(Qt.DashLine)
        painter.setPen(envelope_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), envelope_r, envelope_r)

        # Cross-axis guides
        axis_pen = QPen(QColor(COLORS.surface_border), 1.0)
        painter.setPen(axis_pen)
        painter.drawLine(QPointF(cx - outer_r, cy), QPointF(cx + outer_r, cy))
        painter.drawLine(QPointF(cx, cy - outer_r), QPointF(cx, cy + outer_r))

        # Axis labels
        painter.setPen(QPen(QColor(COLORS.text_muted), 1.0))
        label_font = painter.font()
        label_font.setPointSizeF(label_font.pointSizeF() - 1)
        painter.setFont(label_font)
        painter.drawText(QRectF(cx + outer_r - 18.0, cy + 2.0, 18.0, 14.0), Qt.AlignRight | Qt.AlignVCenter, "+X")
        painter.drawText(QRectF(cx - outer_r, cy + 2.0, 18.0, 14.0), Qt.AlignLeft | Qt.AlignVCenter, "-X")
        painter.drawText(QRectF(cx + 2.0, cy - outer_r, 18.0, 14.0), Qt.AlignLeft | Qt.AlignVCenter, "+Y")
        painter.drawText(QRectF(cx + 2.0, cy + outer_r - 14.0, 18.0, 14.0), Qt.AlignLeft | Qt.AlignVCenter, "-Y")

        # Radius caption inside the disc
        painter.setPen(QPen(QColor(COLORS.text_subtle), 1.0))
        painter.drawText(
            QRectF(cx - outer_r, cy + outer_r - 18.0, 2.0 * outer_r, 16.0),
            Qt.AlignHCenter | Qt.AlignVCenter,
            f"±{self._radius_cm:.2f} cm",
        )

        # Puck
        x_cm, y_cm = self._target_xy_cm
        puck_x = cx + (x_cm / self._radius_cm) * envelope_r
        puck_y = cy - (y_cm / self._radius_cm) * envelope_r
        if not self._enabled_for_motion:
            puck_color = QColor(COLORS.text_subtle)
            border_color = QColor(COLORS.surface_border)
        else:
            puck_color = QColor(COLORS.button_primary_border)
            border_color = QColor(COLORS.text_primary)
        painter.setBrush(QBrush(puck_color))
        painter.setPen(QPen(border_color, 1.5))
        painter.drawEllipse(QPointF(puck_x, puck_y), 9.0, 9.0)

    # ----- mouse handling --------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: D401
        if not self._enabled_for_motion:
            return
        if event.button() != Qt.LeftButton:
            return
        self._dragging = True
        self.setCursor(Qt.ClosedHandCursor)
        self._update_target_from_event(event.position())

    def mouseMoveEvent(self, event) -> None:  # noqa: D401
        if not self._dragging:
            return
        self._update_target_from_event(event.position())

    def mouseReleaseEvent(self, event) -> None:  # noqa: D401
        if event.button() != Qt.LeftButton or not self._dragging:
            return
        self._dragging = False
        self.setCursor(Qt.OpenHandCursor if self._enabled_for_motion else Qt.ForbiddenCursor)
        # Puck stays where placed; no auto-center.

    # ----- helpers ---------------------------------------------------------

    def _update_target_from_event(self, pos: QPointF) -> None:
        size = float(min(self.width(), self.height()))
        cx = self.width() / 2.0
        cy = self.height() / 2.0
        outer_r = size / 2.0 - 6.0
        if outer_r <= 0:
            return
        rel_x_cm = (pos.x() - cx) / outer_r * self._radius_cm
        rel_y_cm = -(pos.y() - cy) / outer_r * self._radius_cm
        radius = math.hypot(rel_x_cm, rel_y_cm)
        if radius > self._radius_cm and radius > 0:
            scale = self._radius_cm / radius
            rel_x_cm *= scale
            rel_y_cm *= scale
        new_target = (float(rel_x_cm), float(rel_y_cm))
        if new_target == self._target_xy_cm:
            return
        self._target_xy_cm = new_target
        self.update()
        self.position_changed.emit(*self._target_xy_cm)
