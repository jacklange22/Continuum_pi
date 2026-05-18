"""Simple 2D top-view selector for registration landmarks.

Two render modes:

* **Selection mode** (the original) -- driven by ``set_landmarks``. Used in the
  main registration tab where the operator picks four landmarks from a
  candidate set. Selected points get a 1..N "selection order" badge.
* **Trial-capture mode** -- driven by ``set_trial_capture_state``. Used in
  the Registration Trial dialog's capture phase. All candidate landmarks are
  drawn with a fixed numeric label (e.g., 1..12 corresponding to L1..L12)
  and a per-landmark state badge: the active capture target is highlighted,
  completed landmarks are dimmed, pending trial landmarks are neutral, and
  non-trial candidates (visible in the YAML but not selected for this trial)
  are drawn very faintly so the operator still sees the full layout.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from continuum_robot.gui.theme import COLORS, qcolor


class RegistrationLandmarkMapWidget(QWidget):
    """Draw candidate registration landmarks in XY and emit click selections."""

    pointToggled = Signal(str)

    _MODE_SELECTION = "selection"
    _MODE_TRIAL_CAPTURE = "trial_capture"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._render_mode: str = self._MODE_SELECTION
        self._points_by_label: dict[str, tuple[float, float, float]] = {}
        self._display_labels: dict[str, str] = {}
        self._enabled_by_label: dict[str, bool] = {}
        self._selected_order: dict[str, int] = {}
        # Trial-capture-mode-only fields.
        self._fixed_numeric_labels: dict[str, int] = {}
        self._active_label: str | None = None
        self._completed_labels: set[str] = set()
        self._trial_labels: set[str] = set()
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
        self._render_mode = self._MODE_SELECTION
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
        # Clear trial-mode state to avoid stale paint.
        self._fixed_numeric_labels = {}
        self._active_label = None
        self._completed_labels = set()
        self._trial_labels = set()
        self.update()

    def set_trial_capture_state(
        self,
        *,
        points_by_label: dict[str, list[float]],
        display_labels: dict[str, str],
        fixed_numeric_labels: dict[str, int],
        trial_labels: list[str],
        active_label: str | None,
        completed_labels: list[str],
    ) -> None:
        """Switch the widget into trial-capture rendering mode.

        Every candidate in ``points_by_label`` is drawn with its fixed numeric
        label (so the operator can always identify "this is point 7 = L7 =
        South Inner"). The visual state per landmark:

        - ``active_label`` -- the landmark currently being captured. Drawn
          with the strongest highlight.
        - ``completed_labels`` -- landmarks whose capture quota has been met.
          Drawn dimmer with a filled marker so the operator sees progress.
        - ``trial_labels`` -- the full set of landmarks selected for this
          trial. Anything in this set but not active or completed is drawn
          as "pending" (neutral outline).
        - Anything in ``points_by_label`` but not in ``trial_labels`` is
          drawn as a faint reference dot (visible context, not part of the
          trial).
        """
        self._render_mode = self._MODE_TRIAL_CAPTURE
        self._points_by_label = {
            str(label): (float(point[0]), float(point[1]), float(point[2]) if len(point) > 2 else 0.0)
            for label, point in points_by_label.items()
            if len(point) >= 2
        }
        self._display_labels = {str(label): str(text) for label, text in display_labels.items()}
        self._fixed_numeric_labels = {str(label): int(value) for label, value in fixed_numeric_labels.items()}
        self._trial_labels = {str(label) for label in trial_labels}
        self._completed_labels = {str(label) for label in completed_labels}
        self._active_label = str(active_label) if active_label is not None else None
        # Reset selection-mode state so paint doesn't accidentally render badges from a prior mode.
        self._enabled_by_label = {}
        self._selected_order = {}
        self.update()

    def point_center_for_label(self, label: str) -> QPoint | None:
        point = self._points_by_label.get(label)
        if point is None:
            return None
        projected = self._project_xy(point[0], point[1])
        return QPoint(int(round(projected.x())), int(round(projected.y())))

    def mousePressEvent(self, event) -> None:  # pragma: no cover - exercised via GUI tests
        # Click-to-toggle is only meaningful in selection mode; trial-capture
        # mode is read-only (the dialog drives state changes via its buttons).
        if self._render_mode != self._MODE_SELECTION:
            return
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
        if self._render_mode == self._MODE_TRIAL_CAPTURE:
            self._paint_trial_capture(painter)
        else:
            self._paint_selection(painter)

    def _paint_selection(self, painter: QPainter) -> None:
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

    def _paint_trial_capture(self, painter: QPainter) -> None:
        # Sort so completed/pending landmarks paint first and the active one
        # paints last (drawn on top, never occluded).
        ordered_items = sorted(
            self._points_by_label.items(),
            key=lambda item: 1 if item[0] == self._active_label else 0,
        )
        for label, (x_mm, y_mm, z_mm) in ordered_items:
            projected = self._project_xy(x_mm, y_mm)
            in_trial = label in self._trial_labels
            is_active = label == self._active_label
            is_completed = label in self._completed_labels

            if is_active:
                radius = 14.0
                fill = qcolor(COLORS.button_primary_border)
                border = qcolor(COLORS.selection_fg)
                number_color = qcolor(COLORS.selection_fg)
                halo_pen = QPen(qcolor(COLORS.button_primary_border), 3)
            elif is_completed:
                radius = 10.0
                fill = qcolor(COLORS.scene_truth)
                border = qcolor(COLORS.text_primary)
                number_color = qcolor(COLORS.text_primary)
                halo_pen = None
            elif in_trial:
                radius = 10.0
                fill = qcolor(COLORS.surface_alt_bg)
                border = qcolor(COLORS.button_primary_border)
                number_color = qcolor(COLORS.text_primary)
                halo_pen = None
            else:
                # Not part of this trial -- faint reference dot so the
                # operator still sees the full 1..N layout.
                radius = 7.0
                fill = qcolor(COLORS.surface_alt_bg)
                fill.setAlpha(110)
                border_color = qcolor(COLORS.text_subtle)
                border_color.setAlpha(140)
                border = border_color
                number_color = qcolor(COLORS.text_subtle)
                halo_pen = None

            # Active landmark gets a halo ring so it pops at a glance.
            if halo_pen is not None:
                painter.setPen(halo_pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(projected, radius + 5.0, radius + 5.0)

            painter.setPen(QPen(border, 2))
            painter.setBrush(fill)
            painter.drawEllipse(projected, radius, radius)

            # Fixed numeric label sits centered inside the dot.
            number = self._fixed_numeric_labels.get(label)
            if number is not None:
                number_font = QFont(painter.font())
                number_font.setPointSize(9 if radius < 12.0 else 10)
                number_font.setBold(is_active or is_completed)
                painter.setFont(number_font)
                painter.setPen(number_color)
                painter.drawText(
                    QRectF(projected.x() - radius, projected.y() - radius, radius * 2.0, radius * 2.0),
                    Qt.AlignCenter,
                    str(number),
                )

            # Caption (display name) drawn to the right of the dot.
            caption_color = qcolor(COLORS.text_primary)
            if not in_trial:
                caption_color = qcolor(COLORS.text_subtle)
                caption_color.setAlpha(180)
            elif is_completed:
                caption_color = qcolor(COLORS.text_muted)
            painter.setPen(caption_color)
            caption = self._display_labels.get(label, label)
            suffix = ""
            if is_active:
                suffix = "  · active"
            elif is_completed:
                suffix = "  · done"
            painter.drawText(
                QPointF(projected.x() + radius + 6.0, projected.y() - 6.0),
                f"{caption} ({z_mm:.0f}){suffix}",
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
