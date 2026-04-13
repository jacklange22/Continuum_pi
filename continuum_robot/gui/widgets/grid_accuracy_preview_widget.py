"""Lightweight 2D preview for the Aurora aligned-grid validation workflow."""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from continuum_robot.gui.theme import COLORS, qcolor


LOGGER = logging.getLogger(__name__)


class GridAccuracyPreviewWidget(QWidget):
    """Display-only aligned-grid overview for the aurora_grid_accuracy page."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._truth_catalog: list[dict[str, Any]] = []
        self._metrics: dict[str, Any] = {}
        self._selected_target_index = 0
        self._expected_samples = 1
        self._signature: tuple[Any, ...] | None = None
        self.setMinimumHeight(320)

    def sizeHint(self) -> QSize:
        return QSize(720, 360)

    def set_preview(
        self,
        *,
        truth_catalog: list[dict[str, Any]],
        metrics: dict[str, Any],
        selected_target_index: int,
        expected_samples: int,
    ) -> None:
        per_point = metrics.get("per_point_metrics", {}) or {}
        signature = (
            int(selected_target_index),
            int(expected_samples),
            tuple(
                (
                    str(entry.get("label", "")),
                    tuple(float(value) for value in entry.get("truth_point_mm", []) or []),
                    tuple(float(value) for value in (per_point.get(str(entry.get("label", "")), {}) or {}).get("aligned_centroid_truth_mm", []) or []),
                    int((per_point.get(str(entry.get("label", "")), {}) or {}).get("raw_sample_count", 0) or 0),
                    int((per_point.get(str(entry.get("label", "")), {}) or {}).get("outlier_count", 0) or 0),
                    (per_point.get(str(entry.get("label", "")), {}) or {}).get("residual_mm"),
                    (per_point.get(str(entry.get("label", "")), {}) or {}).get("sample_spread_rms_mm"),
                )
                for entry in truth_catalog
            ),
            bool(metrics.get("alignment_ready")),
            str(metrics.get("alignment_ready_reason", "")),
        )
        if signature == self._signature:
            return
        self._signature = signature
        self._truth_catalog = list(truth_catalog)
        self._metrics = dict(metrics or {})
        self._selected_target_index = int(selected_target_index)
        self._expected_samples = max(1, int(expected_samples))
        self.update()

    def paintEvent(self, event) -> None:  # pragma: no cover - exercised indirectly in widget tests
        _ = event
        painter = QPainter()
        if not painter.begin(self):
            return
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            self._paint_contents(painter)
        except Exception:
            LOGGER.exception("Grid accuracy preview paint failed.")
            if painter.isActive():
                self._paint_failure_fallback(painter)
        finally:
            if painter.isActive():
                painter.end()

    def _paint_contents(self, painter: QPainter) -> None:
        outer_rect = self.rect().adjusted(0, 0, -1, -1)
        painter.setPen(QPen(qcolor(COLORS.surface_border), 1.0))
        painter.setBrush(qcolor(COLORS.surface_bg))
        painter.drawRoundedRect(outer_rect, 14.0, 14.0)

        header_rect = QRectF(16.0, 14.0, max(10.0, outer_rect.width() - 32.0), 42.0)
        painter.setPen(qcolor(COLORS.text_primary))
        title_font = QFont()
        title_font.setPointSize(11)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.drawText(header_rect, Qt.AlignLeft | Qt.AlignTop, "Live Aligned-Grid Preview")

        body_font = QFont()
        body_font.setPointSize(9)
        painter.setFont(body_font)
        painter.setPen(qcolor(COLORS.text_secondary))
        summary_text = self._header_summary()
        painter.drawText(header_rect.adjusted(0.0, 18.0, 0.0, 0.0), Qt.AlignLeft | Qt.AlignTop, summary_text)

        scene_rect = QRectF(18.0, 66.0, max(10.0, outer_rect.width() - 36.0), max(10.0, outer_rect.height() - 116.0))
        legend_rect = QRectF(18.0, outer_rect.height() - 42.0, max(10.0, outer_rect.width() - 36.0), 24.0)

        self._draw_scene(painter, scene_rect)
        self._draw_legend(painter, legend_rect)

    def _paint_failure_fallback(self, painter: QPainter) -> None:
        painter.resetTransform()
        painter.setClipping(False)
        painter.fillRect(self.rect(), qcolor(COLORS.surface_bg))
        painter.setPen(QPen(qcolor(COLORS.surface_border), 1.0))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 14.0, 14.0)
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(qcolor(COLORS.text_primary))
        painter.drawText(self.rect().adjusted(16, 14, -16, -16), Qt.AlignLeft | Qt.AlignTop, "Live Aligned-Grid Preview")
        font.setBold(False)
        font.setPointSize(9)
        painter.setFont(font)
        painter.setPen(qcolor(COLORS.text_muted))
        painter.drawText(
            self.rect().adjusted(16, 42, -16, -16),
            Qt.TextWordWrap,
            "Preview unavailable because a render error occurred. The experiment data and aligned-grid math are unchanged.",
        )

    def _draw_scene(self, painter: QPainter, rect: QRectF) -> None:
        truth_points = [entry.get("truth_point_mm", []) for entry in self._truth_catalog]
        aligned_points = [
            (self._metrics.get("per_point_metrics", {}) or {}).get(str(entry.get("label", "")), {}).get("aligned_centroid_truth_mm", [])
            for entry in self._truth_catalog
        ]
        all_points = [
            tuple(float(value) for value in point[:2])
            for point in truth_points + aligned_points
            if isinstance(point, list) and len(point) >= 2
        ]
        if not all_points:
            painter.setPen(qcolor(COLORS.text_muted))
            painter.drawText(rect, Qt.AlignCenter, "No truth grid or captured centroids are available yet.")
            return

        min_x, max_x = _expand_range(min(point[0] for point in all_points), max(point[0] for point in all_points), minimum_span=20.0)
        min_y, max_y = _expand_range(min(point[1] for point in all_points), max(point[1] for point in all_points), minimum_span=20.0)

        painter.setPen(QPen(qcolor(COLORS.chart_grid), 1.0, Qt.DashLine))
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = rect.left() + rect.width() * fraction
            y = rect.top() + rect.height() * fraction
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

        per_point = self._metrics.get("per_point_metrics", {}) or {}
        label_font = QFont()
        label_font.setPointSize(8)
        painter.setFont(label_font)

        for entry in self._truth_catalog:
            label = str(entry.get("label", ""))
            point_metrics = per_point.get(label, {}) or {}
            truth = entry.get("truth_point_mm", [])
            if not (isinstance(truth, list) and len(truth) >= 2):
                continue
            truth_point = QPointF(
                _scale(rect, float(truth[0]), min_x, max_x, horizontal=True),
                _scale(rect, float(truth[1]), min_y, max_y, horizontal=False),
            )
            status = _point_status(
                raw_sample_count=int(point_metrics.get("raw_sample_count", 0) or 0),
                expected_samples=self._expected_samples,
                rejected_count=int(point_metrics.get("outlier_count", 0) or 0),
            )
            truth_pen, truth_brush, label_color = _truth_marker_style(status=status)
            if int(entry.get("target_index", -1)) == int(self._selected_target_index):
                painter.setPen(QPen(qcolor(COLORS.accent), 2.2))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(truth_point, 10.0, 10.0)
            painter.setPen(truth_pen)
            painter.setBrush(truth_brush)
            painter.drawRect(QRectF(truth_point.x() - 4.5, truth_point.y() - 4.5, 9.0, 9.0))

            aligned = point_metrics.get("aligned_centroid_truth_mm", [])
            if isinstance(aligned, list) and len(aligned) >= 2:
                aligned_point = QPointF(
                    _scale(rect, float(aligned[0]), min_x, max_x, horizontal=True),
                    _scale(rect, float(aligned[1]), min_y, max_y, horizontal=False),
                )
                painter.setPen(QPen(qcolor(COLORS.scene_residual), 1.4))
                painter.drawLine(aligned_point, truth_point)
                _draw_arrow_head(painter, start=aligned_point, end=truth_point, color=qcolor(COLORS.scene_residual))
                painter.setPen(QPen(qcolor(COLORS.text_primary), 1.0))
                painter.setBrush(qcolor(COLORS.scene_measurement))
                painter.drawEllipse(aligned_point, 4.5, 4.5)
                residual = point_metrics.get("residual_mm")
                if residual is not None and len(self._truth_catalog) <= 16:
                    painter.setPen(qcolor(COLORS.text_secondary))
                    painter.drawText(
                        QPointF(aligned_point.x() + 6.0, aligned_point.y() + 12.0),
                        f"{float(residual):.3f}",
                    )

            painter.setPen(label_color)
            painter.drawText(QPointF(truth_point.x() + 7.0, truth_point.y() - 7.0), label)

    def _draw_legend(self, painter: QPainter, rect: QRectF) -> None:
        legend_items = [
            ("Truth point", qcolor(COLORS.scene_truth), "square"),
            ("Aligned centroid", qcolor(COLORS.scene_measurement), "circle"),
            ("Selected point", qcolor(COLORS.accent), "ring"),
            ("Residual vector", qcolor(COLORS.scene_residual), "line"),
        ]
        x = rect.left()
        painter.setFont(QFont("", 8))
        painter.setPen(qcolor(COLORS.text_muted))
        for label, color, kind in legend_items:
            if kind == "square":
                painter.setPen(Qt.NoPen)
                painter.setBrush(color)
                painter.drawRect(QRectF(x, rect.top() + 6.0, 10.0, 10.0))
            elif kind == "circle":
                painter.setPen(Qt.NoPen)
                painter.setBrush(color)
                painter.drawEllipse(QPointF(x + 5.0, rect.top() + 11.0), 5.0, 5.0)
            elif kind == "ring":
                painter.setPen(QPen(color, 2.0))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPointF(x + 5.0, rect.top() + 11.0), 6.0, 6.0)
            else:
                painter.setPen(QPen(color, 1.6))
                painter.drawLine(QPointF(x, rect.top() + 11.0), QPointF(x + 10.0, rect.top() + 11.0))
            painter.setPen(qcolor(COLORS.text_muted))
            painter.drawText(QPointF(x + 16.0, rect.top() + 15.0), label)
            x += 118.0

    def _header_summary(self) -> str:
        complete = int(self._metrics.get("point_count_complete", 0) or 0)
        partial = int(self._metrics.get("point_count_partial", 0) or 0)
        not_started = int(self._metrics.get("point_count_not_started", 0) or 0)
        solve_text = "aligned residuals ready" if self._metrics.get("alignment_ready") else str(
            self._metrics.get("alignment_ready_reason", "need more complete points")
        )
        return f"{complete} complete • {partial} partial • {not_started} not started • {solve_text}"


def _expand_range(low: float, high: float, *, minimum_span: float) -> tuple[float, float]:
    span = max(float(high - low), float(minimum_span))
    pad = span * 0.16
    center = (float(low) + float(high)) / 2.0
    half = (span / 2.0) + pad
    return center - half, center + half


def _scale(rect: QRectF, value: float, low: float, high: float, *, horizontal: bool) -> float:
    if high <= low:
        return rect.center().x() if horizontal else rect.center().y()
    ratio = (value - low) / (high - low)
    if horizontal:
        return rect.left() + ratio * rect.width()
    return rect.bottom() - ratio * rect.height()


def _point_status(*, raw_sample_count: int, expected_samples: int, rejected_count: int) -> str:
    if raw_sample_count <= 0:
        return "not_started"
    if raw_sample_count < max(1, expected_samples):
        return "partial"
    if rejected_count > 0:
        return "complete_rejected"
    return "complete"


def _truth_marker_style(*, status: str) -> tuple[QPen, QBrush, QColor]:
    if status == "complete":
        return QPen(qcolor(COLORS.scene_truth), 1.2), QBrush(qcolor(COLORS.scene_truth)), qcolor(COLORS.text_secondary)
    if status == "complete_rejected":
        return QPen(qcolor(COLORS.warning_fg), 1.2), QBrush(qcolor(COLORS.warning_bg)), qcolor(COLORS.warning_fg)
    if status == "partial":
        return QPen(qcolor(COLORS.warning_fg), 1.2), QBrush(qcolor(COLORS.warning_bg, alpha=220)), qcolor(COLORS.warning_fg)
    return QPen(qcolor(COLORS.surface_border), 1.2), QBrush(qcolor(COLORS.surface_alt_bg)), qcolor(COLORS.text_muted)


def _draw_arrow_head(painter: QPainter, *, start: QPointF, end: QPointF, color: QColor) -> None:
    dx = end.x() - start.x()
    dy = end.y() - start.y()
    norm = (dx * dx + dy * dy) ** 0.5
    if norm <= 1e-6:
        return
    ux = dx / norm
    uy = dy / norm
    wing = 4.0
    back = 6.0
    left = QPointF(end.x() - (ux * back) - (uy * wing), end.y() - (uy * back) + (ux * wing))
    right = QPointF(end.x() - (ux * back) + (uy * wing), end.y() - (uy * back) - (ux * wing))
    painter.setPen(QPen(color, 1.2))
    path = QPainterPath(end)
    path.lineTo(left)
    path.moveTo(end)
    path.lineTo(right)
    painter.drawPath(path)
