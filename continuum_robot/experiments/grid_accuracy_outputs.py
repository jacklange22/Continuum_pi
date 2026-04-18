"""Extra artifacts for the aligned-grid Aurora validation experiment."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

try:
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
    from PySide6.QtWidgets import QApplication

    _QT_AVAILABLE = True
except ModuleNotFoundError:
    _QT_AVAILABLE = False


_PLOT_QT_APP: QApplication | None = None
LOG = logging.getLogger(__name__)


def build_grid_accuracy_summary_pairs(
    *,
    metrics: dict[str, Any],
    config_used: dict[str, Any],
) -> list[tuple[str, str]]:
    """Return the compact aligned-grid summary rows used by the page and saved artifacts."""
    config_used = dict(config_used or {})
    metrics = dict(metrics or {})
    rows, cols = _grid_dimensions(config_used)
    expected_samples = int(config_used.get("samples_per_point", 0) or 0)
    tip_requested = bool(config_used.get("use_tip_calibration", False))
    point_count_complete, point_count_partial, point_count_not_started = _capture_status_counts(
        metrics=metrics,
        expected_samples=expected_samples,
        point_count_total=int(metrics.get("point_count_total", 0) or 0),
    )
    position_source_summary = ", ".join(
        f"{str(source)}={int(count)}"
        for source, count in sorted((metrics.get("position_source_counts", {}) or {}).items())
    ) or "n/a"
    return [
        ("Grid", f"{rows} x {cols}"),
        ("Spacing", f"{_fmt_float(config_used.get('spacing_mm'))} mm"),
        ("Samples / Point", str(expected_samples)),
        ("Coverage", f"{int(metrics.get('point_count_captured', 0) or 0)} / {int(metrics.get('point_count_total', 0) or 0)}"),
        (
            "Point Status",
            f"{point_count_complete} complete, {point_count_partial} partial, {point_count_not_started} not started",
        ),
        ("Solve Ready", "yes" if metrics.get("alignment_ready") else "no"),
        ("Aligned Points", str(int(metrics.get("point_count_aligned", 0) or 0))),
        ("Raw Samples", str(int(metrics.get("raw_sample_count", 0) or 0))),
        ("Accepted Samples", str(int(metrics.get("accepted_sample_count", 0) or 0))),
        ("Rejected Samples", str(int(metrics.get("rejected_sample_count", metrics.get("outlier_count", 0)) or 0))),
        ("Overall RMS Residual", f"{_fmt_float(metrics.get('overall_rms_residual_mm'))} mm"),
        ("Max Residual", f"{_fmt_float(metrics.get('max_residual_mm'))} mm"),
        ("Mean Within-Point Spread", f"{_fmt_float(metrics.get('mean_within_point_spread_mm'))} mm"),
        ("Max Within-Point Spread", f"{_fmt_float(metrics.get('max_within_point_spread_mm'))} mm"),
        ("Tip Calibration Requested", "yes" if tip_requested else "no"),
        ("Tip Calibration Used", _tip_usage_label(metrics=metrics, tip_requested=tip_requested)),
        ("Position Sources", position_source_summary),
    ]


def build_grid_accuracy_summary_lines(
    *,
    metadata,
    summary,
    metrics: dict[str, Any],
    config_used: dict[str, Any],
) -> list[str]:
    """Return the human-readable summary text for one aligned-grid validation run."""
    metrics = dict(metrics or {})
    lines = [
        "Aurora Grid Accuracy Summary",
        "Aligned-grid consistency check. Measurement method: ideal local truth grid, centroid per labeled point, rigid no-scale best-fit alignment, residuals reported after alignment.",
        "",
        f"Run ID: {metadata.run_id}",
        f"Timestamp: {metadata.timestamp_utc}",
        f"Status: {summary.status}",
    ]
    for label, value in build_grid_accuracy_summary_pairs(metrics=metrics, config_used=config_used):
        lines.append(f"{label}: {value}")
    lines.append(f"Solve Readiness Reason: {metrics.get('alignment_ready_reason', 'n/a')}")

    warning_lines = build_grid_accuracy_warning_lines(metrics=metrics, config_used=config_used)
    if warning_lines:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in warning_lines)

    point_lines = _per_point_summary_lines(metrics=metrics, expected_samples=int(config_used.get("samples_per_point", 0) or 0))
    if point_lines:
        lines.append("")
        lines.append("Per-Point Metrics:")
        lines.extend(point_lines)
    return lines


def build_grid_accuracy_warning_lines(
    *,
    metrics: dict[str, Any],
    config_used: dict[str, Any],
) -> list[str]:
    """Return compact warning lines for incomplete or low-confidence aligned-grid runs."""
    warnings: list[str] = []
    metrics = dict(metrics or {})
    config_used = dict(config_used or {})
    if not metrics.get("alignment_ready"):
        warnings.append(str(metrics.get("alignment_ready_reason", "Aligned residuals are not ready yet.")))
    if int(metrics.get("rejected_sample_count", metrics.get("outlier_count", 0)) or 0) > 0:
        warnings.append(
            f"Rejected/outlier samples were present: {int(metrics.get('rejected_sample_count', metrics.get('outlier_count', 0)) or 0)}."
        )
    if bool(config_used.get("use_tip_calibration", False)) and not bool(metrics.get("tip_calibration_used", False)):
        warnings.append("Tip calibration was requested, but this dataset was captured with coil-origin fallback for at least part of the run.")
    return warnings


def write_grid_accuracy_outputs(
    *,
    output_dir: Path,
    metadata,
    summary,
) -> dict[str, Path]:
    """Write stable figure/text artifacts for one aligned-grid validation run."""
    output_dir = Path(output_dir)
    plot_path = output_dir / "grid_accuracy_alignment.png"
    summary_text_path = output_dir / "grid_accuracy_summary.txt"
    metrics = summary.experiment_metrics if isinstance(summary.experiment_metrics, dict) else {}
    config_used = metadata.config_used if isinstance(metadata.config_used, dict) else {}
    _write_summary_text(
        summary_text_path=summary_text_path,
        metadata=metadata,
        summary=summary,
        metrics=metrics,
        config_used=config_used,
    )
    if _QT_AVAILABLE:
        _ensure_plot_qt_app()
        _write_alignment_plot(
            plot_path=plot_path,
            metrics=metrics,
            config_used=config_used,
        )
    else:
        _write_alignment_plot_placeholder(plot_path=plot_path)
        LOG.warning(
            "Qt plotting backend unavailable; wrote placeholder grid accuracy plot at %s",
            plot_path,
        )
    return {
        "plot_path": plot_path,
        "summary_text_path": summary_text_path,
    }


def _ensure_plot_qt_app() -> QApplication:
    if not _QT_AVAILABLE:
        raise RuntimeError("PySide6 is not installed; grid accuracy plotting is unavailable")
    app = QApplication.instance()
    if app is not None:
        return app
    global _PLOT_QT_APP
    if _PLOT_QT_APP is None:
        _PLOT_QT_APP = QApplication([])
    return _PLOT_QT_APP


def _write_summary_text(
    *,
    summary_text_path: Path,
    metadata,
    summary,
    metrics: dict[str, Any],
    config_used: dict[str, Any],
) -> None:
    lines = build_grid_accuracy_summary_lines(
        metadata=metadata,
        summary=summary,
        metrics=metrics,
        config_used=config_used,
    )
    summary_text_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _write_alignment_plot(
    *,
    plot_path: Path,
    metrics: dict[str, Any],
    config_used: dict[str, Any],
) -> None:
    width = 1280
    height = 860
    image = QImage(width, height, QImage.Format_ARGB32)
    image.fill(QColor("#f8fafc"))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)

    _draw_title_block(painter, QRectF(34.0, 24.0, width - 68.0, 54.0))

    scatter_rect = QRectF(36.0, 98.0, 760.0, 480.0)
    summary_rect = QRectF(824.0, 98.0, 420.0, 242.0)
    residual_rect = QRectF(824.0, 362.0, 420.0, 216.0)
    spread_rect = QRectF(36.0, 610.0, 1208.0, 214.0)

    _draw_panel(painter, scatter_rect, "Aligned Grid Overlay")
    _draw_panel(painter, summary_rect, "Run Summary")
    _draw_panel(painter, residual_rect, "Per-Point Residual")
    _draw_panel(painter, spread_rect, "Within-Point Spread")

    entries = _per_point_entries(metrics)
    _draw_scatter_panel(painter, scatter_rect.adjusted(16.0, 30.0, -14.0, -16.0), entries)
    _draw_summary_panel(painter, summary_rect.adjusted(16.0, 34.0, -16.0, -14.0), metrics, config_used)
    _draw_bar_panel(
        painter,
        residual_rect.adjusted(16.0, 34.0, -16.0, -18.0),
        entries,
        value_key="residual_mm",
        color=QColor("#dc2626"),
        empty_text="Alignment is not ready yet.",
        y_label="Residual (mm)",
    )
    _draw_bar_panel(
        painter,
        spread_rect.adjusted(16.0, 34.0, -16.0, -18.0),
        entries,
        value_key="sample_spread_rms_mm",
        color=QColor("#0f766e"),
        empty_text="No captured point spread is available yet.",
        y_label="Spread RMS (mm)",
    )

    painter.end()
    image.save(str(plot_path))


def _write_alignment_plot_placeholder(*, plot_path: Path) -> None:
    # 1x1 transparent PNG fallback for headless/unit-test environments without Qt.
    plot_path.write_bytes(
        bytes(
            (
                137,
                80,
                78,
                71,
                13,
                10,
                26,
                10,
                0,
                0,
                0,
                13,
                73,
                72,
                68,
                82,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                8,
                6,
                0,
                0,
                0,
                31,
                21,
                196,
                137,
                0,
                0,
                0,
                13,
                73,
                68,
                65,
                84,
                120,
                156,
                99,
                96,
                0,
                0,
                0,
                2,
                0,
                1,
                226,
                33,
                188,
                51,
                0,
                0,
                0,
                0,
                73,
                69,
                78,
                68,
                174,
                66,
                96,
                130,
            )
        )
    )


def _draw_title_block(painter: QPainter, rect: QRectF) -> None:
    title_font = QFont()
    title_font.setPointSize(16)
    title_font.setBold(True)
    painter.setFont(title_font)
    painter.setPen(QColor("#0f172a"))
    painter.drawText(rect, Qt.AlignLeft | Qt.AlignTop, "AURORA GRID ACCURACY")
    subtitle_font = QFont()
    subtitle_font.setPointSize(10)
    painter.setFont(subtitle_font)
    painter.setPen(QColor("#475569"))
    painter.drawText(
        rect.adjusted(0.0, 28.0, 0.0, 0.0),
        Qt.AlignLeft | Qt.AlignTop,
        "Truth grid points and aligned measured centroids in the local grid frame. Residual vectors are shown after rigid best-fit alignment.",
    )


def _draw_panel(painter: QPainter, rect: QRectF, title: str) -> None:
    painter.setPen(QPen(QColor("#dbe4ee"), 1.0))
    painter.setBrush(QColor("#ffffff"))
    painter.drawRoundedRect(rect, 14.0, 14.0)
    font = QFont()
    font.setPointSize(11)
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QColor("#0f172a"))
    painter.drawText(rect.adjusted(14.0, 10.0, -10.0, -10.0), Qt.AlignLeft | Qt.AlignTop, title)


def _draw_scatter_panel(painter: QPainter, rect: QRectF, entries: list[dict[str, Any]]) -> None:
    truth_points = [entry["truth_point_mm"] for entry in entries if entry.get("truth_point_mm") is not None]
    aligned_points = [entry["aligned_centroid_truth_mm"] for entry in entries if entry.get("aligned_centroid_truth_mm") is not None]
    points = truth_points + aligned_points
    if not points:
        painter.setPen(QColor("#64748b"))
        painter.drawText(rect, Qt.AlignCenter, "No truth/aligned points available yet.")
        return

    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    min_x, max_x = _expand_range(min(xs), max(xs), minimum_span=20.0, pad_fraction=0.12)
    min_y, max_y = _expand_range(min(ys), max(ys), minimum_span=20.0, pad_fraction=0.12)

    painter.setPen(QPen(QColor("#e2e8f0"), 1.0, Qt.DashLine))
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = rect.left() + (rect.width() * fraction)
        y = rect.top() + (rect.height() * fraction)
        painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

    for entry in entries:
        truth = entry.get("truth_point_mm")
        aligned = entry.get("aligned_centroid_truth_mm")
        label = str(entry.get("label", ""))
        residual = entry.get("residual_mm")
        if truth is not None:
            truth_point = QPointF(_scale_x(rect, float(truth[0]), min_x, max_x), _scale_y(rect, float(truth[1]), min_y, max_y))
            painter.setPen(QPen(QColor("#0f172a"), 1.0))
            painter.setBrush(QColor("#1d4ed8"))
            painter.drawRect(QRectF(truth_point.x() - 4.5, truth_point.y() - 4.5, 9.0, 9.0))
            painter.setPen(QColor("#0f172a"))
            painter.drawText(QPointF(truth_point.x() + 6.0, truth_point.y() - 6.0), label)
            if aligned is not None:
                aligned_point = QPointF(
                    _scale_x(rect, float(aligned[0]), min_x, max_x),
                    _scale_y(rect, float(aligned[1]), min_y, max_y),
                )
                painter.setPen(QPen(QColor("#dc2626"), 1.3))
                painter.drawLine(aligned_point, truth_point)
                _draw_arrow_head(painter, start=aligned_point, end=truth_point, color=QColor("#dc2626"))
                painter.setPen(QPen(QColor("#ffffff"), 1.0))
                painter.setBrush(QColor("#059669"))
                painter.drawEllipse(aligned_point, 4.6, 4.6)
                if residual is not None:
                    painter.setPen(QColor("#7f1d1d"))
                    painter.drawText(QPointF(aligned_point.x() + 6.0, aligned_point.y() + 12.0), f"{float(residual):.3f} mm")

    legend_y = rect.bottom() - 8.0
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor("#1d4ed8"))
    painter.drawRect(QRectF(rect.left() + 4.0, legend_y - 10.0, 10.0, 10.0))
    painter.setBrush(QColor("#059669"))
    painter.drawEllipse(QPointF(rect.left() + 26.0, legend_y - 5.0), 5.0, 5.0)
    painter.setPen(QColor("#475569"))
    painter.drawText(QPointF(rect.left() + 42.0, legend_y), "Truth points / aligned centroids / residual vectors")


def _draw_summary_panel(painter: QPainter, rect: QRectF, metrics: dict[str, Any], config_used: dict[str, Any]) -> None:
    summary_lookup = dict(build_grid_accuracy_summary_pairs(metrics=metrics, config_used=config_used))
    lines = [
        ("Grid", summary_lookup.get("Grid", "n/a")),
        ("Spacing", summary_lookup.get("Spacing", "n/a")),
        ("Samples / Point", summary_lookup.get("Samples / Point", "n/a")),
        ("Coverage", summary_lookup.get("Coverage", "n/a")),
        ("Overall RMS Residual", summary_lookup.get("Overall RMS Residual", "n/a")),
        ("Max Residual", summary_lookup.get("Max Residual", "n/a")),
        ("Mean Within-Point Spread", summary_lookup.get("Mean Within-Point Spread", "n/a")),
        ("Rejected Samples", summary_lookup.get("Rejected Samples", "n/a")),
        ("Tip Calibration Used", summary_lookup.get("Tip Calibration Used", "n/a")),
    ]
    y = rect.top()
    label_font = QFont()
    label_font.setPointSize(9)
    value_font = QFont()
    value_font.setPointSize(9)
    value_font.setBold(True)
    for label, value in lines:
        painter.setFont(label_font)
        painter.setPen(QColor("#475569"))
        painter.drawText(QRectF(rect.left(), y, rect.width() * 0.44, 20.0), Qt.AlignLeft | Qt.AlignVCenter, label)
        painter.setFont(value_font)
        painter.setPen(QColor("#0f172a"))
        painter.drawText(QRectF(rect.left() + rect.width() * 0.46, y, rect.width() * 0.52, 20.0), Qt.AlignLeft | Qt.AlignVCenter, value)
        y += 24.0
    painter.setFont(label_font)
    painter.setPen(QColor("#64748b"))
    painter.drawText(
        QRectF(rect.left(), y + 8.0, rect.width(), rect.height() - (y - rect.top()) - 8.0),
        Qt.TextWordWrap,
        str(metrics.get("alignment_ready_reason", "Alignment-ready summary unavailable.")),
    )


def _draw_bar_panel(
    painter: QPainter,
    rect: QRectF,
    entries: list[dict[str, Any]],
    *,
    value_key: str,
    color: QColor,
    empty_text: str,
    y_label: str,
) -> None:
    labels = [str(entry.get("label", "")) for entry in entries]
    values = [entry.get(value_key) for entry in entries]
    paired = [(label, float(value)) for label, value in zip(labels, values) if value is not None]
    if not paired:
        painter.setPen(QColor("#64748b"))
        painter.drawText(rect, Qt.AlignCenter, empty_text)
        return

    max_value = max(value for _, value in paired)
    max_value = max(max_value, 0.5)
    chart_rect = rect.adjusted(0.0, 8.0, 0.0, -26.0)
    painter.setPen(QPen(QColor("#e2e8f0"), 1.0))
    painter.drawLine(QPointF(chart_rect.left(), chart_rect.bottom()), QPointF(chart_rect.right(), chart_rect.bottom()))
    painter.drawLine(QPointF(chart_rect.left(), chart_rect.top()), QPointF(chart_rect.left(), chart_rect.bottom()))
    bar_width = chart_rect.width() / max(1, len(paired))
    for index, (label, value) in enumerate(paired):
        left = chart_rect.left() + (index * bar_width) + 8.0
        usable_width = max(8.0, bar_width - 16.0)
        bar_height = (value / max_value) * max(1.0, chart_rect.height() - 10.0)
        bar_rect = QRectF(left, chart_rect.bottom() - bar_height, usable_width, bar_height)
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        painter.drawRoundedRect(bar_rect, 4.0, 4.0)
        painter.setPen(QColor("#475569"))
        painter.drawText(QRectF(left - 6.0, chart_rect.bottom() + 4.0, usable_width + 12.0, 18.0), Qt.AlignHCenter, label)
        painter.drawText(QRectF(left - 6.0, bar_rect.top() - 18.0, usable_width + 12.0, 16.0), Qt.AlignHCenter, f"{value:.3f}")
    painter.drawText(QRectF(rect.left(), rect.top() - 2.0, rect.width(), 16.0), Qt.AlignRight, y_label)


def _draw_arrow_head(painter: QPainter, *, start: QPointF, end: QPointF, color: QColor) -> None:
    dx = end.x() - start.x()
    dy = end.y() - start.y()
    norm = (dx * dx + dy * dy) ** 0.5
    if norm <= 1e-6:
        return
    ux = dx / norm
    uy = dy / norm
    wing = 5.0
    back = 7.0
    left = QPointF(end.x() - (ux * back) - (uy * wing), end.y() - (uy * back) + (ux * wing))
    right = QPointF(end.x() - (ux * back) + (uy * wing), end.y() - (uy * back) - (ux * wing))
    painter.setPen(QPen(color, 1.3))
    painter.drawLine(end, left)
    painter.drawLine(end, right)


def _per_point_entries(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    per_point = metrics.get("per_point_metrics", {}) or {}
    return [per_point[label] for label in sorted(per_point, key=_label_sort_key)]


def _per_point_summary_lines(*, metrics: dict[str, Any], expected_samples: int) -> list[str]:
    lines: list[str] = []
    expected = max(1, int(expected_samples))
    for entry in _per_point_entries(metrics):
        raw_count = int(entry.get("raw_sample_count", 0) or 0)
        accepted_count = int(entry.get("accepted_sample_count", 0) or 0)
        rejected_count = int(entry.get("outlier_count", 0) or 0)
        residual = _fmt_float(entry.get("residual_mm"))
        spread = _fmt_float(entry.get("sample_spread_rms_mm"))
        position_source = str(entry.get("position_source_summary", "n/a"))
        lines.append(
            f"- {entry.get('label', 'P??')}: raw={raw_count}/{expected}, accepted={accepted_count}, "
            f"rejected={rejected_count}, spread={spread} mm, residual={residual} mm, source={position_source}"
        )
    return lines


def _capture_status_counts(
    *,
    metrics: dict[str, Any],
    expected_samples: int,
    point_count_total: int,
) -> tuple[int, int, int]:
    per_point = metrics.get("per_point_metrics", {}) or {}
    complete = 0
    partial = 0
    for entry in per_point.values():
        raw_count = int(entry.get("raw_sample_count", 0) or 0)
        if raw_count >= max(1, expected_samples):
            complete += 1
        elif raw_count > 0:
            partial += 1
    not_started = max(0, int(point_count_total) - complete - partial)
    return complete, partial, not_started


def _tip_usage_label(*, metrics: dict[str, Any], tip_requested: bool) -> str:
    if not tip_requested:
        return "not requested"
    if bool(metrics.get("tip_calibration_used", False)):
        return "yes"
    if bool(metrics.get("coil_origin_fallback_used", False)):
        return "fallback only"
    return "no"


def _grid_dimensions(config_used: dict[str, Any]) -> tuple[int, int]:
    dims = config_used.get("dimensions")
    if isinstance(dims, list) and len(dims) >= 2:
        return int(dims[1]), int(dims[0])
    rows = int(config_used.get("rows", 0) or 0)
    cols = int(config_used.get("cols", 0) or 0)
    return rows, cols


def _label_sort_key(label: str) -> tuple[int, str]:
    digits = "".join(char for char in str(label) if char.isdigit())
    return (int(digits) if digits else 10**9, str(label))


def _fmt_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _expand_range(low: float, high: float, *, minimum_span: float, pad_fraction: float) -> tuple[float, float]:
    span = max(float(high - low), float(minimum_span))
    pad = span * float(pad_fraction)
    center = (float(low) + float(high)) / 2.0
    half = (span / 2.0) + pad
    return center - half, center + half


def _scale_x(rect: QRectF, value: float, low: float, high: float) -> float:
    if high <= low:
        return rect.center().x()
    return rect.left() + ((value - low) / (high - low)) * rect.width()


def _scale_y(rect: QRectF, value: float, low: float, high: float) -> float:
    if high <= low:
        return rect.center().y()
    return rect.bottom() - ((value - low) / (high - low)) * rect.height()
