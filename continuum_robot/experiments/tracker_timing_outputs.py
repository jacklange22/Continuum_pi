"""Saved artifacts for the Aurora backend timing diagnostic."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
from typing import Any

try:
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
    from PySide6.QtWidgets import QApplication

    _QT_AVAILABLE = True
except ModuleNotFoundError:
    _QT_AVAILABLE = False

from continuum_robot.tracking.timing_benchmark import (
    extract_servo_timing_records,
    extract_tracker_timing_records,
)


_PLOT_QT_APP: QApplication | None = None
LOG = logging.getLogger(__name__)


def build_tracker_timing_summary_pairs(*, metrics: dict[str, Any]) -> list[tuple[str, str]]:
    """Return compact backend-timing summary rows for GUI/details surfaces."""
    metrics = dict(metrics or {})
    requested_tools = ",".join(str(value) for value in (metrics.get("requested_tool_ids") or [])) or "n/a"
    per_tool_summary = dict(metrics.get("per_tool_summary", {}) or {})
    per_tool_rate = ", ".join(
        f"{tool_id}={float(summary.get('valid_transform_rate', 0.0) or 0.0) * 100.0:.1f}%"
        for tool_id, summary in sorted(per_tool_summary.items())
    ) or "n/a"
    rows = [
        ("Backend", str(metrics.get("backend_identity", "n/a") or "n/a")),
        ("Configured Backend", str(metrics.get("configured_backend_name", "n/a") or "n/a")),
        ("Selected Backend", str(metrics.get("selected_backend_name", "n/a") or "n/a")),
        ("Tool IDs", requested_tools),
        ("Run Label", str(metrics.get("run_label", "") or "n/a")),
        ("Analyzed Samples", str(int(metrics.get("sample_count_analyzed", 0) or 0))),
        ("Warmup Discarded", str(int(metrics.get("warmup_discarded_count", 0) or 0))),
        ("Effective Loop Hz", _fmt(metrics.get("effective_loop_rate_hz"), suffix=" Hz")),
        ("Unique-Frame Hz", _fmt(metrics.get("unique_frame_rate_hz"), suffix=" Hz")),
        ("Mean Total Time", _fmt(metrics.get("mean_total_cycle_ms"), suffix=" ms")),
        ("P95 Total Time", _fmt(metrics.get("p95_total_cycle_ms"), suffix=" ms")),
        ("P99 Total Time", _fmt(metrics.get("p99_total_cycle_ms"), suffix=" ms")),
        (
            "Under 25 ms",
            (
                f"{float(metrics.get('percent_under_25ms')):.1f}%"
                if metrics.get("percent_under_25ms") is not None
                else "n/a"
            ),
        ),
        (
            "Duplicate Frames",
            (
                f"{int(metrics.get('duplicate_frame_count', 0) or 0)} "
                f"({float(metrics.get('duplicate_frame_ratio', 0.0) or 0.0) * 100.0:.1f}%)"
                if metrics.get("duplicate_frame_ratio") is not None
                else "n/a"
            ),
        ),
        (
            "Invalid/Missing Samples",
            (
                f"{int(metrics.get('invalid_or_missing_requested_tool_sample_count', 0) or 0)} "
                f"({float(metrics.get('invalid_or_missing_requested_tool_ratio', 0.0) or 0.0) * 100.0:.1f}%)"
                if metrics.get("invalid_or_missing_requested_tool_ratio") is not None
                else "n/a"
            ),
        ),
        ("Per-Tool Valid Rate", per_tool_rate),
        ("Backend Errors", str(int(metrics.get("error_sample_count", 0) or 0))),
    ]
    servo_sync = dict(metrics.get("servo_sync", {}) or {})
    if servo_sync.get("enabled"):
        rows.extend(
            [
                ("Servo Sync", "available" if servo_sync.get("available") else "requested but unavailable"),
                ("Servo->Tracker Mean", _fmt(servo_sync.get("servo_to_tracker_mean_offset_ms"), suffix=" ms")),
                ("Servo->Tracker P95", _fmt(servo_sync.get("servo_to_tracker_p95_offset_ms"), suffix=" ms")),
                ("Tracker->Servo Mean", _fmt(servo_sync.get("tracker_to_servo_mean_offset_ms"), suffix=" ms")),
                ("Tracker->Servo P95", _fmt(servo_sync.get("tracker_to_servo_p95_offset_ms"), suffix=" ms")),
            ]
        )
    return rows


def build_tracker_timing_summary_lines(*, metadata, summary, metrics: dict[str, Any]) -> list[str]:
    """Return human-readable summary text for one timing diagnostic run."""
    lines = [
        "Aurora Timing Diagnostic Summary",
        "This run measures the active Python tracker backend acquisition path. GUI refresh rate is not used as the timing signal.",
        "",
        f"Run ID: {metadata.run_id}",
        f"Timestamp: {metadata.timestamp_utc}",
        f"Status: {summary.status}",
    ]
    for label, value in build_tracker_timing_summary_pairs(metrics=metrics):
        lines.append(f"{label}: {value}")
    total_stats = dict(metrics.get("total_cycle_ms_stats", {}) or {})
    lines.append("")
    lines.append(
        "Total cycle timing (ms): "
        f"min={_fmt_number(total_stats.get('min'))}, "
        f"median={_fmt_number(total_stats.get('median'))}, "
        f"mean={_fmt_number(total_stats.get('mean'))}, "
        f"p95={_fmt_number(total_stats.get('p95'))}, "
        f"p99={_fmt_number(total_stats.get('p99'))}, "
        f"max={_fmt_number(total_stats.get('max'))}"
    )
    for stage_key, title in (
        ("backend_call_ms_stats", "Backend get_frame"),
        ("parse_ms_stats", "Parse / transform extraction"),
        ("state_commit_ms_stats", "State commit"),
        ("loop_period_ms_stats", "Observed loop period"),
    ):
        stats = dict(metrics.get(stage_key, {}) or {})
        lines.append(
            f"{title} (ms): "
            f"mean={_fmt_number(stats.get('mean'))}, "
            f"p95={_fmt_number(stats.get('p95'))}, "
            f"max={_fmt_number(stats.get('max'))}"
        )
    stage_definitions = dict(metrics.get("instrumented_stage_definitions", {}) or {})
    if stage_definitions:
        lines.append("")
        lines.append("Stage definitions:")
        for key, message in stage_definitions.items():
            lines.append(f"- {key}: {message}")
    servo_sync = dict(metrics.get("servo_sync", {}) or {})
    if servo_sync.get("enabled"):
        lines.append("")
        lines.append("Servo sync:")
        if servo_sync.get("available"):
            lines.append(
                f"- Servo->tracker offset mean/p95/max (ms): "
                f"{_fmt_number(servo_sync.get('servo_to_tracker_mean_offset_ms'))} / "
                f"{_fmt_number(servo_sync.get('servo_to_tracker_p95_offset_ms'))} / "
                f"{_fmt_number(servo_sync.get('servo_to_tracker_max_offset_ms'))}"
            )
            lines.append(
                f"- Tracker->servo offset mean/p95/max (ms): "
                f"{_fmt_number(servo_sync.get('tracker_to_servo_mean_offset_ms'))} / "
                f"{_fmt_number(servo_sync.get('tracker_to_servo_p95_offset_ms'))} / "
                f"{_fmt_number(servo_sync.get('tracker_to_servo_max_offset_ms'))}"
            )
            lines.append(
                f"- Offsets above {float(servo_sync.get('pairing_threshold_ms', 25.0)):.1f} ms: "
                f"servo={int(servo_sync.get('servo_samples_over_threshold_count', 0) or 0)}, "
                f"tracker={int(servo_sync.get('tracker_samples_over_threshold_count', 0) or 0)}"
            )
        else:
            lines.append("- Servo logging was requested, but no valid tracker-servo pairings were available.")
    return lines


def write_tracker_timing_outputs(*, output_dir: Path, metadata, summary, samples) -> dict[str, Path]:
    """Write stable figure/text artifacts for one timing-diagnostic run."""
    output_dir = Path(output_dir)
    histogram_path = output_dir / "aurora_timing_histogram.png"
    breakdown_path = output_dir / "aurora_timing_breakdown.png"
    timeseries_path = output_dir / "aurora_timing_timeseries.png"
    summary_text_path = output_dir / "aurora_timing_summary.txt"
    sync_plot_path = output_dir / "aurora_timing_sync_offsets.png"
    metrics = summary.experiment_metrics if isinstance(summary.experiment_metrics, dict) else {}
    tracker_records = extract_tracker_timing_records(samples)
    servo_records = extract_servo_timing_records(samples)
    _write_summary_text(summary_text_path=summary_text_path, metadata=metadata, summary=summary, metrics=metrics)
    if _qt_plotting_is_safe():
        _ensure_plot_qt_app()
        _write_histogram_plot(histogram_path=histogram_path, tracker_records=tracker_records, metrics=metrics)
        _write_breakdown_plot(breakdown_path=breakdown_path, metrics=metrics)
        _write_timeseries_plot(timeseries_path=timeseries_path, tracker_records=tracker_records, metrics=metrics)
    else:
        _write_plot_placeholder(histogram_path)
        _write_plot_placeholder(breakdown_path)
        _write_plot_placeholder(timeseries_path)
        LOG.warning("Qt plotting backend unavailable; wrote placeholder timing plots under %s", output_dir)
    written = {
        "histogram_path": histogram_path,
        "breakdown_path": breakdown_path,
        "timeseries_path": timeseries_path,
        "summary_text_path": summary_text_path,
    }
    if metrics.get("servo_sync", {}).get("enabled"):
        if _qt_plotting_is_safe():
            _ensure_plot_qt_app()
            _write_sync_plot(sync_plot_path=sync_plot_path, metrics=metrics, servo_records=servo_records)
        else:
            _write_plot_placeholder(sync_plot_path)
        written["sync_plot_path"] = sync_plot_path
    return written


def _ensure_plot_qt_app() -> QApplication:
    if not _QT_AVAILABLE:
        raise RuntimeError("PySide6 is not installed; tracker timing plotting is unavailable")
    app = QApplication.instance()
    if app is not None:
        return app
    global _PLOT_QT_APP
    if _PLOT_QT_APP is None:
        _PLOT_QT_APP = QApplication([])
    return _PLOT_QT_APP


def _qt_plotting_is_safe() -> bool:
    """Return whether this process can safely construct Qt plot images."""
    if not _QT_AVAILABLE:
        return False
    if QApplication.instance() is not None:
        return True
    if os.environ.get("QT_QPA_PLATFORM"):
        return True
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return False
    if sys.platform == "darwin" and not os.environ.get("DISPLAY"):
        return False
    return True


def _write_summary_text(*, summary_text_path: Path, metadata, summary, metrics: dict[str, Any]) -> None:
    lines = build_tracker_timing_summary_lines(metadata=metadata, summary=summary, metrics=metrics)
    summary_text_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _write_plot_placeholder(path: Path) -> None:
    path.write_bytes(
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


def _write_histogram_plot(*, histogram_path: Path, tracker_records: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    image = _new_image(1280, 720)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_title(painter, QRectF(34.0, 24.0, 1212.0, 56.0), "AURORA TIMING HISTOGRAM", "Total backend sample time distribution for analyzed tracker samples.")
    chart_rect = QRectF(36.0, 104.0, 820.0, 560.0)
    summary_rect = QRectF(888.0, 104.0, 356.0, 560.0)
    _draw_panel(painter, chart_rect, "Total Cycle Time")
    _draw_panel(painter, summary_rect, "Run Summary")
    values = [
        float(record["total_cycle_ms"])
        for record in tracker_records
        if not bool(record.get("warmup_discarded", False)) and record.get("total_cycle_ms") is not None
    ]
    _draw_histogram(
        painter,
        chart_rect.adjusted(18.0, 38.0, -18.0, -28.0),
        values=values,
        color=QColor("#2563eb"),
        x_label="Total Cycle Time (ms)",
        y_label="Count",
        empty_text="No analyzed tracker timing samples were saved.",
    )
    _draw_summary_pairs(painter, summary_rect.adjusted(16.0, 36.0, -16.0, -16.0), build_tracker_timing_summary_pairs(metrics=metrics))
    painter.end()
    image.save(str(histogram_path))


def _write_breakdown_plot(*, breakdown_path: Path, metrics: dict[str, Any]) -> None:
    image = _new_image(1180, 760)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_title(
        painter,
        QRectF(34.0, 24.0, 1112.0, 56.0),
        "AURORA TIMING BREAKDOWN",
        "Mean and p95 stage timing for backend acquisition, parsing, and commit.",
    )
    mean_rect = QRectF(36.0, 104.0, 532.0, 600.0)
    p95_rect = QRectF(612.0, 104.0, 532.0, 600.0)
    _draw_panel(painter, mean_rect, "Stage Mean (ms)")
    _draw_panel(painter, p95_rect, "Stage P95 (ms)")
    stage_labels = ["get_frame", "parse", "commit", "total"]
    mean_values = [
        _maybe_metric(metrics, "backend_call_ms_stats", "mean"),
        _maybe_metric(metrics, "parse_ms_stats", "mean"),
        _maybe_metric(metrics, "state_commit_ms_stats", "mean"),
        _maybe_metric(metrics, "total_cycle_ms_stats", "mean"),
    ]
    p95_values = [
        _maybe_metric(metrics, "backend_call_ms_stats", "p95"),
        _maybe_metric(metrics, "parse_ms_stats", "p95"),
        _maybe_metric(metrics, "state_commit_ms_stats", "p95"),
        _maybe_metric(metrics, "total_cycle_ms_stats", "p95"),
    ]
    _draw_bar_chart(
        painter,
        mean_rect.adjusted(18.0, 38.0, -18.0, -28.0),
        labels=stage_labels,
        values=mean_values,
        color=QColor("#0f766e"),
        y_label="ms",
        empty_text="Stage timing stats were not available.",
    )
    _draw_bar_chart(
        painter,
        p95_rect.adjusted(18.0, 38.0, -18.0, -28.0),
        labels=stage_labels,
        values=p95_values,
        color=QColor("#dc2626"),
        y_label="ms",
        empty_text="Stage timing stats were not available.",
    )
    painter.end()
    image.save(str(breakdown_path))


def _write_timeseries_plot(*, timeseries_path: Path, tracker_records: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    image = _new_image(1280, 760)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_title(
        painter,
        QRectF(34.0, 24.0, 1212.0, 56.0),
        "AURORA TIMING TIMESERIES",
        "Total backend sample time by analyzed tracker sample index. Duplicate frames are highlighted separately.",
    )
    chart_rect = QRectF(36.0, 104.0, 1208.0, 600.0)
    _draw_panel(painter, chart_rect, "Total Cycle Time vs Sample Index")
    points = []
    duplicate_points = []
    analyzed_index = 0
    for record in tracker_records:
        if bool(record.get("warmup_discarded", False)):
            continue
        total_cycle_ms = record.get("total_cycle_ms")
        if total_cycle_ms is None:
            continue
        point = (float(analyzed_index), float(total_cycle_ms))
        if bool(record.get("is_duplicate_frame", False)):
            duplicate_points.append(point)
        else:
            points.append(point)
        analyzed_index += 1
    _draw_line_chart(
        painter,
        chart_rect.adjusted(18.0, 38.0, -18.0, -28.0),
        points=points,
        overlay_points=duplicate_points,
        line_color=QColor("#2563eb"),
        overlay_color=QColor("#dc2626"),
        x_label="Analyzed Sample Index",
        y_label="Total Cycle Time (ms)",
        empty_text="No analyzed tracker timing samples were saved.",
        threshold_y=25.0,
    )
    painter.end()
    image.save(str(timeseries_path))


def _write_sync_plot(*, sync_plot_path: Path, metrics: dict[str, Any], servo_records: list[dict[str, Any]]) -> None:
    image = _new_image(1080, 680)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_title(
        painter,
        QRectF(34.0, 24.0, 1012.0, 56.0),
        "TRACKER / SERVO OFFSET",
        "Nearest-neighbor absolute time offsets between servo telemetry samples and analyzed tracker samples.",
    )
    chart_rect = QRectF(36.0, 104.0, 1008.0, 520.0)
    footer_rect = QRectF(36.0, 634.0, 1008.0, 22.0)
    _draw_panel(painter, chart_rect, "Servo -> Tracker Offset Histogram")
    servo_sync = dict(metrics.get("servo_sync", {}) or {})
    values = [float(value) for value in servo_sync.get("servo_to_tracker_offsets_ms", []) or []]
    _draw_histogram(
        painter,
        chart_rect.adjusted(18.0, 38.0, -18.0, -28.0),
        values=values,
        color=QColor("#7c3aed"),
        x_label="Absolute Offset (ms)",
        y_label="Count",
        empty_text="Servo logging was enabled, but no paired offsets were available.",
    )
    painter.setPen(QColor("#475569"))
    painter.setFont(_body_font())
    painter.drawText(
        footer_rect,
        Qt.AlignLeft | Qt.AlignVCenter,
        f"Servo samples logged: {len(servo_records)} | pairing threshold: {_fmt_number(servo_sync.get('pairing_threshold_ms'))} ms",
    )
    painter.end()
    image.save(str(sync_plot_path))


def _new_image(width: int, height: int) -> QImage:
    image = QImage(int(width), int(height), QImage.Format_ARGB32)
    image.fill(QColor("#f8fafc"))
    return image


def _draw_title(painter: QPainter, rect: QRectF, title: str, subtitle: str) -> None:
    painter.setFont(_title_font())
    painter.setPen(QColor("#0f172a"))
    painter.drawText(rect, Qt.AlignLeft | Qt.AlignTop, title)
    painter.setFont(_body_font())
    painter.setPen(QColor("#475569"))
    painter.drawText(rect.adjusted(0.0, 28.0, 0.0, 0.0), Qt.AlignLeft | Qt.AlignTop, subtitle)


def _draw_panel(painter: QPainter, rect: QRectF, title: str) -> None:
    painter.setPen(QPen(QColor("#dbe4ee"), 1.0))
    painter.setBrush(QColor("#ffffff"))
    painter.drawRoundedRect(rect, 14.0, 14.0)
    painter.setFont(_panel_font())
    painter.setPen(QColor("#0f172a"))
    painter.drawText(rect.adjusted(14.0, 10.0, -10.0, -10.0), Qt.AlignLeft | Qt.AlignTop, title)


def _draw_summary_pairs(painter: QPainter, rect: QRectF, pairs: list[tuple[str, str]]) -> None:
    y = rect.top()
    for label, value in pairs:
        painter.setFont(_body_font())
        painter.setPen(QColor("#475569"))
        painter.drawText(QRectF(rect.left(), y, rect.width() * 0.42, 20.0), Qt.AlignLeft | Qt.AlignVCenter, label)
        painter.setFont(_body_bold_font())
        painter.setPen(QColor("#0f172a"))
        painter.drawText(
            QRectF(rect.left() + rect.width() * 0.44, y, rect.width() * 0.54, 20.0),
            Qt.AlignLeft | Qt.AlignVCenter,
            value,
        )
        y += 23.0
        if y > rect.bottom() - 24.0:
            break


def _draw_histogram(
    painter: QPainter,
    rect: QRectF,
    *,
    values: list[float],
    color: QColor,
    x_label: str,
    y_label: str,
    empty_text: str,
) -> None:
    if not values:
        painter.setFont(_body_font())
        painter.setPen(QColor("#64748b"))
        painter.drawText(rect, Qt.AlignCenter, empty_text)
        return
    minimum = min(values)
    maximum = max(values)
    if maximum <= minimum:
        bins = [(minimum, minimum + 1.0, len(values))]
    else:
        bin_count = min(10, max(4, int(len(values) ** 0.5)))
        width = max(1e-6, (maximum - minimum) / float(bin_count))
        bins = []
        for index in range(bin_count):
            start = minimum + (index * width)
            end = maximum if index == bin_count - 1 else start + width
            count = sum(
                1
                for value in values
                if (
                    (value >= start and value <= end)
                    if index == bin_count - 1
                    else (value >= start and value < end)
                )
            )
            bins.append((start, end, count))
    labels = [f"{start:.1f}-{end:.1f}" for start, end, _count in bins]
    counts = [float(count) for _start, _end, count in bins]
    _draw_bar_chart(painter, rect, labels=labels, values=counts, color=color, y_label=y_label, empty_text=empty_text)
    painter.setPen(QColor("#475569"))
    painter.setFont(_body_font())
    painter.drawText(QRectF(rect.left(), rect.bottom() - 2.0, rect.width(), 18.0), Qt.AlignCenter, x_label)


def _draw_bar_chart(
    painter: QPainter,
    rect: QRectF,
    *,
    labels: list[str],
    values: list[float | None],
    color: QColor,
    y_label: str,
    empty_text: str,
) -> None:
    pairs = [(label, float(value)) for label, value in zip(labels, values) if value is not None]
    if not pairs:
        painter.setFont(_body_font())
        painter.setPen(QColor("#64748b"))
        painter.drawText(rect, Qt.AlignCenter, empty_text)
        return
    chart = rect.adjusted(40.0, 8.0, -10.0, -36.0)
    painter.setPen(QPen(QColor("#e2e8f0"), 1.0))
    painter.drawLine(QPointF(chart.left(), chart.bottom()), QPointF(chart.right(), chart.bottom()))
    painter.drawLine(QPointF(chart.left(), chart.top()), QPointF(chart.left(), chart.bottom()))
    max_value = max(value for _label, value in pairs)
    max_value = max(max_value, 1.0)
    bar_width = chart.width() / max(1, len(pairs))
    for index, (label, value) in enumerate(pairs):
        left = chart.left() + (index * bar_width) + 8.0
        usable_width = max(8.0, bar_width - 16.0)
        bar_height = (value / max_value) * max(1.0, chart.height() - 10.0)
        bar_rect = QRectF(left, chart.bottom() - bar_height, usable_width, bar_height)
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        painter.drawRoundedRect(bar_rect, 4.0, 4.0)
        painter.setFont(_body_font())
        painter.setPen(QColor("#475569"))
        painter.drawText(QRectF(left - 6.0, chart.bottom() + 4.0, usable_width + 12.0, 18.0), Qt.AlignHCenter, label)
        painter.drawText(QRectF(left - 6.0, bar_rect.top() - 18.0, usable_width + 12.0, 16.0), Qt.AlignHCenter, f"{value:.2f}")
    painter.drawText(QRectF(rect.left(), rect.top() - 2.0, rect.width(), 16.0), Qt.AlignRight, y_label)


def _draw_line_chart(
    painter: QPainter,
    rect: QRectF,
    *,
    points: list[tuple[float, float]],
    overlay_points: list[tuple[float, float]],
    line_color: QColor,
    overlay_color: QColor,
    x_label: str,
    y_label: str,
    empty_text: str,
    threshold_y: float | None = None,
) -> None:
    combined = list(points) + list(overlay_points)
    if not combined:
        painter.setFont(_body_font())
        painter.setPen(QColor("#64748b"))
        painter.drawText(rect, Qt.AlignCenter, empty_text)
        return
    chart = rect.adjusted(40.0, 8.0, -10.0, -36.0)
    x_values = [point[0] for point in combined]
    y_values = [point[1] for point in combined]
    min_x, max_x = _expand_range(min(x_values), max(x_values), minimum_span=5.0, pad_fraction=0.04)
    min_y, max_y = _expand_range(min(y_values), max(y_values), minimum_span=5.0, pad_fraction=0.10)
    painter.setPen(QPen(QColor("#e2e8f0"), 1.0))
    painter.drawLine(QPointF(chart.left(), chart.bottom()), QPointF(chart.right(), chart.bottom()))
    painter.drawLine(QPointF(chart.left(), chart.top()), QPointF(chart.left(), chart.bottom()))
    if threshold_y is not None:
        threshold_point_y = _scale_y(chart, float(threshold_y), min_y, max_y)
        painter.setPen(QPen(QColor("#f59e0b"), 1.0, Qt.DashLine))
        painter.drawLine(QPointF(chart.left(), threshold_point_y), QPointF(chart.right(), threshold_point_y))
    if len(points) >= 2:
        painter.setPen(QPen(line_color, 2.0))
        for start, end in zip(points[:-1], points[1:]):
            painter.drawLine(
                QPointF(_scale_x(chart, start[0], min_x, max_x), _scale_y(chart, start[1], min_y, max_y)),
                QPointF(_scale_x(chart, end[0], min_x, max_x), _scale_y(chart, end[1], min_y, max_y)),
            )
    painter.setBrush(line_color)
    painter.setPen(Qt.NoPen)
    for x_value, y_value in points:
        painter.drawEllipse(QPointF(_scale_x(chart, x_value, min_x, max_x), _scale_y(chart, y_value, min_y, max_y)), 3.0, 3.0)
    painter.setBrush(overlay_color)
    for x_value, y_value in overlay_points:
        painter.drawEllipse(QPointF(_scale_x(chart, x_value, min_x, max_x), _scale_y(chart, y_value, min_y, max_y)), 4.0, 4.0)
    painter.setPen(QColor("#475569"))
    painter.setFont(_body_font())
    painter.drawText(QRectF(rect.left(), rect.top() - 2.0, rect.width(), 16.0), Qt.AlignRight, y_label)
    painter.drawText(QRectF(rect.left(), rect.bottom() - 2.0, rect.width(), 18.0), Qt.AlignCenter, x_label)


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


def _maybe_metric(metrics: dict[str, Any], parent_key: str, child_key: str) -> float | None:
    parent = metrics.get(parent_key, {}) or {}
    if not isinstance(parent, dict):
        return None
    value = parent.get(child_key)
    return float(value) if value not in (None, "") else None


def _title_font() -> QFont:
    font = QFont()
    font.setPointSize(16)
    font.setBold(True)
    return font


def _panel_font() -> QFont:
    font = QFont()
    font.setPointSize(11)
    font.setBold(True)
    return font


def _body_font() -> QFont:
    font = QFont()
    font.setPointSize(9)
    return font


def _body_bold_font() -> QFont:
    font = _body_font()
    font.setBold(True)
    return font


def _fmt(value: Any, *, suffix: str = "") -> str:
    if value in (None, ""):
        return "n/a"
    return f"{float(value):.3f}{suffix}"


def _fmt_number(value: Any) -> str:
    if value in (None, ""):
        return "n/a"
    return f"{float(value):.3f}"
