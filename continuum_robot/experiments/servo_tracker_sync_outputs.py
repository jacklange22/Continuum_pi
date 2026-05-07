"""Saved artifacts for servo-tracker synchronization validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from continuum_robot.experiments.dataset_tools import extract_tip_or_tool_position_mm
from continuum_robot.experiments.tracker_timing_outputs import (
    _body_font,
    _draw_bar_chart,
    _draw_histogram,
    _draw_line_chart,
    _draw_panel,
    _draw_summary_pairs,
    _draw_title,
    _ensure_plot_qt_app,
    _fmt,
    _fmt_number,
    _new_image,
    _qt_plotting_is_safe,
)
try:
    from continuum_robot.gui.theme import COLORS
except Exception:
    class _FallbackColors:
        scene_truth = "#2563eb"
        scene_tip = "#f59e0b"
        scene_residual = "#dc2626"
        scene_live_0a = "#0f766e"
        scene_live_0b = "#0891b2"
        scene_measurement = "#7c3aed"

    COLORS = _FallbackColors()
from continuum_robot.tracking.timing_benchmark import (
    extract_servo_command_records,
    extract_servo_timing_records,
    extract_tracker_timing_records,
)
try:
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QColor, QPainter

    _QT_AVAILABLE = True
except ModuleNotFoundError:
    _QT_AVAILABLE = False


LOG = logging.getLogger(__name__)


def build_servo_tracker_sync_summary_pairs(*, metrics: dict[str, Any]) -> list[tuple[str, str]]:
    """Return compact GUI/report rows for one sync-validation run."""
    metrics = dict(metrics or {})
    sync = dict(metrics.get("servo_tracker_sync", {}) or {})
    motion = dict(metrics.get("tracker_motion_metric", {}) or {})
    return [
        ("Backend", str(metrics.get("backend_identity", "n/a") or "n/a")),
        ("Tool IDs", ", ".join(metrics.get("requested_tool_ids", []) or []) or "n/a"),
        ("Servo IDs", ", ".join(str(value) for value in (metrics.get("selected_servo_ids") or [])) or "n/a"),
        ("Motion Protocol", str(metrics.get("motion_protocol", "n/a") or "n/a")),
        ("Analyzed Tracker Samples", str(int(metrics.get("sample_count_analyzed", 0) or 0))),
        ("Servo Telemetry Samples", str(int(metrics.get("servo_telemetry_sample_count", 0) or 0))),
        ("Servo Command Samples", str(int(metrics.get("servo_command_sample_count", 0) or 0))),
        ("Effective Loop Hz", _fmt(metrics.get("effective_loop_rate_hz"), suffix=" Hz")),
        ("Unique-Frame Hz", _fmt(metrics.get("unique_frame_rate_hz"), suffix=" Hz")),
        ("Duplicate Ratio", _percent(metrics.get("duplicate_frame_ratio"))),
        ("Valid Transform Rate", _percent(metrics.get("valid_requested_tool_rate"))),
        (
            "Tracker->Servo Telemetry Mean",
            _fmt(sync.get("tracker_to_servo_telemetry_mean_offset_ms"), suffix=" ms"),
        ),
        (
            "Tracker->Servo Telemetry P95",
            _fmt(sync.get("tracker_to_servo_telemetry_p95_offset_ms"), suffix=" ms"),
        ),
        (
            "Tracker->Servo Command Mean",
            _fmt(sync.get("tracker_to_servo_command_mean_offset_ms"), suffix=" ms"),
        ),
        (
            "Tracker->Servo Command P95",
            _fmt(sync.get("tracker_to_servo_command_p95_offset_ms"), suffix=" ms"),
        ),
        ("Motion Metric Source", str(motion.get("source", "unavailable") or "unavailable")),
        ("Max Motion Metric", _fmt(motion.get("max_displacement_mm"), suffix=" mm")),
    ]


def build_servo_tracker_sync_summary_lines(*, metadata, summary, metrics: dict[str, Any]) -> list[str]:
    """Return human-readable summary text for one sync-validation run."""
    metrics = dict(metrics or {})
    sync = dict(metrics.get("servo_tracker_sync", {}) or {})
    lines = [
        "Servo-Tracker Synchronization Validation Summary",
        (
            "This run measures host-time alignment between tracker samples, servo commands, and servo telemetry "
            "during scripted motion. It does not claim hardware clock synchronization, closed-loop control quality, "
            "or spatial-calibration accuracy."
        ),
        "",
        f"Run ID: {metadata.run_id}",
        f"Timestamp: {metadata.timestamp_utc}",
        f"Status: {summary.status}",
    ]
    for label, value in build_servo_tracker_sync_summary_pairs(metrics=metrics):
        lines.append(f"{label}: {value}")
    lines.extend(
        [
            "",
            "Tracker total-cycle timing (ms): "
            f"min={_fmt_number((metrics.get('total_cycle_ms_stats', {}) or {}).get('min'))}, "
            f"median={_fmt_number((metrics.get('total_cycle_ms_stats', {}) or {}).get('median'))}, "
            f"mean={_fmt_number((metrics.get('total_cycle_ms_stats', {}) or {}).get('mean'))}, "
            f"p95={_fmt_number((metrics.get('total_cycle_ms_stats', {}) or {}).get('p95'))}, "
            f"p99={_fmt_number((metrics.get('total_cycle_ms_stats', {}) or {}).get('p99'))}, "
            f"max={_fmt_number((metrics.get('total_cycle_ms_stats', {}) or {}).get('max'))}",
            "Tracker matching thresholds:",
            f"- <= 5 ms: {_percent(sync.get('tracker_to_servo_telemetry_within_5ms_rate'))}",
            f"- <= 10 ms: {_percent(sync.get('tracker_to_servo_telemetry_within_10ms_rate'))}",
            f"- <= 20 ms: {_percent(sync.get('tracker_to_servo_telemetry_within_20ms_rate'))}",
            f"- <= 25 ms: {_percent(sync.get('tracker_to_servo_telemetry_within_25ms_rate'))}",
            "",
            "Interpretation:",
            "- Use unique-frame Hz, not loop Hz alone, to judge fresh-frame availability during motion.",
            "- Offsets are nearest-neighbor host-time offsets only.",
            "- Duplicate frames and invalid transforms remain visible so the run cannot hide stale backend behavior inside a single average.",
        ]
    )
    return lines


def write_servo_tracker_sync_outputs(*, output_dir: Path, metadata, summary, samples) -> dict[str, Path]:
    """Write stable artifacts for one servo-tracker sync validation run."""
    output_dir = Path(output_dir)
    histogram_path = output_dir / "servo_tracker_offset_histogram.png"
    timeseries_path = output_dir / "servo_tracker_offset_timeseries.png"
    pose_command_path = output_dir / "servo_tracker_pose_command_timeseries.png"
    validity_path = output_dir / "servo_tracker_validity_summary.png"
    summary_text_path = output_dir / "servo_tracker_sync_summary.txt"
    metrics = summary.experiment_metrics if isinstance(summary.experiment_metrics, dict) else {}
    tracker_records = extract_tracker_timing_records(samples)
    servo_telemetry_records = extract_servo_timing_records(samples)
    servo_command_records = extract_servo_command_records(samples)
    summary_text_path.write_text(
        "\n".join(
            build_servo_tracker_sync_summary_lines(
                metadata=metadata,
                summary=summary,
                metrics=metrics,
            )
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    if _qt_plotting_is_safe():
        _ensure_plot_qt_app()
        _write_offset_histogram(histogram_path=histogram_path, metrics=metrics)
        _write_offset_timeseries(timeseries_path=timeseries_path, metrics=metrics)
        _write_pose_command_timeseries(
            pose_command_path=pose_command_path,
            samples=samples,
            metrics=metrics,
        )
        _write_validity_summary(
            validity_path=validity_path,
            metrics=metrics,
            tracker_records=tracker_records,
            servo_telemetry_records=servo_telemetry_records,
            servo_command_records=servo_command_records,
        )
    else:
        _write_plot_placeholder(histogram_path)
        _write_plot_placeholder(timeseries_path)
        _write_plot_placeholder(pose_command_path)
        _write_plot_placeholder(validity_path)
        LOG.warning("Qt plotting backend unavailable; wrote placeholder servo-tracker sync plots under %s", output_dir)
    return {
        "offset_histogram_path": histogram_path,
        "offset_timeseries_path": timeseries_path,
        "pose_command_timeseries_path": pose_command_path,
        "validity_summary_path": validity_path,
        "summary_text_path": summary_text_path,
    }


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


def _primary_offset_series(metrics: dict[str, Any]) -> tuple[list[float], str]:
    sync = dict(metrics.get("servo_tracker_sync", {}) or {})
    telemetry_offsets = list(sync.get("tracker_to_servo_telemetry_offsets_ms", []) or [])
    if telemetry_offsets:
        return [float(value) for value in telemetry_offsets], "tracker to servo telemetry"
    command_offsets = list(sync.get("tracker_to_servo_command_offsets_ms", []) or [])
    if command_offsets:
        return [float(value) for value in command_offsets], "tracker to servo command"
    return [], "tracker to servo telemetry"


def _write_offset_histogram(*, histogram_path: Path, metrics: dict[str, Any]) -> None:
    offsets_ms, label = _primary_offset_series(metrics)
    image = _new_image(1280, 720)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_title(
        painter,
        QRectF(34.0, 24.0, 1212.0, 56.0),
        "SERVO-TRACKER OFFSET HISTOGRAM",
        "Distribution of nearest-neighbor host-time offsets during scripted motion.",
    )
    chart_rect = QRectF(36.0, 104.0, 820.0, 560.0)
    summary_rect = QRectF(888.0, 104.0, 356.0, 560.0)
    _draw_panel(painter, chart_rect, f"Offset Distribution ({label})")
    _draw_panel(painter, summary_rect, "Run Summary")
    _draw_histogram(
        painter,
        chart_rect.adjusted(18.0, 38.0, -18.0, -28.0),
        values=offsets_ms,
        color=QColor(COLORS.scene_truth),
        x_label="Absolute Host-Time Offset (ms)",
        y_label="Count",
        empty_text="No tracker-servo offset series was available for this run.",
    )
    _draw_summary_pairs(
        painter,
        summary_rect.adjusted(16.0, 36.0, -16.0, -16.0),
        build_servo_tracker_sync_summary_pairs(metrics=metrics),
    )
    painter.end()
    image.save(str(histogram_path))


def _write_offset_timeseries(*, timeseries_path: Path, metrics: dict[str, Any]) -> None:
    offsets_ms, label = _primary_offset_series(metrics)
    points = [(float(index), float(value)) for index, value in enumerate(offsets_ms)]
    image = _new_image(1280, 720)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_title(
        painter,
        QRectF(34.0, 24.0, 1212.0, 56.0),
        "SERVO-TRACKER OFFSET TIMESERIES",
        "Nearest-neighbor host-time offset by sample index across the scripted motion window.",
    )
    chart_rect = QRectF(36.0, 104.0, 1208.0, 560.0)
    _draw_panel(painter, chart_rect, f"Offset vs Sample Index ({label})")
    _draw_line_chart(
        painter,
        chart_rect.adjusted(18.0, 38.0, -18.0, -28.0),
        points=points,
        overlay_points=[],
        line_color=QColor(COLORS.scene_tip),
        overlay_color=QColor(COLORS.scene_residual),
        x_label="Sample Index",
        y_label="Absolute Offset (ms)",
        empty_text="No tracker-servo offset series was available for this run.",
        threshold_y=25.0,
    )
    painter.end()
    image.save(str(timeseries_path))


def _write_pose_command_timeseries(*, pose_command_path: Path, samples, metrics: dict[str, Any]) -> None:
    requested_tool_ids = [str(value) for value in (metrics.get("requested_tool_ids") or ["0A"])]
    command_points, measured_points = _servo_motion_series(samples)
    tracker_points, source_label = _tracker_motion_series(
        samples=samples,
        requested_tool_ids=requested_tool_ids,
        prefer_robot_frame_tip=bool(metrics.get("include_robot_frame_tip_pose", True)),
    )
    image = _new_image(1280, 920)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_title(
        painter,
        QRectF(34.0, 24.0, 1212.0, 56.0),
        "SERVO COMMAND / TELEMETRY / TRACKER MOTION",
        "Aligned time-series used to judge whether servo-side and tracker-side motion are usable together in later dynamic experiments.",
    )
    command_rect = QRectF(36.0, 104.0, 1208.0, 224.0)
    measured_rect = QRectF(36.0, 348.0, 1208.0, 224.0)
    tracker_rect = QRectF(36.0, 592.0, 1208.0, 224.0)
    _draw_panel(painter, command_rect, "Servo Command Travel Metric")
    _draw_panel(painter, measured_rect, "Servo Measured Travel Metric")
    _draw_panel(painter, tracker_rect, f"Tracker Motion Metric ({source_label})")
    _draw_line_chart(
        painter,
        command_rect.adjusted(18.0, 38.0, -18.0, -28.0),
        points=command_points,
        overlay_points=[],
        line_color=QColor(COLORS.scene_live_0a),
        overlay_color=QColor(COLORS.scene_live_0a),
        x_label="Monotonic Time (s)",
        y_label="Travel Metric (ticks)",
        empty_text="No servo command samples were saved.",
        threshold_y=None,
    )
    _draw_line_chart(
        painter,
        measured_rect.adjusted(18.0, 38.0, -18.0, -28.0),
        points=measured_points,
        overlay_points=[],
        line_color=QColor(COLORS.scene_live_0b),
        overlay_color=QColor(COLORS.scene_live_0b),
        x_label="Monotonic Time (s)",
        y_label="Travel Metric (ticks)",
        empty_text="No servo telemetry samples were saved.",
        threshold_y=None,
    )
    _draw_line_chart(
        painter,
        tracker_rect.adjusted(18.0, 38.0, -18.0, -28.0),
        points=tracker_points,
        overlay_points=[],
        line_color=QColor(COLORS.scene_tip),
        overlay_color=QColor(COLORS.scene_tip),
        x_label="Monotonic Time (s)",
        y_label="Displacement (mm)",
        empty_text="No tracker motion metric was available for this run.",
        threshold_y=None,
    )
    painter.end()
    image.save(str(pose_command_path))


def _write_validity_summary(
    *,
    validity_path: Path,
    metrics: dict[str, Any],
    tracker_records: list[dict[str, Any]],
    servo_telemetry_records: list[dict[str, Any]],
    servo_command_records: list[dict[str, Any]],
) -> None:
    sync = dict(metrics.get("servo_tracker_sync", {}) or {})
    per_tool_summary = dict(metrics.get("per_tool_summary", {}) or {})
    threshold_labels = ["<=5 ms", "<=10 ms", "<=20 ms", "<=25 ms"]
    threshold_values = [
        100.0 * float(sync.get("tracker_to_servo_telemetry_within_5ms_rate", 0.0) or 0.0),
        100.0 * float(sync.get("tracker_to_servo_telemetry_within_10ms_rate", 0.0) or 0.0),
        100.0 * float(sync.get("tracker_to_servo_telemetry_within_20ms_rate", 0.0) or 0.0),
        100.0 * float(sync.get("tracker_to_servo_telemetry_within_25ms_rate", 0.0) or 0.0),
    ]
    image = _new_image(1240, 760)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_title(
        painter,
        QRectF(34.0, 24.0, 1172.0, 56.0),
        "SERVO-TRACKER VALIDITY SUMMARY",
        "Tracker freshness, transform validity, duplicate behavior, and host-time match-rate thresholds during motion.",
    )
    threshold_rect = QRectF(36.0, 104.0, 560.0, 600.0)
    tool_rect = QRectF(616.0, 104.0, 284.0, 600.0)
    summary_rect = QRectF(920.0, 104.0, 284.0, 600.0)
    _draw_panel(painter, threshold_rect, "Tracker->Servo Telemetry Match Rate")
    _draw_panel(painter, tool_rect, "Per-Tool Valid Transform Rate")
    _draw_panel(painter, summary_rect, "Run Summary")
    _draw_bar_chart(
        painter,
        threshold_rect.adjusted(18.0, 38.0, -18.0, -28.0),
        labels=threshold_labels,
        values=threshold_values,
        color=QColor(COLORS.scene_truth),
        y_label="Matched (%)",
        empty_text="No tracker-servo telemetry match series was available.",
    )
    _draw_bar_chart(
        painter,
        tool_rect.adjusted(18.0, 38.0, -18.0, -28.0),
        labels=list(per_tool_summary.keys()),
        values=[
            100.0 * float(summary.get("valid_transform_rate", 0.0) or 0.0)
            for summary in per_tool_summary.values()
        ],
        color=QColor(COLORS.scene_measurement),
        y_label="Valid (%)",
        empty_text="Per-tool validity was not available.",
    )
    summary_pairs = build_servo_tracker_sync_summary_pairs(metrics=metrics)
    summary_pairs.extend(
        [
            ("Tracker Records", str(len(tracker_records))),
            ("Servo Telemetry", str(len(servo_telemetry_records))),
            ("Servo Commands", str(len(servo_command_records))),
        ]
    )
    _draw_summary_pairs(
        painter,
        summary_rect.adjusted(16.0, 36.0, -16.0, -16.0),
        summary_pairs,
    )
    painter.end()
    image.save(str(validity_path))


def _servo_motion_series(samples) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    command_reference_by_servo: dict[int, float] = {}
    measured_reference_by_servo: dict[int, float] = {}
    command_rows: dict[float, list[float]] = {}
    measured_rows: dict[float, list[float]] = {}
    for sample in samples:
        extra = dict(getattr(sample, "extra", {}) or {})
        record_kind = str(extra.get("record_kind", "")).strip().lower()
        if record_kind == "servo_command":
            servo_id = extra.get("servo_id")
            commanded = extra.get("commanded_position_ticks")
            if servo_id is None or commanded is None:
                continue
            servo_id = int(servo_id)
            commanded_value = float(commanded)
            command_reference_by_servo.setdefault(servo_id, commanded_value)
            command_rows.setdefault(float(sample.monotonic_time_s), []).append(
                abs(commanded_value - command_reference_by_servo[servo_id])
            )
        elif record_kind == "servo_timing":
            servo_id = extra.get("servo_id")
            measured = extra.get("present_position_ticks")
            if servo_id is None or measured is None:
                continue
            servo_id = int(servo_id)
            measured_value = float(measured)
            measured_reference_by_servo.setdefault(servo_id, measured_value)
            measured_rows.setdefault(float(sample.monotonic_time_s), []).append(
                abs(measured_value - measured_reference_by_servo[servo_id])
            )
    command_points = [
        (time_s, float(np.linalg.norm(np.asarray(values, dtype=float))))
        for time_s, values in sorted(command_rows.items())
        if values
    ]
    measured_points = [
        (time_s, float(np.linalg.norm(np.asarray(values, dtype=float))))
        for time_s, values in sorted(measured_rows.items())
        if values
    ]
    return command_points, measured_points


def _tracker_motion_series(
    *,
    samples,
    requested_tool_ids: list[str],
    prefer_robot_frame_tip: bool,
) -> tuple[list[tuple[float, float]], str]:
    reference_position: np.ndarray | None = None
    points: list[tuple[float, float]] = []
    source_label = "unavailable"
    for sample in samples:
        extra = dict(getattr(sample, "extra", {}) or {})
        if str(extra.get("record_kind", "")).strip().lower() != "tracker_timing":
            continue
        if bool(extra.get("warmup_discarded", False)):
            continue
        position_mm = None
        if prefer_robot_frame_tip:
            position_mm, frame_name = extract_tip_or_tool_position_mm(
                sample,
                tool_id=str(requested_tool_ids[0] if requested_tool_ids else "0A"),
                prefer_robot_frame=True,
            )
            if position_mm is not None and frame_name == "robot":
                source_label = "robot tip displacement"
        if position_mm is None:
            for tool_id in requested_tool_ids:
                position_mm, _ = extract_tip_or_tool_position_mm(
                    sample,
                    tool_id=str(tool_id),
                    prefer_robot_frame=False,
                )
                if position_mm is not None:
                    source_label = f"{tool_id} tracker displacement"
                    break
        if position_mm is None:
            continue
        vector = np.asarray(position_mm, dtype=float)
        if reference_position is None:
            reference_position = vector
        points.append((float(sample.monotonic_time_s), float(np.linalg.norm(vector - reference_position))))
    return points, source_label


def _percent(value: Any) -> str:
    if value in (None, ""):
        return "n/a"
    return f"{100.0 * float(value):.1f}%"
