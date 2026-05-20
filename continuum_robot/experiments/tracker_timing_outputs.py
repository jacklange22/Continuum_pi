"""Saved artifacts for the Aurora backend timing diagnostic."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

try:
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
    from PySide6.QtWidgets import QApplication

    _QT_AVAILABLE = True
except ModuleNotFoundError:
    _QT_AVAILABLE = False

from continuum_robot.experiments.plotting import (
    color,
    create_figure,
    legend,
    report_style,
    save_figure,
    style_axes,
)
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


AURORA_THEORETICAL_INTERVAL_MS = 25.0  # 40 Hz Aurora hardware frame rate (ceiling, not a target)


def write_tracker_timing_outputs(*, output_dir: Path, metadata, summary, samples) -> dict[str, Path]:
    """Write canonical artifacts for one timing-diagnostic run.

    Figure contract: 2 thesis-quality PNGs only.
      - thesis_01_cycle_time_distribution.png: histogram + CDF of per-cycle
        total time with reference at 25 ms (40 Hz Aurora theoretical).
      - thesis_02_stage_time_budget.png: horizontal stacked bar at four
        percentiles (median / mean / p95 / p99), each decomposed by stage
        (backend_call / parse / state_commit), explaining the rate gap.
    Everything else (duplicate frame stats, per-tool valid rates, servo-sync
    cross-stream offsets, raw error counts) goes into debug.json. The old
    Qt-rendered histogram / breakdown / timeseries / sync-offsets PNGs and the
    aurora_timing_summary.txt are intentionally NOT written.
    """
    output_dir = Path(output_dir)
    debug_json_path = output_dir / "debug.json"
    thesis_01_path = output_dir / "thesis_01_cycle_time_distribution.png"
    thesis_02_path = output_dir / "thesis_02_stage_time_budget.png"
    metrics = summary.experiment_metrics if isinstance(summary.experiment_metrics, dict) else {}
    tracker_records = extract_tracker_timing_records(samples)
    servo_records = extract_servo_timing_records(samples)

    _write_tracker_debug_json(
        path=debug_json_path,
        output_dir=output_dir,
        metadata=metadata,
        summary=summary,
        metrics=metrics,
        tracker_records=tracker_records,
        servo_records=servo_records,
    )
    for path, writer in [
        (thesis_01_path, lambda: _write_tracker_thesis_01_cycle_distribution(
            path=thesis_01_path, tracker_records=tracker_records, metrics=metrics,
        )),
        (thesis_02_path, lambda: _write_tracker_thesis_02_stage_breakdown(
            path=thesis_02_path, tracker_records=tracker_records, metrics=metrics,
        )),
    ]:
        try:
            writer()
        except Exception:
            _write_plot_placeholder(path)
    return {
        "debug_json_path": debug_json_path,
        "thesis_01_path": thesis_01_path,
        "thesis_02_path": thesis_02_path,
    }


def _filter_analyzed_cycle_times(tracker_records: list[dict[str, Any]]) -> list[float]:
    """Return non-warmup total_cycle_ms values for analysis."""
    values: list[float] = []
    for record in tracker_records:
        if record.get("warmup_discarded"):
            continue
        total = record.get("total_cycle_ms")
        if total is None:
            continue
        try:
            values.append(float(total))
        except (TypeError, ValueError):
            continue
    return values


def _filter_analyzed_stage_arrays(tracker_records: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Return per-stage timing arrays (non-warmup), keyed by stage name."""
    stages = {"backend_call_ms": [], "parse_ms": [], "state_commit_ms": [], "total_cycle_ms": []}
    for record in tracker_records:
        if record.get("warmup_discarded"):
            continue
        for key in stages:
            value = record.get(key)
            if value is None:
                continue
            try:
                stages[key].append(float(value))
            except (TypeError, ValueError):
                continue
    return stages


def _stage_percentile_table(stages: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """For each stage compute median/mean/p95/p99 as a dict."""
    out: dict[str, dict[str, float]] = {}
    for stage, values in stages.items():
        if not values:
            out[stage] = {"median": 0.0, "mean": 0.0, "p95": 0.0, "p99": 0.0}
            continue
        arr = np.asarray(values, dtype=float)
        out[stage] = {
            "median": float(np.median(arr)),
            "mean": float(arr.mean()),
            "p95": float(np.percentile(arr, 95.0)),
            "p99": float(np.percentile(arr, 99.0)),
        }
    return out


def _hz_for_ms(value_ms: float) -> float | None:
    if value_ms is None or value_ms <= 0.0:
        return None
    return 1000.0 / float(value_ms)


def _write_tracker_thesis_01_cycle_distribution(
    *,
    path: Path,
    tracker_records: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    """Thesis figure 1: cycle-time distribution + CDF vs Aurora frame interval.

    Answers "how fast and how consistent is the tracker poll loop?" in one
    chart. Histogram (count) on left axis, cumulative fraction on right axis.
    The 25 ms vertical line is the Aurora hardware frame interval — NOT a
    pipeline target. Cycles shorter than 25 ms mean the loop is over-polling
    a cached frame (duplicate frames likely); cycles longer mean the loop is
    under-polling (Aurora frames will be missed). Observed mean, p95, p99
    marked with vertical lines.
    """
    values = _filter_analyzed_cycle_times(tracker_records)
    fig, ax = create_figure(size="wide", constrained_layout=False)
    fig.subplots_adjust(left=0.085, right=0.92, top=0.90, bottom=0.22)

    if not values:
        ax.text(0.5, 0.5, "No analyzed tracker cycles available",
                transform=ax.transAxes, ha="center", va="center")
        style_axes(ax, xlabel="Total cycle time (ms)", ylabel="Cycle count")
        fig.suptitle("Tracker Cycle Time Distribution vs Aurora Frame Interval",
                     fontsize=13, fontweight="bold", x=0.04, ha="left")
        save_figure(fig, path)
        return

    arr = np.asarray(values, dtype=float)
    mean_ms = float(arr.mean())
    p95_ms = float(np.percentile(arr, 95.0))
    p99_ms = float(np.percentile(arr, 99.0))
    # `_hz_for_ms(mean_ms)` is 1000/mean(ms), which is NOT the same as the
    # mean of (1000/ms) and not the same as the (N-1)/wall_duration rate
    # the GUI surfaces as `effective_loop_rate_hz`. We always prefer the
    # canonical rate from metrics so the figure agrees with the GUI/summary.
    effective_loop_rate_hz = metrics.get("effective_loop_rate_hz")
    if effective_loop_rate_hz is None:
        # Fall back to 1000/mean(ms) only if the canonical rate isn't on
        # the metrics dict; explicit label tells the reader why.
        effective_loop_rate_hz = _hz_for_ms(mean_ms) or 0.0
        effective_rate_label = "1000/mean(ms)"
    else:
        effective_loop_rate_hz = float(effective_loop_rate_hz)
        effective_rate_label = "(N-1)/wall"
    # p95-equivalent rate (1000/p95_ms) is reported separately as a
    # "what's the worst end of the distribution" indicator.
    realized_p95_hz = _hz_for_ms(p95_ms) or 0.0

    x_min = max(0.0, float(arr.min()) * 0.95)
    x_max = max(float(arr.max()) * 1.05, AURORA_THEORETICAL_INTERVAL_MS * 1.2)

    # No shaded zone: the previous green "made 40 Hz budget" shade implied
    # short cycles are good, which is misleading — short cycles indicate the
    # loop is over-polling a cached frame. The 25 ms line (drawn below) and
    # the caption strip explain the duality without the false-good shade.

    bin_count = max(20, min(60, int(np.sqrt(len(arr)) * 2)))
    counts, bins, _patches = ax.hist(
        arr, bins=bin_count, range=(x_min, x_max),
        color=color("measured"), edgecolor="white", linewidth=0.6, alpha=0.85, zorder=2,
    )

    # CDF overlay on twin y-axis
    cdf_ax = ax.twinx()
    sorted_vals = np.sort(arr)
    cdf_y = np.arange(1, len(sorted_vals) + 1) / float(len(sorted_vals))
    cdf_ax.plot(sorted_vals, cdf_y, color=color("reference"), linewidth=1.6, alpha=0.85, zorder=3, label="CDF")
    cdf_ax.set_ylim(0.0, 1.05)
    cdf_ax.set_ylabel("Cumulative fraction", color=color("reference"))
    cdf_ax.tick_params(axis="y", colors=color("reference"))
    cdf_ax.spines["top"].set_visible(False)
    cdf_ax.spines["right"].set_color(color("reference"))

    # Reference lines. The 25 ms line is the Aurora hardware frame interval
    # (the rate at which the device actually produces fresh frames). It is
    # NOT a pipeline target — labeling it as such would suggest shorter
    # cycles are better, which is the opposite of the truth.
    ax.axvline(
        AURORA_THEORETICAL_INTERVAL_MS,
        color=color("threshold"), linestyle="-", linewidth=1.6,
        label=(
            f"Aurora frame interval "
            f"({AURORA_THEORETICAL_INTERVAL_MS:.0f} ms = 40 Hz hardware rate)"
        ),
        zorder=4,
    )
    ax.axvline(mean_ms, color=color("fit"), linestyle="--", linewidth=1.4,
               label=f"Mean ({mean_ms:.1f} ms)", zorder=4)
    ax.axvline(p95_ms, color=color("rejected"), linestyle=":", linewidth=1.4,
               label=f"p95 ({p95_ms:.1f} ms)", zorder=4)
    ax.axvline(p99_ms, color=color("rejected"), linestyle=":", linewidth=1.0, alpha=0.7,
               label=f"p99 ({p99_ms:.1f} ms)", zorder=4)

    style_axes(ax, xlabel="Total cycle time (ms)", ylabel="Cycle count")
    ax.set_xlim(x_min, x_max)
    legend(ax, loc="upper right", ncol=1)

    fig.suptitle("Tracker Cycle Time Distribution vs Aurora Frame Interval",
                 fontsize=13, fontweight="bold", x=0.04, ha="left")

    duplicate_ratio = _safe_ratio(metrics.get("duplicate_frame_ratio"))
    fig.text(
        0.015, 0.02,
        "  •  ".join(
            _strip_empty([
                f"Samples: {len(values)}",
                # Canonical effective rate matches summary.json and the GUI
                # `effective_loop_rate_hz` field. The p95-equivalent uses
                # 1000/p95_ms to characterize the worst end of the
                # distribution. Both rate definitions are spelled out so
                # the reader can reproduce them.
                f"Effective loop rate: {float(effective_loop_rate_hz):.1f} Hz "
                f"[{effective_rate_label}]   ·   1000/p95 ms: {realized_p95_hz:.1f} Hz",
                "Aurora hardware ceiling: 40 Hz (25 ms/frame). "
                "Cycle < 25 ms → over-poll, expect duplicate frames; "
                "cycle > 25 ms → under-poll, expect missed frames.",
                f"Duplicate frames: {duplicate_ratio:.1f}%" if duplicate_ratio is not None else None,
            ])
        ),
        fontsize=9, color=color("text"), ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": color("grid"), "alpha": 0.94},
    )
    save_figure(fig, path)


def _write_tracker_thesis_02_stage_breakdown(
    *,
    path: Path,
    tracker_records: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    """Thesis figure 2: where the per-cycle time is spent at four percentiles.

    Horizontal stacked bars at median / mean / p95 / p99 of total_cycle_ms,
    each decomposed into backend_call / parse / state_commit. Each bar
    labeled with total + equivalent Hz. The 25 ms reference line is the
    Aurora hardware frame interval (NOT a pipeline target — cycles shorter
    than 25 ms mean over-polling a cached frame; cycles longer mean missed
    frames). Reads at a glance: which stage dominates the cycle time.
    """
    stages = _filter_analyzed_stage_arrays(tracker_records)
    table = _stage_percentile_table(stages)

    fig, ax = create_figure(size="wide", constrained_layout=False)
    fig.subplots_adjust(left=0.135, right=0.97, top=0.90, bottom=0.22)

    percentile_labels = ["median", "mean", "p95", "p99"]
    if not stages["total_cycle_ms"]:
        ax.text(0.5, 0.5, "No analyzed stage timings available",
                transform=ax.transAxes, ha="center", va="center")
        style_axes(ax, xlabel="Time per cycle (ms)", ylabel="Percentile")
        fig.suptitle("Tracker Per-Cycle Time Budget by Stage",
                     fontsize=13, fontweight="bold", x=0.04, ha="left")
        save_figure(fig, path)
        return

    stage_colors = {
        "backend_call_ms": color("measured"),
        "parse_ms": color("fit"),
        "state_commit_ms": color("model"),
    }
    stage_display = {
        "backend_call_ms": "backend_call (Aurora I/O)",
        "parse_ms": "parse (decode response)",
        "state_commit_ms": "state_commit (publish to listeners)",
    }

    y_positions = np.arange(len(percentile_labels))[::-1]  # top-to-bottom: median, mean, p95, p99
    bar_height = 0.62
    for stage_index, stage_key in enumerate(["backend_call_ms", "parse_ms", "state_commit_ms"]):
        left_offsets = []
        widths = []
        for pct in percentile_labels:
            cumulative_before = sum(
                table[prior_stage][pct]
                for prior_stage in ["backend_call_ms", "parse_ms", "state_commit_ms"][:stage_index]
            )
            left_offsets.append(cumulative_before)
            widths.append(table[stage_key][pct])
        ax.barh(
            y_positions, widths, bar_height,
            left=left_offsets,
            color=stage_colors[stage_key],
            edgecolor="white", linewidth=0.6,
            label=stage_display[stage_key],
            zorder=2,
        )

    # Per-bar total + Hz label on the right
    max_total = 0.0
    for y_pos, pct in zip(y_positions, percentile_labels):
        total = table["total_cycle_ms"][pct]
        max_total = max(max_total, total)
        hz = _hz_for_ms(total)
        label = f"{total:.1f} ms" + (f" ≈ {hz:.1f} Hz" if hz else "")
        ax.text(total + max(max_total * 0.012, 0.4), y_pos, label,
                va="center", ha="left", fontsize=9, color=color("text"))

    # Aurora hardware frame interval reference (NOT a pipeline target —
    # see the figure docstring). Same labeling convention as thesis_01 so
    # the two figures read consistently.
    ax.axvline(
        AURORA_THEORETICAL_INTERVAL_MS, color=color("threshold"), linestyle="-",
        linewidth=1.6,
        label=(
            f"Aurora frame interval "
            f"({AURORA_THEORETICAL_INTERVAL_MS:.0f} ms = 40 Hz hardware rate)"
        ),
        zorder=3,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([label.upper() for label in percentile_labels])
    style_axes(ax, xlabel="Time per cycle (ms)", ylabel="Percentile")
    # Leave headroom on the right for per-bar labels
    ax.set_xlim(0.0, max(max_total * 1.30, AURORA_THEORETICAL_INTERVAL_MS * 1.2))
    legend(ax, loc="lower right", ncol=1)

    fig.suptitle("Tracker Per-Cycle Time Budget by Stage",
                 fontsize=13, fontweight="bold", x=0.04, ha="left")

    n_samples = len(stages["total_cycle_ms"])
    dominant_stage = max(
        ["backend_call_ms", "parse_ms", "state_commit_ms"],
        key=lambda s: table[s]["mean"],
    )
    dominant_pct = (
        100.0 * table[dominant_stage]["mean"] / table["total_cycle_ms"]["mean"]
        if table["total_cycle_ms"]["mean"] > 0 else 0.0
    )
    fig.text(
        0.015, 0.02,
        "  •  ".join(
            _strip_empty([
                f"Samples: {n_samples}",
                f"Dominant stage: {stage_display[dominant_stage].split(' (')[0]} ({dominant_pct:.0f}% of mean cycle)",
                # 1000/mean(ms) is reported as an EQUIVALENT rate of the
                # mean cycle time; this is NOT the (N-1)/wall_duration
                # `effective_loop_rate_hz` reported in summary.json. The
                # caption spells out which definition is in use.
                f"1000/mean cycle ms: {_hz_for_ms(table['total_cycle_ms']['mean']):.1f} Hz "
                f"(vs 40 Hz Aurora hardware ceiling; see summary.json for "
                "canonical effective_loop_rate_hz)"
                if table["total_cycle_ms"]["mean"] > 0 else None,
            ])
        ),
        fontsize=9, color=color("text"), ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": color("grid"), "alpha": 0.94},
    )
    save_figure(fig, path)


def _write_tracker_debug_json(
    *,
    path: Path,
    output_dir: Path,
    metadata,
    summary,
    metrics: dict[str, Any],
    tracker_records: list[dict[str, Any]],
    servo_records: list[dict[str, Any]],
) -> None:
    """Consolidate every diagnostic that doesn't make it onto thesis figures.

    Duplicate-frame stats, per-tool valid-transform rates, raw error sample
    counts, per-stage detailed stats, servo-sync cross-stream offsets if
    servo logging was enabled, and the run_review sidecar payload.
    """
    duplicate_count = int(metrics.get("duplicate_frame_count", 0) or 0)
    duplicate_ratio = _safe_ratio(metrics.get("duplicate_frame_ratio"))
    per_tool_summary = dict(metrics.get("per_tool_summary", {}) or {})

    review_payload: dict[str, Any] | None = None
    review_path = output_dir / "run_review.json"
    if review_path.exists():
        try:
            review_payload = json.loads(review_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            review_payload = {"parse_error": True}

    sync_block: dict[str, Any] | None = None
    sync_metrics = dict(metrics.get("servo_sync", {}) or {})
    if sync_metrics.get("enabled"):
        sync_block = {
            "enabled": True,
            "servo_telemetry_sample_count": len(servo_records),
            "mean_offset_ms": sync_metrics.get("mean_offset_ms"),
            "p95_offset_ms": sync_metrics.get("p95_offset_ms"),
            "max_offset_ms": sync_metrics.get("max_offset_ms"),
            "offsets_ms": list(sync_metrics.get("offsets_ms", []) or []),
        }

    payload = {
        "schema_version": "1.0",
        "run_id": getattr(metadata, "run_id", None),
        "experiment_name": getattr(metadata, "experiment_name", "tracker_timing_validation"),
        "status": getattr(summary, "status", None),
        "sample_count_analyzed": int(metrics.get("sample_count_analyzed", 0) or 0),
        "warmup_discarded_count": int(metrics.get("warmup_discarded_count", 0) or 0),
        "rate": {
            "effective_loop_rate_hz": metrics.get("effective_loop_rate_hz"),
            "unique_frame_rate_hz": metrics.get("unique_frame_rate_hz"),
            "aurora_theoretical_max_hz": 1000.0 / AURORA_THEORETICAL_INTERVAL_MS,
        },
        "duplicate_frames": {
            "count": duplicate_count,
            "ratio_percent": duplicate_ratio,
            "note": "Aurora returns the same frame_number on consecutive reads when no new hardware frame is ready; these cycles do not deliver fresh data.",
        },
        "stage_stats": {
            "backend_call_ms": dict(metrics.get("backend_call_ms_stats", {}) or {}),
            "parse_ms": dict(metrics.get("parse_ms_stats", {}) or {}),
            "state_commit_ms": dict(metrics.get("state_commit_ms_stats", {}) or {}),
            "total_cycle_ms": dict(metrics.get("total_cycle_ms_stats", {}) or {}),
        },
        "per_tool_valid_rate": {
            tool_id: float(s.get("valid_transform_rate", 0.0) or 0.0)
            for tool_id, s in per_tool_summary.items()
        },
        "errors": {
            "error_sample_count": int(metrics.get("error_sample_count", 0) or 0),
            "invalid_or_missing_requested_tool_count": int(
                metrics.get("invalid_or_missing_requested_tool_sample_count", 0) or 0
            ),
            "invalid_or_missing_requested_tool_ratio": metrics.get(
                "invalid_or_missing_requested_tool_ratio"
            ),
        },
        "servo_sync": sync_block,
        "backend": {
            "backend_identity": metrics.get("backend_identity"),
            "configured_backend_name": metrics.get("configured_backend_name"),
            "selected_backend_name": metrics.get("selected_backend_name"),
            "requested_tool_ids": list(metrics.get("requested_tool_ids", []) or []),
        },
        "run_review": review_payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_debug_json_default), encoding="utf-8")


def _safe_ratio(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value) * 100.0
    except (TypeError, ValueError):
        return None


def _strip_empty(items: list[str | None]) -> list[str]:
    return [str(item) for item in items if item]


def _debug_json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unserialisable type: {type(value).__name__}")


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
