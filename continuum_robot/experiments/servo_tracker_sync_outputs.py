"""Saved artifacts for servo-tracker synchronization validation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from continuum_robot.experiments.dataset_tools import extract_tip_or_tool_position_mm
from continuum_robot.experiments.plotting import (
    color,
    create_figure,
    legend,
    report_style,
    save_figure,
    style_axes,
)
from continuum_robot.experiments.tracker_timing_outputs import _write_plot_placeholder
from continuum_robot.tracking.timing_benchmark import (
    extract_servo_command_records,
    extract_servo_timing_records,
    extract_tracker_timing_records,
)


LOG = logging.getLogger(__name__)


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


def write_servo_tracker_sync_outputs(*, output_dir: Path, metadata, summary, samples) -> dict[str, Path]:
    """Write canonical artifacts for one servo-tracker sync validation run.

    Figure contract: 2 thesis-quality PNGs only.
      - thesis_01_pair_time_alignment.png: histogram + CDF of nearest-neighbor
        host-time offsets between tracker frames and servo telemetry samples,
        with reference thresholds at 5 / 10 / 25 ms and per-threshold rates.
      - thesis_02_motion_correspondence.png: stacked-panel time series. Top
        panel = servo measured position vs time, bottom panel = tracker tip
        displacement vs time. Same X axis. Demonstrates that the motion
        commanded on the servo side actually appears in the tracker stream at
        the same time.
    debug.json holds threshold-bucket rates, per-tool valid-transform rates,
    servo telemetry/command counts, motion magnitudes, motion source label,
    and the run_review sidecar payload. The old Qt-rendered offset
    histogram/timeseries/pose-command/validity-summary PNGs and the .txt
    summary are intentionally NOT written.
    """
    output_dir = Path(output_dir)
    debug_json_path = output_dir / "debug.json"
    thesis_01_path = output_dir / "thesis_01_pair_time_alignment.png"
    thesis_02_path = output_dir / "thesis_02_motion_correspondence.png"
    # Phase-4 supplementary figures (operator-spec request).
    events_on_timeline_path = output_dir / "sync_command_events_on_tracker_timeline.png"
    command_offset_hist_path = output_dir / "sync_command_offset_histogram.png"
    telemetry_offset_hist_path = output_dir / "sync_telemetry_offset_histogram.png"
    frame_age_over_time_path = output_dir / "sync_tracker_frame_age_at_servo_events.png"
    telemetry_interval_path = output_dir / "sync_servo_telemetry_interval_histogram.png"
    combined_summary_path = output_dir / "sync_combined_timing_summary.png"

    metrics = summary.experiment_metrics if isinstance(summary.experiment_metrics, dict) else {}
    tracker_records = extract_tracker_timing_records(samples)
    servo_telemetry_records = extract_servo_timing_records(samples)
    servo_command_records = extract_servo_command_records(samples)

    _write_servo_debug_json(
        path=debug_json_path,
        output_dir=output_dir,
        metadata=metadata,
        summary=summary,
        metrics=metrics,
        samples=samples,
    )
    for path, writer in [
        (thesis_01_path, lambda: _write_servo_thesis_01_alignment(path=thesis_01_path, metrics=metrics)),
        (thesis_02_path, lambda: _write_servo_thesis_02_motion_correspondence(
            path=thesis_02_path, samples=samples, metrics=metrics,
        )),
        (events_on_timeline_path, lambda: _write_sync_command_events_on_tracker_timeline(
            path=events_on_timeline_path,
            tracker_records=tracker_records,
            servo_command_records=servo_command_records,
        )),
        (command_offset_hist_path, lambda: _write_sync_command_offset_histogram(
            path=command_offset_hist_path, metrics=metrics,
        )),
        (telemetry_offset_hist_path, lambda: _write_sync_telemetry_offset_histogram(
            path=telemetry_offset_hist_path, metrics=metrics,
        )),
        (frame_age_over_time_path, lambda: _write_sync_tracker_frame_age_at_servo_events(
            path=frame_age_over_time_path,
            tracker_records=tracker_records,
            servo_command_records=servo_command_records,
            servo_telemetry_records=servo_telemetry_records,
        )),
        (telemetry_interval_path, lambda: _write_sync_servo_telemetry_interval_histogram(
            path=telemetry_interval_path,
            servo_telemetry_records=servo_telemetry_records,
        )),
        (combined_summary_path, lambda: _write_sync_combined_timing_summary(
            path=combined_summary_path,
            metrics=metrics,
            tracker_records=tracker_records,
            servo_command_records=servo_command_records,
            servo_telemetry_records=servo_telemetry_records,
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
        "command_events_on_tracker_timeline_path": events_on_timeline_path,
        "command_offset_histogram_path": command_offset_hist_path,
        "telemetry_offset_histogram_path": telemetry_offset_hist_path,
        "tracker_frame_age_at_servo_events_path": frame_age_over_time_path,
        "servo_telemetry_interval_histogram_path": telemetry_interval_path,
        "combined_timing_summary_path": combined_summary_path,
    }


SYNC_THRESHOLDS_MS = (5.0, 10.0, 25.0)


def _primary_offset_series(metrics: dict[str, Any]) -> tuple[list[float], str]:
    """Pick the most informative offset series — telemetry first, command fallback."""
    sync = dict(metrics.get("servo_tracker_sync", {}) or {})
    telemetry_offsets = list(sync.get("tracker_to_servo_telemetry_offsets_ms", []) or [])
    if telemetry_offsets:
        return [float(value) for value in telemetry_offsets], "tracker → servo telemetry"
    command_offsets = list(sync.get("tracker_to_servo_command_offsets_ms", []) or [])
    if command_offsets:
        return [float(value) for value in command_offsets], "tracker → servo command"
    return [], "tracker → servo telemetry"


def _write_servo_thesis_01_alignment(*, path: Path, metrics: dict[str, Any]) -> None:
    """Thesis figure 1: distribution + CDF of tracker → servo telemetry offsets.

    Histogram on left axis, cumulative fraction on right axis. Vertical
    reference lines at 5 / 10 / 25 ms thresholds with per-threshold cross
    rates annotated in the legend. Title explicitly references that the
    same pairing mechanism is used by collect_pose_command_dataset.
    """
    offsets, series_label = _primary_offset_series(metrics)
    fig, ax = create_figure(size="wide", constrained_layout=False)
    fig.subplots_adjust(left=0.085, right=0.92, top=0.88, bottom=0.22)

    if not offsets:
        ax.text(0.5, 0.5, "No tracker → servo offset series available",
                transform=ax.transAxes, ha="center", va="center")
        style_axes(ax, xlabel="Absolute host-time offset (ms)", ylabel="Sample count")
        fig.suptitle("Servo–Tracker Pair Time Alignment  (Same Mechanism as Collect Pose)",
                     fontsize=13, fontweight="bold", x=0.04, ha="left")
        save_figure(fig, path)
        return

    arr = np.asarray(offsets, dtype=float)
    mean_ms = float(arr.mean())
    p95_ms = float(np.percentile(arr, 95.0))
    p99_ms = float(np.percentile(arr, 99.0))

    x_max = max(float(arr.max()) * 1.05, SYNC_THRESHOLDS_MS[-1] * 1.2)
    bin_count = max(20, min(60, int(np.sqrt(len(arr)) * 2)))
    ax.hist(
        arr, bins=bin_count, range=(0.0, x_max),
        color=color("measured"), edgecolor="white", linewidth=0.6, alpha=0.85, zorder=2,
    )

    cdf_ax = ax.twinx()
    sorted_vals = np.sort(arr)
    cdf_y = np.arange(1, len(sorted_vals) + 1) / float(len(sorted_vals))
    cdf_ax.plot(sorted_vals, cdf_y, color=color("reference"), linewidth=1.6, alpha=0.85, zorder=3, label="CDF")
    cdf_ax.set_ylim(0.0, 1.05)
    cdf_ax.set_ylabel("Cumulative fraction", color=color("reference"))
    cdf_ax.tick_params(axis="y", colors=color("reference"))
    cdf_ax.spines["top"].set_visible(False)
    cdf_ax.spines["right"].set_color(color("reference"))

    threshold_styles = [("--", color("accepted")), ("-.", color("fit")), (":", color("rejected"))]
    threshold_rates: list[tuple[float, float]] = []
    for threshold, (style, threshold_color) in zip(SYNC_THRESHOLDS_MS, threshold_styles):
        rate = float((arr <= threshold).sum()) / float(len(arr))
        threshold_rates.append((threshold, rate))
        ax.axvline(
            threshold, color=threshold_color, linestyle=style, linewidth=1.4,
            label=f"≤ {threshold:.0f} ms  ({rate * 100:.1f}%)", zorder=4,
        )

    style_axes(ax, xlabel="Absolute host-time offset (ms)", ylabel="Sample count")
    ax.set_xlim(0.0, x_max)
    legend(ax, loc="upper right", ncol=1)

    fig.suptitle(
        "Servo–Tracker Pair Time Alignment  (Same Mechanism as Collect Pose Dataset)",
        fontsize=13, fontweight="bold", x=0.04, ha="left",
    )

    fig.text(
        0.015, 0.02,
        "  •  ".join([
            f"Pairs: {len(arr)}  ({series_label})",
            f"Mean: {mean_ms:.2f} ms  •  p95: {p95_ms:.2f} ms  •  p99: {p99_ms:.2f} ms",
            f"Within {SYNC_THRESHOLDS_MS[0]:.0f} ms: {threshold_rates[0][1] * 100:.1f}%",
        ]),
        fontsize=9, color=color("text"), ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": color("grid"), "alpha": 0.94},
    )
    save_figure(fig, path)


def _write_servo_thesis_02_motion_correspondence(
    *,
    path: Path,
    samples,
    metrics: dict[str, Any],
) -> None:
    """Thesis figure 2: stacked-panel time series demonstrating motion-stream alignment.

    Top: servo measured position (travel from start, in ticks).
    Bottom: tracker tip displacement (mm).
    Same X axis (monotonic time). The two curves should rise and fall together
    if the streams are truly co-temporal; visible lag would indicate either
    a sync problem or a kinematic delay between servo motion and tip motion.
    """
    requested_tool_ids = [str(value) for value in (metrics.get("requested_tool_ids") or ["0A"])]
    _command_points, measured_points = _servo_motion_series(samples)
    tracker_points, source_label = _tracker_motion_series(
        samples=samples,
        requested_tool_ids=requested_tool_ids,
        prefer_robot_frame_tip=bool(metrics.get("include_robot_frame_tip_pose", True)),
    )

    with report_style() as plt:
        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, figsize=(8.0, 5.4), sharex=True, constrained_layout=False,
        )
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.18, hspace=0.18)

    if not measured_points and not tracker_points:
        ax_top.text(0.5, 0.5, "No motion data available",
                    transform=ax_top.transAxes, ha="center", va="center")
        ax_bottom.set_visible(False)
        fig.suptitle("Servo–Tracker Motion Correspondence",
                     fontsize=13, fontweight="bold", x=0.04, ha="left")
        save_figure(fig, path)
        return

    if measured_points:
        servo_t = [point[0] for point in measured_points]
        servo_y = [point[1] for point in measured_points]
        ax_top.plot(servo_t, servo_y, color=color("measured"), linewidth=1.4, alpha=0.92, label="Servo telemetry travel")
        ax_top.fill_between(servo_t, 0.0, servo_y, color=color("measured"), alpha=0.10)
    else:
        ax_top.text(0.5, 0.5, "No servo telemetry samples", transform=ax_top.transAxes, ha="center", va="center")
    style_axes(ax_top, ylabel="Servo travel (ticks)")
    ax_top.set_title("Servo measured position (travel from start)", loc="left", pad=6, fontsize=11, fontweight="bold")
    legend(ax_top, loc="upper left")

    if tracker_points:
        tracker_t = [point[0] for point in tracker_points]
        tracker_y = [point[1] for point in tracker_points]
        ax_bottom.plot(tracker_t, tracker_y, color=color("fit"), linewidth=1.4, alpha=0.92,
                       label=f"Tracker displacement ({source_label})")
        ax_bottom.fill_between(tracker_t, 0.0, tracker_y, color=color("fit"), alpha=0.10)
    else:
        ax_bottom.text(0.5, 0.5, "No tracker motion data", transform=ax_bottom.transAxes, ha="center", va="center")
    style_axes(ax_bottom, xlabel="Monotonic time (s)", ylabel="Displacement (mm)")
    ax_bottom.set_title("Tracker tip displacement", loc="left", pad=6, fontsize=11, fontweight="bold")
    legend(ax_bottom, loc="upper left")

    fig.suptitle("Servo–Tracker Motion Correspondence",
                 fontsize=13, fontweight="bold", x=0.04, ha="left")

    n_servo = len(measured_points)
    n_tracker = len(tracker_points)
    max_servo_travel = max((point[1] for point in measured_points), default=0.0)
    max_tracker_disp = max((point[1] for point in tracker_points), default=0.0)
    fig.text(
        0.015, 0.02,
        "  •  ".join([
            f"Servo samples: {n_servo}  •  Tracker samples: {n_tracker}",
            f"Max servo travel: {max_servo_travel:.0f} ticks",
            f"Max tracker displacement: {max_tracker_disp:.2f} mm",
            f"Source: {source_label}",
        ]),
        fontsize=9, color=color("text"), ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": color("grid"), "alpha": 0.94},
    )
    save_figure(fig, path)


def _write_servo_debug_json(
    *,
    path: Path,
    output_dir: Path,
    metadata,
    summary,
    metrics: dict[str, Any],
    samples,
) -> None:
    """Consolidate everything that doesn't make it onto the thesis figures."""
    sync_block = dict(metrics.get("servo_tracker_sync", {}) or {})
    offsets, series_label = _primary_offset_series(metrics)
    threshold_rates = {}
    if offsets:
        arr = np.asarray(offsets, dtype=float)
        for threshold in SYNC_THRESHOLDS_MS + (20.0,):
            threshold_rates[f"within_{int(threshold)}ms_rate"] = float((arr <= threshold).sum()) / float(len(arr))

    per_tool_summary = dict(metrics.get("per_tool_summary", {}) or {})

    tracker_records = extract_tracker_timing_records(samples)
    servo_telemetry_records = extract_servo_timing_records(samples)
    servo_command_records = extract_servo_command_records(samples)

    review_payload: dict[str, Any] | None = None
    review_path = output_dir / "run_review.json"
    if review_path.exists():
        try:
            review_payload = json.loads(review_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            review_payload = {"parse_error": True}

    payload = {
        "schema_version": "1.0",
        "run_id": getattr(metadata, "run_id", None),
        "experiment_name": getattr(metadata, "experiment_name", "servo_tracker_sync_validation"),
        "status": getattr(summary, "status", None),
        "sample_counts": {
            "tracker_samples_analyzed": int(metrics.get("sample_count_analyzed", 0) or 0),
            "tracker_records_extracted": len(tracker_records),
            "servo_telemetry_samples": len(servo_telemetry_records),
            "servo_command_samples": len(servo_command_records),
        },
        "alignment": {
            "primary_series": series_label,
            "mean_offset_ms": sync_block.get("tracker_to_servo_telemetry_mean_offset_ms")
                or sync_block.get("tracker_to_servo_command_mean_offset_ms"),
            "p95_offset_ms": sync_block.get("tracker_to_servo_telemetry_p95_offset_ms")
                or sync_block.get("tracker_to_servo_command_p95_offset_ms"),
            "max_offset_ms": sync_block.get("tracker_to_servo_telemetry_max_offset_ms")
                or sync_block.get("tracker_to_servo_command_max_offset_ms"),
            "threshold_cross_rates": threshold_rates,
        },
        "per_tool_valid_rate": {
            tool_id: float(s.get("valid_transform_rate", 0.0) or 0.0)
            for tool_id, s in per_tool_summary.items()
        },
        "motion": {
            "max_displacement_mm": metrics.get("max_displacement_mm"),
            "requested_tool_ids": list(metrics.get("requested_tool_ids", []) or []),
            "include_robot_frame_tip_pose": metrics.get("include_robot_frame_tip_pose"),
        },
        "errors": {
            "error_sample_count": int(metrics.get("error_sample_count", 0) or 0),
        },
        "run_review": review_payload,
        "note": (
            "Pair alignment is computed by nearest-neighbor matching of tracker "
            "and servo-telemetry samples on host monotonic time. The same matching "
            "is performed during collect_pose_command_dataset; this validation "
            "characterizes its quality on a scripted motion."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_servo_debug_json_default), encoding="utf-8")


def _servo_debug_json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unserialisable type: {type(value).__name__}")


# =========================================================================== #
# Phase-4 supplementary figures (operator-spec request).                       #
# Each answers one question about how well servo events line up with the      #
# tracker stream. Sit alongside thesis_01/02 headlines.                       #
# =========================================================================== #


def _analyzed_tracker_commit_times_ns(tracker_records: list[dict[str, Any]]) -> list[int]:
    return sorted(
        int(record["sample_commit_monotonic_ns"])
        for record in tracker_records
        if not record.get("warmup_discarded")
        and record.get("sample_commit_monotonic_ns") is not None
    )


def _analyzed_servo_command_times_ns(records: list[dict[str, Any]]) -> list[int]:
    return sorted(
        int(record["command_monotonic_ns"])
        for record in records
        if not record.get("warmup_discarded")
        and record.get("command_monotonic_ns") is not None
    )


def _analyzed_servo_telemetry_times_ns(records: list[dict[str, Any]]) -> list[int]:
    return sorted(
        int(record["sample_monotonic_ns"])
        for record in records
        if not record.get("warmup_discarded")
        and record.get("sample_monotonic_ns") is not None
    )


def _nearest_preceding_offset_ms(event_ns: int, sorted_targets_ns: list[int]) -> float | None:
    """Time since the most recent preceding target. None if no preceding target."""
    from bisect import bisect_right
    index = bisect_right(sorted_targets_ns, int(event_ns))
    if index == 0:
        return None
    return float(int(event_ns) - int(sorted_targets_ns[index - 1])) / 1_000_000.0


def _write_sync_command_events_on_tracker_timeline(
    *,
    path: Path,
    tracker_records: list[dict[str, Any]],
    servo_command_records: list[dict[str, Any]],
) -> None:
    """Phase-4 figure 1: servo command events plotted on the tracker timeline.

    Tiny dots = tracker frame commits; vertical lines = servo command
    dispatches. Operator can eyeball at a glance whether commands tend to
    land near a fresh frame or in between.
    """
    tracker_ns = _analyzed_tracker_commit_times_ns(tracker_records)
    command_ns = _analyzed_servo_command_times_ns(servo_command_records)
    fig, ax = create_figure(size="wide", constrained_layout=False)
    fig.subplots_adjust(left=0.085, right=0.97, top=0.90, bottom=0.22)
    if not tracker_ns:
        ax.text(0.5, 0.5, "No analyzed tracker frames", transform=ax.transAxes,
                ha="center", va="center")
        style_axes(ax, xlabel="Time since first analyzed sample (s)", ylabel="")
        fig.suptitle("Servo Command Events on Tracker Frame Timeline",
                     fontsize=13, fontweight="bold", x=0.04, ha="left")
        save_figure(fig, path)
        return
    t0 = tracker_ns[0]
    tracker_seconds = [float(value - t0) / 1_000_000_000.0 for value in tracker_ns]
    command_seconds = [float(value - t0) / 1_000_000_000.0 for value in command_ns]
    ax.scatter(tracker_seconds, [0.5] * len(tracker_seconds), s=4,
               color=color("measured"), alpha=0.5, label="Tracker frame commit")
    for t_sec in command_seconds:
        ax.axvline(t_sec, color=color("rejected"), linewidth=1.0, alpha=0.6)
    ax.plot([], [], color=color("rejected"), linewidth=1.0,
            label=f"Servo command ({len(command_seconds)} events)")
    ax.set_yticks([])
    ax.set_ylim(0.0, 1.0)
    style_axes(ax, xlabel="Time since first analyzed sample (s)", ylabel="")
    fig.suptitle("Servo Command Events on Tracker Frame Timeline",
                 fontsize=13, fontweight="bold", x=0.04, ha="left")
    legend(ax, loc="upper right", ncol=1)
    fig.text(
        0.015, 0.02,
        f"Tracker frames: {len(tracker_seconds)}   •   "
        f"Servo commands: {len(command_seconds)}   •   "
        "Each red line is one command dispatch; the small blue dots are "
        "tracker frame commits. Use sync_command_offset_histogram.png for "
        "the per-event quantification.",
        fontsize=9, color=color("text"), ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
              "edgecolor": color("grid"), "alpha": 0.94},
    )
    save_figure(fig, path)


def _write_offset_histogram(
    *,
    path: Path,
    offsets_ms: list[float],
    title: str,
    xlabel: str,
    caption: str,
) -> None:
    """Shared template for the command/telemetry offset histograms."""
    fig, ax = create_figure(size="wide", constrained_layout=False)
    fig.subplots_adjust(left=0.085, right=0.97, top=0.90, bottom=0.22)
    if not offsets_ms:
        ax.text(0.5, 0.5, "No nearest-tracker offsets available",
                transform=ax.transAxes, ha="center", va="center")
        style_axes(ax, xlabel=xlabel, ylabel="Event count")
        fig.suptitle(title, fontsize=13, fontweight="bold", x=0.04, ha="left")
        save_figure(fig, path)
        return
    arr = np.asarray([float(value) for value in offsets_ms], dtype=float)
    bin_count = max(20, min(60, int(np.sqrt(len(arr)) * 2)))
    ax.hist(arr, bins=bin_count, color=color("measured"), edgecolor="white",
            linewidth=0.6, alpha=0.85, zorder=2)
    median_ms = float(np.median(arr))
    mean_ms = float(arr.mean())
    p95_ms = float(np.percentile(arr, 95.0))
    max_ms = float(arr.max())
    ax.axvline(median_ms, color=color("fit"), linestyle="--", linewidth=1.4,
               label=f"Median = {median_ms:.2f} ms", zorder=3)
    ax.axvline(mean_ms, color=color("threshold"), linestyle=":", linewidth=1.4,
               label=f"Mean = {mean_ms:.2f} ms", zorder=3)
    ax.axvline(p95_ms, color=color("rejected"), linestyle="-.", linewidth=1.2,
               label=f"p95 = {p95_ms:.2f} ms", zorder=3)
    style_axes(ax, xlabel=xlabel, ylabel="Event count")
    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.04, ha="left")
    legend(ax, loc="upper right", ncol=1)
    fig.text(
        0.015, 0.02,
        f"N = {len(arr)} events   •   max = {max_ms:.2f} ms   •   {caption}",
        fontsize=9, color=color("text"), ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
              "edgecolor": color("grid"), "alpha": 0.94},
    )
    save_figure(fig, path)


def _write_sync_command_offset_histogram(
    *,
    path: Path,
    metrics: dict[str, Any],
) -> None:
    """Phase-4 figure 2: histogram of command-to-nearest-tracker-frame offsets."""
    sync = dict(metrics.get("servo_tracker_sync", {}) or {})
    offsets = list(sync.get("tracker_to_servo_command_offsets_ms") or [])
    _write_offset_histogram(
        path=path,
        offsets_ms=offsets,
        title="Servo Command → Nearest Tracker Frame Offset",
        xlabel="|command time − nearest tracker frame| (ms)",
        caption=(
            "Distance in host time between each servo command dispatch and "
            "the closest tracker frame commit. Lower = better sync."
        ),
    )


def _write_sync_telemetry_offset_histogram(
    *,
    path: Path,
    metrics: dict[str, Any],
) -> None:
    """Phase-4 figure 3: histogram of telemetry-to-nearest-tracker-frame offsets."""
    sync = dict(metrics.get("servo_tracker_sync", {}) or {})
    offsets = list(sync.get("tracker_to_servo_telemetry_offsets_ms") or [])
    _write_offset_histogram(
        path=path,
        offsets_ms=offsets,
        title="Servo Telemetry → Nearest Tracker Frame Offset",
        xlabel="|telemetry time − nearest tracker frame| (ms)",
        caption=(
            "Distance in host time between each servo telemetry read and the "
            "closest tracker frame commit. Lower = better sync."
        ),
    )


def _write_sync_tracker_frame_age_at_servo_events(
    *,
    path: Path,
    tracker_records: list[dict[str, Any]],
    servo_command_records: list[dict[str, Any]],
    servo_telemetry_records: list[dict[str, Any]],
) -> None:
    """Phase-4 figure 4: tracker frame age at servo events vs experiment time.

    For each servo event, plots the elapsed time since the most recent
    preceding tracker frame (so positive = stale tracker at the moment of
    the servo event). Reveals whether sync degrades during the run.
    """
    tracker_ns = _analyzed_tracker_commit_times_ns(tracker_records)
    command_ns = _analyzed_servo_command_times_ns(servo_command_records)
    telemetry_ns = _analyzed_servo_telemetry_times_ns(servo_telemetry_records)
    fig, ax = create_figure(size="wide", constrained_layout=False)
    fig.subplots_adjust(left=0.085, right=0.97, top=0.90, bottom=0.22)
    if not tracker_ns or not (command_ns or telemetry_ns):
        ax.text(0.5, 0.5, "Need both tracker frames and servo events",
                transform=ax.transAxes, ha="center", va="center")
        style_axes(ax, xlabel="Time since first analyzed sample (s)",
                   ylabel="Tracker frame age at event (ms)")
        fig.suptitle("Tracker Frame Age at Servo Events",
                     fontsize=13, fontweight="bold", x=0.04, ha="left")
        save_figure(fig, path)
        return
    t0 = min([tracker_ns[0]] + ([command_ns[0]] if command_ns else []) + ([telemetry_ns[0]] if telemetry_ns else []))
    command_ages = [
        (
            float(event_ns - t0) / 1_000_000_000.0,
            _nearest_preceding_offset_ms(event_ns, tracker_ns),
        )
        for event_ns in command_ns
    ]
    telemetry_ages = [
        (
            float(event_ns - t0) / 1_000_000_000.0,
            _nearest_preceding_offset_ms(event_ns, tracker_ns),
        )
        for event_ns in telemetry_ns
    ]
    cmd_x = [t for t, age in command_ages if age is not None]
    cmd_y = [age for t, age in command_ages if age is not None]
    tel_x = [t for t, age in telemetry_ages if age is not None]
    tel_y = [age for t, age in telemetry_ages if age is not None]
    if cmd_x:
        ax.scatter(cmd_x, cmd_y, s=18, color=color("rejected"), alpha=0.7,
                   label=f"Command events ({len(cmd_x)})")
    if tel_x:
        ax.scatter(tel_x, tel_y, s=8, color=color("measured"), alpha=0.5,
                   label=f"Telemetry events ({len(tel_x)})")
    ax.axhline(25.0, color=color("threshold"), linestyle="--", linewidth=1.2,
               label="Aurora frame interval (25 ms)", zorder=2)
    ax.set_ylim(bottom=0.0)
    style_axes(ax, xlabel="Time since first analyzed sample (s)",
               ylabel="Tracker frame age at event (ms)")
    fig.suptitle("Tracker Frame Age at Servo Events",
                 fontsize=13, fontweight="bold", x=0.04, ha="left")
    legend(ax, loc="upper right", ncol=1)
    fig.text(
        0.015, 0.02,
        "For each servo event, elapsed time since the most recent preceding "
        "tracker frame. Sustained values above the 25 ms Aurora interval "
        "indicate the tracker is falling behind and command/telemetry events "
        "are landing on stale frames.",
        fontsize=9, color=color("text"), ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
              "edgecolor": color("grid"), "alpha": 0.94},
    )
    save_figure(fig, path)


def _write_sync_servo_telemetry_interval_histogram(
    *,
    path: Path,
    servo_telemetry_records: list[dict[str, Any]],
) -> None:
    """Phase-4 figure 5: gap distribution between consecutive servo telemetry samples."""
    telemetry_ns = _analyzed_servo_telemetry_times_ns(servo_telemetry_records)
    intervals_ms = [
        float(later - earlier) / 1_000_000.0
        for earlier, later in zip(telemetry_ns[:-1], telemetry_ns[1:])
    ]
    _write_offset_histogram(
        path=path,
        offsets_ms=intervals_ms,
        title="Servo Telemetry Inter-Sample Interval",
        xlabel="Gap between consecutive servo telemetry samples (ms)",
        caption=(
            "If the servo telemetry poll loop falls behind its configured "
            "interval, this distribution shifts right."
        ),
    )


def _write_sync_combined_timing_summary(
    *,
    path: Path,
    metrics: dict[str, Any],
    tracker_records: list[dict[str, Any]],
    servo_command_records: list[dict[str, Any]],
    servo_telemetry_records: list[dict[str, Any]],
) -> None:
    """Phase-4 figure 6: 2x2 combined summary for at-a-glance reporting.

      Top-left:    command-to-tracker offset histogram (compact)
      Top-right:   telemetry-to-tracker offset histogram (compact)
      Bottom-left: rate-comparison bars (tracker frames vs servo telemetry vs servo commands)
      Bottom-right: percentile table rendered as text annotations
    """
    with report_style() as plt:
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.10,
                         hspace=0.40, wspace=0.28)
    sync = dict(metrics.get("servo_tracker_sync", {}) or {})
    cmd_offsets = [float(v) for v in (sync.get("tracker_to_servo_command_offsets_ms") or [])]
    tel_offsets = [float(v) for v in (sync.get("tracker_to_servo_telemetry_offsets_ms") or [])]

    # Top-left: command offset
    ax = axes[0, 0]
    if cmd_offsets:
        ax.hist(cmd_offsets, bins=20, color=color("measured"), edgecolor="white", linewidth=0.5)
        ax.axvline(float(np.median(cmd_offsets)), color=color("fit"), linestyle="--",
                   linewidth=1.2, label=f"Median {np.median(cmd_offsets):.2f} ms")
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No command offsets", transform=ax.transAxes, ha="center", va="center")
    style_axes(ax, title="Command → Tracker offset", xlabel="ms", ylabel="count")

    # Top-right: telemetry offset
    ax = axes[0, 1]
    if tel_offsets:
        ax.hist(tel_offsets, bins=20, color=color("measured"), edgecolor="white", linewidth=0.5)
        ax.axvline(float(np.median(tel_offsets)), color=color("fit"), linestyle="--",
                   linewidth=1.2, label=f"Median {np.median(tel_offsets):.2f} ms")
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No telemetry offsets", transform=ax.transAxes, ha="center", va="center")
    style_axes(ax, title="Telemetry → Tracker offset", xlabel="ms", ylabel="count")

    # Bottom-left: rate comparison
    ax = axes[1, 0]
    tracker_ns = _analyzed_tracker_commit_times_ns(tracker_records)
    telemetry_ns = _analyzed_servo_telemetry_times_ns(servo_telemetry_records)
    command_ns = _analyzed_servo_command_times_ns(servo_command_records)
    def _rate_hz(times_ns: list[int]) -> float:
        if len(times_ns) < 2:
            return 0.0
        duration = (times_ns[-1] - times_ns[0]) / 1_000_000_000.0
        return float(len(times_ns) - 1) / duration if duration > 0 else 0.0
    labels = ["Tracker\n(unique)", "Servo\ntelemetry", "Servo\ncommands"]
    values = [_rate_hz(tracker_ns), _rate_hz(telemetry_ns), _rate_hz(command_ns)]
    bars = ax.bar(labels, values, color=[color("measured"), color("fit"), color("accepted")],
                  edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, labels=[f"{value:.1f} Hz" for value in values], padding=3, fontsize=9)
    ax.axhline(40.0, color=color("threshold"), linestyle="--", linewidth=1.2,
               label="Aurora ceiling (40 Hz)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0.0, max(max(values) * 1.25, 45.0))
    style_axes(ax, title="Stream rates", xlabel="", ylabel="Rate (Hz)")

    # Bottom-right: percentile/stats text
    ax = axes[1, 1]
    ax.axis("off")
    def _fmt(value: Any) -> str:
        try:
            return f"{float(value):.2f} ms"
        except (TypeError, ValueError):
            return "n/a"
    lines = [
        "Per-event nearest-tracker offsets",
        "",
        f"Command   median = {_fmt(sync.get('tracker_to_servo_command_median_offset_ms'))}",
        f"Command   p95    = {_fmt(sync.get('tracker_to_servo_command_p95_offset_ms'))}",
        f"Command   max    = {_fmt(sync.get('tracker_to_servo_command_max_offset_ms'))}",
        "",
        f"Telemetry median = {_fmt(sync.get('tracker_to_servo_telemetry_median_offset_ms'))}",
        f"Telemetry p95    = {_fmt(sync.get('tracker_to_servo_telemetry_p95_offset_ms'))}",
        f"Telemetry max    = {_fmt(sync.get('tracker_to_servo_telemetry_max_offset_ms'))}",
        "",
        f"Command events:   {int(sync.get('tracker_to_servo_command_sample_count', 0) or 0)}",
        f"Telemetry events: {int(sync.get('tracker_to_servo_telemetry_sample_count', 0) or 0)}",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), transform=ax.transAxes,
            ha="left", va="top", family="monospace", fontsize=10, color=color("text"))

    fig.suptitle("Servo-Tracker Sync — Combined Timing Summary",
                 fontsize=13, fontweight="bold", x=0.04, ha="left")
    save_figure(fig, path)
