"""Saved artifacts for servo-tracker synchronization validation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from continuum_robot.experiments.dataset_tools import extract_tip_or_tool_position_mm
from continuum_robot.experiments.plotting import (
    add_metric_box,
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


def _servo_signed_travel_series(samples) -> list[tuple[float, float]]:
    """Signed servo telemetry travel from each servo's first observed position.

    For each servo, the first non-warmup ``present_position_ticks`` is the
    zero; subsequent telemetry samples report ``present - zero`` (signed,
    ticks). When several servos are commanded at the same monotonic time the
    mean of their signed travels is taken so the resulting trace remains
    one-dimensional and oscillates symmetrically around zero.
    """
    measured_zero_by_servo: dict[int, float] = {}
    rows_by_time: dict[float, list[float]] = {}
    for sample in samples:
        extra = dict(getattr(sample, "extra", {}) or {})
        if str(extra.get("record_kind", "")).strip().lower() != "servo_timing":
            continue
        if bool(extra.get("warmup_discarded", False)):
            continue
        servo_id = extra.get("servo_id")
        present = extra.get("present_position_ticks")
        if servo_id is None or present is None:
            continue
        servo_id = int(servo_id)
        present_value = float(present)
        measured_zero_by_servo.setdefault(servo_id, present_value)
        signed = present_value - measured_zero_by_servo[servo_id]
        rows_by_time.setdefault(float(sample.monotonic_time_s), []).append(signed)
    return [
        (time_s, float(np.mean(values)))
        for time_s, values in sorted(rows_by_time.items())
        if values
    ]


def _tracker_signed_projection_series(
    *,
    samples,
    requested_tool_ids: list[str],
    prefer_robot_frame_tip: bool,
    sign_reference: list[tuple[float, float]] | None,
) -> tuple[list[tuple[float, float]], str]:
    """Signed tracker displacement projected onto its principal motion axis.

    Replaces the previous ``norm(translation - first)`` metric, which
    produced a rectified absolute-value fence whenever the robot moved
    symmetrically around a centre. The principal axis is the direction of
    maximum variance of the tracker translation across non-warmup samples;
    the projection's sign is aligned with the supplied servo reference (if
    provided) so positive servo travel maps to positive tracker excursion.

    Important: this samples a SINGLE coordinate frame for the whole series.
    Earlier revisions silently fell back from robot-frame tip to
    tracker-frame tool position when the robot-frame tip was missing on a
    given sample; the resulting frame jumps were ~50 mm coordinate
    offsets, not real motion, and the PCA principal axis would lock onto
    them. If the preferred frame is missing for >50% of analyzed samples
    we fall back to the alternative frame for the whole run, never within
    it.
    """
    primary_tool = str(requested_tool_ids[0]) if requested_tool_ids else "0A"

    def _collect(prefer_robot: bool) -> tuple[list[np.ndarray], list[float], str]:
        kept_positions: list[np.ndarray] = []
        kept_times: list[float] = []
        label = "unavailable"
        for sample in samples:
            extra = dict(getattr(sample, "extra", {}) or {})
            if str(extra.get("record_kind", "")).strip().lower() != "tracker_timing":
                continue
            if bool(extra.get("warmup_discarded", False)):
                continue
            if prefer_robot:
                position_mm, frame_name = extract_tip_or_tool_position_mm(
                    sample, tool_id=primary_tool, prefer_robot_frame=True,
                )
                if position_mm is None or frame_name != "robot":
                    continue  # do not mix frames; skip if robot-frame tip absent
                label = "robot tip"
            else:
                position_mm, _ = extract_tip_or_tool_position_mm(
                    sample, tool_id=primary_tool, prefer_robot_frame=False,
                )
                if position_mm is None:
                    continue
                label = f"tool {primary_tool} (tracker frame)"
            kept_positions.append(np.asarray(position_mm, dtype=float))
            kept_times.append(float(sample.monotonic_time_s))
        return kept_positions, kept_times, label

    positions, times, source_label = ([], [], "unavailable")
    if prefer_robot_frame_tip:
        positions, times, source_label = _collect(prefer_robot=True)
        # Also count total candidates so we can decide whether to fall back
        # to the tracker-frame source. We use 50% coverage as the cutoff: if
        # robot-frame tip is present on most samples, stay with it; if not,
        # the tracker-frame tool gives a denser and more representative
        # series.
        tracker_positions, tracker_times, tracker_label = _collect(prefer_robot=False)
        if len(positions) < 0.5 * len(tracker_positions):
            positions, times, source_label = tracker_positions, tracker_times, tracker_label
    else:
        positions, times, source_label = _collect(prefer_robot=False)

    if not positions:
        return [], source_label

    positions_arr = np.stack(positions)
    centered = positions_arr - positions_arr.mean(axis=0)
    if len(centered) < 2:
        return [(t, 0.0) for t in times], source_label

    cov = centered.T @ centered
    # eigh returns eigenvalues ascending, so the last column is the
    # principal axis (direction of maximum variance).
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_axis = eigvecs[:, -1]
    projected = centered @ principal_axis

    # Sign-align with servo reference so positive servo travel is positive
    # tracker excursion. Without this, the principal axis sign is arbitrary
    # (PCA up to a global flip).
    if sign_reference:
        servo_t = np.array([t for t, _ in sign_reference])
        servo_v = np.array([v for _, v in sign_reference])
        if servo_t.size and servo_v.size and np.std(servo_v) > 0:
            tracker_t = np.array(times)
            nearest = np.searchsorted(servo_t, tracker_t)
            nearest = np.clip(nearest, 0, len(servo_t) - 1)
            servo_at_tracker = servo_v[nearest]
            if np.std(servo_at_tracker) > 0 and np.std(projected) > 0:
                corr = np.corrcoef(servo_at_tracker, projected)[0, 1]
                if not np.isnan(corr) and corr < 0:
                    projected = -projected

    label_suffix = " (principal axis)"
    return list(zip(times, projected.tolist())), source_label + label_suffix


def _pick_motion_window(
    servo_signal: list[tuple[float, float]],
    tracker_signal: list[tuple[float, float]],
    *,
    target_duration_s: float = 2.5,
) -> tuple[float, float]:
    """Pick the window with the most servo direction changes.

    Slides a ``target_duration_s`` window across the servo signal and
    returns the start/end with the maximum number of direction reversals.
    Falls back to the run start when motion never alternates.
    """
    if not servo_signal:
        if tracker_signal:
            t0 = tracker_signal[0][0]
            return t0, min(tracker_signal[-1][0], t0 + target_duration_s)
        return 0.0, target_duration_s

    times = np.array([t for t, _ in servo_signal])
    values = np.array([v for _, v in servo_signal])
    if len(times) < 4:
        return float(times[0]), float(times[0]) + target_duration_s

    diffs = np.diff(values)
    signs = np.sign(diffs)
    direction_changes = np.where(
        (signs[:-1] * signs[1:]) < 0
    )[0]  # indices where the sign flips
    if len(direction_changes) == 0:
        return float(times[0]), float(min(times[-1], times[0] + target_duration_s))

    # times of each direction change (the later of the two diff endpoints)
    change_times = times[direction_changes + 2]

    span_start = float(times[0])
    span_end = float(times[-1])
    if span_end - span_start <= target_duration_s:
        return span_start, span_end

    best_count = -1
    best_start = span_start
    step = max(0.1, target_duration_s / 8.0)
    candidate = span_start
    while candidate + target_duration_s <= span_end:
        in_window = (change_times >= candidate) & (change_times < candidate + target_duration_s)
        count = int(in_window.sum())
        if count > best_count:
            best_count = count
            best_start = float(candidate)
        candidate += step
    return best_start, best_start + target_duration_s


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
    metrics = summary.experiment_metrics if isinstance(summary.experiment_metrics, dict) else {}

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


PAIRING_THRESHOLD_MS = 25.0  # threshold used by collect_pose_command_dataset


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
    """Thesis figure 1: distribution of tracker→servo pair-time offsets.

    Single histogram, no CDF twin axis. Bars beyond the 25 ms pairing
    threshold are coloured rejected-red so the kept/discarded split reads
    at a glance. An on-figure stat box reports N, median, p95, and the
    fraction within the threshold.
    """
    offsets, series_label = _primary_offset_series(metrics)
    fig, ax = create_figure(size="wide", constrained_layout=False)
    fig.subplots_adjust(left=0.085, right=0.97, top=0.86, bottom=0.12)

    if not offsets:
        ax.text(0.5, 0.5, "No tracker → servo offset series available",
                transform=ax.transAxes, ha="center", va="center")
        style_axes(ax, xlabel="Absolute host-time offset (ms)", ylabel="Pair count")
        fig.suptitle("Servo–Tracker Pair Time Alignment",
                     fontsize=13, fontweight="bold", x=0.04, ha="left")
        save_figure(fig, path)
        return

    arr = np.asarray(offsets, dtype=float)
    median_ms = float(np.median(arr))
    p95_ms = float(np.percentile(arr, 95.0))
    within_rate = float((arr <= PAIRING_THRESHOLD_MS).sum()) / float(len(arr))

    x_max = max(float(arr.max()) * 1.05, PAIRING_THRESHOLD_MS * 1.6)
    bin_count = max(20, min(45, int(np.sqrt(len(arr)) * 2)))

    counts, edges = np.histogram(arr, bins=bin_count, range=(0.0, x_max))
    widths = np.diff(edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    bar_colors = [
        color("measured") if c <= PAIRING_THRESHOLD_MS else color("rejected")
        for c in centers
    ]
    ax.bar(
        edges[:-1], counts, width=widths, align="edge",
        color=bar_colors, edgecolor="white", linewidth=0.6, alpha=0.9, zorder=2,
    )

    ax.axvline(PAIRING_THRESHOLD_MS, color=color("threshold"),
               linestyle="-", linewidth=1.6, zorder=3)
    ymax = ax.get_ylim()[1]
    ax.text(PAIRING_THRESHOLD_MS - x_max * 0.005,
            ymax * 0.95,
            f"{PAIRING_THRESHOLD_MS:.0f} ms pairing threshold",
            ha="right", va="top", fontsize=9, color=color("axis"))

    style_axes(ax, xlabel="Absolute host-time offset (ms)", ylabel="Pair count")
    ax.set_xlim(0.0, x_max)

    add_metric_box(ax, [
        f"N = {len(arr)} pairs",
        f"median = {median_ms:.1f} ms",
        f"p95 = {p95_ms:.1f} ms",
        f"within {PAIRING_THRESHOLD_MS:.0f} ms: {within_rate * 100:.1f}%",
        f"({series_label})",
    ], loc="upper right")

    fig.suptitle("Servo–Tracker Pair Time Alignment",
                 fontsize=13, fontweight="bold", x=0.04, ha="left")
    save_figure(fig, path)


def _write_servo_thesis_02_motion_correspondence(
    *,
    path: Path,
    samples,
    metrics: dict[str, Any],
) -> None:
    """Thesis figure 2: zoomed two-panel correspondence between servo and tracker.

    Top panel: signed servo telemetry travel (ticks). Bottom panel: signed
    tracker displacement projected on the run's PCA principal motion axis
    (mm). Both share the X axis and are zoomed to the 2.5 s window of the
    run with the most servo direction reversals. Light vertical guides at
    each servo command transition anchor the eye so co-temporal motion
    reads at a glance even when the dynamics differ between the streams.
    """
    requested_tool_ids = [str(value) for value in (metrics.get("requested_tool_ids") or ["0A"])]
    servo_signal = _servo_signed_travel_series(samples)
    command_transition_times = _servo_command_transition_times(samples)
    tracker_signal, source_label = _tracker_signed_projection_series(
        samples=samples,
        requested_tool_ids=requested_tool_ids,
        prefer_robot_frame_tip=bool(metrics.get("include_robot_frame_tip_pose", True)),
        sign_reference=servo_signal,
    )

    with report_style() as plt:
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(8.0, 5.6), sharex=True, constrained_layout=False,
        )
    fig.subplots_adjust(left=0.14, right=0.97, top=0.87, bottom=0.14, hspace=0.30)

    if not servo_signal and not tracker_signal:
        ax_top.text(0.5, 0.5, "No motion data available",
                    transform=ax_top.transAxes, ha="center", va="center")
        ax_bot.set_visible(False)
        fig.suptitle("Servo–Tracker Motion Correspondence",
                     fontsize=13, fontweight="bold", x=0.04, ha="left")
        save_figure(fig, path)
        return

    window_start, window_end = _pick_motion_window(servo_signal, tracker_signal, target_duration_s=2.5)

    def _in_window(points: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
        if not points:
            return np.array([]), np.array([])
        filtered = [(t, v) for t, v in points if window_start <= t <= window_end]
        if not filtered:
            return np.array([]), np.array([])
        return (
            np.array([t - window_start for t, _ in filtered]),
            np.array([v for _, v in filtered], dtype=float),
        )

    servo_t, servo_v = _in_window(servo_signal)
    tracker_t, tracker_v = _in_window(tracker_signal)
    command_times_rel = np.array(
        [t - window_start for t in command_transition_times
         if window_start <= t <= window_end]
    )

    # Light vertical guides at each servo command transition, on both axes,
    # so the eye can read "the tip moved when the command changed."
    # Slightly darker than the grid colour so they are visible against the
    # white panel background but still stay in the visual background.
    transition_guide_color = "#94a3b8"
    for ax in (ax_top, ax_bot):
        for transition_t in command_times_rel:
            ax.axvline(transition_t, color=transition_guide_color,
                       linewidth=0.9, alpha=0.55, zorder=1)
        ax.axhline(0.0, color=color("axis"), linewidth=0.6, alpha=0.45, zorder=1)

    if servo_t.size:
        ax_top.plot(servo_t, servo_v, color=color("measured"),
                    linewidth=2.0, alpha=0.95)
    else:
        ax_top.text(0.5, 0.5, "No servo telemetry in window",
                    transform=ax_top.transAxes, ha="center", va="center")
    style_axes(ax_top, ylabel="Servo travel (ticks, signed)")
    ax_top.set_title("Servo telemetry  (signed travel from first sample)",
                      loc="left", pad=4, fontsize=11, fontweight="bold")

    if tracker_t.size:
        ax_bot.plot(tracker_t, tracker_v, color=color("fit"),
                    linewidth=2.0, alpha=0.95)
    else:
        ax_bot.text(0.5, 0.5, "No tracker motion in window",
                    transform=ax_bot.transAxes, ha="center", va="center")
    style_axes(
        ax_bot,
        xlabel=f"Time in {window_end - window_start:.1f} s window  (s)",
        ylabel="Tracker displacement (mm, signed)",
    )
    # source_label already ends with " (principal axis)" — strip that for the
    # heading so the panel title isn't "(principal axis) ... (principal axis)".
    source_for_title = source_label.replace(" (principal axis)", "").strip()
    ax_bot.set_title(
        f"Tracker tip  (signed, projected on principal motion axis — source: {source_for_title})",
        loc="left", pad=4, fontsize=11, fontweight="bold",
    )

    ax_bot.set_xlim(0.0, window_end - window_start)

    fig.suptitle("Servo–Tracker Motion Correspondence",
                 fontsize=13, fontweight="bold", x=0.04, ha="left")

    n_commands = int(len(command_times_rel))
    fig.text(
        0.10, 0.02,
        "  •  ".join([
            f"Window: {window_start:.1f}–{window_end:.1f} s of analyzed run",
            f"Servo telemetry samples: {servo_t.size}",
            f"Tracker frames: {tracker_t.size}",
            f"Servo command transitions in window: {n_commands}",
        ]),
        fontsize=9, color=color("text"), ha="left", va="bottom",
    )
    save_figure(fig, path)


def _servo_command_transition_times(samples) -> list[float]:
    """Return monotonic times at which a servo_command's commanded_position_ticks
    changed from the previous command for the same servo.

    Used to draw vertical guides on the motion-correspondence figure so the
    reader can anchor servo→tracker timing without needing to read the
    surrounding text.
    """
    last_by_servo: dict[int, float] = {}
    times: list[float] = []
    for sample in samples:
        extra = dict(getattr(sample, "extra", {}) or {})
        if str(extra.get("record_kind", "")).strip().lower() != "servo_command":
            continue
        servo_id = extra.get("servo_id")
        commanded = extra.get("commanded_position_ticks")
        if servo_id is None or commanded is None:
            continue
        servo_id = int(servo_id)
        commanded_value = float(commanded)
        previous = last_by_servo.get(servo_id)
        if previous is None or abs(previous - commanded_value) > 1e-9:
            times.append(float(sample.monotonic_time_s))
        last_by_servo[servo_id] = commanded_value
    times.sort()
    # Deduplicate near-simultaneous transitions across multiple servos by
    # collapsing entries within 5 ms of each other.
    unique: list[float] = []
    for t in times:
        if not unique or (t - unique[-1]) > 0.005:
            unique.append(t)
    return unique


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
    threshold_rates: dict[str, float] = {}
    if offsets:
        arr = np.asarray(offsets, dtype=float)
        # Diagnostic cross-rates at multiple thresholds for debug.json. The
        # pairing-threshold rate at 25 ms is the figure-of-merit (matches
        # collect_pose_command_dataset); the others give a sense of how the
        # offset distribution falls off.
        for threshold in (5.0, 10.0, 20.0, PAIRING_THRESHOLD_MS):
            threshold_rates[f"within_{int(threshold)}ms_rate"] = (
                float((arr <= threshold).sum()) / float(len(arr))
            )

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
