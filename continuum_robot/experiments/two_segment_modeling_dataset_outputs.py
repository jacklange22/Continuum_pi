"""Thesis-quality figures for the two-segment collect-pose dataset.

The ``thesis_01`` and ``thesis_02`` writers are direct ports of the
single-segment ``modeling_dataset_outputs._write_collect_pose_thesis_*``
functions. The styling (axes, fonts, colorbar layout, title placement,
metric footer) matches the single-segment figures so that side-by-side
comparison in the thesis is visually clean.

Two-segment-specific data shape:

- Distal tip position comes from ``two_segment_pose.distal_tip_pose.T_robot_tip``.
- Per-segment pair command is derived from ``two_segment_command.segments``
  using the canonical tip-target convention (``cable_deltas = [-px, -py, +px, +py]``).
- Two-segment commands carry one pair per segment, so ``thesis_02`` stacks
  three panels (Segment A pair, Segment B pair, Tip XY) instead of two.

Tracker-variability figures mirror the single-segment averaging plots when
the experiment captured multiple samples per command (``samples_per_pattern > 1``).
Per-command spread is computed by grouping the raw post-settle samples by
``(cycle, step_index)`` rather than from an averaged-sample list, because
the two-segment experiment does not maintain a separate averaged-sample
stream — its repeated captures are emitted as independent samples.

The four legacy ``two_segment_*_report.png`` files written by
:func:`write_two_segment_dataset_outputs` are unaffected.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from continuum_robot.experiments.plotting import (
    color,
    create_3d_figure,
    report_style,
    save_figure,
    set_equal_xyz,
    style_3d_axes,
    style_axes,
)
from continuum_robot.experiments.schemas import ExperimentTimeseriesSample


LOG = logging.getLogger(__name__)


# Canonical thesis figure filenames. ``thesis_01`` and ``thesis_02`` mirror
# the single-segment names exactly; ``thesis_03..06`` add two-segment-specific
# diagnostics that have no single-segment equivalent.
THESIS_FIGURE_NAMES: tuple[str, ...] = (
    "thesis_01_workspace_coverage_3d.png",
    "thesis_02_command_and_workspace_2d.png",
    "thesis_03_per_segment_command_pose.png",
    "thesis_04_servo_position_coverage.png",
    "thesis_05_dataset_quality.png",
    "thesis_06_current_load_timeline.png",
)


# Tracker-variability figure names mirror the single-segment names. They are
# only emitted when the run captured more than one sample per command.
VARIABILITY_FIGURE_NAMES: tuple[str, ...] = (
    "tracker_variability_workspace_xy.png",
    "tracker_variability_std_histogram.png",
    "tracker_variability_std_vs_command_index.png",
)


# Mirror the typography constants in the single-segment variability writers so
# the two-segment figures match exactly.
_THESIS_FIG_TITLE_SIZE = 12.5
_THESIS_FIG_AXES_LABEL_SIZE = 10.5
_THESIS_FIG_TICK_SIZE = 9.5
_THESIS_FIG_LEGEND_SIZE = 9.0
_METRIC_LABEL = "3D RMS spread (mm)"


# ----------------------------------------------------------------------------
# Public entrypoint
# ----------------------------------------------------------------------------


def write_two_segment_thesis_figures(
    *,
    output_dir: Path,
    metrics: dict[str, Any],
    samples: list[ExperimentTimeseriesSample],
    sample_failure_events: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    """Write the thesis-quality figure set for a two-segment collect-pose run.

    Returns a dict mapping a short key (e.g. ``thesis_01``) to the written
    path. Each figure is wrapped in a try/except so a single render failure
    does not block the others — the caller already preserves the legacy
    report set.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _build_thesis_records(samples)
    export_rows = _build_export_rows_from_records(records)
    variability_records = _compute_variability_records(records)
    paths: dict[str, Path] = {}

    plot_jobs: list[tuple[str, Path, Callable[[], None]]] = [
        (
            "thesis_01",
            output_dir / "thesis_01_workspace_coverage_3d.png",
            lambda: _write_two_segment_thesis_01_workspace_coverage_3d(
                path=output_dir / "thesis_01_workspace_coverage_3d.png",
                export_rows=export_rows,
                metrics=metrics,
            ),
        ),
        (
            "thesis_02",
            output_dir / "thesis_02_command_and_workspace_2d.png",
            lambda: _write_two_segment_thesis_02_command_and_workspace_2d(
                path=output_dir / "thesis_02_command_and_workspace_2d.png",
                export_rows=export_rows,
                metrics=metrics,
            ),
        ),
        (
            "thesis_03",
            output_dir / "thesis_03_per_segment_command_pose.png",
            lambda: _write_per_segment_command_pose(
                path=output_dir / "thesis_03_per_segment_command_pose.png",
                records=records,
                metrics=metrics,
            ),
        ),
        (
            "thesis_04",
            output_dir / "thesis_04_servo_position_coverage.png",
            lambda: _write_servo_position_coverage(
                path=output_dir / "thesis_04_servo_position_coverage.png",
                records=records,
                metrics=metrics,
            ),
        ),
        (
            "thesis_05",
            output_dir / "thesis_05_dataset_quality.png",
            lambda: _write_dataset_quality(
                path=output_dir / "thesis_05_dataset_quality.png",
                records=records,
                metrics=metrics,
                sample_failure_events=list(sample_failure_events or []),
            ),
        ),
        (
            "thesis_06",
            output_dir / "thesis_06_current_load_timeline.png",
            lambda: _write_current_load_timeline(
                path=output_dir / "thesis_06_current_load_timeline.png",
                records=records,
                metrics=metrics,
            ),
        ),
    ]
    if variability_records:
        plot_jobs.extend(
            [
                (
                    "tracker_variability_workspace_xy",
                    output_dir / "tracker_variability_workspace_xy.png",
                    lambda: _write_tracker_variability_workspace_xy(
                        path=output_dir / "tracker_variability_workspace_xy.png",
                        records=variability_records,
                    ),
                ),
                (
                    "tracker_variability_std_histogram",
                    output_dir / "tracker_variability_std_histogram.png",
                    lambda: _write_tracker_variability_std_histogram(
                        path=output_dir / "tracker_variability_std_histogram.png",
                        records=variability_records,
                    ),
                ),
                (
                    "tracker_variability_std_vs_command_index",
                    output_dir / "tracker_variability_std_vs_command_index.png",
                    lambda: _write_tracker_variability_std_vs_command_index(
                        path=output_dir / "tracker_variability_std_vs_command_index.png",
                        records=variability_records,
                    ),
                ),
            ]
        )
    for key, path, writer in plot_jobs:
        try:
            writer()
        except Exception:
            LOG.exception("Failed to render two-segment thesis figure %s", path)
            _write_plot_placeholder(path)
        paths[key] = path
    return paths


# ----------------------------------------------------------------------------
# Record extraction
# ----------------------------------------------------------------------------


def _build_thesis_records(samples: Iterable[ExperimentTimeseriesSample]) -> list[dict[str, Any]]:
    """Flatten the heavy nested sample structure into plot-friendly dicts.

    One dict per sample, with the keys every figure below needs. Records
    that lack the data a figure requires are simply skipped at plot time.

    ``row_index`` is the monotonic 0-based position of the record in the
    samples sequence — use this for timeline plots. ``sample_index`` and
    ``step_index`` are preserved verbatim from the source sample but are
    not monotonic across the run (they reset per command step).
    """
    records: list[dict[str, Any]] = []
    for row_index, sample in enumerate(samples):
        extra = dict(sample.extra or {})
        command = dict(sample.two_segment_command or {})
        segments = dict(command.get("segments") or {})
        segment_a = [float(value) for value in list(segments.get("segment_a") or [])]
        segment_b = [float(value) for value in list(segments.get("segment_b") or [])]
        pose_payload = dict(sample.two_segment_pose or {})
        distal_pose = dict(pose_payload.get("distal_tip_pose") or {})
        T_robot_tip = distal_pose.get("T_robot_tip")
        tip_xyz: tuple[float, float, float] | None = None
        if isinstance(T_robot_tip, list) and len(T_robot_tip) >= 3:
            try:
                tip_xyz = (
                    float(T_robot_tip[0][3]),
                    float(T_robot_tip[1][3]),
                    float(T_robot_tip[2][3]),
                )
            except (TypeError, IndexError, ValueError):
                tip_xyz = None
        intermediate_payload = dict(pose_payload.get("intermediate_pose") or {})
        intermediate_T = intermediate_payload.get("T_robot_intermediate") or intermediate_payload.get("T_robot_tip")
        intermediate_xyz: tuple[float, float, float] | None = None
        if isinstance(intermediate_T, list) and len(intermediate_T) >= 3:
            try:
                intermediate_xyz = (
                    float(intermediate_T[0][3]),
                    float(intermediate_T[1][3]),
                    float(intermediate_T[2][3]),
                )
            except (TypeError, IndexError, ValueError):
                intermediate_xyz = None
        feedback_raw = dict(extra.get("measured_servo_feedback") or {})
        feedback: dict[int, dict[str, Any]] = {}
        for sid_key, payload in feedback_raw.items():
            try:
                sid = int(sid_key)
            except (TypeError, ValueError):
                continue
            feedback[int(sid)] = dict(payload or {})
        goal_ticks_raw = dict(extra.get("goal_ticks_by_servo") or {})
        goal_ticks: dict[int, int] = {}
        for sid_key, value in goal_ticks_raw.items():
            try:
                sid = int(sid_key)
                goal_ticks[sid] = int(value)
            except (TypeError, ValueError):
                continue
        # Pair-axis convention mirrors the single-segment helper exactly:
        # cable_deltas = [-px, -py, +px, +py] ⇒ pair = (-c0, -c1).
        segment_a_pair = _pair_from_cable_deltas(segment_a)
        segment_b_pair = _pair_from_cable_deltas(segment_b)
        # ``cycle_index`` lets the variability path group repeated samples
        # for the same command across multiple passes through the schedule.
        # ``capture_repeat`` is the same field exposed by single-segment
        # samples; two-segment names it explicitly in ``extra``.
        cycle_index_raw = extra.get("cycle_index", getattr(sample, "cycle_index", None))
        try:
            cycle_index_value = int(cycle_index_raw) if cycle_index_raw is not None else None
        except (TypeError, ValueError):
            cycle_index_value = None
        records.append(
            {
                "row_index": int(row_index),
                "sample_index": int(getattr(sample, "sample_index", 0) or 0),
                "step_index": int(getattr(sample, "step_index", 0) or 0),
                "cycle_index": cycle_index_value,
                "monotonic_time_s": float(getattr(sample, "monotonic_time_s", 0.0) or 0.0),
                "phase": str(getattr(sample, "phase", "") or ""),
                "accepted": bool(extra.get("capture_accepted")),
                "rejection_reason": str(extra.get("capture_rejection_reason") or ""),
                "segment_a_cable_cm": segment_a,
                "segment_b_cable_cm": segment_b,
                "segment_a_pair_cm": segment_a_pair,
                "segment_b_pair_cm": segment_b_pair,
                "segment_a_magnitude_cm": _vector_magnitude(segment_a),
                "segment_b_magnitude_cm": _vector_magnitude(segment_b),
                "tip_xyz_mm": tip_xyz,
                "intermediate_xyz_mm": intermediate_xyz,
                "feedback_by_servo": feedback,
                "goal_ticks_by_servo": goal_ticks,
            }
        )
    return records


def _build_export_rows_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reshape accepted+poseful records into the same shape ``_build_export_rows`` produces in single-segment.

    The single-segment ``thesis_01`` / ``thesis_02`` writers consume an
    ``export_rows`` list of dicts with three keys: ``tip_position_xyz_mm``,
    ``requested_pair_command_cm`` (one pair for single segment), and
    ``sequence_index``. For two-segment we adopt the same shape but
    duplicate the pair into per-segment keys so ``thesis_02`` can show
    both segments without diverging from the single-segment styling.
    """
    out: list[dict[str, Any]] = []
    for record in records:
        if not bool(record.get("accepted")):
            continue
        tip = record.get("tip_xyz_mm")
        if tip is None:
            continue
        seg_a_pair = record.get("segment_a_pair_cm")
        seg_b_pair = record.get("segment_b_pair_cm")
        # ``requested_pair_command_cm`` is filled with segment-B pair (the
        # "second segment that distinguishes this dataset" per the operator
        # request). This keeps the single-segment writer signature satisfied
        # when called as-is. Per-segment keys carry the explicit pairs.
        primary_pair = seg_b_pair if seg_b_pair is not None else seg_a_pair
        out.append(
            {
                "sequence_index": len(out),
                "tip_position_xyz_mm": [float(tip[0]), float(tip[1]), float(tip[2])],
                "requested_pair_command_cm": (
                    [float(primary_pair[0]), float(primary_pair[1])]
                    if primary_pair is not None
                    else []
                ),
                "segment_a_pair_cm": (
                    [float(seg_a_pair[0]), float(seg_a_pair[1])]
                    if seg_a_pair is not None
                    else []
                ),
                "segment_b_pair_cm": (
                    [float(seg_b_pair[0]), float(seg_b_pair[1])]
                    if seg_b_pair is not None
                    else []
                ),
            }
        )
    return out


def _compute_variability_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group raw samples by command and compute per-command position spread.

    The two-segment experiment does not maintain a separate averaged-sample
    stream the way single-segment does; ``samples_per_pattern`` >1 just
    emits N independent samples per command. We group by
    ``(cycle_index, step_index)`` and reduce each group into the same dict
    shape consumed by the single-segment variability writers.

    Returns an empty list when no command has more than one accepted sample
    (i.e. ``samples_per_pattern == 1``), so the variability figures are
    skipped automatically on non-averaged runs.
    """
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in records:
        if not bool(record.get("accepted")):
            continue
        if record.get("tip_xyz_mm") is None:
            continue
        cycle_idx = int(record.get("cycle_index") or 0)
        step_idx = int(record.get("step_index") or 0)
        groups.setdefault((cycle_idx, step_idx), []).append(record)
    out: list[dict[str, Any]] = []
    multi_sample_seen = False
    for (cycle_idx, step_idx), bucket in sorted(groups.items()):
        if len(bucket) < 2:
            # Single-sample commands have no within-command spread to plot.
            # Keep them so a mixed run (some commands repeated, some not)
            # still surfaces every accepted pose if every command happens to
            # repeat — but skip the spread metric. The std fields stay None.
            tip = bucket[0]["tip_xyz_mm"]
            out.append(
                {
                    "command_index": int(step_idx) + int(cycle_idx) * 1_000_000,
                    "averaged_x_mm": float(tip[0]),
                    "averaged_y_mm": float(tip[1]),
                    "averaged_z_mm": float(tip[2]),
                    "position_std_rms_mm": None,
                    "position_max_deviation_mm": None,
                    "first_vs_mean_position_diff_mm": None,
                    "valid_sample_count": 1,
                }
            )
            continue
        multi_sample_seen = True
        positions = np.asarray(
            [list(r["tip_xyz_mm"]) for r in bucket], dtype=float
        )
        mean_xyz = np.mean(positions, axis=0)
        # Sample stdev (ddof=1) per axis, RMS-combined into a 3D scalar to
        # match the single-segment ``position_std_rms_mm`` definition.
        per_axis_std = np.std(positions, axis=0, ddof=1) if positions.shape[0] >= 2 else np.zeros(3)
        std_rms = float(np.sqrt(float(np.mean(per_axis_std ** 2))))
        deviations = np.linalg.norm(positions - mean_xyz, axis=1)
        first_vs_mean = float(np.linalg.norm(positions[0] - mean_xyz))
        out.append(
            {
                "command_index": int(step_idx) + int(cycle_idx) * 1_000_000,
                "averaged_x_mm": float(mean_xyz[0]),
                "averaged_y_mm": float(mean_xyz[1]),
                "averaged_z_mm": float(mean_xyz[2]),
                "position_std_rms_mm": std_rms,
                "position_max_deviation_mm": float(np.max(deviations)),
                "first_vs_mean_position_diff_mm": first_vs_mean,
                "valid_sample_count": int(positions.shape[0]),
            }
        )
    if not multi_sample_seen:
        return []
    return out


def _pair_from_cable_deltas(cable_deltas: list[float]) -> tuple[float, float] | None:
    if not cable_deltas:
        return None
    if len(cable_deltas) < 2:
        return None
    px = -float(cable_deltas[0])
    py = -float(cable_deltas[1])
    return (px, py)


def _vector_magnitude(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(math.sqrt(sum(float(v) * float(v) for v in values)))


def _truncate_reason_label(name: str, *, max_chars: int = 64) -> str:
    text = str(name)
    if len(text) <= max_chars:
        return text
    return text[: max(1, max_chars - 1)] + "…"


def _downsample_indices(point_count: int, *, max_points: int) -> np.ndarray | None:
    if point_count <= max_points or max_points <= 0:
        return None
    return np.linspace(0, point_count - 1, num=max_points, dtype=int)


def _accepted_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if bool(r.get("accepted")) and r.get("tip_xyz_mm") is not None]


def _compact_strip(items: list[str | None]) -> list[str]:
    return [str(item) for item in items if item]


# ----------------------------------------------------------------------------
# Thesis figures 01 + 02 — direct ports of single-segment
# ----------------------------------------------------------------------------


def _write_two_segment_thesis_01_workspace_coverage_3d(
    *,
    path: Path,
    export_rows: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    """Direct port of single-segment ``_write_collect_pose_thesis_01_workspace_coverage_3d``.

    Reads only ``tip_position_xyz_mm`` from the export rows so the source
    behaviour matches what the single-segment writer does for the
    ``collect_pose_command_dataset`` figure of the same name. The figure
    title and footer use the two-segment ``schedule_type`` so the figure
    is unambiguous when filed alongside a single-segment figure.
    """
    points: list[tuple[float, float, float]] = []
    for row in export_rows:
        tip = row.get("tip_position_xyz_mm")
        if not (isinstance(tip, list) and len(tip) >= 3):
            continue
        try:
            points.append((float(tip[0]), float(tip[1]), float(tip[2])))
        except (TypeError, ValueError):
            continue

    fig, ax = create_3d_figure(size="thesis_3d")
    if not points:
        ax.text2D(0.5, 0.5, "No accepted tip positions available",
                  transform=ax.transAxes, ha="center", va="center")
        style_3d_axes(ax, title="")
        fig.suptitle("Two-Segment Workspace Coverage During Babble Collection",
                     fontsize=13, fontweight="bold", x=0.04, ha="left")
        save_figure(fig, path)
        return

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]
    indices = np.arange(len(points), dtype=float)

    scatter = ax.scatter(
        xs, ys, zs,
        c=indices, cmap="viridis",
        s=14, depthshade=True, alpha=0.85, linewidths=0,
    )

    set_equal_xyz(ax, x_values=xs, y_values=ys, z_values=zs, minimum_span=5.0, pad_fraction=0.08)
    style_3d_axes(ax, xlabel="Robot X (mm)", ylabel="Robot Y (mm)", zlabel="Robot Z (mm)")

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.55, pad=0.12)
    cbar.set_label("Sample index (time order)")
    cbar.outline.set_edgecolor(color("grid"))

    mode = str(metrics.get("schedule_type", "unknown") or "unknown").replace("_", " ")
    fig.suptitle(f"Two-Segment Workspace Coverage During Babble Collection  ({mode})",
                 fontsize=13, fontweight="bold", x=0.04, ha="left")

    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    z_span = max(zs) - min(zs)
    accepted = int(metrics.get("accepted_sample_count", len(points)) or len(points))
    rejected = int(metrics.get("rejected_sample_count", 0) or 0)
    total = accepted + rejected
    accept_rate = (accepted / total * 100.0) if total > 0 else 100.0
    fig.text(
        0.015, 0.02,
        "  •  ".join(_compact_strip([
            f"Mode: {mode}",
            f"Accepted: {accepted}/{total}  ({accept_rate:.1f}%)",
            f"Workspace span: X={x_span:.1f} mm  Y={y_span:.1f} mm  Z={z_span:.1f} mm",
        ])),
        fontsize=9, color=color("text"), ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": color("grid"), "alpha": 0.94},
    )
    save_figure(fig, path)


def _write_two_segment_thesis_02_command_and_workspace_2d(
    *,
    path: Path,
    export_rows: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    """Direct port of single-segment ``_write_collect_pose_thesis_02_command_and_workspace_2d``.

    Two-segment commands carry one pair per segment, so this stacks three
    panels instead of two: Segment A pair on top, Segment B pair in the
    middle, distal tip XY on the bottom. All panels share the time-order
    colour scale so a single dot colour traces one accepted sample across
    the three panels. Layout, styling, footer and colorbar placement
    follow the single-segment figure exactly.
    """
    seg_a_cmds: list[tuple[float, float]] = []
    seg_b_cmds: list[tuple[float, float]] = []
    tip_xy: list[tuple[float, float]] = []
    indices: list[int] = []
    for index, row in enumerate(export_rows):
        tip = row.get("tip_position_xyz_mm")
        if not (isinstance(tip, list) and len(tip) >= 2):
            continue
        seg_a = row.get("segment_a_pair_cm")
        seg_b = row.get("segment_b_pair_cm")
        if not (isinstance(seg_a, list) and len(seg_a) >= 2):
            continue
        if not (isinstance(seg_b, list) and len(seg_b) >= 2):
            continue
        try:
            seg_a_cmds.append((float(seg_a[0]), float(seg_a[1])))
            seg_b_cmds.append((float(seg_b[0]), float(seg_b[1])))
            tip_xy.append((float(tip[0]), float(tip[1])))
        except (TypeError, ValueError):
            continue
        indices.append(index)

    with report_style() as plt:
        fig, (ax_a, ax_b, ax_tip) = plt.subplots(
            3, 1, figsize=(7.6, 10.4), constrained_layout=False,
        )
    fig.subplots_adjust(left=0.10, right=0.86, top=0.93, bottom=0.10, hspace=0.32)

    if not seg_a_cmds or not tip_xy:
        for ax in (ax_a, ax_b, ax_tip):
            ax.text(0.5, 0.5, "No accepted samples available",
                    transform=ax.transAxes, ha="center", va="center")
        fig.suptitle("Two-Segment Command Space (top/middle) and Tip Workspace XY (bottom)",
                     fontsize=13, fontweight="bold", x=0.04, ha="left")
        save_figure(fig, path)
        return

    color_values = np.asarray(indices, dtype=float)

    ax_x = [point[0] for point in seg_a_cmds]
    ax_y = [point[1] for point in seg_a_cmds]
    sc = ax_a.scatter(ax_x, ax_y, c=color_values, cmap="viridis",
                      s=14, alpha=0.85, linewidths=0)
    style_axes(ax_a, xlabel="Pair 1/3 command (cm)", ylabel="Pair 2/4 command (cm)")
    ax_a.set_title("Segment A command space", loc="left", pad=8, fontsize=11, fontweight="bold")
    ax_a.set_aspect("equal", adjustable="datalim")

    bx_x = [point[0] for point in seg_b_cmds]
    bx_y = [point[1] for point in seg_b_cmds]
    ax_b.scatter(bx_x, bx_y, c=color_values, cmap="viridis",
                 s=14, alpha=0.85, linewidths=0)
    style_axes(ax_b, xlabel="Pair 1/3 command (cm)", ylabel="Pair 2/4 command (cm)")
    ax_b.set_title("Segment B command space", loc="left", pad=8, fontsize=11, fontweight="bold")
    ax_b.set_aspect("equal", adjustable="datalim")

    tip_x = [point[0] for point in tip_xy]
    tip_y = [point[1] for point in tip_xy]
    ax_tip.scatter(tip_x, tip_y, c=color_values, cmap="viridis",
                   s=14, alpha=0.85, linewidths=0)
    style_axes(ax_tip, xlabel="Robot X (mm)", ylabel="Robot Y (mm)")
    ax_tip.set_title("Distal tip workspace (top-down XY)", loc="left", pad=8, fontsize=11, fontweight="bold")
    ax_tip.set_aspect("equal", adjustable="datalim")

    cbar = fig.colorbar(sc, ax=(ax_a, ax_b, ax_tip), shrink=0.85, pad=0.025)
    cbar.set_label("Sample index (time order)")
    cbar.outline.set_edgecolor(color("grid"))

    mode = str(metrics.get("schedule_type", "unknown") or "unknown").replace("_", " ")
    fig.suptitle(f"Two-Segment Command Space (top/middle) and Tip Workspace XY (bottom)  —  {mode}",
                 fontsize=13, fontweight="bold", x=0.04, ha="left")

    a_x_span = max(ax_x) - min(ax_x)
    a_y_span = max(ax_y) - min(ax_y)
    b_x_span = max(bx_x) - min(bx_x)
    b_y_span = max(bx_y) - min(bx_y)
    tip_x_span = max(tip_x) - min(tip_x)
    tip_y_span = max(tip_y) - min(tip_y)
    fig.text(
        0.015, 0.015,
        "  •  ".join(_compact_strip([
            f"Samples: {len(tip_xy)}",
            f"Seg-A extent: pair_13 {a_x_span:.2f} cm, pair_24 {a_y_span:.2f} cm",
            f"Seg-B extent: pair_13 {b_x_span:.2f} cm, pair_24 {b_y_span:.2f} cm",
            f"Tip XY extent: X {tip_x_span:.1f} mm, Y {tip_y_span:.1f} mm",
        ])),
        fontsize=9, color=color("text"), ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": color("grid"), "alpha": 0.94},
    )
    save_figure(fig, path)


# ----------------------------------------------------------------------------
# Tracker variability figures — direct ports of single-segment
# ----------------------------------------------------------------------------


def _set_variability_title(fig, title: str) -> None:
    fig.suptitle(
        title,
        fontsize=_THESIS_FIG_TITLE_SIZE,
        fontweight="bold",
        x=0.04,
        ha="left",
        y=0.975,
    )


def _set_variability_tick_sizes(ax) -> None:
    ax.tick_params(axis="both", labelsize=_THESIS_FIG_TICK_SIZE)
    ax.xaxis.label.set_size(_THESIS_FIG_AXES_LABEL_SIZE)
    ax.yaxis.label.set_size(_THESIS_FIG_AXES_LABEL_SIZE)


def _write_tracker_variability_workspace_xy(
    *,
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Direct port of single-segment ``_write_tracker_variability_workspace_xy``.

    Each marker is one commanded pose plotted at its averaged tip XY in the
    robot frame; colour encodes the per-command 3D RMS spread across the
    repeated samples at that command. For two-segment, the "repeated
    samples per command" are the ``samples_per_pattern`` post-settle
    captures.
    """
    with report_style() as plt:
        fig, ax = plt.subplots(figsize=(7.2, 6.5), constrained_layout=False)
    fig.subplots_adjust(left=0.105, right=0.88, top=0.90, bottom=0.12)

    valid = [
        (r["averaged_x_mm"], r["averaged_y_mm"], r["position_std_rms_mm"])
        for r in records
        if r["position_std_rms_mm"] is not None
    ]
    if not valid:
        ax.text(0.5, 0.5, "No averaged samples with position spread data",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=_THESIS_FIG_AXES_LABEL_SIZE)
        style_axes(ax, xlabel="Robot X (mm)", ylabel="Robot Y (mm)")
        _set_variability_title(fig, "Workspace Variability")
        save_figure(fig, path)
        return

    xs = [v[0] for v in valid]
    ys = [v[1] for v in valid]
    stds = [v[2] for v in valid]
    scatter = ax.scatter(
        xs, ys,
        c=stds,
        cmap="plasma",
        s=44,
        edgecolors="white",
        linewidths=0.45,
        alpha=0.95,
    )
    ax.set_aspect("equal", adjustable="datalim")
    style_axes(ax, xlabel="Robot X (mm)", ylabel="Robot Y (mm)")
    _set_variability_tick_sizes(ax)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.048, pad=0.025, shrink=0.88)
    cbar.set_label(_METRIC_LABEL, fontsize=_THESIS_FIG_AXES_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=_THESIS_FIG_TICK_SIZE)
    cbar.outline.set_edgecolor(color("grid"))

    _set_variability_title(fig, "Workspace Variability")
    save_figure(fig, path)


def _write_tracker_variability_std_histogram(
    *,
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Direct port of single-segment ``_write_tracker_variability_std_histogram``."""
    with report_style() as plt:
        fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=False)
    fig.subplots_adjust(left=0.105, right=0.98, top=0.86, bottom=0.16)

    stds = [r["position_std_rms_mm"] for r in records if r["position_std_rms_mm"] is not None]
    if not stds:
        ax.text(0.5, 0.5, "No averaged samples with position spread data",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=_THESIS_FIG_AXES_LABEL_SIZE)
        style_axes(ax, xlabel=_METRIC_LABEL, ylabel="Command count")
        _set_variability_title(fig, "Spread Distribution")
        save_figure(fig, path)
        return
    arr = np.asarray(stds, dtype=float)
    bins = min(30, max(10, len(arr) // 4))
    ax.hist(arr, bins=bins, color=color("measured"), edgecolor="white", alpha=0.92)
    median = float(np.median(arr))
    p95 = float(np.percentile(arr, 95))
    ax.axvline(median, color=color("reference"), linestyle="--", linewidth=1.4,
               label=f"median {median:.3f}")
    ax.axvline(p95, color=color("rejected"), linestyle="-.", linewidth=1.3,
               label=f"p95 {p95:.3f}")
    style_axes(ax, xlabel=_METRIC_LABEL, ylabel="Command count")
    _set_variability_tick_sizes(ax)
    legend_obj = ax.legend(loc="upper right", frameon=True, facecolor="white",
                           edgecolor=color("grid"), framealpha=0.95,
                           fontsize=_THESIS_FIG_LEGEND_SIZE)
    if legend_obj is not None:
        legend_obj.get_frame().set_linewidth(0.6)
    _set_variability_title(fig, "Spread Distribution")
    save_figure(fig, path)


def _write_tracker_variability_std_vs_command_index(
    *,
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Direct port of single-segment ``_write_tracker_variability_std_vs_command_index``."""
    with report_style() as plt:
        fig, ax = plt.subplots(figsize=(7.8, 4.4), constrained_layout=False)
    fig.subplots_adjust(left=0.105, right=0.98, top=0.86, bottom=0.16)

    rows_sorted = sorted(
        (r for r in records if r["position_std_rms_mm"] is not None),
        key=lambda r: int(r["command_index"]),
    )
    if not rows_sorted:
        ax.text(0.5, 0.5, "No averaged samples with position spread data",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=_THESIS_FIG_AXES_LABEL_SIZE)
        style_axes(ax, xlabel="Command index (collection order)",
                   ylabel=_METRIC_LABEL)
        _set_variability_title(fig, "Variability Over Collection")
        save_figure(fig, path)
        return
    xs = list(range(len(rows_sorted)))
    ys = [float(r["position_std_rms_mm"]) for r in rows_sorted]
    ax.plot(xs, ys, color=color("measured"), linewidth=0.8,
            marker="o", markersize=3.0, markerfacecolor=color("measured"),
            markeredgecolor="white", markeredgewidth=0.25, alpha=0.80,
            label="command")
    window = max(5, len(ys) // 20)
    if len(ys) >= window:
        kernel = np.ones(window) / float(window)
        smoothed = np.convolve(ys, kernel, mode="valid")
        smooth_x = xs[(window - 1) // 2 : (window - 1) // 2 + len(smoothed)]
        if len(smooth_x) == len(smoothed):
            ax.plot(smooth_x, smoothed, color=color("rejected"),
                    linewidth=1.6, label=f"rolling mean ({window})")
    style_axes(ax, xlabel="Command index (collection order)",
               ylabel=_METRIC_LABEL)
    _set_variability_tick_sizes(ax)
    legend_obj = ax.legend(loc="upper right", frameon=True, facecolor="white",
                           edgecolor=color("grid"), framealpha=0.95,
                           fontsize=_THESIS_FIG_LEGEND_SIZE)
    if legend_obj is not None:
        legend_obj.get_frame().set_linewidth(0.6)
    _set_variability_title(fig, "Variability Over Collection")
    save_figure(fig, path)


# ----------------------------------------------------------------------------
# Two-segment specific diagnostics (thesis_03..06)
# ----------------------------------------------------------------------------


def _write_per_segment_command_pose(
    *,
    path: Path,
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    """Per-segment command magnitude vs distal tip excursion."""
    accepted = [r for r in records if bool(r.get("accepted")) and r.get("tip_xyz_mm") is not None]
    schedule = str(metrics.get("schedule_type") or "unknown").replace("_", " ")
    with report_style() as plt:
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.16, wspace=0.24)

    if not accepted:
        for ax in (ax_a, ax_b):
            ax.text(
                0.5, 0.5, "No accepted samples with distal tip pose",
                transform=ax.transAxes, ha="center", va="center",
            )
        fig.suptitle(
            f"Per-Segment Command Magnitude vs Distal Tip Excursion  ({schedule})",
            fontsize=13, fontweight="bold", x=0.04, ha="left",
        )
        save_figure(fig, path)
        return

    tip_xyz = np.asarray([r["tip_xyz_mm"] for r in accepted], dtype=float)
    origin = tip_xyz[0]
    excursions = np.linalg.norm(tip_xyz - origin, axis=1)
    seg_a_mag = np.asarray([float(r.get("segment_a_magnitude_cm") or 0.0) for r in accepted], dtype=float)
    seg_b_mag = np.asarray([float(r.get("segment_b_magnitude_cm") or 0.0) for r in accepted], dtype=float)
    indices = np.arange(len(accepted), dtype=float)

    sc_a = ax_a.scatter(seg_a_mag, excursions, c=indices, cmap="viridis", s=14, alpha=0.85, linewidths=0)
    style_axes(ax_a, xlabel="Segment A command magnitude (cm)", ylabel="Distal tip excursion from first sample (mm)")
    ax_a.set_title("Segment A — bottom command coupling", loc="left", pad=8, fontsize=11, fontweight="bold")

    ax_b.scatter(seg_b_mag, excursions, c=indices, cmap="viridis", s=14, alpha=0.85, linewidths=0)
    style_axes(ax_b, xlabel="Segment B command magnitude (cm)", ylabel="Distal tip excursion from first sample (mm)")
    ax_b.set_title("Segment B — top command coupling", loc="left", pad=8, fontsize=11, fontweight="bold")

    cbar = fig.colorbar(sc_a, ax=[ax_a, ax_b], shrink=0.85, pad=0.02)
    cbar.set_label("Sample index (time order)")
    cbar.outline.set_edgecolor(color("grid"))

    fig.suptitle(
        f"Per-Segment Command Magnitude vs Distal Tip Excursion  —  {schedule}",
        fontsize=13, fontweight="bold", x=0.04, ha="left",
    )

    def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
        if x.size < 2 or y.size < 2:
            return None
        x_std = float(np.std(x))
        y_std = float(np.std(y))
        if x_std <= 1e-9 or y_std <= 1e-9:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    pearson_a = _pearson(seg_a_mag, excursions)
    pearson_b = _pearson(seg_b_mag, excursions)
    summary = [
        f"Samples: {len(accepted)}",
        f"Pearson r (A) = {('%.3f' % pearson_a) if pearson_a is not None else 'n/a'}",
        f"Pearson r (B) = {('%.3f' % pearson_b) if pearson_b is not None else 'n/a'}",
        f"Excursion p50/p95 = {float(np.percentile(excursions, 50)):.1f}/{float(np.percentile(excursions, 95)):.1f} mm",
    ]
    fig.text(
        0.015, 0.02,
        "  •  ".join(summary),
        fontsize=9, color=color("text"), ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": color("grid"), "alpha": 0.94},
    )
    save_figure(fig, path)


def _write_servo_position_coverage(
    *,
    path: Path,
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    """8-servo position coverage grouped by segment."""
    servo_positions: dict[int, list[int]] = {sid: [] for sid in range(1, 9)}
    servo_deltas: dict[int, list[int]] = {sid: [] for sid in range(1, 9)}
    for record in records:
        for sid, payload in (record.get("feedback_by_servo") or {}).items():
            position = payload.get("position_tick")
            delta = payload.get("delta_from_startup_tick")
            if position is not None:
                try:
                    servo_positions.setdefault(int(sid), []).append(int(position))
                except (TypeError, ValueError):
                    pass
            if delta is not None:
                try:
                    servo_deltas.setdefault(int(sid), []).append(int(delta))
                except (TypeError, ValueError):
                    pass
    servo_ids = sorted(servo_positions)
    ranges = [
        (max(servo_positions[sid]) - min(servo_positions[sid])) if servo_positions[sid] else 0
        for sid in servo_ids
    ]
    deltas_lists = [servo_deltas.get(sid, []) for sid in servo_ids]

    schedule = str(metrics.get("schedule_type") or "unknown").replace("_", " ")
    config_used = dict(metrics.get("config_used") or {})
    max_tick_delta = config_used.get("max_tick_delta_from_startup")
    try:
        max_tick_delta = int(max_tick_delta) if max_tick_delta is not None else None
    except (TypeError, ValueError):
        max_tick_delta = None

    with report_style() as plt:
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9.0, 7.2), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.10, hspace=0.36)

    bar_colors = [color("segment_a") if sid <= 4 else color("segment_b") for sid in servo_ids]
    ax_top.bar(servo_ids, ranges, color=bar_colors, edgecolor="white", linewidth=0.6)
    ax_top.axvline(4.5, color=color("threshold"), linestyle="--", linewidth=1.0)
    if max_tick_delta is not None and max_tick_delta > 0:
        twice_budget = 2 * int(max_tick_delta)
        ax_top.axhline(
            twice_budget,
            color=color("threshold"),
            linestyle=":",
            linewidth=1.2,
            label=f"2 × safety budget = {twice_budget} ticks",
        )
        ax_top.legend(loc="upper right", fontsize=9, framealpha=0.92)
    style_axes(ax_top, xlabel="Servo ID", ylabel="Observed position range (ticks)")
    ax_top.set_title("Per-servo position coverage", loc="left", pad=8, fontsize=11, fontweight="bold")
    ax_top.set_xticks(servo_ids)
    ax_top.text(
        2.5, -max(1.0, 0.04 * max(ranges + [1.0])), "Segment A (1–4)",
        ha="center", va="top", color=color("segment_a"),
        fontsize=9, fontweight="bold",
    )
    ax_top.text(
        6.5, -max(1.0, 0.04 * max(ranges + [1.0])), "Segment B (5–8)",
        ha="center", va="top", color=color("segment_b"),
        fontsize=9, fontweight="bold",
    )

    bp = ax_bot.boxplot(
        deltas_lists, positions=servo_ids, widths=0.6,
        patch_artist=True, manage_ticks=False, showfliers=True,
    )
    for patch, sid in zip(bp["boxes"], servo_ids):
        patch.set_facecolor(color("segment_a") if sid <= 4 else color("segment_b"))
        patch.set_alpha(0.55)
        patch.set_edgecolor(color("axis"))
    for whisker in bp["whiskers"]:
        whisker.set_color(color("axis"))
    for cap in bp["caps"]:
        cap.set_color(color("axis"))
    for median in bp["medians"]:
        median.set_color(color("text"))
        median.set_linewidth(1.4)
    for flier in bp["fliers"]:
        flier.set_markerfacecolor(color("rejected"))
        flier.set_markeredgecolor(color("rejected"))
        flier.set_markersize(3)
    if max_tick_delta is not None and max_tick_delta > 0:
        ax_bot.axhline(int(max_tick_delta), color=color("threshold"), linestyle=":", linewidth=1.0)
        ax_bot.axhline(-int(max_tick_delta), color=color("threshold"), linestyle=":", linewidth=1.0,
                       label=f"±{int(max_tick_delta)} tick safety budget")
        ax_bot.legend(loc="upper right", fontsize=9, framealpha=0.92)
    ax_bot.axhline(0.0, color=color("grid"), linewidth=0.6)
    ax_bot.axvline(4.5, color=color("threshold"), linestyle="--", linewidth=1.0)
    ax_bot.set_xticks(servo_ids)
    style_axes(ax_bot, xlabel="Servo ID", ylabel="Position delta from startup (ticks)")
    ax_bot.set_title("Per-servo delta-from-startup distribution", loc="left", pad=8, fontsize=11, fontweight="bold")

    fig.suptitle(
        f"Two-Segment Servo Position Coverage  —  {schedule}",
        fontsize=13, fontweight="bold", x=0.04, ha="left",
    )
    save_figure(fig, path)


def _write_dataset_quality(
    *,
    path: Path,
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
    sample_failure_events: list[dict[str, Any]],
) -> None:
    """Accepted vs rejected breakdown + rejection-reason bar."""
    accepted = int(metrics.get("accepted_sample_count") or 0)
    rejected = int(metrics.get("rejected_sample_count") or 0)
    failures = int(metrics.get("command_failure_count") or 0)
    total = accepted + rejected
    reason_counts: dict[str, int] = {}
    for record in records:
        if bool(record.get("accepted")):
            continue
        reason = (record.get("rejection_reason") or "").strip() or "unspecified"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    for event in sample_failure_events:
        reason = str(event.get("reason") or event.get("error") or "transport_event").strip()
        if not reason:
            continue
        key = f"event:{reason}"
        reason_counts[key] = reason_counts.get(key, 0) + 1

    schedule = str(metrics.get("schedule_type") or "unknown").replace("_", " ")
    trust_mode = str(metrics.get("run_trust_mode") or "unknown")
    valid_two_segment = bool(metrics.get("valid_for_two_segment_model_training"))
    valid_ann = bool(metrics.get("valid_for_two_segment_ann_training"))

    with report_style() as plt:
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.0, 5.0), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.18, wspace=0.30)

    labels_left = ["Accepted", "Rejected", "Command failures"]
    values_left = [accepted, rejected, failures]
    colors_left = [color("accepted"), color("rejected"), color("threshold")]
    bars = ax_left.bar(labels_left, values_left, color=colors_left, edgecolor="white", linewidth=0.6)
    for bar, value in zip(bars, values_left):
        ax_left.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values_left + [1]) * 0.01,
            f"{value:,}",
            ha="center", va="bottom",
            fontsize=10, color=color("text"),
        )
    accept_pct = (accepted / total * 100.0) if total > 0 else 0.0
    style_axes(ax_left, xlabel="", ylabel="Sample count")
    ax_left.set_title(
        f"Outcome counts  •  acceptance {accept_pct:.1f}%",
        loc="left", pad=8, fontsize=11, fontweight="bold",
    )

    if reason_counts:
        sorted_reasons = sorted(reason_counts.items(), key=lambda item: -item[1])
        top_reasons = sorted_reasons[:8]
        if len(sorted_reasons) > 8:
            other_total = sum(count for _, count in sorted_reasons[8:])
            if other_total > 0:
                top_reasons.append(("other", other_total))
        labels = [_truncate_reason_label(name) for name, _ in top_reasons]
        counts = [int(count) for _, count in top_reasons]
        y_positions = np.arange(len(labels))
        ax_right.barh(y_positions, counts, color=color("rejected"), edgecolor="white", linewidth=0.6, alpha=0.85)
        ax_right.set_yticks(y_positions)
        ax_right.set_yticklabels(labels)
        ax_right.invert_yaxis()
        for y, count in zip(y_positions, counts):
            ax_right.text(
                count + max(counts) * 0.01,
                y,
                f"{count:,}",
                ha="left", va="center",
                fontsize=9, color=color("text"),
            )
        style_axes(ax_right, xlabel="Rejected sample count", ylabel="")
        ax_right.set_title("Top rejection reasons", loc="left", pad=8, fontsize=11, fontweight="bold")
    else:
        ax_right.text(
            0.5, 0.5, "No rejected samples recorded",
            transform=ax_right.transAxes, ha="center", va="center",
        )
        style_axes(ax_right, xlabel="Rejected sample count", ylabel="")
        ax_right.set_title("Top rejection reasons", loc="left", pad=8, fontsize=11, fontweight="bold")

    fig.suptitle(
        f"Two-Segment Dataset Quality  —  {schedule}",
        fontsize=13, fontweight="bold", x=0.04, ha="left",
    )
    valid_chip = "VALID" if (valid_two_segment and valid_ann) else "NOT TRAINING-READY"
    fig.text(
        0.015, 0.02,
        "  •  ".join(
            [
                f"run_trust_mode: {trust_mode}",
                f"valid_for_two_segment_model_training: {valid_two_segment}",
                f"valid_for_two_segment_ann_training: {valid_ann}",
                f"verdict: {valid_chip}",
            ]
        ),
        fontsize=9, color=color("text"), ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": color("grid"), "alpha": 0.94},
    )
    save_figure(fig, path)


def _write_current_load_timeline(
    *,
    path: Path,
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    """Per-servo current/load over collection order."""
    sequence: dict[int, list[tuple[int, float]]] = {sid: [] for sid in range(1, 9)}
    for record in records:
        row_index = int(record.get("row_index") or 0)
        for sid, payload in (record.get("feedback_by_servo") or {}).items():
            load = payload.get("load_proxy_ma")
            if load is None:
                continue
            try:
                sequence.setdefault(int(sid), []).append((row_index, float(load)))
            except (TypeError, ValueError):
                continue
    schedule = str(metrics.get("schedule_type") or "unknown").replace("_", " ")
    config_used = dict(metrics.get("config_used") or {})
    warning_threshold = config_used.get("current_warning_ma")
    sustained_threshold = config_used.get("sustained_overcurrent_ma")
    try:
        warning_threshold = int(warning_threshold) if warning_threshold is not None else None
    except (TypeError, ValueError):
        warning_threshold = None
    try:
        sustained_threshold = int(sustained_threshold) if sustained_threshold is not None else None
    except (TypeError, ValueError):
        sustained_threshold = None

    with report_style() as plt:
        fig, ax = plt.subplots(figsize=(11.0, 5.6), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.14)

    any_data = False
    max_points_per_servo = 2000
    for sid in sorted(sequence):
        points = sequence.get(sid) or []
        if not points:
            continue
        any_data = True
        xs_full = np.asarray([p[0] for p in points], dtype=float)
        ys_full = np.asarray([p[1] for p in points], dtype=float)
        order = np.argsort(xs_full)
        xs_full = xs_full[order]
        ys_full = ys_full[order]
        downsample_idx = _downsample_indices(xs_full.size, max_points=max_points_per_servo)
        xs_plot = xs_full if downsample_idx is None else xs_full[downsample_idx]
        ys_plot = ys_full if downsample_idx is None else ys_full[downsample_idx]
        servo_color = color("segment_a") if sid <= 4 else color("segment_b")
        alpha = max(0.20, 0.85 - 0.10 * ((sid - 1) % 4))
        ax.plot(
            xs_plot, ys_plot,
            color=servo_color, alpha=alpha,
            linewidth=0.7, label=f"servo {sid}",
        )
        if ys_full.size >= 50:
            window = max(25, ys_full.size // 200)
            kernel = np.ones(window) / float(window)
            smoothed = np.convolve(ys_full, kernel, mode="valid")
            offset = (window - 1) // 2
            smoothed_x = xs_full[offset : offset + smoothed.size]
            if smoothed_x.size == smoothed.size:
                ax.plot(
                    smoothed_x, smoothed,
                    color=servo_color, alpha=0.95,
                    linewidth=1.4,
                )
    if warning_threshold is not None and warning_threshold > 0:
        ax.axhline(int(warning_threshold), color=color("threshold"), linestyle=":", linewidth=1.0,
                   label=f"warning {int(warning_threshold)} mA")
    if sustained_threshold is not None and sustained_threshold > 0:
        ax.axhline(int(sustained_threshold), color=color("rejected"), linestyle="-.", linewidth=1.0,
                   label=f"sustained jam {int(sustained_threshold)} mA")
    if not any_data:
        ax.text(
            0.5, 0.5, "No servo load telemetry available",
            transform=ax.transAxes, ha="center", va="center",
        )
    try:
        import matplotlib.ticker as mticker
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _pos: f"{int(value):,}"))
    except Exception:
        pass
    style_axes(ax, xlabel="Sample row (collection order)", ylabel="Load proxy current (mA)")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.92)
    ax.set_title(
        f"Per-servo current/load over collection  —  {schedule}",
        loc="left", pad=8, fontsize=11, fontweight="bold",
    )
    fig.suptitle(
        "Two-Segment Current/Load Timeline",
        fontsize=13, fontweight="bold", x=0.04, ha="left",
    )
    all_values: list[float] = []
    for sid, items in sequence.items():
        all_values.extend(float(v) for _, v in items)
    if all_values:
        arr = np.asarray(all_values, dtype=float)
        warn_count = (
            int(np.sum(arr > int(warning_threshold))) if warning_threshold else 0
        )
        jam_count = (
            int(np.sum(arr > int(sustained_threshold))) if sustained_threshold else 0
        )
        fig.text(
            0.015, 0.02,
            "  •  ".join(
                [
                    f"Load samples: {arr.size:,}",
                    f"Median: {float(np.median(arr)):.1f} mA",
                    f"P95: {float(np.percentile(arr, 95)):.1f} mA",
                    f"Max: {float(np.max(arr)):.1f} mA",
                    f"Above warning: {warn_count:,}",
                    f"Above sustained jam: {jam_count:,}",
                ]
            ),
            fontsize=9, color=color("text"), ha="left", va="bottom",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": color("grid"), "alpha": 0.94},
        )
    save_figure(fig, path)


# ----------------------------------------------------------------------------
# Plot placeholder helper
# ----------------------------------------------------------------------------


_MIN_PNG_BYTES = bytes(
    (
        137, 80, 78, 71, 13, 10, 26, 10,
        0, 0, 0, 13, 73, 72, 68, 82,
        0, 0, 0, 1, 0, 0, 0, 1,
        8, 6, 0, 0, 0, 31, 21, 196, 137,
        0, 0, 0, 13, 73, 68, 65, 84,
        120, 156, 99, 96, 0, 0, 0, 2, 0, 1,
        226, 33, 188, 51,
        0, 0, 0, 0, 73, 69, 78, 68,
        174, 66, 96, 130,
    )
)


def _write_plot_placeholder(path: Path) -> None:
    Path(path).write_bytes(_MIN_PNG_BYTES)
