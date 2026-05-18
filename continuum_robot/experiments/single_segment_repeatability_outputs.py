"""Saved artifacts for the single-segment repeatability experiment."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from continuum_robot.experiments.dataset_tools import extract_tip_or_tool_position_mm
from continuum_robot.experiments.plotting import (
    color,
    create_3d_figure,
    create_figure,
    legend,
    report_style,
    save_figure,
    set_equal_xyz,
    style_3d_axes,
    style_axes,
)
from continuum_robot.experiments.tracker_timing_outputs import _fmt, _write_plot_placeholder


LOG = logging.getLogger(__name__)


def build_single_segment_repeatability_summary_pairs(*, metrics: dict[str, Any]) -> list[tuple[str, str]]:
    """Return compact rows for GUI/report summaries."""
    run_validity = dict(metrics.get("run_validity", {}) or {})
    criteria = dict(run_validity.get("criteria", {}) or {})
    observed = dict(run_validity.get("observed", {}) or {})
    geometry = dict(metrics.get("target_geometry", {}) or {})
    repeat_valid = int(metrics.get("valid_repeat_sample_count", 0) or 0)
    repeat_planned = int(
        criteria.get("planned_repeat_capture_count", metrics.get("planned_visit_count", 0) or 0) or 0
    )
    rejected_count = int(metrics.get("rejected_capture_count", 0) or 0)
    rejected_fraction = observed.get("rejected_capture_fraction")
    pairs = [
        ("Protocol", "Legacy 17 target / all-other approaches"),
        ("Runtime Tip Mode", str(metrics.get("runtime_tip_mode_used", "unknown") or "unknown")),
        ("Runtime Tip Trust", str(metrics.get("runtime_tip_trust_level", "unknown") or "unknown")),
        ("Thesis Trusted", "yes" if metrics.get("thesis_trusted_run") else "no"),
        ("Targets", str(int(metrics.get("target_count", 0) or 0))),
        (
            "Ring Radii",
            (
                f"inner {_fmt(geometry.get('inner_ring_radius_mm'), suffix=' mm')} "
                f"({_fmt(geometry.get('inner_ring_radius_ticks'), suffix=' ticks')}), "
                f"outer {_fmt(geometry.get('outer_ring_radius_mm'), suffix=' mm')} "
                f"({_fmt(geometry.get('outer_ring_radius_ticks'), suffix=' ticks')})"
                if geometry
                else "n/a"
            ),
        ),
        (
            "Amplitude Cap",
            (
                f"{_fmt(geometry.get('max_target_tick_delta_from_startup_cap'), suffix=' ticks')} "
                f"(max target {_fmt(geometry.get('max_target_abs_tick_delta_from_startup'), suffix=' ticks')})"
                if geometry
                else "n/a"
            ),
        ),
        (
            "Spool Diameter",
            _fmt(geometry.get("spool_diameter_mm"), suffix=" mm") if geometry else "n/a",
        ),
        ("Planned Captures", str(int(metrics.get("planned_capture_count", 0) or 0))),
        ("Run Validity", "PASS" if run_validity.get("thesis_valid_run") else "FAIL"),
        (
            "Repeat Coverage",
            f"{repeat_valid}/{repeat_planned}" if repeat_planned > 0 else str(repeat_valid),
        ),
        (
            "Per-Target Minimum",
            (
                f">= {int(criteria.get('min_repeat_captures_per_target', 0) or 0)} repeats"
                if criteria
                else "n/a"
            ),
        ),
        ("Valid Repeat Captures", str(repeat_valid)),
        ("Valid Approach Captures", str(int(metrics.get("valid_approach_sample_count", 0) or 0))),
        (
            "Rejected Captures",
            f"{rejected_count} ({float(rejected_fraction) * 100.0:.1f}%)"
            if rejected_fraction is not None
            else str(rejected_count),
        ),
        ("Pose Frame", str(metrics.get("position_frame", "unknown") or "unknown")),
        ("Overall RMS", _fmt(metrics.get("overall_repeatability_rms_mm"), suffix=" mm")),
        ("Overall Max", _fmt(metrics.get("overall_max_deviation_mm"), suffix=" mm")),
        ("Path Dependence RMS", _fmt(metrics.get("path_dependence_rms_mm"), suffix=" mm")),
        ("Thesis Goal", f"<= {_fmt(metrics.get('thesis_goal_rms_mm'), suffix=' mm')}"),
        ("Goal Result", "PASS" if metrics.get("thesis_goal_pass") else "not met"),
    ]
    comparison = dict(metrics.get("baseline_comparison", {}) or {})
    if comparison.get("available"):
        overall = dict(comparison.get("overall_rms", {}) or {})
        max_dev = dict(comparison.get("overall_max_deviation", {}) or {})
        path = dict(comparison.get("path_dependence_rms", {}) or {})
        rejected = dict(comparison.get("rejected_capture_count", {}) or {})
        pairs.extend(
            [
                ("Baseline Comparison", "available"),
                ("Delta Overall RMS", _signed_fmt(overall.get("delta"), suffix=" mm")),
                ("Delta Overall Max", _signed_fmt(max_dev.get("delta"), suffix=" mm")),
                ("Delta Path RMS", _signed_fmt(path.get("delta"), suffix=" mm")),
                ("Delta Rejected Captures", _signed_int(rejected.get("delta"))),
            ]
        )
    elif comparison:
        pairs.append(("Baseline Comparison", f"unavailable: {comparison.get('error', 'unknown error')}"))
    return pairs


def _repeatability_error_value(row: dict[str, Any]) -> float | None:
    for key in ("spread_rms_mm", "rmse_mm", "XYZ_RMSE_mm", "XY_RMSE_mm"):
        value = _optional_float(row.get(key))
        if value is not None:
            return value
    return None


def _ring_group_rows(metrics: dict[str, Any]) -> list[tuple[str, float]]:
    ring_groups = dict((metrics.get("group_metrics", {}) or {}).get("ring", {}) or {})
    rows: list[tuple[str, float]] = []
    for label in ["center", "inner", "outer"]:
        group = dict(ring_groups.get(label, {}) or {})
        value = _optional_float(group.get("mean_target_rms_mm"))
        if value is not None:
            rows.append((label, value))
    if rows:
        return rows
    values_by_ring: dict[str, list[float]] = {}
    for row in _ordered_per_target(metrics):
        value = _repeatability_error_value(row)
        if value is None:
            continue
        ring = str(row.get("ring", "") or "")
        if not ring:
            target_index = int(row.get("target_index", -1) or -1)
            ring = "center" if target_index == 0 else "inner" if target_index <= 8 else "outer"
        values_by_ring.setdefault(ring, []).append(float(value))
    for label in ["center", "inner", "outer"]:
        values = values_by_ring.get(label, [])
        if values:
            rows.append((label, float(np.mean(values))))
    return rows


def _compact_lines(lines: list[str | None]) -> list[str]:
    return [str(line) for line in lines if line]


def _repeat_points_by_target(*, samples) -> dict[int, list[list[float]]]:
    points: dict[int, list[list[float]]] = {}
    for sample in samples:
        if sample.phase != "repeat":
            continue
        if not bool(sample.extra.get("capture_accepted", True)):
            continue
        position, frame = extract_tip_or_tool_position_mm(sample, tool_id="0A", prefer_robot_frame=True)
        if position is None or frame != "robot":
            continue
        target_index = int(sample.target_index if sample.target_index is not None else -1)
        points.setdefault(target_index, []).append([float(value) for value in position])
    return points


def _ordered_per_target(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [dict(value) for value in (metrics.get("per_target_metrics", {}) or {}).values()],
        key=lambda item: int(item.get("target_index", 10**9)),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None



def _signed_fmt(value: Any, *, suffix: str = "") -> str:
    val = _optional_float(value)
    if val is None:
        return "n/a"
    return f"{val:+.3f}{suffix}"


def _signed_int(value: Any) -> str:
    val = _optional_float(value)
    if val is None:
        return "n/a"
    return f"{int(round(val)):+d}"


def write_single_segment_repeatability_outputs(*, output_dir: Path, metadata, summary, samples) -> dict[str, Path]:
    """Write canonical artifacts for one repeatability run.

    Figure contract: 3 thesis-quality PNGs (justified — each answers a distinct
    thesis claim, see docstrings below). No CSVs, no .txt; everything else lives
    in debug.json.

      - thesis_01_target_returns_3d.png: 3D scatter of all accepted repeat
        captures around their 17 target centroids in robot frame, colored by
        per-target RMS (red highlight for any target > 1.0 mm goal). Answers
        'how tight is the per-target return cluster' in workspace context.
      - thesis_02_per_target_rms_bar.png: per-target RMS bar chart with 1.0 mm
        thesis-goal reference line, bars grouped by ring (center / inner /
        outer). Answers 'do all 17 targets pass individually, not just on
        average'.
      - thesis_03_path_dependence_vs_total.png: paired-metric per target
        (total RMS as circle, path-dep RMS as triangle, connected by a thin
        vertical line). Answers 'is the result approach-direction independent'.

    debug.json consolidates: per-target metrics, path_dependence_by_approach,
    group metrics, run_validity, legacy_style_comparison (was a separate CSV),
    rejection breakdown, provenance, baseline comparison if available,
    run_review sidecar fold-in.

    Dropped: all 11 prior PNGs, repeatability_summary.txt,
    repeatability_debug_samples.csv, repeatability_legacy_style_comparison.csv
    (no external consumers; data preserved in samples.jsonl + debug.json).
    """
    output_dir = Path(output_dir)
    metrics = summary.experiment_metrics if isinstance(summary.experiment_metrics, dict) else {}
    debug_json_path = output_dir / "debug.json"
    thesis_01_path = output_dir / "thesis_01_target_returns_3d.png"
    thesis_02_path = output_dir / "thesis_02_per_target_rms_bar.png"
    thesis_03_path = output_dir / "thesis_03_path_dependence_vs_total.png"

    _write_repeatability_debug_json(
        path=debug_json_path,
        output_dir=output_dir,
        metadata=metadata,
        summary=summary,
        metrics=metrics,
        samples=samples,
    )
    for path, writer in [
        (thesis_01_path, lambda: _write_repeatability_thesis_01_target_returns_3d(
            path=thesis_01_path, samples=samples, metrics=metrics,
        )),
        (thesis_02_path, lambda: _write_repeatability_thesis_02_per_target_rms_bar(
            path=thesis_02_path, metrics=metrics,
        )),
        (thesis_03_path, lambda: _write_repeatability_thesis_03_path_dependence_vs_total(
            path=thesis_03_path, metrics=metrics,
        )),
    ]:
        try:
            writer()
        except Exception:
            LOG.exception("Failed to render %s; writing placeholder", path)
            _write_plot_placeholder(path)
    return {
        "debug_json_path": debug_json_path,
        "thesis_01_path": thesis_01_path,
        "thesis_02_path": thesis_02_path,
        "thesis_03_path": thesis_03_path,
    }


RING_COLORS = {
    "center": "#2563eb",  # blue
    "inner": "#0f766e",   # teal
    "outer": "#d97706",   # orange
}


def _ring_for_target(row: dict[str, Any]) -> str:
    ring = str(row.get("ring", "") or "")
    if ring:
        return ring
    target_index = int(row.get("target_index", -1) or -1)
    if target_index == 0:
        return "center"
    if target_index <= 8:
        return "inner"
    return "outer"


def _write_repeatability_thesis_01_target_returns_3d(
    *,
    path: Path,
    samples,
    metrics: dict[str, Any],
) -> None:
    """Thesis figure 1: 3D scatter of all repeat captures with per-target RMS coloring.

    Every accepted repeat capture appears as a small marker at its 3D position
    in the robot frame. Each target's 16-ish markers cluster around the target
    centroid (drawn as a larger black X). Markers are colored by their target's
    RMS — viridis for targets that meet the 1.0 mm goal, hard red for any
    target that exceeds it. Lets the reader see workspace structure AND
    per-target tightness AND pass/fail in one view.
    """
    points_by_target = _repeat_points_by_target(samples=samples)
    per_target = _ordered_per_target(metrics)
    thesis_goal = float(metrics.get("thesis_goal_rms_mm", 1.0) or 1.0)

    fig, ax = create_3d_figure(size="thesis_3d")
    if not points_by_target or not per_target:
        ax.text2D(0.5, 0.5, "No accepted repeat captures available",
                  transform=ax.transAxes, ha="center", va="center")
        style_3d_axes(ax, title="")
        fig.suptitle("Target Return Repeatability  (17 targets × 16 revisits)",
                     fontsize=13, fontweight="bold", x=0.04, ha="left")
        save_figure(fig, path)
        return

    rms_by_target: dict[int, float | None] = {
        int(row.get("target_index", -1) or -1): _repeatability_error_value(row)
        for row in per_target
    }
    failing_targets = [tidx for tidx, rms in rms_by_target.items() if rms is not None and rms > thesis_goal]

    all_xs: list[float] = []
    all_ys: list[float] = []
    all_zs: list[float] = []
    # Two scatters: passing (viridis on per-target RMS) and failing (hard red).
    pass_xs, pass_ys, pass_zs, pass_c = [], [], [], []
    fail_xs, fail_ys, fail_zs = [], [], []
    for tidx, points in points_by_target.items():
        rms = rms_by_target.get(int(tidx))
        if rms is None:
            continue
        for point in points:
            x, y, z = float(point[0]), float(point[1]), float(point[2])
            all_xs.append(x); all_ys.append(y); all_zs.append(z)
            if rms > thesis_goal:
                fail_xs.append(x); fail_ys.append(y); fail_zs.append(z)
            else:
                pass_xs.append(x); pass_ys.append(y); pass_zs.append(z)
                pass_c.append(rms)

    scatter = None
    if pass_xs:
        scatter = ax.scatter(
            pass_xs, pass_ys, pass_zs,
            c=pass_c, cmap="viridis", vmin=0.0, vmax=thesis_goal,
            s=18, alpha=0.78, linewidths=0,
        )
    if fail_xs:
        ax.scatter(
            fail_xs, fail_ys, fail_zs,
            color=color("rejected"),
            s=24, alpha=0.92, linewidths=0,
            label=f"Targets > {thesis_goal:.1f} mm goal ({len(failing_targets)})",
        )

    # Centroid markers
    centroid_xs, centroid_ys, centroid_zs = [], [], []
    for row in per_target:
        centroid = row.get("centroid_mm")
        if not (isinstance(centroid, list) and len(centroid) >= 3):
            continue
        centroid_xs.append(float(centroid[0]))
        centroid_ys.append(float(centroid[1]))
        centroid_zs.append(float(centroid[2]))
    if centroid_xs:
        ax.scatter(centroid_xs, centroid_ys, centroid_zs,
                   s=80, marker="X", color=color("reference"),
                   edgecolors="white", linewidths=0.6, depthshade=False,
                   label="Target centroids")
        all_xs.extend(centroid_xs); all_ys.extend(centroid_ys); all_zs.extend(centroid_zs)

    set_equal_xyz(ax, x_values=all_xs, y_values=all_ys, z_values=all_zs,
                  minimum_span=5.0, pad_fraction=0.12)
    style_3d_axes(ax, xlabel="Robot X (mm)", ylabel="Robot Y (mm)", zlabel="Robot Z (mm)")

    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.55, pad=0.12)
        cbar.set_label(f"Per-target RMS (mm) — goal ≤ {thesis_goal:.1f}")
        cbar.outline.set_edgecolor(color("grid"))

    legend(ax, loc="upper left")

    overall_rms = _optional_float(metrics.get("overall_repeatability_rms_mm"))
    overall_max = _optional_float(metrics.get("overall_max_deviation_mm"))
    pass_label = "PASS" if metrics.get("thesis_goal_pass") else "FAIL"
    pass_color = color("accepted") if metrics.get("thesis_goal_pass") else color("rejected")
    fig.suptitle("Target Return Repeatability  (17 targets × 16 revisits)",
                 fontsize=13, fontweight="bold", x=0.04, ha="left")
    fig.text(
        0.015, 0.02,
        "  •  ".join(_compact_lines([
            f"Targets: {len(per_target)}",
            f"Repeat captures: {sum(len(p) for p in points_by_target.values())}",
            f"Overall RMS: {_fmt(overall_rms, suffix=' mm')}  •  Max: {_fmt(overall_max, suffix=' mm')}",
            f"Thesis goal ≤ {thesis_goal:.1f} mm: {pass_label}",
        ])),
        fontsize=9, color=color("text"), ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
              "edgecolor": pass_color, "alpha": 0.94, "linewidth": 1.2},
    )
    save_figure(fig, path)


def _write_repeatability_thesis_02_per_target_rms_bar(
    *,
    path: Path,
    metrics: dict[str, Any],
) -> None:
    """Thesis figure 2: per-target RMS bar chart with 1.0 mm thesis-goal line.

    17 bars, one per target, colored by ring group (center / inner / outer).
    Horizontal reference at the thesis goal. Bars exceeding the goal flip to
    the rejected color. Per-target labels along the X axis. The reader sees:
    every target's individual pass/fail vs the singular thesis claim.
    """
    per_target = _ordered_per_target(metrics)
    thesis_goal = float(metrics.get("thesis_goal_rms_mm", 1.0) or 1.0)

    fig, ax = create_figure(size="wide", constrained_layout=False)
    fig.subplots_adjust(left=0.085, right=0.97, top=0.90, bottom=0.22)

    if not per_target:
        ax.text(0.5, 0.5, "No per-target metrics available",
                transform=ax.transAxes, ha="center", va="center")
        style_axes(ax, xlabel="Target", ylabel="RMS (mm)")
        fig.suptitle("Per-Target Repeatability vs Thesis Goal",
                     fontsize=13, fontweight="bold", x=0.04, ha="left")
        save_figure(fig, path)
        return

    labels: list[str] = []
    values: list[float] = []
    bar_colors: list[str] = []
    for row in per_target:
        rms = _repeatability_error_value(row)
        if rms is None:
            continue
        labels.append(str(row.get("label", str(row.get("target_index", "?")))))
        values.append(float(rms))
        ring = _ring_for_target(row)
        bar_colors.append(color("rejected") if rms > thesis_goal else RING_COLORS.get(ring, color("measured")))

    x_positions = np.arange(len(values))
    bars = ax.bar(x_positions, values, color=bar_colors, edgecolor="white", linewidth=0.6, zorder=2)
    # Thesis goal line
    ax.axhline(thesis_goal, color=color("reference"), linestyle="--", linewidth=1.4,
               label=f"Thesis goal ({thesis_goal:.1f} mm)", zorder=3)
    # Optional: overall mean
    overall_rms = _optional_float(metrics.get("overall_repeatability_rms_mm"))
    if overall_rms is not None:
        ax.axhline(overall_rms, color=color("fit"), linestyle=":", linewidth=1.2,
                   label=f"Overall RMS ({overall_rms:.3f} mm)", zorder=3)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    style_axes(ax, xlabel="Target", ylabel="RMS to per-target centroid (mm)")
    # Headroom for the y axis: top of bars + 30% or just above goal line
    y_max = max(max(values), thesis_goal) * 1.20
    ax.set_ylim(0.0, y_max)

    # Inline legend for ring color coding (using proxy artists)
    from matplotlib.patches import Patch
    proxies = [
        Patch(facecolor=RING_COLORS["center"], label="center"),
        Patch(facecolor=RING_COLORS["inner"], label="inner ring"),
        Patch(facecolor=RING_COLORS["outer"], label="outer ring"),
        Patch(facecolor=color("rejected"), label=f"> {thesis_goal:.1f} mm goal"),
    ]
    # Combine ring proxies + the threshold/overall lines
    existing_handles, existing_labels = ax.get_legend_handles_labels()
    ax.legend(handles=proxies + existing_handles, loc="upper right", ncol=2, fontsize=8,
              frameon=True, facecolor="white", edgecolor=color("grid"), framealpha=0.95)

    pass_label = "PASS" if metrics.get("thesis_goal_pass") else "FAIL"
    fig.suptitle("Per-Target Repeatability vs Thesis Goal",
                 fontsize=13, fontweight="bold", x=0.04, ha="left")

    failing_count = sum(1 for v in values if v > thesis_goal)
    fig.text(
        0.015, 0.02,
        "  •  ".join(_compact_lines([
            f"Targets: {len(values)}",
            f"Failing targets: {failing_count}",
            f"Overall RMS: {_fmt(overall_rms, suffix=' mm')}",
            f"Path-dep RMS: {_fmt(metrics.get('path_dependence_rms_mm'), suffix=' mm')}",
            f"Thesis: {pass_label}",
        ])),
        fontsize=9, color=color("text"), ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
              "edgecolor": color("grid"), "alpha": 0.94},
    )
    save_figure(fig, path)


def _write_repeatability_thesis_03_path_dependence_vs_total(
    *,
    path: Path,
    metrics: dict[str, Any],
) -> None:
    """Thesis figure 3: paired metric per target — total RMS vs path-dependence RMS.

    For each target: total RMS (filled circle) and per-target path-dep RMS
    (triangle) at the same X position, connected by a thin vertical line.
    Thesis goal reference line. If for every target the path-dep marker is
    well below the total RMS marker, approach direction is not a significant
    contributor to return error — the headline claim is approach-independent.
    """
    per_target = _ordered_per_target(metrics)
    thesis_goal = float(metrics.get("thesis_goal_rms_mm", 1.0) or 1.0)
    # Per-target path-dep RMS computed from path_dependence_by_approach grouped by target.
    path_entries = list(metrics.get("path_dependence_by_approach", []) or [])
    by_target: dict[int, list[float]] = {}
    for entry in path_entries:
        tidx = entry.get("target_index")
        deviation = _optional_float(entry.get("deviation_mm"))
        if tidx is None or deviation is None:
            continue
        by_target.setdefault(int(tidx), []).append(deviation)
    path_rms_by_target = {
        tidx: float(np.sqrt(np.mean(np.square(vals))))
        for tidx, vals in by_target.items()
        if vals
    }

    fig, ax = create_figure(size="wide", constrained_layout=False)
    fig.subplots_adjust(left=0.085, right=0.97, top=0.90, bottom=0.22)

    if not per_target:
        ax.text(0.5, 0.5, "No per-target metrics available",
                transform=ax.transAxes, ha="center", va="center")
        style_axes(ax, xlabel="Target", ylabel="RMS (mm)")
        fig.suptitle("Per-Target Path-Dependence vs Total Repeatability RMS",
                     fontsize=13, fontweight="bold", x=0.04, ha="left")
        save_figure(fig, path)
        return

    labels: list[str] = []
    total_vals: list[float] = []
    path_vals: list[float] = []
    target_indices: list[int] = []
    for row in per_target:
        tidx = int(row.get("target_index", -1) or -1)
        rms = _repeatability_error_value(row)
        if rms is None:
            continue
        labels.append(str(row.get("label", str(tidx))))
        total_vals.append(float(rms))
        path_vals.append(float(path_rms_by_target.get(tidx, 0.0)))
        target_indices.append(tidx)

    x_positions = np.arange(len(total_vals))

    # Pair connectors
    for x, t_val, p_val in zip(x_positions, total_vals, path_vals):
        ax.plot([x, x], [t_val, p_val], color=color("neutral"), linewidth=0.9, alpha=0.45, zorder=1)

    ax.scatter(x_positions, total_vals, marker="o", s=70, color=color("measured"),
               edgecolors="white", linewidths=0.7, zorder=3,
               label="Total per-target RMS")
    ax.scatter(x_positions, path_vals, marker="^", s=70, color=color("fit"),
               edgecolors="white", linewidths=0.7, zorder=3,
               label="Path-dependence RMS (approach-conditioned)")

    ax.axhline(thesis_goal, color=color("reference"), linestyle="--", linewidth=1.4,
               label=f"Thesis goal ({thesis_goal:.1f} mm)", zorder=2)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    style_axes(ax, xlabel="Target", ylabel="RMS (mm)")
    y_max = max(max(total_vals + path_vals), thesis_goal) * 1.20
    ax.set_ylim(0.0, y_max)
    legend(ax, loc="upper right", ncol=1)

    fig.suptitle("Per-Target Path-Dependence vs Total Repeatability RMS",
                 fontsize=13, fontweight="bold", x=0.04, ha="left")

    overall_path = _optional_float(metrics.get("path_dependence_rms_mm"))
    overall_total = _optional_float(metrics.get("overall_repeatability_rms_mm"))
    ratio_text = ""
    if overall_path is not None and overall_total and overall_total > 0.0:
        ratio_text = f"path/total = {(overall_path / overall_total) * 100.0:.0f}%"
    fig.text(
        0.015, 0.02,
        "  •  ".join(_compact_lines([
            f"Targets: {len(total_vals)}",
            f"Overall total RMS: {_fmt(overall_total, suffix=' mm')}",
            f"Overall path-dep RMS: {_fmt(overall_path, suffix=' mm')}",
            ratio_text or None,
        ])),
        fontsize=9, color=color("text"), ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
              "edgecolor": color("grid"), "alpha": 0.94},
    )
    save_figure(fig, path)


def _write_repeatability_debug_json(
    *,
    path: Path,
    output_dir: Path,
    metadata,
    summary,
    metrics: dict[str, Any],
    samples,
) -> None:
    """Consolidate everything not on thesis figures.

    Folds in: per-target metrics, path_dependence_by_approach, group metrics,
    run_validity, legacy_style_comparison (was its own CSV), rejection
    breakdown, provenance, baseline_comparison if available, run_review.
    """
    accepted_repeat = sum(
        1 for s in samples
        if getattr(s, "phase", "") == "repeat" and bool(getattr(s, "extra", {}).get("capture_accepted", True))
    )
    rejected = int(metrics.get("rejected_capture_count", 0) or 0)
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
        "experiment_name": getattr(metadata, "experiment_name", "single_segment_repeatability"),
        "status": getattr(summary, "status", None),
        "thesis_goal_rms_mm": float(metrics.get("thesis_goal_rms_mm", 1.0) or 1.0),
        "thesis_goal_pass": bool(metrics.get("thesis_goal_pass")),
        "overall": {
            "repeatability_rms_mm": _optional_float(metrics.get("overall_repeatability_rms_mm")),
            "max_deviation_mm": _optional_float(metrics.get("overall_max_deviation_mm")),
            "path_dependence_rms_mm": _optional_float(metrics.get("path_dependence_rms_mm")),
        },
        "sample_counts": {
            "valid_repeat_sample_count": int(metrics.get("valid_repeat_sample_count", 0) or 0),
            "valid_approach_sample_count": int(metrics.get("valid_approach_sample_count", 0) or 0),
            "rejected_capture_count": rejected,
            "accepted_repeat_count_from_samples": accepted_repeat,
        },
        "run_validity": dict(metrics.get("run_validity", {}) or {}),
        "per_target_metrics": dict(metrics.get("per_target_metrics", {}) or {}),
        "path_dependence_by_approach": list(metrics.get("path_dependence_by_approach", []) or []),
        "group_metrics": dict(metrics.get("group_metrics", {}) or {}),
        "legacy_style_comparison": dict(metrics.get("legacy_style_comparison", {}) or {}),
        "baseline_comparison": dict(metrics.get("baseline_comparison", {}) or {}),
        "provenance": {
            "runtime_tip_mode_used": metrics.get("runtime_tip_mode_used"),
            "runtime_tip_trust_level": metrics.get("runtime_tip_trust_level"),
            "thesis_trusted_run": bool(metrics.get("thesis_trusted_run")),
            "position_frame": metrics.get("position_frame"),
            "registration_available": bool(metrics.get("registration_available")),
            "tip_calibration_available": bool(metrics.get("tip_calibration_available")),
        },
        "run_review": review_payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_repeatability_json_default), encoding="utf-8")


def _repeatability_json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unserialisable type: {type(value).__name__}")
