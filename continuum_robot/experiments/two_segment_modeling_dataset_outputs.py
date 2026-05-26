"""Thesis-quality figures for the two-segment collect-pose dataset.

This module mirrors the single-segment ``modeling_dataset_outputs`` figure
contract and extends it with two-segment-specific diagnostics: per-segment
command panels, per-segment command-to-pose correlation, an 8-servo
position-coverage chart grouped by segment, a dataset-quality breakdown
with rejection reasons, and a per-servo current/load timeline.

Figures are intentionally additive — the legacy
``two_segment_*_report.png`` files written by
:func:`write_two_segment_dataset_outputs` remain untouched so existing
tests, run bundles, and operator workflows keep working. Use these
``thesis_*.png`` files as the canonical thesis evidence set going
forward.
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


# Canonical two-segment thesis figure filenames. The numbering matches the
# single-segment ``thesis_01..thesis_02`` set for the first two figures, then
# adds two-segment-specific 03..06.
THESIS_FIGURE_NAMES: tuple[str, ...] = (
    "thesis_01_workspace_coverage_3d.png",
    "thesis_02_command_and_workspace_2d.png",
    "thesis_03_per_segment_command_pose.png",
    "thesis_04_servo_position_coverage.png",
    "thesis_05_dataset_quality.png",
    "thesis_06_current_load_timeline.png",
)


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
    paths: dict[str, Path] = {}

    plot_jobs: list[tuple[str, Path, Callable[[], None]]] = [
        (
            "thesis_01",
            output_dir / "thesis_01_workspace_coverage_3d.png",
            lambda: _write_workspace_coverage_3d(
                path=output_dir / "thesis_01_workspace_coverage_3d.png",
                records=records,
                metrics=metrics,
            ),
        ),
        (
            "thesis_02",
            output_dir / "thesis_02_command_and_workspace_2d.png",
            lambda: _write_command_and_workspace_2d(
                path=output_dir / "thesis_02_command_and_workspace_2d.png",
                records=records,
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
        # Pair-axis convention (mirrors single-segment): cable deltas
        # [c0, c1, c2, c3] correspond to a tip-target pair (px, py) =
        # (-c0, -c1). When the antagonistic pair is intact this is also
        # equivalent to (+c2, +c3). The two-segment dataset uses the same
        # per-segment cable layout, so the same mapping applies per segment.
        segment_a_pair = _pair_from_cable_deltas(segment_a)
        segment_b_pair = _pair_from_cable_deltas(segment_b)
        records.append(
            {
                "row_index": int(row_index),
                "sample_index": int(getattr(sample, "sample_index", 0) or 0),
                "step_index": int(getattr(sample, "step_index", 0) or 0),
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


def _pair_from_cable_deltas(cable_deltas: list[float]) -> tuple[float, float] | None:
    """Map a 4-tendon cable vector to a tip-target pair using the canonical convention."""
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
    """Truncate a rejection reason to a reasonable width without collapsing distinct strings.

    Long reasons get an ellipsis; short reasons are returned unchanged. The
    cap is wide enough that error strings differing in their tail (servo id,
    register name) do not collide into duplicate y-axis ticks.
    """
    text = str(name)
    if len(text) <= max_chars:
        return text
    return text[: max(1, max_chars - 1)] + "…"


def _downsample_indices(point_count: int, *, max_points: int) -> np.ndarray | None:
    """Return evenly spaced indices into a length-N sequence capped at max_points.

    Used by the high-density timeline figure so a 50k-row run does not draw
    50k × 8 polyline segments per servo (the resulting PNG is unreadable and
    expensive to render). Returns ``None`` when no downsampling is needed.
    """
    if point_count <= max_points or max_points <= 0:
        return None
    return np.linspace(0, point_count - 1, num=max_points, dtype=int)


def _accepted_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Subset to accepted samples that carry a tip position."""
    return [r for r in records if bool(r.get("accepted")) and r.get("tip_xyz_mm") is not None]


def _command_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Subset to accepted samples with both segment pairs available."""
    return [
        r for r in records
        if bool(r.get("accepted"))
        and r.get("segment_a_pair_cm") is not None
        and r.get("segment_b_pair_cm") is not None
    ]


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------


def _write_workspace_coverage_3d(
    *,
    path: Path,
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    """Thesis figure 1: 3D distal tip scatter coloured by sample index.

    Mirrors the single-segment ``thesis_01_workspace_coverage_3d`` figure.
    Each point is one accepted distal tip pose; colour encodes order in the
    collection sequence (viridis from first to last). When intermediate
    pose is available it is overlaid as a translucent secondary cloud so
    the operator can see whether the bottom segment articulates separately
    from the distal tip.
    """
    accepted = _accepted_records(records)
    fig, ax = create_3d_figure(size="thesis_3d")
    schedule = str(metrics.get("schedule_type") or "unknown").replace("_", " ")

    if not accepted:
        ax.text2D(
            0.5, 0.5, "No accepted distal tip poses available",
            transform=ax.transAxes, ha="center", va="center",
        )
        style_3d_axes(ax)
        fig.suptitle(
            "Two-Segment Workspace Coverage During Collect-Pose"
            f"  ({schedule})",
            fontsize=13, fontweight="bold", x=0.04, ha="left",
        )
        save_figure(fig, path)
        return

    xs = [float(r["tip_xyz_mm"][0]) for r in accepted]
    ys = [float(r["tip_xyz_mm"][1]) for r in accepted]
    zs = [float(r["tip_xyz_mm"][2]) for r in accepted]
    indices = np.arange(len(accepted), dtype=float)
    scatter = ax.scatter(
        xs, ys, zs,
        c=indices, cmap="viridis",
        s=14, depthshade=True, alpha=0.85, linewidths=0,
        label="distal tip",
    )

    intermediate_pts = [
        (float(r["intermediate_xyz_mm"][0]), float(r["intermediate_xyz_mm"][1]), float(r["intermediate_xyz_mm"][2]))
        for r in accepted
        if r.get("intermediate_xyz_mm") is not None
    ]
    if intermediate_pts:
        ix = [pt[0] for pt in intermediate_pts]
        iy = [pt[1] for pt in intermediate_pts]
        iz = [pt[2] for pt in intermediate_pts]
        ax.scatter(
            ix, iy, iz,
            color=color("segment_a"),
            s=10, depthshade=True, alpha=0.35, linewidths=0,
            label="intermediate (proximal segment tip)",
        )

    set_equal_xyz(
        ax,
        x_values=xs + [pt[0] for pt in intermediate_pts],
        y_values=ys + [pt[1] for pt in intermediate_pts],
        z_values=zs + [pt[2] for pt in intermediate_pts],
        minimum_span=5.0,
        pad_fraction=0.08,
    )
    style_3d_axes(ax, xlabel="Robot X (mm)", ylabel="Robot Y (mm)", zlabel="Robot Z (mm)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.92)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.55, pad=0.12)
    cbar.set_label("Sample index (time order)")
    cbar.outline.set_edgecolor(color("grid"))

    fig.suptitle(
        f"Two-Segment Workspace Coverage During Collect-Pose  ({schedule})",
        fontsize=13, fontweight="bold", x=0.04, ha="left",
    )

    accepted_count = int(metrics.get("accepted_sample_count") or len(accepted))
    rejected_count = int(metrics.get("rejected_sample_count") or 0)
    total = accepted_count + rejected_count
    accept_rate = (accepted_count / total * 100.0) if total > 0 else 100.0
    x_span = (max(xs) - min(xs)) if xs else 0.0
    y_span = (max(ys) - min(ys)) if ys else 0.0
    z_span = (max(zs) - min(zs)) if zs else 0.0
    fig.text(
        0.015, 0.02,
        "  •  ".join(
            [
                f"Schedule: {schedule}",
                f"Accepted: {accepted_count}/{total}  ({accept_rate:.1f}%)",
                f"Workspace span: X={x_span:.1f} mm  Y={y_span:.1f} mm  Z={z_span:.1f} mm",
                (
                    f"Distal+intermediate poses available"
                    if intermediate_pts
                    else "Distal-only pose (intermediate tracker unavailable)"
                ),
            ]
        ),
        fontsize=9, color=color("text"), ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": color("grid"), "alpha": 0.94},
    )
    save_figure(fig, path)


def _write_command_and_workspace_2d(
    *,
    path: Path,
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    """Thesis figure 2: 2x2 panel — per-segment command + tip XY/XZ.

    Two-segment extension of the single-segment ``thesis_02`` figure: the
    top row carries segment-A and segment-B pair commands (tip-target
    convention) so the reader can see each segment's command space
    independently; the bottom row carries the distal tip XY (top-down) and
    XZ (side) projections. All panels share the sample-index colour scale.
    """
    accepted_command = _command_records(records)
    schedule = str(metrics.get("schedule_type") or "unknown").replace("_", " ")
    with report_style() as plt:
        fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.2), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.10, hspace=0.34, wspace=0.30)
    ax_top_a, ax_top_b = axes[0]
    ax_bot_left, ax_bot_right = axes[1]

    if not accepted_command:
        for ax in axes.flatten():
            ax.text(
                0.5, 0.5, "No accepted command/pose samples available",
                transform=ax.transAxes, ha="center", va="center",
            )
        fig.suptitle(
            "Two-Segment Command Space + Distal Tip Workspace"
            f"  ({schedule})",
            fontsize=13, fontweight="bold", x=0.04, ha="left",
        )
        save_figure(fig, path)
        return

    indices = np.arange(len(accepted_command), dtype=float)
    seg_a = [r["segment_a_pair_cm"] for r in accepted_command]
    seg_b = [r["segment_b_pair_cm"] for r in accepted_command]
    seg_a_x = [pair[0] for pair in seg_a]
    seg_a_y = [pair[1] for pair in seg_a]
    seg_b_x = [pair[0] for pair in seg_b]
    seg_b_y = [pair[1] for pair in seg_b]

    sc_top_a = ax_top_a.scatter(
        seg_a_x, seg_a_y, c=indices, cmap="viridis",
        s=14, alpha=0.85, linewidths=0,
    )
    style_axes(ax_top_a, xlabel="Segment A pair X (cm)", ylabel="Segment A pair Y (cm)")
    ax_top_a.set_title("Segment A command space", loc="left", pad=8, fontsize=11, fontweight="bold")
    ax_top_a.set_aspect("equal", adjustable="datalim")

    ax_top_b.scatter(
        seg_b_x, seg_b_y, c=indices, cmap="viridis",
        s=14, alpha=0.85, linewidths=0,
    )
    style_axes(ax_top_b, xlabel="Segment B pair X (cm)", ylabel="Segment B pair Y (cm)")
    ax_top_b.set_title("Segment B command space", loc="left", pad=8, fontsize=11, fontweight="bold")
    ax_top_b.set_aspect("equal", adjustable="datalim")

    tip_records = [r for r in accepted_command if r.get("tip_xyz_mm") is not None]
    if tip_records:
        tip_indices = np.arange(len(tip_records), dtype=float)
        tip_x = [float(r["tip_xyz_mm"][0]) for r in tip_records]
        tip_y = [float(r["tip_xyz_mm"][1]) for r in tip_records]
        tip_z = [float(r["tip_xyz_mm"][2]) for r in tip_records]
        ax_bot_left.scatter(
            tip_x, tip_y, c=tip_indices, cmap="viridis",
            s=14, alpha=0.85, linewidths=0,
        )
        ax_bot_right.scatter(
            tip_x, tip_z, c=tip_indices, cmap="viridis",
            s=14, alpha=0.85, linewidths=0,
        )
    else:
        for ax in (ax_bot_left, ax_bot_right):
            ax.text(
                0.5, 0.5, "No distal tip poses captured",
                transform=ax.transAxes, ha="center", va="center",
            )
    style_axes(ax_bot_left, xlabel="Robot X (mm)", ylabel="Robot Y (mm)")
    ax_bot_left.set_title("Distal tip XY (top-down)", loc="left", pad=8, fontsize=11, fontweight="bold")
    ax_bot_left.set_aspect("equal", adjustable="datalim")
    style_axes(ax_bot_right, xlabel="Robot X (mm)", ylabel="Robot Z (mm)")
    ax_bot_right.set_title("Distal tip XZ (side)", loc="left", pad=8, fontsize=11, fontweight="bold")
    ax_bot_right.set_aspect("equal", adjustable="datalim")

    cbar = fig.colorbar(sc_top_a, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("Sample index (time order)")
    cbar.outline.set_edgecolor(color("grid"))

    fig.suptitle(
        f"Two-Segment Command Space + Distal Tip Workspace  —  {schedule}",
        fontsize=13, fontweight="bold", x=0.04, ha="left",
    )

    a_x_span = max(seg_a_x) - min(seg_a_x)
    a_y_span = max(seg_a_y) - min(seg_a_y)
    b_x_span = max(seg_b_x) - min(seg_b_x)
    b_y_span = max(seg_b_y) - min(seg_b_y)
    summary = [
        f"Samples: {len(accepted_command)}",
        f"Seg-A pair extent: X {a_x_span:.2f} cm, Y {a_y_span:.2f} cm",
        f"Seg-B pair extent: X {b_x_span:.2f} cm, Y {b_y_span:.2f} cm",
    ]
    if tip_records:
        tip_x_arr = np.asarray([r["tip_xyz_mm"][0] for r in tip_records], dtype=float)
        tip_y_arr = np.asarray([r["tip_xyz_mm"][1] for r in tip_records], dtype=float)
        tip_z_arr = np.asarray([r["tip_xyz_mm"][2] for r in tip_records], dtype=float)
        summary.append(
            f"Tip extent: X {float(np.ptp(tip_x_arr)):.1f} mm, "
            f"Y {float(np.ptp(tip_y_arr)):.1f} mm, Z {float(np.ptp(tip_z_arr)):.1f} mm"
        )
    fig.text(
        0.015, 0.02,
        "  •  ".join(summary),
        fontsize=9, color=color("text"), ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": color("grid"), "alpha": 0.94},
    )
    save_figure(fig, path)


def _write_per_segment_command_pose(
    *,
    path: Path,
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    """Thesis figure 3: per-segment command magnitude vs distal tip excursion.

    Two diagnostic scatters that surface how each segment couples into the
    distal tip position. A two-segment robot where the top segment is
    actually articulating should show a stronger correlation between
    segment-B (top) command magnitude and tip-from-startup excursion than
    segment-A alone. Useful for sanity-checking the routing compensation
    and the operator's segment assignment.
    """
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
            "Per-Segment Command Magnitude vs Distal Tip Excursion"
            f"  ({schedule})",
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
    """Thesis figure 4: 8-servo position coverage grouped by segment.

    Top panel: per-servo observed-range bar chart, with the safety
    ``max_tick_delta_from_startup`` envelope drawn as a horizontal dashed
    reference. Bottom panel: per-servo ``delta_from_startup`` box plot so
    the operator can see distribution shape (sample spread, asymmetry)
    rather than just min/max.
    """
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
        # max_tick_delta_from_startup is a per-servo *one-sided* tick budget;
        # the worst-case observed range from end to end is 2*budget if the
        # trajectory swings through both extremes.
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
    """Thesis figure 5: accepted vs rejected breakdown + rejection-reason bar.

    Replaces the bare 3-bar legacy figure with a thesis-quality two-panel
    layout that surfaces (1) the high-level outcome counts and (2) the
    rejection-reason distribution so the operator can argue *why* a run is
    or isn't trainable, not just the headline number.
    """
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
    # Include transport-event reasons from the failure events log when sample
    # rejection didn't surface them directly.
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
        # Render reasons at full length up to a wide cap so distinct error
        # strings (e.g. "Failed to write goal position for servo 3" vs
        # "Failed to write torque enable for servo 7") never collapse into
        # the same row. Matplotlib will wrap them visually if needed.
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
    """Thesis figure 6: per-servo current/load over collection order.

    For each servo the figure draws a downsampled load-proxy trace plus a
    per-servo rolling-mean envelope so the reader can pick out trends in
    high-density runs without being overwhelmed by point-level jitter.
    The configured warning and sustained-jam thresholds appear as
    horizontal references. Operator-facing: useful for spotting a servo
    that drifted toward jam over a long run, or a transport event that
    landed the trace flat at zero.
    """
    sequence: dict[int, list[tuple[int, float]]] = {sid: [] for sid in range(1, 9)}
    for record in records:
        # row_index is monotonic across the whole run; sample_index resets
        # each command step in the two-segment schedule so it makes a poor
        # x-axis on its own (the figure would only span 0..N-per-step).
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
    # Per-servo display budget for raw points; a long run (50k+ rows) will
    # otherwise produce an unreadable solid block of overlapping lines.
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
        # Rolling envelope so trend is visible through the haze. Skip when
        # we have very few samples (avoid a meaningless 1-point line).
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
    # Lock the x-axis to plain integer notation so the sample-index scale
    # never reads as 10^4 multipliers in the rendered PNG.
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
    # Stat strip footer
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
# Plot placeholder helper (mirrors single-segment for graceful failures)
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
