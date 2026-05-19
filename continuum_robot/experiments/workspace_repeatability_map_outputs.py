"""Output bundle writer for the workspace repeatability map experiment.

Consumes the per-visit timeseries samples emitted by
:class:`continuum_robot.experiments.workspace_repeatability_map.WorkspaceRepeatabilityMapExperiment`
and produces three thesis-style figures that match the existing
``thesis_NN_*.png`` naming and visual style used by
``modeling_dataset_outputs.py`` and ``single_segment_repeatability_outputs.py``:

* ``thesis_01_workspace_rms_3d.png`` -- 3D centroid scatter, color = per-target
  RMS spread (mm). Cubic XYZ bounding box so the disk reads as a disk and Z-axis
  noise does not get auto-stretched.
* ``thesis_02_workspace_rms_map.png`` -- top-down 2D map with Delaunay
  ``tricontourf`` smooth fill clipped to the 12 mm disk + centroid dots on top,
  shared colormap with figure 1.
* ``thesis_03_rms_vs_amplitude.png`` -- per-target RMS as a function of
  commanded deflection amplitude. Scatter colored by ``z_stddev`` plus a
  binned median with the p25-p75 band drawn underneath, so the reader sees
  the individual targets AND the workspace-wide radial trend.
* ``thesis_04_2d_repeatability_map.png`` -- top-down 2D scatter, one dot per
  target colored by per-target RMS spread. Same data and colormap as figure 1,
  flattened to XY for thesis prints (no 3D rotation), and no spatial
  interpolation (unlike figure 2's contour fill) so the reader sees each
  measured target directly.

Plus the non-figure artifacts:

* ``workspace_map_per_target.csv``   per-target dispersion table
* ``workspace_map_visits.jsonl``     per-visit raw record
* ``workspace_map_summary.json``     workspace-wide stats + worst-5 list

No pass/fail visual coloring: the figures stay neutral and let the operator
read the actual numbers from the footer caption.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from continuum_robot.experiments.plotting import (
    SEMANTIC_COLORS,
    color,
    create_3d_figure,
    create_figure,
    import_matplotlib,
    report_style,
    save_figure,
    set_equal_xyz,
    style_3d_axes,
    style_axes,
)


# Bundle filenames. Match the ``thesis_NN_<name>.png`` convention used by
# the rest of the experiment package's thesis figures.
PER_TARGET_CSV = "workspace_map_per_target.csv"
VISITS_JSONL = "workspace_map_visits.jsonl"
SUMMARY_JSON = "workspace_map_summary.json"
THESIS_01_PNG = "thesis_01_workspace_rms_3d.png"
THESIS_02_PNG = "thesis_02_workspace_rms_map.png"
THESIS_03_PNG = "thesis_03_rms_vs_amplitude.png"
THESIS_04_PNG = "thesis_04_2d_repeatability_map.png"


# ---------------------------------------------------------------------------
# Per-target metric computation
# ---------------------------------------------------------------------------


def compute_workspace_repeatability_metrics(
    visits_by_target: Mapping[int, Sequence[Mapping[str, Any]]],
    targets_by_index: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return per-target dispersion rows.

    Each input visit must carry a ``position_mm`` triple in the same robot
    frame across all visits to a given target (otherwise the centroid is
    meaningless). Visits with ``rejected=True`` or non-finite positions are
    excluded from the metrics; the count of rejections is preserved in the
    row so the operator can spot tracker-flaky targets.
    """

    rows: list[dict[str, Any]] = []
    for target_index in sorted(visits_by_target.keys()):
        visits = list(visits_by_target.get(int(target_index), []))
        target_meta = dict(targets_by_index.get(int(target_index), {}) or {})
        accepted_positions: list[np.ndarray] = []
        rejected_visits = 0
        for visit in visits:
            if bool(visit.get("rejected", False)):
                rejected_visits += 1
                continue
            position = visit.get("position_mm")
            if not isinstance(position, (list, tuple)) or len(position) != 3:
                rejected_visits += 1
                continue
            try:
                arr = np.asarray([float(value) for value in position], dtype=float)
            except (TypeError, ValueError):
                rejected_visits += 1
                continue
            if not np.all(np.isfinite(arr)):
                rejected_visits += 1
                continue
            accepted_positions.append(arr)
        n_kept = len(accepted_positions)
        n_total = len(visits)
        if n_kept == 0:
            rows.append(
                {
                    "target_index": int(target_index),
                    "target_label": str(target_meta.get("label", f"T{target_index:03d}")),
                    "target_x_mm": float(target_meta.get("x_mm") or 0.0),
                    "target_y_mm": float(target_meta.get("y_mm") or 0.0),
                    "target_amplitude_mm": float(target_meta.get("amplitude_mm") or 0.0),
                    "target_angle_deg": float(target_meta.get("angle_deg") or 0.0),
                    "n_visits_total": int(n_total),
                    "n_visits_kept": 0,
                    "n_visits_rejected": int(rejected_visits),
                    "centroid_x_mm": None,
                    "centroid_y_mm": None,
                    "centroid_z_mm": None,
                    "rms_spread_mm": None,
                    "max_spread_mm": None,
                    "x_stddev_mm": None,
                    "y_stddev_mm": None,
                    "z_stddev_mm": None,
                }
            )
            continue
        positions = np.stack(accepted_positions, axis=0)
        centroid = positions.mean(axis=0)
        deviations = positions - centroid
        norm_distances = np.linalg.norm(deviations, axis=1)
        rms_spread = float(np.sqrt(np.mean(norm_distances * norm_distances)))
        rows.append(
            {
                "target_index": int(target_index),
                "target_label": str(target_meta.get("label", f"T{target_index:03d}")),
                "target_x_mm": float(target_meta.get("x_mm") or 0.0),
                "target_y_mm": float(target_meta.get("y_mm") or 0.0),
                "target_amplitude_mm": float(target_meta.get("amplitude_mm") or 0.0),
                "target_angle_deg": float(target_meta.get("angle_deg") or 0.0),
                "n_visits_total": int(n_total),
                "n_visits_kept": int(n_kept),
                "n_visits_rejected": int(rejected_visits),
                "centroid_x_mm": float(centroid[0]),
                "centroid_y_mm": float(centroid[1]),
                "centroid_z_mm": float(centroid[2]),
                "rms_spread_mm": float(rms_spread),
                "max_spread_mm": float(np.max(norm_distances)),
                "x_stddev_mm": float(np.std(positions[:, 0], ddof=0)),
                "y_stddev_mm": float(np.std(positions[:, 1], ddof=0)),
                "z_stddev_mm": float(np.std(positions[:, 2], ddof=0)),
            }
        )
    return rows


def summarize_workspace_repeatability(
    per_target_rows: Sequence[Mapping[str, Any]],
    *,
    worst_n: int = 5,
    thesis_goal_rms_mm: float | None = None,
) -> dict[str, Any]:
    """Aggregate per-target rows into a workspace-wide summary."""
    rms_values = [
        float(row["rms_spread_mm"])
        for row in per_target_rows
        if row.get("rms_spread_mm") is not None and math.isfinite(float(row.get("rms_spread_mm") or float("nan")))
    ]
    max_values = [
        float(row["max_spread_mm"])
        for row in per_target_rows
        if row.get("max_spread_mm") is not None and math.isfinite(float(row.get("max_spread_mm") or float("nan")))
    ]
    n_targets = len(per_target_rows)
    n_targets_with_data = len(rms_values)
    if not rms_values:
        return {
            "target_count": int(n_targets),
            "targets_with_data": 0,
            "workspace_rms_mean_mm": None,
            "workspace_rms_median_mm": None,
            "workspace_rms_p95_mm": None,
            "workspace_rms_max_mm": None,
            "workspace_max_spread_max_mm": None,
            "worst_targets": [],
            "thesis_goal_rms_mm": float(thesis_goal_rms_mm) if thesis_goal_rms_mm is not None else None,
            "fraction_above_thesis_goal": None,
        }
    rms_array = np.asarray(rms_values, dtype=float)
    sorted_rows = sorted(
        (row for row in per_target_rows if row.get("rms_spread_mm") is not None),
        key=lambda row: float(row["rms_spread_mm"]),
        reverse=True,
    )
    worst_rows = [
        {
            "target_index": int(row["target_index"]),
            "target_label": str(row["target_label"]),
            "rms_spread_mm": float(row["rms_spread_mm"]),
            "max_spread_mm": float(row["max_spread_mm"]) if row.get("max_spread_mm") is not None else None,
            "n_visits_kept": int(row["n_visits_kept"]),
        }
        for row in sorted_rows[: max(0, int(worst_n))]
    ]
    summary: dict[str, Any] = {
        "target_count": int(n_targets),
        "targets_with_data": int(n_targets_with_data),
        "workspace_rms_mean_mm": float(np.mean(rms_array)),
        "workspace_rms_median_mm": float(np.median(rms_array)),
        "workspace_rms_p95_mm": float(np.percentile(rms_array, 95)),
        "workspace_rms_max_mm": float(np.max(rms_array)),
        "workspace_max_spread_max_mm": float(np.max(max_values)) if max_values else None,
        "worst_targets": worst_rows,
        "thesis_goal_rms_mm": float(thesis_goal_rms_mm) if thesis_goal_rms_mm is not None else None,
    }
    if thesis_goal_rms_mm is not None and rms_array.size > 0:
        summary["fraction_above_thesis_goal"] = float(
            np.mean(rms_array > float(thesis_goal_rms_mm))
        )
    return summary


def group_visits_by_target(
    samples: Sequence[Mapping[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    """Bucket per-visit sample-extras (already in ``sample.extra`` form) by target_index."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        target_index = sample.get("target_index")
        if target_index is None:
            continue
        try:
            grouped[int(target_index)].append(dict(sample))
        except (TypeError, ValueError):
            continue
    return dict(grouped)


# ---------------------------------------------------------------------------
# Bundle entry-point
# ---------------------------------------------------------------------------


def write_workspace_repeatability_map_outputs(
    *,
    output_dir: Path,
    target_catalog: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    max_amplitude_mm: float | None = None,
    thesis_goal_rms_mm: float | None = None,
    quality: str | None = None,
) -> dict[str, Path]:
    """Write CSV / JSONL / summary.json + three thesis figures into ``output_dir``.

    Returns a dict of artifact-name -> path. Missing inputs are handled
    quietly (zero-visit runs still write summary.json + empty CSV/JSONL);
    the figures are skipped only when no per-target data exists.

    ``max_amplitude_mm`` lets the figures draw the configured workspace
    boundary as a reference circle / disk -- pass ``WorkspaceRepeatabilityMapConfig.max_amplitude_mm``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets_by_index = {int(target.get("target_index", index)): dict(target) for index, target in enumerate(target_catalog)}
    visits_by_target = group_visits_by_target(samples)
    per_target_rows = compute_workspace_repeatability_metrics(
        visits_by_target=visits_by_target,
        targets_by_index=targets_by_index,
    )
    summary = summarize_workspace_repeatability(
        per_target_rows,
        worst_n=5,
        thesis_goal_rms_mm=thesis_goal_rms_mm,
    )

    paths: dict[str, Path] = {}
    paths["per_target_csv"] = _write_per_target_csv(output_dir / PER_TARGET_CSV, per_target_rows)
    paths["visits_jsonl"] = _write_visits_jsonl(output_dir / VISITS_JSONL, samples)
    paths["summary_json"] = _write_summary_json(
        output_dir / SUMMARY_JSON,
        summary=summary,
        per_target_rows=per_target_rows,
    )

    rows_with_data = [row for row in per_target_rows if row.get("rms_spread_mm") is not None]
    if rows_with_data:
        # Resolve disk radius for the workspace boundary if the caller didn't
        # pass it; fall back to the largest commanded amplitude.
        resolved_max_amplitude = (
            float(max_amplitude_mm)
            if max_amplitude_mm is not None and float(max_amplitude_mm) > 0.0
            else float(max(float(row.get("target_amplitude_mm") or 0.0) for row in rows_with_data) or 1.0)
        )
        paths["thesis_01"] = _write_workspace_thesis_01_rms_3d(
            output_dir / THESIS_01_PNG,
            rows_with_data,
            summary=summary,
            max_amplitude_mm=resolved_max_amplitude,
            quality=quality,
        )
        paths["thesis_02"] = _write_workspace_thesis_02_rms_map(
            output_dir / THESIS_02_PNG,
            rows_with_data,
            summary=summary,
            max_amplitude_mm=resolved_max_amplitude,
            quality=quality,
        )
        paths["thesis_03"] = _write_workspace_thesis_03_rms_vs_amplitude(
            output_dir / THESIS_03_PNG,
            rows_with_data,
            summary=summary,
            max_amplitude_mm=resolved_max_amplitude,
            quality=quality,
        )
        paths["thesis_04"] = _write_workspace_thesis_04_rms_map_scatter(
            output_dir / THESIS_04_PNG,
            rows_with_data,
            summary=summary,
            max_amplitude_mm=resolved_max_amplitude,
            quality=quality,
        )
    return paths


# ---------------------------------------------------------------------------
# CSV / JSON writers
# ---------------------------------------------------------------------------


def _write_per_target_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    fieldnames = [
        "target_index",
        "target_label",
        "target_x_mm",
        "target_y_mm",
        "target_amplitude_mm",
        "target_angle_deg",
        "n_visits_total",
        "n_visits_kept",
        "n_visits_rejected",
        "centroid_x_mm",
        "centroid_y_mm",
        "centroid_z_mm",
        "rms_spread_mm",
        "max_spread_mm",
        "x_stddev_mm",
        "y_stddev_mm",
        "z_stddev_mm",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    return path


def _write_visits_jsonl(path: Path, samples: Sequence[Mapping[str, Any]]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            target_index = sample.get("target_index")
            if target_index is None:
                continue
            handle.write(json.dumps(dict(sample), default=_json_safe) + "\n")
    return path


def _write_summary_json(
    path: Path,
    *,
    summary: Mapping[str, Any],
    per_target_rows: Sequence[Mapping[str, Any]],
) -> Path:
    payload = {
        "schema_version": "workspace_repeatability_map_v1",
        "summary": dict(summary),
        "per_target_rows": [dict(row) for row in per_target_rows],
    }
    path.write_text(json.dumps(payload, indent=2, default=_json_safe), encoding="utf-8")
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable: {value!r}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _color_vmax(values: Sequence[float]) -> float:
    """Pick a color-scale maximum that doesn't let one outlier wash the figure out.

    Returns the actual max when no significant outlier exists, otherwise the
    95th percentile. The threshold ``max <= p95 * 1.25`` is the "no outlier"
    band; above that, the figure switches to p95-clamped to keep most of the
    workspace visible. The colormap norm is configured to clip values above
    this maximum so they appear as the brightest viridis but don't break the
    scale.
    """
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return 1.0
    p95 = float(np.percentile(arr, 95))
    p_max = float(arr.max())
    if p_max <= p95 * 1.25 or p95 <= 0:
        return max(p_max, 1e-9)
    return max(p95, 1e-9)


def _format_mm(value: float | None, *, decimals: int = 3) -> str:
    if value is None:
        return "n/a"
    if not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{decimals}f} mm"


def _footer_caption_lines(
    *,
    summary: Mapping[str, Any],
    rows_with_data: Sequence[Mapping[str, Any]],
    max_amplitude_mm: float,
) -> list[str]:
    """Workspace-wide RMS numbers + worst target. These are the headline stats."""
    total_visits_kept = sum(int(row.get("n_visits_kept") or 0) for row in rows_with_data)
    visits_per_target = max(
        (int(row.get("n_visits_total") or 0) for row in rows_with_data), default=0
    )
    parts = [
        f"Targets: {summary.get('target_count', 0)}  ({summary.get('targets_with_data', 0)} with data)",
        f"Visits/target: up to {visits_per_target}  (kept {total_visits_kept} captures)",
        f"Workspace R: {max_amplitude_mm:.1f} mm",
        f"RMS mean: {_format_mm(summary.get('workspace_rms_mean_mm'))}",
        f"median: {_format_mm(summary.get('workspace_rms_median_mm'))}",
        f"p95: {_format_mm(summary.get('workspace_rms_p95_mm'))}",
        f"max: {_format_mm(summary.get('workspace_rms_max_mm'))}",
    ]
    worst = list(summary.get("worst_targets") or [])
    if worst:
        worst_top = worst[0]
        parts.append(
            f"worst: {worst_top['target_label']} ({_format_mm(worst_top.get('rms_spread_mm'))})"
        )
    return parts


def _draw_footer(fig, lines: Sequence[str]) -> None:
    fig.text(
        0.015,
        0.012,
        "  •  ".join(lines),
        fontsize=9,
        color=color("text"),
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": color("grid"), "alpha": 0.94},
    )


def _apply_suptitle(fig, text: str) -> None:
    """Consistent left-anchored title for the thesis-figure set."""
    fig.suptitle(text, fontsize=13, fontweight="bold", x=0.04, ha="left", y=0.965)


# ---------------------------------------------------------------------------
# Thesis figure 1 -- 3D centroid scatter
# ---------------------------------------------------------------------------


def _write_workspace_thesis_01_rms_3d(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
    max_amplitude_mm: float,
    quality: str | None,
) -> Path:
    """Thesis figure 1: 3D scatter of per-target centroids colored by RMS spread.

    One dot per target at its measured centroid, colored by the per-target
    3D RMS distance from that centroid. Cubic XYZ bounding box (via
    ``set_equal_xyz``) prevents Z-axis auto-stretching, so an essentially
    planar workspace reads as planar instead of looking like the robot is
    drifting wildly out of plane.
    """
    centroids = np.asarray(
        [
            [float(row["centroid_x_mm"]), float(row["centroid_y_mm"]), float(row["centroid_z_mm"])]
            for row in rows
        ],
        dtype=float,
    )
    rms_values = np.asarray([float(row["rms_spread_mm"]) for row in rows], dtype=float)
    vmax = _color_vmax(rms_values.tolist())

    fig, ax = create_3d_figure(size="thesis_3d", constrained_layout=False)
    # Manual margins so the 3D axes use the full figure (constrained_layout
    # otherwise leaves the 3D axes oddly squished to the left).
    fig.subplots_adjust(left=0.02, right=0.92, top=0.92, bottom=0.10)
    scatter = ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        centroids[:, 2],
        c=rms_values,
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        s=46,
        edgecolors="white",
        linewidths=0.5,
        depthshade=True,
        alpha=0.92,
    )
    # Cube around BOTH the centroids and the configured disk radius so the
    # figure never trims the workspace boundary off the corners.
    span_xs = list(centroids[:, 0]) + [-max_amplitude_mm, max_amplitude_mm]
    span_ys = list(centroids[:, 1]) + [-max_amplitude_mm, max_amplitude_mm]
    span_zs = list(centroids[:, 2])
    set_equal_xyz(
        ax,
        x_values=span_xs,
        y_values=span_ys,
        z_values=span_zs,
        minimum_span=max(2.0 * max_amplitude_mm, 4.0),
        pad_fraction=0.10,
    )
    style_3d_axes(ax, xlabel="Robot X (mm)", ylabel="Robot Y (mm)", zlabel="Robot Z (mm)")

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.62, pad=0.04, fraction=0.04)
    cbar.set_label("Per-target RMS spread (mm)")
    cbar.outline.set_edgecolor(color("grid"))

    _apply_suptitle(fig, "Workspace Repeatability Map  —  per-target 3D RMS spread")
    _draw_footer(
        fig,
        _footer_caption_lines(summary=summary, rows_with_data=rows, max_amplitude_mm=max_amplitude_mm),
    )
    return save_figure(fig, Path(path), quality=quality)


# ---------------------------------------------------------------------------
# Thesis figure 2 -- top-down tricontourf map
# ---------------------------------------------------------------------------


def _write_workspace_thesis_02_rms_map(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
    max_amplitude_mm: float,
    quality: str | None,
) -> Path:
    """Thesis figure 2: top-down ``tricontourf`` heatmap clipped to the disk.

    Delaunay-triangulates the 100 (or N) centroid XY positions and contour-fills
    the per-target RMS spread continuously across the disk. The fill is clipped
    to the configured workspace boundary so the figure shows the actual
    operational region. Centroid dots are drawn on top with the same colormap
    so the reader sees both the smooth field and the individual data points.
    """
    plt = import_matplotlib()
    from matplotlib import tri as _mtri
    from matplotlib.patches import Circle

    xs = np.asarray([float(row["centroid_x_mm"]) for row in rows], dtype=float)
    ys = np.asarray([float(row["centroid_y_mm"]) for row in rows], dtype=float)
    rms_values = np.asarray([float(row["rms_spread_mm"]) for row in rows], dtype=float)
    vmax = _color_vmax(rms_values.tolist())

    with report_style() as plt_styled:
        fig, ax = plt_styled.subplots(figsize=(7.6, 6.4), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.92, top=0.92, bottom=0.11)

    # Triangulate and contour-fill. tricontourf needs at least 3 non-collinear
    # points; the Vogel-disk-fill has 100 by construction, but guard anyway.
    contour = None
    if len(xs) >= 3:
        try:
            triang = _mtri.Triangulation(xs, ys)
            contour = ax.tricontourf(
                triang,
                rms_values,
                levels=22,
                cmap="viridis",
                vmin=0.0,
                vmax=vmax,
                extend="max",
                zorder=1,
            )
        except RuntimeError:
            contour = None

    # Disk reference circle + clip the contour to the disk so contour-fill
    # doesn't bulge past the workspace edge along Delaunay scallops.
    disk_outline = Circle(
        (0.0, 0.0),
        max_amplitude_mm,
        fill=False,
        edgecolor=color("reference"),
        linestyle="--",
        linewidth=1.4,
        zorder=4,
    )
    ax.add_patch(disk_outline)
    if contour is not None:
        disk_clip = Circle((0.0, 0.0), max_amplitude_mm, transform=ax.transData)
        ax.add_patch(disk_clip)
        disk_clip.set_visible(False)
        # matplotlib >=3.8 deprecated .collections in favour of an iterable; both still work.
        try:
            for coll in contour.collections:
                coll.set_clip_path(disk_clip)
        except AttributeError:
            # On newer matplotlib, the contour set is itself clippable.
            try:
                contour.set_clip_path(disk_clip)
            except Exception:
                pass

    centroids = ax.scatter(
        xs,
        ys,
        c=rms_values,
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        s=42,
        edgecolors="white",
        linewidths=0.6,
        zorder=5,
    )

    pad = max_amplitude_mm * 0.10
    ax.set_xlim(-max_amplitude_mm - pad, max_amplitude_mm + pad)
    ax.set_ylim(-max_amplitude_mm - pad, max_amplitude_mm + pad)
    ax.set_aspect("equal", adjustable="box")
    style_axes(ax, xlabel="Robot X (mm)", ylabel="Robot Y (mm)")

    color_source = contour if contour is not None else centroids
    cbar = fig.colorbar(color_source, ax=ax, shrink=0.85, pad=0.025)
    cbar.set_label("Per-target RMS spread (mm)")
    cbar.outline.set_edgecolor(color("grid"))

    _apply_suptitle(fig, "Workspace Repeatability Map  —  top-down RMS field")
    _draw_footer(
        fig,
        _footer_caption_lines(summary=summary, rows_with_data=rows, max_amplitude_mm=max_amplitude_mm),
    )
    return save_figure(fig, Path(path), quality=quality)


# ---------------------------------------------------------------------------
# Thesis figure 3 -- RMS vs commanded amplitude
# ---------------------------------------------------------------------------


def _write_workspace_thesis_03_rms_vs_amplitude(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
    max_amplitude_mm: float,
    quality: str | None,
) -> Path:
    """Thesis figure 3: per-target RMS spread as a function of commanded amplitude.

    Scatter colored by ``z_stddev`` (so the reader sees whether bad targets are
    out-of-plane) plus a binned median with a p25-p75 shaded band drawn behind
    the scatter. The trend line answers the central physical question for a
    cable-driven robot: does repeatability worsen near the rim, or is it
    uniform across the workspace?
    """
    amplitudes = np.asarray([float(row["target_amplitude_mm"]) for row in rows], dtype=float)
    rms_values = np.asarray([float(row["rms_spread_mm"]) for row in rows], dtype=float)
    z_std = np.asarray([float(row["z_stddev_mm"] or 0.0) for row in rows], dtype=float)

    with report_style() as plt_styled:
        fig, ax = plt_styled.subplots(figsize=(7.8, 4.3), constrained_layout=False)
    fig.subplots_adjust(left=0.095, right=0.93, top=0.90, bottom=0.18)

    # Binned median + p25/p75 band drawn underneath the scatter so the
    # individual targets stay readable on top.
    n_bins = max(3, min(8, len(rows) // 12))
    if amplitudes.size > 1 and float(amplitudes.max()) > float(amplitudes.min()):
        bin_edges = np.linspace(amplitudes.min(), amplitudes.max(), n_bins + 1)
        bin_centers: list[float] = []
        medians: list[float] = []
        p25s: list[float] = []
        p75s: list[float] = []
        counts: list[int] = []
        for i in range(n_bins):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]
            if i == n_bins - 1:
                mask = (amplitudes >= lower) & (amplitudes <= upper)
            else:
                mask = (amplitudes >= lower) & (amplitudes < upper)
            if not mask.any():
                continue
            bin_centers.append(0.5 * (lower + upper))
            medians.append(float(np.median(rms_values[mask])))
            p25s.append(float(np.percentile(rms_values[mask], 25)))
            p75s.append(float(np.percentile(rms_values[mask], 75)))
            counts.append(int(mask.sum()))
        if bin_centers:
            ax.fill_between(
                bin_centers,
                p25s,
                p75s,
                color=color("model"),
                alpha=0.18,
                linewidth=0,
                label="p25-p75 band",
                zorder=1,
            )
            ax.plot(
                bin_centers,
                medians,
                color=color("model"),
                linewidth=2.0,
                marker="o",
                markersize=6,
                label="binned median",
                zorder=2,
            )

    scatter = ax.scatter(
        amplitudes,
        rms_values,
        c=z_std,
        cmap="viridis",
        s=30,
        alpha=0.82,
        linewidths=0,
        zorder=3,
        label="per-target RMS",
    )

    style_axes(
        ax,
        xlabel="Commanded amplitude (mm from neutral)",
        ylabel="Per-target RMS spread (mm)",
    )

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.85, pad=0.025)
    cbar.set_label("Z stddev (mm)")
    cbar.outline.set_edgecolor(color("grid"))

    # X axis extends slightly past the largest amplitude so the rim dots aren't on the edge.
    if amplitudes.size > 0:
        ax.set_xlim(-0.5, max(float(amplitudes.max()), float(max_amplitude_mm)) * 1.05 + 0.5)
    # Y axis floor at 0; force a minimum positive ceiling so all-zero RMS
    # data (synthetic edge cases) doesn't trigger matplotlib's "identical
    # ylims" singular-transform warning.
    y_max_raw = float(np.max(rms_values)) if rms_values.size > 0 else 1.0
    ax.set_ylim(0.0, max(y_max_raw * 1.15, 0.05))

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            loc="upper left",
            frameon=True,
            facecolor="white",
            edgecolor="#cbd5e1",
            framealpha=0.95,
        )

    _apply_suptitle(fig, "Repeatability vs commanded deflection")
    _draw_footer(
        fig,
        _footer_caption_lines(summary=summary, rows_with_data=rows, max_amplitude_mm=max_amplitude_mm),
    )
    return save_figure(fig, Path(path), quality=quality)


# ---------------------------------------------------------------------------
# Thesis figure 4 -- 2D per-target scatter (flat XY companion to figure 1)
# ---------------------------------------------------------------------------


def _write_workspace_thesis_04_rms_map_scatter(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
    max_amplitude_mm: float,
    quality: str | None,
) -> Path:
    """Thesis figure 4: top-down per-target scatter colored by RMS spread.

    The 2D companion to figure 1: same one-dot-per-target visualization and
    same colormap, but flattened to XY. Two practical differences from the
    other figures in the set:

    - vs figure 1 (3D scatter): prints flat, no rotation needed for a thesis
      PDF, and a top-down view makes spatial structure (rim vs center,
      directional bias) immediately readable.
    - vs figure 2 (contour-filled field): no Delaunay smoothing -- the reader
      sees every measured target directly, with no spatial interpolation
      hiding which targets are actually carrying the workspace-level numbers.

    Axes are autoscaled to the actual tip workspace extent, not the
    commanded cable amplitude. ``max_amplitude_mm`` is the commanded cable
    deflection ceiling (tendon-space input), while ``centroid_x/y_mm`` are
    tip positions in robot frame; a curving continuum robot's tip moves
    much further than its cable input, so the two are not the same workspace
    and a cable-amplitude reference disk would visually misrepresent the
    physical region.
    """
    plt = import_matplotlib()
    from matplotlib.patches import Circle

    xs = np.asarray([float(row["centroid_x_mm"]) for row in rows], dtype=float)
    ys = np.asarray([float(row["centroid_y_mm"]) for row in rows], dtype=float)
    rms_values = np.asarray([float(row["rms_spread_mm"]) for row in rows], dtype=float)
    vmax = _color_vmax(rms_values.tolist())

    with report_style() as plt_styled:
        fig, ax = plt_styled.subplots(figsize=(7.6, 6.4), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.92, top=0.92, bottom=0.11)

    # Auto-scale axes to the actual tip workspace, with symmetric square
    # bounds so X and Y read at the same scale. Pad by 8% so the rim dots
    # don't kiss the edge.
    if xs.size > 0:
        xy_extent = max(float(np.max(np.abs(xs))), float(np.max(np.abs(ys))), 1.0)
    else:
        xy_extent = 1.0
    pad = xy_extent * 0.08
    ax.set_xlim(-xy_extent - pad, xy_extent + pad)
    ax.set_ylim(-xy_extent - pad, xy_extent + pad)
    ax.set_aspect("equal", adjustable="box")

    # Small dashed reference circle at the commanded cable-amplitude radius
    # ONLY if it sits clearly inside the tip workspace (otherwise it implies
    # a workspace boundary that does not exist in robot frame). Labelled to
    # avoid the "is that the workspace?" misread.
    if 0 < max_amplitude_mm < xy_extent * 0.85:
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                max_amplitude_mm,
                fill=False,
                edgecolor=color("reference"),
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                zorder=2,
            )
        )
        ax.annotate(
            f"cable amplitude\n({max_amplitude_mm:.0f} mm)",
            xy=(max_amplitude_mm * 0.707, max_amplitude_mm * 0.707),
            xytext=(max_amplitude_mm * 0.95, max_amplitude_mm * 0.95),
            fontsize=8,
            color=color("axis"),
            ha="left",
            va="bottom",
            alpha=0.85,
        )

    scatter = ax.scatter(
        xs,
        ys,
        c=rms_values,
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        s=72,
        edgecolors="white",
        linewidths=0.7,
        alpha=0.95,
        zorder=3,
    )

    style_axes(ax, xlabel="Robot X (mm)", ylabel="Robot Y (mm)")

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.85, pad=0.025)
    cbar.set_label("Per-target RMS spread (mm)")
    cbar.outline.set_edgecolor(color("grid"))

    _apply_suptitle(fig, "Workspace Repeatability Map  —  per-target 2D RMS spread (tip frame)")
    _draw_footer(
        fig,
        _footer_caption_lines(summary=summary, rows_with_data=rows, max_amplitude_mm=max_amplitude_mm),
    )
    return save_figure(fig, Path(path), quality=quality)
