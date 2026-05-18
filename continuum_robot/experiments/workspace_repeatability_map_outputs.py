"""Output bundle writer for the workspace repeatability map experiment.

Consumes the per-visit timeseries samples emitted by
:class:`continuum_robot.experiments.workspace_repeatability_map.WorkspaceRepeatabilityMapExperiment`
and produces the analysis artifacts the operator wants:

* ``workspace_map_per_target.csv``    -- one row per target: centroid xyz,
  RMS spread, per-axis stddev, max-of-15 deviation, visit count.
* ``workspace_map_visits.jsonl``      -- one row per visit: target index,
  cycle, visit-in-cycle, raw position, accepted/rejected.
* ``workspace_map_summary.json``      -- workspace-wide metrics: mean RMS,
  max RMS, p95 RMS, plus the worst-N target list.
* ``workspace_map_3d_report.png``     -- 3D scatter, color = RMS spread.
* ``workspace_map_xy_heatmap_report.png`` -- top-down XY view, same colormap.
* ``workspace_map_axis_stddev_report.png`` -- per-target stacked bars of
  the X/Y/Z standard deviations sorted by RMS spread.

The plotting helpers use matplotlib's Agg backend (no GUI dependency) so the
artifacts can be regenerated from a saved run on any machine. The 3D scatter
is the centerpiece figure -- it's what the operator asked for.
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
    create_figure,
    figure_dpi,
    import_matplotlib,
    report_style,
    save_figure,
    style_axes,
)


# Bundle filenames mirror the rest of the experiment package's report-naming
# convention so a fresh operator can find them by pattern. Anything that gets
# embedded in a thesis figure gets a ``_report.png`` suffix.
PER_TARGET_CSV = "workspace_map_per_target.csv"
VISITS_JSONL = "workspace_map_visits.jsonl"
SUMMARY_JSON = "workspace_map_summary.json"
SCATTER_3D_PNG = "workspace_map_3d_report.png"
XY_HEATMAP_PNG = "workspace_map_xy_heatmap_report.png"
AXIS_STDDEV_PNG = "workspace_map_axis_stddev_report.png"


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
        # ``sample`` here is the ``extra`` dict already projected -- callers pass
        # ``[sample.extra for sample in session.samples]``. That decouples this
        # module from the timeseries-sample dataclass and makes it easy to feed
        # synthetic dicts in tests.
        target_index = sample.get("target_index")
        if target_index is None:
            continue
        try:
            grouped[int(target_index)].append(dict(sample))
        except (TypeError, ValueError):
            continue
    return dict(grouped)


def write_workspace_repeatability_map_outputs(
    *,
    output_dir: Path,
    target_catalog: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    thesis_goal_rms_mm: float | None = None,
    quality: str | None = None,
) -> dict[str, Path]:
    """Write CSV / JSONL / summary.json / PNG bundle into ``output_dir``.

    Returns a dict of artifact-name → path. Missing inputs are handled
    silently (e.g. a run that captured zero visits still gets a summary.json
    and the empty CSVs / JSONL); the figures are skipped only when there is
    no data at all to plot.
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

    rows_with_data = [
        row for row in per_target_rows if row.get("rms_spread_mm") is not None
    ]
    if rows_with_data:
        paths["scatter_3d"] = _plot_3d_scatter(
            output_dir / SCATTER_3D_PNG, rows_with_data, quality=quality
        )
        paths["xy_heatmap"] = _plot_xy_heatmap(
            output_dir / XY_HEATMAP_PNG, rows_with_data, quality=quality
        )
        paths["axis_stddev"] = _plot_axis_stddev_bars(
            output_dir / AXIS_STDDEV_PNG, rows_with_data, quality=quality
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
# Plotting
# ---------------------------------------------------------------------------


def _vmin_vmax(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    vmin = float(min(values))
    vmax = float(max(values))
    if math.isclose(vmin, vmax):
        # Avoid all-one-color renders when every target has identical spread.
        vmax = vmin + max(0.05, abs(vmin) * 0.05 + 0.05)
    return vmin, vmax


def _plot_3d_scatter(path: Path, rows: Sequence[Mapping[str, Any]], *, quality: str | None) -> Path:
    plt = import_matplotlib()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the 3d projection

    centroids = np.asarray(
        [[float(row["centroid_x_mm"]), float(row["centroid_y_mm"]), float(row["centroid_z_mm"])] for row in rows],
        dtype=float,
    )
    rms_values = [float(row["rms_spread_mm"]) for row in rows]
    vmin, vmax = _vmin_vmax(rms_values)
    with report_style() as plt:
        fig = plt.figure(figsize=(8.4, 6.6))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            centroids[:, 2],
            c=rms_values,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            s=44,
            edgecolors="white",
            linewidths=0.5,
        )
        ax.set_xlabel("X centroid (mm)")
        ax.set_ylabel("Y centroid (mm)")
        ax.set_zlabel("Z centroid (mm)")
        ax.set_title("Workspace repeatability map — per-target 3D RMS spread (mm)", loc="left", pad=14, fontweight="bold")
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.08)
        cbar.set_label("RMS spread (mm)")
        # Annotate the five worst points so the operator sees them at a glance.
        worst = sorted(rows, key=lambda row: float(row["rms_spread_mm"]), reverse=True)[:5]
        for row in worst:
            ax.text(
                float(row["centroid_x_mm"]),
                float(row["centroid_y_mm"]),
                float(row["centroid_z_mm"]),
                f" {row['target_label']} ({float(row['rms_spread_mm']):.2f} mm)",
                fontsize=8,
                color=SEMANTIC_COLORS["text"],
            )
    return save_figure(fig, Path(path), quality=quality)


def _plot_xy_heatmap(path: Path, rows: Sequence[Mapping[str, Any]], *, quality: str | None) -> Path:
    rms_values = [float(row["rms_spread_mm"]) for row in rows]
    vmin, vmax = _vmin_vmax(rms_values)
    with report_style() as plt:
        fig, ax = plt.subplots(figsize=(7.6, 6.4), constrained_layout=True)
        xs = [float(row["centroid_x_mm"]) for row in rows]
        ys = [float(row["centroid_y_mm"]) for row in rows]
        scatter = ax.scatter(
            xs,
            ys,
            c=rms_values,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            s=110,
            edgecolors="white",
            linewidths=0.5,
        )
        ax.set_aspect("equal", adjustable="box")
        style_axes(
            ax,
            title="Workspace repeatability map — top-down (color = RMS spread)",
            xlabel="X centroid (mm)",
            ylabel="Y centroid (mm)",
        )
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label("RMS spread (mm)")
        # Mark the five worst points; subtle text so the dots stay readable.
        worst = sorted(rows, key=lambda row: float(row["rms_spread_mm"]), reverse=True)[:5]
        for row in worst:
            ax.annotate(
                f"{row['target_label']}\n{float(row['rms_spread_mm']):.2f} mm",
                xy=(float(row["centroid_x_mm"]), float(row["centroid_y_mm"])),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
                color=SEMANTIC_COLORS["text"],
            )
    return save_figure(fig, Path(path), quality=quality)


def _plot_axis_stddev_bars(path: Path, rows: Sequence[Mapping[str, Any]], *, quality: str | None) -> Path:
    sorted_rows = sorted(rows, key=lambda row: float(row["rms_spread_mm"]), reverse=True)
    labels = [str(row["target_label"]) for row in sorted_rows]
    x_std = [float(row["x_stddev_mm"]) for row in sorted_rows]
    y_std = [float(row["y_stddev_mm"]) for row in sorted_rows]
    z_std = [float(row["z_stddev_mm"]) for row in sorted_rows]
    indices = np.arange(len(labels), dtype=float)
    with report_style() as plt:
        fig, ax = plt.subplots(figsize=(max(8.4, len(labels) * 0.1), 4.6), constrained_layout=True)
        ax.bar(indices, x_std, color=SEMANTIC_COLORS["segment_a"], label="X stddev")
        ax.bar(indices, y_std, bottom=x_std, color=SEMANTIC_COLORS["segment_b"], label="Y stddev")
        bottom_xy = [a + b for a, b in zip(x_std, y_std)]
        ax.bar(indices, z_std, bottom=bottom_xy, color=SEMANTIC_COLORS["model"], label="Z stddev")
        # Only label every Nth tick so the axis stays readable for 100 targets.
        step = max(1, len(labels) // 20)
        tick_positions = indices[::step]
        tick_labels = [labels[int(i)] for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
        style_axes(
            ax,
            title="Per-target standard deviation by axis (sorted by RMS spread, descending)",
            xlabel="Target (sorted by RMS spread)",
            ylabel="standard deviation (mm)",
        )
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", framealpha=0.95)
    return save_figure(fig, Path(path), quality=quality)
