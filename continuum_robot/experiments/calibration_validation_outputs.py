"""Saved artifacts for offline registration/pivot validation experiments."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from continuum_robot.experiments.plotting import (
    add_metric_box,
    color,
    create_3d_figure,
    create_figure,
    figure_dpi,
    legend,
    report_style,
    save_figure,
    set_equal_xy,
    set_equal_xyz,
    style_3d_axes,
    style_axes,
)
from continuum_robot.experiments.tracker_timing_outputs import (
    _body_font,
    _body_bold_font,
    _draw_bar_chart,
    _draw_histogram,
    _draw_panel,
    _draw_summary_pairs,
    _draw_title,
    _ensure_plot_qt_app,
    _new_image,
    _panel_font,
    _qt_plotting_is_safe,
    _write_plot_placeholder,
)

try:
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QColor, QPainter, QPen

    _QT_AVAILABLE = True
except ModuleNotFoundError:
    _QT_AVAILABLE = False


def write_registration_validation_outputs(*, output_dir: Path, metadata, summary) -> dict[str, Path]:
    """Write canonical output artifacts for registration validation.

    Figure contract (mirrors pivot_validation):
      - 2 thesis-quality PNGs only
      - debug.json folds in outliers, rotation deltas, and the run_review sidecar
      - per-run rows live in summary.json under experiment_metrics.per_run_rows
        (NOT in samples.jsonl, which is the typed timeseries contract loaded by
        the GUI; aggregation experiments keep it empty)
      - metrics.csv and registration_validation_summary.txt are intentionally
        NOT written (GUI controller shows "not written" gracefully; bundle
        exporter only picks them up if present)
    """
    metrics = summary.experiment_metrics if isinstance(summary.experiment_metrics, dict) else {}
    output_dir = Path(output_dir)
    debug_json_path = output_dir / "debug.json"
    thesis_01_path = output_dir / "thesis_01_robot_origins_3d.png"
    thesis_02_path = output_dir / "thesis_02_within_vs_cross_run_quality.png"

    _write_registration_debug_json(
        path=debug_json_path, output_dir=output_dir, metadata=metadata, summary=summary, metrics=metrics,
    )
    for path, writer in [
        (thesis_01_path, lambda: _write_registration_thesis_01_robot_origins_3d(path=thesis_01_path, metrics=metrics)),
        (thesis_02_path, lambda: _write_registration_thesis_02_within_vs_cross_run_quality(path=thesis_02_path, metrics=metrics)),
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


def write_pivot_validation_outputs(*, output_dir: Path, metadata, summary) -> dict[str, Path]:
    """Write canonical output artifacts for pivot validation.

    Figure contract: 2 thesis-quality PNGs only. All diagnostic content lives in
    debug.json. The per-run rows are NOT written to samples.jsonl (typed
    contract for ExperimentTimeseriesSample records re-loaded by the GUI); they
    live in summary.json under experiment_metrics.per_run_rows. metrics.csv and
    pivot_validation_summary.txt are intentionally NOT written: the GUI shows
    \"not written\" gracefully and the bundle exporter only picks them up if
    they exist.
    """
    metrics = summary.experiment_metrics if isinstance(summary.experiment_metrics, dict) else {}
    output_dir = Path(output_dir)
    debug_json_path = output_dir / "debug.json"
    thesis_01_path = output_dir / "thesis_01_tip_vectors_3d.png"
    thesis_02_path = output_dir / "thesis_02_sample_count_vs_quality.png"

    _write_pivot_debug_json(path=debug_json_path, output_dir=output_dir, metadata=metadata, summary=summary, metrics=metrics)
    for path, writer in [
        (thesis_01_path, lambda: _write_pivot_thesis_01_tip_vectors_3d(path=thesis_01_path, metrics=metrics)),
        (thesis_02_path, lambda: _write_pivot_thesis_02_sample_count_vs_quality(path=thesis_02_path, metrics=metrics)),
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


def _write_rows_csv(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    header_overrides: dict[str, str] | None = None,
) -> None:
    fieldnames: list[str] = []
    header_overrides = dict(header_overrides or {})
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    output_headers = [header_overrides.get(key, key) for key in fieldnames]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_headers or ["empty"])
        writer.writeheader()
        if not fieldnames:
            writer.writerow({"empty": ""})
            return
        for row in rows:
            writer.writerow(
                {
                    header_overrides.get(key, key): _csv_value(row.get(key))
                    for key in fieldnames
                }
            )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return str(value)
    return value


def _pivot_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in list(metrics.get("per_run_rows") or [])]


def _write_pivot_debug_json(
    *,
    path: Path,
    output_dir: Path,
    metadata,
    summary,
    metrics: dict[str, Any],
) -> None:
    """Consolidate every diagnostic / failure signal into a single JSON file.

    Folds in run_review.json (if present), per-run sample-rejection rates,
    outlier runs (distance-to-consensus exceeds threshold), and any source
    artifacts that were skipped during analysis.
    """
    rows = _pivot_rows(metrics)
    distance_summary = dict(metrics.get("distance_to_consensus_summary_mm", {}) or {})
    mean_dist = _optional_float(distance_summary.get("mean")) or 0.0
    std_dist = _optional_float(distance_summary.get("std")) or 0.0
    outlier_threshold_mm = mean_dist + 2.0 * std_dist

    outliers: list[dict[str, Any]] = []
    for row in rows:
        distance = _optional_float(row.get("distance_to_consensus_mm"))
        if distance is None or std_dist <= 0.0:
            continue
        if distance > outlier_threshold_mm:
            outliers.append(
                {
                    "label": row.get("label"),
                    "path": row.get("path"),
                    "distance_to_consensus_mm": distance,
                    "rmse_mm": _optional_float(row.get("rmse_mm")),
                    "z_score": (distance - mean_dist) / std_dist if std_dist > 0 else None,
                }
            )

    rejection_rows = []
    for row in rows:
        total = int(row.get("sample_count_total") or 0)
        rejected = int(row.get("sample_count_rejected") or 0)
        if total <= 0:
            continue
        rejection_rows.append(
            {
                "label": row.get("label"),
                "sample_count_total": total,
                "sample_count_rejected": rejected,
                "rejection_rate": rejected / total,
            }
        )

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
        "experiment_name": getattr(metadata, "experiment_name", "pivot_validation"),
        "status": getattr(summary, "status", None),
        "outlier_detection": {
            "method": "distance_to_consensus_mm > mean + 2*std",
            "mean_distance_mm": mean_dist,
            "std_distance_mm": std_dist,
            "threshold_mm": outlier_threshold_mm,
            "outlier_runs": outliers,
            "outlier_count": len(outliers),
        },
        "sample_rejection_by_run": rejection_rows,
        "invalid_source_runs": list(metrics.get("invalid_runs") or []),
        "selected_run_count": int(metrics.get("selected_run_count", 0) or 0),
        "valid_run_count": int(metrics.get("valid_run_count", 0) or 0),
        "run_review": review_payload,
    }
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _write_pivot_thesis_01_tip_vectors_3d(*, path: Path, metrics: dict[str, Any]) -> None:
    """Thesis figure 1: 3D scatter of solved tip-offset vectors in tool-local frame.

    Each point is one source pivot run; color encodes that run's solve RMSE.
    A consensus X marks the cross-run mean. A translucent wireframe sphere at
    radius = max distance to consensus visually answers "how tight is the
    cluster" without forcing the reader to parse the metric box.
    """
    rows = _pivot_rows(metrics)
    points: list[tuple[float, float, float]] = []
    rmse_for_color: list[float] = []
    labels: list[str] = []
    for index, row in enumerate(rows):
        vector = row.get("tip_vector_local_mm")
        if not (isinstance(vector, list) and len(vector) >= 3):
            continue
        try:
            point = (float(vector[0]), float(vector[1]), float(vector[2]))
        except (TypeError, ValueError):
            continue
        points.append(point)
        rmse_for_color.append(float(row.get("rmse_mm") or 0.0))
        labels.append(f"R{index + 1}")

    fig, ax = create_3d_figure(size="thesis_3d")
    if not points:
        ax.text2D(
            0.5,
            0.5,
            "No valid pivot tip-offset vectors available",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        style_3d_axes(ax, title="")
        fig.suptitle("Pivot Tip-Offset Consistency (Tool-Local Frame)", fontsize=13, fontweight="bold", x=0.04, ha="left")
        save_figure(fig, path)
        return

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]

    consensus = metrics.get("consensus_tip_vector_local_mm")
    consensus_xyz: tuple[float, float, float] | None = None
    if isinstance(consensus, list) and len(consensus) >= 3:
        try:
            consensus_xyz = (float(consensus[0]), float(consensus[1]), float(consensus[2]))
        except (TypeError, ValueError):
            consensus_xyz = None

    # Wireframe spread reference: 3 great circles at radius = max distance to consensus.
    if consensus_xyz is not None and len(points) >= 2:
        cx, cy, cz = consensus_xyz
        radius = max(
            float(np.linalg.norm(np.asarray(point, dtype=float) - np.asarray(consensus_xyz, dtype=float)))
            for point in points
        )
        if radius > 0.0:
            theta = np.linspace(0.0, 2.0 * np.pi, 96)
            circle_kwargs = {"color": color("neutral"), "alpha": 0.32, "linewidth": 0.9, "linestyle": "--"}
            ax.plot(cx + radius * np.cos(theta), cy + radius * np.sin(theta), np.full_like(theta, cz), **circle_kwargs)
            ax.plot(cx + radius * np.cos(theta), np.full_like(theta, cy), cz + radius * np.sin(theta), **circle_kwargs)
            ax.plot(np.full_like(theta, cx), cy + radius * np.cos(theta), cz + radius * np.sin(theta),
                    label=f"Max-spread sphere ({radius:.3f} mm)", **circle_kwargs)

    scatter = ax.scatter(
        xs, ys, zs,
        c=rmse_for_color, cmap="viridis",
        s=70, depthshade=True, edgecolors="white", linewidths=0.6,
    )

    # Drop-lines from each point to the consensus.
    if consensus_xyz is not None:
        cx, cy, cz = consensus_xyz
        for px, py, pz in points:
            ax.plot([cx, px], [cy, py], [cz, pz], color=color("neutral"), linewidth=0.9, alpha=0.55)
        ax.scatter(
            [cx], [cy], [cz],
            s=200, marker="X",
            color=color("reference"), edgecolors="white", linewidths=1.2,
            depthshade=False, label="Consensus",
        )

    # Per-point labels (cheap legend without cluttering)
    if len(points) <= 12:
        for (px, py, pz), label_text in zip(points, labels):
            ax.text(px, py, pz, f"  {label_text}", fontsize=8, color=color("text"), zorder=10)

    # Equal cubic box around all the geometry, including consensus + sphere radius pad.
    pad_xs, pad_ys, pad_zs = list(xs), list(ys), list(zs)
    if consensus_xyz is not None:
        pad_xs.append(consensus_xyz[0])
        pad_ys.append(consensus_xyz[1])
        pad_zs.append(consensus_xyz[2])
    set_equal_xyz(ax, x_values=pad_xs, y_values=pad_ys, z_values=pad_zs, minimum_span=1.0, pad_fraction=0.20)
    style_3d_axes(ax, xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.55, pad=0.12)
    cbar.set_label("Per-run pivot RMSE (mm)")
    cbar.outline.set_edgecolor(color("grid"))

    legend(ax, loc="upper left")

    fig.suptitle("Pivot Tip-Offset Consistency (Tool-Local Frame)", fontsize=13, fontweight="bold", x=0.04, ha="left")

    distance_summary = dict(metrics.get("distance_to_consensus_summary_mm", {}) or {})
    rmse_summary = dict(metrics.get("rmse_summary_mm", {}) or {})
    fig.text(
        0.015, 0.02,
        "  •  ".join(
            _compact_metric_lines(
                [
                    f"Runs: {int(metrics.get('valid_run_count', len(points)) or 0)}",
                    f"Mean RMSE: {_fmt(rmse_summary.get('mean'))} mm",
                    f"Mean spread: {_fmt(distance_summary.get('mean'))} mm",
                    f"Max spread: {_fmt(distance_summary.get('max'))} mm",
                ]
            )
        ),
        fontsize=9, color=color("text"), ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": color("grid"), "alpha": 0.94},
    )
    save_figure(fig, path)


def _write_pivot_thesis_02_sample_count_vs_quality(*, path: Path, metrics: dict[str, Any]) -> None:
    """Thesis figure 2: does more pivot-dance data improve solve quality?

    Single 2D panel with two co-plotted series against samples-used:
      - Distance to consensus (filled circle) — cross-run repeatability
      - Within-run RMSE (filled triangle) — solver residual quality
    A thin vertical line per run pairs the two markers so each run reads as a
    single data point with two quality faces. Dashed regression lines per
    series quantify the trend; correlation coefficients sit in the metric
    strip. 2D because the two metrics share a unit (mm), so a shared error
    axis is more honest than forcing them into separate spatial dimensions.
    """
    rows = _pivot_rows(metrics)
    runs: list[tuple[int, float, float, str]] = []  # (used, distance, rmse, label)
    for index, row in enumerate(rows):
        sample_used = row.get("sample_count_used")
        distance = _optional_float(row.get("distance_to_consensus_mm"))
        rmse = _optional_float(row.get("rmse_mm"))
        if sample_used is None or distance is None:
            continue
        try:
            used_int = int(sample_used)
        except (TypeError, ValueError):
            continue
        runs.append((used_int, distance, rmse if rmse is not None else 0.0, f"R{index + 1}"))

    # constrained_layout=False so we can reserve bottom space for the metric
    # strip placed via fig.text (constrained_layout doesn't account for fig.text).
    fig, ax = create_figure(size="wide", constrained_layout=False)
    fig.subplots_adjust(left=0.085, right=0.965, top=0.90, bottom=0.22)
    if not runs:
        ax.text(0.5, 0.5, "No per-run sample-count data available", transform=ax.transAxes, ha="center", va="center")
        style_axes(ax, xlabel="Samples used per pivot dance", ylabel="Error (mm)")
        fig.suptitle("Pivot Convergence — Does More Sampling Help?", fontsize=13, fontweight="bold", x=0.04, ha="left")
        save_figure(fig, path)
        return

    runs.sort(key=lambda entry: entry[0])
    used = [entry[0] for entry in runs]
    distances = [entry[1] for entry in runs]
    rmses = [entry[2] for entry in runs]
    labels = [entry[3] for entry in runs]

    # Per-run pair connectors (vertical line linking the two markers at same X)
    for x_val, d_val, r_val in zip(used, distances, rmses):
        ax.plot([x_val, x_val], [d_val, r_val], color=color("neutral"), linewidth=0.9, alpha=0.45, zorder=1)

    ax.scatter(
        used, distances,
        marker="o", s=95, color=color("measured"),
        edgecolors="white", linewidths=0.8, zorder=3,
        label="Distance to consensus (cross-run)",
    )
    ax.scatter(
        used, rmses,
        marker="^", s=95, color=color("fit"),
        edgecolors="white", linewidths=0.8, zorder=3,
        label="Within-run RMSE (solver residual)",
    )

    # Regression lines + correlations (only when X actually varies).
    correlations: dict[str, float | None] = {"distance": None, "rmse": None}
    if len(set(used)) > 1:
        x_min, x_max = float(min(used)), float(max(used))
        x_pad = max((x_max - x_min) * 0.08, 1.0)
        x_line = np.linspace(x_min - x_pad, x_max + x_pad, 64)
        for series_name, values, series_color in [
            ("distance", distances, color("measured")),
            ("rmse", rmses, color("fit")),
        ]:
            if len({round(v, 9) for v in values}) <= 1:
                continue
            try:
                slope, intercept = np.polyfit(used, values, 1)
                ax.plot(
                    x_line, slope * x_line + intercept,
                    color=series_color, linewidth=1.4, linestyle="--", alpha=0.6, zorder=2,
                )
                with np.errstate(invalid="ignore", divide="ignore"):
                    corr = float(np.corrcoef(used, values)[0, 1])
                if np.isfinite(corr):
                    correlations[series_name] = corr
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                pass

    # Per-run labels above the higher of each pair.
    for x_val, d_val, r_val, label_text in zip(used, distances, rmses, labels):
        top_y = max(d_val, r_val)
        ax.annotate(
            label_text,
            xy=(x_val, top_y),
            xytext=(0, 9),
            textcoords="offset points",
            fontsize=9, fontweight="bold",
            color=color("text"), ha="center",
        )

    # Y range with headroom for labels and a 0 floor so error magnitude is honest.
    y_max = max(max(distances), max(rmses))
    ax.set_ylim(0.0, y_max * 1.22 + 0.02)

    style_axes(ax, xlabel="Samples used per pivot dance", ylabel="Error (mm)")
    # Upper-left: trends go up-and-to-the-right, so the legend sits in the
    # empty quadrant and never collides with per-run labels.
    legend(ax, loc="upper left", ncol=1)
    fig.suptitle("Pivot Convergence — Does More Sampling Help?", fontsize=13, fontweight="bold", x=0.04, ha="left")

    def _corr_phrase(metric_label: str, value: float | None) -> str | None:
        if value is None:
            return None
        arrow = "↘" if value < 0 else "↗"
        return f"corr(used, {metric_label}) = {value:+.2f} {arrow}"

    distance_summary = dict(metrics.get("distance_to_consensus_summary_mm", {}) or {})
    rmse_summary = dict(metrics.get("rmse_summary_mm", {}) or {})
    fig.text(
        0.015, 0.02,
        "  •  ".join(
            _compact_metric_lines(
                [
                    f"Runs: {len(runs)}",
                    f"Samples (min/med/max): {min(used)} / {int(np.median(used))} / {max(used)}",
                    f"Mean dist: {_fmt(distance_summary.get('mean'))} mm",
                    f"Mean RMSE: {_fmt(rmse_summary.get('mean'))} mm",
                    _corr_phrase("dist", correlations["distance"]),
                    _corr_phrase("rmse", correlations["rmse"]),
                ]
            )
        ),
        fontsize=9, color=color("text"), ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": color("grid"), "alpha": 0.94},
    )
    save_figure(fig, path)


def _registration_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in list(metrics.get("per_run_rows") or [])]


def _write_registration_debug_json(
    *,
    path: Path,
    output_dir: Path,
    metadata,
    summary,
    metrics: dict[str, Any],
) -> None:
    """Consolidate registration diagnostics into a single JSON file.

    Outlier detection (per-run origin_distance_to_consensus_mm beyond mean+2σ),
    full per-run translation + rotation deltas (which don't fit on the main
    thesis figures), any skipped source artifacts, and the run_review sidecar
    payload if present.
    """
    rows = _registration_rows(metrics)
    origin_distance_summary = dict(metrics.get("origin_distance_to_consensus_summary_mm", {}) or {})
    mean_dist = _optional_float(origin_distance_summary.get("mean")) or 0.0
    std_dist = _optional_float(origin_distance_summary.get("std")) or 0.0
    outlier_threshold_mm = mean_dist + 2.0 * std_dist

    outliers: list[dict[str, Any]] = []
    for row in rows:
        distance = _optional_float(row.get("origin_distance_to_consensus_mm"))
        if distance is None or std_dist <= 0.0:
            continue
        if distance > outlier_threshold_mm:
            outliers.append(
                {
                    "label": row.get("label"),
                    "path": row.get("path"),
                    "origin_distance_to_consensus_mm": distance,
                    "fre_mm": _optional_float(row.get("fre_mm")),
                    "translation_delta_to_consensus_mm": _optional_float(row.get("translation_delta_to_consensus_mm")),
                    "rotation_delta_to_consensus_deg": _optional_float(row.get("rotation_delta_to_consensus_deg")),
                    "z_score": (distance - mean_dist) / std_dist if std_dist > 0 else None,
                }
            )

    per_run_deltas = [
        {
            "label": row.get("label"),
            "fre_mm": _optional_float(row.get("fre_mm")),
            "translation_delta_to_consensus_mm": _optional_float(row.get("translation_delta_to_consensus_mm")),
            "rotation_delta_to_consensus_deg": _optional_float(row.get("rotation_delta_to_consensus_deg")),
            "origin_distance_to_consensus_mm": _optional_float(row.get("origin_distance_to_consensus_mm")),
            "landmark_count": int(row.get("landmark_count", 0) or 0) or None,
        }
        for row in rows
    ]

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
        "experiment_name": getattr(metadata, "experiment_name", "registration_validation"),
        "status": getattr(summary, "status", None),
        "outlier_detection": {
            "method": "origin_distance_to_consensus_mm > mean + 2*std",
            "mean_distance_mm": mean_dist,
            "std_distance_mm": std_dist,
            "threshold_mm": outlier_threshold_mm,
            "outlier_runs": outliers,
            "outlier_count": len(outliers),
        },
        "per_run_deltas": per_run_deltas,
        "rotation_summary_deg": dict(metrics.get("rotation_delta_to_consensus_summary_deg", {}) or {}),
        "invalid_source_runs": list(metrics.get("invalid_runs") or []),
        "selected_run_count": int(metrics.get("selected_run_count", 0) or 0),
        "valid_run_count": int(metrics.get("valid_run_count", 0) or 0),
        "run_review": review_payload,
    }
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _write_registration_thesis_01_robot_origins_3d(*, path: Path, metrics: dict[str, Any]) -> None:
    """Thesis figure 1: 3D scatter of solved robot-frame origins in Aurora space.

    Each point is one registration solve; color encodes that run's FRE.
    A consensus X marks the mean robot-origin, and a wireframe sphere at
    radius = max per-run distance to consensus visually answers
    "how consistently is the robot frame placed across re-solves?"
    """
    rows = _registration_rows(metrics)
    points: list[tuple[float, float, float]] = []
    fre_for_color: list[float] = []
    labels: list[str] = []
    for index, row in enumerate(rows):
        origin = row.get("robot_origin_in_aurora_mm")
        if not (isinstance(origin, list) and len(origin) >= 3):
            continue
        try:
            point = (float(origin[0]), float(origin[1]), float(origin[2]))
        except (TypeError, ValueError):
            continue
        points.append(point)
        fre_for_color.append(float(row.get("fre_mm") or 0.0))
        labels.append(f"R{index + 1}")

    fig, ax = create_3d_figure(size="thesis_3d")
    if not points:
        ax.text2D(0.5, 0.5, "No valid solved frame origins available",
                  transform=ax.transAxes, ha="center", va="center")
        style_3d_axes(ax, title="")
        fig.suptitle("Registration Robot-Origin Consistency (Aurora Frame)",
                     fontsize=13, fontweight="bold", x=0.04, ha="left")
        save_figure(fig, path)
        return

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]

    consensus = metrics.get("consensus_robot_origin_in_aurora_mm")
    consensus_xyz: tuple[float, float, float] | None = None
    if isinstance(consensus, list) and len(consensus) >= 3:
        try:
            consensus_xyz = (float(consensus[0]), float(consensus[1]), float(consensus[2]))
        except (TypeError, ValueError):
            consensus_xyz = None

    # Wireframe spread reference: 3 great circles at radius = max distance to consensus.
    if consensus_xyz is not None and len(points) >= 2:
        cx, cy, cz = consensus_xyz
        radius = max(
            float(np.linalg.norm(np.asarray(point, dtype=float) - np.asarray(consensus_xyz, dtype=float)))
            for point in points
        )
        if radius > 0.0:
            theta = np.linspace(0.0, 2.0 * np.pi, 96)
            circle_kwargs = {"color": color("neutral"), "alpha": 0.32, "linewidth": 0.9, "linestyle": "--"}
            ax.plot(cx + radius * np.cos(theta), cy + radius * np.sin(theta), np.full_like(theta, cz), **circle_kwargs)
            ax.plot(cx + radius * np.cos(theta), np.full_like(theta, cy), cz + radius * np.sin(theta), **circle_kwargs)
            ax.plot(np.full_like(theta, cx), cy + radius * np.cos(theta), cz + radius * np.sin(theta),
                    label=f"Max-spread sphere ({radius:.3f} mm)", **circle_kwargs)

    scatter = ax.scatter(
        xs, ys, zs,
        c=fre_for_color, cmap="viridis",
        s=70, depthshade=True, edgecolors="white", linewidths=0.6,
    )

    if consensus_xyz is not None:
        cx, cy, cz = consensus_xyz
        for px, py, pz in points:
            ax.plot([cx, px], [cy, py], [cz, pz], color=color("neutral"), linewidth=0.9, alpha=0.55)
        ax.scatter(
            [cx], [cy], [cz],
            s=200, marker="X",
            color=color("reference"), edgecolors="white", linewidths=1.2,
            depthshade=False, label="Consensus",
        )

    if len(points) <= 12:
        for (px, py, pz), label_text in zip(points, labels):
            ax.text(px, py, pz, f"  {label_text}", fontsize=8, color=color("text"), zorder=10)

    pad_xs, pad_ys, pad_zs = list(xs), list(ys), list(zs)
    if consensus_xyz is not None:
        pad_xs.append(consensus_xyz[0])
        pad_ys.append(consensus_xyz[1])
        pad_zs.append(consensus_xyz[2])
    set_equal_xyz(ax, x_values=pad_xs, y_values=pad_ys, z_values=pad_zs, minimum_span=1.0, pad_fraction=0.20)
    style_3d_axes(ax, xlabel="Aurora X (mm)", ylabel="Aurora Y (mm)", zlabel="Aurora Z (mm)")

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.55, pad=0.12)
    cbar.set_label("Per-run FRE (mm)")
    cbar.outline.set_edgecolor(color("grid"))

    legend(ax, loc="upper left")

    fig.suptitle("Registration Robot-Origin Consistency (Aurora Frame)",
                 fontsize=13, fontweight="bold", x=0.04, ha="left")

    fre_summary = dict(metrics.get("fre_summary_mm", {}) or {})
    distance_summary = dict(metrics.get("origin_distance_to_consensus_summary_mm", {}) or {})
    spread = dict(metrics.get("robot_origin_spread_mm", {}) or {})
    fig.text(
        0.015, 0.02,
        "  •  ".join(
            _compact_metric_lines(
                [
                    f"Runs: {int(metrics.get('valid_run_count', len(points)) or 0)}",
                    f"Mean FRE: {_fmt(fre_summary.get('mean'))} mm",
                    f"Mean origin spread: {_fmt(distance_summary.get('mean') or spread.get('mean_distance_mm'))} mm",
                    f"Max origin spread: {_fmt(distance_summary.get('max') or spread.get('max_distance_mm'))} mm",
                ]
            )
        ),
        fontsize=9, color=color("text"), ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": color("grid"), "alpha": 0.94},
    )
    save_figure(fig, path)


def _write_registration_thesis_02_within_vs_cross_run_quality(*, path: Path, metrics: dict[str, Any]) -> None:
    """Thesis figure 2: is within-run FRE a useful proxy for cross-run consistency?

    2D scatter answering: does a low FRE solve also land close to the consensus?
      - X axis: per-run FRE (mm) — solver's reported within-run residual
      - Y axis: per-run origin distance to consensus (mm) — cross-run consistency
      - Marker color: per-run rotation delta to consensus (deg) — rotation-quality face
    Dashed regression line + correlation coefficient quantify the answer. 2D
    because all three axes here are quality metrics in three different unit
    families (mm/mm/deg); forcing them into XYZ would mislead by implying a
    cubic space.
    """
    rows = _registration_rows(metrics)
    fres: list[float] = []
    distances: list[float] = []
    rotations: list[float] = []
    labels: list[str] = []
    for index, row in enumerate(rows):
        fre = _optional_float(row.get("fre_mm"))
        dist = _optional_float(row.get("origin_distance_to_consensus_mm"))
        rot = _optional_float(row.get("rotation_delta_to_consensus_deg"))
        if fre is None or dist is None:
            continue
        fres.append(fre)
        distances.append(dist)
        rotations.append(rot if rot is not None else 0.0)
        labels.append(f"R{index + 1}")

    # constrained_layout=False so we can reserve bottom space for the metric strip.
    fig, ax = create_figure(size="wide", constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.92, top=0.90, bottom=0.22)
    if not fres:
        ax.text(0.5, 0.5, "No per-run FRE / consensus-distance data available",
                transform=ax.transAxes, ha="center", va="center")
        style_axes(ax, xlabel="Per-run FRE (mm)", ylabel="Origin distance to consensus (mm)")
        fig.suptitle("Registration Within-Run vs Cross-Run Quality",
                     fontsize=13, fontweight="bold", x=0.04, ha="left")
        save_figure(fig, path)
        return

    scatter = ax.scatter(
        fres, distances,
        c=rotations, cmap="viridis",
        s=110, edgecolors="white", linewidths=0.8, zorder=3,
    )

    # Regression line + correlation (only when X actually varies).
    correlation: float | None = None
    if len(set(round(v, 9) for v in fres)) > 1 and len(set(round(v, 9) for v in distances)) > 1:
        try:
            slope, intercept = np.polyfit(fres, distances, 1)
            x_min, x_max = float(min(fres)), float(max(fres))
            x_pad = max((x_max - x_min) * 0.08, 0.01)
            x_line = np.linspace(x_min - x_pad, x_max + x_pad, 64)
            ax.plot(
                x_line, slope * x_line + intercept,
                color=color("neutral"), linewidth=1.4, linestyle="--", alpha=0.6, zorder=2,
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = float(np.corrcoef(fres, distances)[0, 1])
            if np.isfinite(corr):
                correlation = corr
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            pass

    # Per-run labels above each marker.
    for x_val, y_val, label_text in zip(fres, distances, labels):
        ax.annotate(
            label_text,
            xy=(x_val, y_val),
            xytext=(0, 9),
            textcoords="offset points",
            fontsize=9, fontweight="bold",
            color=color("text"), ha="center",
        )

    # Y range with headroom for labels and a 0 floor.
    y_max = max(distances)
    ax.set_ylim(0.0, y_max * 1.25 + 0.02)
    x_max = max(fres)
    x_min = min(fres)
    x_pad = max((x_max - x_min) * 0.12, 0.02)
    ax.set_xlim(max(0.0, x_min - x_pad), x_max + x_pad)

    style_axes(ax, xlabel="Per-run FRE (mm)", ylabel="Origin distance to consensus (mm)")
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Rotation delta (deg)")
    cbar.outline.set_edgecolor(color("grid"))

    fig.suptitle("Registration Within-Run vs Cross-Run Quality",
                 fontsize=13, fontweight="bold", x=0.04, ha="left")

    def _corr_phrase(value: float | None) -> str | None:
        if value is None:
            return None
        arrow = "↗" if value >= 0 else "↘"
        sense = "FRE predicts spread" if value > 0.4 else (
            "FRE inversely tracks spread" if value < -0.4 else "weak / no relationship"
        )
        return f"corr(FRE, spread) = {value:+.2f} {arrow}  ({sense})"

    fre_summary = dict(metrics.get("fre_summary_mm", {}) or {})
    distance_summary = dict(metrics.get("origin_distance_to_consensus_summary_mm", {}) or {})
    rotation_summary = dict(metrics.get("rotation_delta_to_consensus_summary_deg", {}) or {})
    fig.text(
        0.015, 0.02,
        "  •  ".join(
            _compact_metric_lines(
                [
                    f"Runs: {len(fres)}",
                    f"Mean FRE: {_fmt(fre_summary.get('mean'))} mm",
                    f"Mean spread: {_fmt(distance_summary.get('mean'))} mm",
                    f"Mean rotation delta: {_fmt(rotation_summary.get('mean'))} deg",
                    _corr_phrase(correlation),
                ]
            )
        ),
        fontsize=9, color=color("text"), ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": color("grid"), "alpha": 0.94},
    )
    save_figure(fig, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Unserialisable type: {type(value).__name__}")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compact_metric_lines(lines: list[str | None]) -> list[str]:
    return [str(line) for line in lines if line]


def _draw_projected_scatter(
    painter: QPainter,
    rect: QRectF,
    *,
    points: list[np.ndarray],
    point_color: QColor,
    point_radius: float,
    empty_text: str,
    axis_note: str,
) -> None:
    if not points:
        painter.setFont(_body_font())
        painter.setPen(QColor("#64748b"))
        painter.drawText(rect, Qt.AlignCenter, empty_text)
        return
    projected = np.asarray([_project_point(point) for point in points], dtype=float)
    minimum = projected.min(axis=0)
    maximum = projected.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-6)
    shared_span = float(max(span[0], span[1], 1e-6))
    painter.setPen(QPen(QColor("#cbd5e1"), 1.0))
    painter.drawRect(rect)
    for point in projected:
        x_norm = float((point[0] - minimum[0]) / shared_span)
        y_norm = float((point[1] - minimum[1]) / shared_span)
        x = rect.left() + (x_norm * rect.width())
        y = rect.bottom() - (y_norm * rect.height())
        painter.setPen(Qt.NoPen)
        painter.setBrush(point_color)
        painter.drawEllipse(QRectF(x - point_radius, y - point_radius, point_radius * 2.0, point_radius * 2.0))
    painter.setPen(QColor("#475569"))
    painter.setFont(_body_font())
    painter.drawText(rect.adjusted(8.0, 8.0, -8.0, -8.0), Qt.AlignLeft | Qt.AlignTop, axis_note)


def _new_publication_image(width: int, height: int):
    image = _new_image(width, height)
    dots_per_meter = int(round(float(figure_dpi()) / 0.0254))
    image.setDotsPerMeterX(dots_per_meter)
    image.setDotsPerMeterY(dots_per_meter)
    return image


def _project_point(point: np.ndarray) -> tuple[float, float]:
    x, y, z = [float(value) for value in np.asarray(point, dtype=float).reshape(3)]
    return (x - (0.55 * y), z + (0.35 * y))


def _metric_triplet(summary: Any) -> str:
    summary = dict(summary or {})
    return f"{_fmt(summary.get('mean'))} / {_fmt(summary.get('std'))} / {_fmt(summary.get('max'))}"


def _metric_stat(summary: Any, key: str) -> str:
    summary = dict(summary or {})
    return _fmt(summary.get(key))


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
