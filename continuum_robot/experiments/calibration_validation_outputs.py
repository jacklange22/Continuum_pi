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
    """Write canonical output artifacts for registration validation."""
    metrics = summary.experiment_metrics if isinstance(summary.experiment_metrics, dict) else {}
    output_dir = Path(output_dir)
    csv_path = output_dir / "metrics.csv"
    text_path = output_dir / "registration_validation_summary.txt"
    origin_report_path = output_dir / "registration_frame_origins_report.png"
    fre_report_path = output_dir / "registration_fre_report.png"
    spread_report_path = output_dir / "registration_transform_spread_report.png"
    fre_plot_path = output_dir / "registration_fre_histogram.png"
    origin_plot_path = output_dir / "registration_frame_origins.png"
    spread_plot_path = output_dir / "registration_transform_spread.png"
    _write_rows_csv(
        csv_path,
        rows=metrics.get("per_run_rows") or [],
        header_overrides={
            "path": "path",
            "label": "label",
            "timestamp_utc": "timestamp_utc",
            "measurement_tool_id": "measurement_tool_id",
            "coil_tool_id": "coil_tool_id",
            "fre_mm": "fre_mm",
            "landmark_count": "landmark_count",
            "robot_origin_in_aurora_mm": "robot_origin_in_aurora_mm",
            "translation_delta_to_consensus_mm": "translation_delta_to_consensus_mm",
            "rotation_delta_to_consensus_deg": "rotation_delta_to_consensus_deg",
            "origin_distance_to_consensus_mm": "origin_distance_to_consensus_mm",
            "run_index": "run_index",
        },
    )
    text_path.write_text(
        "\n".join(build_registration_validation_summary_lines(metadata=metadata, summary=summary, metrics=metrics)).strip() + "\n",
        encoding="utf-8",
    )
    for path, writer in [
        (origin_report_path, lambda: _write_registration_origin_report(path=origin_report_path, metrics=metrics)),
        (fre_report_path, lambda: _write_registration_fre_report(path=fre_report_path, metrics=metrics)),
        (spread_report_path, lambda: _write_registration_transform_spread_report(path=spread_report_path, metrics=metrics)),
    ]:
        try:
            writer()
        except Exception:
            _write_plot_placeholder(path)
    if _qt_plotting_is_safe():
        _ensure_plot_qt_app()
        _write_registration_fre_histogram(fre_plot_path=fre_plot_path, metrics=metrics)
        _write_registration_origin_plot(origin_plot_path=origin_plot_path, metrics=metrics)
        _write_registration_spread_plot(spread_plot_path=spread_plot_path, metrics=metrics)
    else:
        _write_plot_placeholder(fre_plot_path)
        _write_plot_placeholder(origin_plot_path)
        _write_plot_placeholder(spread_plot_path)
    return {
        "metrics_csv_path": csv_path,
        "summary_text_path": text_path,
        "origin_report_path": origin_report_path,
        "fre_report_path": fre_report_path,
        "spread_report_path": spread_report_path,
        "fre_plot_path": fre_plot_path,
        "origin_plot_path": origin_plot_path,
        "spread_plot_path": spread_plot_path,
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


def build_registration_validation_summary_lines(*, metadata, summary, metrics: dict[str, Any]) -> list[str]:
    """Return human-readable registration-validation summary text."""
    spread = dict(metrics.get("robot_origin_spread_mm", {}) or {})
    lines = [
        "Registration Validation Summary",
        "Offline analysis of repeated saved registration solves. Metrics quantify FRE spread, transform spread, and where the downstream robot/model frame origin lands in Aurora space (all distances in mm, rotation in degrees).",
        "",
        f"Run ID: {metadata.run_id}",
        f"Timestamp: {metadata.timestamp_utc}",
        f"Status: {summary.status}",
        f"Selected runs: {int(metrics.get('selected_run_count', 0) or 0)}",
        f"Valid runs: {int(metrics.get('valid_run_count', 0) or 0)}",
        f"Skipped runs: {int(metrics.get('invalid_run_count', 0) or 0)}",
        "",
        "Definitions:",
        "- FRE is residual error on the fiducials used by each registration solve (not an external ground-truth held-out error).",
        "- Translation/rotation spread values are transform deltas relative to a consensus transform across selected runs.",
        "",
        f"FRE mean / std / max (mm): {_metric_triplet(metrics.get('fre_summary_mm'))}",
        f"Translation spread mean / std / max (mm): {_metric_triplet(metrics.get('translation_delta_to_consensus_summary_mm'))}",
        f"Rotation spread mean / std / max (deg): {_metric_triplet(metrics.get('rotation_delta_to_consensus_summary_deg'))}",
        f"Robot-origin RMS / max distance to consensus (mm): {_fmt(spread.get('rms_distance_mm'))} / {_fmt(spread.get('max_distance_mm'))}",
        f"Robot-origin XYZ span (mm): x={_fmt(spread.get('span_x_mm'))}, y={_fmt(spread.get('span_y_mm'))}, z={_fmt(spread.get('span_z_mm'))}",
        "",
        "Saved plots:",
        "- registration_frame_origins_report.png",
        "- registration_fre_report.png",
        "- registration_transform_spread_report.png",
        "- registration_fre_histogram.png",
        "- registration_frame_origins.png",
        "- registration_transform_spread.png",
    ]
    invalid_runs = metrics.get("invalid_runs") or []
    if invalid_runs:
        lines.extend(["", "Skipped source artifacts:"])
        for row in invalid_runs:
            lines.append(f"- {row.get('path')}: {row.get('reason')}")
    return lines


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


def _write_registration_origin_report(*, path: Path, metrics: dict[str, Any]) -> None:
    rows = _registration_rows(metrics)
    origins = [
        [float(value) for value in row.get("robot_origin_in_aurora_mm")]
        for row in rows
        if isinstance(row.get("robot_origin_in_aurora_mm"), list) and len(row.get("robot_origin_in_aurora_mm")) >= 3
    ]
    with report_style() as plt:
        fig, ax = plt.subplots(figsize=(5.4, 5.0), constrained_layout=True)
    if not origins:
        ax.text(0.5, 0.5, "No valid solved frame origins available", transform=ax.transAxes, ha="center", va="center")
    else:
        xs = [point[0] for point in origins]
        ys = [point[1] for point in origins]
        ax.scatter(xs, ys, s=34, color=color("measured"), alpha=0.78, linewidths=0, label="Solved origins")
        consensus = metrics.get("consensus_robot_origin_in_aurora_mm")
        if isinstance(consensus, list) and len(consensus) >= 2:
            cx = float(consensus[0])
            cy = float(consensus[1])
            for x_value, y_value in zip(xs, ys):
                ax.plot([cx, x_value], [cy, y_value], color=color("neutral"), linewidth=0.7, alpha=0.35)
            ax.scatter(
                [cx],
                [cy],
                s=70,
                marker="X",
                color=color("reference"),
                edgecolors="white",
                linewidths=0.8,
                zorder=4,
                label="Consensus origin",
            )
            xs.append(cx)
            ys.append(cy)
        set_equal_xy(ax, x_values=xs, y_values=ys, minimum_span=1.0)
    style_axes(
        ax,
        title="Registration Frame Origin Consistency",
        xlabel="Aurora X (mm)",
        ylabel="Aurora Y (mm)",
    )
    legend(ax, loc="best")
    spread = dict(metrics.get("robot_origin_spread_mm", {}) or {})
    fre_summary = dict(metrics.get("fre_summary_mm", {}) or {})
    add_metric_box(
        ax,
        _compact_metric_lines(
            [
                f"Mean FRE: {_fmt(fre_summary.get('mean'))} mm",
                f"RMS origin spread: {_fmt(spread.get('rms_distance_mm'))} mm",
                f"Max origin spread: {_fmt(spread.get('max_distance_mm'))} mm",
                f"Runs: {int(metrics.get('valid_run_count', len(origins)) or 0)}",
            ]
        ),
        loc="upper right",
    )
    save_figure(fig, path)


def _write_registration_fre_report(*, path: Path, metrics: dict[str, Any]) -> None:
    values = [
        float(row.get("fre_mm"))
        for row in _registration_rows(metrics)
        if row.get("fre_mm") is not None
    ]
    with report_style() as plt:
        fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    if not values:
        ax.text(0.5, 0.5, "No valid FRE values available", transform=ax.transAxes, ha="center", va="center")
    else:
        bins = max(3, min(10, int(np.ceil(np.sqrt(len(values))))))
        ax.hist(values, bins=bins, color=color("measured"), edgecolor="white", linewidth=0.8, alpha=0.88)
        mean_fre = _optional_float((metrics.get("fre_summary_mm") or {}).get("mean"))
        if mean_fre is not None:
            ax.axvline(
                mean_fre,
                color=color("reference"),
                linestyle="-",
                linewidth=1.4,
                label=f"Mean FRE ({mean_fre:.3f} mm)",
            )
            legend(ax, loc="upper right")
    style_axes(
        ax,
        title="Registration FRE Distribution",
        xlabel="FRE (mm)",
        ylabel="Run count",
    )
    save_figure(fig, path)


def _write_registration_transform_spread_report(*, path: Path, metrics: dict[str, Any]) -> None:
    rows = _registration_rows(metrics)
    labels = [f"R{index + 1}" for index, _row in enumerate(rows)]
    translation_values = [
        float(row.get("translation_delta_to_consensus_mm", 0.0) or 0.0)
        for row in rows
    ]
    rotation_values = [
        float(row.get("rotation_delta_to_consensus_deg", 0.0) or 0.0)
        for row in rows
    ]
    with report_style() as plt:
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.8), constrained_layout=True, sharex=True)
    top_ax, bottom_ax = axes
    if not rows:
        top_ax.text(0.5, 0.5, "No valid transform spread values available", transform=top_ax.transAxes, ha="center", va="center")
        bottom_ax.set_visible(False)
    else:
        x_positions = list(range(len(rows)))
        top_bars = top_ax.bar(
            x_positions,
            translation_values,
            color=color("measured"),
            edgecolor="white",
            linewidth=0.7,
        )
        bottom_bars = bottom_ax.bar(
            x_positions,
            rotation_values,
            color=color("fit"),
            edgecolor="white",
            linewidth=0.7,
        )
        if len(rows) <= 12:
            top_ax.bar_label(top_bars, labels=[f"{value:.2f}" for value in translation_values], padding=2, fontsize=8)
            bottom_ax.bar_label(bottom_bars, labels=[f"{value:.2f}" for value in rotation_values], padding=2, fontsize=8)
        top_ax.set_ylim(0.0, max(translation_values + [0.0]) * 1.18 + 0.02)
        bottom_ax.set_ylim(0.0, max(rotation_values + [0.0]) * 1.18 + 0.02)
        bottom_ax.set_xticks(x_positions)
        bottom_ax.set_xticklabels(labels)
    style_axes(
        top_ax,
        title="Registration Transform Spread",
        xlabel="",
        ylabel="Translation delta (mm)",
    )
    style_axes(
        bottom_ax,
        title="",
        xlabel="Registration run",
        ylabel="Rotation delta (deg)",
    )
    save_figure(fig, path)


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


def _registration_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in list(metrics.get("per_run_rows") or [])]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compact_metric_lines(lines: list[str | None]) -> list[str]:
    return [str(line) for line in lines if line]


def _write_registration_fre_histogram(*, fre_plot_path: Path, metrics: dict[str, Any]) -> None:
    image = _new_publication_image(1180, 720)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_title(
        painter,
        QRectF(34.0, 24.0, 1112.0, 56.0),
        "REGISTRATION FRE DISTRIBUTION (MM)",
        "Histogram of per-run fiducial registration error (FRE) across the selected saved solves.",
    )
    chart_rect = QRectF(36.0, 104.0, 740.0, 560.0)
    summary_rect = QRectF(808.0, 104.0, 338.0, 560.0)
    _draw_panel(painter, chart_rect, "FRE Histogram")
    _draw_panel(painter, summary_rect, "Summary")
    values = [float(row.get("fre_mm")) for row in (metrics.get("per_run_rows") or []) if row.get("fre_mm") is not None]
    _draw_histogram(
        painter,
        chart_rect.adjusted(18.0, 38.0, -18.0, -28.0),
        values=values,
        color=QColor("#2563eb"),
        x_label="FRE (mm)",
        y_label="Count",
        empty_text="No valid FRE values were available.",
    )
    _draw_summary_pairs(
        painter,
        summary_rect.adjusted(16.0, 36.0, -16.0, -16.0),
        [
            ("Valid Runs", str(int(metrics.get("valid_run_count", 0) or 0))),
            ("Mean FRE (mm)", _metric_stat(metrics.get("fre_summary_mm"), "mean")),
            ("Std FRE (mm)", _metric_stat(metrics.get("fre_summary_mm"), "std")),
            ("Max FRE (mm)", _metric_stat(metrics.get("fre_summary_mm"), "max")),
            ("Mean dT (mm)", _metric_stat(metrics.get("translation_delta_to_consensus_summary_mm"), "mean")),
            ("Mean dR (deg)", _metric_stat(metrics.get("rotation_delta_to_consensus_summary_deg"), "mean")),
        ],
    )
    painter.end()
    image.save(str(fre_plot_path))


def _write_registration_origin_plot(*, origin_plot_path: Path, metrics: dict[str, Any]) -> None:
    image = _new_publication_image(1180, 760)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_title(
        painter,
        QRectF(34.0, 24.0, 1112.0, 56.0),
        "REGISTRATION FRAME ORIGINS (MM)",
        "Projected 3D scatter of solved robot/model-frame origins expressed in Aurora space with equal projected scaling.",
    )
    scatter_rect = QRectF(36.0, 104.0, 820.0, 600.0)
    summary_rect = QRectF(888.0, 104.0, 258.0, 600.0)
    _draw_panel(painter, scatter_rect, "Solved Frame Origins (mm)")
    _draw_panel(painter, summary_rect, "Spread")
    points = [
        np.asarray(row.get("robot_origin_in_aurora_mm") or [], dtype=float)
        for row in (metrics.get("per_run_rows") or [])
        if isinstance(row.get("robot_origin_in_aurora_mm"), list) and len(row.get("robot_origin_in_aurora_mm")) == 3
    ]
    _draw_projected_scatter(
        painter,
        scatter_rect.adjusted(18.0, 38.0, -18.0, -28.0),
        points=points,
        point_color=QColor("#0f766e"),
        point_radius=5.0,
        empty_text="No valid registration origins were available.",
        axis_note="Projected axes in mm (equal projected scale)",
    )
    spread = dict(metrics.get("robot_origin_spread_mm", {}) or {})
    _draw_summary_pairs(
        painter,
        summary_rect.adjusted(16.0, 36.0, -16.0, -16.0),
        [
            ("Std X (mm)", _fmt(spread.get("std_x_mm"))),
            ("Std Y (mm)", _fmt(spread.get("std_y_mm"))),
            ("Std Z (mm)", _fmt(spread.get("std_z_mm"))),
            ("RMS Dist (mm)", _fmt(spread.get("rms_distance_mm"))),
            ("Max Dist (mm)", _fmt(spread.get("max_distance_mm"))),
            ("Span X (mm)", _fmt(spread.get("span_x_mm"))),
            ("Span Y (mm)", _fmt(spread.get("span_y_mm"))),
            ("Span Z (mm)", _fmt(spread.get("span_z_mm"))),
        ],
    )
    painter.end()
    image.save(str(origin_plot_path))


def _write_registration_spread_plot(*, spread_plot_path: Path, metrics: dict[str, Any]) -> None:
    rows = list(metrics.get("per_run_rows") or [])
    labels = [f"R{index + 1}" for index in range(len(rows))]
    translation_values = [float(row.get("translation_delta_to_consensus_mm", 0.0) or 0.0) for row in rows]
    rotation_values = [float(row.get("rotation_delta_to_consensus_deg", 0.0) or 0.0) for row in rows]
    image = _new_publication_image(1180, 760)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_title(
        painter,
        QRectF(34.0, 24.0, 1112.0, 56.0),
        "REGISTRATION TRANSFORM SPREAD (MM / DEG)",
        "Per-run translation and rotation deltas relative to the consensus transform.",
    )
    top_rect = QRectF(36.0, 104.0, 1108.0, 280.0)
    bottom_rect = QRectF(36.0, 414.0, 1108.0, 280.0)
    _draw_panel(painter, top_rect, "Translation Delta To Consensus (mm)")
    _draw_panel(painter, bottom_rect, "Rotation Delta To Consensus (deg)")
    _draw_bar_chart(
        painter,
        top_rect.adjusted(18.0, 38.0, -18.0, -28.0),
        labels=labels,
        values=translation_values,
        color=QColor("#2563eb"),
        y_label="mm",
        empty_text="No valid runs were available.",
    )
    _draw_bar_chart(
        painter,
        bottom_rect.adjusted(18.0, 38.0, -18.0, -28.0),
        labels=labels,
        values=rotation_values,
        color=QColor("#dc2626"),
        y_label="deg",
        empty_text="No valid runs were available.",
    )
    painter.end()
    image.save(str(spread_plot_path))


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
