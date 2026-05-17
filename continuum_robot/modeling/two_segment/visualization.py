"""Comparison visualizations for the hybrid-residual two-segment modeling path.

Every figure in this module is rendered to a PNG file (matplotlib Agg backend)
and never opens a GUI -- safe to call from headless runs and from tests. All
inputs are plain numpy arrays and label metadata; no project-specific dataclass
is required, which keeps the module trivially reusable.

The figures are designed for thesis-grade reporting:

* ``write_hybrid_before_after_scatter`` -- per-axis Measured vs Predicted
  scatter for Mike CC, ANN, and Hybrid on the same axes. Diagonal reference
  line plus per-model legends with axis-wise RMSE.
* ``write_hybrid_residual_histograms`` -- per-axis residual distributions for
  Mike CC, ANN, and Hybrid with shared bin edges so the eye can directly
  compare tail width and centering.
* ``write_hybrid_workspace_error_map`` -- planar XY scatter colored by 3D
  distal error, one subplot per model, common color scale.
* ``write_hybrid_convergence_overlay`` -- ANN training loss curve (residual
  target) with the Mike-only RMSE drawn as a horizontal baseline so it's
  obvious when learning crosses the geometric baseline.
* ``write_hybrid_improvement_bars`` -- grouped bar chart of headline metrics
  (xyz_rmse, p95, max) for Mike vs ANN vs Hybrid.

``write_hybrid_visualization_bundle`` runs every applicable figure and returns
the dict of paths written, skipping cleanly when an input slice is missing.
"""
from __future__ import annotations

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


# Pinned model-to-color map; mirrors the registration plot palette so the
# thesis figures stay consistent without callers having to thread colors in.
MODEL_COLORS: dict[str, str] = {
    "mike_constant_curvature": SEMANTIC_COLORS["model"],
    "mike": SEMANTIC_COLORS["model"],
    "ann": SEMANTIC_COLORS["prediction"],
    "hybrid_residual": SEMANTIC_COLORS["fit"],
    "hybrid": SEMANTIC_COLORS["fit"],
    "linear_baseline": SEMANTIC_COLORS["neutral"],
}

MODEL_LABELS: dict[str, str] = {
    "mike_constant_curvature": "Mike CC",
    "mike": "Mike CC",
    "ann": "ANN",
    "hybrid_residual": "Hybrid (Mike + ANN residual)",
    "hybrid": "Hybrid (Mike + ANN residual)",
    "linear_baseline": "Linear baseline",
}


def _distal_slice(label_metadata: Mapping[str, Any] | None) -> list[int]:
    """Return the [x, y, z] column indices for the distal-tip position label."""
    slices = (
        label_metadata.get("label_slices")
        if isinstance(label_metadata, Mapping) and isinstance(label_metadata.get("label_slices"), Mapping)
        else {}
    )
    role = slices.get("distal_tip") if isinstance(slices, Mapping) else None
    indices = role.get("position") if isinstance(role, Mapping) else None
    if isinstance(indices, list) and len(indices) == 3:
        return [int(value) for value in indices]
    return [0, 1, 2]


def _rmse(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values))))


def _normalize_predictions_dict(
    predictions_by_model: Mapping[str, np.ndarray | None],
) -> dict[str, np.ndarray]:
    """Filter out None or empty arrays; keeps insertion order from caller."""
    cleaned: dict[str, np.ndarray] = {}
    for key, value in predictions_by_model.items():
        if value is None:
            continue
        arr = np.asarray(value, dtype=float)
        if arr.size == 0:
            continue
        cleaned[str(key)] = arr
    return cleaned


def write_hybrid_before_after_scatter(
    *,
    output_path: Path,
    y_true: np.ndarray,
    predictions_by_model: Mapping[str, np.ndarray | None],
    label_metadata: Mapping[str, Any] | None = None,
    quality: str | None = None,
) -> Path:
    """Per-axis Measured vs Predicted scatter overlay (Mike / ANN / Hybrid).

    Three subplots (one per axis). For each axis, every available model is
    overlaid as a scatter against the same measured values; the unit-slope
    reference line shows ideal prediction. The legend reports per-axis RMSE
    so the relative ordering of models is unambiguous from the figure alone.
    """
    cleaned = _normalize_predictions_dict(predictions_by_model)
    if not cleaned:
        raise ValueError("write_hybrid_before_after_scatter requires at least one model prediction array.")
    distal = _distal_slice(label_metadata)
    truth = np.asarray(y_true, dtype=float)[:, distal]
    axis_names = ["X (mm)", "Y (mm)", "Z (mm)"]

    with report_style() as plt:
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6), constrained_layout=True)
        for axis_index, axis_label in enumerate(axis_names):
            ax = axes[axis_index]
            measured_axis = truth[:, axis_index]
            for model_key, prediction in cleaned.items():
                pred_arr = np.asarray(prediction, dtype=float)[:, distal][:, axis_index]
                rmse = _rmse(pred_arr - measured_axis)
                ax.scatter(
                    measured_axis,
                    pred_arr,
                    s=22,
                    alpha=0.7,
                    color=MODEL_COLORS.get(model_key, SEMANTIC_COLORS["prediction"]),
                    label=f"{MODEL_LABELS.get(model_key, model_key)} (RMSE={rmse:.2f} mm)",
                    edgecolors="white",
                    linewidths=0.4,
                )
            lo = float(min(measured_axis.min(), *(np.asarray(p, dtype=float)[:, distal][:, axis_index].min() for p in cleaned.values())))
            hi = float(max(measured_axis.max(), *(np.asarray(p, dtype=float)[:, distal][:, axis_index].max() for p in cleaned.values())))
            span = max(hi - lo, 1e-3)
            pad = 0.06 * span
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=SEMANTIC_COLORS["reference"], linewidth=1.0, linestyle="--", label="ideal")
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_aspect("equal", adjustable="box")
            style_axes(ax, title=f"Distal {axis_label}", xlabel="measured (mm)", ylabel="predicted (mm)")
            ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1", framealpha=0.95, fontsize=8)
        fig.suptitle("Hybrid vs Mike vs ANN — Distal-Tip Measured vs Predicted", fontsize=13, fontweight="bold")
    return save_figure(fig, Path(output_path), quality=quality)


def write_hybrid_residual_histograms(
    *,
    output_path: Path,
    y_true: np.ndarray,
    predictions_by_model: Mapping[str, np.ndarray | None],
    label_metadata: Mapping[str, Any] | None = None,
    bins: int = 24,
    quality: str | None = None,
) -> Path:
    """Per-axis residual histograms (predicted − measured) for each model."""
    cleaned = _normalize_predictions_dict(predictions_by_model)
    if not cleaned:
        raise ValueError("write_hybrid_residual_histograms requires at least one model prediction array.")
    distal = _distal_slice(label_metadata)
    truth = np.asarray(y_true, dtype=float)[:, distal]
    axis_names = ["X (mm)", "Y (mm)", "Z (mm)"]

    # Use shared bin edges per axis so the comparison is visually faithful.
    per_axis_residuals: dict[int, dict[str, np.ndarray]] = {}
    for axis_index in range(3):
        per_axis_residuals[axis_index] = {}
        for model_key, prediction in cleaned.items():
            pred_arr = np.asarray(prediction, dtype=float)[:, distal][:, axis_index]
            per_axis_residuals[axis_index][model_key] = pred_arr - truth[:, axis_index]

    with report_style() as plt:
        fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4), constrained_layout=True)
        for axis_index, axis_label in enumerate(axis_names):
            ax = axes[axis_index]
            stacked = np.concatenate(list(per_axis_residuals[axis_index].values()))
            if stacked.size == 0:
                style_axes(ax, title=f"Residuals {axis_label}", xlabel="residual (mm)", ylabel="count")
                continue
            extent = float(max(np.abs(stacked.min()), np.abs(stacked.max()), 1e-3))
            edges = np.linspace(-extent * 1.05, extent * 1.05, max(8, int(bins) + 1))
            for model_key, residuals in per_axis_residuals[axis_index].items():
                ax.hist(
                    residuals,
                    bins=edges,
                    alpha=0.55,
                    color=MODEL_COLORS.get(model_key, SEMANTIC_COLORS["prediction"]),
                    label=f"{MODEL_LABELS.get(model_key, model_key)} (RMSE={_rmse(residuals):.2f} mm)",
                    edgecolor="white",
                    linewidth=0.6,
                )
            ax.axvline(0.0, color=SEMANTIC_COLORS["reference"], linewidth=1.0, linestyle="--", alpha=0.8)
            style_axes(ax, title=f"Residuals {axis_label}", xlabel="residual (mm)", ylabel="count")
            ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1", framealpha=0.95, fontsize=8)
        fig.suptitle("Residual distribution: Mike vs ANN vs Hybrid", fontsize=13, fontweight="bold")
    return save_figure(fig, Path(output_path), quality=quality)


def write_hybrid_workspace_error_map(
    *,
    output_path: Path,
    y_true: np.ndarray,
    predictions_by_model: Mapping[str, np.ndarray | None],
    label_metadata: Mapping[str, Any] | None = None,
    quality: str | None = None,
) -> Path:
    """One subplot per model: distal targets in XY, colored by 3D error.

    All subplots share the same color scale so the eye can compare error
    magnitudes across models directly. Targets are colored by the Euclidean
    distance ``||predicted - measured||``; samples with zero error draw at
    the colormap minimum.
    """
    cleaned = _normalize_predictions_dict(predictions_by_model)
    if not cleaned:
        raise ValueError("write_hybrid_workspace_error_map requires at least one model prediction array.")
    distal = _distal_slice(label_metadata)
    truth = np.asarray(y_true, dtype=float)[:, distal]

    errors_by_model: dict[str, np.ndarray] = {}
    for model_key, prediction in cleaned.items():
        pred_arr = np.asarray(prediction, dtype=float)[:, distal]
        errors_by_model[model_key] = np.linalg.norm(pred_arr - truth, axis=1)
    max_error = float(max((np.max(errors) if errors.size else 0.0 for errors in errors_by_model.values()), default=0.0))
    color_max = max(max_error, 0.5)

    n_models = len(cleaned)
    with report_style() as plt:
        fig, axes = plt.subplots(1, max(1, n_models), figsize=(max(4.6, 4.6 * n_models), 5.0), constrained_layout=True)
        if n_models == 1:
            axes = [axes]
        scatter_handle = None
        for ax, (model_key, errors) in zip(axes, errors_by_model.items()):
            scatter_handle = ax.scatter(
                truth[:, 0],
                truth[:, 1],
                c=errors,
                cmap="viridis",
                vmin=0.0,
                vmax=color_max,
                s=46,
                edgecolors="white",
                linewidths=0.4,
            )
            style_axes(
                ax,
                title=f"{MODEL_LABELS.get(model_key, model_key)} — distal error",
                xlabel="X measured (mm)",
                ylabel="Y measured (mm)",
            )
            ax.set_aspect("equal", adjustable="box")
        if scatter_handle is not None:
            cbar = fig.colorbar(scatter_handle, ax=axes, location="right", shrink=0.85, pad=0.02)
            cbar.set_label("‖predicted − measured‖ (mm)")
        fig.suptitle("Per-target distal error across models", fontsize=13, fontweight="bold")
    return save_figure(fig, Path(output_path), quality=quality)


def write_hybrid_convergence_overlay(
    *,
    output_path: Path,
    loss_history: Sequence[Mapping[str, float]],
    mike_only_xyz_rmse_mm: float | None,
    hybrid_xyz_rmse_mm: float | None = None,
    quality: str | None = None,
) -> Path:
    """ANN-on-residual training curve with Mike-only RMSE as a baseline line.

    Loss is plotted in normalized residual units (whatever the ANN trained
    against), so the absolute scale is not directly comparable to the
    millimeter-space RMSE baselines. Annotations report the absolute RMSE
    of the final Hybrid prediction and the Mike-only baseline so the reader
    can see the millimeter-space delta the curve corresponds to.
    """
    history = [dict(row) for row in loss_history if "train_loss" in row]
    if not history:
        raise ValueError("write_hybrid_convergence_overlay requires a non-empty loss history.")
    epochs = np.asarray([float(row.get("epoch", index + 1)) for index, row in enumerate(history)], dtype=float)
    train_loss = np.asarray([float(row["train_loss"]) for row in history], dtype=float)
    val_loss = np.asarray([float(row.get("validation_loss", row["train_loss"])) for row in history], dtype=float)

    with report_style() as plt:
        fig, ax = plt.subplots(figsize=(8.4, 4.6), constrained_layout=True)
        ax.plot(epochs, train_loss, color=SEMANTIC_COLORS["model"], label="train residual loss (normalized)")
        ax.plot(epochs, val_loss, color=SEMANTIC_COLORS["prediction"], label="validation residual loss (normalized)")
        annotations: list[str] = []
        if mike_only_xyz_rmse_mm is not None and float(mike_only_xyz_rmse_mm) >= 0.0:
            annotations.append(f"Mike CC baseline = {float(mike_only_xyz_rmse_mm):.2f} mm XYZ RMSE")
        if hybrid_xyz_rmse_mm is not None and float(hybrid_xyz_rmse_mm) >= 0.0:
            annotations.append(f"Hybrid (final) = {float(hybrid_xyz_rmse_mm):.2f} mm XYZ RMSE")
        for index, text in enumerate(annotations):
            ax.text(
                0.02,
                0.95 - 0.07 * index,
                text,
                transform=ax.transAxes,
                color=SEMANTIC_COLORS["text"],
                fontsize=10,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
            )
        style_axes(
            ax,
            title="Hybrid ANN-on-residual training (vs Mike CC baseline)",
            xlabel="epoch",
            ylabel="MSE loss (normalized residual units)",
        )
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", framealpha=0.95)
    return save_figure(fig, Path(output_path), quality=quality)


def write_hybrid_improvement_bars(
    *,
    output_path: Path,
    metrics_by_model: Mapping[str, Mapping[str, float | None]],
    quality: str | None = None,
) -> Path:
    """Grouped bar chart of XYZ RMSE / p95 / max error across models."""
    metric_keys = ("xyz_rmse_mm", "p95_error_mm", "max_error_mm")
    metric_labels = {"xyz_rmse_mm": "XYZ RMSE", "p95_error_mm": "P95", "max_error_mm": "Max"}
    available_models = [
        model_key
        for model_key, metrics in metrics_by_model.items()
        if any(isinstance(metrics.get(key), (int, float)) for key in metric_keys)
    ]
    if not available_models:
        raise ValueError("write_hybrid_improvement_bars requires at least one model with numeric metrics.")

    with report_style() as plt:
        fig, ax = plt.subplots(figsize=(8.6, 4.6), constrained_layout=True)
        bar_width = 0.8 / max(1, len(available_models))
        x_positions = np.arange(len(metric_keys), dtype=float)
        for model_index, model_key in enumerate(available_models):
            metrics = metrics_by_model[model_key]
            values = [
                float(metrics[key]) if isinstance(metrics.get(key), (int, float)) else 0.0
                for key in metric_keys
            ]
            offsets = x_positions + (model_index - (len(available_models) - 1) / 2.0) * bar_width
            bars = ax.bar(
                offsets,
                values,
                width=bar_width * 0.92,
                color=MODEL_COLORS.get(model_key, SEMANTIC_COLORS["prediction"]),
                label=MODEL_LABELS.get(model_key, model_key),
                edgecolor="white",
                linewidth=0.5,
            )
            for bar_obj, value in zip(bars, values):
                ax.text(
                    bar_obj.get_x() + bar_obj.get_width() / 2.0,
                    bar_obj.get_height(),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=SEMANTIC_COLORS["text"],
                )
        ax.set_xticks(x_positions)
        ax.set_xticklabels([metric_labels[key] for key in metric_keys])
        style_axes(ax, title="Distal error breakdown across models", xlabel="metric", ylabel="error (mm)")
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", framealpha=0.95)
    return save_figure(fig, Path(output_path), quality=quality)


def write_hybrid_visualization_bundle(
    *,
    output_dir: Path,
    y_true: np.ndarray,
    predictions_by_model: Mapping[str, np.ndarray | None],
    label_metadata: Mapping[str, Any] | None = None,
    hybrid_loss_history: Sequence[Mapping[str, float]] | None = None,
    mike_only_xyz_rmse_mm: float | None = None,
    hybrid_xyz_rmse_mm: float | None = None,
    metrics_by_model: Mapping[str, Mapping[str, float | None]] | None = None,
    quality: str | None = None,
) -> dict[str, Path]:
    """Render every available comparison visualization for the hybrid path.

    Returns a dict of figure-name → path. Figures whose inputs are missing
    are silently skipped (so caller can pass partial inputs without guarding
    each call site).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned = _normalize_predictions_dict(predictions_by_model)
    produced: dict[str, Path] = {}
    if cleaned and np.asarray(y_true).size > 0:
        produced["scatter"] = write_hybrid_before_after_scatter(
            output_path=output_dir / "two_segment_hybrid_before_after_scatter_report.png",
            y_true=y_true,
            predictions_by_model=cleaned,
            label_metadata=label_metadata,
            quality=quality,
        )
        produced["residual_histograms"] = write_hybrid_residual_histograms(
            output_path=output_dir / "two_segment_hybrid_residual_histograms_report.png",
            y_true=y_true,
            predictions_by_model=cleaned,
            label_metadata=label_metadata,
            quality=quality,
        )
        produced["workspace_error_map"] = write_hybrid_workspace_error_map(
            output_path=output_dir / "two_segment_hybrid_workspace_error_map_report.png",
            y_true=y_true,
            predictions_by_model=cleaned,
            label_metadata=label_metadata,
            quality=quality,
        )
    if hybrid_loss_history:
        produced["convergence"] = write_hybrid_convergence_overlay(
            output_path=output_dir / "two_segment_hybrid_convergence_overlay_report.png",
            loss_history=hybrid_loss_history,
            mike_only_xyz_rmse_mm=mike_only_xyz_rmse_mm,
            hybrid_xyz_rmse_mm=hybrid_xyz_rmse_mm,
            quality=quality,
        )
    if metrics_by_model:
        produced["improvement_bars"] = write_hybrid_improvement_bars(
            output_path=output_dir / "two_segment_hybrid_improvement_bars_report.png",
            metrics_by_model=metrics_by_model,
            quality=quality,
        )
    _ = (figure_dpi, import_matplotlib, create_figure)
    return produced
