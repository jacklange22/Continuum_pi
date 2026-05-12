"""Output bundle writer and report figures for two-segment modeling."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from continuum_robot.experiments.dataset_io import canonical_experiment_output_root, canonical_timestamped_path
from continuum_robot.experiments.plotting import add_metric_box, color, create_figure, import_matplotlib, legend, save_figure, set_equal_xy, style_axes


EXPERIMENT_NAME = "two_segment_modeling"
EXPECTED_FIGURES = [
    "two_segment_model_comparison_report.png",
    "two_segment_distal_measured_vs_predicted_xy_report.png",
    "two_segment_position_error_distribution_report.png",
    "two_segment_axis_error_report.png",
    "two_segment_two_coil_error_report.png",
]


def allocate_output_dir(*, project_root: Path, output_root: Path | None = None, output_dir: Path | None = None) -> Path:
    if output_dir is not None:
        candidate = _collision_safe_dir(Path(output_dir))
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate
    root = Path(output_root) if output_root is not None else Path(project_root) / "data" / "experiments"
    experiment_root = canonical_experiment_output_root(root, EXPERIMENT_NAME)
    experiment_root.mkdir(parents=True, exist_ok=True)
    output_dir = canonical_timestamped_path(experiment_root, EXPERIMENT_NAME)
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _collision_safe_dir(path: Path) -> Path:
    candidate = Path(path)
    if not candidate.exists():
        return candidate
    suffix = 1
    while True:
        named = candidate.with_name(f"{candidate.name}_{suffix:02d}")
        if not named.exists():
            return named
        suffix += 1


def write_output_bundle(
    *,
    output_dir: Path,
    config: Any,
    dataset: Any,
    bundle: Any,
    split: dict[str, Any],
    model_results: list[Any],
    predictions_rows: list[dict[str, Any]],
    plot_quality: str = "production",
) -> dict[str, Path]:
    """Write the canonical two-segment modeling analysis bundle."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    paths = {
        "summary": output_dir / "summary.json",
        "metadata": output_dir / "metadata.json",
        "metrics": output_dir / "metrics.csv",
        "predictions": output_dir / "predictions.csv",
        "summary_text": output_dir / "two_segment_modeling_summary.txt",
        "model_config": output_dir / "model_config.yaml",
        "feature_metadata": output_dir / "feature_metadata.json",
        "label_metadata": output_dir / "label_metadata.json",
        "split": output_dir / "train_test_split.json",
        "rejected": output_dir / "rejected_samples.jsonl",
        "run_provenance": output_dir / "run_provenance.json",
        "config_snapshot": output_dir / "config_snapshot.yaml",
        "model_status": output_dir / "model_status.json",
        "physics_parameter_report": output_dir / "physics_model_parameter_report.json",
        "physics_parameter_report_text": output_dir / "physics_model_parameter_report.txt",
    }
    metrics_rows = _metrics_rows(model_results)
    summary = _summary_payload(config=config, dataset=dataset, bundle=bundle, split=split, model_results=model_results)
    metadata = _metadata_payload(config=config, dataset=dataset, bundle=bundle, split=split, model_results=model_results)
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    paths["feature_metadata"].write_text(json.dumps(bundle.feature_metadata, indent=2), encoding="utf-8")
    paths["label_metadata"].write_text(json.dumps(bundle.label_metadata, indent=2), encoding="utf-8")
    paths["split"].write_text(json.dumps(split, indent=2), encoding="utf-8")
    paths["run_provenance"].write_text(json.dumps(dataset.run_provenance, indent=2), encoding="utf-8")
    model_status = _model_status_payload(model_results)
    physics_report = _physics_parameter_report_payload(model_results, config=config)
    paths["model_status"].write_text(json.dumps(model_status, indent=2), encoding="utf-8")
    paths["physics_parameter_report"].write_text(json.dumps(physics_report, indent=2), encoding="utf-8")
    paths["physics_parameter_report_text"].write_text(_physics_parameter_report_text(physics_report), encoding="utf-8")
    paths["model_config"].write_text(yaml.safe_dump(_config_payload(config), sort_keys=False), encoding="utf-8")
    paths["config_snapshot"].write_text(yaml.safe_dump({"two_segment_modeling_config": _config_payload(config)}, sort_keys=False), encoding="utf-8")
    _write_metrics_csv(paths["metrics"], metrics_rows)
    _write_predictions_csv(paths["predictions"], predictions_rows)
    with paths["rejected"].open("w", encoding="utf-8") as handle:
        for item in dataset.rejected_samples:
            handle.write(json.dumps({"run_dir": str(item.run_dir), "sample_index": item.sample_index, "reason": item.reason, "payload": item.payload}) + "\n")
    paths["summary_text"].write_text(_summary_text(summary), encoding="utf-8")
    figure_paths = write_report_figures(
        output_dir=output_dir,
        bundle=bundle,
        split=split,
        model_results=model_results,
        predictions_rows=predictions_rows,
        quality=plot_quality,
    )
    paths.update(figure_paths)
    return paths


def write_report_figures(
    *,
    output_dir: Path,
    bundle: Any,
    split: dict[str, Any],
    model_results: list[Any],
    predictions_rows: list[dict[str, Any]],
    quality: str = "production",
) -> dict[str, Path]:
    figures: dict[str, Path] = {}
    figures["comparison"] = _plot_model_comparison(output_dir / "two_segment_model_comparison_report.png", model_results, quality=quality)
    figures["distal_measured_vs_predicted_xy"] = _plot_measured_vs_predicted_xy(
        output_dir / "two_segment_distal_measured_vs_predicted_xy_report.png",
        predictions_rows,
        role_prefix="distal",
        title="Measured vs Predicted Distal-Tip XY",
        quality=quality,
    )
    figures["measured_vs_predicted_xy_legacy_alias"] = _plot_measured_vs_predicted_xy(
        output_dir / "two_segment_measured_vs_predicted_xy_report.png",
        predictions_rows,
        role_prefix="distal",
        title="Measured vs Predicted Distal-Tip XY",
        quality=quality,
    )
    if bool(bundle.label_metadata.get("includes_intermediate_label")):
        figures["intermediate_measured_vs_predicted_xy"] = _plot_measured_vs_predicted_xy(
            output_dir / "two_segment_intermediate_measured_vs_predicted_xy_report.png",
            predictions_rows,
            role_prefix="intermediate",
            title="Measured vs Predicted Intermediate-Segment XY",
            quality=quality,
        )
    figures["position_error_distribution"] = _plot_error_distribution(
        output_dir / "two_segment_position_error_distribution_report.png",
        predictions_rows,
        quality=quality,
    )
    figures["axis_error"] = _plot_axis_error(output_dir / "two_segment_axis_error_report.png", model_results, quality=quality)
    figures["two_coil_error"] = _plot_two_coil_error(output_dir / "two_segment_two_coil_error_report.png", model_results, quality=quality)
    if bool(bundle.label_metadata.get("orientation_available")):
        figures["orientation_error"] = _plot_orientation_error(output_dir / "two_segment_orientation_error_report.png", model_results, quality=quality)
    ann_result = next((result for result in model_results if result.model_key == "ann" and result.loss_history), None)
    if ann_result is not None:
        figures["ann_loss_curve"] = _plot_ann_loss(output_dir / "two_segment_ann_loss_curve_report.png", ann_result.loss_history, quality=quality)
    if any("ann_sweep_candidate_count" in dict(result.metrics or {}) for result in model_results):
        figures["ann_model_size"] = _plot_ann_model_size(output_dir / "two_segment_ann_model_size_report.png", model_results, quality=quality)
    best = _best_completed_model(model_results)
    if best is not None:
        figures["workspace_residual_vectors"] = _plot_workspace_residuals(
            output_dir / "two_segment_workspace_residual_vectors_report.png",
            predictions_rows,
            model_key=best.model_key,
            quality=quality,
        )
    _ = split
    return figures


def _summary_payload(*, config: Any, dataset: Any, bundle: Any, split: dict[str, Any], model_results: list[Any]) -> dict[str, Any]:
    best = _best_completed_model(model_results)
    source_ids = _source_dataset_run_ids(dataset.run_provenance)
    metrics = {
        "dataset_type": EXPERIMENT_NAME,
        "run_trust_mode": "lower_trust" if dataset.allow_lower_trust else "offline_modeling_analysis",
        "valid_for_model_training": bool(not dataset.allow_lower_trust and dataset.accepted_count > 0),
        "valid_for_two_segment_model_training": bool(not dataset.allow_lower_trust and dataset.accepted_count > 0),
        "valid_for_thesis_repeatability": False,
        "accepted_sample_count": int(dataset.accepted_count),
        "rejected_sample_count": int(dataset.rejected_count),
        "rejected_sample_reasons": dataset.rejection_counts(),
        "source_run_count": int(dataset.run_count),
        "source_dataset_run_ids": source_ids,
        "best_model_by_xyz_rmse": (
            {
                "model_key": best.model_key,
                "xyz_rmse_mm": best.metrics.get("xyz_rmse_mm"),
                "orientation_mean_error_deg": best.metrics.get("orientation_mean_error_deg"),
            }
            if best is not None
            else None
        ),
        "split_method": split.get("method"),
        "train_samples": len(split.get("train_indices", [])),
        "test_samples": len(split.get("test_indices", [])),
        "distal_only": bool(dataset.distal_only),
        "includes_intermediate_pose": bool(dataset.includes_intermediate_pose),
        "orientation_available": bool(dataset.orientation_available),
        "label_mode": bundle.label_metadata.get("label_mode"),
        "direct_global_model": bool(bundle.label_metadata.get("direct_global_model", True)),
        "includes_intermediate_label": bool(bundle.label_metadata.get("includes_intermediate_label", False)),
        "structured_composition_model": bool(bundle.label_metadata.get("structured_composition_model", False)),
        "models": {result.model_key: _model_result_payload(result) for result in model_results},
        "physics_model_status": _physics_status_summary(model_results),
        "data_quality_warnings": _data_quality_warnings(dataset=dataset, split=split),
        "run_provenance": {
            "operating_mode": "offline_analysis",
            "hardware_profile": "offline",
            "source_experiment": "two_segment_collect_pose_command_dataset",
        },
    }
    return {
        "schema_version": "1.0",
        "experiment_name": EXPERIMENT_NAME,
        "run_id": Path(str(config.output_dir or "")).name if getattr(config, "output_dir", None) else "",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "success": bool(dataset.accepted_count > 0),
        "status": "success" if dataset.accepted_count > 0 else "failed",
        "sample_counts": {"total": int(dataset.accepted_count + dataset.rejected_count), "accepted": int(dataset.accepted_count), "rejected": int(dataset.rejected_count)},
        "experiment_metrics": metrics,
    }


def _metadata_payload(*, config: Any, dataset: Any, bundle: Any, split: dict[str, Any], model_results: list[Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "experiment_name": EXPERIMENT_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "trust_info": {
            "run_trust_mode": "lower_trust" if dataset.allow_lower_trust else "offline_modeling_analysis",
            "valid_for_model_training": bool(not dataset.allow_lower_trust and dataset.accepted_count > 0),
            "valid_for_two_segment_model_training": bool(not dataset.allow_lower_trust and dataset.accepted_count > 0),
            "valid_for_thesis_repeatability": False,
            "data_quality_warnings": _data_quality_warnings(dataset=dataset, split=split),
        },
        "provenance_info": {
            "operating_mode": "offline_analysis",
            "hardware_profile": "offline",
            "source_runs": [item.get("run_dir") for item in dataset.run_provenance],
            "allow_lower_trust": bool(dataset.allow_lower_trust),
        },
        "modeling_config": _config_payload(config),
        "label_metadata": dict(bundle.label_metadata or {}),
        "models": {result.model_key: _model_result_payload(result) for result in model_results},
        "physics_model_status": _physics_status_summary(model_results),
    }


def _config_payload(config: Any) -> dict[str, Any]:
    if hasattr(config, "__dataclass_fields__"):
        return asdict(config)
    return dict(config or {})


def _model_result_payload(result: Any) -> dict[str, Any]:
    return {
        "label": result.label,
        "status": result.status,
        "reason": result.reason,
        "metrics": dict(result.metrics or {}),
        "artifact_paths": dict(result.artifact_paths or {}),
        "status_details": dict(result.status_details or {}),
    }


def _metrics_rows(model_results: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in model_results:
        row = {"model_key": result.model_key, "label": result.label, "status": result.status, "reason": result.reason}
        status_details = dict(result.status_details or {})
        if status_details:
            row["missing_parameters"] = json.dumps(status_details.get("missing_parameters", []))
            row["hardware_validated"] = status_details.get("hardware_validated")
            row["predictions_generated"] = status_details.get("predictions_generated")
        row.update(dict(result.metrics or {}))
        rows.append(row)
    return rows


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summary_text(summary: dict[str, Any]) -> str:
    metrics = dict(summary.get("experiment_metrics", {}) or {})
    lines = [
        "Two-Segment Modeling Scaffold",
        f"status: {summary.get('status')}",
        f"accepted_sample_count: {metrics.get('accepted_sample_count')}",
        f"rejected_sample_count: {metrics.get('rejected_sample_count')}",
        f"split_method: {metrics.get('split_method')}",
        f"valid_for_two_segment_model_training: {metrics.get('valid_for_two_segment_model_training')}",
        f"allow_lower_trust: {metrics.get('run_trust_mode') == 'lower_trust'}",
        f"orientation_available: {metrics.get('orientation_available')}",
        f"label_mode: {metrics.get('label_mode')}",
        f"direct_global_model: {metrics.get('direct_global_model')}",
        f"structured_composition_model: {metrics.get('structured_composition_model')}",
        f"physics_model_status: {metrics.get('physics_model_status')}",
        "",
        "This is offline modeling/data analysis only. It does not enable two-segment control, chasing, or automatic pretension.",
    ]
    return "\n".join(lines).strip() + "\n"


def _plot_model_comparison(path: Path, model_results: list[Any], *, quality: str) -> Path:
    completed = [result for result in model_results if result.status == "completed"]
    fig, ax = create_figure(size="wide")
    if completed:
        labels = [result.label for result in completed]
        x = np.arange(len(labels))
        distal_values = [float(result.metrics.get("distal_xyz_rmse_mm", result.metrics.get("xyz_rmse_mm", 0.0))) for result in completed]
        ax.bar(x - 0.16, distal_values, width=0.32, color=color("model"), label="distal")
        if any("intermediate_xyz_rmse_mm" in result.metrics for result in completed):
            inter_values = [float(result.metrics.get("intermediate_xyz_rmse_mm", 0.0)) for result in completed]
            ax.bar(x + 0.16, inter_values, width=0.32, color=color("prediction"), label="intermediate")
            legend(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
    else:
        ax.text(0.5, 0.5, "No completed models", ha="center", va="center", transform=ax.transAxes)
    style_axes(ax, title="Two-Segment Model Comparison", xlabel="Model", ylabel="XYZ RMSE (mm)")
    return save_figure(fig, path, quality=quality)


def _plot_measured_vs_predicted_xy(path: Path, rows: list[dict[str, Any]], *, role_prefix: str, title: str, quality: str) -> Path:
    fig, ax = create_figure(size="square")
    completed_models = sorted({str(row.get("model_key")) for row in rows if row.get("status") == "completed"})
    if rows and completed_models:
        mx = f"measured_{role_prefix}_x_mm"
        my = f"measured_{role_prefix}_y_mm"
        px = f"predicted_{role_prefix}_x_mm"
        py = f"predicted_{role_prefix}_y_mm"
        available = [row for row in rows if mx in row and my in row and px in row and py in row]
        xs = [float(row[mx]) for row in available]
        ys = [float(row[my]) for row in available]
        ax.scatter(xs, ys, color=color("measured"), label="measured", s=18)
        for model_key in completed_models[:3]:
            model_rows = [row for row in available if row.get("model_key") == model_key]
            ax.scatter([float(row[px]) for row in model_rows], [float(row[py]) for row in model_rows], s=14, label=model_key)
        set_equal_xy(ax, x_values=xs, y_values=ys)
        legend(ax)
    else:
        ax.text(0.5, 0.5, "No predictions", ha="center", va="center", transform=ax.transAxes)
    style_axes(ax, title=title, xlabel="X (mm)", ylabel="Y (mm)")
    return save_figure(fig, path, quality=quality)


def _plot_error_distribution(path: Path, rows: list[dict[str, Any]], *, quality: str) -> Path:
    fig, ax = create_figure(size="wide")
    completed_models = sorted({str(row.get("model_key")) for row in rows if row.get("status") == "completed"})
    if completed_models:
        data = [[float(row["distal_position_error_mm"]) for row in rows if row.get("model_key") == model_key and "distal_position_error_mm" in row] for model_key in completed_models]
        try:
            ax.boxplot(data, tick_labels=completed_models, showfliers=True)
        except TypeError:  # pragma: no cover - older matplotlib compatibility
            ax.boxplot(data, labels=completed_models, showfliers=True)
    else:
        ax.text(0.5, 0.5, "No completed model errors", ha="center", va="center", transform=ax.transAxes)
    style_axes(ax, title="Position Error Distribution", xlabel="Model", ylabel="3D error (mm)")
    return save_figure(fig, path, quality=quality)


def _plot_axis_error(path: Path, model_results: list[Any], *, quality: str) -> Path:
    completed = [result for result in model_results if result.status == "completed"]
    fig, ax = create_figure(size="wide")
    if completed:
        x = np.arange(len(completed))
        width = 0.24
        for offset, key, label in [(-width, "distal_x_rmse_mm", "distal X"), (0.0, "distal_y_rmse_mm", "distal Y"), (width, "distal_z_rmse_mm", "distal Z")]:
            ax.bar(x + offset, [float(result.metrics.get(key, 0.0)) for result in completed], width=width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([result.model_key for result in completed])
        legend(ax)
    else:
        ax.text(0.5, 0.5, "No completed model axis metrics", ha="center", va="center", transform=ax.transAxes)
    style_axes(ax, title="Axis RMSE by Model", xlabel="Model", ylabel="RMSE (mm)")
    return save_figure(fig, path, quality=quality)


def _plot_orientation_error(path: Path, model_results: list[Any], *, quality: str) -> Path:
    completed = [result for result in model_results if result.status == "completed" and "distal_orientation_mean_error_deg" in result.metrics]
    fig, ax = create_figure(size="wide")
    if completed:
        ax.bar([result.model_key for result in completed], [float(result.metrics.get("distal_orientation_mean_error_deg", 0.0)) for result in completed], color=color("prediction"))
    else:
        ax.text(0.5, 0.5, "Orientation labels unavailable", ha="center", va="center", transform=ax.transAxes)
    style_axes(ax, title="Orientation Error", xlabel="Model", ylabel="Mean angular error (deg)")
    return save_figure(fig, path, quality=quality)


def _plot_ann_loss(path: Path, history: list[dict[str, float]], *, quality: str) -> Path:
    fig, ax = create_figure(size="wide")
    if history:
        ax.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="train")
        ax.plot([row["epoch"] for row in history], [row["validation_loss"] for row in history], label="validation")
        legend(ax)
    else:
        ax.text(0.5, 0.5, "ANN not trained", ha="center", va="center", transform=ax.transAxes)
    style_axes(ax, title="ANN Loss Curve", xlabel="Epoch", ylabel="Standardized MSE")
    return save_figure(fig, path, quality=quality)


def _plot_workspace_residuals(path: Path, rows: list[dict[str, Any]], *, model_key: str, quality: str) -> Path:
    fig, ax = create_figure(size="square")
    selected = [row for row in rows if row.get("model_key") == model_key and row.get("status") == "completed"]
    if selected:
        xs = [float(row["measured_distal_x_mm"]) for row in selected]
        ys = [float(row["measured_distal_y_mm"]) for row in selected]
        dx = [float(row["predicted_distal_x_mm"]) - float(row["measured_distal_x_mm"]) for row in selected]
        dy = [float(row["predicted_distal_y_mm"]) - float(row["measured_distal_y_mm"]) for row in selected]
        ax.scatter(xs, ys, color=color("measured"), s=18)
        ax.quiver(xs, ys, dx, dy, angles="xy", scale_units="xy", scale=1.0, color=color("prediction"), width=0.004)
        set_equal_xy(ax, x_values=xs, y_values=ys)
        add_metric_box(ax, [f"model: {model_key}", f"n={len(selected)}"], loc="upper left")
    else:
        ax.text(0.5, 0.5, "No residual vectors", ha="center", va="center", transform=ax.transAxes)
    style_axes(ax, title="Workspace Residual Vectors", xlabel="X (mm)", ylabel="Y (mm)")
    return save_figure(fig, path, quality=quality)


def _best_completed_model(model_results: list[Any]):
    completed = [result for result in model_results if result.status == "completed" and "xyz_rmse_mm" in result.metrics]
    return min(completed, key=lambda result: float(result.metrics.get("xyz_rmse_mm", float("inf")))) if completed else None


def _model_status_payload(model_results: list[Any]) -> dict[str, Any]:
    return {
        "schema_version": "two_segment_model_status_v1",
        "models": {result.model_key: _model_result_payload(result) for result in model_results},
    }


def _physics_status_summary(model_results: list[Any]) -> dict[str, Any]:
    physics_keys = {"mike_constant_curvature", "camarillo"}
    return {
        result.model_key: {
            "status": result.status,
            "reason": result.reason,
            "missing_parameters": dict(result.status_details or {}).get("missing_parameters", []),
            "hardware_validated": dict(result.status_details or {}).get("hardware_validated"),
            "predictions_generated": dict(result.status_details or {}).get("predictions_generated", bool(result.predictions is not None)),
        }
        for result in model_results
        if result.model_key in physics_keys
    }


def _physics_parameter_report_payload(model_results: list[Any], *, config: Any) -> dict[str, Any]:
    statuses = _physics_status_summary(model_results)
    model_payloads = {
        key: dict(result.status_details or {"model_key": result.model_key, "status": result.status, "reason": result.reason})
        for key, result in ((result.model_key, result) for result in model_results)
        if key in {"mike_constant_curvature", "camarillo"}
    }
    return {
        "schema_version": "two_segment_physics_parameter_report_v1",
        "config_physics_models": dict(_config_payload(config).get("model_config", {}).get("physics_models", {}) or {}),
        "known_design_parameters": _aggregate_status_list(model_payloads, "known_design_parameters"),
        "measured_parameters": _aggregate_status_list(model_payloads, "measured_parameters"),
        "estimated_parameters": _aggregate_status_list(model_payloads, "estimated_parameters"),
        "unknown_parameters": _aggregate_status_list(model_payloads, "unknown_parameters"),
        "hardware_validation_required": _aggregate_status_list(model_payloads, "hardware_validation_required"),
        "blocking_missing_parameters": _aggregate_status_list(model_payloads, "blocking_missing_parameters"),
        "nonblocking_uncertain_parameters": _aggregate_status_list(model_payloads, "nonblocking_uncertain_parameters"),
        "models": model_payloads,
        "summary": statuses,
        "note": "Unavailable physics adapters do not generate predictions and are excluded from numeric model ranking.",
    }


def _physics_parameter_report_text(report: dict[str, Any]) -> str:
    lines = [
        "Two-Segment Physics Model Parameter Report",
        "Unavailable physics adapters do not generate predictions.",
        "",
        f"known_design_parameters: {report.get('known_design_parameters', [])}",
        f"measured_parameters: {report.get('measured_parameters', [])}",
        f"estimated_parameters: {report.get('estimated_parameters', [])}",
        f"unknown_parameters: {report.get('unknown_parameters', [])}",
        f"hardware_validation_required: {report.get('hardware_validation_required', [])}",
        f"blocking_missing_parameters: {report.get('blocking_missing_parameters', [])}",
        f"nonblocking_uncertain_parameters: {report.get('nonblocking_uncertain_parameters', [])}",
        "",
    ]
    models = report.get("models") if isinstance(report.get("models"), dict) else {}
    for key, payload in models.items():
        data = dict(payload or {})
        lines.extend(
            [
                f"model: {key}",
                f"status: {data.get('status')}",
                f"reason: {data.get('reason')}",
                f"hardware_validated: {data.get('hardware_validated')}",
                f"predictions_generated: {data.get('predictions_generated')}",
                f"missing_parameters: {data.get('missing_parameters', [])}",
                f"blocking_missing_parameters: {data.get('blocking_missing_parameters', [])}",
                f"nonblocking_uncertain_parameters: {data.get('nonblocking_uncertain_parameters', [])}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def _aggregate_status_list(model_payloads: dict[str, dict[str, Any]], key: str) -> list[str]:
    values: list[str] = []
    for payload in model_payloads.values():
        values.extend(str(item) for item in list(payload.get(key, []) or []))
    return sorted(set(values))


def _plot_two_coil_error(path: Path, model_results: list[Any], *, quality: str) -> Path:
    completed = [result for result in model_results if result.status == "completed"]
    fig, ax = create_figure(size="wide")
    if completed:
        x = np.arange(len(completed))
        width = 0.32
        ax.bar(x - width / 2, [float(result.metrics.get("distal_xyz_rmse_mm", result.metrics.get("xyz_rmse_mm", 0.0))) for result in completed], width=width, label="distal")
        ax.bar(x + width / 2, [float(result.metrics.get("intermediate_xyz_rmse_mm", 0.0)) for result in completed], width=width, label="intermediate")
        ax.set_xticks(x)
        ax.set_xticklabels([result.model_key for result in completed])
        legend(ax)
    else:
        ax.text(0.5, 0.5, "No completed model errors", ha="center", va="center", transform=ax.transAxes)
    style_axes(ax, title="Two-Coil Position Error", xlabel="Model", ylabel="XYZ RMSE (mm)")
    return save_figure(fig, path, quality=quality)


def _plot_ann_model_size(path: Path, model_results: list[Any], *, quality: str) -> Path:
    fig, ax = create_figure(size="wide")
    ann = next((result for result in model_results if result.model_key == "ann"), None)
    summary_path = dict(getattr(ann, "artifact_paths", {}) or {}).get("sweep_summary") if ann is not None else None
    rows: list[dict[str, Any]] = []
    if summary_path and Path(summary_path).exists():
        rows = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    completed = [row for row in rows if row.get("status") == "completed" and row.get("xyz_rmse_mm") is not None]
    if completed:
        labels = [str(row.get("hidden_layers")) for row in completed]
        values = [float(row.get("xyz_rmse_mm")) for row in completed]
        ax.bar(labels, values, color=color("model"))
    else:
        ax.text(0.5, 0.5, "ANN sweep not available", ha="center", va="center", transform=ax.transAxes)
    style_axes(ax, title="ANN Model Size Sweep", xlabel="Hidden layers", ylabel="Distal XYZ RMSE (mm)")
    return save_figure(fig, path, quality=quality)


def _data_quality_warnings(*, dataset: Any, split: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if dataset.allow_lower_trust:
        warnings.append("allow_lower_trust_used_outputs_not_thesis_trusted")
    if split.get("method") == "single_run_random":
        warnings.append("single_run_random_split_can_overestimate_generalization")
    if not dataset.orientation_available:
        warnings.append("orientation_labels_unavailable_position_only")
    if dataset.rejected_count:
        warnings.append("some_samples_rejected")
    return sorted(set(warnings))


def _source_dataset_run_ids(run_provenance: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for item in run_provenance:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        summary = item.get("summary") if isinstance(item.get("summary"), dict) else {}
        run_id = metadata.get("run_id") or summary.get("run_id") or item.get("run_name") or item.get("run_dir")
        if run_id:
            ids.append(str(run_id))
    return sorted(set(ids))
