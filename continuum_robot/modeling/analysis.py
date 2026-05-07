"""Modeling dataset evaluation and comparison helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from continuum_robot.experiments.dataset_io import canonical_timestamped_path
from continuum_robot.experiments.plotting import add_metric_box, color, create_figure, import_matplotlib, legend, save_figure, set_equal_xy, style_axes
from continuum_robot.modeling.ann_training import (
    LEGACY_FULL_POSE_INPUT_DIM,
    LEGACY_FULL_POSE_OUTPUT_DIM,
    ModelingDatasetSummary,
    TorchUnavailableError,
    TrainedArtifactSummary,
    _build_legacy_ann_model,
    _load_export_rows,
    _require_torch,
    _torch_dtype,
    discover_trained_artifacts,
    load_loss_history,
    load_modeling_dataset_summary,
    load_training_metadata,
    prepare_legacy_ann_dataset,
)

try:
    from continuum_robot.gui.experiment_visualization import ChartModel, ChartSeriesModel, VisualizationModel
except Exception:
    @dataclass
    class ChartSeriesModel:
        name: str
        points_xy: list[tuple[float, float]] = field(default_factory=list)
        color_hex: str = "#58718a"

    @dataclass
    class ChartModel:
        kind: str
        title: str
        x_title: str
        y_title: str
        caption: str = ""
        categories: list[str] = field(default_factory=list)
        values: list[float] = field(default_factory=list)
        points_xy: list[tuple[float, float]] = field(default_factory=list)
        series_xy: list[ChartSeriesModel] = field(default_factory=list)
        color_hex: str = "#58718a"

    @dataclass
    class VisualizationModel:
        series_3d: list[Any] = field(default_factory=list)
        charts: list[ChartModel] = field(default_factory=list)
        summary_lines: list[str] = field(default_factory=list)

try:
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QColor, QImage, QPainter, QPen
    from PySide6.QtWidgets import QApplication

    _QT_AVAILABLE = True
except Exception:  # pragma: no cover - optional during headless use
    _QT_AVAILABLE = False


DEFAULT_RESULTS_ROOT = "data/modeling_results"
DEFAULT_CAMARILLO_STIFFNESS_PATH = "tools/camarillo_stiffness"
DEFAULT_CABLE_POSITIONS_MM = ((4.0, 0.0), (0.0, 4.0), (-4.0, 0.0), (0.0, -4.0))
DEFAULT_SEGMENT_LENGTH_MM = 64.0
DEFAULT_ADDITIONAL_CABLE_LENGTH_MM = 50.0


@dataclass(frozen=True)
class ModelingGeometryConfig:
    """Legacy-model geometry and stiffness assumptions used for evaluation."""

    cable_positions_mm: tuple[tuple[float, float], ...] = DEFAULT_CABLE_POSITIONS_MM
    segment_length_mm: float = DEFAULT_SEGMENT_LENGTH_MM
    additional_cable_length_mm: float = DEFAULT_ADDITIONAL_CABLE_LENGTH_MM
    camarillo_stiffness_path: str = DEFAULT_CAMARILLO_STIFFNESS_PATH


@dataclass(frozen=True)
class ArtifactDetails:
    """Expanded ANN artifact metadata used by the modeling tab."""

    summary: TrainedArtifactSummary
    metadata: dict[str, Any]
    training_config: dict[str, Any]
    split_manifest: dict[str, Any]
    train_losses: list[float]
    validation_losses: list[float]


@dataclass(frozen=True)
class ModelMetrics:
    """Compact error summary for one evaluated model."""

    model_key: str
    label: str
    status: str
    reason: str = ""
    sample_count: int = 0
    position_rmse_mm: float | None = None
    mean_position_error_mm: float | None = None
    max_position_error_mm: float | None = None
    tangent_mean_error_deg: float | None = None
    tangent_rmse_deg: float | None = None
    tangent_max_error_deg: float | None = None
    axis_position_rmse_mm: list[float] = field(default_factory=list)
    axis_tangent_rmse: list[float] = field(default_factory=list)
    phase_metrics: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelEvaluation:
    """One model's predictions and derived metrics."""

    metrics: ModelMetrics
    predictions: np.ndarray | None = None
    position_errors_mm: list[float] = field(default_factory=list)
    tangent_errors_deg: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class ModelingEvaluationConfig:
    """Requested evaluation configuration."""

    include_mike: bool = True
    include_camarillo: bool = True
    include_ann: bool = True
    evaluation_scope: str = "artifact_test_split"
    results_root: str = DEFAULT_RESULTS_ROOT
    geometry: ModelingGeometryConfig = field(default_factory=ModelingGeometryConfig)


@dataclass(frozen=True)
class ModelingEvaluationResult:
    """Saved evaluation bundle and GUI-facing visualization."""

    dataset_summary: ModelingDatasetSummary
    artifact_details: ArtifactDetails | None
    evaluation_scope_requested: str
    evaluation_scope_used: str
    evaluation_scope_note: str
    selected_sample_count: int
    output_dir: Path
    summary_path: Path
    metadata_path: Path
    comparison_csv_path: Path
    phase_csv_path: Path | None
    plot_paths: dict[str, Path]
    model_evaluations: dict[str, ModelEvaluation]
    visualization_model: VisualizationModel


def default_results_root(project_root: Path) -> Path:
    """Return the default evaluation output root."""
    return Path(project_root) / DEFAULT_RESULTS_ROOT


def load_trained_artifact_details(path: Path) -> ArtifactDetails:
    """Load metadata, config, split manifest, and loss history for one artifact."""
    artifact_dir = Path(path)
    metadata = load_training_metadata(artifact_dir)
    training_config_path = artifact_dir / "training_config.json"
    split_manifest_path = artifact_dir / "split_manifest.json"
    training_config = (
        json.loads(training_config_path.read_text(encoding="utf-8"))
        if training_config_path.exists()
        else {}
    )
    split_manifest = (
        json.loads(split_manifest_path.read_text(encoding="utf-8"))
        if split_manifest_path.exists()
        else {}
    )
    train_losses, validation_losses = load_loss_history(artifact_dir)
    matching = [
        artifact
        for artifact in discover_trained_artifacts(artifact_root=artifact_dir.parent)
        if artifact.path == artifact_dir
    ]
    if matching:
        summary = matching[0]
    else:
        dataset = dict(metadata.get("dataset", {}) or {})
        backend = dict(metadata.get("backend", {}) or {})
        training = dict(metadata.get("training", {}) or {})
        files = dict(metadata.get("files", {}) or {})
        model_path_raw = files.get("model_path")
        summary = TrainedArtifactSummary(
            path=artifact_dir,
            artifact_name=artifact_dir.name,
            created_at_utc=str(metadata.get("created_at_utc", "") or ""),
            status=str(metadata.get("status", "unknown") or "unknown"),
            dataset_name=str(dataset.get("run_name", "unknown") or "unknown"),
            backend_name=str(backend.get("selected_backend", "unknown") or "unknown"),
            epochs_completed=int(training.get("epochs_completed", 0) or 0),
            best_validation_loss=(
                float(training["best_validation_loss"])
                if training.get("best_validation_loss") not in (None, "")
                else None
            ),
            metadata_path=artifact_dir / "training_metadata.json",
            model_path=(Path(model_path_raw) if model_path_raw else None),
        )
    return ArtifactDetails(
        summary=summary,
        metadata=metadata,
        training_config=training_config,
        split_manifest=split_manifest,
        train_losses=train_losses,
        validation_losses=validation_losses,
    )


def evaluate_models(
    *,
    project_root: Path,
    dataset_path: Path,
    artifact_path: Path | None,
    config: ModelingEvaluationConfig,
) -> ModelingEvaluationResult:
    """Evaluate Mike, Camarillo, and ANN models on one canonical dataset."""
    dataset_summary = load_modeling_dataset_summary(dataset_path)
    prepared = prepare_legacy_ann_dataset(dataset_path)
    if prepared.inputs.shape[0] == 0:
        raise ValueError("Selected dataset has no accepted full-pose samples after legacy filtering.")
    artifact_details = load_trained_artifact_details(artifact_path) if artifact_path is not None else None
    selected_indices, scope_used, scope_note = _resolve_evaluation_indices(
        dataset_path=dataset_path,
        prepared_sample_count=int(prepared.inputs.shape[0]),
        artifact_details=artifact_details,
        requested_scope=config.evaluation_scope,
    )
    if not selected_indices:
        raise ValueError("No samples were available for the requested evaluation scope.")

    inputs = prepared.inputs[selected_indices, :]
    truths = prepared.outputs[selected_indices, :]
    phases = _selected_phases(dataset_path=dataset_path, prepared=prepared, selected_indices=selected_indices)
    evaluations: dict[str, ModelEvaluation] = {}

    if bool(config.include_mike):
        evaluations["mike"] = _evaluate_mike(
            inputs=inputs,
            truths=truths,
            phases=phases,
            geometry=config.geometry,
        )
    if bool(config.include_camarillo):
        evaluations["camarillo"] = _evaluate_camarillo(
            project_root=project_root,
            inputs=inputs,
            truths=truths,
            phases=phases,
            geometry=config.geometry,
        )
    if bool(config.include_ann):
        evaluations["ann"] = _evaluate_ann(
            inputs=inputs,
            truths=truths,
            phases=phases,
            artifact_details=artifact_details,
        )

    visualization = build_modeling_visualization(
        dataset_summary=dataset_summary,
        artifact_details=artifact_details,
        evaluation_scope_requested=config.evaluation_scope,
        evaluation_scope_used=scope_used,
        evaluation_scope_note=scope_note,
        truths=truths,
        evaluations=evaluations,
        phases=phases,
    )
    output_dir = _allocate_results_dir(
        project_root=Path(project_root),
        results_root_raw=config.results_root,
        dataset_name=dataset_summary.run_name,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    comparison_csv_path = output_dir / "comparison_metrics.csv"
    phase_csv_path = output_dir / "phase_metrics.csv"
    summary_path = output_dir / "summary.json"
    metadata_path = output_dir / "evaluation_metadata.json"
    plot_paths = _write_plots(
        output_dir=output_dir,
        truths=truths,
        evaluations=evaluations,
        phases=phases,
    )
    _write_comparison_csv(comparison_csv_path, evaluations)
    if _phase_rows(evaluations):
        _write_phase_csv(phase_csv_path, evaluations)
    else:
        phase_csv_path = None
    metadata_payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(dataset_summary.path),
            "run_id": dataset_summary.run_id,
            "run_name": dataset_summary.run_name,
            "dataset_mode": dataset_summary.dataset_mode,
            "dataset_mode_summary": dataset_summary.dataset_mode_summary,
            "accepted_count": dataset_summary.accepted_count,
            "rejected_count": dataset_summary.rejected_count,
            "trust_summary": dataset_summary.trust_summary,
        },
        "artifact": (
            {
                "path": str(artifact_details.summary.path),
                "artifact_name": artifact_details.summary.artifact_name,
                "dataset_name": artifact_details.summary.dataset_name,
                "backend_name": artifact_details.summary.backend_name,
                "metadata": artifact_details.metadata,
                "split_manifest": artifact_details.split_manifest,
            }
            if artifact_details is not None
            else None
        ),
        "evaluation": {
            "requested_scope": config.evaluation_scope,
            "used_scope": scope_used,
            "scope_note": scope_note,
            "selected_sample_count": len(selected_indices),
            "models_requested": {
                "mike": bool(config.include_mike),
                "camarillo": bool(config.include_camarillo),
                "ann": bool(config.include_ann),
            },
            "geometry": asdict(config.geometry),
            "legacy_references": [
                "@references/comparison.py",
                "@references/mike_cc.py",
                "@references/camarillo_cc.py",
                "@references/utils_cc.py",
                "@references/ANN.py",
                "@references/multi_input.py",
                "@references/hysteresis_model_learning.py",
            ],
        },
        "results": {
            key: _metrics_to_payload(evaluation.metrics)
            for key, evaluation in evaluations.items()
        },
        "files": {
            "comparison_csv_path": str(comparison_csv_path),
            "phase_csv_path": (str(phase_csv_path) if phase_csv_path is not None else None),
            "plot_paths": {key: str(value) for key, value in plot_paths.items()},
        },
    }
    summary_payload = {
        "dataset_run_name": dataset_summary.run_name,
        "dataset_mode": dataset_summary.dataset_mode,
        "selected_sample_count": len(selected_indices),
        "evaluation_scope_used": scope_used,
        "models": {
            key: _metrics_to_payload(evaluation.metrics)
            for key, evaluation in evaluations.items()
        },
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return ModelingEvaluationResult(
        dataset_summary=dataset_summary,
        artifact_details=artifact_details,
        evaluation_scope_requested=config.evaluation_scope,
        evaluation_scope_used=scope_used,
        evaluation_scope_note=scope_note,
        selected_sample_count=len(selected_indices),
        output_dir=output_dir,
        summary_path=summary_path,
        metadata_path=metadata_path,
        comparison_csv_path=comparison_csv_path,
        phase_csv_path=phase_csv_path,
        plot_paths=plot_paths,
        model_evaluations=evaluations,
        visualization_model=visualization,
    )


def build_modeling_visualization(
    *,
    dataset_summary: ModelingDatasetSummary,
    artifact_details: ArtifactDetails | None,
    evaluation_scope_requested: str,
    evaluation_scope_used: str,
    evaluation_scope_note: str,
    truths: np.ndarray,
    evaluations: dict[str, ModelEvaluation],
    phases: list[str],
) -> VisualizationModel:
    """Build GUI-facing plots and summary text for modeling comparison."""
    available = [evaluation for evaluation in evaluations.values() if evaluation.metrics.status == "completed"]
    summary_lines = [
        f"Dataset: {dataset_summary.run_name}",
        f"Dataset mode: {dataset_summary.dataset_mode}",
        f"Accepted samples in dataset: {dataset_summary.accepted_count}",
        f"Selected evaluation samples: {truths.shape[0]}",
        f"Requested scope: {evaluation_scope_requested}",
        f"Used scope: {evaluation_scope_used}",
        f"Scope note: {evaluation_scope_note}",
        f"ANN artifact: {artifact_details.summary.artifact_name if artifact_details is not None else 'none selected'}",
    ]
    for evaluation in evaluations.values():
        metrics = evaluation.metrics
        if metrics.status != "completed":
            summary_lines.append(f"{metrics.label}: {metrics.status} ({metrics.reason or 'unavailable'})")
            continue
        summary_lines.append(
            (
                f"{metrics.label}: position RMSE {_fmt(metrics.position_rmse_mm)} mm, "
                f"mean {_fmt(metrics.mean_position_error_mm)} mm, "
                f"max {_fmt(metrics.max_position_error_mm)} mm, "
                f"tangent mean {_fmt(metrics.tangent_mean_error_deg)} deg"
            )
        )
    charts: list[ChartModel] = []
    if truths.size and available:
        scatter_series = [
            ChartSeriesModel(
                name="Measured",
                points_xy=[(float(row[0]), float(row[1])) for row in truths[:, :3]],
                color_hex="#0f766e",
            )
        ]
        scatter_palette = {"mike": "#2563eb", "camarillo": "#dc2626", "ann": "#a16207"}
        for evaluation in available:
            scatter_series.append(
                ChartSeriesModel(
                    name=evaluation.metrics.label,
                    points_xy=[
                        (float(row[0]), float(row[1]))
                        for row in (evaluation.predictions[:, :3] if evaluation.predictions is not None else np.zeros((0, 3)))
                    ],
                    color_hex=scatter_palette.get(evaluation.metrics.model_key, "#58718a"),
                )
            )
        charts.append(
            ChartModel(
                kind="scatter",
                title="Workspace XY",
                x_title="X (mm)",
                y_title="Y (mm)",
                series_xy=scatter_series,
                caption="Measured workspace samples and model predictions on the same selected dataset slice.",
            )
        )
    comparison_labels = [evaluation.metrics.label for evaluation in available]
    if comparison_labels:
        charts.append(
            ChartModel(
                kind="bar",
                title="Position RMSE",
                x_title="Model",
                y_title="RMSE (mm)",
                categories=comparison_labels,
                values=[float(evaluation.metrics.position_rmse_mm or 0.0) for evaluation in available],
                color_hex="#2563eb",
                caption="RMS tip-position error on the selected evaluation set.",
            )
        )
        charts.append(
            ChartModel(
                kind="bar",
                title="Mean Tangent Error",
                x_title="Model",
                y_title="Error (deg)",
                categories=comparison_labels,
                values=[float(evaluation.metrics.tangent_mean_error_deg or 0.0) for evaluation in available],
                color_hex="#dc2626",
                caption="Mean angular error between predicted and measured tip tangents.",
            )
        )
    for evaluation in available:
        histogram = _histogram(
            np.asarray(evaluation.position_errors_mm, dtype=float),
            bins=10,
        )
        charts.append(
            ChartModel(
                kind="bar",
                title=f"{evaluation.metrics.label} Position Error Histogram",
                x_title="Bin",
                y_title="Count",
                categories=histogram["labels"],
                values=histogram["counts"],
                color_hex="#0f766e",
                caption="Position-error distribution for the selected evaluation samples.",
            )
        )
    phase_labels = sorted({phase for phase in phases if phase})
    if len(phase_labels) > 1 and available:
        phase_index = {phase: index for index, phase in enumerate(phase_labels)}
        charts.append(
            ChartModel(
                kind="line",
                title="Per-Phase Position RMSE",
                x_title="Phase Index",
                y_title="RMSE (mm)",
                series_xy=[
                    ChartSeriesModel(
                        name=evaluation.metrics.label,
                        points_xy=[
                            (
                                float(phase_index[phase]),
                                float(metrics.get("position_rmse_mm", 0.0)),
                            )
                            for phase, metrics in sorted(
                                evaluation.metrics.phase_metrics.items(),
                                key=lambda item: phase_index.get(item[0], 10**6),
                            )
                        ],
                        color_hex=color,
                    )
                    for evaluation, color in zip(
                        available,
                        ["#2563eb", "#dc2626", "#a16207"],
                    )
                ],
                caption=(
                    "Phase order: "
                    + ", ".join(f"{phase_index[label]}={label}" for label in phase_labels)
                ),
            )
        )
    return VisualizationModel(charts=charts, summary_lines=summary_lines)


def build_dataset_summary_pairs(summary: ModelingDatasetSummary) -> list[tuple[str, str]]:
    """Format dataset metadata for the modeling tab."""
    exports = [
        label
        for label, present in (
            ("export.jsonl", summary.export_jsonl_path is not None),
            ("legacy.dat", summary.legacy_dat_path is not None),
            ("summary.txt", summary.summary_text_path is not None),
        )
        if present
    ]
    return [
        ("Dataset Mode", summary.dataset_mode),
        ("Mode Summary", summary.dataset_mode_summary or "n/a"),
        ("Samples", str(summary.sample_count)),
        ("Accepted / Rejected", f"{summary.accepted_count} / {summary.rejected_count}"),
        ("Trust / Provenance", summary.trust_summary),
        ("Exports Present", ", ".join(exports) if exports else "none"),
        ("Full Pose", "yes" if summary.full_pose_available else "no"),
        ("Tangent Data", "yes" if summary.tangent_available else "no"),
        ("Sequential Context", "yes" if summary.sequential_context_available else "no"),
    ]


def build_artifact_summary_pairs(details: ArtifactDetails | None) -> list[tuple[str, str]]:
    """Format ANN artifact metadata for the modeling tab."""
    if details is None:
        return [
            ("Artifact", "No ANN artifact selected."),
        ]
    metadata = dict(details.metadata or {})
    model_payload = dict(metadata.get("model", {}) or {})
    backend = dict(metadata.get("backend", {}) or {})
    training = dict(metadata.get("training", {}) or {})
    dataset = dict(metadata.get("dataset", {}) or {})
    validation_losses = [value for value in details.validation_losses if not math.isnan(float(value))]
    return [
        ("Artifact", details.summary.artifact_name),
        ("Linked Dataset", str(dataset.get("run_name", "unknown") or "unknown")),
        ("Model Config", f"{model_payload.get('hidden_layers', [])} -> {model_payload.get('output_dim', 'n/a')}"),
        (
            "Training Config",
            (
                f"epochs={training.get('epochs_completed', 'n/a')}, "
                f"batch={training.get('batch_size', 'n/a')}, "
                f"lr={training.get('learning_rate', 'n/a')}"
            ),
        ),
        ("Backend", f"{backend.get('selected_backend', 'unknown')} | torch {backend.get('torch_version', 'n/a')}"),
        (
            "Loss Summary",
            (
                f"best val {_fmt(training.get('best_validation_loss'))}, "
                f"last val {_fmt(validation_losses[-1] if validation_losses else None)}"
            ),
        ),
        (
            "Held-Out Split",
            str(len(details.split_manifest.get("test_indices", []) or []))
            if details.split_manifest
            else "n/a",
        ),
    ]


def build_evaluation_summary_pairs(result: ModelingEvaluationResult | None) -> list[tuple[str, str]]:
    """Format last evaluation status for the modeling tab."""
    if result is None:
        return [("Evaluation", "No results yet.")]
    best = sorted(
        (
            evaluation.metrics
            for evaluation in result.model_evaluations.values()
            if evaluation.metrics.status == "completed" and evaluation.metrics.position_rmse_mm is not None
        ),
        key=lambda metrics: float(metrics.position_rmse_mm),
    )
    best_label = best[0].label if best else "n/a"
    best_rmse = _fmt(best[0].position_rmse_mm) if best else "n/a"
    return [
        ("Scope Used", f"{result.evaluation_scope_used} ({result.selected_sample_count} samples)"),
        ("Best Position RMSE", f"{best_label} | {best_rmse} mm"),
        ("Output Folder", str(result.output_dir)),
    ]


def _resolve_evaluation_indices(
    *,
    dataset_path: Path,
    prepared_sample_count: int,
    artifact_details: ArtifactDetails | None,
    requested_scope: str,
) -> tuple[list[int], str, str]:
    all_indices = list(range(prepared_sample_count))
    requested = str(requested_scope or "full_dataset").strip().lower()
    if requested != "artifact_test_split":
        return all_indices, "full_dataset", "Using all accepted legacy-compatible samples from the selected dataset."
    if artifact_details is None:
        return all_indices, "full_dataset", "No ANN artifact selected, so the full accepted dataset was used."
    split_manifest = dict(artifact_details.split_manifest or {})
    dataset = dict(artifact_details.metadata.get("dataset", {}) or {})
    linked_path = str(dataset.get("path", "") or "")
    if linked_path:
        try:
            if Path(linked_path).resolve() != Path(dataset_path).resolve():
                return (
                    all_indices,
                    "full_dataset",
                    "The selected ANN artifact was trained on a different dataset, so its held-out split was not reused.",
                )
        except Exception:
            pass
    test_indices = [int(value) for value in split_manifest.get("test_indices", []) or [] if int(value) < prepared_sample_count]
    if not test_indices:
        return (
            all_indices,
            "full_dataset",
            "The selected ANN artifact does not expose a reusable held-out test split, so the full accepted dataset was used.",
        )
    return (
        test_indices,
        "artifact_test_split",
        "Using the selected ANN artifact's saved held-out test split.",
    )


def _selected_phases(
    *,
    dataset_path: Path,
    prepared,
    selected_indices: list[int],
) -> list[str]:
    rows = _load_export_rows(dataset_path)
    phase_by_sequence = {
        int(row.get("sequence_index", index) or index): str(row.get("phase", "unknown") or "unknown")
        for index, row in enumerate(rows)
    }
    return [
        phase_by_sequence.get(int(prepared.sequence_indices[index]), "unknown")
        for index in selected_indices
    ]


def _evaluate_mike(
    *,
    inputs: np.ndarray,
    truths: np.ndarray,
    phases: list[str],
    geometry: ModelingGeometryConfig,
) -> ModelEvaluation:
    predictions = np.zeros_like(truths)
    for index, command in enumerate(inputs):
        webster = _mike_forward_webster(command=command, geometry=geometry)
        transform = _calculate_transform(webster)
        predictions[index, :3] = transform[:3, 3]
        predictions[index, 3:] = transform[:3, 2]
    return _complete_model_evaluation(
        model_key="mike",
        label="Mike",
        predictions=predictions,
        truths=truths,
        phases=phases,
    )


def _evaluate_camarillo(
    *,
    project_root: Path,
    inputs: np.ndarray,
    truths: np.ndarray,
    phases: list[str],
    geometry: ModelingGeometryConfig,
) -> ModelEvaluation:
    stiffness_path = Path(project_root) / geometry.camarillo_stiffness_path
    if not stiffness_path.exists():
        return ModelEvaluation(
            metrics=ModelMetrics(
                model_key="camarillo",
                label="Camarillo",
                status="unavailable",
                reason=f"Missing stiffness file: {stiffness_path}",
            ),
            predictions=None,
        )
    stiffness = np.loadtxt(stiffness_path, delimiter=",", dtype=float)
    if np.asarray(stiffness).shape[0] < 3:
        return ModelEvaluation(
            metrics=ModelMetrics(
                model_key="camarillo",
                label="Camarillo",
                status="unavailable",
                reason=f"Invalid stiffness file: {stiffness_path}",
            ),
            predictions=None,
        )
    ka = float(stiffness[0])
    kb = float(stiffness[1])
    kt = float(stiffness[2])
    predictions = np.zeros_like(truths)
    for index, command in enumerate(inputs):
        camarillo_params = _camarillo_no_slack(
            delta_lengths=[tuple(float(value) for value in command)],
            cable_positions=[geometry.cable_positions_mm],
            segment_stiffness_vals=[(ka, kb)],
            cable_stiffness_vals=[(kt, kt, kt, kt)],
            segment_lengths=[float(geometry.segment_length_mm)],
            additional_cable_length=float(geometry.additional_cable_length_mm),
        )
        webster = _camarillo_to_webster(camarillo_params.reshape(-1), [float(geometry.segment_length_mm)])
        transform = _calculate_transform(webster[:3])
        predictions[index, :3] = transform[:3, 3]
        predictions[index, 3:] = transform[:3, 2]
    return _complete_model_evaluation(
        model_key="camarillo",
        label="Camarillo",
        predictions=predictions,
        truths=truths,
        phases=phases,
    )


def _evaluate_ann(
    *,
    inputs: np.ndarray,
    truths: np.ndarray,
    phases: list[str],
    artifact_details: ArtifactDetails | None,
) -> ModelEvaluation:
    if artifact_details is None:
        return ModelEvaluation(
            metrics=ModelMetrics(
                model_key="ann",
                label="ANN",
                status="unavailable",
                reason="No ANN artifact selected.",
            ),
            predictions=None,
        )
    model_path = artifact_details.summary.model_path
    if model_path is None or not Path(model_path).exists():
        return ModelEvaluation(
            metrics=ModelMetrics(
                model_key="ann",
                label="ANN",
                status="unavailable",
                reason="Selected ANN artifact does not include `model.pt`.",
            ),
            predictions=None,
        )
    try:
        torch = _require_torch()
    except TorchUnavailableError as exc:
        return ModelEvaluation(
            metrics=ModelMetrics(
                model_key="ann",
                label="ANN",
                status="unavailable",
                reason=str(exc),
            ),
            predictions=None,
        )
    metadata = dict(artifact_details.metadata or {})
    model_payload = dict(metadata.get("model", {}) or {})
    hidden_layers = [int(value) for value in model_payload.get("hidden_layers", [32, 32]) or [32, 32]]
    dtype = _torch_dtype(torch, str(model_payload.get("dtype", "float64") or "float64"))
    model = _build_legacy_ann_model(
        torch=torch,
        input_dim=int(model_payload.get("input_dim", LEGACY_FULL_POSE_INPUT_DIM) or LEGACY_FULL_POSE_INPUT_DIM),
        output_dim=int(model_payload.get("output_dim", LEGACY_FULL_POSE_OUTPUT_DIM) or LEGACY_FULL_POSE_OUTPUT_DIM),
        hidden_layers=hidden_layers,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    state_dict = torch.load(Path(model_path), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    with torch.inference_mode():
        tensor_inputs = torch.tensor(inputs, dtype=dtype, device=torch.device("cpu"))
        predictions = model(tensor_inputs).detach().cpu().numpy()
    return _complete_model_evaluation(
        model_key="ann",
        label="ANN",
        predictions=np.asarray(predictions, dtype=float),
        truths=truths,
        phases=phases,
    )


def _complete_model_evaluation(
    *,
    model_key: str,
    label: str,
    predictions: np.ndarray,
    truths: np.ndarray,
    phases: list[str],
) -> ModelEvaluation:
    predictions = np.asarray(predictions, dtype=float)
    truths = np.asarray(truths, dtype=float)
    position_error_vectors = predictions[:, :3] - truths[:, :3]
    position_errors = np.linalg.norm(position_error_vectors, axis=1)
    tangent_errors = _tangent_errors_deg(predictions[:, 3:], truths[:, 3:])
    axis_position_rmse = np.sqrt(np.mean(np.square(position_error_vectors), axis=0))
    axis_tangent_rmse = np.sqrt(np.mean(np.square(predictions[:, 3:] - truths[:, 3:]), axis=0))
    phase_metrics = _phase_metrics(
        phases=phases,
        position_errors=position_errors,
        tangent_errors=tangent_errors,
    )
    metrics = ModelMetrics(
        model_key=model_key,
        label=label,
        status="completed",
        sample_count=int(predictions.shape[0]),
        position_rmse_mm=float(np.sqrt(np.mean(np.square(position_errors)))) if position_errors.size else None,
        mean_position_error_mm=float(np.mean(position_errors)) if position_errors.size else None,
        max_position_error_mm=float(np.max(position_errors)) if position_errors.size else None,
        tangent_mean_error_deg=float(np.mean(tangent_errors)) if tangent_errors.size else None,
        tangent_rmse_deg=float(np.sqrt(np.mean(np.square(tangent_errors)))) if tangent_errors.size else None,
        tangent_max_error_deg=float(np.max(tangent_errors)) if tangent_errors.size else None,
        axis_position_rmse_mm=[float(value) for value in axis_position_rmse],
        axis_tangent_rmse=[float(value) for value in axis_tangent_rmse],
        phase_metrics=phase_metrics,
    )
    return ModelEvaluation(
        metrics=metrics,
        predictions=predictions,
        position_errors_mm=[float(value) for value in position_errors],
        tangent_errors_deg=[float(value) for value in tangent_errors],
    )


def _phase_metrics(
    *,
    phases: list[str],
    position_errors: np.ndarray,
    tangent_errors: np.ndarray,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[int]] = {}
    for index, phase in enumerate(phases):
        grouped.setdefault(str(phase or "unknown"), []).append(index)
    payload: dict[str, dict[str, float]] = {}
    for phase, indices in grouped.items():
        payload[phase] = {
            "sample_count": float(len(indices)),
            "position_rmse_mm": float(np.sqrt(np.mean(np.square(position_errors[indices])))),
            "mean_position_error_mm": float(np.mean(position_errors[indices])),
            "tangent_mean_error_deg": float(np.mean(tangent_errors[indices])),
        }
    return payload


def _write_comparison_csv(path: Path, evaluations: dict[str, ModelEvaluation]) -> None:
    rows = [_metrics_to_payload(evaluation.metrics) for evaluation in evaluations.values()]
    fieldnames = [
        "model_key",
        "label",
        "status",
        "reason",
        "sample_count",
        "position_rmse_mm",
        "mean_position_error_mm",
        "max_position_error_mm",
        "tangent_mean_error_deg",
        "tangent_rmse_deg",
        "tangent_max_error_deg",
        "axis_position_rmse_mm",
        "axis_tangent_rmse",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _phase_rows(evaluations: dict[str, ModelEvaluation]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evaluation in evaluations.values():
        for phase, payload in sorted(evaluation.metrics.phase_metrics.items()):
            rows.append(
                {
                    "model_key": evaluation.metrics.model_key,
                    "label": evaluation.metrics.label,
                    "phase": phase,
                    "sample_count": payload.get("sample_count"),
                    "position_rmse_mm": payload.get("position_rmse_mm"),
                    "mean_position_error_mm": payload.get("mean_position_error_mm"),
                    "tangent_mean_error_deg": payload.get("tangent_mean_error_deg"),
                }
            )
    return rows


def _write_phase_csv(path: Path, evaluations: dict[str, ModelEvaluation]) -> None:
    rows = _phase_rows(evaluations)
    fieldnames = [
        "model_key",
        "label",
        "phase",
        "sample_count",
        "position_rmse_mm",
        "mean_position_error_mm",
        "tangent_mean_error_deg",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _allocate_results_dir(*, project_root: Path, results_root_raw: str, dataset_name: str) -> Path:
    results_root = Path(results_root_raw)
    if not results_root.is_absolute():
        results_root = Path(project_root) / results_root
    results_root.mkdir(parents=True, exist_ok=True)
    return canonical_timestamped_path(results_root, _slugify(dataset_name))


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value or "").strip())
    return cleaned.strip("_").lower() or "modeling_eval"


def _metrics_to_payload(metrics: ModelMetrics) -> dict[str, Any]:
    return {
        "model_key": metrics.model_key,
        "label": metrics.label,
        "status": metrics.status,
        "reason": metrics.reason,
        "sample_count": metrics.sample_count,
        "position_rmse_mm": metrics.position_rmse_mm,
        "mean_position_error_mm": metrics.mean_position_error_mm,
        "max_position_error_mm": metrics.max_position_error_mm,
        "tangent_mean_error_deg": metrics.tangent_mean_error_deg,
        "tangent_rmse_deg": metrics.tangent_rmse_deg,
        "tangent_max_error_deg": metrics.tangent_max_error_deg,
        "axis_position_rmse_mm": json.dumps(metrics.axis_position_rmse_mm),
        "axis_tangent_rmse": json.dumps(metrics.axis_tangent_rmse),
    }


def _histogram(values: np.ndarray, *, bins: int) -> dict[str, list[float] | list[str]]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"labels": [], "counts": []}
    if np.allclose(values.max(initial=0.0), values.min(initial=0.0)):
        minimum = float(values.min(initial=0.0))
        maximum = minimum + 1.0
    else:
        minimum = float(values.min(initial=0.0))
        maximum = float(values.max(initial=0.0))
    counts, edges = np.histogram(values, bins=bins, range=(minimum, maximum))
    labels = [
        f"{edges[index]:.2f}-{edges[index + 1]:.2f}"
        for index in range(len(edges) - 1)
    ]
    return {"labels": labels, "counts": [float(value) for value in counts]}


def _tangent_errors_deg(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    pred_norm = _normalize_rows(predicted)
    truth_norm = _normalize_rows(truth)
    dots = np.sum(pred_norm * truth_norm, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    return values / norms


def _mike_forward_webster(*, command: np.ndarray, geometry: ModelingGeometryConfig) -> np.ndarray:
    delta_la = float(command[0])
    delta_lb = float(command[1])
    pos_a = geometry.cable_positions_mm[0]
    pos_b = geometry.cable_positions_mm[1]
    phi_numerator = delta_la * pos_b[0] - delta_lb * pos_a[0]
    phi_denominator = delta_lb * pos_a[1] - delta_la * pos_b[1]
    phi = math.atan2(phi_numerator, phi_denominator)
    theta_numerator = (delta_la**2) + (delta_lb**2)
    theta_denominator = ((pos_a[0] * math.cos(phi) + pos_a[1] * math.sin(phi)) ** 2) + (
        (pos_b[0] * math.cos(phi) + pos_b[1] * math.sin(phi)) ** 2
    )
    theta = math.sqrt(theta_numerator / theta_denominator) if theta_denominator != 0 else 0.0
    kappa = theta / float(geometry.segment_length_mm) if geometry.segment_length_mm else 0.0
    return np.asarray([float(geometry.segment_length_mm), kappa, phi], dtype=float)


def _camarillo_no_slack(
    *,
    delta_lengths: list[tuple[float, ...]],
    cable_positions: list[tuple[tuple[float, ...], ...]],
    segment_stiffness_vals: list[tuple[float, ...]],
    cable_stiffness_vals: list[tuple[float, ...]],
    segment_lengths: list[float],
    additional_cable_length: float,
) -> np.ndarray:
    num_segments = len(delta_lengths)
    cables_per_segment = [len(values) for values in delta_lengths]
    k_m_inv_diag: list[float] = []
    l0_diag: list[float] = []
    lt_diag: list[float] = []
    kt_inv_diag: list[float] = []
    d_blocks: list[np.ndarray] = []
    for segment_index in range(num_segments):
        ka, kb = segment_stiffness_vals[segment_index]
        k_m_inv_diag.extend([1.0 / kb, 1.0 / kb, 1.0 / ka])
        positions = cable_positions[segment_index]
        d_blocks.append(
            np.concatenate(
                [
                    np.asarray([[position[0] for position in positions]], dtype=float),
                    np.asarray([[position[1] for position in positions]], dtype=float),
                    np.ones((1, cables_per_segment[segment_index]), dtype=float),
                ],
                axis=0,
            )
        )
        l0_diag.extend([float(segment_lengths[segment_index])] * 3)
        lt_diag.extend(
            [float(additional_cable_length + sum(segment_lengths[: segment_index + 1]))]
            * cables_per_segment[segment_index]
        )
        kt_inv_diag.extend([1.0 / float(value) for value in cable_stiffness_vals[segment_index]])
    total_cables = sum(cables_per_segment)
    d_matrix = np.zeros((3 * num_segments, total_cables), dtype=float)
    for root_index in range(num_segments):
        for segment_index in range(root_index, num_segments):
            cable_offset = sum(cables_per_segment[:segment_index])
            count = cables_per_segment[segment_index]
            d_matrix[3 * root_index : 3 * root_index + 3, cable_offset : cable_offset + count] = d_blocks[segment_index]
    k_m_inv = np.diag(np.asarray(k_m_inv_diag, dtype=float))
    l0 = np.diag(np.asarray(l0_diag, dtype=float))
    lt = np.diag(np.asarray(lt_diag, dtype=float))
    kt_inv = np.diag(np.asarray(kt_inv_diag, dtype=float))
    y = np.concatenate(
        [-np.asarray(values, dtype=float).reshape((-1, 1)) for values in delta_lengths],
        axis=0,
    )
    c_m = (d_matrix.T @ l0 @ k_m_inv @ d_matrix) + (lt @ kt_inv)
    a_matrix = k_m_inv @ d_matrix @ np.linalg.inv(c_m)
    return a_matrix @ y


def _camarillo_to_webster(camarillo_params: np.ndarray, segment_lengths: list[float]) -> np.ndarray:
    params = np.asarray(camarillo_params, dtype=float).reshape((-1,))
    if params.size % 3 != 0:
        raise ValueError("Expected Camarillo parameters in groups of three.")
    webster: list[float] = []
    for segment_index in range(len(segment_lengths)):
        kappa_x = float(params[3 * segment_index])
        kappa_y = float(params[3 * segment_index + 1])
        axial_strain = float(params[3 * segment_index + 2])
        kappa = math.sqrt((kappa_x**2) + (kappa_y**2))
        phi = math.atan2(kappa_y, kappa_x)
        length = (1.0 - axial_strain) * float(segment_lengths[segment_index])
        webster.extend([length, kappa, phi])
    return np.asarray(webster, dtype=float)


def _calculate_transform(webster_params: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    for theta, d, r, alpha in _dh_parameters(np.asarray(webster_params, dtype=float).reshape((3,))):
        transform = transform @ _dh_transform(theta=theta, d=d, r=r, alpha=alpha)
    return transform


def _dh_parameters(webster_params: np.ndarray) -> list[tuple[float, float, float, float]]:
    length = float(webster_params[0])
    kappa = float(webster_params[1])
    phi = float(webster_params[2])
    if abs(kappa) <= 1e-12:
        return [(0.0, length, 0.0, 0.0)]
    return [
        (phi, 0.0, 0.0, -math.pi / 2.0),
        (kappa * length / 2.0, 0.0, 0.0, math.pi / 2.0),
        (0.0, (2.0 / kappa) * math.sin(kappa * length / 2.0), 0.0, -math.pi / 2.0),
        (kappa * length / 2.0, 0.0, 0.0, math.pi / 2.0),
        (-phi, 0.0, 0.0, 0.0),
    ]


def _dh_transform(*, theta: float, d: float, r: float, alpha: float) -> np.ndarray:
    z_transform = np.asarray(
        [
            [math.cos(theta), -math.sin(theta), 0.0, 0.0],
            [math.sin(theta), math.cos(theta), 0.0, 0.0],
            [0.0, 0.0, 1.0, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    x_transform = np.asarray(
        [
            [1.0, 0.0, 0.0, r],
            [0.0, math.cos(alpha), -math.sin(alpha), 0.0],
            [0.0, math.sin(alpha), math.cos(alpha), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return z_transform @ x_transform


def _write_plots(
    *,
    output_dir: Path,
    truths: np.ndarray,
    evaluations: dict[str, ModelEvaluation],
    phases: list[str],
) -> dict[str, Path]:
    plot_paths = {
        "workspace_xy": output_dir / "workspace_xy.png",
        "position_histograms": output_dir / "position_histograms.png",
        "comparison_summary": output_dir / "comparison_summary.png",
        "model_workspace_prediction_report": output_dir / "model_workspace_prediction_report.png",
        "model_comparison_summary_report": output_dir / "model_comparison_summary_report.png",
    }
    try:
        _write_workspace_plot(plot_paths["workspace_xy"], truths, evaluations)
        _write_histogram_plot(plot_paths["position_histograms"], evaluations)
        _write_comparison_plot(plot_paths["comparison_summary"], evaluations, phases)
        _write_workspace_plot(plot_paths["model_workspace_prediction_report"], truths, evaluations)
        _write_comparison_plot(plot_paths["model_comparison_summary_report"], evaluations, phases)
    except Exception:
        for path in plot_paths.values():
            _write_plot_placeholder(path)
    return plot_paths


def _ensure_plot_qt_app() -> None:
    app = QApplication.instance()
    if app is None:
        QApplication([])


def _write_workspace_plot(path: Path, truths: np.ndarray, evaluations: dict[str, ModelEvaluation]) -> None:
    measured = truths[:, :2] if truths.size else np.zeros((0, 2))
    fig, ax = create_figure(size="square")
    if measured.size:
        ax.scatter(measured[:, 0], measured[:, 1], s=18, color=color("measured"), alpha=0.55, linewidths=0, label="Measured")
    for evaluation, series_color in zip(
        [value for value in evaluations.values() if value.metrics.status == "completed"],
        [color("prediction"), color("fit"), color("target")],
    ):
        points = evaluation.predictions[:, :2] if evaluation.predictions is not None else np.zeros((0, 2))
        if points.size:
            ax.scatter(points[:, 0], points[:, 1], s=12, color=series_color, alpha=0.42, linewidths=0, label=evaluation.metrics.label)
    all_points = [tuple(row) for row in measured.tolist()]
    for evaluation in evaluations.values():
        if evaluation.predictions is not None:
            all_points.extend(tuple(row[:2]) for row in evaluation.predictions.tolist())
    if all_points:
        set_equal_xy(ax, x_values=[point[0] for point in all_points], y_values=[point[1] for point in all_points], minimum_span=5.0)
    else:
        ax.text(0.5, 0.5, "No model comparison points available", transform=ax.transAxes, ha="center", va="center")
    style_axes(ax, title="Model Prediction Workspace", xlabel="Robot-frame X position (mm)", ylabel="Robot-frame Y position (mm)")
    legend(ax, loc="best")
    save_figure(fig, path)


def _write_histogram_plot(path: Path, evaluations: dict[str, ModelEvaluation]) -> None:
    completed = [value for value in evaluations.values() if value.metrics.status == "completed"]
    plt = import_matplotlib()

    fig, axes = plt.subplots(max(1, len(completed)), 1, figsize=(7.2, max(3.0, 2.4 * max(1, len(completed)))), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    if not completed:
        axes[0].text(0.5, 0.5, "No completed model evaluations available", transform=axes[0].transAxes, ha="center", va="center")
        style_axes(axes[0], title="Position Error Distribution", xlabel="Position error (mm)", ylabel="Count")
    for ax, evaluation in zip(axes, completed):
        values = np.asarray(evaluation.position_errors_mm, dtype=float)
        ax.hist(values, bins=12, color=color("measured"), alpha=0.85, edgecolor="white")
        style_axes(ax, title=f"{evaluation.metrics.label} Position Error", xlabel="Position error (mm)", ylabel="Count")
        add_metric_box(ax, [f"RMSE: {float(evaluation.metrics.position_rmse_mm or 0.0):.2f} mm"], loc="upper right")
    save_figure(fig, path)


def _write_comparison_plot(path: Path, evaluations: dict[str, ModelEvaluation], phases: list[str]) -> None:
    completed = [value for value in evaluations.values() if value.metrics.status == "completed"]
    labels = [value.metrics.label for value in completed]
    plt = import_matplotlib()

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), constrained_layout=True)
    axes[0].bar(labels, [float(value.metrics.position_rmse_mm or 0.0) for value in completed], color=color("measured"))
    style_axes(axes[0], title="Model Position RMSE", xlabel="Model", ylabel="Position RMSE (mm)")
    axes[1].bar(labels, [float(value.metrics.tangent_mean_error_deg or 0.0) for value in completed], color=color("target"))
    style_axes(axes[1], title="Mean Tangent Error", xlabel="Model", ylabel="Tangent error (deg)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    save_figure(fig, path)


def _draw_panel(painter: QPainter, rect: QRectF, title: str) -> None:
    painter.save()
    painter.setPen(QPen(QColor("#334155"), 1.2))
    painter.setBrush(QColor("#111827"))
    painter.drawRoundedRect(rect, 18.0, 18.0)
    painter.setPen(QColor("#f8fafc"))
    painter.drawText(QRectF(rect.left() + 16.0, rect.top() + 12.0, rect.width() - 32.0, 20.0), title)
    painter.restore()


def _draw_legend(painter: QPainter, rect: QRectF, entries: list[tuple[str, str]]) -> None:
    painter.save()
    y = rect.top()
    for label, color in entries:
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(color))
        painter.drawEllipse(QPointF(rect.left() + 8.0, y + 8.0), 5.0, 5.0)
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(QRectF(rect.left() + 24.0, y, rect.width() - 24.0, 20.0), Qt.AlignLeft, label)
        y += 26.0
    painter.restore()


def _draw_xy_series(painter: QPainter, rect: QRectF, series: list[tuple[str, np.ndarray, str]]) -> None:
    all_points = np.concatenate([points for _name, points, _color in series if points.size], axis=0) if any(points.size for _name, points, _color in series) else np.zeros((0, 2))
    if all_points.size == 0:
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(rect, Qt.AlignCenter, "No points available.")
        return
    min_x = float(np.min(all_points[:, 0]))
    max_x = float(np.max(all_points[:, 0]))
    min_y = float(np.min(all_points[:, 1]))
    max_y = float(np.max(all_points[:, 1]))
    if abs(max_x - min_x) < 1e-6:
        min_x -= 1.0
        max_x += 1.0
    if abs(max_y - min_y) < 1e-6:
        min_y -= 1.0
        max_y += 1.0
    painter.save()
    painter.setPen(QPen(QColor("#334155"), 1.0))
    painter.drawRect(rect)
    for _name, points, color in series:
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(color))
        for x_value, y_value in points:
            point = _map_xy(float(x_value), float(y_value), rect, min_x, max_x, min_y, max_y)
            painter.drawEllipse(point, 4.0, 4.0)
    painter.restore()


def _draw_bar_values(
    painter: QPainter,
    rect: QRectF,
    *,
    labels: list[str],
    values: list[float],
    color: str,
    y_label: str,
) -> None:
    painter.save()
    if not labels:
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(rect, Qt.AlignCenter, "No values available.")
        painter.restore()
        return
    plot_rect = rect.adjusted(0.0, 0.0, 0.0, -32.0)
    painter.setPen(QPen(QColor("#334155"), 1.0))
    painter.drawRect(plot_rect)
    max_value = max(max(values, default=0.0), 1e-6)
    bar_width = plot_rect.width() / max(1, len(values))
    painter.setBrush(QColor(color))
    painter.setPen(Qt.NoPen)
    for index, value in enumerate(values):
        height = (float(value) / max_value) * max(1.0, plot_rect.height() - 20.0)
        left = plot_rect.left() + (index * bar_width) + 6.0
        width = max(10.0, bar_width - 12.0)
        top = plot_rect.bottom() - height
        painter.drawRoundedRect(QRectF(left, top, width, height), 6.0, 6.0)
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(QRectF(left, plot_rect.bottom() + 6.0, width, 24.0), Qt.AlignCenter | Qt.TextWordWrap, labels[index])
        painter.setPen(Qt.NoPen)
    painter.setPen(QColor("#94a3b8"))
    painter.drawText(QRectF(rect.left(), rect.top() - 20.0, rect.width(), 18.0), Qt.AlignRight, y_label)
    painter.restore()


def _draw_line_series(
    painter: QPainter,
    rect: QRectF,
    series: list[tuple[str, list[tuple[float, float]], str]],
    *,
    x_axis_labels: list[str],
    y_label: str,
) -> None:
    all_points = [point for _name, points, _color in series for point in points]
    if not all_points:
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(rect, Qt.AlignCenter, "No per-phase values available.")
        return
    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    if abs(max_x - min_x) < 1e-6:
        min_x -= 1.0
        max_x += 1.0
    if abs(max_y - min_y) < 1e-6:
        min_y = 0.0
        max_y += 1.0
    painter.save()
    painter.setPen(QPen(QColor("#334155"), 1.0))
    painter.drawRect(rect)
    for series_index, (name, points, color) in enumerate(series):
        painter.setPen(QPen(QColor(color), 2.0))
        previous = None
        for x_value, y_value in points:
            point = _map_xy(float(x_value), float(y_value), rect, min_x, max_x, min_y, max_y)
            if previous is not None:
                painter.drawLine(previous, point)
            painter.setBrush(QColor(color))
            painter.drawEllipse(point, 4.0, 4.0)
            previous = point
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(
            QRectF(rect.right() - 140.0, rect.top() + 10.0 + (18.0 * series_index), 130.0, 16.0),
            Qt.AlignRight,
            name,
        )
    painter.setPen(QColor("#94a3b8"))
    painter.drawText(QRectF(rect.left(), rect.top() - 20.0, rect.width(), 18.0), Qt.AlignRight, y_label)
    for index, label in enumerate(x_axis_labels):
        x_value = float(index)
        point = _map_xy(x_value, min_y, rect, min_x, max_x, min_y, max_y)
        painter.drawText(QRectF(point.x() - 30.0, rect.bottom() + 6.0, 60.0, 28.0), Qt.AlignCenter | Qt.TextWordWrap, label)
    painter.restore()


def _map_xy(x_value: float, y_value: float, rect: QRectF, min_x: float, max_x: float, min_y: float, max_y: float) -> QPointF:
    x_ratio = (x_value - min_x) / (max_x - min_x)
    y_ratio = (y_value - min_y) / (max_y - min_y)
    return QPointF(
        rect.left() + (rect.width() * x_ratio),
        rect.bottom() - (rect.height() * y_ratio),
    )


def _write_plot_placeholder(path: Path) -> None:
    path.write_bytes(
        bytes(
            (
                137, 80, 78, 71, 13, 10, 26, 10,
                0, 0, 0, 13, 73, 72, 68, 82,
                0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0, 0, 0, 31, 21, 196, 137,
                0, 0, 0, 13, 73, 68, 65, 84, 120, 156, 99, 96, 0, 0, 0, 2, 0, 1, 226, 33, 188, 51,
                0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
            )
        )
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value in (None, "") else f"{float(value):.3f}"
