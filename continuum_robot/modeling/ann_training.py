"""Legacy ANN training support for canonical modeling datasets."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import csv
import json
import math
import platform
from pathlib import Path
import re
import time
from typing import Any, Callable

import numpy as np

from continuum_robot.experiments.dataset_io import canonical_experiment_output_root, canonical_timestamped_path
from continuum_robot.experiments.plotting import color, create_figure, legend, save_figure, style_axes

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
except Exception:  # pragma: no cover - optional during headless/test use
    _QT_AVAILABLE = False


TRAINING_SCHEMA_VERSION = "1.0"
DEFAULT_ARTIFACT_ROOT = "data/models/ann"
LEGACY_POSITION_THRESHOLD_MM = 128.0
LEGACY_TANGENT_THRESHOLD_RAD = float(np.pi)
LEGACY_FULL_POSE_INPUT_DIM = 4
LEGACY_FULL_POSE_OUTPUT_DIM = 6
DEFAULT_HIDDEN_LAYERS = [32, 32]
DEFAULT_POSE_SCALE = 10.0


class TorchUnavailableError(RuntimeError):
    """Raised when PyTorch-backed training features are requested but unavailable."""


@dataclass(frozen=True)
class BackendOption:
    """One candidate training backend."""

    name: str
    label: str
    available: bool
    recommended: bool
    dtype: str
    reason: str = ""


@dataclass(frozen=True)
class BackendReport:
    """Resolved backend availability and recommendation."""

    python_version: str
    platform_summary: str
    torch_available: bool
    torch_version: str | None
    selected_backend: str
    recommended_backend: str
    selected_dtype: str
    backend_options: list[BackendOption] = field(default_factory=list)


@dataclass(frozen=True)
class ModelingDatasetSummary:
    """Compact summary for one canonical modeling dataset run."""

    path: Path
    run_id: str
    run_name: str
    timestamp_utc: str
    status: str
    dataset_mode: str
    dataset_mode_summary: str
    run_label: str
    dataset_tag: str
    sample_count: int
    accepted_count: int
    rejected_count: int
    acceptance_rate: float | None
    full_pose_available: bool
    tangent_available: bool
    sequential_context_available: bool
    export_jsonl_path: Path | None
    legacy_dat_path: Path | None
    summary_text_path: Path | None
    trust_summary: str
    runtime_tip_summary: str
    pretension_summary: str
    metadata_payload: dict[str, Any] = field(default_factory=dict)
    summary_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedLegacyAnnDataset:
    """Accepted and cleaned data ready for the legacy ANN."""

    summary: ModelingDatasetSummary
    inputs: np.ndarray
    outputs: np.ndarray
    step_groups: list[int]
    sequence_indices: list[int]
    filtered_reason_counts: dict[str, int]
    original_accepted_count: int


@dataclass(frozen=True)
class DatasetSplit:
    """Reproducible grouped split manifest."""

    strategy: str
    train_indices: list[int]
    validation_indices: list[int]
    test_indices: list[int]
    group_ids: list[int]


@dataclass
class AnnTrainingConfig:
    """V1 training parameters for the legacy full-pose ANN."""

    hidden_layers: list[int] = field(default_factory=lambda: list(DEFAULT_HIDDEN_LAYERS))
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 256
    loss_kind: str = "pose"
    pose_orientation_scale: float = DEFAULT_POSE_SCALE
    random_seed: int = 0
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    checkpointing: bool = False
    artifact_name: str = "legacy_ann_full_pose"
    artifact_root: str = DEFAULT_ARTIFACT_ROOT

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingEstimate:
    """Warmup-benchmark runtime estimate."""

    estimated_total_s: float
    estimated_epoch_s: float
    train_batch_time_s: float
    validation_batch_time_s: float
    benchmark_train_batches: int
    benchmark_validation_batches: int
    train_batch_count: int
    validation_batch_count: int


@dataclass(frozen=True)
class TrainingProgress:
    """Live progress callback payload."""

    epoch: int
    total_epochs: int
    train_loss: float
    validation_loss: float | None
    elapsed_s: float
    remaining_s: float | None
    status: str


@dataclass(frozen=True)
class TrainingResult:
    """Finished training run summary."""

    artifact_dir: Path
    model_path: Path | None
    metadata_path: Path
    loss_history_path: Path
    loss_plot_path: Path
    split_manifest_path: Path
    summary_text_path: Path
    status: str
    best_epoch: int | None
    best_validation_loss: float | None
    test_loss: float | None
    epochs_completed: int
    train_losses: list[float]
    validation_losses: list[float]
    estimate: TrainingEstimate | None


@dataclass(frozen=True)
class TrainedArtifactSummary:
    """List entry for a saved ANN artifact bundle."""

    path: Path
    artifact_name: str
    created_at_utc: str
    status: str
    dataset_name: str
    backend_name: str
    epochs_completed: int
    best_validation_loss: float | None
    metadata_path: Path
    model_path: Path | None


def default_artifact_root(project_root: Path) -> Path:
    """Return the default training artifact root under canonical data/."""
    return Path(project_root) / DEFAULT_ARTIFACT_ROOT


def discover_modeling_datasets(*, output_root: Path) -> list[ModelingDatasetSummary]:
    """List canonical modeling datasets from the experiment output root."""
    dataset_root = canonical_experiment_output_root(Path(output_root), "collect_pose_command_dataset")
    if not dataset_root.exists():
        return []
    items: list[ModelingDatasetSummary] = []
    for child in sorted(dataset_root.iterdir(), key=lambda value: value.name, reverse=True):
        if not child.is_dir():
            continue
        try:
            items.append(load_modeling_dataset_summary(child))
        except Exception:
            continue
    items.sort(key=lambda entry: (entry.timestamp_utc, entry.run_name), reverse=True)
    return items


def load_modeling_dataset_summary(path: Path) -> ModelingDatasetSummary:
    """Load summary metadata for one modeling dataset run."""
    run_dir = Path(path)
    metadata_payload = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    summary_payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    metrics = (
        summary_payload.get("experiment_metrics", {})
        if isinstance(summary_payload.get("experiment_metrics"), dict)
        else {}
    )
    export_rows = _load_export_rows(run_dir)
    accepted_rows = [row for row in export_rows if bool(row.get("accepted"))]
    full_pose_available = all(
        len(row.get("resolved_cable_command_cm", []) or []) == LEGACY_FULL_POSE_INPUT_DIM
        and len(row.get("tip_position_xyz_mm", []) or []) == 3
        and len(row.get("tip_tangent_xyz", []) or []) == 3
        for row in accepted_rows
    ) if accepted_rows else False
    tangent_available = all(len(row.get("tip_tangent_xyz", []) or []) == 3 for row in accepted_rows) if accepted_rows else False
    sequential_context_available = any(len(row.get("previous_pair_command_cm", []) or []) == 2 for row in export_rows)
    provenance = dict(metrics.get("run_provenance", {}) or {})
    runtime_tip = dict(provenance.get("runtime_tip_calibration", {}) or {})
    pretension = dict(provenance.get("pretension_artifact", {}) or {})
    trust_summary = (
        f"runtime tip {runtime_tip.get('trust_level', 'unknown')}, "
        f"pretension {pretension.get('status', 'unknown')}"
    )
    return ModelingDatasetSummary(
        path=run_dir,
        run_id=str(metadata_payload.get("run_id", "") or ""),
        run_name=run_dir.name,
        timestamp_utc=str(metadata_payload.get("timestamp_utc", "") or ""),
        status=str(summary_payload.get("status", "unknown") or "unknown"),
        dataset_mode=str(metrics.get("dataset_mode", "unknown") or "unknown"),
        dataset_mode_summary=str(metrics.get("dataset_mode_summary", "") or ""),
        run_label=str(metrics.get("run_label", "") or ""),
        dataset_tag=str(metrics.get("dataset_tag", "") or ""),
        sample_count=int(summary_payload.get("sample_counts", {}).get("total", len(export_rows)) or len(export_rows)),
        accepted_count=int(metrics.get("accepted_sample_count", len(accepted_rows)) or len(accepted_rows)),
        rejected_count=int(metrics.get("rejected_sample_count", max(0, len(export_rows) - len(accepted_rows))) or max(0, len(export_rows) - len(accepted_rows))),
        acceptance_rate=(
            float(metrics.get("accepted_capture_rate"))
            if metrics.get("accepted_capture_rate") is not None
            else _safe_rate(len(accepted_rows), len(export_rows))
        ),
        full_pose_available=bool(full_pose_available),
        tangent_available=bool(tangent_available),
        sequential_context_available=bool(sequential_context_available),
        export_jsonl_path=(run_dir / "modeling_dataset_export.jsonl") if (run_dir / "modeling_dataset_export.jsonl").exists() else None,
        legacy_dat_path=(run_dir / "modeling_dataset_legacy_compat.dat") if (run_dir / "modeling_dataset_legacy_compat.dat").exists() else None,
        summary_text_path=(run_dir / "modeling_dataset_summary.txt") if (run_dir / "modeling_dataset_summary.txt").exists() else None,
        trust_summary=trust_summary,
        runtime_tip_summary=f"{runtime_tip.get('mode', 'unknown')} ({runtime_tip.get('trust_level', 'unknown')})",
        pretension_summary=f"{pretension.get('active_source_type', 'unknown')} ({pretension.get('status', 'unknown')})",
        metadata_payload=metadata_payload,
        summary_payload=summary_payload,
    )


def prepare_legacy_ann_dataset(path: Path) -> PreparedLegacyAnnDataset:
    """Adapt a canonical modeling dataset into the legacy ANN full-pose contract."""
    summary = load_modeling_dataset_summary(path)
    export_rows = _load_export_rows(path)
    filtered_counts: dict[str, int] = {}
    inputs: list[np.ndarray] = []
    outputs: list[np.ndarray] = []
    step_groups: list[int] = []
    sequence_indices: list[int] = []
    accepted_count = 0
    for row in export_rows:
        if not bool(row.get("accepted")):
            continue
        accepted_count += 1
        reason = _row_filter_reason(row)
        if reason is not None:
            filtered_counts[reason] = int(filtered_counts.get(reason, 0)) + 1
            continue
        command = np.asarray(row.get("resolved_cable_command_cm", []) or [], dtype=float)
        pose = np.asarray(
            list(row.get("tip_position_xyz_mm", []) or []) + list(row.get("tip_tangent_xyz", []) or []),
            dtype=float,
        )
        inputs.append(command)
        outputs.append(pose)
        step_groups.append(int(row.get("step_index", row.get("sequence_index", len(inputs) - 1)) or 0))
        sequence_indices.append(int(row.get("sequence_index", len(inputs) - 1) or 0))

    prepared_inputs = np.stack(inputs, axis=0) if inputs else np.zeros((0, LEGACY_FULL_POSE_INPUT_DIM), dtype=float)
    prepared_outputs = np.stack(outputs, axis=0) if outputs else np.zeros((0, LEGACY_FULL_POSE_OUTPUT_DIM), dtype=float)
    return PreparedLegacyAnnDataset(
        summary=summary,
        inputs=prepared_inputs,
        outputs=prepared_outputs,
        step_groups=step_groups,
        sequence_indices=sequence_indices,
        filtered_reason_counts=filtered_counts,
        original_accepted_count=accepted_count,
    )


def build_grouped_split(prepared: PreparedLegacyAnnDataset, config: AnnTrainingConfig) -> DatasetSplit:
    """Build a leak-safer contiguous grouped split by command step."""
    validate_training_config(config)
    groups_in_order: list[int] = []
    group_to_indices: dict[int, list[int]] = {}
    for index, group_id in enumerate(prepared.step_groups):
        group_key = int(group_id)
        if group_key not in group_to_indices:
            groups_in_order.append(group_key)
            group_to_indices[group_key] = []
        group_to_indices[group_key].append(index)
    counts = _allocate_partition_counts(
        total=len(groups_in_order),
        ratios=[config.train_ratio, config.validation_ratio, config.test_ratio],
    )
    train_group_count, validation_group_count, test_group_count = counts
    train_groups = groups_in_order[:train_group_count]
    validation_groups = groups_in_order[train_group_count : train_group_count + validation_group_count]
    test_groups = groups_in_order[train_group_count + validation_group_count : train_group_count + validation_group_count + test_group_count]
    train_indices = [index for group_id in train_groups for index in group_to_indices[group_id]]
    validation_indices = [index for group_id in validation_groups for index in group_to_indices[group_id]]
    test_indices = [index for group_id in test_groups for index in group_to_indices[group_id]]
    return DatasetSplit(
        strategy="ordered_step_group",
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        group_ids=list(groups_in_order),
    )


def detect_training_backends(
    *,
    preferred_backend: str | None = None,
    platform_name: str | None = None,
) -> BackendReport:
    """Detect torch backend availability and choose a recommended backend."""
    platform_name = str(platform_name or platform.system()).lower()
    python_version = platform.python_version()
    platform_summary = f"{platform.system()} {platform.release()} | {platform.machine()}"
    try:
        torch = _require_torch()
    except TorchUnavailableError:
        options = [
            BackendOption(
                name="cpu",
                label="CPU",
                available=False,
                recommended=True,
                dtype="float64",
                reason="PyTorch is not installed.",
            ),
            BackendOption(
                name="mps",
                label="MPS",
                available=False,
                recommended=False,
                dtype="float32",
                reason="PyTorch is not installed.",
            ),
            BackendOption(
                name="cuda",
                label="CUDA",
                available=False,
                recommended=False,
                dtype="float64",
                reason="PyTorch is not installed.",
            ),
        ]
        return BackendReport(
            python_version=python_version,
            platform_summary=platform_summary,
            torch_available=False,
            torch_version=None,
            selected_backend="cpu",
            recommended_backend="cpu",
            selected_dtype="float64",
            backend_options=options,
        )
    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())
    order = ["mps", "cuda", "cpu"] if platform_name == "darwin" else ["cuda", "mps", "cpu"]
    recommended_backend = next(
        (
            candidate
            for candidate in order
            if (candidate == "cpu")
            or (candidate == "cuda" and cuda_available)
            or (candidate == "mps" and mps_available)
        ),
        "cpu",
    )
    selected_backend = str(preferred_backend or recommended_backend or "cpu").strip().lower() or recommended_backend
    option_map = {
        "cpu": BackendOption(
            name="cpu",
            label="CPU",
            available=True,
            recommended=(recommended_backend == "cpu"),
            dtype="float64",
            reason="Universal fallback backend.",
        ),
        "mps": BackendOption(
            name="mps",
            label="MPS",
            available=mps_available,
            recommended=(recommended_backend == "mps"),
            dtype="float32",
            reason=(
                "Recommended on Apple Silicon for faster local iteration."
                if mps_available
                else "MPS is unavailable on this machine or torch build."
            ),
        ),
        "cuda": BackendOption(
            name="cuda",
            label="CUDA",
            available=cuda_available,
            recommended=(recommended_backend == "cuda"),
            dtype="float64",
            reason=(
                "Recommended when a CUDA device is available."
                if cuda_available
                else "CUDA is unavailable on this machine or torch build."
            ),
        ),
    }
    if selected_backend not in option_map or not option_map[selected_backend].available:
        selected_backend = recommended_backend
    selected_dtype = option_map[selected_backend].dtype
    return BackendReport(
        python_version=python_version,
        platform_summary=platform_summary,
        torch_available=True,
        torch_version=str(getattr(torch, "__version__", "unknown")),
        selected_backend=selected_backend,
        recommended_backend=recommended_backend,
        selected_dtype=selected_dtype,
        backend_options=[option_map["cpu"], option_map["mps"], option_map["cuda"]],
    )


def validate_training_config(config: AnnTrainingConfig) -> None:
    """Validate one training configuration."""
    if not config.hidden_layers or any(int(value) <= 0 for value in config.hidden_layers):
        raise ValueError("Hidden layers must contain one or more positive integers.")
    if float(config.learning_rate) <= 0.0:
        raise ValueError("Learning rate must be positive.")
    if int(config.batch_size) <= 0:
        raise ValueError("Batch size must be positive.")
    if int(config.epochs) <= 0:
        raise ValueError("Epoch count must be positive.")
    if str(config.loss_kind).strip().lower() not in {"pose", "position", "orientation"}:
        raise ValueError("Loss kind must be one of: pose, position, orientation.")
    if float(config.pose_orientation_scale) <= 0.0:
        raise ValueError("Pose orientation scale must be positive.")
    ratios = [float(config.train_ratio), float(config.validation_ratio), float(config.test_ratio)]
    if any(value < 0.0 for value in ratios):
        raise ValueError("Split ratios must be non-negative.")
    ratio_sum = sum(ratios)
    if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("Train, validation, and test ratios must sum to 1.0.")
    if not str(config.artifact_name).strip():
        raise ValueError("Artifact name must not be empty.")


def estimate_runtime(
    *,
    prepared: PreparedLegacyAnnDataset,
    split: DatasetSplit,
    config: AnnTrainingConfig,
    backend_name: str,
    time_fn: Callable[[], float] = time.perf_counter,
) -> TrainingEstimate:
    """Warm up a real training step and estimate full runtime."""
    if prepared.inputs.shape[0] == 0:
        raise ValueError("Dataset has no accepted full-pose samples after filtering.")
    torch = _require_torch()
    backend = detect_training_backends(preferred_backend=backend_name)
    device = _torch_device(torch, backend.selected_backend)
    dtype = _torch_dtype(torch, backend.selected_dtype)
    model = _build_legacy_ann_model(
        torch=torch,
        input_dim=LEGACY_FULL_POSE_INPUT_DIM,
        output_dim=LEGACY_FULL_POSE_OUTPUT_DIM,
        hidden_layers=config.hidden_layers,
        device=device,
        dtype=dtype,
    )
    loss_module = _build_loss_module(torch=torch, config=config, device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.learning_rate))
    train_loader, validation_loader, _test_loader = _build_dataloaders(
        torch=torch,
        prepared=prepared,
        split=split,
        config=config,
        dtype=dtype,
    )
    benchmark_train_batches = min(6, len(train_loader))
    benchmark_validation_batches = min(4, len(validation_loader))
    train_iter = iter(train_loader)
    validation_iter = iter(validation_loader)
    if benchmark_train_batches == 0:
        raise ValueError("Training split is empty. Increase the dataset size or adjust the split ratios.")
    _run_train_batch(torch=torch, model=model, loss_module=loss_module, optimizer=optimizer, batch=next(train_iter), device=device)
    if benchmark_validation_batches > 0:
        _run_validation_batch(torch=torch, model=model, loss_module=loss_module, batch=next(validation_iter), device=device)
    train_iter = iter(train_loader)
    validation_iter = iter(validation_loader)
    train_elapsed = 0.0
    measured_train_batches = 0
    for _index in range(benchmark_train_batches):
        batch = next(train_iter, None)
        if batch is None:
            break
        started = time_fn()
        _run_train_batch(torch=torch, model=model, loss_module=loss_module, optimizer=optimizer, batch=batch, device=device)
        train_elapsed += max(0.0, time_fn() - started)
        measured_train_batches += 1
    validation_elapsed = 0.0
    measured_validation_batches = 0
    for _index in range(benchmark_validation_batches):
        batch = next(validation_iter, None)
        if batch is None:
            break
        started = time_fn()
        _run_validation_batch(torch=torch, model=model, loss_module=loss_module, batch=batch, device=device)
        validation_elapsed += max(0.0, time_fn() - started)
        measured_validation_batches += 1
    measured_train_batches = max(1, measured_train_batches)
    measured_validation_batches = max(1, measured_validation_batches) if benchmark_validation_batches > 0 else 1
    train_batch_time = train_elapsed / float(measured_train_batches)
    validation_batch_time = (
        validation_elapsed / float(measured_validation_batches)
        if benchmark_validation_batches > 0
        else 0.0
    )
    estimated_epoch_s = (
        float(len(train_loader)) * train_batch_time
        + float(len(validation_loader)) * validation_batch_time
    )
    return TrainingEstimate(
        estimated_total_s=float(config.epochs) * estimated_epoch_s,
        estimated_epoch_s=estimated_epoch_s,
        train_batch_time_s=train_batch_time,
        validation_batch_time_s=validation_batch_time,
        benchmark_train_batches=benchmark_train_batches,
        benchmark_validation_batches=benchmark_validation_batches,
        train_batch_count=len(train_loader),
        validation_batch_count=len(validation_loader),
    )


def train_legacy_ann(
    *,
    project_root: Path,
    dataset_path: Path,
    config: AnnTrainingConfig,
    backend_name: str,
    progress_callback: Callable[[TrainingProgress], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    time_fn: Callable[[], float] = time.perf_counter,
) -> TrainingResult:
    """Train the legacy full-pose ANN and save an artifact bundle."""
    validate_training_config(config)
    torch = _require_torch()
    prepared = prepare_legacy_ann_dataset(dataset_path)
    if prepared.inputs.shape[0] == 0:
        raise ValueError("Dataset has no accepted full-pose samples after filtering.")
    split = build_grouped_split(prepared, config)
    backend_report = detect_training_backends(preferred_backend=backend_name)
    device = _torch_device(torch, backend_report.selected_backend)
    dtype = _torch_dtype(torch, backend_report.selected_dtype)
    np.random.seed(int(config.random_seed))
    torch.manual_seed(int(config.random_seed))
    artifact_dir = _allocate_artifact_dir(
        project_root=Path(project_root),
        artifact_root_raw=config.artifact_root,
        artifact_name=config.artifact_name,
    )
    checkpoints_dir = artifact_dir / "checkpoints"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    if bool(config.checkpointing):
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
    loss_history_path = artifact_dir / "loss_history.csv"
    loss_plot_path = artifact_dir / "loss_curve.png"
    loss_report_plot_path = artifact_dir / "ann_loss_curve_report.png"
    metadata_path = artifact_dir / "training_metadata.json"
    config_path = artifact_dir / "training_config.json"
    split_manifest_path = artifact_dir / "split_manifest.json"
    summary_text_path = artifact_dir / "training_summary.txt"
    model_path = artifact_dir / "model.pt"

    estimate = estimate_runtime(
        prepared=prepared,
        split=split,
        config=config,
        backend_name=backend_report.selected_backend,
        time_fn=time_fn,
    )
    train_loader, validation_loader, test_loader = _build_dataloaders(
        torch=torch,
        prepared=prepared,
        split=split,
        config=config,
        dtype=dtype,
    )
    model = _build_legacy_ann_model(
        torch=torch,
        input_dim=LEGACY_FULL_POSE_INPUT_DIM,
        output_dim=LEGACY_FULL_POSE_OUTPUT_DIM,
        hidden_layers=config.hidden_layers,
        device=device,
        dtype=dtype,
    )
    loss_module = _build_loss_module(torch=torch, config=config, device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.learning_rate))
    train_losses: list[float] = []
    validation_losses: list[float] = []
    epoch_rows: list[dict[str, Any]] = []
    best_state_dict = None
    best_epoch: int | None = None
    best_validation_loss: float | None = None
    status = "completed"
    started_at = time_fn()
    epochs_completed = 0
    for epoch in range(1, int(config.epochs) + 1):
        if stop_requested is not None and stop_requested():
            status = "cancelled"
            break
        train_loss = _run_training_epoch(
            torch=torch,
            model=model,
            loss_module=loss_module,
            optimizer=optimizer,
            dataloader=train_loader,
            device=device,
            stop_requested=stop_requested,
        )
        if stop_requested is not None and stop_requested():
            status = "cancelled"
            break
        validation_loss = (
            _run_validation_epoch(
                torch=torch,
                model=model,
                loss_module=loss_module,
                dataloader=validation_loader,
                device=device,
            )
            if len(validation_loader) > 0
            else None
        )
        train_losses.append(float(train_loss))
        validation_losses.append(float(validation_loss) if validation_loss is not None else float("nan"))
        epochs_completed = epoch
        current_objective = float(validation_loss) if validation_loss is not None else float(train_loss)
        if best_validation_loss is None or current_objective < best_validation_loss:
            best_validation_loss = current_objective
            best_epoch = epoch
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if bool(config.checkpointing):
            torch.save(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "validation_loss": (float(validation_loss) if validation_loss is not None else None),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoints_dir / f"epoch_{epoch:04d}.pt",
            )
        elapsed_s = max(0.0, time_fn() - started_at)
        average_epoch_s = elapsed_s / float(max(1, epochs_completed))
        remaining_s = max(0.0, average_epoch_s * float(max(0, int(config.epochs) - epoch)))
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "validation_loss": (float(validation_loss) if validation_loss is not None else None),
                "elapsed_s": elapsed_s,
            }
        )
        if progress_callback is not None:
            progress_callback(
                TrainingProgress(
                    epoch=epoch,
                    total_epochs=int(config.epochs),
                    train_loss=float(train_loss),
                    validation_loss=(float(validation_loss) if validation_loss is not None else None),
                    elapsed_s=elapsed_s,
                    remaining_s=remaining_s,
                    status=status,
                )
            )
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        torch.save(best_state_dict, model_path)
    elif model_path.exists():
        model_path.unlink()
    test_loss = (
        _run_validation_epoch(
            torch=torch,
            model=model,
            loss_module=loss_module,
            dataloader=test_loader,
            device=device,
        )
        if len(test_loader) > 0 and best_state_dict is not None
        else None
    )
    _write_loss_history_csv(loss_history_path, epoch_rows)
    _write_loss_plot(loss_plot_path, train_losses, validation_losses)
    _write_loss_plot(loss_report_plot_path, train_losses, validation_losses)
    split_manifest = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "strategy": split.strategy,
        "train_indices": list(split.train_indices),
        "validation_indices": list(split.validation_indices),
        "test_indices": list(split.test_indices),
        "group_ids": list(split.group_ids),
        "sequence_indices": list(prepared.sequence_indices),
    }
    split_manifest_path.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    dataset_metadata = _dataset_metadata_for_artifact(prepared=prepared)
    metadata_payload = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "artifact_kind": "legacy_ann_full_pose_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "legacy_references": [
            "@references/ANN.py",
            "@references/real_model_learning.py",
            "@references/utils_data.py",
            "@references/multi_input.py",
            "@references/hysteresis_model_learning.py",
        ],
        "dataset": dataset_metadata,
        "model": {
            "family": "legacy_ann",
            "variant": "full_pose",
            "input_dim": LEGACY_FULL_POSE_INPUT_DIM,
            "output_dim": LEGACY_FULL_POSE_OUTPUT_DIM,
            "hidden_layers": list(config.hidden_layers),
            "activation": "relu",
            "dtype": backend_report.selected_dtype,
        },
        "loss": {
            "kind": str(config.loss_kind),
            "pose_orientation_scale": float(config.pose_orientation_scale),
        },
        "backend": {
            "selected_backend": backend_report.selected_backend,
            "recommended_backend": backend_report.recommended_backend,
            "torch_version": backend_report.torch_version,
            "python_version": backend_report.python_version,
            "platform_summary": backend_report.platform_summary,
        },
        "training": {
            "epochs_requested": int(config.epochs),
            "epochs_completed": int(epochs_completed),
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "test_loss": (float(test_loss) if test_loss is not None else None),
            "checkpointing": bool(config.checkpointing),
            "learning_rate": float(config.learning_rate),
            "batch_size": int(config.batch_size),
            "random_seed": int(config.random_seed),
        },
        "split": {
            "strategy": split.strategy,
            "train_ratio": float(config.train_ratio),
            "validation_ratio": float(config.validation_ratio),
            "test_ratio": float(config.test_ratio),
            "train_count": len(split.train_indices),
            "validation_count": len(split.validation_indices),
            "test_count": len(split.test_indices),
        },
        "estimate": (asdict(estimate) if estimate is not None else None),
        "files": {
            "model_path": (str(model_path) if model_path.exists() else None),
            "loss_history_path": str(loss_history_path),
            "loss_plot_path": str(loss_plot_path),
            "loss_report_plot_path": str(loss_report_plot_path),
            "split_manifest_path": str(split_manifest_path),
            "summary_text_path": str(summary_text_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    config_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    summary_text_path.write_text(_render_summary_text(metadata_payload), encoding="utf-8")
    return TrainingResult(
        artifact_dir=artifact_dir,
        model_path=(model_path if model_path.exists() else None),
        metadata_path=metadata_path,
        loss_history_path=loss_history_path,
        loss_plot_path=loss_plot_path,
        split_manifest_path=split_manifest_path,
        summary_text_path=summary_text_path,
        status=status,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        test_loss=(float(test_loss) if test_loss is not None else None),
        epochs_completed=epochs_completed,
        train_losses=train_losses,
        validation_losses=validation_losses,
        estimate=estimate,
    )


def discover_trained_artifacts(*, artifact_root: Path) -> list[TrainedArtifactSummary]:
    """List saved ANN training artifact bundles."""
    root = Path(artifact_root)
    if not root.exists():
        return []
    artifacts: list[TrainedArtifactSummary] = []
    for child in sorted(root.iterdir(), key=lambda value: value.name, reverse=True):
        metadata_path = child / "training_metadata.json"
        if not child.is_dir() or not metadata_path.exists():
            continue
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        training = dict(payload.get("training", {}) or {})
        dataset = dict(payload.get("dataset", {}) or {})
        backend = dict(payload.get("backend", {}) or {})
        model_path_raw = dict(payload.get("files", {}) or {}).get("model_path")
        model_path = Path(model_path_raw) if model_path_raw else None
        artifacts.append(
            TrainedArtifactSummary(
                path=child,
                artifact_name=child.name,
                created_at_utc=str(payload.get("created_at_utc", "") or ""),
                status=str(payload.get("status", "unknown") or "unknown"),
                dataset_name=str(dataset.get("run_name", dataset.get("dataset_mode", "unknown")) or "unknown"),
                backend_name=str(backend.get("selected_backend", "unknown") or "unknown"),
                epochs_completed=int(training.get("epochs_completed", 0) or 0),
                best_validation_loss=(
                    float(training["best_validation_loss"])
                    if training.get("best_validation_loss") not in (None, "")
                    else None
                ),
                metadata_path=metadata_path,
                model_path=model_path,
            )
        )
    artifacts.sort(key=lambda entry: (entry.created_at_utc, entry.artifact_name), reverse=True)
    return artifacts


def load_training_metadata(path: Path) -> dict[str, Any]:
    """Load one saved training metadata payload."""
    candidate = Path(path)
    metadata_path = candidate / "training_metadata.json" if candidate.is_dir() else candidate
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_loss_history(path: Path) -> tuple[list[float], list[float]]:
    """Load loss history from an artifact directory or CSV path."""
    candidate = Path(path)
    csv_path = candidate / "loss_history.csv" if candidate.is_dir() else candidate
    train_losses: list[float] = []
    validation_losses: list[float] = []
    if not csv_path.exists():
        return train_losses, validation_losses
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            train_losses.append(float(row.get("train_loss", "nan")))
            raw_validation = row.get("validation_loss", "")
            validation_losses.append(float(raw_validation) if raw_validation not in ("", "None", None) else float("nan"))
    return train_losses, validation_losses


def build_training_visualization(
    *,
    status: str,
    train_losses: list[float],
    validation_losses: list[float],
    metadata: dict[str, Any] | None = None,
) -> VisualizationModel:
    """Build a reusable chart payload for live or saved training history."""
    metadata = dict(metadata or {})
    training_payload = dict(metadata.get("training", {}) or {})
    summary_lines = [
        f"Status: {status}",
        f"Epochs completed: {training_payload.get('epochs_completed', len(train_losses))}",
        f"Best epoch: {training_payload.get('best_epoch', 'n/a')}",
        f"Best validation loss: {_fmt_number(training_payload.get('best_validation_loss'))}",
        f"Test loss: {_fmt_number(training_payload.get('test_loss'))}",
    ]
    charts = [
        ChartModel(
            kind="line",
            title="Train Loss",
            x_title="Epoch",
            y_title="Loss",
            points_xy=[(index + 1, float(value)) for index, value in enumerate(train_losses)],
            color_hex="#0f766e",
            caption="Legacy ANN training loss across epochs.",
        )
    ]
    if any(not math.isnan(float(value)) for value in validation_losses):
        charts.append(
            ChartModel(
                kind="line",
                title="Validation Loss",
                x_title="Epoch",
                y_title="Loss",
                points_xy=[
                    (index + 1, float(value))
                    for index, value in enumerate(validation_losses)
                    if not math.isnan(float(value))
                ],
                color_hex="#2563eb",
                caption="Validation loss for the held-out grouped split.",
            )
        )
    return VisualizationModel(charts=charts, summary_lines=summary_lines)


def estimate_memory_bytes(*, sample_count: int, batch_size: int, hidden_layers: list[int], dtype_name: str) -> int:
    """Approximate tensor memory used by dataset, model parameters, and one active batch."""
    bytes_per_value = 8 if str(dtype_name) == "float64" else 4
    layer_sizes = [LEGACY_FULL_POSE_INPUT_DIM, *[int(value) for value in hidden_layers], LEGACY_FULL_POSE_OUTPUT_DIM]
    parameter_count = 0
    for input_dim, output_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
        parameter_count += (int(input_dim) * int(output_dim)) + int(output_dim)
    dataset_values = int(sample_count) * (LEGACY_FULL_POSE_INPUT_DIM + LEGACY_FULL_POSE_OUTPUT_DIM)
    batch_values = int(batch_size) * (LEGACY_FULL_POSE_INPUT_DIM + LEGACY_FULL_POSE_OUTPUT_DIM)
    return int((dataset_values + parameter_count + (3 * batch_values)) * bytes_per_value)


def _load_export_rows(path: Path) -> list[dict[str, Any]]:
    export_path = Path(path) / "modeling_dataset_export.jsonl"
    rows: list[dict[str, Any]] = []
    if not export_path.exists():
        return rows
    with export_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
    return rows


def _row_filter_reason(row: dict[str, Any]) -> str | None:
    command = np.asarray(row.get("resolved_cable_command_cm", []) or [], dtype=float)
    position = np.asarray(row.get("tip_position_xyz_mm", []) or [], dtype=float)
    tangent = np.asarray(row.get("tip_tangent_xyz", []) or [], dtype=float)
    if command.shape != (LEGACY_FULL_POSE_INPUT_DIM,):
        return "missing_command"
    if position.shape != (3,):
        return "missing_position"
    if tangent.shape != (3,):
        return "missing_tangent"
    if not np.isfinite(command).all() or not np.isfinite(position).all() or not np.isfinite(tangent).all():
        return "non_finite"
    if np.abs(position).max(initial=0.0) > LEGACY_POSITION_THRESHOLD_MM:
        return "legacy_position_threshold"
    if np.abs(tangent).max(initial=0.0) > LEGACY_TANGENT_THRESHOLD_RAD:
        return "legacy_tangent_threshold"
    return None


def _allocate_partition_counts(*, total: int, ratios: list[float]) -> list[int]:
    if total <= 0:
        return [0 for _ in ratios]
    if len(ratios) != 3:
        raise ValueError("Expected train, validation, and test ratios.")
    positive_partitions = sum(1 for ratio in ratios if ratio > 0.0)
    base_counts = [int(math.floor(total * ratio)) for ratio in ratios]
    if total >= positive_partitions:
        for index, ratio in enumerate(ratios):
            if ratio > 0.0 and base_counts[index] == 0:
                base_counts[index] = 1
    while sum(base_counts) > total:
        candidate_index = max(
            (index for index, count in enumerate(base_counts) if count > 0),
            key=lambda index: base_counts[index],
        )
        base_counts[candidate_index] -= 1
    remainders = [
        (index, (total * ratios[index]) - math.floor(total * ratios[index]))
        for index in range(len(ratios))
    ]
    remainders.sort(key=lambda item: item[1], reverse=True)
    while sum(base_counts) < total:
        for index, _remainder in remainders:
            base_counts[index] += 1
            if sum(base_counts) == total:
                break
    return base_counts


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - import failure depends on environment
        raise TorchUnavailableError(
            "PyTorch is not installed. Install `torch` to enable ANN training."
        ) from exc
    return torch


def _torch_device(torch, backend_name: str):
    backend_name = str(backend_name or "cpu").strip().lower()
    if backend_name == "cuda":
        if not bool(torch.cuda.is_available()):
            raise ValueError("CUDA backend requested but unavailable.")
        return torch.device("cuda")
    if backend_name == "mps":
        backend = getattr(getattr(torch, "backends", None), "mps", None)
        if not bool(backend and backend.is_available()):
            raise ValueError("MPS backend requested but unavailable.")
        return torch.device("mps")
    return torch.device("cpu")


def _torch_dtype(torch, dtype_name: str):
    return torch.float64 if str(dtype_name) == "float64" else torch.float32


def _build_legacy_ann_model(*, torch, input_dim: int, output_dim: int, hidden_layers: list[int], device, dtype):
    nn = torch.nn
    layers = OrderedDict(
        [
            ("input", nn.Linear(int(input_dim), int(hidden_layers[0]))),
            ("input_activation", nn.ReLU()),
        ]
    )
    for index, (in_features, out_features) in enumerate(zip(hidden_layers[:-1], hidden_layers[1:]), start=1):
        layers[f"hidden{index}"] = nn.Linear(int(in_features), int(out_features))
        layers[f"activation{index}"] = nn.ReLU()
    layers["output"] = nn.Linear(int(hidden_layers[-1]), int(output_dim))
    model = nn.Sequential(layers)
    return model.to(device=device, dtype=dtype)


def _build_loss_module(*, torch, config: AnnTrainingConfig, device, dtype):
    loss_kind = str(config.loss_kind).strip().lower()
    if loss_kind == "position":
        return _PositionLoss(torch=torch)
    if loss_kind == "orientation":
        return _OrientationLoss(torch=torch)
    return _PoseLoss(
        torch=torch,
        scale=float(config.pose_orientation_scale),
        num_outputs=LEGACY_FULL_POSE_OUTPUT_DIM,
        device=device,
        dtype=dtype,
    )


class _PoseLoss:
    def __init__(self, *, torch, scale: float, num_outputs: int, device, dtype) -> None:
        orientation_dims = max(0, int(num_outputs) - 3)
        self.weights = torch.tensor(
            [1.0, 1.0, 1.0, *([float(scale)] * orientation_dims)],
            device=device,
            dtype=dtype,
        )
        self._torch = torch

    def __call__(self, pred, target):
        weights = self.weights.unsqueeze(0).expand(pred.size(0), -1)
        return self._torch.nn.functional.mse_loss(pred * weights, target * weights)


class _PositionLoss:
    def __init__(self, *, torch) -> None:
        self._torch = torch

    def __call__(self, pred, target):
        return self._torch.sqrt(self._torch.nn.functional.mse_loss(pred[:, :3], target[:, :3]) * 3.0)


class _OrientationLoss:
    def __init__(self, *, torch) -> None:
        self._torch = torch

    def __call__(self, pred, target):
        return self._torch.sqrt(self._torch.nn.functional.mse_loss(pred[:, 3:], target[:, 3:]) * 3.0)


def _build_dataloaders(*, torch, prepared: PreparedLegacyAnnDataset, split: DatasetSplit, config: AnnTrainingConfig, dtype):
    tensor_inputs = torch.tensor(prepared.inputs, dtype=dtype)
    tensor_outputs = torch.tensor(prepared.outputs, dtype=dtype)
    dataset = torch.utils.data.TensorDataset(tensor_inputs, tensor_outputs)
    generator = torch.Generator()
    generator.manual_seed(int(config.random_seed))
    train_subset = torch.utils.data.Subset(dataset, list(split.train_indices))
    validation_subset = torch.utils.data.Subset(dataset, list(split.validation_indices))
    test_subset = torch.utils.data.Subset(dataset, list(split.test_indices))
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config.batch_size),
        shuffle=True,
        generator=generator,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_subset,
        batch_size=int(config.batch_size),
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=int(config.batch_size),
        shuffle=False,
    )
    return train_loader, validation_loader, test_loader


def _run_train_batch(*, torch, model, loss_module, optimizer, batch, device):
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    model.train()
    predictions = model(inputs)
    loss = loss_module(predictions, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if device.type == "mps":
        getattr(torch, "mps", None).synchronize() if getattr(torch, "mps", None) is not None else None
    elif device.type == "cuda":
        torch.cuda.synchronize()
    return float(loss.item())


def _run_validation_batch(*, torch, model, loss_module, batch, device):
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(inputs)
        loss = loss_module(predictions, targets)
    if device.type == "mps":
        getattr(torch, "mps", None).synchronize() if getattr(torch, "mps", None) is not None else None
    elif device.type == "cuda":
        torch.cuda.synchronize()
    return float(loss.item())


def _run_training_epoch(*, torch, model, loss_module, optimizer, dataloader, device, stop_requested):
    losses: list[float] = []
    for batch in dataloader:
        if stop_requested is not None and stop_requested():
            break
        losses.append(
            _run_train_batch(
                torch=torch,
                model=model,
                loss_module=loss_module,
                optimizer=optimizer,
                batch=batch,
                device=device,
            )
        )
    if not losses:
        return float("nan")
    return float(sum(losses) / float(len(losses)))


def _run_validation_epoch(*, torch, model, loss_module, dataloader, device):
    losses: list[float] = []
    for batch in dataloader:
        losses.append(
            _run_validation_batch(
                torch=torch,
                model=model,
                loss_module=loss_module,
                batch=batch,
                device=device,
            )
        )
    if not losses:
        return float("nan")
    return float(sum(losses) / float(len(losses)))


def _allocate_artifact_dir(*, project_root: Path, artifact_root_raw: str, artifact_name: str) -> Path:
    artifact_root = Path(artifact_root_raw)
    if not artifact_root.is_absolute():
        artifact_root = Path(project_root) / artifact_root
    artifact_root.mkdir(parents=True, exist_ok=True)
    safe_name = _slugify(artifact_name) or "legacy_ann"
    return canonical_timestamped_path(artifact_root, safe_name)


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value or "").strip())
    return cleaned.strip("_").lower()


def _write_loss_history_csv(path: Path, epoch_rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "validation_loss", "elapsed_s"])
        writer.writeheader()
        for row in epoch_rows:
            writer.writerow(row)


def _dataset_metadata_for_artifact(*, prepared: PreparedLegacyAnnDataset) -> dict[str, Any]:
    summary = prepared.summary
    return {
        "path": str(summary.path),
        "run_id": summary.run_id,
        "run_name": summary.run_name,
        "timestamp_utc": summary.timestamp_utc,
        "dataset_mode": summary.dataset_mode,
        "dataset_mode_summary": summary.dataset_mode_summary,
        "sample_count": summary.sample_count,
        "accepted_count": summary.accepted_count,
        "rejected_count": summary.rejected_count,
        "full_pose_available": summary.full_pose_available,
        "sequential_context_available": summary.sequential_context_available,
        "run_label": summary.run_label,
        "dataset_tag": summary.dataset_tag,
        "trust_summary": summary.trust_summary,
        "runtime_tip_summary": summary.runtime_tip_summary,
        "pretension_summary": summary.pretension_summary,
        "filtered_reason_counts": dict(prepared.filtered_reason_counts),
        "prepared_sample_count": int(prepared.inputs.shape[0]),
    }


def _render_summary_text(metadata_payload: dict[str, Any]) -> str:
    dataset = dict(metadata_payload.get("dataset", {}) or {})
    backend = dict(metadata_payload.get("backend", {}) or {})
    training = dict(metadata_payload.get("training", {}) or {})
    estimate = dict(metadata_payload.get("estimate", {}) or {})
    lines = [
        "Legacy ANN Full-Pose Training Summary",
        "",
        f"Status: {metadata_payload.get('status', 'unknown')}",
        f"Created: {metadata_payload.get('created_at_utc', 'n/a')}",
        f"Dataset: {dataset.get('run_name', 'n/a')}",
        f"Dataset mode: {dataset.get('dataset_mode', 'n/a')}",
        f"Prepared samples: {dataset.get('prepared_sample_count', 'n/a')}",
        f"Backend: {backend.get('selected_backend', 'n/a')} ({backend.get('platform_summary', 'n/a')})",
        f"Epochs completed: {training.get('epochs_completed', 'n/a')}",
        f"Best epoch: {training.get('best_epoch', 'n/a')}",
        f"Best validation loss: {_fmt_number(training.get('best_validation_loss'))}",
        f"Test loss: {_fmt_number(training.get('test_loss'))}",
    ]
    if estimate:
        lines.append(f"Warmup estimate: {_fmt_seconds(estimate.get('estimated_total_s'))}")
    lines.extend(
        [
            "",
            "Legacy references:",
            "- @references/ANN.py",
            "- @references/real_model_learning.py",
            "- @references/utils_data.py",
            "- @references/multi_input.py",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _write_loss_plot(path: Path, train_losses: list[float], validation_losses: list[float]) -> None:
    fig, ax = create_figure(size="wide")
    values = [float(value) for value in train_losses if not math.isnan(float(value))]
    values.extend(float(value) for value in validation_losses if not math.isnan(float(value)))
    if not values:
        ax.text(0.5, 0.5, "No loss history was saved", transform=ax.transAxes, ha="center", va="center")
        style_axes(ax, title="ANN Training Loss", xlabel="Epoch", ylabel="Loss")
        save_figure(fig, path)
        return
    if train_losses:
        ax.plot(range(len(train_losses)), train_losses, color=color("measured"), label="Train")
    if validation_losses and any(not math.isnan(float(value)) for value in validation_losses):
        ax.plot(range(len(validation_losses)), validation_losses, color=color("fit"), label="Validation")
    style_axes(ax, title="ANN Training Loss", xlabel="Epoch", ylabel="Loss")
    legend(ax, loc="best")
    save_figure(fig, path)


def _draw_loss_polyline(*, painter: QPainter, plot_rect: QRectF, values: list[float], color: str, min_y: float, max_y: float) -> None:
    points: list[QPointF] = []
    numeric_values = [float(value) for value in values]
    if len(numeric_values) == 1:
        numeric_values = numeric_values + numeric_values
    for index, value in enumerate(numeric_values):
        if math.isnan(value):
            continue
        x_ratio = float(index) / float(max(1, len(numeric_values) - 1))
        y_ratio = (value - float(min_y)) / float(max_y - min_y)
        points.append(
            QPointF(
                plot_rect.left() + (plot_rect.width() * x_ratio),
                plot_rect.bottom() - (plot_rect.height() * y_ratio),
            )
        )
    if len(points) < 2:
        return
    painter.setPen(QPen(QColor(color), 2.2))
    for start, end in zip(points[:-1], points[1:]):
        painter.drawLine(start, end)


def _ensure_qt_app() -> None:
    if not _QT_AVAILABLE:
        return
    app = QApplication.instance()
    if app is None:
        QApplication([])


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


def _fmt_number(value: Any) -> str:
    if value in (None, "", "None"):
        return "n/a"
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def _fmt_seconds(value: Any) -> str:
    if value in (None, "", "None"):
        return "n/a"
    total_s = float(value)
    if total_s < 60.0:
        return f"{total_s:.1f} s"
    minutes, seconds = divmod(total_s, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {seconds:04.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m"
