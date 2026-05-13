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
from typing import Any, Callable, Sequence

import numpy as np

from continuum_robot.experiments.dataset_io import canonical_experiment_output_root, canonical_timestamped_path
from continuum_robot.experiments.plotting import color, create_figure, import_matplotlib, legend, save_figure, style_axes

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


def format_hidden_layers_for_ann_ui(layers: list[int]) -> str:
    """Canonical comma+space display for ANN hidden-layer lists (GUI + config text)."""
    return ", ".join(str(int(x)) for x in layers)


def parse_hidden_layers_text(raw: str) -> tuple[list[int] | None, str | None]:
    """Parse comma/semicolon-separated positive integers for ANN hidden layers.

    Returns ``(layers, None)`` on success, or ``(None, error_message)`` on failure.
    """
    stripped = str(raw or "").strip()
    if not stripped:
        return None, "Hidden layers must not be empty."
    try:
        segments = [segment.strip() for segment in stripped.replace(";", ",").split(",")]
        parsed = [int(value) for value in segments if value]
    except ValueError:
        return None, "Hidden layers must be a comma-separated list of integers."
    if not parsed:
        return None, "Hidden layers must contain at least one integer."
    if any(int(value) <= 0 for value in parsed):
        return None, "Each hidden layer size must be a positive integer."
    return parsed, None


def parse_sweep_extra_hidden_layers_groups(raw: str) -> tuple[list[list[int]] | None, str | None]:
    """Parse optional extra ANN architectures for model sweeps.

    Groups are separated by ``|`` or newlines. Each group is comma/semicolon-separated widths,
    e.g. ``"48,48 | 96"`` → ``[[48, 48], [96]]``.
    """
    stripped = str(raw or "").strip()
    if not stripped:
        return [], None
    groups_out: list[list[int]] = []
    for segment in re.split(r"[\n|]+", stripped):
        piece = segment.strip()
        if not piece:
            continue
        layers, err = parse_hidden_layers_text(piece)
        if err or layers is None:
            return None, err or f"Invalid sweep architecture group: {piece!r}"
        groups_out.append(list(layers))
    return groups_out, None


DEFAULT_POSE_SCALE = 10.0

MODEL_SWEEP_DEFAULT_ANN_HIDDEN_LAYERS: tuple[tuple[int, ...], ...] = ((32, 32), (64, 64), (128, 128))

MODEL_SWEEP_SINGLE_SPLIT_WARNING = (
    "All models were trained and compared on one grouped train/validation/test split from this "
    "dataset only. Metrics do not assess cross-session or cross-run generalization."
)

COLLECT_POSE_COMMAND_DATASET = "collect_pose_command_dataset"

# Default ANN catalog policy: training is blocked for these trust modes unless UI enables lower-trust debug training.
_LEGACY_ANN_LOW_TRUST_MODES = frozenset({"lower_trust", "debug"})
_LEGACY_ANN_SERVO_BLOCKED_MODES = frozenset({"servo_only"})


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
    dataset_scan_root: str = "experiments"
    catalog_experiment_name: str = ""
    mock_mode_flag: bool | None = None
    run_trust_mode: str = "unknown"
    valid_for_model_training_flag: bool | None = None
    structurally_ready_for_legacy_ann: bool = False
    accepted_legacy_trainable_count: int = 0
    trainable_for_legacy_ann: bool = False
    legacy_ann_rejection_reasons: tuple[str, ...] = ()
    discovery_warnings: tuple[str, ...] = ()
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
    model_sweep_include_linear_baseline: bool = True
    model_sweep_extra_hidden_layers_text: str = ""
    linear_ridge_alpha: float = 1e-6

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
class ModelSweepResult:
    """Outputs from :func:`run_model_sweep`."""

    sweep_root: Path
    summary_json_path: Path
    summary_csv_path: Path
    summary_txt_path: Path
    comparison_png_path: Path | None
    rows: tuple[dict[str, Any], ...]
    best_model: dict[str, Any] | None
    warnings: tuple[str, ...]


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


def _is_data_trash_path(path: Path) -> bool:
    """True if the path lives under ``data/trash`` (operators use Trash for quarantine)."""
    try:
        resolved = Path(path).resolve()
    except Exception:
        resolved = Path(path)
    parts = resolved.parts
    for index, part in enumerate(parts):
        if part == "data" and index + 1 < len(parts) and parts[index + 1] == "trash":
            return True
    return False


def _coalesce_model_training_valid(*values: Any) -> bool | None:
    for value in values:
        if value is None:
            continue
        return bool(value)
    return None


def _read_mock_mode_flag(metadata_payload: dict[str, Any], metrics: dict[str, Any]) -> bool | None:
    mp = dict(metadata_payload.get("run_provenance", {}) or {})
    rp = dict(metrics.get("run_provenance", {}) or {})
    raw = (
        mp.get("mock_mode")
        if mp.get("mock_mode") is not None
        else rp.get("mock_mode")
        if rp.get("mock_mode") is not None
        else _as_dict(metadata_payload.get("backend_info")).get("mock_mode")
    )
    if raw is None:
        return None
    return bool(raw)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _read_run_trust_mode(metadata_payload: dict[str, Any], metrics: dict[str, Any]) -> str:
    run_trust = _as_dict(metrics.get("run_trust"))
    meta_trust = _as_dict(metadata_payload.get("trust_info"))
    raw = (
        metrics.get("run_trust_mode")
        if metrics.get("run_trust_mode") is not None
        else run_trust.get("run_trust_mode")
        if run_trust.get("run_trust_mode") is not None
        else meta_trust.get("run_trust_mode")
    )
    return str(raw or "unknown").strip().lower() or "unknown"


def _finalize_legacy_ann_trainability(
    *,
    dataset_scan_root: str,
    structurally_ready_for_legacy_ann: bool,
    export_jsonl_path: Path | None,
    accepted_legacy_trainable_count: int,
    mock_mode_flag: bool | None,
    run_trust_mode: str,
    valid_for_model_training_flag: bool | None,
    full_pose_available: bool,
    accepted_count: int,
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []

    if dataset_scan_root == "mock":
        reasons.append("mock run (under data/mock_experiments)")

    if mock_mode_flag is True:
        reasons.append("mock run")

    if export_jsonl_path is None:
        reasons.append("No modeling_dataset_export.jsonl")

    if export_jsonl_path is not None and accepted_legacy_trainable_count <= 0 and accepted_count > 0:
        reasons.append("0 accepted full-pose samples (after ANN field validation)")
    elif export_jsonl_path is not None and accepted_count == 0:
        reasons.append("0 accepted samples in export")

    if (
        export_jsonl_path is not None
        and accepted_count > 0
        and not full_pose_available
        and accepted_legacy_trainable_count > 0
    ):
        reasons.append("mixed-quality export: some accepted rows lack full ANN fields")

    if run_trust_mode in _LEGACY_ANN_SERVO_BLOCKED_MODES:
        reasons.append("servo-only/no tracker labels")

    if run_trust_mode in _LEGACY_ANN_LOW_TRUST_MODES:
        reasons.append("lower-trust run; enable include lower-trust if you really want debug training")

    if valid_for_model_training_flag is False:
        reasons.append("not marked valid_for_model_training")

    trainable = True
    if not structurally_ready_for_legacy_ann:
        trainable = False
    elif dataset_scan_root == "mock":
        trainable = False
    elif mock_mode_flag is True:
        trainable = False
    elif export_jsonl_path is None:
        trainable = False
    elif accepted_legacy_trainable_count <= 0:
        trainable = False
    elif run_trust_mode in _LEGACY_ANN_SERVO_BLOCKED_MODES:
        trainable = False
    elif run_trust_mode in _LEGACY_ANN_LOW_TRUST_MODES:
        trainable = False
    elif valid_for_model_training_flag is False:
        trainable = False

    # De-duplicate while keeping order
    ordered: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        if reason not in seen:
            ordered.append(reason)
            seen.add(reason)
    return trainable, tuple(ordered)


def effective_legacy_ann_training_allowed(
    summary: ModelingDatasetSummary | None,
    *,
    allow_mock_training: bool = False,
    allow_lower_trust_training: bool = False,
) -> bool:
    """Return whether Benchmark/Train should be enabled for this dataset under optional debug overrides."""
    if summary is None:
        return False
    if not summary.structurally_ready_for_legacy_ann:
        return False
    if summary.dataset_scan_root == "mock" and not allow_mock_training:
        return False
    if summary.mock_mode_flag is True and not allow_mock_training:
        return False
    if summary.valid_for_model_training_flag is False and not allow_lower_trust_training:
        return False
    if summary.run_trust_mode in _LEGACY_ANN_SERVO_BLOCKED_MODES:
        return False
    if summary.run_trust_mode in _LEGACY_ANN_LOW_TRUST_MODES and not allow_lower_trust_training:
        return False
    return True


def _compile_modeling_dataset_summary(
    run_dir: Path,
    *,
    dataset_scan_root: str,
    strict: bool,
) -> ModelingDatasetSummary:
    run_dir = Path(run_dir)
    meta_path = run_dir / "metadata.json"
    sum_path = run_dir / "summary.json"
    export_path = run_dir / "modeling_dataset_export.jsonl"

    warnings: list[str] = []
    metadata_payload: dict[str, Any] = {}
    summary_payload: dict[str, Any] = {}

    if strict:
        if not meta_path.is_file():
            raise FileNotFoundError(str(meta_path))
        if not sum_path.is_file():
            raise FileNotFoundError(str(sum_path))
        metadata_payload = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(metadata_payload, dict):
            raise ValueError("metadata.json must be a JSON object.")
        summary_payload = json.loads(sum_path.read_text(encoding="utf-8"))
        if not isinstance(summary_payload, dict):
            raise ValueError("summary.json must be a JSON object.")
    else:
        if meta_path.is_file():
            try:
                raw = json.loads(meta_path.read_text(encoding="utf-8"))
                metadata_payload = raw if isinstance(raw, dict) else {}
                if not isinstance(raw, dict):
                    warnings.append("metadata.json was not a JSON object.")
            except Exception as exc:
                warnings.append(f"metadata.json unreadable: {exc}")
        else:
            warnings.append("missing metadata.json")
        if sum_path.is_file():
            try:
                raw = json.loads(sum_path.read_text(encoding="utf-8"))
                summary_payload = raw if isinstance(raw, dict) else {}
                if not isinstance(raw, dict):
                    warnings.append("summary.json was not a JSON object.")
            except Exception as exc:
                warnings.append(f"summary.json unreadable: {exc}")
        else:
            warnings.append("missing summary.json")

    metrics = (
        summary_payload.get("experiment_metrics", {})
        if isinstance(summary_payload.get("experiment_metrics"), dict)
        else {}
    )
    export_rows: list[dict[str, Any]] = []
    if export_path.is_file():
        try:
            export_rows = _load_export_rows(run_dir)
        except Exception as exc:
            warnings.append(f"modeling_dataset_export.jsonl unreadable: {exc}")
    accepted_rows = [row for row in export_rows if bool(row.get("accepted"))]
    accepted_legacy_trainable_count = 0
    for row in accepted_rows:
        if _row_filter_reason(row) is None:
            accepted_legacy_trainable_count += 1

    full_pose_available = (
        all(
            len(row.get("resolved_cable_command_cm", []) or []) == LEGACY_FULL_POSE_INPUT_DIM
            and len(row.get("tip_position_xyz_mm", []) or []) == 3
            and len(row.get("tip_tangent_xyz", []) or []) == 3
            for row in accepted_rows
        )
        if accepted_rows
        else False
    )
    tangent_available = all(len(row.get("tip_tangent_xyz", []) or []) == 3 for row in accepted_rows) if accepted_rows else False
    sequential_context_available = any(len(row.get("previous_pair_command_cm", []) or []) == 2 for row in export_rows)

    provenance = dict(metrics.get("run_provenance", {}) or {})
    runtime_tip = dict(provenance.get("runtime_tip_calibration", {}) or {})
    pretension = dict(provenance.get("pretension_artifact", {}) or {})
    trust_summary = (
        f"runtime tip {runtime_tip.get('trust_level', 'unknown')}, pretension {pretension.get('status', 'unknown')}"
    )

    run_trust = _as_dict(metrics.get("run_trust"))
    meta_trust = _as_dict(metadata_payload.get("trust_info"))
    valid_for_model_training_flag = _coalesce_model_training_valid(
        metrics.get("valid_for_model_training"),
        run_trust.get("valid_for_model_training"),
        meta_trust.get("valid_for_model_training"),
    )
    mock_mode_flag = _read_mock_mode_flag(metadata_payload, metrics)
    run_trust_mode = _read_run_trust_mode(metadata_payload, metrics)

    structurally_ready_for_legacy_ann = bool(
        meta_path.is_file()
        and sum_path.is_file()
        and export_path.is_file()
        and isinstance(metadata_payload, dict)
        and isinstance(summary_payload, dict)
        and len(metadata_payload) > 0
        and len(summary_payload) > 0
        and accepted_legacy_trainable_count > 0
    )

    trainable_for_legacy_ann, legacy_reasons = _finalize_legacy_ann_trainability(
        dataset_scan_root=dataset_scan_root,
        structurally_ready_for_legacy_ann=structurally_ready_for_legacy_ann,
        export_jsonl_path=export_path if export_path.is_file() else None,
        accepted_legacy_trainable_count=accepted_legacy_trainable_count,
        mock_mode_flag=mock_mode_flag,
        run_trust_mode=run_trust_mode,
        valid_for_model_training_flag=valid_for_model_training_flag,
        full_pose_available=bool(full_pose_available),
        accepted_count=len(accepted_rows),
    )

    export_jsonl_path = export_path if export_path.is_file() else None
    catalog_experiment_name = str(metadata_payload.get("experiment_name") or "").strip()

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
        export_jsonl_path=export_jsonl_path,
        legacy_dat_path=(run_dir / "modeling_dataset_legacy_compat.dat") if (run_dir / "modeling_dataset_legacy_compat.dat").exists() else None,
        summary_text_path=(run_dir / "modeling_dataset_summary.txt") if (run_dir / "modeling_dataset_summary.txt").exists() else None,
        trust_summary=trust_summary,
        runtime_tip_summary=f"{runtime_tip.get('mode', 'unknown')} ({runtime_tip.get('trust_level', 'unknown')})",
        pretension_summary=f"{pretension.get('active_source_type', 'unknown')} ({pretension.get('status', 'unknown')})",
        dataset_scan_root=dataset_scan_root,
        catalog_experiment_name=catalog_experiment_name,
        mock_mode_flag=mock_mode_flag,
        run_trust_mode=run_trust_mode,
        valid_for_model_training_flag=valid_for_model_training_flag,
        structurally_ready_for_legacy_ann=bool(structurally_ready_for_legacy_ann),
        accepted_legacy_trainable_count=int(accepted_legacy_trainable_count),
        trainable_for_legacy_ann=bool(trainable_for_legacy_ann),
        legacy_ann_rejection_reasons=legacy_reasons,
        discovery_warnings=tuple(warnings),
        metadata_payload=metadata_payload,
        summary_payload=summary_payload,
    )


def _discovery_sort_key(entry: ModelingDatasetSummary) -> tuple:
    kind_order = {"experiments": 0, "real": 0, "legacy": 1, "archived": 2, "mock": 3}
    return (
        0 if entry.trainable_for_legacy_ann else 1,
        kind_order.get(entry.dataset_scan_root, 4),
        str(entry.timestamp_utc or ""),
        str(entry.run_name or ""),
    )


def discover_modeling_datasets(
    *,
    output_root: Path | None = None,
    project_root: Path | None = None,
    include_mock_experiments: bool = False,
    include_archived_experiments: bool = False,
) -> list[ModelingDatasetSummary]:
    """List collect_pose_command_dataset runs from canonical data roots.

    When ``project_root`` is provided, discovery merges:

    - ``data/experiments/collect_pose_command_dataset/*``
    - optional ``data/mock_experiments/...`` (``include_mock_experiments``)
    - optional ``data/experiments_archived/...`` (``include_archived_experiments``)
    - optional legacy relative path via ``canonical_experiment_output_root(output_root, ...)``

    Runs under ``data/trash`` are skipped. Every run folder yields an entry even when incomplete,
    with trainability spelled out on the summary record.
    """
    merged_roots: list[tuple[Path, str]] = []
    pr = Path(project_root) if project_root is not None else None
    if pr is not None:
        merged_roots.append((pr / "data" / "experiments" / COLLECT_POSE_COMMAND_DATASET, "experiments"))
        if include_mock_experiments:
            merged_roots.append((pr / "data" / "mock_experiments" / COLLECT_POSE_COMMAND_DATASET, "mock"))
        if include_archived_experiments:
            merged_roots.append((pr / "data" / "experiments_archived" / COLLECT_POSE_COMMAND_DATASET, "archived"))
    legacy_root: Path | None = None
    if output_root is not None:
        legacy_root = canonical_experiment_output_root(Path(output_root), COLLECT_POSE_COMMAND_DATASET)

    seen: set[str] = set()
    run_dirs: list[tuple[Path, str]] = []
    for collect_root, scan_kind in merged_roots:
        if not collect_root.exists() or not collect_root.is_dir():
            continue
        for child in sorted(collect_root.iterdir(), key=lambda value: value.name, reverse=True):
            if not child.is_dir() or child.name.startswith("."):
                continue
            if _is_data_trash_path(child):
                continue
            key = str(child.resolve())
            if key in seen:
                continue
            seen.add(key)
            run_dirs.append((child, scan_kind))

    if legacy_root is not None and legacy_root.exists():
        for child in sorted(legacy_root.iterdir(), key=lambda value: value.name, reverse=True):
            if not child.is_dir() or child.name.startswith("."):
                continue
            if _is_data_trash_path(child):
                continue
            key = str(child.resolve())
            if key in seen:
                continue
            seen.add(key)
            run_dirs.append((child, "legacy"))

    items: list[ModelingDatasetSummary] = []
    for run_dir, scan_kind in run_dirs:
        items.append(_compile_modeling_dataset_summary(run_dir, dataset_scan_root=scan_kind, strict=False))

    items.sort(key=_discovery_sort_key)
    return items


def load_modeling_dataset_summary(path: Path) -> ModelingDatasetSummary:
    """Load summary metadata for one modeling dataset run.

    Raises when core JSON artifacts are missing/unreadable (training prep expects a healthy bundle).
    """
    return _compile_modeling_dataset_summary(Path(path), dataset_scan_root="experiments", strict=True)


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


def _prepare_empty_artifact_directory(path: Path) -> None:
    """Create ``path`` as a new empty directory (parents must exist)."""
    candidate = Path(path)
    if candidate.exists():
        try:
            next(candidate.iterdir())
        except StopIteration:
            return
        raise ValueError(f"Artifact directory must be empty: {candidate}")
    candidate.mkdir(parents=False, exist_ok=False)


def _tangent_angle_errors_rad(pred_t: np.ndarray, targ_t: np.ndarray) -> np.ndarray:
    eps = 1e-12
    pnorm = np.linalg.norm(pred_t, axis=1)
    tnorm = np.linalg.norm(targ_t, axis=1)
    mask = (pnorm > eps) & (tnorm > eps)
    out = np.full(int(pred_t.shape[0]), np.nan, dtype=float)
    if not np.any(mask):
        return out
    pn = pred_t[mask] / pnorm[mask, None]
    tn = targ_t[mask] / tnorm[mask, None]
    dots = np.clip(np.sum(pn * tn, axis=1), -1.0, 1.0)
    out[mask] = np.arccos(dots)
    return out


def compute_pose_evaluation_metrics(pred: np.ndarray, targ: np.ndarray) -> dict[str, Any]:
    """Geometric error metrics for full-pose predictions (pos mm + tangent xyz)."""
    pred = np.asarray(pred, dtype=float)
    targ = np.asarray(targ, dtype=float)
    dpos = pred[:, :3] - targ[:, :3]
    pos_mags = np.linalg.norm(dpos, axis=1)
    rmse_xyz = float(np.sqrt(np.mean(np.sum(dpos**2, axis=1))))
    rmse_xy = float(np.sqrt(np.mean(np.sum(dpos[:, :2] ** 2, axis=1))))
    rmse_z = float(np.sqrt(np.mean(dpos[:, 2] ** 2)))
    ang = _tangent_angle_errors_rad(pred[:, 3:6], targ[:, 3:6])
    finite_ang = ang[np.isfinite(ang)]
    return {
        "position_rmse_xyz_mm": rmse_xyz,
        "position_rmse_xy_mm": rmse_xy,
        "position_rmse_z_mm": rmse_z,
        "position_error_l2_mm": {
            "mean": float(np.mean(pos_mags)),
            "median": float(np.median(pos_mags)),
            "p95": float(np.percentile(pos_mags, 95)),
            "max": float(np.max(pos_mags)),
        },
        "tangent_angular_error_rad": (
            {
                "mean": float(np.mean(finite_ang)),
                "median": float(np.median(finite_ang)),
                "p95": float(np.percentile(finite_ang, 95)),
                "max": float(np.max(finite_ang)),
            }
            if finite_ang.size
            else None
        ),
    }


def _eval_torch_loader_metrics(
    *,
    torch: Any,
    model: Any,
    loss_module: Any,
    dataloader: Any,
    device: Any,
) -> dict[str, Any] | None:
    """Mean per-batch scalar loss plus pose metrics for one split."""
    if len(dataloader) == 0:
        return None
    preds: list[np.ndarray] = []
    targs: list[np.ndarray] = []
    batch_losses: list[float] = []
    for batch in dataloader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        model.eval()
        with torch.no_grad():
            predictions = model(inputs)
            batch_losses.append(float(loss_module(predictions, targets).item()))
        preds.append(predictions.detach().cpu().numpy())
        targs.append(targets.detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    targ = np.concatenate(targs, axis=0)
    metrics = compute_pose_evaluation_metrics(pred, targ)
    return {"loss_mean": float(np.mean(batch_losses)), **metrics}


def merge_ann_sweep_architectures(
    *,
    defaults: Sequence[Sequence[int]],
    extras: list[list[int]],
) -> list[list[int]]:
    """Deduplicate architecture lists while preserving order (defaults first)."""
    seen: set[tuple[int, ...]] = set()
    out: list[list[int]] = []
    for group in (*defaults, *extras):
        key = tuple(int(x) for x in group)
        if key in seen:
            continue
        seen.add(key)
        out.append(list(key))
    return out


def select_best_sweep_row_by_test_position_rmse(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the sweep row with lowest ``test_position_rmse_xyz_mm`` when present, else lowest ``test_loss``."""
    best: dict[str, Any] | None = None
    best_key: tuple[float, float] | None = None
    for row in rows:
        xyz = row.get("test_position_rmse_xyz_mm")
        if xyz is not None and isinstance(xyz, (int, float)) and not math.isnan(float(xyz)):
            key = (float(xyz), float(row.get("test_loss") or 1e308))
        else:
            tl = row.get("test_loss")
            if tl is None or (isinstance(tl, float) and math.isnan(tl)):
                key = (1e308, 1e308)
            else:
                key = (1e308, float(tl))
        if best is None or key < (best_key or (1e308, 1e308)):
            best = dict(row)
            best_key = key
    return best


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
    split: DatasetSplit | None = None,
    artifact_dir: Path | None = None,
) -> TrainingResult:
    """Train the legacy full-pose ANN and save an artifact bundle.

    When ``split`` is provided (e.g. model sweep), that split is used instead of rebuilding.
    When ``artifact_dir`` is provided, artifacts are written there (directory must not exist or must be empty);
    otherwise a timestamped folder is allocated under the configured artifact root.
    """
    validate_training_config(config)
    torch = _require_torch()
    prepared = prepare_legacy_ann_dataset(dataset_path)
    if prepared.inputs.shape[0] == 0:
        raise ValueError("Dataset has no accepted full-pose samples after filtering.")
    split_used = split if split is not None else build_grouped_split(prepared, config)
    backend_report = detect_training_backends(preferred_backend=backend_name)
    device = _torch_device(torch, backend_report.selected_backend)
    dtype = _torch_dtype(torch, backend_report.selected_dtype)
    np.random.seed(int(config.random_seed))
    torch.manual_seed(int(config.random_seed))
    if artifact_dir is None:
        artifact_dir_path = _allocate_artifact_dir(
            project_root=Path(project_root),
            artifact_root_raw=config.artifact_root,
            artifact_name=config.artifact_name,
        )
        artifact_dir_path.mkdir(parents=True, exist_ok=False)
    else:
        artifact_dir_path = Path(artifact_dir).resolve()
        _prepare_empty_artifact_directory(artifact_dir_path)
    checkpoints_dir = artifact_dir_path / "checkpoints"
    if bool(config.checkpointing):
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
    loss_history_path = artifact_dir_path / "loss_history.csv"
    loss_plot_path = artifact_dir_path / "loss_curve.png"
    loss_report_plot_path = artifact_dir_path / "ann_loss_curve_report.png"
    metadata_path = artifact_dir_path / "training_metadata.json"
    config_path = artifact_dir_path / "training_config.json"
    split_manifest_path = artifact_dir_path / "split_manifest.json"
    summary_text_path = artifact_dir_path / "training_summary.txt"
    model_path = artifact_dir_path / "model.pt"

    estimate = estimate_runtime(
        prepared=prepared,
        split=split_used,
        config=config,
        backend_name=backend_report.selected_backend,
        time_fn=time_fn,
    )
    train_loader, validation_loader, test_loader = _build_dataloaders(
        torch=torch,
        prepared=prepared,
        split=split_used,
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
    training_wall_time_s = max(0.0, time_fn() - started_at)
    evaluation_payload: dict[str, Any] = {
        "single_dataset_split_note": MODEL_SWEEP_SINGLE_SPLIT_WARNING,
    }
    if best_state_dict is not None:
        val_metrics = _eval_torch_loader_metrics(
            torch=torch,
            model=model,
            loss_module=loss_module,
            dataloader=validation_loader,
            device=device,
        )
        test_metrics = (
            _eval_torch_loader_metrics(
                torch=torch,
                model=model,
                loss_module=loss_module,
                dataloader=test_loader,
                device=device,
            )
            if len(test_loader) > 0
            else None
        )
        evaluation_payload["validation"] = val_metrics
        evaluation_payload["test"] = test_metrics
    _write_loss_history_csv(loss_history_path, epoch_rows)
    _write_loss_plot(loss_plot_path, train_losses, validation_losses)
    _write_loss_plot(loss_report_plot_path, train_losses, validation_losses)
    split_manifest = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "strategy": split_used.strategy,
        "train_indices": list(split_used.train_indices),
        "validation_indices": list(split_used.validation_indices),
        "test_indices": list(split_used.test_indices),
        "group_ids": list(split_used.group_ids),
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
            "training_wall_time_s": float(training_wall_time_s),
            "checkpointing": bool(config.checkpointing),
            "learning_rate": float(config.learning_rate),
            "batch_size": int(config.batch_size),
            "random_seed": int(config.random_seed),
        },
        "split": {
            "strategy": split_used.strategy,
            "train_ratio": float(config.train_ratio),
            "validation_ratio": float(config.validation_ratio),
            "test_ratio": float(config.test_ratio),
            "train_count": len(split_used.train_indices),
            "validation_count": len(split_used.validation_indices),
            "test_count": len(split_used.test_indices),
        },
        "estimate": (asdict(estimate) if estimate is not None else None),
        "evaluation": evaluation_payload,
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
        artifact_dir=artifact_dir_path,
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


def _ann_sweep_subdir_name(hidden_layers: list[int]) -> str:
    return "ann_" + "_".join(str(int(x)) for x in hidden_layers)


def _write_model_comparison_report_png(path: Path, rows: Sequence[dict[str, Any]]) -> bool:
    """Bar chart of test XYZ RMSE (mm) by model. Returns False if matplotlib or metrics unavailable."""
    try:
        plt = import_matplotlib()
    except Exception:
        return False
    labels: list[str] = []
    values: list[float] = []
    for row in rows:
        labels.append(str(row.get("model_label") or row.get("model_key") or "?"))
        raw = row.get("test_position_rmse_xyz_mm")
        if raw is None:
            values.append(float("nan"))
        else:
            values.append(float(raw))
    if not labels or not any(not math.isnan(v) for v in values):
        return False
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    plot_vals = [0.0 if math.isnan(v) else v for v in values]
    ax.bar(range(len(labels)), plot_vals, color=color("measured"))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    style_axes(ax, title="Model sweep — test position RMSE (XYZ)", xlabel="Model", ylabel="mm (RMS of 3D error)")
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)
    return True


def _write_model_sweep_summary_artifacts(
    *,
    sweep_root: Path,
    rows: list[dict[str, Any]],
    best: dict[str, Any] | None,
    warnings: tuple[str, ...],
) -> tuple[Path, Path, Path, Path | None]:
    json_path = sweep_root / "model_sweep_summary.json"
    csv_path = sweep_root / "model_sweep_summary.csv"
    txt_path = sweep_root / "model_sweep_summary.txt"
    png_path = sweep_root / "model_comparison_report.png"
    payload = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "warnings": list(warnings),
        "rows": rows,
        "best_model": best,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
    lines = [
        "ANN / linear ridge model sweep summary",
        "",
        f"Created (UTC): {payload['created_at_utc']}",
        f"Sweep folder: {sweep_root}",
        "",
    ]
    for warn in warnings:
        lines.append(f"Warning: {warn}")
    if warnings:
        lines.append("")
    if best:
        lines.append(
            "Best model (lowest test_position_rmse_xyz_mm when available, else test loss): "
            f"{best.get('model_label') or best.get('model_key')} "
            f"| subdir={best.get('artifact_subdir')}"
        )
        lines.append("")
    for row in rows:
        lines.append(
            f"- {row.get('model_label')}: val_loss={_fmt_number(row.get('validation_loss_mean'))} "
            f"test_loss={_fmt_number(row.get('test_loss'))} "
            f"test_xyz_rmse_mm={_fmt_number(row.get('test_position_rmse_xyz_mm'))} "
            f"time_s={_fmt_number(row.get('training_wall_time_s'))}"
        )
    txt_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    png_written = _write_model_comparison_report_png(png_path, rows)
    return json_path, csv_path, txt_path, (png_path if png_written else None)


def _sweep_summary_row_from_metadata(
    metadata: dict[str, Any],
    *,
    model_key: str,
    model_label: str,
    artifact_subdir: str,
) -> dict[str, Any]:
    training = dict(metadata.get("training", {}) or {})
    evaluation = dict(metadata.get("evaluation", {}) or {})
    val_block = dict(evaluation.get("validation") or {}) if evaluation.get("validation") else {}
    test_block = dict(evaluation.get("test") or {}) if evaluation.get("test") else {}
    pos_err = dict(test_block.get("position_error_l2_mm") or {})
    ang = test_block.get("tangent_angular_error_rad")
    ang_d = dict(ang) if isinstance(ang, dict) else {}
    return {
        "model_key": model_key,
        "model_label": model_label,
        "artifact_subdir": artifact_subdir,
        "validation_loss_mean": val_block.get("loss_mean"),
        "test_loss": training.get("test_loss"),
        "test_position_rmse_xyz_mm": test_block.get("position_rmse_xyz_mm"),
        "test_position_rmse_xy_mm": test_block.get("position_rmse_xy_mm"),
        "test_position_rmse_z_mm": test_block.get("position_rmse_z_mm"),
        "test_position_error_l2_mean_mm": pos_err.get("mean"),
        "test_position_error_l2_median_mm": pos_err.get("median"),
        "test_position_error_l2_p95_mm": pos_err.get("p95"),
        "test_position_error_l2_max_mm": pos_err.get("max"),
        "test_tangent_angular_error_mean_rad": ang_d.get("mean"),
        "test_tangent_angular_error_median_rad": ang_d.get("median"),
        "training_wall_time_s": training.get("training_wall_time_s"),
        "hidden_layers": list(dict(metadata.get("model", {}) or {}).get("hidden_layers") or []),
    }


def train_linear_ridge_full_pose(
    *,
    project_root: Path,
    dataset_path: Path,
    prepared: PreparedLegacyAnnDataset,
    split: DatasetSplit,
    config: AnnTrainingConfig,
    backend_name: str,
    artifact_dir: Path,
    ridge_alpha: float | None = None,
    stop_requested: Callable[[], bool] | None = None,
    time_fn: Callable[[], float] = time.perf_counter,
) -> TrainingResult:
    """Closed-form ridge linear map (4 cable cm -> 6 pose) for sweep baseline comparison."""
    if stop_requested is not None and stop_requested():
        raise RuntimeError("Cancelled before linear ridge baseline started.")
    torch = _require_torch()
    alpha = float(ridge_alpha if ridge_alpha is not None else config.linear_ridge_alpha)
    backend_report = detect_training_backends(preferred_backend=backend_name)
    device = _torch_device(torch, backend_report.selected_backend)
    dtype = _torch_dtype(torch, backend_report.selected_dtype)
    np.random.seed(int(config.random_seed))
    torch.manual_seed(int(config.random_seed))
    artifact_dir_path = Path(artifact_dir).resolve()
    _prepare_empty_artifact_directory(artifact_dir_path)
    started_at = time_fn()
    Xi = prepared.inputs[np.asarray(split.train_indices, dtype=int)]
    Yi = prepared.outputs[np.asarray(split.train_indices, dtype=int)]
    if Xi.shape[0] < 1:
        raise ValueError("Ridge baseline requires at least one training sample.")
    n = Xi.shape[0]
    x_aug = np.concatenate([np.ones((n, 1), dtype=float), Xi], axis=1)
    d = int(x_aug.shape[1])
    reg = alpha * np.eye(d, dtype=float)
    ata = x_aug.T @ x_aug + reg
    aty = x_aug.T @ Yi
    try:
        coeffs = np.linalg.solve(ata, aty)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(ata, aty, rcond=None)[0]
    lin = torch.nn.Linear(LEGACY_FULL_POSE_INPUT_DIM, LEGACY_FULL_POSE_OUTPUT_DIM).to(device=device, dtype=dtype)
    with torch.no_grad():
        lin.weight.copy_(torch.tensor(coeffs[1:, :].T, dtype=dtype, device=device))
        lin.bias.copy_(torch.tensor(coeffs[0, :], dtype=dtype, device=device))
    train_loader, validation_loader, test_loader = _build_dataloaders(
        torch=torch,
        prepared=prepared,
        split=split,
        config=config,
        dtype=dtype,
    )
    loss_module = _build_loss_module(torch=torch, config=config, device=device, dtype=dtype)
    train_loss = (
        _run_validation_epoch(
            torch=torch,
            model=lin,
            loss_module=loss_module,
            dataloader=train_loader,
            device=device,
        )
        if len(train_loader) > 0
        else float("nan")
    )
    val_loss = (
        _run_validation_epoch(
            torch=torch,
            model=lin,
            loss_module=loss_module,
            dataloader=validation_loader,
            device=device,
        )
        if len(validation_loader) > 0
        else None
    )
    test_loss = (
        _run_validation_epoch(
            torch=torch,
            model=lin,
            loss_module=loss_module,
            dataloader=test_loader,
            device=device,
        )
        if len(test_loader) > 0
        else None
    )
    training_wall_time_s = max(0.0, time_fn() - started_at)
    val_metrics = _eval_torch_loader_metrics(
        torch=torch,
        model=lin,
        loss_module=loss_module,
        dataloader=validation_loader,
        device=device,
    )
    test_metrics = (
        _eval_torch_loader_metrics(
            torch=torch,
            model=lin,
            loss_module=loss_module,
            dataloader=test_loader,
            device=device,
        )
        if len(test_loader) > 0
        else None
    )
    evaluation_payload: dict[str, Any] = {
        "single_dataset_split_note": MODEL_SWEEP_SINGLE_SPLIT_WARNING,
        "validation": val_metrics,
        "test": test_metrics,
    }
    loss_history_path = artifact_dir_path / "loss_history.csv"
    loss_plot_path = artifact_dir_path / "loss_curve.png"
    loss_report_plot_path = artifact_dir_path / "ann_loss_curve_report.png"
    metadata_path = artifact_dir_path / "training_metadata.json"
    config_path = artifact_dir_path / "training_config.json"
    split_manifest_path = artifact_dir_path / "split_manifest.json"
    summary_text_path = artifact_dir_path / "training_summary.txt"
    model_path = artifact_dir_path / "model.pt"
    epoch_rows = [
        {
            "epoch": 1,
            "train_loss": float(train_loss),
            "validation_loss": (float(val_loss) if val_loss is not None else None),
            "elapsed_s": float(training_wall_time_s),
        }
    ]
    _write_loss_history_csv(loss_history_path, epoch_rows)
    train_losses = [float(train_loss)]
    validation_losses = [float(val_loss) if val_loss is not None else float("nan")]
    _write_loss_plot(loss_plot_path, train_losses, validation_losses)
    _write_loss_plot(loss_report_plot_path, train_losses, validation_losses)
    torch.save(lin.state_dict(), model_path)
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
    config_dump = dict(config.to_dict())
    config_dump["hidden_layers"] = []
    config_dump["linear_ridge_baseline"] = True
    metadata_payload = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "artifact_kind": "linear_ridge_full_pose_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "dataset": dataset_metadata,
        "model": {
            "family": "linear_ridge_full_pose",
            "variant": "full_pose",
            "input_dim": LEGACY_FULL_POSE_INPUT_DIM,
            "output_dim": LEGACY_FULL_POSE_OUTPUT_DIM,
            "hidden_layers": [],
            "ridge_alpha": alpha,
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
            "epochs_requested": 1,
            "epochs_completed": 1,
            "best_epoch": 1,
            "best_validation_loss": (float(val_loss) if val_loss is not None else None),
            "test_loss": (float(test_loss) if test_loss is not None else None),
            "training_wall_time_s": float(training_wall_time_s),
            "checkpointing": False,
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
        "estimate": None,
        "evaluation": evaluation_payload,
        "files": {
            "model_path": str(model_path),
            "loss_history_path": str(loss_history_path),
            "loss_plot_path": str(loss_plot_path),
            "loss_report_plot_path": str(loss_report_plot_path),
            "split_manifest_path": str(split_manifest_path),
            "summary_text_path": str(summary_text_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    config_path.write_text(json.dumps(config_dump, indent=2), encoding="utf-8")
    summary_text_path.write_text(_render_summary_text(metadata_payload), encoding="utf-8")
    return TrainingResult(
        artifact_dir=artifact_dir_path,
        model_path=model_path,
        metadata_path=metadata_path,
        loss_history_path=loss_history_path,
        loss_plot_path=loss_plot_path,
        split_manifest_path=split_manifest_path,
        summary_text_path=summary_text_path,
        status="completed",
        best_epoch=1,
        best_validation_loss=(float(val_loss) if val_loss is not None else None),
        test_loss=(float(test_loss) if test_loss is not None else None),
        epochs_completed=1,
        train_losses=train_losses,
        validation_losses=validation_losses,
        estimate=None,
    )


def run_model_sweep(
    *,
    project_root: Path,
    dataset_path: Path,
    base_config: AnnTrainingConfig,
    backend_name: str,
    include_linear_baseline: bool | None = None,
    ann_hidden_layers_list: list[list[int]] | None = None,
    extra_hidden_layers_text: str | None = None,
    status_callback: Callable[[str], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    time_fn: Callable[[], float] = time.perf_counter,
) -> ModelSweepResult:
    """Train linear ridge (optional) and several ANNs sharing one split; write sweep summaries."""
    _require_torch()
    prepared = prepare_legacy_ann_dataset(dataset_path)
    if prepared.inputs.shape[0] == 0:
        raise ValueError("Dataset has no accepted full-pose samples after filtering.")
    validate_training_config(base_config)
    split = build_grouped_split(prepared, base_config)
    extras_raw = (
        str(extra_hidden_layers_text)
        if extra_hidden_layers_text is not None
        else str(base_config.model_sweep_extra_hidden_layers_text or "")
    )
    defaults = [list(t) for t in MODEL_SWEEP_DEFAULT_ANN_HIDDEN_LAYERS]
    if ann_hidden_layers_list is not None:
        ann_list = [list(int(x) for x in group) for group in ann_hidden_layers_list]
    else:
        extras, extra_err = parse_sweep_extra_hidden_layers_groups(extras_raw)
        if extra_err or extras is None:
            raise ValueError(extra_err or "Invalid sweep extra architectures text.")
        ann_list = merge_ann_sweep_architectures(defaults=defaults, extras=list(extras))
    use_linear = (
        bool(include_linear_baseline)
        if include_linear_baseline is not None
        else bool(base_config.model_sweep_include_linear_baseline)
    )
    sweep_root = _allocate_artifact_dir(
        project_root=Path(project_root),
        artifact_root_raw=base_config.artifact_root,
        artifact_name=f"{base_config.artifact_name.strip()}_model_sweep",
    )
    sweep_root.mkdir(parents=True, exist_ok=False)
    shared_split_path = sweep_root / "shared_split_manifest.json"
    shared_split_path.write_text(
        json.dumps(
            {
                "schema_version": TRAINING_SCHEMA_VERSION,
                "strategy": split.strategy,
                "train_indices": list(split.train_indices),
                "validation_indices": list(split.validation_indices),
                "test_indices": list(split.test_indices),
                "group_ids": list(split.group_ids),
                "sequence_indices": list(prepared.sequence_indices),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    warnings_list: list[str] = [MODEL_SWEEP_SINGLE_SPLIT_WARNING]
    if len(split.test_indices) < 3:
        warnings_list.append("Test split has very few samples; sweep metrics may be noisy.")
    rows: list[dict[str, Any]] = []

    def _status(msg: str) -> None:
        if status_callback is not None:
            status_callback(msg)

    if use_linear:
        if stop_requested is not None and stop_requested():
            raise RuntimeError("Model sweep cancelled.")
        _status("Sweep: training linear ridge baseline.")
        lin_dir = sweep_root / "linear_ridge_full_pose"
        train_linear_ridge_full_pose(
            project_root=project_root,
            dataset_path=dataset_path,
            prepared=prepared,
            split=split,
            config=base_config,
            backend_name=backend_name,
            artifact_dir=lin_dir,
            stop_requested=stop_requested,
            time_fn=time_fn,
        )
        meta = json.loads((lin_dir / "training_metadata.json").read_text(encoding="utf-8"))
        rows.append(
            _sweep_summary_row_from_metadata(
                meta,
                model_key="linear_ridge_full_pose",
                model_label="linear_ridge",
                artifact_subdir="linear_ridge_full_pose",
            )
        )

    for hidden in ann_list:
        if stop_requested is not None and stop_requested():
            raise RuntimeError("Model sweep cancelled.")
        label = format_hidden_layers_for_ann_ui(hidden)
        sub = _ann_sweep_subdir_name(hidden)
        _status(f"Sweep: training ANN [{label}] -> {sub}")
        cfg = AnnTrainingConfig(**base_config.to_dict())
        cfg.hidden_layers = list(hidden)
        train_legacy_ann(
            project_root=project_root,
            dataset_path=dataset_path,
            config=cfg,
            backend_name=backend_name,
            split=split,
            artifact_dir=sweep_root / sub,
            progress_callback=None,
            stop_requested=stop_requested,
            time_fn=time_fn,
        )
        meta = json.loads((sweep_root / sub / "training_metadata.json").read_text(encoding="utf-8"))
        rows.append(
            _sweep_summary_row_from_metadata(
                meta,
                model_key=f"legacy_ann_{sub}",
                model_label=f"ANN [{label}]",
                artifact_subdir=sub,
            )
        )

    best = select_best_sweep_row_by_test_position_rmse(rows)
    warnings_t = tuple(warnings_list)
    json_path, csv_path, txt_path, png_path = _write_model_sweep_summary_artifacts(
        sweep_root=sweep_root,
        rows=rows,
        best=best,
        warnings=warnings_t,
    )
    return ModelSweepResult(
        sweep_root=sweep_root,
        summary_json_path=json_path,
        summary_csv_path=csv_path,
        summary_txt_path=txt_path,
        comparison_png_path=png_path,
        rows=tuple(rows),
        best_model=best,
        warnings=warnings_t,
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
    model = dict(metadata_payload.get("model", {}) or {})
    artifact_kind = str(metadata_payload.get("artifact_kind", "") or "")
    title = (
        "Linear ridge full-pose baseline summary"
        if "linear_ridge" in artifact_kind
        else "Legacy ANN full-pose training summary"
    )
    hl = model.get("hidden_layers")
    model_line = (
        f"Model: {model.get('family', 'n/a')} | ridge_alpha={model.get('ridge_alpha', 'n/a')}"
        if str(model.get("family", "")).startswith("linear_ridge")
        else f"Hidden layers: {hl if hl is not None else 'n/a'}"
    )
    lines = [
        title,
        "",
        f"Status: {metadata_payload.get('status', 'unknown')}",
        f"Created: {metadata_payload.get('created_at_utc', 'n/a')}",
        f"Dataset: {dataset.get('run_name', 'n/a')}",
        f"Dataset mode: {dataset.get('dataset_mode', 'n/a')}",
        f"Prepared samples: {dataset.get('prepared_sample_count', 'n/a')}",
        f"Backend: {backend.get('selected_backend', 'n/a')} ({backend.get('platform_summary', 'n/a')})",
        model_line,
        f"Epochs completed: {training.get('epochs_completed', 'n/a')}",
        f"Best epoch: {training.get('best_epoch', 'n/a')}",
        f"Best validation loss: {_fmt_number(training.get('best_validation_loss'))}",
        f"Test loss: {_fmt_number(training.get('test_loss'))}",
        f"Training wall time (s): {_fmt_number(training.get('training_wall_time_s'))}",
    ]
    if estimate:
        lines.append(f"Warmup estimate: {_fmt_seconds(estimate.get('estimated_total_s'))}")
    evaluation = dict(metadata_payload.get("evaluation", {}) or {})
    test_block = dict(evaluation.get("test") or {}) if evaluation.get("test") else {}
    if test_block:
        pe = dict(test_block.get("position_error_l2_mm") or {})
        lines.extend(
            [
                "",
                "Held-out test metrics (same split as training):",
                f"  Loss (batch mean): {_fmt_number(test_block.get('loss_mean'))}",
                f"  Position RMSE XYZ (mm): {_fmt_number(test_block.get('position_rmse_xyz_mm'))}",
                f"  Position RMSE XY (mm): {_fmt_number(test_block.get('position_rmse_xy_mm'))}",
                f"  Position RMSE Z (mm): {_fmt_number(test_block.get('position_rmse_z_mm'))}",
                f"  Position error L2 mean / median / p95 / max (mm): "
                f"{_fmt_number(pe.get('mean'))} / {_fmt_number(pe.get('median'))} / "
                f"{_fmt_number(pe.get('p95'))} / {_fmt_number(pe.get('max'))}",
            ]
        )
        ang = test_block.get("tangent_angular_error_rad")
        if isinstance(ang, dict) and ang:
            lines.append(
                "  Tangent angular error mean / median (rad): "
                f"{_fmt_number(ang.get('mean'))} / {_fmt_number(ang.get('median'))}"
            )
    note = evaluation.get("single_dataset_split_note")
    if note:
        lines.extend(["", f"Note: {note}"])
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
