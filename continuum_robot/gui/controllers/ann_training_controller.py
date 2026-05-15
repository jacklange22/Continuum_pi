"""Controller for the ANN training popout."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import threading
from typing import Any

from continuum_robot.gui.experiment_visualization import VisualizationModel
from continuum_robot.modeling.ann_training import (
    ANN_TRAINING_CATEGORY_BLOCKED,
    ANN_TRAINING_CATEGORY_EXPLORATORY_ONLY,
    ANN_TRAINING_CATEGORY_TRAINABLE,
    ANN_TRAINING_CATEGORY_TRAINABLE_WITH_WARNINGS,
    AnnTrainingConfig,
    BackendReport,
    MODEL_SWEEP_SINGLE_SPLIT_WARNING,
    ModelingDatasetSummary,
    TorchUnavailableError,
    TrainedArtifactSummary,
    TrainingEstimate,
    TrainingProgress,
    ann_training_will_be_exploratory,
    build_grouped_split,
    build_training_visualization,
    default_artifact_root,
    detect_training_backends,
    discover_modeling_datasets,
    discover_trained_artifacts,
    effective_legacy_ann_training_allowed,
    estimate_memory_bytes,
    estimate_runtime,
    format_hidden_layers_for_ann_ui,
    load_loss_history,
    load_modeling_dataset_summary,
    load_training_metadata,
    prepare_legacy_ann_dataset,
    parse_hidden_layers_text,
    run_model_sweep,
    train_legacy_ann,
    validate_training_config,
)


@dataclass
class AnnTrainingViewState:
    """UI-facing snapshot for the training popout."""

    datasets: list[ModelingDatasetSummary] = field(default_factory=list)
    catalog_all_datasets: list[ModelingDatasetSummary] = field(default_factory=list)
    catalog_trainable_total: int = 0
    catalog_non_trainable_total: int = 0
    dataset_catalog_message: str = ""
    show_non_trainable_datasets: bool = False
    show_mock_dataset_roots: bool = False
    include_archived_datasets: bool = False
    allow_mock_training: bool = False
    allow_lower_trust_training: bool = False
    allow_exploratory_incomplete_target: bool = False
    allow_parallel_single_demo_training: bool = False
    selected_dataset_path: str = ""
    dataset_summary_pairs: list[tuple[str, str]] = field(default_factory=list)
    ann_training_category: str = ANN_TRAINING_CATEGORY_BLOCKED
    ann_training_blocking_reasons: list[str] = field(default_factory=list)
    ann_training_warnings: list[str] = field(default_factory=list)
    will_train_exploratory: bool = False
    exploratory_training_warning: str = ""
    backend_report: BackendReport | None = None
    system_summary_pairs: list[tuple[str, str]] = field(default_factory=list)
    artifacts: list[TrainedArtifactSummary] = field(default_factory=list)
    selected_artifact_path: str = ""
    artifact_summary_pairs: list[tuple[str, str]] = field(default_factory=list)
    config_error: str | None = None
    estimate: TrainingEstimate | None = None
    estimate_stale: bool = True
    estimate_summary_pairs: list[tuple[str, str]] = field(default_factory=list)
    benchmark_active: bool = False
    training_active: bool = False
    status_message: str = "Select a modeling dataset to prepare ANN training."
    current_epoch: int = 0
    total_epochs: int = 0
    current_train_loss: float | None = None
    current_validation_loss: float | None = None
    elapsed_s: float = 0.0
    remaining_s: float | None = None
    last_output_path: str | None = None
    can_benchmark: bool = False
    can_train: bool = False
    sweep_active: bool = False
    can_run_sweep: bool = False
    sweep_best_model_text: str = ""
    sweep_split_warning: str = ""
    visualization_model: VisualizationModel = field(
        default_factory=lambda: VisualizationModel(summary_lines=["No training history loaded."])
    )


class AnnTrainingController:
    """Owns ANN popout state and runs benchmark/training work in background threads."""

    def __init__(
        self,
        *,
        project_root: Path,
        dataset_output_root: Path,
        artifact_root: Path | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.dataset_output_root = Path(dataset_output_root)
        self.artifact_root = Path(artifact_root) if artifact_root is not None else default_artifact_root(self.project_root)
        self._lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._catalog_dirty = True
        self._selected_dataset_summary: ModelingDatasetSummary | None = None
        self._selected_artifact_metadata: dict[str, Any] | None = None
        self._estimate_signature: tuple[Any, ...] | None = None
        self._training_history_train: list[float] = []
        self._training_history_validation: list[float] = []
        self._training_status = "idle"
        self._hidden_layers_text = "32, 32"
        self._hidden_layers_parse_error: str | None = None
        self.config = AnnTrainingConfig(artifact_root=str(self.artifact_root))
        self._hidden_layers_text = format_hidden_layers_for_ann_ui(self.config.hidden_layers)
        self.state = AnnTrainingViewState()

    def refresh(self) -> AnnTrainingViewState:
        with self._lock:
            catalog_dirty = self._catalog_dirty
            selected_dataset_path = self.state.selected_dataset_path
            selected_artifact_path = self.state.selected_artifact_path
            hidden_layers_text = self._hidden_layers_text
            config = AnnTrainingConfig(**self.config.to_dict())
            training_active = self.state.training_active
            benchmark_active = self.state.benchmark_active
            sweep_active = self.state.sweep_active
            estimate = self.state.estimate
            estimate_signature = self._estimate_signature
            selected_dataset_summary = self._selected_dataset_summary
            selected_artifact_metadata = self._selected_artifact_metadata
            show_non_trainable = self.state.show_non_trainable_datasets
            show_mock_roots = self.state.show_mock_dataset_roots
            include_archived = self.state.include_archived_datasets
            allow_mock_training = self.state.allow_mock_training
            allow_lower_trust = self.state.allow_lower_trust_training
            allow_exploratory = self.state.allow_exploratory_incomplete_target
            allow_parallel_single_demo = self.state.allow_parallel_single_demo_training

        datasets = self.state.datasets
        artifacts = self.state.artifacts
        if catalog_dirty:
            catalog_all = discover_modeling_datasets(
                project_root=self.project_root,
                output_root=self.dataset_output_root,
                include_mock_experiments=show_mock_roots,
                include_archived_experiments=include_archived,
            )
            trainable_total = sum(1 for item in catalog_all if item.trainable_for_legacy_ann)
            non_trainable_total = max(0, len(catalog_all) - trainable_total)
            visible = [
                item
                for item in catalog_all
                if (
                    item.trainable_for_legacy_ann
                    or item.ann_training_category == ANN_TRAINING_CATEGORY_EXPLORATORY_ONLY
                    or show_non_trainable
                )
                and (show_mock_roots or item.dataset_scan_root != "mock")
            ]
            datasets = visible
            catalog_message = ""
            if not catalog_all:
                catalog_message = (
                    "No collect_pose_command_dataset folders found under data/experiments (and optional mock/archived roots)."
                )
            elif trainable_total == 0 and non_trainable_total > 0 and not show_non_trainable:
                catalog_message = (
                    f"No trainable canonical collect-pose datasets found. Found {non_trainable_total} non-trainable run(s). "
                    "Enable “Show non-trainable / debug datasets” to inspect why."
                )
            self.state.catalog_all_datasets = catalog_all
            self.state.catalog_trainable_total = trainable_total
            self.state.catalog_non_trainable_total = non_trainable_total
            self.state.dataset_catalog_message = catalog_message
            artifacts = discover_trained_artifacts(artifact_root=self.artifact_root)
            if not datasets:
                selected_dataset_path = ""
                selected_dataset_summary = None
            else:
                visible_paths = {str(item.path) for item in datasets}
                if not selected_dataset_path:
                    preferred_trainable = next((item for item in datasets if item.trainable_for_legacy_ann), None)
                    if preferred_trainable is not None:
                        selected_dataset_path = str(preferred_trainable.path)
                    else:
                        selected_dataset_path = str(datasets[0].path)
                elif selected_dataset_path not in visible_paths:
                    preferred_trainable = next((item for item in datasets if item.trainable_for_legacy_ann), None)
                    if preferred_trainable is not None:
                        selected_dataset_path = str(preferred_trainable.path)
                    else:
                        selected_dataset_path = str(datasets[0].path)
                selected_dataset_summary = self._resolve_dataset_summary(selected_dataset_path, datasets)
            if not selected_artifact_path and artifacts:
                selected_artifact_path = str(artifacts[0].path)
            selected_artifact_metadata = self._resolve_artifact_metadata(selected_artifact_path, artifacts)
        elif selected_dataset_summary is None and selected_dataset_path:
            selected_dataset_summary = self._resolve_dataset_summary(selected_dataset_path, datasets)
        elif selected_artifact_metadata is None and selected_artifact_path:
            selected_artifact_metadata = self._resolve_artifact_metadata(selected_artifact_path, artifacts)

        config_error = self._config_error(config=config, hidden_layers_text=hidden_layers_text)
        backend_report = detect_training_backends(preferred_backend=getattr(self, "_selected_backend_name", None))
        dataset_pairs = self._dataset_pairs(selected_dataset_summary)
        system_pairs = self._system_pairs(
            backend_report=backend_report,
            dataset_summary=selected_dataset_summary,
            config=config,
        )
        estimate_pairs = self._estimate_pairs(estimate=estimate, stale=self._estimate_is_stale(estimate_signature, selected_dataset_summary, config, backend_report))
        artifact_pairs = self._artifact_pairs(selected_artifact_metadata)
        training_allowed = effective_legacy_ann_training_allowed(
            selected_dataset_summary,
            allow_mock_training=allow_mock_training,
            allow_lower_trust_training=allow_lower_trust,
            allow_exploratory_incomplete_target=allow_exploratory,
            allow_parallel_single_demo_training=allow_parallel_single_demo,
        )
        will_train_exploratory = ann_training_will_be_exploratory(
            selected_dataset_summary,
            allow_exploratory_incomplete_target=allow_exploratory,
        )
        exploratory_warning_text = ""
        if will_train_exploratory and selected_dataset_summary is not None:
            exploratory_warning_text = self._build_exploratory_warning_text(selected_dataset_summary)
        can_benchmark = (
            training_allowed
            and config_error is None
            and not training_active
            and not benchmark_active
            and not sweep_active
            and backend_report.torch_available
        )
        can_train = can_benchmark
        can_run_sweep = can_train
        visualization_model = self._visualization_model(selected_artifact_metadata)

        with self._lock:
            self._catalog_dirty = False
            self._selected_dataset_summary = selected_dataset_summary
            self._selected_artifact_metadata = selected_artifact_metadata
            self.state.datasets = datasets
            self.state.selected_dataset_path = selected_dataset_path
            self.state.dataset_summary_pairs = dataset_pairs
            self.state.backend_report = backend_report
            self.state.system_summary_pairs = system_pairs
            self.state.artifacts = artifacts
            self.state.selected_artifact_path = selected_artifact_path
            self.state.artifact_summary_pairs = artifact_pairs
            self.state.config_error = config_error
            self.state.estimate_summary_pairs = estimate_pairs
            self.state.estimate_stale = self._estimate_is_stale(estimate_signature, selected_dataset_summary, config, backend_report)
            self.state.can_benchmark = can_benchmark
            self.state.can_train = can_train
            self.state.can_run_sweep = can_run_sweep
            self.state.visualization_model = visualization_model
            self.state.will_train_exploratory = bool(will_train_exploratory)
            self.state.exploratory_training_warning = exploratory_warning_text
            if selected_dataset_summary is not None:
                self.state.ann_training_category = selected_dataset_summary.ann_training_category
                self.state.ann_training_blocking_reasons = list(
                    selected_dataset_summary.ann_training_blocking_reasons
                )
                self.state.ann_training_warnings = list(selected_dataset_summary.ann_training_warnings)
            else:
                self.state.ann_training_category = ANN_TRAINING_CATEGORY_BLOCKED
                self.state.ann_training_blocking_reasons = []
                self.state.ann_training_warnings = []
            if catalog_dirty and not self.state.status_message:
                self.state.status_message = "Select a modeling dataset to prepare ANN training."
            return self.state

    def set_dataset_output_root(self, path: Path) -> None:
        with self._lock:
            self.dataset_output_root = Path(path)
            self._catalog_dirty = True

    def set_show_non_trainable_datasets(self, value: bool) -> None:
        with self._lock:
            self.state.show_non_trainable_datasets = bool(value)
            self._catalog_dirty = True

    def set_show_mock_dataset_roots(self, value: bool) -> None:
        with self._lock:
            self.state.show_mock_dataset_roots = bool(value)
            self._catalog_dirty = True

    def set_include_archived_datasets(self, value: bool) -> None:
        with self._lock:
            self.state.include_archived_datasets = bool(value)
            self._catalog_dirty = True

    def set_allow_mock_training(self, value: bool) -> None:
        with self._lock:
            self.state.allow_mock_training = bool(value)

    def set_allow_lower_trust_training(self, value: bool) -> None:
        with self._lock:
            self.state.allow_lower_trust_training = bool(value)

    def set_allow_exploratory_incomplete_target(self, value: bool) -> None:
        with self._lock:
            self.state.allow_exploratory_incomplete_target = bool(value)

    def set_allow_parallel_single_demo_training(self, value: bool) -> None:
        with self._lock:
            self.state.allow_parallel_single_demo_training = bool(value)

    def invalidate_catalog(self) -> None:
        with self._lock:
            self._catalog_dirty = True

    def set_artifact_root(self, raw_path: str) -> None:
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.project_root / path
        with self._lock:
            self.artifact_root = path
            self.config.artifact_root = str(path)
            self._catalog_dirty = True
            self._invalidate_estimate_locked()

    def select_dataset(self, path: str) -> None:
        with self._lock:
            self.state.selected_dataset_path = str(path)
            self._selected_dataset_summary = None
            self._invalidate_estimate_locked()

    def select_artifact(self, path: str) -> None:
        with self._lock:
            self.state.selected_artifact_path = str(path)
            self._selected_artifact_metadata = None

    def select_backend(self, backend_name: str) -> None:
        with self._lock:
            self._selected_backend_name = str(backend_name)
            self._invalidate_estimate_locked()

    def set_hidden_layers_text(self, raw_value: str) -> None:
        with self._lock:
            parsed, err = parse_hidden_layers_text(raw_value)
            self._hidden_layers_parse_error = err
            if parsed is None:
                self._hidden_layers_text = str(raw_value)
                self.config.hidden_layers = []
            else:
                self.config.hidden_layers = list(parsed)
                self._hidden_layers_text = format_hidden_layers_for_ann_ui(parsed)
            self._invalidate_estimate_locked()

    def set_learning_rate(self, value: float) -> None:
        with self._lock:
            self.config.learning_rate = float(value)
            self._invalidate_estimate_locked()

    def set_batch_size(self, value: int) -> None:
        with self._lock:
            self.config.batch_size = int(value)
            self._invalidate_estimate_locked()

    def set_epochs(self, value: int) -> None:
        with self._lock:
            self.config.epochs = int(value)
            self._invalidate_estimate_locked()

    def set_loss_kind(self, value: str) -> None:
        with self._lock:
            self.config.loss_kind = str(value)
            self._invalidate_estimate_locked()

    def set_pose_orientation_scale(self, value: float) -> None:
        with self._lock:
            self.config.pose_orientation_scale = float(value)
            self._invalidate_estimate_locked()

    def set_random_seed(self, value: int) -> None:
        with self._lock:
            self.config.random_seed = int(value)
            self._invalidate_estimate_locked()

    def set_train_ratio(self, value: float) -> None:
        with self._lock:
            self.config.train_ratio = float(value)
            self._invalidate_estimate_locked()

    def set_validation_ratio(self, value: float) -> None:
        with self._lock:
            self.config.validation_ratio = float(value)
            self._invalidate_estimate_locked()

    def set_test_ratio(self, value: float) -> None:
        with self._lock:
            self.config.test_ratio = float(value)
            self._invalidate_estimate_locked()

    def set_checkpointing(self, value: bool) -> None:
        with self._lock:
            self.config.checkpointing = bool(value)
            self._invalidate_estimate_locked()

    def set_artifact_name(self, value: str) -> None:
        with self._lock:
            self.config.artifact_name = str(value).strip()
            self._invalidate_estimate_locked()

    def set_model_sweep_include_linear_baseline(self, value: bool) -> None:
        with self._lock:
            self.config.model_sweep_include_linear_baseline = bool(value)

    def set_model_sweep_extra_hidden_layers_text(self, value: str) -> None:
        with self._lock:
            self.config.model_sweep_extra_hidden_layers_text = str(value)

    def run_sweep(self) -> None:
        if self._job_active():
            return
        state = self.refresh()
        if not state.can_run_sweep:
            return
        with self._lock:
            self.state.sweep_active = True
            self.state.status_message = "Starting model sweep (shared split)."
            self._cancel_event.clear()
            dataset_path = self.state.selected_dataset_path
            config = AnnTrainingConfig(**self.config.to_dict())
            selected_backend = self._selected_backend_name_or_report()
            selected_summary = self._selected_dataset_summary
            training_provenance = self._build_training_provenance(
                summary=selected_summary,
                exploratory_override=bool(state.will_train_exploratory),
            )

        def _worker() -> None:
            try:

                def _on_status(message: str) -> None:
                    with self._lock:
                        self.state.status_message = message

                result = run_model_sweep(
                    project_root=self.project_root,
                    dataset_path=Path(dataset_path),
                    base_config=config,
                    backend_name=selected_backend,
                    include_linear_baseline=config.model_sweep_include_linear_baseline,
                    extra_hidden_layers_text=config.model_sweep_extra_hidden_layers_text,
                    status_callback=_on_status,
                    stop_requested=self._cancel_event.is_set,
                    training_provenance=training_provenance,
                )
                best = result.best_model
                best_text = ""
                if best is not None:
                    rmse = best.get("test_position_rmse_xyz_mm")
                    best_text = (
                        f"Best by test XYZ RMSE: {best.get('model_label')} "
                        f"({best.get('artifact_subdir')})"
                        + (f", RMSE={float(rmse):.4f} mm" if rmse is not None else "")
                    )
                with self._lock:
                    self.state.last_output_path = str(result.sweep_root)
                    self._catalog_dirty = True
                    self.state.sweep_best_model_text = best_text
                    self.state.sweep_split_warning = MODEL_SWEEP_SINGLE_SPLIT_WARNING
                    if best is not None and best.get("artifact_subdir"):
                        best_path = result.sweep_root / str(best["artifact_subdir"])
                        self.state.selected_artifact_path = str(best_path)
                        try:
                            self._selected_artifact_metadata = load_training_metadata(best_path)
                        except Exception:
                            self._selected_artifact_metadata = None
                    else:
                        self.state.selected_artifact_path = str(result.sweep_root)
                        self._selected_artifact_metadata = None
                    self.state.status_message = f"Model sweep complete. Results in {result.sweep_root.name}."
            except TorchUnavailableError as exc:
                with self._lock:
                    self.state.status_message = str(exc)
            except Exception as exc:
                with self._lock:
                    self.state.status_message = f"Model sweep failed: {exc}"
            finally:
                with self._lock:
                    self.state.sweep_active = False

        self._worker_thread = threading.Thread(target=_worker, daemon=True)
        self._worker_thread.start()

    def benchmark(self) -> None:
        if self._job_active():
            return
        state = self.refresh()
        if not state.can_benchmark:
            return
        with self._lock:
            self.state.benchmark_active = True
            self.state.status_message = "Benchmarking current ANN configuration."
            dataset_path = self.state.selected_dataset_path
            config = AnnTrainingConfig(**self.config.to_dict())
            selected_backend = self._selected_backend_name_or_report()

        def _worker() -> None:
            try:
                prepared = prepare_legacy_ann_dataset(Path(dataset_path))
                split = build_grouped_split(prepared, config)
                estimate = estimate_runtime(
                    prepared=prepared,
                    split=split,
                    config=config,
                    backend_name=selected_backend,
                )
                with self._lock:
                    self.state.estimate = estimate
                    self._estimate_signature = self._current_signature_locked(
                        dataset_summary=prepared.summary,
                        config=config,
                        backend_name=selected_backend,
                    )
                    self.state.status_message = (
                        f"Warmup estimate complete: {_fmt_seconds(estimate.estimated_total_s)} total."
                    )
            except Exception as exc:
                with self._lock:
                    self.state.status_message = f"Benchmark failed: {exc}"
            finally:
                with self._lock:
                    self.state.benchmark_active = False

        self._worker_thread = threading.Thread(target=_worker, daemon=True)
        self._worker_thread.start()

    def train(self) -> None:
        if self._job_active():
            return
        state = self.refresh()
        if not state.can_train:
            return
        with self._lock:
            self.state.training_active = True
            self.state.status_message = "Preparing ANN training."
            self.state.current_epoch = 0
            self.state.total_epochs = int(self.config.epochs)
            self.state.current_train_loss = None
            self.state.current_validation_loss = None
            self.state.elapsed_s = 0.0
            self.state.remaining_s = None
            self._training_history_train = []
            self._training_history_validation = []
            self._training_status = "running"
            self._cancel_event.clear()
            dataset_path = self.state.selected_dataset_path
            config = AnnTrainingConfig(**self.config.to_dict())
            selected_backend = self._selected_backend_name_or_report()
            estimate_signature = self._estimate_signature
            selected_dataset_summary = self._selected_dataset_summary
            training_provenance = self._build_training_provenance(
                summary=selected_dataset_summary,
                exploratory_override=bool(state.will_train_exploratory),
            )

        def _worker() -> None:
            try:
                if self._estimate_is_stale(estimate_signature, selected_dataset_summary, config, detect_training_backends(preferred_backend=selected_backend)):
                    prepared = prepare_legacy_ann_dataset(Path(dataset_path))
                    split = build_grouped_split(prepared, config)
                    estimate = estimate_runtime(
                        prepared=prepared,
                        split=split,
                        config=config,
                        backend_name=selected_backend,
                    )
                    with self._lock:
                        self.state.estimate = estimate
                        self._estimate_signature = self._current_signature_locked(
                            dataset_summary=prepared.summary,
                            config=config,
                            backend_name=selected_backend,
                        )
                        self.state.status_message = (
                            f"Estimate complete: {_fmt_seconds(estimate.estimated_total_s)} total. Starting training."
                        )
                result = train_legacy_ann(
                    project_root=self.project_root,
                    dataset_path=Path(dataset_path),
                    config=config,
                    backend_name=selected_backend,
                    progress_callback=self._on_training_progress,
                    stop_requested=self._cancel_event.is_set,
                    training_provenance=training_provenance,
                )
                metadata = load_training_metadata(result.metadata_path)
                with self._lock:
                    self.state.last_output_path = str(result.artifact_dir)
                    self.state.selected_artifact_path = str(result.artifact_dir)
                    self._selected_artifact_metadata = metadata
                    self._catalog_dirty = True
                    self.state.status_message = (
                        f"Training {result.status}. Saved artifact to {result.artifact_dir.name}."
                    )
                    self._training_status = result.status
            except TorchUnavailableError as exc:
                with self._lock:
                    self.state.status_message = str(exc)
                    self._training_status = "failed"
            except Exception as exc:
                with self._lock:
                    self.state.status_message = f"Training failed: {exc}"
                    self._training_status = "failed"
            finally:
                with self._lock:
                    self.state.training_active = False

        self._worker_thread = threading.Thread(target=_worker, daemon=True)
        self._worker_thread.start()

    def cancel(self) -> None:
        self._cancel_event.set()
        with self._lock:
            if self.state.training_active:
                self.state.status_message = "Cancellation requested. Training will stop after the current batch."

    def shutdown(self, timeout_s: float = 2.0) -> None:
        self._cancel_event.set()
        thread = self._worker_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout_s)

    def hidden_layers_text(self) -> str:
        with self._lock:
            return self._hidden_layers_text

    def artifact_root_text(self) -> str:
        with self._lock:
            return str(self.artifact_root)

    def config_snapshot(self) -> AnnTrainingConfig:
        with self._lock:
            return AnnTrainingConfig(**self.config.to_dict())

    def _job_active(self) -> bool:
        with self._lock:
            return bool(self.state.training_active or self.state.benchmark_active or self.state.sweep_active)

    def _build_exploratory_warning_text(self, summary: ModelingDatasetSummary) -> str:
        complete = int(summary.complete_training_row_count)
        export_rows = int(summary.modeling_export_row_count)
        target = int(summary.target_valid_sample_count)
        usable = complete if complete > 0 else max(0, export_rows)
        target_text = f"target_valid_sample_count={target}" if target > 0 else "target sample count"
        return (
            "This run is not target-complete for thesis model training. "
            f"Training will use {usable} complete exported rows only ({target_text}). "
            "Do not cite as final thesis-valid dataset."
        )

    def _build_training_provenance(
        self,
        *,
        summary: ModelingDatasetSummary | None,
        exploratory_override: bool,
    ) -> dict[str, Any]:
        if summary is None:
            return {"exploratory_training_override": bool(exploratory_override)}
        return {
            "exploratory_training_override": bool(exploratory_override),
            "source_run_valid_for_model_training": summary.valid_for_model_training_flag,
            "source_run_model_training_validity_status": summary.model_training_validity_status,
            "source_validity_reason": summary.model_training_validity_reason,
            "source_validity_warnings": list(summary.model_training_warnings),
            "source_validity_hard_invalidation_reasons": list(
                summary.model_training_hard_invalidation_reasons
            ),
            "ann_training_category": summary.ann_training_category,
            "complete_training_row_count": int(summary.complete_training_row_count),
            "target_valid_sample_count": int(summary.target_valid_sample_count),
            "modeling_export_row_count": int(summary.modeling_export_row_count),
            "modeling_legacy_row_count": int(summary.modeling_legacy_row_count),
            "excluded_incomplete_rows_count": int(summary.incomplete_rows_excluded_from_training),
            "source_run_id": summary.run_id,
            "source_run_path": str(summary.path),
            "source_run_trust_mode": summary.run_trust_mode,
            "source_run_mock_mode_flag": summary.mock_mode_flag,
        }

    def _resolve_dataset_summary(self, selected_path: str, datasets: list[ModelingDatasetSummary]) -> ModelingDatasetSummary | None:
        for dataset in datasets:
            if str(dataset.path) == str(selected_path):
                return dataset
        if selected_path:
            try:
                return load_modeling_dataset_summary(Path(selected_path))
            except Exception:
                return None
        return None

    def _resolve_artifact_metadata(self, selected_path: str, artifacts: list[TrainedArtifactSummary]) -> dict[str, Any] | None:
        for artifact in artifacts:
            if str(artifact.path) == str(selected_path):
                try:
                    return load_training_metadata(artifact.path)
                except Exception:
                    return None
        if selected_path:
            try:
                return load_training_metadata(Path(selected_path))
            except Exception:
                return None
        return None

    def _config_error(self, *, config: AnnTrainingConfig, hidden_layers_text: str) -> str | None:
        if not hidden_layers_text.strip():
            return "Hidden layers must not be empty."
        if self._hidden_layers_parse_error:
            return self._hidden_layers_parse_error
        try:
            validate_training_config(config)
        except Exception as exc:
            return str(exc)
        return None

    def _dataset_pairs(self, summary: ModelingDatasetSummary | None) -> list[tuple[str, str]]:
        if summary is None:
            return [("Dataset", "No modeling dataset selected.")]
        roots = {
            "experiments": "real (data/experiments)",
            "mock": "mock (data/mock_experiments)",
            "archived": "archived",
            "legacy": "legacy / custom output root",
        }
        root_label = roots.get(summary.dataset_scan_root, summary.dataset_scan_root)
        reason_text = (
            "; ".join(summary.legacy_ann_rejection_reasons) if summary.legacy_ann_rejection_reasons else "none"
        )
        export_files = []
        if summary.export_jsonl_path is not None:
            export_files.append("export jsonl")
        if summary.legacy_dat_path is not None:
            export_files.append("legacy dat")
        warnings_text = (
            "; ".join(summary.ann_training_warnings) if summary.ann_training_warnings else "none"
        )
        blocking_text = (
            "; ".join(summary.ann_training_blocking_reasons)
            if summary.ann_training_blocking_reasons
            else "none"
        )
        return [
            ("Run", summary.run_name),
            ("Experiment", summary.catalog_experiment_name or "collect_pose_command_dataset"),
            ("Data root", root_label),
            ("Run ID", summary.run_id or "n/a"),
            ("Timestamp", summary.timestamp_utc or "n/a"),
            ("Mock", str(summary.mock_mode_flag)),
            ("Trust mode", summary.run_trust_mode),
            ("Valid for model training", str(summary.valid_for_model_training_flag)),
            ("Model training validity", summary.model_training_validity_status or "unknown"),
            ("ANN trainability", summary.ann_training_category),
            ("ANN trainable (catalog)", "yes" if summary.trainable_for_legacy_ann else "no"),
            ("ANN blocking reasons", blocking_text),
            ("ANN warnings", warnings_text),
            ("Legacy train rows", str(summary.accepted_legacy_trainable_count)),
            ("Complete training rows", str(summary.complete_training_row_count)),
            ("Target valid samples", str(summary.target_valid_sample_count)),
            ("Export rows", str(summary.modeling_export_row_count)),
            ("Legacy export rows", str(summary.modeling_legacy_row_count)),
            ("Incomplete rows excluded", str(summary.incomplete_rows_excluded_from_training)),
            ("Rejection reasons", reason_text),
            ("Mode", summary.dataset_mode.replace("_", " ")),
            ("Accepted", str(summary.accepted_count)),
            ("Rejected", str(summary.rejected_count)),
            (
                "Acceptance",
                f"{summary.acceptance_rate * 100.0:.1f}%"
                if summary.acceptance_rate is not None
                else "n/a",
            ),
            ("Trust", summary.trust_summary),
            ("Runtime Tip", summary.runtime_tip_summary),
            ("Pretension", summary.pretension_summary),
            ("Exports", ", ".join(export_files) if export_files else "none"),
            ("Full Pose", "yes" if summary.full_pose_available else "no"),
            ("Ordered Context", "yes" if summary.sequential_context_available else "no"),
        ]

    def _system_pairs(
        self,
        *,
        backend_report: BackendReport,
        dataset_summary: ModelingDatasetSummary | None,
        config: AnnTrainingConfig,
    ) -> list[tuple[str, str]]:
        sample_count = (
            int(dataset_summary.accepted_legacy_trainable_count)
            if dataset_summary is not None and int(dataset_summary.accepted_legacy_trainable_count) > 0
            else (int(dataset_summary.accepted_count) if dataset_summary is not None else 0)
        )
        memory_bytes = estimate_memory_bytes(
            sample_count=sample_count,
            batch_size=config.batch_size,
            hidden_layers=config.hidden_layers,
            dtype_name=backend_report.selected_dtype,
        )
        return [
            ("Python", backend_report.python_version),
            ("Torch", backend_report.torch_version or "not installed"),
            ("Platform", backend_report.platform_summary),
            ("Selected Backend", backend_report.selected_backend.upper()),
            ("Recommended", backend_report.recommended_backend.upper()),
            ("Tensor DType", backend_report.selected_dtype),
            ("Accepted Samples", str(sample_count)),
            ("Approx Memory", _fmt_bytes(memory_bytes)),
        ]

    def _estimate_pairs(self, *, estimate: TrainingEstimate | None, stale: bool) -> list[tuple[str, str]]:
        if estimate is None:
            return [("Estimate", "Run a warmup benchmark to estimate runtime.")]
        pairs = [
            ("Estimated Total", _fmt_seconds(estimate.estimated_total_s)),
            ("Estimated / Epoch", _fmt_seconds(estimate.estimated_epoch_s)),
            ("Train Batch Time", _fmt_seconds(estimate.train_batch_time_s)),
            ("Validation Batch Time", _fmt_seconds(estimate.validation_batch_time_s)),
            ("Measured Batches", f"{estimate.benchmark_train_batches} train, {estimate.benchmark_validation_batches} val"),
        ]
        if stale:
            pairs.append(("Estimate State", "stale"))
        else:
            pairs.append(("Estimate State", "current"))
        return pairs

    def _artifact_pairs(self, metadata: dict[str, Any] | None) -> list[tuple[str, str]]:
        if metadata is None:
            return [("Artifact", "No ANN artifact selected.")]
        training = dict(metadata.get("training", {}) or {})
        dataset = dict(metadata.get("dataset", {}) or {})
        backend = dict(metadata.get("backend", {}) or {})
        model = dict(metadata.get("model", {}) or {})
        hl = model.get("hidden_layers")
        return [
            ("Status", str(metadata.get("status", "unknown") or "unknown")),
            ("Dataset", str(dataset.get("run_name", "unknown") or "unknown")),
            ("Backend", str(backend.get("selected_backend", "unknown") or "unknown")),
            ("Hidden layers", str(hl) if hl is not None else "n/a"),
            ("Epochs", str(training.get("epochs_completed", "n/a"))),
            ("Best Epoch", str(training.get("best_epoch", "n/a"))),
            ("Best Val", _fmt_metric(training.get("best_validation_loss"))),
            ("Test Loss", _fmt_metric(training.get("test_loss"))),
        ]

    def _visualization_model(self, metadata: dict[str, Any] | None) -> VisualizationModel:
        if self._training_history_train:
            payload = {
                "training": {
                    "epochs_completed": len(self._training_history_train),
                    "best_epoch": self.state.current_epoch or len(self._training_history_train),
                    "best_validation_loss": self.state.current_validation_loss,
                }
            }
            return build_training_visualization(
                status=self._training_status,
                train_losses=self._training_history_train,
                validation_losses=self._training_history_validation,
                metadata=payload,
            )
        if metadata is not None:
            train_losses, validation_losses = load_loss_history(Path(metadata.get("files", {}).get("loss_history_path", "")))
            return build_training_visualization(
                status=str(metadata.get("status", "unknown") or "unknown"),
                train_losses=train_losses,
                validation_losses=validation_losses,
                metadata=metadata,
            )
        return VisualizationModel(summary_lines=["No training history loaded."])

    def _on_training_progress(self, progress: TrainingProgress) -> None:
        with self._lock:
            self.state.current_epoch = int(progress.epoch)
            self.state.total_epochs = int(progress.total_epochs)
            self.state.current_train_loss = float(progress.train_loss)
            self.state.current_validation_loss = (
                float(progress.validation_loss)
                if progress.validation_loss is not None and not math.isnan(float(progress.validation_loss))
                else None
            )
            self.state.elapsed_s = float(progress.elapsed_s)
            self.state.remaining_s = progress.remaining_s
            self.state.status_message = (
                f"Training epoch {progress.epoch}/{progress.total_epochs} | "
                f"train={progress.train_loss:.6f}"
                + (
                    f" | val={progress.validation_loss:.6f}"
                    if progress.validation_loss is not None and not math.isnan(float(progress.validation_loss))
                    else ""
                )
            )
            self._training_history_train.append(float(progress.train_loss))
            self._training_history_validation.append(
                float(progress.validation_loss)
                if progress.validation_loss is not None
                else float("nan")
            )
            self._training_status = progress.status

    def _selected_backend_name_or_report(self) -> str:
        selected = getattr(self, "_selected_backend_name", "")
        if selected:
            return str(selected)
        report = self.state.backend_report or detect_training_backends()
        return report.selected_backend

    def _invalidate_estimate_locked(self) -> None:
        self.state.estimate = None
        self._estimate_signature = None

    def _current_signature_locked(
        self,
        *,
        dataset_summary: ModelingDatasetSummary | None,
        config: AnnTrainingConfig,
        backend_name: str,
    ) -> tuple[Any, ...]:
        dataset_signature = None
        if dataset_summary is not None:
            dataset_signature = (
                str(dataset_summary.path),
                dataset_summary.accepted_count,
                dataset_summary.full_pose_available,
            )
        return (
            dataset_signature,
            tuple(int(value) for value in config.hidden_layers),
            float(config.learning_rate),
            int(config.batch_size),
            int(config.epochs),
            str(config.loss_kind),
            float(config.pose_orientation_scale),
            int(config.random_seed),
            float(config.train_ratio),
            float(config.validation_ratio),
            float(config.test_ratio),
            str(backend_name),
        )

    def _estimate_is_stale(
        self,
        estimate_signature: tuple[Any, ...] | None,
        dataset_summary: ModelingDatasetSummary | None,
        config: AnnTrainingConfig,
        backend_report: BackendReport,
    ) -> bool:
        if estimate_signature is None:
            return True
        current_signature = self._current_signature_locked(
            dataset_summary=dataset_summary,
            config=config,
            backend_name=backend_report.selected_backend,
        )
        return current_signature != estimate_signature


def _fmt_metric(value: Any) -> str:
    if value in (None, "", "None"):
        return "n/a"
    return f"{float(value):.6f}"


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


def _fmt_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"
