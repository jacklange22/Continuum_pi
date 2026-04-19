"""Controller for the ANN training popout."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import threading
from typing import Any

from continuum_robot.gui.experiment_visualization import VisualizationModel
from continuum_robot.modeling.ann_training import (
    AnnTrainingConfig,
    BackendReport,
    ModelingDatasetSummary,
    TorchUnavailableError,
    TrainedArtifactSummary,
    TrainingEstimate,
    TrainingProgress,
    build_grouped_split,
    build_training_visualization,
    default_artifact_root,
    detect_training_backends,
    discover_modeling_datasets,
    discover_trained_artifacts,
    estimate_memory_bytes,
    estimate_runtime,
    load_loss_history,
    load_modeling_dataset_summary,
    load_training_metadata,
    prepare_legacy_ann_dataset,
    train_legacy_ann,
    validate_training_config,
)


@dataclass
class AnnTrainingViewState:
    """UI-facing snapshot for the training popout."""

    datasets: list[ModelingDatasetSummary] = field(default_factory=list)
    selected_dataset_path: str = ""
    dataset_summary_pairs: list[tuple[str, str]] = field(default_factory=list)
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
            estimate = self.state.estimate
            estimate_signature = self._estimate_signature
            selected_dataset_summary = self._selected_dataset_summary
            selected_artifact_metadata = self._selected_artifact_metadata

        datasets = self.state.datasets
        artifacts = self.state.artifacts
        if catalog_dirty:
            datasets = discover_modeling_datasets(output_root=self.dataset_output_root)
            artifacts = discover_trained_artifacts(artifact_root=self.artifact_root)
            if not selected_dataset_path and datasets:
                selected_dataset_path = str(datasets[0].path)
            if not selected_artifact_path and artifacts:
                selected_artifact_path = str(artifacts[0].path)
            selected_dataset_summary = self._resolve_dataset_summary(selected_dataset_path, datasets)
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
        can_benchmark = bool(selected_dataset_summary) and config_error is None and not training_active and not benchmark_active and backend_report.torch_available
        can_train = can_benchmark
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
            self.state.visualization_model = visualization_model
            if catalog_dirty and not self.state.status_message:
                self.state.status_message = "Select a modeling dataset to prepare ANN training."
            return self.state

    def set_dataset_output_root(self, path: Path) -> None:
        with self._lock:
            self.dataset_output_root = Path(path)
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
            self._hidden_layers_text = str(raw_value)
            try:
                layers = [segment.strip() for segment in str(raw_value).replace(";", ",").split(",")]
                parsed = [int(value) for value in layers if value]
            except ValueError:
                self._hidden_layers_parse_error = "Hidden layers must be a comma-separated list of integers."
                self.config.hidden_layers = []
            else:
                self._hidden_layers_parse_error = None
                self.config.hidden_layers = parsed
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
            return bool(self.state.training_active or self.state.benchmark_active)

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
        export_files = []
        if summary.export_jsonl_path is not None:
            export_files.append("export jsonl")
        if summary.legacy_dat_path is not None:
            export_files.append("legacy dat")
        return [
            ("Run", summary.run_name),
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
        sample_count = dataset_summary.accepted_count if dataset_summary is not None else 0
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
        return [
            ("Status", str(metadata.get("status", "unknown") or "unknown")),
            ("Dataset", str(dataset.get("run_name", "unknown") or "unknown")),
            ("Backend", str(backend.get("selected_backend", "unknown") or "unknown")),
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
