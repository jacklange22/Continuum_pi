"""Controller for the Modeling analysis tab."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import threading

from continuum_robot.data.export_run_bundle import ExportBundleResult, export_run_bundle
from continuum_robot.gui.experiment_visualization import VisualizationModel
from continuum_robot.data.run_management import discover_experiment_run_dirs
from continuum_robot.modeling import (
    ArtifactDetails,
    ModelingDatasetSummary,
    ModelingEvaluationConfig,
    ModelingEvaluationResult,
    TorchUnavailableError,
    TrainedArtifactSummary,
    build_artifact_summary_pairs,
    build_dataset_summary_pairs,
    build_evaluation_summary_pairs,
    default_artifact_root,
    default_results_root,
    discover_modeling_datasets,
    discover_trained_artifacts,
    evaluate_models,
    load_trained_artifact_details,
)
from continuum_robot.modeling.analysis import PastEvaluation, discover_past_evaluations
from continuum_robot.modeling.ann_training import (
    MIN_COMPLETE_ROWS_FOR_TRAINING,
    RowFilterReport,
    validate_legacy_ann_rows,
)


# Operator-visible threshold for "eval is too small to trust". RMSE on <100 samples is
# noisy enough that the headline number can swing by tens of percent across re-runs.
MIN_EVAL_SAMPLES_WARN_THRESHOLD = 100
from continuum_robot.modeling.two_segment import (
    TwoSegmentModelingConfig,
    TwoSegmentModelingResult,
    load_two_segment_modeling_dataset,
    run_two_segment_modeling,
)


@dataclass(frozen=True)
class HeadlineMetric:
    """One per-model RMSE for the big-number headline row in the Modeling tab.

    The tab uses ``rmse_mm`` to color-code the block (green ≤ 1mm, amber ≤ 3mm,
    red > 3mm) so the operator gets at-a-glance verdict on each model's performance
    against the sub-millimeter surgical-accuracy target. ``per_axis_rmse_mm`` powers
    the tile tooltip so hovering shows the X/Y/Z breakdown without scrolling.
    """

    label: str
    model_key: str
    rmse_mm: float | None
    status: str
    reason: str = ""
    per_axis_rmse_mm: tuple[float, float, float] | None = None
    mean_position_error_mm: float | None = None
    max_position_error_mm: float | None = None


@dataclass
class ModelingViewState:
    """UI-facing modeling tab snapshot."""

    datasets: list[ModelingDatasetSummary] = field(default_factory=list)
    selected_dataset_path: str = ""
    dataset_summary_pairs: list[tuple[str, str]] = field(default_factory=list)
    artifacts: list[TrainedArtifactSummary] = field(default_factory=list)
    selected_artifact_path: str = ""
    artifact_summary_pairs: list[tuple[str, str]] = field(default_factory=list)
    evaluation_summary_pairs: list[tuple[str, str]] = field(default_factory=list)
    headline_rmse_pairs: list[tuple[str, str]] = field(default_factory=list)
    headline_metrics: list[HeadlineMetric] = field(default_factory=list)
    past_evaluations: list[PastEvaluation] = field(default_factory=list)
    # Optional separate test dataset (Wolfe MS thesis §3.2.3 cross-acquisition eval).
    # When set, the ANN artifact is trained on selected_dataset_path but evaluated
    # against this dataset instead. Empty string = no override (default behavior).
    selected_test_dataset_path: str = ""
    # Eval-dataset sample-count warning. The UI shows a chip with this text whenever the
    # *effective* evaluation dataset has fewer than ``MIN_EVAL_SAMPLES_WARN_THRESHOLD``
    # complete rows — small evals produce noisy RMSE numbers that won't reproduce.
    eval_sample_count_warning: str = ""
    # Set when the selected ANN artifact's linked training dataset doesn't match the
    # currently-selected dataset on this tab. Common operator mistake — surface it as a
    # chip so the resulting evaluation numbers aren't read as "ANN trained on this".
    artifact_dataset_mismatch: str = ""
    # Set by validate_rows_for_eval(). UI surfaces the row-filter status next to the
    # Validate Rows button so operators can audit dataset health without leaving the tab.
    row_filter_status_text: str = ""
    # Friendly headline label shown at the top of the Per-Model RMSE card. Updates as
    # the operator changes selection — before any eval, it shows "Ready to compare on
    # <dataset_name>"; after an eval, it shows the eval scope + sample count.
    headline_status_label: str = ""
    include_mike: bool = True
    include_camarillo: bool = True
    include_ann: bool = True
    evaluation_scope: str = "artifact_test_split"
    evaluation_active: bool = False
    status_message: str = "Select a dataset, choose models, then run a comparison."
    last_output_path: str | None = None
    can_evaluate: bool = False
    # True when the last evaluation reused training samples as its test split (any scope
    # other than a separately collected test dataset). UI surfaces this as a chip so the
    # operator never silently cites same-session numbers as thesis-grade.
    last_eval_same_session: bool = True
    # True when the last evaluation used a separate test acquisition AND that acquisition
    # is an angular_test_mesh — Wolfe §3.2.3 cross-acquisition methodology achieved.
    last_eval_thesis_grade: bool = False
    artifact_details: ArtifactDetails | None = None
    visualization_model: VisualizationModel = field(
        default_factory=lambda: VisualizationModel(summary_lines=["No modeling results loaded."])
    )
    two_segment_dataset_runs: list[str] = field(default_factory=list)
    selected_two_segment_run_paths: list[str] = field(default_factory=list)
    two_segment_summary_pairs: list[tuple[str, str]] = field(default_factory=list)
    two_segment_include_linear: bool = True
    two_segment_include_ann: bool = True
    two_segment_include_camarillo: bool = False
    two_segment_include_mike: bool = False
    two_segment_strict_mode: bool = True
    two_segment_allow_lower_trust: bool = False
    two_segment_label_mode: str = "auto"
    two_segment_include_orientation_if_available: bool = False
    two_segment_ann_sweep_enabled: bool = False
    two_segment_ann_hidden_layers: str = "128,128"
    two_segment_ann_epochs: int = 200
    two_segment_test_fraction: float = 0.25
    two_segment_active: bool = False
    two_segment_can_run: bool = False
    two_segment_last_output_path: str | None = None
    two_segment_can_open_output: bool = False
    two_segment_can_export_output: bool = False
    two_segment_export_path: str | None = None
    # ── External Model Comparison card ───────────────────────────────────────
    # The operator picks Model A from the artifact dropdown (above) and uploads
    # a .pt for Model B via a file dialog. The comparison runs on the currently
    # selected dataset and produces two side-by-side 3D scatter plots colored
    # by per-sample tip-position error. State below tracks the upload path,
    # async status, error/status messages, and the most recent result object
    # which the tab passes to build_comparison_figure() for rendering.
    comparison_external_model_path: str = ""
    comparison_active: bool = False
    comparison_status_message: str = (
        "Pick a dataset, select an ANN artifact above for Model A, "
        "upload a .pt for Model B, then click Generate."
    )
    comparison_error_message: str = ""
    comparison_result_id: int = 0  # bump-each-render counter the tab can watch


class ModelingController:
    """Owns Modeling tab state and runs evaluations asynchronously."""

    def __init__(
        self,
        *,
        project_root: Path,
        dataset_output_root: Path,
        artifact_root: Path | None = None,
        results_root: Path | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.dataset_output_root = Path(dataset_output_root)
        self.artifact_root = Path(artifact_root) if artifact_root is not None else default_artifact_root(self.project_root)
        self.results_root = Path(results_root) if results_root is not None else default_results_root(self.project_root)
        self._lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None
        self._catalog_dirty = True
        self._selected_dataset_summary: ModelingDatasetSummary | None = None
        self._selected_artifact_details: ArtifactDetails | None = None
        self._last_result: ModelingEvaluationResult | None = None
        self._last_two_segment_result: TwoSegmentModelingResult | None = None
        self.config = ModelingEvaluationConfig(results_root=str(self.results_root))
        self.state = ModelingViewState()
        # Per-tick disk-read caches keyed by (path, mtime). The 5 Hz refresh timer was
        # hammering summary.json reads and JSONL parses on every tick, which blocked the
        # GUI paint thread and made scrolling feel choppy. mtime-keyed caches keep the
        # cache fresh when data actually changes on disk while making steady-state ticks
        # nearly free (microsecond stat() vs. multi-millisecond read+parse).
        self._eval_warn_cache: tuple[tuple[str, float, str], str] | None = None
        self._past_eval_cache: tuple[tuple[str, float, str], list[PastEvaluation]] | None = None
        self._trainability_cache: tuple[tuple, list[tuple[str, str]]] | None = None

    def refresh(self) -> ModelingViewState:
        with self._lock:
            catalog_dirty = self._catalog_dirty
            selected_dataset_path = self.state.selected_dataset_path
            selected_artifact_path = self.state.selected_artifact_path
            selected_two_segment_run_paths = list(self.state.selected_two_segment_run_paths)
            selected_dataset_summary = self._selected_dataset_summary
            selected_artifact_details = self._selected_artifact_details
            evaluation_active = self.state.evaluation_active
            two_segment_active = self.state.two_segment_active

        datasets = self.state.datasets
        artifacts = self.state.artifacts
        two_segment_runs = list(self.state.two_segment_dataset_runs)
        if catalog_dirty:
            datasets = discover_modeling_datasets(
                project_root=self.project_root,
                output_root=self.dataset_output_root,
                include_mock_experiments=False,
                include_archived_experiments=False,
            )
            # The Modeling tab compares forward models (cable → pose) against Mike/Camarillo.
            # Inverse (xyz → cable) artifacts can't fit that workflow; hide them from the
            # picker rather than letting the operator hit an "unavailable" later.
            artifacts = discover_trained_artifacts(
                artifact_root=self.artifact_root,
                include_inverse=False,
            )
            two_segment_runs = [str(path) for path in self.discover_two_segment_dataset_runs()]
            if not selected_dataset_path and datasets:
                selected_dataset_path = str(datasets[0].path)
            if artifacts and selected_artifact_path == "":
                selected_artifact_path = str(artifacts[0].path)
            if not selected_two_segment_run_paths and two_segment_runs:
                selected_two_segment_run_paths = [two_segment_runs[0]]
            selected_dataset_summary = self._resolve_dataset_summary(selected_dataset_path, datasets)
            selected_artifact_details = self._resolve_artifact_details(selected_artifact_path, artifacts)
        if (not catalog_dirty) and selected_dataset_summary is None and selected_dataset_path:
            selected_dataset_summary = self._resolve_dataset_summary(selected_dataset_path, datasets)
        if (not catalog_dirty) and selected_artifact_details is None and selected_artifact_path:
            selected_artifact_details = self._resolve_artifact_details(selected_artifact_path, artifacts)

        can_evaluate = bool(selected_dataset_summary) and (
            bool(self.config.include_mike)
            or bool(self.config.include_camarillo)
            or (bool(self.config.include_ann) and bool(selected_artifact_details))
        ) and not evaluation_active
        two_segment_trainability = self._two_segment_trainability_pairs(
            selected_two_segment_run_paths,
            allow_lower_trust=bool(self.state.two_segment_allow_lower_trust and not self.state.two_segment_strict_mode),
        )
        two_segment_can_run = bool(selected_two_segment_run_paths) and not two_segment_active and bool(self._two_segment_model_keys())

        with self._lock:
            self._catalog_dirty = False
            self._selected_dataset_summary = selected_dataset_summary
            self._selected_artifact_details = selected_artifact_details
            self.state.datasets = datasets
            self.state.selected_dataset_path = selected_dataset_path
            self.state.dataset_summary_pairs = build_dataset_summary_pairs(selected_dataset_summary) if selected_dataset_summary is not None else []
            self.state.artifacts = artifacts
            self.state.selected_artifact_path = selected_artifact_path
            self.state.artifact_details = selected_artifact_details
            self.state.artifact_summary_pairs = build_artifact_summary_pairs(selected_artifact_details)
            self.state.evaluation_summary_pairs = build_evaluation_summary_pairs(self._last_result)
            self.state.headline_rmse_pairs = self._build_headline_rmse_pairs(self._last_result)
            self.state.headline_metrics = self._build_headline_metrics(self._last_result)
            self.state.past_evaluations = self._discover_past_evaluations(selected_dataset_summary)
            self.state.last_eval_same_session = self._eval_used_same_session(self._last_result)
            self.state.last_eval_thesis_grade = self._eval_is_thesis_grade(self._last_result)
            self.state.eval_sample_count_warning = self._eval_sample_count_warning(
                selected_dataset_summary, selected_test_dataset_path=self.state.selected_test_dataset_path
            )
            self.state.artifact_dataset_mismatch = self._artifact_dataset_mismatch_warning(
                dataset_summary=selected_dataset_summary,
                artifact_details=selected_artifact_details,
            )
            self.state.headline_status_label = self._headline_status_label(
                dataset_summary=selected_dataset_summary,
                last_result=self._last_result,
            )
            self.state.include_mike = bool(self.config.include_mike)
            self.state.include_camarillo = bool(self.config.include_camarillo)
            self.state.include_ann = bool(self.config.include_ann)
            self.state.evaluation_scope = str(self.config.evaluation_scope)
            self.state.can_evaluate = can_evaluate
            self.state.two_segment_dataset_runs = two_segment_runs
            self.state.selected_two_segment_run_paths = selected_two_segment_run_paths
            self.state.two_segment_summary_pairs = two_segment_trainability
            self.state.two_segment_can_run = two_segment_can_run
            self.state.two_segment_can_open_output = bool(self.state.two_segment_last_output_path)
            self.state.two_segment_can_export_output = bool(self.state.two_segment_last_output_path)
            return self.state

    def select_dataset(self, path: str) -> None:
        with self._lock:
            self.state.selected_dataset_path = str(path)
            self._selected_dataset_summary = None

    def select_artifact(self, path: str) -> None:
        with self._lock:
            self.state.selected_artifact_path = str(path)
            self._selected_artifact_details = None

    def set_include_mike(self, value: bool) -> None:
        with self._lock:
            self.config.include_mike = bool(value)

    def set_include_camarillo(self, value: bool) -> None:
        with self._lock:
            self.config.include_camarillo = bool(value)

    def set_include_ann(self, value: bool) -> None:
        with self._lock:
            self.config.include_ann = bool(value)

    def validate_rows_for_eval(self) -> RowFilterReport | None:
        """Run the complete_rows_only row filter on the effective eval dataset.

        Picks the test dataset if set (Wolfe-style override), else the selected
        training dataset. Updates ``state.row_filter_status_text`` with the
        result summary. Returns the report so callers can inspect details.
        """
        with self._lock:
            test_path = self.state.selected_test_dataset_path
            dataset_summary = self._selected_dataset_summary
        target: Path | None = None
        if test_path:
            target = Path(test_path)
        elif dataset_summary is not None:
            target = Path(dataset_summary.path)
        if target is None:
            with self._lock:
                self.state.row_filter_status_text = "Select a dataset first."
            return None
        try:
            report = validate_legacy_ann_rows(target)
        except Exception as exc:
            with self._lock:
                self.state.row_filter_status_text = f"Row validation failed: {exc}"
            return None
        status = "ok" if report.can_train else f"blocked: {report.block_reason}"
        text = (
            f"Row filter on {target.name}: {report.complete_row_count} complete, "
            f"{report.excluded_row_count} excluded "
            f"(target={report.target_complete_row_count}, {status})."
        )
        with self._lock:
            self.state.row_filter_status_text = text
            self.state.status_message = text
        return report

    def set_test_dataset_path(self, value: str) -> None:
        """Optional override dataset to evaluate against (Wolfe §3.2.3).

        Empty string clears the override, falling back to the artifact's split or
        the full selected dataset (per the evaluation_scope combo).
        """
        with self._lock:
            self.state.selected_test_dataset_path = str(value or "").strip()

    def set_evaluation_scope(self, value: str) -> None:
        with self._lock:
            self.config.evaluation_scope = str(value)

    def set_dataset_output_root(self, path: Path) -> None:
        with self._lock:
            self.dataset_output_root = Path(path)
            self._catalog_dirty = True

    def set_artifact_root(self, path: Path) -> None:
        with self._lock:
            self.artifact_root = Path(path)
            self._catalog_dirty = True

    def select_two_segment_runs(self, paths: list[str]) -> None:
        selected = {str(path) for path in paths if str(path).strip()}
        with self._lock:
            visible = set(self.state.two_segment_dataset_runs)
            self.state.selected_two_segment_run_paths = [
                path for path in self.state.two_segment_dataset_runs if path in selected or path not in visible
            ]
            if not self.state.selected_two_segment_run_paths:
                self.state.selected_two_segment_run_paths = [path for path in paths if str(path).strip()]

    def set_two_segment_model_enabled(self, model_key: str, value: bool) -> None:
        with self._lock:
            key = str(model_key)
            if key == "linear_baseline":
                self.state.two_segment_include_linear = bool(value)
            elif key == "ann":
                self.state.two_segment_include_ann = bool(value)
            elif key == "camarillo":
                self.state.two_segment_include_camarillo = bool(value)
            elif key in {"mike", "mike_constant_curvature"}:
                self.state.two_segment_include_mike = bool(value)

    def set_two_segment_strict_mode(self, value: bool) -> None:
        with self._lock:
            self.state.two_segment_strict_mode = bool(value)
            if bool(value):
                self.state.two_segment_allow_lower_trust = False

    def set_two_segment_allow_lower_trust(self, value: bool) -> None:
        with self._lock:
            self.state.two_segment_allow_lower_trust = bool(value)
            if bool(value):
                self.state.two_segment_strict_mode = False

    def discover_two_segment_dataset_runs(self) -> list[Path]:
        """Return two-segment collect-pose runs for future GUI integration."""
        return discover_experiment_run_dirs(
            self.project_root,
            experiment_name="two_segment_collect_pose_command_dataset",
        )

    def validate_two_segment_modeling_trainability(
        self,
        run_dirs: list[Path],
        *,
        allow_lower_trust: bool = False,
    ) -> dict[str, object]:
        """Return a controller-level trainability summary without running models."""
        dataset = load_two_segment_modeling_dataset(run_dirs, allow_lower_trust=bool(allow_lower_trust))
        return {
            "runs_scanned": len(run_dirs),
            "samples_scanned": dataset.accepted_count + dataset.rejected_count,
            "samples_accepted": dataset.accepted_count,
            "samples_rejected": dataset.rejected_count,
            "rejection_reasons": dataset.rejection_counts(),
            "trainable": dataset.accepted_count >= 2,
            "orientation_available": dataset.orientation_available,
            "includes_intermediate_pose": dataset.includes_intermediate_pose,
            "two_coil_xyz_available": dataset.two_coil_xyz_available,
            "two_coil_orientation_available": dataset.two_coil_orientation_available,
        }

    def set_two_segment_label_mode(self, value: str) -> None:
        with self._lock:
            self.state.two_segment_label_mode = str(value or "auto")

    def set_two_segment_include_orientation_if_available(self, value: bool) -> None:
        with self._lock:
            self.state.two_segment_include_orientation_if_available = bool(value)

    def set_two_segment_ann_sweep_enabled(self, value: bool) -> None:
        with self._lock:
            self.state.two_segment_ann_sweep_enabled = bool(value)

    def set_two_segment_ann_hidden_layers(self, value: str) -> None:
        with self._lock:
            self.state.two_segment_ann_hidden_layers = str(value or "128,128")

    def set_two_segment_ann_epochs(self, value: int) -> None:
        with self._lock:
            self.state.two_segment_ann_epochs = max(1, int(value))

    def set_two_segment_test_fraction(self, value: float) -> None:
        with self._lock:
            self.state.two_segment_test_fraction = min(0.9, max(0.05, float(value)))

    def run_two_segment_modeling_analysis(self) -> None:
        with self._lock:
            if self.state.two_segment_active:
                return
            run_dirs = [Path(path) for path in self.state.selected_two_segment_run_paths]
            model_keys = self._two_segment_model_keys()
            allow_lower_trust = bool(self.state.two_segment_allow_lower_trust and not self.state.two_segment_strict_mode)
            label_mode = str(self.state.two_segment_label_mode)
            include_orientation = bool(self.state.two_segment_include_orientation_if_available)
            ann_sweep = bool(self.state.two_segment_ann_sweep_enabled)
            ann_hidden_layers = str(self.state.two_segment_ann_hidden_layers)
            ann_epochs = int(self.state.two_segment_ann_epochs)
            test_fraction = float(self.state.two_segment_test_fraction)
            self.state.two_segment_active = True
            self.state.status_message = "Running two-segment modeling analysis..."
        if not run_dirs:
            with self._lock:
                self.state.two_segment_active = False
                self.state.status_message = "Select one or more two-segment dataset runs first."
            return
        if not model_keys:
            with self._lock:
                self.state.two_segment_active = False
                self.state.status_message = "Select at least one two-segment model family."
            return
        self._worker_thread = threading.Thread(
            target=self._two_segment_modeling_worker,
            kwargs={
                "run_dirs": run_dirs,
                "model_keys": model_keys,
                "allow_lower_trust": allow_lower_trust,
                "label_mode": label_mode,
                "include_orientation": include_orientation,
                "ann_sweep": ann_sweep,
                "ann_hidden_layers": ann_hidden_layers,
                "ann_epochs": ann_epochs,
                "test_fraction": test_fraction,
            },
            daemon=True,
        )
        self._worker_thread.start()

    def export_last_two_segment_modeling_bundle(self) -> ExportBundleResult:
        with self._lock:
            output_path = self.state.two_segment_last_output_path
        if not output_path:
            raise ValueError("No two-segment modeling output is available to export.")
        result = export_run_bundle(
            run_dir=Path(output_path),
            project_root=self.project_root,
            include_samples=False,
            include_debug=False,
            make_zip=True,
        )
        with self._lock:
            self.state.two_segment_export_path = str(result.final_path)
            self.state.status_message = f"Exported two-segment modeling bundle to {result.final_path}."
        return result

    def evaluate(self) -> None:
        with self._lock:
            if self.state.evaluation_active:
                return
            dataset_summary = self._selected_dataset_summary
            artifact_details = self._selected_artifact_details
            test_dataset_path = self.state.selected_test_dataset_path
            config = ModelingEvaluationConfig(
                include_mike=bool(self.config.include_mike),
                include_camarillo=bool(self.config.include_camarillo),
                include_ann=bool(self.config.include_ann),
                evaluation_scope=str(self.config.evaluation_scope),
                results_root=str(self.results_root),
                geometry=self.config.geometry,
            )
            self.state.evaluation_active = True
            self.state.status_message = "Running modeling comparison..."
        if dataset_summary is None:
            with self._lock:
                self.state.evaluation_active = False
                self.state.status_message = "Select a modeling dataset first."
            return
        self._worker_thread = threading.Thread(
            target=self._evaluate_worker,
            kwargs={
                "dataset_path": dataset_summary.path,
                "artifact_path": (artifact_details.summary.path if artifact_details is not None else None),
                "config": config,
                "test_dataset_path": Path(test_dataset_path) if test_dataset_path else None,
            },
            daemon=True,
        )
        self._worker_thread.start()

    # ── External Model Comparison ──────────────────────────────────────────
    def set_comparison_external_model_path(self, path: str) -> None:
        """Operator picked a .pt file via the Upload dialog. Just store the path.

        We don't validate or load the model here — that happens lazily when the
        operator clicks Generate, so a stale path doesn't block the rest of
        refresh() and a bad file produces an inline error rather than a startup
        crash.
        """
        with self._lock:
            self.comparison_external_model_path = str(path or "").strip()
            self.state.comparison_external_model_path = self.comparison_external_model_path
            # Clear stale status when the operator picks something new.
            self.state.comparison_error_message = ""
            if self.comparison_external_model_path:
                self.state.comparison_status_message = (
                    f"Ready: Model B = {Path(self.comparison_external_model_path).name}. "
                    "Click Generate to compute the side-by-side plot."
                )
            else:
                self.state.comparison_status_message = (
                    "Pick a dataset, select an ANN artifact above for Model A, "
                    "upload a .pt for Model B, then click Generate."
                )

    def run_external_comparison(self) -> None:
        """Kick off the side-by-side comparison on a background thread.

        Validates that Model A (current ANN selection), Model B (uploaded .pt),
        and a dataset are all selected. The worker fills ``self._last_comparison``
        and bumps ``comparison_result_id`` so the tab notices and re-renders.
        """
        with self._lock:
            if self.state.comparison_active:
                return
            dataset_path = (
                Path(self.state.selected_dataset_path)
                if self.state.selected_dataset_path
                else None
            )
            artifact_a = self._selected_artifact_details
            external_b = (
                Path(self.state.comparison_external_model_path)
                if self.state.comparison_external_model_path
                else None
            )
        if dataset_path is None:
            with self._lock:
                self.state.comparison_error_message = "Select a dataset first."
            return
        if artifact_a is None:
            with self._lock:
                self.state.comparison_error_message = (
                    "Select an ANN artifact (Model A) from the dropdown above first."
                )
            return
        if external_b is None:
            with self._lock:
                self.state.comparison_error_message = (
                    "Upload a .pt file for Model B first."
                )
            return
        with self._lock:
            self.state.comparison_active = True
            self.state.comparison_error_message = ""
            self.state.comparison_status_message = "Running side-by-side comparison…"
        self._worker_thread = threading.Thread(
            target=self._comparison_worker,
            kwargs={
                "model_a_path": Path(artifact_a.summary.path),
                "model_b_path": external_b,
                "dataset_path": dataset_path,
            },
            daemon=True,
        )
        self._worker_thread.start()

    def _comparison_worker(
        self,
        *,
        model_a_path: Path,
        model_b_path: Path,
        dataset_path: Path,
    ) -> None:
        """Background worker — loads both models, runs inference, builds result.

        Lazy-imports model_comparison so the controller stays importable even on
        machines without PyTorch (the lazy import keeps test surfaces small).
        """
        try:
            from continuum_robot.modeling.model_comparison import (
                run_side_by_side_comparison,
            )

            result = run_side_by_side_comparison(
                model_a_path=model_a_path,
                model_b_path=model_b_path,
                dataset_path=dataset_path,
            )
        except Exception as exc:  # noqa: BLE001 — surface anything in the UI rather than crash
            with self._lock:
                self.state.comparison_active = False
                self.state.comparison_error_message = (
                    f"Side-by-side comparison failed: {exc}"
                )
                self.state.comparison_status_message = (
                    "Comparison failed — see error above."
                )
            return
        with self._lock:
            self._last_comparison = result
            self.state.comparison_active = False
            self.state.comparison_status_message = (
                f"Comparison complete: {result.dataset_run_name} ({result.a_stats.sample_count} samples). "
                f"Model A mean={result.a_stats.mean_mm:.2f} mm, "
                f"Model B mean={result.b_stats.mean_mm:.2f} mm."
            )
            self.state.comparison_result_id += 1

    def get_last_comparison_result(self):
        """Return the most recent comparison result, or None if not run yet.

        Kept as a method (not a state field) because matplotlib Figure / numpy
        arrays inside a dataclass-typed state would force the entire state
        snapshot pattern to copy these heavy objects every refresh. Tab code
        pulls this on demand when ``comparison_result_id`` changes.
        """
        with self._lock:
            return getattr(self, "_last_comparison", None)

    def save_last_comparison_png(self, target_path: Path, *, dpi: int = 300) -> Path | None:
        """Save the most recent comparison as a PNG at the requested path.

        Returns the actual saved path, or None if there's nothing to save. Errors
        are surfaced into ``comparison_error_message`` so the tab can echo them
        without a tracebacks-in-dialogs situation.
        """
        result = self.get_last_comparison_result()
        if result is None:
            with self._lock:
                self.state.comparison_error_message = (
                    "Run a comparison first — nothing to save yet."
                )
            return None
        try:
            from continuum_robot.modeling.model_comparison import save_comparison_png

            saved = save_comparison_png(result, Path(target_path), dpi=int(dpi))
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self.state.comparison_error_message = f"Save failed: {exc}"
            return None
        with self._lock:
            self.state.comparison_status_message = f"Saved comparison PNG to {saved}."
        return saved

    def shutdown(self) -> None:
        thread = self._worker_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def _evaluate_worker(
        self,
        *,
        dataset_path: Path,
        artifact_path: Path | None,
        config: ModelingEvaluationConfig,
        test_dataset_path: Path | None = None,
    ) -> None:
        try:
            result = evaluate_models(
                project_root=self.project_root,
                dataset_path=dataset_path,
                artifact_path=artifact_path,
                config=config,
                test_dataset_path=test_dataset_path,
            )
        except (RuntimeError, ValueError, TorchUnavailableError) as exc:
            with self._lock:
                self.state.evaluation_active = False
                self.state.status_message = str(exc)
            return
        with self._lock:
            self._last_result = result
            self.state.visualization_model = result.visualization_model
            self.state.evaluation_active = False
            self.state.last_output_path = str(result.output_dir)
            self.state.status_message = (
                f"Modeling comparison saved to {result.output_dir.name} "
                f"using {result.selected_sample_count} samples."
            )

    @staticmethod
    def _build_headline_metrics(
        result: ModelingEvaluationResult | None,
    ) -> list[HeadlineMetric]:
        """Structured per-model metrics for the big-number headline row.

        Carries enough information for the UI to color-code each block by RMSE
        threshold and to render "unavailable" tiles with a clear reason rather
        than blanks.
        """
        if result is None:
            return []
        out: list[HeadlineMetric] = []
        for evaluation in result.model_evaluations.values():
            metrics = evaluation.metrics
            axis_rmse = list(metrics.axis_position_rmse_mm or [])
            per_axis = (
                (float(axis_rmse[0]), float(axis_rmse[1]), float(axis_rmse[2]))
                if len(axis_rmse) >= 3
                else None
            )
            out.append(
                HeadlineMetric(
                    label=metrics.label,
                    model_key=metrics.model_key,
                    rmse_mm=(
                        float(metrics.position_rmse_mm)
                        if metrics.status == "completed" and metrics.position_rmse_mm is not None
                        else None
                    ),
                    status=str(metrics.status),
                    reason=str(metrics.reason or ""),
                    per_axis_rmse_mm=per_axis,
                    mean_position_error_mm=(
                        float(metrics.mean_position_error_mm)
                        if metrics.mean_position_error_mm is not None
                        else None
                    ),
                    max_position_error_mm=(
                        float(metrics.max_position_error_mm)
                        if metrics.max_position_error_mm is not None
                        else None
                    ),
                )
            )
        return out

    @staticmethod
    def _build_headline_rmse_pairs(
        result: ModelingEvaluationResult | None,
    ) -> list[tuple[str, str]]:
        """Compact per-model RMSE row for the top of the Comparison panel.

        Renders as ``[("Mike", "9.42 mm"), ("Camarillo", "9.18 mm"), ("ANN", "1.87 mm")]``
        so the operator sees the headline number for every model side-by-side without
        having to scroll the full evaluation pairs block.
        """
        if result is None:
            return []
        pairs: list[tuple[str, str]] = []
        for evaluation in result.model_evaluations.values():
            metrics = evaluation.metrics
            if metrics.status == "completed" and metrics.position_rmse_mm is not None:
                pairs.append((metrics.label, f"{float(metrics.position_rmse_mm):.2f} mm"))
            elif metrics.status != "completed":
                pairs.append((metrics.label, f"unavailable ({metrics.reason or ''})"))
        return pairs

    def _discover_past_evaluations(
        self, dataset_summary: ModelingDatasetSummary | None
    ) -> list[PastEvaluation]:
        """List the most recent saved Modeling evaluations for the selected dataset.

        Lets the operator compare today's run to yesterday's without leaving the tab.
        Cheap — only reads each evaluation's ``summary.json``.

        Cached by (results_root, dir mtime, dataset_run_name) so the 5 Hz refresh timer
        doesn't re-scan the whole evaluations directory on every tick. Directory mtime
        updates when an eval is added/removed, so new runs surface within one tick.
        """
        if dataset_summary is None:
            return []
        try:
            dir_mtime = self.results_root.stat().st_mtime if self.results_root.exists() else 0.0
        except OSError:
            dir_mtime = 0.0
        cache_key = (str(self.results_root), dir_mtime, str(dataset_summary.run_name))
        if self._past_eval_cache is not None and self._past_eval_cache[0] == cache_key:
            return self._past_eval_cache[1]
        try:
            result = discover_past_evaluations(
                project_root=self.project_root,
                results_root=self.results_root,
                dataset_run_name=dataset_summary.run_name,
                limit=10,
            )
        except Exception:
            result = []
        self._past_eval_cache = (cache_key, result)
        return result

    def _eval_sample_count_warning(
        self,
        dataset_summary: ModelingDatasetSummary | None,
        *,
        selected_test_dataset_path: str,
    ) -> str:
        """Return a warning string when the effective evaluation dataset is too small.

        Empty string when the dataset is healthy. Looks at the test dataset path first
        (Wolfe-style override), then falls back to the selected training dataset.
        """
        effective_path: Path | None = None
        if selected_test_dataset_path:
            try:
                effective_path = Path(selected_test_dataset_path)
            except Exception:
                effective_path = None
        elif dataset_summary is not None:
            effective_path = Path(dataset_summary.path)
        if effective_path is None:
            return ""
        # Count rows cheaply from summary.json metrics (no jsonl parse needed).
        # Cache by (path, summary.json mtime) so the 5 Hz refresh tick doesn't re-read
        # the same file on every fire — stat() is microseconds, json.loads is milliseconds.
        summary_path = effective_path / "summary.json"
        try:
            mtime = summary_path.stat().st_mtime
        except OSError:
            return ""
        cache_key = (str(summary_path), mtime, "eval_warn_v1")
        if self._eval_warn_cache is not None and self._eval_warn_cache[0] == cache_key:
            return self._eval_warn_cache[1]
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = payload.get("experiment_metrics", {}) if isinstance(payload, dict) else {}
            accepted = int(metrics.get("accepted_sample_count", 0) or 0)
        except Exception:
            self._eval_warn_cache = (cache_key, "")
            return ""
        if accepted < MIN_EVAL_SAMPLES_WARN_THRESHOLD and accepted > 0:
            result = (
                f"⚠ Evaluation dataset has only {accepted} accepted samples "
                f"(< {MIN_EVAL_SAMPLES_WARN_THRESHOLD}). RMSE on this few rows is noisy; "
                "collect more before citing the number."
            )
        else:
            result = ""
        self._eval_warn_cache = (cache_key, result)
        return result

    @staticmethod
    def _headline_status_label(
        *,
        dataset_summary: ModelingDatasetSummary | None,
        last_result: ModelingEvaluationResult | None,
    ) -> str:
        """Build the friendly status label shown on the Per-Model RMSE card.

        Before any eval: "Ready to compare on <dataset_name>".
        After an eval: "Last eval: <best_label> <rmse> mm on <samples> samples [scope]".
        """
        if last_result is not None:
            best = sorted(
                (
                    evaluation.metrics
                    for evaluation in last_result.model_evaluations.values()
                    if evaluation.metrics.status == "completed"
                    and evaluation.metrics.position_rmse_mm is not None
                ),
                key=lambda m: float(m.position_rmse_mm),
            )
            scope = str(last_result.evaluation_scope_used or "").strip().lower()
            scope_label = {
                "separate_test_dataset": "separate test dataset",
                "artifact_test_split": "held-out split",
                "full_dataset": "full dataset",
            }.get(scope, scope or "unknown scope")
            if best:
                return (
                    f"Last eval: {best[0].label} {float(best[0].position_rmse_mm):.2f} mm "
                    f"on {last_result.selected_sample_count} samples ({scope_label})."
                )
            return f"Last eval completed on {last_result.selected_sample_count} samples ({scope_label})."
        if dataset_summary is not None:
            return f"Ready to compare on {dataset_summary.run_name}."
        return "Select a dataset to begin."

    @staticmethod
    def _artifact_dataset_mismatch_warning(
        *,
        dataset_summary: ModelingDatasetSummary | None,
        artifact_details: ArtifactDetails | None,
    ) -> str:
        """Detect when the selected ANN artifact was trained on a different dataset.

        Operators routinely pick an artifact intending "the model trained on this run"
        but actually point at one trained elsewhere. The resulting numbers then mean
        something different. Surface this as a chip rather than letting it propagate
        into a published RMSE.
        """
        if dataset_summary is None or artifact_details is None:
            return ""
        metadata = dict(artifact_details.metadata or {})
        artifact_dataset = dict(metadata.get("dataset", {}) or {})
        linked_run_name = str(artifact_dataset.get("run_name", "") or "").strip()
        linked_path = str(artifact_dataset.get("path", "") or "").strip()
        if not linked_run_name and not linked_path:
            return ""
        current_run_name = str(dataset_summary.run_name or "").strip()
        current_path = str(dataset_summary.path)
        # Match by run_name OR by resolved path — either is sufficient.
        try:
            if linked_path and Path(linked_path).resolve() == Path(current_path).resolve():
                return ""
        except Exception:
            pass
        if linked_run_name and linked_run_name == current_run_name:
            return ""
        return (
            f"⚠ Selected ANN artifact was trained on '{linked_run_name or 'a different dataset'}'. "
            f"You're evaluating against '{current_run_name}' — that's a cross-acquisition test "
            "(legitimate, but the numbers are NOT 'how well this model fits its own training data')."
        )

    @staticmethod
    def _eval_is_thesis_grade(result: ModelingEvaluationResult | None) -> bool:
        """True when the last evaluation matches Wolfe's §3.2.3 cross-acquisition setup.

        Strict gate (all conditions must hold):
          1. ``evaluation_scope_used == "separate_test_dataset"`` — a real test override.
          2. Train and test refer to different runs — both ``run_id`` AND resolved
             ``path`` must differ. Defends against accidental clone-of-folder loops.
          3. Test dataset is a real-hardware acquisition:
             ``run_trust_mode == "thesis_trusted"``, ``mock_mode_flag is not True``,
             ``"servo_only" not in run_trust_mode``. Picked per operator answer to the
             "Real-hardware rule" audit question.
          4. Test dataset has enough complete rows to evaluate against:
             ``validate_legacy_ann_rows(...).complete_row_count >=
             MIN_COMPLETE_ROWS_FOR_TRAINING``. Tolerates some excluded rows per the
             operator's "resistant to <100% success" preference.
          5. ANN artifact details MUST be loaded — Wolfe §3.2.3 is an ANN methodology,
             so a result with no artifact_details (Mike/Camarillo-only or a stale
             selection that failed to load) cannot be certified thesis-grade. The chip
             is meant to signal "this evaluation is comparable to Wolfe's ANN numbers"
             and that requires an actual ANN provenance to inspect.
          6. ANN artifact was NOT trained with the exploratory override:
             ``training_provenance.exploratory_training_override is not True``.
             Exploratory artifacts carry their own warning per design.

        Any failure ⇒ False ⇒ the green chip stays hidden and the amber same-session
        caveat (or no chip) takes over.
        """
        if result is None:
            return False
        scope = str(result.evaluation_scope_used or "").strip().lower()
        if scope != "separate_test_dataset":
            return False
        train_summary = result.dataset_summary
        test_summary = result.test_dataset_summary
        if train_summary is None or test_summary is None:
            return False
        # 2 — train and test must differ on both run_id AND path.
        train_run_id = str(train_summary.run_id or "").strip()
        test_run_id = str(test_summary.run_id or "").strip()
        if train_run_id and test_run_id and train_run_id == test_run_id:
            return False
        try:
            if Path(train_summary.path).resolve() == Path(test_summary.path).resolve():
                return False
        except Exception:
            if str(train_summary.path) == str(test_summary.path):
                return False
        # 3 — test dataset must be real hardware.
        if test_summary.mock_mode_flag is True:
            return False
        test_trust_mode = str(test_summary.run_trust_mode or "").strip().lower()
        if test_trust_mode != "thesis_trusted":
            return False
        # 4 — test dataset must have enough complete rows.
        try:
            row_filter = validate_legacy_ann_rows(Path(test_summary.path))
        except Exception:
            return False
        if int(row_filter.complete_row_count) < int(MIN_COMPLETE_ROWS_FOR_TRAINING):
            return False
        # 5 — ANN artifact details must be present (Wolfe §3.2.3 is an ANN methodology).
        artifact_details = result.artifact_details
        if artifact_details is None:
            return False
        # 6 — artifact must NOT carry the exploratory override.
        metadata = dict(artifact_details.metadata or {})
        provenance = dict(metadata.get("training_provenance", {}) or {})
        if bool(provenance.get("exploratory_training_override")):
            return False
        return True

    @staticmethod
    def _eval_used_same_session(result: ModelingEvaluationResult | None) -> bool:
        """True when the last evaluation pulled its test samples from the training run.

        Both ``artifact_test_split`` and ``full_dataset`` scopes share the training data
        with the evaluated model, so they're "same-session" by Wolfe's standard (thesis
        p85: he re-collects on a separate test mesh). Only an explicit separate dataset
        designation would clear this flag — not implemented in this pass.
        """
        if result is None:
            return False
        scope = str(result.evaluation_scope_used or "").strip().lower()
        return scope != "separate_test_dataset"

    def _two_segment_modeling_worker(
        self,
        *,
        run_dirs: list[Path],
        model_keys: list[str],
        allow_lower_trust: bool,
        label_mode: str,
        include_orientation: bool,
        ann_sweep: bool,
        ann_hidden_layers: str,
        ann_epochs: int,
        test_fraction: float,
    ) -> None:
        try:
            hidden_layers = _parse_hidden_layers(ann_hidden_layers)
            result = run_two_segment_modeling(
                run_dirs=run_dirs,
                project_root=self.project_root,
                config=TwoSegmentModelingConfig(
                    allow_lower_trust=bool(allow_lower_trust),
                    model_keys=list(model_keys),
                    label_mode=str(label_mode),
                    include_orientation_if_available=bool(include_orientation),
                    test_fraction=float(test_fraction),
                    model_config={
                        "ann": {
                            "hidden_layers": hidden_layers,
                            "hidden_layer_options": [[32, 32], [64, 64], [128, 128]],
                            "sweep_enabled": bool(ann_sweep),
                            "epochs": int(ann_epochs),
                            "batch_size": 64,
                            "learning_rate": 0.001,
                            "patience": 20,
                            "seeds": [42],
                        }
                    },
                    output_root=str(self.project_root / "data" / "experiments"),
                    random_seed=42,
                ),
            )
        except ValueError as exc:
            requirements = (
                "Required: dual_segment dataset, accepted all-8 startup, distal_tip pose role, "
                "non-servo-only trusted run, successful commands, and valid_for_two_segment_model_training=true."
            )
            with self._lock:
                self.state.two_segment_active = False
                self.state.status_message = f"Two-segment modeling could not start: {exc} {requirements}"
            return
        with self._lock:
            self._last_two_segment_result = result
            self.state.two_segment_active = False
            self.state.two_segment_last_output_path = str(result.output_dir)
            self.state.status_message = self._two_segment_result_message(result)

    def _two_segment_model_keys(self) -> list[str]:
        keys: list[str] = []
        if self.state.two_segment_include_linear:
            keys.append("linear_baseline")
        if self.state.two_segment_include_ann:
            keys.append("ann")
        if self.state.two_segment_include_camarillo:
            keys.append("camarillo")
        if self.state.two_segment_include_mike:
            keys.append("mike_constant_curvature")
        return keys

    def _two_segment_trainability_pairs(self, run_paths: list[str], *, allow_lower_trust: bool) -> list[tuple[str, str]]:
        if not run_paths:
            return [
                ("Selection", "Select one or more two_segment_collect_pose_command_dataset runs."),
                ("Required", "dual_segment, all-8 startup, distal_tip robot-frame pose, trusted non-servo-only samples."),
            ]
        # Cache by (sorted paths tuple, per-path dir mtimes, allow_lower_trust, label_mode).
        # The validation reads summary.json, metadata.json, config_snapshot.yaml AND iterates
        # samples.jsonl line-by-line for each selected run — the single most expensive thing
        # the refresh tick can do. Dir-mtime keying invalidates when any of those files change.
        mtimes: list[float] = []
        for path in run_paths:
            try:
                mtimes.append(Path(path).stat().st_mtime)
            except OSError:
                mtimes.append(0.0)
        cache_key = (
            tuple(sorted(str(p) for p in run_paths)),
            tuple(mtimes),
            bool(allow_lower_trust),
            str(self.state.two_segment_label_mode),
            "trainability_v1",
        )
        if self._trainability_cache is not None and self._trainability_cache[0] == cache_key:
            return self._trainability_cache[1]
        try:
            summary = self.validate_two_segment_modeling_trainability(
                [Path(path) for path in run_paths],
                allow_lower_trust=bool(allow_lower_trust),
            )
        except Exception as exc:
            pairs: list[tuple[str, str]] = [("Trainability", f"Could not inspect selected runs: {exc}")]
            self._trainability_cache = (cache_key, pairs)
            return pairs
        pairs = [
            ("Runs", str(summary["runs_scanned"])),
            ("Samples", f"{summary['samples_accepted']} accepted / {summary['samples_rejected']} rejected"),
            ("Trainable", str(summary["trainable"])),
            ("Rejection Reasons", str(summary["rejection_reasons"] or {})),
            ("Orientation", "available" if summary["orientation_available"] else "XYZ only"),
            ("Intermediate Pose", str(summary["includes_intermediate_pose"])),
            ("Two-Coil Labels", str(summary.get("two_coil_xyz_available", False))),
            ("Label Mode", str(self.state.two_segment_label_mode)),
            ("Input Features", "8 tendon displacements (mm)"),
            ("Physics Models", "Mike/Camarillo require validated geometry, stiffness, sign, and frame config."),
        ]
        self._trainability_cache = (cache_key, pairs)
        return pairs

    @staticmethod
    def _two_segment_result_message(result: TwoSegmentModelingResult) -> str:
        completed = [item for item in result.model_results if item.status == "completed"]
        unavailable = [f"{item.model_key}: {item.reason}" for item in result.model_results if str(item.status).startswith("unavailable")]
        best = min(
            completed,
            key=lambda item: float(item.metrics.get("xyz_rmse_mm", float("inf"))),
        ) if completed else None
        pieces = [
            f"Two-segment modeling saved to {result.output_dir.name}.",
            f"Accepted {result.dataset.accepted_count}, rejected {result.dataset.rejected_count}.",
        ]
        if best is not None:
            pieces.append(f"Best XYZ RMSE: {best.model_key} {best.metrics.get('xyz_rmse_mm'):.4g} mm.")
        if unavailable:
            pieces.append("Unavailable: " + "; ".join(unavailable[:3]))
        return " ".join(pieces)

    @staticmethod
    def _resolve_dataset_summary(
        selected_path: str,
        datasets: list[ModelingDatasetSummary],
    ) -> ModelingDatasetSummary | None:
        for dataset in datasets:
            if str(dataset.path) == str(selected_path):
                return dataset
        return datasets[0] if datasets else None

    @staticmethod
    def _resolve_artifact_details(
        selected_path: str,
        artifacts: list[TrainedArtifactSummary],
    ) -> ArtifactDetails | None:
        for artifact in artifacts:
            if str(artifact.path) == str(selected_path):
                try:
                    return load_trained_artifact_details(artifact.path)
                except Exception:
                    return None
        if artifacts:
            try:
                return load_trained_artifact_details(artifacts[0].path)
            except Exception:
                return None
        return None


def _parse_hidden_layers(raw: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    except ValueError:
        return [128, 128]
    return values or [128, 128]
