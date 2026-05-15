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
    against the sub-millimeter surgical-accuracy target.
    """

    label: str
    model_key: str
    rmse_mm: float | None
    status: str
    reason: str = ""


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
        """
        if dataset_summary is None:
            return []
        try:
            return discover_past_evaluations(
                project_root=self.project_root,
                results_root=self.results_root,
                dataset_run_name=dataset_summary.run_name,
                limit=10,
            )
        except Exception:
            return []

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
        try:
            payload = json.loads((effective_path / "summary.json").read_text(encoding="utf-8"))
            metrics = payload.get("experiment_metrics", {}) if isinstance(payload, dict) else {}
            accepted = int(metrics.get("accepted_sample_count", 0) or 0)
        except Exception:
            return ""
        if accepted < MIN_EVAL_SAMPLES_WARN_THRESHOLD and accepted > 0:
            return (
                f"⚠ Evaluation dataset has only {accepted} accepted samples "
                f"(< {MIN_EVAL_SAMPLES_WARN_THRESHOLD}). RMSE on this few rows is noisy; "
                "collect more before citing the number."
            )
        return ""

    @staticmethod
    def _eval_is_thesis_grade(result: ModelingEvaluationResult | None) -> bool:
        """True when the last evaluation matches Wolfe's §3.2.3 cross-acquisition setup.

        Requires: ``evaluation_scope_used == "separate_test_dataset"`` (artifact and test
        come from different runs). Stronger than ``not same_session`` because a non-mesh
        test dataset is still legitimate cross-acquisition data.
        """
        if result is None:
            return False
        return str(result.evaluation_scope_used or "").strip().lower() == "separate_test_dataset"

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
        try:
            summary = self.validate_two_segment_modeling_trainability(
                [Path(path) for path in run_paths],
                allow_lower_trust=bool(allow_lower_trust),
            )
        except Exception as exc:
            return [("Trainability", f"Could not inspect selected runs: {exc}")]
        return [
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
