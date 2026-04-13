"""Canonical experiment workspace controller for the operator GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any

import yaml

from continuum_robot.experiments.critical_experiments import (
    GridDefinitionConfig,
    RepeatabilityDatasetConfig,
    build_grid_accuracy_preview,
    build_repeatability_preview,
)
from continuum_robot.experiments.dataset_io import canonical_experiment_output_root
from continuum_robot.tracking.timing_benchmark import (
    compute_servo_sync_summary,
    compute_tracker_timing_summary,
    extract_servo_timing_records,
    extract_tracker_timing_records,
)
from continuum_robot.gui.experiment_parameters import apply_field_value, dump_payload, parse_field_value
from continuum_robot.gui.experiment_preflight import PreflightReport, RUN_BLOCKED, evaluate_preflight
from continuum_robot.gui.experiment_visualization import VisualizationModel, build_visualization_model


@dataclass(frozen=True)
class ExperimentOption:
    """GUI-facing definition for one visible experiment."""

    name: str
    title: str
    description: str
    category: str
    badges: list[str]
    default_config_path: str | None = None


@dataclass
class RunHistoryEntry:
    """Lightweight summary of one prior experiment run."""

    path: str
    experiment_name: str
    timestamp_utc: str
    status: str
    label: str
    metric_summary: str = ""


@dataclass
class ExperimentViewState:
    """UI-facing state for the generic experiment workspace."""

    experiment_options: list[ExperimentOption] = field(default_factory=list)
    selected_experiment: str = ""
    experiment_title: str = ""
    experiment_description: str = ""
    experiment_category: str = ""
    experiment_badges: list[str] = field(default_factory=list)
    config_text: str = ""
    config_error: str | None = None
    operator_notes: str = ""
    output_root: str = ""
    planned_output_dir: str = ""
    preflight_report: PreflightReport = field(default_factory=lambda: PreflightReport(overall_status=RUN_BLOCKED))
    run_checklist: list[tuple[str, str]] = field(default_factory=list)
    result_details: list[tuple[str, str]] = field(default_factory=list)
    run_active: bool = False
    progress_current: int = 0
    progress_total: int = 0
    status_message: str = "Select an experiment to open its validation workspace."
    last_error: str | None = None
    last_output_path: str | None = None
    loaded_run_path: str | None = None
    result_summary_lines: list[str] = field(default_factory=lambda: ["No run loaded."])
    history: list[RunHistoryEntry] = field(default_factory=list)
    color_mode: str = "target_point"
    show_axes: bool = True
    show_labels: bool = False
    show_centroids: bool = True
    show_truth: bool = True
    visualization_model: VisualizationModel = field(
        default_factory=lambda: VisualizationModel(summary_lines=["No run loaded."])
    )


class ExperimentController:
    """Owns experiment workspace state, execution, and history loading."""

    def __init__(
        self,
        experiment_loader,
        experiment_runner,
        registration_path: Path,
        servo_service,
        tracking_service,
    ) -> None:
        _ = experiment_loader
        self.experiment_runner = experiment_runner
        self.registration_path = Path(registration_path)
        self.servo_service = servo_service
        self.tracking_service = tracking_service
        self.settings = experiment_runner.settings
        self.project_root = Path(experiment_runner.project_root)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._history_dirty = True
        self._visualization_dirty = True
        self._current_bundle = None
        self._live_samples = []
        self._planned_output_dir_name = ""
        self._selected_payload: dict[str, Any] = {}
        self._field_drafts: dict[str, str] = {}
        self._field_errors: dict[str, str] = {}
        self._raw_config_error: str | None = None

        self._options_by_name = self._build_workspace_options()
        self.state = ExperimentViewState(
            experiment_options=list(self._options_by_name.values()),
            output_root=str(experiment_runner.output_dir),
        )
        if not self._options_by_name:
            raise RuntimeError("No workspace-visible experiments are registered.")

    def refresh(self) -> ExperimentViewState:
        with self._lock:
            selected_experiment = self.state.selected_experiment
            output_root = self._resolve_repo_path(self.state.output_root)
            current_bundle = self._current_bundle
            live_samples = list(self._live_samples)
            color_mode = self.state.color_mode
            show_centroids = self.state.show_centroids
            show_truth = self.state.show_truth
            config_payload = dict(self._selected_payload)
            config_error = self._current_config_error_locked()
            history_dirty = self._history_dirty
            visualization_dirty = self._visualization_dirty
            planned_output_dir = self._planned_output_dir(output_root, selected_experiment)
            cached_visualization_model = self.state.visualization_model

        if not selected_experiment:
            with self._lock:
                if history_dirty:
                    self.state.history = []
                    self._history_dirty = False
                self.state.experiment_title = "Select An Experiment"
                self.state.experiment_description = (
                    "Choose a structured validation or data-generation run from the dropdown above."
                )
                self.state.experiment_category = ""
                self.state.experiment_badges = []
                self.state.config_error = None
                self.state.preflight_report = PreflightReport(
                    overall_status=RUN_BLOCKED,
                    experiment_name="",
                    planned_output_dir="",
                )
                self.state.run_checklist = []
                self.state.result_details = [("Workspace", "Select an experiment to load its custom page.")]
                self.state.planned_output_dir = ""
                self.state.visualization_model = VisualizationModel(
                    summary_lines=["Select an experiment to view run outputs and analysis."]
                )
                self.state.result_summary_lines = list(self.state.visualization_model.summary_lines)
                self.state.config_text = ""
                return self.state

        history = None
        if history_dirty:
            history = self._scan_run_history(output_root, selected_experiment)

        tracking_snapshot = self.tracking_service.get_snapshot()
        neutral_setpoints = self.servo_service.load_neutral_setpoints()
        preflight = evaluate_preflight(
            experiment_name=selected_experiment,
            config_payload=config_payload,
            config_error=config_error,
            settings=self.settings,
            tracking_snapshot=tracking_snapshot,
            servo_connected=bool(self.servo_service.is_connected),
            neutral_setpoints=neutral_setpoints,
            registration_path=self.registration_path,
            output_root=output_root,
            planned_output_dir=planned_output_dir,
            project_root=self.project_root,
        )

        preview = None
        preview_metrics = None
        if selected_experiment == "aurora_grid_accuracy":
            try:
                preview = build_grid_accuracy_preview(
                    config=GridDefinitionConfig.from_dict(config_payload),
                    project_root=self.project_root,
                )
            except Exception:
                preview = None
            preview_metrics = preview.metrics if preview is not None else None
        elif selected_experiment == "tracker_timing_validation" and current_bundle is None and live_samples:
            preview_metrics = self._build_tracker_timing_preview_metrics(
                live_samples=live_samples,
                config_payload=config_payload,
            )

        if visualization_dirty:
            visualization_model = self._build_visualization_model(
                experiment_name=selected_experiment,
                bundle=current_bundle,
                live_samples=live_samples,
                color_mode=color_mode,
                show_centroids=show_centroids,
                show_truth=show_truth,
                config_payload=config_payload,
                preview=preview,
            )
        else:
            visualization_model = cached_visualization_model
        checklist = self._build_run_checklist(
            experiment_name=selected_experiment,
            config_payload=config_payload,
            tracking_snapshot=tracking_snapshot,
            planned_output_dir=planned_output_dir,
        )
        result_details = self._build_result_details(
            current_bundle,
            experiment_name=selected_experiment,
            preview_metrics=preview_metrics,
        )

        with self._lock:
            if history is not None:
                self.state.history = history
                self._history_dirty = False
            option = self._options_by_name[selected_experiment]
            self.state.experiment_title = option.title
            self.state.experiment_description = option.description
            self.state.experiment_category = option.category
            self.state.experiment_badges = list(option.badges)
            self.state.config_error = config_error
            self.state.preflight_report = preflight
            self.state.run_checklist = checklist
            self.state.planned_output_dir = str(planned_output_dir)
            self.state.visualization_model = visualization_model
            self.state.result_summary_lines = list(visualization_model.summary_lines)
            self.state.result_details = result_details
            self._visualization_dirty = False
            return self.state

    def refresh_prerequisites(self) -> ExperimentViewState:
        """Compatibility alias for the main window refresh loop."""
        return self.refresh()

    def select_experiment(self, experiment_name: str) -> None:
        if experiment_name not in self._options_by_name:
            raise KeyError(f"Unsupported experiment: {experiment_name}")
        payload = self._load_default_payload(experiment_name)
        with self._lock:
            self.state.selected_experiment = experiment_name
            self.state.loaded_run_path = None
            self.state.last_error = None
            self.state.last_output_path = None
            self.state.status_message = f"Loaded defaults for {self._options_by_name[experiment_name].title}."
            self._current_bundle = None
            self._live_samples = []
            self._history_dirty = True
            self._visualization_dirty = True
            self._apply_payload_locked(payload)
            self._reset_planned_output_dir_locked()

    def load_defaults(self) -> None:
        """Reload the default example config for the selected experiment."""
        if not self.state.selected_experiment:
            return
        self.select_experiment(self.state.selected_experiment)

    def clear_selection(self) -> None:
        """Return the experiment workspace to its empty selector state."""
        with self._lock:
            self.state.selected_experiment = ""
            self.state.loaded_run_path = None
            self.state.last_output_path = None
            self.state.last_error = None
            self.state.status_message = "Select an experiment to open its validation workspace."
            self._selected_payload = {}
            self._field_drafts.clear()
            self._field_errors.clear()
            self._raw_config_error = None
            self._current_bundle = None
            self._live_samples = []
            self._history_dirty = True
            self._planned_output_dir_name = ""
            self.state.config_text = ""

    def set_parameter_value(self, key: str, raw_value: str) -> None:
        with self._lock:
            self._field_drafts[key] = str(raw_value)
            try:
                parsed_value = parse_field_value(value_kind="yaml", raw_value=raw_value)
            except ValueError as exc:
                self._field_errors[key] = str(exc)
                self.state.config_error = self._current_config_error_locked()
                return
            self._field_errors.pop(key, None)
            self._raw_config_error = None
            self._selected_payload = apply_field_value(self._selected_payload, key=key, value=parsed_value)
            self._current_bundle = None
            self._live_samples = []
            self._visualization_dirty = True
            self._reset_planned_output_dir_locked()
            self._sync_config_text_locked()

    def set_config_value(self, key: str, value: Any) -> None:
        """Update one typed config value from a custom experiment page."""
        with self._lock:
            self._field_errors.pop(key, None)
            self._raw_config_error = None
            self._selected_payload = apply_field_value(self._selected_payload, key=key, value=value)
            self._current_bundle = None
            self._live_samples = []
            self._visualization_dirty = True
            self._reset_planned_output_dir_locked()
            self._sync_config_text_locked()

    def config_payload(self) -> dict[str, Any]:
        """Return a copy of the current selected experiment payload."""
        with self._lock:
            return dict(self._selected_payload)

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Return one nested config value using dot-path lookup."""
        with self._lock:
            current: Any = self._selected_payload
            for segment in [part for part in str(key).split(".") if part]:
                if not isinstance(current, dict) or segment not in current:
                    return default
                current = current[segment]
            return current

    def set_config_text(self, text: str) -> None:
        payload, error = self._parse_config_text(text)
        with self._lock:
            if error:
                self._raw_config_error = error
                self.state.config_text = str(text)
                self.state.config_error = error
                self.state.status_message = "Config text is invalid. Fix the YAML before running."
                return
            self._raw_config_error = None
            self._selected_payload = dict(payload)
            self._field_errors.clear()
            self._field_drafts.clear()
            self._current_bundle = None
            self._live_samples = []
            self._visualization_dirty = True
            self._reset_planned_output_dir_locked()
            self._sync_config_text_locked()

    def set_operator_notes(self, notes: str) -> None:
        with self._lock:
            self.state.operator_notes = str(notes)

    def set_output_root(self, raw_path: str) -> None:
        with self._lock:
            self.state.output_root = str(raw_path).strip() or str(self.experiment_runner.output_dir)
            self._history_dirty = True
            self._reset_planned_output_dir_locked()

    def set_color_mode(self, color_mode: str) -> None:
        with self._lock:
            self.state.color_mode = str(color_mode)
            self._visualization_dirty = True

    def set_show_axes(self, value: bool) -> None:
        with self._lock:
            self.state.show_axes = bool(value)

    def set_show_labels(self, value: bool) -> None:
        with self._lock:
            self.state.show_labels = bool(value)

    def set_show_centroids(self, value: bool) -> None:
        with self._lock:
            self.state.show_centroids = bool(value)
            self._visualization_dirty = True

    def set_show_truth(self, value: bool) -> None:
        with self._lock:
            self.state.show_truth = bool(value)
            self._visualization_dirty = True

    def run(self, *, confirm_overwrite: bool = False) -> None:
        state = self.refresh()
        if state.preflight_report.overall_status == RUN_BLOCKED:
            raise RuntimeError(state.preflight_report.summary)
        if state.preflight_report.requires_confirmation and not confirm_overwrite:
            raise RuntimeError("Run requires explicit overwrite confirmation.")
        if state.run_active:
            raise RuntimeError("Experiment is already running.")

        with self._lock:
            self._stop_event.clear()
            self.state.run_active = True
            self.state.progress_current = 0
            self.state.progress_total = 0
            self.state.status_message = f"Running {self.state.selected_experiment}."
            self.state.last_error = None
            self.state.last_output_path = None
            self._live_samples = []
            self._current_bundle = None
            experiment_name = self.state.selected_experiment
            config_payload = dict(self._selected_payload)
            operator_notes = self.state.operator_notes
            output_root = self._resolve_repo_path(self.state.output_root)
            output_dir_name = self._planned_output_dir_name

        def _worker() -> None:
            try:
                result = self.experiment_runner.run_experiment(
                    experiment_name,
                    config=config_payload,
                    operator_notes=operator_notes,
                    output_dir=output_root,
                    output_dir_name=output_dir_name,
                    progress_callback=self._on_progress,
                    stop_requested=self._stop_event.is_set,
                    sample_callback=self._on_sample,
                )
                bundle = self.experiment_runner.load_dataset(result.paths.output_dir)
                partial_saved = bool(not result.success and result.summary.sample_counts.get("total", 0) > 0)
                stopped = self._stop_event.is_set() and "stopped by operator" in result.message.lower()
                with self._lock:
                    self._current_bundle = bundle
                    self.state.last_output_path = str(result.paths.output_dir)
                    self.state.loaded_run_path = str(result.paths.output_dir)
                    if stopped and partial_saved:
                        self.state.status_message = (
                            f"Run stopped. Partial results were saved to {result.paths.output_dir.name}."
                        )
                        self.state.last_error = None
                    elif partial_saved:
                        self.state.status_message = (
                            f"Run completed with partial results in {result.paths.output_dir.name}. "
                            "Review the summary before using the dataset."
                        )
                        self.state.last_error = result.message
                    else:
                        self.state.status_message = result.message
                        self.state.last_error = None if result.success else result.message
                    self._history_dirty = True
                    self._visualization_dirty = True
                    self._reset_planned_output_dir_locked()
            except Exception as exc:
                with self._lock:
                    self.state.last_error = str(exc)
                    self.state.status_message = f"Experiment failed: {exc}"
            finally:
                with self._lock:
                    self.state.run_active = False

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            self.state.status_message = "Stop requested."

    def load_run(self, path: Path) -> None:
        bundle = self.experiment_runner.load_dataset(Path(path))
        experiment_name = bundle.metadata.experiment_name
        if experiment_name not in self._options_by_name:
            raise RuntimeError(
                f"Run {path} belongs to {experiment_name}, which is not exposed in the generic Experiment workspace."
            )
        with self._lock:
            self.state.selected_experiment = experiment_name
            self.state.operator_notes = str(bundle.metadata.operator_notes or "")
            self.state.loaded_run_path = str(bundle.paths.output_dir)
            self.state.last_output_path = str(bundle.paths.output_dir)
            self.state.status_message = f"Loaded prior run {bundle.paths.output_dir.name}."
            self.state.last_error = None
            self._current_bundle = bundle
            self._live_samples = []
            self._history_dirty = True
            self._visualization_dirty = True
            self._selected_payload = dict(bundle.metadata.config_used)
            self._field_errors.clear()
            self._field_drafts.clear()
            self._raw_config_error = None
            self._sync_config_text_locked()
            self._reset_planned_output_dir_locked()

    def shutdown(self, timeout_s: float = 2.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout_s)

    def _build_workspace_options(self) -> dict[str, ExperimentOption]:
        options: dict[str, ExperimentOption] = {}
        for descriptor in self.experiment_runner.available_experiments():
            if not bool(getattr(descriptor, "workspace_visible", True)):
                continue
            badges = [str(descriptor.category).title(), *list(descriptor.tags)]
            options[descriptor.name] = ExperimentOption(
                name=descriptor.name,
                title=descriptor.title,
                description=descriptor.description,
                category=descriptor.category,
                badges=badges,
                default_config_path=descriptor.default_config_path,
            )
        preferred_order = [
            "repeatability_dataset",
            "aurora_grid_accuracy",
            "tracker_timing_validation",
            "pretension_validation",
            "command_schedule_validation",
            "collect_pose_command_dataset",
            "replay_runner",
        ]
        ordered_names = [name for name in preferred_order if name in options]
        ordered_names.extend(sorted(name for name in options if name not in ordered_names))
        return {name: options[name] for name in ordered_names}

    def _default_config_path(self, experiment_name: str) -> Path:
        option = self._options_by_name[experiment_name]
        if option.default_config_path:
            return self._resolve_repo_path(option.default_config_path)
        return self.project_root / "config" / f"experiment_{experiment_name}.example.yaml"

    def _load_default_payload(self, experiment_name: str) -> dict[str, Any]:
        example_path = self._default_config_path(experiment_name)
        if not example_path.exists():
            return {}
        payload, error = self._parse_config_text(example_path.read_text(encoding="utf-8"))
        if error:
            raise RuntimeError(f"Default experiment config is invalid: {example_path}: {error}")
        return payload

    def _resolve_repo_path(self, raw_path: str | Path) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return self.project_root / path

    def _parse_config_text(self, text: str) -> tuple[dict[str, Any], str | None]:
        raw = str(text or "").strip()
        if not raw:
            return {}, None
        try:
            payload = yaml.safe_load(raw) or {}
        except Exception as exc:
            return {}, str(exc)
        if not isinstance(payload, dict):
            return {}, "Experiment config must be a mapping."
        return dict(payload), None

    def _apply_payload_locked(self, payload: dict[str, Any]) -> None:
        self._selected_payload = dict(payload)
        self._field_errors.clear()
        self._field_drafts.clear()
        self._raw_config_error = None
        self._sync_config_text_locked()

    def _sync_config_text_locked(self) -> None:
        self.state.config_text = dump_payload(self._selected_payload)
        self.state.config_error = self._current_config_error_locked()

    def _current_config_error_locked(self) -> str | None:
        if self._raw_config_error:
            return self._raw_config_error
        if not self._field_errors:
            return None
        rendered = "; ".join(
            f"{key}: {message}"
            for key, message in list(self._field_errors.items())[:3]
        )
        return f"Parameter edits are invalid. {rendered}"

    def _reset_planned_output_dir_locked(self) -> None:
        if not self.state.selected_experiment:
            self._planned_output_dir_name = ""
            return
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_name = self.state.selected_experiment.replace(" ", "_")
        self._planned_output_dir_name = f"{timestamp}_{safe_name}"

    def _planned_output_dir(self, output_root: Path, experiment_name: str) -> Path:
        experiment_root = canonical_experiment_output_root(output_root, experiment_name)
        if not self._planned_output_dir_name:
            return experiment_root
        return experiment_root / self._planned_output_dir_name

    def _on_progress(self, current: int, total: int, payload: dict[str, Any]) -> None:
        with self._lock:
            self.state.progress_current = int(current)
            self.state.progress_total = int(total)
            phase = payload.get("phase")
            phase_text = f" phase={phase}" if phase else ""
            self.state.status_message = f"Running {self.state.selected_experiment}:{phase_text} {current}/{total}"

    def _on_sample(self, sample) -> None:
        with self._lock:
            self._live_samples.append(sample)
            self._visualization_dirty = True

    def _scan_run_history(self, output_root: Path, experiment_name: str) -> list[RunHistoryEntry]:
        entries: list[RunHistoryEntry] = []
        experiment_root = canonical_experiment_output_root(output_root, experiment_name)
        if not experiment_root.exists():
            return entries
        run_dirs = sorted(
            {metadata_path.parent for metadata_path in experiment_root.rglob("metadata.json")},
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for run_dir in run_dirs:
            summary_path = run_dir / "summary.json"
            metadata_path = run_dir / "metadata.json"
            if not summary_path.exists():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            entry_experiment = str(metadata.get("experiment_name", "unknown"))
            if entry_experiment != experiment_name:
                continue
            timestamp = str(metadata.get("timestamp_utc", ""))
            status = str(summary.get("status", "unknown"))
            entries.append(
                RunHistoryEntry(
                    path=str(run_dir),
                    experiment_name=entry_experiment,
                    timestamp_utc=timestamp,
                    status=status,
                    label=self._format_history_label(
                        run_dir=run_dir,
                        experiment_name=entry_experiment,
                        timestamp_utc=timestamp,
                        summary=summary,
                    ),
                    metric_summary=self._history_metric_label(
                        experiment_name=entry_experiment,
                        metrics=summary.get("experiment_metrics", {})
                        if isinstance(summary.get("experiment_metrics"), dict)
                        else {},
                    ),
                )
            )
            if len(entries) >= 25:
                break
        return entries

    def _format_history_label(
        self,
        *,
        run_dir: Path,
        experiment_name: str,
        timestamp_utc: str,
        summary: dict[str, Any],
    ) -> str:
        stamp = timestamp_utc.replace("T", " ").replace("+00:00", "Z")
        status = str(summary.get("status", "unknown"))
        metrics = summary.get("experiment_metrics", {}) if isinstance(summary.get("experiment_metrics"), dict) else {}
        metric_label = self._history_metric_label(experiment_name=experiment_name, metrics=metrics)
        suffix = f" | {metric_label}" if metric_label else ""
        return f"{stamp} | {experiment_name} | {status}{suffix} | {run_dir.name}"

    @staticmethod
    def _history_metric_label(*, experiment_name: str, metrics: dict[str, Any]) -> str:
        if experiment_name == "repeatability_dataset":
            value = metrics.get("overall_repeatability_rms_mm")
            return f"repeatability={float(value):.3f} mm" if value is not None else ""
        if experiment_name == "aurora_grid_accuracy":
            value = metrics.get("overall_rms_error_mm")
            return f"grid_rms={float(value):.3f} mm" if value is not None else ""
        if experiment_name == "command_schedule_validation":
            value = metrics.get("point_count")
            return f"points={int(value)}" if value is not None else ""
        if experiment_name == "collect_pose_command_dataset":
            value = metrics.get("schedule_point_count")
            return f"schedule_points={int(value)}" if value is not None else ""
        if experiment_name == "replay_runner":
            source = metrics.get("source_experiment_name")
            count = metrics.get("source_sample_count")
            if source and count is not None:
                return f"source={source} | samples={int(count)}"
            if source:
                return f"source={source}"
            return ""
        if experiment_name == "pretension_validation":
            value = metrics.get("travel_used_ticks")
            if value is not None:
                return f"travel={int(value)} ticks"
            value = metrics.get("trigger_current_ma")
            return f"trigger={float(value):.1f} mA" if value is not None else ""
        if experiment_name == "tracker_timing_validation":
            value = metrics.get("unique_frame_rate_hz")
            if value is not None:
                return f"unique={float(value):.2f} Hz"
            value = metrics.get("mean_total_cycle_ms")
            return f"mean={float(value):.2f} ms" if value is not None else ""
        return ""

    def _build_visualization_model(
        self,
        *,
        experiment_name: str,
        bundle,
        live_samples,
        color_mode: str,
        show_centroids: bool,
        show_truth: bool,
        config_payload: dict[str, Any],
        preview=None,
    ) -> VisualizationModel:
        if bundle is not None:
            metrics = bundle.summary.experiment_metrics if isinstance(bundle.summary.experiment_metrics, dict) else {}
            return build_visualization_model(
                experiment_name=experiment_name,
                samples=bundle.samples,
                metrics=metrics,
                config_payload=config_payload or bundle.metadata.config_used,
                color_mode=color_mode,
                show_centroids=show_centroids,
                show_truth=show_truth,
            )
        if preview is not None and experiment_name == "aurora_grid_accuracy":
            return build_visualization_model(
                experiment_name=experiment_name,
                samples=preview.samples,
                metrics=preview.metrics,
                config_payload=config_payload,
                color_mode=color_mode,
                show_centroids=show_centroids,
                show_truth=show_truth,
            )
        if live_samples:
            return build_visualization_model(
                experiment_name=experiment_name,
                samples=live_samples,
                metrics={},
                config_payload=config_payload,
                color_mode=color_mode,
                show_centroids=show_centroids,
                show_truth=show_truth,
            )
        return VisualizationModel(summary_lines=["No run loaded."])

    def _build_tracker_timing_preview_metrics(
        self,
        *,
        live_samples,
        config_payload: dict[str, Any],
    ) -> dict[str, Any]:
        tracker_records = extract_tracker_timing_records(live_samples)
        servo_records = extract_servo_timing_records(live_samples)
        requested_tool_ids = [
            str(value).strip().upper()
            for value in (config_payload.get("requested_tool_ids") or ["0A", "0B"])
            if str(value).strip()
        ] or ["0A", "0B"]
        servo_sync = compute_servo_sync_summary(tracker_records, servo_records)
        servo_sync["enabled"] = bool(config_payload.get("enable_servo_logging", False))
        backend_identity = next(
            (str(record.get("backend_identity")) for record in tracker_records if record.get("backend_identity")),
            "",
        )
        return compute_tracker_timing_summary(
            tracker_records,
            requested_tool_ids=requested_tool_ids,
            backend_identity=backend_identity,
            configured_backend_name="",
            selected_backend_name="",
            run_duration_s=config_payload.get("run_duration_s"),
            run_label=str(config_payload.get("run_label", "") or ""),
            servo_sync_summary=servo_sync,
        )

    def _build_run_checklist(
        self,
        *,
        experiment_name: str,
        config_payload: dict[str, Any],
        tracking_snapshot,
        planned_output_dir: Path,
    ) -> list[tuple[str, str]]:
        backend = tracking_snapshot.selected_backend_name or tracking_snapshot.backend_identity or "unknown"
        option = self._options_by_name[experiment_name]
        return [
            ("Experiment", option.title),
            ("Category", option.category.title()),
            ("Tracker Backend", backend),
            ("Run Mode", self._mode_label(experiment_name, config_payload)),
            ("Output Path", str(planned_output_dir)),
            ("Config Summary", self._config_summary_label(experiment_name, config_payload)),
        ]

    def _build_result_details(
        self,
        bundle,
        *,
        experiment_name: str,
        preview_metrics: dict[str, Any] | None = None,
    ) -> list[tuple[str, str]]:
        if bundle is None:
            if preview_metrics and experiment_name == "aurora_grid_accuracy":
                return [
                    (
                        "Coverage",
                        f"{int(preview_metrics.get('point_count_captured', 0) or 0)} / "
                        f"{int(preview_metrics.get('point_count_total', 0) or 0)}",
                    ),
                    (
                        "Point Status",
                        f"{int(preview_metrics.get('point_count_complete', 0) or 0)} complete, "
                        f"{int(preview_metrics.get('point_count_partial', 0) or 0)} partial, "
                        f"{int(preview_metrics.get('point_count_not_started', 0) or 0)} not started",
                    ),
                    ("Solve Ready", "Yes" if preview_metrics.get("alignment_ready") else "Not Yet"),
                    ("Aligned Points", str(int(preview_metrics.get("point_count_aligned", 0) or 0))),
                    ("Raw Samples", str(int(preview_metrics.get("raw_sample_count", 0) or 0))),
                    ("Accepted Samples", str(int(preview_metrics.get("accepted_sample_count", 0) or 0))),
                    ("Rejected Samples", str(int(preview_metrics.get("rejected_sample_count", 0) or 0))),
                    (
                        "RMS Residual",
                        self._format_metric_value(preview_metrics.get("overall_rms_residual_mm")),
                    ),
                    ("Max Residual", self._format_metric_value(preview_metrics.get("max_residual_mm"))),
                    (
                        "Mean Spread",
                        self._format_metric_value(preview_metrics.get("mean_within_point_spread_mm")),
                    ),
                    (
                        "Max Spread",
                        self._format_metric_value(preview_metrics.get("max_within_point_spread_mm")),
                    ),
                    (
                        "Tip Calibration Used",
                        "Yes"
                        if preview_metrics.get("tip_calibration_used")
                        else ("Fallback Only" if preview_metrics.get("coil_origin_fallback_used") else "No"),
                    ),
                ]
            if preview_metrics and experiment_name == "tracker_timing_validation":
                return [
                    ("Backend", str(preview_metrics.get("backend_identity", "n/a") or "n/a")),
                    ("Analyzed Samples", str(int(preview_metrics.get("sample_count_analyzed", 0) or 0))),
                    ("Warmup Discarded", str(int(preview_metrics.get("warmup_discarded_count", 0) or 0))),
                    ("Effective Loop Hz", self._format_metric_value(preview_metrics.get("effective_loop_rate_hz"))),
                    ("Unique-Frame Hz", self._format_metric_value(preview_metrics.get("unique_frame_rate_hz"))),
                    ("Mean Total Time", self._format_metric_value(preview_metrics.get("mean_total_cycle_ms"))),
                    ("P95 Total Time", self._format_metric_value(preview_metrics.get("p95_total_cycle_ms"))),
                    (
                        "Duplicate Frames",
                        (
                            f"{int(preview_metrics.get('duplicate_frame_count', 0) or 0)} "
                            f"({100.0 * float(preview_metrics.get('duplicate_frame_ratio', 0.0) or 0.0):.1f}%)"
                        ),
                    ),
                    (
                        "Valid Requested Tools",
                        (
                            f"{100.0 * float(preview_metrics.get('valid_requested_tool_rate', 0.0) or 0.0):.1f}%"
                            if preview_metrics.get("valid_requested_tool_rate") is not None
                            else "n/a"
                        ),
                    ),
                    ("Backend Errors", str(int(preview_metrics.get("error_sample_count", 0) or 0))),
                ]
            if preview_metrics:
                return [
                    ("Preview Status", self._format_metric_value(preview_metrics.get("status"))),
                ]
            return [("Last Run", "No run loaded yet.")]
        bundle_experiment_name = str(bundle.metadata.experiment_name or experiment_name)
        pairs = [
            ("Status", bundle.summary.status),
            ("Run ID", bundle.metadata.run_id),
            ("Output Dir", str(bundle.paths.output_dir)),
            ("Metadata", str(bundle.paths.metadata_path)),
            ("Summary", str(bundle.paths.summary_path)),
            ("Samples", str(bundle.paths.samples_path)),
        ]
        if bundle.paths.config_snapshot_path is not None:
            pairs.append(("Config Snapshot", str(bundle.paths.config_snapshot_path)))
        if bundle_experiment_name == "aurora_grid_accuracy":
            metrics = bundle.summary.experiment_metrics if isinstance(bundle.summary.experiment_metrics, dict) else {}
            plot_path = bundle.paths.output_dir / "grid_accuracy_alignment.png"
            summary_text_path = bundle.paths.output_dir / "grid_accuracy_summary.txt"
            pairs.extend(
                [
                    (
                        "Coverage",
                        f"{int(metrics.get('point_count_captured', 0) or 0)} / "
                        f"{int(metrics.get('point_count_total', 0) or 0)}",
                    ),
                    (
                        "Point Status",
                        f"{int(metrics.get('point_count_complete', 0) or 0)} complete, "
                        f"{int(metrics.get('point_count_partial', 0) or 0)} partial, "
                        f"{int(metrics.get('point_count_not_started', 0) or 0)} not started",
                    ),
                    ("Aligned Points", str(int(metrics.get("point_count_aligned", 0) or 0))),
                    ("Raw Samples", str(int(metrics.get("raw_sample_count", 0) or 0))),
                    ("Accepted Samples", str(int(metrics.get("accepted_sample_count", 0) or 0))),
                    ("Rejected Samples", str(int(metrics.get("rejected_sample_count", 0) or 0))),
                    ("RMS Residual", self._format_metric_value(metrics.get("overall_rms_residual_mm"))),
                    ("Max Residual", self._format_metric_value(metrics.get("max_residual_mm"))),
                    ("Mean Spread", self._format_metric_value(metrics.get("mean_within_point_spread_mm"))),
                    (
                        "Tip Calibration Used",
                        "Yes"
                        if metrics.get("tip_calibration_used")
                        else ("Fallback Only" if metrics.get("coil_origin_fallback_used") else "No"),
                    ),
                    ("Alignment Plot", str(plot_path) if plot_path.exists() else "not written"),
                    ("Summary Note", str(summary_text_path) if summary_text_path.exists() else "not written"),
                ]
            )
            return pairs
        if bundle_experiment_name == "pretension_validation":
            metrics = bundle.summary.experiment_metrics if isinstance(bundle.summary.experiment_metrics, dict) else {}
            plot_path = bundle.paths.output_dir / "pretension_response.png"
            summary_text_path = bundle.paths.output_dir / "pretension_summary.txt"
            pairs.extend(
                [
                    ("Servo", str(metrics.get("servo_id", "n/a"))),
                    ("Accepted", "Yes" if metrics.get("accepted") else "No"),
                    ("Stop Reason", str(metrics.get("stop_reason", "n/a"))),
                    ("Final Position", self._format_metric_value(metrics.get("final_position_tick"))),
                    ("Travel Used (ticks)", self._format_metric_value(metrics.get("travel_used_ticks"))),
                    ("Travel Used (mm)", self._format_metric_value(metrics.get("travel_used_mm"))),
                    ("Baseline Current", self._format_metric_value(metrics.get("baseline_current_ma"))),
                    ("Effective Trigger", self._format_metric_value(metrics.get("effective_trigger_current_ma"))),
                    ("Trigger Current", self._format_metric_value(metrics.get("trigger_current_ma"))),
                    ("Max Displacement", self._format_metric_value(metrics.get("max_observed_displacement_mm"))),
                    ("Response Plot", str(plot_path) if plot_path.exists() else "not written"),
                    ("Summary Note", str(summary_text_path) if summary_text_path.exists() else "not written"),
                ]
            )
            return pairs
        if bundle_experiment_name == "tracker_timing_validation":
            metrics = bundle.summary.experiment_metrics if isinstance(bundle.summary.experiment_metrics, dict) else {}
            histogram_path = bundle.paths.output_dir / "aurora_timing_histogram.png"
            breakdown_path = bundle.paths.output_dir / "aurora_timing_breakdown.png"
            timeseries_path = bundle.paths.output_dir / "aurora_timing_timeseries.png"
            summary_text_path = bundle.paths.output_dir / "aurora_timing_summary.txt"
            sync_plot_path = bundle.paths.output_dir / "aurora_timing_sync_offsets.png"
            pairs.extend(
                [
                    ("Backend", str(metrics.get("backend_identity", "n/a") or "n/a")),
                    ("Tool IDs", ", ".join(metrics.get("requested_tool_ids", []) or [])),
                    ("Analyzed Samples", str(int(metrics.get("sample_count_analyzed", 0) or 0))),
                    ("Warmup Discarded", str(int(metrics.get("warmup_discarded_count", 0) or 0))),
                    ("Effective Loop Hz", self._format_metric_value(metrics.get("effective_loop_rate_hz"))),
                    ("Unique-Frame Hz", self._format_metric_value(metrics.get("unique_frame_rate_hz"))),
                    ("Mean Total Time", self._format_metric_value(metrics.get("mean_total_cycle_ms"))),
                    ("P95 Total Time", self._format_metric_value(metrics.get("p95_total_cycle_ms"))),
                    (
                        "Duplicate Frames",
                        (
                            f"{int(metrics.get('duplicate_frame_count', 0) or 0)} "
                            f"({100.0 * float(metrics.get('duplicate_frame_ratio', 0.0) or 0.0):.1f}%)"
                        ),
                    ),
                    (
                        "Under 25 ms",
                        (
                            f"{float(metrics.get('percent_under_25ms', 0.0) or 0.0):.1f}%"
                            if metrics.get("percent_under_25ms") is not None
                            else "n/a"
                        ),
                    ),
                    ("Histogram", str(histogram_path) if histogram_path.exists() else "not written"),
                    ("Breakdown Plot", str(breakdown_path) if breakdown_path.exists() else "not written"),
                    ("Timeseries Plot", str(timeseries_path) if timeseries_path.exists() else "not written"),
                    ("Summary Note", str(summary_text_path) if summary_text_path.exists() else "not written"),
                    (
                        "Sync Plot",
                        str(sync_plot_path)
                        if sync_plot_path.exists()
                        else ("not written" if metrics.get("servo_sync", {}).get("enabled") else "not requested"),
                    ),
                ]
            )
            return pairs
        scalar_metrics = [
            (key, value)
            for key, value in bundle.summary.experiment_metrics.items()
            if isinstance(value, (str, int, float, bool))
        ]
        for key, value in scalar_metrics[:6]:
            rendered = f"{float(value):.3f}" if isinstance(value, float) else str(value)
            pairs.append((self._label_from_metric_key(key), rendered))
        return pairs

    @staticmethod
    def _mode_label(experiment_name: str, config_payload: dict[str, Any]) -> str:
        if experiment_name == "replay_runner":
            return "offline"
        if experiment_name == "command_schedule_validation":
            return "software validation"
        if experiment_name == "tracker_timing_validation":
            return "backend diagnostic"
        if experiment_name == "pivot_calibration":
            return "offline" if config_payload.get("input_path") else ("dry-run" if bool(config_payload.get("dry_run", False)) else "live")
        return "dry-run" if bool(config_payload.get("dry_run", False)) else "live"

    @staticmethod
    def _config_summary_label(experiment_name: str, config_payload: dict[str, Any]) -> str:
        if experiment_name == "repeatability_dataset":
            schedule = RepeatabilityDatasetConfig.from_dict(config_payload).schedule
            try:
                preview = build_repeatability_preview(
                    RepeatabilityDatasetConfig.from_dict(config_payload),
                    tendon_count=None,
                )
                target_count = int(preview.summary.get("target_count", 0) or 0)
            except Exception:
                target_count = len(schedule.target_points_cm or [])
            return (
                f"{str(schedule.target_set).replace('_', ' ')}, "
                f"{target_count} targets, "
                f"{int(schedule.revisit_count)} revisits, "
                f"{int(schedule.samples_per_point)} samples/visit"
            )
        if experiment_name == "aurora_grid_accuracy":
            captured_points = config_payload.get("captured_points", []) or []
            captured_count = len(captured_points) if captured_points else int(config_payload.get("captured_point_count", 0) or 0)
            return (
                f"{config_payload.get('dimensions', [])} @ {config_payload.get('spacing_mm', 'n/a')} mm, "
                f"{captured_count} captured points, "
                f"{int(config_payload.get('samples_per_point', 0) or 0)} samples/point"
            )
        if experiment_name == "command_schedule_validation":
            schedule = config_payload.get("schedule", {}) or {}
            return (
                f"{schedule.get('kind', 'unknown')} schedule, "
                f"{int(schedule.get('dimensions', 0) or 0)} dimensions, "
                f"{int(schedule.get('repeats', 0) or 0)} repeat(s)"
            )
        if experiment_name == "collect_pose_command_dataset":
            if config_payload.get("command_points"):
                return f"{len(config_payload.get('command_points', []) or [])} explicit command points"
            schedule = config_payload.get("command_schedule", {}) or {}
            return (
                f"{schedule.get('kind', 'unknown')} schedule, "
                f"{int(schedule.get('dimensions', 0) or 0)} dimensions, "
                f"{int(config_payload.get('sample_count_per_point', 0) or 0)} samples/point"
            )
        if experiment_name == "replay_runner":
            return str(config_payload.get("dataset_path", "select an existing run"))
        if experiment_name == "pretension_validation":
            return (
                f"servo {config_payload.get('servo_id', 'n/a')}, "
                f"step {config_payload.get('step_ticks', 'live default')} ticks, "
                f"max travel {config_payload.get('max_travel_ticks', 'live default')} ticks, "
                f"tracker={'on' if bool(config_payload.get('include_tracker_displacement', True)) else 'off'}"
            )
        if experiment_name == "tracker_timing_validation":
            tool_ids = ",".join(str(value) for value in (config_payload.get("requested_tool_ids") or ["0A", "0B"]))
            sample_target = config_payload.get("sample_count_target")
            stop_mode = (
                f"{int(sample_target)} analyzed samples"
                if sample_target not in (None, "", 0, "0")
                else f"{float(config_payload.get('run_duration_s', 8.0) or 8.0):.1f} s"
            )
            return (
                f"tools {tool_ids}, "
                f"warmup {int(config_payload.get('warmup_samples', 10) or 0)}, "
                f"stop {stop_mode}, "
                f"servo sync={'on' if bool(config_payload.get('enable_servo_logging', False)) else 'off'}"
            )
        return "See experiment parameters."

    @staticmethod
    def _label_from_metric_key(key: str) -> str:
        return " ".join(segment.capitalize() for segment in str(key).split("_"))

    @staticmethod
    def _format_metric_value(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)
