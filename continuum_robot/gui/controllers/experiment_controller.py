"""Canonical experiment workspace controller for the operator GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any

import yaml

from continuum_robot.gui.experiment_parameters import (
    ExperimentParameterField,
    apply_field_value,
    build_parameter_fields,
    dump_payload,
    parse_field_value,
)
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
    parameter_fields: list[ExperimentParameterField] = field(default_factory=list)
    parameter_error_count: int = 0
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
    status_message: str = "Select an experiment and review the validation checks."
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
        default_experiment = (
            "repeatability_dataset"
            if "repeatability_dataset" in self._options_by_name
            else next(iter(self._options_by_name), "")
        )
        if not default_experiment:
            raise RuntimeError("No workspace-visible experiments are registered.")
        self.select_experiment(default_experiment)

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
            planned_output_dir = output_root / self._planned_output_dir_name

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

        visualization_model = self._build_visualization_model(
            experiment_name=selected_experiment,
            bundle=current_bundle,
            live_samples=live_samples,
            color_mode=color_mode,
            show_centroids=show_centroids,
            show_truth=show_truth,
            config_payload=config_payload,
        )
        checklist = self._build_run_checklist(
            experiment_name=selected_experiment,
            config_payload=config_payload,
            tracking_snapshot=tracking_snapshot,
            planned_output_dir=planned_output_dir,
        )
        result_details = self._build_result_details(current_bundle)

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
        self.select_experiment(self.state.selected_experiment)

    def set_parameter_value(self, key: str, raw_value: str) -> None:
        with self._lock:
            field = next((item for item in self.state.parameter_fields if item.key == key), None)
            if field is None:
                return
            self._field_drafts[key] = str(raw_value)
            try:
                parsed_value = parse_field_value(value_kind=field.value_kind, raw_value=raw_value)
            except ValueError as exc:
                self._field_errors[key] = str(exc)
                self.state.config_error = self._current_config_error_locked()
                self.state.parameter_fields = build_parameter_fields(
                    self._selected_payload,
                    drafts=self._field_drafts,
                    errors=self._field_errors,
                )
                self.state.parameter_error_count = len(self._field_errors)
                return
            self._field_errors.pop(key, None)
            self._raw_config_error = None
            self._selected_payload = apply_field_value(self._selected_payload, key=key, value=parsed_value)
            self._current_bundle = None
            self._live_samples = []
            self._visualization_dirty = True
            self._reset_planned_output_dir_locked()
            self._sync_parameter_state_locked()

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
            self._sync_parameter_state_locked()

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
            self._sync_parameter_state_locked()
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
        return options

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
        self._sync_parameter_state_locked()

    def _sync_parameter_state_locked(self) -> None:
        fields = build_parameter_fields(
            self._selected_payload,
            drafts=self._field_drafts,
            errors=self._field_errors,
        )
        self.state.parameter_fields = fields
        self.state.parameter_error_count = len([field for field in fields if field.error])
        self.state.config_text = dump_payload(self._selected_payload)
        self.state.config_error = self._current_config_error_locked()
        for field in fields:
            self._field_drafts.setdefault(field.key, field.raw_value)

    def _current_config_error_locked(self) -> str | None:
        if self._raw_config_error:
            return self._raw_config_error
        invalid_fields = [field for field in build_parameter_fields(self._selected_payload, drafts=self._field_drafts, errors=self._field_errors) if field.error]
        if not invalid_fields:
            return None
        rendered = "; ".join(f"{field.label}: {field.error}" for field in invalid_fields[:3])
        return f"Parameter edits are invalid. {rendered}"

    def _reset_planned_output_dir_locked(self) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_name = self.state.selected_experiment.replace(" ", "_")
        self._planned_output_dir_name = f"{timestamp}_{safe_name}"

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
        if not output_root.exists():
            return entries
        run_dirs = sorted(
            {metadata_path.parent for metadata_path in output_root.rglob("metadata.json")},
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
            value = metrics.get("validation_displacement_rms_mm")
            if value is not None:
                return f"disp_rms={float(value):.3f} mm"
            value = metrics.get("final_position_spread_ticks")
            return f"spread={int(value)} ticks" if value is not None else ""
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

    def _build_result_details(self, bundle) -> list[tuple[str, str]]:
        if bundle is None:
            return [("Last Run", "No run loaded yet.")]
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
        if experiment_name == "pivot_calibration":
            return "offline" if config_payload.get("input_path") else ("dry-run" if bool(config_payload.get("dry_run", False)) else "live")
        return "dry-run" if bool(config_payload.get("dry_run", False)) else "live"

    @staticmethod
    def _config_summary_label(experiment_name: str, config_payload: dict[str, Any]) -> str:
        if experiment_name == "repeatability_dataset":
            schedule = config_payload.get("schedule", {}) or {}
            targets = len(schedule.get("target_points_cm", []) or [])
            return (
                f"{targets} targets, "
                f"{int(schedule.get('revisit_count', 0) or 0)} revisits, "
                f"{int(schedule.get('samples_per_point', 0) or 0)} samples/point"
            )
        if experiment_name == "aurora_grid_accuracy":
            return (
                f"{config_payload.get('dimensions', [])} @ {config_payload.get('spacing_mm', 'n/a')} mm, "
                f"{int(config_payload.get('repetitions_per_point', 0) or 0)} repetitions"
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
                f"{int(config_payload.get('run_count', 0) or 0)} runs, "
                f"{config_payload.get('validation_direction', 'n/a')} {config_payload.get('validation_delta_ticks', 'n/a')} ticks"
            )
        return "See experiment parameters."

    @staticmethod
    def _label_from_metric_key(key: str) -> str:
        return " ".join(segment.capitalize() for segment in str(key).split("_"))
