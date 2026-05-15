"""Canonical experiment workspace controller for the operator GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any

import yaml

from continuum_robot.experiments.critical_experiments import (
    GridDefinitionConfig,
    build_grid_accuracy_preview,
)
from continuum_robot.experiments.builtins import ServoTrackerSyncValidationConfig
from continuum_robot.experiments.dataset_io import (
    canonical_experiment_output_root,
    canonical_timestamp_token,
    canonical_timestamped_name,
    sanitize_output_name,
)
from continuum_robot.experiments.single_segment_repeatability import SingleSegmentRepeatabilityConfig
from continuum_robot.tracking.timing_benchmark import (
    compute_servo_tracker_sync_summary,
    compute_servo_sync_summary,
    extract_servo_command_records,
    compute_tracker_timing_summary,
    extract_servo_timing_records,
    extract_tracker_timing_records,
)
from continuum_robot.gui.experiment_parameters import apply_field_value, dump_payload, parse_field_value
from continuum_robot.gui.experiment_preflight import PreflightReport, RUN_BLOCKED, evaluate_preflight
from continuum_robot.gui.experiment_visualization import VisualizationModel, build_visualization_model


LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentOption:
    """GUI-facing definition for one visible experiment."""

    name: str
    title: str
    description: str
    category: str
    badges: list[str]
    default_config_path: str | None = None


MODE_EXPERIMENT_VISIBILITY: dict[str, set[str]] = {
    "one_servo": {
        "command_schedule_validation",
        "tracker_timing_validation",
        "servo_tracker_sync_validation",
        "aurora_grid_accuracy",
        "registration_validation",
        "pivot_validation",
    },
    "single_segment": {
        "pretension_validation",
        "single_segment_repeatability",
        "collect_pose_command_dataset",
        "penprobe_chasing_demo",
        "registration_validation",
        "pivot_validation",
        "aurora_grid_accuracy",
        "tracker_timing_validation",
        "servo_tracker_sync_validation",
        "command_schedule_validation",
        "replay_runner",
    },
    "dual_segment": {
        "two_segment_startup_validation",
        "two_segment_collect_pose_command_dataset",
        "registration_validation",
        "pivot_validation",
        "aurora_grid_accuracy",
        "tracker_timing_validation",
        "servo_tracker_sync_validation",
        "command_schedule_validation",
        "replay_runner",
    },
    "parallel_single": {
        "collect_pose_command_dataset",
        "command_schedule_validation",
        "tracker_timing_validation",
        "servo_tracker_sync_validation",
    },
}


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
    experiment_filter_summary: str = ""
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
    history_loading: bool = False
    color_mode: str = "target_point"
    show_axes: bool = True
    show_labels: bool = False
    show_centroids: bool = True
    show_truth: bool = True
    visualization_model: VisualizationModel = field(
        default_factory=lambda: VisualizationModel(summary_lines=["No run loaded."])
    )
    penprobe_chase_live_summary: str = ""


class ExperimentController:
    """Owns experiment workspace state, execution, and history loading."""

    HISTORY_LIMIT = 25
    HISTORY_SCAN_CANDIDATE_MULTIPLIER = 4
    PREREQUISITE_REFRESH_INTERVAL_S = 0.25
    PREREQUISITE_REFRESH_WHILE_PENPROBE_RUNNING_S = 0.35
    MANUAL_REFRESH_EXPERIMENTS = {
        "registration_validation",
        "pivot_validation",
        "single_segment_repeatability",
        "pretension_validation",
    }

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
        self._active_run_output_dir: Path | None = None
        self._selected_payload: dict[str, Any] = {}
        self._field_drafts: dict[str, str] = {}
        self._field_errors: dict[str, str] = {}
        self._raw_config_error: str | None = None
        self._history_cache: dict[tuple[str, str], list[RunHistoryEntry]] = {}
        self._history_loading_key: tuple[str, str] | None = None
        self._preflight_cache_key: tuple[Any, ...] | None = None
        self._preflight_cache_report: PreflightReport | None = None
        self._last_prerequisite_refresh_s = 0.0
        self._penprobe_gui_hz_smooth: float | None = None
        self._penprobe_gui_last_wall_s = 0.0

        self._options_by_name = self._build_workspace_options()
        visible_options = self._mode_filtered_workspace_options()
        self.state = ExperimentViewState(
            experiment_options=list(visible_options.values()),
            experiment_filter_summary=self._mode_filter_summary(visible_options),
            output_root=str(self._mock_aware_output_root(Path(experiment_runner.output_dir))),
        )
        if not self._options_by_name:
            raise RuntimeError("No workspace-visible experiments are registered.")
        self.experiment_runner.penprobe_live_gui_hz = self.peek_penprobe_gui_hz

    def peek_penprobe_gui_hz(self) -> float | None:
        with self._lock:
            return self._penprobe_gui_hz_smooth

    def note_penprobe_gui_pulse(self) -> None:
        with self._lock:
            if str(self.state.selected_experiment or "") != "penprobe_chasing_demo" or not bool(self.state.run_active):
                return
            wall = time.monotonic()
            prev = float(self._penprobe_gui_last_wall_s)
            self._penprobe_gui_last_wall_s = wall
            if prev <= 0.0:
                return
            interval_s = wall - prev
            instant_hz = 1.0 / max(1e-6, interval_s)
            beta = 0.35
            if self._penprobe_gui_hz_smooth is None:
                self._penprobe_gui_hz_smooth = float(instant_hz)
            else:
                smooth = float(self._penprobe_gui_hz_smooth)
                self._penprobe_gui_hz_smooth = float(beta * instant_hz + (1.0 - beta) * smooth)

    def refresh(self) -> ExperimentViewState:
        started = time.monotonic()
        visible_options = self._mode_filtered_workspace_options()
        filter_summary = self._mode_filter_summary(visible_options)
        with self._lock:
            self.state.experiment_options = list(visible_options.values())
            self.state.experiment_filter_summary = filter_summary
            if self.state.selected_experiment and self.state.selected_experiment not in visible_options:
                previous = self.state.selected_experiment
                self.state.selected_experiment = ""
                self.state.loaded_run_path = None
                self.state.last_output_path = None
                self.state.last_error = None
                self.state.status_message = (
                    f"{previous} is hidden for operating_mode={self.settings.robot.operating_mode()}. "
                    "Select a workflow listed for the current mode."
                )
                self._selected_payload = {}
                self._field_drafts.clear()
                self._field_errors.clear()
                self._raw_config_error = None
                self._current_bundle = None
                self._live_samples = []
                self._history_dirty = True
                self._planned_output_dir_name = ""
                self.state.config_text = ""
                self.state.history_loading = False
                self._invalidate_preflight_cache_locked()
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
            active_run_output_dir = self._active_run_output_dir
            cached_visualization_model = self.state.visualization_model
            cached_history = self._history_cache.get((str(output_root), selected_experiment), [])
            history_loading = self._history_loading_key == (str(output_root), selected_experiment)

        if not selected_experiment:
            with self._lock:
                if history_dirty:
                    self.state.history = []
                    self._history_dirty = False
                    self.state.history_loading = False
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
            history = list(cached_history)
            self._ensure_history_scan(output_root=output_root, experiment_name=selected_experiment)
            history_loading = True

        tracking_snapshot = self.tracking_service.get_snapshot()
        neutral_setpoints = self.servo_service.load_neutral_setpoints()
        servo_calibration_summary = self.servo_service.neutral_calibration.get_calibration_summary()
        preflight = self._get_preflight_report(
            experiment_name=selected_experiment,
            config_payload=config_payload,
            config_error=config_error,
            tracking_snapshot=tracking_snapshot,
            neutral_setpoints=neutral_setpoints,
            servo_calibration_summary=servo_calibration_summary,
            output_root=output_root,
            planned_output_dir=planned_output_dir,
            active_run_output_dir=active_run_output_dir,
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
        elif selected_experiment == "servo_tracker_sync_validation" and current_bundle is None and live_samples:
            preview_metrics = self._build_servo_tracker_sync_preview_metrics(
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
            self.state.history_loading = bool(history_loading)
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
        with self._lock:
            now = time.monotonic()
            if (
                self.state.selected_experiment == "penprobe_chasing_demo"
                and self.state.run_active
                and not self._history_dirty
                and not self._visualization_dirty
                and self._preflight_cache_report is not None
                and (now - self._last_prerequisite_refresh_s) < float(self.PREREQUISITE_REFRESH_WHILE_PENPROBE_RUNNING_S)
            ):
                return self.state
            if (
                self.state.selected_experiment
                and not self.state.run_active
                and not self._visualization_dirty
                and self._preflight_cache_report is not None
                and (now - self._last_prerequisite_refresh_s) < self.PREREQUISITE_REFRESH_INTERVAL_S
            ):
                return self.state
        return self.refresh()

    def refresh_policy_for(self, experiment_name: str | None = None) -> str:
        name = str(experiment_name or self.state.selected_experiment or "")
        if name in self.MANUAL_REFRESH_EXPERIMENTS:
            return "manual"
        return "live"

    def should_periodically_refresh_selected_experiment(self) -> bool:
        with self._lock:
            selected_experiment = str(self.state.selected_experiment or "")
            if not selected_experiment:
                return True
            if self.refresh_policy_for(selected_experiment) != "manual":
                return True
            if self.state.run_active or self.state.history_loading:
                return True
            if self._history_dirty or self._visualization_dirty or self._preflight_cache_report is None:
                return True
            return False

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
            self._invalidate_preflight_cache_locked()
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
            self.state.history_loading = False
            self._invalidate_preflight_cache_locked()

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
            self._invalidate_preflight_cache_locked()
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
            self._invalidate_preflight_cache_locked()
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
            self._invalidate_preflight_cache_locked()
            self._reset_planned_output_dir_locked()
            self._sync_config_text_locked()

    def set_operator_notes(self, notes: str) -> None:
        with self._lock:
            self.state.operator_notes = str(notes)

    def set_output_root(self, raw_path: str) -> None:
        with self._lock:
            self.state.output_root = str(raw_path).strip() or str(self.experiment_runner.output_dir)
            self._history_dirty = True
            self.state.history_loading = False
            self._invalidate_preflight_cache_locked()
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
            self._active_run_output_dir = self._planned_output_dir(output_root, experiment_name)
            self._invalidate_preflight_cache_locked()
            self._visualization_dirty = True
            self.state.penprobe_chase_live_summary = ""
            self._penprobe_gui_last_wall_s = 0.0
            self._penprobe_gui_hz_smooth = None

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
                    accepted_run_count = result.summary.experiment_metrics.get("accepted_run_count")
                    no_accepted_pretension = (
                        str(experiment_name) == "pretension_validation"
                        and accepted_run_count is not None
                        and int(accepted_run_count) == 0
                    )
                    if no_accepted_pretension:
                        self.state.status_message = (
                            f"No accepted pretension startup produced. Review {result.paths.output_dir.name}."
                        )
                        self.state.last_error = result.message
                    elif stopped and partial_saved:
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
                    self._invalidate_preflight_cache_locked()
                    self._reset_planned_output_dir_locked()
            except Exception as exc:
                LOG.exception(
                    "Experiment workspace run failed | experiment=%s | error=%s",
                    experiment_name,
                    exc,
                )
                with self._lock:
                    self.state.last_error = str(exc)
                    self.state.status_message = f"Experiment failed: {exc}"
            finally:
                with self._lock:
                    self.state.run_active = False
                    self.state.penprobe_chase_live_summary = ""
                    self._active_run_output_dir = None
                    self._invalidate_preflight_cache_locked()

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
            self._invalidate_preflight_cache_locked()
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
            "single_segment_repeatability",
            "penprobe_chasing_demo",
            "collect_pose_command_dataset",
            "registration_validation",
            "pivot_validation",
            "aurora_grid_accuracy",
            "tracker_timing_validation",
            "servo_tracker_sync_validation",
            "two_segment_startup_validation",
            "two_segment_collect_pose_command_dataset",
            "pretension_validation",
            "command_schedule_validation",
            "replay_runner",
        ]
        ordered_names = [name for name in preferred_order if name in options]
        ordered_names.extend(sorted(name for name in options if name not in ordered_names))
        return {name: options[name] for name in ordered_names}

    def _mode_filtered_workspace_options(self) -> dict[str, ExperimentOption]:
        mode = self.settings.robot.operating_mode()
        allowed = MODE_EXPERIMENT_VISIBILITY.get(mode)
        if not allowed:
            return dict(self._options_by_name)
        return {name: option for name, option in self._options_by_name.items() if name in allowed}

    def _mode_filter_summary(self, visible_options: dict[str, ExperimentOption]) -> str:
        mode = self.settings.robot.operating_mode()
        active_label = self.settings.robot.active_segment_label()
        if mode == "single_segment":
            return (
                f"Showing single_segment workflows for {active_label}: pretension, repeatability, "
                "collect-pose, penprobe chasing, and calibration/tracker validation."
            )
        if mode == "dual_segment":
            return (
                "Showing dual_segment foundation workflows only: all-8 startup validation, "
                "two-segment collect-pose dataset, and supporting validation/diagnostics."
            )
        if mode == "parallel_single":
            return (
                "Showing parallel_single demo workflows only: synchronized single-segment command playback "
                "across Spine 1 and Spine 2; pretension/control/chasing are hidden."
            )
        if mode == "one_servo":
            return "Showing one_servo-safe diagnostics/validation only; multi-servo experiments are hidden."
        return f"Showing {len(visible_options)} workflow(s) for operating_mode={mode}."

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
            resolved = path
        else:
            resolved = self.project_root / path
        return self._mock_aware_output_root(resolved)

    def _mock_aware_output_root(self, path: Path) -> Path:
        if not bool(getattr(self.settings.runtime, "mock_mode", False)):
            return Path(path)
        candidate = Path(path)
        try:
            relative = candidate.resolve().relative_to((self.project_root / "data" / "experiments").resolve())
        except ValueError:
            if candidate.name == "experiments" and candidate.parent.name == "data":
                return candidate.parent / "mock_experiments"
            return candidate
        return self.project_root / "data" / "mock_experiments" / relative

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
        experiment_name = self.state.selected_experiment
        try:
            output_root = self._resolve_repo_path(self.state.output_root)
            experiment_root = canonical_experiment_output_root(output_root, experiment_name)
            experiment_root.mkdir(parents=True, exist_ok=True)
            # Collision-safe: appends _NN suffix if a folder with the same timestamp already exists.
            self._planned_output_dir_name = canonical_timestamped_name(experiment_root, experiment_name)
        except Exception:
            timestamp = canonical_timestamp_token()
            safe_name = sanitize_output_name(experiment_name, default="experiment")
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
            detail_parts = [f"phase={phase}"] if phase else []
            if payload.get("current_target") is not None:
                detail_parts.append(f"target=T{int(payload.get('current_target')):02d}")
            if payload.get("source_target") is not None:
                detail_parts.append(f"source=T{int(payload.get('source_target')):02d}")
            if payload.get("revisit_index") is not None:
                detail_parts.append(f"revisit={int(payload.get('revisit_index')) + 1}")
            if payload.get("run_index") is not None:
                detail_parts.append(f"run={int(payload.get('run_index')) + 1}")
            if payload.get("quality_score_0_100") is not None:
                detail_parts.append(f"quality={float(payload.get('quality_score_0_100')):.1f}/100")
            detail_text = " " + " ".join(detail_parts) if detail_parts else ""
            self.state.status_message = f"Running {self.state.selected_experiment}:{detail_text} {current}/{total}"

    def _on_sample(self, sample) -> None:
        with self._lock:
            self._live_samples.append(sample)
            selected = str(self.state.selected_experiment or "")
            chasing = selected == "penprobe_chasing_demo" and bool(self.state.run_active)
            if chasing:
                extra = getattr(sample, "extra", None) if sample is not None else None
                if isinstance(extra, dict) and extra.get("chase_live_status_line"):
                    self.state.penprobe_chase_live_summary = str(extra.get("chase_live_status_line"))
                iteration = getattr(sample, "sample_index", None)
                dirty_every = 25
                if isinstance(iteration, int) and int(iteration) > 0 and int(iteration) % int(dirty_every) == 0:
                    self._visualization_dirty = True
            else:
                self._visualization_dirty = True

    def _scan_run_history(self, output_root: Path, experiment_name: str) -> list[RunHistoryEntry]:
        entries: list[RunHistoryEntry] = []
        experiment_root = canonical_experiment_output_root(output_root, experiment_name)
        if not experiment_root.exists():
            return entries
        candidate_limit = max(self.HISTORY_LIMIT, self.HISTORY_LIMIT * self.HISTORY_SCAN_CANDIDATE_MULTIPLIER)
        run_dirs: list[Path] = []
        try:
            with os.scandir(experiment_root) as iterator:
                run_dirs = [
                    Path(entry.path)
                    for entry in iterator
                    if entry.is_dir()
                ]
        except Exception:
            LOG.exception("Could not scan experiment history root %s", experiment_root)
            return entries
        # Canonical run folders are timestamp-prefixed, so lexical ordering gives
        # a cheap and stable recency approximation without stat()-ing every run dir.
        run_dirs.sort(key=lambda item: item.name, reverse=True)
        run_dirs = run_dirs[:candidate_limit]
        for run_dir in run_dirs:
            summary_path = run_dir / "summary.json"
            metadata_path = run_dir / "metadata.json"
            if not summary_path.exists() or not metadata_path.exists():
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
            if len(entries) >= self.HISTORY_LIMIT:
                break
        return entries

    def _ensure_history_scan(self, *, output_root: Path, experiment_name: str) -> None:
        key = (str(output_root), str(experiment_name))
        with self._lock:
            if self._history_loading_key == key:
                return
            self._history_loading_key = key
        LOG.debug("Scheduling experiment history scan for %s", key)

        def _worker() -> None:
            started = time.monotonic()
            try:
                entries = self._scan_run_history(output_root, experiment_name)
            except Exception:
                LOG.exception("Experiment history scan failed for %s", key)
                entries = []
            duration_ms = (time.monotonic() - started) * 1000.0
            with self._lock:
                self._history_cache[key] = entries
                if self.state.selected_experiment == experiment_name and str(self._resolve_repo_path(self.state.output_root)) == str(output_root):
                    self.state.history = list(entries)
                    self.state.history_loading = False
                    self._history_dirty = False
                if self._history_loading_key == key:
                    self._history_loading_key = None
            LOG.debug(
                "Experiment history scan finished for %s in %.1f ms with %d entries",
                key,
                duration_ms,
                len(entries),
            )

        threading.Thread(target=_worker, daemon=True).start()

    def _get_preflight_report(
        self,
        *,
        experiment_name: str,
        config_payload: dict[str, Any],
        config_error: str | None,
        tracking_snapshot,
        neutral_setpoints: dict[int, int],
        servo_calibration_summary,
        output_root: Path,
        planned_output_dir: Path,
        active_run_output_dir: Path | None,
    ) -> PreflightReport:
        key = self._build_preflight_cache_key(
            experiment_name=experiment_name,
            config_payload=config_payload,
            config_error=config_error,
            tracking_snapshot=tracking_snapshot,
            neutral_setpoints=neutral_setpoints,
            servo_calibration_summary=servo_calibration_summary,
            output_root=output_root,
            planned_output_dir=planned_output_dir,
            active_run_output_dir=active_run_output_dir,
        )
        with self._lock:
            if key == self._preflight_cache_key and self._preflight_cache_report is not None:
                return self._preflight_cache_report
        started = time.monotonic()
        report = evaluate_preflight(
            experiment_name=experiment_name,
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
            servo_calibration_summary=servo_calibration_summary,
            active_run_output_dir=active_run_output_dir,
        )
        duration_ms = (time.monotonic() - started) * 1000.0
        with self._lock:
            self._preflight_cache_key = key
            self._preflight_cache_report = report
            self._last_prerequisite_refresh_s = time.monotonic()
        LOG.debug("Experiment preflight refreshed for %s in %.1f ms", experiment_name, duration_ms)
        return report

    def _build_preflight_cache_key(
        self,
        *,
        experiment_name: str,
        config_payload: dict[str, Any],
        config_error: str | None,
        tracking_snapshot,
        neutral_setpoints: dict[int, int],
        servo_calibration_summary,
        output_root: Path,
        planned_output_dir: Path,
        active_run_output_dir: Path | None = None,
    ) -> tuple[Any, ...]:
        tools_signature = tuple(
            (
                str(tool_id),
                bool(getattr(tool, "present", False)),
                getattr(tool, "valid", None),
                str(getattr(tool, "tracking_state", "")),
            )
            for tool_id, tool in sorted(getattr(tracking_snapshot, "tools", {}).items())
        )
        servo_summary_signature = tuple(
            (
                int(servo_id),
                str(entry.pretension_result_status or ""),
                entry.neutral_setpoint,
                entry.pretension_final_position_tick,
                str(entry.pretension_completed_at_utc or ""),
            )
            for servo_id, entry in sorted(getattr(servo_calibration_summary, "servo_entries", {}).items())
        )
        baseline_signature = None
        if experiment_name == "single_segment_repeatability":
            baseline_path = str(config_payload.get("baseline_run_path", "") or "").strip()
            baseline_signature = self._path_signature(
                self._resolve_repo_path(baseline_path) if baseline_path else None
            )
        return (
            experiment_name,
            json.dumps(config_payload, sort_keys=True, default=str),
            str(config_error or ""),
            str(output_root),
            str(planned_output_dir),
            str(active_run_output_dir) if active_run_output_dir is not None else "",
            bool(self.servo_service.is_connected),
            tuple(sorted((int(key), int(value)) for key, value in neutral_setpoints.items())),
            str(getattr(tracking_snapshot, "canonical_state", "")),
            str(getattr(tracking_snapshot, "backend_identity", "")),
            str(getattr(tracking_snapshot, "selected_backend_name", "")),
            bool(getattr(tracking_snapshot, "tracker_data_stale", False)),
            round(float(getattr(tracking_snapshot, "tracker_data_age_s", 0.0) or 0.0), 1),
            str(getattr(tracking_snapshot, "registration_state", "")),
            str(getattr(tracking_snapshot, "runtime_tip_calibration_state", "")),
            bool(getattr(tracking_snapshot, "runtime_tip_identity_fallback", False)),
            str(getattr(tracking_snapshot, "tip_pose_status", "")),
            tools_signature,
            self._path_signature(self.registration_path),
            self._path_signature(self._configured_pivot_tip_path()),
            str(getattr(servo_calibration_summary, "status", "")),
            str(getattr(servo_calibration_summary, "updated_at_utc", "")),
            bool(getattr(servo_calibration_summary, "compatible", False)),
            str(getattr(servo_calibration_summary, "message", "")),
            servo_summary_signature,
            baseline_signature,
        )

    def _configured_pivot_tip_path(self) -> Path | None:
        raw_path = getattr(self.settings.registration, "penprobe_file", None)
        if not raw_path:
            return None
        return self._resolve_repo_path(raw_path)

    @staticmethod
    def _path_signature(path: Path | None) -> tuple[str, bool, int | None, int | None] | None:
        if path is None:
            return None
        candidate = Path(path)
        if not candidate.exists():
            return (str(candidate), False, None, None)
        stat = candidate.stat()
        return (str(candidate), True, int(stat.st_mtime_ns), int(stat.st_size))

    def _invalidate_preflight_cache_locked(self) -> None:
        self._preflight_cache_key = None
        self._preflight_cache_report = None

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
        if experiment_name == "single_segment_repeatability":
            value = metrics.get("overall_repeatability_rms_mm")
            if value is not None:
                return f"repeatability={float(value):.3f} mm"
            return f"repeat_samples={int(metrics.get('valid_repeat_sample_count', 0) or 0)}"
        if experiment_name == "aurora_grid_accuracy":
            value = metrics.get("overall_rms_error_mm")
            return f"grid_rms={float(value):.3f} mm" if value is not None else ""
        if experiment_name == "command_schedule_validation":
            value = metrics.get("point_count")
            return f"points={int(value)}" if value is not None else ""
        if experiment_name == "collect_pose_command_dataset":
            value = metrics.get("accepted_sample_count")
            if value is not None:
                return f"accepted={int(value)}"
            mode = metrics.get("dataset_mode")
            return str(mode).replace("_", " ") if mode else ""
        if experiment_name == "registration_validation":
            summary = metrics.get("translation_delta_to_consensus_summary_mm", {}) or {}
            value = summary.get("mean")
            return f"mean_dT={float(value):.3f} mm" if value is not None else f"valid={int(metrics.get('valid_run_count', 0) or 0)}"
        if experiment_name == "pivot_validation":
            summary = metrics.get("distance_to_consensus_summary_mm", {}) or {}
            value = summary.get("mean")
            return f"mean_dC={float(value):.3f} mm" if value is not None else f"valid={int(metrics.get('valid_run_count', 0) or 0)}"
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
        if experiment_name == "two_segment_startup_validation":
            completion = dict(metrics.get("stage_completion", {}) or {})
            completed = sum(1 for value in completion.values() if bool(value))
            return f"stages={completed}/5"
        if experiment_name == "two_segment_collect_pose_command_dataset":
            accepted = metrics.get("accepted_sample_count")
            valid = metrics.get("valid_for_two_segment_model_training")
            return f"accepted={int(accepted or 0)} | two_seg_train={valid}"
        if experiment_name == "penprobe_chasing_demo":
            value = metrics.get("final_error_norm_mm")
            if value is not None:
                return f"final_error={float(value):.2f} mm"
            reason = metrics.get("stop_reason")
            return str(reason) if reason else ""
        if experiment_name == "tracker_timing_validation":
            value = metrics.get("unique_frame_rate_hz")
            if value is not None:
                return f"unique={float(value):.2f} Hz"
            value = metrics.get("mean_total_cycle_ms")
            return f"mean={float(value):.2f} ms" if value is not None else ""
        if experiment_name == "servo_tracker_sync_validation":
            sync = metrics.get("servo_tracker_sync", {}) or {}
            value = sync.get("tracker_to_servo_telemetry_p95_offset_ms")
            if value is not None:
                return f"p95_offset={float(value):.2f} ms"
            value = metrics.get("unique_frame_rate_hz")
            return f"unique={float(value):.2f} Hz" if value is not None else ""
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

    def _build_servo_tracker_sync_preview_metrics(
        self,
        *,
        live_samples,
        config_payload: dict[str, Any],
    ) -> dict[str, Any]:
        tracker_records = extract_tracker_timing_records(live_samples)
        servo_records = extract_servo_timing_records(live_samples)
        command_records = extract_servo_command_records(live_samples)
        config = ServoTrackerSyncValidationConfig.from_dict(config_payload)
        sync_summary = compute_servo_tracker_sync_summary(
            tracker_records,
            servo_records,
            command_records,
        )
        backend_identity = next(
            (str(record.get("backend_identity")) for record in tracker_records if record.get("backend_identity")),
            "",
        )
        tracker_metrics = compute_tracker_timing_summary(
            tracker_records,
            requested_tool_ids=config.requested_tool_ids,
            backend_identity=backend_identity,
            configured_backend_name="",
            selected_backend_name="",
            run_duration_s=config.run_duration_s,
            run_label=str(config.run_label or ""),
        )
        tracker_metrics.update(
            {
                "selected_servo_ids": [int(servo_id) for servo_id in config.servo_ids],
                "motion_protocol": str(config.motion_mode),
                "servo_telemetry_sample_count": len(servo_records),
                "servo_command_sample_count": len(command_records),
                "include_robot_frame_tip_pose": bool(config.include_robot_frame_tip_pose),
                "servo_tracker_sync": sync_summary,
            }
        )
        return tracker_metrics

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
            if preview_metrics and experiment_name == "servo_tracker_sync_validation":
                sync = dict(preview_metrics.get("servo_tracker_sync", {}) or {})
                return [
                    ("Backend", str(preview_metrics.get("backend_identity", "n/a") or "n/a")),
                    ("Servo IDs", ", ".join(str(value) for value in (preview_metrics.get("selected_servo_ids") or []))),
                    ("Analyzed Tracker Samples", str(int(preview_metrics.get("sample_count_analyzed", 0) or 0))),
                    ("Servo Telemetry Samples", str(int(preview_metrics.get("servo_telemetry_sample_count", 0) or 0))),
                    ("Servo Command Samples", str(int(preview_metrics.get("servo_command_sample_count", 0) or 0))),
                    ("Effective Loop Hz", self._format_metric_value(preview_metrics.get("effective_loop_rate_hz"))),
                    ("Unique-Frame Hz", self._format_metric_value(preview_metrics.get("unique_frame_rate_hz"))),
                    (
                        "Duplicate Frames",
                        (
                            f"{int(preview_metrics.get('duplicate_frame_count', 0) or 0)} "
                            f"({100.0 * float(preview_metrics.get('duplicate_frame_ratio', 0.0) or 0.0):.1f}%)"
                        ),
                    ),
                    (
                        "Tracker->Servo Telemetry Mean",
                        self._format_metric_value(sync.get("tracker_to_servo_telemetry_mean_offset_ms")),
                    ),
                    (
                        "Tracker->Servo Telemetry P95",
                        self._format_metric_value(sync.get("tracker_to_servo_telemetry_p95_offset_ms")),
                    ),
                    (
                        "Tracker->Servo <= 10 ms",
                        (
                            f"{100.0 * float(sync.get('tracker_to_servo_telemetry_within_10ms_rate', 0.0) or 0.0):.1f}%"
                            if sync.get("tracker_to_servo_telemetry_within_10ms_rate") is not None
                            else "n/a"
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
                ]
            if preview_metrics and experiment_name in {"registration_validation", "pivot_validation"}:
                return [
                    ("Selected Runs", str(int(preview_metrics.get("selected_run_count", 0) or 0))),
                    ("Valid Runs", str(int(preview_metrics.get("valid_run_count", 0) or 0))),
                    ("Skipped Runs", str(int(preview_metrics.get("invalid_run_count", 0) or 0))),
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
        if bundle_experiment_name == "single_segment_repeatability":
            from continuum_robot.experiments.single_segment_repeatability_outputs import (
                build_single_segment_repeatability_summary_pairs,
            )

            metrics = bundle.summary.experiment_metrics if isinstance(bundle.summary.experiment_metrics, dict) else {}
            report_clusters_path = bundle.paths.output_dir / "repeatability_clusters_report.png"
            report_error_path = bundle.paths.output_dir / "repeatability_error_by_target_report.png"
            report_group_path = bundle.paths.output_dir / "repeatability_group_summary_report.png"
            clusters_path = bundle.paths.output_dir / "repeatability_clusters.png"
            rmse_path = bundle.paths.output_dir / "repeatability_rmse_summary.png"
            path_path = bundle.paths.output_dir / "repeatability_path_dependence.png"
            summary_text_path = bundle.paths.output_dir / "repeatability_summary.txt"
            pairs.extend(build_single_segment_repeatability_summary_pairs(metrics=metrics))
            pairs.extend(
                [
                    ("Clusters Report", str(report_clusters_path) if report_clusters_path.exists() else "not written"),
                    ("Error Report", str(report_error_path) if report_error_path.exists() else "not written"),
                    ("Group Report", str(report_group_path) if report_group_path.exists() else "not written"),
                    ("Clusters Plot", str(clusters_path) if clusters_path.exists() else "not written"),
                    ("RMSE Plot", str(rmse_path) if rmse_path.exists() else "not written"),
                    ("Path Plot", str(path_path) if path_path.exists() else "not written"),
                    ("Summary Note", str(summary_text_path) if summary_text_path.exists() else "not written"),
                ]
            )
            return pairs
        if bundle_experiment_name == "aurora_grid_accuracy":
            metrics = bundle.summary.experiment_metrics if isinstance(bundle.summary.experiment_metrics, dict) else {}
            report_path = bundle.paths.output_dir / "aurora_grid_alignment_report.png"
            dashboard_path = bundle.paths.output_dir / "aurora_grid_accuracy_dashboard.png"
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
                    ("Alignment Report", str(report_path) if report_path.exists() else "not written"),
                    ("Dashboard Plot", str(dashboard_path) if dashboard_path.exists() else "not written"),
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
        if bundle_experiment_name == "two_segment_startup_validation":
            metrics = bundle.summary.experiment_metrics if isinstance(bundle.summary.experiment_metrics, dict) else {}
            completion = dict(metrics.get("stage_completion", {}) or {})
            completed = sum(1 for value in completion.values() if bool(value))
            pairs.extend(
                [
                    ("Startup Type", str(metrics.get("startup_type", "manual_two_segment_startup"))),
                    ("Stages Complete", f"{completed} / 5"),
                    ("Captured This Run", ", ".join(str(value) for value in list(metrics.get("captured_this_run", []) or [])) or "n/a"),
                    ("Final Accepted", "Yes" if metrics.get("final_accepted") else "No"),
                    ("Startup Artifact", str(metrics.get("final_startup_artifact_path") or "not written")),
                    ("Workflow State", str(metrics.get("workflow_state_path") or "n/a")),
                    ("Summary Note", str(bundle.paths.output_dir / "two_segment_startup_summary.txt")),
                    ("Positions Report", str(bundle.paths.output_dir / "two_segment_startup_positions_report.png")),
                    ("Load Proxy Report", str(bundle.paths.output_dir / "two_segment_startup_load_proxy_report.png")),
                    ("Stage Drift Report", str(bundle.paths.output_dir / "two_segment_startup_stage_drift_report.png")),
                ]
            )
            return pairs
        if bundle_experiment_name == "two_segment_collect_pose_command_dataset":
            metrics = bundle.summary.experiment_metrics if isinstance(bundle.summary.experiment_metrics, dict) else {}
            pose = dict(metrics.get("pose_label_summary", {}) or {})
            missing_required = ", ".join(str(value) for value in list(pose.get("missing_required_roles", []) or [])) or "none"
            available_roles = ", ".join(str(value) for value in list(pose.get("available_roles", []) or [])) or "none"
            pairs.extend(
                [
                    ("Dataset Type", str(metrics.get("dataset_type", "two_segment_collect_pose_command_dataset"))),
                    ("Schedule", str(metrics.get("schedule_type", "n/a"))),
                    ("Accepted / Rejected", f"{metrics.get('accepted_sample_count', 0)} / {metrics.get('rejected_sample_count', 0)}"),
                    ("Trust Mode", str(metrics.get("run_trust_mode", "unknown"))),
                    ("Two-Segment Model Training Valid", str(metrics.get("valid_for_two_segment_model_training"))),
                    ("Generic Model Training Valid", str(metrics.get("valid_for_model_training"))),
                    ("Pose Samples", str(pose.get("pose_observation_sample_count", 0))),
                    ("Available Pose Roles", available_roles),
                    ("Missing Required Roles", missing_required),
                    ("Distal Only", str(metrics.get("distal_only"))),
                    ("Includes Intermediate Pose", str(metrics.get("includes_intermediate_pose"))),
                    ("Summary Note", str(bundle.paths.output_dir / "two_segment_dataset_summary.txt")),
                    ("Command Report", str(bundle.paths.output_dir / "two_segment_command_coverage_report.png")),
                    ("Servo Report", str(bundle.paths.output_dir / "two_segment_servo_position_coverage_report.png")),
                    ("Pose Report", str(bundle.paths.output_dir / "two_segment_pose_coverage_report.png")),
                ]
            )
            return pairs
        if bundle_experiment_name == "collect_pose_command_dataset":
            from continuum_robot.experiments.modeling_dataset_outputs import build_modeling_dataset_summary_pairs

            metrics = bundle.summary.experiment_metrics if isinstance(bundle.summary.experiment_metrics, dict) else {}
            workspace_plot_path = bundle.paths.output_dir / "modeling_workspace_coverage.png"
            command_plot_path = bundle.paths.output_dir / "modeling_command_distribution.png"
            export_jsonl_path = bundle.paths.output_dir / "modeling_dataset_export.jsonl"
            summary_text_path = bundle.paths.output_dir / "modeling_dataset_summary.txt"
            legacy_dat_path = bundle.paths.output_dir / "modeling_dataset_legacy_compat.dat"
            pairs.extend(build_modeling_dataset_summary_pairs(metrics=metrics))
            pairs.extend(
                [
                    ("Workspace Plot", str(workspace_plot_path) if workspace_plot_path.exists() else "not written"),
                    ("Command Plot", str(command_plot_path) if command_plot_path.exists() else "not written"),
                    ("Export JSONL", str(export_jsonl_path) if export_jsonl_path.exists() else "not written"),
                    ("Legacy DAT", str(legacy_dat_path) if legacy_dat_path.exists() else "not written"),
                    ("Summary Note", str(summary_text_path) if summary_text_path.exists() else "not written"),
                ]
            )
            return pairs
        if bundle_experiment_name == "registration_validation":
            metrics = bundle.summary.experiment_metrics if isinstance(bundle.summary.experiment_metrics, dict) else {}
            csv_path = bundle.paths.output_dir / "metrics.csv"
            summary_text_path = bundle.paths.output_dir / "registration_validation_summary.txt"
            origin_report_path = bundle.paths.output_dir / "registration_frame_origins_report.png"
            fre_report_path = bundle.paths.output_dir / "registration_fre_report.png"
            spread_report_path = bundle.paths.output_dir / "registration_transform_spread_report.png"
            pairs.extend(
                [
                    ("Selected Runs", str(int(metrics.get("selected_run_count", 0) or 0))),
                    ("Valid Runs", str(int(metrics.get("valid_run_count", 0) or 0))),
                    ("Skipped Runs", str(int(metrics.get("invalid_run_count", 0) or 0))),
                    ("Mean FRE", self._format_metric_value((metrics.get("fre_summary_mm", {}) or {}).get("mean"))),
                    (
                        "Mean dT",
                        self._format_metric_value((metrics.get("translation_delta_to_consensus_summary_mm", {}) or {}).get("mean")),
                    ),
                    (
                        "Mean dR",
                        self._format_metric_value((metrics.get("rotation_delta_to_consensus_summary_deg", {}) or {}).get("mean")),
                    ),
                    ("Metrics CSV", str(csv_path) if csv_path.exists() else "not written"),
                    ("Origin Report", str(origin_report_path) if origin_report_path.exists() else "not written"),
                    ("FRE Report", str(fre_report_path) if fre_report_path.exists() else "not written"),
                    ("Spread Report", str(spread_report_path) if spread_report_path.exists() else "not written"),
                    ("Summary Note", str(summary_text_path) if summary_text_path.exists() else "not written"),
                ]
            )
            return pairs
        if bundle_experiment_name == "pivot_validation":
            metrics = bundle.summary.experiment_metrics if isinstance(bundle.summary.experiment_metrics, dict) else {}
            csv_path = bundle.paths.output_dir / "metrics.csv"
            summary_text_path = bundle.paths.output_dir / "pivot_validation_summary.txt"
            pairs.extend(
                [
                    ("Selected Runs", str(int(metrics.get("selected_run_count", 0) or 0))),
                    ("Valid Runs", str(int(metrics.get("valid_run_count", 0) or 0))),
                    ("Skipped Runs", str(int(metrics.get("invalid_run_count", 0) or 0))),
                    (
                        "Mean Norm",
                        self._format_metric_value((metrics.get("tip_norm_summary_mm", {}) or {}).get("mean")),
                    ),
                    (
                        "Mean RMSE",
                        self._format_metric_value((metrics.get("rmse_summary_mm", {}) or {}).get("mean")),
                    ),
                    (
                        "Mean dC",
                        self._format_metric_value((metrics.get("distance_to_consensus_summary_mm", {}) or {}).get("mean")),
                    ),
                    ("Metrics CSV", str(csv_path) if csv_path.exists() else "not written"),
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
        if bundle_experiment_name == "servo_tracker_sync_validation":
            metrics = bundle.summary.experiment_metrics if isinstance(bundle.summary.experiment_metrics, dict) else {}
            sync = dict(metrics.get("servo_tracker_sync", {}) or {})
            histogram_path = bundle.paths.output_dir / "servo_tracker_offset_histogram.png"
            timeseries_path = bundle.paths.output_dir / "servo_tracker_offset_timeseries.png"
            pose_path = bundle.paths.output_dir / "servo_tracker_pose_command_timeseries.png"
            validity_path = bundle.paths.output_dir / "servo_tracker_validity_summary.png"
            summary_text_path = bundle.paths.output_dir / "servo_tracker_sync_summary.txt"
            pairs.extend(
                [
                    ("Backend", str(metrics.get("backend_identity", "n/a") or "n/a")),
                    ("Tool IDs", ", ".join(metrics.get("requested_tool_ids", []) or [])),
                    ("Servo IDs", ", ".join(str(value) for value in (metrics.get("selected_servo_ids") or []))),
                    ("Motion Protocol", str(metrics.get("motion_protocol", "n/a") or "n/a")),
                    ("Analyzed Tracker Samples", str(int(metrics.get("sample_count_analyzed", 0) or 0))),
                    ("Servo Telemetry Samples", str(int(metrics.get("servo_telemetry_sample_count", 0) or 0))),
                    ("Servo Command Samples", str(int(metrics.get("servo_command_sample_count", 0) or 0))),
                    ("Effective Loop Hz", self._format_metric_value(metrics.get("effective_loop_rate_hz"))),
                    ("Unique-Frame Hz", self._format_metric_value(metrics.get("unique_frame_rate_hz"))),
                    ("Mean Total Time", self._format_metric_value(metrics.get("mean_total_cycle_ms"))),
                    ("P95 Total Time", self._format_metric_value(metrics.get("p95_total_cycle_ms"))),
                    (
                        "Tracker->Servo Telemetry Mean",
                        self._format_metric_value(sync.get("tracker_to_servo_telemetry_mean_offset_ms")),
                    ),
                    (
                        "Tracker->Servo Telemetry P95",
                        self._format_metric_value(sync.get("tracker_to_servo_telemetry_p95_offset_ms")),
                    ),
                    (
                        "Tracker->Servo Command P95",
                        self._format_metric_value(sync.get("tracker_to_servo_command_p95_offset_ms")),
                    ),
                    (
                        "Tracker->Servo <= 10 ms",
                        (
                            f"{100.0 * float(sync.get('tracker_to_servo_telemetry_within_10ms_rate', 0.0) or 0.0):.1f}%"
                            if sync.get("tracker_to_servo_telemetry_within_10ms_rate") is not None
                            else "n/a"
                        ),
                    ),
                    (
                        "Motion Metric Source",
                        str((metrics.get("tracker_motion_metric", {}) or {}).get("source", "unavailable") or "unavailable"),
                    ),
                    ("Offset Histogram", str(histogram_path) if histogram_path.exists() else "not written"),
                    ("Offset Timeseries", str(timeseries_path) if timeseries_path.exists() else "not written"),
                    ("Pose / Command Plot", str(pose_path) if pose_path.exists() else "not written"),
                    ("Validity Plot", str(validity_path) if validity_path.exists() else "not written"),
                    ("Summary Note", str(summary_text_path) if summary_text_path.exists() else "not written"),
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
        if experiment_name in {"registration_validation", "pivot_validation"}:
            return "offline thesis validation"
        if experiment_name == "single_segment_repeatability":
            return "live thesis repeatability"
        if experiment_name == "command_schedule_validation":
            return "software validation"
        if experiment_name == "tracker_timing_validation":
            return "backend diagnostic"
        if experiment_name == "servo_tracker_sync_validation":
            return "motion sync validation"
        if experiment_name == "penprobe_chasing_demo":
            return "live bounded demo"
        if experiment_name == "pivot_calibration":
            return "offline" if config_payload.get("input_path") else ("dry-run" if bool(config_payload.get("dry_run", False)) else "live")
        return "dry-run" if bool(config_payload.get("dry_run", False)) else "live"

    @staticmethod
    def _config_summary_label(experiment_name: str, config_payload: dict[str, Any]) -> str:
        if experiment_name == "single_segment_repeatability":
            config = SingleSegmentRepeatabilityConfig.from_dict(config_payload)
            return (
                f"legacy 17 targets ({float(config.inner_ring_radius_mm):.1f}/{float(config.outer_ring_radius_mm):.1f} mm rings), "
                f"272 visits, 544 captures, "
                f"cap {int(config.max_target_tick_delta_from_startup)} ticks, "
                f"settle {float(config.settle_time_s):.2f}s, "
                f"tool {config.tool_id}"
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
            mode = str(config_payload.get("dataset_mode", "workspace_coverage") or "workspace_coverage")
            samples_per_command = int(config_payload.get("samples_per_command", config_payload.get("sample_count_per_point", 1)) or 1)
            if config_payload.get("command_points"):
                return f"{len(config_payload.get('command_points', []) or [])} explicit command points, {samples_per_command} samples/command"
            if mode == "hysteresis_path_dependence":
                return (
                    f"hysteresis, "
                    f"{int(config_payload.get('hysteresis_target_count', 0) or 0)} targets, "
                    f"{int(config_payload.get('hysteresis_cycle_count', 0) or 0)} cycle(s)"
                )
            if mode == "repeatability_linked":
                return (
                    f"repeatability-linked, "
                    f"{int(config_payload.get('repeatability_block_count', 0) or 0)} block(s), "
                    f"{int(config_payload.get('sample_count_target', 0) or 0)} target samples"
                )
            return (
                f"workspace coverage, "
                f"target {int(config_payload.get('sample_count_target', 0) or 0)} samples, "
                f"{samples_per_command} samples/command"
            )
        if experiment_name == "registration_validation":
            return f"{len(config_payload.get('run_paths', []) or [])} saved registration runs selected"
        if experiment_name == "pivot_validation":
            return f"{len(config_payload.get('run_paths', []) or [])} saved pivot-calibration runs selected"
        if experiment_name == "replay_runner":
            return str(config_payload.get("dataset_path", "select an existing run"))
        if experiment_name == "pretension_validation":
            run_mode = str(config_payload.get("mode", "single_servo_trace") or "single_servo_trace")
            start_mode = str(config_payload.get("pretension_start_mode", "live_default") or "live_default")
            return (
                f"{run_mode.replace('_', ' ')}, servo {config_payload.get('servo_id', 'n/a')}, "
                f"start {start_mode.replace('_', ' ')}, "
                f"step {config_payload.get('step_ticks', 'live default')} ticks, "
                f"max travel {config_payload.get('max_travel_ticks', 'live default')} ticks, "
                f"tracker={'on' if bool(config_payload.get('include_tracker_displacement', True)) else 'off'}"
            )
        if experiment_name == "penprobe_chasing_demo":
            return (
                f"0A-to-0B XY chase, cap {int(config_payload.get('max_tick_delta_from_startup', 500) or 500)} ticks, "
                f"period {float(config_payload.get('loop_period_s', 0.075) or 0.075):.3f}s, "
                f"mapping {str(config_payload.get('mapping_mode', 'aggressive_tick_demo'))}"
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
        if experiment_name == "servo_tracker_sync_validation":
            config = ServoTrackerSyncValidationConfig.from_dict(config_payload)
            return (
                f"servos {','.join(str(value) for value in config.servo_ids)}, "
                f"tools {','.join(config.requested_tool_ids)}, "
                f"amplitude {int(config.command_amplitude_ticks)} ticks, "
                f"{float(config.step_period_s):.3f} s cadence"
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
