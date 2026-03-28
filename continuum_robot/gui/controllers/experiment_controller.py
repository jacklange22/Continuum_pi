"""Canonical experiment workspace controller for the operator GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any
from uuid import uuid4

import yaml

from continuum_robot.gui.experiment_preflight import PreflightReport, RUN_BLOCKED, evaluate_preflight
from continuum_robot.gui.experiment_visualization import VisualizationModel, build_visualization_model


@dataclass(frozen=True)
class ExperimentOption:
    """Static GUI definition for one critical experiment."""

    name: str
    title: str
    description: str
    badges: list[str]


@dataclass
class RunHistoryEntry:
    """Lightweight summary of one prior experiment run."""

    path: str
    experiment_name: str
    timestamp_utc: str
    status: str
    label: str


@dataclass
class ExperimentViewState:
    """UI-facing state for the canonical experiment workspace."""

    experiment_options: list[ExperimentOption] = field(default_factory=list)
    selected_experiment: str = ""
    experiment_title: str = ""
    experiment_description: str = ""
    experiment_badges: list[str] = field(default_factory=list)
    config_text: str = ""
    config_error: str | None = None
    operator_notes: str = ""
    output_root: str = ""
    planned_output_dir: str = ""
    preflight_report: PreflightReport = field(default_factory=lambda: PreflightReport(overall_status=RUN_BLOCKED))
    run_checklist: list[tuple[str, str]] = field(default_factory=list)
    run_active: bool = False
    progress_current: int = 0
    progress_total: int = 0
    status_message: str = "Select an experiment and review preflight checks."
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
    """Owns the experiment workspace state, execution, and history loading."""

    _EXPERIMENTS = {
        "repeatability_dataset": ExperimentOption(
            name="repeatability_dataset",
            title="Repeatability Dataset",
            description="Main robot dataset collection with revisit scheduling, repeated samples, and repeatability metrics.",
            badges=["Tracking", "Repeatability", "Registration optional", "Dry-run or live"],
        ),
        "aurora_grid_accuracy": ExperimentOption(
            name="aurora_grid_accuracy",
            title="Aurora Grid Accuracy",
            description="Tracker-only accuracy and precision characterization against a physical or synthetic grid.",
            badges=["Tracking", "Grid truth", "Tip calibration aware", "Registration optional"],
        ),
        "pivot_calibration": ExperimentOption(
            name="pivot_calibration",
            title="Pivot Calibration",
            description="Generate the pen-probe tip file from live or offline pivot samples before registration.",
            badges=["Tip file generation", "Offline or live", "No registration required"],
        ),
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
        self.state = ExperimentViewState(
            experiment_options=list(self._EXPERIMENTS.values()),
            output_root=str(experiment_runner.output_dir),
        )

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._history_dirty = True
        self._visualization_dirty = True
        self._current_bundle = None
        self._live_samples = []
        self._planned_output_dir_name = ""
        self._selected_payload: dict[str, Any] = {}

        self.select_experiment("repeatability_dataset")

    def refresh(self) -> ExperimentViewState:
        with self._lock:
            config_text = self.state.config_text
            selected_experiment = self.state.selected_experiment
            output_root = self._resolve_repo_path(self.state.output_root)
            current_bundle = self._current_bundle
            live_samples = list(self._live_samples)
            color_mode = self.state.color_mode
            show_centroids = self.state.show_centroids
            show_truth = self.state.show_truth
            if self._history_dirty:
                self.state.history = self._scan_run_history(output_root)
                self._history_dirty = False

        config_payload, config_error = self._parse_config_text(config_text)
        self._selected_payload = dict(config_payload or {})
        planned_output_dir = output_root / self._planned_output_dir_name
        preflight = evaluate_preflight(
            experiment_name=selected_experiment,
            config_payload=config_payload,
            config_error=config_error,
            settings=self.settings,
            tracking_snapshot=self.tracking_service.get_snapshot(),
            servo_connected=bool(self.servo_service.is_connected),
            neutral_setpoints=self.servo_service.load_neutral_setpoints(),
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
            tracking_snapshot=self.tracking_service.get_snapshot(),
            planned_output_dir=planned_output_dir,
        )

        with self._lock:
            option = self._EXPERIMENTS[selected_experiment]
            self.state.experiment_title = option.title
            self.state.experiment_description = option.description
            self.state.experiment_badges = list(option.badges)
            self.state.config_error = config_error
            self.state.preflight_report = preflight
            self.state.run_checklist = checklist
            self.state.planned_output_dir = str(planned_output_dir)
            self.state.visualization_model = visualization_model
            self.state.result_summary_lines = list(visualization_model.summary_lines)
            return self.state

    def refresh_prerequisites(self) -> ExperimentViewState:
        """Compatibility alias for the main window refresh loop."""
        return self.refresh()

    def select_experiment(self, experiment_name: str) -> None:
        if experiment_name not in self._EXPERIMENTS:
            raise KeyError(f"Unsupported experiment: {experiment_name}")
        example_path = self._example_config_path(experiment_name)
        config_text = example_path.read_text(encoding="utf-8") if example_path.exists() else "{}\n"
        with self._lock:
            self.state.selected_experiment = experiment_name
            self.state.config_text = config_text
            self.state.loaded_run_path = None
            self.state.last_error = None
            self.state.last_output_path = None
            self._current_bundle = None
            self._live_samples = []
            self._visualization_dirty = True
            self._reset_planned_output_dir_locked()

    def set_config_text(self, text: str) -> None:
        with self._lock:
            self.state.config_text = str(text)
            self._current_bundle = None
            self._live_samples = []
            self._visualization_dirty = True
            self._reset_planned_output_dir_locked()

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
                with self._lock:
                    self._current_bundle = bundle
                    self.state.last_output_path = str(result.paths.output_dir)
                    self.state.loaded_run_path = str(result.paths.output_dir)
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
        if experiment_name not in self._EXPERIMENTS:
            raise RuntimeError(f"Run {path} is not one of the canonical experiment workspace types.")
        config_text = yaml.safe_dump(bundle.metadata.config_used, sort_keys=False) or "{}\n"
        with self._lock:
            self.state.selected_experiment = experiment_name
            self.state.config_text = config_text
            self.state.operator_notes = str(bundle.metadata.operator_notes or "")
            self.state.loaded_run_path = str(bundle.paths.output_dir)
            self.state.last_output_path = str(bundle.paths.output_dir)
            self.state.status_message = f"Loaded prior run {bundle.paths.output_dir.name}."
            self.state.last_error = None
            self._current_bundle = bundle
            self._live_samples = []
            self._visualization_dirty = True
            self._reset_planned_output_dir_locked()

    def shutdown(self, timeout_s: float = 2.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout_s)

    def _example_config_path(self, experiment_name: str) -> Path:
        return self.project_root / "config" / f"experiment_{experiment_name}.example.yaml"

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

    def _reset_planned_output_dir_locked(self) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = uuid4().hex[:12]
        safe_name = self.state.selected_experiment.replace(" ", "_")
        self._planned_output_dir_name = f"{timestamp}_{safe_name}_{run_id}"

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

    def _scan_run_history(self, output_root: Path) -> list[RunHistoryEntry]:
        entries: list[RunHistoryEntry] = []
        if not output_root.exists():
            return entries
        for run_dir in sorted(output_root.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
            metadata_path = run_dir / "metadata.json"
            summary_path = run_dir / "summary.json"
            if not (run_dir.is_dir() and metadata_path.exists() and summary_path.exists()):
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            experiment_name = str(metadata.get("experiment_name", "unknown"))
            timestamp = str(metadata.get("timestamp_utc", ""))
            status = str(summary.get("status", "unknown"))
            label = f"{run_dir.name} | {experiment_name} | {status}"
            entries.append(
                RunHistoryEntry(
                    path=str(run_dir),
                    experiment_name=experiment_name,
                    timestamp_utc=timestamp,
                    status=status,
                    label=label,
                )
            )
            if len(entries) >= 25:
                break
        return entries

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

    def _build_run_checklist(self, *, experiment_name: str, config_payload: dict[str, Any], tracking_snapshot, planned_output_dir: Path) -> list[tuple[str, str]]:
        backend = tracking_snapshot.selected_backend_name or tracking_snapshot.backend_identity or "unknown"
        if experiment_name == "repeatability_dataset":
            tool_ids = str(config_payload.get("tool_id", "0A"))
            mode = "dry-run" if bool(config_payload.get("dry_run", True)) else "live"
            tip_file = "n/a"
        elif experiment_name == "aurora_grid_accuracy":
            tool_ids = str(config_payload.get("tool_id", "0B"))
            mode = "dry-run" if bool(config_payload.get("dry_run", True)) else "live"
            tip_file = str(config_payload.get("tip_file") or config_payload.get("tip_vector_mm") or "optional")
        else:
            tool_ids = str(config_payload.get("tool_id", "0B"))
            mode = (
                "offline"
                if config_payload.get("input_path")
                else ("dry-run" if bool(config_payload.get("dry_run", False)) else "live")
            )
            tip_file = str(config_payload.get("output_tip_file", "data/tip_cals/generated_penprobe_tip.csv"))
        return [
            ("Experiment", experiment_name),
            ("Backend", backend),
            ("Mode", mode),
            ("Tool IDs", tool_ids),
            ("Tip File", tip_file),
            ("Registration", str(self.registration_path) if self.registration_path.exists() else "missing"),
            ("Output", str(planned_output_dir)),
        ]
