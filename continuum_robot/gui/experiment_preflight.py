"""Pure-Python experiment preflight validation for the GUI workspace."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import os
from typing import Any

from continuum_robot.experiments.builtins import (
    CollectPoseCommandDatasetConfig,
    CommandScheduleValidationConfig,
    PretensionValidationExperimentConfig,
    ReplayRunnerConfig,
    TrackerTimingValidationConfig,
)
from continuum_robot.experiments.critical_experiments import (
    GridDefinitionConfig,
    PivotCalibrationConfig,
    RepeatabilityDatasetConfig,
    build_repeatability_preview,
)
from continuum_robot.experiments.schedules import generate_command_schedule


PREFLIGHT_OK = "ok"
PREFLIGHT_WARNING = "warning"
PREFLIGHT_BLOCKED = "blocked"
PREFLIGHT_INFO = "info"

RUN_OK = "ok_to_run"
RUN_WARNING = "ok_with_warning"
RUN_BLOCKED = "blocked"


@dataclass
class PreflightCheck:
    """One operator-facing preflight check."""

    key: str
    label: str
    status: str
    message: str


@dataclass
class PreflightReport:
    """Structured preflight report for the GUI."""

    overall_status: str
    checks: list[PreflightCheck] = field(default_factory=list)
    overwrite_targets: list[str] = field(default_factory=list)
    experiment_name: str = ""
    planned_output_dir: str = ""

    @property
    def requires_confirmation(self) -> bool:
        return bool(self.overwrite_targets)

    @property
    def blocking_messages(self) -> list[str]:
        return [check.message for check in self.checks if check.status == PREFLIGHT_BLOCKED]

    @property
    def warning_messages(self) -> list[str]:
        return [check.message for check in self.checks if check.status == PREFLIGHT_WARNING]

    @property
    def info_messages(self) -> list[str]:
        return [check.message for check in self.checks if check.status == PREFLIGHT_INFO]

    @property
    def summary(self) -> str:
        if self.overall_status == RUN_BLOCKED:
            return "Blocked. Fix the required items before running."
        if self.overall_status == RUN_WARNING:
            return "Ready with warnings. Review the highlighted items."
        return "Ready. Required checks passed."


def evaluate_preflight(
    *,
    experiment_name: str,
    config_payload: dict[str, Any] | None,
    config_error: str | None,
    settings,
    tracking_snapshot,
    servo_connected: bool,
    neutral_setpoints: dict[int, int],
    registration_path: Path,
    output_root: Path,
    planned_output_dir: Path,
    project_root: Path,
) -> PreflightReport:
    """Evaluate experiment-specific guardrails for the GUI workspace."""
    checks: list[PreflightCheck] = []
    overwrite_targets: list[str] = []

    if config_error:
        checks.append(
            PreflightCheck(
                key="config",
                label="Config",
                status=PREFLIGHT_BLOCKED,
                message=f"Config YAML is invalid. Fix it, then refresh checks. Details: {config_error}",
            )
        )
        return _finalize_report(
            experiment_name=experiment_name,
            planned_output_dir=planned_output_dir,
            checks=checks,
            overwrite_targets=overwrite_targets,
        )

    payload = dict(config_payload or {})
    backend_name = tracking_snapshot.selected_backend_name or tracking_snapshot.backend_identity or "unknown"
    tracker_ready = tracking_snapshot.canonical_state in {"mock", "streaming_healthy", "streaming_degraded"}

    checks.append(
        PreflightCheck(
            key="output_root",
            label="Output Path",
            status=_output_status(output_root),
            message=_output_message(output_root, planned_output_dir),
        )
    )
    if planned_output_dir.exists():
        checks.append(
            _blocked(
                "planned_output_dir",
                "Planned Output",
                f"Planned output folder already exists: {planned_output_dir}",
            )
        )

    if experiment_name == "repeatability_dataset":
        checks.append(_tracking_state_check(settings=settings, tracker_ready=tracker_ready, backend_name=backend_name, tracking_snapshot=tracking_snapshot))
        config = RepeatabilityDatasetConfig.from_dict(payload)
        expected_dims = len(settings.robot.tendon_to_servo or settings.robot.servo_ids)
        try:
            preview = build_repeatability_preview(config, tendon_count=expected_dims)
        except ValueError as exc:
            checks.append(
                _blocked(
                    "config_targets",
                    "Schedule",
                    str(exc),
                )
            )
        else:
            if not preview.target_catalog:
                checks.append(_blocked("config_targets", "Schedule", "Repeatability schedule has no target points."))
            else:
                checks.append(
                    _ok(
                        "dimensions",
                        "Dimensionality",
                        f"Repeatability targets match the configured {expected_dims}-tendon robot.",
                    )
                )
                checks.append(
                    _info(
                        "schedule",
                        "Schedule",
                        f"{preview.summary['target_count']} targets, "
                        f"{preview.summary['visit_count']} revisits, "
                        f"{preview.summary['planned_sample_count']} planned measurement samples "
                        f"using target set '{config.schedule.target_set}'.",
                    )
                )

        checks.append(
            _tool_check(
                tool_id=config.tool_id,
                snapshot=tracking_snapshot,
                mock_mode=bool(settings.runtime.mock_mode),
            )
        )

        if config.dry_run:
            checks.append(
                _info(
                    "mode",
                    "Run Mode",
                    "This run is in dry-run mode. Commands will be computed and logged, but not sent to hardware."
                    if servo_connected
                    else "This run is in dry-run mode and does not require live servo hardware.",
                )
            )
        elif not servo_connected:
            checks.append(
                _blocked(
                    "mode",
                    "Run Mode",
                    "Live repeatability requires a connected servo service. Connect OpenRB and verify servo status first.",
                )
            )
        else:
            checks.append(_ok("mode", "Run Mode", "Live repeatability will use the connected servo service."))

        neutral_count = len(neutral_setpoints)
        if config.dry_run:
            status = PREFLIGHT_OK if neutral_count == expected_dims else PREFLIGHT_WARNING
            message = (
                f"Neutral setpoints are available for all {expected_dims} tendons."
                if neutral_count == expected_dims
                else (
                    "Neutral setpoints are missing or incomplete. Dry-run output can still be recorded, "
                    "but motor values may be incomplete."
                )
            )
            checks.append(PreflightCheck("neutral_setpoints", "Neutral Setpoints", status, message))
        else:
            status = PREFLIGHT_OK if neutral_count == expected_dims else PREFLIGHT_BLOCKED
            message = (
                f"Neutral setpoints are available for all {expected_dims} tendons."
                if neutral_count == expected_dims
                else "Live repeatability needs neutral setpoints for every tendon. Capture and save neutral first."
            )
            checks.append(PreflightCheck("neutral_setpoints", "Neutral Setpoints", status, message))

        checks.append(
            _registration_quality_check(
                registration_path=registration_path,
                tracking_snapshot=tracking_snapshot,
                missing_message=(
                    "Registration file is missing. The run can continue, but pose will stay in tracker frame only."
                ),
            )
        )

    elif experiment_name == "aurora_grid_accuracy":
        checks.append(_tracking_state_check(settings=settings, tracker_ready=tracker_ready, backend_name=backend_name, tracking_snapshot=tracking_snapshot))
        config = GridDefinitionConfig.from_dict(payload)
        dims = list(config.dimensions)
        if len(dims) not in {2, 3} or any(value <= 0 for value in dims) or config.spacing_mm <= 0.0:
            checks.append(
                _blocked(
                    "grid_definition",
                    "Grid Definition",
                    "Grid definition is invalid. Use 2 or 3 positive dimensions and a positive spacing value.",
                )
            )
        else:
            checks.append(
                _ok(
                    "grid_definition",
                    "Grid Definition",
                    f"Ideal truth grid is {dims} with {config.spacing_mm:.2f} mm spacing.",
                )
            )
        if config.truth_points_file:
            truth_path = _resolve_repo_path(project_root, config.truth_points_file)
            if truth_path.exists():
                checks.append(_ok("truth_grid", "Truth Grid", f"Truth grid file found: {truth_path}"))
            else:
                checks.append(
                    _blocked(
                        "truth_grid",
                        "Truth Grid",
                        f"Truth grid file is missing: {truth_path}. Select a valid file or remove truth_points_file to generate the grid from the config.",
                    )
                )
        else:
            checks.append(
                _info(
                    "truth_grid",
                    "Truth Grid",
                    "Truth grid will be generated in a local grid frame and aligned in code to the measured centroids. Registration is not required.",
                )
            )
        checks.append(
            _tool_check(
                tool_id=config.tool_id,
                snapshot=tracking_snapshot,
                mock_mode=bool(settings.runtime.mock_mode),
            )
        )
        if config.use_tip_calibration:
            tip_path = _resolve_repo_path(project_root, config.tip_file) if config.tip_file else None
            if config.tip_vector_mm is not None:
                checks.append(_ok("tip_calibration", "Tip Calibration", "Tip calibration is provided directly in config."))
            elif tip_path is not None and tip_path.exists():
                checks.append(_ok("tip_calibration", "Tip Calibration", f"Tip calibration file found: {tip_path}"))
            elif config.allow_coil_origin_fallback:
                checks.append(
                    _warning(
                        "tip_calibration",
                        "Tip Calibration",
                        "Tip calibration is missing. The experiment can run with coil-origin fallback, but tip-based accuracy metrics will be partial.",
                    )
                )
            else:
                checks.append(
                    _blocked(
                        "tip_calibration",
                        "Tip Calibration",
                        "Tip calibration is required for this run. Provide tip_vector_mm, point to a valid tip_file, or enable allow_coil_origin_fallback.",
                    )
                )
        else:
            checks.append(_info("tip_calibration", "Tip Calibration", "Tip calibration is not required for this run."))

        checks.append(
            _info(
                "registration",
                "Registration",
                "Aligned grid-consistency validation does not require registration. Board placement is solved from the captured labeled points.",
            )
        )

        if config.dry_run:
            checks.append(_info("mode", "Run Mode", "Grid accuracy will run in dry-run/mock mode."))
        elif tracker_ready:
            checks.append(_ok("mode", "Run Mode", "Grid accuracy will use live tracking input."))
        else:
            checks.append(
                _blocked(
                    "mode",
                    "Run Mode",
                    "Live grid accuracy requires healthy tracker streaming. Confirm frames and tool visibility before running.",
                )
            )

        complete_points = 0
        partial_points = 0
        raw_samples = 0
        required_samples = max(1, int(config.samples_per_point))
        for point in config.captured_points:
            if not isinstance(point, dict):
                continue
            point_samples = point.get("raw_samples", []) or []
            sample_count = len(point_samples)
            raw_samples += sample_count
            if sample_count >= required_samples:
                complete_points += 1
            elif sample_count > 0:
                partial_points += 1
        if complete_points == 0 and partial_points == 0 and raw_samples == 0:
            complete_points = int(payload.get("captured_point_count", 0) or 0)
            raw_samples = int(payload.get("captured_sample_count", 0) or 0)
        if complete_points >= 3:
            checks.append(
                _ok(
                    "captured_points",
                    "Captured Points",
                    f"{complete_points} grid points are complete ({raw_samples} raw samples). The aligned residual solve is ready.",
                )
            )
        elif partial_points > 0 or raw_samples > 0:
            checks.append(
                _blocked(
                    "captured_points",
                    "Captured Points",
                    f"Capture at least 3 complete labeled points before saving. "
                    f"Currently {complete_points} complete, {partial_points} partial, {raw_samples} raw samples.",
                )
            )
        else:
            checks.append(
                _blocked(
                    "captured_points",
                    "Captured Points",
                    "No labeled grid points are captured yet. Select a point on the custom page and capture samples with tool 0B.",
                )
            )

    elif experiment_name == "pivot_calibration":
        checks.append(_tracking_state_check(settings=settings, tracker_ready=tracker_ready, backend_name=backend_name, tracking_snapshot=tracking_snapshot))
        config = PivotCalibrationConfig.from_dict(payload)
        if config.input_path:
            input_path = _resolve_repo_path(project_root, config.input_path)
            if input_path.exists():
                checks.append(_ok("input_file", "Pivot Input", f"Offline pivot input found: {input_path}"))
            else:
                checks.append(
                    _blocked(
                        "input_file",
                        "Pivot Input",
                        f"Pivot input file is missing: {input_path}. Choose a valid recorded file or clear input_path for live collection.",
                    )
                )
        elif config.dry_run:
            checks.append(_info("input_file", "Pivot Input", "Pivot calibration will use synthetic dry-run samples."))
        elif tracker_ready:
            checks.append(_ok("input_file", "Pivot Input", "Pivot calibration will collect live tracker samples."))
        else:
            checks.append(
                _blocked(
                    "input_file",
                    "Pivot Input",
                    "Live pivot calibration requires healthy tracker streaming. Start tracking and confirm tool visibility first.",
                )
            )

        if not config.input_path:
            checks.append(
                _tool_check(
                    tool_id=config.tool_id,
                    snapshot=tracking_snapshot,
                    mock_mode=bool(settings.runtime.mock_mode or config.dry_run),
                )
            )

        output_tip_path = _resolve_repo_path(project_root, config.output_tip_file)
        try:
            output_tip_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        status = PREFLIGHT_OK if os.access(output_tip_path.parent, os.W_OK) else PREFLIGHT_BLOCKED
        message = (
            f"Tip file will be written to {output_tip_path}."
            if status == PREFLIGHT_OK
            else f"Tip output directory is not writable: {output_tip_path.parent}. Pick a writable path before running."
        )
        checks.append(PreflightCheck("tip_output", "Tip Output", status, message))
        if output_tip_path.exists():
            overwrite_targets.append(str(output_tip_path))
            checks.append(
                _warning(
                    "overwrite",
                    "Overwrite Protection",
                    f"Tip output file already exists and will be overwritten after explicit confirmation: {output_tip_path}",
                )
            )

        if config.input_path:
            checks.append(_info("registration", "Registration", "Pivot calibration does not require registration."))
        else:
            checks.append(_info("registration", "Registration", "Pivot calibration does not require registration."))

    elif experiment_name == "command_schedule_validation":
        config = CommandScheduleValidationConfig.from_dict(payload)
        try:
            points = generate_command_schedule(config.schedule)
        except Exception as exc:
            checks.append(
                _blocked(
                    "schedule",
                    "Schedule",
                    f"Command schedule is invalid: {exc}",
                )
            )
        else:
            checks.append(
                _ok(
                    "schedule",
                    "Schedule",
                    f"{len(points)} command points generated from a {config.schedule.kind} schedule in {config.schedule.dimensions} dimensions.",
                )
            )
        checks.append(_info("tracking_state", "Tracking", "Tracker input is not required for schedule validation."))
        checks.append(_info("mode", "Run Mode", "Schedule validation is a pure software validation run."))
        checks.append(_info("registration", "Registration", "Registration is not required for schedule validation."))

    elif experiment_name == "collect_pose_command_dataset":
        checks.append(_tracking_state_check(settings=settings, tracker_ready=tracker_ready, backend_name=backend_name, tracking_snapshot=tracking_snapshot))
        config = CollectPoseCommandDatasetConfig.from_dict(payload)
        try:
            points = (
                list(config.command_points)
                if config.command_points
                else generate_command_schedule(config.command_schedule)
            )
        except Exception as exc:
            checks.append(
                _blocked(
                    "schedule",
                    "Command Schedule",
                    f"Dataset collection schedule is invalid: {exc}",
                )
            )
            points = []
        if points:
            checks.append(
                _ok(
                    "schedule",
                    "Command Schedule",
                    f"{len(points)} command point(s) will be sampled with {max(1, int(config.sample_count_per_point))} sample(s) per point.",
                )
            )
        expected_dims = len(settings.robot.tendon_to_servo or settings.robot.servo_ids)
        neutral_count = len(neutral_setpoints)
        if config.dry_run:
            checks.append(_info("mode", "Run Mode", "Dataset collection will run in dry-run mode."))
            checks.append(
                PreflightCheck(
                    "neutral_setpoints",
                    "Neutral Setpoints",
                    PREFLIGHT_OK if neutral_count == expected_dims else PREFLIGHT_WARNING,
                    (
                        f"Neutral setpoints are available for all {expected_dims} tendons."
                        if neutral_count == expected_dims
                        else "Neutral setpoints are incomplete. Dry-run output can still be recorded, but live command replay would be partial."
                    ),
                )
            )
        elif not servo_connected:
            checks.append(
                _blocked(
                    "mode",
                    "Run Mode",
                    "Live dataset collection requires a connected servo service.",
                )
            )
        else:
            checks.append(_ok("mode", "Run Mode", "Live dataset collection will use the connected servo service."))
            checks.append(
                PreflightCheck(
                    "neutral_setpoints",
                    "Neutral Setpoints",
                    PREFLIGHT_OK if neutral_count == expected_dims else PREFLIGHT_BLOCKED,
                    (
                        f"Neutral setpoints are available for all {expected_dims} tendons."
                        if neutral_count == expected_dims
                        else "Live dataset collection requires neutral setpoints for every tendon."
                    ),
                )
            )
        checks.append(
            _registration_quality_check(
                registration_path=registration_path,
                tracking_snapshot=tracking_snapshot,
                missing_message=(
                    "Registration file is missing. Samples will still be saved, but robot-frame pose will be unavailable."
                ),
            )
        )

    elif experiment_name == "replay_runner":
        config = ReplayRunnerConfig.from_dict(payload)
        if not config.dataset_path:
            checks.append(_blocked("dataset_path", "Replay Dataset", "Replay runner requires dataset_path."))
        else:
            dataset_path = _resolve_repo_path(project_root, config.dataset_path)
            if dataset_path.exists():
                checks.append(_ok("dataset_path", "Replay Dataset", f"Replay dataset found: {dataset_path}"))
            else:
                checks.append(_blocked("dataset_path", "Replay Dataset", f"Replay dataset is missing: {dataset_path}"))
        checks.append(_info("tracking_state", "Tracking", "Replay runner uses saved datasets and does not require live tracking."))
        checks.append(_info("mode", "Run Mode", "Replay runner is an offline analysis workflow."))
        checks.append(_info("registration", "Registration", "Replay runner uses the saved dataset state."))

    elif experiment_name == "pretension_validation":
        config = PretensionValidationExperimentConfig.from_dict(payload)
        if not servo_connected:
            checks.append(_blocked("servo_service", "Servo Service", "Pretension validation requires a connected servo service."))
        else:
            checks.append(_ok("servo_service", "Servo Service", "Pretension validation can use the connected servo service."))
        configured_ids = [int(value) for value in (settings.robot.servo_ids or [])]
        if configured_ids and int(config.servo_id) not in configured_ids:
            checks.append(
                _blocked(
                    "servo_id",
                    "Servo ID",
                    f"Servo {config.servo_id} is not part of the configured robot servo IDs {configured_ids}.",
                )
            )
        else:
            checks.append(_ok("servo_id", "Servo ID", f"Pretension validation will target servo {config.servo_id}."))
        if bool(config.include_tracker_displacement):
            if tracker_ready:
                checks.append(
                    _ok(
                        "tracking_state",
                        "Tracking",
                        "Tracker displacement will be sampled alongside the pretension trace when frames remain fresh.",
                    )
                )
            else:
                checks.append(
                    _warning(
                        "tracking_state",
                        "Tracking",
                        "Tracker displacement was requested, but live tracking is not currently ready. "
                        "The run can still save current-versus-travel data only.",
                    )
                )
        else:
            checks.append(
                _info(
                    "tracking_state",
                    "Tracking",
                    "Tracker displacement is disabled for this run. Only servo current-versus-travel data will be collected.",
                )
            )
        if int(config.step_ticks or 1) <= 0:
            checks.append(
                _blocked(
                    "step_ticks",
                    "Step Ticks",
                    "step_ticks must be positive.",
                )
            )
        else:
            checks.append(
                _ok(
                    "step_ticks",
                    "Step Ticks",
                    "Pretension response will be recorded against increasing travel from the untensioned reference.",
                )
            )
        checks.append(
            _info(
                "registration",
                "Registration",
                "Registration is optional. When present, tracker displacement will prefer robot-frame tip motion; otherwise it falls back to tracker-frame tool motion when available.",
            )
        )
        checks.append(
            _ok(
                "scientific_framing",
                "Scientific Framing",
                "This experiment validates current-displacement response and startup-state reproducibility. It does not claim true tendon-force sensing.",
            )
        )

    elif experiment_name == "tracker_timing_validation":
        config = TrackerTimingValidationConfig.from_dict(payload)
        configured_backend = str(settings.serial.tracker_backend or "").strip().lower()
        selected_backend = str(
            tracking_snapshot.selected_backend_name or tracking_snapshot.backend_identity or ""
        ).strip().lower()
        if bool(settings.runtime.mock_mode):
            checks.append(
                _blocked(
                    "mock_mode",
                    "Runtime Mode",
                    "Tracker timing validation is blocked in mock mode. Use the live Python NDI backend for a meaningful benchmark.",
                )
            )
        else:
            checks.append(
                _ok(
                    "runtime_mode",
                    "Runtime Mode",
                    "Live runtime mode is enabled for a real backend timing benchmark.",
                )
            )
        if configured_backend not in {"ndi", ""}:
            checks.append(
                _blocked(
                    "configured_backend",
                    "Configured Backend",
                    f"Configured tracker backend is '{configured_backend}'. This diagnostic targets the Python NDI backend path only.",
                )
            )
        else:
            checks.append(
                _ok(
                    "configured_backend",
                    "Configured Backend",
                    "Configured tracker backend is the Python NDI path.",
                )
            )
        if selected_backend in {"bridge", "tracker_bridge_json"}:
            checks.append(
                _blocked(
                    "selected_backend",
                    "Selected Backend",
                    "The active tracker backend is the legacy bridge path. This diagnostic should run against the Python NDI backend instead.",
                )
            )
        elif tracker_ready:
            checks.append(
                _ok(
                    "selected_backend",
                    "Selected Backend",
                    f"Live tracking is ready via {tracking_snapshot.backend_identity or tracking_snapshot.selected_backend_name or 'ndi'}.",
                )
            )
        else:
            checks.append(
                _warning(
                    "selected_backend",
                    "Selected Backend",
                    "Live tracking is not currently streaming. The experiment can start the backend, but the recorded run will only be meaningful if the Python NDI backend connects successfully.",
                )
            )
        requested_tools = ", ".join(config.requested_tool_ids) or "n/a"
        stop_mode = (
            f"{int(config.sample_count_target)} analyzed samples"
            if config.sample_count_target is not None
            else f"{float(config.run_duration_s):.1f} s duration"
        )
        checks.append(
            _info(
                "benchmark_scope",
                "Benchmark Scope",
                f"Will benchmark requested tools {requested_tools}, discard {int(config.warmup_samples)} warmup sample(s), and stop after {stop_mode}.",
            )
        )
        if bool(config.enable_servo_logging):
            if servo_connected:
                checks.append(
                    _ok(
                        "servo_logging",
                        "Servo Sync Logging",
                        "Servo telemetry will be logged alongside tracker timing samples for timestamp-alignment analysis.",
                    )
                )
            else:
                checks.append(
                    _blocked(
                        "servo_logging",
                        "Servo Sync Logging",
                        "Servo sync logging was requested, but the servo service is not connected.",
                    )
                )
        else:
            checks.append(
                _info(
                    "servo_logging",
                    "Servo Sync Logging",
                    "Servo sync logging is disabled for this run.",
                )
            )
        checks.append(
            _ok(
                "scientific_framing",
                "Scientific Framing",
                "This diagnostic measures backend acquisition timing and duplicate/stale frame behavior. It does not use GUI refresh rate as a tracker-performance metric.",
            )
        )

    else:
        checks.append(_blocked("experiment", "Experiment", f"Unsupported experiment selection: {experiment_name}"))

    return _finalize_report(
        experiment_name=experiment_name,
        planned_output_dir=planned_output_dir,
        checks=checks,
        overwrite_targets=overwrite_targets,
    )


def _resolve_repo_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return Path(project_root) / path


def _output_status(output_root: Path) -> str:
    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except Exception:
        return PREFLIGHT_BLOCKED
    return PREFLIGHT_OK if os.access(output_root, os.W_OK) else PREFLIGHT_BLOCKED


def _output_message(output_root: Path, planned_output_dir: Path) -> str:
    if _output_status(output_root) != PREFLIGHT_OK:
        return f"Output root is not writable: {output_root}"
    return f"Run data will be written to {planned_output_dir}"


def _tool_check(*, tool_id: str, snapshot, mock_mode: bool) -> PreflightCheck:
    tool = snapshot.tools.get(tool_id)
    if tool is not None and tool.tracking_state == "tracked":
        return _ok("tool_ids", "Tool IDs", f"Required tool {tool_id} is currently tracked.")
    if mock_mode:
        return _info(
            "tool_ids",
            "Tool IDs",
            f"Tool {tool_id} is not currently visible. Mock or dry-run mode can still generate the expected stream for this run.",
        )
    return _blocked(
        "tool_ids",
        "Tool IDs",
        f"Required tool {tool_id} is not currently tracked. Confirm the tool is enabled in Aurora and visible in the Tracking tab.",
    )


def _tracking_state_check(*, settings, tracker_ready: bool, backend_name: str, tracking_snapshot) -> PreflightCheck:
    return PreflightCheck(
        key="tracking_state",
        label="Tracking",
        status=(
            PREFLIGHT_OK
            if tracker_ready
            else (PREFLIGHT_INFO if settings.runtime.mock_mode else PREFLIGHT_BLOCKED)
        ),
        message=(
            f"Tracker is ready on backend {backend_name} with state {tracking_snapshot.canonical_state}."
            if tracker_ready
            else (
                f"App is in mock mode. The {backend_name} backend will self-start for dry-run execution."
                if settings.runtime.mock_mode
                else (
                    f"Tracker is not ready. Current state is {tracking_snapshot.canonical_state} on backend {backend_name}. "
                    "Start tracking and confirm live frames before running."
                )
            )
        ),
    )


def _registration_quality_check(*, registration_path: Path, tracking_snapshot, missing_message: str) -> PreflightCheck:
    if not registration_path.exists():
        return _warning("registration", "Registration", missing_message)

    try:
        payload = json.loads(registration_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _warning(
            "registration",
            "Registration",
            f"Registration file exists but could not be read: {exc}",
        )

    metrics = payload.get("validation_metrics", {}) if isinstance(payload.get("validation_metrics"), dict) else {}
    fre_mm = metrics.get("overall_fre_mm", payload.get("fre_mm"))
    max_residual_mm = metrics.get("max_residual_mm")
    worst_label = metrics.get("worst_landmark_label")
    if worst_label is None and isinstance(metrics.get("residual_summary"), dict):
        worst_label = metrics["residual_summary"].get("worst_landmark_label")
    worst_residual_mm = metrics.get("worst_landmark_residual_mm")
    if worst_residual_mm is None and isinstance(metrics.get("residual_summary"), dict):
        worst_residual_mm = metrics["residual_summary"].get("worst_landmark_residual_mm")
    configured_limit = metrics.get("configured_max_fre_mm")

    parts = [f"Registration file found: {registration_path}"]
    if fre_mm is not None:
        parts.append(f"FRE={float(fre_mm):.3f} mm")
    if max_residual_mm is not None:
        parts.append(f"max residual={float(max_residual_mm):.3f} mm")
    if worst_label and worst_residual_mm is not None:
        parts.append(f"worst landmark {worst_label}={float(worst_residual_mm):.3f} mm")

    status = PREFLIGHT_OK
    if configured_limit is not None and fre_mm is not None and float(fre_mm) > float(configured_limit):
        status = PREFLIGHT_WARNING
        parts.append(f"configured limit {float(configured_limit):.3f} mm exceeded")
    elif configured_limit is not None and max_residual_mm is not None and float(max_residual_mm) > float(configured_limit):
        status = PREFLIGHT_WARNING
        parts.append("inspect landmark residual distribution")

    canonical_state = getattr(tracking_snapshot, "canonical_state", "disconnected")
    if canonical_state not in {"disconnected", "mock"}:
        tracking_registration_state = getattr(tracking_snapshot, "registration_state", "missing_registration")
        tip_pose_status = getattr(tracking_snapshot, "tip_pose_status", "missing_registration")
        if tracking_registration_state != "loaded":
            status = PREFLIGHT_WARNING
            parts.append(f"tracking runtime registration state is {tracking_registration_state}")
        elif tip_pose_status != "ok":
            status = PREFLIGHT_WARNING
            parts.append(f"live tip pose status is {tip_pose_status}")

    return PreflightCheck(
        "registration",
        "Registration",
        status,
        "; ".join(parts),
    )


def _ok(key: str, label: str, message: str) -> PreflightCheck:
    return PreflightCheck(key=key, label=label, status=PREFLIGHT_OK, message=message)


def _warning(key: str, label: str, message: str) -> PreflightCheck:
    return PreflightCheck(key=key, label=label, status=PREFLIGHT_WARNING, message=message)


def _blocked(key: str, label: str, message: str) -> PreflightCheck:
    return PreflightCheck(key=key, label=label, status=PREFLIGHT_BLOCKED, message=message)


def _info(key: str, label: str, message: str) -> PreflightCheck:
    return PreflightCheck(key=key, label=label, status=PREFLIGHT_INFO, message=message)


def _finalize_report(
    *,
    experiment_name: str,
    planned_output_dir: Path,
    checks: list[PreflightCheck],
    overwrite_targets: list[str],
) -> PreflightReport:
    if any(check.status == PREFLIGHT_BLOCKED for check in checks):
        overall_status = RUN_BLOCKED
    elif any(check.status == PREFLIGHT_WARNING for check in checks):
        overall_status = RUN_WARNING
    else:
        overall_status = RUN_OK
    return PreflightReport(
        overall_status=overall_status,
        checks=checks,
        overwrite_targets=list(overwrite_targets),
        experiment_name=experiment_name,
        planned_output_dir=str(planned_output_dir),
    )
