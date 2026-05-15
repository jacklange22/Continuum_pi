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
    ServoTrackerSyncValidationConfig,
    TrackerTimingValidationConfig,
)
from continuum_robot.experiments.calibration_validation import (
    PivotValidationConfig,
    RegistrationValidationConfig,
)
from continuum_robot.experiments.single_segment_repeatability import (
    LEGACY_CAPTURE_COUNT,
    LEGACY_TARGET_COUNT,
    LEGACY_VISIT_COUNT,
    SingleSegmentRepeatabilityConfig,
    build_legacy_17_point_targets,
    build_targets_for_preset,
    generate_legacy_revisit_sequence,
    load_repeatability_metrics_from_run,
    repeatability_target_tick_profile,
)
from continuum_robot.experiments.penprobe_chasing_demo import (
    MAPPING_AGGRESSIVE_TICK_DEMO,
    MAPPING_PAIRED_XY_PROPORTIONAL,
    PenprobeChasingDemoConfig,
    _validate_legacy_mapping_config,
)
from continuum_robot.experiments.two_segment_startup_validation import DUAL_SEGMENT_BLOCK_MESSAGE
from continuum_robot.experiments.two_segment_collect_pose_dataset import (
    BLOCK_MESSAGE as TWO_SEGMENT_DATASET_BLOCK_MESSAGE,
    TwoSegmentCollectPoseDatasetConfig,
)
from continuum_robot.tracking.two_segment_roles import (
    resolve_two_segment_tracking_roles,
    two_segment_role_readiness_rows,
)
from continuum_robot.servos.displacement_mapper import TendonDisplacementMapper
from continuum_robot.experiments.critical_experiments import (
    GridDefinitionConfig,
    PivotCalibrationConfig,
)
from continuum_robot.experiments.schedules import generate_command_schedule
from continuum_robot.tracking.runtime_tip_policy import (
    WORKFLOW_MODELING_DATASET,
    WORKFLOW_PRETENSION_VALIDATION,
    evaluate_runtime_tip_trust,
)
from continuum_robot.servos.segment_readiness import (
    evaluate_selected_segment_readiness,
)
from continuum_robot.servos.sign_mapping_check import ServoMappingCheckRepository


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


PLANNED_OUTPUT_NOT_CREATED_YET = "not_created_yet"
PLANNED_OUTPUT_AVAILABLE = "available"
PLANNED_OUTPUT_OWNED_BY_CURRENT_RUN = "owned_by_current_run"
PLANNED_OUTPUT_EXISTING_PREVIOUS_RUN = "existing_previous_run"
PLANNED_OUTPUT_CONFLICT = "conflict"


@dataclass
class PreflightReport:
    """Structured preflight report for the GUI."""

    overall_status: str
    checks: list[PreflightCheck] = field(default_factory=list)
    overwrite_targets: list[str] = field(default_factory=list)
    experiment_name: str = ""
    planned_output_dir: str = ""
    planned_output_state: str = PLANNED_OUTPUT_NOT_CREATED_YET

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
    servo_calibration_summary=None,
    active_run_output_dir: Path | None = None,
) -> PreflightReport:
    """Evaluate experiment-specific guardrails for the GUI workspace.

    ``active_run_output_dir`` lets the caller mark one directory as belonging to the
    in-flight run so it is treated as ``owned_by_current_run`` rather than a stale
    conflict.
    """
    checks: list[PreflightCheck] = []
    overwrite_targets: list[str] = []
    planned_output_state = classify_planned_output_state(
        planned_output_dir=planned_output_dir,
        active_run_output_dir=active_run_output_dir,
    )

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
            planned_output_state=planned_output_state,
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
    if planned_output_state == PLANNED_OUTPUT_OWNED_BY_CURRENT_RUN:
        checks.append(
            _ok(
                "planned_output_dir",
                "Planned Output",
                f"Output folder created for active run: {planned_output_dir}",
            )
        )
    elif planned_output_state == PLANNED_OUTPUT_EXISTING_PREVIOUS_RUN:
        checks.append(
            _blocked(
                "planned_output_dir",
                "Planned Output",
                f"Planned output folder already exists from a previous run: {planned_output_dir}. "
                "Refresh to allocate a unique timestamp before running.",
            )
        )
    elif planned_output_state == PLANNED_OUTPUT_CONFLICT:
        checks.append(
            _blocked(
                "planned_output_dir",
                "Planned Output",
                f"Planned output folder is in a conflict state: {planned_output_dir}.",
            )
        )

    if experiment_name == "single_segment_repeatability":
        config = SingleSegmentRepeatabilityConfig.from_dict(payload)
        configured_backend = str(settings.serial.tracker_backend or "").strip().lower()
        selected_backend = str(
            tracking_snapshot.selected_backend_name or tracking_snapshot.backend_identity or ""
        ).strip().lower()
        if bool(settings.runtime.mock_mode):
            checks.append(
                _blocked(
                    "mock_mode",
                    "Runtime Mode",
                    "Single-segment repeatability is a live thesis experiment. Disable mock mode before running.",
                )
            )
        else:
            checks.append(_ok("runtime_mode", "Runtime Mode", "Live runtime mode is enabled."))
        if configured_backend not in {"ndi", ""}:
            checks.append(
                _blocked(
                    "configured_backend",
                    "Configured Backend",
                    f"Configured tracker backend is '{configured_backend}'. Repeatability must use the active Python NDI path.",
                )
            )
        elif "bridge" in selected_backend:
            checks.append(
                _blocked(
                    "selected_backend",
                    "Selected Backend",
                    "The active tracker backend appears to be tracker_bridge. Use the Python NDI backend before running repeatability.",
                )
            )
        elif tracking_snapshot.canonical_state == "streaming_healthy":
            checks.append(
                _ok(
                    "tracking_state",
                    "Tracking",
                    f"Tracker is healthy via {tracking_snapshot.backend_identity or tracking_snapshot.selected_backend_name or 'ndi'}.",
                )
            )
        else:
            checks.append(
                _blocked(
                    "tracking_state",
                    "Tracking",
                    f"Tracker must be streaming healthy. Current state is {tracking_snapshot.canonical_state}.",
                )
            )
        checks.append(
            _strict_tool_gate_check(
                tool_id=config.tool_id,
                snapshot=tracking_snapshot,
                max_tracker_age_s=float(config.max_tracker_age_s),
            )
        )
        if tracking_snapshot.registration_state == "loaded" and tracking_snapshot.T_robot_aurora is not None:
            checks.append(
                _ok(
                    "registration",
                    "Base Registration",
                    "Accepted base registration is loaded and available to the live transform chain.",
                )
            )
        else:
            checks.append(
                _blocked(
                    "registration",
                    "Base Registration",
                    f"Accepted base registration must be loaded. Runtime state is {tracking_snapshot.registration_state}.",
                )
            )
        pivot_tip_file = getattr(settings.registration, "penprobe_file", None)
        if pivot_tip_file:
            pivot_tip_path = _resolve_repo_path(project_root, pivot_tip_file)
            if pivot_tip_path.exists():
                checks.append(
                    _ok(
                        "pivot_tip",
                        "0B Pivot Tip Calibration",
                        f"Pivot-calibrated 0B tip file is present: {pivot_tip_path}",
                    )
                )
            else:
                checks.append(
                    _blocked(
                        "pivot_tip",
                        "0B Pivot Tip Calibration",
                        f"Pivot-calibrated 0B tip file is missing: {pivot_tip_path}. Run and accept 0B pivot calibration first.",
                    )
                )
        else:
            checks.append(
                _blocked(
                    "pivot_tip",
                    "0B Pivot Tip Calibration",
                    "No penprobe_file is configured for the 0B pivot-calibrated tip.",
                )
            )
        runtime_tip_policy = evaluate_runtime_tip_trust(
            snapshot=tracking_snapshot,
            workflow=experiment_name,
            allow_lower_trust=bool(config.allow_debug_coil_as_tip),
        )
        if runtime_tip_policy.allowed_for_workflow and runtime_tip_policy.thesis_trusted:
            checks.append(
                _ok(
                    "runtime_tip",
                    "Runtime Tip Policy",
                    runtime_tip_policy.status_message,
                )
            )
        elif runtime_tip_policy.allowed_for_workflow:
            checks.append(
                _blocked(
                    "runtime_tip",
                    "Runtime Tip Policy",
                    "Repeatability requires thesis_trusted runtime tip policy output. "
                    f"Requested workflow/platform={runtime_tip_policy.requested_workflow}, "
                    f"resolved canonical workflow={runtime_tip_policy.workflow}, "
                    f"mode={runtime_tip_policy.mode}, trust={runtime_tip_policy.trust_label}.",
                )
            )
        else:
            mode = str(getattr(tracking_snapshot, "runtime_tip_mode", "latest_accepted") or "latest_accepted")
            checks.append(
                _blocked(
                    "runtime_tip",
                    "Runtime Tip Policy",
                    "Repeatability requires a thesis_trusted runtime tip policy outcome. "
                    f"Requested workflow/platform={runtime_tip_policy.requested_workflow}, "
                    f"resolved canonical workflow={runtime_tip_policy.workflow}. "
                    f"Mode={mode}, trust={runtime_tip_policy.trust_label}, "
                    f"Runtime state={tracking_snapshot.runtime_tip_calibration_state}, "
                    f"tip pose={tracking_snapshot.tip_pose_status}, "
                    f"fallback={tracking_snapshot.runtime_tip_identity_fallback}. "
                    f"Reasons={runtime_tip_policy.reasons or ['policy_not_allowed']}.",
                )
            )
        servo_ids = [int(value) for value in settings.robot.active_segment_servo_ids()]
        if len(servo_ids) == 4:
            checks.append(
                _ok(
                    "single_segment",
                    "Single-Segment Assumption",
                    f"Active segment {settings.robot.active_segment_label()} ({settings.robot.active_segment_key()}) servo IDs: {servo_ids}.",
                )
            )
        else:
            checks.append(
                _blocked(
                    "single_segment",
                    "Single-Segment Assumption",
                    f"This protocol requires exactly 4 active segment servos/tendons. Found {servo_ids}.",
                )
            )
        if float(config.outer_ring_radius_mm) < float(config.inner_ring_radius_mm):
            checks.append(
                _blocked(
                    "target_geometry",
                    "Target Geometry",
                    "outer_ring_radius_mm must be >= inner_ring_radius_mm.",
                )
            )
        else:
            target_catalog = build_targets_for_preset(
                config.target_preset,
                inner_ring_radius_mm=float(config.inner_ring_radius_mm),
                outer_ring_radius_mm=float(config.outer_ring_radius_mm),
            )
            mapper = TendonDisplacementMapper(
                spool_diameter_cm=float(settings.robot.spool_diameter_cm),
                ticks_per_rev=int(settings.robot.ticks_per_revolution),
            )
            target_profile = repeatability_target_tick_profile(
                targets=target_catalog,
                mapper=mapper,
            )
            max_tick_delta = int(target_profile.get("max_abs_tick_delta", 0) or 0)
            cap = int(config.max_target_tick_delta_from_startup)
            if max_tick_delta > cap:
                checks.append(
                    _blocked(
                        "target_geometry",
                        "Target Geometry",
                        "Repeatability target amplitude exceeds the configured experiment cap: "
                        f"{max_tick_delta} ticks requested > {cap} ticks cap. "
                        "Reduce ring radii or raise max_target_tick_delta_from_startup intentionally.",
                    )
                )
            else:
                checks.append(
                    _ok(
                        "target_geometry",
                        "Target Geometry",
                        (
                            f"Inner/outer ring={float(config.inner_ring_radius_mm):.1f}/"
                            f"{float(config.outer_ring_radius_mm):.1f} mm, "
                            f"max target delta={max_tick_delta} ticks, "
                            f"cap={cap} ticks, spool={float(settings.robot.spool_diameter_cm) * 10.0:.1f} mm."
                        ),
                    )
                )
        if not servo_connected:
            checks.append(
                _blocked(
                    "servo_connection",
                    "Servo Connection",
                    "OpenRB / DYNAMIXEL service must be connected for live repeatability commands.",
                )
            )
        else:
            checks.append(_ok("servo_connection", "Servo Connection", "Servo service is connected."))
        missing_neutral = [servo_id for servo_id in servo_ids if int(servo_id) not in neutral_setpoints]
        if missing_neutral:
            checks.append(
                _blocked(
                    "neutral_setpoints",
                    "Neutral Setpoints",
                    "Neutral setpoints are missing for servo(s): " + ", ".join(str(value) for value in missing_neutral),
                )
            )
        else:
            checks.append(_ok("neutral_setpoints", "Neutral Setpoints", "Neutral setpoints exist for all configured tendons."))
        checks.extend(
            _selected_segment_truth_checks(
                settings=settings,
                servo_calibration_summary=servo_calibration_summary,
                project_root=project_root,
                servo_connected=servo_connected,
                require_startup=True,
                strict_startup=True,
                allow_calibration_creation=False,
                label_prefix="Repeatability",
            )
        )
        checks.append(_pretension_artifact_check(servo_ids=servo_ids, servo_calibration_summary=servo_calibration_summary))
        if config.baseline_run_path:
            baseline_path = _resolve_repo_path(project_root, config.baseline_run_path)
            try:
                baseline_metrics = load_repeatability_metrics_from_run(baseline_path)
            except Exception as exc:
                checks.append(
                    _blocked(
                        "baseline_run",
                        "Baseline Comparison",
                        f"Selected baseline run cannot be used for comparison: {exc}",
                    )
                )
            else:
                baseline_status = str(baseline_metrics.get("status", "unknown") or "unknown")
                baseline_message = (
                    f"Baseline loaded from {baseline_path}; "
                    f"baseline RMS={float(baseline_metrics.get('overall_repeatability_rms_mm', 0.0) or 0.0):.3f} mm; "
                    f"status={baseline_status}."
                )
                if baseline_status != "success":
                    checks.append(
                        _warning(
                            "baseline_run",
                            "Baseline Comparison",
                            baseline_message + " Compare carefully because the selected baseline was not a fully valid run.",
                        )
                    )
                else:
                    checks.append(
                        _ok(
                            "baseline_run",
                            "Baseline Comparison",
                            baseline_message,
                        )
                    )
        else:
            checks.append(
                _info(
                    "baseline_run",
                    "Baseline Comparison",
                    "No baseline selected. The run will save full metrics, but no improvement delta will be computed.",
                )
            )
        preset_targets = build_targets_for_preset(
            config.target_preset,
            inner_ring_radius_mm=float(config.inner_ring_radius_mm),
            outer_ring_radius_mm=float(config.outer_ring_radius_mm),
        )
        preset_visits = generate_legacy_revisit_sequence(
            preset_targets,
            seed=int(config.random_seed),
            visits_per_target=int(config.visits_per_target),
        )
        checks.append(
            _ok(
                "protocol",
                "Protocol",
                f"{config.target_preset} preset: {len(preset_targets)} targets, "
                f"{len(preset_visits)} approach/repeat visits, {len(preset_visits) * 2} planned captures.",
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

    elif experiment_name == "registration_validation":
        config = RegistrationValidationConfig.from_dict(payload)
        checks.append(_info("tracking_state", "Tracking", "Registration validation is an offline analysis run. Live tracking is not required."))
        checks.append(_info("mode", "Run Mode", "Registration validation analyzes previously saved registration solves."))
        checks.append(_info("registration", "Registration", "This run consumes saved registration artifacts rather than the live registration state."))
        if len(config.run_paths) < 2:
            checks.append(
                _blocked(
                    "run_paths",
                    "Selected Runs",
                    "Select at least 2 saved registration runs before running the validation analysis.",
                )
            )
        else:
            missing = [
                str(_resolve_repo_path(project_root, raw_path))
                for raw_path in config.run_paths
                if not _resolve_repo_path(project_root, raw_path).exists()
            ]
            if missing:
                checks.append(
                    _blocked(
                        "run_paths",
                        "Selected Runs",
                        f"Some selected registration runs are missing: {', '.join(missing[:3])}",
                    )
                )
            else:
                checks.append(
                    _ok(
                        "run_paths",
                        "Selected Runs",
                        f"{len(config.run_paths)} saved registration runs will be analyzed offline.",
                    )
                )

    elif experiment_name == "pivot_validation":
        config = PivotValidationConfig.from_dict(payload)
        checks.append(_info("tracking_state", "Tracking", "Pivot validation is an offline analysis run. Live tracking is not required."))
        checks.append(_info("mode", "Run Mode", "Pivot validation analyzes previously saved pivot-calibration runs."))
        checks.append(_info("registration", "Registration", "Pivot validation does not require live registration state."))
        if len(config.run_paths) < 2:
            checks.append(
                _blocked(
                    "run_paths",
                    "Selected Runs",
                    "Select at least 2 saved pivot-calibration runs before running the validation analysis.",
                )
            )
        else:
            missing = [
                str(_resolve_repo_path(project_root, raw_path))
                for raw_path in config.run_paths
                if not _resolve_repo_path(project_root, raw_path).exists()
            ]
            if missing:
                checks.append(
                    _blocked(
                        "run_paths",
                        "Selected Runs",
                        f"Some selected pivot runs are missing: {', '.join(missing[:3])}",
                    )
                )
            else:
                checks.append(
                    _ok(
                        "run_paths",
                        "Selected Runs",
                        f"{len(config.run_paths)} saved pivot-calibration runs will be analyzed offline.",
                    )
                )

    elif experiment_name == "collect_pose_command_dataset":
        config = CollectPoseCommandDatasetConfig.from_dict(payload)
        no_tracker_servo_only = bool(
            (bool(config.allow_no_tracker_test_run) or str(config.run_trust_mode or "").strip().lower() == "servo_only")
            and not tracker_ready
        )
        if no_tracker_servo_only:
            checks.append(
                _warning(
                    "tracking_state",
                    "Tracking",
                    "Tracker is not connected. This run is allowed as a servo-only hardware test. "
                    "Tip position, robot-frame pose, modeling labels, and thesis repeatability metrics will not be produced.",
                )
            )
        else:
            checks.append(_tracking_state_check(settings=settings, tracker_ready=tracker_ready, backend_name=backend_name, tracking_snapshot=tracking_snapshot))
        try:
            points = list(config.command_points) if config.command_points else (
                generate_command_schedule(config.command_schedule)
                if bool(config.legacy_schedule_override)
                else []
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
                    "Command Protocol",
                    f"Legacy schedule override will run {len(points)} explicit command point(s) with {max(1, int(config.samples_per_command))} sample(s) per command.",
                )
            )
            planned_command_count = len(points)
        else:
            if config.dataset_mode == "workspace_coverage":
                planned_command_count = max(1, int((int(config.sample_count_target) + max(1, int(config.samples_per_command)) - 1) / max(1, int(config.samples_per_command))))
                mode_message = (
                    f"Workspace coverage will target about {int(config.sample_count_target)} accepted samples "
                    f"using bounded {'quasi-random' if config.quasi_random else 'random'} pair commands."
                )
            elif config.dataset_mode == "hysteresis_path_dependence":
                planned_command_count = max(1, int(config.hysteresis_target_count)) * max(1, int(config.hysteresis_cycle_count)) * max(2, int(config.hysteresis_prior_family_count)) * 2
                mode_message = (
                    f"Hysteresis path-dependence will run {int(config.hysteresis_cycle_count)} cycle(s), "
                    f"{int(config.hysteresis_target_count)} target family points, and "
                    f"{int(config.hysteresis_prior_family_count)} prior-state families."
                )
            else:
                planned_command_count = max(1, int(config.repeatability_block_count)) * max(1, int((int(config.sample_count_target) + max(1, int(config.samples_per_command)) - 1) / max(1, int(config.samples_per_command))))
                mode_message = (
                    f"Repeatability-linked collection will run {int(config.repeatability_block_count)} trusted startup block(s) "
                    f"with about {int(config.sample_count_target)} accepted samples per block."
                )
            checks.append(_ok("schedule", "Command Protocol", mode_message))
        checks.append(
            _ok(
                "capture_plan",
                "Capture Plan",
                f"Planned commands={int(planned_command_count)}, samples/command={max(1, int(config.samples_per_command))}, planned captures={int(planned_command_count * max(1, int(config.samples_per_command)) + 2)}.",
            )
        )
        operating_context = settings.robot.operating_context()
        commanded_servo_ids = [int(value) for value in operating_context.commanded_servo_ids]
        expected_dims = len(commanded_servo_ids)
        missing_active_neutral = [servo_id for servo_id in commanded_servo_ids if int(servo_id) not in neutral_setpoints]
        if config.dry_run:
            checks.append(_warning("mode", "Run Mode", "Dry-run is enabled. This is useful for protocol rehearsal, but the run is not thesis-grade modeling data."))
            checks.append(
                PreflightCheck(
                    "neutral_setpoints",
                    "Neutral Setpoints",
                    PREFLIGHT_OK if not missing_active_neutral else PREFLIGHT_INFO,
                    (
                        f"Neutral setpoints are available for commanded servos {commanded_servo_ids}."
                        if not missing_active_neutral
                        else f"Neutral setpoints are incomplete for commanded servos {missing_active_neutral}. Dry-run output can still be recorded, but live replay would not be ready."
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
            checks.append(_ok("mode", "Run Mode", "Live Motor Babble collection will use the connected servo service."))
            checks.append(
                PreflightCheck(
                    "neutral_setpoints",
                    "Neutral Setpoints",
                    PREFLIGHT_OK if not missing_active_neutral else PREFLIGHT_BLOCKED,
                    (
                        f"Neutral setpoints are available for commanded servos {commanded_servo_ids}."
                        if not missing_active_neutral
                        else f"Live dataset collection requires neutral setpoints for every commanded servo; missing {missing_active_neutral}."
                    ),
                )
            )
        if operating_context.operating_mode == "parallel_single":
            if expected_dims != 8 or not operating_context.mirror_pairs:
                checks.append(_blocked("single_segment", "Parallel Single", f"Parallel-single Motor Babble requires 8 commanded servos and mirror pairs; found {commanded_servo_ids}."))
            else:
                checks.append(
                    _ok(
                        "single_segment",
                        "Parallel Single Demo",
                        "Parallel Single Demo: the same single-segment command is sent to Spine 1 and Spine 2. "
                        f"Mirror pairs: {operating_context.mirror_pairs}.",
                    )
                )
                checks.append(
                    _warning(
                        "parallel_single_scope",
                        "Parallel Single Scope",
                        "This is not two-segment kinematics. It is synchronized single-segment command playback.",
                    )
                )
                amplitude_cm = abs(float(config.workspace_amplitude_cm))
                if amplitude_cm > 0.25:
                    checks.append(
                        _warning(
                            "parallel_single_amplitude",
                            "Parallel Single Amplitude",
                            "Start parallel_single demo conservatively at ±0.25 cm and increase only after both spines "
                            "move correctly. Current workspace_amplitude_cm is "
                            f"±{amplitude_cm:.2f} cm.",
                        )
                    )
        elif expected_dims != 4:
            checks.append(_blocked("single_segment", "Single Segment", f"Motor Babble currently supports exactly 4 active segment servos; found {commanded_servo_ids}."))
        else:
            checks.append(
                _ok(
                    "single_segment",
                    "Single Segment",
                    f"Single Segment: only {settings.robot.active_segment_key()} {commanded_servo_ids}. "
                    "This is not all-8 parallel_single demo mode.",
                )
            )
        collect_debug_mode = bool(
            config.dry_run
            or no_tracker_servo_only
            or bool(config.allow_lower_trust_pretension)
            or str(config.run_trust_mode or "").strip().lower() in {"servo_only", "current_only", "lower_trust", "debug", "debug_only", "mock"}
        )
        checks.extend(
            _selected_segment_truth_checks(
                settings=settings,
                servo_calibration_summary=servo_calibration_summary,
                project_root=project_root,
                servo_connected=servo_connected,
                require_startup=True,
                strict_startup=not collect_debug_mode,
                allow_calibration_creation=collect_debug_mode,
                label_prefix="Collect Pose",
            )
        )
        if no_tracker_servo_only:
            checks.append(
                _warning(
                    "registration",
                    "Base Registration",
                    "Registration is not required for this servo-only hardware test. No robot-frame pose labels will be produced.",
                )
            )
            checks.append(
                _warning(
                    "runtime_tip",
                    "Runtime Tip Policy",
                    "Runtime tip is unavailable. This run will be marked servo_only, lower_trust, not_thesis_trusted, and not valid for model training.",
                )
            )
            if bool(config.export_legacy_dat):
                checks.append(
                    _warning(
                        "legacy_export",
                        "Legacy Export",
                        "Legacy .dat export will be disabled for servo-only no-tracker output because no valid pose labels exist.",
                    )
                )
        else:
            checks.append(_strict_tool_gate_check(tool_id=config.tool_id, snapshot=tracking_snapshot, max_tracker_age_s=float(config.max_tracker_age_s)))
            if tracking_snapshot.registration_state == "loaded" and tracking_snapshot.T_robot_aurora is not None:
                checks.append(_ok("registration", "Base Registration", "Accepted base registration is loaded and active."))
            elif not bool(config.require_robot_frame_tip) and bool(config.allow_lower_trust_runtime_tip):
                checks.append(
                    _warning(
                        "registration",
                        "Base Registration",
                        f"Base registration is not loaded (state={tracking_snapshot.registration_state}). This lower-trust run will fall back to tracker-frame-only outputs.",
                    )
                )
            else:
                checks.append(_blocked("registration", "Base Registration", f"Accepted base registration must be loaded. Runtime state is {tracking_snapshot.registration_state}."))
            runtime_tip_policy = evaluate_runtime_tip_trust(
                snapshot=tracking_snapshot,
                workflow=WORKFLOW_MODELING_DATASET,
                allow_lower_trust=bool(config.allow_lower_trust_runtime_tip),
            )
            if runtime_tip_policy.allowed_for_workflow and runtime_tip_policy.thesis_trusted:
                checks.append(_ok("runtime_tip", "Runtime Tip Policy", runtime_tip_policy.status_message))
            elif runtime_tip_policy.allowed_for_workflow:
                checks.append(
                    _warning(
                        "runtime_tip",
                        "Runtime Tip Policy",
                        f"{runtime_tip_policy.status_message} This run is explicitly {runtime_tip_policy.trust_label}.",
                    )
                )
            elif not bool(config.require_robot_frame_tip) and bool(config.allow_lower_trust_runtime_tip):
                checks.append(
                    _warning(
                        "runtime_tip",
                        "Runtime Tip Policy",
                        f"Live robot-frame tip pose is not required for this explicitly lower-trust run. Current policy={runtime_tip_policy.trust_label}, mode={runtime_tip_policy.mode}, state={tracking_snapshot.runtime_tip_calibration_state}, tip pose={tracking_snapshot.tip_pose_status}.",
                    )
                )
            else:
                checks.append(
                    _blocked(
                        "runtime_tip",
                        "Runtime Tip Policy",
                        (
                            f"Runtime tip policy must allow modeling data. "
                            f"Mode={runtime_tip_policy.mode}, trust={runtime_tip_policy.trust_label}, state={tracking_snapshot.runtime_tip_calibration_state}, tip pose={tracking_snapshot.tip_pose_status}, reasons={runtime_tip_policy.reasons or ['policy_not_allowed']}."
                        ),
                    )
                )
        if servo_calibration_summary is None:
            checks.append(
                _warning("pretension", "Pretension Source", "Pretension/startup artifact state is unavailable for this dry-run.")
                if config.dry_run
                else _blocked("pretension", "Pretension Source", "Pretension/startup artifact state is unavailable.")
            )
        else:
            source_summary = servo_calibration_summary.pretension_source_summary(
                [int(value) for value in settings.robot.active_segment_servo_ids()]
            )
            if source_summary.accepted and source_summary.usable:
                checks.append(_ok("pretension", "Pretension Source", source_summary.message))
            elif config.dry_run:
                checks.append(
                    _warning(
                        "pretension",
                        "Pretension Source",
                        source_summary.message + " Dry-run remains available for protocol rehearsal only.",
                    )
                )
            elif config.allow_lower_trust_pretension:
                checks.append(_warning("pretension", "Pretension Source", source_summary.message + " Lower-trust override is enabled for this run."))
            else:
                checks.append(_blocked("pretension", "Pretension Source", source_summary.message))

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

    elif experiment_name == "two_segment_startup_validation":
        operating_context = settings.robot.operating_context()
        expected_ids = [int(value) for value in getattr(operating_context, "expected_servo_ids", [])]
        commanded_ids = [int(value) for value in getattr(operating_context, "commanded_servo_ids", [])]
        if operating_context.operating_mode != "dual_segment":
            checks.append(_blocked("operating_mode", "Operating Mode", DUAL_SEGMENT_BLOCK_MESSAGE))
        else:
            checks.append(_ok("operating_mode", "Operating Mode", "dual_segment all-8 manual startup validation is selected."))
        checks.append(_physical_assembly_check(operating_context))
        checks.append(_baud_advisory_check(settings))
        if not servo_connected:
            checks.append(_blocked("servo_service", "Servo Service", "Two-segment startup validation requires a connected ServoService."))
        else:
            checks.append(_ok("servo_service", "Servo Service", "ServoService is connected for all-8 telemetry capture."))
        if expected_ids != [1, 2, 3, 4, 5, 6, 7, 8] or commanded_ids != expected_ids:
            checks.append(
                _blocked(
                    "servo_ids",
                    "All-8 Servo IDs",
                    f"Expected/commanded IDs must both be [1,2,3,4,5,6,7,8]; resolved expected={expected_ids}, commanded={commanded_ids}.",
                )
            )
        else:
            checks.append(_ok("servo_ids", "All-8 Servo IDs", "All 8 expected/commanded servo IDs are present in the operating context."))
        segments = dict(operating_context.metadata().get("segments", {}) or {})
        segment_a = dict(segments.get("segment_a", {}) or {})
        segment_b = dict(segments.get("segment_b", {}) or {})
        if [int(value) for value in segment_a.get("servo_ids", [])] == [1, 2, 3, 4]:
            checks.append(_ok("segment_a", "Segment A", "Segment A/proximal uses servos [1,2,3,4] with configured pairs."))
        else:
            checks.append(_blocked("segment_a", "Segment A", f"Segment A/proximal must use servos [1,2,3,4]; resolved {segment_a.get('servo_ids')}."))
        if [int(value) for value in segment_b.get("servo_ids", [])] == [5, 6, 7, 8]:
            checks.append(_ok("segment_b", "Segment B", "Segment B/distal uses servos [5,6,7,8] with configured pairs."))
        else:
            checks.append(_blocked("segment_b", "Segment B", f"Segment B/distal must use servos [5,6,7,8]; resolved {segment_b.get('servo_ids')}."))
        checks.append(
            _info(
                "telemetry_capture",
                "Telemetry Capture",
                "Live position freshness is checked at each stage. Missing positions block capture; missing current estimates warn by default.",
            )
        )
        if tracker_ready:
            checks.append(_ok("tracking", "Tracking", f"Tracker is available via {backend_name}. Pose snapshots will be recorded when exposed by the backend."))
        else:
            checks.append(
                _warning(
                    "tracking",
                    "Tracking",
                    "Tracker is optional for this workflow. The run remains servo/current-only and lower-trust for pose if no tracker snapshot is available.",
                )
            )
        checks.append(
            _info(
                "motion_scope",
                "Motion Scope",
                "This workflow records manual all-8 startup stages and does not command automatic two-segment pretension/control.",
            )
        )

    elif experiment_name == "two_segment_collect_pose_command_dataset":
        config = TwoSegmentCollectPoseDatasetConfig.from_dict(payload)
        operating_context = settings.robot.operating_context()
        expected_ids = [int(value) for value in getattr(operating_context, "expected_servo_ids", [])]
        commanded_ids = [int(value) for value in getattr(operating_context, "commanded_servo_ids", [])]
        servo_only_override = bool(
            config.dry_run
            or config.allow_servo_only_test_run
            or str(config.run_trust_mode or "").strip().lower() == "servo_only"
        )
        if operating_context.operating_mode != "dual_segment":
            checks.append(_blocked("operating_mode", "Operating Mode", TWO_SEGMENT_DATASET_BLOCK_MESSAGE))
        else:
            checks.append(_ok("operating_mode", "Operating Mode", "dual_segment two-segment dataset mode is selected."))
        checks.append(_physical_assembly_check(operating_context))
        checks.append(_baud_advisory_check(settings))
        if expected_ids != [1, 2, 3, 4, 5, 6, 7, 8] or commanded_ids != expected_ids:
            checks.append(
                _blocked(
                    "servo_ids",
                    "All-8 Servo IDs",
                    f"Expected/commanded IDs must both be [1,2,3,4,5,6,7,8]; resolved expected={expected_ids}, commanded={commanded_ids}.",
                )
            )
        else:
            checks.append(_ok("servo_ids", "All-8 Servo IDs", "All 8 expected/commanded servo IDs are present."))
        if not servo_connected and not config.dry_run:
            checks.append(_blocked("servo_service", "Servo Service", "Live two-segment dataset collection requires a connected ServoService."))
        elif servo_connected:
            checks.append(_ok("servo_service", "Servo Service", "ServoService is connected for all-8 telemetry/command checks."))
        else:
            checks.append(_warning("servo_service", "Servo Service", "Dry-run mode can resolve commands without connected hardware."))
        if float(config.max_segment_displacement_mm) <= 0.0 and config.schedule_type != "zero":
            checks.append(_blocked("command_limits", "Command Limits", "max_segment_displacement_mm must be greater than zero unless using schedule_type=zero."))
        elif int(config.max_tick_delta_from_startup) <= 0:
            checks.append(_blocked("command_limits", "Command Limits", "max_tick_delta_from_startup must be greater than zero."))
        else:
            checks.append(
                _ok(
                    "command_limits",
                    "Command Limits",
                    f"Commands are bounded to <= {float(config.max_segment_displacement_mm):.3f} mm/segment and <= {int(config.max_tick_delta_from_startup)} ticks from startup.",
                )
            )
        if servo_calibration_summary is None:
            if servo_only_override:
                checks.append(_warning("startup_artifact", "Startup Artifact", "Startup artifact state is unavailable; servo-only/dry-run output will be marked not trainable."))
            else:
                checks.append(_blocked("startup_artifact", "Startup Artifact", "Trusted two-segment dataset collection requires an accepted all-8 startup artifact."))
        else:
            startup = servo_calibration_summary.pretension_source_summary(expected_ids)
            if startup.accepted and startup.usable:
                checks.append(_ok("startup_artifact", "Startup Artifact", startup.message))
            elif servo_only_override:
                checks.append(_warning("startup_artifact", "Startup Artifact", startup.message + " Servo-only/dry-run output will be marked not trainable."))
            else:
                checks.append(_blocked("startup_artifact", "Startup Artifact", startup.message))
        if tracker_ready:
            role_resolution = resolve_two_segment_tracking_roles(
                snapshot=tracking_snapshot,
                role_configs=getattr(settings.registration, "two_segment_tracking_roles", {}),
                require_model_training_roles=not servo_only_override,
            )
            missing_required = list(role_resolution.get("missing_required_roles", []) or [])
            if missing_required and not servo_only_override:
                checks.append(
                    _blocked(
                        "tracking_roles",
                        "Two-Segment Pose Roles",
                        f"Missing required two-segment pose role(s): {missing_required}. Configure/live-track distal_tip before a training-intended run.",
                    )
                )
            elif missing_required:
                checks.append(
                    _warning(
                        "tracking_roles",
                        "Two-Segment Pose Roles",
                        f"Missing required role(s) {missing_required}; servo-only/dry-run output will be marked not trainable.",
                    )
                )
            else:
                checks.append(_ok("tracking_roles", "Two-Segment Pose Roles", "Required two-segment pose roles are live and fresh."))
            role_rows = two_segment_role_readiness_rows(
                snapshot=tracking_snapshot,
                role_configs=getattr(settings.registration, "two_segment_tracking_roles", {}),
            )
            role_text = "; ".join(
                f"{row['role']}={row['tool_id']}:{row['status']}"
                for row in role_rows
            )
            checks.append(_ok("tracking", "Tracking", f"Tracker is available via {backend_name}. Roles: {role_text}"))
        elif servo_only_override:
            checks.append(
                _warning(
                    "tracking",
                    "Tracking",
                    "No tracker labels are required for explicit servo-only/dry-run collection. The run will be marked not valid for two-segment model training.",
                )
            )
        else:
            checks.append(_blocked("tracking", "Tracking", "Trusted two-segment model-training data requires tracker pose labels."))
        checks.append(
            _info(
                "scope",
                "Scope",
                "This experiment records structured two-segment command/pose data only; no kinematics, control, chasing, or automatic pretension is implemented.",
            )
        )

    elif experiment_name == "two_segment_repeatability":
        operating_context = settings.robot.operating_context()
        if operating_context.operating_mode != "dual_segment":
            checks.append(_blocked("operating_mode", "Operating Mode", DUAL_SEGMENT_BLOCK_MESSAGE))
        else:
            checks.append(_ok("operating_mode", "Operating Mode", "dual_segment two-segment repeatability is selected."))
        checks.append(_physical_assembly_check(operating_context))
        checks.append(_baud_advisory_check(settings))
        if not servo_connected:
            checks.append(_blocked("servo_service", "Servo Service", "Two-segment repeatability requires a connected ServoService."))
        else:
            checks.append(_ok("servo_service", "Servo Service", "ServoService is connected for all-8 telemetry."))
        if not tracker_ready:
            checks.append(
                _warning(
                    "tracking",
                    "Tracking",
                    "Tracker is unavailable. The run will execute but distal/intermediate scatter cannot be computed.",
                )
            )
        else:
            checks.append(_ok("tracking", "Tracking", "Tracker is available for distal/intermediate scatter."))
        checks.append(
            _info(
                "scope",
                "Scope",
                "Open-loop repeatability: bottom/top tendon commands; no closed-loop control. Output reports scatter, not validated control accuracy.",
            )
        )

    elif experiment_name == "pretension_validation":
        config = PretensionValidationExperimentConfig.from_dict(payload)
        if not servo_connected:
            checks.append(_blocked("servo_service", "Servo Service", "Pretension validation requires a connected servo service."))
        else:
            checks.append(_ok("servo_service", "Servo Service", "Pretension validation can use the connected servo service."))
        operating_context = settings.robot.operating_context()
        configured_ids = [int(value) for value in operating_context.all_configured_servo_ids]
        active_ids = [int(value) for value in operating_context.active_segment_servo_ids]
        staged_ids = [int(value) for value in (config.servo_ids or [])]
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
        if config.mode in {"single_segment_staged", "single_segment_characterization", "staged", "four_servo_staged"}:
            selected_ids = staged_ids or active_ids
            if operating_context.operating_mode != "single_segment":
                message = (
                    "dual_segment currently supports all-8 readiness and manual startup capture. "
                    "Automatic two-segment pretension/control is not implemented yet."
                    if operating_context.operating_mode == "dual_segment"
                    else (
                        "Automatic staged pretension currently requires operating_mode=single_segment. "
                        "Select Segment A or Segment B."
                    )
                )
                checks.append(
                    _blocked(
                        "active_segment",
                        "Active Segment",
                        message,
                    )
                )
            elif len(selected_ids) != 4:
                checks.append(
                    _blocked(
                        "active_segment",
                        "Active Segment",
                        f"Staged pretension requires exactly 4 active segment servos. Active segment {settings.robot.active_segment_label()} has {selected_ids}.",
                    )
                )
            else:
                checks.append(
                    _ok(
                        "active_segment",
                        "Active Segment",
                        f"Staged pretension will use {settings.robot.active_segment_label()} ({settings.robot.active_segment_key()}): {selected_ids}.",
                    )
                )
            checks.extend(
                _selected_segment_truth_checks(
                    settings=settings,
                    servo_calibration_summary=servo_calibration_summary,
                    project_root=project_root,
                    servo_connected=servo_connected,
                    require_startup=False,
                    strict_startup=False,
                    allow_calibration_creation=True,
                    label_prefix="Pretension Validation",
                )
            )
        pretension_no_tracker_allowed = bool(
            (bool(config.allow_no_tracker_test_run)
            or bool(config.allow_current_only_when_tracker_missing)
            or str(config.run_trust_mode or "").strip().lower() in {"current_only", "servo_only", "lower_trust"})
            and not tracker_ready
        )
        if bool(config.include_tracker_displacement):
            if tracker_ready:
                checks.append(
                    _ok(
                        "tracking_state",
                        "Tracking",
                        "Tracker displacement will be sampled alongside the pretension trace when frames remain fresh.",
                    )
                )
            elif pretension_no_tracker_allowed:
                checks.append(
                    _warning(
                        "tracking_state",
                        "Tracking",
                        "Tracker is not connected. This run is explicitly allowed as current-only/lower-trust pretension testing. "
                        "Tip centering and centered-startup claims will not be produced.",
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
            if pretension_no_tracker_allowed:
                checks.append(
                    _warning(
                        "runtime_tip",
                        "Runtime Tip Policy",
                        "Runtime tip is unavailable. Pretension output will be marked current_only/lower_trust and not thesis-trusted.",
                    )
                )
            else:
                runtime_tip_policy = evaluate_runtime_tip_trust(
                    snapshot=tracking_snapshot,
                    workflow=WORKFLOW_PRETENSION_VALIDATION,
                    allow_lower_trust=True,
                )
                if runtime_tip_policy.thesis_trusted:
                    checks.append(_ok("runtime_tip", "Runtime Tip Policy", runtime_tip_policy.status_message))
                elif runtime_tip_policy.allowed_for_workflow:
                    checks.append(
                        _warning(
                            "runtime_tip",
                            "Runtime Tip Policy",
                            f"{runtime_tip_policy.status_message} Pretension validation will record this as {runtime_tip_policy.trust_label}.",
                        )
                    )
                elif str(config.mode).strip().lower() in {"single_segment_staged", "advanced_4servo", "staged"} and bool(config.enable_tip_centering):
                    checks.append(
                        _blocked(
                            "runtime_tip",
                            "Runtime Tip Policy",
                            f"Tip centering requires an allowed runtime tip policy outcome. Mode={runtime_tip_policy.mode}, trust={runtime_tip_policy.trust_label}, reasons={runtime_tip_policy.reasons or ['policy_not_allowed']}.",
                        )
                    )
                else:
                    checks.append(
                        _warning(
                            "runtime_tip",
                            "Runtime Tip Policy",
                            f"Runtime tip is not available for robot-frame tip metrics. Mode={runtime_tip_policy.mode}, trust={runtime_tip_policy.trust_label}.",
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

    elif experiment_name == "servo_tracker_sync_validation":
        config = ServoTrackerSyncValidationConfig.from_dict(payload)
        configured_backend = str(settings.serial.tracker_backend or "").strip().lower()
        selected_backend = str(
            tracking_snapshot.selected_backend_name or tracking_snapshot.backend_identity or ""
        ).strip().lower()
        if bool(settings.runtime.mock_mode):
            checks.append(
                _blocked(
                    "mock_mode",
                    "Runtime Mode",
                    "Servo-tracker sync validation is blocked in mock mode. Use the live Python NDI backend and live servo connection.",
                )
            )
        else:
            checks.append(
                _ok(
                    "runtime_mode",
                    "Runtime Mode",
                    "Live runtime mode is enabled for a meaningful motion-and-sync validation run.",
                )
            )
        if configured_backend not in {"ndi", ""}:
            checks.append(
                _blocked(
                    "configured_backend",
                    "Configured Backend",
                    f"Configured tracker backend is '{configured_backend}'. This validation targets the Python NDI backend path only.",
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
                    "The active tracker backend is the legacy bridge path. This validation should run against the Python NDI backend instead.",
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
                    "Live tracking is not currently streaming. The experiment can start the backend, but the saved run is only meaningful if the Python NDI backend connects successfully.",
                )
            )
        if not servo_connected:
            checks.append(
                _blocked(
                    "servo_connection",
                    "Servo Connection",
                    "Servo-tracker sync validation requires a connected servo service.",
                )
            )
        else:
            checks.append(
                _ok(
                    "servo_connection",
                    "Servo Connection",
                    "Servo service is connected for command and telemetry logging.",
                )
            )
        if not config.servo_ids:
            checks.append(
                _blocked(
                    "servo_ids",
                    "Servo IDs",
                    "At least one servo ID is required.",
                )
            )
        else:
            checks.append(
                _ok(
                    "servo_ids",
                    "Servo IDs",
                    f"Will run the scripted motion on servo IDs {', '.join(str(value) for value in config.servo_ids)}.",
                )
            )
        checks.append(
            _info(
                "motion_scope",
                "Motion Scope",
                (
                    f"Will run {config.motion_mode.replace('_', ' ')} motion for {float(config.run_duration_s):.1f} s "
                    f"after {float(config.warmup_duration_s):.1f} s warmup, using amplitude {int(config.command_amplitude_ticks)} ticks "
                    f"and {float(config.step_period_s):.3f} s command cadence."
                ),
            )
        )
        checks.append(
            _ok(
                "scientific_framing",
                "Scientific Framing",
                "This experiment validates host-time alignment and logging usability during motion. It does not claim closed-loop control quality or hardware clock synchronization.",
            )
        )

    elif experiment_name == "penprobe_chasing_demo":
        config = PenprobeChasingDemoConfig.from_dict(payload)
        operating_context = settings.robot.operating_context()
        active_ids = [int(value) for value in operating_context.active_segment_servo_ids]
        if operating_context.operating_mode != "single_segment":
            checks.append(
                _blocked(
                    "operating_mode",
                    "Operating Mode",
                    f"Penprobe chasing demo v1 requires single_segment mode. Current mode is {operating_context.operating_mode}.",
                )
            )
        elif len(active_ids) != 4:
            checks.append(
                _blocked(
                    "active_segment",
                    "Active Segment",
                    f"Penprobe chasing demo requires exactly four active segment servos; found {active_ids}.",
                )
            )
        else:
            checks.append(
                _ok(
                    "active_segment",
                    "Active Segment",
                    f"Demo will use {operating_context.active_segment_label} ({operating_context.active_segment_key}): {active_ids}; pairs {operating_context.active_pairs}.",
                )
            )
        if not servo_connected:
            checks.append(_blocked("servo_service", "Servo Service", "Penprobe chasing requires a connected ServoService."))
        else:
            checks.append(_ok("servo_service", "Servo Service", "ServoService is connected."))
        checks.extend(
            _selected_segment_truth_checks(
                settings=settings,
                servo_calibration_summary=servo_calibration_summary,
                project_root=project_root,
                servo_connected=servo_connected,
                require_startup=True,
                strict_startup=True,
                allow_calibration_creation=False,
                label_prefix="Penprobe",
            )
        )
        if int(config.max_tick_delta_from_startup) > int(config.hard_max_tick_delta_from_startup):
            checks.append(
                _blocked(
                    "tick_cap",
                    "Tick Cap",
                    f"max_tick_delta_from_startup={int(config.max_tick_delta_from_startup)} exceeds hard cap {int(config.hard_max_tick_delta_from_startup)}.",
                )
            )
        else:
            checks.append(
                _ok(
                    "tick_cap",
                    "Tick Cap",
                    f"Commands are capped to +/-{int(config.max_tick_delta_from_startup)} ticks from the accepted startup reference.",
                )
            )
        if servo_calibration_summary is None:
            checks.append(_blocked("pretension", "Startup Reference", "Pretension/startup artifact state is unavailable."))
        else:
            checks.append(_pretension_artifact_check(servo_ids=active_ids, servo_calibration_summary=servo_calibration_summary))
        checks.append(_strict_tool_gate_check(tool_id="0A", snapshot=tracking_snapshot, max_tracker_age_s=float(config.loop_period_s) * 4.0))
        checks.append(_strict_tool_gate_check(tool_id="0B", snapshot=tracking_snapshot, max_tracker_age_s=float(config.loop_period_s) * 4.0))
        if tracking_snapshot.registration_state == "loaded" and tracking_snapshot.T_robot_aurora is not None:
            checks.append(_ok("registration", "Robot Frame", "Accepted base registration is loaded; 0A and 0B will be compared in robot frame."))
        else:
            checks.append(
                _blocked(
                    "registration",
                    "Robot Frame",
                    f"Penprobe chasing requires an accepted robot-frame transform. Registration state is {tracking_snapshot.registration_state}.",
                )
            )
        try:
            _validate_legacy_mapping_config(config, project_root)
            if config.mapping_mode == "legacy_polynomial_workspace":
                checks.append(
                    _warning(
                        "mapping",
                        "Mapping Mode",
                        "Legacy polynomial workspace files are present, but v1 still runs through the bounded paired-command safety envelope.",
                    )
                )
            elif str(config.mapping_mode).strip().lower() == MAPPING_AGGRESSIVE_TICK_DEMO:
                checks.append(
                    _ok(
                        "mapping",
                        "Mapping Mode",
                        "aggressive_tick_demo: direct goal writes; per-axis max_tick_step_per_cycle; bypasses paired XY L2 vector cap.",
                    )
                )
            elif str(config.mapping_mode).strip().lower() == MAPPING_PAIRED_XY_PROPORTIONAL:
                checks.append(
                    _ok(
                        "mapping",
                        "Mapping Mode",
                        "Using bounded paired_xy_proportional mapping (displacement command envelope).",
                    )
                )
            else:
                checks.append(_ok("mapping", "Mapping Mode", f"Using mapping_mode={config.mapping_mode!r}."))
        except Exception as exc:
            checks.append(_blocked("mapping", "Mapping Mode", str(exc)))
        checks.append(
            _info(
                "control_scope",
                "Control Scope",
                "MVP demo controls XY only using 0A coil-as-tip and 0B tool origin. Z is logged but does not drive commands.",
            )
        )

    else:
        checks.append(_blocked("experiment", "Experiment", f"Unsupported experiment selection: {experiment_name}"))

    return _finalize_report(
        experiment_name=experiment_name,
        planned_output_dir=planned_output_dir,
        checks=checks,
        overwrite_targets=overwrite_targets,
        planned_output_state=planned_output_state,
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


def _strict_tool_gate_check(*, tool_id: str, snapshot, max_tracker_age_s: float) -> PreflightCheck:
    tool_key = str(tool_id or "0A").upper()
    tool = snapshot.tools.get(tool_key)
    reasons: list[str] = []
    if snapshot.tracker_data_stale:
        reasons.append("tracker data is stale")
    if snapshot.tracker_data_age_s is None:
        reasons.append("tracker data age is unknown")
    elif float(snapshot.tracker_data_age_s) > float(max_tracker_age_s):
        reasons.append(
            f"tracker data age {float(snapshot.tracker_data_age_s):.3f}s exceeds {float(max_tracker_age_s):.3f}s"
        )
    if tool is None:
        reasons.append(f"tool {tool_key} is missing")
    else:
        if not tool.present:
            reasons.append(f"tool {tool_key} is not present")
        if tool.valid is False:
            reasons.append(f"tool {tool_key} is invalid")
        if tool.translation_mm is None:
            reasons.append(f"tool {tool_key} has no translation")
        if str(tool.tracking_state).lower() in {"invalid", "missing", "out_of_volume"}:
            reasons.append(f"tool {tool_key} state is {tool.tracking_state}")
    if reasons:
        return _blocked(
            "tool_gate",
            "Tracker Capture Gate",
            "Capture gate is blocked: " + "; ".join(reasons),
        )
    return _ok(
        "tool_gate",
        "Tracker Capture Gate",
        f"Tool {tool_key} is visible and tracker data age is within {float(max_tracker_age_s):.3f}s.",
    )


def _selected_segment_truth_checks(
    *,
    settings,
    servo_calibration_summary,
    project_root: Path,
    servo_connected: bool,
    require_startup: bool,
    strict_startup: bool,
    allow_calibration_creation: bool,
    label_prefix: str,
) -> list[PreflightCheck]:
    context = settings.robot.operating_context()
    if context.operating_mode != "single_segment":
        return []
    expected_ids = [int(value) for value in context.active_segment_servo_ids or context.expected_servo_ids]
    sign_mapping = _latest_sign_mapping_check(
        project_root=project_root,
        active_segment_key=str(context.active_segment_key or ""),
        expected_servo_ids=expected_ids,
    )
    readiness = evaluate_selected_segment_readiness(
        operating_mode=context.operating_mode,
        active_segment_key=context.active_segment_key,
        active_segment_label=context.active_segment_label,
        expected_servo_ids=expected_ids,
        calibration_summary=servo_calibration_summary,
        mock_mode=bool(settings.runtime.mock_mode),
        servo_connected=bool(servo_connected),
        sign_mapping=sign_mapping,
    )
    checks: list[PreflightCheck] = []
    if bool(settings.runtime.mock_mode):
        checks.append(
            _warning(
                "mock_mode",
                "Runtime Mode",
                "mock_mode=true. Runs may execute only as mock/debug output and must remain non-thesis/non-training.",
            )
        )
    neutral = readiness.neutral_safe_calibration
    neutral_label = f"{label_prefix} Neutral/Safe Calibration"
    if neutral.ready:
        checks.append(_ok("neutral_safe_calibration", neutral_label, neutral.message))
    elif allow_calibration_creation and not neutral.mismatch_reasons and not neutral.is_mock:
        checks.append(
            _warning(
                "neutral_safe_calibration",
                neutral_label,
                neutral.message
                + " Raw tiny jog is allowed for bring-up; calibrated experiments remain blocked until bounds are captured.",
            )
        )
    elif neutral.is_mock and bool(settings.runtime.mock_mode):
        checks.append(
            _warning(
                "neutral_safe_calibration",
                neutral_label,
                neutral.message + " This run can only be mock/debug and is not hardware-valid.",
            )
        )
    else:
        checks.append(_blocked("neutral_safe_calibration", neutral_label, neutral.message))
    startup = readiness.startup_pretension
    if require_startup:
        startup_label = f"{label_prefix} Startup Reference"
        if startup.ready:
            checks.append(_ok("startup_reference", startup_label, startup.message))
        elif strict_startup:
            checks.append(_blocked("startup_reference", startup_label, startup.message))
        else:
            checks.append(
                _warning(
                    "startup_reference",
                    startup_label,
                    startup.message + " This explicit debug/servo-only run must not be used as thesis/training data.",
                )
            )
    elif not startup.ready:
        checks.append(
            _info(
                "startup_reference",
                f"{label_prefix} Startup Reference",
                startup.message + " Pretension validation may proceed only as calibration/startup creation.",
            )
        )
    if sign_mapping is not None and not sign_mapping.confirmed:
        checks.append(
            _warning(
                "servo_sign_mapping",
                "Servo Sign/Mapping",
                sign_mapping.message,
            )
        )
    return checks


def _latest_sign_mapping_check(
    *,
    project_root: Path,
    active_segment_key: str,
    expected_servo_ids: list[int],
):
    try:
        repo = ServoMappingCheckRepository(project_root / "data" / "calibration" / "servo_mapping_checks")
        return repo.latest_for_segment(
            active_segment_key=str(active_segment_key or ""),
            expected_servo_ids=[int(value) for value in expected_servo_ids],
        )
    except Exception:
        return None


def _pretension_artifact_check(*, servo_ids: list[int], servo_calibration_summary) -> PreflightCheck:
    if servo_calibration_summary is None:
        return _blocked(
            "pretension",
            "Pretension State",
            "Pretension artifact state is unavailable. Refresh servo state before running repeatability.",
        )
    if not servo_calibration_summary.exists or not servo_calibration_summary.compatible:
        return _blocked(
            "pretension",
            "Pretension State",
            f"Servo calibration artifact is not ready: {servo_calibration_summary.message}",
        )
    source_summary = servo_calibration_summary.pretension_source_summary([int(value) for value in servo_ids])
    if not source_summary.accepted or not source_summary.usable:
        return _blocked(
            "pretension",
            "Pretension State",
            source_summary.message,
        )
    return _ok(
        "pretension",
        "Pretension State",
        source_summary.message,
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


def _baud_advisory_check(settings) -> PreflightCheck:
    """Soft-warn when 8-servo work runs at non-1 Mbps baud rates.

    XC330 + OpenRB-150 supports up to 1 Mbps. Eight-servo work (`dual_segment`,
    `parallel_single`) benefits substantially from the higher baud. This check
    never blocks: it returns INFO at 1 Mbps, WARNING below 1 Mbps for 8-servo
    modes, and INFO for 1-4 servo modes regardless.
    """
    baud = int(getattr(getattr(settings, "serial", None), "baudrate", 57600) or 57600)
    mode = str(getattr(getattr(settings, "robot", None), "operating_mode", lambda: "single_segment")())
    eight_servo_mode = mode in {"dual_segment", "parallel_single"}
    if baud >= 1_000_000:
        return _ok(
            "baud_rate",
            "Bus Baud",
            f"DYNAMIXEL bus at {baud} bps. Confirm every servo is reflashed at this baud (DYNAMIXEL Wizard).",
        )
    if eight_servo_mode:
        return _warning(
            "baud_rate",
            "Bus Baud",
            (
                f"DYNAMIXEL bus is at {baud} bps. Eight-servo work is recommended at 1 000 000 bps; "
                "reflash every servo (DYNAMIXEL Wizard) and update `baudrate:` in system.yaml/system.local.yaml."
            ),
        )
    return _info(
        "baud_rate",
        "Bus Baud",
        f"DYNAMIXEL bus at {baud} bps (single-segment use).",
    )


def _physical_assembly_check(operating_context) -> PreflightCheck:
    """Inspect the resolved bottom/top physical-role assignment for two-segment runs."""
    assembly = dict(getattr(operating_context, "physical_assembly", {}) or {})
    issues = list(getattr(operating_context, "physical_assembly_issues", []) or [])
    bottom_key = assembly.get("bottom_segment_key") or ""
    top_key = assembly.get("top_segment_key") or ""
    bottom_label = assembly.get("bottom_segment_label") or bottom_key or "?"
    top_label = assembly.get("top_segment_label") or top_key or "?"
    bottom_ids = list(assembly.get("bottom_servo_ids") or [])
    top_ids = list(assembly.get("top_servo_ids") or [])
    if issues:
        return _blocked(
            "physical_assembly",
            "Bottom/Top Assignment",
            "Invalid bottom/top physical assembly: " + "; ".join(issues),
        )
    if not bottom_key or not top_key:
        return _warning(
            "physical_assembly",
            "Bottom/Top Assignment",
            "Bottom/top physical assembly is not configured. Configure `physical_assembly` in the robot config.",
        )
    return _ok(
        "physical_assembly",
        "Bottom/Top Assignment",
        f"Bottom={bottom_label} ({bottom_ids}), Top={top_label} ({top_ids}).",
    )


def _finalize_report(
    *,
    experiment_name: str,
    planned_output_dir: Path,
    checks: list[PreflightCheck],
    overwrite_targets: list[str],
    planned_output_state: str = PLANNED_OUTPUT_NOT_CREATED_YET,
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
        planned_output_state=str(planned_output_state),
    )


def classify_planned_output_state(
    *,
    planned_output_dir: Path,
    active_run_output_dir: Path | None,
) -> str:
    """Classify a planned output directory in its run-lifecycle state.

    States:
    - ``not_created_yet``: directory does not yet exist on disk.
    - ``available``: directory does not exist; safe to allocate.
    - ``owned_by_current_run``: directory exists and matches the in-flight run's output dir.
    - ``existing_previous_run``: directory exists and belongs to a prior run.
    - ``conflict``: directory exists in an unexpected state (e.g., not a directory).
    """
    planned = Path(planned_output_dir)
    if active_run_output_dir is not None:
        try:
            planned_resolved = planned.resolve()
            active_resolved = Path(active_run_output_dir).resolve()
        except Exception:
            planned_resolved = planned
            active_resolved = Path(active_run_output_dir)
        if planned_resolved == active_resolved:
            return PLANNED_OUTPUT_OWNED_BY_CURRENT_RUN
    if not planned.exists():
        return PLANNED_OUTPUT_NOT_CREATED_YET
    if planned.is_dir():
        return PLANNED_OUTPUT_EXISTING_PREVIOUS_RUN
    return PLANNED_OUTPUT_CONFLICT
