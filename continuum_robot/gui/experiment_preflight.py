"""Pure-Python experiment preflight validation for the GUI workspace."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import Any

from continuum_robot.experiments.critical_experiments import (
    GridDefinitionConfig,
    PivotCalibrationConfig,
    RepeatabilityDatasetConfig,
)


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
            return "Run blocked. Resolve the red items before starting."
        if self.overall_status == RUN_WARNING:
            return "Ready with warnings. Review the amber items before starting."
        return "Ready to run. All required checks passed."


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
                message=f"Config could not be parsed. Fix the YAML error and refresh preflight. Details: {config_error}",
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
                        "Start tracking from the System or Tracking tab and confirm live frames before running."
                    )
                )
            ),
        )
    )

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
        config = RepeatabilityDatasetConfig.from_dict(payload)
        expected_dims = len(settings.robot.tendon_to_servo or settings.robot.servo_ids)
        target_lengths = {len(point) for point in config.schedule.target_points_cm}
        if not config.schedule.target_points_cm:
            checks.append(_blocked("config_targets", "Schedule", "Repeatability schedule has no target points."))
        elif target_lengths != {expected_dims}:
            checks.append(
                _blocked(
                    "dimensions",
                    "Dimensionality",
                    f"Repeatability targets do not match the configured robot. "
                    f"Expected {expected_dims} tendon values per point, found {sorted(target_lengths)}. "
                    "Fix schedule.target_points_cm before running.",
                )
            )
        else:
            checks.append(
                _ok(
                    "dimensions",
                    "Dimensionality",
                    f"Repeatability targets match the configured {expected_dims}-tendon robot.",
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
                    "but commanded motor values may be missing."
                )
            )
            checks.append(PreflightCheck("neutral_setpoints", "Neutral Setpoints", status, message))
        else:
            status = PREFLIGHT_OK if neutral_count == expected_dims else PREFLIGHT_BLOCKED
            message = (
                f"Neutral setpoints are available for all {expected_dims} tendons."
                if neutral_count == expected_dims
                else "Live repeatability requires saved neutral setpoints for every tendon. Capture and save neutral first."
            )
            checks.append(PreflightCheck("neutral_setpoints", "Neutral Setpoints", status, message))

        if registration_path.exists():
            checks.append(_ok("registration", "Registration", f"Registration file found: {registration_path}"))
        else:
            checks.append(
                _warning(
                    "registration",
                    "Registration",
                    "Registration file is missing. The run can continue, but only tracker-frame pose will be available until registration is saved.",
                )
            )

    elif experiment_name == "aurora_grid_accuracy":
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
                    f"Grid definition is {dims} with {config.spacing_mm:.2f} mm spacing.",
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
                    "Truth grid will be generated from spacing and dimensions in the config.",
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

        if config.truth_frame == "robot":
            if registration_path.exists():
                checks.append(_ok("registration", "Registration", f"Registration file found: {registration_path}"))
            else:
                checks.append(
                    _blocked(
                        "registration",
                        "Registration",
                        "Robot-frame grid accuracy requires registration. Save a registration file first or change truth_frame to tracker.",
                    )
                )
        else:
            checks.append(_info("registration", "Registration", "Tracker-frame truth does not require registration."))

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

    elif experiment_name == "pivot_calibration":
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
