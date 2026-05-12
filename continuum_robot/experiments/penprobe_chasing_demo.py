"""MVP one-segment 0A coil-origin chasing demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from continuum_robot.experiments.framework import BaseExperiment, ExperimentHardwareRequirements, ExperimentSession
from continuum_robot.experiments.schemas import ExperimentTimeseriesSample
from continuum_robot.servos.displacement_mapper import TendonDisplacementMapper
from continuum_robot.servos.segment_readiness import evaluate_selected_segment_readiness
from continuum_robot.servos.servo_service import RAW_POSITION_MAX_TICK, RAW_POSITION_MIN_TICK, SINGLE_SEGMENT_WORKFLOW_EXPERIMENT


MAPPING_PAIRED_XY_PROPORTIONAL = "paired_xy_proportional"
MAPPING_LEGACY_POLYNOMIAL_WORKSPACE = "legacy_polynomial_workspace"
SUPPORTED_MAPPING_MODES = {MAPPING_PAIRED_XY_PROPORTIONAL, MAPPING_LEGACY_POLYNOMIAL_WORKSPACE}


@dataclass
class PenprobeChasingDemoConfig:
    """Operator config for the single-segment penprobe chasing MVP."""

    loop_period_s: float = 0.075
    max_duration_s: float = 30.0
    max_iterations: int = 400
    max_tick_delta_from_startup: int = 500
    hard_max_tick_delta_from_startup: int = 500
    max_step_ticks: int = 10
    max_target_radius_mm: float = 45.0
    target_deadband_mm: float = 1.0
    displacement_gain_cm_per_mm: float = 0.002
    x_axis_sign: int = 1
    y_axis_sign: int = 1
    mapping_mode: str = MAPPING_PAIRED_XY_PROPORTIONAL
    legacy_polynomial_workspace: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PenprobeChasingDemoConfig":
        payload = dict(payload or {})
        mode = str(payload.get("mapping_mode", MAPPING_PAIRED_XY_PROPORTIONAL) or MAPPING_PAIRED_XY_PROPORTIONAL)
        mode = mode.strip().lower()
        if mode not in SUPPORTED_MAPPING_MODES:
            mode = MAPPING_PAIRED_XY_PROPORTIONAL
        return cls(
            loop_period_s=max(0.02, float(payload.get("loop_period_s", 0.075))),
            max_duration_s=max(0.0, float(payload.get("max_duration_s", 30.0))),
            max_iterations=max(1, int(payload.get("max_iterations", 400))),
            max_tick_delta_from_startup=max(0, int(payload.get("max_tick_delta_from_startup", 500))),
            hard_max_tick_delta_from_startup=max(0, int(payload.get("hard_max_tick_delta_from_startup", 500))),
            max_step_ticks=max(1, int(payload.get("max_step_ticks", 10))),
            max_target_radius_mm=max(0.0, float(payload.get("max_target_radius_mm", 45.0))),
            target_deadband_mm=max(0.0, float(payload.get("target_deadband_mm", 1.0))),
            displacement_gain_cm_per_mm=max(0.0, float(payload.get("displacement_gain_cm_per_mm", 0.002))),
            x_axis_sign=(1 if int(payload.get("x_axis_sign", 1)) >= 0 else -1),
            y_axis_sign=(1 if int(payload.get("y_axis_sign", 1)) >= 0 else -1),
            mapping_mode=mode,
            legacy_polynomial_workspace={
                str(key): str(value)
                for key, value in dict(payload.get("legacy_polynomial_workspace", {}) or {}).items()
                if str(value).strip()
            },
        )


@dataclass
class ChasingPose:
    """One robot-frame tool point used by the demo controller."""

    tool_id: str
    position_mm: list[float]
    tracking_state: str
    frame_number: int | None


class PenprobeChasingDemoExperiment(BaseExperiment):
    """Move one active segment's 0A coil origin toward the live 0B tool-origin target."""

    name = "penprobe_chasing_demo"
    description = (
        "MVP demo: 0A coil-origin chasing toward a live 0B tool-origin XY target. "
        "This is not physical tip chasing unless a validated T_robot_tip transform is used."
    )
    hardware_requirements = ExperimentHardwareRequirements(
        tracking_required=True,
        servo_required=True,
        registration_required=True,
        mock_compatible=True,
    )

    def __init__(self, config: PenprobeChasingDemoConfig) -> None:
        super().__init__(config=config)
        self._startup_reference_by_servo: dict[int, int] = {}
        self._current_tick_delta_by_servo: dict[int, int] = {}
        self._stop_reason = "not_started"
        self._mapping_mode_used = str(config.mapping_mode)
        self._mapping_warnings: list[str] = []

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "PenprobeChasingDemoExperiment":
        return cls(config=PenprobeChasingDemoConfig.from_dict(payload))

    def precheck(self, session: ExperimentSession) -> None:
        context = session.context.settings.robot.operating_context()
        if context.operating_mode != "single_segment":
            raise RuntimeError(
                "penprobe_chasing_demo only supports operating_mode=single_segment in v1. "
                f"Current mode is {context.operating_mode}."
            )
        servo_ids = [int(value) for value in context.active_segment_servo_ids]
        if len(servo_ids) != 4:
            raise RuntimeError(f"penprobe_chasing_demo requires exactly 4 active segment servos; found {servo_ids}.")
        if session.context.servo_service is None or not getattr(session.context.servo_service, "is_connected", False):
            raise RuntimeError("Servo service is not connected.")
        if session.context.tracking_service is None:
            raise RuntimeError("Tracking service is not available.")
        if int(self.config.max_tick_delta_from_startup) > int(self.config.hard_max_tick_delta_from_startup):
            raise RuntimeError(
                "max_tick_delta_from_startup exceeds the configured hard cap: "
                f"{self.config.max_tick_delta_from_startup} > {self.config.hard_max_tick_delta_from_startup}."
            )
        calibration_summary = session.context.servo_service.neutral_calibration.get_calibration_summary()
        if hasattr(calibration_summary, "servo_entries"):
            readiness = evaluate_selected_segment_readiness(
                operating_mode=context.operating_mode,
                active_segment_key=context.active_segment_key,
                active_segment_label=context.active_segment_label,
                expected_servo_ids=servo_ids,
                calibration_summary=calibration_summary,
                mock_mode=bool(session.context.settings.runtime.mock_mode),
                servo_connected=True,
            )
            if not readiness.neutral_safe_calibration.ready:
                raise RuntimeError(readiness.neutral_safe_calibration.message)
        self._startup_reference_by_servo = _startup_reference_from_calibration(
            session.context.servo_service,
            servo_ids,
        )
        _validate_legacy_mapping_config(self.config, session.context.project_root)
        snapshot = session.context.tracking_service.get_snapshot()
        _extract_robot_frame_tool_pose(snapshot, "0A", max_tracker_age_s=float(self.config.loop_period_s) * 4.0)
        _extract_robot_frame_tool_pose(snapshot, "0B", max_tracker_age_s=float(self.config.loop_period_s) * 4.0)
        _validate_servo_telemetry_ready(session.context.servo_service, servo_ids)

    def execute(self, session: ExperimentSession) -> None:
        context = session.context.settings.robot.operating_context()
        servo_ids = [int(value) for value in context.active_segment_servo_ids]
        pairs = {str(key): [int(value) for value in values] for key, values in dict(context.active_pairs or {}).items()}
        mapper = TendonDisplacementMapper(
            spool_diameter_cm=float(session.context.settings.robot.spool_diameter_cm),
            ticks_per_rev=int(session.context.settings.robot.ticks_per_revolution),
        )
        self._mapping_mode_used, self._mapping_warnings = _resolve_mapping_mode(self.config, session.context.project_root)
        self._current_tick_delta_by_servo = {int(servo_id): 0 for servo_id in servo_ids}
        total = int(self.config.max_iterations)
        deadline = (
            float(session.context.monotonic_fn()) + float(self.config.max_duration_s)
            if float(self.config.max_duration_s) > 0.0
            else None
        )
        start_monotonic = float(session.context.monotonic_fn())
        self._stop_reason = "running"
        last_loop_monotonic = start_monotonic

        try:
            with session.context.servo_service.exclusive_bus_operation(
                owner="penprobe_chasing_demo",
                reason="bounded active single-segment 0A-to-0B chasing demo",
            ):
                for iteration in range(total):
                    session.raise_if_stop_requested()
                    now = float(session.context.monotonic_fn())
                    if deadline is not None and now >= deadline:
                        self._stop_reason = "max_duration_reached"
                        break
                    loop_dt_s = max(0.0, now - last_loop_monotonic)
                    last_loop_monotonic = now

                    snapshot = session.context.tracking_service.get_snapshot()
                    tip = _extract_robot_frame_tool_pose(snapshot, "0A", max_tracker_age_s=float(self.config.loop_period_s) * 4.0)
                    target = _extract_robot_frame_tool_pose(snapshot, "0B", max_tracker_age_s=float(self.config.loop_period_s) * 4.0)
                    telemetry = _validate_servo_telemetry_ready(session.context.servo_service, servo_ids)

                    desired_delta_by_servo, mapping_debug = paired_xy_proportional_tick_request(
                        tip_xy_mm=tip.position_mm[:2],
                        target_xy_mm=target.position_mm[:2],
                        servo_ids=servo_ids,
                        pairs=pairs,
                        mapper=mapper,
                        config=self.config,
                    )
                    stepped_delta_by_servo = _step_and_clamp_tick_deltas(
                        current_delta_by_servo=self._current_tick_delta_by_servo,
                        desired_delta_by_servo=desired_delta_by_servo,
                        max_step_ticks=int(self.config.max_step_ticks),
                        max_abs_delta_ticks=int(self.config.max_tick_delta_from_startup),
                    )
                    bounded_delta_by_servo, raw_bound_clips = _clamp_tick_deltas_to_startup_raw_bounds(
                        stepped_delta_by_servo,
                        startup_reference_by_servo=self._startup_reference_by_servo,
                    )
                    self._current_tick_delta_by_servo = dict(bounded_delta_by_servo)
                    commanded_ticks = {
                        int(servo_id): int(self._startup_reference_by_servo[int(servo_id)] + bounded_delta_by_servo[int(servo_id)])
                        for servo_id in servo_ids
                    }
                    tendon_displacements_cm = [
                        float(mapper.ticks_to_displacement_mm(bounded_delta_by_servo[int(servo_id)]) / 10.0)
                        for servo_id in servo_ids
                    ]
                    command_result = session.context.servo_service.command_displacement(
                        tendon_displacements_cm,
                        [int(self._startup_reference_by_servo[int(servo_id)]) for servo_id in servo_ids],
                        servo_ids,
                        motion_workflow=SINGLE_SEGMENT_WORKFLOW_EXPERIMENT,
                    )
                    measured_after = {
                        int(servo_id): (
                            int(command_result.telemetry_by_id[int(servo_id)].present_position)
                            if command_result.telemetry_by_id[int(servo_id)].present_position is not None
                            else None
                        )
                        for servo_id in servo_ids
                    }
                    error_xy = [
                        float(target.position_mm[0] - tip.position_mm[0]),
                        float(target.position_mm[1] - tip.position_mm[1]),
                    ]
                    error_norm = float(math.hypot(error_xy[0], error_xy[1]))
                    max_tick_delta_used = max(abs(int(value)) for value in bounded_delta_by_servo.values()) if bounded_delta_by_servo else 0
                    sample = _build_chasing_sample(
                        session=session,
                        snapshot=snapshot,
                        iteration=iteration,
                        tip=tip,
                        target=target,
                        error_xy_mm=error_xy,
                        error_norm_mm=error_norm,
                        requested_tick_delta_by_servo=desired_delta_by_servo,
                        commanded_tick_delta_by_servo=bounded_delta_by_servo,
                        commanded_ticks=commanded_ticks,
                        measured_ticks=measured_after,
                        tendon_displacements_cm=tendon_displacements_cm,
                        startup_reference_by_servo=self._startup_reference_by_servo,
                        loop_dt_s=loop_dt_s,
                        mapping_debug={
                            **dict(mapping_debug),
                            "raw_bound_clips": {str(k): v for k, v in raw_bound_clips.items()},
                            "servo_command_positions_by_id": {str(k): int(v) for k, v in dict(command_result.positions_by_id).items()},
                            "servo_clamp_reasons_by_id": {str(k): str(v) for k, v in dict(command_result.clamp_reasons_by_id).items()},
                        },
                        mapping_mode_used=self._mapping_mode_used,
                        max_tick_delta_used=max_tick_delta_used,
                        telemetry_by_id=telemetry,
                    )
                    session.add_sample(sample)
                    session.update_progress(iteration + 1, total, {"phase": "chasing", "error_mm": error_norm})
                    if error_norm <= float(self.config.target_deadband_mm):
                        session.set_metric("last_deadband_iteration", int(iteration))
                    sleep_s = max(0.0, float(self.config.loop_period_s) - (float(session.context.monotonic_fn()) - now))
                    if sleep_s > 0.0:
                        session.context.sleep_fn(sleep_s)
                else:
                    self._stop_reason = "max_iterations_reached"
        except Exception as exc:
            self._stop_reason = _classify_stop_reason(exc)
            self._record_metrics(session=session, context=context, servo_ids=servo_ids, pairs=pairs)
            raise

        self._record_metrics(session=session, context=context, servo_ids=servo_ids, pairs=pairs)

    def summarize(self, session: ExperimentSession) -> dict[str, Any]:
        metrics = dict(session.metrics)
        if session.samples:
            last = session.samples[-1]
            metrics["final_error_norm_mm"] = last.extra.get("xy_error_norm_mm")
            metrics["final_tip_xy_mm"] = last.extra.get("tip_xy_mm")
            metrics["final_target_xy_mm"] = last.extra.get("target_xy_mm")
        return metrics

    def write_outputs(self, session: ExperimentSession, paths, summary) -> None:
        lines = [
            "Penprobe Chasing Demo Summary",
            "",
            f"Status: {summary.status}",
            f"Success: {summary.success}",
            f"Stop reason: {summary.experiment_metrics.get('stop_reason', 'unknown')}",
            f"Mapping mode: {summary.experiment_metrics.get('mapping_mode_used', 'unknown')}",
            f"Active segment: {summary.experiment_metrics.get('active_segment_label')} ({summary.experiment_metrics.get('active_segment_key')})",
            f"Active servos: {summary.experiment_metrics.get('active_servo_ids')}",
            f"Max tick delta cap: {summary.experiment_metrics.get('max_tick_delta_from_startup')}",
            f"Max tick delta used: {summary.experiment_metrics.get('max_tick_delta_used')}",
            f"Final XY error mm: {summary.experiment_metrics.get('final_error_norm_mm')}",
            "",
            "This MVP uses the 0A coil origin in robot frame as the controlled point and the 0B robot-frame tool origin as the target. It does not claim physical tip chasing unless a validated T_robot_tip transform is explicitly used.",
        ]
        (paths.output_dir / "penprobe_chasing_summary.txt").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def _record_metrics(self, *, session: ExperimentSession, context, servo_ids: list[int], pairs: dict[str, list[int]]) -> None:
        session.set_metric("stop_reason", self._stop_reason)
        session.set_metric("mapping_mode_used", self._mapping_mode_used)
        session.set_metric("mapping_warnings", list(self._mapping_warnings))
        session.set_metric("operating_context", context.metadata())
        session.set_metric("active_segment_key", context.active_segment_key)
        session.set_metric("active_segment_label", context.active_segment_label)
        session.set_metric("active_servo_ids", [int(value) for value in servo_ids])
        session.set_metric("active_pairs", {str(key): [int(value) for value in values] for key, values in dict(pairs).items()})
        session.set_metric("controlled_point_source", "0A coil origin in robot frame")
        session.set_metric("target_point_source", "0B tool origin in robot frame; not a validated physical pen tip")
        session.set_metric("startup_reference_by_servo", {str(k): int(v) for k, v in self._startup_reference_by_servo.items()})
        session.set_metric("max_tick_delta_from_startup", int(self.config.max_tick_delta_from_startup))
        session.set_metric(
            "max_tick_delta_used",
            max((abs(int(value)) for value in self._current_tick_delta_by_servo.values()), default=0),
        )


def _startup_reference_from_calibration(servo_service, servo_ids: list[int]) -> dict[int, int]:
    summary = (
        servo_service.neutral_calibration.get_calibration_summary()
        if getattr(servo_service, "neutral_calibration", None) is not None
        else None
    )
    if summary is None:
        pretension_reader = getattr(servo_service, "pretension_source_summary", None)
        pretension = pretension_reader([int(value) for value in servo_ids]) if callable(pretension_reader) else None
    else:
        pretension = summary.pretension_source_summary([int(value) for value in servo_ids])
    if pretension is None:
        raise RuntimeError("Accepted startup reference is unavailable from the servo calibration service.")
    if not pretension.usable:
        raise RuntimeError(pretension.message)
    positions = {int(key): value for key, value in dict(pretension.positions_by_servo).items()}
    missing = [int(servo_id) for servo_id in servo_ids if positions.get(int(servo_id)) is None]
    if missing:
        raise RuntimeError("Accepted startup reference is missing final position tick(s): " + ", ".join(str(v) for v in missing))
    return {int(servo_id): int(positions[int(servo_id)]) for servo_id in servo_ids}


def _extract_robot_frame_tool_pose(snapshot, tool_id: str, *, max_tracker_age_s: float) -> ChasingPose:
    if getattr(snapshot, "tracker_data_stale", False):
        raise RuntimeError("tracker_stale")
    age_s = getattr(snapshot, "tracker_data_age_s", None)
    if age_s is None or float(age_s) > float(max_tracker_age_s):
        raise RuntimeError(f"tracker_stale: age {age_s} s exceeds {max_tracker_age_s:.3f} s")
    T_robot_aurora = getattr(snapshot, "T_robot_aurora", None)
    if T_robot_aurora is None:
        raise RuntimeError("robot_frame_transform_unavailable")
    tool_key = str(tool_id or "").upper()
    tools = {str(key).upper(): value for key, value in dict(getattr(snapshot, "tools", {}) or {}).items()}
    tool = tools.get(tool_key)
    if tool is None:
        raise RuntimeError(f"tracker_tool_missing:{tool_key}")
    if not bool(getattr(tool, "present", True)):
        raise RuntimeError(f"tracker_tool_missing:{tool_key}")
    if getattr(tool, "valid", True) is False:
        raise RuntimeError(f"tracker_tool_invalid:{tool_key}")
    if str(getattr(tool, "tracking_state", "unknown") or "unknown") not in {"tracked", "valid"}:
        raise RuntimeError(f"tracker_tool_not_tracked:{tool_key}:{getattr(tool, 'tracking_state', 'unknown')}")
    T_aurora_tool = getattr(tool, "T_aurora_tool", None)
    if T_aurora_tool is None:
        translation = getattr(tool, "translation_mm", None)
        if translation is None:
            raise RuntimeError(f"tracker_tool_transform_missing:{tool_key}")
        T_aurora_tool = np.eye(4)
        T_aurora_tool[0][3] = float(translation[0])
        T_aurora_tool[1][3] = float(translation[1])
        T_aurora_tool[2][3] = float(translation[2])
    robot_point = np.asarray(T_robot_aurora, dtype=float) @ np.asarray(T_aurora_tool, dtype=float)
    return ChasingPose(
        tool_id=tool_key,
        position_mm=[float(robot_point[0][3]), float(robot_point[1][3]), float(robot_point[2][3])],
        tracking_state=str(getattr(tool, "tracking_state", "unknown") or "unknown"),
        frame_number=getattr(tool, "frame_number", None),
    )


def _validate_servo_telemetry_ready(servo_service, servo_ids: list[int]) -> dict[int, Any]:
    telemetry = servo_service.read_live_telemetry([int(value) for value in servo_ids])
    for servo_id in servo_ids:
        current = telemetry.get(int(servo_id))
        if current is None:
            raise RuntimeError(f"missing_telemetry:{servo_id}")
        if current.present_position is None:
            raise RuntimeError(f"missing_position:{servo_id}")
        if current.hardware_error_code not in (None, 0) or current.hardware_error:
            raise RuntimeError(f"hardware_error:{servo_id}:{current.hardware_error or current.hardware_error_code}")
        servo_service.safety_guard.validate_telemetry_freshness(current.last_read_monotonic_s)
        servo_service.safety_guard.validate_currents([current.present_current_ma], require_present=True)
        servo_service.safety_guard.validate_voltage(current.present_voltage_mv, require_present=True)
        servo_service.safety_guard.validate_temperature(current.present_temperature_c, require_present=True)
    return telemetry


def _classify_stop_reason(exc: Exception) -> str:
    message = str(exc)
    if "tracker_tool_missing" in message:
        return "tracker_tool_missing"
    if "tracker_tool_not_tracked" in message or "tracker_tool_invalid" in message:
        return "tracker_tool_invalid"
    if "tracker_stale" in message:
        return "tracker_stale"
    if "robot_frame_transform_unavailable" in message:
        return "robot_frame_transform_unavailable"
    if "missing_position" in message or "missing_telemetry" in message:
        return "servo_telemetry_missing"
    if "hardware_error" in message:
        return "servo_hardware_error"
    if "stopped by operator" in message.lower():
        return "operator_stop"
    return "failed"


def paired_xy_proportional_tick_request(
    *,
    tip_xy_mm: list[float],
    target_xy_mm: list[float],
    servo_ids: list[int],
    pairs: dict[str, list[int]],
    mapper: TendonDisplacementMapper,
    config: PenprobeChasingDemoConfig,
) -> tuple[dict[int, int], dict[str, Any]]:
    """Map XY error to antagonistic paired tick deltas relative to startup."""
    error_x = float(target_xy_mm[0]) - float(tip_xy_mm[0])
    error_y = float(target_xy_mm[1]) - float(tip_xy_mm[1])
    clamped_x, clamped_y, clamped_radius = _clamp_xy(error_x, error_y, float(config.max_target_radius_mm))
    axis_a = list(dict(pairs or {}).get("axis_a", []))
    axis_b = list(dict(pairs or {}).get("axis_b", []))
    if len(axis_a) != 2 or len(axis_b) != 2:
        if len(servo_ids) != 4:
            raise ValueError(f"Expected four active segment servo IDs; found {servo_ids}.")
        axis_a = [int(servo_ids[0]), int(servo_ids[2])]
        axis_b = [int(servo_ids[1]), int(servo_ids[3])]
    request_cm_by_servo = {int(servo_id): 0.0 for servo_id in servo_ids}
    _apply_axis_pair_request(
        request_cm_by_servo,
        pair=[int(axis_a[0]), int(axis_a[1])],
        axis_request_cm=float(clamped_x) * float(config.displacement_gain_cm_per_mm) * int(config.x_axis_sign),
    )
    _apply_axis_pair_request(
        request_cm_by_servo,
        pair=[int(axis_b[0]), int(axis_b[1])],
        axis_request_cm=float(clamped_y) * float(config.displacement_gain_cm_per_mm) * int(config.y_axis_sign),
    )
    request_ticks = {
        int(servo_id): int(mapper.displacement_cm_to_ticks(request_cm_by_servo[int(servo_id)]))
        for servo_id in servo_ids
    }
    return request_ticks, {
        "xy_error_mm": [error_x, error_y],
        "clamped_xy_error_mm": [clamped_x, clamped_y],
        "clamped_radius_mm": clamped_radius,
        "request_cm_by_servo": {str(k): float(v) for k, v in request_cm_by_servo.items()},
        "axis_a": axis_a,
        "axis_b": axis_b,
    }


def _apply_axis_pair_request(request_cm_by_servo: dict[int, float], *, pair: list[int], axis_request_cm: float) -> None:
    request_cm_by_servo[int(pair[0])] = request_cm_by_servo.get(int(pair[0]), 0.0) - float(axis_request_cm)
    request_cm_by_servo[int(pair[1])] = request_cm_by_servo.get(int(pair[1]), 0.0) + float(axis_request_cm)


def _clamp_xy(x_mm: float, y_mm: float, max_radius_mm: float) -> tuple[float, float, float]:
    radius = float(math.hypot(float(x_mm), float(y_mm)))
    if max_radius_mm <= 0.0 or radius <= max_radius_mm:
        return float(x_mm), float(y_mm), radius
    scale = float(max_radius_mm) / radius
    return float(x_mm) * scale, float(y_mm) * scale, float(max_radius_mm)


def _step_and_clamp_tick_deltas(
    *,
    current_delta_by_servo: dict[int, int],
    desired_delta_by_servo: dict[int, int],
    max_step_ticks: int,
    max_abs_delta_ticks: int,
) -> dict[int, int]:
    output: dict[int, int] = {}
    for servo_id, desired in desired_delta_by_servo.items():
        current = int(current_delta_by_servo.get(int(servo_id), 0))
        diff = int(desired) - current
        if diff > int(max_step_ticks):
            diff = int(max_step_ticks)
        elif diff < -int(max_step_ticks):
            diff = -int(max_step_ticks)
        stepped = current + diff
        output[int(servo_id)] = max(-int(max_abs_delta_ticks), min(int(max_abs_delta_ticks), int(stepped)))
    return output


def _clamp_tick_deltas_to_startup_raw_bounds(
    delta_by_servo: dict[int, int],
    *,
    startup_reference_by_servo: dict[int, int],
) -> tuple[dict[int, int], dict[int, str]]:
    output: dict[int, int] = {}
    clips: dict[int, str] = {}
    for servo_id, delta in dict(delta_by_servo).items():
        startup = int(startup_reference_by_servo[int(servo_id)])
        raw_goal = startup + int(delta)
        clamped_goal = max(RAW_POSITION_MIN_TICK, min(RAW_POSITION_MAX_TICK, raw_goal))
        if clamped_goal != raw_goal:
            clips[int(servo_id)] = f"raw_goal_{raw_goal}_clamped_to_{clamped_goal}"
        output[int(servo_id)] = int(clamped_goal - startup)
    return output, clips


def _resolve_mapping_mode(config: PenprobeChasingDemoConfig, project_root: Path) -> tuple[str, list[str]]:
    if str(config.mapping_mode) != MAPPING_LEGACY_POLYNOMIAL_WORKSPACE:
        return MAPPING_PAIRED_XY_PROPORTIONAL, []
    _validate_legacy_mapping_config(config, project_root)
    return MAPPING_LEGACY_POLYNOMIAL_WORKSPACE, [
        "legacy_polynomial_workspace files are present, but v1 still uses the bounded paired command safety envelope."
    ]


def _validate_legacy_mapping_config(config: PenprobeChasingDemoConfig, project_root: Path) -> None:
    if str(config.mapping_mode) != MAPPING_LEGACY_POLYNOMIAL_WORKSPACE:
        return
    required = [
        "x_coeffs_full",
        "zfits_poly3_full_circle",
        "known_surface",
    ]
    missing_keys = [key for key in required if not str(config.legacy_polynomial_workspace.get(key, "")).strip()]
    if missing_keys:
        raise RuntimeError("legacy_polynomial_workspace is missing required file config key(s): " + ", ".join(missing_keys))
    missing_files: list[str] = []
    for key in required:
        path = Path(config.legacy_polynomial_workspace[key])
        resolved = path if path.is_absolute() else Path(project_root) / path
        if not resolved.exists():
            missing_files.append(f"{key}={resolved}")
    if missing_files:
        raise RuntimeError("legacy_polynomial_workspace file(s) are missing: " + "; ".join(missing_files))


def _build_chasing_sample(
    *,
    session: ExperimentSession,
    snapshot,
    iteration: int,
    tip: ChasingPose,
    target: ChasingPose,
    error_xy_mm: list[float],
    error_norm_mm: float,
    requested_tick_delta_by_servo: dict[int, int],
    commanded_tick_delta_by_servo: dict[int, int],
    commanded_ticks: dict[int, int],
    measured_ticks: dict[int, int | None],
    tendon_displacements_cm: list[float],
    startup_reference_by_servo: dict[int, int],
    loop_dt_s: float,
    mapping_debug: dict[str, Any],
    mapping_mode_used: str,
    max_tick_delta_used: int,
    telemetry_by_id: dict[int, Any],
) -> ExperimentTimeseriesSample:
    return ExperimentTimeseriesSample(
        monotonic_time_s=session.elapsed_s(),
        wall_time_utc=datetime.now(timezone.utc).isoformat(),
        phase="chasing",
        step_index=int(iteration),
        sample_index=int(iteration),
        commanded_motor_values={str(k): int(v) for k, v in commanded_ticks.items()},
        commanded_cable_deltas_cm=[float(value) for value in tendon_displacements_cm],
        tracker_frame_id=getattr(snapshot, "last_frame_number", None),
        tool_ids_seen=list(getattr(snapshot, "normalized_live_tool_ids", []) or []),
        transform_validity={
            str(tool_id): str(getattr(tool, "tracking_state", "unknown"))
            for tool_id, tool in dict(getattr(snapshot, "tools", {}) or {}).items()
        },
        pose_in_tracker_frame={},
        pose_in_robot_frame={
            "tip_0A_coil": {"translation_mm": list(tip.position_mm), "source": "0A_coil_as_tip"},
            "controlled_0A_coil_origin": {"translation_mm": list(tip.position_mm), "source": "0A_coil_origin_robot_frame"},
            "target_0B_tool_origin": {"translation_mm": list(target.position_mm), "source": "0B_tool_origin_robot_frame"},
            "target_0B_tool_origin_unpivoted": {"translation_mm": list(target.position_mm), "source": "0B_tool_origin_not_physical_tip"},
        },
        freshness_s=getattr(snapshot, "tracker_data_age_s", None),
        latency_s=getattr(snapshot, "tracker_data_age_s", None),
        status_flags=[],
        backend_health={
            "canonical_state": getattr(snapshot, "canonical_state", "unknown"),
            "tracker_data_stale": bool(getattr(snapshot, "tracker_data_stale", False)),
        },
        extra={
            "tip_xy_mm": [float(tip.position_mm[0]), float(tip.position_mm[1])],
            "target_xy_mm": [float(target.position_mm[0]), float(target.position_mm[1])],
            "tip_z_mm": float(tip.position_mm[2]),
            "target_z_mm": float(target.position_mm[2]),
            "xy_error_mm": [float(error_xy_mm[0]), float(error_xy_mm[1])],
            "xy_error_norm_mm": float(error_norm_mm),
            "requested_tick_delta_by_servo": {str(k): int(v) for k, v in requested_tick_delta_by_servo.items()},
            "commanded_tick_delta_by_servo": {str(k): int(v) for k, v in commanded_tick_delta_by_servo.items()},
            "startup_reference_by_servo": {str(k): int(v) for k, v in startup_reference_by_servo.items()},
            "measured_ticks_by_servo": {str(k): v for k, v in measured_ticks.items()},
            "max_tick_delta_used": int(max_tick_delta_used),
            "loop_dt_s": float(loop_dt_s),
            "mapping_mode": str(mapping_mode_used),
            "controlled_point_source": "0A coil origin in robot frame",
            "target_point_source": "0B tool origin in robot frame; not a validated physical pen tip",
            "mapping_debug": dict(mapping_debug),
            "telemetry_age_s_by_servo": {
                str(servo_id): (
                    session.context.servo_service.telemetry_age_s(telemetry)
                    if telemetry is not None
                    else None
                )
                for servo_id, telemetry in dict(telemetry_by_id or {}).items()
            },
        },
    )


def write_penprobe_chasing_config_template(path: Path) -> None:
    """Write the default YAML payload used by docs/tests if needed."""
    payload = PenprobeChasingDemoConfig.from_dict({}).__dict__
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
