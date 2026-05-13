"""MVP one-segment 0A coil-origin chasing demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from continuum_robot.experiments.framework import BaseExperiment, ExperimentHardwareRequirements, ExperimentSession
from continuum_robot.experiments.schemas import ExperimentTimeseriesSample
from continuum_robot.servos.displacement_mapper import TendonDisplacementMapper
from continuum_robot.servos.segment_readiness import evaluate_selected_segment_readiness
from continuum_robot.servos.servo_service import (
    RAW_POSITION_MAX_TICK,
    RAW_POSITION_MIN_TICK,
    SINGLE_SEGMENT_WORKFLOW_EXPERIMENT,
    is_wrap_risk,
)


MAPPING_PAIRED_XY_PROPORTIONAL = "paired_xy_proportional"
MAPPING_LEGACY_POLYNOMIAL_WORKSPACE = "legacy_polynomial_workspace"
SUPPORTED_MAPPING_MODES = {MAPPING_PAIRED_XY_PROPORTIONAL, MAPPING_LEGACY_POLYNOMIAL_WORKSPACE}


@dataclass
class PenprobeChasingDemoConfig:
    """Operator config for the single-segment penprobe chasing MVP."""

    loop_period_s: float = 0.04
    max_duration_s: float = 30.0
    max_iterations: int = 750
    max_tick_delta_from_startup: int = 500
    hard_max_tick_delta_from_startup: int = 500
    max_step_ticks: int = 25
    max_tick_step_per_cycle: int = 25
    max_target_radius_mm: float = 45.0
    target_deadband_mm: float = 1.0
    xy_deadband_mm: float = 1.0
    proportional_gain_ticks_per_mm: float = 8.0
    displacement_gain_cm_per_mm: float = 0.002
    stale_tracker_timeout_s: float = 0.2
    stale_tracker_persist_cycles_before_stop: int = 12
    gui_status_nominal_hz: float = 8.0
    max_servo_write_hz: float = 25.0
    saturation_stop_cycles: int = 40
    x_axis_sign: int = 1
    y_axis_sign: int = 1
    flip_x: bool = False
    flip_y: bool = False
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
            loop_period_s=max(0.02, float(payload.get("loop_period_s", payload.get("loop_target_period_s", 0.04)))),
            max_duration_s=max(0.0, float(payload.get("max_duration_s", 30.0))),
            max_iterations=max(1, int(payload.get("max_iterations", 750))),
            max_tick_delta_from_startup=max(0, int(payload.get("max_tick_delta_from_startup", 500))),
            hard_max_tick_delta_from_startup=max(0, int(payload.get("hard_max_tick_delta_from_startup", 500))),
            max_step_ticks=max(1, min(100, int(payload.get("max_step_ticks", payload.get("max_tick_step_per_cycle", 25))))),
            max_tick_step_per_cycle=max(1, min(100, int(payload.get("max_tick_step_per_cycle", payload.get("max_step_ticks", 25))))),
            max_target_radius_mm=max(0.0, float(payload.get("max_target_radius_mm", 45.0))),
            target_deadband_mm=max(0.0, float(payload.get("target_deadband_mm", 1.0))),
            xy_deadband_mm=max(0.0, float(payload.get("xy_deadband_mm", payload.get("target_deadband_mm", 1.0)))),
            proportional_gain_ticks_per_mm=max(0.0, float(payload.get("proportional_gain_ticks_per_mm", 8.0))),
            displacement_gain_cm_per_mm=max(0.0, float(payload.get("displacement_gain_cm_per_mm", 0.002))),
            stale_tracker_timeout_s=max(0.02, float(payload.get("stale_tracker_timeout_s", 0.2))),
            stale_tracker_persist_cycles_before_stop=max(1, int(payload.get("stale_tracker_persist_cycles_before_stop", 12))),
            gui_status_nominal_hz=max(1.0, float(payload.get("gui_status_nominal_hz", 8.0))),
            max_servo_write_hz=max(1.0, float(payload.get("max_servo_write_hz", 25.0))),
            saturation_stop_cycles=max(1, int(payload.get("saturation_stop_cycles", 40))),
            x_axis_sign=(1 if int(payload.get("x_axis_sign", 1)) >= 0 else -1),
            y_axis_sign=(1 if int(payload.get("y_axis_sign", 1)) >= 0 else -1),
            flip_x=bool(payload.get("flip_x", False)),
            flip_y=bool(payload.get("flip_y", False)),
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
        self._chase_perf: dict[str, Any] = {}

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
        perf: dict[str, Any] = {
            "control_loop_dt_s": [],
            "servo_write_dt_s": [],
            "tracker_age_s": [],
            "command_compute_s": [],
            "servo_command_write_s": [],
            "commanded_tick_step_norm": [],
            "command_saturated_bool": [],
            "skipped_write_reason": [],
            "actual_control_hz_instant": [],
            "actual_servo_write_hz_instant": [],
        }
        total = int(self.config.max_iterations)
        deadline = (
            float(session.context.monotonic_fn()) + float(self.config.max_duration_s)
            if float(self.config.max_duration_s) > 0.0
            else None
        )
        start_monotonic = float(session.context.monotonic_fn())
        self._stop_reason = "running"
        last_servo_write_monotonic = start_monotonic - (1.0 / max(1e-6, float(self.config.max_servo_write_hz)))
        saturated_cycles = 0
        stale_tracker_cycles = 0
        last_good_tip: ChasingPose | None = None
        last_good_target: ChasingPose | None = None
        stale_timeout = float(self.config.stale_tracker_timeout_s)
        loop_period_s = float(self.config.loop_period_s)
        persist = max(1, int(self.config.stale_tracker_persist_cycles_before_stop))
        max_step = max(1, int(self.config.max_tick_step_per_cycle or self.config.max_step_ticks))
        last_known_telemetry: dict[int, Any] = {}
        progress_stride = max(1, int(round(0.5 / max(loop_period_s, 1e-6))))
        last_burst_servo_write_hz: float | None = None

        try:
            with session.context.servo_service.exclusive_bus_operation(
                owner="penprobe_chasing_demo",
                reason="bounded active single-segment 0A-to-0B chasing demo",
            ):
                last_known_telemetry = _validate_servo_telemetry_ready(session.context.servo_service, servo_ids)

                for iteration in range(total):
                    session.raise_if_stop_requested()
                    iteration_start_mono = float(session.context.monotonic_fn())
                    if deadline is not None and iteration_start_mono >= deadline:
                        self._stop_reason = "max_duration_reached"
                        break

                    snapshot = session.context.tracking_service.get_snapshot()
                    tracker_age_s = getattr(snapshot, "tracker_data_age_s", None)
                    if tracker_age_s is not None:
                        perf["tracker_age_s"].append(float(tracker_age_s))

                    peek_gui_hz = getattr(session.context, "penprobe_live_gui_hz", None)
                    actual_gui_hz: float | None = None
                    if callable(peek_gui_hz):
                        raw_hz = peek_gui_hz()
                        if isinstance(raw_hz, (int, float)):
                            actual_gui_hz = float(raw_hz)

                    tip_pose, tip_err = _extract_chasing_pose_optional(snapshot, "0A", max_tracker_age_s=stale_timeout)
                    tgt_pose, tgt_err = _extract_chasing_pose_optional(snapshot, "0B", max_tracker_age_s=stale_timeout)

                    transient_detail: list[str] = []
                    if tip_pose is None:
                        transient_detail.append(tip_err or "tracker_tip_unavailable")
                    if tgt_pose is None:
                        transient_detail.append(tgt_err or "tracker_target_unavailable")
                    transient_hold = tip_pose is None or tgt_pose is None

                    if transient_hold:
                        stale_tracker_cycles += 1
                        if stale_tracker_cycles >= persist:
                            detail_txt = "; ".join(transient_detail) if transient_detail else "persisted_without_detail"
                            raise RuntimeError(f"tracker_stale_persisted:{detail_txt}")
                    else:
                        stale_tracker_cycles = 0
                        last_good_tip, last_good_target = tip_pose, tgt_pose

                    live_tip = tip_pose if tip_pose is not None else last_good_tip
                    live_tgt = tgt_pose if tgt_pose is not None else last_good_target
                    sample_tip = live_tip if live_tip is not None else ChasingPose("0A", [0.0, 0.0, 0.0], "unknown", None)
                    sample_tgt = live_tgt if live_tgt is not None else ChasingPose("0B", [0.0, 0.0, 0.0], "unknown", None)

                    compute_started = float(session.context.monotonic_fn())

                    skipped_write_reason: str | None = None
                    desired_delta_by_servo = {int(servo_id): 0 for servo_id in servo_ids}
                    mapping_debug: dict[str, Any] = {}
                    bounded_candidate_delta = dict(self._current_tick_delta_by_servo)
                    raw_bound_clips: dict[int, str] = {}
                    command_result_message: dict[str, Any] = {}

                    stable_before = dict(self._current_tick_delta_by_servo)

                    command_write_duration_s = 0.0
                    telemetry_for_sample = dict(last_known_telemetry)

                    if transient_hold:
                        skipped_write_reason = "tracking_hold"
                        mapping_debug = {
                            "transient_tracking_hold": True,
                            "transient_tracker_detail": list(transient_detail),
                        }
                        tendon_displacements_cm = [
                            float(mapper.ticks_to_displacement_mm(stable_before[int(servo_id)]) / 10.0) for servo_id in servo_ids
                        ]
                        _validate_cached_chase_telemetry(session.context.servo_service, servo_ids, last_known_telemetry)
                    else:
                        assert tip_pose is not None and tgt_pose is not None
                        _validate_cached_chase_telemetry(session.context.servo_service, servo_ids, last_known_telemetry)
                        desired_delta_by_servo, mapping_debug = paired_xy_proportional_tick_request(
                            tip_xy_mm=tip_pose.position_mm[:2],
                            target_xy_mm=tgt_pose.position_mm[:2],
                            servo_ids=servo_ids,
                            pairs=pairs,
                            mapper=mapper,
                            config=self.config,
                        )
                        stepped_delta_by_servo = _step_and_clamp_tick_deltas(
                            current_delta_by_servo=self._current_tick_delta_by_servo,
                            desired_delta_by_servo=desired_delta_by_servo,
                            max_step_ticks=max_step,
                            max_abs_delta_ticks=int(self.config.max_tick_delta_from_startup),
                        )
                        bounded_candidate_delta, raw_bound_clips = _clamp_tick_deltas_to_startup_raw_bounds(
                            stepped_delta_by_servo,
                            startup_reference_by_servo=self._startup_reference_by_servo,
                        )
                        hypo_command_ticks = {
                            int(servo_id): int(self._startup_reference_by_servo[int(servo_id)] + bounded_candidate_delta[int(servo_id)])
                            for servo_id in servo_ids
                        }
                        wrap_risk_payload: dict[str, int] = {}
                        for servo_id in servo_ids:
                            telem = last_known_telemetry[int(servo_id)]
                            present = telem.present_position
                            if present is None:
                                raise RuntimeError(f"missing_position:{servo_id}")
                            hypo = hypo_command_ticks[int(servo_id)]
                            if is_wrap_risk(int(present), int(hypo), (RAW_POSITION_MIN_TICK, RAW_POSITION_MAX_TICK)):
                                wrap_risk_payload[str(int(servo_id))] = int(hypo)
                        if wrap_risk_payload:
                            raise RuntimeError("wrap_risk:" + json.dumps(wrap_risk_payload, sort_keys=True))

                        tendon_displacements_cm = [
                            float(mapper.ticks_to_displacement_mm(bounded_candidate_delta[int(servo_id)]) / 10.0)
                            for servo_id in servo_ids
                        ]

                        now_cmd = float(session.context.monotonic_fn())
                        ok_write, gate_reason = _servo_write_allowed(
                            now_s=now_cmd,
                            last_write_monotonic_s=float(last_servo_write_monotonic),
                            max_hz=float(self.config.max_servo_write_hz),
                        )
                        if ok_write:
                            t_wr_start = float(session.context.monotonic_fn())
                            command_result = session.context.servo_service.command_displacement(
                                tendon_displacements_cm,
                                [int(self._startup_reference_by_servo[int(servo_id)]) for servo_id in servo_ids],
                                servo_ids,
                                motion_workflow=SINGLE_SEGMENT_WORKFLOW_EXPERIMENT,
                                chase_tight_loop_writes=True,
                            )
                            command_write_duration_s = max(0.0, float(session.context.monotonic_fn()) - t_wr_start)
                            last_known_telemetry = dict(command_result.telemetry_by_id)
                            telemetry_for_sample = dict(last_known_telemetry)
                            last_servo_write_monotonic = float(session.context.monotonic_fn())
                            self._current_tick_delta_by_servo = dict(bounded_candidate_delta)
                            positions_by_id = dict(command_result.positions_by_id)
                            clamp_reasons_by_id = dict(command_result.clamp_reasons_by_id)
                            command_result_message = {
                                "servo_command_positions_by_id": {str(k): int(v) for k, v in positions_by_id.items()},
                                "servo_clamp_reasons_by_id": {str(k): str(v) for k, v in clamp_reasons_by_id.items()},
                            }
                            perf["servo_command_write_s"].append(float(command_write_duration_s))
                            if command_write_duration_s > 1e-9:
                                perf["servo_write_dt_s"].append(float(command_write_duration_s))
                                last_burst_servo_write_hz = 1.0 / float(command_write_duration_s)
                                perf["actual_servo_write_hz_instant"].append(last_burst_servo_write_hz)
                        else:
                            skipped_write_reason = gate_reason or "rate_limit"
                            tendon_displacements_cm = [
                                float(mapper.ticks_to_displacement_mm(stable_before[int(servo_id)]) / 10.0)
                                for servo_id in servo_ids
                            ]
                            self._current_tick_delta_by_servo = dict(stable_before)

                    bounded_applied_delta = dict(self._current_tick_delta_by_servo)
                    commanded_ticks = {
                        int(servo_id): int(self._startup_reference_by_servo[int(servo_id)] + bounded_applied_delta[int(servo_id)])
                        for servo_id in servo_ids
                    }

                    planned_tick_step_by_servo = {
                        int(servo_id): int(bounded_candidate_delta[int(servo_id)] - stable_before.get(int(servo_id), 0))
                        for servo_id in servo_ids
                    }

                    commanded_step_by_servo = {
                        int(servo_id): int(bounded_applied_delta[int(servo_id)] - stable_before.get(int(servo_id), 0))
                        for servo_id in servo_ids
                    }

                    command_compute_duration_s = max(0.0, float(session.context.monotonic_fn()) - compute_started)
                    perf["command_compute_s"].append(float(command_compute_duration_s))

                    planned_tick_step_norm = float(_command_tick_step_norm(servo_ids, planned_tick_step_by_servo))
                    perf["commanded_tick_step_norm"].append(planned_tick_step_norm)

                    error_xy = [
                        float(sample_tgt.position_mm[0] - sample_tip.position_mm[0]),
                        float(sample_tgt.position_mm[1] - sample_tip.position_mm[1]),
                    ]
                    error_norm = float(math.hypot(error_xy[0], error_xy[1]))
                    max_tick_delta_used = max((abs(int(v)) for v in bounded_applied_delta.values()), default=0)
                    distance_to_cap = max(0, int(self.config.max_tick_delta_from_startup) - int(max_tick_delta_used))
                    saturation_event = distance_to_cap <= 0 and not transient_hold
                    perf["command_saturated_bool"].append(bool(saturation_event))
                    if saturation_event:
                        saturated_cycles += 1
                    else:
                        saturated_cycles = 0
                    if saturated_cycles >= int(self.config.saturation_stop_cycles):
                        raise RuntimeError("saturation_limit_persisted")

                    measured_ticks = {
                        int(servo_id): (
                            int(telemetry_for_sample[int(servo_id)].present_position)
                            if telemetry_for_sample.get(int(servo_id)) is not None
                            and telemetry_for_sample[int(servo_id)].present_position is not None
                            else None
                        )
                        for servo_id in servo_ids
                    }

                    sleep_remaining = loop_period_s - (float(session.context.monotonic_fn()) - iteration_start_mono)
                    if sleep_remaining > 0.0:
                        session.context.sleep_fn(sleep_remaining)
                    loop_completed_dt_s = max(1e-9, float(session.context.monotonic_fn()) - iteration_start_mono)
                    perf["control_loop_dt_s"].append(float(loop_completed_dt_s))
                    perf["actual_control_hz_instant"].append(1.0 / loop_completed_dt_s)
                    skipped_token = skipped_write_reason or "servo_written"
                    perf["skipped_write_reason"].append(str(skipped_token))

                    instantaneous_control_hz = 1.0 / loop_completed_dt_s
                    burst_servo_hz = last_burst_servo_write_hz

                    commanded_tick_step_norm = planned_tick_step_norm
                    saturated_flag = bool(saturation_event)

                    chase_line = (
                        f"Control: {instantaneous_control_hz:.1f} Hz, "
                        f"Servo writes: {burst_servo_hz:.1f} Hz"
                        if burst_servo_hz is not None
                        else f"Control: {instantaneous_control_hz:.1f} Hz, Servo writes: n/a"
                    )
                    if tracker_age_s is None:
                        chase_line += ", Tracker age: n/a ms"
                    else:
                        chase_line += f", Tracker age: {float(tracker_age_s) * 1000.0:.0f} ms"

                    chase_instrument_extra = {
                        "actual_control_hz": float(instantaneous_control_hz),
                        "actual_servo_write_hz": burst_servo_hz,
                        "actual_gui_update_hz": actual_gui_hz,
                        "tracker_age_s": float(tracker_age_s) if tracker_age_s is not None else None,
                        "commanded_tick_step_norm": float(commanded_tick_step_norm),
                        "command_saturated_true_false": bool(saturated_flag),
                        "skipped_write_reason": skipped_token,
                        "bus_ownership_snapshot": _bus_ownership_dict(session.context.servo_service),
                        "chase_live_status_line": chase_line,
                        "planned_tick_step_norm": planned_tick_step_norm,
                    }

                    sample = _build_chasing_sample(
                        session=session,
                        snapshot=snapshot,
                        iteration=iteration,
                        tip=sample_tip,
                        target=sample_tgt,
                        error_xy_mm=error_xy,
                        error_norm_mm=error_norm,
                        requested_tick_delta_by_servo=dict(desired_delta_by_servo),
                        commanded_tick_delta_by_servo=bounded_applied_delta,
                        commanded_tick_step_by_servo=commanded_step_by_servo,
                        commanded_ticks=commanded_ticks,
                        measured_ticks=measured_ticks,
                        tendon_displacements_cm=list(tendon_displacements_cm),
                        startup_reference_by_servo=self._startup_reference_by_servo,
                        loop_dt_s=float(loop_completed_dt_s),
                        mapping_debug={
                            **dict(mapping_debug),
                            **command_result_message,
                            "command_compute_duration_s": float(command_compute_duration_s),
                            "servo_command_write_duration_s": float(command_write_duration_s),
                            "raw_bound_clips": {str(k): v for k, v in raw_bound_clips.items()},
                            "distance_to_cap_ticks": int(distance_to_cap),
                            "skipped_write_reason": skipped_token,
                        },
                        mapping_mode_used=self._mapping_mode_used,
                        max_tick_delta_used=max_tick_delta_used,
                        telemetry_by_id=dict(telemetry_for_sample),
                        chase_instrument_extra=dict(chase_instrument_extra),
                    )
                    session.add_sample(sample)
                    if iteration == 0 or (iteration % progress_stride == 0):
                        session.update_progress(iteration + 1, total, {"phase": "chasing", "error_mm": error_norm})
                    if error_norm <= float(self.config.xy_deadband_mm):
                        session.set_metric("last_deadband_iteration", int(iteration))
                else:
                    self._stop_reason = "max_iterations_reached"
        except Exception as exc:
            self._stop_reason = _classify_stop_reason(exc)
            self._chase_perf = perf
            self._record_metrics(session=session, context=context, servo_ids=servo_ids, pairs=pairs)
            raise

        self._chase_perf = perf
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
        em = dict(summary.experiment_metrics or {})
        hz_ctrl = em.get("achieved_mean_control_hz")
        hz_servo = em.get("achieved_mean_servo_write_hz")
        hz_gui = em.get("achieved_mean_gui_hz")
        lines = [
            "Penprobe Chasing Demo Summary",
            "",
            f"Status: {summary.status}",
            f"Success: {summary.success}",
            f"Stop reason: {em.get('stop_reason', 'unknown')}",
            f"Mapping mode: {em.get('mapping_mode_used', 'unknown')}",
            f"Active segment: {em.get('active_segment_label')} ({em.get('active_segment_key')})",
            f"Active servos: {em.get('active_servo_ids')}",
            f"Max tick delta cap: {em.get('max_tick_delta_from_startup')}",
            f"Max tick delta used: {em.get('max_tick_delta_used')}",
            f"Final XY error mm: {em.get('final_error_norm_mm')}",
            "",
            "Loop instrumentation (session mean / max aggregates):",
            (
                f"Achieved mean control Hz: {float(hz_ctrl):.2f}"
                if isinstance(hz_ctrl, (int, float))
                else "Achieved mean control Hz: n/a"
            ),
            (
                f"Achieved mean servo write Hz: {float(hz_servo):.2f}"
                if isinstance(hz_servo, (int, float))
                else "Achieved mean servo write Hz: n/a"
            ),
            (
                f"Mean GUI telemetry Hz snapshot: {float(hz_gui):.2f}"
                if isinstance(hz_gui, (int, float))
                else "Mean GUI telemetry Hz snapshot: n/a"
            ),
            f"Mean / max tracker age (s): {em.get('mean_tracker_age_s')} / {em.get('max_tracker_age_s')}",
            f"Mean / max commanded tick step norm (ticks/cycle planned): "
            f"{em.get('mean_commanded_tick_step_norm')} / {em.get('max_commanded_tick_step_norm')}",
            f"Saturation sample fraction (distance-to-cap exhausted while tracking): "
            f"{em.get('chase_saturation_fraction')}",
            f"Skipped write counts: {em.get('chase_skipped_write_counts')}",
            "",
            "This MVP uses the 0A coil origin in robot frame as the controlled point and the 0B robot-frame "
            "tool origin as the target. It does not claim physical tip chasing unless a validated T_robot_tip "
            "transform is explicitly used.",
        ]
        text = "\n".join(lines).rstrip() + "\n"
        (paths.output_dir / "penprobe_chasing_summary.txt").write_text(text, encoding="utf-8")

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
        perf = dict(getattr(self, "_chase_perf", {}) or {})
        ctrl_dts = list(perf.get("control_loop_dt_s") or [])
        servo_dts = list(perf.get("servo_write_dt_s") or [])
        trackers = list(perf.get("tracker_age_s") or [])
        norms = list(perf.get("commanded_tick_step_norm") or [])
        saturation_flags = list(perf.get("command_saturated_bool") or [])
        skipped = Counter(str(value) for value in (perf.get("skipped_write_reason") or []))
        gui_samples = []
        if session.samples:
            for sample in session.samples:
                extra = getattr(sample, "extra", {}) or {}
                gz = extra.get("actual_gui_update_hz")
                if isinstance(gz, (int, float)):
                    gui_samples.append(float(gz))

        mean_ctrl_hz = float(len(ctrl_dts) / sum(ctrl_dts)) if ctrl_dts and sum(ctrl_dts) > 1e-12 else None
        mean_servo_hz = float(len(servo_dts) / sum(servo_dts)) if servo_dts and sum(servo_dts) > 1e-12 else None

        session.set_metric("achieved_mean_control_hz", mean_ctrl_hz)
        session.set_metric("achieved_mean_servo_write_hz", mean_servo_hz)
        session.set_metric("mean_tracker_age_s", float(sum(trackers)) / len(trackers) if trackers else None)
        session.set_metric("max_tracker_age_s", float(max(trackers)) if trackers else None)
        session.set_metric(
            "mean_commanded_tick_step_norm",
            float(sum(norms)) / len(norms) if norms else None,
        )
        session.set_metric("max_commanded_tick_step_norm", float(max(norms)) if norms else None)
        session.set_metric(
            "chase_saturation_fraction",
            float(sum(1 for value in saturation_flags if value)) / float(len(saturation_flags))
            if saturation_flags
            else None,
        )
        session.set_metric(
            "chase_skipped_write_counts",
            {str(reason): int(count) for reason, count in sorted(skipped.items())},
        )
        session.set_metric(
            "achieved_mean_gui_hz",
            float(sum(gui_samples)) / len(gui_samples) if gui_samples else None,
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


_FATAL_CHASE_TOOL_TAGS = (
    "tracker_tool_missing:",
    "tracker_tool_transform_missing:",
    "robot_frame_transform_unavailable",
)


def _fatal_tracker_exc_for_chase(exc: RuntimeError) -> bool:
    return any(tag in str(exc) for tag in _FATAL_CHASE_TOOL_TAGS)


def _extract_chasing_pose_optional(
    snapshot,
    tool_id: str,
    *,
    max_tracker_age_s: float,
) -> tuple[ChasingPose | None, str | None]:
    try:
        return _extract_robot_frame_tool_pose(snapshot, tool_id, max_tracker_age_s=max_tracker_age_s), None
    except RuntimeError as exc:
        if _fatal_tracker_exc_for_chase(exc):
            raise
        return None, str(exc)


def _servo_write_allowed(*, now_s: float, last_write_monotonic_s: float, max_hz: float) -> tuple[bool, str | None]:
    interval_s = 1.0 / max(1e-6, float(max_hz))
    if float(now_s) + 1e-12 >= float(last_write_monotonic_s) + float(interval_s):
        return True, None
    return False, "rate_limit"


def _bus_ownership_dict(servo_service) -> dict[str, Any]:
    try:
        st = servo_service.bus_ownership_status()
    except Exception as exc:
        return {"ownership_error": str(exc)}
    return {
        "active": bool(st.active),
        "owner": st.owner,
        "reason": st.reason,
        "servo_id": st.servo_id,
        "held_by_current_thread": bool(st.held_by_current_thread),
        "started_at_monotonic_s": st.started_at_monotonic_s,
    }


def _command_tick_step_norm(servo_ids: list[int], step_by_id: dict[int, int]) -> float:
    vec = [float(step_by_id.get(int(servo_id), 0)) for servo_id in servo_ids]
    return float(math.sqrt(sum(component * component for component in vec)))


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


def _clamp_xy_tick_vector_l2(x_ticks: float, y_ticks: float, max_norm: float) -> tuple[float, float]:
    """Clamp (x,y) tick-space axis requests to Euclidean norm max_norm."""
    cap = float(max_norm)
    if cap <= 0.0:
        return 0.0, 0.0
    nx = float(x_ticks)
    ny = float(y_ticks)
    length = float(math.hypot(nx, ny))
    if length <= cap or length <= 0.0:
        return nx, ny
    scale = cap / length
    return nx * scale, ny * scale


def _validate_cached_chase_telemetry(servo_service, servo_ids: list[int], telemetry_by_id: dict[int, Any]) -> None:
    """Lightweight hot-loop guard using already-cached telemetry (no live bus read)."""
    for servo_id in servo_ids:
        current = telemetry_by_id.get(int(servo_id))
        if current is None:
            raise RuntimeError(f"missing_telemetry:{servo_id}")
        if current.present_position is None:
            raise RuntimeError(f"missing_position:{servo_id}")
        if current.hardware_error_code not in (None, 0) or current.hardware_error:
            raise RuntimeError(f"hardware_error:{servo_id}:{current.hardware_error or current.hardware_error_code}")


def _classify_stop_reason(exc: Exception) -> str:
    message = str(exc)
    if "tracker_stale_persisted" in message:
        return "tracker_stale_persisted"
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
    if "wrap_risk" in message:
        return "wrap_risk"
    if "saturation_limit_persisted" in message:
        return "saturation_limit_persisted"
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
    if float(math.hypot(error_x, error_y)) <= float(config.xy_deadband_mm):
        error_x = 0.0
        error_y = 0.0
    clamped_x, clamped_y, clamped_radius = _clamp_xy(error_x, error_y, float(config.max_target_radius_mm))
    axis_a = list(dict(pairs or {}).get("axis_a", []))
    axis_b = list(dict(pairs or {}).get("axis_b", []))
    if len(axis_a) != 2 or len(axis_b) != 2:
        if len(servo_ids) != 4:
            raise ValueError(f"Expected four active segment servo IDs; found {servo_ids}.")
        axis_a = [int(servo_ids[0]), int(servo_ids[2])]
        axis_b = [int(servo_ids[1]), int(servo_ids[3])]
    gain = float(config.proportional_gain_ticks_per_mm)
    raw_axis_x_ticks = float(clamped_x) * gain * int(config.x_axis_sign) * (-1 if config.flip_x else 1)
    raw_axis_y_ticks = float(clamped_y) * gain * int(config.y_axis_sign) * (-1 if config.flip_y else 1)
    cap = float(config.max_tick_step_per_cycle)
    axis_x_ticks, axis_y_ticks = _clamp_xy_tick_vector_l2(raw_axis_x_ticks, raw_axis_y_ticks, cap)
    request_cm_by_servo = {int(servo_id): 0.0 for servo_id in servo_ids}
    _apply_axis_pair_request(
        request_cm_by_servo,
        pair=[int(axis_a[0]), int(axis_a[1])],
        axis_request_ticks=int(round(axis_x_ticks)),
    )
    _apply_axis_pair_request(
        request_cm_by_servo,
        pair=[int(axis_b[0]), int(axis_b[1])],
        axis_request_ticks=int(round(axis_y_ticks)),
    )
    request_ticks = {int(servo_id): int(request_cm_by_servo[int(servo_id)]) for servo_id in servo_ids}
    return request_ticks, {
        "xy_error_mm": [error_x, error_y],
        "clamped_xy_error_mm": [clamped_x, clamped_y],
        "clamped_radius_mm": clamped_radius,
        "raw_axis_request_ticks_xy": [raw_axis_x_ticks, raw_axis_y_ticks],
        "norm_clamped_axis_request_ticks_xy": [axis_x_ticks, axis_y_ticks],
        "tick_norm_cap_per_cycle": cap,
        "request_ticks_by_servo": {str(k): int(v) for k, v in request_ticks.items()},
        "lower_tick_means_tension": True,
        "proportional_gain_ticks_per_mm": float(config.proportional_gain_ticks_per_mm),
        "axis_a": axis_a,
        "axis_b": axis_b,
    }


def _apply_axis_pair_request(request_cm_by_servo: dict[int, float], *, pair: list[int], axis_request_ticks: int) -> None:
    request_cm_by_servo[int(pair[0])] = request_cm_by_servo.get(int(pair[0]), 0.0) - int(axis_request_ticks)
    request_cm_by_servo[int(pair[1])] = request_cm_by_servo.get(int(pair[1]), 0.0) + int(axis_request_ticks)


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
    commanded_tick_step_by_servo: dict[int, int],
    commanded_ticks: dict[int, int],
    measured_ticks: dict[int, int | None],
    tendon_displacements_cm: list[float],
    startup_reference_by_servo: dict[int, int],
    loop_dt_s: float,
    mapping_debug: dict[str, Any],
    mapping_mode_used: str,
    max_tick_delta_used: int,
    telemetry_by_id: dict[int, Any],
    chase_instrument_extra: dict[str, Any] | None = None,
) -> ExperimentTimeseriesSample:
    chase_instrument_extra = dict(chase_instrument_extra or {})
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
            "commanded_tick_step_by_servo": {str(k): int(v) for k, v in commanded_tick_step_by_servo.items()},
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
            **chase_instrument_extra,
        },
    )


def write_penprobe_chasing_config_template(path: Path) -> None:
    """Write the default YAML payload used by docs/tests if needed."""
    payload = PenprobeChasingDemoConfig.from_dict({}).__dict__
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
