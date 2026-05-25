"""Two-segment penprobe lookup demo experiment.

Uses the 0B penprobe tool as a desired target XYZ in robot base frame,
finds the nearest map point from a previously-built workspace lookup map,
and feedforward-commands the associated all-8 servo goal ticks. This is
NOT closed-loop control; it is a presentation-quality nearest-reachable
playback of recorded data.

The experiment is explicitly demo-only:
- ``valid_for_model_training = False`` (always)
- ``valid_for_thesis_repeatability = False`` (always)
- ``not_closed_loop_validated = True`` (always)
- run summary text + JSON spell this out

Safety contract:
- never writes goals when the tracker is stale, when the 0B tool is missing,
  when the map is incompatible with the runtime, or when the controller's
  safety limiter trips on any servo
- always uses the existing ``ServoService._write_goal_positions`` path so the
  bus seam, hardware-error stops, and current/load checks are unchanged
- never torque-offs as part of the demo
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from continuum_robot.demo.two_segment_lookup_controller import (
    LookupControllerConfig,
    LookupDecision,
    TwoSegmentWorkspaceLookupController,
)
from continuum_robot.demo.two_segment_workspace_lookup import load_workspace_lookup_map
from continuum_robot.experiments.framework import (
    BaseExperiment,
    ExperimentHardwareRequirements,
    ExperimentSession,
)
from continuum_robot.experiments.schemas import ExperimentTimeseriesSample


EXPERIMENT_NAME = "two_segment_penprobe_lookup_demo"
DEMO_SCHEMA_VERSION = "two_segment_penprobe_lookup_demo_v1"
BLOCK_MESSAGE = (
    "Two-segment penprobe lookup demo requires operating_mode=dual_segment, all 8 servos, "
    "an accepted all-8 startup, and a loaded workspace lookup map whose servo IDs + "
    "bottom/top assignment match the current runtime."
)


@dataclass
class TwoSegmentPenprobeLookupDemoConfig:
    """Operator configuration for the penprobe lookup demo experiment."""

    map_path: str = ""
    target_tool_id: str = "0B"
    target_tool_role: str = "penprobe_target"
    control_rate_hz: float = 3.0
    max_duration_s: float = 60.0
    max_iterations: int = 600
    command_update_deadband_mm: float = 1.0
    min_target_motion_mm: float = 1.0
    nearest_distance_warning_mm: float = 5.0
    max_nearest_distance_mm: float = 25.0
    interpolation_mode: str = "nearest"
    knn: int = 1
    allow_interpolation: bool = False
    require_dual_segment: bool = True
    require_all_8_startup: bool = True
    demo_only: bool = True
    max_current_warning_ma: int = 800
    hard_current_stop_ma: int = 1200
    hard_current_stop_duration_s: float = 2.0
    transport_burst_recovery_enabled: bool = True
    stale_tracker_timeout_s: float = 0.25
    absolute_min_tick: int = 0
    absolute_max_tick: int = 4095
    extra_per_servo_tick_headroom: int = 0
    allow_servo_only_test_run: bool = False
    run_trust_mode: str = "demo_only"
    physical_assembly_confirmed_by_operator: bool = False
    dry_run: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "TwoSegmentPenprobeLookupDemoConfig":
        payload = dict(payload or {})
        interp = str(payload.get("interpolation_mode", "nearest") or "nearest").lower()
        if interp not in {"nearest", "inverse_distance"}:
            interp = "nearest"
        return cls(
            map_path=str(payload.get("map_path", "") or ""),
            target_tool_id=str(payload.get("target_tool_id", "0B") or "0B").upper(),
            target_tool_role=str(payload.get("target_tool_role", "penprobe_target") or "penprobe_target"),
            control_rate_hz=max(0.5, float(payload.get("control_rate_hz", 3.0))),
            max_duration_s=max(0.0, float(payload.get("max_duration_s", 60.0))),
            max_iterations=max(1, int(payload.get("max_iterations", 600))),
            command_update_deadband_mm=max(0.0, float(payload.get("command_update_deadband_mm", 1.0))),
            min_target_motion_mm=max(0.0, float(payload.get("min_target_motion_mm", 1.0))),
            nearest_distance_warning_mm=max(0.0, float(payload.get("nearest_distance_warning_mm", 5.0))),
            max_nearest_distance_mm=max(0.0, float(payload.get("max_nearest_distance_mm", 25.0))),
            interpolation_mode=interp,
            knn=max(1, int(payload.get("knn", 1))),
            allow_interpolation=bool(payload.get("allow_interpolation", False)),
            require_dual_segment=bool(payload.get("require_dual_segment", True)),
            require_all_8_startup=bool(payload.get("require_all_8_startup", True)),
            demo_only=bool(payload.get("demo_only", True)),
            max_current_warning_ma=max(0, int(payload.get("max_current_warning_ma", 800))),
            hard_current_stop_ma=max(0, int(payload.get("hard_current_stop_ma", 1200))),
            hard_current_stop_duration_s=max(0.1, float(payload.get("hard_current_stop_duration_s", 2.0))),
            transport_burst_recovery_enabled=bool(payload.get("transport_burst_recovery_enabled", True)),
            stale_tracker_timeout_s=max(0.05, float(payload.get("stale_tracker_timeout_s", 0.25))),
            absolute_min_tick=max(0, int(payload.get("absolute_min_tick", 0))),
            absolute_max_tick=min(4095, int(payload.get("absolute_max_tick", 4095))),
            extra_per_servo_tick_headroom=max(0, int(payload.get("extra_per_servo_tick_headroom", 0))),
            allow_servo_only_test_run=bool(payload.get("allow_servo_only_test_run", False)),
            run_trust_mode=str(payload.get("run_trust_mode", "demo_only") or "demo_only").lower(),
            physical_assembly_confirmed_by_operator=bool(
                payload.get("physical_assembly_confirmed_by_operator", False)
            ),
            dry_run=bool(payload.get("dry_run", False)),
        )


@dataclass
class _DemoTraceRow:
    iteration: int
    monotonic_time_s: float
    wall_time_utc: str
    target_xyz_robot_mm: list[float] | None
    target_age_s: float | None
    selected_map_index: int | None
    selected_point_xyz_robot_mm: list[float] | None
    nearest_distance_mm: float | None
    all_8_goal_ticks: dict[str, int] | None
    command_sent: bool
    command_skip_reason: str | None
    block_reason: str | None
    safety_limiter_reasons: list[str]
    is_extrapolating: bool
    inside_workspace_hint: bool
    max_current_proxy_ma: float | None
    tracker_freshness_s: float | None


class TwoSegmentPenprobeLookupDemoExperiment(BaseExperiment):
    """Two-segment nearest-reachable feedforward demo. NOT closed-loop control."""

    name = EXPERIMENT_NAME
    description = (
        "DEMO ONLY. Uses the 0B penprobe pose as a target and commands the all-8 servo "
        "goal ticks of the nearest map point from a previously-built workspace lookup "
        "map. Not closed-loop control, not validated for thesis/model-training data."
    )
    hardware_requirements = ExperimentHardwareRequirements(
        tracking_required=True,
        servo_required=True,
        registration_required=True,
        mock_compatible=True,
    )

    def __init__(self, config: TwoSegmentPenprobeLookupDemoConfig) -> None:
        super().__init__(config)
        self.config: TwoSegmentPenprobeLookupDemoConfig = config
        self._controller: TwoSegmentWorkspaceLookupController | None = None
        self._map_payload: dict[str, Any] | None = None
        self._trace_rows: list[_DemoTraceRow] = []
        self._last_commanded_ticks: dict[str, int] | None = None
        self._last_target_xyz: list[float] | None = None
        self._stop_reason: str | None = None
        self._sustained_overcurrent_window: list[tuple[float, float]] = []  # (monotonic_s, max_load_ma)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "TwoSegmentPenprobeLookupDemoExperiment":
        return cls(TwoSegmentPenprobeLookupDemoConfig.from_dict(payload))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self, session: ExperimentSession) -> None:
        if not self.config.map_path:
            raise RuntimeError("Two-segment penprobe lookup demo requires `map_path`.")
        map_path = Path(self.config.map_path)
        if not map_path.is_absolute():
            map_path = session.context.project_root / map_path
        if not map_path.exists():
            raise RuntimeError(f"Workspace lookup map not found at {map_path}.")
        self._map_payload = load_workspace_lookup_map(map_path)
        context = session.context.settings.robot.operating_context()
        controller_config = LookupControllerConfig(
            interpolation_mode=self.config.interpolation_mode if bool(self.config.allow_interpolation) else "nearest",
            knn=int(self.config.knn) if bool(self.config.allow_interpolation) else 1,
            nearest_distance_warning_mm=float(self.config.nearest_distance_warning_mm),
            max_nearest_distance_mm=float(self.config.max_nearest_distance_mm),
            expected_commanded_servo_ids=[int(v) for v in context.commanded_servo_ids],
            expected_bottom_segment_key=getattr(context, "bottom_segment_key", None) or None,
            expected_top_segment_key=getattr(context, "top_segment_key", None) or None,
            absolute_min_tick=int(self.config.absolute_min_tick),
            absolute_max_tick=int(self.config.absolute_max_tick),
            extra_per_servo_tick_headroom=int(self.config.extra_per_servo_tick_headroom),
        )
        self._controller = TwoSegmentWorkspaceLookupController(
            map_payload=self._map_payload, config=controller_config
        )

    def precheck(self, session: ExperimentSession) -> None:
        context = session.context.settings.robot.operating_context()
        if bool(self.config.require_dual_segment) and context.operating_mode != "dual_segment":
            raise RuntimeError(BLOCK_MESSAGE)
        expected_ids = [int(v) for v in context.expected_servo_ids]
        if expected_ids != [1, 2, 3, 4, 5, 6, 7, 8]:
            raise RuntimeError(
                f"Two-segment penprobe lookup demo requires all 8 servo IDs [1..8]; resolved {expected_ids}."
            )
        assembly_issues = list(getattr(context, "physical_assembly_issues", []) or [])
        if assembly_issues:
            raise RuntimeError("Invalid physical assembly: " + "; ".join(assembly_issues))
        if session.context.servo_service is None or (
            not bool(self.config.dry_run) and not bool(getattr(session.context.servo_service, "is_connected", False))
        ):
            raise RuntimeError("Two-segment penprobe lookup demo requires a connected ServoService.")
        if self._controller is None:
            raise RuntimeError("Workspace lookup map not loaded; call setup() first.")
        compatible, reasons = self._controller.check_compatibility()
        if not compatible:
            raise RuntimeError(
                "Map/runtime mismatch: " + "; ".join(reasons)
                + ". Build a fresh map after bottom/top reassembly or fix the servo IDs."
            )
        if bool(self.config.require_all_8_startup) and not bool(self.config.allow_servo_only_test_run):
            startup = self._resolve_startup_provenance(session=session)
            if not bool(startup.get("accepted_all_8_startup")):
                raise RuntimeError(
                    "Demo requires an accepted all-8 manual startup artifact. Run "
                    "two_segment_startup_validation first, or set allow_servo_only_test_run=true for a dry inspection."
                )
        session.set_stage("precheck", "passed", "two-segment penprobe lookup demo precheck passed.")

    def execute(self, session: ExperimentSession) -> None:
        if self._controller is None:
            raise RuntimeError("setup() did not initialize the lookup controller.")
        context = session.context.settings.robot.operating_context()
        loop_period_s = 1.0 / max(0.5, float(self.config.control_rate_hz))
        run_started_s = session.elapsed_s()
        iteration = 0
        last_tracker_age = None
        while True:
            session.raise_if_stop_requested()
            iteration += 1
            now_s = session.elapsed_s()
            if iteration > int(self.config.max_iterations):
                self._stop_reason = "max_iterations_reached"
                break
            if float(self.config.max_duration_s) > 0 and (now_s - run_started_s) > float(self.config.max_duration_s):
                self._stop_reason = "max_duration_reached"
                break

            target_xyz, target_age, tracker_stale = self._read_target_pose(session=session)
            decision = self._controller.decide(
                target_xyz_robot_mm=target_xyz,
                tracker_stale=tracker_stale,
            )
            max_current_ma, tracker_freshness_s = self._max_current_proxy(session=session)
            sustained_stop = self._update_sustained_overcurrent(now_s=now_s, max_current_ma=max_current_ma)
            command_sent = False
            command_skip_reason: str | None = None
            commanded_ticks_for_row: dict[str, int] | None = None
            if sustained_stop:
                command_skip_reason = "sustained_overcurrent_stop"
                self._stop_reason = "sustained_overcurrent_stop"
                self._record_trace_row(
                    iteration=iteration,
                    monotonic_time_s=now_s,
                    target_xyz=target_xyz,
                    target_age=target_age,
                    decision=decision,
                    command_sent=command_sent,
                    command_skip_reason=command_skip_reason,
                    commanded_ticks=commanded_ticks_for_row,
                    max_current=max_current_ma,
                    tracker_freshness=tracker_freshness_s,
                )
                break
            if not decision.command_allowed:
                command_skip_reason = decision.block_reason or "command_not_allowed"
            else:
                # Deadband: skip the write if neither the target nor the chosen
                # ticks have changed materially.
                if not self._target_moved_enough(target_xyz):
                    command_skip_reason = "target_within_deadband"
                elif self._ticks_within_deadband(decision.all_8_goal_ticks):
                    command_skip_reason = "command_within_deadband"
                else:
                    try:
                        commanded_ticks_for_row = self._issue_goal_ticks(
                            session=session, decision=decision
                        )
                        command_sent = True
                        self._last_commanded_ticks = dict(commanded_ticks_for_row)
                        self._last_target_xyz = list(decision.target_xyz_robot_mm)
                    except Exception as exc:
                        command_skip_reason = f"write_failed:{exc.__class__.__name__}:{exc}"
                        self._stop_reason = "write_failed"
            self._record_trace_row(
                iteration=iteration,
                monotonic_time_s=now_s,
                target_xyz=target_xyz,
                target_age=target_age,
                decision=decision,
                command_sent=command_sent,
                command_skip_reason=command_skip_reason,
                commanded_ticks=commanded_ticks_for_row,
                max_current=max_current_ma,
                tracker_freshness=tracker_freshness_s,
            )
            # Emit a lightweight per-iteration ExperimentTimeseriesSample so
            # the framework's sample-count gate is satisfied AND the run
            # appears in standard data-tab summaries. The sample carries the
            # same data as the trace row so downstream tools see one canonical
            # row stream per iteration.
            session.add_sample(
                self._build_session_sample(
                    session=session,
                    iteration=iteration,
                    target_xyz=target_xyz,
                    decision=decision,
                    command_sent=command_sent,
                    command_skip_reason=command_skip_reason,
                    commanded_ticks=commanded_ticks_for_row,
                    max_current=max_current_ma,
                    tracker_freshness=tracker_freshness_s,
                )
            )
            if self._stop_reason:
                break
            session.update_progress(
                iteration,
                int(self.config.max_iterations),
                {
                    "phase": "penprobe_lookup",
                    "command_sent": int(command_sent),
                    "nearest_distance_mm": (
                        round(float(decision.nearest_distance_mm), 3) if decision.nearest_distance_mm is not None else None
                    ),
                },
            )
            session.context.sleep_fn(loop_period_s)
        if self._stop_reason is None:
            self._stop_reason = "operator_stop"
        self._record_metrics(session=session, context=context)

    def write_outputs(self, session: ExperimentSession, paths, summary) -> None:
        write_two_segment_penprobe_lookup_demo_outputs(
            output_dir=paths.output_dir,
            metrics=dict(summary.experiment_metrics or {}),
            trace_rows=self._trace_rows,
            map_payload=self._map_payload or {},
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _read_target_pose(
        self, *, session: ExperimentSession
    ) -> tuple[list[float] | None, float | None, bool]:
        tracking_service = session.context.tracking_service
        if tracking_service is None:
            return None, None, True
        try:
            snapshot = getattr(tracking_service, "peek_snapshot", None)
            snapshot = snapshot() if callable(snapshot) else tracking_service.get_snapshot()
        except Exception:
            return None, None, True
        if snapshot is None:
            return None, None, True
        # Reuse the proven robot-frame helper from penprobe_chasing_demo so
        # the transform math is identical to existing single-segment chasing.
        from continuum_robot.experiments.penprobe_chasing_demo import _extract_chasing_pose_optional

        pose, err = _extract_chasing_pose_optional(
            snapshot,
            str(self.config.target_tool_id or "0B"),
            max_tracker_age_s=float(self.config.stale_tracker_timeout_s),
        )
        tracker_age = getattr(snapshot, "tracker_data_age_s", None)
        tracker_stale = bool(getattr(snapshot, "tracker_data_stale", False)) or pose is None
        if pose is None:
            return None, tracker_age, True
        return [float(v) for v in pose.position_mm], (float(tracker_age) if tracker_age is not None else None), tracker_stale

    def _issue_goal_ticks(
        self, *, session: ExperimentSession, decision: LookupDecision
    ) -> dict[str, int]:
        if decision.all_8_goal_ticks is None:
            raise RuntimeError("decision has no goal ticks")
        ticks_by_servo = {int(k): int(v) for k, v in decision.all_8_goal_ticks.items()}
        writer = getattr(session.context.servo_service, "_write_goal_positions", None)
        if not callable(writer):
            raise RuntimeError("ServoService all-8 raw goal writer is unavailable.")
        if bool(self.config.dry_run):
            return {str(k): int(v) for k, v in ticks_by_servo.items()}
        writer(ticks_by_servo)
        return {str(k): int(v) for k, v in ticks_by_servo.items()}

    def _target_moved_enough(self, target_xyz: list[float] | None) -> bool:
        if target_xyz is None:
            return False
        if self._last_target_xyz is None:
            return True
        prev = np.asarray(self._last_target_xyz, dtype=float)
        cur = np.asarray(target_xyz, dtype=float)
        return float(np.linalg.norm(cur - prev)) >= float(self.config.min_target_motion_mm)

    def _ticks_within_deadband(self, new_ticks: dict[str, int] | None) -> bool:
        if not new_ticks:
            return False
        if self._last_commanded_ticks is None:
            return False
        # The deadband is per-segment distance — if the goal ticks haven't
        # moved at all, skip the write. We use a tight integer comparison
        # rather than a deadband_mm because ticks are integer commands.
        try:
            return all(int(new_ticks[k]) == int(self._last_commanded_ticks.get(k, -1)) for k in new_ticks)
        except (KeyError, ValueError, TypeError):
            return False

    def _max_current_proxy(self, *, session: ExperimentSession) -> tuple[float | None, float | None]:
        servo_service = session.context.servo_service
        if servo_service is None or not bool(getattr(servo_service, "is_connected", False)):
            return None, None
        try:
            telemetry = servo_service.read_live_telemetry([int(v) for v in range(1, 9)])
        except Exception:
            return None, None
        max_load_ma: float | None = None
        tracker_freshness: float | None = None
        for item in telemetry.values():
            load = getattr(item, "present_current_ma", None)
            if load is None:
                continue
            try:
                val = abs(float(load))
            except (TypeError, ValueError):
                continue
            if max_load_ma is None or val > max_load_ma:
                max_load_ma = val
        try:
            tracking_service = session.context.tracking_service
            if tracking_service is not None:
                snapshot = (
                    tracking_service.peek_snapshot()
                    if hasattr(tracking_service, "peek_snapshot")
                    else tracking_service.get_snapshot()
                )
                tracker_freshness = getattr(snapshot, "tracker_data_age_s", None)
        except Exception:
            pass
        return max_load_ma, tracker_freshness

    def _update_sustained_overcurrent(self, *, now_s: float, max_current_ma: float | None) -> bool:
        if max_current_ma is None:
            self._sustained_overcurrent_window = []
            return False
        if max_current_ma >= float(self.config.hard_current_stop_ma):
            self._sustained_overcurrent_window.append((float(now_s), float(max_current_ma)))
            self._sustained_overcurrent_window = [
                (t, v)
                for (t, v) in self._sustained_overcurrent_window
                if (now_s - t) <= float(self.config.hard_current_stop_duration_s)
            ]
            if self._sustained_overcurrent_window:
                first_t, _ = self._sustained_overcurrent_window[0]
                return (now_s - first_t) >= float(self.config.hard_current_stop_duration_s)
        else:
            self._sustained_overcurrent_window = []
        return False

    def _record_trace_row(
        self,
        *,
        iteration: int,
        monotonic_time_s: float,
        target_xyz: list[float] | None,
        target_age: float | None,
        decision: LookupDecision,
        command_sent: bool,
        command_skip_reason: str | None,
        commanded_ticks: dict[str, int] | None,
        max_current: float | None,
        tracker_freshness: float | None,
    ) -> None:
        row = _DemoTraceRow(
            iteration=int(iteration),
            monotonic_time_s=float(monotonic_time_s),
            wall_time_utc=datetime.now(timezone.utc).isoformat(),
            target_xyz_robot_mm=list(target_xyz) if target_xyz is not None else None,
            target_age_s=float(target_age) if target_age is not None else None,
            selected_map_index=decision.selected_map_index,
            selected_point_xyz_robot_mm=list(decision.selected_point_xyz_robot_mm)
            if decision.selected_point_xyz_robot_mm is not None
            else None,
            nearest_distance_mm=decision.nearest_distance_mm,
            all_8_goal_ticks=dict(commanded_ticks) if commanded_ticks else None,
            command_sent=bool(command_sent),
            command_skip_reason=command_skip_reason,
            block_reason=decision.block_reason,
            safety_limiter_reasons=list(decision.safety_limiter_reasons),
            is_extrapolating=bool(decision.is_extrapolating),
            inside_workspace_hint=bool(decision.inside_workspace_hint),
            max_current_proxy_ma=float(max_current) if max_current is not None else None,
            tracker_freshness_s=float(tracker_freshness) if tracker_freshness is not None else None,
        )
        self._trace_rows.append(row)

    def _build_session_sample(
        self,
        *,
        session: ExperimentSession,
        iteration: int,
        target_xyz: list[float] | None,
        decision: LookupDecision,
        command_sent: bool,
        command_skip_reason: str | None,
        commanded_ticks: dict[str, int] | None,
        max_current: float | None,
        tracker_freshness: float | None,
    ) -> ExperimentTimeseriesSample:
        flags = ["demo_only", "not_closed_loop_validated", "two_segment_penprobe_lookup_demo"]
        if command_sent:
            flags.append("command_sent")
        else:
            flags.append("command_skipped")
        if decision.is_extrapolating:
            flags.append("extrapolating")
        return ExperimentTimeseriesSample(
            monotonic_time_s=session.elapsed_s(),
            wall_time_utc=datetime.now(timezone.utc).isoformat(),
            phase="penprobe_lookup",
            step_index=int(iteration),
            sample_index=int(iteration),
            commanded_motor_values=dict(commanded_ticks or {}),
            commanded_cable_deltas_cm=[0.0] * 8,
            tracker_frame_id=None,
            tool_ids_seen=[str(self.config.target_tool_id)],
            transform_validity={"target": "available" if target_xyz is not None else "missing"},
            pose_in_tracker_frame={},
            pose_in_robot_frame={
                "roles": {
                    self.config.target_tool_role: {
                        "translation_mm": list(target_xyz) if target_xyz is not None else [],
                    }
                }
            },
            status_flags=sorted(set(flags)),
            backend_health={
                "tracker_freshness_s": tracker_freshness,
                "max_current_proxy_ma": max_current,
            },
            extra={
                "record_kind": "two_segment_penprobe_lookup_demo_step",
                "demo_only": True,
                "not_closed_loop_validated": True,
                "iteration": int(iteration),
                "command_sent": bool(command_sent),
                "command_skip_reason": command_skip_reason,
                "block_reason": decision.block_reason,
                "safety_limiter_reasons": list(decision.safety_limiter_reasons),
                "nearest_distance_mm": decision.nearest_distance_mm,
                "selected_map_index": decision.selected_map_index,
                "selected_point_xyz_robot_mm": (
                    list(decision.selected_point_xyz_robot_mm)
                    if decision.selected_point_xyz_robot_mm is not None
                    else None
                ),
                "is_extrapolating": bool(decision.is_extrapolating),
                "inside_workspace_hint": bool(decision.inside_workspace_hint),
                "interpolation_mode": decision.interpolation_mode,
                "knn_used": decision.knn_used,
                "all_8_goal_ticks": dict(commanded_ticks) if commanded_ticks else None,
                "bottom_top_assignment": dict(decision.bottom_top_assignment),
            },
        )

    def _resolve_startup_provenance(self, *, session: ExperimentSession) -> dict[str, Any]:
        servo_service = session.context.servo_service
        if servo_service is None:
            return {}
        try:
            artifact = servo_service.neutral_calibration.load_calibration_artifact()
        except Exception:
            return {}
        robot = dict(getattr(artifact, "robot", {}) or {})
        return {
            "accepted_all_8_startup": bool(robot.get("startup_type") == "manual_two_segment_startup"),
            "startup_artifact_path": getattr(artifact, "path", None),
            "manual_two_segment_startup": bool(robot.get("manual_two_segment_startup", False)),
        }

    def _record_metrics(self, *, session: ExperimentSession, context) -> None:
        controller = self._controller
        assembly = controller.bottom_top_assignment if controller else {}
        metadata = dict(self._map_payload.get("metadata") or {}) if self._map_payload else {}
        commands_sent = sum(1 for row in self._trace_rows if row.command_sent)
        accepted_iterations = sum(
            1 for row in self._trace_rows if row.command_sent or row.command_skip_reason in {"target_within_deadband", "command_within_deadband"}
        )
        nearest_distances = [
            row.nearest_distance_mm for row in self._trace_rows if row.nearest_distance_mm is not None
        ]
        max_current_seen = [row.max_current_proxy_ma for row in self._trace_rows if row.max_current_proxy_ma is not None]
        session.set_metric("schema_version", DEMO_SCHEMA_VERSION)
        session.set_metric("demo_only", True)
        session.set_metric("not_closed_loop_validated", True)
        session.set_metric("valid_for_model_training", False)
        session.set_metric("valid_for_thesis_repeatability", False)
        session.set_metric("controlled_point", "map distal pose (from previously collected dataset)")
        session.set_metric("target_point", f"{self.config.target_tool_id} tool origin in robot frame")
        session.set_metric("physical_tip_chasing", False)
        session.set_metric("stop_reason", self._stop_reason or "operator_stop")
        session.set_metric("iterations", len(self._trace_rows))
        session.set_metric("commands_sent", commands_sent)
        session.set_metric("commands_skipped_in_deadband", accepted_iterations - commands_sent)
        session.set_metric("commands_blocked", sum(1 for row in self._trace_rows if row.block_reason))
        session.set_metric(
            "nearest_distance_summary_mm",
            {
                "min": float(min(nearest_distances)) if nearest_distances else None,
                "max": float(max(nearest_distances)) if nearest_distances else None,
                "mean": float(sum(nearest_distances) / len(nearest_distances)) if nearest_distances else None,
            },
        )
        session.set_metric(
            "max_current_proxy_summary_ma",
            {
                "min": float(min(max_current_seen)) if max_current_seen else None,
                "max": float(max(max_current_seen)) if max_current_seen else None,
            },
        )
        session.set_metric("map_metadata", metadata)
        session.set_metric("bottom_top_assignment", dict(assembly or {}))
        session.set_metric("operating_mode", context.operating_mode)
        session.set_metric("commanded_servo_ids", list(context.commanded_servo_ids))
        session.set_metric("run_trust_mode", self.config.run_trust_mode)
        session.set_metric("target_tool_id", str(self.config.target_tool_id))


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_two_segment_penprobe_lookup_demo_outputs(
    *,
    output_dir: Path,
    metrics: dict[str, Any],
    trace_rows: list[_DemoTraceRow],
    map_payload: dict[str, Any],
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {
        "summary_text": output_dir / "two_segment_penprobe_lookup_demo_summary.txt",
        "trace_csv": output_dir / "demo_trace.csv",
        "trace_jsonl": output_dir / "demo_trace.jsonl",
        "map_metadata_json": output_dir / "map_used.json",
    }
    _write_summary_text(paths["summary_text"], metrics)
    _write_trace_csv(paths["trace_csv"], trace_rows)
    _write_trace_jsonl(paths["trace_jsonl"], trace_rows)
    map_meta = dict(map_payload.get("metadata") or {})
    paths["map_metadata_json"].write_text(json.dumps(map_meta, indent=2), encoding="utf-8")
    try:
        from continuum_robot.experiments.plotting import create_figure, save_figure, style_axes

        fig_path = output_dir / "penprobe_lookup_demo_path_report.png"
        _write_path_figure(fig_path, trace_rows, create_figure, save_figure, style_axes)
        paths["path_figure"] = fig_path
    except Exception:
        pass
    return paths


def _write_summary_text(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "Two-Segment Penprobe Lookup Demo (DEMO ONLY)",
        f"schema_version: {metrics.get('schema_version', DEMO_SCHEMA_VERSION)}",
        f"demo_only: {metrics.get('demo_only', True)}",
        f"not_closed_loop_validated: {metrics.get('not_closed_loop_validated', True)}",
        f"valid_for_model_training: {metrics.get('valid_for_model_training', False)}",
        f"valid_for_thesis_repeatability: {metrics.get('valid_for_thesis_repeatability', False)}",
        "",
        f"controlled_point: {metrics.get('controlled_point', '')}",
        f"target_point: {metrics.get('target_point', '')}",
        f"physical_tip_chasing: {metrics.get('physical_tip_chasing', False)}",
        "",
        f"stop_reason: {metrics.get('stop_reason', '')}",
        f"iterations: {metrics.get('iterations', 0)}",
        f"commands_sent: {metrics.get('commands_sent', 0)}",
        f"commands_skipped_in_deadband: {metrics.get('commands_skipped_in_deadband', 0)}",
        f"commands_blocked: {metrics.get('commands_blocked', 0)}",
        "",
        "Nearest-distance summary (mm):",
        f"  min:  {(metrics.get('nearest_distance_summary_mm') or {}).get('min')}",
        f"  max:  {(metrics.get('nearest_distance_summary_mm') or {}).get('max')}",
        f"  mean: {(metrics.get('nearest_distance_summary_mm') or {}).get('mean')}",
        "",
        f"target_tool_id: {metrics.get('target_tool_id', '')}",
        f"bottom_top_assignment: {metrics.get('bottom_top_assignment', {})}",
        "",
        "This run is a presentation-only demo. It is NOT a closed-loop",
        "controller and is NOT valid for thesis or model-training evaluation.",
    ]
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _write_trace_csv(path: Path, trace_rows: list[_DemoTraceRow]) -> None:
    fields = [
        "iteration",
        "monotonic_time_s",
        "wall_time_utc",
        "target_x_mm",
        "target_y_mm",
        "target_z_mm",
        "target_age_s",
        "selected_map_index",
        "selected_x_mm",
        "selected_y_mm",
        "selected_z_mm",
        "nearest_distance_mm",
        "command_sent",
        "command_skip_reason",
        "block_reason",
        "is_extrapolating",
        "inside_workspace_hint",
        "max_current_proxy_ma",
        "tracker_freshness_s",
    ] + [f"goal_tick_servo_{i}" for i in range(1, 9)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in trace_rows:
            tgt = row.target_xyz_robot_mm or [None, None, None]
            sel = row.selected_point_xyz_robot_mm or [None, None, None]
            ticks = row.all_8_goal_ticks or {}
            csv_row = {
                "iteration": row.iteration,
                "monotonic_time_s": row.monotonic_time_s,
                "wall_time_utc": row.wall_time_utc,
                "target_x_mm": tgt[0],
                "target_y_mm": tgt[1],
                "target_z_mm": tgt[2],
                "target_age_s": row.target_age_s,
                "selected_map_index": row.selected_map_index,
                "selected_x_mm": sel[0],
                "selected_y_mm": sel[1],
                "selected_z_mm": sel[2],
                "nearest_distance_mm": row.nearest_distance_mm,
                "command_sent": int(row.command_sent),
                "command_skip_reason": row.command_skip_reason or "",
                "block_reason": row.block_reason or "",
                "is_extrapolating": int(row.is_extrapolating),
                "inside_workspace_hint": int(row.inside_workspace_hint),
                "max_current_proxy_ma": row.max_current_proxy_ma,
                "tracker_freshness_s": row.tracker_freshness_s,
            }
            for sid in range(1, 9):
                csv_row[f"goal_tick_servo_{sid}"] = ticks.get(str(sid)) if ticks else None
            writer.writerow(csv_row)


def _write_trace_jsonl(path: Path, trace_rows: list[_DemoTraceRow]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in trace_rows:
            handle.write(
                json.dumps(
                    {
                        "iteration": row.iteration,
                        "monotonic_time_s": row.monotonic_time_s,
                        "wall_time_utc": row.wall_time_utc,
                        "target_xyz_robot_mm": row.target_xyz_robot_mm,
                        "target_age_s": row.target_age_s,
                        "selected_map_index": row.selected_map_index,
                        "selected_point_xyz_robot_mm": row.selected_point_xyz_robot_mm,
                        "nearest_distance_mm": row.nearest_distance_mm,
                        "all_8_goal_ticks": row.all_8_goal_ticks,
                        "command_sent": row.command_sent,
                        "command_skip_reason": row.command_skip_reason,
                        "block_reason": row.block_reason,
                        "safety_limiter_reasons": row.safety_limiter_reasons,
                        "is_extrapolating": row.is_extrapolating,
                        "inside_workspace_hint": row.inside_workspace_hint,
                        "max_current_proxy_ma": row.max_current_proxy_ma,
                        "tracker_freshness_s": row.tracker_freshness_s,
                    }
                )
                + "\n"
            )


def _write_path_figure(path: Path, trace_rows: list[_DemoTraceRow], create_figure, save_figure, style_axes) -> None:
    fig, ax = create_figure(size="square")
    if not trace_rows:
        ax.text(0.5, 0.5, "No trace rows captured.", ha="center", va="center", transform=ax.transAxes)
        style_axes(ax, title="Penprobe Lookup Demo Path (DEMO)", xlabel="X (mm)", ylabel="Y (mm)")
        save_figure(fig, path)
        return
    targets = [row.target_xyz_robot_mm for row in trace_rows if row.target_xyz_robot_mm is not None]
    selecteds = [row.selected_point_xyz_robot_mm for row in trace_rows if row.selected_point_xyz_robot_mm is not None]
    if targets:
        ax.plot([t[0] for t in targets], [t[1] for t in targets], "-o", markersize=2, label="target (0B)")
    if selecteds:
        ax.plot([s[0] for s in selecteds], [s[1] for s in selecteds], "-x", markersize=3, label="selected map point")
    style_axes(ax, title="Penprobe Lookup Demo Path (DEMO ONLY)", xlabel="X (mm)", ylabel="Y (mm)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8)
    save_figure(fig, path)
