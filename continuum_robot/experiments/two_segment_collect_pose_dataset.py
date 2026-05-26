"""Conservative two-segment command/pose dataset MVP.

This experiment records structured Segment A/B commands, all-8 servo telemetry,
and optional tracking observations. It deliberately does not implement
two-segment kinematics, learned control, chasing, or automatic pretension.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from continuum_robot.experiments.framework import BaseExperiment, ExperimentHardwareRequirements, ExperimentSession
from continuum_robot.experiments.plotting import color, create_figure, save_figure, style_axes
from continuum_robot.experiments.schemas import ExperimentTimeseriesSample
from continuum_robot.two_segment import (
    TwoSegmentCommand,
    build_two_segment_foundation_metadata,
    two_segment_command_schema,
)
from continuum_robot.tracking.two_segment_roles import (
    resolve_two_segment_tracking_roles,
    role_config_records,
)


EXPERIMENT_NAME = "two_segment_collect_pose_command_dataset"
TWO_SEGMENT_DATASET_SCHEMA_VERSION = "two_segment_collect_pose_dataset_v1"
BLOCK_MESSAGE = (
    "Two-segment dataset collection requires operating_mode=dual_segment and an all-8 startup foundation. "
    "This experiment collects structured two-segment command/pose data; it does not implement two-segment control."
)
# Bottom/top-aware schedules. The legacy `single_axis_micro`/`segment_isolation`/
# `small_combined` schedules remain supported for backwards compatibility, but
# the canonical thesis-grade schedules are bottom/top-oriented.
SCHEDULE_TYPES = {
    "zero",
    "single_axis_micro",
    "segment_isolation",
    "small_combined",
    "bottom_only_sweep",
    "top_only_sweep",
    "workspace_coverage",
    "random_babble",
    "structured_grid",
    "mixed_training",
}
# Schedules that benefit from the higher-throughput long-run loop.
LONG_RUN_FRIENDLY_SCHEDULES = {
    "workspace_coverage",
    "random_babble",
    "structured_grid",
    "mixed_training",
    "bottom_only_sweep",
    "top_only_sweep",
}


@dataclass
class TwoSegmentCollectPoseDatasetConfig:
    """Configuration for a two-segment dataset collection run.

    The default `max_segment_displacement_cm` is conservative; thesis-grade runs
    are expected to ramp up to ±1.0 cm once the safety limits and motion are
    validated on hardware. The configured command amplitude is honored at the
    schedule level; safety still gates tick deltas via
    `max_tick_delta_from_startup` and `SafetyConfig`.
    """

    schedule_type: str = "single_axis_micro"
    dry_run: bool = True
    # Bounded tendon displacement amplitude. The legacy `max_segment_displacement_mm`
    # field is still accepted on input but is converted to cm; one canonical
    # field avoids the silent ÷10 cap that the original MVP had.
    max_segment_displacement_cm: float = 0.25
    max_tick_delta_from_startup: int = 320
    samples_per_pattern: int = 1
    capture_repeats: int = 1
    # 2026-05-25 default: 3.0 s settle between commanded patterns. Tendons +
    # constant-curvature mechanics have a few hundred ms of relaxation time
    # after each new goal-position write; 3 s is conservatively above that
    # window and gives the tracker time to converge on a stable pose. Operator
    # can shorten in config when throughput matters (random_babble / large
    # workspace_coverage) but the default is the safe-bench-day value.
    settle_time_s: float = 3.0
    # Top-segment tendon routing compensation. When True (default), every
    # actuator-level write adds the bottom-segment per-tendon command to the
    # top-segment per-tendon command before computing goal ticks. This is the
    # physical reality of a stacked two-segment continuum robot whose top
    # tendons route through the bottom segment: bottom-segment displacement
    # drags top tendons by the same amount, so the top servo must pull
    # `top_intended + bottom_intended` to make the top achieve its intended
    # displacement RELATIVE to the intermediate plate. The user-intended
    # command (`TwoSegmentCommand.segment_a` / `segment_b`) is recorded as
    # the modeling feature; the compensated command is recorded under
    # `extra.servo_flat_command_cm` for full audit. Setting this False (only
    # for benches with independently routed tendons) restores the legacy
    # passthrough behaviour.
    top_segment_tendon_routing_compensation: bool = True
    allow_servo_only_test_run: bool = True
    run_trust_mode: str = "servo_only"
    capture_tracker_snapshot: bool = True
    requested_tool_roles: dict[str, str] = field(default_factory=dict)
    run_label: str = ""
    dataset_tag: str = ""
    # Long-run loop. When `continue_until_valid_samples` is true the experiment
    # cycles through the schedule until at least `target_valid_sample_count`
    # accepted samples have been collected (or a budget is exhausted).
    continue_until_valid_samples: bool = False
    target_valid_sample_count: int = 0
    max_total_packet_failures: int | None = None
    max_consecutive_packet_failures: int | None = None
    max_total_dropped_samples: int | None = None
    long_run_recovery_enabled: bool = False
    drop_sample_on_transport_error: bool = True
    pattern_count_budget: int = 0
    # Random schedules use this seed for reproducibility.
    random_seed: int = 0
    # Optional grid density for `structured_grid` / `workspace_coverage`.
    grid_points_per_axis: int = 5
    # Optional command ramping. When set, large jumps between consecutive
    # schedule steps are split into intermediate sub-commands of at most this
    # tendon-displacement step size (in cm). Only the final target generates a
    # sample; intermediate ramp steps preserve final range while reducing peak
    # current draw / tendon shock. ``None`` disables ramping.
    command_ramp_step_cm: float | None = 0.05
    command_ramp_settle_time_s: float = 0.10
    # Two-segment current/load policy (lock-in):
    #   - warning around 800 mA per servo (descriptive, not a stop)
    #   - hard stop around 1200 mA per servo ONLY if SUSTAINED
    #   - sustained duration target = ~2 seconds (operator-tunable)
    #   - never hard-stop on a single transient spike
    #
    # The thresholds below feed the post-run current/load summary. Live
    # hard-stop during motion is still handled by the lower-level servo
    # safety guard configured in SafetyConfig; this experiment-level policy
    # supplements that with run-aggregate visibility.
    #
    # `sustained_overcurrent_sample_count` is the number of consecutive
    # samples above the warning threshold needed to flag a servo as
    # sustained-jam. The "2 seconds" target translates to:
    #   capture rate ~5 Hz  -> ~10 samples
    #   capture rate ~10 Hz -> ~20 samples
    # The default (3) is a transient-rejecting floor; bump it for slower
    # captures, or leave at 3 for the typical bench-day workspace_coverage
    # cadence where 2 seconds tends to be over-protective.
    current_warning_ma: int = 800
    sustained_overcurrent_ma: int = 1200
    sustained_overcurrent_sample_count: int = 3
    # Trusted dual-segment sessions should explicitly confirm which physical
    # segment is bottom/proximal and which is top/distal before collecting data.
    physical_assembly_confirmed_by_operator: bool = False
    physical_assembly_confirmed_at_utc: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "TwoSegmentCollectPoseDatasetConfig":
        payload = dict(payload or {})
        requested_roles = payload.get("requested_tool_roles") or payload.get("tool_roles") or {}
        # Backwards-compat: accept legacy `max_segment_displacement_mm` (mm) and
        # convert to cm.
        if "max_segment_displacement_cm" in payload:
            amp_cm = max(0.0, float(payload.get("max_segment_displacement_cm", 0.25)))
        elif "max_segment_displacement_mm" in payload:
            amp_cm = max(0.0, float(payload.get("max_segment_displacement_mm", 0.1))) / 10.0
        else:
            amp_cm = 0.25
        if "command_ramp_step_cm" not in payload:
            command_ramp_step_cm = 0.05
        elif payload.get("command_ramp_step_cm") in (None, ""):
            command_ramp_step_cm = None
        else:
            command_ramp_step_cm = max(0.0, float(payload.get("command_ramp_step_cm")))
        return cls(
            schedule_type=str(payload.get("schedule_type", "single_axis_micro") or "single_axis_micro").strip().lower(),
            dry_run=bool(payload.get("dry_run", True)),
            max_segment_displacement_cm=amp_cm,
            max_tick_delta_from_startup=max(0, int(payload.get("max_tick_delta_from_startup", 320))),
            samples_per_pattern=max(1, int(payload.get("samples_per_pattern", payload.get("samples_per_command", 1)))),
            capture_repeats=max(1, int(payload.get("capture_repeats", 1))),
            settle_time_s=max(0.0, float(payload.get("settle_time_s", 3.0))),
            allow_servo_only_test_run=bool(payload.get("allow_servo_only_test_run", True)),
            run_trust_mode=str(payload.get("run_trust_mode", "servo_only") or "servo_only").strip().lower(),
            capture_tracker_snapshot=bool(payload.get("capture_tracker_snapshot", True)),
            requested_tool_roles={str(key).upper(): str(value) for key, value in dict(requested_roles or {}).items()},
            run_label=str(payload.get("run_label", "") or ""),
            dataset_tag=str(payload.get("dataset_tag", "") or ""),
            continue_until_valid_samples=bool(payload.get("continue_until_valid_samples", False)),
            target_valid_sample_count=max(0, int(payload.get("target_valid_sample_count", 0) or 0)),
            max_total_packet_failures=(
                None if payload.get("max_total_packet_failures") in (None, "")
                else max(0, int(payload.get("max_total_packet_failures")))
            ),
            max_consecutive_packet_failures=(
                None if payload.get("max_consecutive_packet_failures") in (None, "")
                else max(0, int(payload.get("max_consecutive_packet_failures")))
            ),
            max_total_dropped_samples=(
                None if payload.get("max_total_dropped_samples") in (None, "")
                else max(0, int(payload.get("max_total_dropped_samples")))
            ),
            long_run_recovery_enabled=bool(payload.get("long_run_recovery_enabled", False)),
            drop_sample_on_transport_error=bool(payload.get("drop_sample_on_transport_error", True)),
            pattern_count_budget=max(0, int(payload.get("pattern_count_budget", 0) or 0)),
            random_seed=int(payload.get("random_seed", 0) or 0),
            grid_points_per_axis=max(1, int(payload.get("grid_points_per_axis", 5) or 5)),
            command_ramp_step_cm=command_ramp_step_cm,
            command_ramp_settle_time_s=max(0.0, float(payload.get("command_ramp_settle_time_s", 0.10))),
            current_warning_ma=max(1, int(payload.get("current_warning_ma", payload.get("max_current_warning_ma", 800)) or 800)),
            sustained_overcurrent_ma=max(
                1,
                int(payload.get("sustained_overcurrent_ma", payload.get("sustained_jam_current_ma", 1200)) or 1200),
            ),
            sustained_overcurrent_sample_count=max(
                1,
                int(payload.get("sustained_overcurrent_sample_count", payload.get("sustained_jam_cycles", 3)) or 3),
            ),
            physical_assembly_confirmed_by_operator=bool(
                payload.get("physical_assembly_confirmed_by_operator", False)
                or payload.get("physical_assembly_confirmed", False)
            ),
            physical_assembly_confirmed_at_utc=str(payload.get("physical_assembly_confirmed_at_utc", "") or ""),
            top_segment_tendon_routing_compensation=bool(
                payload.get("top_segment_tendon_routing_compensation", True)
            ),
        )

    @property
    def max_segment_displacement_mm(self) -> float:
        return float(self.max_segment_displacement_cm) * 10.0


@dataclass(frozen=True)
class TwoSegmentCommandStep:
    index: int
    label: str
    phase: str
    command: TwoSegmentCommand


class TwoSegmentCollectPoseCommandDatasetExperiment(BaseExperiment):
    """Collect lower-risk two-segment command/pose data in dual_segment mode."""

    name = EXPERIMENT_NAME
    description = (
        "Conservative two-segment command/pose dataset MVP for dual_segment mode. "
        "Stores Segment A/B commands and optional pose labels; no two-segment control is implemented."
    )
    hardware_requirements = ExperimentHardwareRequirements(
        tracking_required=False,
        servo_required=True,
        registration_required=False,
        mock_compatible=True,
    )

    def __init__(self, config: TwoSegmentCollectPoseDatasetConfig) -> None:
        super().__init__(config)
        self.config: TwoSegmentCollectPoseDatasetConfig = config
        self._startup_ticks_by_servo: dict[int, int] = {}
        self._startup_provenance: dict[str, Any] = {}
        self._command_steps: list[TwoSegmentCommandStep] = []
        self._sample_failure_events: list[dict[str, Any]] = []
        self._long_run_health: dict[str, Any] = {}
        self._transport_recovery_report: dict[str, Any] = {}
        # Tracks the last-issued flat command (in cm) for ramping. Starts at
        # zero (startup-relative zero), updated to the target after each
        # successful issue.
        self._previous_flat_cm: list[float] = [0.0] * 8
        self._ramp_substep_count: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "TwoSegmentCollectPoseCommandDatasetExperiment":
        return cls(TwoSegmentCollectPoseDatasetConfig.from_dict(payload))

    def setup(self, session: ExperimentSession) -> None:
        context = session.context.settings.robot.operating_context()
        self._startup_ticks_by_servo, self._startup_provenance = _resolve_startup_ticks(
            session=session,
            config=self.config,
            expected_ids=[int(value) for value in context.expected_servo_ids],
        )
        self._command_steps = build_two_segment_command_schedule(self.config, context=context)

    def precheck(self, session: ExperimentSession) -> None:
        context = session.context.settings.robot.operating_context()
        if context.operating_mode != "dual_segment":
            raise RuntimeError(BLOCK_MESSAGE)
        # Honest bottom/top assembly is required; otherwise the schedule
        # builder silently falls back to defaults and the resulting data
        # misrepresents which segment was commanded.
        assembly_issues = list(getattr(context, "physical_assembly_issues", []) or [])
        if assembly_issues:
            raise RuntimeError(
                "Two-segment dataset collection requires a valid bottom/top physical assembly. "
                f"Issues: {'; '.join(assembly_issues)}"
            )
        expected_ids = [int(value) for value in context.expected_servo_ids]
        commanded_ids = [int(value) for value in context.commanded_servo_ids]
        if expected_ids != [1, 2, 3, 4, 5, 6, 7, 8] or commanded_ids != expected_ids:
            raise RuntimeError(
                "Two-segment dataset collection requires all 8 expected/commanded servo IDs [1,2,3,4,5,6,7,8]; "
                f"resolved expected={expected_ids}, commanded={commanded_ids}."
            )
        if session.context.servo_service is None or (not self.config.dry_run and not session.context.servo_service.is_connected):
            raise RuntimeError("Two-segment dataset collection requires ServoService for all-8 startup and telemetry.")
        if self.config.schedule_type not in SCHEDULE_TYPES:
            raise RuntimeError(f"Unknown two-segment schedule_type '{self.config.schedule_type}'. Expected one of {sorted(SCHEDULE_TYPES)}.")
        trusted_requested = _trusted_dataset_requested(self.config)
        assembly = dict(getattr(context, "physical_assembly", {}) or {})
        assembly_confirmed = bool(
            self.config.physical_assembly_confirmed_by_operator
            or assembly.get("confirmed_by_operator", False)
        )
        if trusted_requested and not assembly_confirmed:
            raise RuntimeError(
                "Trusted two-segment dataset collection requires operator confirmation of the bottom/top physical assembly. "
                "Confirm Bottom/proximal and Top/distal in the GUI before collecting thesis-intended data."
            )
        if trusted_requested and not bool(self._startup_provenance.get("accepted_all_8_startup")):
            raise RuntimeError(
                "Trusted two-segment dataset collection requires an accepted all-8 manual startup artifact. "
                "Use two_segment_startup_validation first or run with allow_servo_only_test_run=true and run_trust_mode=servo_only."
            )
        if trusted_requested:
            _observation, pose_fields = _two_segment_pose_observation(
                session=session,
                config=self.config,
                run_trust_mode=_run_trust_mode(self.config),
            )
            missing_required = list(pose_fields.get("missing_required_roles", []) or [])
            if "distal_tip" in missing_required:
                raise RuntimeError(
                    "Trusted two-segment dataset collection requires a live distal_tip tracker role. "
                    "Configure and track the distal/top tool before collecting thesis-intended data."
                )
            if not pose_fields_have_distal_robot_frame(pose_fields):
                raise RuntimeError(
                    "Trusted two-segment dataset collection requires a live robot-frame distal_tip pose. "
                    "Load an accepted registration/runtime-tip path so T_robot_tip is available before collecting model-training data."
                )
        violations = _command_limit_violations(
            steps=self._command_steps,
            context=context,
            startup_ticks_by_servo=self._startup_ticks_by_servo,
            max_tick_delta=int(self.config.max_tick_delta_from_startup),
            mapper=session.context.servo_service.mapper,
            apply_top_routing_compensation=bool(self.config.top_segment_tendon_routing_compensation),
        )
        if violations:
            raise RuntimeError("Two-segment command schedule exceeds configured tick limits: " + "; ".join(violations))
        if not self.config.dry_run:
            telemetry = session.context.servo_service.read_live_telemetry(expected_ids)
            missing = [servo_id for servo_id in expected_ids if telemetry.get(servo_id) is None or telemetry[servo_id].present_position is None]
            if missing:
                raise RuntimeError(f"Fresh all-8 servo positions are required; missing position telemetry for {missing}.")
        session.set_stage("precheck", "passed", "dual_segment two-segment dataset precheck passed.")

    def execute(self, session: ExperimentSession) -> None:
        context = session.context.settings.robot.operating_context()
        run_trust_mode = _run_trust_mode(self.config)
        role_configs = _effective_role_configs(session=session, config=self.config)
        tool_roles = _tool_roles_from_configs(role_configs)
        foundation = build_two_segment_foundation_metadata(context, tool_roles=tool_roles)
        startup_provenance = dict(self._startup_provenance)
        startup_provenance.setdefault("startup_ticks_by_servo", {str(key): int(value) for key, value in sorted(self._startup_ticks_by_servo.items())})
        # Long-run state
        continue_until_valid = bool(self.config.continue_until_valid_samples) and int(self.config.target_valid_sample_count or 0) > 0
        target_valid = int(self.config.target_valid_sample_count or 0)
        max_total_failures = self.config.max_total_packet_failures
        max_consec_failures = self.config.max_consecutive_packet_failures
        max_total_dropped = self.config.max_total_dropped_samples
        accepted = 0
        rejected = 0
        command_failures = 0
        transport_failures = 0
        consecutive_transport_failures = 0
        progress = 0
        cycle = 0
        sample_failure_events: list[dict[str, Any]] = []
        total_target = (
            int(target_valid)
            if continue_until_valid
            else len(self._command_steps) * max(1, int(self.config.capture_repeats)) * max(1, int(self.config.samples_per_pattern))
        )
        stop_reason: str = ""
        repeats_to_run = max(1, int(self.config.capture_repeats))
        while True:
            if continue_until_valid and accepted >= int(target_valid):
                stop_reason = "target_valid_sample_count_reached"
                break
            cycle += 1
            for step in self._command_steps:
                session.raise_if_stop_requested()
                if continue_until_valid and accepted >= int(target_valid):
                    stop_reason = "target_valid_sample_count_reached"
                    break
                command_result = self._issue_command(session=session, command=step.command)
                command_ok = bool(command_result.get("success", False))
                if not command_ok:
                    command_failures += 1
                    sample_failure_events.append({
                        "phase": str(step.phase),
                        "step_index": int(step.index),
                        "cycle": int(cycle),
                        "failure_kind": "command_failed",
                        "message": str(command_result.get("command_message") or ""),
                    })
                if float(self.config.settle_time_s) > 0.0:
                    session.context.sleep_fn(float(self.config.settle_time_s))
                for sample_index in range(max(1, int(self.config.samples_per_pattern))):
                    try:
                        sample = self._capture_sample(
                            session=session,
                            step=step,
                            command_result=command_result,
                            repeat_index=int(cycle - 1),
                            sample_index=sample_index,
                            run_trust_mode=run_trust_mode,
                            foundation=foundation,
                            startup_provenance=startup_provenance,
                        )
                    except Exception as exc:
                        transport_failures += 1
                        consecutive_transport_failures += 1
                        sample_failure_events.append({
                            "phase": str(step.phase),
                            "step_index": int(step.index),
                            "cycle": int(cycle),
                            "failure_kind": "sample_capture_error",
                            "message": str(exc),
                        })
                        if max_consec_failures is not None and consecutive_transport_failures > int(max_consec_failures):
                            stop_reason = f"max_consecutive_packet_failures_exceeded:{consecutive_transport_failures}"
                            break
                        if max_total_failures is not None and transport_failures > int(max_total_failures):
                            stop_reason = f"max_total_packet_failures_exceeded:{transport_failures}"
                            break
                        if not bool(self.config.long_run_recovery_enabled):
                            raise
                        continue
                    consecutive_transport_failures = 0
                    session.add_sample(sample)
                    accepted_flag = bool(sample.extra.get("capture_accepted"))
                    accepted += int(accepted_flag)
                    rejected += int(not accepted_flag)
                    # Avoid double-counting: when the command write itself failed,
                    # the `command_failed` event already captured the root cause
                    # for this pattern. The downstream sample is rejected as a
                    # consequence, so a second `capture_rejected` event would
                    # double-count in the events log. Per-sample counters
                    # (rejected, sample.extra.capture_accepted) remain accurate.
                    if not accepted_flag and command_ok:
                        sample_failure_events.append({
                            "phase": str(step.phase),
                            "step_index": int(step.index),
                            "cycle": int(cycle),
                            "failure_kind": "capture_rejected",
                            "message": str(sample.extra.get("capture_rejection_reason") or ""),
                        })
                    if max_total_dropped is not None and rejected > int(max_total_dropped):
                        stop_reason = f"max_total_dropped_samples_exceeded:{rejected}"
                        break
                    progress += 1
                    progress_total = max(progress, total_target)
                    session.update_progress(
                        progress,
                        progress_total,
                        {
                            "phase": step.phase,
                            "step_index": int(step.index),
                            "accepted_samples": int(accepted),
                            "rejected_samples": int(rejected),
                            "cycle": int(cycle),
                        },
                    )
                    if continue_until_valid and accepted >= int(target_valid):
                        stop_reason = "target_valid_sample_count_reached"
                        break
                if stop_reason:
                    break
            if stop_reason:
                break
            if not continue_until_valid:
                if cycle >= repeats_to_run:
                    stop_reason = "scheduled_repeat_count_reached"
                    break
        self._sample_failure_events = sample_failure_events
        self._long_run_health = {
            "schema_version": "two_segment_long_run_health_v1",
            "stop_reason": stop_reason,
            "cycles_completed": int(cycle),
            "command_failures": int(command_failures),
            "transport_failures": int(transport_failures),
            "consecutive_transport_failures_at_stop": int(consecutive_transport_failures),
            "rejected_samples": int(rejected),
            "accepted_samples": int(accepted),
            "target_valid_sample_count": int(target_valid) if continue_until_valid else None,
            "continue_until_valid_samples": bool(continue_until_valid),
            "max_total_packet_failures": max_total_failures,
            "max_consecutive_packet_failures": max_consec_failures,
            "max_total_dropped_samples": max_total_dropped,
            "long_run_recovery_enabled": bool(self.config.long_run_recovery_enabled),
            "command_ramp_step_cm": self.config.command_ramp_step_cm,
            "command_ramp_substeps_total": int(self._ramp_substep_count),
        }
        self._transport_recovery_report = {
            "schema_version": "two_segment_transport_recovery_v1",
            "long_run_recovery_enabled": bool(self.config.long_run_recovery_enabled),
            "transport_failures": int(transport_failures),
            "events": list(sample_failure_events),
            "stop_reason": stop_reason,
        }
        pose_summary = _pose_label_summary(session.samples)
        data_quality_warnings = _data_quality_warnings(
            config=self.config,
            pose_summary=pose_summary,
            startup_provenance=startup_provenance,
        )
        session.set_metric("dataset_schema_version", TWO_SEGMENT_DATASET_SCHEMA_VERSION)
        session.set_metric("dataset_type", "two_segment_collect_pose_command_dataset")
        session.set_metric("schedule_type", str(self.config.schedule_type))
        session.set_metric("command_step_count", len(self._command_steps))
        session.set_metric("capture_repeats", int(self.config.capture_repeats))
        session.set_metric("samples_per_pattern", int(self.config.samples_per_pattern))
        session.set_metric("accepted_sample_count", int(accepted))
        session.set_metric("rejected_sample_count", int(rejected))
        session.set_metric("command_failure_count", int(command_failures))
        session.set_metric("run_trust_mode", run_trust_mode)
        valid_for_ann_training = _valid_for_two_segment_model_training(
            self.config,
            pose_summary,
            startup_provenance,
            command_failures,
        )
        session.set_metric("valid_for_two_segment_model_training", valid_for_ann_training)
        session.set_metric("valid_for_two_segment_ann_training", valid_for_ann_training)
        session.set_metric("label_mode_used", _dataset_label_mode_from_pose_summary(pose_summary))
        session.set_metric("valid_for_model_training", False)
        session.set_metric("valid_for_thesis_repeatability", False)
        session.set_metric("data_quality_warnings", data_quality_warnings)
        session.set_metric("two_segment_command_schema", two_segment_command_schema(context))
        session.set_metric("two_segment_foundation", foundation)
        session.set_metric("two_segment_tracking_role_config", role_config_records(role_configs))
        session.set_metric("startup_artifact_provenance", startup_provenance)
        session.set_metric("pose_label_summary", pose_summary)
        session.set_metric("distal_only", bool(pose_summary.get("distal_pose_sample_count", 0) and not pose_summary.get("intermediate_pose_sample_count", 0)))
        session.set_metric("includes_intermediate_pose", bool(pose_summary.get("intermediate_pose_sample_count", 0)))
        session.set_metric("includes_orientation", bool(pose_summary.get("includes_orientation", False)))
        session.set_metric("automatic_two_segment_pretension_validated", False)
        session.set_metric("two_segment_control_validated", False)
        session.set_metric("long_run_health", dict(self._long_run_health))
        session.set_metric("transport_recovery_report", dict(self._transport_recovery_report))
        session.set_metric("sample_failure_event_count", len(self._sample_failure_events))
        current_load_summary = _two_segment_current_load_summary(
            samples=session.samples,
            warning_ma=int(self.config.current_warning_ma or 800),
            hard_ma=int(self.config.sustained_overcurrent_ma or 1200),
            sustained_sample_count=int(self.config.sustained_overcurrent_sample_count or 3),
        )
        session.set_metric("current_load_summary", current_load_summary)
        for servo_id in current_load_summary.get("sustained_jam_servo_ids", []) or []:
            session.add_warning(f"servo {servo_id} showed sustained current >= warning threshold")
        tracker_freshness_summary = _two_segment_tracker_freshness_summary(
            samples=session.samples,
            stale_threshold_s=float(
                getattr(getattr(session.context.settings, "serial", None), "tracker_max_stale_interval_s", None) or 0.25
            ),
        )
        session.set_metric("tracker_freshness_summary", tracker_freshness_summary)
        if tracker_freshness_summary.get("samples_stale", 0):
            session.add_warning(
                f"tracker reported {tracker_freshness_summary['samples_stale']} stale samples "
                f"(threshold={tracker_freshness_summary['stale_threshold_s']:.3f} s, "
                f"max freshness={tracker_freshness_summary['max_freshness_s']:.3f} s)"
            )
        # Registration provenance: same block as repeatability for cross-run
        # comparability. Two collect-pose runs done with different registrations
        # produce datasets that should be compared cautiously.
        registration_summary = _two_segment_registration_summary(session=session)
        session.set_metric("registration_summary", registration_summary)
        if registration_summary.get("registration_state") not in {"loaded", "trusted"}:
            session.add_warning(
                f"Registration not fully loaded during collect-pose "
                f"(state={registration_summary.get('registration_state')}); distal/intermediate "
                "labels may be in tracker frame only."
            )
        assembly = dict(context.metadata().get("physical_assembly") or {})
        if self.config.physical_assembly_confirmed_by_operator:
            assembly["confirmed_by_operator"] = True
            assembly["confirmed_at_utc"] = str(
                self.config.physical_assembly_confirmed_at_utc
                or datetime.now(timezone.utc).isoformat()
            )
            assembly["confirmation_source"] = "experiment_config"
        session.set_metric("physical_assembly", assembly)
        session.set_metric("physical_assembly_confirmed_by_operator", bool(assembly.get("confirmed_by_operator", False)))
        session.set_metric("bottom_segment_key", context.bottom_segment_key)
        session.set_metric("top_segment_key", context.top_segment_key)
        session.set_metric("bottom_servo_ids", list(context.bottom_servo_ids or []))
        session.set_metric("top_servo_ids", list(context.top_servo_ids or []))
        for warning in data_quality_warnings:
            session.add_warning(str(warning))

    def finalize(self, session: ExperimentSession) -> None:
        if self.config.dry_run or not self._startup_ticks_by_servo:
            return
        try:
            writer = getattr(session.context.servo_service, "_write_goal_positions", None)
            if callable(writer) and session.context.servo_service.is_connected:
                writer({int(servo_id): int(tick) for servo_id, tick in self._startup_ticks_by_servo.items()})
        except Exception as exc:
            session.add_warning(f"Failed to return all 8 servos to startup ticks during finalize: {exc}")

    def _issue_command(self, *, session: ExperimentSession, command: TwoSegmentCommand) -> dict[str, Any]:
        context = session.context.settings.robot.operating_context()
        # `intended_flat_cm` is the user/scheduler-intended 8-vector in
        # commanded_servo_ids order. It is recorded as the modeling feature
        # (`ordered_8_displacements_cm`) so the trained model maps from
        # "operator intent" → measured pose.
        intended_flat_cm = command.to_flat(context=context)
        # `servo_flat_cm` is what the bus actually receives. Top-segment
        # tendons route through the bottom segment, so the top servo must
        # pull `top_intended + bottom_intended` to make the top achieve its
        # intended displacement relative to the intermediate plate. Disable
        # via `top_segment_tendon_routing_compensation: false` only if your
        # rig's tendons are independently routed (no parallel pass-through).
        if bool(self.config.top_segment_tendon_routing_compensation):
            servo_flat_cm, compensation_info = _apply_top_routing_compensation(
                intended_flat_cm, context=context,
            )
        else:
            servo_flat_cm = list(intended_flat_cm)
            compensation_info = {"applied": False, "reason": "disabled_by_config"}
        commanded_ids = [int(value) for value in context.commanded_servo_ids]
        startup_ticks = [int(self._startup_ticks_by_servo[servo_id]) for servo_id in commanded_ids]
        goal_ticks = session.context.servo_service.mapper.to_goal_positions(servo_flat_cm, startup_ticks)
        goals_by_servo = {int(servo_id): int(goal) for servo_id, goal in zip(commanded_ids, goal_ticks)}
        result = {
            "success": True,
            "requested_flat_command_cm": list(intended_flat_cm),
            "ordered_8_displacements_cm": list(intended_flat_cm),
            "servo_flat_command_cm": list(servo_flat_cm),
            "top_routing_compensation": dict(compensation_info),
            "commanded_servo_ids": list(commanded_ids),
            "startup_ticks_by_servo": {str(key): int(value) for key, value in sorted(self._startup_ticks_by_servo.items())},
            "goal_ticks_by_servo": {str(key): int(value) for key, value in sorted(goals_by_servo.items())},
            "command_message": "Dry-run command resolved relative to startup ticks.",
            "dry_run": bool(self.config.dry_run),
        }
        if self.config.dry_run:
            self._previous_flat_cm = list(intended_flat_cm)
            return result
        # Optional command ramping: if any tendon jump exceeds the configured
        # step, write intermediate sub-commands first. Intermediate writes do
        # NOT generate samples — only the final target counts toward
        # accepted/rejected. Ramping interpolates in user-intended space and
        # applies compensation per interim write so the bus always sees
        # physically consistent goal ticks.
        ramp_substeps = self._ramp_command_writes(
            session=session,
            target_flat_cm=list(intended_flat_cm),
            commanded_ids=commanded_ids,
            startup_ticks=startup_ticks,
        )
        try:
            writer = getattr(session.context.servo_service, "_write_goal_positions", None)
            if not callable(writer):
                raise RuntimeError("ServoService all-8 raw goal writer is unavailable.")
            writer(goals_by_servo)
            self._previous_flat_cm = list(intended_flat_cm)
            self._ramp_substep_count += int(ramp_substeps)
            result["ramp_substeps"] = int(ramp_substeps)
            result["command_message"] = (
                f"Wrote bounded all-8 raw goal ticks relative to manual startup artifact "
                f"({ramp_substeps} ramp substeps)."
                if ramp_substeps
                else "Wrote bounded all-8 raw goal ticks relative to manual startup artifact."
            )
            return result
        except Exception as exc:
            result["success"] = False
            result["command_message"] = str(exc)
            result["ramp_substeps"] = int(ramp_substeps)
            return result

    def _ramp_command_writes(
        self,
        *,
        session: ExperimentSession,
        target_flat_cm: list[float],
        commanded_ids: list[int],
        startup_ticks: list[int],
    ) -> int:
        """Write intermediate goal-tick steps when the per-tendon jump exceeds the ramp threshold.

        Returns the number of intermediate substeps written. Returns 0 when
        ramping is disabled or no substeps are needed.
        """
        step_cm = self.config.command_ramp_step_cm
        if step_cm is None or float(step_cm) <= 0.0:
            return 0
        previous = list(self._previous_flat_cm) if self._previous_flat_cm else [0.0] * len(target_flat_cm)
        deltas = [float(t) - float(p) for t, p in zip(target_flat_cm, previous)]
        max_abs_delta = max((abs(value) for value in deltas), default=0.0)
        if max_abs_delta <= float(step_cm):
            return 0
        import math as _math

        substep_count = max(1, int(_math.ceil(max_abs_delta / float(step_cm))) - 1)
        if substep_count <= 0:
            return 0
        writer = getattr(session.context.servo_service, "_write_goal_positions", None)
        if not callable(writer):
            return 0
        mapper = session.context.servo_service.mapper
        sleep_fn = session.context.sleep_fn
        settle = float(self.config.command_ramp_settle_time_s)
        context = session.context.settings.robot.operating_context()
        apply_compensation = bool(self.config.top_segment_tendon_routing_compensation)
        for k in range(1, substep_count + 1):
            fraction = float(k) / float(substep_count + 1)  # never reaches 1.0
            # Interpolate in user-intended space, then compensate the interim
            # write so the bus sees physically consistent goal ticks at every
            # ramp substep (consistent with the final write).
            interim_flat = [float(p) + float(d) * fraction for p, d in zip(previous, deltas)]
            if apply_compensation:
                servo_interim_flat, _info = _apply_top_routing_compensation(interim_flat, context=context)
            else:
                servo_interim_flat = list(interim_flat)
            interim_goals = mapper.to_goal_positions(servo_interim_flat, startup_ticks)
            interim_goals_by_servo = {int(sid): int(goal) for sid, goal in zip(commanded_ids, interim_goals)}
            try:
                writer(interim_goals_by_servo)
            except Exception:
                # If an intermediate write fails, abort the ramp and let the
                # final write attempt report failure with full context.
                return k - 1
            if settle > 0.0:
                sleep_fn(settle)
        return substep_count

    def _capture_sample(
        self,
        *,
        session: ExperimentSession,
        step: TwoSegmentCommandStep,
        command_result: dict[str, Any],
        repeat_index: int,
        sample_index: int,
        run_trust_mode: str,
        foundation: dict[str, Any],
        startup_provenance: dict[str, Any],
    ) -> ExperimentTimeseriesSample:
        context = session.context.settings.robot.operating_context()
        commanded_ids = [int(value) for value in context.commanded_servo_ids]
        telemetry = _read_live_telemetry(session=session, servo_ids=commanded_ids, dry_run=bool(self.config.dry_run))
        servo_feedback = _servo_feedback_payload(session=session, telemetry=telemetry, servo_ids=commanded_ids, startup_ticks_by_servo=self._startup_ticks_by_servo)
        pose_observation, pose_fields = _two_segment_pose_observation(
            session=session,
            config=self.config,
            run_trust_mode=run_trust_mode,
        )
        command_record = step.command.to_record(context=context)
        ordered_8 = step.command.to_flat(context=context)
        command_success = bool(command_result.get("success", False))
        # Identify any commanded servo with a missing measured position. This
        # mirrors the downstream modeling-loader rejection so the operator
        # learns about telemetry drops at collection time, not after.
        missing_measured_servo_ids = sorted(
            int(servo_id)
            for servo_id in commanded_ids
            if dict(servo_feedback.get(str(int(servo_id)), {}) or {}).get("position_tick") is None
        )
        servo_only_run = _servo_only_mode(self.config)
        # Trusted (non-servo-only/dry-run) runs reject the sample at capture
        # time. Servo-only/dry-run runs record the missing IDs but stay
        # accepted (the modeling loader still excludes them later).
        telemetry_complete_failure = bool(missing_measured_servo_ids) and not servo_only_run
        capture_accepted = bool(command_success) and not telemetry_complete_failure
        if not command_success:
            rejection_reason = str(command_result.get("command_message") or "command_failed")
        elif telemetry_complete_failure:
            rejection_reason = f"measured_servo_position_missing:{','.join(str(x) for x in missing_measured_servo_ids)}"
        else:
            rejection_reason = None
        missing_pose = not bool(pose_fields.get("pose_observations_present"))
        valid_for_two_segment_model_training = bool(
            capture_accepted
            and _valid_for_two_segment_model_training(self.config, _pose_label_summary_from_fields(pose_fields), startup_provenance, 0)
        )
        label_mode_used = _dataset_label_mode_from_pose_summary(_pose_label_summary_from_fields(pose_fields))
        flags = ["capture_accepted" if capture_accepted else "capture_rejected"]
        if self.config.dry_run:
            flags.append("dry_run")
        if servo_only_run:
            flags.extend(["servo_only", "lower_trust", "not_model_training_ready"])
        if missing_pose:
            flags.append("pose_labels_missing")
        if missing_measured_servo_ids:
            flags.append("measured_servo_position_missing")
        return ExperimentTimeseriesSample(
            monotonic_time_s=session.elapsed_s(),
            wall_time_utc=datetime.now(timezone.utc).isoformat(),
            phase=str(step.phase),
            step_index=int(step.index),
            sample_index=int(sample_index),
            cycle_index=int(repeat_index),
            commanded_motor_values=dict(command_result.get("goal_ticks_by_servo", {}) or {}),
            commanded_cable_deltas_cm=list(ordered_8),
            two_segment_command=command_record,
            tracker_frame_id=pose_fields.get("tracker_frame_id"),
            tool_ids_seen=list(pose_fields.get("tool_ids_seen", []) or []),
            transform_validity=dict(pose_fields.get("transform_validity", {}) or {}),
            pose_in_tracker_frame=dict(pose_fields.get("pose_in_tracker_frame", {}) or {}),
            pose_in_robot_frame=dict(pose_fields.get("pose_in_robot_frame", {}) or {}),
            two_segment_pose=pose_observation,
            freshness_s=pose_fields.get("freshness_s"),
            latency_s=None,
            status_flags=sorted(set(flags)),
            backend_health=dict(pose_fields.get("backend_health", {}) or {}),
            extra={
                "record_kind": "two_segment_dataset_capture",
                "dataset_schema_version": TWO_SEGMENT_DATASET_SCHEMA_VERSION,
                "run_trust_mode": run_trust_mode,
                "schedule_type": str(self.config.schedule_type),
                "command_label": str(step.label),
                "command_schema_version": command_record.get("schema_version"),
                "ordered_8_displacements_cm": list(ordered_8),
                "command_units": "cm",
                "commanded_servo_ids": list(commanded_ids),
                "segment_order": list(context.segment_order),
                "segments": dict(context.metadata().get("segments", {}) or {}),
                "goal_ticks_by_servo": dict(command_result.get("goal_ticks_by_servo", {}) or {}),
                # Audit trail for the top-segment tendon routing compensation
                # applied at servo-write time. `servo_flat_command_cm` is what
                # the bus actually saw; `ordered_8_displacements_cm` (above)
                # is what the user intended and is what the modeling feature
                # path reads. Both are recorded so post-hoc analysis can
                # reconstruct the actuator-level command and verify policy.
                "servo_flat_command_cm": list(command_result.get("servo_flat_command_cm") or ordered_8),
                "top_routing_compensation": dict(command_result.get("top_routing_compensation") or {}),
                "startup_artifact_provenance": dict(startup_provenance),
                "measured_servo_feedback": servo_feedback,
                "missing_measured_servo_ids": list(missing_measured_servo_ids),
                "pose_observations_present": bool(pose_fields.get("pose_observations_present")),
                "available_pose_roles": list(pose_fields.get("available_roles", []) or []),
                "missing_pose_roles": list(pose_fields.get("missing_roles", []) or []),
                "stale_pose_roles": list(pose_fields.get("stale_roles", []) or []),
                "invalid_pose_roles": list(pose_fields.get("invalid_roles", []) or []),
                "missing_required_pose_roles": list(pose_fields.get("missing_required_roles", []) or []),
                "role_observations": dict(pose_fields.get("role_observations", {}) or {}),
                "role_config": dict(pose_fields.get("role_config", {}) or {}),
                "pose_trust_by_role": dict(pose_fields.get("pose_trust_by_role", {}) or {}),
                "tracker_freshness_by_role": dict(pose_fields.get("tracker_freshness_by_role", {}) or {}),
                "distal_only": bool(pose_fields.get("distal_only")),
                "includes_intermediate_pose": bool(pose_fields.get("includes_intermediate_pose")),
                "includes_orientation": bool(_pose_fields_include_orientation(pose_fields)),
                "label_mode_used": label_mode_used,
                "capture_accepted": bool(capture_accepted),
                "capture_rejection_reason": rejection_reason,
                "command_success": bool(command_success),
                "command_message": str(command_result.get("command_message", "") or ""),
                "valid_for_two_segment_model_training": bool(valid_for_two_segment_model_training),
                "valid_for_two_segment_ann_training": bool(valid_for_two_segment_model_training),
                "valid_for_model_training": False,
                "valid_for_thesis_repeatability": False,
                "data_quality_warnings": _sample_warnings(config=self.config, missing_pose=missing_pose, startup_provenance=startup_provenance),
            },
        )

    def summarize(self, session: ExperimentSession) -> dict[str, Any]:
        return dict(session.metrics)

    def write_outputs(self, session: ExperimentSession, paths, summary) -> None:
        write_two_segment_dataset_outputs(
            output_dir=paths.output_dir,
            metrics=dict(summary.experiment_metrics or {}),
            samples=session.samples,
            sample_failure_events=list(self._sample_failure_events),
            long_run_health=dict(self._long_run_health),
            transport_recovery_report=dict(self._transport_recovery_report),
        )


def build_two_segment_command_schedule(config: TwoSegmentCollectPoseDatasetConfig, *, context) -> list[TwoSegmentCommandStep]:
    """Build the per-segment command schedule.

    The amplitude is taken from the configured ``max_segment_displacement_cm``;
    no silent cap is applied (the legacy MVP capped at 0.01 cm). Hardware safety
    is enforced separately via ``max_tick_delta_from_startup`` and the safety
    guard. Schedules can be bottom/top-aware: when a physical assembly is
    defined, ``bottom_only_sweep`` / ``top_only_sweep`` route the displacements
    to the correct fixed segment.
    """

    amp_cm = max(0.0, float(config.max_segment_displacement_cm))
    schedule_type = str(config.schedule_type)
    zero_step = TwoSegmentCommandStep(
        index=0,
        label="zero_startup",
        phase="zero",
        command=TwoSegmentCommand.from_mapping({"segment_a": [0.0, 0.0, 0.0, 0.0], "segment_b": [0.0, 0.0, 0.0, 0.0]}),
    )
    if amp_cm <= 0.0 or schedule_type == "zero":
        return [zero_step]

    assembly = dict(context.metadata().get("physical_assembly") or {}) if context is not None else {}
    bottom_key = str(assembly.get("bottom_segment_key") or "segment_a")
    top_key = str(assembly.get("top_segment_key") or "segment_b")

    def _seg(values_bottom: list[float], values_top: list[float]) -> dict[str, list[float]]:
        # values_bottom / values_top are tendon-vectors for the bottom and top
        # *physical* segments; map them back to segment_a/segment_b by physical role.
        mapping: dict[str, list[float]] = {bottom_key: list(values_bottom), top_key: list(values_top)}
        # Ensure both keys exist, even when assembly metadata is empty.
        mapping.setdefault("segment_a", [0.0, 0.0, 0.0, 0.0])
        mapping.setdefault("segment_b", [0.0, 0.0, 0.0, 0.0])
        return {"segment_a": mapping["segment_a"], "segment_b": mapping["segment_b"]}

    patterns: list[tuple[str, str, list[float], list[float]]] = []
    if schedule_type in {"single_axis_micro", "segment_isolation", "small_combined", "bottom_only_sweep", "top_only_sweep", "workspace_coverage", "structured_grid", "mixed_training"}:
        single_axis_patterns: list[tuple[str, str, list[float], list[float]]] = [
            ("bottom_axis_a_pos", "single_axis_micro", [amp_cm, 0.0, -amp_cm, 0.0], [0.0, 0.0, 0.0, 0.0]),
            ("bottom_axis_a_neg", "single_axis_micro", [-amp_cm, 0.0, amp_cm, 0.0], [0.0, 0.0, 0.0, 0.0]),
            ("bottom_axis_b_pos", "single_axis_micro", [0.0, amp_cm, 0.0, -amp_cm], [0.0, 0.0, 0.0, 0.0]),
            ("bottom_axis_b_neg", "single_axis_micro", [0.0, -amp_cm, 0.0, amp_cm], [0.0, 0.0, 0.0, 0.0]),
            ("top_axis_a_pos", "single_axis_micro", [0.0, 0.0, 0.0, 0.0], [amp_cm, 0.0, -amp_cm, 0.0]),
            ("top_axis_a_neg", "single_axis_micro", [0.0, 0.0, 0.0, 0.0], [-amp_cm, 0.0, amp_cm, 0.0]),
            ("top_axis_b_pos", "single_axis_micro", [0.0, 0.0, 0.0, 0.0], [0.0, amp_cm, 0.0, -amp_cm]),
            ("top_axis_b_neg", "single_axis_micro", [0.0, 0.0, 0.0, 0.0], [0.0, -amp_cm, 0.0, amp_cm]),
        ]
        patterns = list(single_axis_patterns)
    if schedule_type == "segment_isolation":
        patterns = [item for item in patterns if item[0].startswith("bottom")][:2] + [item for item in patterns if item[0].startswith("top")][:2]
    elif schedule_type == "small_combined":
        half = amp_cm / 2.0
        patterns.extend(
            [
                ("combined_bottom_top_axis_a_pos", "small_combined", [half, 0.0, -half, 0.0], [half, 0.0, -half, 0.0]),
                ("combined_bottom_top_axis_b_pos", "small_combined", [0.0, half, 0.0, -half], [0.0, half, 0.0, -half]),
                ("combined_cross_axes", "small_combined", [half, 0.0, -half, 0.0], [0.0, -half, 0.0, half]),
            ]
        )
    elif schedule_type == "bottom_only_sweep":
        patterns = [item for item in patterns if item[0].startswith("bottom")]
    elif schedule_type == "top_only_sweep":
        patterns = [item for item in patterns if item[0].startswith("top")]
    elif schedule_type == "workspace_coverage":
        patterns = _workspace_coverage_patterns(amp_cm)
    elif schedule_type == "structured_grid":
        patterns = _structured_grid_patterns(amp_cm, points=max(2, int(config.grid_points_per_axis)))
    elif schedule_type == "random_babble":
        patterns = _random_babble_patterns(
            amp_cm,
            sample_count=_random_babble_unique_command_count(config),
            seed=int(config.random_seed),
        )
    elif schedule_type == "mixed_training":
        # Mix structured single-axis with workspace coverage and a few random samples.
        mixed = list(patterns)
        mixed.extend(_workspace_coverage_patterns(amp_cm))
        mixed.extend(
            _random_babble_patterns(
                amp_cm,
                sample_count=_random_babble_unique_command_count(config, default_when_unset=16),
                seed=int(config.random_seed),
            )
        )
        patterns = mixed

    steps: list[TwoSegmentCommandStep] = [zero_step]
    for index, (label, phase, bottom_values, top_values) in enumerate(patterns, start=1):
        segments_mapping = _seg(bottom_values, top_values)
        steps.append(
            TwoSegmentCommandStep(
                index=index,
                label=label,
                phase=phase,
                command=TwoSegmentCommand.from_mapping(segments_mapping),
            )
        )
    if int(config.pattern_count_budget) > 0:
        steps = steps[: max(1, int(config.pattern_count_budget) + 1)]  # +1 for zero step
    return steps


def _workspace_coverage_patterns(amp_cm: float) -> list[tuple[str, str, list[float], list[float]]]:
    """Cardinal + diagonal directions for both segments to cover workspace."""

    if amp_cm <= 0.0:
        return []
    half = amp_cm / 2.0
    cardinals_bottom = [
        ("bottom_only_pos_x", "workspace_coverage", [amp_cm, 0.0, -amp_cm, 0.0], [0.0, 0.0, 0.0, 0.0]),
        ("bottom_only_pos_y", "workspace_coverage", [0.0, amp_cm, 0.0, -amp_cm], [0.0, 0.0, 0.0, 0.0]),
        ("bottom_only_neg_x", "workspace_coverage", [-amp_cm, 0.0, amp_cm, 0.0], [0.0, 0.0, 0.0, 0.0]),
        ("bottom_only_neg_y", "workspace_coverage", [0.0, -amp_cm, 0.0, amp_cm], [0.0, 0.0, 0.0, 0.0]),
    ]
    cardinals_top = [
        ("top_only_pos_x", "workspace_coverage", [0.0, 0.0, 0.0, 0.0], [amp_cm, 0.0, -amp_cm, 0.0]),
        ("top_only_pos_y", "workspace_coverage", [0.0, 0.0, 0.0, 0.0], [0.0, amp_cm, 0.0, -amp_cm]),
        ("top_only_neg_x", "workspace_coverage", [0.0, 0.0, 0.0, 0.0], [-amp_cm, 0.0, amp_cm, 0.0]),
        ("top_only_neg_y", "workspace_coverage", [0.0, 0.0, 0.0, 0.0], [0.0, -amp_cm, 0.0, amp_cm]),
    ]
    combos = [
        ("combo_pos_x_pos_x", "workspace_coverage", [half, 0.0, -half, 0.0], [half, 0.0, -half, 0.0]),
        ("combo_pos_x_pos_y", "workspace_coverage", [half, 0.0, -half, 0.0], [0.0, half, 0.0, -half]),
        ("combo_pos_y_pos_x", "workspace_coverage", [0.0, half, 0.0, -half], [half, 0.0, -half, 0.0]),
        ("combo_pos_y_pos_y", "workspace_coverage", [0.0, half, 0.0, -half], [0.0, half, 0.0, -half]),
        ("combo_neg_x_pos_x", "workspace_coverage", [-half, 0.0, half, 0.0], [half, 0.0, -half, 0.0]),
        ("combo_neg_y_pos_y", "workspace_coverage", [0.0, -half, 0.0, half], [0.0, half, 0.0, -half]),
        ("combo_neg_xy_neg_xy", "workspace_coverage", [-half, -half, half, half], [-half, -half, half, half]),
    ]
    return cardinals_bottom + cardinals_top + combos


def _structured_grid_patterns(amp_cm: float, *, points: int) -> list[tuple[str, str, list[float], list[float]]]:
    """Grid in 4D (bottom_x, bottom_y, top_x, top_y) tendon space at amp_cm scale."""

    if amp_cm <= 0.0 or points < 2:
        return []
    grid_values = [amp_cm * (-1.0 + 2.0 * idx / (points - 1)) for idx in range(points)]
    patterns: list[tuple[str, str, list[float], list[float]]] = []
    # Cap the grid: a full Cartesian product across 4 axes would explode. Sweep
    # bottom-X, bottom-Y, top-X, top-Y one at a time, fixing others at 0.
    for label_prefix, axis in (("bottom_x", 0), ("bottom_y", 1), ("top_x", 0), ("top_y", 1)):
        for value in grid_values:
            if abs(value) < 1e-9:
                continue
            bottom = [0.0, 0.0, 0.0, 0.0]
            top = [0.0, 0.0, 0.0, 0.0]
            if label_prefix.startswith("bottom"):
                bottom[axis] = value
                bottom[axis + 2] = -value
            else:
                top[axis] = value
                top[axis + 2] = -value
            patterns.append((f"{label_prefix}_{value:+.3f}", "structured_grid", bottom, top))
    return patterns


def _random_babble_unique_command_count(
    config: "TwoSegmentCollectPoseDatasetConfig",
    *,
    default_when_unset: int = 5000,
) -> int:
    """Resolve the number of UNIQUE random commands the schedule should generate.

    The pre-2026-05-26 behaviour was ``max(8, pattern_count_budget or 32)``,
    which silently capped random_babble at 32 unique commands even when the
    operator asked for ``target_valid_sample_count=50000``. That made the
    run cycle through the same 32 commands ~150 times — the clusters were
    in the data, not the plot.

    Resolution order (highest priority first):

    1. ``pattern_count_budget`` > 0 → use it verbatim.
    2. ``continue_until_valid_samples`` + ``target_valid_sample_count`` →
       generate ``ceil(target / max(1, samples_per_pattern))`` unique
       commands so a single pass through the schedule yields the requested
       sample count. A small headroom margin is added so a sub-100 %
       acceptance rate doesn't force a second pass through identical
       commands.
    3. Otherwise default to ``default_when_unset`` (5000 — matches the
       single-segment ``workspace_coverage`` ladder).

    The floor of 8 is preserved so trivially-small runs still get a few
    distinct commands.
    """
    if int(config.pattern_count_budget or 0) > 0:
        return max(8, int(config.pattern_count_budget))
    if bool(config.continue_until_valid_samples) and int(config.target_valid_sample_count or 0) > 0:
        per_pattern = max(1, int(config.samples_per_pattern or 1))
        target = int(config.target_valid_sample_count)
        # 5 % headroom (rounded up to >= 1) — generous enough that an 80 %+
        # acceptance run finishes in one pass; cheap if 100 % acceptance.
        unique = (target + per_pattern - 1) // per_pattern
        headroom = max(1, unique // 20)
        return max(8, unique + headroom)
    return max(8, int(default_when_unset))


def _random_babble_patterns(amp_cm: float, *, sample_count: int, seed: int) -> list[tuple[str, str, list[float], list[float]]]:
    """Reproducible random tendon babble within the amp_cm hypercube.

    Bottom and top tendons are sampled independently in tendon space; opposite
    tendons are anti-correlated so total tendon length per segment is conserved
    on average (this is descriptive, not a controller — kinematics may differ
    on hardware until validated).
    """

    if amp_cm <= 0.0 or sample_count <= 0:
        return []
    import random
    rng = random.Random(int(seed) or 0)
    patterns: list[tuple[str, str, list[float], list[float]]] = []
    for index in range(int(sample_count)):
        bottom_x = rng.uniform(-amp_cm, amp_cm)
        bottom_y = rng.uniform(-amp_cm, amp_cm)
        top_x = rng.uniform(-amp_cm, amp_cm)
        top_y = rng.uniform(-amp_cm, amp_cm)
        bottom = [bottom_x, bottom_y, -bottom_x, -bottom_y]
        top = [top_x, top_y, -top_x, -top_y]
        patterns.append((f"random_{index:04d}", "random_babble", bottom, top))
    return patterns


def write_two_segment_dataset_outputs(
    *,
    output_dir: Path,
    metrics: dict[str, Any],
    samples: list[ExperimentTimeseriesSample],
    sample_failure_events: list[dict[str, Any]] | None = None,
    long_run_health: dict[str, Any] | None = None,
    transport_recovery_report: dict[str, Any] | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_text": output_dir / "two_segment_dataset_summary.txt",
        "role_provenance": output_dir / "two_segment_tracking_role_provenance.json",
        "metrics_csv": output_dir / "metrics.csv",
        "command_coverage": output_dir / "two_segment_command_coverage_report.png",
        "servo_position_coverage": output_dir / "two_segment_servo_position_coverage_report.png",
        "pose_coverage": output_dir / "two_segment_pose_coverage_report.png",
        "quality": output_dir / "two_segment_dataset_quality_report.png",
        "long_run_health": output_dir / "long_run_health.json",
        "sample_failure_events": output_dir / "sample_failure_events.jsonl",
        "transport_recovery_report": output_dir / "transport_recovery_report.json",
    }
    _write_summary_text(paths["summary_text"], metrics)
    paths["role_provenance"].write_text(
        json.dumps(
            {
                "two_segment_tracking_role_config": metrics.get("two_segment_tracking_role_config", {}),
                "pose_label_summary": metrics.get("pose_label_summary", {}),
                "physical_assembly": metrics.get("physical_assembly", {}),
                "bottom_segment_key": metrics.get("bottom_segment_key"),
                "top_segment_key": metrics.get("top_segment_key"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_metrics_csv(paths["metrics_csv"], samples)
    _write_command_coverage(paths["command_coverage"], samples)
    _write_servo_position_coverage(paths["servo_position_coverage"], samples)
    _write_pose_coverage(paths["pose_coverage"], samples, metrics)
    _write_quality_report(paths["quality"], metrics)
    paths["long_run_health"].write_text(
        json.dumps(long_run_health or {}, indent=2),
        encoding="utf-8",
    )
    paths["transport_recovery_report"].write_text(
        json.dumps(transport_recovery_report or {}, indent=2),
        encoding="utf-8",
    )
    with paths["sample_failure_events"].open("w", encoding="utf-8") as handle:
        for event in list(sample_failure_events or []):
            handle.write(json.dumps(event) + "\n")
    # Thesis-quality figure set, additive alongside the legacy reports. Mirrors
    # the single-segment collect_pose_command_dataset ``thesis_*`` contract and
    # adds per-segment + dataset-quality + current-load panels specific to the
    # two-segment data shape. Failures here must not block the legacy outputs.
    try:
        from continuum_robot.experiments.two_segment_modeling_dataset_outputs import (
            write_two_segment_thesis_figures,
        )
        thesis_paths = write_two_segment_thesis_figures(
            output_dir=output_dir,
            metrics=metrics,
            samples=samples,
            sample_failure_events=list(sample_failure_events or []),
        )
        for key, path in thesis_paths.items():
            paths[key] = path
    except Exception:
        # Operator already has the legacy reports; emit a warning but do not
        # propagate so the rest of the bundle still lands.
        import logging as _logging
        _logging.getLogger(__name__).exception(
            "Two-segment thesis figure set failed to render; legacy reports unaffected"
        )
    return paths


def _resolve_startup_ticks(
    *,
    session: ExperimentSession,
    config: TwoSegmentCollectPoseDatasetConfig,
    expected_ids: list[int],
) -> tuple[dict[int, int], dict[str, Any]]:
    servo_service = session.context.servo_service
    trusted_requested = _trusted_dataset_requested(config)
    try:
        reference = servo_service.resolve_startup_reference_ticks(expected_ids)
        source_summary = servo_service.pretension_source_summary(expected_ids)
        accepted_all_8 = bool(source_summary.accepted and source_summary.usable and len(reference.ticks_by_servo) == 8)
        provenance = {
            "source": reference.source,
            "message": reference.message,
            "accepted_all_8_startup": accepted_all_8,
            "pretension_source_type": source_summary.source_type,
            "pretension_message": source_summary.message,
            "artifact_path": str(getattr(servo_service.neutral_calibration, "path", "")),
            "startup_ticks_by_servo": {str(key): int(value) for key, value in sorted(reference.ticks_by_servo.items())},
        }
        return {int(key): int(value) for key, value in reference.ticks_by_servo.items()}, provenance
    except Exception as exc:
        if trusted_requested:
            raise RuntimeError(
                "Trusted two-segment dataset collection requires an accepted all-8 manual startup artifact. "
                "Use two_segment_startup_validation first or run with allow_servo_only_test_run=true and run_trust_mode=servo_only."
            ) from exc
        telemetry_ticks: dict[int, int] = {}
        try:
            if servo_service is not None and servo_service.is_connected:
                telemetry = servo_service.read_live_telemetry(expected_ids)
                telemetry_ticks = {
                    int(servo_id): int(item.present_position)
                    for servo_id, item in telemetry.items()
                    if item is not None and item.present_position is not None
                }
        except Exception:
            telemetry_ticks = {}
        ticks = {int(servo_id): int(telemetry_ticks.get(int(servo_id), 2048)) for servo_id in expected_ids}
        return ticks, {
            "source": "telemetry_or_synthetic_servo_only_fallback",
            "message": f"Startup artifact unavailable for lower-trust run: {exc}",
            "accepted_all_8_startup": False,
            "artifact_path": str(getattr(getattr(servo_service, "neutral_calibration", None), "path", "")),
            "startup_ticks_by_servo": {str(key): int(value) for key, value in sorted(ticks.items())},
        }


def _trusted_dataset_requested(config: TwoSegmentCollectPoseDatasetConfig) -> bool:
    return (
        not bool(config.dry_run)
        and not bool(config.allow_servo_only_test_run)
        and str(config.run_trust_mode or "").strip().lower() not in {"servo_only", "current_only", "lower_trust", "debug"}
    )


def _servo_only_mode(config: TwoSegmentCollectPoseDatasetConfig) -> bool:
    return bool(config.dry_run or config.allow_servo_only_test_run or str(config.run_trust_mode).strip().lower() == "servo_only")


def _run_trust_mode(config: TwoSegmentCollectPoseDatasetConfig) -> str:
    if _servo_only_mode(config):
        return "servo_only"
    return str(config.run_trust_mode or "thesis_trusted")


def tick_budget_for_segment_amplitude_cm(
    amplitude_cm: float,
    *,
    spool_diameter_cm: float,
    ticks_per_rev: int = 4096,
    safety_factor: float = 1.2,
    compensation_doubles_top: bool = True,
) -> int:
    """Compute a per-servo tick budget that comfortably covers an amplitude.

    The two-segment top-tendon routing compensation can almost double the
    top-segment per-servo displacement (top servo receives
    ``top_intended + bottom_intended``). The safety budget the GUI/preflight
    enforces must therefore account for that doubling plus headroom for
    settling/jitter.

    The formula is:
        effective_cm  = amplitude_cm * (2 if compensation_doubles_top else 1)
        ticks         = effective_cm * ticks_per_rev / (π * spool_diameter_cm)
        budget        = ceil(ticks * safety_factor)

    Defaults to ``safety_factor=1.2`` (20 % headroom) and
    ``compensation_doubles_top=True``. Returns a positive integer tick
    delta budget — the operator's chosen amplitude maps deterministically
    to this number so the GUI never has to ask the operator to set it
    manually.
    """
    import math

    effective_cm = max(0.0, float(amplitude_cm)) * (2.0 if compensation_doubles_top else 1.0)
    circumference_cm = math.pi * max(1e-9, float(spool_diameter_cm))
    raw_ticks = effective_cm * float(ticks_per_rev) / circumference_cm
    return max(1, int(math.ceil(raw_ticks * max(1.0, float(safety_factor)))))


def _command_limit_violations(
    *,
    steps: list[TwoSegmentCommandStep],
    context,
    startup_ticks_by_servo: dict[int, int],
    max_tick_delta: int,
    mapper,
    apply_top_routing_compensation: bool = True,
) -> list[str]:
    """Return safety violations for the configured schedule.

    When ``apply_top_routing_compensation`` is True (the default and the
    policy for trusted dual-segment work), the per-step tick delta is
    computed from the COMPENSATED flat command — i.e. the values actually
    written to servos. This is important because compensation can almost
    double the top-segment per-servo displacement (top tendon shares the
    bottom's pull) and the safety budget must be checked against what the
    bus will actually receive.
    """
    violations: list[str] = []
    commanded_ids = [int(value) for value in context.commanded_servo_ids]
    startup_ticks = [int(startup_ticks_by_servo[int(servo_id)]) for servo_id in commanded_ids]
    for step in steps:
        intended_flat = step.command.to_flat(context=context)
        if apply_top_routing_compensation:
            servo_flat, _info = _apply_top_routing_compensation(intended_flat, context=context)
        else:
            servo_flat = list(intended_flat)
        goals = mapper.to_goal_positions(servo_flat, startup_ticks)
        for servo_id, startup, goal in zip(commanded_ids, startup_ticks, goals):
            delta = abs(int(goal) - int(startup))
            if delta > int(max_tick_delta):
                violations.append(f"{step.label}: servo {servo_id} delta {delta} > {int(max_tick_delta)}")
    return violations


def _apply_top_routing_compensation(
    flat_cm: list[float],
    *,
    context,
) -> tuple[list[float], dict[str, Any]]:
    """Compensate top-segment tendon displacements for routing through the bottom segment.

    Physical reality of the stacked rig: top-segment tendons route through
    the bottom segment in parallel with the bottom-segment tendons. When the
    bottom +X tendon shortens by Δ (bending the bottom toward +X), the top
    +X tendon (which passes through the bottom on the same parallel path)
    is dragged by the same Δ. To make the top segment achieve its
    operator-intended displacement ``t`` RELATIVE to the intermediate plate,
    the top servo must therefore pull ``t + Δ``.

    Convention: tendon order within each 4-vector is ``[+X, +Y, -X, -Y]``
    and matches by-list-index across bottom and top. Per-tendon compensation
    is ``top_servo[i] = top_intended[i] + bottom_intended[i]``.

    Inputs:
        ``flat_cm`` — 8-vector in ``context.commanded_servo_ids`` order
        (user-intended per-segment displacements; the canonical
        TwoSegmentCommand.to_flat output).

    Returns ``(compensated_flat_cm, info)``. When the physical assembly is
    missing or malformed the function returns the input unchanged with
    ``info["applied"] = False`` and a descriptive reason.
    """
    commanded_ids = [int(value) for value in getattr(context, "commanded_servo_ids", [])]
    metadata = context.metadata() if callable(getattr(context, "metadata", None)) else {}
    assembly = dict(metadata.get("physical_assembly") or {})
    bottom_servo_ids = [int(value) for value in list(assembly.get("bottom_servo_ids") or [])]
    top_servo_ids = [int(value) for value in list(assembly.get("top_servo_ids") or [])]
    if (
        len(bottom_servo_ids) != 4
        or len(top_servo_ids) != 4
        or len(flat_cm) != 8
        or len(commanded_ids) != 8
    ):
        return list(flat_cm), {
            "applied": False,
            "reason": "physical_assembly_incomplete_or_unexpected_command_shape",
            "bottom_servo_ids": list(bottom_servo_ids),
            "top_servo_ids": list(top_servo_ids),
        }
    by_servo = {int(servo_id): float(value) for servo_id, value in zip(commanded_ids, flat_cm)}
    compensated_by_servo = dict(by_servo)
    bottom_contribution_by_top: dict[int, float] = {}
    for tendon_index in range(4):
        bottom_id = int(bottom_servo_ids[tendon_index])
        top_id = int(top_servo_ids[tendon_index])
        bottom_value = float(by_servo.get(bottom_id, 0.0))
        compensated_by_servo[top_id] = float(by_servo.get(top_id, 0.0)) + bottom_value
        bottom_contribution_by_top[top_id] = bottom_value
    compensated_flat = [float(compensated_by_servo[int(servo_id)]) for servo_id in commanded_ids]
    return compensated_flat, {
        "applied": True,
        "schema_version": "two_segment_top_routing_compensation_v1",
        "bottom_servo_ids": list(bottom_servo_ids),
        "top_servo_ids": list(top_servo_ids),
        "bottom_contribution_added_to_top_servo_cm": {
            str(servo_id): float(value)
            for servo_id, value in sorted(bottom_contribution_by_top.items())
        },
        "convention": (
            "top_servo[i] = top_intended[i] + bottom_intended[i]; "
            "tendon order [+X, +Y, -X, -Y]; matches by-list-index across bottom and top."
        ),
    }


def _read_live_telemetry(*, session: ExperimentSession, servo_ids: list[int], dry_run: bool) -> dict[int, Any]:
    if dry_run and (session.context.servo_service is None or not session.context.servo_service.is_connected):
        return {}
    try:
        return session.context.servo_service.read_live_telemetry(servo_ids)
    except Exception:
        return {}


def _servo_feedback_payload(*, session: ExperimentSession, telemetry: dict[int, Any], servo_ids: list[int], startup_ticks_by_servo: dict[int, int]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for servo_id in servo_ids:
        item = telemetry.get(int(servo_id))
        position = getattr(item, "present_position", None) if item is not None else None
        current = getattr(item, "present_current_ma", None) if item is not None else None
        payload[str(servo_id)] = {
            "servo_id": int(servo_id),
            "position_tick": int(position) if position is not None else None,
            "delta_from_startup_tick": (
                int(position) - int(startup_ticks_by_servo[int(servo_id)])
                if position is not None and int(servo_id) in startup_ticks_by_servo
                else None
            ),
            "signed_raw_current_ma": int(current) if current is not None else None,
            "load_proxy_ma": abs(int(current)) if current is not None else None,
            "voltage_mv": int(item.present_voltage_mv) if item is not None and item.present_voltage_mv is not None else None,
            "temperature_c": int(item.present_temperature_c) if item is not None and item.present_temperature_c is not None else None,
            "hardware_error": getattr(item, "hardware_error", None) if item is not None else "missing telemetry",
            "telemetry_stale": (
                not bool(session.context.servo_service.telemetry_is_fresh(item))
                if item is not None and session.context.servo_service is not None
                else True
            ),
            "telemetry_age_s": (
                session.context.servo_service.telemetry_age_s(item)
                if item is not None and session.context.servo_service is not None
                else None
            ),
        }
    return payload


def _effective_role_configs(*, session: ExperimentSession, config: TwoSegmentCollectPoseDatasetConfig) -> dict[str, Any]:
    role_configs = role_config_records(
        getattr(session.context.settings.registration, "two_segment_tracking_roles", {}) or {}
    )
    for tool_id, role_name in dict(config.requested_tool_roles or {}).items():
        role_key = str(role_name)
        record = dict(role_configs.get(role_key, {}) or {})
        record.setdefault("role_name", role_key)
        record["tool_id"] = str(tool_id).strip().upper()
        record["enabled"] = True
        record.setdefault("source", "experiment_config.requested_tool_roles")
        role_configs[role_key] = record
    return role_configs


def _tool_roles_from_configs(role_configs: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for role_name, raw_config in role_config_records(role_configs).items():
        tool_id = str(raw_config.get("tool_id", "") or "").strip().upper()
        if tool_id:
            result[tool_id] = str(role_name)
    return result


def _two_segment_pose_observation(
    *,
    session: ExperimentSession,
    config: TwoSegmentCollectPoseDatasetConfig,
    run_trust_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    role_configs = _effective_role_configs(session=session, config=config)
    if not bool(config.capture_tracker_snapshot) or session.context.tracking_service is None:
        resolved = resolve_two_segment_tracking_roles(
            snapshot=None,
            role_configs=role_configs,
            require_model_training_roles=not _servo_only_mode(config),
        )
        observation = dict(resolved.get("two_segment_pose", {}) or {})
        observation["status"] = "tracker_unavailable"
        return observation, {
            "pose_observations_present": False,
            "backend_health": {"canonical_state": "disabled", "run_trust_mode": run_trust_mode},
            "tool_ids_seen": [],
            "transform_validity": {},
            "pose_in_tracker_frame": {},
            "pose_in_robot_frame": {},
            "pose_trust_by_role": dict(resolved.get("pose_trust_by_role", {}) or {}),
            "tracker_freshness_by_role": dict(resolved.get("tracker_freshness_by_role", {}) or {}),
            "available_roles": list(resolved.get("available_roles", []) or []),
            "missing_roles": list(resolved.get("missing_roles", []) or []),
            "stale_roles": list(resolved.get("stale_roles", []) or []),
            "invalid_roles": list(resolved.get("invalid_roles", []) or []),
            "missing_required_roles": list(resolved.get("missing_required_roles", []) or []),
            "role_observations": dict(resolved.get("role_observations", {}) or {}),
            "role_config": dict(resolved.get("role_config", {}) or {}),
            "distal_only": bool((resolved.get("dataset_validity", {}) or {}).get("distal_only", False)),
            "includes_intermediate_pose": bool((resolved.get("dataset_validity", {}) or {}).get("includes_intermediate_pose", False)),
        }
    try:
        reader = getattr(session.context.tracking_service, "peek_snapshot", None)
        snapshot = reader() if callable(reader) else session.context.tracking_service.get_snapshot()
    except Exception as exc:
        resolved = resolve_two_segment_tracking_roles(
            snapshot=None,
            role_configs=role_configs,
            require_model_training_roles=not _servo_only_mode(config),
        )
        observation = dict(resolved.get("two_segment_pose", {}) or {})
        observation["status"] = f"tracker_error:{exc}"
        return observation, {
            "pose_observations_present": False,
            "backend_health": {"error": str(exc)},
            "tool_ids_seen": [],
            "available_roles": [],
            "missing_roles": list(resolved.get("missing_roles", []) or []),
            "stale_roles": [],
            "invalid_roles": [],
            "missing_required_roles": list(resolved.get("missing_required_roles", []) or []),
            "role_observations": dict(resolved.get("role_observations", {}) or {}),
            "role_config": dict(resolved.get("role_config", {}) or {}),
        }
    resolved = resolve_two_segment_tracking_roles(
        snapshot=snapshot,
        role_configs=role_configs,
        require_model_training_roles=not _servo_only_mode(config),
    )
    observation = dict(resolved.get("two_segment_pose", {}) or {})
    dataset_validity = dict(resolved.get("dataset_validity", {}) or {})
    return observation, {
        "pose_observations_present": bool(resolved.get("available_roles")),
        "backend_health": {
            "canonical_state": getattr(snapshot, "canonical_state", ""),
            "registration_state": getattr(snapshot, "registration_state", ""),
            "run_trust_mode": run_trust_mode,
        },
        "tracker_frame_id": getattr(snapshot, "frame_id", None),
        "freshness_s": getattr(snapshot, "tracker_data_age_s", None),
        "tool_ids_seen": list(resolved.get("tool_ids_seen", []) or []),
        "transform_validity": dict(resolved.get("transform_validity", {}) or {}),
        "pose_in_tracker_frame": dict(resolved.get("pose_in_tracker_frame", {}) or {}),
        "pose_in_robot_frame": dict(resolved.get("pose_in_robot_frame", {}) or {}),
        "pose_trust_by_role": dict(resolved.get("pose_trust_by_role", {}) or {}),
        "tracker_freshness_by_role": dict(resolved.get("tracker_freshness_by_role", {}) or {}),
        "available_roles": list(resolved.get("available_roles", []) or []),
        "missing_roles": list(resolved.get("missing_roles", []) or []),
        "stale_roles": list(resolved.get("stale_roles", []) or []),
        "invalid_roles": list(resolved.get("invalid_roles", []) or []),
        "missing_required_roles": list(resolved.get("missing_required_roles", []) or []),
        "role_observations": dict(resolved.get("role_observations", {}) or {}),
        "role_config": dict(resolved.get("role_config", {}) or {}),
        "distal_only": bool(dataset_validity.get("distal_only")),
        "includes_intermediate_pose": bool(dataset_validity.get("includes_intermediate_pose")),
    }


def _two_segment_registration_summary(*, session) -> dict[str, Any]:
    """Capture registration provenance from the tracking snapshot if available.

    Mirrors the same helper in two_segment_repeatability.py. Surfaces the
    registration file path, FRE, timestamp, and state so two runs can be
    honestly compared. Returns a placeholder block when no tracker is connected.
    """
    placeholder = {
        "schema_version": "two_segment_registration_summary_v1",
        "registration_state": "tracker_unavailable",
        "registration_path": None,
        "stored_registration_fre_mm": None,
        "stored_registration_timestamp_utc": None,
        "stored_registration_measurement_tool_id": None,
        "stored_registration_coil_tool_id": None,
        "runtime_tip_mode": None,
        "runtime_tip_trust_level": None,
    }
    tracking_service = getattr(session.context, "tracking_service", None)
    if tracking_service is None:
        return placeholder
    try:
        reader = getattr(tracking_service, "peek_snapshot", None)
        snapshot = reader() if callable(reader) else tracking_service.get_snapshot()
    except Exception as exc:
        placeholder["registration_state"] = f"tracker_error:{exc}"
        return placeholder
    if snapshot is None:
        return placeholder
    return {
        "schema_version": "two_segment_registration_summary_v1",
        "registration_state": getattr(snapshot, "registration_state", "missing_registration"),
        "registration_path": getattr(snapshot, "registration_path", None),
        "stored_registration_fre_mm": getattr(snapshot, "stored_registration_fre_mm", None),
        "stored_registration_timestamp_utc": getattr(snapshot, "stored_registration_timestamp_utc", None),
        "stored_registration_measurement_tool_id": getattr(snapshot, "stored_registration_measurement_tool_id", None),
        "stored_registration_coil_tool_id": getattr(snapshot, "stored_registration_coil_tool_id", None),
        "runtime_tip_mode": getattr(snapshot, "runtime_tip_mode", None),
        "runtime_tip_trust_level": getattr(snapshot, "runtime_tip_trust_level", None),
    }


def pose_fields_have_distal_robot_frame(fields: dict[str, Any]) -> bool:
    """Return True when the resolved pose fields include a robot-frame distal label."""

    roles = dict(dict(fields.get("pose_in_robot_frame") or {}).get("roles") or {})
    distal = dict(roles.get("distal_tip") or {})
    if _matrix_has_translation(distal.get("T_robot_tip")):
        return True
    translation = distal.get("translation_mm")
    return bool(isinstance(translation, list) and len(translation) >= 3)


def _matrix_has_translation(value: Any) -> bool:
    if not isinstance(value, list) or len(value) < 3:
        return False
    try:
        return all(len(list(value[index])) >= 4 for index in range(3))
    except (TypeError, ValueError):
        return False


def _two_segment_tracker_freshness_summary(
    *,
    samples: list[ExperimentTimeseriesSample],
    stale_threshold_s: float = 0.25,
) -> dict[str, Any]:
    """Aggregate per-run tracker freshness statistics.

    Captures the operator-visible distinction between "tracker is fast" and
    "tracker is silently dropping frames." Counts samples where
    ``freshness_s`` exceeded ``stale_threshold_s`` and reports the worst
    observation. Descriptive only — does not change safety enforcement.
    """

    freshness_values: list[float] = []
    samples_with_freshness = 0
    samples_stale = 0
    samples_with_stale_role_observation = 0
    for sample in samples:
        freshness = getattr(sample, "freshness_s", None)
        if freshness is not None:
            try:
                value = float(freshness)
            except (TypeError, ValueError):
                value = None
            if value is not None:
                samples_with_freshness += 1
                freshness_values.append(value)
                if value > float(stale_threshold_s):
                    samples_stale += 1
        extra = dict(getattr(sample, "extra", {}) or {})
        if list(extra.get("stale_pose_roles") or []):
            samples_with_stale_role_observation += 1
    if not freshness_values:
        return {
            "schema_version": "two_segment_tracker_freshness_summary_v1",
            "stale_threshold_s": float(stale_threshold_s),
            "samples_with_freshness": 0,
            "samples_stale": 0,
            "samples_with_stale_role_observation": int(samples_with_stale_role_observation),
            "max_freshness_s": None,
            "p95_freshness_s": None,
            "median_freshness_s": None,
            "mean_freshness_s": None,
            "any_stale_observed": False,
        }
    arr = sorted(freshness_values)
    n = len(arr)
    p95_index = max(0, min(n - 1, int(round(0.95 * (n - 1)))))
    return {
        "schema_version": "two_segment_tracker_freshness_summary_v1",
        "stale_threshold_s": float(stale_threshold_s),
        "samples_with_freshness": int(samples_with_freshness),
        "samples_stale": int(samples_stale),
        "samples_with_stale_role_observation": int(samples_with_stale_role_observation),
        "max_freshness_s": float(arr[-1]),
        "p95_freshness_s": float(arr[p95_index]),
        "median_freshness_s": float(arr[n // 2]),
        "mean_freshness_s": float(sum(arr) / n),
        "any_stale_observed": bool(samples_stale > 0 or samples_with_stale_role_observation > 0),
    }


def _two_segment_current_load_summary(
    *,
    samples: list[ExperimentTimeseriesSample],
    warning_ma: int,
    hard_ma: int,
    sustained_sample_count: int = 3,
) -> dict[str, Any]:
    """Aggregate per-servo current/load statistics across the run.

    Surfaces mean / median / p95 / max load_proxy_ma per servo, plus the count
    of samples exceeding warning / hard thresholds and any servo with
    `sustained_sample_count` consecutive samples >= the warning threshold.
    Descriptive only — does not change safety enforcement.
    """

    per_servo_load: dict[int, list[float]] = {}
    per_servo_signed: dict[int, list[float]] = {}
    per_servo_warning_count: dict[int, int] = {}
    per_servo_hard_count: dict[int, int] = {}
    per_servo_consecutive_warning: dict[int, int] = {}
    sustained_jam_servo_ids: set[int] = set()

    for sample in samples:
        extra = dict(getattr(sample, "extra", {}) or {})
        feedback = dict(extra.get("measured_servo_feedback") or {})
        # Track per-servo consecutive warning counts in capture order so
        # transient spikes don't get flagged as sustained.
        seen_servo_ids: set[int] = set()
        for raw_id, data in feedback.items():
            try:
                servo_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            seen_servo_ids.add(servo_id)
            entry = dict(data or {})
            load = entry.get("load_proxy_ma")
            signed = entry.get("signed_raw_current_ma")
            if load is None:
                # Don't reset the consecutive counter when telemetry is missing;
                # a packet drop ≠ a known-low current. Keep the count flat.
                continue
            per_servo_load.setdefault(servo_id, []).append(float(load))
            if signed is not None:
                per_servo_signed.setdefault(servo_id, []).append(float(signed))
            if float(load) >= float(warning_ma):
                per_servo_warning_count[servo_id] = per_servo_warning_count.get(servo_id, 0) + 1
                per_servo_consecutive_warning[servo_id] = per_servo_consecutive_warning.get(servo_id, 0) + 1
                if per_servo_consecutive_warning[servo_id] >= int(sustained_sample_count):
                    sustained_jam_servo_ids.add(servo_id)
            else:
                per_servo_consecutive_warning[servo_id] = 0
            if float(load) >= float(hard_ma):
                per_servo_hard_count[servo_id] = per_servo_hard_count.get(servo_id, 0) + 1
        # Reset consecutive counters for servos not seen at all in this sample
        # (sample-level absence == reset).
        for servo_id in list(per_servo_consecutive_warning):
            if servo_id not in seen_servo_ids:
                per_servo_consecutive_warning[servo_id] = 0

    def _stats(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"mean_ma": None, "median_ma": None, "p95_ma": None, "max_ma": None, "min_ma": None}
        arr = sorted(values)
        n = len(arr)
        p95_index = max(0, min(n - 1, int(round(0.95 * (n - 1)))))
        return {
            "mean_ma": float(sum(arr) / n),
            "median_ma": float(arr[n // 2]),
            "p95_ma": float(arr[p95_index]),
            "max_ma": float(arr[-1]),
            "min_ma": float(arr[0]),
        }

    per_servo_summary = {}
    for servo_id in sorted(set(list(per_servo_load) + list(per_servo_signed))):
        per_servo_summary[str(servo_id)] = {
            "servo_id": int(servo_id),
            "load_proxy": _stats(per_servo_load.get(servo_id, [])),
            "signed_current": _stats(per_servo_signed.get(servo_id, [])),
            "warning_threshold_exceeded_count": int(per_servo_warning_count.get(servo_id, 0)),
            "hard_threshold_exceeded_count": int(per_servo_hard_count.get(servo_id, 0)),
            "sample_count": len(per_servo_load.get(servo_id, [])),
        }

    return {
        "schema_version": "two_segment_current_load_summary_v1",
        "warning_threshold_ma": int(warning_ma),
        "hard_threshold_ma": int(hard_ma),
        "sustained_sample_count": int(sustained_sample_count),
        "per_servo": per_servo_summary,
        "sustained_jam_servo_ids": sorted(sustained_jam_servo_ids),
        "any_warning_observed": any(int(count) > 0 for count in per_servo_warning_count.values()),
        "any_hard_observed": any(int(count) > 0 for count in per_servo_hard_count.values()),
        "note": (
            "Descriptive per-servo current/load aggregation across the run. "
            "Does not change safety enforcement; see SafetyConfig for hard stops."
        ),
    }


def _pose_label_summary(samples: list[ExperimentTimeseriesSample]) -> dict[str, Any]:
    distal = 0
    distal_robot_frame = 0
    intermediate = 0
    observed = 0
    orientation = 0
    missing_required: set[str] = set()
    available_roles: set[str] = set()
    for sample in samples:
        pose = dict(sample.two_segment_pose or {})
        if dict(pose.get("distal_tip_pose", {}) or {}):
            distal += 1
        if pose_fields_have_distal_robot_frame(
            {"pose_in_robot_frame": dict(getattr(sample, "pose_in_robot_frame", {}) or {})}
        ):
            distal_robot_frame += 1
        if dict(pose.get("intermediate_pose", {}) or {}):
            intermediate += 1
        if bool(sample.extra.get("pose_observations_present")):
            observed += 1
        if bool(sample.extra.get("includes_orientation", False)):
            orientation += 1
        missing_required.update(str(value) for value in list(sample.extra.get("missing_required_pose_roles", []) or []))
        available_roles.update(str(value) for value in list(sample.extra.get("available_pose_roles", []) or []))
    return {
        "pose_observation_sample_count": int(observed),
        "distal_pose_sample_count": int(distal),
        "distal_robot_frame_pose_sample_count": int(distal_robot_frame),
        "intermediate_pose_sample_count": int(intermediate),
        "includes_intermediate_pose": bool(intermediate),
        "distal_only": bool(distal and not intermediate),
        "includes_orientation": bool(orientation),
        "available_roles": sorted(available_roles),
        "missing_required_roles": sorted(missing_required),
    }


def _pose_label_summary_from_fields(fields: dict[str, Any]) -> dict[str, Any]:
    available_roles = set(str(value) for value in list(fields.get("available_roles", []) or []))
    includes_orientation = _pose_fields_include_orientation(fields)
    distal_robot_frame = pose_fields_have_distal_robot_frame(fields)
    return {
        "pose_observation_sample_count": int(bool(fields.get("pose_observations_present"))),
        "distal_pose_sample_count": int("distal_tip" in available_roles),
        "distal_robot_frame_pose_sample_count": int(distal_robot_frame),
        "intermediate_pose_sample_count": int("intermediate_segment" in available_roles),
        "missing_required_roles": list(fields.get("missing_required_roles", []) or []),
        "includes_intermediate_pose": "intermediate_segment" in available_roles,
        "distal_only": "distal_tip" in available_roles and "intermediate_segment" not in available_roles,
        "includes_orientation": bool(includes_orientation),
    }


def _pose_fields_include_orientation(fields: dict[str, Any]) -> bool:
    for role_payload in dict(fields.get("role_observations", {}) or {}).values():
        record = dict(role_payload or {})
        pose = dict(record.get("robot_frame_pose") or record.get("pose") or {})
        if pose.get("tangent_xyz") or pose.get("orientation_tangent_xyz"):
            return True
    return False


def _dataset_label_mode_from_pose_summary(pose_summary: dict[str, Any]) -> str:
    distal = int(pose_summary.get("distal_pose_sample_count", 0) or 0) > 0
    intermediate = int(pose_summary.get("intermediate_pose_sample_count", 0) or 0) > 0
    orientation = bool(pose_summary.get("includes_orientation", False))
    if intermediate and orientation:
        return "two_coil_pose12"
    if intermediate:
        return "two_coil_xyz"
    if distal and orientation:
        return "distal_pose6"
    return "distal_xyz" if distal else "unavailable"


def _valid_for_two_segment_model_training(
    config: TwoSegmentCollectPoseDatasetConfig,
    pose_summary: dict[str, Any],
    startup_provenance: dict[str, Any],
    command_failures: int,
) -> bool:
    return bool(
        not _servo_only_mode(config)
        and bool(startup_provenance.get("accepted_all_8_startup"))
        and int(pose_summary.get("distal_pose_sample_count", 0) or 0) > 0
        and int(pose_summary.get("distal_robot_frame_pose_sample_count", 0) or 0) > 0
        and "distal_tip" not in list(pose_summary.get("missing_required_roles", []) or [])
        and int(command_failures) == 0
    )


def _data_quality_warnings(*, config: TwoSegmentCollectPoseDatasetConfig, pose_summary: dict[str, Any], startup_provenance: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if _servo_only_mode(config):
        warnings.append("servo_only_or_dry_run_not_model_training_valid")
    if not bool(startup_provenance.get("accepted_all_8_startup")):
        warnings.append("accepted_all_8_startup_artifact_not_available")
    if int(pose_summary.get("pose_observation_sample_count", 0) or 0) == 0:
        warnings.append("pose_labels_missing")
    if int(pose_summary.get("distal_robot_frame_pose_sample_count", 0) or 0) == 0:
        warnings.append("distal_tip_robot_frame_pose_missing")
    missing_required = list(pose_summary.get("missing_required_roles", []) or [])
    if "distal_tip" in missing_required:
        warnings.append("required_pose_roles_missing")
    if not bool(pose_summary.get("includes_intermediate_pose")):
        warnings.append("distal_only_no_intermediate_pose")
    return sorted(set(warnings))


def _sample_warnings(*, config: TwoSegmentCollectPoseDatasetConfig, missing_pose: bool, startup_provenance: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if _servo_only_mode(config):
        warnings.append("servo_only_or_dry_run")
    if missing_pose:
        warnings.append("pose_labels_missing")
    if not bool(startup_provenance.get("accepted_all_8_startup")):
        warnings.append("startup_artifact_not_accepted")
    return sorted(set(warnings))


def _write_summary_text(path: Path, metrics: dict[str, Any]) -> None:
    pose = dict(metrics.get("pose_label_summary", {}) or {})
    long_run = dict(metrics.get("long_run_health") or {})
    current_load = dict(metrics.get("current_load_summary") or {})
    tracker_fresh = dict(metrics.get("tracker_freshness_summary") or {})
    lines = [
        "Two-Segment Collect-Pose Command Dataset",
        f"dataset_schema_version: {metrics.get('dataset_schema_version', TWO_SEGMENT_DATASET_SCHEMA_VERSION)}",
        f"schedule_type: {metrics.get('schedule_type')}",
        f"run_trust_mode: {metrics.get('run_trust_mode')}",
        f"valid_for_two_segment_model_training: {metrics.get('valid_for_two_segment_model_training')}",
        f"valid_for_two_segment_ann_training: {metrics.get('valid_for_two_segment_ann_training')}",
        f"label_mode_used: {metrics.get('label_mode_used')}",
        f"valid_for_model_training: {metrics.get('valid_for_model_training')}",
        f"bottom_segment_key: {metrics.get('bottom_segment_key', 'n/a')}",
        f"top_segment_key: {metrics.get('top_segment_key', 'n/a')}",
        f"accepted_sample_count: {metrics.get('accepted_sample_count')}",
        f"rejected_sample_count: {metrics.get('rejected_sample_count')}",
        f"command_failure_count: {metrics.get('command_failure_count', 0)}",
        f"pose_observation_sample_count: {pose.get('pose_observation_sample_count', 0)}",
        f"distal_only: {metrics.get('distal_only')}",
        f"includes_intermediate_pose: {metrics.get('includes_intermediate_pose')}",
        f"includes_orientation: {metrics.get('includes_orientation')}",
        f"physical_assembly_confirmed_by_operator: {metrics.get('physical_assembly_confirmed_by_operator')}",
        "automatic_two_segment_pretension_validated: false",
        "two_segment_control_validated: false",
    ]
    if long_run:
        lines.extend([
            "",
            "Long-Run Health:",
            f"  stop_reason: {long_run.get('stop_reason', 'n/a')}",
            f"  cycles_completed: {long_run.get('cycles_completed', 0)}",
            f"  transport_failures: {long_run.get('transport_failures', 0)}",
            f"  continue_until_valid_samples: {long_run.get('continue_until_valid_samples', False)}",
            f"  target_valid_sample_count: {long_run.get('target_valid_sample_count', 'n/a')}",
        ])
    if current_load:
        sustained = list(current_load.get("sustained_jam_servo_ids") or [])
        lines.extend([
            "",
            "Current/Load Summary:",
            f"  warning_threshold_ma: {current_load.get('warning_threshold_ma', 'n/a')}",
            f"  hard_threshold_ma: {current_load.get('hard_threshold_ma', 'n/a')}",
            f"  any_warning_observed: {current_load.get('any_warning_observed', False)}",
            f"  any_hard_observed: {current_load.get('any_hard_observed', False)}",
            f"  sustained_jam_servo_ids: {sustained or 'none'}",
        ])
    if tracker_fresh:
        lines.extend([
            "",
            "Tracker Freshness:",
            f"  stale_threshold_s: {tracker_fresh.get('stale_threshold_s', 'n/a')}",
            f"  samples_with_freshness: {tracker_fresh.get('samples_with_freshness', 0)}",
            f"  samples_stale: {tracker_fresh.get('samples_stale', 0)}",
            f"  max_freshness_s: {tracker_fresh.get('max_freshness_s', 'n/a')}",
            f"  p95_freshness_s: {tracker_fresh.get('p95_freshness_s', 'n/a')}",
            f"  any_stale_observed: {tracker_fresh.get('any_stale_observed', False)}",
        ])
    registration = dict(metrics.get("registration_summary") or {})
    if registration:
        lines.extend([
            "",
            "Registration Provenance:",
            f"  registration_state: {registration.get('registration_state', 'unknown')}",
            f"  registration_path: {registration.get('registration_path', 'n/a')}",
            f"  stored_registration_fre_mm: {registration.get('stored_registration_fre_mm', 'n/a')}",
            f"  stored_registration_timestamp_utc: {registration.get('stored_registration_timestamp_utc', 'n/a')}",
            f"  runtime_tip_mode: {registration.get('runtime_tip_mode', 'n/a')}",
            f"  runtime_tip_trust_level: {registration.get('runtime_tip_trust_level', 'n/a')}",
        ])
    lines.extend([
        "",
        "This dataset stores structured Segment A/B commands and optional pose observations.",
        "It does not implement two-segment kinematics, control, modeling, chasing, or automatic pretension.",
    ])
    Path(path).write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _write_metrics_csv(path: Path, samples: list[ExperimentTimeseriesSample]) -> None:
    fields = [
        "step_index",
        "sample_index",
        "phase",
        "servo_id",
        "segment",
        "command_cm",
        "goal_tick",
        "position_tick",
        "signed_raw_current_ma",
        "load_proxy_ma",
        "capture_accepted",
        "valid_for_two_segment_model_training",
        "valid_for_two_segment_ann_training",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            extra = dict(sample.extra or {})
            segments = dict(extra.get("segments", {}) or {})
            segment_by_servo = {
                int(servo_id): str(key)
                for key, segment in segments.items()
                for servo_id in list(dict(segment or {}).get("servo_ids", []) or [])
            }
            command_by_servo = {
                int(key): float(value)
                for key, value in dict((sample.two_segment_command or {}).get("servo_command_cm", {}) or {}).items()
            }
            feedback = dict(extra.get("measured_servo_feedback", {}) or {})
            goals = dict(extra.get("goal_ticks_by_servo", {}) or {})
            for raw_servo_id in sorted(set(command_by_servo) | {int(value) for value in feedback.keys()}):
                data = dict(feedback.get(str(raw_servo_id), {}) or {})
                writer.writerow(
                    {
                        "step_index": sample.step_index,
                        "sample_index": sample.sample_index,
                        "phase": sample.phase,
                        "servo_id": int(raw_servo_id),
                        "segment": segment_by_servo.get(int(raw_servo_id), ""),
                        "command_cm": command_by_servo.get(int(raw_servo_id)),
                        "goal_tick": goals.get(str(raw_servo_id)),
                        "position_tick": data.get("position_tick"),
                        "signed_raw_current_ma": data.get("signed_raw_current_ma"),
                        "load_proxy_ma": data.get("load_proxy_ma"),
                        "capture_accepted": extra.get("capture_accepted"),
                        "valid_for_two_segment_model_training": extra.get("valid_for_two_segment_model_training"),
                        "valid_for_two_segment_ann_training": extra.get("valid_for_two_segment_ann_training"),
                    }
                )


def _write_command_coverage(path: Path, samples: list[ExperimentTimeseriesSample]) -> None:
    a_mags: list[float] = []
    b_mags: list[float] = []
    for sample in samples:
        segments = dict((sample.two_segment_command or {}).get("segments", {}) or {})
        a = [float(value) for value in list(segments.get("segment_a", []) or [])]
        b = [float(value) for value in list(segments.get("segment_b", []) or [])]
        if a and b:
            a_mags.append(sum(value * value for value in a) ** 0.5 * 10.0)
            b_mags.append(sum(value * value for value in b) ** 0.5 * 10.0)
    fig, ax = create_figure(size="wide")
    if a_mags and b_mags:
        ax.scatter(a_mags, b_mags, color=color("measured"))
    else:
        ax.text(0.5, 0.5, "No command samples", ha="center", va="center", transform=ax.transAxes)
    style_axes(ax, title="Two-Segment Command Coverage", xlabel="Segment A command norm (mm)", ylabel="Segment B command norm (mm)")
    save_figure(fig, path)


def _write_servo_position_coverage(path: Path, samples: list[ExperimentTimeseriesSample]) -> None:
    values: dict[int, list[int]] = {servo_id: [] for servo_id in range(1, 9)}
    for sample in samples:
        feedback = dict((sample.extra or {}).get("measured_servo_feedback", {}) or {})
        for raw_id, data in feedback.items():
            position = dict(data or {}).get("position_tick")
            if position is not None:
                values.setdefault(int(raw_id), []).append(int(position))
    servo_ids = sorted(values)
    ranges = [(max(values[sid]) - min(values[sid])) if values[sid] else 0 for sid in servo_ids]
    fig, ax = create_figure(size="wide")
    ax.bar(servo_ids, ranges, color=[color("segment_a") if sid <= 4 else color("segment_b") for sid in servo_ids])
    ax.axvline(4.5, color=color("threshold"), linestyle="--", linewidth=1.0)
    style_axes(ax, title="Servo Position Coverage", xlabel="Servo ID", ylabel="Observed range (ticks)")
    ax.set_xticks(servo_ids)
    save_figure(fig, path)


def _write_pose_coverage(path: Path, samples: list[ExperimentTimeseriesSample], metrics: dict[str, Any]) -> None:
    xs: list[float] = []
    ys: list[float] = []
    for sample in samples:
        pose = dict(sample.two_segment_pose or {})
        distal = dict(pose.get("distal_tip_pose", {}) or {})
        point = distal.get("translation_mm")
        if isinstance(point, list) and len(point) >= 2:
            xs.append(float(point[0]))
            ys.append(float(point[1]))
    fig, ax = create_figure(size="wide")
    if xs and ys:
        ax.scatter(xs, ys, color=color("accepted"))
        message = "distal/intermediate pose labels present"
    else:
        warnings = ", ".join(str(value) for value in list(metrics.get("data_quality_warnings", []) or []))
        message = f"Servo-only / missing pose labels. Not trainable. {warnings}".strip()
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, wrap=True)
    style_axes(ax, title="Two-Segment Pose Coverage", xlabel="X (mm)", ylabel="Y (mm)")
    save_figure(fig, path)


def _write_quality_report(path: Path, metrics: dict[str, Any]) -> None:
    labels = ["accepted", "rejected", "command failures"]
    values = [
        int(metrics.get("accepted_sample_count", 0) or 0),
        int(metrics.get("rejected_sample_count", 0) or 0),
        int(metrics.get("command_failure_count", 0) or 0),
    ]
    fig, ax = create_figure(size="wide")
    ax.bar(labels, values, color=[color("accepted"), color("rejected"), color("threshold")])
    title = "Dataset Quality"
    subtitle = f"valid_for_two_segment_model_training={metrics.get('valid_for_two_segment_model_training')}"
    style_axes(ax, title=f"{title}\n{subtitle}", xlabel="", ylabel="Samples")
    save_figure(fig, path)
