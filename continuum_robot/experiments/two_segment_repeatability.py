"""Two-segment repeatability scaffold experiment.

This experiment defines a structured target set for a stacked two-segment
robot and records per-target distal/intermediate poses plus all-8 servo
telemetry. It does NOT implement closed-loop chasing or two-segment control:
targets are open-loop tendon-displacement commands, applied bottom and top.

The scaffold is honest about provenance and trust:
- Acceptance is open-loop (commanded vs measured pose).
- The result is repeatability of the open-loop motion plus tracker, not of a
  closed-loop controller.
- No <1 mm thesis target is hardcoded; success criteria are configured by the
  operator and reported alongside the measured metrics.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from continuum_robot.experiments.framework import BaseExperiment, ExperimentHardwareRequirements, ExperimentSession
from continuum_robot.experiments.plotting import color, create_figure, save_figure, style_axes
from continuum_robot.experiments.schemas import ExperimentTimeseriesSample
from continuum_robot.experiments.two_segment_collect_pose_dataset import (
    _effective_role_configs,
    _read_live_telemetry,
    _resolve_startup_ticks,
    _servo_feedback_payload,
    _two_segment_pose_observation,
)
from continuum_robot.two_segment import (
    TwoSegmentCommand,
    build_two_segment_foundation_metadata,
    two_segment_command_schema,
)


EXPERIMENT_NAME = "two_segment_repeatability"
REPEATABILITY_SCHEMA_VERSION = "two_segment_repeatability_v1"
BLOCK_MESSAGE = (
    "Two-segment repeatability requires operating_mode=dual_segment, all 8 servos, "
    "and an accepted all-8 manual startup foundation. Open-loop repeatability only; no closed-loop control is implemented."
)


@dataclass(frozen=True)
class TwoSegmentRepeatabilityTarget:
    """One bottom/top target in tendon space."""

    target_index: int
    label: str
    group_tag: str
    bottom_tendon_cm: list[float]
    top_tendon_cm: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_index": int(self.target_index),
            "label": str(self.label),
            "group_tag": str(self.group_tag),
            "bottom_tendon_cm": list(self.bottom_tendon_cm),
            "top_tendon_cm": list(self.top_tendon_cm),
        }


@dataclass
class TwoSegmentRepeatabilityConfig:
    """Configuration for the two-segment repeatability scaffold."""

    target_set: str = "default_demo"
    bottom_inner_radius_cm: float = 0.10
    bottom_outer_radius_cm: float = 0.20
    top_inner_radius_cm: float = 0.10
    top_outer_radius_cm: float = 0.20
    points_per_ring: int = 4
    repeat_visits: int = 3
    settle_time_s: float = 1.5
    samples_per_visit: int = 1
    max_tick_delta_from_startup: int = 320
    allow_servo_only_test_run: bool = True
    run_trust_mode: str = "servo_only"
    capture_tracker_snapshot: bool = True
    requested_tool_roles: dict[str, str] = field(default_factory=dict)
    dataset_tag: str = ""
    target_distal_rms_mm: float | None = None  # operator-configured success target; reported, not hardcoded
    target_intermediate_rms_mm: float | None = None
    # The shared startup-tick resolver from the collect-pose experiment expects
    # these attributes. Repeatability runs are always live (no `dry_run`).
    dry_run: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "TwoSegmentRepeatabilityConfig":
        payload = dict(payload or {})
        requested_roles = payload.get("requested_tool_roles") or payload.get("tool_roles") or {}
        return cls(
            target_set=str(payload.get("target_set", "default_demo") or "default_demo").strip().lower(),
            bottom_inner_radius_cm=max(0.0, float(payload.get("bottom_inner_radius_cm", 0.10))),
            bottom_outer_radius_cm=max(0.0, float(payload.get("bottom_outer_radius_cm", 0.20))),
            top_inner_radius_cm=max(0.0, float(payload.get("top_inner_radius_cm", 0.10))),
            top_outer_radius_cm=max(0.0, float(payload.get("top_outer_radius_cm", 0.20))),
            points_per_ring=max(1, int(payload.get("points_per_ring", 4))),
            repeat_visits=max(1, int(payload.get("repeat_visits", 3))),
            settle_time_s=max(0.0, float(payload.get("settle_time_s", 1.5))),
            samples_per_visit=max(1, int(payload.get("samples_per_visit", 1))),
            max_tick_delta_from_startup=max(0, int(payload.get("max_tick_delta_from_startup", 320))),
            allow_servo_only_test_run=bool(payload.get("allow_servo_only_test_run", True)),
            run_trust_mode=str(payload.get("run_trust_mode", "servo_only") or "servo_only").strip().lower(),
            capture_tracker_snapshot=bool(payload.get("capture_tracker_snapshot", True)),
            requested_tool_roles={str(key).upper(): str(value) for key, value in dict(requested_roles or {}).items()},
            dataset_tag=str(payload.get("dataset_tag", "") or ""),
            target_distal_rms_mm=(
                None if payload.get("target_distal_rms_mm") in (None, "")
                else float(payload.get("target_distal_rms_mm"))
            ),
            target_intermediate_rms_mm=(
                None if payload.get("target_intermediate_rms_mm") in (None, "")
                else float(payload.get("target_intermediate_rms_mm"))
            ),
        )


def build_two_segment_repeatability_targets(config: TwoSegmentRepeatabilityConfig) -> list[TwoSegmentRepeatabilityTarget]:
    """Build the structured target set in tendon space.

    Each target is an open-loop tendon displacement for the bottom and top
    segments. The default set covers: center; inner/outer rings on the bottom
    alone; inner/outer rings on the top alone; and a small set of combined
    bottom+top targets.
    """

    targets: list[TwoSegmentRepeatabilityTarget] = []
    targets.append(
        TwoSegmentRepeatabilityTarget(
            target_index=0,
            label="center",
            group_tag="center",
            bottom_tendon_cm=[0.0, 0.0, 0.0, 0.0],
            top_tendon_cm=[0.0, 0.0, 0.0, 0.0],
        )
    )

    def _ring_points(radius_cm: float, points: int) -> list[tuple[float, float]]:
        if radius_cm <= 0.0 or points < 1:
            return []
        return [
            (
                float(radius_cm * np.cos(2.0 * np.pi * idx / points)),
                float(radius_cm * np.sin(2.0 * np.pi * idx / points)),
            )
            for idx in range(points)
        ]

    def _tendon_vector(x_cm: float, y_cm: float) -> list[float]:
        return [float(x_cm), float(y_cm), -float(x_cm), -float(y_cm)]

    next_index = 1
    # Bottom-only inner ring
    for idx, (x, y) in enumerate(_ring_points(config.bottom_inner_radius_cm, config.points_per_ring)):
        targets.append(
            TwoSegmentRepeatabilityTarget(
                target_index=next_index,
                label=f"bottom_inner_{idx:02d}",
                group_tag="bottom_only_inner",
                bottom_tendon_cm=_tendon_vector(x, y),
                top_tendon_cm=[0.0, 0.0, 0.0, 0.0],
            )
        )
        next_index += 1
    # Bottom-only outer ring
    for idx, (x, y) in enumerate(_ring_points(config.bottom_outer_radius_cm, config.points_per_ring)):
        targets.append(
            TwoSegmentRepeatabilityTarget(
                target_index=next_index,
                label=f"bottom_outer_{idx:02d}",
                group_tag="bottom_only_outer",
                bottom_tendon_cm=_tendon_vector(x, y),
                top_tendon_cm=[0.0, 0.0, 0.0, 0.0],
            )
        )
        next_index += 1
    # Top-only inner ring
    for idx, (x, y) in enumerate(_ring_points(config.top_inner_radius_cm, config.points_per_ring)):
        targets.append(
            TwoSegmentRepeatabilityTarget(
                target_index=next_index,
                label=f"top_inner_{idx:02d}",
                group_tag="top_only_inner",
                bottom_tendon_cm=[0.0, 0.0, 0.0, 0.0],
                top_tendon_cm=_tendon_vector(x, y),
            )
        )
        next_index += 1
    # Top-only outer ring
    for idx, (x, y) in enumerate(_ring_points(config.top_outer_radius_cm, config.points_per_ring)):
        targets.append(
            TwoSegmentRepeatabilityTarget(
                target_index=next_index,
                label=f"top_outer_{idx:02d}",
                group_tag="top_only_outer",
                bottom_tendon_cm=[0.0, 0.0, 0.0, 0.0],
                top_tendon_cm=_tendon_vector(x, y),
            )
        )
        next_index += 1
    # Combined bottom+top (smaller magnitude to stay safe)
    half_inner_bot = config.bottom_inner_radius_cm / 2.0
    half_inner_top = config.top_inner_radius_cm / 2.0
    for idx, theta in enumerate([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0]):
        targets.append(
            TwoSegmentRepeatabilityTarget(
                target_index=next_index,
                label=f"combined_{idx:02d}",
                group_tag="combined",
                bottom_tendon_cm=_tendon_vector(half_inner_bot * float(np.cos(theta)), half_inner_bot * float(np.sin(theta))),
                top_tendon_cm=_tendon_vector(half_inner_top * float(np.cos(theta)), half_inner_top * float(np.sin(theta))),
            )
        )
        next_index += 1
    return targets


class TwoSegmentRepeatabilityExperiment(BaseExperiment):
    """Open-loop two-segment repeatability scaffold."""

    name = EXPERIMENT_NAME
    description = (
        "Bonus/demo scaffold for two-segment repeatability. Open-loop tendon "
        "commands across structured bottom/top targets; reports per-target "
        "scatter for distal and (if available) intermediate poses. No closed-"
        "loop control or chasing is implemented."
    )
    hardware_requirements = ExperimentHardwareRequirements(
        tracking_required=False,
        servo_required=True,
        registration_required=False,
        mock_compatible=True,
    )

    def __init__(self, config: TwoSegmentRepeatabilityConfig) -> None:
        super().__init__(config)
        self.config: TwoSegmentRepeatabilityConfig = config
        self._startup_ticks_by_servo: dict[int, int] = {}
        self._startup_provenance: dict[str, Any] = {}
        self._targets: list[TwoSegmentRepeatabilityTarget] = []

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "TwoSegmentRepeatabilityExperiment":
        return cls(TwoSegmentRepeatabilityConfig.from_dict(payload))

    def setup(self, session: ExperimentSession) -> None:
        context = session.context.settings.robot.operating_context()
        self._startup_ticks_by_servo, self._startup_provenance = _resolve_startup_ticks(
            session=session,
            config=self.config,
            expected_ids=[int(value) for value in context.expected_servo_ids],
        )
        self._targets = build_two_segment_repeatability_targets(self.config)

    def precheck(self, session: ExperimentSession) -> None:
        context = session.context.settings.robot.operating_context()
        if context.operating_mode != "dual_segment":
            raise RuntimeError(BLOCK_MESSAGE)
        expected_ids = [int(value) for value in context.expected_servo_ids]
        if expected_ids != [1, 2, 3, 4, 5, 6, 7, 8]:
            raise RuntimeError(
                "Two-segment repeatability requires all 8 expected servo IDs; "
                f"resolved {expected_ids}."
            )
        assembly_issues = list(getattr(context, "physical_assembly_issues", []) or [])
        if assembly_issues:
            raise RuntimeError("Invalid physical assembly: " + "; ".join(assembly_issues))
        if session.context.servo_service is None or not bool(getattr(session.context.servo_service, "is_connected", False)):
            raise RuntimeError("Two-segment repeatability requires a connected ServoService.")
        if not bool(self._startup_provenance.get("accepted_all_8_startup")) and not bool(self.config.allow_servo_only_test_run):
            raise RuntimeError(
                "Two-segment repeatability requires an accepted all-8 manual startup artifact. "
                "Run two_segment_startup_validation first, or set allow_servo_only_test_run=true for a lower-trust demo."
            )
        session.set_stage("precheck", "passed", "two-segment repeatability precheck passed.")

    def execute(self, session: ExperimentSession) -> None:
        context = session.context.settings.robot.operating_context()
        bottom_key = context.bottom_segment_key or "segment_a"
        top_key = context.top_segment_key or "segment_b"
        role_configs = _effective_role_configs(session=session, config=self.config)
        foundation = build_two_segment_foundation_metadata(context)
        total_visits = len(self._targets) * max(1, int(self.config.repeat_visits)) * max(1, int(self.config.samples_per_visit))
        progress = 0
        run_trust_mode = self.config.run_trust_mode or "servo_only"
        target_records: list[dict[str, Any]] = []
        accepted = 0
        rejected = 0
        for visit_index in range(max(1, int(self.config.repeat_visits))):
            for target in self._targets:
                session.raise_if_stop_requested()
                # Issue tendon command in segment_a/segment_b coordinates
                mapping = {bottom_key: list(target.bottom_tendon_cm), top_key: list(target.top_tendon_cm)}
                mapping.setdefault("segment_a", [0.0, 0.0, 0.0, 0.0])
                mapping.setdefault("segment_b", [0.0, 0.0, 0.0, 0.0])
                command = TwoSegmentCommand.from_mapping(
                    {"segment_a": mapping["segment_a"], "segment_b": mapping["segment_b"]}
                )
                command_result = self._issue_command(session=session, command=command)
                if float(self.config.settle_time_s) > 0.0:
                    session.context.sleep_fn(float(self.config.settle_time_s))
                for sample_index in range(max(1, int(self.config.samples_per_visit))):
                    sample, pose_summary = self._capture_sample(
                        session=session,
                        command=command,
                        command_result=command_result,
                        target=target,
                        visit_index=visit_index,
                        sample_index=sample_index,
                        run_trust_mode=run_trust_mode,
                    )
                    session.add_sample(sample)
                    target_records.append({
                        "visit_index": int(visit_index),
                        "sample_index": int(sample_index),
                        "target": target.to_dict(),
                        "pose_summary": pose_summary,
                        "capture_accepted": bool(sample.extra.get("capture_accepted")),
                    })
                    accepted += int(bool(sample.extra.get("capture_accepted")))
                    rejected += int(not bool(sample.extra.get("capture_accepted")))
                    progress += 1
                    session.update_progress(progress, total_visits, {
                        "phase": "repeat_visit",
                        "target_label": target.label,
                        "accepted_samples": int(accepted),
                        "rejected_samples": int(rejected),
                    })
        scatter_metrics = _compute_repeatability_metrics(target_records)
        session.set_metric("schema_version", REPEATABILITY_SCHEMA_VERSION)
        session.set_metric("dataset_type", "two_segment_repeatability")
        session.set_metric("target_set", str(self.config.target_set))
        session.set_metric("target_count", len(self._targets))
        session.set_metric("repeat_visits", int(self.config.repeat_visits))
        session.set_metric("accepted_sample_count", int(accepted))
        session.set_metric("rejected_sample_count", int(rejected))
        session.set_metric("two_segment_foundation", foundation)
        session.set_metric("two_segment_command_schema", two_segment_command_schema(context))
        session.set_metric("bottom_segment_key", context.bottom_segment_key)
        session.set_metric("top_segment_key", context.top_segment_key)
        session.set_metric("bottom_servo_ids", list(context.bottom_servo_ids or []))
        session.set_metric("top_servo_ids", list(context.top_servo_ids or []))
        session.set_metric("physical_assembly", dict(context.metadata().get("physical_assembly") or {}))
        session.set_metric("startup_artifact_provenance", dict(self._startup_provenance))
        session.set_metric("run_trust_mode", run_trust_mode)
        session.set_metric("scatter_metrics", scatter_metrics)
        session.set_metric("target_distal_rms_mm", self.config.target_distal_rms_mm)
        session.set_metric("target_intermediate_rms_mm", self.config.target_intermediate_rms_mm)
        session.set_metric("valid_for_thesis_repeatability", False)
        session.set_metric("valid_for_two_segment_model_training", False)
        session.set_metric("automatic_two_segment_pretension_validated", False)
        session.set_metric("two_segment_control_validated", False)
        session.set_metric("target_records", target_records)
        session.set_metric("two_segment_tracking_role_config", dict(role_configs))
        # Registration provenance: every repeatability run should record what
        # registration was loaded (file path, FRE, age) so two runs from
        # different days can be honestly compared. If no tracker / no
        # registration, the fields default to descriptive placeholders.
        registration_summary = _two_segment_registration_summary(session=session)
        session.set_metric("registration_summary", registration_summary)
        if registration_summary.get("registration_state") not in {"loaded", "trusted"}:
            session.add_warning(
                f"Registration not fully loaded during repeatability "
                f"(state={registration_summary.get('registration_state')}); distal/intermediate "
                "scatter is in tracker frame only."
            )
        # Reuse the collect-pose helpers for per-servo current/load and tracker
        # freshness aggregation so the repeatability run has the same
        # observability bar.
        from continuum_robot.experiments.two_segment_collect_pose_dataset import (
            _two_segment_current_load_summary,
            _two_segment_tracker_freshness_summary,
        )

        safety = getattr(session.context.settings, "safety", None)
        current_load_summary = _two_segment_current_load_summary(
            samples=session.samples,
            warning_ma=int(getattr(safety, "servo_reported_current_warning_ma", None) or 800),
            hard_ma=int(getattr(safety, "servo_reported_current_hard_limit_ma", None) or 850),
        )
        session.set_metric("current_load_summary", current_load_summary)
        for servo_id in current_load_summary.get("sustained_jam_servo_ids", []) or []:
            session.add_warning(f"servo {servo_id} showed sustained current >= warning threshold during repeatability run")
        serial_cfg = getattr(session.context.settings, "serial", None)
        tracker_freshness_summary = _two_segment_tracker_freshness_summary(
            samples=session.samples,
            stale_threshold_s=float(getattr(serial_cfg, "tracker_max_stale_interval_s", None) or 0.25),
        )
        session.set_metric("tracker_freshness_summary", tracker_freshness_summary)
        if tracker_freshness_summary.get("samples_stale", 0):
            session.add_warning(
                f"tracker reported {tracker_freshness_summary['samples_stale']} stale samples during repeatability "
                f"(threshold={tracker_freshness_summary['stale_threshold_s']:.3f} s, "
                f"max freshness={tracker_freshness_summary['max_freshness_s']:.3f} s)"
            )

    def _issue_command(self, *, session: ExperimentSession, command: TwoSegmentCommand) -> dict[str, Any]:
        context = session.context.settings.robot.operating_context()
        flat_cm = command.to_flat(context=context)
        commanded_ids = [int(value) for value in context.commanded_servo_ids]
        startup_ticks = [int(self._startup_ticks_by_servo[servo_id]) for servo_id in commanded_ids]
        goal_ticks = session.context.servo_service.mapper.to_goal_positions(flat_cm, startup_ticks)
        goals_by_servo = {int(servo_id): int(goal) for servo_id, goal in zip(commanded_ids, goal_ticks)}
        result = {
            "success": True,
            "requested_flat_command_cm": list(flat_cm),
            "goal_ticks_by_servo": {str(key): int(value) for key, value in sorted(goals_by_servo.items())},
        }
        # Validate tick delta safety
        for servo_id, startup_tick, goal_tick in zip(commanded_ids, startup_ticks, goal_ticks):
            delta = abs(int(goal_tick) - int(startup_tick))
            if delta > int(self.config.max_tick_delta_from_startup):
                result["success"] = False
                result["command_message"] = (
                    f"servo {servo_id} delta {delta} exceeds max_tick_delta_from_startup {self.config.max_tick_delta_from_startup}"
                )
                return result
        try:
            writer = getattr(session.context.servo_service, "_write_goal_positions", None)
            if not callable(writer):
                raise RuntimeError("ServoService all-8 raw goal writer is unavailable.")
            writer(goals_by_servo)
            return result
        except Exception as exc:
            result["success"] = False
            result["command_message"] = str(exc)
            return result

    def _capture_sample(
        self,
        *,
        session: ExperimentSession,
        command: TwoSegmentCommand,
        command_result: dict[str, Any],
        target: TwoSegmentRepeatabilityTarget,
        visit_index: int,
        sample_index: int,
        run_trust_mode: str,
    ) -> tuple[ExperimentTimeseriesSample, dict[str, Any]]:
        context = session.context.settings.robot.operating_context()
        commanded_ids = [int(value) for value in context.commanded_servo_ids]
        telemetry = _read_live_telemetry(session=session, servo_ids=commanded_ids, dry_run=False)
        servo_feedback = _servo_feedback_payload(
            session=session,
            telemetry=telemetry,
            servo_ids=commanded_ids,
            startup_ticks_by_servo=self._startup_ticks_by_servo,
        )
        pose_observation, pose_fields = _two_segment_pose_observation(
            session=session,
            config=self.config,
            run_trust_mode=run_trust_mode,
        )
        command_success = bool(command_result.get("success", True))
        # Apply the same live measured-position completeness gate as collect-pose:
        # if any of the 8 commanded servos is missing a position, the sample is
        # rejected in trusted runs and recorded-but-kept in servo-only runs.
        missing_measured_servo_ids = sorted(
            int(servo_id)
            for servo_id in commanded_ids
            if dict(servo_feedback.get(str(int(servo_id)), {}) or {}).get("position_tick") is None
        )
        servo_only_run = bool(self.config.allow_servo_only_test_run or str(self.config.run_trust_mode).strip().lower() == "servo_only")
        telemetry_complete_failure = bool(missing_measured_servo_ids) and not servo_only_run
        capture_accepted = bool(command_success) and not telemetry_complete_failure
        if not command_success:
            rejection_reason = str(command_result.get("command_message") or "command_failed")
        elif telemetry_complete_failure:
            rejection_reason = f"measured_servo_position_missing:{','.join(str(x) for x in missing_measured_servo_ids)}"
        else:
            rejection_reason = None
        flags = ["capture_accepted" if capture_accepted else "capture_rejected"]
        if missing_measured_servo_ids:
            flags.append("measured_servo_position_missing")
        pose_summary = {
            "distal_translation_mm": dict(pose_observation.get("distal_tip_pose", {})).get("translation_mm"),
            "intermediate_translation_mm": dict(pose_observation.get("intermediate_pose", {})).get("translation_mm"),
        }
        sample = ExperimentTimeseriesSample(
            monotonic_time_s=session.elapsed_s(),
            wall_time_utc=datetime.now(timezone.utc).isoformat(),
            phase="repeat_visit",
            step_index=int(target.target_index),
            sample_index=int(sample_index),
            cycle_index=int(visit_index),
            commanded_motor_values=dict(command_result.get("goal_ticks_by_servo", {}) or {}),
            commanded_cable_deltas_cm=list(command.to_flat(context=context)),
            two_segment_command=command.to_record(context=context),
            tracker_frame_id=pose_fields.get("tracker_frame_id"),
            tool_ids_seen=list(pose_fields.get("tool_ids_seen", []) or []),
            transform_validity=dict(pose_fields.get("transform_validity", {}) or {}),
            pose_in_tracker_frame=dict(pose_fields.get("pose_in_tracker_frame", {}) or {}),
            pose_in_robot_frame=dict(pose_fields.get("pose_in_robot_frame", {}) or {}),
            two_segment_pose=pose_observation,
            freshness_s=pose_fields.get("freshness_s"),
            status_flags=sorted(set(flags)),
            backend_health=dict(pose_fields.get("backend_health", {}) or {}),
            extra={
                "record_kind": "two_segment_repeatability_visit",
                "target": target.to_dict(),
                "capture_accepted": bool(capture_accepted),
                "capture_rejection_reason": rejection_reason,
                "command_success": bool(command_success),
                "measured_servo_feedback": servo_feedback,
                "missing_measured_servo_ids": list(missing_measured_servo_ids),
                "visit_index": int(visit_index),
                "run_trust_mode": str(run_trust_mode),
                "pose_summary": pose_summary,
                "physical_assembly": dict(context.metadata().get("physical_assembly") or {}),
            },
        )
        return sample, pose_summary

    def summarize(self, session: ExperimentSession) -> dict[str, Any]:
        return dict(session.metrics)

    def write_outputs(self, session: ExperimentSession, paths, summary) -> None:
        write_two_segment_repeatability_outputs(
            output_dir=paths.output_dir,
            metrics=dict(summary.experiment_metrics or {}),
        )


def _two_segment_registration_summary(*, session) -> dict[str, Any]:
    """Capture registration provenance from the tracking snapshot if available.

    Surfaces the registration file path, FRE, timestamp, and state so two
    repeatability runs can be compared honestly. If no tracker is connected
    or no registration was loaded, returns placeholders with the state field
    set so the operator-visible warning fires.
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


def _compute_repeatability_metrics(target_records: list[dict[str, Any]]) -> dict[str, Any]:
    by_target: dict[int, list[dict[str, Any]]] = {}
    for record in target_records:
        if not bool(record.get("capture_accepted")):
            continue
        target_idx = int(record["target"]["target_index"])
        by_target.setdefault(target_idx, []).append(record)
    per_target: list[dict[str, Any]] = []
    distal_squared_errors: list[float] = []
    intermediate_squared_errors: list[float] = []
    for idx, records in sorted(by_target.items()):
        distal_points = [record["pose_summary"].get("distal_translation_mm") for record in records]
        distal_points = [list(point) for point in distal_points if isinstance(point, list) and len(point) >= 3]
        intermediate_points = [record["pose_summary"].get("intermediate_translation_mm") for record in records]
        intermediate_points = [list(point) for point in intermediate_points if isinstance(point, list) and len(point) >= 3]
        target_entry: dict[str, Any] = {
            "target_index": int(idx),
            "label": str(records[0]["target"]["label"]),
            "group_tag": str(records[0]["target"]["group_tag"]),
            "visits": len(records),
            "distal_scatter_mm": _scatter_metrics(distal_points),
            "intermediate_scatter_mm": _scatter_metrics(intermediate_points),
        }
        per_target.append(target_entry)
        for pose in distal_points:
            mean = np.mean(np.asarray(distal_points), axis=0)
            distal_squared_errors.append(float(np.linalg.norm(np.asarray(pose) - mean) ** 2))
        for pose in intermediate_points:
            mean = np.mean(np.asarray(intermediate_points), axis=0)
            intermediate_squared_errors.append(float(np.linalg.norm(np.asarray(pose) - mean) ** 2))
    return {
        "per_target": per_target,
        "aggregate_distal_rms_mm": float(np.sqrt(np.mean(distal_squared_errors))) if distal_squared_errors else None,
        "aggregate_intermediate_rms_mm": float(np.sqrt(np.mean(intermediate_squared_errors))) if intermediate_squared_errors else None,
        "target_count_with_distal": sum(1 for entry in per_target if entry["distal_scatter_mm"]["visits"] > 0),
        "target_count_with_intermediate": sum(1 for entry in per_target if entry["intermediate_scatter_mm"]["visits"] > 0),
    }


def _scatter_metrics(points: list[list[float]]) -> dict[str, Any]:
    if not points:
        return {"visits": 0, "rms_mm": None, "max_mm": None, "mean_xyz_mm": None}
    arr = np.asarray(points, dtype=float)
    mean = np.mean(arr, axis=0)
    deltas = arr - mean
    distances = np.linalg.norm(deltas, axis=1)
    return {
        "visits": int(arr.shape[0]),
        "rms_mm": float(np.sqrt(np.mean(distances ** 2))),
        "max_mm": float(np.max(distances)),
        "mean_xyz_mm": [float(value) for value in mean.tolist()],
    }


def write_two_segment_repeatability_outputs(*, output_dir: Path, metrics: dict[str, Any]) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_text": output_dir / "two_segment_repeatability_summary.txt",
        "scatter_metrics_json": output_dir / "two_segment_repeatability_scatter_metrics.json",
        "per_target_csv": output_dir / "two_segment_repeatability_per_target.csv",
        "distal_scatter_figure": output_dir / "two_segment_repeatability_distal_scatter.png",
        "rms_bar_figure": output_dir / "two_segment_repeatability_per_target_rms.png",
    }
    scatter = dict(metrics.get("scatter_metrics") or {})
    paths["scatter_metrics_json"].write_text(json.dumps(scatter, indent=2), encoding="utf-8")
    _write_repeatability_summary(paths["summary_text"], metrics, scatter)
    _write_per_target_csv(paths["per_target_csv"], scatter)
    _write_distal_scatter_figure(paths["distal_scatter_figure"], metrics, scatter)
    _write_per_target_rms_figure(paths["rms_bar_figure"], scatter)
    return paths


def _write_repeatability_summary(path: Path, metrics: dict[str, Any], scatter: dict[str, Any]) -> None:
    target_distal = metrics.get("target_distal_rms_mm")
    target_intermediate = metrics.get("target_intermediate_rms_mm")
    distal = scatter.get("aggregate_distal_rms_mm")
    intermediate = scatter.get("aggregate_intermediate_rms_mm")
    registration = dict(metrics.get("registration_summary") or {})
    lines = [
        "Two-Segment Repeatability",
        f"schema_version: {metrics.get('schema_version', REPEATABILITY_SCHEMA_VERSION)}",
        f"target_set: {metrics.get('target_set', '')}",
        f"target_count: {metrics.get('target_count', 0)}",
        f"repeat_visits: {metrics.get('repeat_visits', 0)}",
        f"accepted_sample_count: {metrics.get('accepted_sample_count', 0)}",
        f"rejected_sample_count: {metrics.get('rejected_sample_count', 0)}",
        f"bottom_segment_key: {metrics.get('bottom_segment_key', '')}",
        f"top_segment_key: {metrics.get('top_segment_key', '')}",
        "",
        f"aggregate_distal_rms_mm: {distal if distal is not None else 'n/a'}",
        f"aggregate_intermediate_rms_mm: {intermediate if intermediate is not None else 'n/a'}",
        f"operator_target_distal_rms_mm: {target_distal if target_distal is not None else 'not set'}",
        f"operator_target_intermediate_rms_mm: {target_intermediate if target_intermediate is not None else 'not set'}",
    ]
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
        "This is an open-loop repeatability scaffold. It does not validate two-segment closed-loop control.",
        "valid_for_thesis_repeatability: false",
        "automatic_two_segment_pretension_validated: false",
        "two_segment_control_validated: false",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_per_target_csv(path: Path, scatter: dict[str, Any]) -> None:
    fields = [
        "target_index",
        "label",
        "group_tag",
        "visits",
        "distal_rms_mm",
        "distal_max_mm",
        "intermediate_rms_mm",
        "intermediate_max_mm",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in list(scatter.get("per_target") or []):
            distal = dict(entry.get("distal_scatter_mm") or {})
            intermediate = dict(entry.get("intermediate_scatter_mm") or {})
            writer.writerow({
                "target_index": entry.get("target_index"),
                "label": entry.get("label"),
                "group_tag": entry.get("group_tag"),
                "visits": entry.get("visits"),
                "distal_rms_mm": distal.get("rms_mm"),
                "distal_max_mm": distal.get("max_mm"),
                "intermediate_rms_mm": intermediate.get("rms_mm"),
                "intermediate_max_mm": intermediate.get("max_mm"),
            })


def _write_distal_scatter_figure(path: Path, metrics: dict[str, Any], scatter: dict[str, Any]) -> None:
    per_target = list(scatter.get("per_target") or [])
    fig, ax = create_figure(size="wide")
    if not per_target:
        ax.text(0.5, 0.5, "No accepted repeatability samples.", ha="center", va="center", transform=ax.transAxes)
        style_axes(ax, title="Two-Segment Distal Repeatability Scatter", xlabel="X (mm)", ylabel="Y (mm)")
        save_figure(fig, path)
        return
    for entry in per_target:
        distal = dict(entry.get("distal_scatter_mm") or {})
        mean = distal.get("mean_xyz_mm")
        rms = distal.get("rms_mm")
        if not isinstance(mean, list) or len(mean) < 2:
            continue
        ax.scatter([mean[0]], [mean[1]], s=30, color=color("measured"))
        if rms is not None:
            circle = np.linspace(0.0, 2.0 * np.pi, 64)
            ax.plot(
                mean[0] + float(rms) * np.cos(circle),
                mean[1] + float(rms) * np.sin(circle),
                color=color("threshold"),
                linewidth=0.7,
            )
    style_axes(ax, title="Two-Segment Distal Repeatability Scatter", xlabel="X (mm)", ylabel="Y (mm)")
    ax.set_aspect("equal", adjustable="datalim")
    save_figure(fig, path)


def _write_per_target_rms_figure(path: Path, scatter: dict[str, Any]) -> None:
    per_target = list(scatter.get("per_target") or [])
    fig, ax = create_figure(size="wide")
    if not per_target:
        ax.text(0.5, 0.5, "No accepted repeatability samples.", ha="center", va="center", transform=ax.transAxes)
        style_axes(ax, title="Per-Target Repeatability RMS", xlabel="Target", ylabel="RMS (mm)")
        save_figure(fig, path)
        return
    labels = [str(entry.get("label", "")) for entry in per_target]
    rms_values = [float(dict(entry.get("distal_scatter_mm") or {}).get("rms_mm") or 0.0) for entry in per_target]
    ax.bar(range(len(labels)), rms_values, color=color("measured"))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    style_axes(ax, title="Per-Target Distal Repeatability RMS", xlabel="Target", ylabel="RMS (mm)")
    save_figure(fig, path)
