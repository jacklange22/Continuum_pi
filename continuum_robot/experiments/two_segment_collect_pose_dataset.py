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
SCHEDULE_TYPES = {"zero", "single_axis_micro", "segment_isolation", "small_combined"}


@dataclass
class TwoSegmentCollectPoseDatasetConfig:
    """Configuration for the bounded two-segment dataset MVP."""

    schedule_type: str = "single_axis_micro"
    dry_run: bool = True
    max_segment_displacement_mm: float = 0.1
    max_tick_delta_from_startup: int = 20
    samples_per_pattern: int = 1
    capture_repeats: int = 1
    settle_time_s: float = 0.0
    allow_servo_only_test_run: bool = True
    run_trust_mode: str = "servo_only"
    capture_tracker_snapshot: bool = True
    requested_tool_roles: dict[str, str] = field(default_factory=dict)
    run_label: str = ""
    dataset_tag: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "TwoSegmentCollectPoseDatasetConfig":
        payload = dict(payload or {})
        requested_roles = payload.get("requested_tool_roles") or payload.get("tool_roles") or {}
        return cls(
            schedule_type=str(payload.get("schedule_type", "single_axis_micro") or "single_axis_micro").strip().lower(),
            dry_run=bool(payload.get("dry_run", True)),
            max_segment_displacement_mm=max(0.0, float(payload.get("max_segment_displacement_mm", 0.1))),
            max_tick_delta_from_startup=max(0, int(payload.get("max_tick_delta_from_startup", 20))),
            samples_per_pattern=max(1, int(payload.get("samples_per_pattern", payload.get("samples_per_command", 1)))),
            capture_repeats=max(1, int(payload.get("capture_repeats", 1))),
            settle_time_s=max(0.0, float(payload.get("settle_time_s", 0.0))),
            allow_servo_only_test_run=bool(payload.get("allow_servo_only_test_run", True)),
            run_trust_mode=str(payload.get("run_trust_mode", "servo_only") or "servo_only").strip().lower(),
            capture_tracker_snapshot=bool(payload.get("capture_tracker_snapshot", True)),
            requested_tool_roles={str(key).upper(): str(value) for key, value in dict(requested_roles or {}).items()},
            run_label=str(payload.get("run_label", "") or ""),
            dataset_tag=str(payload.get("dataset_tag", "") or ""),
        )


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
        if trusted_requested and not bool(self._startup_provenance.get("accepted_all_8_startup")):
            raise RuntimeError(
                "Trusted two-segment dataset collection requires an accepted all-8 manual startup artifact. "
                "Use two_segment_startup_validation first or run with allow_servo_only_test_run=true and run_trust_mode=servo_only."
            )
        violations = _command_limit_violations(
            steps=self._command_steps,
            context=context,
            startup_ticks_by_servo=self._startup_ticks_by_servo,
            max_tick_delta=int(self.config.max_tick_delta_from_startup),
            mapper=session.context.servo_service.mapper,
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
        total = len(self._command_steps) * max(1, int(self.config.capture_repeats)) * max(1, int(self.config.samples_per_pattern))
        accepted = 0
        rejected = 0
        command_failures = 0
        progress = 0
        run_trust_mode = _run_trust_mode(self.config)
        role_configs = _effective_role_configs(session=session, config=self.config)
        tool_roles = _tool_roles_from_configs(role_configs)
        foundation = build_two_segment_foundation_metadata(context, tool_roles=tool_roles)
        startup_provenance = dict(self._startup_provenance)
        startup_provenance.setdefault("startup_ticks_by_servo", {str(key): int(value) for key, value in sorted(self._startup_ticks_by_servo.items())})
        for repeat_index in range(max(1, int(self.config.capture_repeats))):
            for step in self._command_steps:
                session.raise_if_stop_requested()
                command_result = self._issue_command(session=session, command=step.command)
                if not bool(command_result.get("success", False)):
                    command_failures += 1
                if float(self.config.settle_time_s) > 0.0:
                    session.context.sleep_fn(float(self.config.settle_time_s))
                for sample_index in range(max(1, int(self.config.samples_per_pattern))):
                    sample = self._capture_sample(
                        session=session,
                        step=step,
                        command_result=command_result,
                        repeat_index=repeat_index,
                        sample_index=sample_index,
                        run_trust_mode=run_trust_mode,
                        foundation=foundation,
                        startup_provenance=startup_provenance,
                    )
                    session.add_sample(sample)
                    accepted += int(bool(sample.extra.get("capture_accepted")))
                    rejected += int(not bool(sample.extra.get("capture_accepted")))
                    progress += 1
                    session.update_progress(
                        progress,
                        total,
                        {
                            "phase": step.phase,
                            "step_index": int(step.index),
                            "accepted_samples": int(accepted),
                            "rejected_samples": int(rejected),
                        },
                    )
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
        session.set_metric("valid_for_two_segment_model_training", _valid_for_two_segment_model_training(self.config, pose_summary, startup_provenance, command_failures))
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
        session.set_metric("automatic_two_segment_pretension_validated", False)
        session.set_metric("two_segment_control_validated", False)
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
        flat_cm = command.to_flat(context=context)
        commanded_ids = [int(value) for value in context.commanded_servo_ids]
        startup_ticks = [int(self._startup_ticks_by_servo[servo_id]) for servo_id in commanded_ids]
        goal_ticks = session.context.servo_service.mapper.to_goal_positions(flat_cm, startup_ticks)
        goals_by_servo = {int(servo_id): int(goal) for servo_id, goal in zip(commanded_ids, goal_ticks)}
        result = {
            "success": True,
            "requested_flat_command_cm": list(flat_cm),
            "ordered_8_displacements_cm": list(flat_cm),
            "commanded_servo_ids": list(commanded_ids),
            "startup_ticks_by_servo": {str(key): int(value) for key, value in sorted(self._startup_ticks_by_servo.items())},
            "goal_ticks_by_servo": {str(key): int(value) for key, value in sorted(goals_by_servo.items())},
            "command_message": "Dry-run command resolved relative to startup ticks.",
            "dry_run": bool(self.config.dry_run),
        }
        if self.config.dry_run:
            return result
        try:
            writer = getattr(session.context.servo_service, "_write_goal_positions", None)
            if not callable(writer):
                raise RuntimeError("ServoService all-8 raw goal writer is unavailable.")
            writer(goals_by_servo)
            result["command_message"] = "Wrote bounded all-8 raw goal ticks relative to manual startup artifact."
            return result
        except Exception as exc:
            result["success"] = False
            result["command_message"] = str(exc)
            return result

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
        capture_accepted = bool(command_success)
        rejection_reason = None if command_success else str(command_result.get("command_message") or "command_failed")
        missing_pose = not bool(pose_fields.get("pose_observations_present"))
        valid_for_two_segment_model_training = bool(
            capture_accepted
            and _valid_for_two_segment_model_training(self.config, _pose_label_summary_from_fields(pose_fields), startup_provenance, 0)
        )
        flags = ["capture_accepted" if capture_accepted else "capture_rejected"]
        if self.config.dry_run:
            flags.append("dry_run")
        if _servo_only_mode(self.config):
            flags.extend(["servo_only", "lower_trust", "not_model_training_ready"])
        if missing_pose:
            flags.append("pose_labels_missing")
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
                "startup_artifact_provenance": dict(startup_provenance),
                "measured_servo_feedback": servo_feedback,
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
                "capture_accepted": bool(capture_accepted),
                "capture_rejection_reason": rejection_reason,
                "command_success": bool(command_success),
                "command_message": str(command_result.get("command_message", "") or ""),
                "valid_for_two_segment_model_training": bool(valid_for_two_segment_model_training),
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
        )


def build_two_segment_command_schedule(config: TwoSegmentCollectPoseDatasetConfig, *, context) -> list[TwoSegmentCommandStep]:
    amp_cm = min(float(config.max_segment_displacement_mm) / 10.0, 0.01)
    if amp_cm <= 0.0 or str(config.schedule_type) == "zero":
        return [
            TwoSegmentCommandStep(
                index=0,
                label="zero_startup",
                phase="zero",
                command=TwoSegmentCommand.from_mapping({"segment_a": [0.0, 0.0, 0.0, 0.0], "segment_b": [0.0, 0.0, 0.0, 0.0]}),
            )
        ]
    patterns: list[tuple[str, str, list[float], list[float]]] = [
        ("segment_a_axis_a_pos", "single_axis_micro", [amp_cm, 0.0, -amp_cm, 0.0], [0.0, 0.0, 0.0, 0.0]),
        ("segment_a_axis_a_neg", "single_axis_micro", [-amp_cm, 0.0, amp_cm, 0.0], [0.0, 0.0, 0.0, 0.0]),
        ("segment_a_axis_b_pos", "single_axis_micro", [0.0, amp_cm, 0.0, -amp_cm], [0.0, 0.0, 0.0, 0.0]),
        ("segment_a_axis_b_neg", "single_axis_micro", [0.0, -amp_cm, 0.0, amp_cm], [0.0, 0.0, 0.0, 0.0]),
        ("segment_b_axis_a_pos", "single_axis_micro", [0.0, 0.0, 0.0, 0.0], [amp_cm, 0.0, -amp_cm, 0.0]),
        ("segment_b_axis_a_neg", "single_axis_micro", [0.0, 0.0, 0.0, 0.0], [-amp_cm, 0.0, amp_cm, 0.0]),
        ("segment_b_axis_b_pos", "single_axis_micro", [0.0, 0.0, 0.0, 0.0], [0.0, amp_cm, 0.0, -amp_cm]),
        ("segment_b_axis_b_neg", "single_axis_micro", [0.0, 0.0, 0.0, 0.0], [0.0, -amp_cm, 0.0, amp_cm]),
    ]
    schedule_type = str(config.schedule_type)
    if schedule_type == "segment_isolation":
        patterns = [item for item in patterns if item[0].startswith("segment_a")][:2] + [item for item in patterns if item[0].startswith("segment_b")][:2]
    elif schedule_type == "small_combined":
        half = amp_cm / 2.0
        patterns.extend(
            [
                ("combined_a_axis_a_b_axis_a_pos", "small_combined", [half, 0.0, -half, 0.0], [half, 0.0, -half, 0.0]),
                ("combined_a_axis_b_b_axis_b_pos", "small_combined", [0.0, half, 0.0, -half], [0.0, half, 0.0, -half]),
                ("combined_cross_axes", "small_combined", [half, 0.0, -half, 0.0], [0.0, -half, 0.0, half]),
            ]
        )
    steps: list[TwoSegmentCommandStep] = [
        TwoSegmentCommandStep(
            index=0,
            label="zero_startup",
            phase="zero",
            command=TwoSegmentCommand.from_mapping({"segment_a": [0.0, 0.0, 0.0, 0.0], "segment_b": [0.0, 0.0, 0.0, 0.0]}),
        )
    ]
    max_patterns = max(1, int(config.capture_repeats)) * 9999
    for index, (label, phase, segment_a, segment_b) in enumerate(patterns[:max_patterns], start=1):
        steps.append(
            TwoSegmentCommandStep(
                index=index,
                label=label,
                phase=phase,
                command=TwoSegmentCommand.from_mapping({"segment_a": segment_a, "segment_b": segment_b}),
            )
        )
    _ = context
    return steps


def write_two_segment_dataset_outputs(*, output_dir: Path, metrics: dict[str, Any], samples: list[ExperimentTimeseriesSample]) -> dict[str, Path]:
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
    }
    _write_summary_text(paths["summary_text"], metrics)
    paths["role_provenance"].write_text(
        json.dumps(
            {
                "two_segment_tracking_role_config": metrics.get("two_segment_tracking_role_config", {}),
                "pose_label_summary": metrics.get("pose_label_summary", {}),
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


def _command_limit_violations(*, steps: list[TwoSegmentCommandStep], context, startup_ticks_by_servo: dict[int, int], max_tick_delta: int, mapper) -> list[str]:
    violations: list[str] = []
    commanded_ids = [int(value) for value in context.commanded_servo_ids]
    startup_ticks = [int(startup_ticks_by_servo[int(servo_id)]) for servo_id in commanded_ids]
    for step in steps:
        goals = mapper.to_goal_positions(step.command.to_flat(context=context), startup_ticks)
        for servo_id, startup, goal in zip(commanded_ids, startup_ticks, goals):
            delta = abs(int(goal) - int(startup))
            if delta > int(max_tick_delta):
                violations.append(f"{step.label}: servo {servo_id} delta {delta} > {int(max_tick_delta)}")
    return violations


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


def _pose_label_summary(samples: list[ExperimentTimeseriesSample]) -> dict[str, Any]:
    distal = 0
    intermediate = 0
    observed = 0
    missing_required: set[str] = set()
    available_roles: set[str] = set()
    for sample in samples:
        pose = dict(sample.two_segment_pose or {})
        if dict(pose.get("distal_tip_pose", {}) or {}):
            distal += 1
        if dict(pose.get("intermediate_pose", {}) or {}):
            intermediate += 1
        if bool(sample.extra.get("pose_observations_present")):
            observed += 1
        missing_required.update(str(value) for value in list(sample.extra.get("missing_required_pose_roles", []) or []))
        available_roles.update(str(value) for value in list(sample.extra.get("available_pose_roles", []) or []))
    return {
        "pose_observation_sample_count": int(observed),
        "distal_pose_sample_count": int(distal),
        "intermediate_pose_sample_count": int(intermediate),
        "includes_intermediate_pose": bool(intermediate),
        "distal_only": bool(distal and not intermediate),
        "available_roles": sorted(available_roles),
        "missing_required_roles": sorted(missing_required),
    }


def _pose_label_summary_from_fields(fields: dict[str, Any]) -> dict[str, Any]:
    available_roles = set(str(value) for value in list(fields.get("available_roles", []) or []))
    return {
        "pose_observation_sample_count": int(bool(fields.get("pose_observations_present"))),
        "distal_pose_sample_count": int("distal_tip" in available_roles),
        "intermediate_pose_sample_count": int("intermediate_segment" in available_roles),
        "missing_required_roles": list(fields.get("missing_required_roles", []) or []),
    }


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
        and not list(pose_summary.get("missing_required_roles", []) or [])
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
    if list(pose_summary.get("missing_required_roles", []) or []):
        warnings.append("required_pose_roles_missing")
    if not bool(pose_summary.get("includes_intermediate_pose")):
        warnings.append("intermediate_pose_missing")
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
    lines = [
        "Two-Segment Collect-Pose Command Dataset",
        f"dataset_schema_version: {metrics.get('dataset_schema_version', TWO_SEGMENT_DATASET_SCHEMA_VERSION)}",
        f"schedule_type: {metrics.get('schedule_type')}",
        f"run_trust_mode: {metrics.get('run_trust_mode')}",
        f"valid_for_two_segment_model_training: {metrics.get('valid_for_two_segment_model_training')}",
        f"valid_for_model_training: {metrics.get('valid_for_model_training')}",
        f"accepted_sample_count: {metrics.get('accepted_sample_count')}",
        f"rejected_sample_count: {metrics.get('rejected_sample_count')}",
        f"pose_observation_sample_count: {pose.get('pose_observation_sample_count', 0)}",
        f"distal_only: {metrics.get('distal_only')}",
        f"includes_intermediate_pose: {metrics.get('includes_intermediate_pose')}",
        "automatic_two_segment_pretension_validated: false",
        "two_segment_control_validated: false",
        "",
        "This dataset stores structured Segment A/B commands and optional pose observations.",
        "It does not implement two-segment kinematics, control, modeling, chasing, or automatic pretension.",
    ]
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
