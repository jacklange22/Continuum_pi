"""Canonical live single-segment repeatability experiment.

This module recreates the legacy thesis repeatability protocol while using the
current tracker, servo, registration, runtime-tip, and dataset infrastructure.
The legacy reference used direct Aurora/Arduino access; this implementation
does not.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import random
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from continuum_robot.experiments.framework import BaseExperiment, ExperimentHardwareRequirements, ExperimentSession
from continuum_robot.experiments.sample_builders import sample_from_tracking_snapshot
from continuum_robot.experiments.schemas import ExperimentTimeseriesSample
from continuum_robot.experiments.validation import (
    STATUS_INVALID_INSUFFICIENT_SAMPLES,
    STATUS_INVALID_REPEATABILITY_COVERAGE,
    STATUS_PARTIAL_SUCCESS,
    STATUS_SUCCESS,
)
from continuum_robot.servos.servo_service import ServoBusBusyError
from continuum_robot.tracking.runtime_tip_policy import evaluate_runtime_tip_trust


LEGACY_TARGET_COUNT = 17
LEGACY_APPROACHES_PER_TARGET = 16
LEGACY_VISIT_COUNT = LEGACY_TARGET_COUNT * LEGACY_APPROACHES_PER_TARGET
LEGACY_CAPTURE_COUNT = LEGACY_VISIT_COUNT * 2
DEFAULT_INNER_RING_RADIUS_MM = 4.0
DEFAULT_OUTER_RING_RADIUS_MM = 8.0
DEFAULT_MAX_TARGET_TICK_DELTA_FROM_STARTUP = 650
RUNTIME_TIP_POLICY_WORKFLOW = "single_segment_repeatability"


LOG = logging.getLogger(__name__)


_REPEATABILITY_METRICS_CACHE_LOCK = threading.Lock()
_REPEATABILITY_METRICS_CACHE: dict[tuple[str, int, int], dict[str, Any]] = {}


@dataclass(frozen=True)
class LegacyRepeatabilityTarget:
    """One legacy single-segment target."""

    target_index: int
    label: str
    ring: str
    ring_radius_mm: float
    angle_rad: float
    angle_deg: float
    cable_deltas_mm: list[float]
    cable_deltas_cm: list[float]
    group_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LegacyRepeatabilityVisit:
    """One legacy revisit pair: approach target, then desired target."""

    sequence_index: int
    target_index: int
    approach_index: int
    revisit_index: int
    approach_target: LegacyRepeatabilityTarget
    repeat_target: LegacyRepeatabilityTarget

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["approach_target"] = self.approach_target.to_dict()
        payload["repeat_target"] = self.repeat_target.to_dict()
        return payload


@dataclass
class SingleSegmentRepeatabilityConfig:
    """Config for the live legacy 17-point single-segment repeatability run."""

    tool_id: str = "0A"
    settle_time_s: float = 4.0
    capture_timeout_s: float = 1.0
    capture_poll_interval_s: float = 0.02
    max_tracker_age_s: float = 0.25
    random_seed: int = 0
    run_label: str = ""
    require_robot_frame_tip: bool = True
    allow_debug_coil_as_tip: bool = False
    return_to_center_on_finalize: bool = True
    fail_on_rejected_capture: bool = False
    thesis_goal_rms_mm: float = 1.0
    min_repeat_captures_per_target: int = 12
    min_repeat_capture_fraction: float = 0.85
    max_rejected_capture_fraction: float = 0.10
    baseline_run_path: str = ""
    inner_ring_radius_mm: float = DEFAULT_INNER_RING_RADIUS_MM
    outer_ring_radius_mm: float = DEFAULT_OUTER_RING_RADIUS_MM
    max_target_tick_delta_from_startup: int = DEFAULT_MAX_TARGET_TICK_DELTA_FROM_STARTUP

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SingleSegmentRepeatabilityConfig":
        payload = dict(payload or {})
        return cls(
            tool_id=str(payload.get("tool_id", "0A") or "0A").strip().upper(),
            settle_time_s=max(0.0, float(payload.get("settle_time_s", 4.0))),
            capture_timeout_s=max(0.0, float(payload.get("capture_timeout_s", 1.0))),
            capture_poll_interval_s=max(0.001, float(payload.get("capture_poll_interval_s", 0.02))),
            max_tracker_age_s=max(0.0, float(payload.get("max_tracker_age_s", 0.25))),
            random_seed=int(payload.get("random_seed", 0)),
            run_label=str(payload.get("run_label", "") or ""),
            require_robot_frame_tip=bool(payload.get("require_robot_frame_tip", True)),
            allow_debug_coil_as_tip=bool(payload.get("allow_debug_coil_as_tip", False)),
            return_to_center_on_finalize=bool(payload.get("return_to_center_on_finalize", True)),
            fail_on_rejected_capture=bool(payload.get("fail_on_rejected_capture", False)),
            thesis_goal_rms_mm=float(payload.get("thesis_goal_rms_mm", 1.0)),
            min_repeat_captures_per_target=max(1, int(payload.get("min_repeat_captures_per_target", 12))),
            min_repeat_capture_fraction=max(0.0, min(1.0, float(payload.get("min_repeat_capture_fraction", 0.85)))),
            max_rejected_capture_fraction=max(0.0, min(1.0, float(payload.get("max_rejected_capture_fraction", 0.10)))),
            baseline_run_path=str(payload.get("baseline_run_path", "") or "").strip(),
            inner_ring_radius_mm=max(
                0.0,
                float(payload.get("inner_ring_radius_mm", DEFAULT_INNER_RING_RADIUS_MM)),
            ),
            outer_ring_radius_mm=max(
                0.0,
                float(payload.get("outer_ring_radius_mm", DEFAULT_OUTER_RING_RADIUS_MM)),
            ),
            max_target_tick_delta_from_startup=max(
                1,
                int(
                    payload.get(
                        "max_target_tick_delta_from_startup",
                        DEFAULT_MAX_TARGET_TICK_DELTA_FROM_STARTUP,
                    )
                ),
            ),
        )


class SingleSegmentRepeatabilityExperiment(BaseExperiment):
    """Live repeatability experiment preserving the old 17-target revisit protocol."""

    name = "single_segment_repeatability"
    description = (
        "Faithful live recreation of the legacy single-segment 17-target repeatability protocol: "
        "center, inner ring, outer ring, and every-target revisit from all other targets."
    )
    hardware_requirements = ExperimentHardwareRequirements(
        tracking_required=True,
        servo_required=True,
        registration_required=True,
        mock_compatible=False,
    )

    def __init__(self, config: SingleSegmentRepeatabilityConfig) -> None:
        super().__init__(config)
        self.config: SingleSegmentRepeatabilityConfig
        self._targets = build_legacy_17_point_targets(
            inner_ring_radius_mm=float(self.config.inner_ring_radius_mm),
            outer_ring_radius_mm=float(self.config.outer_ring_radius_mm),
        )
        self._visits = generate_legacy_revisit_sequence(self._targets, seed=self.config.random_seed)
        self._neutral_ticks: list[int] = []
        self._servo_ids: list[int] = []

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "SingleSegmentRepeatabilityExperiment":
        return cls(SingleSegmentRepeatabilityConfig.from_dict(payload))

    def setup(self, session: ExperimentSession) -> None:
        self._servo_ids = _configured_single_segment_servo_ids(session)
        self._neutral_ticks = _load_neutral_ticks(session, self._servo_ids)
        tick_profile = repeatability_target_tick_profile(
            targets=self._targets,
            mapper=session.context.servo_service.mapper,
        )
        session.set_metric(
            "target_catalog",
            [
                {
                    **target.to_dict(),
                    "cable_deltas_ticks": list(
                        tick_profile["per_target_cable_deltas_ticks"].get(int(target.target_index), [])
                    ),
                }
                for target in self._targets
            ],
        )
        ring_ticks = repeatability_ring_tick_defaults(
            mapper=session.context.servo_service.mapper,
            inner_ring_radius_mm=float(self.config.inner_ring_radius_mm),
            outer_ring_radius_mm=float(self.config.outer_ring_radius_mm),
        )
        session.set_metric(
            "target_geometry",
            {
                "inner_ring_radius_mm": float(self.config.inner_ring_radius_mm),
                "outer_ring_radius_mm": float(self.config.outer_ring_radius_mm),
                "inner_ring_radius_ticks": int(ring_ticks["inner_ring_radius_ticks"]),
                "outer_ring_radius_ticks": int(ring_ticks["outer_ring_radius_ticks"]),
                "max_target_abs_tick_delta_from_startup": int(tick_profile["max_abs_tick_delta"]),
                "max_target_tick_delta_from_startup_cap": int(
                    self.config.max_target_tick_delta_from_startup
                ),
                "spool_diameter_mm": float(session.context.settings.robot.spool_diameter_cm) * 10.0,
            },
        )
        session.set_metric("planned_visit_count", LEGACY_VISIT_COUNT)
        session.set_metric("planned_capture_count", LEGACY_CAPTURE_COUNT)
        session.set_metric("protocol", "legacy_17_target_single_segment_all_other_approaches")
        session.set_metric("run_label", str(self.config.run_label or ""))
        session.set_metric(
            "active_segment",
            {
                "key": session.context.settings.robot.active_segment_key(),
                "label": session.context.settings.robot.active_segment_label(),
                "servo_ids": session.context.settings.robot.active_segment_servo_ids(),
                "pairs": session.context.settings.robot.active_segment_pairs(),
                "robot_mode": session.context.settings.robot.mode,
            },
        )
        session.set_metric("operating_context", session.context.settings.robot.operating_context().metadata())

    def precheck(self, session: ExperimentSession) -> None:
        _precheck_single_segment_repeatability(
            session=session,
            config=self.config,
            servo_ids=self._servo_ids,
            neutral_ticks=self._neutral_ticks,
        )
        _record_repeatability_run_provenance(
            session=session,
            config=self.config,
            servo_ids=self._servo_ids,
            neutral_ticks=self._neutral_ticks,
        )

    def execute(self, session: ExperimentSession) -> None:
        total_captures = LEGACY_CAPTURE_COUNT
        sample_index = 0
        rejected_count = 0
        with session.context.servo_service.exclusive_bus_operation(
            owner=self.name,
            reason="single-segment repeatability",
        ):
            for visit in self._visits:
                session.raise_if_stop_requested()
                approach_payload = self._command_target(session, visit.approach_target)
                session.context.sleep_fn(float(self.config.settle_time_s))
                approach_sample = self._capture_after_move(
                    session,
                    visit=visit,
                    target=visit.approach_target,
                    phase="approach",
                    sample_index=sample_index,
                    command_payload=approach_payload,
                )
                sample_index += 1
                if not bool(approach_sample.extra.get("capture_accepted", False)):
                    rejected_count += 1
                session.add_sample(approach_sample)
                session.update_progress(
                    sample_index,
                    total_captures,
                    _progress_payload(visit=visit, phase="approach", sample_index=sample_index),
                )
                self._raise_if_rejected_and_fatal(approach_sample)

                session.raise_if_stop_requested()
                repeat_payload = self._command_target(session, visit.repeat_target)
                session.context.sleep_fn(float(self.config.settle_time_s))
                repeat_sample = self._capture_after_move(
                    session,
                    visit=visit,
                    target=visit.repeat_target,
                    phase="repeat",
                    sample_index=sample_index,
                    command_payload=repeat_payload,
                )
                sample_index += 1
                if not bool(repeat_sample.extra.get("capture_accepted", False)):
                    rejected_count += 1
                session.add_sample(repeat_sample)
                session.update_progress(
                    sample_index,
                    total_captures,
                    _progress_payload(visit=visit, phase="repeat", sample_index=sample_index),
                )
                self._raise_if_rejected_and_fatal(repeat_sample)

        if rejected_count:
            session.add_warning(
                f"{rejected_count} capture(s) failed tracker freshness/validity gating and were saved as rejected."
            )

    def finalize(self, session: ExperimentSession) -> None:
        if not self.config.return_to_center_on_finalize:
            return
        if not getattr(session.context.servo_service, "is_connected", False):
            return
        if not self._neutral_ticks or not self._servo_ids:
            return
        try:
            self._command_target(session, self._targets[0])
        except Exception as exc:
            session.add_warning(f"Could not return robot to center target during finalize: {exc}")

    def summarize(self, session: ExperimentSession) -> dict[str, Any]:
        metrics = compute_single_segment_repeatability_metrics(
            session.samples,
            targets=self._targets,
            tool_id=self.config.tool_id,
            thesis_goal_rms_mm=float(self.config.thesis_goal_rms_mm),
            require_robot_frame_tip=bool(self.config.require_robot_frame_tip),
            min_repeat_captures_per_target=int(self.config.min_repeat_captures_per_target),
            min_repeat_capture_fraction=float(self.config.min_repeat_capture_fraction),
            max_rejected_capture_fraction=float(self.config.max_rejected_capture_fraction),
        )
        baseline_path = str(self.config.baseline_run_path or "").strip()
        if baseline_path:
            try:
                baseline_metrics = load_repeatability_metrics_from_run(
                    _resolve_repo_path(session.context.project_root, baseline_path)
                )
                metrics["baseline_comparison"] = compute_repeatability_baseline_comparison(
                    current_metrics=metrics,
                    baseline_metrics=baseline_metrics,
                    baseline_path=str(_resolve_repo_path(session.context.project_root, baseline_path)),
                )
            except Exception as exc:
                metrics["baseline_comparison"] = {
                    "available": False,
                    "baseline_path": baseline_path,
                    "error": str(exc),
                }
                session.add_warning(f"Baseline comparison failed: {exc}")
        accepted_repeat_count = int(metrics.get("valid_repeat_sample_count", 0) or 0)
        run_validity = dict(metrics.get("run_validity", {}) or {})
        provenance = dict(session.metrics.get("run_provenance", {}) or {})
        runtime_tip_trust = str(provenance.get("runtime_tip_trust_level", "unknown") or "unknown")
        runtime_tip_mode = str(provenance.get("runtime_tip_mode", "unknown") or "unknown")
        thesis_trusted_tip = bool(provenance.get("thesis_trusted_runtime_tip", False))
        metrics["runtime_tip_mode_used"] = runtime_tip_mode
        metrics["runtime_tip_trust_level"] = runtime_tip_trust
        metrics["thesis_trusted_runtime_tip"] = thesis_trusted_tip
        metrics["thesis_trusted_run"] = bool(thesis_trusted_tip)
        if provenance:
            runtime_tip = dict(provenance.get("runtime_tip_calibration", {}) or {})
            registration = dict(provenance.get("base_registration", {}) or {})
            transform_chain = dict(provenance.get("transform_chain_summary", {}) or {})
            metrics["transform_chain_audit"] = {
                "registration_artifact_path": registration.get("path"),
                "registration_artifact_sha256": registration.get("sha256"),
                "registration_fre_mm": registration.get("stored_fre_mm"),
                "runtime_tip_artifact_path": runtime_tip.get("path"),
                "runtime_tip_artifact_sha256": runtime_tip.get("sha256"),
                "runtime_tip_mode": runtime_tip.get("mode", runtime_tip_mode),
                "runtime_tip_trust_level": runtime_tip.get("trust_level", runtime_tip_trust),
                "pose_frame_used_for_analysis": (
                    dict(metrics.get("legacy_style_comparison", {}) or {}).get("pose_frame_used")
                    or metrics.get("position_frame")
                ),
                "identity_or_fallback": bool(
                    runtime_tip.get("identity_fallback")
                    or runtime_tip.get("identity_marker")
                    or transform_chain.get("identity_fallback")
                ),
                "transform_expression": transform_chain.get("expression"),
            }
        if not thesis_trusted_tip:
            run_validity["thesis_valid_run"] = False
            failure_reasons = list(run_validity.get("failure_reasons", []) or [])
            failure_reasons.append(
                "runtime tip mode/trust is lower than thesis policy "
                f"(mode={runtime_tip_mode}, trust={runtime_tip_trust})"
            )
            run_validity["failure_reasons"] = sorted(set(str(reason) for reason in failure_reasons))
            metrics["run_validity"] = run_validity
        if accepted_repeat_count <= 0:
            force_status = STATUS_INVALID_INSUFFICIENT_SAMPLES
        elif bool(run_validity.get("thesis_valid_run", False)):
            force_status = STATUS_SUCCESS
        elif not thesis_trusted_tip:
            force_status = STATUS_PARTIAL_SUCCESS
        else:
            force_status = STATUS_INVALID_REPEATABILITY_COVERAGE
        metrics["status"] = force_status
        metrics["summary_requirements"] = {
            "force_status": force_status,
            "min_sample_count": 1,
            "require_registration": True,
            "registration_available": bool(metrics.get("registration_available", False)),
            "require_tip_calibration": True,
            "tip_calibration_available": bool(metrics.get("tip_calibration_available", False)),
            "invalid_transforms_are_fatal": False,
        }
        return metrics

    def write_outputs(self, session: ExperimentSession, paths, summary) -> None:
        from continuum_robot.experiments.single_segment_repeatability_outputs import (
            write_single_segment_repeatability_outputs,
        )

        write_single_segment_repeatability_outputs(
            output_dir=paths.output_dir,
            metadata=session.metadata,
            summary=summary,
            samples=session.samples,
        )

    def _command_target(
        self,
        session: ExperimentSession,
        target: LegacyRepeatabilityTarget,
    ) -> dict[str, Any]:
        requested = [float(value) for value in target.cable_deltas_cm]
        requested_ticks = [
            int(session.context.servo_service.mapper.displacement_cm_to_ticks(float(value)))
            for value in requested
        ]
        before_telemetry = _read_servo_telemetry_payload(
            session,
            self._servo_ids,
            stage="before_command",
        )
        positions_before = {
            str(servo_id): row.get("present_position_ticks")
            for servo_id, row in dict(before_telemetry.get("by_servo", {}) or {}).items()
            if isinstance(row, dict)
        }
        max_tick_delta = max((abs(int(value)) for value in requested_ticks), default=0)
        if max_tick_delta > int(self.config.max_target_tick_delta_from_startup):
            raise RuntimeError(
                "Repeatability command blocked by experiment-safe target envelope: "
                f"target {target.label} requests {max_tick_delta} ticks but cap is "
                f"{int(self.config.max_target_tick_delta_from_startup)} ticks."
            )
        LOG.info(
            "Repeatability command dispatch | target=%s | target_index=%s | requested_cable_cm=%s | requested_ticks=%s",
            target.label,
            int(target.target_index),
            requested,
            requested_ticks,
        )
        try:
            command = session.context.servo_service.command_displacement(
                tendon_displacements_cm=list(requested),
                neutral_ticks=list(self._neutral_ticks),
                servo_ids=list(self._servo_ids),
                motion_workflow="experiment_motion",
            )
        except Exception as exc:
            telemetry_issue_code = _classify_repeatability_exception(exc)
            LOG.exception(
                "Repeatability command failed | target=%s | target_index=%s | requested_cable_cm=%s | failure_code=%s | error=%s",
                target.label,
                int(target.target_index),
                requested,
                telemetry_issue_code or "none",
                exc,
            )
            if telemetry_issue_code:
                raise RuntimeError(
                    "runtime_telemetry_invalid: "
                    f"stage=command_dispatch code={telemetry_issue_code} "
                    f"target={target.label} detail={exc}"
                ) from exc
            raise
        debug_entry = next(iter((command.debug_entries_by_id or {}).values()), None)
        LOG.info(
            "Repeatability command success | target=%s | target_index=%s | resolved_cable_cm=%s | final_goals=%s | clamp_reasons=%s",
            target.label,
            int(target.target_index),
            list(command.resolved_displacements_cm or requested),
            {str(servo_id): int(goal) for servo_id, goal in command.positions_by_id.items()},
            {str(servo_id): str(reason) for servo_id, reason in command.clamp_reasons_by_id.items()},
        )
        return {
            "commanded_motor_values": {str(servo_id): int(goal) for servo_id, goal in command.positions_by_id.items()},
            "requested_cable_ticks": list(requested_ticks),
            "servo_positions_before_command": positions_before,
            "servo_telemetry_before_command": before_telemetry,
            "motion_profile": {
                "operating_mode_label": (
                    session.context.servo_service.operating_mode_label(debug_entry.operating_mode)
                    if debug_entry is not None and debug_entry.operating_mode is not None
                    else "unknown"
                ),
                "operating_mode": debug_entry.operating_mode if debug_entry is not None else None,
                "goal_current_ma": debug_entry.goal_current_ma if debug_entry is not None else None,
                "profile_velocity": debug_entry.profile_velocity if debug_entry is not None else None,
                "profile_acceleration": debug_entry.profile_acceleration if debug_entry is not None else None,
            },
            "message": str(command.message or ""),
        }

    def _capture_after_move(
        self,
        session: ExperimentSession,
        *,
        visit: LegacyRepeatabilityVisit,
        target: LegacyRepeatabilityTarget,
        phase: str,
        sample_index: int,
        command_payload: dict[str, Any],
    ) -> ExperimentTimeseriesSample:
        snapshot, gate = _wait_for_valid_capture(
            session=session,
            tool_id=self.config.tool_id,
            max_tracker_age_s=float(self.config.max_tracker_age_s),
            timeout_s=float(self.config.capture_timeout_s),
            poll_interval_s=float(self.config.capture_poll_interval_s),
            require_robot_frame_tip=bool(self.config.require_robot_frame_tip),
            allow_debug_coil_as_tip=bool(self.config.allow_debug_coil_as_tip),
        )
        telemetry_payload = _read_servo_telemetry_payload(
            session,
            self._servo_ids,
            stage="runtime_capture",
        )
        target_ticks = [
            int(session.context.servo_service.mapper.displacement_cm_to_ticks(float(value)))
            for value in target.cable_deltas_cm
        ]
        robot_tip_payload = dict(getattr(snapshot, "T_robot_tip", None) and {"matrix": snapshot.T_robot_tip} or {})
        robot_tip_translation = _matrix_translation_mm(getattr(snapshot, "T_robot_tip", None))
        raw_tool_pose = _raw_tracker_tool_pose(snapshot, str(self.config.tool_id or "0A"))
        runtime_tip_policy = evaluate_runtime_tip_trust(
            snapshot=snapshot,
            workflow=RUNTIME_TIP_POLICY_WORKFLOW,
            allow_lower_trust=bool(self.config.allow_debug_coil_as_tip),
        )
        flags = [] if gate["accepted"] else ["capture_rejected", str(gate["reason"])]
        if getattr(snapshot, "registration_state", "") != "loaded":
            flags.append("registration_missing")
        if (
            getattr(snapshot, "T_robot_tip", None) is not None
            and getattr(snapshot, "tip_pose_status", "") in {"ok", "coil_as_tip"}
        ):
            flags.append("full_pose_available")
        if not bool(telemetry_payload.get("ok", False)):
            flags.append("runtime_telemetry_invalid")
            failure_code = str(telemetry_payload.get("failure_code", "") or "").strip()
            if failure_code:
                flags.append(f"runtime_telemetry_{failure_code}")
            LOG.warning(
                "Repeatability runtime telemetry invalid | phase=%s | sample_index=%s | code=%s | message=%s | issues=%s",
                phase,
                int(sample_index),
                failure_code or "unknown",
                telemetry_payload.get("message"),
                telemetry_payload.get("issues"),
            )
        extra = {
            "protocol": "legacy_17_target_single_segment",
            "capture_role": phase,
            "capture_accepted": bool(gate["accepted"]),
            "capture_reject_reason": None if gate["accepted"] else str(gate["reason"]),
            "tracker_gate": gate,
            "target_label": target.label,
            "target_ring": target.ring,
            "target_ring_radius_mm": float(target.ring_radius_mm),
            "target_angle_deg": float(target.angle_deg),
            "target_cable_deltas_mm": [float(value) for value in target.cable_deltas_mm],
            "target_cable_deltas_cm": [float(value) for value in target.cable_deltas_cm],
            "target_cable_deltas_ticks": list(target_ticks),
            "command_target_id": str(target.label),
            "command_target_index": int(target.target_index),
            "commanded_cable_displacement_mm": [float(value) for value in target.cable_deltas_mm],
            "commanded_cable_displacement_cm": [float(value) for value in target.cable_deltas_cm],
            "resolved_servo_goal_ticks": dict(command_payload.get("commanded_motor_values", {}) or {}),
            "current_servo_positions_before_command": dict(
                command_payload.get("servo_positions_before_command", {}) or {}
            ),
            "current_servo_positions_after_command": {
                str(servo_id): row.get("present_position_ticks")
                for servo_id, row in dict(telemetry_payload.get("by_servo", {}) or {}).items()
                if isinstance(row, dict)
            },
            "raw_tracker_tool_pose": raw_tool_pose,
            "robot_frame_tip_pose_used": {
                **robot_tip_payload,
                "translation_mm": robot_tip_translation,
                "status": getattr(snapshot, "tip_pose_status", None),
            },
            "tracker_stale_age_s": getattr(snapshot, "tracker_data_age_s", None),
            "tracker_frame_number": getattr(snapshot, "last_frame_number", None),
            "accept_reject_reason": "ok" if gate["accepted"] else str(gate["reason"]),
            "runtime_tip_mode": str(getattr(snapshot, "runtime_tip_mode", "latest_accepted") or "latest_accepted"),
            "runtime_tip_trust_level": runtime_tip_policy.trust_label,
            "runtime_tip_policy": runtime_tip_policy.to_dict(),
            "runtime_tip_identity_fallback": bool(getattr(snapshot, "runtime_tip_identity_fallback", False)),
            "source_target_index": int(visit.approach_index),
            "desired_target_index": int(visit.target_index),
            "legacy_sequence_index": int(visit.sequence_index),
            "settle_time_s": float(self.config.settle_time_s),
            "servo_telemetry": telemetry_payload,
            "runtime_telemetry_status": {
                "ok": bool(telemetry_payload.get("ok", False)),
                "failure_code": telemetry_payload.get("failure_code"),
                "failure_stage": telemetry_payload.get("failure_stage"),
                "message": telemetry_payload.get("message"),
                "issues": list(telemetry_payload.get("issues", []) or []),
            },
            "servo_motion_profile": dict(command_payload.get("motion_profile", {}) or {}),
            "servo_command_message": str(command_payload.get("message", "") or ""),
        }
        return sample_from_tracking_snapshot(
            session,
            snapshot=snapshot,
            phase=phase,
            step_index=int(sample_index),
            sample_index=int(sample_index),
            commanded_cable_deltas_cm=[float(value) for value in target.cable_deltas_cm],
            commanded_motor_values=dict(command_payload.get("commanded_motor_values", {}) or {}),
            status_flags=flags,
            extra=extra,
            target_index=int(target.target_index),
            revisit_index=int(visit.revisit_index),
            approach_index=int(visit.approach_index),
            tracker_tool_id=str(self.config.tool_id),
        )

    def _raise_if_rejected_and_fatal(self, sample: ExperimentTimeseriesSample) -> None:
        if self.config.fail_on_rejected_capture and not bool(sample.extra.get("capture_accepted", False)):
            raise RuntimeError(
                f"Capture rejected at sample {sample.sample_index}: "
                f"{sample.extra.get('capture_reject_reason', 'unknown')}"
            )


def register_single_segment_repeatability(registry) -> None:
    """Register the canonical live single-segment repeatability experiment."""
    registry.register(
        name=SingleSegmentRepeatabilityExperiment.name,
        title="Single-Segment Repeatability",
        description=SingleSegmentRepeatabilityExperiment.description,
        category="validation",
        tags=["Repeatability", "Thesis", "Tracking", "Servo"],
        default_config_path="config/experiment_single_segment_repeatability.example.yaml",
        factory=SingleSegmentRepeatabilityExperiment.from_dict,
    )


def build_legacy_17_point_targets(
    *,
    inner_ring_radius_mm: float = DEFAULT_INNER_RING_RADIUS_MM,
    outer_ring_radius_mm: float = DEFAULT_OUTER_RING_RADIUS_MM,
) -> list[LegacyRepeatabilityTarget]:
    """Return the exact legacy 17-target single-segment target catalog.

    The target math mirrors the reference repeatability script:
    ``[0] + [inner] * 8 + [outer] * 8`` and 16 angular slots. Tendon commands
    follow the legacy four-cable delta convention
    ``-length * [cos(theta), sin(theta), -cos(theta), -sin(theta)]``.
    """
    inner = max(0.0, float(inner_ring_radius_mm))
    outer = max(0.0, float(outer_ring_radius_mm))
    cable_lengths_mm = [0.0] + [inner] * 8 + [outer] * 8
    cable_angles_rad = [0.0] + [float(index) * math.pi / 4.0 for index in range(16)]
    targets: list[LegacyRepeatabilityTarget] = []
    for index, (length_mm, angle_rad) in enumerate(zip(cable_lengths_mm, cable_angles_rad)):
        cos_theta = math.cos(float(angle_rad))
        sin_theta = math.sin(float(angle_rad))
        deltas_mm = [
            -float(length_mm) * cos_theta,
            -float(length_mm) * sin_theta,
            float(length_mm) * cos_theta,
            float(length_mm) * sin_theta,
        ]
        if index == 0:
            ring = "center"
            angle_deg = 0.0
            group_tags = ["center"]
        else:
            ring = "inner" if index <= 8 else "outer"
            angle_deg = float(((index - 1) % 8) * 45.0)
            axis_tag = "on_axis" if int(angle_deg) % 90 == 0 else "off_axis"
            group_tags = [ring, axis_tag]
        targets.append(
            LegacyRepeatabilityTarget(
                target_index=int(index),
                label=f"T{index:02d}",
                ring=ring,
                ring_radius_mm=float(length_mm),
                angle_rad=float(angle_rad),
                angle_deg=float(angle_deg),
                cable_deltas_mm=[_clean_float(value) for value in deltas_mm],
                cable_deltas_cm=[_clean_float(value / 10.0) for value in deltas_mm],
                group_tags=group_tags,
            )
        )
    return targets


def repeatability_ring_tick_defaults(
    *,
    mapper,
    inner_ring_radius_mm: float = DEFAULT_INNER_RING_RADIUS_MM,
    outer_ring_radius_mm: float = DEFAULT_OUTER_RING_RADIUS_MM,
) -> dict[str, int]:
    """Return ring radii converted to absolute per-servo tick deltas."""
    inner_ticks = abs(int(mapper.displacement_cm_to_ticks(float(inner_ring_radius_mm) / 10.0)))
    outer_ticks = abs(int(mapper.displacement_cm_to_ticks(float(outer_ring_radius_mm) / 10.0)))
    return {
        "inner_ring_radius_ticks": int(inner_ticks),
        "outer_ring_radius_ticks": int(outer_ticks),
    }


def repeatability_target_tick_profile(*, targets: list[LegacyRepeatabilityTarget], mapper) -> dict[str, Any]:
    """Return per-target cable tick deltas plus maximum absolute amplitude."""
    per_target: dict[int, list[int]] = {}
    max_abs_tick_delta = 0
    for target in targets:
        cable_ticks = [
            int(mapper.displacement_cm_to_ticks(float(delta_cm)))
            for delta_cm in target.cable_deltas_cm
        ]
        per_target[int(target.target_index)] = cable_ticks
        if cable_ticks:
            max_abs_tick_delta = max(max_abs_tick_delta, max(abs(value) for value in cable_ticks))
    return {
        "per_target_cable_deltas_ticks": per_target,
        "max_abs_tick_delta": int(max_abs_tick_delta),
    }


def generate_legacy_revisit_sequence(
    targets: list[LegacyRepeatabilityTarget] | None = None,
    *,
    seed: int = 0,
) -> list[LegacyRepeatabilityVisit]:
    """Generate the legacy all-other-target approach sequence.

    For each desired target, all other 16 targets are randomized once. Each
    visit is executed as an approach capture at the randomized prior target,
    followed by a repeat capture at the desired target.
    """
    catalog = list(targets or build_legacy_17_point_targets())
    by_index = {int(target.target_index): target for target in catalog}
    if sorted(by_index) != list(range(LEGACY_TARGET_COUNT)):
        raise ValueError("Legacy repeatability sequence requires target indices 0..16.")
    rng = random.Random(int(seed))
    visits: list[LegacyRepeatabilityVisit] = []
    sequence_index = 0
    for desired_index in range(LEGACY_TARGET_COUNT):
        approach_indices = [index for index in range(LEGACY_TARGET_COUNT) if index != desired_index]
        rng.shuffle(approach_indices)
        for revisit_index, approach_index in enumerate(approach_indices):
            visits.append(
                LegacyRepeatabilityVisit(
                    sequence_index=int(sequence_index),
                    target_index=int(desired_index),
                    approach_index=int(approach_index),
                    revisit_index=int(revisit_index),
                    approach_target=by_index[int(approach_index)],
                    repeat_target=by_index[int(desired_index)],
                )
            )
            sequence_index += 1
    return visits


def compute_single_segment_repeatability_metrics(
    samples: list[ExperimentTimeseriesSample],
    *,
    targets: list[LegacyRepeatabilityTarget] | None = None,
    tool_id: str = "0A",
    thesis_goal_rms_mm: float = 1.0,
    require_robot_frame_tip: bool = True,
    min_repeat_captures_per_target: int = 12,
    min_repeat_capture_fraction: float = 0.85,
    max_rejected_capture_fraction: float = 0.10,
) -> dict[str, Any]:
    """Compute legacy-style repeatability metrics from accepted repeat captures."""
    catalog = list(targets or build_legacy_17_point_targets())
    catalog_by_index = {int(target.target_index): target for target in catalog}
    accepted_repeat: list[tuple[ExperimentTimeseriesSample, np.ndarray, str]] = []
    accepted_approach_count = 0
    rejected_count = 0
    for sample in samples:
        accepted = bool(sample.extra.get("capture_accepted", True)) and "capture_rejected" not in sample.status_flags
        if not accepted:
            rejected_count += 1
            continue
        position, frame = _extract_repeatability_position(
            sample,
            tool_id=tool_id,
            require_robot_frame_tip=require_robot_frame_tip,
        )
        if position is None:
            rejected_count += 1
            continue
        if sample.phase == "repeat":
            accepted_repeat.append((sample, np.asarray(position, dtype=float), str(frame)))
        elif sample.phase == "approach":
            accepted_approach_count += 1

    per_target: dict[str, Any] = {}
    all_deviations: list[float] = []
    all_positions: list[np.ndarray] = []
    position_frames: set[str] = set()
    path_entries: list[dict[str, Any]] = []
    for target in catalog:
        target_rows = [
            (sample, position, frame)
            for sample, position, frame in accepted_repeat
            if int(sample.target_index if sample.target_index is not None else -1) == int(target.target_index)
        ]
        if target_rows:
            positions = np.vstack([position for _sample, position, _frame in target_rows])
            centroid = np.mean(positions, axis=0)
            deviations = positions - centroid.reshape((1, 3))
            distances = np.linalg.norm(deviations, axis=1)
            xy_distances = np.linalg.norm(deviations[:, 0:2], axis=1)
            all_deviations.extend(float(value) for value in distances.tolist())
            all_positions.extend([np.asarray(row, dtype=float) for row in positions])
            position_frames.update(str(frame) for _sample, _position, frame in target_rows)
            for (sample, position, _frame), deviation in zip(target_rows, distances):
                path_entries.append(
                    {
                        "target_index": int(target.target_index),
                        "approach_index": int(sample.approach_index if sample.approach_index is not None else -1),
                        "revisit_index": int(sample.revisit_index if sample.revisit_index is not None else -1),
                        "deviation_mm": float(deviation),
                    }
                )
            rmse = float(np.sqrt(np.mean(np.square(distances)))) if len(distances) else None
            xy_rmse = float(np.sqrt(np.mean(np.square(xy_distances)))) if len(xy_distances) else None
            max_dev = float(np.max(distances)) if len(distances) else None
            centroid_list = [float(value) for value in centroid.tolist()]
        else:
            rmse = None
            xy_rmse = None
            max_dev = None
            centroid_list = []
            distances = np.asarray([], dtype=float)
        per_target[str(target.target_index)] = {
            "target_index": int(target.target_index),
            "label": target.label,
            "ring": target.ring,
            "ring_radius_mm": float(target.ring_radius_mm),
            "angle_deg": float(target.angle_deg),
            "group_tags": list(target.group_tags),
            "repeat_sample_count": int(len(target_rows)),
            "centroid_mm": centroid_list,
            "XY_RMSE_mm": xy_rmse,
            "XYZ_RMSE_mm": rmse,
            "spread_rms_mm": rmse,
            "rmse_mm": rmse,
            "max_deviation_mm": max_dev,
            "mean_deviation_mm": float(np.mean(distances)) if len(distances) else None,
        }

    overall_rms = _safe_rms(all_deviations)
    overall_max = float(max(all_deviations)) if all_deviations else None
    path_rms = _safe_rms([entry["deviation_mm"] for entry in path_entries])
    group_metrics = _compute_group_metrics(per_target)
    run_validity = _assess_repeatability_run_validity(
        per_target=per_target,
        targets=catalog,
        valid_repeat_sample_count=int(len(accepted_repeat)),
        rejected_capture_count=int(rejected_count),
        min_repeat_captures_per_target=int(min_repeat_captures_per_target),
        min_repeat_capture_fraction=float(min_repeat_capture_fraction),
        max_rejected_capture_fraction=float(max_rejected_capture_fraction),
    )
    legacy_style = compute_legacy_style_repeatability_analysis(
        samples,
        targets=catalog,
        tool_id=tool_id,
        require_robot_frame_tip=require_robot_frame_tip,
    )
    return {
        "protocol": "legacy_17_target_single_segment_all_other_approaches",
        "metric_definitions": {
            "thesis_strict": (
                "Accepted repeat captures only; position selected by require_robot_frame_tip; "
                "per-target centroid is mean XYZ; RMSE/max are 3D Euclidean distances to centroid."
            ),
            "legacy_style": (
                "Legacy comparison of desired-target/repeat captures only. The original script used "
                "T_tip_2_model XYZ positions, filtered abs(x)<100 mm, used mean XYZ centroids, "
                "computed 3D Euclidean RMSE/max, and plotted XY clusters."
            ),
        },
        "target_count": LEGACY_TARGET_COUNT,
        "planned_visit_count": LEGACY_VISIT_COUNT,
        "planned_capture_count": LEGACY_CAPTURE_COUNT,
        "valid_repeat_sample_count": int(len(accepted_repeat)),
        "valid_approach_sample_count": int(accepted_approach_count),
        "invalid_sample_count": int(rejected_count),
        "rejected_capture_count": int(rejected_count),
        "per_target_metrics": per_target,
        "overall_repeatability_rms_mm": overall_rms,
        "overall_max_deviation_mm": overall_max,
        "path_dependence_rms_mm": path_rms,
        "path_dependence_by_approach": path_entries,
        "group_metrics": group_metrics,
        "run_validity": run_validity,
        "target_catalog": [target.to_dict() for target in catalog],
        "legacy_style_comparison": legacy_style,
        "position_frame": _position_frame_label(position_frames, require_robot_frame_tip=require_robot_frame_tip),
        "registration_available": any("registration_missing" not in sample.status_flags for sample in samples),
        "tip_calibration_available": any("full_pose_available" in sample.status_flags for sample in samples),
        "thesis_goal_rms_mm": float(thesis_goal_rms_mm),
        "thesis_goal_pass": bool(overall_rms is not None and overall_rms <= float(thesis_goal_rms_mm)),
    }


def load_repeatability_metrics_from_run(path: Path) -> dict[str, Any]:
    """Load experiment metrics from a canonical repeatability run directory or summary file."""
    root = Path(path)
    summary_path = root if root.is_file() and root.name == "summary.json" else root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Repeatability baseline summary not found: {summary_path}")
    summary_stat = summary_path.stat()
    resolved_summary_path = str(summary_path.resolve())
    cache_key = (resolved_summary_path, int(summary_stat.st_mtime_ns), int(summary_stat.st_size))
    with _REPEATABILITY_METRICS_CACHE_LOCK:
        cached_metrics = _REPEATABILITY_METRICS_CACHE.get(cache_key)
        if cached_metrics is not None:
            return dict(cached_metrics)
    started = time.monotonic()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = payload.get("experiment_metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError(f"Repeatability baseline summary has no experiment_metrics mapping: {summary_path}")
    metrics = dict(metrics)
    metrics.setdefault("status", payload.get("status"))
    metrics.setdefault("success", payload.get("success"))
    metrics.setdefault("run_id", payload.get("run_id"))
    protocol = str(metrics.get("protocol", ""))
    experiment_name = str(payload.get("experiment_name", ""))
    if experiment_name not in {"single_segment_repeatability", "repeatability_dataset"}:
        raise ValueError(f"Baseline run is not a repeatability experiment: {experiment_name}")
    if experiment_name == "single_segment_repeatability" and protocol != "legacy_17_target_single_segment_all_other_approaches":
        raise ValueError(f"Baseline run does not use the legacy 17-target protocol: {protocol}")
    with _REPEATABILITY_METRICS_CACHE_LOCK:
        stale_keys = [
            key
            for key in _REPEATABILITY_METRICS_CACHE
            if key[0] == resolved_summary_path and key != cache_key
        ]
        for stale_key in stale_keys:
            _REPEATABILITY_METRICS_CACHE.pop(stale_key, None)
        _REPEATABILITY_METRICS_CACHE[cache_key] = dict(metrics)
    LOG.debug(
        "Loaded repeatability baseline metrics from %s in %.1f ms",
        summary_path,
        (time.monotonic() - started) * 1000.0,
    )
    return dict(metrics)


def compute_legacy_style_repeatability_analysis(
    samples: list[ExperimentTimeseriesSample],
    *,
    targets: list[LegacyRepeatabilityTarget] | None = None,
    tool_id: str = "0A",
    require_robot_frame_tip: bool = True,
    require_capture_accepted: bool = True,
    outlier_abs_x_limit_mm: float = 100.0,
) -> dict[str, Any]:
    """Compute an explicit comparison layer matching the legacy repeatability figure.

    The old analysis used the desired-target captures only, computed a mean XYZ
    centroid per target, measured Euclidean error to that centroid in 3D, and
    plotted the XY coordinates with labels showing the 3D RMSE.
    """
    catalog = list(targets or build_legacy_17_point_targets())
    rows_by_target: dict[int, list[np.ndarray]] = {int(target.target_index): [] for target in catalog}
    selected = 0
    rejected = 0
    frame_counts: dict[str, int] = {}
    for sample in samples:
        if str(getattr(sample, "phase", "")) != "repeat":
            continue
        if require_capture_accepted and (
            not bool(sample.extra.get("capture_accepted", True)) or "capture_rejected" in sample.status_flags
        ):
            rejected += 1
            continue
        position, frame = _extract_repeatability_position(
            sample,
            tool_id=tool_id,
            require_robot_frame_tip=require_robot_frame_tip,
        )
        if position is None:
            rejected += 1
            continue
        position_arr = np.asarray(position, dtype=float)
        if position_arr.shape != (3,) or not np.all(np.isfinite(position_arr)):
            rejected += 1
            continue
        if abs(float(position_arr[0])) >= float(outlier_abs_x_limit_mm):
            rejected += 1
            continue
        target_index = int(sample.target_index if sample.target_index is not None else -1)
        if target_index not in rows_by_target:
            rejected += 1
            continue
        rows_by_target[target_index].append(position_arr)
        selected += 1
        frame_counts[str(frame or "unknown")] = int(frame_counts.get(str(frame or "unknown"), 0)) + 1

    per_target: dict[str, Any] = {}
    all_xy_deviations: list[float] = []
    all_xyz_deviations: list[float] = []
    for target in catalog:
        positions = rows_by_target[int(target.target_index)]
        if positions:
            matrix = np.vstack(positions)
            centroid = np.mean(matrix, axis=0)
            deviations = matrix - centroid.reshape((1, 3))
            xy_distances = np.linalg.norm(deviations[:, 0:2], axis=1)
            xyz_distances = np.linalg.norm(deviations, axis=1)
            xy_rmse = float(np.sqrt(np.mean(np.square(xy_distances))))
            xyz_rmse = float(np.sqrt(np.mean(np.square(xyz_distances))))
            max_deviation = float(np.max(xyz_distances))
            all_xy_deviations.extend(float(value) for value in xy_distances.tolist())
            all_xyz_deviations.extend(float(value) for value in xyz_distances.tolist())
            centroid_list = [float(value) for value in centroid.tolist()]
            samples_xyz = [[float(value) for value in row.tolist()] for row in matrix]
        else:
            xy_rmse = None
            xyz_rmse = None
            max_deviation = None
            centroid_list = []
            samples_xyz = []
        per_target[str(target.target_index)] = {
            "target_id": int(target.target_index),
            "target_index": int(target.target_index),
            "label": target.label,
            "n_samples": int(len(positions)),
            "XY_RMSE_mm": xy_rmse,
            "XYZ_RMSE_mm": xyz_rmse,
            "max_deviation_mm": max_deviation,
            "centroid_x_mm": centroid_list[0] if len(centroid_list) >= 1 else None,
            "centroid_y_mm": centroid_list[1] if len(centroid_list) >= 2 else None,
            "centroid_z_mm": centroid_list[2] if len(centroid_list) >= 3 else None,
            "centroid_mm": centroid_list,
            "sample_positions_mm": samples_xyz,
            "analysis_mode": "legacy_style",
        }

    return {
        "analysis_mode": "legacy_style",
        "definition": (
            "Desired-target/repeat captures only; mean XYZ centroid per target; XY_RMSE is XY-plane "
            "distance to centroid; XYZ_RMSE and max_deviation are 3D Euclidean distances to centroid. "
            "Legacy figure labels used XYZ_RMSE while plotting XY."
        ),
        "sample_selection": {
            "phase": "repeat",
            "require_capture_accepted": bool(require_capture_accepted),
            "require_robot_frame_tip": bool(require_robot_frame_tip),
            "outlier_filter": f"abs(x_mm) < {float(outlier_abs_x_limit_mm):.3f}",
            "selected_repeat_sample_count": int(selected),
            "rejected_repeat_sample_count": int(rejected),
            "frame_counts": frame_counts,
        },
        "pose_frame_used": _position_frame_label(set(frame_counts), require_robot_frame_tip=require_robot_frame_tip),
        "overall_XY_RMSE_mm": _safe_rms(all_xy_deviations),
        "overall_XYZ_RMSE_mm": _safe_rms(all_xyz_deviations),
        "overall_max_deviation_mm": float(max(all_xyz_deviations)) if all_xyz_deviations else None,
        "per_target": per_target,
    }


def compute_repeatability_baseline_comparison(
    *,
    current_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    baseline_path: str = "",
) -> dict[str, Any]:
    """Compare two repeatability summaries using thesis-facing scalar and per-target deltas."""
    current = dict(current_metrics or {})
    baseline = dict(baseline_metrics or {})
    per_target_current = dict(current.get("per_target_metrics", {}) or {})
    per_target_baseline = dict(baseline.get("per_target_metrics", {}) or {})
    per_target_delta: dict[str, Any] = {}
    for key in sorted(set(per_target_current) | set(per_target_baseline), key=lambda value: int(value) if str(value).isdigit() else 10**9):
        c_row = dict(per_target_current.get(str(key), {}) or {})
        b_row = dict(per_target_baseline.get(str(key), {}) or {})
        per_target_delta[str(key)] = {
            "target_index": int(c_row.get("target_index", b_row.get("target_index", key))),
            "label": str(c_row.get("label", b_row.get("label", f"T{int(key):02d}" if str(key).isdigit() else key))),
            "current_rmse_mm": _as_optional_float(c_row.get("spread_rms_mm", c_row.get("rmse_mm"))),
            "baseline_rmse_mm": _as_optional_float(b_row.get("spread_rms_mm", b_row.get("rmse_mm"))),
            "delta_rmse_mm": _delta(c_row.get("spread_rms_mm", c_row.get("rmse_mm")), b_row.get("spread_rms_mm", b_row.get("rmse_mm"))),
            "current_max_deviation_mm": _as_optional_float(c_row.get("max_deviation_mm")),
            "baseline_max_deviation_mm": _as_optional_float(b_row.get("max_deviation_mm")),
            "delta_max_deviation_mm": _delta(c_row.get("max_deviation_mm"), b_row.get("max_deviation_mm")),
        }
    group_delta = _compare_group_metrics(
        dict(current.get("group_metrics", {}) or {}),
        dict(baseline.get("group_metrics", {}) or {}),
    )
    overall_delta = _delta(current.get("overall_repeatability_rms_mm"), baseline.get("overall_repeatability_rms_mm"))
    return {
        "available": True,
        "baseline_path": str(baseline_path or ""),
        "baseline_status": str(baseline.get("status", "unknown") or "unknown"),
        "overall_rms": _comparison_row(current.get("overall_repeatability_rms_mm"), baseline.get("overall_repeatability_rms_mm")),
        "overall_max_deviation": _comparison_row(current.get("overall_max_deviation_mm"), baseline.get("overall_max_deviation_mm")),
        "path_dependence_rms": _comparison_row(current.get("path_dependence_rms_mm"), baseline.get("path_dependence_rms_mm")),
        "rejected_capture_count": {
            "current": int(current.get("rejected_capture_count", 0) or 0),
            "baseline": int(baseline.get("rejected_capture_count", 0) or 0),
            "delta": int(current.get("rejected_capture_count", 0) or 0) - int(baseline.get("rejected_capture_count", 0) or 0),
        },
        "per_target_rmse": per_target_delta,
        "group_metrics": group_delta,
        "improved_overall_rms": bool(overall_delta is not None and overall_delta < 0.0),
    }


def _precheck_single_segment_repeatability(
    *,
    session: ExperimentSession,
    config: SingleSegmentRepeatabilityConfig,
    servo_ids: list[int],
    neutral_ticks: list[int],
) -> None:
    settings = session.context.settings
    snapshot = session.context.tracking_service.get_snapshot()
    backend_name = str(snapshot.selected_backend_name or snapshot.backend_identity or "").lower()
    if "bridge" in backend_name:
        raise RuntimeError("Single-segment repeatability must use the active Python NDI tracker backend, not tracker_bridge.")
    if bool(settings.runtime.mock_mode):
        raise RuntimeError("Single-segment repeatability is a live thesis experiment. Disable mock mode before running.")
    if snapshot.canonical_state != "streaming_healthy":
        raise RuntimeError(f"Tracker must be connected and healthy; current state is {snapshot.canonical_state}.")
    if snapshot.registration_state != "loaded":
        raise RuntimeError("Accepted base registration must be loaded before repeatability.")
    pivot_tip_file = getattr(settings.registration, "penprobe_file", None)
    if not pivot_tip_file:
        raise RuntimeError("No 0B pen-probe pivot tip file is configured.")
    pivot_tip_path = _resolve_repo_path(session.context.project_root, pivot_tip_file)
    if not pivot_tip_path.exists():
        raise RuntimeError(f"0B pen-probe pivot tip file is missing: {pivot_tip_path}")
    runtime_tip_policy = evaluate_runtime_tip_trust(
        snapshot=snapshot,
        workflow=RUNTIME_TIP_POLICY_WORKFLOW,
        allow_lower_trust=bool(config.allow_debug_coil_as_tip),
    )
    if config.require_robot_frame_tip and not runtime_tip_policy.allowed_for_workflow:
        raise RuntimeError(
            "Single-segment repeatability requires a thesis_trusted runtime tip policy outcome; "
            f"requested workflow/platform={runtime_tip_policy.requested_workflow}, "
            f"resolved canonical workflow={runtime_tip_policy.workflow}, "
            f"mode={runtime_tip_policy.mode}, trust={runtime_tip_policy.trust_label}, "
            f"state={snapshot.runtime_tip_calibration_state}, tip pose={snapshot.tip_pose_status}, "
            f"reasons={runtime_tip_policy.reasons or ['policy_not_allowed']}."
        )
    if config.require_robot_frame_tip and not runtime_tip_policy.thesis_trusted:
        raise RuntimeError(
            "Single-segment repeatability is blocked because the active runtime tip path is not thesis_trusted; "
            f"requested workflow/platform={runtime_tip_policy.requested_workflow}, "
            f"resolved canonical workflow={runtime_tip_policy.workflow}, "
            f"mode={runtime_tip_policy.mode}, trust={runtime_tip_policy.trust_label}."
        )
    if config.require_robot_frame_tip and snapshot.T_robot_tip is None:
        raise RuntimeError(f"Live robot-frame tip pose must be active; tip pose status is {snapshot.tip_pose_status}.")
    runtime_tip_policy = evaluate_runtime_tip_trust(
        snapshot=snapshot,
        workflow=RUNTIME_TIP_POLICY_WORKFLOW,
        allow_lower_trust=bool(config.allow_debug_coil_as_tip),
    )
    gate = _tracker_gate_status(
        snapshot=snapshot,
        tool_id=config.tool_id,
        max_tracker_age_s=float(config.max_tracker_age_s),
        require_robot_frame_tip=bool(config.require_robot_frame_tip),
        allow_debug_coil_as_tip=bool(config.allow_debug_coil_as_tip),
    )
    if not gate["accepted"]:
        raise RuntimeError(f"Tracker capture gate is blocked before run start: {gate['reason']}.")
    if not getattr(session.context.servo_service, "is_connected", False):
        raise RuntimeError("OpenRB / DYNAMIXEL servo service must be connected.")
    if len(servo_ids) != 4:
        raise RuntimeError(
            "Single-segment repeatability requires exactly 4 configured servos for active segment "
            f"{settings.robot.active_segment_label()} ({settings.robot.active_segment_key()}); found {servo_ids}."
        )
    if len(neutral_ticks) != len(servo_ids):
        raise RuntimeError("Neutral setpoints are missing for one or more configured servos.")
    if float(config.outer_ring_radius_mm) < float(config.inner_ring_radius_mm):
        raise RuntimeError(
            "Repeatability target config is invalid: outer_ring_radius_mm must be >= inner_ring_radius_mm."
        )
    target_catalog = build_legacy_17_point_targets(
        inner_ring_radius_mm=float(config.inner_ring_radius_mm),
        outer_ring_radius_mm=float(config.outer_ring_radius_mm),
    )
    tick_profile = repeatability_target_tick_profile(
        targets=target_catalog,
        mapper=session.context.servo_service.mapper,
    )
    max_target_abs_tick_delta = int(tick_profile["max_abs_tick_delta"])
    if max_target_abs_tick_delta > int(config.max_target_tick_delta_from_startup):
        raise RuntimeError(
            "Repeatability target envelope exceeds the configured experiment-safe amplitude cap: "
            f"{max_target_abs_tick_delta} ticks requested > "
            f"{int(config.max_target_tick_delta_from_startup)} ticks cap. "
            "Lower the ring radii or increase max_target_tick_delta_from_startup intentionally."
        )
    calibration_summary = session.context.servo_service.neutral_calibration.get_calibration_summary()
    if not calibration_summary.exists or not calibration_summary.compatible:
        raise RuntimeError(f"Servo calibration artifact is not ready: {calibration_summary.message}")
    pretension_source = calibration_summary.pretension_source_summary([int(value) for value in servo_ids])
    if not pretension_source.accepted or not pretension_source.usable:
        raise RuntimeError(pretension_source.message)
    with session.context.servo_service.exclusive_bus_operation(
        owner="single_segment_repeatability_precheck",
        reason="single-segment repeatability preflight telemetry validation",
    ):
        telemetry_report = _read_servo_telemetry_payload(
            session,
            servo_ids,
            stage="preflight",
        )
        session.set_metric("preflight_servo_telemetry", telemetry_report)
        if not bool(telemetry_report.get("ok", False)):
            LOG.error(
                "Repeatability preflight telemetry invalid | code=%s | message=%s | issues=%s",
                telemetry_report.get("failure_code"),
                telemetry_report.get("message"),
                telemetry_report.get("issues"),
            )
            raise RuntimeError(
                "preflight_telemetry_invalid: "
                f"code={telemetry_report.get('failure_code') or 'unknown'} "
                f"detail={telemetry_report.get('message') or 'telemetry validation failed'}"
            )
        LOG.info(
            "Repeatability preflight telemetry valid | servo_ids=%s | freshness_threshold_s=%.3f",
            servo_ids,
            float(telemetry_report.get("freshness_threshold_s", 0.0) or 0.0),
        )
    for servo_id in servo_ids:
        assessment = session.context.servo_service.assess_experiment_motion(int(servo_id))
        if not assessment.ready:
            raise RuntimeError(f"Servo {servo_id} is not ready for repeatability commands: {assessment.reason}")


def _record_repeatability_run_provenance(
    *,
    session: ExperimentSession,
    config: SingleSegmentRepeatabilityConfig,
    servo_ids: list[int],
    neutral_ticks: list[int],
) -> None:
    settings = session.context.settings
    snapshot = session.context.tracking_service.get_snapshot()
    servo_calibration_summary = session.context.servo_service.get_calibration_summary()
    startup_reference = session.context.servo_service.resolve_startup_reference_ticks(list(servo_ids))
    pivot_tip_file = getattr(settings.registration, "penprobe_file", None)
    pivot_tip_path = (
        _resolve_repo_path(session.context.project_root, pivot_tip_file)
        if pivot_tip_file
        else None
    )
    registration_path = Path(
        getattr(snapshot, "registration_path", None) or session.context.registration_path
    )
    runtime_tip_path_raw = getattr(snapshot, "runtime_tip_calibration_path", None)
    runtime_tip_path = Path(runtime_tip_path_raw) if runtime_tip_path_raw else None
    gate = _tracker_gate_status(
        snapshot=snapshot,
        tool_id=config.tool_id,
        max_tracker_age_s=float(config.max_tracker_age_s),
        require_robot_frame_tip=bool(config.require_robot_frame_tip),
        allow_debug_coil_as_tip=bool(config.allow_debug_coil_as_tip),
    )
    target_catalog = build_legacy_17_point_targets(
        inner_ring_radius_mm=float(config.inner_ring_radius_mm),
        outer_ring_radius_mm=float(config.outer_ring_radius_mm),
    )
    target_tick_profile = repeatability_target_tick_profile(
        targets=target_catalog,
        mapper=session.context.servo_service.mapper,
    )
    ring_ticks = repeatability_ring_tick_defaults(
        mapper=session.context.servo_service.mapper,
        inner_ring_radius_mm=float(config.inner_ring_radius_mm),
        outer_ring_radius_mm=float(config.outer_ring_radius_mm),
    )
    pretension_source = servo_calibration_summary.pretension_source_summary([int(value) for value in servo_ids])
    pretension_entries: dict[str, Any] = {}
    for servo_id in servo_ids:
        entry = servo_calibration_summary.servo_entries.get(int(servo_id))
        if entry is None:
            pretension_entries[str(servo_id)] = {"exists": False}
            continue
        pretension_entries[str(servo_id)] = {
            "exists": True,
            "neutral_setpoint": entry.neutral_setpoint,
            "safe_min_tick": entry.safe_min_tick,
            "safe_max_tick": entry.safe_max_tick,
            "pretension_current_threshold_ma": entry.pretension_current_threshold_ma,
            "pretension_final_position_tick": entry.pretension_final_position_tick,
            "pretension_result_status": entry.pretension_result_status,
            "pretension_source": entry.pretension_source,
            "pretension_note": entry.pretension_note,
            "pretension_completed_at_utc": entry.pretension_completed_at_utc,
            "latest_pretension_run": dict(entry.latest_pretension_run or {}) or None,
            "capture_source": entry.capture_source,
            "calibrated_at_utc": entry.calibrated_at_utc,
            "status": entry.status,
            "valid": bool(entry.valid),
        }
    precheck_summary = {
        "overall_status": "ready",
        "checks": [
            {
                "key": "tracker_backend",
                "status": "ok",
                "message": (
                    f"Backend {snapshot.backend_identity or snapshot.selected_backend_name or 'unknown'} "
                    f"selected={snapshot.selected_backend_name or 'unknown'}."
                ),
            },
            {
                "key": "tracking_state",
                "status": "ok",
                "message": f"Tracker state {snapshot.canonical_state}; frame age {gate.get('tracker_age_s')}.",
            },
            {
                "key": "tracker_gate",
                "status": "ok" if gate.get("accepted") else "blocked",
                "message": str(gate.get("reason", "unknown")),
            },
            {
                "key": "registration",
                "status": "ok",
                "message": f"Base registration state={snapshot.registration_state}.",
            },
            {
                "key": "runtime_tip",
                "status": "ok",
                "message": (
                    f"Runtime tip mode={getattr(snapshot, 'runtime_tip_mode', 'latest_accepted')}; "
                    f"trust={runtime_tip_policy.trust_label}; "
                    f"Runtime tip state={snapshot.runtime_tip_calibration_state}; "
                    f"tip pose={snapshot.tip_pose_status}; fallback={snapshot.runtime_tip_identity_fallback}."
                ),
            },
            {
                "key": "servos",
                "status": "ok",
                "message": (
                    f"Active segment {settings.robot.active_segment_label()} "
                    f"({settings.robot.active_segment_key()}) servo IDs={servo_ids}; "
                    f"connected={bool(getattr(session.context.servo_service, 'is_connected', False))}."
                ),
            },
            {
                "key": "pretension",
                "status": "ok",
                "message": pretension_source.message,
            },
        ],
    }
    provenance = {
        "backend_identity": str(snapshot.backend_identity or ""),
        "selected_backend_name": str(snapshot.selected_backend_name or ""),
        "configured_backend_name": str(getattr(settings.serial, "tracker_backend", "") or ""),
        "tracking_state": str(snapshot.canonical_state or ""),
        "tool_id": str(config.tool_id or "0A"),
        "require_robot_frame_tip": bool(config.require_robot_frame_tip),
        "allow_debug_coil_as_tip": bool(config.allow_debug_coil_as_tip),
        "runtime_tip_mode": str(getattr(snapshot, "runtime_tip_mode", "latest_accepted") or "latest_accepted"),
        "runtime_tip_trust_level": runtime_tip_policy.trust_label,
        "runtime_tip_policy": runtime_tip_policy.to_dict(),
        "thesis_trusted_runtime_tip": bool(runtime_tip_policy.thesis_trusted),
        "thesis_trusted_run": bool(runtime_tip_policy.thesis_trusted),
        "startup_reference_source": str(startup_reference.source or "neutral"),
        "startup_reference_ticks_by_servo": {
            str(servo_id): int(tick)
            for servo_id, tick in sorted(startup_reference.ticks_by_servo.items())
        },
        "neutral_ticks_by_servo": {
            str(servo_id): int(neutral_tick)
            for servo_id, neutral_tick in zip(servo_ids, neutral_ticks)
        },
        "pivot_tip": _file_provenance(pivot_tip_path),
        "base_registration": {
            **_file_provenance(registration_path),
            "state": str(snapshot.registration_state or ""),
            "tracking_path": str(getattr(snapshot, "registration_path", "") or ""),
            "stored_timestamp_utc": getattr(snapshot, "stored_registration_timestamp_utc", None),
            "stored_fre_mm": getattr(snapshot, "stored_registration_fre_mm", None),
        },
        "runtime_tip_calibration": {
            **_file_provenance(runtime_tip_path),
            "state": str(snapshot.runtime_tip_calibration_state or ""),
            "mode": str(getattr(snapshot, "runtime_tip_mode", "latest_accepted") or "latest_accepted"),
            "trust_level": runtime_tip_policy.trust_label,
            "policy": runtime_tip_policy.to_dict(),
            "artifact_used": bool(runtime_tip_path is not None and runtime_tip_path.exists()),
            "identity_marker": (
                "T_coil_tip_identity__coil_pose_used_as_tip"
                if str(getattr(snapshot, "runtime_tip_mode", "")) == "coil_as_tip"
                else None
            ),
            "thesis_trusted": bool(runtime_tip_policy.thesis_trusted),
            "mode_message": str(getattr(snapshot, "runtime_tip_mode_message", "") or ""),
            "artifact_kind": getattr(snapshot, "runtime_tip_selected_artifact_kind", None),
            "selected_artifact_path": getattr(snapshot, "runtime_tip_selected_artifact_path", None),
            "stored_timestamp_utc": getattr(snapshot, "stored_runtime_tip_timestamp_utc", None),
            "measurement_tool_id": getattr(snapshot, "stored_runtime_tip_measurement_tool_id", None),
            "coil_tool_id": getattr(snapshot, "stored_runtime_tip_coil_tool_id", None),
            "identity_fallback": bool(getattr(snapshot, "runtime_tip_identity_fallback", False)),
            "tip_pose_status": str(getattr(snapshot, "tip_pose_status", "")),
        },
        "pretension_artifact": {
            **_file_provenance(Path(servo_calibration_summary.path)),
            "status": servo_calibration_summary.status,
            "message": servo_calibration_summary.message,
            "compatible": bool(servo_calibration_summary.compatible),
            "updated_at_utc": servo_calibration_summary.updated_at_utc,
            "schema_version": int(servo_calibration_summary.schema_version),
            "active_source_type": pretension_source.source_type,
            "accepted": bool(pretension_source.accepted),
            "usable": bool(pretension_source.usable),
            "active_source_message": pretension_source.message,
            "active_source_updated_at_utc": pretension_source.updated_at_utc,
            "active_source_note": pretension_source.note,
            "active_source_by_servo": {
                str(servo_id): source
                for servo_id, source in sorted(pretension_source.source_by_servo.items())
            },
            "servos": pretension_entries,
        },
        "precheck_trust_summary": precheck_summary,
        "repeatability_target_geometry": {
            "inner_ring_radius_mm": float(config.inner_ring_radius_mm),
            "outer_ring_radius_mm": float(config.outer_ring_radius_mm),
            "inner_ring_radius_ticks": int(ring_ticks["inner_ring_radius_ticks"]),
            "outer_ring_radius_ticks": int(ring_ticks["outer_ring_radius_ticks"]),
            "max_target_abs_tick_delta_from_startup": int(target_tick_profile["max_abs_tick_delta"]),
            "max_target_tick_delta_from_startup_cap": int(config.max_target_tick_delta_from_startup),
            "spool_diameter_mm": float(settings.robot.spool_diameter_cm) * 10.0,
            "ticks_per_revolution": int(settings.robot.ticks_per_revolution),
        },
        "transform_chain_summary": _repeatability_transform_chain_summary(snapshot),
    }
    session.metadata.backend_info["run_provenance"] = {
        "backend_identity": provenance["backend_identity"],
        "selected_backend_name": provenance["selected_backend_name"],
        "configured_backend_name": provenance["configured_backend_name"],
        "tracking_state": provenance["tracking_state"],
        "pretension_artifact": provenance["pretension_artifact"],
        "precheck_trust_summary": precheck_summary,
    }
    session.metadata.registration_info["pivot_tip"] = provenance["pivot_tip"]
    session.metadata.registration_info["base_registration"] = provenance["base_registration"]
    session.metadata.registration_info["runtime_tip_calibration"] = provenance["runtime_tip_calibration"]
    session.metadata.registration_info["runtime_tip_mode"] = provenance["runtime_tip_mode"]
    session.metadata.registration_info["runtime_tip_trust_level"] = provenance["runtime_tip_trust_level"]
    session.metadata.registration_info["runtime_tip_policy"] = provenance["runtime_tip_policy"]
    session.metadata.registration_info["thesis_trusted_runtime_tip"] = provenance["thesis_trusted_runtime_tip"]
    session.metadata.registration_info["transform_chain_summary"] = provenance["transform_chain_summary"]
    session.metadata.backend_info["pretension_source"] = {
        "source_type": pretension_source.source_type,
        "accepted": bool(pretension_source.accepted),
        "usable": bool(pretension_source.usable),
        "message": pretension_source.message,
        "updated_at_utc": pretension_source.updated_at_utc,
        "note": pretension_source.note,
    }
    session.set_metric("run_provenance", provenance)


def _configured_single_segment_servo_ids(session: ExperimentSession) -> list[int]:
    robot = session.context.settings.robot
    context = robot.operating_context()
    if context.operating_mode != "single_segment":
        raise RuntimeError(
            f"Single-segment repeatability requires single_segment operating mode; current mode is {context.operating_mode}."
        )
    servo_ids = [int(value) for value in context.active_segment_servo_ids]
    return servo_ids


def _repeatability_transform_chain_summary(snapshot) -> dict[str, Any]:
    runtime_tip_policy = evaluate_runtime_tip_trust(snapshot=snapshot, workflow=RUNTIME_TIP_POLICY_WORKFLOW)
    return {
        "expression": "T_robot_tip = T_robot_aurora @ T_aurora_coil_0A @ T_coil_tip",
        "runtime_tip_mode": str(getattr(snapshot, "runtime_tip_mode", "latest_accepted") or "latest_accepted"),
        "runtime_tip_trust_level": runtime_tip_policy.trust_label,
        "runtime_tip_policy": runtime_tip_policy.to_dict(),
        "registration_state": str(getattr(snapshot, "registration_state", "") or ""),
        "runtime_tip_state": str(getattr(snapshot, "runtime_tip_calibration_state", "") or ""),
        "tip_pose_status": str(getattr(snapshot, "tip_pose_status", "") or ""),
        "registration_path": str(getattr(snapshot, "registration_path", "") or ""),
        "runtime_tip_artifact_path": str(getattr(snapshot, "runtime_tip_calibration_path", "") or ""),
        "identity_fallback": bool(getattr(snapshot, "runtime_tip_identity_fallback", False)),
        "coil_as_tip_identity_marker": (
            "T_coil_tip_identity__coil_pose_used_as_tip"
            if str(getattr(snapshot, "runtime_tip_mode", "")) == "coil_as_tip"
            else None
        ),
        "stored_registration_fre_mm": getattr(snapshot, "stored_registration_fre_mm", None),
        "stored_registration_timestamp_utc": getattr(snapshot, "stored_registration_timestamp_utc", None),
        "stored_runtime_tip_timestamp_utc": getattr(snapshot, "stored_runtime_tip_timestamp_utc", None),
        "runtime_coil_tool_id": str(getattr(snapshot, "runtime_coil_tool_id", "0A") or "0A"),
    }


def _load_neutral_ticks(session: ExperimentSession, servo_ids: list[int]) -> list[int]:
    reference = session.context.servo_service.resolve_startup_reference_ticks(list(servo_ids))
    return [int(reference.ticks_by_servo[servo_id]) for servo_id in servo_ids if servo_id in reference.ticks_by_servo]


def _wait_for_valid_capture(
    *,
    session: ExperimentSession,
    tool_id: str,
    max_tracker_age_s: float,
    timeout_s: float,
    poll_interval_s: float,
    require_robot_frame_tip: bool,
    allow_debug_coil_as_tip: bool = False,
) -> tuple[Any, dict[str, Any]]:
    deadline = session.context.monotonic_fn() + float(timeout_s)
    last_snapshot = session.context.tracking_service.get_snapshot()
    last_gate = _tracker_gate_status(
        snapshot=last_snapshot,
        tool_id=tool_id,
        max_tracker_age_s=max_tracker_age_s,
        require_robot_frame_tip=require_robot_frame_tip,
        allow_debug_coil_as_tip=allow_debug_coil_as_tip,
    )
    if last_gate["accepted"]:
        return last_snapshot, last_gate
    while session.context.monotonic_fn() < deadline:
        session.raise_if_stop_requested()
        session.context.sleep_fn(float(poll_interval_s))
        last_snapshot = session.context.tracking_service.get_snapshot()
        last_gate = _tracker_gate_status(
            snapshot=last_snapshot,
            tool_id=tool_id,
            max_tracker_age_s=max_tracker_age_s,
            require_robot_frame_tip=require_robot_frame_tip,
            allow_debug_coil_as_tip=allow_debug_coil_as_tip,
        )
        if last_gate["accepted"]:
            return last_snapshot, last_gate
    return last_snapshot, last_gate


def _tracker_gate_status(
    *,
    snapshot,
    tool_id: str,
    max_tracker_age_s: float,
    require_robot_frame_tip: bool,
    allow_debug_coil_as_tip: bool = False,
) -> dict[str, Any]:
    tool_key = str(tool_id or "0A").upper()
    tool = snapshot.tools.get(tool_key)
    reasons: list[str] = []
    age = snapshot.tracker_data_age_s
    if snapshot.canonical_state not in {"streaming_healthy", "streaming_degraded"}:
        reasons.append(f"tracker_state={snapshot.canonical_state}")
    if snapshot.tracker_data_stale:
        reasons.append("tracker_data_stale")
    if age is None:
        reasons.append("missing_tracker_age")
    elif float(age) > float(max_tracker_age_s):
        reasons.append(f"tracker_age_{float(age):.3f}s_exceeds_{float(max_tracker_age_s):.3f}s")
    if tool is None:
        reasons.append(f"missing_tool_{tool_key}")
    else:
        if not tool.present:
            reasons.append(f"tool_{tool_key}_not_present")
        if tool.valid is False:
            reasons.append(f"tool_{tool_key}_invalid")
        if tool.translation_mm is None:
            reasons.append(f"tool_{tool_key}_missing_translation")
        if str(tool.tracking_state).lower() in {"invalid", "missing", "out_of_volume"}:
            reasons.append(f"tool_{tool_key}_state_{tool.tracking_state}")
    if require_robot_frame_tip:
        runtime_tip_policy = evaluate_runtime_tip_trust(
            snapshot=snapshot,
            workflow=RUNTIME_TIP_POLICY_WORKFLOW,
            allow_lower_trust=bool(allow_debug_coil_as_tip),
        )
        if not runtime_tip_policy.allowed_for_workflow:
            reasons.extend(runtime_tip_policy.reasons or ["runtime_tip_policy_not_allowed"])
        if runtime_tip_policy.allowed_for_workflow and not runtime_tip_policy.thesis_trusted:
            reasons.append(f"runtime_tip_trust_{runtime_tip_policy.trust_label}")
        if snapshot.T_robot_tip is None:
            reasons.append("missing_T_robot_tip")
    return {
        "accepted": not reasons,
        "reason": "ok" if not reasons else "; ".join(reasons),
        "tracker_age_s": None if age is None else float(age),
        "tracker_frame_id": snapshot.last_frame_number,
        "tip_pose_status": snapshot.tip_pose_status,
        "runtime_tip_calibration_state": snapshot.runtime_tip_calibration_state,
        "runtime_tip_mode": getattr(snapshot, "runtime_tip_mode", None),
        "runtime_tip_trust_level": (
            evaluate_runtime_tip_trust(
                snapshot=snapshot,
                workflow=RUNTIME_TIP_POLICY_WORKFLOW,
                allow_lower_trust=bool(allow_debug_coil_as_tip),
            ).trust_label
        ),
        "tool_id": tool_key,
        "tool_present": bool(tool.present) if tool is not None else False,
        "tool_valid": tool.valid if tool is not None else None,
        "tool_tracking_state": tool.tracking_state if tool is not None else "missing",
    }


def _classify_repeatability_exception(exc: Exception) -> str | None:
    if isinstance(exc, ServoBusBusyError):
        return "owner_conflict" if str(exc.owner or "").strip() else "bus_contention"
    detail = str(exc).lower()
    if "present position is unavailable" in detail:
        return "missing_position"
    if "current telemetry is unavailable" in detail or "present current" in detail:
        return "missing_current"
    if "input voltage" in detail:
        return "missing_voltage"
    if "telemetry is stale" in detail or "telemetry freshness" in detail:
        return "stale_telemetry"
    if "did not respond to ping" in detail or "missing servo" in detail:
        return "servo_missing"
    if "telemetry" in detail or "read" in detail:
        return "telemetry_read_error"
    return None


def _classify_repeatability_telemetry(
    *,
    session: ExperimentSession,
    servo_ids: list[int],
    telemetry_by_id: dict[int, Any] | None,
    stage: str,
    read_error: Exception | None = None,
) -> dict[str, Any]:
    selected_ids = [int(value) for value in servo_ids]
    payload: dict[str, Any] = {}
    issues: list[dict[str, Any]] = []
    failure_code: str | None = None

    if read_error is not None:
        error_code = _classify_repeatability_exception(read_error)
        if error_code in {"owner_conflict", "bus_contention"}:
            failure_code = error_code
        else:
            failure_code = "telemetry_read_error"
        return {
            "ok": False,
            "failure_code": failure_code,
            "failure_stage": str(stage),
            "message": str(read_error),
            "issues": [
                {
                    "code": failure_code,
                    "detail": str(read_error),
                }
            ],
            "servo_ids": selected_ids,
            "by_servo": payload,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "freshness_threshold_s": float(session.context.servo_service.telemetry_freshness_threshold_s()),
        }

    telemetry_by_id = dict(telemetry_by_id or {})
    freshness_threshold_s = float(session.context.servo_service.telemetry_freshness_threshold_s())
    for servo_id in selected_ids:
        telemetry = telemetry_by_id.get(int(servo_id))
        if telemetry is None:
            issues.append(
                {
                    "servo_id": int(servo_id),
                    "code": "servo_missing",
                    "detail": "No telemetry payload returned for configured servo ID.",
                }
            )
            if failure_code is None:
                failure_code = "servo_missing"
            payload[str(servo_id)] = {
                "present_position_ticks": None,
                "present_current_ma": None,
                "torque_enabled": None,
                "operating_mode": None,
                "hardware_error": "telemetry_missing",
                "present_voltage_mv": None,
                "present_temperature_c": None,
                "telemetry_error": "telemetry_missing",
                "telemetry_age_s": None,
                "telemetry_fresh": None,
            }
            continue

        telemetry_age_s = session.context.servo_service.telemetry_age_s(telemetry)
        telemetry_fresh = session.context.servo_service.telemetry_is_fresh(telemetry)
        payload[str(servo_id)] = {
            "present_position_ticks": telemetry.present_position,
            "present_current_ma": telemetry.present_current_ma,
            "torque_enabled": telemetry.torque_enabled,
            "operating_mode": telemetry.operating_mode,
            "hardware_error": telemetry.hardware_error,
            "present_voltage_mv": telemetry.present_voltage_mv,
            "present_temperature_c": telemetry.present_temperature_c,
            "telemetry_error": telemetry.telemetry_error,
            "telemetry_age_s": telemetry_age_s,
            "telemetry_fresh": telemetry_fresh,
        }
        missing_fields: list[str] = []
        if telemetry.present_position is None:
            missing_fields.append("position")
        if telemetry.present_current_ma is None:
            missing_fields.append("current")
        if telemetry.present_voltage_mv is None:
            missing_fields.append("voltage")
        if telemetry_fresh is False:
            issues.append(
                {
                    "servo_id": int(servo_id),
                    "code": "stale_telemetry",
                    "detail": (
                        f"Telemetry is stale (age={float(telemetry_age_s or 0.0):.3f}s, "
                        f"threshold={freshness_threshold_s:.3f}s)."
                    ),
                }
            )
            if failure_code is None:
                failure_code = "stale_telemetry"
        if missing_fields:
            if "position" in missing_fields:
                code = "missing_position"
            elif "current" in missing_fields:
                code = "missing_current"
            else:
                code = "missing_voltage"
            issues.append(
                {
                    "servo_id": int(servo_id),
                    "code": code,
                    "detail": f"Missing telemetry field(s): {', '.join(missing_fields)}.",
                }
            )
            if failure_code is None:
                failure_code = code
        if telemetry.telemetry_error:
            issues.append(
                {
                    "servo_id": int(servo_id),
                    "code": "telemetry_read_error",
                    "detail": str(telemetry.telemetry_error),
                }
            )
            if failure_code is None:
                failure_code = "telemetry_read_error"

    if failure_code is None:
        message = "Servo telemetry is valid."
        ok = True
    else:
        issue_text = "; ".join(
            f"servo {entry.get('servo_id', 'n/a')}: {entry.get('code')} ({entry.get('detail', '')})"
            if "servo_id" in entry
            else f"{entry.get('code')} ({entry.get('detail', '')})"
            for entry in issues
        )
        message = issue_text or "Servo telemetry is invalid."
        ok = False
    return {
        "ok": bool(ok),
        "failure_code": failure_code,
        "failure_stage": str(stage),
        "message": str(message),
        "issues": issues,
        "servo_ids": selected_ids,
        "by_servo": payload,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "freshness_threshold_s": freshness_threshold_s,
    }


def _read_servo_telemetry_payload(
    session: ExperimentSession,
    servo_ids: list[int],
    *,
    stage: str,
) -> dict[str, Any]:
    try:
        telemetry_by_id = session.context.servo_service.read_live_telemetry([int(value) for value in servo_ids])
    except Exception as exc:
        return _classify_repeatability_telemetry(
            session=session,
            servo_ids=servo_ids,
            telemetry_by_id=None,
            stage=stage,
            read_error=exc,
        )
    return _classify_repeatability_telemetry(
        session=session,
        servo_ids=servo_ids,
        telemetry_by_id=telemetry_by_id,
        stage=stage,
    )


def _extract_repeatability_position(
    sample: ExperimentTimeseriesSample,
    *,
    tool_id: str,
    require_robot_frame_tip: bool,
) -> tuple[list[float] | None, str | None]:
    tip_payload = sample.pose_in_robot_frame.get("tip", {})
    tip_position = tip_payload.get("translation_mm")
    if isinstance(tip_position, list) and len(tip_position) == 3:
        return [float(value) for value in tip_position], "robot"
    if require_robot_frame_tip:
        return None, None
    tool_payload = sample.pose_in_tracker_frame.get(str(tool_id).upper(), {})
    tracker_position = tool_payload.get("translation_mm")
    if isinstance(tracker_position, list) and len(tracker_position) == 3:
        return [float(value) for value in tracker_position], "tracker"
    return None, None


def _matrix_translation_mm(matrix: Any) -> list[float] | None:
    if not isinstance(matrix, list) or len(matrix) < 3:
        return None
    try:
        return [float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3])]
    except (IndexError, TypeError, ValueError):
        return None


def _raw_tracker_tool_pose(snapshot, tool_id: str) -> dict[str, Any]:
    tool_key = str(tool_id or "0A").upper()
    tool = getattr(snapshot, "tools", {}).get(tool_key)
    if tool is None:
        return {"tool_id": tool_key, "present": False, "valid": None, "tracking_state": "missing"}
    return {
        "tool_id": tool_key,
        "present": bool(getattr(tool, "present", False)),
        "valid": getattr(tool, "valid", None),
        "tracking_state": str(getattr(tool, "tracking_state", "unknown") or "unknown"),
        "translation_mm": (
            [float(value) for value in getattr(tool, "translation_mm")]
            if getattr(tool, "translation_mm", None) is not None
            else None
        ),
        "quaternion_wxyz": (
            [float(value) for value in getattr(tool, "quaternion_wxyz")]
            if getattr(tool, "quaternion_wxyz", None) is not None
            else None
        ),
        "frame_number": getattr(tool, "frame_number", None),
    }


def _compute_group_metrics(per_target: dict[str, Any]) -> dict[str, Any]:
    group_values: dict[str, dict[str, list[float]]] = {
        "ring": {},
        "axis_class": {},
        "magnitude_class": {},
    }
    for target_metrics in per_target.values():
        rms = target_metrics.get("spread_rms_mm")
        if rms is None:
            continue
        ring = str(target_metrics.get("ring", "unknown"))
        group_values["ring"].setdefault(ring, []).append(float(rms))
        tags = set(str(value) for value in (target_metrics.get("group_tags") or []))
        if "on_axis" in tags or "off_axis" in tags:
            axis = "on_axis" if "on_axis" in tags else "off_axis"
            group_values["axis_class"].setdefault(axis, []).append(float(rms))
        if ring in {"inner", "outer"}:
            magnitude = "low" if ring == "inner" else "high"
            group_values["magnitude_class"].setdefault(magnitude, []).append(float(rms))
    metrics: dict[str, Any] = {}
    for group_name, values_by_label in group_values.items():
        metrics[group_name] = {
            label: {
                "target_count": int(len(values)),
                "mean_target_rms_mm": float(np.mean(values)) if values else None,
                "max_target_rms_mm": float(np.max(values)) if values else None,
            }
            for label, values in sorted(values_by_label.items())
        }
    return metrics


def _assess_repeatability_run_validity(
    *,
    per_target: dict[str, Any],
    targets: list[LegacyRepeatabilityTarget],
    valid_repeat_sample_count: int,
    rejected_capture_count: int,
    min_repeat_captures_per_target: int,
    min_repeat_capture_fraction: float,
    max_rejected_capture_fraction: float,
) -> dict[str, Any]:
    planned_repeat_capture_count = int(LEGACY_VISIT_COUNT)
    planned_total_capture_count = int(LEGACY_CAPTURE_COUNT)
    min_repeat_captures_per_target = max(1, int(min_repeat_captures_per_target))
    min_repeat_capture_fraction = max(0.0, min(1.0, float(min_repeat_capture_fraction)))
    max_rejected_capture_fraction = max(0.0, min(1.0, float(max_rejected_capture_fraction)))
    min_required_repeat_capture_count = int(
        math.ceil(float(planned_repeat_capture_count) * float(min_repeat_capture_fraction))
    )
    accepted_repeat_fraction = (
        float(valid_repeat_sample_count) / float(planned_repeat_capture_count)
        if planned_repeat_capture_count > 0
        else 0.0
    )
    rejected_capture_fraction = (
        float(rejected_capture_count) / float(planned_total_capture_count)
        if planned_total_capture_count > 0
        else 0.0
    )
    targets_below_min_count: list[dict[str, Any]] = []
    targets_missing_repeat_coverage: list[dict[str, Any]] = []
    ring_coverage: dict[str, dict[str, Any]] = {}
    for target in targets:
        row = dict(per_target.get(str(target.target_index), {}) or {})
        repeat_count = int(row.get("repeat_sample_count", 0) or 0)
        if repeat_count < min_repeat_captures_per_target:
            targets_below_min_count.append(
                {
                    "target_index": int(target.target_index),
                    "label": target.label,
                    "ring": target.ring,
                    "repeat_sample_count": repeat_count,
                }
            )
        if repeat_count <= 0:
            targets_missing_repeat_coverage.append(
                {
                    "target_index": int(target.target_index),
                    "label": target.label,
                    "ring": target.ring,
                }
            )
        ring_row = ring_coverage.setdefault(
            str(target.ring),
            {
                "expected_target_count": 0,
                "targets_with_any_repeat": 0,
                "targets_meeting_min_repeat_count": 0,
                "accepted_repeat_capture_count": 0,
            },
        )
        ring_row["expected_target_count"] += 1
        ring_row["accepted_repeat_capture_count"] += repeat_count
        if repeat_count > 0:
            ring_row["targets_with_any_repeat"] += 1
        if repeat_count >= min_repeat_captures_per_target:
            ring_row["targets_meeting_min_repeat_count"] += 1
    critical_missing_target_groups = [
        ring
        for ring, row in sorted(ring_coverage.items())
        if int(row.get("targets_with_any_repeat", 0) or 0) <= 0
    ]
    failure_reasons: list[str] = []
    if valid_repeat_sample_count < min_required_repeat_capture_count:
        failure_reasons.append(
            f"accepted repeat coverage {valid_repeat_sample_count}/{planned_repeat_capture_count} is below "
            f"the required minimum {min_required_repeat_capture_count}/{planned_repeat_capture_count}"
        )
    if targets_below_min_count:
        failure_reasons.append(
            "per-target repeat coverage below minimum on "
            + ", ".join(
                f"{row['label']}={int(row['repeat_sample_count'])}"
                for row in targets_below_min_count
            )
        )
    if rejected_capture_fraction > max_rejected_capture_fraction:
        failure_reasons.append(
            f"rejected capture rate {rejected_capture_fraction:.3f} exceeds maximum {max_rejected_capture_fraction:.3f}"
        )
    if critical_missing_target_groups:
        failure_reasons.append(
            "critical target group coverage missing for " + ", ".join(critical_missing_target_groups)
        )
    warning_reasons: list[str] = []
    if valid_repeat_sample_count < planned_repeat_capture_count:
        warning_reasons.append(
            f"accepted repeat coverage is partial at {valid_repeat_sample_count}/{planned_repeat_capture_count}"
        )
    if rejected_capture_count > 0:
        warning_reasons.append(
            f"{rejected_capture_count} capture(s) were rejected by tracker freshness/validity gating"
        )
    if targets_below_min_count and not failure_reasons:
        warning_reasons.append("some targets met the minimum threshold but did not reach full 16-repeat coverage")
    return {
        "thesis_valid_run": not failure_reasons,
        "criteria": {
            "planned_repeat_capture_count": planned_repeat_capture_count,
            "planned_total_capture_count": planned_total_capture_count,
            "min_repeat_capture_fraction": float(min_repeat_capture_fraction),
            "min_required_repeat_capture_count": min_required_repeat_capture_count,
            "min_repeat_captures_per_target": min_repeat_captures_per_target,
            "max_rejected_capture_fraction": float(max_rejected_capture_fraction),
            "require_all_target_groups": True,
        },
        "observed": {
            "valid_repeat_sample_count": int(valid_repeat_sample_count),
            "accepted_repeat_fraction": float(accepted_repeat_fraction),
            "rejected_capture_count": int(rejected_capture_count),
            "rejected_capture_fraction": float(rejected_capture_fraction),
            "targets_below_min_count": int(len(targets_below_min_count)),
            "targets_missing_repeat_coverage_count": int(len(targets_missing_repeat_coverage)),
            "critical_missing_target_groups": list(critical_missing_target_groups),
            "ring_coverage": ring_coverage,
        },
        "targets_below_min_repeat_count": targets_below_min_count,
        "targets_missing_repeat_coverage": targets_missing_repeat_coverage,
        "critical_missing_target_groups": list(critical_missing_target_groups),
        "failure_reasons": failure_reasons,
        "warning_reasons": warning_reasons,
    }


def _progress_payload(*, visit: LegacyRepeatabilityVisit, phase: str, sample_index: int) -> dict[str, Any]:
    return {
        "phase": phase,
        "sample_index": int(sample_index),
        "current_target": int(visit.target_index),
        "source_target": int(visit.approach_index),
        "revisit_index": int(visit.revisit_index),
    }


def _safe_rms(values: list[float]) -> float | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.square(array))))


def _position_frame_label(frames: set[str], *, require_robot_frame_tip: bool) -> str:
    if not frames:
        return "robot" if require_robot_frame_tip else "unknown"
    if frames == {"robot"}:
        return "robot"
    if frames == {"tracker"}:
        return "tracker"
    return "mixed"


def _clean_float(value: float) -> float:
    numeric = float(value)
    return 0.0 if abs(numeric) < 1e-12 else numeric


def _file_provenance(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False}
    file_path = Path(path)
    info: dict[str, Any] = {
        "path": str(file_path),
        "exists": bool(file_path.exists()),
    }
    if not file_path.exists():
        return info
    stat = file_path.stat()
    info["size_bytes"] = int(stat.st_size)
    info["modified_at_utc"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    info["sha256"] = _file_sha256(file_path)
    return info


def _file_sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _resolve_repo_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return Path(project_root) / path


def _comparison_row(current_value: Any, baseline_value: Any) -> dict[str, float | None]:
    current = _as_optional_float(current_value)
    baseline = _as_optional_float(baseline_value)
    delta = _delta(current, baseline)
    percent = None
    if delta is not None and baseline not in (None, 0.0):
        percent = float((delta / float(baseline)) * 100.0)
    return {
        "current": current,
        "baseline": baseline,
        "delta": delta,
        "percent_delta": percent,
    }


def _compare_group_metrics(current_groups: dict[str, Any], baseline_groups: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group_name in sorted(set(current_groups) | set(baseline_groups)):
        current_rows = dict(current_groups.get(group_name, {}) or {})
        baseline_rows = dict(baseline_groups.get(group_name, {}) or {})
        result[group_name] = {}
        for label in sorted(set(current_rows) | set(baseline_rows)):
            c_value = (current_rows.get(label, {}) or {}).get("mean_target_rms_mm")
            b_value = (baseline_rows.get(label, {}) or {}).get("mean_target_rms_mm")
            result[group_name][label] = _comparison_row(c_value, b_value)
    return result


def _delta(current_value: Any, baseline_value: Any) -> float | None:
    current = _as_optional_float(current_value)
    baseline = _as_optional_float(baseline_value)
    if current is None or baseline is None:
        return None
    return float(current - baseline)


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
