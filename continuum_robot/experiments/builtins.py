"""Built-in canonical experiments."""

from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import re
import shutil
import threading
import time
from typing import Any

from continuum_robot.data.model_training_validity import NON_TRAINING_PHASES, sample_has_complete_command_servo_tip
from continuum_robot.experiments.dataset_io import ExperimentDatasetLoader, ExperimentDatasetWriter
from continuum_robot.experiments.experiment_models import ExperimentPoint
from continuum_robot.experiments.framework import BaseExperiment, ExperimentHardwareRequirements, ExperimentSession
from continuum_robot.experiments.critical_experiments import register_critical_experiments
from continuum_robot.experiments.calibration_validation import register_calibration_validation_experiments
from continuum_robot.experiments.registration_trial import register_registration_trial_experiment
from continuum_robot.experiments.modeling_dataset_outputs import write_modeling_dataset_outputs
from continuum_robot.registration.legacy_compat import average_quaternions
from continuum_robot.tracking.transforms import quat_wxyz_to_rotmat, rotmat_to_quat_wxyz
from continuum_robot.experiments.pretension_validation_outputs import write_pretension_validation_outputs
from continuum_robot.experiments.penprobe_chasing_demo import PenprobeChasingDemoExperiment
from continuum_robot.experiments.registration_sampling_study import (
    RegistrationSamplingStudyExperiment,
)
from continuum_robot.experiments.servo_tracker_sync_outputs import write_servo_tracker_sync_outputs
from continuum_robot.experiments.single_segment_repeatability import register_single_segment_repeatability
from continuum_robot.experiments.workspace_repeatability_map import register_workspace_repeatability_map
from continuum_robot.experiments.tracker_timing_outputs import write_tracker_timing_outputs
from continuum_robot.experiments.two_segment_startup_validation import (
    TwoSegmentStartupValidationExperiment,
)
from continuum_robot.experiments.two_segment_collect_pose_dataset import (
    TwoSegmentCollectPoseCommandDatasetExperiment,
)
from continuum_robot.experiments.two_segment_repeatability import (
    TwoSegmentRepeatabilityExperiment,
)
from continuum_robot.experiments.schedules import (
    CommandScheduleConfig,
    command_schedule_checksum,
    generate_command_schedule,
)
from continuum_robot.experiments.sample_builders import sample_from_tracking_snapshot
from continuum_robot.experiments.schemas import ExperimentMetadata, ExperimentSummary, ExperimentTimeseriesSample
from continuum_robot.servos.segment_readiness import evaluate_selected_segment_readiness
from continuum_robot.servos.servo_service import PretensionParameters, ServoTelemetryRetryError
from continuum_robot.tracking.timing_benchmark import (
    compute_servo_tracker_sync_summary,
    compute_servo_sync_summary,
    extract_servo_command_records,
    compute_tracker_timing_summary,
    extract_servo_timing_records,
    extract_tracker_timing_records,
)
from continuum_robot.tracking.runtime_tip_policy import (
    WORKFLOW_MODELING_DATASET,
    WORKFLOW_PRETENSION_VALIDATION,
    evaluate_runtime_tip_trust,
)

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is a declared dependency
    np = None


LOG = logging.getLogger(__name__)


@dataclass
class TrackerPipelineMockConfig:
    """Config for the tracker pipeline mock experiment."""

    sample_count: int = 10
    sample_period_s: float = 0.05

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "TrackerPipelineMockConfig":
        payload = dict(payload or {})
        return cls(
            sample_count=int(payload.get("sample_count", 10)),
            sample_period_s=float(payload.get("sample_period_s", 0.05)),
        )


@dataclass
class TransformChainValidationConfig:
    """Config for transform chain validation."""

    robot_translation_mm: list[float] = field(default_factory=lambda: [10.0, 0.0, 0.0])
    tracker_translation_mm: list[float] = field(default_factory=lambda: [0.0, 20.0, 0.0])
    tip_offset_mm: list[float] = field(default_factory=lambda: [5.0, 0.0, 15.0])

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "TransformChainValidationConfig":
        payload = dict(payload or {})
        return cls(
            robot_translation_mm=[float(value) for value in payload.get("robot_translation_mm", [10.0, 0.0, 0.0])],
            tracker_translation_mm=[float(value) for value in payload.get("tracker_translation_mm", [0.0, 20.0, 0.0])],
            tip_offset_mm=[float(value) for value in payload.get("tip_offset_mm", [5.0, 0.0, 15.0])],
        )


@dataclass
class CommandScheduleValidationConfig:
    """Config for command schedule validation."""

    schedule: CommandScheduleConfig = field(default_factory=CommandScheduleConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CommandScheduleValidationConfig":
        payload = dict(payload or {})
        return cls(schedule=CommandScheduleConfig.from_dict(payload.get("schedule")))


@dataclass
class DatasetSchemaRoundtripConfig:
    """Config for dataset schema roundtrip validation."""

    sample_count: int = 3

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "DatasetSchemaRoundtripConfig":
        payload = dict(payload or {})
        return cls(sample_count=int(payload.get("sample_count", 3)))


@dataclass
class ReplayRunnerConfig:
    """Config for replaying a previously written dataset."""

    dataset_path: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ReplayRunnerConfig":
        payload = dict(payload or {})
        return cls(dataset_path=str(payload.get("dataset_path", "")))


@dataclass
class CollectPoseCommandDatasetConfig:
    """Config for the authoritative Motor Babble modeling dataset collector."""

    dataset_mode: str = "workspace_coverage"
    dry_run: bool = False
    sample_count_target: int = 120
    samples_per_command: int = 1
    # Multi-frame post-settle averaging knobs. When tracker_samples_per_command > 1
    # the per-command capture changes from "take one snapshot after settle" to
    # "after one settle, collect N deduplicated tracker frames and emit both a
    # first-frame label row and an averaged-frame label row". Existing
    # samples_per_command behaviour is unchanged (it still emits N separate
    # rows from N separate cached-snapshot reads). The two knobs are mutually
    # exclusive — setting both > 1 fails the experiment precheck.
    tracker_samples_per_command: int = 1
    # Per-frame patience budget. Each new fresh frame must arrive within this
    # many seconds of the previous one; total wall time scales as N * budget.
    # Hard-fail (per the user's spec) only if a single frame exceeds this.
    tracker_per_frame_max_wait_s: float = 1.0
    # Optional inner-loop poll interval used while waiting for a new frame_id;
    # None means reuse capture_poll_interval_s.
    tracker_sample_period_s: float | None = None
    # When None: auto-true iff tracker_samples_per_command > 1.
    averaged_label_enabled: bool | None = None
    export_first_sample_label: bool = True
    # When None: auto-true iff averaged_label_enabled is on.
    export_averaged_sample_label: bool | None = None
    settle_time_s: float = 0.15
    max_tracker_age_s: float = 0.15
    capture_timeout_s: float = 1.0
    capture_poll_interval_s: float = 0.02
    tool_id: str = "0A"
    require_robot_frame_tip: bool = True
    workspace_amplitude_cm: float = 1.0
    envelope_utilization: float = 0.75
    quasi_random: bool = True
    random_seed: int = 0
    hysteresis_target_count: int = 8
    hysteresis_cycle_count: int = 2
    hysteresis_prior_family_count: int = 4
    repeatability_block_count: int = 3
    # angular_test_mesh mode (Wolfe MS thesis §3.2.3 p85). Produces a θ × φ grid where
    # the cable pair commands trace circles of constant bending angle at evenly spaced
    # plane-rotation angles. With theta_count=12, phi_count=24, samples_per_command=5,
    # the schedule is the 288-position × 5-rep mesh Wolfe used to compare models.
    test_mesh_theta_count: int = 12
    test_mesh_phi_count: int = 24
    test_mesh_amplitude_cm: float | None = None  # None ⇒ workspace_amplitude_cm
    allow_lower_trust_runtime_tip: bool = False
    allow_lower_trust_pretension: bool = False
    allow_no_tracker_test_run: bool = False
    run_trust_mode: str = "thesis_trusted"
    export_legacy_dat: bool = True
    run_label: str = ""
    dataset_tag: str = ""
    post_write_settle_s: float = 0.0
    telemetry_retry_count: int = 2
    telemetry_retry_delay_s: float = 0.03
    allow_recovered_packet_errors: bool = True
    max_recovered_packet_errors_per_run: int | None = None
    max_current_warning_ma: int | None = 500
    current_warning_ma: int | None = 500
    transient_current_spike_ma: int | None = None
    sustained_jam_current_ma: int | None = None
    sustained_jam_cycles: int = 3
    transient_spike_policy: str = "warn_drop_sample_continue"
    sustained_jam_policy: str = "stop_safely"
    current_spike_resync_enabled: bool = True
    current_spike_cooldown_s: float = 0.25
    current_spike_return_to_previous_safe_goal: bool = False
    current_spike_max_events_per_servo: int = 6
    continue_until_valid_samples: bool = False
    target_valid_sample_count: int | None = None
    max_total_attempts: int | None = None
    max_dropped_fraction: float | None = None
    max_consecutive_failures: int | None = None
    long_run_recovery_enabled: bool = False
    transport_burst_recovery_enabled: bool = False
    transport_burst_cooldown_s: float = 0.35
    transport_burst_resync_attempts: int = 5
    transport_burst_resync_delay_s: float = 0.1
    retry_same_command_after_resync: bool = False
    max_consecutive_transport_bursts: int = 3
    max_total_dropped_samples_fraction: float | None = 0.05
    max_consecutive_packet_failures: int = 3
    max_total_packet_failures: int | None = None
    on_unrecovered_post_motion_telemetry: str = "fail_run"
    resync_read_attempts: int = 3
    resync_delay_s: float = 0.05
    return_to_neutral_on_resync_failure: bool = False
    command_transition_ramp_enabled: bool = False
    max_delta_cm_per_transition: float | None = None
    max_delta_cm_per_ramp_step: float | None = None
    ramp_step_settle_s: float = 0.0
    ramp_include_telemetry_checks: bool = True
    ramp_log_intermediate_telemetry: bool = False
    profile_velocity_ticks_per_s: int | None = None
    profile_acceleration: int | None = None
    chunk_flush_every_n_commands: int | None = None
    resume_from_command_index: int = 0
    long_run_health_write_interval_samples: int = 100
    goal_write_retry_attempts: int | None = None
    allow_partial_dataset_after_telemetry_drops: bool = False
    legacy_schedule_override: bool = False
    command_points: list[dict[str, Any]] = field(default_factory=list)
    command_schedule: CommandScheduleConfig = field(
        default_factory=lambda: CommandScheduleConfig(kind="babble", dimensions=4, amplitude_cm=1.0, babble_count=120)
    )

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CollectPoseCommandDatasetConfig":
        payload = dict(payload or {})
        schedule_payload = payload.get("command_schedule") or payload.get("schedule")
        return cls(
            dataset_mode=str(payload.get("dataset_mode", "workspace_coverage") or "workspace_coverage").strip().lower(),
            dry_run=bool(payload.get("dry_run", False)),
            sample_count_target=max(1, int(payload.get("sample_count_target", 120))),
            samples_per_command=max(1, int(payload.get("samples_per_command", payload.get("sample_count_per_point", 1)))),
            tracker_samples_per_command=max(1, int(payload.get("tracker_samples_per_command", 1))),
            tracker_per_frame_max_wait_s=max(0.05, float(payload.get("tracker_per_frame_max_wait_s", 1.0))),
            tracker_sample_period_s=(
                None
                if payload.get("tracker_sample_period_s") in (None, "")
                else max(0.001, float(payload.get("tracker_sample_period_s")))
            ),
            averaged_label_enabled=(
                None
                if payload.get("averaged_label_enabled") in (None, "")
                else bool(payload.get("averaged_label_enabled"))
            ),
            export_first_sample_label=bool(payload.get("export_first_sample_label", True)),
            export_averaged_sample_label=(
                None
                if payload.get("export_averaged_sample_label") in (None, "")
                else bool(payload.get("export_averaged_sample_label"))
            ),
            settle_time_s=float(payload.get("settle_time_s", 0.15)),
            max_tracker_age_s=float(payload.get("max_tracker_age_s", 0.15)),
            capture_timeout_s=float(payload.get("capture_timeout_s", 1.0)),
            capture_poll_interval_s=float(payload.get("capture_poll_interval_s", 0.02)),
            tool_id=str(payload.get("tool_id", "0A") or "0A").strip().upper(),
            require_robot_frame_tip=bool(payload.get("require_robot_frame_tip", True)),
            workspace_amplitude_cm=max(0.0, float(payload.get("workspace_amplitude_cm", 1.0))),
            envelope_utilization=max(0.05, min(1.0, float(payload.get("envelope_utilization", 0.75)))),
            quasi_random=bool(payload.get("quasi_random", True)),
            random_seed=int(payload.get("random_seed", payload.get("seed", 0))),
            hysteresis_target_count=max(1, int(payload.get("hysteresis_target_count", 8))),
            hysteresis_cycle_count=max(1, int(payload.get("hysteresis_cycle_count", 2))),
            hysteresis_prior_family_count=max(2, int(payload.get("hysteresis_prior_family_count", 4))),
            repeatability_block_count=max(1, int(payload.get("repeatability_block_count", 3))),
            test_mesh_theta_count=max(2, int(payload.get("test_mesh_theta_count", 12))),
            test_mesh_phi_count=max(3, int(payload.get("test_mesh_phi_count", 24))),
            test_mesh_amplitude_cm=(
                float(payload["test_mesh_amplitude_cm"])
                if payload.get("test_mesh_amplitude_cm") not in (None, "")
                else None
            ),
            allow_lower_trust_runtime_tip=bool(payload.get("allow_lower_trust_runtime_tip", False)),
            allow_lower_trust_pretension=bool(payload.get("allow_lower_trust_pretension", False)),
            allow_no_tracker_test_run=bool(payload.get("allow_no_tracker_test_run", False)),
            run_trust_mode=str(payload.get("run_trust_mode", "thesis_trusted") or "thesis_trusted").strip().lower(),
            export_legacy_dat=bool(payload.get("export_legacy_dat", True)),
            run_label=str(payload.get("run_label", "") or ""),
            dataset_tag=str(payload.get("dataset_tag", "") or ""),
            post_write_settle_s=max(0.0, float(payload.get("post_write_settle_s", payload.get("post_command_settle_s", 0.0)))),
            telemetry_retry_count=max(0, int(payload.get("telemetry_retry_count", 2))),
            telemetry_retry_delay_s=max(0.0, float(payload.get("telemetry_retry_delay_s", 0.03))),
            allow_recovered_packet_errors=bool(payload.get("allow_recovered_packet_errors", True)),
            max_recovered_packet_errors_per_run=(
                None
                if payload.get("max_recovered_packet_errors_per_run") in (None, "")
                else max(0, int(payload.get("max_recovered_packet_errors_per_run")))
            ),
            long_run_recovery_enabled=bool(payload.get("long_run_recovery_enabled", False)),
            transport_burst_recovery_enabled=bool(
                payload.get(
                    "transport_burst_recovery_enabled",
                    payload.get("long_run_recovery_enabled", False),
                )
            ),
            transport_burst_cooldown_s=max(0.0, float(payload.get("transport_burst_cooldown_s", 0.35))),
            transport_burst_resync_attempts=max(
                1,
                int(payload.get("transport_burst_resync_attempts", payload.get("resync_read_attempts", 5))),
            ),
            transport_burst_resync_delay_s=max(
                0.0,
                float(payload.get("transport_burst_resync_delay_s", payload.get("resync_delay_s", 0.1))),
            ),
            retry_same_command_after_resync=bool(payload.get("retry_same_command_after_resync", False)),
            max_consecutive_transport_bursts=max(
                1,
                int(payload.get("max_consecutive_transport_bursts", payload.get("max_consecutive_packet_failures", 3))),
            ),
            max_total_dropped_samples_fraction=(
                None
                if payload.get("max_total_dropped_samples_fraction") in (None, "")
                else max(0.0, float(payload.get("max_total_dropped_samples_fraction")))
            ),
            max_consecutive_packet_failures=max(1, int(payload.get("max_consecutive_packet_failures", 3))),
            max_total_packet_failures=(
                None
                if payload.get("max_total_packet_failures") in (None, "")
                else max(0, int(payload.get("max_total_packet_failures")))
            ),
            on_unrecovered_post_motion_telemetry=str(
                payload.get("on_unrecovered_post_motion_telemetry", "fail_run") or "fail_run"
            ).strip().lower(),
            resync_read_attempts=max(1, int(payload.get("resync_read_attempts", 3))),
            resync_delay_s=max(0.0, float(payload.get("resync_delay_s", 0.05))),
            return_to_neutral_on_resync_failure=bool(payload.get("return_to_neutral_on_resync_failure", False)),
            command_transition_ramp_enabled=bool(payload.get("command_transition_ramp_enabled", False)),
            max_delta_cm_per_transition=(
                None
                if (
                    payload.get("max_delta_cm_per_transition") in (None, "")
                    and payload.get("max_delta_cm_per_ramp_step") in (None, "")
                )
                else max(
                    0.0,
                    float(
                        payload.get(
                            "max_delta_cm_per_ramp_step",
                            payload.get("max_delta_cm_per_transition"),
                        )
                    ),
                )
            ),
            max_delta_cm_per_ramp_step=(
                None
                if (
                    payload.get("max_delta_cm_per_ramp_step") in (None, "")
                    and payload.get("max_delta_cm_per_transition") in (None, "")
                )
                else max(
                    0.0,
                    float(
                        payload.get(
                            "max_delta_cm_per_ramp_step",
                            payload.get("max_delta_cm_per_transition"),
                        )
                    ),
                )
            ),
            ramp_step_settle_s=max(0.0, float(payload.get("ramp_step_settle_s", 0.0))),
            ramp_include_telemetry_checks=bool(payload.get("ramp_include_telemetry_checks", True)),
            ramp_log_intermediate_telemetry=bool(payload.get("ramp_log_intermediate_telemetry", False)),
            profile_velocity_ticks_per_s=(
                None
                if payload.get("profile_velocity_ticks_per_s") in (None, "")
                else max(0, int(payload.get("profile_velocity_ticks_per_s")))
            ),
            profile_acceleration=(
                None
                if payload.get("profile_acceleration") in (None, "")
                else max(0, int(payload.get("profile_acceleration")))
            ),
            chunk_flush_every_n_commands=(
                None
                if payload.get("chunk_flush_every_n_commands") in (None, "")
                else max(1, int(payload.get("chunk_flush_every_n_commands")))
            ),
            resume_from_command_index=max(0, int(payload.get("resume_from_command_index", 0))),
            long_run_health_write_interval_samples=max(1, int(payload.get("long_run_health_write_interval_samples", 100))),
            goal_write_retry_attempts=(
                None
                if payload.get("goal_write_retry_attempts") in (None, "")
                else max(1, int(payload.get("goal_write_retry_attempts")))
            ),
            allow_partial_dataset_after_telemetry_drops=bool(
                payload.get("allow_partial_dataset_after_telemetry_drops", False)
            ),
            max_current_warning_ma=(
                max(0, int(payload.get("max_current_warning_ma", 500)))
                if payload.get("max_current_warning_ma", 500) not in (None, "")
                else None
            ),
            current_warning_ma=(
                max(0, int(payload.get("current_warning_ma", payload.get("max_current_warning_ma", 500))))
                if payload.get("current_warning_ma", payload.get("max_current_warning_ma", 500)) not in (None, "")
                else None
            ),
            transient_current_spike_ma=(
                None
                if payload.get("transient_current_spike_ma") in (None, "")
                else int(payload.get("transient_current_spike_ma"))
            ),
            sustained_jam_current_ma=(
                None
                if payload.get("sustained_jam_current_ma") in (None, "")
                else int(payload.get("sustained_jam_current_ma"))
            ),
            sustained_jam_cycles=max(1, int(payload.get("sustained_jam_cycles", 3))),
            transient_spike_policy=str(
                payload.get("transient_spike_policy", "warn_drop_sample_continue") or "warn_drop_sample_continue"
            ).strip().lower(),
            sustained_jam_policy=str(
                payload.get("sustained_jam_policy", "stop_safely") or "stop_safely"
            ).strip().lower(),
            current_spike_resync_enabled=bool(payload.get("current_spike_resync_enabled", True)),
            current_spike_cooldown_s=max(0.0, float(payload.get("current_spike_cooldown_s", 0.25))),
            current_spike_return_to_previous_safe_goal=bool(
                payload.get("current_spike_return_to_previous_safe_goal", False)
            ),
            current_spike_max_events_per_servo=max(1, int(payload.get("current_spike_max_events_per_servo", 6))),
            continue_until_valid_samples=bool(payload.get("continue_until_valid_samples", False)),
            target_valid_sample_count=(
                None
                if payload.get("target_valid_sample_count") in (None, "")
                else max(1, int(payload.get("target_valid_sample_count")))
            ),
            max_total_attempts=(
                None
                if payload.get("max_total_attempts") in (None, "")
                else max(1, int(payload.get("max_total_attempts")))
            ),
            max_dropped_fraction=(
                None
                if payload.get("max_dropped_fraction") in (None, "")
                else max(0.0, float(payload.get("max_dropped_fraction")))
            ),
            max_consecutive_failures=(
                None
                if payload.get("max_consecutive_failures") in (None, "")
                else max(1, int(payload.get("max_consecutive_failures")))
            ),
            legacy_schedule_override=bool(
                ("command_schedule" in payload)
                or ("schedule" in payload)
                or bool(payload.get("command_points"))
            ),
            command_points=[
                {
                    "index": int(point.get("index", index)),
                    "tendon_displacement_cm": [float(value) for value in point.get("tendon_displacement_cm", [])],
                    "settle_time_s": (
                        float(point["settle_time_s"])
                        if point.get("settle_time_s") not in (None, "")
                        else None
                    ),
                    "repeat": int(point.get("repeat", 1)),
                    "label": str(point.get("label", "")),
                }
                for index, point in enumerate(payload.get("command_points", []) or [])
            ],
            command_schedule=CommandScheduleConfig.from_dict(schedule_payload),
        )


@dataclass
class ModelingCommandStep:
    """One ordered single-segment modeling command/capture step."""

    index: int
    phase: str
    label: str
    pair_command_cm: list[float]
    cable_command_cm: list[float]
    settle_time_s: float
    block_index: int | None = None
    prior_family: str | None = None
    previous_pair_command_cm: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


PRETENSION_TIP_VARIANT_CURRENT_ONLY = "current_only"
PRETENSION_TIP_VARIANT_PAIRED_THEN_TIP = "paired_then_tip"
PRETENSION_TIP_VARIANT_JACOBIAN_LEARNED_TIP = "jacobian_learned_tip"
PRETENSION_TIP_VARIANT_OPTIONS = (
    PRETENSION_TIP_VARIANT_CURRENT_ONLY,
    PRETENSION_TIP_VARIANT_PAIRED_THEN_TIP,
    PRETENSION_TIP_VARIANT_JACOBIAN_LEARNED_TIP,
)


@dataclass
class PretensionValidationExperimentConfig:
    """Config for one-servo pretension response validation."""

    mode: str = "single_servo_trace"
    staged_strategy: str = "conservative_startup"
    allow_legacy_staged_strategy: bool = False
    servo_id: int = 1
    servo_ids: list[int] = field(default_factory=list)
    repeat_runs: int = 1
    move_to_reference: bool = True
    include_tracker_displacement: bool = True
    allow_current_only_when_tracker_missing: bool = False
    allow_no_tracker_test_run: bool = False
    run_trust_mode: str = "thesis_trusted"
    enable_tip_centering: bool = True
    tip_center_tolerance_mm: float = 1.0
    tip_center_max_iterations: int = 16
    tip_center_step_ticks: int = 10
    tip_center_x_sign: int = 1
    tip_center_y_sign: int = 1
    tip_target_xy_mm: list[float] = field(default_factory=lambda: [0.0, 0.0])
    max_tip_displacement_mm: float = 50.0
    tip_divergence_stop_mm: float = 2.0
    equalization_step_ticks: int = 2
    equalization_max_iterations: int = 24
    conservative_max_iterations: int = 24
    conservative_step_ticks: int = 10
    conservative_max_cumulative_travel_ticks: int = 80
    coarse_slack_step_ticks: int = 50
    soft_release_step_ticks: int = 50
    soft_release_max_travel_ticks: int = 40
    soft_release_current_target_ma: float = 10.0
    takeup_target_load_proxy_ma: float = 25.0
    high_load_proxy_ma: float = 40.0
    takeup_max_iterations: int = 16
    staged_packet_retry_budget: int = 3
    characterization_step_ticks: int = 1
    characterization_pair_cycles: int = 1
    current_characterization_sample_count: int = 20
    current_noise_multiplier: float = 3.0
    min_meaningful_current_delta_ma: float = 2.0
    load_balance_tolerance_ma: float = 10.0
    pair_balance_tolerance_ma: float = 10.0
    settle_verify_time_s: float = 0.2
    accept_max_load_balance_error_ma: float = 30.0
    accept_max_pair_balance_error_ma: float = 30.0
    accept_max_final_tip_xy_offset_mm: float = 3.0
    max_runtime_s: float = 60.0
    tracker_tool_id: str = "0A"
    pretension_start_mode: str | None = None
    untensioned_reference_tick: int | None = None
    step_ticks: int | None = None
    settle_time_s: float | None = None
    baseline_sample_count: int | None = None
    current_filter_window: int | None = None
    current_delta_threshold_ma: int | None = None
    absolute_trigger_current_ma: int | None = None
    hard_current_stop_ma: int | None = None
    max_travel_ticks: int | None = None
    timeout_s: float | None = None
    # --- Engagement-scan take-up knobs --------------------------------------
    # The take-up phase now uses a 3-step pattern (engagement scan + back-off
    # + fine take-up). See _run_symmetric_paired_takeup for the algorithm.
    engagement_scan_step_ticks: int = 50
    """Coarse inward step (ticks) during the engagement scan. All four servos
    step inward together by this amount each iteration until each engages."""
    engagement_rise_threshold_ma: float = 5.0
    """A servo is declared ENGAGED when its filtered current rises above its
    baseline by this many mA. The effective threshold is
    max(this, min_meaningful_current_delta_ma) so it never sits inside the
    measured noise floor."""
    engagement_back_off_ticks: int = 50
    """After every servo engages, back each one off by this many ticks before
    the fine phase. A small slack margin keeps the fine phase from immediately
    overshooting on the very next inward step."""
    # --- Tip-centering variant + Jacobian-learning knobs --------------------
    tip_centering_variant: str = PRETENSION_TIP_VARIANT_PAIRED_THEN_TIP
    """Which post-takeup tip-centering routine to run.

    - ``current_only``: stop after the symmetric paired take-up reaches the
      target band. No tip feedback. Tip XY centering is whatever the symmetric
      paired motion produces.
    - ``paired_then_tip``: existing conservative pair-stepping with sign-flip
      recovery. Robust but slow when the tip-vs-pair sign is unknown.
    - ``jacobian_learned_tip``: probe each pair ``jacobian_probe_step_ticks``
      ticks, measure tip XY delta, build a 2x2 Jacobian, then take an inverse-
      Jacobian step toward the target XY. Fewer iterations on average; needs a
      modestly accurate tracker.
    """
    jacobian_probe_step_ticks: int = 30
    """How many ticks to probe each pair during Jacobian construction."""
    jacobian_min_observable_tip_delta_mm: float = 0.20
    """Minimum tip displacement (mm) required for a Jacobian probe to be trusted.
    If a probe moves the tip by less than this, the Jacobian column is treated
    as unreliable and the variant falls back to ``paired_then_tip``."""
    jacobian_step_gain: float = 0.8
    """Inverse-Jacobian step gain. 1.0 = move full predicted delta in one step;
    0.8 leaves margin for nonlinearity."""
    jacobian_max_pair_step_ticks: int = 25
    """Max pair-tick delta per inverse-Jacobian step. Caps overshoot."""
    # --- Manual-baseline capture (operator hand-tensions, we record state) --
    manual_baseline_capture_count: int = 0
    """Number of operator-paced manual-baseline records to capture BEFORE the
    algorithm runs. 0 disables manual baseline capture. The experiment pauses
    ``manual_baseline_pause_s`` seconds between captures so the operator can
    re-tension the spine manually between them. Recorded baselines are written
    into the run folder and used by the comparison report."""
    manual_baseline_pause_s: float = 15.0
    """Seconds between manual baseline captures (operator re-tensions in this
    window). Long enough for human reaction; can be overridden for GUI-paced
    flows that drive the experiment via session callbacks."""
    manual_baseline_record_path: str = ""
    """Optional path to a pre-recorded manual baselines JSON. If set, the
    experiment skips the manual-capture phase and uses these records for the
    comparison report. Format: list of dicts with keys
    ``positions_by_servo``, ``currents_ma_by_servo``, ``tip_xy_mm``,
    ``timestamp_utc``. Lets the GUI capture manual baselines separately."""

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PretensionValidationExperimentConfig":
        payload = dict(payload or {})
        raw_servo_ids = payload.get("servo_ids")
        if isinstance(raw_servo_ids, str):
            tokens = [segment.strip() for segment in raw_servo_ids.replace(";", ",").split(",")]
        elif isinstance(raw_servo_ids, (list, tuple, set)):
            tokens = [str(value).strip() for value in raw_servo_ids]
        else:
            tokens = []
        servo_ids: list[int] = []
        for token in tokens:
            if not token:
                continue
            try:
                parsed = int(token)
            except Exception:
                continue
            if parsed > 0 and parsed not in servo_ids:
                servo_ids.append(parsed)
        raw_target_xy = [
            float(value)
            for value in (payload.get("tip_target_xy_mm") or [0.0, 0.0])
        ][:2]
        while len(raw_target_xy) < 2:
            raw_target_xy.append(0.0)
        return cls(
            mode=str(payload.get("mode", "single_servo_trace") or "single_servo_trace").strip().lower(),
            staged_strategy=str(payload.get("staged_strategy", "conservative_startup") or "conservative_startup").strip().lower(),
            allow_legacy_staged_strategy=bool(payload.get("allow_legacy_staged_strategy", False)),
            servo_id=int(payload.get("servo_id", 1)),
            servo_ids=servo_ids,
            repeat_runs=max(1, int(payload.get("repeat_runs", 1))),
            move_to_reference=bool(payload.get("move_to_reference", True)),
            include_tracker_displacement=bool(payload.get("include_tracker_displacement", True)),
            allow_current_only_when_tracker_missing=bool(payload.get("allow_current_only_when_tracker_missing", False)),
            allow_no_tracker_test_run=bool(payload.get("allow_no_tracker_test_run", False)),
            run_trust_mode=str(payload.get("run_trust_mode", "thesis_trusted") or "thesis_trusted").strip().lower(),
            enable_tip_centering=bool(payload.get("enable_tip_centering", True)),
            tip_center_tolerance_mm=max(0.0, float(payload.get("tip_center_tolerance_mm", 1.0))),
            tip_center_max_iterations=max(0, int(payload.get("tip_center_max_iterations", 16))),
            tip_center_step_ticks=max(1, int(payload.get("tip_center_step_ticks", 10))),
            tip_center_x_sign=(1 if int(payload.get("tip_center_x_sign", 1)) >= 0 else -1),
            tip_center_y_sign=(1 if int(payload.get("tip_center_y_sign", 1)) >= 0 else -1),
            tip_target_xy_mm=raw_target_xy,
            max_tip_displacement_mm=max(0.0, float(payload.get("max_tip_displacement_mm", 50.0))),
            tip_divergence_stop_mm=max(0.0, float(payload.get("tip_divergence_stop_mm", 2.0))),
            equalization_step_ticks=max(1, int(payload.get("equalization_step_ticks", 2))),
            equalization_max_iterations=max(0, int(payload.get("equalization_max_iterations", 24))),
            conservative_max_iterations=max(0, int(payload.get("conservative_max_iterations", 24))),
            conservative_step_ticks=max(1, int(payload.get("conservative_step_ticks", payload.get("tip_center_step_ticks", 10)))),
            conservative_max_cumulative_travel_ticks=max(
                0,
                int(payload.get("conservative_max_cumulative_travel_ticks", 80)),
            ),
            coarse_slack_step_ticks=max(1, int(payload.get("coarse_slack_step_ticks", 50))),
            soft_release_step_ticks=max(1, int(payload.get("soft_release_step_ticks", payload.get("coarse_slack_step_ticks", 50)))),
            soft_release_max_travel_ticks=max(0, int(payload.get("soft_release_max_travel_ticks", 40))),
            soft_release_current_target_ma=max(0.0, float(payload.get("soft_release_current_target_ma", 10.0))),
            takeup_target_load_proxy_ma=max(
                0.0,
                float(payload.get("takeup_target_load_proxy_ma", payload.get("engaged_load_proxy_target_ma", 25.0))),
            ),
            high_load_proxy_ma=max(0.0, float(payload.get("high_load_proxy_ma", 40.0))),
            takeup_max_iterations=max(0, int(payload.get("takeup_max_iterations", 16))),
            staged_packet_retry_budget=max(1, int(payload.get("staged_packet_retry_budget", 3))),
            characterization_step_ticks=max(1, int(payload.get("characterization_step_ticks", 1))),
            characterization_pair_cycles=max(1, int(payload.get("characterization_pair_cycles", 1))),
            current_characterization_sample_count=max(2, int(payload.get("current_characterization_sample_count", 20))),
            current_noise_multiplier=max(1.0, float(payload.get("current_noise_multiplier", 3.0))),
            min_meaningful_current_delta_ma=max(0.0, float(payload.get("min_meaningful_current_delta_ma", 2.0))),
            load_balance_tolerance_ma=max(0.0, float(payload.get("load_balance_tolerance_ma", 10.0))),
            pair_balance_tolerance_ma=max(0.0, float(payload.get("pair_balance_tolerance_ma", 10.0))),
            settle_verify_time_s=max(0.0, float(payload.get("settle_verify_time_s", 0.2))),
            accept_max_load_balance_error_ma=max(0.0, float(payload.get("accept_max_load_balance_error_ma", 30.0))),
            accept_max_pair_balance_error_ma=max(0.0, float(payload.get("accept_max_pair_balance_error_ma", 30.0))),
            accept_max_final_tip_xy_offset_mm=max(0.0, float(payload.get("accept_max_final_tip_xy_offset_mm", 3.0))),
            max_runtime_s=max(1.0, float(payload.get("max_runtime_s", 60.0))),
            tracker_tool_id=str(payload.get("tracker_tool_id", "0A")),
            pretension_start_mode=(
                str(payload.get("pretension_start_mode")).strip().lower()
                if payload.get("pretension_start_mode") not in (None, "")
                else None
            ),
            untensioned_reference_tick=(
                int(payload["untensioned_reference_tick"])
                if payload.get("untensioned_reference_tick") not in (None, "")
                else None
            ),
            step_ticks=int(payload["step_ticks"]) if payload.get("step_ticks") not in (None, "") else None,
            settle_time_s=(
                float(payload["settle_time_s"]) if payload.get("settle_time_s") not in (None, "") else None
            ),
            baseline_sample_count=(
                int(payload["baseline_sample_count"])
                if payload.get("baseline_sample_count") not in (None, "")
                else None
            ),
            current_filter_window=(
                int(payload["current_filter_window"])
                if payload.get("current_filter_window") not in (None, "")
                else None
            ),
            current_delta_threshold_ma=(
                int(payload["current_delta_threshold_ma"])
                if payload.get("current_delta_threshold_ma") not in (None, "")
                else None
            ),
            absolute_trigger_current_ma=(
                int(payload["absolute_trigger_current_ma"])
                if payload.get("absolute_trigger_current_ma") not in (None, "")
                else None
            ),
            hard_current_stop_ma=(
                int(payload["hard_current_stop_ma"])
                if payload.get("hard_current_stop_ma") not in (None, "")
                else None
            ),
            max_travel_ticks=(
                int(payload["max_travel_ticks"]) if payload.get("max_travel_ticks") not in (None, "") else None
            ),
            timeout_s=float(payload["timeout_s"]) if payload.get("timeout_s") not in (None, "") else None,
            tip_centering_variant=(
                str(payload.get("tip_centering_variant", PRETENSION_TIP_VARIANT_PAIRED_THEN_TIP)).strip().lower()
                if payload.get("tip_centering_variant") not in (None, "")
                else PRETENSION_TIP_VARIANT_PAIRED_THEN_TIP
            ),
            jacobian_probe_step_ticks=max(1, int(payload.get("jacobian_probe_step_ticks", 30))),
            jacobian_min_observable_tip_delta_mm=max(
                0.0, float(payload.get("jacobian_min_observable_tip_delta_mm", 0.20))
            ),
            jacobian_step_gain=max(0.05, min(1.5, float(payload.get("jacobian_step_gain", 0.8)))),
            jacobian_max_pair_step_ticks=max(1, int(payload.get("jacobian_max_pair_step_ticks", 25))),
            manual_baseline_capture_count=max(0, int(payload.get("manual_baseline_capture_count", 0))),
            manual_baseline_pause_s=max(0.0, float(payload.get("manual_baseline_pause_s", 15.0))),
            manual_baseline_record_path=str(payload.get("manual_baseline_record_path") or ""),
            engagement_scan_step_ticks=max(1, int(payload.get("engagement_scan_step_ticks", 50))),
            engagement_rise_threshold_ma=max(0.5, float(payload.get("engagement_rise_threshold_ma", 5.0))),
            engagement_back_off_ticks=max(0, int(payload.get("engagement_back_off_ticks", 50))),
        )


@dataclass
class TrackerTimingValidationConfig:
    """Config for backend Aurora timing and throughput validation."""

    requested_tool_ids: list[str] = field(default_factory=lambda: ["0A", "0B"])
    run_duration_s: float = 8.0
    sample_count_target: int | None = None
    warmup_samples: int = 10
    timeout_s: float = 20.0
    enable_servo_logging: bool = False
    run_label: str = ""
    servo_poll_interval_s: float = 0.02

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "TrackerTimingValidationConfig":
        payload = dict(payload or {})
        requested_tool_ids = [
            str(value).strip().upper()
            for value in (payload.get("requested_tool_ids") or ["0A", "0B"])
            if str(value).strip()
        ]
        requested_tool_ids = [
            tool_id
            for tool_id in requested_tool_ids
            if tool_id in {"0A", "0B"}
        ] or ["0A", "0B"]
        deduped_tool_ids: list[str] = []
        for tool_id in requested_tool_ids:
            if tool_id not in deduped_tool_ids:
                deduped_tool_ids.append(tool_id)
        return cls(
            requested_tool_ids=deduped_tool_ids,
            run_duration_s=float(payload.get("run_duration_s", 8.0)),
            sample_count_target=(
                int(payload["sample_count_target"])
                if payload.get("sample_count_target") not in (None, "", 0, "0")
                else None
            ),
            warmup_samples=max(0, int(payload.get("warmup_samples", 10))),
            timeout_s=float(payload.get("timeout_s", 20.0)),
            enable_servo_logging=bool(payload.get("enable_servo_logging", False)),
            run_label=str(payload.get("run_label", "") or ""),
            servo_poll_interval_s=float(payload.get("servo_poll_interval_s", 0.02)),
        )


@dataclass
class ServoTrackerSyncValidationConfig:
    """Config for shared-clock servo/tracker motion-sync validation."""

    servo_ids: list[int] = field(default_factory=lambda: [1])
    requested_tool_ids: list[str] = field(default_factory=lambda: ["0A"])
    motion_mode: str = "alternating_step"
    run_duration_s: float = 8.0
    warmup_duration_s: float = 1.0
    command_amplitude_ticks: int = 32
    step_period_s: float = 0.35
    telemetry_poll_interval_s: float = 0.02
    timeout_s: float = 20.0
    include_robot_frame_tip_pose: bool = True
    run_label: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ServoTrackerSyncValidationConfig":
        payload = dict(payload or {})
        raw_servo_ids = payload.get("servo_ids", [1])
        if isinstance(raw_servo_ids, str):
            tokens = [segment.strip() for segment in raw_servo_ids.replace(";", ",").split(",")]
        elif isinstance(raw_servo_ids, (list, tuple, set)):
            tokens = [str(value).strip() for value in raw_servo_ids]
        else:
            tokens = [str(raw_servo_ids).strip()]
        servo_ids: list[int] = []
        for token in tokens:
            if not token:
                continue
            try:
                servo_id = int(token)
            except Exception:
                continue
            if servo_id > 0 and servo_id not in servo_ids:
                servo_ids.append(servo_id)
        if not servo_ids:
            servo_ids = [1]
        requested_tool_ids = [
            str(value).strip().upper()
            for value in (payload.get("requested_tool_ids") or ["0A"])
            if str(value).strip()
        ]
        requested_tool_ids = [
            tool_id
            for tool_id in requested_tool_ids
            if tool_id in {"0A", "0B"}
        ] or ["0A"]
        deduped_tool_ids: list[str] = []
        for tool_id in requested_tool_ids:
            if tool_id not in deduped_tool_ids:
                deduped_tool_ids.append(tool_id)
        motion_mode = str(payload.get("motion_mode", "alternating_step") or "alternating_step").strip().lower()
        if motion_mode != "alternating_step":
            motion_mode = "alternating_step"
        return cls(
            servo_ids=servo_ids,
            requested_tool_ids=deduped_tool_ids,
            motion_mode=motion_mode,
            run_duration_s=float(payload.get("run_duration_s", 8.0)),
            warmup_duration_s=max(0.0, float(payload.get("warmup_duration_s", 1.0))),
            command_amplitude_ticks=max(1, int(payload.get("command_amplitude_ticks", 32))),
            step_period_s=max(0.05, float(payload.get("step_period_s", 0.35))),
            telemetry_poll_interval_s=max(0.005, float(payload.get("telemetry_poll_interval_s", 0.02))),
            timeout_s=float(payload.get("timeout_s", 20.0)),
            include_robot_frame_tip_pose=bool(payload.get("include_robot_frame_tip_pose", True)),
            run_label=str(payload.get("run_label", "") or ""),
        )


class TrackerPipelineMockExperiment(BaseExperiment):
    """Validate runner, logging, and summaries against mock tracking."""

    name = "tracker_pipeline_mock"
    description = "Validate the canonical runner and logging against mock tracking."
    hardware_requirements = ExperimentHardwareRequirements(tracking_required=True, mock_compatible=True)

    def __init__(self, config: TrackerPipelineMockConfig) -> None:
        super().__init__(config=config)
        self._tracking_started_here = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "TrackerPipelineMockExperiment":
        return cls(config=TrackerPipelineMockConfig.from_dict(payload))

    def setup(self, session: ExperimentSession) -> None:
        tracking_service = session.context.tracking_service
        if tracking_service is not None and getattr(tracking_service, "_thread", None) is None:
            tracking_service.start()
            self._tracking_started_here = True

    def precheck(self, session: ExperimentSession) -> None:
        if not session.context.settings.runtime.mock_mode:
            raise RuntimeError("tracker_pipeline_mock requires mock_mode=true.")

    def execute(self, session: ExperimentSession) -> None:
        total = max(1, int(self.config.sample_count))
        for sample_index in range(total):
            session.raise_if_stop_requested()
            snapshot = session.context.tracking_service.get_snapshot()
            sample = _sample_from_tracking_snapshot(
                session,
                snapshot=snapshot,
                phase="sample",
                step_index=sample_index,
                sample_index=sample_index,
                commanded_cable_deltas_cm=[],
                commanded_motor_values={},
                status_flags=["diagnostic_experiment"],
            )
            session.add_sample(sample)
            session.update_progress(sample_index + 1, total, {"phase": "sample", "step_index": sample_index})
            if self.config.sample_period_s > 0:
                session.context.sleep_fn(float(self.config.sample_period_s))
        session.set_metric(
            "unique_frames_observed_final",
            session.context.tracking_service.get_snapshot().unique_frames_observed,
        )

    def finalize(self, session: ExperimentSession) -> None:
        if self._tracking_started_here:
            session.context.tracking_service.stop()
            self._tracking_started_here = False


class TransformChainValidationExperiment(BaseExperiment):
    """Validate transform composition using synthetic transforms."""

    name = "transform_chain_validation"
    description = "Validate synthetic transform composition and output pose correctness."
    hardware_requirements = ExperimentHardwareRequirements(mock_compatible=True)

    def __init__(self, config: TransformChainValidationConfig) -> None:
        super().__init__(config=config)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "TransformChainValidationExperiment":
        return cls(config=TransformChainValidationConfig.from_dict(payload))

    def execute(self, session: ExperimentSession) -> None:
        if np is None:
            raise RuntimeError("numpy is required for transform_chain_validation")
        T_robot_aurora = np.eye(4)
        T_robot_aurora[0:3, 3] = np.asarray(self.config.robot_translation_mm, dtype=float)
        theta = math.pi / 2.0
        T_aurora_coil = np.array(
            [
                [math.cos(theta), -math.sin(theta), 0.0, self.config.tracker_translation_mm[0]],
                [math.sin(theta), math.cos(theta), 0.0, self.config.tracker_translation_mm[1]],
                [0.0, 0.0, 1.0, self.config.tracker_translation_mm[2]],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        T_coil_tip = np.eye(4)
        T_coil_tip[0:3, 3] = np.asarray(self.config.tip_offset_mm, dtype=float)
        T_robot_tip = T_robot_aurora @ T_aurora_coil @ T_coil_tip
        expected_translation = np.array(
            [
                self.config.robot_translation_mm[0] + self.config.tracker_translation_mm[0] - self.config.tip_offset_mm[1],
                self.config.robot_translation_mm[1] + self.config.tracker_translation_mm[1] + self.config.tip_offset_mm[0],
                self.config.robot_translation_mm[2] + self.config.tracker_translation_mm[2] + self.config.tip_offset_mm[2],
            ],
            dtype=float,
        )
        error_mm = float(np.linalg.norm(T_robot_tip[0:3, 3] - expected_translation))
        session.add_sample(
            ExperimentTimeseriesSample(
                monotonic_time_s=session.elapsed_s(),
                wall_time_utc=_utc_now_iso(),
                phase="validation",
                step_index=0,
                sample_index=0,
                transform_validity={"synthetic_chain": "valid"},
                pose_in_tracker_frame={"synthetic_coil": {"matrix": T_aurora_coil.tolist()}},
                pose_in_robot_frame={"synthetic_tip": {"matrix": T_robot_tip.tolist()}},
                status_flags=[],
                extra={"expected_translation_mm": expected_translation.tolist()},
            )
        )
        session.set_metric("translation_error_mm", error_mm)
        session.set_metric("expected_tip_translation_mm", expected_translation.tolist())


class CommandScheduleValidationExperiment(BaseExperiment):
    """Validate generated command schedules and schedule determinism."""

    name = "command_schedule_validation"
    description = "Validate sweep, grid, trajectory, and babble schedule generation."
    hardware_requirements = ExperimentHardwareRequirements(mock_compatible=True)

    def __init__(self, config: CommandScheduleValidationConfig) -> None:
        super().__init__(config=config)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "CommandScheduleValidationExperiment":
        return cls(config=CommandScheduleValidationConfig.from_dict(payload))

    def execute(self, session: ExperimentSession) -> None:
        points = generate_command_schedule(self.config.schedule)
        checksum = command_schedule_checksum(points)
        max_abs_delta = max((abs(float(value)) for point in points for value in point.tendon_displacement_cm), default=0.0)
        for point in points:
            session.add_sample(
                ExperimentTimeseriesSample(
                    monotonic_time_s=session.elapsed_s(),
                    wall_time_utc=_utc_now_iso(),
                    phase="schedule_point",
                    step_index=int(point.index),
                    sample_index=0,
                    commanded_cable_deltas_cm=[float(value) for value in point.tendon_displacement_cm],
                    status_flags=["schedule_generated"],
                    extra={"repeat": point.repeat, "label": point.label},
                )
            )
        session.set_metric("schedule_checksum", checksum)
        session.set_metric("point_count", len(points))
        session.set_metric("max_abs_cable_delta_cm", max_abs_delta)


class DatasetSchemaRoundtripExperiment(BaseExperiment):
    """Write and reload a synthetic dataset to validate schema integrity."""

    name = "dataset_schema_roundtrip"
    description = "Write a dataset and reload it to validate schema integrity."
    hardware_requirements = ExperimentHardwareRequirements(mock_compatible=True)

    def __init__(self, config: DatasetSchemaRoundtripConfig) -> None:
        super().__init__(config=config)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "DatasetSchemaRoundtripExperiment":
        return cls(config=DatasetSchemaRoundtripConfig.from_dict(payload))

    def execute(self, session: ExperimentSession) -> None:
        for sample_index in range(max(1, int(self.config.sample_count))):
            session.add_sample(
                ExperimentTimeseriesSample(
                    monotonic_time_s=float(sample_index) * 0.1,
                    wall_time_utc=_utc_now_iso(),
                    phase="roundtrip_source",
                    step_index=sample_index,
                    sample_index=sample_index,
                    commanded_cable_deltas_cm=[0.1 * float(sample_index)],
                    status_flags=["schema_validation"],
                    extra={"value": sample_index},
                )
            )
        roundtrip_root = session.context.output_root / "_schema_roundtrip_tmp"
        writer = ExperimentDatasetWriter(roundtrip_root)
        metadata = ExperimentMetadata(
            schema_version=session.metadata.schema_version,
            experiment_name=f"{self.name}_internal",
            run_id=f"{session.metadata.run_id}_internal",
            timestamp_utc=_utc_now_iso(),
            git_commit=session.metadata.git_commit,
            backend_info={"mode": "synthetic"},
            registration_info={},
            config_used={"sample_count": self.config.sample_count},
            operator_notes="internal roundtrip validation",
        )
        summary = ExperimentSummary(
            schema_version=session.metadata.schema_version,
            experiment_name=metadata.experiment_name,
            run_id=metadata.run_id,
            success=True,
            sample_counts={"total": len(session.samples)},
            dropped_frames=0,
            invalid_transforms=0,
            stage_pass_fail={"roundtrip": "passed"},
            experiment_metrics={"sample_count": len(session.samples)},
        )
        paths = writer.write_dataset(metadata, list(session.samples), summary)
        bundle = ExperimentDatasetLoader().load_dataset(paths.output_dir)
        session.set_metric("roundtrip_output_dir", str(paths.output_dir))
        session.set_metric("roundtrip_ok", bundle.summary.sample_counts.get("total") == len(session.samples))
        session.set_metric("roundtrip_sample_count", len(bundle.samples))


class ReplayRunnerExperiment(BaseExperiment):
    """Replay a recorded dataset through offline analysis."""

    name = "replay_runner"
    description = "Replay a recorded dataset through the canonical loader and analysis path."
    hardware_requirements = ExperimentHardwareRequirements(mock_compatible=True)

    def __init__(self, config: ReplayRunnerConfig) -> None:
        super().__init__(config=config)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "ReplayRunnerExperiment":
        return cls(config=ReplayRunnerConfig.from_dict(payload))

    def precheck(self, session: ExperimentSession) -> None:
        if not self.config.dataset_path:
            raise RuntimeError("replay_runner requires config.dataset_path.")
        if not Path(self.config.dataset_path).exists():
            raise RuntimeError(f"Replay dataset path does not exist: {self.config.dataset_path}")

    def execute(self, session: ExperimentSession) -> None:
        bundle = ExperimentDatasetLoader().load_dataset(Path(self.config.dataset_path))
        for index, sample in enumerate(bundle.samples):
            replayed = ExperimentTimeseriesSample.from_dict(sample.to_dict())
            replayed.phase = f"replay::{sample.phase}"
            replayed.step_index = index
            replayed.extra = {**dict(replayed.extra), "source_run_id": bundle.metadata.run_id}
            session.add_sample(replayed)
        session.set_metric("source_experiment_name", bundle.metadata.experiment_name)
        session.set_metric("source_sample_count", len(bundle.samples))
        session.set_metric("source_success", bundle.summary.success)


class TrackerTimingValidationExperiment(BaseExperiment):
    """Benchmark backend Aurora acquisition timing on the active Python tracker path."""

    name = "tracker_timing_validation"
    description = (
        "Measure backend Aurora acquisition timing, duplicate-frame behavior, and optional "
        "tracker-servo timestamp alignment using the active Python tracking backend."
    )
    hardware_requirements = ExperimentHardwareRequirements(
        tracking_required=True,
        servo_required=False,
        mock_compatible=False,
    )

    def __init__(self, config: TrackerTimingValidationConfig) -> None:
        super().__init__(config=config)
        self._tracking_started_here = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "TrackerTimingValidationExperiment":
        return cls(config=TrackerTimingValidationConfig.from_dict(payload))

    def setup(self, session: ExperimentSession) -> None:
        tracking_service = session.context.tracking_service
        if tracking_service is not None and getattr(tracking_service, "_thread", None) is None:
            tracking_service.start()
            self._tracking_started_here = True

    def precheck(self, session: ExperimentSession) -> None:
        tracking_service = session.context.tracking_service
        if tracking_service is None:
            raise RuntimeError("tracker_timing_validation requires tracking_service.")
        if not hasattr(tracking_service, "register_timing_listener"):
            raise RuntimeError("Configured tracking service does not expose backend timing listeners.")
        snapshot = tracking_service.get_snapshot()
        backend_identity = str(snapshot.backend_identity or "")
        selected_backend = str(snapshot.selected_backend_name or "")
        if backend_identity == "tracker_bridge_json" or selected_backend == "bridge":
            raise RuntimeError(
                "tracker_timing_validation targets the Python NDI backend path. "
                "Legacy tracker_bridge is not a valid benchmark backend for this diagnostic."
            )

    def execute(self, session: ExperimentSession) -> None:
        tracking_service = session.context.tracking_service
        servo_service = session.context.servo_service
        requested_tool_ids = list(self.config.requested_tool_ids or ["0A", "0B"])
        warmup_samples = max(0, int(self.config.warmup_samples))
        run_duration_s = max(0.1, float(self.config.run_duration_s))
        sample_count_target = (
            int(self.config.sample_count_target)
            if self.config.sample_count_target not in (None, "", 0)
            else None
        )
        timeout_s = max(run_duration_s, float(self.config.timeout_s))
        run_start_ns = time.monotonic_ns()
        pending_tracker_records: list[dict[str, Any]] = []
        pending_servo_records: list[dict[str, Any]] = []
        queue_lock = threading.Lock()
        servo_stop_event = threading.Event()
        tracker_sample_index = 0
        servo_sample_index = 0
        analyzed_tracker_count = 0
        timed_out = False

        def timing_listener(record: dict[str, Any]) -> None:
            normalized = dict(record or {})
            normalized["requested_tool_ids"] = list(requested_tool_ids)
            with queue_lock:
                pending_tracker_records.append(normalized)

        def drain_pending_records() -> None:
            nonlocal tracker_sample_index, servo_sample_index, analyzed_tracker_count
            with queue_lock:
                tracker_batch = list(pending_tracker_records)
                servo_batch = list(pending_servo_records)
                pending_tracker_records.clear()
                pending_servo_records.clear()
            for record in tracker_batch:
                warmup_discarded = tracker_sample_index < warmup_samples
                record["warmup_discarded"] = bool(warmup_discarded)
                if not warmup_discarded:
                    analyzed_tracker_count += 1
                session.add_sample(
                    self._build_tracker_timing_sample(
                        session,
                        record=record,
                        run_start_ns=run_start_ns,
                        sample_index=tracker_sample_index,
                    )
                )
                tracker_sample_index += 1
            for record in servo_batch:
                session.add_sample(
                    TrackerTimingValidationExperiment._build_servo_timing_sample(
                        session,
                        record=record,
                        run_start_ns=run_start_ns,
                        sample_index=servo_sample_index,
                    )
                )
                servo_sample_index += 1

        def poll_servo_loop() -> None:
            servo_ids = [int(value) for value in (session.context.settings.robot.servo_ids or [])]
            poll_interval_s = max(0.01, float(self.config.servo_poll_interval_s))
            while not servo_stop_event.is_set():
                sample_monotonic_ns = time.monotonic_ns()
                try:
                    telemetry_by_id = servo_service.read_live_telemetry(servo_ids)
                    goal_positions = servo_service.last_goal_positions()
                    goal_times = servo_service.last_goal_command_times()
                    for servo_id in servo_ids:
                        telemetry = telemetry_by_id.get(int(servo_id))
                        pending_record = {
                            "sample_monotonic_ns": int(sample_monotonic_ns),
                            "servo_id": int(servo_id),
                            "commanded_position_ticks": goal_positions.get(int(servo_id)),
                            "commanded_position_age_s": (
                                float(max(0.0, (sample_monotonic_ns / 1_000_000_000.0) - float(goal_times[int(servo_id)])))
                                if int(servo_id) in goal_times
                                else None
                            ),
                            "present_position_ticks": (
                                int(telemetry.present_position)
                                if telemetry is not None and telemetry.present_position is not None
                                else None
                            ),
                            "present_current_ma": (
                                int(telemetry.present_current_ma)
                                if telemetry is not None and telemetry.present_current_ma is not None
                                else None
                            ),
                            "telemetry_age_s": (
                                servo_service.telemetry_age_s(telemetry)
                                if telemetry is not None
                                else None
                            ),
                            "reported_servo_id": (
                                int(telemetry.reported_servo_id)
                                if telemetry is not None and telemetry.reported_servo_id is not None
                                else None
                            ),
                            "error_flag": False,
                            "error_message": None,
                        }
                        with queue_lock:
                            pending_servo_records.append(pending_record)
                except Exception as exc:
                    with queue_lock:
                        pending_servo_records.append(
                            {
                                "sample_monotonic_ns": int(sample_monotonic_ns),
                                "servo_id": None,
                                "commanded_position_ticks": None,
                                "present_position_ticks": None,
                                "present_current_ma": None,
                                "telemetry_age_s": None,
                                "reported_servo_id": None,
                                "error_flag": True,
                                "error_message": str(exc),
                            }
                        )
                    return
                time.sleep(poll_interval_s)

        tracking_service.register_timing_listener(timing_listener)
        servo_thread: threading.Thread | None = None
        if bool(self.config.enable_servo_logging) and bool(getattr(servo_service, "is_connected", False)):
            servo_thread = threading.Thread(target=poll_servo_loop, daemon=True)
            servo_thread.start()

        try:
            while True:
                session.raise_if_stop_requested()
                drain_pending_records()
                elapsed_s = max(0.0, float(time.monotonic_ns() - run_start_ns) / 1_000_000_000.0)
                if sample_count_target is not None:
                    progress_total = max(1, int(sample_count_target))
                    progress_current = min(progress_total, int(analyzed_tracker_count))
                else:
                    progress_total = max(1, int(round(run_duration_s * 1000.0)))
                    progress_current = min(progress_total, int(round(elapsed_s * 1000.0)))
                session.update_progress(
                    progress_current,
                    progress_total,
                    {
                        "phase": "timing_capture",
                        "analyzed_tracker_samples": analyzed_tracker_count,
                        "total_tracker_samples": tracker_sample_index,
                    },
                )
                if sample_count_target is not None and analyzed_tracker_count >= sample_count_target:
                    break
                if sample_count_target is None and elapsed_s >= run_duration_s:
                    break
                if elapsed_s >= timeout_s:
                    timed_out = True
                    session.add_warning(
                        "Timing diagnostic timed out before the configured stop condition completed."
                    )
                    break
                time.sleep(0.005)
        finally:
            servo_stop_event.set()
            if servo_thread is not None:
                servo_thread.join(timeout=1.0)
            tracking_service.unregister_timing_listener(timing_listener)
            time.sleep(0.01)
            drain_pending_records()

        tracker_records = extract_tracker_timing_records(session.samples)
        servo_records = extract_servo_timing_records(session.samples)
        servo_sync = compute_servo_sync_summary(tracker_records, servo_records)
        servo_sync["enabled"] = bool(self.config.enable_servo_logging)
        if bool(self.config.enable_servo_logging) and not servo_records:
            session.add_warning("Servo sync logging was requested, but no servo telemetry samples were captured.")
            servo_sync["available"] = False
        snapshot = tracking_service.get_snapshot()
        force_status = "success"
        if not tracker_records:
            force_status = "invalid_due_to_insufficient_samples"
        elif timed_out:
            force_status = "partial_success"
        metrics = compute_tracker_timing_summary(
            tracker_records,
            requested_tool_ids=requested_tool_ids,
            backend_identity=str(snapshot.backend_identity or ""),
            configured_backend_name=str(snapshot.configured_backend_name or ""),
            selected_backend_name=str(snapshot.selected_backend_name or ""),
            run_duration_s=run_duration_s if sample_count_target is None else None,
            run_label=str(self.config.run_label or ""),
            servo_sync_summary=servo_sync,
        )
        metrics.update(
            {
                "stop_mode": "sample_count" if sample_count_target is not None else "duration",
                "sample_count_target": sample_count_target,
                "warmup_samples_requested": int(warmup_samples),
                "timeout_s": float(timeout_s),
                "enable_servo_logging": bool(self.config.enable_servo_logging),
                "servo_sample_count": len(servo_records),
                "instrumented_stage_definitions": {
                    "backend_call_ms": "Host monotonic time spent inside backend get_frame().",
                    "parse_ms": "Host monotonic time from backend return to parsed tool transforms/runtime fields.",
                    "state_commit_ms": "Host monotonic time from parsed payload to committed backend runtime state.",
                    "total_cycle_ms": "Host monotonic time from cycle start to committed backend sample record.",
                },
                "summary_requirements": {"force_status": force_status},
            }
        )
        session.metrics.update(metrics)

    def finalize(self, session: ExperimentSession) -> None:
        if self._tracking_started_here:
            session.context.tracking_service.stop()
            self._tracking_started_here = False

    def write_outputs(self, session: ExperimentSession, paths, summary) -> None:
        write_tracker_timing_outputs(
            output_dir=paths.output_dir,
            metadata=session.metadata,
            summary=summary,
            samples=session.samples,
        )

    @staticmethod
    def _build_tracker_timing_sample(
        session: ExperimentSession,
        *,
        record: dict[str, Any],
        run_start_ns: int,
        sample_index: int,
    ) -> ExperimentTimeseriesSample:
        commit_ns = int(record.get("sample_commit_monotonic_ns") or record.get("state_commit_complete_ns") or run_start_ns)
        monotonic_time_s = max(0.0, float(commit_ns - int(run_start_ns)) / 1_000_000_000.0)
        tool_validity = {
            str(key): str(value)
            for key, value in dict(record.get("tool_validity", {}) or {}).items()
        }
        status_flags: list[str] = ["tracker_timing"]
        if bool(record.get("is_duplicate_frame", False)):
            status_flags.append("duplicate_frame")
        if bool(record.get("error_flag", False)):
            status_flags.append("backend_error")
        return ExperimentTimeseriesSample(
            monotonic_time_s=monotonic_time_s,
            wall_time_utc=str(record.get("observed_at_utc") or _utc_now_iso()),
            phase="tracker_timing",
            step_index=int(sample_index),
            sample_index=int(sample_index),
            tracker_frame_id=(
                int(record["frame_number"])
                if record.get("frame_number") is not None
                else None
            ),
            tool_ids_seen=[str(value) for value in (record.get("tools_visible") or [])],
            transform_validity=tool_validity,
            freshness_s=None,
            latency_s=None,
            status_flags=status_flags,
            backend_health={
                "backend_identity": str(record.get("backend_identity", "")),
                "error_flag": bool(record.get("error_flag", False)),
                "error_stage": record.get("error_stage"),
            },
            extra={**dict(record), "record_kind": "tracker_timing"},
        )

    @staticmethod
    def _build_servo_timing_sample(
        session: ExperimentSession,
        *,
        record: dict[str, Any],
        run_start_ns: int,
        sample_index: int,
    ) -> ExperimentTimeseriesSample:
        sample_ns = int(record.get("sample_monotonic_ns") or run_start_ns)
        monotonic_time_s = max(0.0, float(sample_ns - int(run_start_ns)) / 1_000_000_000.0)
        servo_id = record.get("servo_id")
        status_flags = ["servo_timing"]
        if bool(record.get("error_flag", False)):
            status_flags.append("servo_error")
        commanded_motor_values = {}
        if servo_id not in (None, ""):
            commanded_motor_values["servo_id"] = int(servo_id)
        if record.get("commanded_position_ticks") is not None:
            commanded_motor_values["commanded_position_ticks"] = int(record["commanded_position_ticks"])
        return ExperimentTimeseriesSample(
            monotonic_time_s=monotonic_time_s,
            wall_time_utc=_utc_now_iso(),
            phase="servo_timing",
            step_index=int(sample_index),
            sample_index=int(sample_index),
            tracker_frame_id=-1,
            commanded_motor_values=commanded_motor_values,
            status_flags=status_flags,
            backend_health={"source": "servo_service"},
            extra={**dict(record), "record_kind": "servo_timing"},
        )


class ServoTrackerSyncValidationExperiment(BaseExperiment):
    """Validate host-time alignment between servo motion and tracker samples."""

    name = "servo_tracker_sync_validation"
    description = (
        "Run a bounded servo motion schedule while logging tracker frames, servo commands, and servo telemetry "
        "on one host monotonic clock to validate synchronization quality for later robot-motion experiments."
    )
    hardware_requirements = ExperimentHardwareRequirements(
        tracking_required=True,
        servo_required=True,
        mock_compatible=False,
    )

    def __init__(self, config: ServoTrackerSyncValidationConfig) -> None:
        super().__init__(config=config)
        self._tracking_started_here = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "ServoTrackerSyncValidationExperiment":
        return cls(config=ServoTrackerSyncValidationConfig.from_dict(payload))

    def setup(self, session: ExperimentSession) -> None:
        tracking_service = session.context.tracking_service
        if tracking_service is not None and getattr(tracking_service, "_thread", None) is None:
            tracking_service.start()
            self._tracking_started_here = True

    def precheck(self, session: ExperimentSession) -> None:
        tracking_service = session.context.tracking_service
        servo_service = session.context.servo_service
        if tracking_service is None:
            raise RuntimeError("servo_tracker_sync_validation requires tracking_service.")
        if servo_service is None or not bool(getattr(servo_service, "is_connected", False)):
            raise RuntimeError("servo_tracker_sync_validation requires a connected servo service.")
        if not hasattr(tracking_service, "register_timing_listener"):
            raise RuntimeError("Configured tracking service does not expose backend timing listeners.")
        if bool(session.context.settings.runtime.mock_mode):
            raise RuntimeError("servo_tracker_sync_validation requires live runtime mode.")
        if not self.config.servo_ids:
            raise RuntimeError("servo_tracker_sync_validation requires at least one servo ID.")
        snapshot = tracking_service.get_snapshot()
        backend_identity = str(snapshot.backend_identity or "")
        selected_backend = str(snapshot.selected_backend_name or "")
        if backend_identity == "tracker_bridge_json" or selected_backend == "bridge":
            raise RuntimeError(
                "servo_tracker_sync_validation targets the Python NDI backend path. "
                "Legacy tracker_bridge is not a valid backend for this validation."
            )

    def execute(self, session: ExperimentSession) -> None:
        tracking_service = session.context.tracking_service
        servo_service = session.context.servo_service
        requested_tool_ids = list(self.config.requested_tool_ids or ["0A"])
        servo_ids = [int(servo_id) for servo_id in self.config.servo_ids]
        total_duration_s = max(0.5, float(self.config.run_duration_s))
        warmup_duration_s = max(0.0, float(self.config.warmup_duration_s))
        capture_duration_s = total_duration_s + warmup_duration_s
        step_period_s = max(0.05, float(self.config.step_period_s))
        telemetry_poll_interval_s = max(0.005, float(self.config.telemetry_poll_interval_s))
        timeout_s = max(capture_duration_s, float(self.config.timeout_s))
        amplitude_ticks = max(1, int(self.config.command_amplitude_ticks))
        run_start_ns = time.monotonic_ns()
        warmup_end_ns = run_start_ns + int(round(warmup_duration_s * 1_000_000_000.0))
        pending_tracker_records: list[dict[str, Any]] = []
        pending_servo_records: list[dict[str, Any]] = []
        pending_command_records: list[dict[str, Any]] = []
        queue_lock = threading.Lock()
        state_lock = threading.Lock()
        tracker_sample_index = 0
        servo_sample_index = 0
        command_sample_index = 0
        timed_out = False
        motion_phase = "warmup"
        command_step_index = -1
        commanded_positions_by_servo: dict[int, int] = {}

        initial_telemetry = servo_service.read_live_telemetry(servo_ids)
        centers_by_servo: dict[int, int] = {}
        amplitude_by_servo: dict[int, int] = {}
        for servo_id in servo_ids:
            telemetry = initial_telemetry.get(int(servo_id))
            assessment = servo_service.assess_motion(
                int(servo_id),
                require_calibrated_bounds=False,
                telemetry=telemetry,
            )
            if not assessment.ready:
                raise RuntimeError(f"Servo {servo_id} is not motion-ready: {assessment.reason}")
            if telemetry is None or telemetry.present_position is None:
                raise RuntimeError(f"Servo {servo_id} present position is unavailable.")
            current_position = int(telemetry.present_position)
            safe_min = int(assessment.safe_min_tick if assessment.safe_min_tick is not None else current_position)
            safe_max = int(assessment.safe_max_tick if assessment.safe_max_tick is not None else current_position)
            usable_amplitude = int(min(amplitude_ticks, current_position - safe_min, safe_max - current_position))
            if usable_amplitude <= 0:
                raise RuntimeError(
                    f"Servo {servo_id} cannot execute the requested bounded motion from position {current_position} "
                    f"within safe range [{safe_min}, {safe_max}]."
                )
            centers_by_servo[int(servo_id)] = current_position
            amplitude_by_servo[int(servo_id)] = usable_amplitude
            commanded_positions_by_servo[int(servo_id)] = current_position

        def timing_listener(record: dict[str, Any]) -> None:
            normalized = dict(record or {})
            normalized["requested_tool_ids"] = list(requested_tool_ids)
            normalized["received_monotonic_ns"] = int(time.monotonic_ns())
            with state_lock:
                normalized["motion_phase"] = motion_phase
                normalized["command_step_index"] = command_step_index
                normalized["commanded_positions_by_servo"] = {
                    str(servo_id): int(goal_tick)
                    for servo_id, goal_tick in commanded_positions_by_servo.items()
                }
            with queue_lock:
                pending_tracker_records.append(normalized)

        def queue_servo_telemetry_records(*, telemetry_by_id: dict[int, Any], sample_ns: int) -> None:
            nonlocal pending_servo_records
            goal_positions = servo_service.last_goal_positions()
            goal_times = servo_service.last_goal_command_times()
            with state_lock:
                phase = str(motion_phase)
                step_index_value = int(command_step_index)
            warmup_discarded = int(sample_ns) < int(warmup_end_ns)
            for servo_id in servo_ids:
                telemetry = telemetry_by_id.get(int(servo_id))
                goal_time_s = goal_times.get(int(servo_id))
                pending_record = {
                    "sample_monotonic_ns": int(sample_ns),
                    "servo_id": int(servo_id),
                    "commanded_position_ticks": goal_positions.get(int(servo_id)),
                    "commanded_position_age_s": (
                        float(max(0.0, (sample_ns / 1_000_000_000.0) - float(goal_time_s)))
                        if goal_time_s is not None
                        else None
                    ),
                    "present_position_ticks": (
                        int(telemetry.present_position)
                        if telemetry is not None and telemetry.present_position is not None
                        else None
                    ),
                    "present_current_ma": (
                        int(telemetry.present_current_ma)
                        if telemetry is not None and telemetry.present_current_ma is not None
                        else None
                    ),
                    "telemetry_age_s": servo_service.telemetry_age_s(telemetry) if telemetry is not None else None,
                    "reported_servo_id": (
                        int(telemetry.reported_servo_id)
                        if telemetry is not None and telemetry.reported_servo_id is not None
                        else None
                    ),
                    "torque_enabled": (
                        bool(telemetry.torque_enabled)
                        if telemetry is not None and telemetry.torque_enabled is not None
                        else None
                    ),
                    "motion_phase": phase,
                    "command_step_index": step_index_value,
                    "warmup_discarded": bool(warmup_discarded),
                    "error_flag": False,
                    "error_message": None,
                }
                pending_servo_records.append(pending_record)

        def drain_pending_records() -> None:
            nonlocal tracker_sample_index, servo_sample_index, command_sample_index
            with queue_lock:
                tracker_batch = list(pending_tracker_records)
                servo_batch = list(pending_servo_records)
                command_batch = list(pending_command_records)
                pending_tracker_records.clear()
                pending_servo_records.clear()
                pending_command_records.clear()
            latest_snapshot = tracking_service.get_snapshot()
            for record in tracker_batch:
                commit_ns = self._effective_tracker_commit_ns(record, fallback_ns=run_start_ns)
                record["sample_commit_monotonic_ns"] = int(commit_ns)
                record["warmup_discarded"] = bool(commit_ns < int(warmup_end_ns))
                session.add_sample(
                    self._build_tracker_sync_sample(
                        record=record,
                        latest_snapshot=latest_snapshot,
                        run_start_ns=run_start_ns,
                        sample_index=tracker_sample_index,
                        include_robot_frame_tip_pose=bool(self.config.include_robot_frame_tip_pose),
                    )
                )
                tracker_sample_index += 1
            for record in servo_batch:
                session.add_sample(
                    TrackerTimingValidationExperiment._build_servo_timing_sample(
                        session,
                        record=record,
                        run_start_ns=run_start_ns,
                        sample_index=servo_sample_index,
                    )
                )
                servo_sample_index += 1
            for record in command_batch:
                session.add_sample(
                    self._build_servo_command_sample(
                        record=record,
                        run_start_ns=run_start_ns,
                        sample_index=command_sample_index,
                    )
                )
                command_sample_index += 1

        tracking_service.register_timing_listener(timing_listener)
        next_telemetry_poll_ns = int(run_start_ns)
        next_command_ns = int(warmup_end_ns)

        try:
            with servo_service.exclusive_bus_operation(
                owner=self.name,
                reason="servo-tracker sync validation",
            ):
                queue_servo_telemetry_records(telemetry_by_id=initial_telemetry, sample_ns=int(run_start_ns))
                while True:
                    session.raise_if_stop_requested()
                    now_ns = time.monotonic_ns()
                    elapsed_s = max(0.0, float(now_ns - run_start_ns) / 1_000_000_000.0)
                    if now_ns >= next_telemetry_poll_ns:
                        telemetry_by_id = servo_service.read_live_telemetry(servo_ids)
                        queue_servo_telemetry_records(telemetry_by_id=telemetry_by_id, sample_ns=int(now_ns))
                        next_telemetry_poll_ns = int(now_ns + int(round(telemetry_poll_interval_s * 1_000_000_000.0)))
                    if now_ns >= next_command_ns and elapsed_s < capture_duration_s:
                        with state_lock:
                            command_step_index += 1
                            direction_sign = 1 if (command_step_index % 2 == 0) else -1
                            motion_phase = "step_positive" if direction_sign > 0 else "step_negative"
                        for servo_id in servo_ids:
                            target_tick = int(centers_by_servo[int(servo_id)] + direction_sign * amplitude_by_servo[int(servo_id)])
                            result = servo_service.move_servo_to_raw_target(
                                servo_id=int(servo_id),
                                target_tick=int(target_tick),
                                reason=self.name,
                            )
                            if not result.success:
                                raise RuntimeError(result.message)
                            goal_positions = servo_service.last_goal_positions()
                            goal_times = servo_service.last_goal_command_times()
                            with state_lock:
                                commanded_positions_by_servo[int(servo_id)] = int(goal_positions.get(int(servo_id), target_tick))
                            pending_command_records.append(
                                {
                                    "command_monotonic_ns": int(round(goal_times.get(int(servo_id), time.monotonic()) * 1_000_000_000.0)),
                                    "servo_id": int(servo_id),
                                    "commanded_position_ticks": int(commanded_positions_by_servo[int(servo_id)]),
                                    "motion_phase": str(motion_phase),
                                    "command_step_index": int(command_step_index),
                                    "warmup_discarded": False,
                                    "error_flag": False,
                                    "error_message": None,
                                }
                            )
                        next_command_ns = int(now_ns + int(round(step_period_s * 1_000_000_000.0)))
                    drain_pending_records()
                    progress_total = max(1, int(round(capture_duration_s * 1000.0)))
                    progress_current = min(progress_total, int(round(elapsed_s * 1000.0)))
                    session.update_progress(
                        progress_current,
                        progress_total,
                        {
                            "phase": motion_phase,
                            "command_step_index": int(command_step_index),
                            "tracker_samples": tracker_sample_index,
                            "servo_telemetry_samples": servo_sample_index,
                            "servo_command_samples": command_sample_index,
                        },
                    )
                    if elapsed_s >= capture_duration_s:
                        break
                    if elapsed_s >= timeout_s:
                        timed_out = True
                        session.add_warning("Sync validation timed out before the configured motion window completed.")
                        break
                    time.sleep(0.005)
        finally:
            tracking_service.unregister_timing_listener(timing_listener)
            time.sleep(0.01)
            drain_pending_records()

        tracker_records = extract_tracker_timing_records(session.samples)
        servo_telemetry_records = extract_servo_timing_records(session.samples)
        servo_command_records = extract_servo_command_records(session.samples)
        snapshot = tracking_service.get_snapshot()
        sync_summary = compute_servo_tracker_sync_summary(
            tracker_records,
            servo_telemetry_records,
            servo_command_records,
        )
        tracker_metrics = compute_tracker_timing_summary(
            tracker_records,
            requested_tool_ids=requested_tool_ids,
            backend_identity=str(snapshot.backend_identity or ""),
            configured_backend_name=str(snapshot.configured_backend_name or ""),
            selected_backend_name=str(snapshot.selected_backend_name or ""),
            run_duration_s=total_duration_s,
            run_label=str(self.config.run_label or ""),
        )
        motion_metric = self._motion_metric_summary(
            samples=session.samples,
            requested_tool_ids=requested_tool_ids,
            prefer_robot_frame_tip=bool(self.config.include_robot_frame_tip_pose),
        )
        force_status = "success"
        if not tracker_records or not servo_telemetry_records or not servo_command_records:
            force_status = "invalid_due_to_insufficient_samples"
        elif not bool(sync_summary.get("available")):
            force_status = "invalid_due_to_insufficient_samples"
        elif timed_out:
            force_status = "partial_success"
        tracker_metrics.update(
            {
                "motion_protocol": str(self.config.motion_mode),
                "selected_servo_ids": [int(servo_id) for servo_id in servo_ids],
                "run_duration_s": float(total_duration_s),
                "warmup_duration_s": float(warmup_duration_s),
                "command_amplitude_ticks": int(amplitude_ticks),
                "step_period_s": float(step_period_s),
                "telemetry_poll_interval_s": float(telemetry_poll_interval_s),
                "timeout_s": float(timeout_s),
                "include_robot_frame_tip_pose": bool(self.config.include_robot_frame_tip_pose),
                "servo_telemetry_sample_count": len(servo_telemetry_records),
                "servo_command_sample_count": len(servo_command_records),
                "command_step_count": max(0, int(command_step_index) + 1),
                "center_positions_ticks": {str(key): int(value) for key, value in centers_by_servo.items()},
                "amplitude_by_servo_ticks": {str(key): int(value) for key, value in amplitude_by_servo.items()},
                "servo_tracker_sync": sync_summary,
                "tracker_motion_metric": motion_metric,
                "summary_requirements": {"force_status": force_status},
            }
        )
        session.metrics.update(tracker_metrics)

    def finalize(self, session: ExperimentSession) -> None:
        if self._tracking_started_here:
            session.context.tracking_service.stop()
            self._tracking_started_here = False

    def write_outputs(self, session: ExperimentSession, paths, summary) -> None:
        write_servo_tracker_sync_outputs(
            output_dir=paths.output_dir,
            metadata=session.metadata,
            summary=summary,
            samples=session.samples,
        )

    @staticmethod
    def _build_tracker_sync_sample(
        *,
        record: dict[str, Any],
        latest_snapshot,
        run_start_ns: int,
        sample_index: int,
        include_robot_frame_tip_pose: bool,
    ) -> ExperimentTimeseriesSample:
        commit_ns = int(record.get("sample_commit_monotonic_ns") or record.get("state_commit_complete_ns") or run_start_ns)
        monotonic_time_s = max(0.0, float(commit_ns - int(run_start_ns)) / 1_000_000_000.0)
        tool_validity = {
            str(key): str(value)
            for key, value in dict(record.get("tool_validity", {}) or {}).items()
        }
        pose_in_tracker_frame: dict[str, Any] = {}
        for tool_id, payload in dict(record.get("tool_pose_payload", {}) or {}).items():
            pose_in_tracker_frame[str(tool_id)] = {
                "tracking_state": str(payload.get("tracking_state", "unknown") or "unknown"),
                "translation_mm": list(payload.get("translation_mm", [])) if isinstance(payload.get("translation_mm"), list) else None,
                "quaternion_wxyz": list(payload.get("quaternion_wxyz", [])) if isinstance(payload.get("quaternion_wxyz"), list) else None,
                "frame_number": (
                    int(payload["frame_number"])
                    if payload.get("frame_number") is not None
                    else None
                ),
            }
        pose_in_robot_frame: dict[str, Any] = {}
        tip_pose_matched = False
        if (
            include_robot_frame_tip_pose
            and latest_snapshot is not None
            and latest_snapshot.T_robot_tip is not None
            and latest_snapshot.last_frame_number is not None
            and record.get("frame_number") is not None
            and int(latest_snapshot.last_frame_number) == int(record.get("frame_number"))
        ):
            pose_in_robot_frame["tip"] = {
                "matrix": latest_snapshot.T_robot_tip,
                "translation_mm": [
                    float(latest_snapshot.T_robot_tip[0][3]),
                    float(latest_snapshot.T_robot_tip[1][3]),
                    float(latest_snapshot.T_robot_tip[2][3]),
                ],
            }
            tip_pose_matched = True
        status_flags: list[str] = ["tracker_sync"]
        if bool(record.get("is_duplicate_frame", False)):
            status_flags.append("duplicate_frame")
        if bool(record.get("error_flag", False)):
            status_flags.append("backend_error")
        if bool(record.get("warmup_discarded", False)):
            status_flags.append("warmup_discarded")
        return ExperimentTimeseriesSample(
            monotonic_time_s=monotonic_time_s,
            wall_time_utc=str(record.get("observed_at_utc") or _utc_now_iso()),
            phase="tracker_sync",
            step_index=int(sample_index),
            sample_index=int(sample_index),
            tracker_frame_id=(
                int(record["frame_number"])
                if record.get("frame_number") is not None
                else None
            ),
            tool_ids_seen=[str(value) for value in (record.get("tools_visible") or [])],
            transform_validity=tool_validity,
            pose_in_tracker_frame=pose_in_tracker_frame,
            pose_in_robot_frame=pose_in_robot_frame,
            freshness_s=None,
            latency_s=None,
            status_flags=status_flags,
            backend_health={
                "backend_identity": str(record.get("backend_identity", "")),
                "error_flag": bool(record.get("error_flag", False)),
                "error_stage": record.get("error_stage"),
                "tip_pose_matched": bool(tip_pose_matched),
            },
            extra={**dict(record), "record_kind": "tracker_timing"},
        )

    @staticmethod
    def _build_servo_command_sample(
        *,
        record: dict[str, Any],
        run_start_ns: int,
        sample_index: int,
    ) -> ExperimentTimeseriesSample:
        command_ns = int(record.get("command_monotonic_ns") or run_start_ns)
        monotonic_time_s = max(0.0, float(command_ns - int(run_start_ns)) / 1_000_000_000.0)
        servo_id = record.get("servo_id")
        commanded_motor_values = {}
        if servo_id not in (None, ""):
            commanded_motor_values["servo_id"] = int(servo_id)
        if record.get("commanded_position_ticks") is not None:
            commanded_motor_values["commanded_position_ticks"] = int(record["commanded_position_ticks"])
        status_flags = ["servo_command"]
        if bool(record.get("error_flag", False)):
            status_flags.append("servo_error")
        if bool(record.get("warmup_discarded", False)):
            status_flags.append("warmup_discarded")
        return ExperimentTimeseriesSample(
            monotonic_time_s=monotonic_time_s,
            wall_time_utc=_utc_now_iso(),
            phase="servo_command",
            step_index=int(record.get("command_step_index") if record.get("command_step_index") is not None else sample_index),
            sample_index=int(sample_index),
            tracker_frame_id=-1,
            commanded_motor_values=commanded_motor_values,
            status_flags=status_flags,
            backend_health={"source": "servo_service"},
            extra={**dict(record), "record_kind": "servo_command"},
        )

    @staticmethod
    def _motion_metric_summary(
        *,
        samples,
        requested_tool_ids: list[str],
        prefer_robot_frame_tip: bool,
    ) -> dict[str, Any]:
        reference_position: np.ndarray | None = None
        source_label = "unavailable"
        displacements_mm: list[float] = []
        sample_count = 0
        for sample in samples:
            extra = dict(getattr(sample, "extra", {}) or {})
            if str(extra.get("record_kind", "")).strip().lower() != "tracker_timing":
                continue
            if bool(extra.get("warmup_discarded", False)):
                continue
            position_mm: list[float] | None = None
            if prefer_robot_frame_tip:
                tip_payload = dict(getattr(sample, "pose_in_robot_frame", {}) or {}).get("tip", {}) or {}
                translation = tip_payload.get("translation_mm")
                if isinstance(translation, list) and len(translation) == 3:
                    position_mm = [float(value) for value in translation]
                    source_label = "robot_tip_translation_mm"
            if position_mm is None:
                for tool_id in requested_tool_ids:
                    tool_payload = dict(getattr(sample, "pose_in_tracker_frame", {}) or {}).get(str(tool_id), {}) or {}
                    translation = tool_payload.get("translation_mm")
                    if isinstance(translation, list) and len(translation) == 3:
                        position_mm = [float(value) for value in translation]
                        source_label = f"tracker_tool_{tool_id}_translation_mm"
                        break
            if position_mm is None:
                continue
            vector = np.asarray(position_mm, dtype=float)
            if reference_position is None:
                reference_position = vector
            displacements_mm.append(float(np.linalg.norm(vector - reference_position)))
            sample_count += 1
        return {
            "source": source_label,
            "sample_count": int(sample_count),
            "max_displacement_mm": max(displacements_mm) if displacements_mm else None,
        }

    @staticmethod
    def _effective_tracker_commit_ns(record: dict[str, Any], *, fallback_ns: int) -> int:
        commit_ns = record.get("sample_commit_monotonic_ns") or record.get("state_commit_complete_ns")
        received_ns = record.get("received_monotonic_ns")
        if commit_ns is None:
            return int(received_ns if received_ns is not None else fallback_ns)
        commit_value = int(commit_ns)
        if received_ns is None:
            return commit_value
        # Some backends/tests provide synthetic timing-domain values; if commit and received
        # timestamps diverge materially, trust the host-received monotonic timestamp.
        if abs(commit_value - int(received_ns)) > 250_000_000:
            return int(received_ns)
        return commit_value


class PretensionValidationExperiment(BaseExperiment):
    """Capture one pretension response trace versus commanded travel."""

    name = "pretension_validation"
    description = "Validate pretension response versus commanded travel using current as an engagement proxy."
    hardware_requirements = ExperimentHardwareRequirements(
        servo_required=True,
        tracking_required=False,
        mock_compatible=True,
    )

    def __init__(self, config: PretensionValidationExperimentConfig) -> None:
        super().__init__(config=config)
        self._tracking_started_here = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "PretensionValidationExperiment":
        return cls(config=PretensionValidationExperimentConfig.from_dict(payload))

    def setup(self, session: ExperimentSession) -> None:
        tracking_service = session.context.tracking_service
        if (
            bool(self.config.include_tracker_displacement)
            and tracking_service is not None
            and getattr(tracking_service, "_thread", None) is None
        ):
            tracking_service.start()
            self._tracking_started_here = True

    def precheck(self, session: ExperimentSession) -> None:
        if not session.context.servo_service.is_connected:
            raise RuntimeError("pretension_validation requires a connected servo service.")
        tracker_service = session.context.tracking_service
        if bool(self.config.include_tracker_displacement) and tracker_service is not None:
            snapshot_reader = getattr(tracker_service, "peek_snapshot", None)
            snapshot = snapshot_reader() if callable(snapshot_reader) else tracker_service.get_snapshot()
            if (
                _pretension_current_only_explicit(self.config)
                and str(getattr(snapshot, "canonical_state", "") or "") not in {"mock", "connected", "streaming_healthy", "streaming_degraded"}
            ):
                session.add_warning(
                    "Tracker is not connected. Pretension validation is explicitly allowed as current-only/lower-trust testing."
                )
                session.metadata.registration_info["runtime_tip_policy"] = None
                return
            runtime_tip_policy = evaluate_runtime_tip_trust(
                snapshot=snapshot,
                workflow=WORKFLOW_PRETENSION_VALIDATION,
                allow_lower_trust=True,
            )
            session.metadata.registration_info["runtime_tip_policy"] = runtime_tip_policy.to_dict()
            if self._is_staged_mode() and bool(self.config.enable_tip_centering) and not runtime_tip_policy.allowed_for_workflow:
                raise RuntimeError(
                    "Pretension tip centering requires an allowed runtime tip policy outcome; "
                    f"mode={runtime_tip_policy.mode}, trust={runtime_tip_policy.trust_label}, "
                    f"reasons={runtime_tip_policy.reasons or ['policy_not_allowed']}."
                )
            if runtime_tip_policy.allowed_for_workflow and not runtime_tip_policy.thesis_trusted:
                session.add_warning(
                    "Pretension validation is using a lower-trust runtime tip path: "
                    f"mode={runtime_tip_policy.mode}, trust={runtime_tip_policy.trust_label}."
                )

    def execute(self, session: ExperimentSession) -> None:
        if self._is_staged_mode():
            self._execute_single_segment_staged(session)
            return
        servo_service = session.context.servo_service
        tracker_service = session.context.tracking_service
        validation_service = servo_service.pretension_validation
        parameters = self._pretension_parameters(session)
        mapper = getattr(servo_service, "mapper", None)
        include_tracker = bool(self.config.include_tracker_displacement and tracker_service is not None)
        if bool(self.config.include_tracker_displacement) and tracker_service is None:
            session.add_warning(
                "Tracker displacement was requested, but tracking_service is unavailable. "
                "This run will save current-versus-travel data only."
            )
        runtime_tip_policy = (
            evaluate_runtime_tip_trust(
                snapshot=(
                    tracker_service.peek_snapshot()
                    if hasattr(tracker_service, "peek_snapshot")
                    else tracker_service.get_snapshot()
                ),
                workflow=WORKFLOW_PRETENSION_VALIDATION,
                allow_lower_trust=True,
            )
            if tracker_service is not None
            else None
        )

        sample_index = 0
        progress_index = 0
        estimated_total = max(4, int(parameters.max_travel_ticks // max(1, parameters.step_ticks)) + 4)
        start_snapshot = tracker_service.get_snapshot() if include_tracker else None

        if self.config.move_to_reference:
            move_result = servo_service.move_servo_to_pretension_reference(
                servo_id=int(self.config.servo_id),
                parameters=parameters,
            )
            if move_result.blocked or not move_result.success:
                raise RuntimeError(move_result.message)
            start_snapshot = tracker_service.get_snapshot() if include_tracker else start_snapshot
            move_payload = self._trace_payload(
                servo_id=int(self.config.servo_id),
                run_state="move_to_reference",
                commanded_position_ticks=move_result.goal_tick,
                current_position_ticks=(
                    move_result.telemetry.present_position if move_result.telemetry is not None else None
                ),
                raw_current_ma=(
                    move_result.telemetry.present_current_ma if move_result.telemetry is not None else None
                ),
                filtered_current_ma=(
                    float(move_result.telemetry.present_current_ma)
                    if move_result.telemetry is not None and move_result.telemetry.present_current_ma is not None
                    else None
                ),
                baseline_current_ma=None,
                effective_trigger_current_ma=None,
                hard_current_stop_ma=int(parameters.hard_current_stop_ma),
                untensioned_reference_tick=int(parameters.untensioned_reference_tick),
                trigger_met=False,
                stop_reason=None,
                tracker_metric=(
                    validation_service.compute_tracker_displacement(
                        start_snapshot,
                        start_snapshot,
                        tracker_tool_id=self.config.tracker_tool_id,
                    )
                    if include_tracker and start_snapshot is not None
                    else None
                ),
                mapper=mapper,
            )
            session.add_sample(
                self._build_trace_sample(
                    session,
                    snapshot=start_snapshot if include_tracker else None,
                    phase="move_to_reference",
                    step_index=0,
                    sample_index=sample_index,
                    payload=move_payload,
                )
            )
            sample_index += 1

        def _on_progress(result) -> None:
            nonlocal sample_index, progress_index
            tracker_snapshot = tracker_service.get_snapshot() if include_tracker else None
            tracker_metric = (
                validation_service.compute_tracker_displacement(
                    start_snapshot,
                    tracker_snapshot,
                    tracker_tool_id=self.config.tracker_tool_id,
                )
                if include_tracker and start_snapshot is not None and tracker_snapshot is not None
                else None
            )
            phase = self._phase_for_result_status(str(result.status))
            payload = self._trace_payload(
                servo_id=int(self.config.servo_id),
                run_state=str(result.status),
                commanded_position_ticks=result.last_commanded_target_tick,
                current_position_ticks=result.current_position_tick,
                raw_current_ma=result.final_current_ma,
                filtered_current_ma=result.filtered_current_ma,
                baseline_current_ma=result.baseline_current_ma,
                effective_trigger_current_ma=result.threshold_ma,
                hard_current_stop_ma=result.hard_current_stop_ma,
                untensioned_reference_tick=result.untensioned_reference_tick,
                trigger_met=bool(result.success and result.stop_reason in {"baseline_delta_trigger", "absolute_trigger", "combined_trigger"}),
                stop_reason=result.stop_reason,
                tracker_metric=tracker_metric,
                mapper=mapper,
            )
            session.add_sample(
                self._build_trace_sample(
                    session,
                    snapshot=tracker_snapshot if include_tracker else None,
                    phase=phase,
                    step_index=progress_index,
                    sample_index=sample_index,
                    payload=payload,
                )
            )
            sample_index += 1
            progress_index += 1
            session.update_progress(
                min(progress_index, estimated_total),
                estimated_total,
                {
                    "phase": phase,
                    "servo_id": int(self.config.servo_id),
                    "run_state": str(result.status),
                },
            )

        pretension_result = servo_service.run_pretension_routine(
            servo_id=int(self.config.servo_id),
            parameters=parameters,
            progress_callback=_on_progress,
            stop_requested=session.stop_requested,
        )

        trace_points = [
            sample.extra
            for sample in session.samples
            if sample.phase in {"move_to_reference", "pretension_baseline", "pretension_step", "pretension_result"}
        ]
        tracker_displacements = [
            float(point["tracker_displacement_mm"])
            for point in trace_points
            if point.get("tracker_displacement_mm") is not None
        ]
        max_observed_current_ma = max(
            (
                float(point["raw_current_ma"])
                for point in trace_points
                if point.get("raw_current_ma") is not None
            ),
            default=None,
        )
        max_observed_filtered_current_ma = max(
            (
                float(point["filtered_current_ma"])
                for point in trace_points
                if point.get("filtered_current_ma") is not None
            ),
            default=None,
        )
        trigger_point = next((point for point in trace_points if bool(point.get("trigger_met"))), None)
        travel_used_ticks = (
            None
            if pretension_result.final_position_tick is None or pretension_result.untensioned_reference_tick is None
            else int(pretension_result.untensioned_reference_tick) - int(pretension_result.final_position_tick)
        )
        travel_used_mm = (
            float(mapper.ticks_to_displacement_mm(travel_used_ticks))
            if travel_used_ticks is not None and mapper is not None
            else None
        )
        metrics = {
            "servo_id": int(self.config.servo_id),
            "accepted": bool(pretension_result.success),
            "pretension_success": bool(pretension_result.success),
            "stop_reason": pretension_result.stop_reason,
            "status": pretension_result.status,
            "final_position_tick": pretension_result.final_position_tick,
            "travel_used_ticks": travel_used_ticks,
            "travel_used_mm": travel_used_mm,
            "baseline_current_ma": pretension_result.baseline_current_ma,
            "effective_trigger_current_ma": pretension_result.threshold_ma,
            "trigger_current_ma": (
                pretension_result.filtered_current_ma if pretension_result.success else None
            ),
            "final_raw_current_ma": pretension_result.final_current_ma,
            "final_filtered_current_ma": pretension_result.filtered_current_ma,
            "hard_current_stop_ma": pretension_result.hard_current_stop_ma,
            "max_observed_current_ma": max_observed_current_ma,
            "max_observed_filtered_current_ma": max_observed_filtered_current_ma,
            "max_observed_displacement_mm": max(tracker_displacements) if tracker_displacements else None,
            "trigger_displacement_mm": (
                float(trigger_point["tracker_displacement_mm"])
                if trigger_point is not None and trigger_point.get("tracker_displacement_mm") is not None
                else None
            ),
            "tracker_metric_frame": (
                trigger_point.get("tracker_metric_frame")
                if trigger_point is not None and trigger_point.get("tracker_metric_frame")
                else next(
                    (
                        point.get("tracker_metric_frame")
                        for point in trace_points
                        if point.get("tracker_metric_frame") not in (None, "", "unavailable", "frame_mismatch")
                    ),
                    "unavailable",
                )
            ),
            "tracker_metric_sample_count": len(tracker_displacements),
            "runtime_tip_policy": runtime_tip_policy.to_dict() if runtime_tip_policy is not None else None,
            "runtime_tip_mode_used": runtime_tip_policy.mode if runtime_tip_policy is not None else None,
            "runtime_tip_trust_level": runtime_tip_policy.trust_label if runtime_tip_policy is not None else "unavailable",
            "thesis_trusted_runtime_tip": bool(runtime_tip_policy.thesis_trusted) if runtime_tip_policy is not None else False,
            "trace_sample_count": len(trace_points),
            "parameters": {
                "servo_id": int(self.config.servo_id),
                "move_to_reference": bool(self.config.move_to_reference),
                "include_tracker_displacement": bool(self.config.include_tracker_displacement),
                "tracker_tool_id": str(self.config.tracker_tool_id),
                "pretension_start_mode": str(parameters.start_mode),
                "untensioned_reference_tick": int(parameters.untensioned_reference_tick),
                "step_ticks": int(parameters.step_ticks),
                "settle_time_s": float(parameters.settle_time_s),
                "baseline_sample_count": int(parameters.baseline_sample_count),
                "current_filter_window": int(parameters.current_filter_window),
                "current_delta_threshold_ma": int(parameters.current_delta_threshold_ma),
                "absolute_trigger_current_ma": parameters.absolute_trigger_current_ma,
                "hard_current_stop_ma": int(parameters.hard_current_stop_ma),
                "max_travel_ticks": int(parameters.max_travel_ticks),
                "timeout_s": float(parameters.timeout_s),
            },
        }
        if not pretension_result.success:
            session.add_warning(
                f"Pretension validation ended with {pretension_result.status}: "
                f"{pretension_result.stop_reason or pretension_result.message}"
            )
            metrics["summary_requirements"] = {"force_status": "partial_success"}
        session.metrics.update(metrics)

    def finalize(self, session: ExperimentSession) -> None:
        if self._tracking_started_here:
            session.context.tracking_service.stop()
            self._tracking_started_here = False

    def write_outputs(self, session: ExperimentSession, paths, summary) -> None:
        write_pretension_validation_outputs(
            output_dir=paths.output_dir,
            metadata=session.metadata,
            summary=summary,
            samples=session.samples,
        )
        self._write_pretension_debug_export_bundle(session=session, paths=paths, summary=summary)

    def _is_staged_mode(self) -> bool:
        return str(getattr(self.config, "mode", "single_servo_trace") or "").strip().lower() in {
            "single_segment_staged",
            "staged",
            "four_servo_staged",
            "single_segment_characterization",
            "pretension_characterization",
            "characterization",
            "advanced_startup",
            "conservative_startup",
        }

    def _execute_single_segment_staged(self, session: ExperimentSession) -> None:
        strategy = str(getattr(self.config, "staged_strategy", "conservative_startup") or "").strip().lower()
        if strategy != "legacy":
            self._execute_reliable_staged_pretension(session)
            return
        if not bool(getattr(self.config, "allow_legacy_staged_strategy", False)):
            raise RuntimeError(
                "Legacy staged pretension is disabled for normal operation because it can create "
                "one-sided tendon loading. Use conservative staged pretension or manual startup."
            )
        self._execute_legacy_single_segment_staged(session)

    def _execute_reliable_staged_pretension(self, session: ExperimentSession) -> None:
        """Run the evidence-first 4-servo pretension workflow.

        This path intentionally avoids the selected-servo current-threshold take-up loop.
        Current is characterized first, then used only as a load proxy for small
        paired moves while tracker centering remains the primary objective.
        """
        servo_service = session.context.servo_service
        tracker_service = session.context.tracking_service
        operating_context = session.context.settings.robot.operating_context()
        if operating_context.operating_mode != "single_segment":
            if operating_context.operating_mode == "dual_segment":
                raise RuntimeError(
                    "dual_segment currently supports all-8 readiness and manual startup capture. "
                    "Automatic two-segment pretension/control is not implemented yet."
                )
            raise RuntimeError(
                "Automatic staged pretension currently requires operating_mode=single_segment. "
                "Select Segment A or Segment B."
            )
        include_tracker = bool(self.config.include_tracker_displacement and tracker_service is not None)
        if not bool(self.config.include_tracker_displacement) and not _pretension_current_only_explicit(self.config):
            raise RuntimeError(
                "Current-only staged pretension requires an explicit lower-trust/current-only override. "
                "Enable allow_current_only_when_tracker_missing or set run_trust_mode=current_only."
            )
        if include_tracker:
            tracker_snapshot = _optional_tracking_snapshot(tracker_service)
            if tracker_snapshot is None or str(getattr(tracker_snapshot, "canonical_state", "") or "") not in {"mock", "connected", "streaming_healthy", "streaming_degraded"}:
                include_tracker = False
                if not _pretension_current_only_explicit(self.config):
                    raise RuntimeError("Tracker tip pose is required for advanced pretension, but tracking is not connected.")
                session.add_warning(
                    "Tracker is not connected; advanced pretension will run in explicitly current-only lower-trust mode. "
                    "Tip-centering and centered-startup claims are disabled."
                )
        if bool(self.config.include_tracker_displacement) and tracker_service is None:
            if not _pretension_current_only_explicit(self.config):
                raise RuntimeError("Tracker tip pose is required for advanced pretension, but tracking_service is unavailable.")
            session.add_warning(
                "Tracker tip pose is unavailable; advanced pretension will run in explicitly current-only lower-trust mode. "
                "Tip-centering and centered-startup claims are disabled."
            )
        if include_tracker and tracker_service.get_snapshot() is None:
            include_tracker = False
            if not _pretension_current_only_explicit(self.config):
                raise RuntimeError("Tracker snapshot is unavailable for advanced pretension.")
            session.add_warning(
                "Tracker snapshot is unavailable; advanced pretension will run in explicitly current-only lower-trust mode. "
                "Tip-centering and centered-startup claims are disabled."
            )

        servo_ids = self._staged_servo_ids(session)
        repeat_runs = max(1, int(self.config.repeat_runs))
        mode_kind = self._reliable_pretension_mode()
        total_progress = max(1, repeat_runs * (5 if mode_kind == "characterization" else 6))
        progress = 0
        run_rows: list[dict[str, Any]] = []
        trace_rows: list[dict[str, Any]] = []
        failure_counts: dict[str, int] = {}
        final_tip_xy_points_mm: list[list[float]] = []
        quality_scores: list[float] = []
        accepted_runs = 0
        manual_startup_artifact = self._manual_startup_artifact_snapshot(servo_service, servo_ids)
        manual_baseline_records: list[dict[str, Any]] = self._load_or_capture_manual_baselines(
            session=session,
            servo_service=servo_service,
            tracker_service=tracker_service if include_tracker else None,
            servo_ids=servo_ids,
        )

        with servo_service.exclusive_bus_operation(
            owner="pretension_validation",
            servo_id=None,
            reason=f"advanced pretension {mode_kind}",
        ):
            for run_index in range(repeat_runs):
                session.raise_if_stop_requested()
                target_xy = self._target_xy()
                run_label = f"run_{run_index + 1:02d}"
                missing_fields: list[str] = []
                reject_reasons: list[str] = []
                clipped_move_count = 0
                correction_move_count = 0
                correction_travel_ticks = 0
                stop_reason = ""
                converged = False
                telemetry_event_counts: dict[str, int] = {}
                packet_retry_count = 0
                fail_closed_stop_reason: str | None = None
                trace_start_index = len(trace_rows)
                run_deadline_monotonic = (
                    float(session.context.monotonic_fn())
                    + max(1.0, float(getattr(self.config, "max_runtime_s", 60.0)))
                )

                initial = self._advanced_measurement(
                    servo_service=servo_service,
                    tracker_service=tracker_service if include_tracker else None,
                    servo_ids=servo_ids,
                    baseline_current_ma_by_servo={},
                    target_xy_mm=target_xy,
                    trust_status=("runtime_tip" if include_tracker else "current_only_lower_trust"),
                )
                start_position_ticks = dict(initial["measured_positions_ticks"])
                startup_reference_ticks = dict(start_position_ticks)
                missing_fields.extend(initial["missing_fields"])
                self._merge_event_counts(telemetry_event_counts, initial.get("telemetry_event_counts"))
                packet_retry_count += int(initial.get("packet_retry_count", 0) or 0)
                if initial.get("telemetry_fail_closed_reason"):
                    fail_closed_stop_reason = str(initial["telemetry_fail_closed_reason"])
                    reject_reasons.append(fail_closed_stop_reason)
                row = self._advanced_stage_row(
                    mode_kind=mode_kind,
                    run_index=run_index,
                    stage="initial_state",
                    target_xy_mm=target_xy,
                    measurement=initial,
                    extra={"manual_startup_artifact": manual_startup_artifact},
                )
                trace_rows.append(row)
                self._add_staged_sample(session, phase="pretension_stage_initial", run_index=run_index, step_index=len(trace_rows) - 1, payload=row)
                progress += 1
                session.update_progress(progress, total_progress, {"phase": "initial_state", "run_index": run_index})

                start_mode_used = self._resolved_staged_start_mode(session, servo_ids)
                if start_mode_used == "full_release_4095":
                    start_result = self._run_explicit_full_release_start(
                        session=session,
                        servo_service=servo_service,
                        tracker_service=tracker_service if include_tracker else None,
                        servo_ids=servo_ids,
                        run_index=run_index,
                        target_xy_mm=target_xy,
                        startup_reference_ticks_by_servo=startup_reference_ticks,
                        mode_kind=mode_kind,
                        trace_rows=trace_rows,
                    )
                else:
                    start_result = self._run_soft_release_start(
                        session=session,
                        servo_service=servo_service,
                        tracker_service=tracker_service if include_tracker else None,
                        servo_ids=servo_ids,
                        run_index=run_index,
                        target_xy_mm=target_xy,
                        startup_reference_ticks_by_servo=startup_reference_ticks,
                        mode_kind=mode_kind,
                        trace_rows=trace_rows,
                        deadline_monotonic=run_deadline_monotonic,
                    )
                clipped_move_count += int(start_result.get("clipped_move_count", 0))
                correction_move_count += int(start_result.get("move_count", 0))
                correction_travel_ticks += int(start_result.get("travel_ticks", 0))
                self._merge_event_counts(telemetry_event_counts, start_result.get("telemetry_event_counts"))
                packet_retry_count += int(start_result.get("packet_retry_count", 0) or 0)
                if start_result.get("stop_reason") not in (None, "", "soft_release_complete", "full_release_complete"):
                    fail_closed_stop_reason = str(start_result["stop_reason"])
                    reject_reasons.append(fail_closed_stop_reason)
                after_start = self._advanced_measurement(
                    servo_service=servo_service,
                    tracker_service=tracker_service if include_tracker else None,
                    servo_ids=servo_ids,
                    baseline_current_ma_by_servo={},
                    target_xy_mm=target_xy,
                    startup_reference_ticks_by_servo=startup_reference_ticks,
                    trust_status=("runtime_tip" if include_tracker else "current_only_lower_trust"),
                )
                self._merge_event_counts(telemetry_event_counts, after_start.get("telemetry_event_counts"))
                packet_retry_count += int(after_start.get("packet_retry_count", 0) or 0)
                if after_start.get("telemetry_fail_closed_reason"):
                    fail_closed_stop_reason = str(after_start["telemetry_fail_closed_reason"])
                    reject_reasons.append(fail_closed_stop_reason)
                row = self._advanced_stage_row(
                    mode_kind=mode_kind,
                    run_index=run_index,
                    stage="low_load_start",
                    target_xy_mm=target_xy,
                    measurement=after_start,
                    extra={
                        "startup_source": start_mode_used,
                        "start_result": start_result,
                        "clipped_move_count": clipped_move_count,
                    },
                )
                trace_rows.append(row)
                self._add_staged_sample(session, phase="pretension_stage_low_load_start", run_index=run_index, step_index=len(trace_rows) - 1, payload=row)
                progress += 1
                session.update_progress(progress, total_progress, {"phase": "low_load_start", "run_index": run_index})

                current_characterization = self._characterize_current_noise(
                    session=session,
                    servo_service=servo_service,
                    tracker_service=tracker_service if include_tracker else None,
                    servo_ids=servo_ids,
                    run_index=run_index,
                    target_xy_mm=target_xy,
                    mode_kind=mode_kind,
                    trace_rows=trace_rows,
                    startup_reference_ticks_by_servo=startup_reference_ticks,
                    trust_status=("runtime_tip" if include_tracker else "current_only_lower_trust"),
                )
                self._merge_event_counts(telemetry_event_counts, current_characterization.get("telemetry_event_counts"))
                packet_retry_count += int(current_characterization.get("packet_retry_count", 0) or 0)
                if current_characterization.get("telemetry_fail_closed_reason"):
                    fail_closed_stop_reason = str(current_characterization["telemetry_fail_closed_reason"])
                    reject_reasons.append(fail_closed_stop_reason)
                baseline_current_ma_by_servo = {
                    int(servo_id): float(stats["mean_current_ma"])
                    for servo_id, stats in current_characterization["by_servo"].items()
                    if stats.get("mean_current_ma") is not None
                }
                effective_load_tolerance_ma = max(
                    float(self.config.load_balance_tolerance_ma),
                    float(current_characterization.get("max_useful_current_delta_ma") or 0.0),
                )
                progress += 1
                session.update_progress(progress, total_progress, {"phase": "current_characterization", "run_index": run_index})

                if fail_closed_stop_reason:
                    mode_result = {
                        "stop_reason": fail_closed_stop_reason,
                        "move_count": 0,
                        "travel_ticks": 0,
                        "clipped_move_count": 0,
                        "converged": False,
                    }
                    stop_reason = fail_closed_stop_reason
                    converged = False
                elif mode_kind == "characterization":
                    mode_result = self._run_pair_characterization_sequence(
                        session=session,
                        servo_service=servo_service,
                        tracker_service=tracker_service if include_tracker else None,
                        servo_ids=servo_ids,
                        run_index=run_index,
                        target_xy_mm=target_xy,
                        baseline_current_ma_by_servo=baseline_current_ma_by_servo,
                        trace_rows=trace_rows,
                        startup_reference_ticks_by_servo=startup_reference_ticks,
                        deadline_monotonic=run_deadline_monotonic,
                    )
                    clipped_move_count += int(mode_result.get("clipped_move_count", 0))
                    correction_move_count += int(mode_result.get("move_count", 0))
                    correction_travel_ticks += int(mode_result.get("travel_ticks", 0))
                    stop_reason = str(mode_result.get("stop_reason") or "characterization_complete")
                    converged = False
                else:
                    takeup_result = self._run_symmetric_paired_takeup(
                        session=session,
                        servo_service=servo_service,
                        tracker_service=tracker_service if include_tracker else None,
                        servo_ids=servo_ids,
                        run_index=run_index,
                        target_xy_mm=target_xy,
                        baseline_current_ma_by_servo=baseline_current_ma_by_servo,
                        startup_reference_ticks_by_servo=startup_reference_ticks,
                        effective_load_tolerance_ma=effective_load_tolerance_ma,
                        trace_rows=trace_rows,
                        deadline_monotonic=run_deadline_monotonic,
                    )
                    clipped_move_count += int(takeup_result.get("clipped_move_count", 0))
                    correction_move_count += int(takeup_result.get("move_count", 0))
                    correction_travel_ticks += int(takeup_result.get("travel_ticks", 0))
                    self._merge_event_counts(telemetry_event_counts, takeup_result.get("telemetry_event_counts"))
                    packet_retry_count += int(takeup_result.get("packet_retry_count", 0) or 0)
                    if takeup_result.get("stop_reason") not in (
                        None,
                        "",
                        "takeup_complete",
                        "within_load_target",
                        "high_load_transition_to_trim",
                    ):
                        reject_reasons.append(str(takeup_result["stop_reason"]))
                    variant = str(
                        getattr(self.config, "tip_centering_variant", PRETENSION_TIP_VARIANT_PAIRED_THEN_TIP)
                        or PRETENSION_TIP_VARIANT_PAIRED_THEN_TIP
                    ).strip().lower()
                    if variant not in PRETENSION_TIP_VARIANT_OPTIONS:
                        variant = PRETENSION_TIP_VARIANT_PAIRED_THEN_TIP
                    if variant == PRETENSION_TIP_VARIANT_CURRENT_ONLY or not include_tracker:
                        # Stop here. Current take-up already reached the target
                        # band; without tracker we cannot reliably center the
                        # tip. Mark as 'converged' if the takeup phase reported
                        # a successful stop reason.
                        mode_result = {
                            "stop_reason": "current_only_complete"
                            if variant == PRETENSION_TIP_VARIANT_CURRENT_ONLY
                            else "current_only_no_tracker",
                            "move_count": 0,
                            "travel_ticks": 0,
                            "clipped_move_count": 0,
                            "converged": variant == PRETENSION_TIP_VARIANT_CURRENT_ONLY,
                            "packet_retry_count": 0,
                            "telemetry_event_counts": {},
                            "variant": variant,
                        }
                    elif variant == PRETENSION_TIP_VARIANT_JACOBIAN_LEARNED_TIP:
                        mode_result = self._run_jacobian_tip_centering(
                            session=session,
                            servo_service=servo_service,
                            tracker_service=tracker_service,
                            servo_ids=servo_ids,
                            run_index=run_index,
                            target_xy_mm=target_xy,
                            baseline_current_ma_by_servo=baseline_current_ma_by_servo,
                            effective_load_tolerance_ma=effective_load_tolerance_ma,
                            trace_rows=trace_rows,
                            startup_reference_ticks_by_servo=startup_reference_ticks,
                            deadline_monotonic=run_deadline_monotonic,
                        )
                        mode_result["variant"] = variant
                    else:  # PRETENSION_TIP_VARIANT_PAIRED_THEN_TIP (default)
                        mode_result = self._run_conservative_startup_sequence(
                            session=session,
                            servo_service=servo_service,
                            tracker_service=tracker_service if include_tracker else None,
                            servo_ids=servo_ids,
                            run_index=run_index,
                            target_xy_mm=target_xy,
                            baseline_current_ma_by_servo=baseline_current_ma_by_servo,
                            effective_load_tolerance_ma=effective_load_tolerance_ma,
                            trace_rows=trace_rows,
                            startup_reference_ticks_by_servo=startup_reference_ticks,
                            deadline_monotonic=run_deadline_monotonic,
                        )
                        mode_result["variant"] = variant
                    clipped_move_count += int(mode_result.get("clipped_move_count", 0))
                    correction_move_count += int(mode_result.get("move_count", 0))
                    correction_travel_ticks += int(mode_result.get("travel_ticks", 0))
                    self._merge_event_counts(telemetry_event_counts, mode_result.get("telemetry_event_counts"))
                    packet_retry_count += int(mode_result.get("packet_retry_count", 0) or 0)
                    stop_reason = str(mode_result.get("stop_reason") or "")
                    converged = bool(mode_result.get("converged", False))
                    if stop_reason and stop_reason not in {
                        "converged",
                        "within_tolerance",
                        "current_only_complete",
                        "jacobian_converged",
                    }:
                        reject_reasons.append(stop_reason)
                progress += 1
                session.update_progress(progress, total_progress, {"phase": mode_kind, "run_index": run_index})

                pre_settle = self._advanced_measurement(
                    servo_service=servo_service,
                    tracker_service=tracker_service if include_tracker else None,
                    servo_ids=servo_ids,
                    baseline_current_ma_by_servo=baseline_current_ma_by_servo,
                    target_xy_mm=target_xy,
                    startup_reference_ticks_by_servo=startup_reference_ticks,
                    trust_status=("runtime_tip" if include_tracker else "current_only_lower_trust"),
                )
                if float(self.config.settle_verify_time_s) > 0.0:
                    session.context.sleep_fn(float(self.config.settle_verify_time_s))
                final = self._advanced_measurement(
                    servo_service=servo_service,
                    tracker_service=tracker_service if include_tracker else None,
                    servo_ids=servo_ids,
                    baseline_current_ma_by_servo=baseline_current_ma_by_servo,
                    target_xy_mm=target_xy,
                    startup_reference_ticks_by_servo=startup_reference_ticks,
                    trust_status=("runtime_tip" if include_tracker else "current_only_lower_trust"),
                )
                self._merge_event_counts(telemetry_event_counts, pre_settle.get("telemetry_event_counts"))
                self._merge_event_counts(telemetry_event_counts, final.get("telemetry_event_counts"))
                packet_retry_count += int(pre_settle.get("packet_retry_count", 0) or 0)
                packet_retry_count += int(final.get("packet_retry_count", 0) or 0)
                missing_fields.extend(final["missing_fields"])
                if final.get("telemetry_fail_closed_reason"):
                    reject_reasons.append(str(final["telemetry_fail_closed_reason"]))
                settle_tip_drift_mm = self._distance_mm(pre_settle.get("tip_xyz_mm"), final.get("tip_xyz_mm"))
                final_position_ticks = dict(final["measured_positions_ticks"])
                final_currents_ma = dict(final["raw_current_ma"])
                current_above_baseline_ma = dict(final["current_above_baseline_ma"])
                tip_xy_offset_mm = final.get("tip_xy_error_mm")
                if final.get("tip_xy_mm") is not None:
                    final_tip_xy_points_mm.append(list(final["tip_xy_mm"]))
                if missing_fields:
                    reject_reasons.append("missing_telemetry")
                if mode_kind != "characterization" and not converged:
                    reject_reasons.append("not_converged")
                if (
                    mode_kind != "characterization"
                    and include_tracker
                    and tip_xy_offset_mm is not None
                    and float(tip_xy_offset_mm) > float(self.config.accept_max_final_tip_xy_offset_mm)
                ):
                    reject_reasons.append("tip_center_error")
                quality_flags: list[str] = []
                if (
                    mode_kind != "characterization"
                    and final.get("load_balance_error_ma") is not None
                    and float(final["load_balance_error_ma"]) > effective_load_tolerance_ma
                ):
                    quality_flags.append("load_proxy_imbalance_flagged")
                if (
                    mode_kind != "characterization"
                    and final.get("load_balance_error_ma") is not None
                    and float(final["load_balance_error_ma"]) > float(self.config.accept_max_load_balance_error_ma)
                ):
                    reject_reasons.append("load_balance_error")
                if mode_kind != "characterization" and not include_tracker:
                    reject_reasons.append("current_only_lower_trust")
                if include_tracker and tip_xy_offset_mm is None and not bool(self.config.allow_current_only_when_tracker_missing):
                    reject_reasons.append("missing_tip_pose")
                reject_reasons = sorted(set(reason for reason in reject_reasons if reason))

                quality_components, quality_score = self._staged_quality_score(
                    tip_xy_offset_mm=tip_xy_offset_mm,
                    settle_tip_drift_mm=settle_tip_drift_mm,
                    load_balance_error_ma=final.get("load_balance_error_ma"),
                    pair_balance_error_ma=final.get("pair_balance_error_ma"),
                    final_position_ticks=final_position_ticks,
                    start_position_ticks=start_position_ticks,
                    missing_fields=missing_fields,
                    clipped_move_count=clipped_move_count,
                    correction_move_count=correction_move_count,
                    correction_travel_ticks=correction_travel_ticks,
                    takeup_success=bool(converged or mode_kind == "characterization"),
                )
                accepted = bool(mode_kind == "characterization" or not reject_reasons)
                if accepted:
                    accepted_runs += 1
                if quality_score is not None:
                    quality_scores.append(float(quality_score))
                for reason in reject_reasons:
                    failure_counts[reason] = int(failure_counts.get(reason, 0)) + 1

                run_trace_rows = trace_rows[trace_start_index:]
                # Flatten the per-servo position/current dicts for the comparison
                # report. The comparison helper expects integer-keyed dicts; the
                # string-keyed dicts above are kept for legacy schema readers.
                positions_int_keyed = {
                    int(k): (int(v) if v is not None else None)
                    for k, v in final_position_ticks.items()
                }
                currents_int_keyed = {
                    int(k): (float(v) if v is not None else None)
                    for k, v in final_currents_ma.items()
                }
                run_row = {
                    "run_index": int(run_index),
                    "run_label": run_label,
                    "mode": mode_kind,
                    "variant": str(mode_result.get("variant") or ""),
                    "accepted": bool(accepted),
                    "reject_reasons": reject_reasons,
                    "stop_reason": stop_reason or ("characterization_complete" if mode_kind == "characterization" else "converged"),
                    "converged": bool(converged),
                    "servo_ids": list(servo_ids),
                    "target_xy_mm": list(target_xy),
                    "current_characterization": current_characterization,
                    "effective_load_tolerance_ma": float(effective_load_tolerance_ma),
                    "baseline_current_ma_by_servo": {str(k): float(v) for k, v in baseline_current_ma_by_servo.items()},
                    "final_current_ma_by_servo": {str(k): v for k, v in final_currents_ma.items()},
                    "current_above_baseline_ma_by_servo": {str(k): v for k, v in current_above_baseline_ma.items()},
                    "load_proxy_current_ma_by_servo": {str(k): v for k, v in current_above_baseline_ma.items()},
                    "start_position_ticks_by_servo": {str(k): v for k, v in start_position_ticks.items()},
                    "final_position_ticks_by_servo": {str(k): v for k, v in final_position_ticks.items()},
                    # Flattened convenience keys for the comparison helper.
                    "positions_by_servo": positions_int_keyed,
                    "currents_ma_by_servo": currents_int_keyed,
                    "final_tip_xy_mm": (
                        [float(final.get("tip_xy_mm")[0]), float(final.get("tip_xy_mm")[1])]
                        if isinstance(final.get("tip_xy_mm"), (list, tuple)) and len(final.get("tip_xy_mm")) >= 2
                        else None
                    ),
                    "startup_reference_ticks_by_servo": {str(k): v for k, v in startup_reference_ticks.items()},
                    "final_tendon_displacement_mm_by_servo": self._json_safe_keyed(final.get("tendon_displacement_mm") or {}),
                    "load_balance_error_ma": final.get("load_balance_error_ma"),
                    "pair_balance_error_ma": final.get("pair_balance_error_ma"),
                    "initial_tip_xyz_mm": initial.get("tip_xyz_mm"),
                    "pre_settle_tip_xyz_mm": pre_settle.get("tip_xyz_mm"),
                    "final_tip_xyz_mm": final.get("tip_xyz_mm"),
                    "final_tip_xy_offset_mm": tip_xy_offset_mm,
                    "settle_tip_drift_mm": settle_tip_drift_mm,
                    "clipped_move_count": int(clipped_move_count),
                    "correction_move_count": int(correction_move_count),
                    "correction_travel_ticks": int(correction_travel_ticks),
                    "quality_score_0_100": float(quality_score),
                    "quality_components": dict(quality_components),
                    "quality_flags": sorted(set(quality_flags)),
                    "good_but_flagged_reasons": (
                        sorted(set(quality_flags)) if accepted and quality_flags else []
                    ),
                    "missing_fields": sorted(set(missing_fields)),
                    "telemetry_event_counts": dict(sorted(telemetry_event_counts.items())),
                    "packet_retry_count": int(packet_retry_count),
                    "sign_flip_count": int(mode_result.get("sign_flip_count", 0) or 0),
                    "sign_flip_used_by_axis": dict(mode_result.get("sign_flip_used_by_axis") or {}),
                    "trust_status": "runtime_tip" if include_tracker else "current_only_lower_trust",
                    "startup_source": start_mode_used,
                    "trace_sample_count": len(run_trace_rows),
                }
                run_rows.append(run_row)
                row = self._advanced_stage_row(
                    mode_kind=mode_kind,
                    run_index=run_index,
                    stage="settle_verify",
                    target_xy_mm=target_xy,
                    measurement=final,
                    extra={
                        "accepted": bool(accepted),
                        "reject_reasons": reject_reasons,
                        "quality_score_0_100": float(quality_score),
                        "quality_components": dict(quality_components),
                        "quality_flags": sorted(set(quality_flags)),
                        "good_but_flagged_reasons": run_row["good_but_flagged_reasons"],
                        "settle_tip_drift_mm": settle_tip_drift_mm,
                        "stop_reason": run_row["stop_reason"],
                        "telemetry_event_counts": dict(sorted(telemetry_event_counts.items())),
                        "packet_retry_count": int(packet_retry_count),
                        "sign_flip_count": int(run_row["sign_flip_count"]),
                        "sign_flip_used_by_axis": dict(run_row["sign_flip_used_by_axis"]),
                    },
                )
                trace_rows.append(row)
                self._add_staged_sample(session, phase="pretension_stage_verify", run_index=run_index, step_index=len(trace_rows) - 1, payload=row)
                if mode_kind != "characterization" and accepted:
                    self._save_advanced_startup_artifact(servo_service=servo_service, servo_ids=servo_ids, run_row=run_row)
                progress += 1
                session.update_progress(
                    progress,
                    total_progress,
                    {"phase": "settle_verify", "run_index": run_index, "quality_score_0_100": float(quality_score)},
                )

        final_tracking_snapshot = tracker_service.get_snapshot() if tracker_service is not None else None
        runtime_tip_policy = (
            evaluate_runtime_tip_trust(
                snapshot=final_tracking_snapshot,
                workflow=WORKFLOW_PRETENSION_VALIDATION,
                allow_lower_trust=True,
            )
            if final_tracking_snapshot is not None
            else None
        )
        staged_metrics = self._reliable_pretension_metrics(
            mode_kind=mode_kind,
            servo_ids=servo_ids,
            repeat_runs=repeat_runs,
            accepted_runs=accepted_runs,
            failure_counts=failure_counts,
            run_rows=run_rows,
            trace_rows=trace_rows,
            final_tip_xy_points_mm=final_tip_xy_points_mm,
            quality_scores=quality_scores,
            manual_startup_artifact=manual_startup_artifact,
        )
        staged_metrics["tip_centering_variant"] = str(
            getattr(self.config, "tip_centering_variant", PRETENSION_TIP_VARIANT_PAIRED_THEN_TIP)
            or PRETENSION_TIP_VARIANT_PAIRED_THEN_TIP
        )
        staged_metrics["manual_baseline_records"] = list(manual_baseline_records)
        staged_metrics["manual_baseline_record_count"] = int(len(manual_baseline_records))
        staged_metrics["pretension_comparison_report"] = _build_pretension_comparison_report(
            algorithm_run_rows=run_rows,
            manual_baseline_records=manual_baseline_records,
            servo_ids=servo_ids,
            tip_target_xy_mm=self._target_xy(),
            target_load_band_ma=(
                float(self.config.takeup_target_load_proxy_ma),
                float(self.config.high_load_proxy_ma),
            ),
        )
        staged_metrics["runtime_tip_policy"] = runtime_tip_policy.to_dict() if runtime_tip_policy is not None else None
        staged_metrics["runtime_tip_mode_used"] = runtime_tip_policy.mode if runtime_tip_policy is not None else None
        staged_metrics["runtime_tip_trust_level"] = (
            runtime_tip_policy.trust_label if runtime_tip_policy is not None else "unavailable"
        )
        staged_metrics["thesis_trusted_runtime_tip"] = (
            bool(runtime_tip_policy.thesis_trusted) if runtime_tip_policy is not None else False
        )
        staged_metrics["tracker_connected"] = bool(tracker_service is not None)
        staged_metrics["registration_available"] = bool(
            final_tracking_snapshot is not None and getattr(final_tracking_snapshot, "registration_state", None) == "loaded"
        )
        staged_metrics["run_trust_mode"] = "thesis_trusted" if runtime_tip_policy is not None and runtime_tip_policy.thesis_trusted else "current_only"
        staged_metrics["valid_for_model_training"] = False
        staged_metrics["valid_for_thesis_repeatability"] = bool(
            runtime_tip_policy is not None and runtime_tip_policy.thesis_trusted
        )
        staged_metrics["not_thesis_trusted"] = not bool(
            runtime_tip_policy is not None and runtime_tip_policy.thesis_trusted
        )
        staged_metrics["active_segment"] = {
            "key": session.context.settings.robot.active_segment_key(),
            "label": session.context.settings.robot.active_segment_label(),
            "servo_ids": session.context.settings.robot.active_segment_servo_ids(),
            "pairs": session.context.settings.robot.active_segment_pairs(),
            "robot_mode": session.context.settings.robot.mode,
        }
        staged_metrics["operating_context"] = session.context.settings.robot.operating_context().metadata()
        session.metrics.update(staged_metrics)

    def _execute_legacy_single_segment_staged(self, session: ExperimentSession) -> None:
        servo_service = session.context.servo_service
        tracker_service = session.context.tracking_service
        include_tracker = bool(self.config.include_tracker_displacement and tracker_service is not None)
        if include_tracker:
            tracker_snapshot = _optional_tracking_snapshot(tracker_service)
            if tracker_snapshot is None or str(getattr(tracker_snapshot, "canonical_state", "") or "") not in {"mock", "connected", "streaming_healthy", "streaming_degraded"}:
                include_tracker = False
                if not _pretension_current_only_explicit(self.config):
                    raise RuntimeError("Tracker displacement is required for staged pretension, but tracking is not connected.")
                session.add_warning(
                    "Tracker is not connected; staged pretension will run in explicitly current-only lower-trust mode."
                )
        if bool(self.config.include_tracker_displacement) and tracker_service is None:
            if not _pretension_current_only_explicit(self.config):
                raise RuntimeError(
                    "Tracker displacement is required for staged pretension, but tracking_service is unavailable."
                )
            session.add_warning(
                "Tracker displacement is unavailable; staged pretension will run in explicitly current-only lower-trust mode."
            )
        if include_tracker:
            snapshot = tracker_service.get_snapshot()
            if snapshot is None:
                include_tracker = False
                if not _pretension_current_only_explicit(self.config):
                    raise RuntimeError("Tracker snapshot is unavailable for staged pretension.")
                session.add_warning(
                    "Tracker snapshot is unavailable at staged pretension start; falling back to explicitly current-only lower-trust mode."
                )

        servo_ids = self._staged_servo_ids(session)
        repeat_runs = max(1, int(self.config.repeat_runs))
        total_progress = max(1, repeat_runs * 6)
        stage_progress = 0
        manual_startup_artifact = self._manual_startup_artifact_snapshot(servo_service, servo_ids)

        run_rows: list[dict[str, Any]] = []
        trace_rows: list[dict[str, Any]] = []
        failure_counts: dict[str, int] = {}
        final_tip_xy_points_mm: list[list[float]] = []
        accepted_runs = 0

        for run_index in range(repeat_runs):
            session.raise_if_stop_requested()
            run_prefix = f"run_{run_index + 1:02d}"
            step_records: dict[int, Any] = {}
            baselines_ma: dict[int, float] = {}
            start_position_ticks: dict[int, int | None] = {}
            final_position_ticks: dict[int, int | None] = {}
            final_currents_ma: dict[int, int | None] = {}
            stop_reasons: dict[int, str] = {}
            current_above_baseline_ma: dict[int, float | None] = {}
            missing_fields: list[str] = []
            run_trace_rows: list[dict[str, Any]] = []
            clipped_move_count = 0
            correction_move_count = 0
            correction_travel_ticks = 0
            initial_tip_xyz_mm = self._staged_tip_position_mm(tracker_service) if include_tracker else None

            # Stage A: baseline measurement
            for servo_id in servo_ids:
                parameters = self._staged_parameters_for_servo(session, servo_id)
                baseline = servo_service.measure_pretension_baseline(
                    servo_id=int(servo_id),
                    sample_count=int(parameters.baseline_sample_count),
                    filter_window=int(parameters.current_filter_window),
                    parameters=parameters,
                )
                baselines_ma[int(servo_id)] = float(baseline.filtered_current_ma)
                start_position_ticks[int(servo_id)] = (
                    int(baseline.position_tick) if baseline.position_tick is not None else None
                )
                row = {
                    "mode": "single_segment_staged",
                    "run_index": int(run_index),
                    "stage": "baseline",
                    "servo_id": int(servo_id),
                    "baseline_current_ma": float(baseline.filtered_current_ma),
                    "baseline_samples_count": int(baseline.sample_count),
                    "position_tick": (
                        int(baseline.position_tick) if baseline.position_tick is not None else None
                    ),
                    "initial_tip_xyz_mm": list(initial_tip_xyz_mm) if initial_tip_xyz_mm is not None else None,
                }
                run_trace_rows.append(row)
                trace_rows.append(dict(row))
                self._add_staged_sample(
                    session,
                    phase="pretension_stage_baseline",
                    run_index=run_index,
                    step_index=len(run_trace_rows) - 1,
                    payload=row,
                )
            stage_progress += 1
            session.update_progress(stage_progress, total_progress, {"phase": "baseline", "run_index": run_index})

            # Stage B: explicit custom start mode. Release-200 is skipped when the tip is already centered.
            custom_start_status = "skipped"
            custom_start_reason = "move_to_reference_disabled"
            if bool(self.config.move_to_reference):
                target_xy = self._target_xy()
                initial_tip_offset = self._tip_xy_offset(initial_tip_xyz_mm, target_xy)
                requested_modes = {
                    str(self._staged_parameters_for_servo(session, servo_id).start_mode)
                    for servo_id in servo_ids
                }
                if (
                    "release_200_from_current" in requested_modes
                    and initial_tip_offset is not None
                    and initial_tip_offset <= float(self.config.tip_center_tolerance_mm)
                ):
                    custom_start_status = "seeded_current_tip_near_target"
                    custom_start_reason = "tip_already_within_center_tolerance"
                else:
                    custom_start_status = "attempted"
                    custom_start_reason = "explicit_start_mode"
                    for pair_index, (left_index, right_index) in enumerate(((0, 2), (1, 3)), start=1):
                        for servo_id in (int(servo_ids[left_index]), int(servo_ids[right_index])):
                            parameters = self._staged_parameters_for_servo(session, servo_id)
                            move = servo_service.move_servo_to_pretension_reference(
                                servo_id=int(servo_id),
                                parameters=parameters,
                            )
                            if move.clamped:
                                clipped_move_count += 1
                            if move.success:
                                correction_move_count += 1
                                correction_travel_ticks += abs(int(move.delta_ticks))
                            row = {
                                "mode": "single_segment_staged",
                                "run_index": int(run_index),
                                "stage": "custom_start",
                                "pair_label": f"{pair_index}",
                                "servo_id": int(servo_id),
                                "start_mode": str(parameters.start_mode),
                                "status": str(move.status),
                                "success": bool(move.success),
                                "current_position_tick": move.current_position_tick,
                                "goal_tick": move.goal_tick,
                                "delta_ticks": int(move.delta_ticks),
                                "clamped": bool(move.clamped),
                                "reason": custom_start_reason,
                            }
                            run_trace_rows.append(row)
                            trace_rows.append(dict(row))
                            self._add_staged_sample(
                                session,
                                phase="pretension_stage_custom_start",
                                run_index=run_index,
                                step_index=len(run_trace_rows) - 1,
                                payload=row,
                            )
            row = {
                "mode": "single_segment_staged",
                "run_index": int(run_index),
                "stage": "custom_start_summary",
                "custom_start_status": custom_start_status,
                "custom_start_reason": custom_start_reason,
                "clipped_move_count": int(clipped_move_count),
            }
            run_trace_rows.append(row)
            trace_rows.append(dict(row))
            self._add_staged_sample(
                session,
                phase="pretension_stage_custom_start",
                run_index=run_index,
                step_index=len(run_trace_rows) - 1,
                payload=row,
            )
            stage_progress += 1
            session.update_progress(stage_progress, total_progress, {"phase": "custom_start", "run_index": run_index})

            # Stage C: paired slack take-up. Pair 1/3 is handled before pair 2/4.
            for pair_index, (left_index, right_index) in enumerate(((0, 2), (1, 3)), start=1):
                for servo_id in (int(servo_ids[left_index]), int(servo_ids[right_index])):
                    parameters = self._staged_parameters_for_servo(session, servo_id)
                    result = servo_service.run_pretension_routine(
                        servo_id=int(servo_id),
                        parameters=parameters,
                        stop_requested=session.stop_requested,
                    )
                    step_records[int(servo_id)] = result
                    final_position_ticks[int(servo_id)] = (
                        int(result.final_position_tick) if result.final_position_tick is not None else None
                    )
                    final_currents_ma[int(servo_id)] = (
                        int(result.final_current_ma) if result.final_current_ma is not None else None
                    )
                    stop_reason = str(result.stop_reason or result.primary_reason or result.status or "unknown")
                    stop_reasons[int(servo_id)] = stop_reason
                    if not result.success:
                        failure_counts[stop_reason] = int(failure_counts.get(stop_reason, 0)) + 1
                    travel_used_ticks = (
                        max(0, int(result.untensioned_reference_tick) - int(result.final_position_tick))
                        if result.untensioned_reference_tick is not None and result.final_position_tick is not None
                        else None
                    )
                    row = {
                        "mode": "single_segment_staged",
                        "run_index": int(run_index),
                        "stage": "paired_takeup",
                        "pair_label": f"{pair_index}",
                        "servo_id": int(servo_id),
                        "status": str(result.status),
                        "success": bool(result.success),
                        "stop_reason": stop_reason,
                        "final_position_tick": (
                            int(result.final_position_tick) if result.final_position_tick is not None else None
                        ),
                        "final_current_ma": (
                            int(result.final_current_ma) if result.final_current_ma is not None else None
                        ),
                        "baseline_current_ma": (
                            float(result.baseline_current_ma) if result.baseline_current_ma is not None else None
                        ),
                        "current_above_baseline_ma": (
                            float(result.filtered_current_ma - baselines_ma[int(servo_id)])
                            if (
                                result.filtered_current_ma is not None
                                and int(servo_id) in baselines_ma
                            )
                            else None
                        ),
                        "travel_used_ticks": travel_used_ticks,
                        "travel_used_mm": (
                            float(servo_service.mapper.ticks_to_displacement_mm(travel_used_ticks))
                            if travel_used_ticks is not None and getattr(servo_service, "mapper", None) is not None
                            else None
                        ),
                    }
                    run_trace_rows.append(row)
                    trace_rows.append(dict(row))
                    self._add_staged_sample(
                        session,
                        phase="pretension_stage_paired_takeup",
                        run_index=run_index,
                        step_index=len(run_trace_rows) - 1,
                        payload=row,
                    )
            stage_progress += 1
            session.update_progress(stage_progress, total_progress, {"phase": "paired_takeup", "run_index": run_index})

            # Stage D: pair-first load equalization
            equalization_iteration_count = 0
            equalization_note = "not_needed"
            load_balance_error_ma: float | None = None
            pair_balance_error_ma: float | None = None
            for iteration in range(max(0, int(self.config.equalization_max_iterations))):
                telemetry = servo_service.read_live_telemetry(servo_ids)
                loads: dict[int, float] = {}
                for servo_id in servo_ids:
                    current_value = telemetry[int(servo_id)].present_current_ma
                    if current_value is None:
                        missing_fields.append(f"servo_{int(servo_id)}_present_current_ma")
                        continue
                    loads[int(servo_id)] = float(current_value) - float(baselines_ma.get(int(servo_id), 0.0))
                if len(loads) != len(servo_ids):
                    equalization_note = "missing_current"
                    break
                current_above_baseline_ma = dict(loads)
                load_values = list(loads.values())
                load_balance_error_ma = float(max(load_values) - min(load_values))
                if len(servo_ids) >= 4:
                    pair_deltas = [
                        abs(float(loads[int(servo_ids[a])] - loads[int(servo_ids[b])]))
                        for a, b in ((0, 2), (1, 3))
                    ]
                    pair_balance_error_ma = float(max(pair_deltas))
                else:
                    pair_balance_error_ma = load_balance_error_ma
                if load_balance_error_ma <= float(self.config.load_balance_tolerance_ma):
                    equalization_note = "within_tolerance"
                    break
                moved = 0
                for pair_index, (left_index, right_index) in enumerate(((0, 2), (1, 3)), start=1):
                    sid_a = int(servo_ids[left_index])
                    sid_b = int(servo_ids[right_index])
                    pair_diff = float(loads[sid_a] - loads[sid_b])
                    if abs(pair_diff) <= float(self.config.pair_balance_tolerance_ma):
                        continue
                    pos_a = telemetry[sid_a].present_position
                    pos_b = telemetry[sid_b].present_position
                    if pos_a is None or pos_b is None:
                        missing_fields.append(f"pair_{pair_index}_present_position")
                        continue
                    step = int(self.config.equalization_step_ticks)
                    delta_a = -step if pair_diff < 0.0 else step
                    delta_b = step if pair_diff < 0.0 else -step
                    move_a = servo_service.move_servo_to_raw_target(
                        servo_id=sid_a,
                        target_tick=int(pos_a) + int(delta_a),
                        reason="pretension_pair_equalization",
                    )
                    move_b = servo_service.move_servo_to_raw_target(
                        servo_id=sid_b,
                        target_tick=int(pos_b) + int(delta_b),
                        reason="pretension_pair_equalization",
                    )
                    if move_a.clamped:
                        clipped_move_count += 1
                    if move_b.clamped:
                        clipped_move_count += 1
                    for move in (move_a, move_b):
                        if move.success:
                            moved += 1
                            correction_move_count += 1
                            correction_travel_ticks += abs(int(move.delta_ticks))
                    row = {
                        "mode": "single_segment_staged",
                        "run_index": int(run_index),
                        "stage": "pair_equalization",
                        "iteration": int(iteration + 1),
                        "pair_label": f"{pair_index}",
                        "servo_a": sid_a,
                        "servo_b": sid_b,
                        "pair_current_mismatch_ma": abs(pair_diff),
                        "delta_a_ticks": int(delta_a),
                        "delta_b_ticks": int(delta_b),
                        "move_a_status": str(move_a.status),
                        "move_b_status": str(move_b.status),
                    }
                    run_trace_rows.append(row)
                    trace_rows.append(dict(row))
                    self._add_staged_sample(
                        session,
                        phase="pretension_stage_pair_equalization",
                        run_index=run_index,
                        step_index=len(run_trace_rows) - 1,
                        payload=row,
                    )
                if moved > 0:
                    equalization_iteration_count = int(iteration + 1)
                    equalization_note = "pair_adjusted"
                    continue
                target = float(sum(load_values) / len(load_values))
                for servo_id in servo_ids:
                    diff = float(loads[int(servo_id)] - target)
                    if abs(diff) <= (0.5 * float(self.config.load_balance_tolerance_ma)):
                        continue
                    telemetry_entry = telemetry[int(servo_id)]
                    if telemetry_entry.present_position is None:
                        missing_fields.append(f"servo_{int(servo_id)}_present_position")
                        continue
                    tighten = diff < 0.0
                    delta = -int(self.config.equalization_step_ticks) if tighten else int(self.config.equalization_step_ticks)
                    target_tick = int(telemetry_entry.present_position) + int(delta)
                    move = servo_service.move_servo_to_raw_target(
                        servo_id=int(servo_id),
                        target_tick=int(target_tick),
                        reason="pretension_equalization",
                    )
                    if move.success:
                        moved += 1
                        correction_move_count += 1
                        correction_travel_ticks += abs(int(move.delta_ticks))
                    if move.clamped:
                        clipped_move_count += 1
                equalization_iteration_count = int(iteration + 1)
                if moved <= 0:
                    equalization_note = "no_safe_moves"
                    break
            row = {
                "mode": "single_segment_staged",
                "run_index": int(run_index),
                "stage": "equalization",
                "equalization_iterations": int(equalization_iteration_count),
                "equalization_status": equalization_note,
                "load_balance_error_ma": load_balance_error_ma,
                "pair_balance_error_ma": pair_balance_error_ma,
                "clipped_move_count": int(clipped_move_count),
                "correction_move_count": int(correction_move_count),
                "correction_travel_ticks": int(correction_travel_ticks),
            }
            run_trace_rows.append(row)
            trace_rows.append(dict(row))
            self._add_staged_sample(
                session,
                phase="pretension_stage_equalization",
                run_index=run_index,
                step_index=len(run_trace_rows) - 1,
                payload=row,
            )
            stage_progress += 1
            session.update_progress(stage_progress, total_progress, {"phase": "equalization", "run_index": run_index})

            # Stage D: optional tip centering / verticalization proxy (XY centering)
            tip_xy_offset_mm: float | None = None
            initial_tip_xyz_mm = self._staged_tip_position_mm(tracker_service) if include_tracker else None
            final_tip_xyz_mm = initial_tip_xyz_mm
            tip_centering_iterations = 0
            tip_centering_status = "skipped_no_tracker"
            if include_tracker and bool(self.config.enable_tip_centering):
                tip_centering_status = "attempted"
                target_xy = list(self.config.tip_target_xy_mm or [0.0, 0.0])
                if len(target_xy) < 2:
                    target_xy = [0.0, 0.0]
                for iteration in range(max(0, int(self.config.tip_center_max_iterations))):
                    current_tip = self._staged_tip_position_mm(tracker_service)
                    if current_tip is None:
                        tip_centering_status = "tip_unavailable"
                        break
                    final_tip_xyz_mm = list(current_tip)
                    tip_xy = [float(current_tip[0]) - float(target_xy[0]), float(current_tip[1]) - float(target_xy[1])]
                    tip_xy_offset_mm = float(math.sqrt((tip_xy[0] * tip_xy[0]) + (tip_xy[1] * tip_xy[1])))
                    row = {
                        "mode": "single_segment_staged",
                        "run_index": int(run_index),
                        "stage": "tip_centering_iteration",
                        "iteration": int(iteration + 1),
                        "tip_xyz_mm": list(current_tip),
                        "tip_x_error_mm": float(tip_xy[0]),
                        "tip_y_error_mm": float(tip_xy[1]),
                        "tip_xy_offset_mm": float(tip_xy_offset_mm),
                    }
                    run_trace_rows.append(row)
                    trace_rows.append(dict(row))
                    self._add_staged_sample(
                        session,
                        phase="pretension_stage_centering",
                        run_index=run_index,
                        step_index=len(run_trace_rows) - 1,
                        payload=row,
                    )
                    if tip_xy_offset_mm <= float(self.config.tip_center_tolerance_mm):
                        tip_centering_status = "within_tolerance"
                        break
                    if (
                        initial_tip_xyz_mm is not None
                        and float(
                            math.sqrt(
                                (float(current_tip[0]) - float(initial_tip_xyz_mm[0])) ** 2
                                + (float(current_tip[1]) - float(initial_tip_xyz_mm[1])) ** 2
                                + (float(current_tip[2]) - float(initial_tip_xyz_mm[2])) ** 2
                            )
                        )
                        > float(self.config.max_tip_displacement_mm)
                    ):
                        tip_centering_status = "tip_displacement_limit"
                        break
                    # Axis-aligned pair differential adjustment.
                    x_sign = int(self.config.tip_center_x_sign) if int(self.config.tip_center_x_sign) != 0 else 1
                    y_sign = int(self.config.tip_center_y_sign) if int(self.config.tip_center_y_sign) != 0 else 1
                    x_delta = int(self.config.tip_center_step_ticks) * x_sign
                    y_delta = int(self.config.tip_center_step_ticks) * y_sign
                    if tip_xy[0] > 0.0:
                        x_delta *= -1
                    if tip_xy[1] > 0.0:
                        y_delta *= -1
                    # Pair 1/3 drives one bend axis; pair 2/4 drives orthogonal axis.
                    pair_commands = (
                        (int(servo_ids[0]), -int(x_delta), int(servo_ids[2]), int(x_delta)),
                        (int(servo_ids[1]), -int(y_delta), int(servo_ids[3]), int(y_delta)),
                    )
                    safe_moves = 0
                    telemetry = servo_service.read_live_telemetry(servo_ids)
                    for sid_a, delta_a, sid_b, delta_b in pair_commands:
                        pos_a = telemetry[int(sid_a)].present_position
                        pos_b = telemetry[int(sid_b)].present_position
                        if pos_a is None or pos_b is None:
                            continue
                        move_a = servo_service.move_servo_to_raw_target(
                            servo_id=int(sid_a),
                            target_tick=int(pos_a) + int(delta_a),
                            reason="pretension_tip_centering",
                        )
                        move_b = servo_service.move_servo_to_raw_target(
                            servo_id=int(sid_b),
                            target_tick=int(pos_b) + int(delta_b),
                            reason="pretension_tip_centering",
                        )
                        if move_a.success and move_b.success:
                            safe_moves += 1
                            correction_move_count += 2
                            correction_travel_ticks += abs(int(delta_a)) + abs(int(delta_b))
                        if move_a.clamped:
                            clipped_move_count += 1
                        if move_b.clamped:
                            clipped_move_count += 1
                    tip_centering_iterations = int(iteration + 1)
                    if safe_moves <= 0:
                        tip_centering_status = "no_safe_moves"
                        break
                final_tip_xyz_mm = self._staged_tip_position_mm(tracker_service) or final_tip_xyz_mm
                if final_tip_xyz_mm is not None and len(final_tip_xyz_mm) >= 2:
                    target_xy = list(self.config.tip_target_xy_mm or [0.0, 0.0])
                    if len(target_xy) < 2:
                        target_xy = [0.0, 0.0]
                    tip_xy_offset_mm = float(
                        math.sqrt(
                            (float(final_tip_xyz_mm[0]) - float(target_xy[0])) ** 2
                            + (float(final_tip_xyz_mm[1]) - float(target_xy[1])) ** 2
                        )
                    )
            elif include_tracker:
                tip_centering_status = "disabled"
                final_tip_xyz_mm = self._staged_tip_position_mm(tracker_service)
                if final_tip_xyz_mm is not None and len(final_tip_xyz_mm) >= 2:
                    target_xy = list(self.config.tip_target_xy_mm or [0.0, 0.0])
                    if len(target_xy) < 2:
                        target_xy = [0.0, 0.0]
                    tip_xy_offset_mm = float(
                        math.sqrt(
                            (float(final_tip_xyz_mm[0]) - float(target_xy[0])) ** 2
                            + (float(final_tip_xyz_mm[1]) - float(target_xy[1])) ** 2
                        )
                    )

            row = {
                "mode": "single_segment_staged",
                "run_index": int(run_index),
                "stage": "tip_centering",
                "tip_centering_iterations": int(tip_centering_iterations),
                "tip_centering_status": tip_centering_status,
                "initial_tip_xyz_mm": list(initial_tip_xyz_mm) if initial_tip_xyz_mm is not None else None,
                "final_tip_xyz_mm": list(final_tip_xyz_mm) if final_tip_xyz_mm is not None else None,
                "final_tip_xy_offset_mm": tip_xy_offset_mm,
            }
            run_trace_rows.append(row)
            trace_rows.append(dict(row))
            self._add_staged_sample(
                session,
                phase="pretension_stage_centering",
                run_index=run_index,
                step_index=len(run_trace_rows) - 1,
                payload=row,
            )
            stage_progress += 1
            session.update_progress(stage_progress, total_progress, {"phase": "tip_centering", "run_index": run_index})

            # Stage F: settle and verify
            pre_settle_tip_xyz_mm = list(final_tip_xyz_mm) if final_tip_xyz_mm is not None else None
            if float(self.config.settle_verify_time_s) > 0.0:
                session.context.sleep_fn(float(self.config.settle_verify_time_s))
            telemetry_final = servo_service.read_live_telemetry(servo_ids)
            for servo_id in servo_ids:
                telemetry_entry = telemetry_final[int(servo_id)]
                final_position_ticks[int(servo_id)] = telemetry_entry.present_position
                final_currents_ma[int(servo_id)] = telemetry_entry.present_current_ma
                baseline_value = float(baselines_ma.get(int(servo_id), 0.0))
                current_value = telemetry_entry.present_current_ma
                current_above_baseline_ma[int(servo_id)] = (
                    float(current_value) - baseline_value
                    if current_value is not None
                    else None
                )
                if telemetry_entry.present_position is None:
                    missing_fields.append(f"servo_{int(servo_id)}_present_position_final")
                if telemetry_entry.present_current_ma is None:
                    missing_fields.append(f"servo_{int(servo_id)}_present_current_final")
            above_values = [
                float(value)
                for value in current_above_baseline_ma.values()
                if value is not None
            ]
            if above_values:
                load_balance_error_ma = float(max(above_values) - min(above_values))
            if len(servo_ids) >= 4 and all(current_above_baseline_ma.get(int(sid)) is not None for sid in servo_ids):
                pair_balance_error_ma = max(
                    abs(float(current_above_baseline_ma[int(servo_ids[0])] - current_above_baseline_ma[int(servo_ids[2])])),
                    abs(float(current_above_baseline_ma[int(servo_ids[1])] - current_above_baseline_ma[int(servo_ids[3])])),
                )
            final_tip_xyz_mm = self._staged_tip_position_mm(tracker_service) if include_tracker else final_tip_xyz_mm
            settle_tip_drift_mm = self._distance_mm(pre_settle_tip_xyz_mm, final_tip_xyz_mm)
            if final_tip_xyz_mm is not None and len(final_tip_xyz_mm) >= 2:
                target_xy = list(self.config.tip_target_xy_mm or [0.0, 0.0])
                if len(target_xy) < 2:
                    target_xy = [0.0, 0.0]
                tip_xy_offset_mm = float(
                    math.sqrt(
                        (float(final_tip_xyz_mm[0]) - float(target_xy[0])) ** 2
                        + (float(final_tip_xyz_mm[1]) - float(target_xy[1])) ** 2
                    )
                )
                final_tip_xy_points_mm.append([float(final_tip_xyz_mm[0]), float(final_tip_xyz_mm[1])])
            accepted = True
            reject_reasons: list[str] = []
            if missing_fields:
                accepted = False
                reject_reasons.append("missing_telemetry")
            if any(not bool(step_records.get(int(sid)).success) for sid in servo_ids if int(sid) in step_records):
                accepted = False
                reject_reasons.append("takeup_failed")
            if load_balance_error_ma is not None and load_balance_error_ma > float(self.config.accept_max_load_balance_error_ma):
                accepted = False
                reject_reasons.append("load_balance_error")
            if pair_balance_error_ma is not None and pair_balance_error_ma > float(self.config.accept_max_pair_balance_error_ma):
                accepted = False
                reject_reasons.append("pair_balance_error")
            if (
                include_tracker
                and tip_xy_offset_mm is not None
                and tip_xy_offset_mm > float(self.config.accept_max_final_tip_xy_offset_mm)
            ):
                accepted = False
                reject_reasons.append("tip_center_error")
            if include_tracker and tip_xy_offset_mm is None and not bool(self.config.allow_current_only_when_tracker_missing):
                accepted = False
                reject_reasons.append("missing_tip_pose")
            quality_components, quality_score = self._staged_quality_score(
                tip_xy_offset_mm=tip_xy_offset_mm,
                settle_tip_drift_mm=settle_tip_drift_mm,
                load_balance_error_ma=load_balance_error_ma,
                pair_balance_error_ma=pair_balance_error_ma,
                final_position_ticks=final_position_ticks,
                start_position_ticks=start_position_ticks,
                missing_fields=missing_fields,
                clipped_move_count=clipped_move_count,
                correction_move_count=correction_move_count,
                correction_travel_ticks=correction_travel_ticks,
                takeup_success=not any(
                    not bool(step_records.get(int(sid)).success)
                    for sid in servo_ids
                    if int(sid) in step_records
                ),
            )
            if accepted:
                accepted_runs += 1
            if reject_reasons:
                for reason in reject_reasons:
                    failure_counts[reason] = int(failure_counts.get(reason, 0)) + 1

            run_row = {
                "run_index": int(run_index),
                "run_label": run_prefix,
                "accepted": bool(accepted),
                "reject_reasons": list(reject_reasons),
                "servo_ids": list(servo_ids),
                "baseline_current_ma_by_servo": {str(k): float(v) for k, v in baselines_ma.items()},
                "final_current_ma_by_servo": {
                    str(k): (None if v is None else int(v))
                    for k, v in final_currents_ma.items()
                },
                "current_above_baseline_ma_by_servo": {
                    str(k): (None if v is None else float(v))
                    for k, v in current_above_baseline_ma.items()
                },
                "start_position_ticks_by_servo": {
                    str(k): (None if v is None else int(v))
                    for k, v in start_position_ticks.items()
                },
                "final_position_ticks_by_servo": {
                    str(k): (None if v is None else int(v))
                    for k, v in final_position_ticks.items()
                },
                "stop_reason_by_servo": {str(k): str(v) for k, v in stop_reasons.items()},
                "load_balance_error_ma": load_balance_error_ma,
                "pair_balance_error_ma": pair_balance_error_ma,
                "equalization_iterations": int(equalization_iteration_count),
                "equalization_status": equalization_note,
                "tip_centering_iterations": int(tip_centering_iterations),
                "tip_centering_status": tip_centering_status,
                "initial_tip_xyz_mm": list(initial_tip_xyz_mm) if initial_tip_xyz_mm is not None else None,
                "pre_settle_tip_xyz_mm": list(pre_settle_tip_xyz_mm) if pre_settle_tip_xyz_mm is not None else None,
                "final_tip_xyz_mm": list(final_tip_xyz_mm) if final_tip_xyz_mm is not None else None,
                "final_tip_xy_offset_mm": tip_xy_offset_mm,
                "settle_tip_drift_mm": settle_tip_drift_mm,
                "clipped_move_count": int(clipped_move_count),
                "correction_move_count": int(correction_move_count),
                "correction_travel_ticks": int(correction_travel_ticks),
                "quality_score_0_100": float(quality_score),
                "quality_components": dict(quality_components),
                "missing_fields": sorted(set(missing_fields)),
            }
            run_rows.append(run_row)
            if accepted:
                self._save_advanced_startup_artifact(
                    servo_service=servo_service,
                    servo_ids=servo_ids,
                    run_row=run_row,
                )
            for servo_id in servo_ids:
                run_trace_rows.append(
                    {
                        "mode": "single_segment_staged",
                        "run_index": int(run_index),
                        "stage": "verify",
                        "servo_id": int(servo_id),
                        "baseline_current_ma": float(baselines_ma.get(int(servo_id), 0.0)),
                        "final_current_ma": final_currents_ma.get(int(servo_id)),
                        "current_above_baseline_ma": current_above_baseline_ma.get(int(servo_id)),
                        "start_position_tick": start_position_ticks.get(int(servo_id)),
                        "final_position_tick": final_position_ticks.get(int(servo_id)),
                        "travel_used_ticks": (
                            None
                            if (
                                start_position_ticks.get(int(servo_id)) is None
                                or final_position_ticks.get(int(servo_id)) is None
                            )
                            else int(start_position_ticks[int(servo_id)]) - int(final_position_ticks[int(servo_id)])
                        ),
                        "stop_reason": stop_reasons.get(int(servo_id)),
                        "accepted": bool(accepted),
                        "quality_score_0_100": float(quality_score),
                    }
                )
            for row in run_trace_rows:
                if row.get("stage") != "verify":
                    continue
                trace_rows.append(dict(row))
                self._add_staged_sample(
                    session,
                    phase="pretension_stage_verify",
                    run_index=run_index,
                    step_index=len(trace_rows) - 1,
                    payload=row,
                )
            stage_progress += 1
            session.update_progress(
                stage_progress,
                total_progress,
                {
                    "phase": "verify",
                    "run_index": run_index,
                    "quality_score_0_100": float(quality_score),
                },
            )

        # Aggregate repeatability metrics for thesis-style summaries.
        def _std(values: list[float]) -> float | None:
            if len(values) < 2:
                return 0.0 if values else None
            mean_value = float(sum(values) / len(values))
            return float(math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values)))

        per_servo_position_std: dict[str, float | None] = {}
        per_servo_current_std: dict[str, float | None] = {}
        for servo_id in servo_ids:
            final_positions = [
                float(row["final_position_ticks_by_servo"][str(int(servo_id))])
                for row in run_rows
                if row.get("final_position_ticks_by_servo", {}).get(str(int(servo_id))) is not None
            ]
            final_currents = [
                float(row["final_current_ma_by_servo"][str(int(servo_id))])
                for row in run_rows
                if row.get("final_current_ma_by_servo", {}).get(str(int(servo_id))) is not None
            ]
            per_servo_position_std[str(int(servo_id))] = _std(final_positions)
            per_servo_current_std[str(int(servo_id))] = _std(final_currents)

        tip_xy_std_mm: float | None = None
        if final_tip_xy_points_mm:
            x_values = [float(point[0]) for point in final_tip_xy_points_mm]
            y_values = [float(point[1]) for point in final_tip_xy_points_mm]
            x_std = _std(x_values) or 0.0
            y_std = _std(y_values) or 0.0
            tip_xy_std_mm = float(math.sqrt((x_std * x_std) + (y_std * y_std)))
        quality_scores = [
            float(row["quality_score_0_100"])
            for row in run_rows
            if row.get("quality_score_0_100") is not None
        ]
        quality_score_mean = (
            float(sum(quality_scores) / len(quality_scores))
            if quality_scores
            else None
        )

        runtime_tip_policy = (
            evaluate_runtime_tip_trust(
                snapshot=tracker_service.get_snapshot(),
                workflow=WORKFLOW_PRETENSION_VALIDATION,
                allow_lower_trust=True,
            )
            if tracker_service is not None
            else None
        )
        session.metrics.update(
            {
                "mode": "single_segment_staged",
                "algorithm": "advanced_4servo_pretension",
                "servo_ids": list(servo_ids),
                "repeat_runs": int(repeat_runs),
                "pretension_start_mode": (
                    str(self.config.pretension_start_mode).strip().lower()
                    if self.config.pretension_start_mode not in (None, "")
                    else "live_default"
                ),
                "run_count": int(len(run_rows)),
                "accepted_run_count": int(accepted_runs),
                "accepted_run_fraction": (
                    float(accepted_runs / len(run_rows))
                    if run_rows
                    else 0.0
                ),
                "failure_reason_counts": dict(sorted(failure_counts.items())),
                "final_position_std_ticks_by_servo": per_servo_position_std,
                "final_current_std_ma_by_servo": per_servo_current_std,
                "final_tip_xy_std_mm": tip_xy_std_mm,
                "quality_score_mean_0_100": quality_score_mean,
                "quality_score_std_0_100": _std(quality_scores),
                "quality_scores_0_100": quality_scores,
                "manual_startup_artifact": manual_startup_artifact,
                "runtime_tip_policy": runtime_tip_policy.to_dict() if runtime_tip_policy is not None else None,
                "runtime_tip_mode_used": runtime_tip_policy.mode if runtime_tip_policy is not None else None,
                "runtime_tip_trust_level": (
                    runtime_tip_policy.trust_label if runtime_tip_policy is not None else "unavailable"
                ),
                "thesis_trusted_runtime_tip": (
                    bool(runtime_tip_policy.thesis_trusted) if runtime_tip_policy is not None else False
                ),
                "advanced_startup_artifacts": [
                    {
                        "run_index": int(row.get("run_index", 0)),
                        "accepted": bool(row.get("accepted")),
                        "quality_score_0_100": row.get("quality_score_0_100"),
                        "final_position_ticks_by_servo": row.get("final_position_ticks_by_servo"),
                        "final_current_ma_by_servo": row.get("final_current_ma_by_servo"),
                        "final_tip_xyz_mm": row.get("final_tip_xyz_mm"),
                        "final_tip_xy_offset_mm": row.get("final_tip_xy_offset_mm"),
                    }
                    for row in run_rows
                ],
                "run_rows": run_rows,
                "trace_rows": trace_rows,
                "units": {
                    "baseline_current_ma": "mA",
                    "trigger_current_ma": "mA",
                    "current_above_baseline_ma": "mA",
                    "start_position_ticks": "ticks",
                    "final_position_ticks": "ticks",
                    "travel_used_ticks": "ticks",
                    "travel_used_mm": "mm",
                    "tip_position_mm": "mm",
                    "tip_xy_offset_mm": "mm",
                    "load_balance_error_ma": "mA",
                    "pair_balance_error_ma": "mA",
                    "quality_score_0_100": "score",
                },
                "summary_requirements": {
                    "force_status": (
                        "success"
                        if run_rows and accepted_runs == len(run_rows)
                        else "failed"
                        if accepted_runs == 0
                        else "partial_success"
                    )
                },
            }
        )

    def _staged_servo_ids(self, session: ExperimentSession) -> list[int]:
        configured_ids = [int(value) for value in (self.config.servo_ids or []) if int(value) > 0]
        if not configured_ids:
            configured_ids = [
                int(value)
                for value in session.context.settings.robot.active_segment_servo_ids()
                if int(value) > 0
            ]
        deduped: list[int] = []
        for servo_id in configured_ids:
            if servo_id not in deduped:
                deduped.append(int(servo_id))
        if len(deduped) != 4:
            active_label = session.context.settings.robot.active_segment_label()
            raise RuntimeError(
                "Staged pretension validation requires exactly 4 servo IDs for the active single-segment tendon set "
                f"({active_label}); found {deduped}."
            )
        return deduped

    def _staged_parameters_for_servo(self, session: ExperimentSession, servo_id: int) -> PretensionParameters:
        defaults = session.context.servo_service.default_pretension_parameters(int(servo_id))
        default_start_mode = str(defaults.start_mode)
        if str(getattr(self.config, "staged_strategy", "") or "").strip().lower() != "legacy":
            default_start_mode = "current_position"
        return PretensionParameters(
            untensioned_reference_tick=(
                int(self.config.untensioned_reference_tick)
                if self.config.untensioned_reference_tick is not None
                else int(defaults.untensioned_reference_tick)
            ),
            start_mode=(
                str(self.config.pretension_start_mode).strip().lower()
                if self.config.pretension_start_mode not in (None, "")
                else default_start_mode
            ),
            step_ticks=int(self.config.step_ticks) if self.config.step_ticks is not None else int(defaults.step_ticks),
            settle_time_s=(
                float(self.config.settle_time_s)
                if self.config.settle_time_s is not None
                else float(defaults.settle_time_s)
            ),
            baseline_sample_count=(
                int(self.config.baseline_sample_count)
                if self.config.baseline_sample_count is not None
                else int(defaults.baseline_sample_count)
            ),
            current_filter_window=(
                int(self.config.current_filter_window)
                if self.config.current_filter_window is not None
                else int(defaults.current_filter_window)
            ),
            current_delta_threshold_ma=(
                int(self.config.current_delta_threshold_ma)
                if self.config.current_delta_threshold_ma is not None
                else int(defaults.current_delta_threshold_ma)
            ),
            absolute_trigger_current_ma=(
                int(self.config.absolute_trigger_current_ma)
                if self.config.absolute_trigger_current_ma is not None
                else defaults.absolute_trigger_current_ma
            ),
            hard_current_stop_ma=(
                int(self.config.hard_current_stop_ma)
                if self.config.hard_current_stop_ma is not None
                else int(defaults.hard_current_stop_ma)
            ),
            max_travel_ticks=(
                int(self.config.max_travel_ticks)
                if self.config.max_travel_ticks is not None
                else int(defaults.max_travel_ticks)
            ),
            timeout_s=float(self.config.timeout_s) if self.config.timeout_s is not None else float(defaults.timeout_s),
        )

    def _staged_tip_position_mm(self, tracker_service) -> list[float] | None:
        if tracker_service is None:
            return None
        snapshot = tracker_service.get_snapshot()
        matrix = getattr(snapshot, "T_robot_tip", None)
        if isinstance(matrix, list) and len(matrix) == 4:
            try:
                return [float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3])]
            except Exception:
                return None
        tool = getattr(snapshot, "tools", {}).get(str(self.config.tracker_tool_id))
        if tool is None:
            return None
        translation = getattr(tool, "translation_mm", None)
        if translation is None or len(translation) != 3:
            return None
        return [float(translation[0]), float(translation[1]), float(translation[2])]

    @staticmethod
    def _merge_event_counts(target: dict[str, int], source: dict[str, Any] | None) -> None:
        for key, value in dict(source or {}).items():
            target[str(key)] = int(target.get(str(key), 0)) + int(value or 0)

    def _resolved_staged_start_mode(self, session: ExperimentSession, servo_ids: list[int]) -> str:
        modes = {
            str(self._staged_parameters_for_servo(session, int(servo_id)).start_mode).strip().lower()
            for servo_id in servo_ids
        }
        if "full_release_4095" in modes:
            return "full_release_4095"
        if "manual_startup_artifact" in modes:
            return "manual_startup_artifact"
        if "release_200_from_current" in modes:
            return "release_200_from_current"
        return "current_position"

    @staticmethod
    def _classify_telemetry_error(error: Any) -> str | None:
        text = str(error or "").strip().lower()
        if not text:
            return None
        if "incorrect status packet" in text:
            return "incorrect_status_packet"
        if "no status packet" in text:
            return "no_status_packet"
        if "txrx" in text or "communication" in text or "packet" in text:
            return "packet_status_error"
        return "telemetry_error"

    def _telemetry_policy_for_snapshot(self, servo_service, telemetry: dict[int, Any], servo_ids: list[int]) -> dict[str, Any]:
        counts: dict[str, int] = {}
        missing_current: list[int] = []
        missing_position: list[int] = []
        stale: list[int] = []
        hard_errors: list[int] = []
        for servo_id in servo_ids:
            entry = telemetry.get(int(servo_id))
            if entry is None:
                missing_current.append(int(servo_id))
                missing_position.append(int(servo_id))
                counts["missing_servo"] = int(counts.get("missing_servo", 0)) + 1
                continue
            if entry.present_current_ma is None:
                missing_current.append(int(servo_id))
            if entry.present_position is None:
                missing_position.append(int(servo_id))
            classified = self._classify_telemetry_error(getattr(entry, "telemetry_error", None))
            if classified:
                counts[classified] = int(counts.get(classified, 0)) + 1
            if getattr(entry, "hardware_error", None) or getattr(entry, "hardware_error_code", None) not in (None, 0):
                hard_errors.append(int(servo_id))
                counts["hardware_error"] = int(counts.get("hardware_error", 0)) + 1
            try:
                servo_service.safety_guard.validate_telemetry_freshness(getattr(entry, "last_read_monotonic_s", None))
            except Exception:
                stale.append(int(servo_id))
                counts["stale_telemetry"] = int(counts.get("stale_telemetry", 0)) + 1
        if missing_current and missing_position:
            counts["missing_current_and_position"] = int(counts.get("missing_current_and_position", 0)) + 1
        elif missing_current:
            counts["missing_current"] = int(counts.get("missing_current", 0)) + 1
        elif missing_position:
            counts["missing_position"] = int(counts.get("missing_position", 0)) + 1
        fail_reason = None
        retryable = False
        if stale:
            fail_reason = "stale_telemetry"
        elif hard_errors:
            fail_reason = "safety_limit_rejected"
        elif missing_current and missing_position:
            fail_reason = "missing_current_and_position"
            retryable = True
        elif missing_current:
            fail_reason = "missing_current"
            retryable = True
        elif missing_position:
            fail_reason = "missing_position"
            retryable = True
        else:
            packet_reasons = [
                key
                for key in ("no_status_packet", "incorrect_status_packet", "packet_status_error", "telemetry_error")
                if counts.get(key)
            ]
            if packet_reasons:
                fail_reason = str(packet_reasons[0])
                retryable = fail_reason != "telemetry_error"
        return {
            "ok": fail_reason is None,
            "retryable": bool(retryable),
            "fail_reason": fail_reason,
            "event_counts": counts,
            "missing_current_servo_ids": missing_current,
            "missing_position_servo_ids": missing_position,
            "stale_servo_ids": stale,
        }

    def _read_live_telemetry_with_policy(self, servo_service, servo_ids: list[int]) -> tuple[dict[int, Any], dict[str, Any]]:
        budget = max(1, int(self.config.staged_packet_retry_budget))
        merged_counts: dict[str, int] = {}
        last_telemetry: dict[int, Any] = {}
        last_policy: dict[str, Any] = {}
        retry_count = 0
        for _attempt in range(1, budget + 1):
            try:
                telemetry = servo_service.read_live_telemetry(servo_ids)
            except Exception as exc:
                reason = self._classify_telemetry_error(str(exc)) or "packet_status_error"
                telemetry = {}
                last_policy = {
                    "ok": False,
                    "retryable": True,
                    "fail_reason": reason,
                    "event_counts": {reason: 1},
                    "missing_current_servo_ids": list(servo_ids),
                    "missing_position_servo_ids": list(servo_ids),
                    "stale_servo_ids": [],
                }
            else:
                last_policy = self._telemetry_policy_for_snapshot(servo_service, telemetry, servo_ids)
            last_telemetry = telemetry
            self._merge_event_counts(merged_counts, last_policy.get("event_counts"))
            if last_policy.get("ok"):
                return last_telemetry, {
                    **last_policy,
                    "event_counts": merged_counts,
                    "packet_retry_count": retry_count,
                    "telemetry_fail_closed_reason": None,
                }
            if last_policy.get("fail_reason") == "stale_telemetry":
                return last_telemetry, {
                    **last_policy,
                    "event_counts": merged_counts,
                    "packet_retry_count": retry_count,
                    "telemetry_fail_closed_reason": "stale_telemetry",
                }
            if not bool(last_policy.get("retryable")):
                return last_telemetry, {
                    **last_policy,
                    "event_counts": merged_counts,
                    "packet_retry_count": retry_count,
                    "telemetry_fail_closed_reason": last_policy.get("fail_reason"),
                }
            retry_count += 1
        reason = str(last_policy.get("fail_reason") or "packet_retry_budget_exhausted")
        if retry_count >= budget:
            reason = "packet_retry_budget_exhausted"
        merged_counts[reason] = int(merged_counts.get(reason, 0)) + 1
        return last_telemetry, {
            **last_policy,
            "event_counts": merged_counts,
            "packet_retry_count": retry_count,
            "telemetry_fail_closed_reason": reason,
        }

    def _preflight_raw_targets(
        self,
        *,
        servo_service,
        telemetry: dict[int, Any],
        targets_by_servo: dict[int, int],
    ) -> tuple[bool, dict[str, Any]]:
        details: dict[str, Any] = {}
        for servo_id, target_tick in targets_by_servo.items():
            entry = telemetry.get(int(servo_id))
            if entry is None or entry.present_position is None:
                return False, {"stop_reason": "missing_position", "details": details}
            assessment = servo_service.assess_motion(
                int(servo_id),
                require_calibrated_bounds=False,
                telemetry=entry,
            )
            details[str(int(servo_id))] = {
                "ready": bool(assessment.ready),
                "reason": assessment.reason,
                "safe_min_tick": assessment.safe_min_tick,
                "safe_max_tick": assessment.safe_max_tick,
                "target_tick": int(target_tick),
                "current_position_tick": int(entry.present_position),
            }
            if not assessment.ready:
                reason = str(assessment.reason or "").lower()
                if "torque" in reason and "enable" in reason:
                    return False, {"stop_reason": "torque_enable_failed", "details": details}
                if "stale" in reason:
                    return False, {"stop_reason": "stale_telemetry", "details": details}
                return False, {"stop_reason": "safety_limit_rejected", "details": details}
            if assessment.safe_min_tick is None or assessment.safe_max_tick is None:
                return False, {"stop_reason": "safety_limit_rejected", "details": details}
            if int(target_tick) < int(assessment.safe_min_tick) or int(target_tick) > int(assessment.safe_max_tick):
                return False, {"stop_reason": "safety_limit_rejected", "details": details}
        return True, {"stop_reason": "", "details": details}

    # Tolerance for "did the servo reach the commanded tick?" after a group write.
    # DYNAMIXEL position-mode writes accept a goal and start moving; the present
    # position will not be exactly equal to the goal immediately afterward. A small
    # tolerance lets us declare success once the servo is within a few ticks of the
    # goal (the goal is being tracked, just not arrived-at-exactly). Without this
    # tolerance the entire paired pretension algorithm fails on every step with
    # `partial_pair_failure`, which was the historical operational blocker.
    POSITION_REACHED_TOLERANCE_TICKS = 8

    def _apply_group_command(
        self,
        *,
        servo_service,
        telemetry: dict[int, Any],
        targets_by_servo: dict[int, int],
        reason: str,
        require_opposed_pair: bool = False,
    ) -> dict[str, Any]:
        targets = {int(k): int(v) for k, v in dict(targets_by_servo or {}).items()}
        if not targets:
            return {"success": False, "stop_reason": "no_command", "move_count": 0, "travel_ticks": 0, "commanded_positions_ticks": {}}
        if require_opposed_pair and len(targets) == 2:
            deltas = [
                int(targets[int(servo_id)]) - int(telemetry[int(servo_id)].present_position)
                for servo_id in targets
                if telemetry.get(int(servo_id)) is not None and telemetry[int(servo_id)].present_position is not None
            ]
            if len(deltas) != 2 or deltas[0] == 0 or deltas[1] == 0 or (deltas[0] > 0) == (deltas[1] > 0):
                return {
                    "success": False,
                    "stop_reason": "safety_limit_rejected",
                    "message": "Paired correction commands must move opposing tendons in opposite raw-count directions.",
                    "move_count": 0,
                    "travel_ticks": 0,
                    "commanded_positions_ticks": targets,
                }
        ok, preflight = self._preflight_raw_targets(
            servo_service=servo_service,
            telemetry=telemetry,
            targets_by_servo=targets,
        )
        if not ok:
            return {
                "success": False,
                "stop_reason": str(preflight.get("stop_reason") or "safety_limit_rejected"),
                "preflight": preflight,
                "move_count": 0,
                "travel_ticks": 0,
                "commanded_positions_ticks": targets,
            }
        start_positions = {int(servo_id): int(telemetry[int(servo_id)].present_position) for servo_id in targets}
        tolerance = int(self.POSITION_REACHED_TOLERANCE_TICKS)

        def _reached(present: int | None, goal: int) -> bool:
            """A servo is considered to have reached its goal once it is within
            ``POSITION_REACHED_TOLERANCE_TICKS`` of the commanded position. The
            servo will continue to track the goal; we only need to know that the
            write was accepted and motion is underway."""
            if present is None:
                return False
            return abs(int(present) - int(goal)) <= tolerance

        try:
            if hasattr(servo_service, "_write_goal_positions"):
                servo_service._write_goal_positions(targets)
            else:
                for servo_id, target_tick in targets.items():
                    servo_service.move_servo_to_raw_target(
                        servo_id=int(servo_id),
                        target_tick=int(target_tick),
                        reason=reason,
                    )
        except Exception as exc:
            after_failure = servo_service.read_live_telemetry(list(targets))
            reached = {
                int(servo_id): _reached(
                    after_failure.get(int(servo_id)).present_position
                    if after_failure.get(int(servo_id)) is not None
                    else None,
                    int(target_tick),
                )
                for servo_id, target_tick in targets.items()
            }
            return {
                "success": False,
                "stop_reason": "partial_pair_failure" if any(reached.values()) else "safety_limit_rejected",
                "message": str(exc),
                "preflight": preflight,
                "reached_targets": reached,
                "move_count": sum(1 for reached_target in reached.values() if reached_target),
                "travel_ticks": sum(abs(int(targets[sid]) - int(start_positions[sid])) for sid in targets if reached.get(sid)),
                "commanded_positions_ticks": targets,
            }
        after = servo_service.read_live_telemetry(list(targets))
        reached = {
            int(servo_id): _reached(
                after.get(int(servo_id)).present_position
                if after.get(int(servo_id)) is not None
                else None,
                int(target_tick),
            )
            for servo_id, target_tick in targets.items()
        }
        move_count = sum(1 for reached_target in reached.values() if reached_target)
        success = move_count == len(targets)
        return {
            "success": bool(success),
            "stop_reason": "" if success else "partial_pair_failure",
            "preflight": preflight,
            "reached_targets": reached,
            "position_tolerance_ticks": tolerance,
            "move_count": int(move_count),
            "travel_ticks": sum(abs(int(targets[sid]) - int(start_positions[sid])) for sid, did_reach in reached.items() if did_reach),
            "commanded_positions_ticks": targets,
            "delta_ticks": {str(sid): int(targets[sid]) - int(start_positions[sid]) for sid in targets},
            "move_status": {str(sid): ("moved" if reached[sid] else "not_reached") for sid in targets},
        }

    def _target_xy(self) -> list[float]:
        target_xy = list(self.config.tip_target_xy_mm or [0.0, 0.0])
        if len(target_xy) < 2:
            target_xy = [0.0, 0.0]
        return [float(target_xy[0]), float(target_xy[1])]

    @staticmethod
    def _tip_xy_offset(tip_xyz_mm: list[float] | None, target_xy_mm: list[float]) -> float | None:
        if tip_xyz_mm is None or len(tip_xyz_mm) < 2:
            return None
        return float(
            math.sqrt(
                (float(tip_xyz_mm[0]) - float(target_xy_mm[0])) ** 2
                + (float(tip_xyz_mm[1]) - float(target_xy_mm[1])) ** 2
            )
        )

    @staticmethod
    def _distance_mm(left_xyz_mm: list[float] | None, right_xyz_mm: list[float] | None) -> float | None:
        if left_xyz_mm is None or right_xyz_mm is None or len(left_xyz_mm) < 3 or len(right_xyz_mm) < 3:
            return None
        return float(
            math.sqrt(
                (float(left_xyz_mm[0]) - float(right_xyz_mm[0])) ** 2
                + (float(left_xyz_mm[1]) - float(right_xyz_mm[1])) ** 2
                + (float(left_xyz_mm[2]) - float(right_xyz_mm[2])) ** 2
            )
        )

    def _run_soft_release_start(
        self,
        *,
        session: ExperimentSession,
        servo_service,
        tracker_service,
        servo_ids: list[int],
        run_index: int,
        target_xy_mm: list[float],
        startup_reference_ticks_by_servo: dict[int, int | None],
        mode_kind: str,
        trace_rows: list[dict[str, Any]],
        deadline_monotonic: float | None = None,
    ) -> dict[str, Any]:
        result = {
            "stop_reason": "soft_release_complete",
            "move_count": 0,
            "travel_ticks": 0,
            "clipped_move_count": 0,
            "released_ticks_by_servo": {str(int(servo_id)): 0 for servo_id in servo_ids},
            "low_load_reached_by_servo": {str(int(servo_id)): False for servo_id in servo_ids},
            "packet_retry_count": 0,
            "telemetry_event_counts": {},
        }
        step = max(1, int(self.config.soft_release_step_ticks))
        max_travel = max(0, int(self.config.soft_release_max_travel_ticks))
        target_current = max(0.0, float(self.config.soft_release_current_target_ma))
        if max_travel <= 0:
            result["stop_reason"] = "soft_release_disabled"
            return result
        iterations = max(1, int(math.ceil(max_travel / float(step))))
        for iteration in range(iterations):
            session.raise_if_stop_requested()
            if deadline_monotonic is not None and float(session.context.monotonic_fn()) > float(deadline_monotonic):
                result["stop_reason"] = "runtime_budget_exhausted"
                return result
            measurement = self._advanced_measurement(
                servo_service=servo_service,
                tracker_service=tracker_service,
                servo_ids=servo_ids,
                baseline_current_ma_by_servo={},
                target_xy_mm=target_xy_mm,
                startup_reference_ticks_by_servo=startup_reference_ticks_by_servo,
                trust_status=("runtime_tip" if tracker_service is not None else "current_only_lower_trust"),
            )
            self._merge_event_counts(result["telemetry_event_counts"], measurement.get("telemetry_event_counts"))
            result["packet_retry_count"] += int(measurement.get("packet_retry_count", 0) or 0)
            if measurement.get("telemetry_fail_closed_reason"):
                result["stop_reason"] = str(measurement["telemetry_fail_closed_reason"])
                return result
            raw_current = dict(measurement.get("raw_current_ma") or {})
            positions = dict(measurement.get("measured_positions_ticks") or {})
            release_targets: dict[int, int] = {}
            for servo_id in servo_ids:
                current = raw_current.get(int(servo_id))
                position = positions.get(int(servo_id))
                released = int(result["released_ticks_by_servo"][str(int(servo_id))])
                low_load = current is not None and abs(float(current)) <= target_current
                capped = released >= max_travel
                result["low_load_reached_by_servo"][str(int(servo_id))] = bool(low_load)
                if low_load or capped or position is None:
                    continue
                delta = min(step, max_travel - released)
                release_targets[int(servo_id)] = int(position) + int(delta)
            row = self._advanced_stage_row(
                mode_kind=mode_kind,
                run_index=run_index,
                stage="soft_release_check",
                target_xy_mm=target_xy_mm,
                measurement=measurement,
                extra={
                    "iteration": int(iteration + 1),
                    "soft_release_current_target_ma": float(target_current),
                    "released_ticks_by_servo": dict(result["released_ticks_by_servo"]),
                    "low_load_reached_by_servo": dict(result["low_load_reached_by_servo"]),
                },
            )
            trace_rows.append(row)
            self._add_staged_sample(session, phase="pretension_soft_release", run_index=run_index, step_index=len(trace_rows) - 1, payload=row)
            if not release_targets:
                return result
            telemetry, policy = self._read_live_telemetry_with_policy(servo_service, servo_ids)
            self._merge_event_counts(result["telemetry_event_counts"], policy.get("event_counts"))
            result["packet_retry_count"] += int(policy.get("packet_retry_count", 0) or 0)
            if policy.get("telemetry_fail_closed_reason"):
                result["stop_reason"] = str(policy["telemetry_fail_closed_reason"])
                return result
            move = self._apply_group_command(
                servo_service=servo_service,
                telemetry=telemetry,
                targets_by_servo=release_targets,
                reason="pretension_soft_release_low_load_start",
            )
            result["move_count"] += int(move.get("move_count", 0))
            result["travel_ticks"] += int(move.get("travel_ticks", 0))
            if not bool(move.get("success")):
                result["stop_reason"] = str(move.get("stop_reason") or "safety_limit_rejected")
                return result
            for sid, delta in dict(move.get("delta_ticks") or {}).items():
                result["released_ticks_by_servo"][str(int(sid))] = (
                    int(result["released_ticks_by_servo"][str(int(sid))]) + abs(int(delta))
                )
        for servo_id in servo_ids:
            if int(result["released_ticks_by_servo"][str(int(servo_id))]) >= max_travel:
                result["low_load_reached_by_servo"][str(int(servo_id))] = bool(
                    result["low_load_reached_by_servo"][str(int(servo_id))]
                )
        return result

    def _run_explicit_full_release_start(
        self,
        *,
        session: ExperimentSession,
        servo_service,
        tracker_service,
        servo_ids: list[int],
        run_index: int,
        target_xy_mm: list[float],
        startup_reference_ticks_by_servo: dict[int, int | None],
        mode_kind: str,
        trace_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        telemetry, policy = self._read_live_telemetry_with_policy(servo_service, servo_ids)
        result = {
            "stop_reason": "full_release_complete",
            "move_count": 0,
            "travel_ticks": 0,
            "clipped_move_count": 0,
            "packet_retry_count": int(policy.get("packet_retry_count", 0) or 0),
            "telemetry_event_counts": dict(policy.get("event_counts") or {}),
            "explicit_full_release": True,
        }
        if policy.get("telemetry_fail_closed_reason"):
            result["stop_reason"] = str(policy["telemetry_fail_closed_reason"])
            return result
        targets: dict[int, int] = {}
        for servo_id in servo_ids:
            parameters = self._staged_parameters_for_servo(session, int(servo_id))
            window = servo_service.pretension_window_for_servo(servo_id=int(servo_id), parameters=parameters)
            current_position = telemetry[int(servo_id)].present_position
            if current_position is None:
                result["stop_reason"] = "missing_position"
                return result
            targets[int(servo_id)] = int(window.effective_max_target_tick)
            if int(window.effective_max_target_tick) != int(parameters.untensioned_reference_tick):
                result["clipped_move_count"] += 1
        move = self._apply_group_command(
            servo_service=servo_service,
            telemetry=telemetry,
            targets_by_servo=targets,
            reason="pretension_explicit_full_release_start",
        )
        result["move_count"] += int(move.get("move_count", 0))
        result["travel_ticks"] += int(move.get("travel_ticks", 0))
        if not bool(move.get("success")):
            result["stop_reason"] = str(move.get("stop_reason") or "safety_limit_rejected")
        measurement = self._advanced_measurement(
            servo_service=servo_service,
            tracker_service=tracker_service,
            servo_ids=servo_ids,
            baseline_current_ma_by_servo={},
            target_xy_mm=target_xy_mm,
            commanded_positions_ticks=targets,
            startup_reference_ticks_by_servo=startup_reference_ticks_by_servo,
            trust_status=("runtime_tip" if tracker_service is not None else "current_only_lower_trust"),
        )
        row = self._advanced_stage_row(
            mode_kind=mode_kind,
            run_index=run_index,
            stage="explicit_full_release",
            target_xy_mm=target_xy_mm,
            measurement=measurement,
            extra={"move": move, **result},
        )
        trace_rows.append(row)
        self._add_staged_sample(session, phase="pretension_explicit_full_release", run_index=run_index, step_index=len(trace_rows) - 1, payload=row)
        return result

    def _run_symmetric_paired_takeup(
        self,
        *,
        session: ExperimentSession,
        servo_service,
        tracker_service,
        servo_ids: list[int],
        run_index: int,
        target_xy_mm: list[float],
        baseline_current_ma_by_servo: dict[int, float],
        startup_reference_ticks_by_servo: dict[int, int | None],
        effective_load_tolerance_ma: float,
        trace_rows: list[dict[str, Any]],
        deadline_monotonic: float | None = None,
    ) -> dict[str, Any]:
        """Find each servo's engagement edge, then fine-tune to the target band.

        Algorithm (per operator request 2026-05-18):

        1. Engagement scan. All four servos step inward together by
           ``engagement_scan_step_ticks`` per iteration (default 50). After
           each step, read each servo's filtered current. When a servo's
           ``current - baseline`` rises above ``engagement_rise_threshold_ma``
           (default 5 mA), mark that servo as ENGAGED, record the engagement
           tick, and stop stepping it inward. Continue with the remaining
           servos until they all engage OR the travel budget is exhausted.
        2. Back-off. Command each engaged servo back to
           ``engagement_tick + engagement_back_off_ticks`` (default 50). This
           gives a small slack margin so the fine phase has room to add tension
           without immediately overshooting.
        3. Fine. Step inward in ``conservative_step_ticks`` (default 8)
           increments. Stop when every servo's load proxy is in the target
           band [takeup_target_load_proxy_ma, high_load_proxy_ma] OR a servo
           overshoots the high edge (transition to trim).

        The engagement-scan approach replaces the prior "monotonic step +
        slack-vs-refine" heuristic which relied on absolute load thresholds
        and lost the engagement signal in baseline noise. The new code looks
        for the TRANSITION away from baseline, which is what makes a tendon
        physically taut.
        """
        result: dict[str, Any] = {
            "move_count": 0,
            "travel_ticks": 0,
            "clipped_move_count": 0,
            "stop_reason": "takeup_complete",
            "packet_retry_count": 0,
            "telemetry_event_counts": {},
            "engagement_scan": {
                "engagement_tick_by_servo": {},
                "iterations": 0,
                "iterations_to_engage_by_servo": {},
                "rise_threshold_ma": 0.0,
                "back_off_targets_by_servo": {},
            },
        }
        coarse_step = max(1, int(getattr(self.config, "engagement_scan_step_ticks", 50)))
        back_off_ticks = max(0, int(getattr(self.config, "engagement_back_off_ticks", 50)))
        rise_threshold_ma = max(
            float(self.config.min_meaningful_current_delta_ma),
            float(getattr(self.config, "engagement_rise_threshold_ma", 5.0)),
        )
        result["engagement_scan"]["rise_threshold_ma"] = float(rise_threshold_ma)
        refine_step = max(1, int(self.config.conservative_step_ticks))
        target_load = max(float(self.config.takeup_target_load_proxy_ma), float(effective_load_tolerance_ma) * 0.5)
        high_load = max(target_load, float(getattr(self.config, "high_load_proxy_ma", 40.0)))
        max_travel = max(0, int(self.config.conservative_max_cumulative_travel_ticks))
        max_iterations = max(1, int(self.config.takeup_max_iterations))

        engagement_tick_by_servo: dict[int, int] = {}
        iterations_to_engage_by_servo: dict[int, int] = {}

        # --- Phase 1: engagement scan ----------------------------------
        for iteration in range(max_iterations):
            session.raise_if_stop_requested()
            if deadline_monotonic is not None and float(session.context.monotonic_fn()) > float(deadline_monotonic):
                result["stop_reason"] = "runtime_budget_exhausted"
                return result
            measurement = self._advanced_measurement(
                servo_service=servo_service,
                tracker_service=tracker_service,
                servo_ids=servo_ids,
                baseline_current_ma_by_servo=baseline_current_ma_by_servo,
                target_xy_mm=target_xy_mm,
                startup_reference_ticks_by_servo=startup_reference_ticks_by_servo,
                trust_status=("runtime_tip" if tracker_service is not None else "current_only_lower_trust"),
            )
            self._merge_event_counts(result["telemetry_event_counts"], measurement.get("telemetry_event_counts"))
            result["packet_retry_count"] += int(measurement.get("packet_retry_count", 0) or 0)
            if measurement.get("telemetry_fail_closed_reason"):
                result["stop_reason"] = str(measurement["telemetry_fail_closed_reason"])
                return result
            positions = dict(measurement.get("measured_positions_ticks") or {})
            loads = dict(measurement.get("load_proxy_current_ma") or measurement.get("current_above_baseline_ma") or {})

            # Mark newly-engaged servos based on this step's measurement.
            for servo_id in servo_ids:
                sid = int(servo_id)
                if sid in engagement_tick_by_servo:
                    continue
                load = loads.get(sid)
                if load is None or positions.get(sid) is None:
                    continue
                if float(load) >= float(rise_threshold_ma):
                    engagement_tick_by_servo[sid] = int(positions[sid])
                    iterations_to_engage_by_servo[sid] = int(iteration + 1)

            still_unengaged = [sid for sid in (int(s) for s in servo_ids) if sid not in engagement_tick_by_servo]
            if not still_unengaged:
                result["stop_reason"] = "all_servos_engaged"
                result["engagement_scan"]["iterations"] = int(iteration + 1)
                break
            if max_travel > 0 and int(result["travel_ticks"]) >= max_travel:
                result["stop_reason"] = "travel_budget_exhausted_during_engagement_scan"
                result["engagement_scan"]["iterations"] = int(iteration + 1)
                return result

            telemetry, policy = self._read_live_telemetry_with_policy(servo_service, servo_ids)
            self._merge_event_counts(result["telemetry_event_counts"], policy.get("event_counts"))
            result["packet_retry_count"] += int(policy.get("packet_retry_count", 0) or 0)
            if policy.get("telemetry_fail_closed_reason"):
                result["stop_reason"] = str(policy["telemetry_fail_closed_reason"])
                return result

            # Step inward ONLY the servos that have not yet engaged.
            targets: dict[int, int] = {}
            for servo_id in still_unengaged:
                entry = telemetry.get(int(servo_id))
                if entry is None or entry.present_position is None:
                    result["stop_reason"] = "missing_position"
                    return result
                targets[int(servo_id)] = int(entry.present_position) - int(coarse_step)
            move = self._apply_group_command(
                servo_service=servo_service,
                telemetry=telemetry,
                targets_by_servo=targets,
                reason="pretension_engagement_scan",
            )
            result["move_count"] += int(move.get("move_count", 0))
            result["travel_ticks"] += int(move.get("travel_ticks", 0))
            if not bool(move.get("success")):
                result["stop_reason"] = str(move.get("stop_reason") or "engagement_scan_move_failed")
                return result
            after = self._advanced_measurement(
                servo_service=servo_service,
                tracker_service=tracker_service,
                servo_ids=servo_ids,
                baseline_current_ma_by_servo=baseline_current_ma_by_servo,
                target_xy_mm=target_xy_mm,
                commanded_positions_ticks=targets,
                startup_reference_ticks_by_servo=startup_reference_ticks_by_servo,
                trust_status=("runtime_tip" if tracker_service is not None else "current_only_lower_trust"),
            )
            row = self._advanced_stage_row(
                mode_kind="conservative_startup",
                run_index=run_index,
                stage="engagement_scan",
                target_xy_mm=target_xy_mm,
                measurement=after,
                extra={
                    "iteration": int(iteration + 1),
                    "engagement_scan_step_ticks": int(coarse_step),
                    "engagement_rise_threshold_ma": float(rise_threshold_ma),
                    "engaged_servo_ids": sorted(int(sid) for sid in engagement_tick_by_servo),
                    "unengaged_servo_ids": sorted(int(sid) for sid in still_unengaged if int(sid) not in engagement_tick_by_servo),
                    "engagement_tick_by_servo": {str(k): int(v) for k, v in engagement_tick_by_servo.items()},
                    "move": move,
                },
            )
            trace_rows.append(row)
            self._add_staged_sample(session, phase="pretension_engagement_scan", run_index=run_index, step_index=len(trace_rows) - 1, payload=row)
            result["engagement_scan"]["iterations"] = int(iteration + 1)
        else:
            # Loop exhausted without all-engaged.
            still_unengaged = [sid for sid in (int(s) for s in servo_ids) if sid not in engagement_tick_by_servo]
            if still_unengaged:
                result["stop_reason"] = "engagement_scan_iteration_limit"
                return result

        result["engagement_scan"]["engagement_tick_by_servo"] = {
            str(k): int(v) for k, v in engagement_tick_by_servo.items()
        }
        result["engagement_scan"]["iterations_to_engage_by_servo"] = {
            str(k): int(v) for k, v in iterations_to_engage_by_servo.items()
        }

        # --- Phase 2: back off to engagement_tick + back_off_ticks -------
        if back_off_ticks > 0 and engagement_tick_by_servo:
            telemetry, policy = self._read_live_telemetry_with_policy(servo_service, servo_ids)
            self._merge_event_counts(result["telemetry_event_counts"], policy.get("event_counts"))
            result["packet_retry_count"] += int(policy.get("packet_retry_count", 0) or 0)
            if policy.get("telemetry_fail_closed_reason"):
                result["stop_reason"] = str(policy["telemetry_fail_closed_reason"])
                return result
            back_off_targets: dict[int, int] = {
                int(sid): int(engagement_tick_by_servo[int(sid)]) + int(back_off_ticks)
                for sid in servo_ids
                if int(sid) in engagement_tick_by_servo
            }
            result["engagement_scan"]["back_off_targets_by_servo"] = {
                str(k): int(v) for k, v in back_off_targets.items()
            }
            move = self._apply_group_command(
                servo_service=servo_service,
                telemetry=telemetry,
                targets_by_servo=back_off_targets,
                reason="pretension_engagement_back_off",
            )
            result["move_count"] += int(move.get("move_count", 0))
            result["travel_ticks"] += int(move.get("travel_ticks", 0))
            if not bool(move.get("success")):
                result["stop_reason"] = str(move.get("stop_reason") or "engagement_back_off_failed")
                return result
            after = self._advanced_measurement(
                servo_service=servo_service,
                tracker_service=tracker_service,
                servo_ids=servo_ids,
                baseline_current_ma_by_servo=baseline_current_ma_by_servo,
                target_xy_mm=target_xy_mm,
                commanded_positions_ticks=back_off_targets,
                startup_reference_ticks_by_servo=startup_reference_ticks_by_servo,
                trust_status=("runtime_tip" if tracker_service is not None else "current_only_lower_trust"),
            )
            row = self._advanced_stage_row(
                mode_kind="conservative_startup",
                run_index=run_index,
                stage="engagement_back_off",
                target_xy_mm=target_xy_mm,
                measurement=after,
                extra={
                    "back_off_ticks": int(back_off_ticks),
                    "back_off_targets_by_servo": {str(k): int(v) for k, v in back_off_targets.items()},
                    "engagement_tick_by_servo": {str(k): int(v) for k, v in engagement_tick_by_servo.items()},
                    "move": move,
                },
            )
            trace_rows.append(row)
            self._add_staged_sample(session, phase="pretension_engagement_back_off", run_index=run_index, step_index=len(trace_rows) - 1, payload=row)

        # --- Phase 3: fine take-up to target band -----------------------
        for iteration in range(max_iterations):
            session.raise_if_stop_requested()
            if deadline_monotonic is not None and float(session.context.monotonic_fn()) > float(deadline_monotonic):
                result["stop_reason"] = "runtime_budget_exhausted"
                return result
            measurement = self._advanced_measurement(
                servo_service=servo_service,
                tracker_service=tracker_service,
                servo_ids=servo_ids,
                baseline_current_ma_by_servo=baseline_current_ma_by_servo,
                target_xy_mm=target_xy_mm,
                startup_reference_ticks_by_servo=startup_reference_ticks_by_servo,
                trust_status=("runtime_tip" if tracker_service is not None else "current_only_lower_trust"),
            )
            self._merge_event_counts(result["telemetry_event_counts"], measurement.get("telemetry_event_counts"))
            result["packet_retry_count"] += int(measurement.get("packet_retry_count", 0) or 0)
            if measurement.get("telemetry_fail_closed_reason"):
                result["stop_reason"] = str(measurement["telemetry_fail_closed_reason"])
                return result
            loads = dict(measurement.get("load_proxy_current_ma") or measurement.get("current_above_baseline_ma") or {})
            valid_loads = [float(loads.get(int(sid))) for sid in servo_ids if loads.get(int(sid)) is not None]
            if len(valid_loads) == len(servo_ids) and min(valid_loads) >= target_load:
                result["stop_reason"] = "within_load_target"
                return result
            if len(valid_loads) == len(servo_ids) and max(valid_loads) >= high_load:
                result["stop_reason"] = "high_load_transition_to_trim"
                return result
            if max_travel > 0 and int(result["travel_ticks"]) >= max_travel:
                result["stop_reason"] = "travel_budget_exhausted"
                return result
            telemetry, policy = self._read_live_telemetry_with_policy(servo_service, servo_ids)
            self._merge_event_counts(result["telemetry_event_counts"], policy.get("event_counts"))
            result["packet_retry_count"] += int(policy.get("packet_retry_count", 0) or 0)
            if policy.get("telemetry_fail_closed_reason"):
                result["stop_reason"] = str(policy["telemetry_fail_closed_reason"])
                return result
            # Step inward only servos that are still below the target band.
            targets = {}
            for servo_id in servo_ids:
                sid = int(servo_id)
                load = loads.get(sid)
                entry = telemetry.get(sid)
                if entry is None or entry.present_position is None:
                    result["stop_reason"] = "missing_position"
                    return result
                if load is None or float(load) < float(target_load):
                    targets[sid] = int(entry.present_position) - int(refine_step)
            if not targets:
                # Every servo already at or above target_load — convergence.
                result["stop_reason"] = "within_load_target"
                return result
            move = self._apply_group_command(
                servo_service=servo_service,
                telemetry=telemetry,
                targets_by_servo=targets,
                reason="pretension_fine_take_up",
            )
            result["move_count"] += int(move.get("move_count", 0))
            result["travel_ticks"] += int(move.get("travel_ticks", 0))
            if not bool(move.get("success")):
                result["stop_reason"] = str(move.get("stop_reason") or "fine_take_up_move_failed")
                return result
            after = self._advanced_measurement(
                servo_service=servo_service,
                tracker_service=tracker_service,
                servo_ids=servo_ids,
                baseline_current_ma_by_servo=baseline_current_ma_by_servo,
                target_xy_mm=target_xy_mm,
                commanded_positions_ticks=targets,
                startup_reference_ticks_by_servo=startup_reference_ticks_by_servo,
                trust_status=("runtime_tip" if tracker_service is not None else "current_only_lower_trust"),
            )
            row = self._advanced_stage_row(
                mode_kind="conservative_startup",
                run_index=run_index,
                stage="fine_take_up",
                target_xy_mm=target_xy_mm,
                measurement=after,
                extra={
                    "iteration": int(iteration + 1),
                    "takeup_target_load_proxy_ma": float(target_load),
                    "high_load_proxy_ma": float(high_load),
                    "fine_step_ticks": int(refine_step),
                    "stepped_servo_ids": sorted(int(sid) for sid in targets),
                    "move": move,
                },
            )
            trace_rows.append(row)
            self._add_staged_sample(session, phase="pretension_fine_take_up", run_index=run_index, step_index=len(trace_rows) - 1, payload=row)
        result["stop_reason"] = "fine_take_up_iteration_limit"
        return result

    def _reliable_pretension_mode(self) -> str:
        mode = str(getattr(self.config, "mode", "") or "").strip().lower()
        strategy = str(getattr(self.config, "staged_strategy", "") or "").strip().lower()
        if mode in {"single_segment_characterization", "pretension_characterization", "characterization"}:
            return "characterization"
        if strategy in {"characterization", "identification"}:
            return "characterization"
        return "conservative_startup"

    @staticmethod
    def _current_stats(values: list[float]) -> dict[str, float | int | None]:
        cleaned = [float(value) for value in values if value is not None]
        if not cleaned:
            return {
                "sample_count": 0,
                "mean_current_ma": None,
                "std_current_ma": None,
                "min_current_ma": None,
                "max_current_ma": None,
                "peak_to_peak_current_ma": None,
            }
        mean_value = float(sum(cleaned) / len(cleaned))
        std_value = (
            float(math.sqrt(sum((value - mean_value) ** 2 for value in cleaned) / len(cleaned)))
            if len(cleaned) > 1
            else 0.0
        )
        return {
            "sample_count": int(len(cleaned)),
            "mean_current_ma": mean_value,
            "std_current_ma": std_value,
            "min_current_ma": float(min(cleaned)),
            "max_current_ma": float(max(cleaned)),
            "peak_to_peak_current_ma": float(max(cleaned) - min(cleaned)),
        }

    def _advanced_measurement(
        self,
        *,
        servo_service,
        tracker_service,
        servo_ids: list[int],
        baseline_current_ma_by_servo: dict[int, float],
        target_xy_mm: list[float],
        commanded_positions_ticks: dict[int, int | None] | None = None,
        startup_reference_ticks_by_servo: dict[int, int | None] | None = None,
        trust_status: str = "runtime_tip",
    ) -> dict[str, Any]:
        telemetry, telemetry_policy = self._read_live_telemetry_with_policy(servo_service, servo_ids)
        positions: dict[int, int | None] = {}
        raw_current: dict[int, int | None] = {}
        signed_raw_current: dict[int, int | None] = {}
        filtered_current: dict[int, float | None] = {}
        baseline_current: dict[int, float | None] = {}
        above_baseline: dict[int, float | None] = {}
        tendon_displacement_mm: dict[int, float | None] = {}
        current_validity: dict[int, str] = {}
        position_error: dict[int, int | None] = {}
        missing_fields: list[str] = []
        startup_refs = dict(startup_reference_ticks_by_servo or {})
        for servo_id in servo_ids:
            entry = telemetry.get(int(servo_id))
            if entry is None:
                positions[int(servo_id)] = None
                raw_current[int(servo_id)] = None
                signed_raw_current[int(servo_id)] = None
                baseline_current[int(servo_id)] = baseline_current_ma_by_servo.get(int(servo_id))
                filtered_current[int(servo_id)] = None
                above_baseline[int(servo_id)] = None
                tendon_displacement_mm[int(servo_id)] = None
                current_validity[int(servo_id)] = "missing"
                position_error[int(servo_id)] = None
                missing_fields.append(f"servo_{int(servo_id)}_present_current_ma")
                missing_fields.append(f"servo_{int(servo_id)}_present_position")
                continue
            positions[int(servo_id)] = entry.present_position
            raw_current[int(servo_id)] = entry.present_current_ma
            signed_raw_current[int(servo_id)] = entry.present_current_ma
            baseline = baseline_current_ma_by_servo.get(int(servo_id))
            baseline_current[int(servo_id)] = baseline
            filtered_current[int(servo_id)] = float(entry.present_current_ma) if entry.present_current_ma is not None else None
            above_baseline[int(servo_id)] = (
                abs(float(entry.present_current_ma) - float(baseline))
                if entry.present_current_ma is not None and baseline is not None
                else None
            )
            reference_tick = startup_refs.get(int(servo_id))
            tendon_displacement_mm[int(servo_id)] = (
                servo_service.mapper.ticks_to_displacement_mm(int(reference_tick) - int(entry.present_position))
                if reference_tick is not None and entry.present_position is not None
                else None
            )
            if entry.present_current_ma is None:
                current_validity[int(servo_id)] = "missing"
                missing_fields.append(f"servo_{int(servo_id)}_present_current_ma")
            elif entry.telemetry_error:
                current_validity[int(servo_id)] = self._classify_telemetry_error(entry.telemetry_error) or "telemetry_error"
            elif int(telemetry_policy.get("packet_retry_count", 0) or 0) > 0:
                current_validity[int(servo_id)] = "valid_after_retry"
            else:
                current_validity[int(servo_id)] = "valid"
            if entry.present_position is None:
                missing_fields.append(f"servo_{int(servo_id)}_present_position")
            commanded = (commanded_positions_ticks or {}).get(int(servo_id))
            position_error[int(servo_id)] = (
                int(entry.present_position) - int(commanded)
                if entry.present_position is not None and commanded is not None
                else None
            )
        load_values = [float(value) for value in above_baseline.values() if value is not None]
        load_balance_error_ma = float(max(load_values) - min(load_values)) if load_values else None
        pair_balance_error_ma = None
        if len(servo_ids) >= 4 and all(above_baseline.get(int(sid)) is not None for sid in servo_ids):
            pair_balance_error_ma = max(
                abs(float(above_baseline[int(servo_ids[0])] - above_baseline[int(servo_ids[2])])),
                abs(float(above_baseline[int(servo_ids[1])] - above_baseline[int(servo_ids[3])])),
            )
        tip_xyz = self._staged_tip_position_mm(tracker_service) if tracker_service is not None else None
        tip_xy = [float(tip_xyz[0]), float(tip_xyz[1])] if tip_xyz is not None and len(tip_xyz) >= 2 else None
        tip_error = self._tip_xy_offset(tip_xyz, target_xy_mm)
        ownership = servo_service.bus_ownership_status()
        return {
            "timestamp_utc": _utc_now_iso(),
            "trust_status": str(trust_status),
            "target_xy_mm": list(target_xy_mm),
            "tip_xyz_mm": list(tip_xyz) if tip_xyz is not None else None,
            "tip_xy_mm": tip_xy,
            "tip_xy_error_mm": tip_error,
            "commanded_positions_ticks": {int(k): v for k, v in dict(commanded_positions_ticks or {}).items()},
            "measured_positions_ticks": positions,
            "position_error_ticks": position_error,
            "raw_current_ma": raw_current,
            "signed_raw_current_ma": signed_raw_current,
            "filtered_current_ma": filtered_current,
            "current_validity": current_validity,
            "baseline_current_ma": baseline_current,
            "current_above_baseline_ma": above_baseline,
            "load_proxy_current_ma": above_baseline,
            "startup_reference_ticks": {int(k): v for k, v in startup_refs.items()},
            "tendon_displacement_mm": tendon_displacement_mm,
            "pair_balance_error_ma": pair_balance_error_ma,
            "load_balance_error_ma": load_balance_error_ma,
            "ownership_state": {
                "active": bool(ownership.active),
                "owner": ownership.owner,
                "reason": ownership.reason,
                "held_by_current_thread": bool(ownership.held_by_current_thread),
            },
            "missing_fields": sorted(set(missing_fields)),
            "telemetry_event_counts": dict(telemetry_policy.get("event_counts") or {}),
            "packet_retry_count": int(telemetry_policy.get("packet_retry_count", 0) or 0),
            "telemetry_fail_closed_reason": telemetry_policy.get("telemetry_fail_closed_reason"),
            "missing_current_servo_ids": list(telemetry_policy.get("missing_current_servo_ids") or []),
            "missing_position_servo_ids": list(telemetry_policy.get("missing_position_servo_ids") or []),
            "stale_servo_ids": list(telemetry_policy.get("stale_servo_ids") or []),
        }

    @staticmethod
    def _json_safe_keyed(mapping: dict[Any, Any]) -> dict[str, Any]:
        return {str(k): v for k, v in dict(mapping or {}).items()}

    def _advanced_stage_row(
        self,
        *,
        mode_kind: str,
        run_index: int,
        stage: str,
        target_xy_mm: list[float],
        measurement: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = {
            "timestamp_utc": measurement.get("timestamp_utc", _utc_now_iso()),
            "run_id": f"run_{int(run_index) + 1:02d}",
            "run_index": int(run_index),
            "stage": str(stage),
            "mode": str(mode_kind),
            "servo_ids": [
                int(servo_id)
                for servo_id in sorted((measurement.get("measured_positions_ticks") or {}).keys())
            ],
            "target_xy_mm": list(target_xy_mm),
            "tip_xy_mm": measurement.get("tip_xy_mm"),
            "tip_xyz_mm": measurement.get("tip_xyz_mm"),
            "tip_xy_error_mm": measurement.get("tip_xy_error_mm"),
            "commanded_positions_ticks": self._json_safe_keyed(measurement.get("commanded_positions_ticks") or {}),
            "measured_positions_ticks": self._json_safe_keyed(measurement.get("measured_positions_ticks") or {}),
            "position_error_ticks": self._json_safe_keyed(measurement.get("position_error_ticks") or {}),
            "raw_current_ma": self._json_safe_keyed(measurement.get("raw_current_ma") or {}),
            "signed_raw_current_ma": self._json_safe_keyed(measurement.get("signed_raw_current_ma") or {}),
            "filtered_current_ma": self._json_safe_keyed(measurement.get("filtered_current_ma") or {}),
            "current_validity": self._json_safe_keyed(measurement.get("current_validity") or {}),
            "baseline_current_ma": self._json_safe_keyed(measurement.get("baseline_current_ma") or {}),
            "current_above_baseline_ma": self._json_safe_keyed(measurement.get("current_above_baseline_ma") or {}),
            "load_proxy_current_ma": self._json_safe_keyed(measurement.get("load_proxy_current_ma") or {}),
            "startup_reference_ticks": self._json_safe_keyed(measurement.get("startup_reference_ticks") or {}),
            "tendon_displacement_mm": self._json_safe_keyed(measurement.get("tendon_displacement_mm") or {}),
            "pair_balance_error_ma": measurement.get("pair_balance_error_ma"),
            "load_balance_error_ma": measurement.get("load_balance_error_ma"),
            "ownership_state": dict(measurement.get("ownership_state") or {}),
            "missing_fields": list(measurement.get("missing_fields") or []),
            "telemetry_event_counts": dict(measurement.get("telemetry_event_counts") or {}),
            "packet_retry_count": int(measurement.get("packet_retry_count", 0) or 0),
            "telemetry_fail_closed_reason": measurement.get("telemetry_fail_closed_reason"),
            "trust_status": measurement.get("trust_status"),
        }
        row.update(dict(extra or {}))
        return row

    def _characterize_current_noise(
        self,
        *,
        session: ExperimentSession,
        servo_service,
        tracker_service,
        servo_ids: list[int],
        run_index: int,
        target_xy_mm: list[float],
        mode_kind: str,
        trace_rows: list[dict[str, Any]],
        startup_reference_ticks_by_servo: dict[int, int | None] | None = None,
        trust_status: str = "runtime_tip",
    ) -> dict[str, Any]:
        sample_count = max(2, int(self.config.current_characterization_sample_count))
        samples_by_servo: dict[int, list[float]] = {int(servo_id): [] for servo_id in servo_ids}
        telemetry_event_counts: dict[str, int] = {}
        packet_retry_count = 0
        telemetry_fail_closed_reason: str | None = None
        for sample_index in range(sample_count):
            measurement = self._advanced_measurement(
                servo_service=servo_service,
                tracker_service=tracker_service,
                servo_ids=servo_ids,
                baseline_current_ma_by_servo={},
                target_xy_mm=target_xy_mm,
                startup_reference_ticks_by_servo=startup_reference_ticks_by_servo,
                trust_status=trust_status,
            )
            self._merge_event_counts(telemetry_event_counts, measurement.get("telemetry_event_counts"))
            packet_retry_count += int(measurement.get("packet_retry_count", 0) or 0)
            if measurement.get("telemetry_fail_closed_reason") and telemetry_fail_closed_reason is None:
                telemetry_fail_closed_reason = str(measurement["telemetry_fail_closed_reason"])
            for servo_id, current in measurement["raw_current_ma"].items():
                if current is not None:
                    samples_by_servo[int(servo_id)].append(float(current))
            row = self._advanced_stage_row(
                mode_kind=mode_kind,
                run_index=run_index,
                stage="current_characterization_sample",
                target_xy_mm=target_xy_mm,
                measurement=measurement,
                extra={"sample_index": int(sample_index)},
            )
            trace_rows.append(row)
            self._add_staged_sample(session, phase="pretension_current_characterization", run_index=run_index, step_index=len(trace_rows) - 1, payload=row)
        by_servo = {int(servo_id): self._current_stats(values) for servo_id, values in samples_by_servo.items()}
        noise_floor = max(
            [
                float(stats.get("std_current_ma") or 0.0)
                for stats in by_servo.values()
            ]
            or [0.0]
        )
        peak_to_peak = max(
            [
                float(stats.get("peak_to_peak_current_ma") or 0.0)
                for stats in by_servo.values()
            ]
            or [0.0]
        )
        useful_delta = max(
            float(self.config.min_meaningful_current_delta_ma),
            peak_to_peak,
            noise_floor * float(self.config.current_noise_multiplier),
        )
        summary = {
            "sample_count": int(sample_count),
            "by_servo": {str(k): v for k, v in by_servo.items()},
            "max_noise_std_ma": float(noise_floor),
            "max_peak_to_peak_ma": float(peak_to_peak),
            "noise_multiplier": float(self.config.current_noise_multiplier),
            "max_useful_current_delta_ma": float(useful_delta),
            "telemetry_event_counts": dict(sorted(telemetry_event_counts.items())),
            "packet_retry_count": int(packet_retry_count),
            "telemetry_fail_closed_reason": telemetry_fail_closed_reason,
            "current_units": "mA servo-reported current estimate; signed raw samples retained and load proxy uses absolute delta from baseline",
            "interpretation": "current is treated as a relative load proxy; deltas below useful threshold are not considered reliable",
        }
        row = {
            "timestamp_utc": _utc_now_iso(),
            "run_id": f"run_{int(run_index) + 1:02d}",
            "run_index": int(run_index),
            "stage": "current_characterization",
            "mode": str(mode_kind),
            "target_xy_mm": list(target_xy_mm),
            **summary,
        }
        trace_rows.append(row)
        self._add_staged_sample(session, phase="pretension_current_characterization", run_index=run_index, step_index=len(trace_rows) - 1, payload=row)
        return summary

    def _apply_pair_command(
        self,
        *,
        servo_service,
        telemetry: dict[int, Any],
        sid_a: int,
        delta_a: int,
        sid_b: int,
        delta_b: int,
        reason: str,
    ) -> dict[str, Any]:
        pos_a = telemetry[int(sid_a)].present_position
        pos_b = telemetry[int(sid_b)].present_position
        if pos_a is None or pos_b is None:
            return {
                "success": False,
                "stop_reason": "missing_position",
                "clipped_move_count": 0,
                "move_count": 0,
                "travel_ticks": 0,
                "commanded_positions_ticks": {},
            }
        target_a = int(pos_a) + int(delta_a)
        target_b = int(pos_b) + int(delta_b)
        move = self._apply_group_command(
            servo_service=servo_service,
            telemetry=telemetry,
            targets_by_servo={int(sid_a): target_a, int(sid_b): target_b},
            reason=reason,
            require_opposed_pair=True,
        )
        move["clipped_move_count"] = 0
        return move

    def _run_pair_characterization_sequence(
        self,
        *,
        session: ExperimentSession,
        servo_service,
        tracker_service,
        servo_ids: list[int],
        run_index: int,
        target_xy_mm: list[float],
        baseline_current_ma_by_servo: dict[int, float],
        trace_rows: list[dict[str, Any]],
        startup_reference_ticks_by_servo: dict[int, int | None] | None = None,
        deadline_monotonic: float | None = None,
    ) -> dict[str, Any]:
        total = {"move_count": 0, "travel_ticks": 0, "clipped_move_count": 0, "stop_reason": ""}
        step = max(1, int(self.config.characterization_step_ticks))
        cycles = max(1, int(self.config.characterization_pair_cycles))
        commands = [
            ("pair_1_3_positive", int(servo_ids[0]), -step, int(servo_ids[2]), step),
            ("pair_1_3_negative", int(servo_ids[0]), step, int(servo_ids[2]), -step),
            ("pair_2_4_positive", int(servo_ids[1]), -step, int(servo_ids[3]), step),
            ("pair_2_4_negative", int(servo_ids[1]), step, int(servo_ids[3]), -step),
        ]
        for cycle_index in range(cycles):
            for label, sid_a, delta_a, sid_b, delta_b in commands:
                session.raise_if_stop_requested()
                if deadline_monotonic is not None and float(session.context.monotonic_fn()) > float(deadline_monotonic):
                    total["stop_reason"] = "runtime_budget_exhausted"
                    return total
                before = self._advanced_measurement(
                    servo_service=servo_service,
                    tracker_service=tracker_service,
                    servo_ids=servo_ids,
                    baseline_current_ma_by_servo=baseline_current_ma_by_servo,
                    target_xy_mm=target_xy_mm,
                    startup_reference_ticks_by_servo=startup_reference_ticks_by_servo,
                    trust_status=("runtime_tip" if tracker_service is not None else "current_only_lower_trust"),
                )
                telemetry, policy = self._read_live_telemetry_with_policy(servo_service, servo_ids)
                self._merge_event_counts(total.setdefault("telemetry_event_counts", {}), policy.get("event_counts"))
                total["packet_retry_count"] = int(total.get("packet_retry_count", 0)) + int(policy.get("packet_retry_count", 0) or 0)
                if policy.get("telemetry_fail_closed_reason"):
                    total["stop_reason"] = str(policy["telemetry_fail_closed_reason"])
                    return total
                move = self._apply_pair_command(
                    servo_service=servo_service,
                    telemetry=telemetry,
                    sid_a=sid_a,
                    delta_a=delta_a,
                    sid_b=sid_b,
                    delta_b=delta_b,
                    reason="pretension_characterization_pair_step",
                )
                after = self._advanced_measurement(
                    servo_service=servo_service,
                    tracker_service=tracker_service,
                    servo_ids=servo_ids,
                    baseline_current_ma_by_servo=baseline_current_ma_by_servo,
                    target_xy_mm=target_xy_mm,
                    commanded_positions_ticks=move.get("commanded_positions_ticks"),
                    startup_reference_ticks_by_servo=startup_reference_ticks_by_servo,
                    trust_status=("runtime_tip" if tracker_service is not None else "current_only_lower_trust"),
                )
                self._merge_event_counts(total.setdefault("telemetry_event_counts", {}), after.get("telemetry_event_counts"))
                total["packet_retry_count"] = int(total.get("packet_retry_count", 0)) + int(after.get("packet_retry_count", 0) or 0)
                total["move_count"] += int(move.get("move_count", 0))
                total["travel_ticks"] += int(move.get("travel_ticks", 0))
                total["clipped_move_count"] += int(move.get("clipped_move_count", 0))
                if not bool(move.get("success")):
                    total["stop_reason"] = str(move.get("stop_reason") or "move_failed")
                tip_before = before.get("tip_xy_mm")
                tip_after = after.get("tip_xy_mm")
                tip_delta = (
                    [float(tip_after[0]) - float(tip_before[0]), float(tip_after[1]) - float(tip_before[1])]
                    if tip_before is not None and tip_after is not None
                    else None
                )
                row = self._advanced_stage_row(
                    mode_kind="characterization",
                    run_index=run_index,
                    stage="pair_characterization_step",
                    target_xy_mm=target_xy_mm,
                    measurement=after,
                    extra={
                        "cycle_index": int(cycle_index),
                        "pair_label": label,
                        "move": move,
                        "tip_xy_delta_mm": tip_delta,
                        "tip_xy_delta_per_tick_mm": (
                            None
                            if tip_delta is None or int(move.get("travel_ticks", 0)) <= 0
                            else [float(tip_delta[0]) / float(move["travel_ticks"]), float(tip_delta[1]) / float(move["travel_ticks"])]
                        ),
                    },
                )
                trace_rows.append(row)
                self._add_staged_sample(session, phase="pretension_pair_characterization", run_index=run_index, step_index=len(trace_rows) - 1, payload=row)
                if total["stop_reason"]:
                    return total
        total["stop_reason"] = "characterization_complete"
        return total

    def _run_conservative_startup_sequence(
        self,
        *,
        session: ExperimentSession,
        servo_service,
        tracker_service,
        servo_ids: list[int],
        run_index: int,
        target_xy_mm: list[float],
        baseline_current_ma_by_servo: dict[int, float],
        effective_load_tolerance_ma: float,
        trace_rows: list[dict[str, Any]],
        startup_reference_ticks_by_servo: dict[int, int | None] | None = None,
        deadline_monotonic: float | None = None,
    ) -> dict[str, Any]:
        result = {
            "move_count": 0,
            "travel_ticks": 0,
            "clipped_move_count": 0,
            "stop_reason": "",
            "converged": False,
            "packet_retry_count": 0,
            "telemetry_event_counts": {},
            "sign_flip_count": 0,
            "sign_flip_used_by_axis": {"x": False, "y": False},
        }
        step = max(1, int(self.config.conservative_step_ticks))
        best_error = math.inf
        worse_count = 0
        wrong_direction_count = 0
        axis_sign_correction = {"x": 1, "y": 1}
        for iteration in range(max(0, int(self.config.conservative_max_iterations))):
            session.raise_if_stop_requested()
            if deadline_monotonic is not None and float(session.context.monotonic_fn()) > float(deadline_monotonic):
                result["stop_reason"] = "runtime_budget_exhausted"
                break
            measurement = self._advanced_measurement(
                servo_service=servo_service,
                tracker_service=tracker_service,
                servo_ids=servo_ids,
                baseline_current_ma_by_servo=baseline_current_ma_by_servo,
                target_xy_mm=target_xy_mm,
                startup_reference_ticks_by_servo=startup_reference_ticks_by_servo,
                trust_status=("runtime_tip" if tracker_service is not None else "current_only_lower_trust"),
            )
            self._merge_event_counts(result["telemetry_event_counts"], measurement.get("telemetry_event_counts"))
            result["packet_retry_count"] += int(measurement.get("packet_retry_count", 0) or 0)
            if measurement.get("telemetry_fail_closed_reason"):
                result["stop_reason"] = str(measurement["telemetry_fail_closed_reason"])
                break
            tip_error = measurement.get("tip_xy_error_mm")
            load_error = measurement.get("load_balance_error_ma")
            pair_error = measurement.get("pair_balance_error_ma")
            if tip_error is not None and float(tip_error) <= float(self.config.tip_center_tolerance_mm):
                result["converged"] = True
                result["stop_reason"] = "within_tolerance"
                break
            if (
                result["travel_ticks"] >= int(self.config.conservative_max_cumulative_travel_ticks)
                and int(self.config.conservative_max_cumulative_travel_ticks) > 0
            ):
                result["stop_reason"] = "travel_budget_exhausted"
                break
            if tip_error is not None:
                if float(tip_error) < best_error:
                    best_error = float(tip_error)
                    worse_count = 0
                elif float(tip_error) > best_error + float(self.config.tip_divergence_stop_mm):
                    worse_count += 1
                    if worse_count >= 2:
                        result["stop_reason"] = "tip_diverging"
                        break
            if load_error is not None and float(load_error) > (2.0 * float(effective_load_tolerance_ma)):
                result["stop_reason"] = "load_proxy_spread_limit"
                break

            telemetry, policy = self._read_live_telemetry_with_policy(servo_service, servo_ids)
            self._merge_event_counts(result["telemetry_event_counts"], policy.get("event_counts"))
            result["packet_retry_count"] += int(policy.get("packet_retry_count", 0) or 0)
            if policy.get("telemetry_fail_closed_reason"):
                result["stop_reason"] = str(policy["telemetry_fail_closed_reason"])
                break
            commands: list[tuple[str, int, int, int, int]] = []
            tip_xy = measurement.get("tip_xy_mm")
            if tip_xy is not None:
                x_error = float(tip_xy[0]) - float(target_xy_mm[0])
                y_error = float(tip_xy[1]) - float(target_xy_mm[1])
                if abs(x_error) > float(self.config.tip_center_tolerance_mm) * 0.5:
                    x_delta = (
                        step
                        * (1 if x_error < 0.0 else -1)
                        * int(self.config.tip_center_x_sign or 1)
                        * int(axis_sign_correction["x"])
                    )
                    commands.append(("center_pair_1_3", int(servo_ids[0]), -int(x_delta), int(servo_ids[2]), int(x_delta)))
                if abs(y_error) > float(self.config.tip_center_tolerance_mm) * 0.5:
                    y_delta = (
                        step
                        * (1 if y_error < 0.0 else -1)
                        * int(self.config.tip_center_y_sign or 1)
                        * int(axis_sign_correction["y"])
                    )
                    commands.append(("center_pair_2_4", int(servo_ids[1]), -int(y_delta), int(servo_ids[3]), int(y_delta)))
            elif pair_error is not None and float(pair_error) > float(effective_load_tolerance_ma):
                loads = measurement.get("current_above_baseline_ma") or {}
                for label, left_index, right_index in (("balance_pair_1_3", 0, 2), ("balance_pair_2_4", 1, 3)):
                    sid_a = int(servo_ids[left_index])
                    sid_b = int(servo_ids[right_index])
                    if loads.get(sid_a) is None or loads.get(sid_b) is None:
                        continue
                    diff = float(loads[sid_a]) - float(loads[sid_b])
                    if abs(diff) <= float(effective_load_tolerance_ma):
                        continue
                    delta_a = step if diff > 0.0 else -step
                    delta_b = -step if diff > 0.0 else step
                    commands.append((label, sid_a, delta_a, sid_b, delta_b))
            if not commands:
                result["stop_reason"] = "no_actionable_error"
                break
            iteration_moves = []
            for label, sid_a, delta_a, sid_b, delta_b in commands[:2]:
                move = self._apply_pair_command(
                    servo_service=servo_service,
                    telemetry=telemetry,
                    sid_a=sid_a,
                    delta_a=delta_a,
                    sid_b=sid_b,
                    delta_b=delta_b,
                    reason="pretension_conservative_pair_step",
                )
                iteration_moves.append({"pair_label": label, **move})
                result["move_count"] += int(move.get("move_count", 0))
                result["travel_ticks"] += int(move.get("travel_ticks", 0))
                result["clipped_move_count"] += int(move.get("clipped_move_count", 0))
                if not bool(move.get("success")):
                    result["stop_reason"] = str(move.get("stop_reason") or "move_failed")
                    break
            after = self._advanced_measurement(
                servo_service=servo_service,
                tracker_service=tracker_service,
                servo_ids=servo_ids,
                baseline_current_ma_by_servo=baseline_current_ma_by_servo,
                target_xy_mm=target_xy_mm,
                startup_reference_ticks_by_servo=startup_reference_ticks_by_servo,
                trust_status=("runtime_tip" if tracker_service is not None else "current_only_lower_trust"),
            )
            self._merge_event_counts(result["telemetry_event_counts"], after.get("telemetry_event_counts"))
            result["packet_retry_count"] += int(after.get("packet_retry_count", 0) or 0)
            after_tip_error = after.get("tip_xy_error_mm")
            if tip_error is not None and after_tip_error is not None and iteration_moves:
                if float(after_tip_error) > float(tip_error) + float(self.config.tip_divergence_stop_mm):
                    moved_axes = []
                    for move_entry in iteration_moves:
                        label = str(move_entry.get("pair_label", ""))
                        if label == "center_pair_1_3" and "x" not in moved_axes:
                            moved_axes.append("x")
                        if label == "center_pair_2_4" and "y" not in moved_axes:
                            moved_axes.append("y")
                    axes_to_flip = [
                        axis
                        for axis in moved_axes
                        if not bool(result["sign_flip_used_by_axis"].get(axis))
                    ]
                    if axes_to_flip:
                        for axis in axes_to_flip:
                            axis_sign_correction[axis] *= -1
                            result["sign_flip_used_by_axis"][axis] = True
                            result["sign_flip_count"] += 1
                        wrong_direction_count = 0
                    else:
                        wrong_direction_count += 1
                        result["stop_reason"] = "tip_response_wrong_direction"
                else:
                    wrong_direction_count = 0
            row = self._advanced_stage_row(
                mode_kind="conservative_startup",
                run_index=run_index,
                stage="conservative_pair_step",
                target_xy_mm=target_xy_mm,
                measurement=after,
                extra={
                    "iteration": int(iteration + 1),
                    "effective_load_tolerance_ma": float(effective_load_tolerance_ma),
                    "moves": iteration_moves,
                    "sign_flip_count": int(result["sign_flip_count"]),
                    "sign_flip_used_by_axis": dict(result["sign_flip_used_by_axis"]),
                    "axis_sign_correction": dict(axis_sign_correction),
                    "partial_quality_components": {
                        "tip_centering": self._bounded_score(after.get("tip_xy_error_mm"), float(self.config.accept_max_final_tip_xy_offset_mm), missing_score=45.0),
                        "pair_balance": self._bounded_score(after.get("pair_balance_error_ma"), float(effective_load_tolerance_ma), missing_score=55.0),
                        "load_balance": self._bounded_score(after.get("load_balance_error_ma"), float(effective_load_tolerance_ma), missing_score=55.0),
                    },
                },
            )
            trace_rows.append(row)
            self._add_staged_sample(session, phase="pretension_conservative_startup", run_index=run_index, step_index=len(trace_rows) - 1, payload=row)
            if result["stop_reason"]:
                break
        if not result["stop_reason"]:
            result["stop_reason"] = "iteration_limit"
        return result

    # ---------------------------------------------------------------
    # New: manual-baseline capture + Jacobian-learned tip centering
    # ---------------------------------------------------------------

    def _load_or_capture_manual_baselines(
        self,
        *,
        session: ExperimentSession,
        servo_service,
        tracker_service,
        servo_ids: list[int],
    ) -> list[dict[str, Any]]:
        """Return manual-baseline records.

        Resolution order:
        - If ``manual_baseline_record_path`` is set, load it (the GUI captures
          baselines separately and writes the file).
        - Else if ``manual_baseline_capture_count > 0``, run the operator-paced
          inline capture phase. The phase pauses ``manual_baseline_pause_s``
          seconds between captures so the operator can re-tension manually
          between them. ``session.raise_if_stop_requested()`` is consulted each
          pause tick so the GUI can advance early.
        - Else return an empty list (no manual baseline comparison).
        """
        path = str(getattr(self.config, "manual_baseline_record_path", "") or "").strip()
        if path:
            resolved = Path(path)
            if not resolved.is_absolute():
                resolved = Path(session.context.project_root) / resolved
            if not resolved.exists():
                raise RuntimeError(
                    f"manual_baseline_record_path is set but the file does not exist: {resolved}."
                )
            try:
                payload = json.loads(resolved.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to parse manual_baseline_record_path {resolved}: {exc}"
                ) from exc
            records = payload if isinstance(payload, list) else payload.get("records") or []
            if not isinstance(records, list):
                raise RuntimeError(
                    f"manual_baseline_record_path payload must be a list of records: {resolved}"
                )
            return [dict(item) for item in records if isinstance(item, dict)]

        capture_count = max(0, int(getattr(self.config, "manual_baseline_capture_count", 0) or 0))
        if capture_count <= 0:
            return []
        pause_s = max(0.0, float(getattr(self.config, "manual_baseline_pause_s", 15.0) or 15.0))
        records: list[dict[str, Any]] = []
        for index in range(capture_count):
            session.raise_if_stop_requested()
            session.add_warning(
                f"Pretension manual baseline capture {index + 1}/{capture_count}: "
                "hand-tension the spine, then wait. The next capture happens automatically "
                f"after {pause_s:.0f}s, or click Stop to advance early."
            )
            # Use the standard measurement helper so the record carries the same
            # fields as algorithm runs (positions, currents, tip XY).
            measurement = self._advanced_measurement(
                servo_service=servo_service,
                tracker_service=tracker_service,
                servo_ids=servo_ids,
                baseline_current_ma_by_servo={},
                target_xy_mm=self._target_xy(),
                trust_status="runtime_tip" if tracker_service is not None else "current_only_lower_trust",
            )
            record = {
                "index": int(index),
                "timestamp_utc": _utc_now_iso(),
                "source": "manual_inline_capture",
                "positions_by_servo": dict(measurement.get("measured_positions_ticks") or {}),
                "currents_ma_by_servo": dict(measurement.get("raw_current_ma") or {}),
                "tip_xy_mm": measurement.get("tip_xy_mm"),
                "tip_xyz_mm": measurement.get("tip_xyz_mm"),
                "tip_xy_error_mm": measurement.get("tip_xy_error_mm"),
                "load_balance_error_ma": measurement.get("load_balance_error_ma"),
                "pair_balance_error_ma": measurement.get("pair_balance_error_ma"),
            }
            records.append(record)
            # Pause to let the operator re-tension before the next capture.
            elapsed = 0.0
            poll_step = 0.5
            while elapsed < pause_s and index + 1 < capture_count:
                try:
                    session.raise_if_stop_requested()
                except Exception:
                    raise
                session.context.sleep_fn(poll_step)
                elapsed += poll_step
        return records

    def _run_jacobian_tip_centering(
        self,
        *,
        session: ExperimentSession,
        servo_service,
        tracker_service,
        servo_ids: list[int],
        run_index: int,
        target_xy_mm: list[float],
        baseline_current_ma_by_servo: dict[int, float],
        effective_load_tolerance_ma: float,
        trace_rows: list[dict[str, Any]],
        startup_reference_ticks_by_servo: dict[int, int | None] | None = None,
        deadline_monotonic: float | None = None,
    ) -> dict[str, Any]:
        """Jacobian-based tip centering.

        Phase 1 (probe): with the spine in its current take-up state, command
        each pair +``jacobian_probe_step_ticks`` (axis_a then axis_b), measure
        tip XY delta, then command back to the pre-probe position. Build the
        2x2 Jacobian J = [[dx/da, dx/db], [dy/da, dy/db]] (units mm per tick).

        Phase 2 (drive): until tip error is within tolerance or iteration limit:
          read tip XY -> compute error = current - target -> compute pair deltas
          [da, db] = -J^-1 * error * step_gain, clipped to
          ``jacobian_max_pair_step_ticks``. Apply paired commands.

        If the probe response is below ``jacobian_min_observable_tip_delta_mm``,
        the column is unreliable and the routine returns immediately with
        ``stop_reason='jacobian_probe_unobservable'`` so the caller can fall
        back to the paired_then_tip routine.
        """
        result: dict[str, Any] = {
            "move_count": 0,
            "travel_ticks": 0,
            "clipped_move_count": 0,
            "stop_reason": "",
            "converged": False,
            "packet_retry_count": 0,
            "telemetry_event_counts": {},
            "jacobian": None,
            "jacobian_probe_response_mm": {},
            "jacobian_iterations": 0,
        }
        if tracker_service is None:
            result["stop_reason"] = "jacobian_requires_tracker"
            return result
        probe_ticks = max(1, int(getattr(self.config, "jacobian_probe_step_ticks", 30)))
        min_response_mm = max(0.0, float(getattr(self.config, "jacobian_min_observable_tip_delta_mm", 0.20)))
        step_gain = max(0.05, min(1.5, float(getattr(self.config, "jacobian_step_gain", 0.8))))
        max_pair_step = max(1, int(getattr(self.config, "jacobian_max_pair_step_ticks", 25)))
        max_iterations = max(1, int(getattr(self.config, "tip_center_max_iterations", 16)))
        tip_tolerance_mm = max(0.0, float(getattr(self.config, "tip_center_tolerance_mm", 1.0)))
        divergence_stop_mm = max(0.0, float(getattr(self.config, "tip_divergence_stop_mm", 2.0)))

        pairs_map = session.context.settings.robot.active_segment_pairs() or {}
        # Resolve axis-a / axis-b pair servo IDs from the explicit pairs map so
        # this routine respects the segment's declared pairing instead of
        # assuming positional indexing.
        axis_a = [int(v) for v in (pairs_map.get("axis_a") or [])]
        axis_b = [int(v) for v in (pairs_map.get("axis_b") or [])]
        if len(axis_a) != 2 or len(axis_b) != 2:
            # Fall back to the positional convention used by paired_then_tip
            # so callers without explicit pair metadata still work.
            if len(servo_ids) >= 4:
                axis_a = [int(servo_ids[0]), int(servo_ids[2])]
                axis_b = [int(servo_ids[1]), int(servo_ids[3])]
            else:
                result["stop_reason"] = "jacobian_requires_pair_metadata"
                return result

        def _read_tip() -> list[float] | None:
            m = self._advanced_measurement(
                servo_service=servo_service,
                tracker_service=tracker_service,
                servo_ids=servo_ids,
                baseline_current_ma_by_servo=baseline_current_ma_by_servo,
                target_xy_mm=target_xy_mm,
                startup_reference_ticks_by_servo=startup_reference_ticks_by_servo,
                trust_status="runtime_tip",
            )
            self._merge_event_counts(result["telemetry_event_counts"], m.get("telemetry_event_counts"))
            result["packet_retry_count"] += int(m.get("packet_retry_count", 0) or 0)
            tip_xy = m.get("tip_xy_mm")
            if tip_xy is None:
                return None
            return [float(tip_xy[0]), float(tip_xy[1])]

        def _apply_probe(axis_pair: list[int], delta: int) -> dict[str, Any]:
            telemetry, policy = self._read_live_telemetry_with_policy(servo_service, axis_pair)
            self._merge_event_counts(result["telemetry_event_counts"], policy.get("event_counts"))
            result["packet_retry_count"] += int(policy.get("packet_retry_count", 0) or 0)
            move = self._apply_pair_command(
                servo_service=servo_service,
                telemetry=telemetry,
                sid_a=int(axis_pair[0]),
                delta_a=-int(delta),
                sid_b=int(axis_pair[1]),
                delta_b=int(delta),
                reason="pretension_jacobian_probe",
            )
            result["move_count"] += int(move.get("move_count", 0))
            result["travel_ticks"] += int(move.get("travel_ticks", 0))
            result["clipped_move_count"] += int(move.get("clipped_move_count", 0))
            return move

        # --- Phase 1: probe -------------------------------------------------
        pre_tip = _read_tip()
        if pre_tip is None:
            result["stop_reason"] = "jacobian_missing_tip_pose"
            return result

        # Probe axis A: +probe_ticks on first pair member, -probe_ticks on second.
        move_a = _apply_probe(axis_a, probe_ticks)
        if not bool(move_a.get("success")):
            result["stop_reason"] = f"jacobian_probe_a_failed:{move_a.get('stop_reason')}"
            # Try to restore before bailing.
            _apply_probe(axis_a, -probe_ticks)
            return result
        session.context.sleep_fn(max(0.05, float(self.config.settle_verify_time_s)))
        tip_after_a = _read_tip()
        # Restore axis A to baseline.
        restore_a = _apply_probe(axis_a, -probe_ticks)
        if tip_after_a is None or not bool(restore_a.get("success")):
            result["stop_reason"] = "jacobian_probe_a_unreadable"
            return result
        dxA = float(tip_after_a[0]) - float(pre_tip[0])
        dyA = float(tip_after_a[1]) - float(pre_tip[1])

        # Probe axis B.
        move_b = _apply_probe(axis_b, probe_ticks)
        if not bool(move_b.get("success")):
            result["stop_reason"] = f"jacobian_probe_b_failed:{move_b.get('stop_reason')}"
            _apply_probe(axis_b, -probe_ticks)
            return result
        session.context.sleep_fn(max(0.05, float(self.config.settle_verify_time_s)))
        tip_after_b = _read_tip()
        restore_b = _apply_probe(axis_b, -probe_ticks)
        if tip_after_b is None or not bool(restore_b.get("success")):
            result["stop_reason"] = "jacobian_probe_b_unreadable"
            return result
        dxB = float(tip_after_b[0]) - float(pre_tip[0])
        dyB = float(tip_after_b[1]) - float(pre_tip[1])

        # Build 2x2 Jacobian (tip mm per pair tick).
        per_tick = float(probe_ticks)
        J = [
            [dxA / per_tick, dxB / per_tick],
            [dyA / per_tick, dyB / per_tick],
        ]
        result["jacobian"] = J
        result["jacobian_probe_response_mm"] = {
            "axis_a_response_mm": [dxA, dyA],
            "axis_b_response_mm": [dxB, dyB],
            "probe_ticks": int(probe_ticks),
        }
        # Reject if either probe column is below the minimum observable response.
        norm_a = (dxA * dxA + dyA * dyA) ** 0.5
        norm_b = (dxB * dxB + dyB * dyB) ** 0.5
        if norm_a < min_response_mm or norm_b < min_response_mm:
            result["stop_reason"] = "jacobian_probe_unobservable"
            return result
        det = J[0][0] * J[1][1] - J[0][1] * J[1][0]
        if abs(det) < 1e-9:
            result["stop_reason"] = "jacobian_singular"
            return result
        inv_det = 1.0 / det
        Jinv = [
            [J[1][1] * inv_det, -J[0][1] * inv_det],
            [-J[1][0] * inv_det, J[0][0] * inv_det],
        ]

        # --- Phase 2: drive -------------------------------------------------
        best_error_mm = math.inf
        for iteration in range(max_iterations):
            session.raise_if_stop_requested()
            if deadline_monotonic is not None and float(session.context.monotonic_fn()) > float(deadline_monotonic):
                result["stop_reason"] = "runtime_budget_exhausted"
                break
            tip_xy = _read_tip()
            if tip_xy is None:
                result["stop_reason"] = "jacobian_missing_tip_pose"
                break
            x_err = float(tip_xy[0]) - float(target_xy_mm[0])
            y_err = float(tip_xy[1]) - float(target_xy_mm[1])
            err_mag = (x_err * x_err + y_err * y_err) ** 0.5
            if err_mag <= tip_tolerance_mm:
                result["converged"] = True
                result["stop_reason"] = "jacobian_converged"
                break
            if err_mag < best_error_mm:
                best_error_mm = err_mag
            elif err_mag > best_error_mm + divergence_stop_mm:
                result["stop_reason"] = "jacobian_tip_diverging"
                break

            # [da, db] = -Jinv * [x_err, y_err] * step_gain (in tick units).
            da_raw = -(Jinv[0][0] * x_err + Jinv[0][1] * y_err) * step_gain
            db_raw = -(Jinv[1][0] * x_err + Jinv[1][1] * y_err) * step_gain
            da = max(-max_pair_step, min(max_pair_step, int(round(da_raw))))
            db = max(-max_pair_step, min(max_pair_step, int(round(db_raw))))
            if da == 0 and db == 0:
                # Numerical clamp landed at zero: nudge in the direction of error.
                da = (1 if da_raw > 0 else -1) if abs(da_raw) > 0 else 0
                db = (1 if db_raw > 0 else -1) if abs(db_raw) > 0 else 0
                if da == 0 and db == 0:
                    result["stop_reason"] = "jacobian_step_underflow"
                    break

            telemetry, policy = self._read_live_telemetry_with_policy(servo_service, servo_ids)
            self._merge_event_counts(result["telemetry_event_counts"], policy.get("event_counts"))
            result["packet_retry_count"] += int(policy.get("packet_retry_count", 0) or 0)
            iteration_moves: list[dict[str, Any]] = []
            if da != 0:
                move = self._apply_pair_command(
                    servo_service=servo_service,
                    telemetry=telemetry,
                    sid_a=int(axis_a[0]),
                    delta_a=-int(da),
                    sid_b=int(axis_a[1]),
                    delta_b=int(da),
                    reason="pretension_jacobian_drive_axis_a",
                )
                iteration_moves.append({"pair_label": "axis_a", **move})
                result["move_count"] += int(move.get("move_count", 0))
                result["travel_ticks"] += int(move.get("travel_ticks", 0))
                if not bool(move.get("success")):
                    result["stop_reason"] = f"jacobian_drive_a_failed:{move.get('stop_reason')}"
                    break
            if db != 0:
                telemetry, policy2 = self._read_live_telemetry_with_policy(servo_service, servo_ids)
                self._merge_event_counts(result["telemetry_event_counts"], policy2.get("event_counts"))
                move = self._apply_pair_command(
                    servo_service=servo_service,
                    telemetry=telemetry,
                    sid_a=int(axis_b[0]),
                    delta_a=-int(db),
                    sid_b=int(axis_b[1]),
                    delta_b=int(db),
                    reason="pretension_jacobian_drive_axis_b",
                )
                iteration_moves.append({"pair_label": "axis_b", **move})
                result["move_count"] += int(move.get("move_count", 0))
                result["travel_ticks"] += int(move.get("travel_ticks", 0))
                if not bool(move.get("success")):
                    result["stop_reason"] = f"jacobian_drive_b_failed:{move.get('stop_reason')}"
                    break

            session.context.sleep_fn(max(0.05, float(self.config.settle_verify_time_s)))
            after_m = self._advanced_measurement(
                servo_service=servo_service,
                tracker_service=tracker_service,
                servo_ids=servo_ids,
                baseline_current_ma_by_servo=baseline_current_ma_by_servo,
                target_xy_mm=target_xy_mm,
                startup_reference_ticks_by_servo=startup_reference_ticks_by_servo,
                trust_status="runtime_tip",
            )
            self._merge_event_counts(result["telemetry_event_counts"], after_m.get("telemetry_event_counts"))
            result["packet_retry_count"] += int(after_m.get("packet_retry_count", 0) or 0)
            row = self._advanced_stage_row(
                mode_kind="conservative_startup",
                run_index=run_index,
                stage="jacobian_tip_centering",
                target_xy_mm=target_xy_mm,
                measurement=after_m,
                extra={
                    "iteration": int(iteration + 1),
                    "jacobian": J,
                    "jacobian_inverse": Jinv,
                    "moves": iteration_moves,
                    "axis_a_pair": list(axis_a),
                    "axis_b_pair": list(axis_b),
                    "pair_deltas": {"axis_a_ticks": int(da), "axis_b_ticks": int(db)},
                    "raw_pair_deltas_unclipped": {"axis_a_ticks": float(da_raw), "axis_b_ticks": float(db_raw)},
                    "tip_error_mm": [float(x_err), float(y_err)],
                    "tip_error_magnitude_mm": float(err_mag),
                },
            )
            trace_rows.append(row)
            self._add_staged_sample(
                session,
                phase="pretension_jacobian_tip_centering",
                run_index=run_index,
                step_index=len(trace_rows) - 1,
                payload=row,
            )
            result["jacobian_iterations"] = int(iteration + 1)
        if not result["stop_reason"]:
            result["stop_reason"] = "jacobian_iteration_limit"
        return result

    @staticmethod
    def _series_std(values: list[float]) -> float | None:
        if len(values) < 2:
            return 0.0 if values else None
        mean_value = float(sum(values) / len(values))
        return float(math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values)))

    def _reliable_pretension_metrics(
        self,
        *,
        mode_kind: str,
        servo_ids: list[int],
        repeat_runs: int,
        accepted_runs: int,
        failure_counts: dict[str, int],
        run_rows: list[dict[str, Any]],
        trace_rows: list[dict[str, Any]],
        final_tip_xy_points_mm: list[list[float]],
        quality_scores: list[float],
        manual_startup_artifact: dict[str, Any],
    ) -> dict[str, Any]:
        per_servo_position_std: dict[str, float | None] = {}
        per_servo_current_std: dict[str, float | None] = {}
        for servo_id in servo_ids:
            final_positions = [
                float(row["final_position_ticks_by_servo"][str(int(servo_id))])
                for row in run_rows
                if row.get("final_position_ticks_by_servo", {}).get(str(int(servo_id))) is not None
            ]
            final_currents = [
                float(row["final_current_ma_by_servo"][str(int(servo_id))])
                for row in run_rows
                if row.get("final_current_ma_by_servo", {}).get(str(int(servo_id))) is not None
            ]
            per_servo_position_std[str(int(servo_id))] = self._series_std(final_positions)
            per_servo_current_std[str(int(servo_id))] = self._series_std(final_currents)
        tip_xy_std_mm = None
        if final_tip_xy_points_mm:
            x_std = self._series_std([float(point[0]) for point in final_tip_xy_points_mm]) or 0.0
            y_std = self._series_std([float(point[1]) for point in final_tip_xy_points_mm]) or 0.0
            tip_xy_std_mm = float(math.sqrt((x_std * x_std) + (y_std * y_std)))
        aggregate_telemetry_counts: dict[str, int] = {}
        aggregate_packet_retries = 0
        for row in run_rows:
            self._merge_event_counts(aggregate_telemetry_counts, row.get("telemetry_event_counts"))
            aggregate_packet_retries += int(row.get("packet_retry_count", 0) or 0)
        return {
            "mode": "single_segment_staged",
            "staged_strategy": str(getattr(self.config, "staged_strategy", "conservative_startup") or "conservative_startup"),
            "algorithm_mode": str(mode_kind),
            "algorithm": "reliable_advanced_4servo_pretension",
            "servo_ids": list(servo_ids),
            "repeat_runs": int(repeat_runs),
            "pretension_start_mode": (
                str(self.config.pretension_start_mode).strip().lower()
                if self.config.pretension_start_mode not in (None, "")
                else "live_default"
            ),
            "run_count": int(len(run_rows)),
            "accepted_run_count": int(accepted_runs),
            "accepted_run_fraction": float(accepted_runs / len(run_rows)) if run_rows else 0.0,
            "failure_reason_counts": dict(sorted(failure_counts.items())),
            "telemetry_event_counts": dict(sorted(aggregate_telemetry_counts.items())),
            "packet_retry_count": int(aggregate_packet_retries),
            "final_position_std_ticks_by_servo": per_servo_position_std,
            "final_current_std_ma_by_servo": per_servo_current_std,
            "final_tip_xy_std_mm": tip_xy_std_mm,
            "quality_score_mean_0_100": float(sum(quality_scores) / len(quality_scores)) if quality_scores else None,
            "quality_score_std_0_100": self._series_std(quality_scores),
            "quality_scores_0_100": list(quality_scores),
            "manual_startup_artifact": manual_startup_artifact,
            "advanced_startup_artifacts": [
                {
                    "run_index": int(row.get("run_index", 0)),
                    "accepted": bool(row.get("accepted")),
                    "quality_score_0_100": row.get("quality_score_0_100"),
                    "final_position_ticks_by_servo": row.get("final_position_ticks_by_servo"),
                    "final_current_ma_by_servo": row.get("final_current_ma_by_servo"),
                    "final_tip_xyz_mm": row.get("final_tip_xyz_mm"),
                    "final_tip_xy_offset_mm": row.get("final_tip_xy_offset_mm"),
                }
                for row in run_rows
            ],
            "run_rows": run_rows,
            "trace_rows": trace_rows,
            "units": {
                "current_ma": "mA",
                "signed_raw_current_ma": "mA servo-reported current estimate",
                "baseline_current_ma": "mA",
                "current_above_baseline_ma": "mA absolute delta from signed baseline; load proxy only",
                "load_proxy_current_ma": "mA absolute delta from signed baseline; load proxy only",
                "position_ticks": "ticks",
                "tendon_displacement_mm": "mm relative to startup reference",
                "packet_retry_count": "count",
                "travel_used_ticks": "ticks",
                "travel_used_mm": "mm",
                "tip_position_mm": "mm",
                "tip_xy_offset_mm": "mm",
                "load_balance_error_ma": "mA",
                "pair_balance_error_ma": "mA",
                "quality_score_0_100": "score",
            },
            "summary_requirements": {
                "force_status": (
                    "success"
                    if run_rows and accepted_runs == len(run_rows)
                    else "failed"
                    if accepted_runs == 0
                    else "partial_success"
                )
            },
        }

    @staticmethod
    def _bounded_score(value: float | None, tolerance: float, *, missing_score: float = 50.0) -> float:
        if value is None:
            return float(missing_score)
        tolerance = max(float(tolerance), 1e-6)
        value = max(0.0, float(value))
        if value <= tolerance:
            return 100.0
        return max(0.0, 100.0 * (1.0 - ((value - tolerance) / (2.0 * tolerance))))

    def _staged_quality_score(
        self,
        *,
        tip_xy_offset_mm: float | None,
        settle_tip_drift_mm: float | None,
        load_balance_error_ma: float | None,
        pair_balance_error_ma: float | None,
        final_position_ticks: dict[int, int | None],
        start_position_ticks: dict[int, int | None],
        missing_fields: list[str],
        clipped_move_count: int,
        correction_move_count: int,
        correction_travel_ticks: int,
        takeup_success: bool,
    ) -> tuple[dict[str, float], float]:
        max_travel_ticks = max(1.0, float(self.config.max_travel_ticks if self.config.max_travel_ticks is not None else 320))
        observed_travel = [
            abs(float(start_position_ticks[int(servo_id)]) - float(final_position_ticks[int(servo_id)]))
            for servo_id in final_position_ticks
            if start_position_ticks.get(int(servo_id)) is not None
            and final_position_ticks.get(int(servo_id)) is not None
        ]
        max_observed_travel = max(observed_travel) if observed_travel else None
        travel_score = (
            60.0
            if max_observed_travel is None
            else max(0.0, 100.0 - (60.0 * min(1.0, float(max_observed_travel) / float(max_travel_ticks))))
        )
        components = {
            "tip_centering": self._bounded_score(
                tip_xy_offset_mm,
                float(self.config.accept_max_final_tip_xy_offset_mm),
                missing_score=45.0,
            ),
            "settle_stability": self._bounded_score(
                settle_tip_drift_mm,
                max(0.5, float(self.config.tip_center_tolerance_mm) * 0.5),
                missing_score=65.0,
            ),
            "pair_balance": self._bounded_score(
                pair_balance_error_ma,
                float(self.config.accept_max_pair_balance_error_ma),
                missing_score=55.0,
            ),
            "load_balance": self._bounded_score(
                load_balance_error_ma,
                float(self.config.accept_max_load_balance_error_ma),
                missing_score=55.0,
            ),
            "travel_used": travel_score,
            "safety_guards": max(0.0, 100.0 - (35.0 * max(0, int(clipped_move_count)))),
            "telemetry_quality": max(0.0, 100.0 - (25.0 * len(set(missing_fields)))),
            "correction_required": max(
                0.0,
                100.0
                - min(80.0, (2.0 * max(0, int(correction_move_count))) + (0.25 * max(0, int(correction_travel_ticks)))),
            ),
            "takeup_success": 100.0 if takeup_success else 30.0,
        }
        weights = {
            "tip_centering": 0.30,
            "settle_stability": 0.12,
            "pair_balance": 0.14,
            "load_balance": 0.14,
            "travel_used": 0.08,
            "safety_guards": 0.08,
            "telemetry_quality": 0.08,
            "correction_required": 0.04,
            "takeup_success": 0.02,
        }
        score = sum(float(components[name]) * float(weight) for name, weight in weights.items())
        return {name: round(float(value), 3) for name, value in components.items()}, round(max(0.0, min(100.0, score)), 3)

    @staticmethod
    def _manual_startup_artifact_snapshot(servo_service, servo_ids: list[int]) -> dict[str, Any]:
        try:
            summary = servo_service.pretension_source_summary(list(servo_ids))
        except Exception as exc:
            return {"available": False, "error": str(exc)}
        return {
            "available": bool(summary.usable and summary.source_type == "manual"),
            "source_type": summary.source_type,
            "accepted": bool(summary.accepted),
            "usable": bool(summary.usable),
            "message": summary.message,
            "updated_at_utc": summary.updated_at_utc,
            "note": summary.note,
            "positions_by_servo": {str(k): v for k, v in dict(summary.positions_by_servo or {}).items()},
            "currents_ma_by_servo": {str(k): v for k, v in dict(summary.currents_by_servo or {}).items()},
        }

    @staticmethod
    def _save_advanced_startup_artifact(*, servo_service, servo_ids: list[int], run_row: dict[str, Any]) -> None:
        final_positions = dict(run_row.get("final_position_ticks_by_servo") or {})
        final_currents = dict(run_row.get("final_current_ma_by_servo") or {})
        thresholds: dict[int, int | None] = {}
        try:
            summary = servo_service.get_calibration_summary()
            for servo_id in servo_ids:
                entry = summary.servo_entries.get(int(servo_id)) if summary.exists else None
                thresholds[int(servo_id)] = (
                    int(entry.pretension_current_threshold_ma)
                    if entry is not None and entry.pretension_current_threshold_ma is not None
                    else None
                )
        except Exception:
            thresholds = {int(servo_id): None for servo_id in servo_ids}
        for servo_id in servo_ids:
            key = str(int(servo_id))
            servo_service.neutral_calibration.save_pretension_result(
                servo_id=int(servo_id),
                final_position_tick=final_positions.get(key),
                final_current_ma=final_currents.get(key),
                threshold_ma=thresholds.get(int(servo_id)),
                result_status="accepted",
                pretension_source="algorithmic",
                pretension_note="advanced_4servo_pretension",
                run_record={
                    "source": "algorithmic",
                    "mode": "advanced_4servo_pretension",
                    "run_index": int(run_row.get("run_index", 0)),
                    "quality_score_0_100": run_row.get("quality_score_0_100"),
                    "quality_components": dict(run_row.get("quality_components") or {}),
                    "final_tip_xyz_mm": run_row.get("final_tip_xyz_mm"),
                    "final_tip_xy_offset_mm": run_row.get("final_tip_xy_offset_mm"),
                    "final_position_ticks_by_servo": final_positions,
                    "final_current_ma_by_servo": final_currents,
                },
            )

    def _write_pretension_debug_export_bundle(self, *, session: ExperimentSession, paths, summary) -> None:
        metrics = getattr(summary, "experiment_metrics", {}) or {}
        if not isinstance(metrics, dict) or "pretension" not in str(metrics.get("algorithm", "")):
            return
        export_dir = Path(paths.output_dir) / "pretension_debug_bundle"
        export_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {
            "created_at_utc": _utc_now_iso(),
            "source_output_dir": str(paths.output_dir),
            "files": [],
            "note": "Upload this folder or zip when reviewing pretension hardware behavior.",
        }
        for filename in ("summary.json", "metrics.csv", "samples.jsonl", "pretension_summary.txt", "metadata.json", "config_snapshot.yaml"):
            source = Path(paths.output_dir) / filename
            if not source.exists():
                continue
            destination = export_dir / filename
            shutil.copy2(source, destination)
            manifest["files"].append(str(destination.name))
        for plot_path in sorted(Path(paths.output_dir).glob("*.png")):
            destination = export_dir / plot_path.name
            shutil.copy2(plot_path, destination)
            manifest["files"].append(str(destination.name))
        diagnostics = {
            "experiment_name": getattr(session.metadata, "experiment_name", "pretension_validation"),
            "status": getattr(summary, "status", None),
            "message": getattr(summary, "message", None),
            "algorithm": metrics.get("algorithm"),
            "algorithm_mode": metrics.get("algorithm_mode"),
            "failure_reason_counts": metrics.get("failure_reason_counts"),
            "accepted_run_count": metrics.get("accepted_run_count"),
            "run_count": metrics.get("run_count"),
            "quality_scores_0_100": metrics.get("quality_scores_0_100"),
        }
        (export_dir / "debug_manifest.json").write_text(
            json.dumps({"manifest": manifest, "diagnostics": diagnostics}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (export_dir / "session_log_excerpt.txt").write_text(
            "\n".join(
                [
                    "Pretension debug bundle created by pretension_validation.",
                    f"Output directory: {paths.output_dir}",
                    f"Algorithm: {metrics.get('algorithm')}",
                    f"Mode: {metrics.get('algorithm_mode')}",
                    f"Failures: {metrics.get('failure_reason_counts')}",
                    "Attach this bundle plus the GUI/operator log if available.",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _add_staged_sample(
        session: ExperimentSession,
        *,
        phase: str,
        run_index: int,
        step_index: int,
        payload: dict[str, Any],
    ) -> None:
        servo_id = payload.get("servo_id")
        commanded_motor_values = {}
        if servo_id not in (None, ""):
            commanded_motor_values["servo_id"] = int(servo_id)
        if payload.get("final_position_tick") is not None:
            commanded_motor_values["commanded_position_ticks"] = int(payload["final_position_tick"])
        session.add_sample(
            ExperimentTimeseriesSample(
                monotonic_time_s=session.elapsed_s(),
                wall_time_utc=_utc_now_iso(),
                phase=str(phase),
                step_index=int(step_index),
                sample_index=int(step_index),
                commanded_motor_values=commanded_motor_values,
                status_flags=["pretension_validation", "single_segment_staged"],
                extra={"run_index": int(run_index), **dict(payload)},
            )
        )

    def _pretension_parameters(self, session: ExperimentSession) -> PretensionParameters:
        defaults = session.context.servo_service.default_pretension_parameters(int(self.config.servo_id))
        return PretensionParameters(
            untensioned_reference_tick=(
                int(self.config.untensioned_reference_tick)
                if self.config.untensioned_reference_tick is not None
                else int(defaults.untensioned_reference_tick)
            ),
            start_mode=(
                str(self.config.pretension_start_mode).strip().lower()
                if self.config.pretension_start_mode not in (None, "")
                else str(defaults.start_mode)
            ),
            step_ticks=int(self.config.step_ticks) if self.config.step_ticks is not None else int(defaults.step_ticks),
            settle_time_s=(
                float(self.config.settle_time_s)
                if self.config.settle_time_s is not None
                else float(defaults.settle_time_s)
            ),
            baseline_sample_count=(
                int(self.config.baseline_sample_count)
                if self.config.baseline_sample_count is not None
                else int(defaults.baseline_sample_count)
            ),
            current_filter_window=(
                int(self.config.current_filter_window)
                if self.config.current_filter_window is not None
                else int(defaults.current_filter_window)
            ),
            current_delta_threshold_ma=(
                int(self.config.current_delta_threshold_ma)
                if self.config.current_delta_threshold_ma is not None
                else int(defaults.current_delta_threshold_ma)
            ),
            absolute_trigger_current_ma=(
                int(self.config.absolute_trigger_current_ma)
                if self.config.absolute_trigger_current_ma is not None
                else defaults.absolute_trigger_current_ma
            ),
            hard_current_stop_ma=(
                int(self.config.hard_current_stop_ma)
                if self.config.hard_current_stop_ma is not None
                else int(defaults.hard_current_stop_ma)
            ),
            max_travel_ticks=(
                int(self.config.max_travel_ticks)
                if self.config.max_travel_ticks is not None
                else int(defaults.max_travel_ticks)
            ),
            timeout_s=float(self.config.timeout_s) if self.config.timeout_s is not None else float(defaults.timeout_s),
        )

    @staticmethod
    def _phase_for_result_status(status: str) -> str:
        normalized = str(status or "").strip().lower()
        if normalized == "baseline_ready":
            return "pretension_baseline"
        if normalized == "running":
            return "pretension_step"
        return "pretension_result"

    @staticmethod
    def _build_trace_sample(
        session: ExperimentSession,
        *,
        snapshot,
        phase: str,
        step_index: int,
        sample_index: int,
        payload: dict[str, Any],
    ) -> ExperimentTimeseriesSample:
        commanded_position = payload.get("commanded_position_ticks")
        commanded_motor_values = {
            "servo_id": int(payload["servo_id"]),
            "commanded_position_ticks": commanded_position,
        }
        if snapshot is None:
            return ExperimentTimeseriesSample(
                monotonic_time_s=session.elapsed_s(),
                wall_time_utc=_utc_now_iso(),
                phase=phase,
                step_index=int(step_index),
                sample_index=int(sample_index),
                commanded_motor_values=commanded_motor_values,
                status_flags=["pretension_validation"],
                extra=dict(payload),
            )
        return _sample_from_tracking_snapshot(
            session,
            snapshot=snapshot,
            phase=phase,
            step_index=int(step_index),
            sample_index=int(sample_index),
            commanded_cable_deltas_cm=[],
            commanded_motor_values=commanded_motor_values,
            status_flags=["pretension_validation"],
            extra=dict(payload),
        )

    @staticmethod
    def _trace_payload(
        *,
        servo_id: int,
        run_state: str,
        commanded_position_ticks: int | None,
        current_position_ticks: int | None,
        raw_current_ma: int | None,
        filtered_current_ma: float | None,
        baseline_current_ma: float | None,
        effective_trigger_current_ma: int | None,
        hard_current_stop_ma: int | None,
        untensioned_reference_tick: int | None,
        trigger_met: bool,
        stop_reason: str | None,
        tracker_metric,
        mapper,
    ) -> dict[str, Any]:
        travel_ticks = (
            None
            if untensioned_reference_tick is None or current_position_ticks is None
            else int(untensioned_reference_tick) - int(current_position_ticks)
        )
        travel_mm = (
            float(mapper.ticks_to_displacement_mm(travel_ticks))
            if travel_ticks is not None and mapper is not None
            else None
        )
        return {
            "servo_id": int(servo_id),
            "run_state": str(run_state),
            "commanded_position_ticks": commanded_position_ticks,
            "current_position_ticks": current_position_ticks,
            "travel_from_untensioned_ticks": travel_ticks,
            "travel_from_untensioned_mm": travel_mm,
            "raw_current_ma": (int(raw_current_ma) if raw_current_ma is not None else None),
            "filtered_current_ma": (float(filtered_current_ma) if filtered_current_ma is not None else None),
            "baseline_current_ma": (float(baseline_current_ma) if baseline_current_ma is not None else None),
            "effective_trigger_current_ma": (
                int(effective_trigger_current_ma) if effective_trigger_current_ma is not None else None
            ),
            "hard_current_stop_ma": (int(hard_current_stop_ma) if hard_current_stop_ma is not None else None),
            "trigger_met": bool(trigger_met),
            "stop_reason": stop_reason,
            "tracker_displacement_mm": (
                float(tracker_metric.displacement_magnitude_mm)
                if tracker_metric is not None and tracker_metric.displacement_magnitude_mm is not None
                else None
            ),
            "tracker_displacement_vector_mm": (
                list(tracker_metric.displacement_vector_mm)
                if tracker_metric is not None and tracker_metric.displacement_vector_mm is not None
                else None
            ),
            "tracker_metric_frame": (
                str(tracker_metric.metric_frame)
                if tracker_metric is not None and tracker_metric.metric_frame not in (None, "")
                else None
            ),
            "tracker_metric_message": (
                str(tracker_metric.message) if tracker_metric is not None and tracker_metric.message else None
            ),
        }


def _collect_pose_output_root(session: ExperimentSession) -> Path:
    run_dir = getattr(session.context, "run_output_dir", None)
    if run_dir is not None:
        return Path(run_dir)
    return Path(session.context.output_root)


def _append_collect_pose_jsonl_event(output_root: Path, filename: str, event: dict[str, Any]) -> None:
    path = Path(output_root) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n")


def _write_collect_pose_checkpoint(
    *,
    output_root: Path,
    last_completed_command_index: int | None,
    accepted_sample_count: int,
    dropped_post_motion_telemetry_samples: int,
    next_command_index_to_resume: int,
    total_packet_failure_events: int,
) -> None:
    payload = {
        "schema_version": "collect_pose_checkpoint_v1",
        "last_completed_command_index": last_completed_command_index,
        "accepted_sample_count": int(accepted_sample_count),
        "dropped_post_motion_telemetry_samples": int(dropped_post_motion_telemetry_samples),
        "next_command_index_to_resume": int(next_command_index_to_resume),
        "total_packet_failure_events": int(total_packet_failure_events),
    }
    (Path(output_root) / "collect_pose_checkpoint.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_collect_pose_long_run_health(
    *,
    output_root: Path,
    session: ExperimentSession,
    metrics: dict[str, Any],
    last_command_step_index: int | None,
    estimated_total_commands: int,
    samples: list[ExperimentTimeseriesSample] | None = None,
) -> None:
    accepted = int(metrics.get("accepted_sample_count", 0) or 0)
    complete_training = int(metrics.get("complete_training_row_count", 0) or 0)
    target_complete_training = int(metrics.get("target_valid_sample_count", 0) or 0)
    remaining_complete_training = max(0, target_complete_training - complete_training)
    dropped = int(metrics.get("dropped_post_motion_telemetry_samples", 0) or 0)
    dropped_pre_motion = int(metrics.get("dropped_pre_motion_telemetry_samples", 0) or 0)
    rate = metrics.get("mean_sample_hz_estimate")
    remaining_cmds = max(0, int(estimated_total_commands) - int(last_command_step_index or -1) - 1)
    sample_list = list(samples or [])
    currents = _collect_pose_current_values(sample_list) if sample_list else []
    max_abs_current = max((abs(int(value)) for value in currents), default=None)
    tracker_stale = (
        sum(
            1
            for sample in sample_list
            if sample.extra.get("capture_accepted") is False
            and (
                "stale" in str(sample.extra.get("capture_rejection_reason", "")).lower()
                or "tracker_age" in str(sample.extra.get("capture_rejection_reason", "")).lower()
            )
        )
        if sample_list
        else 0
    )
    payload = {
        "schema_version": "collect_pose_long_run_health_v1",
        "wall_time_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_session_s": float(session.elapsed_s()),
        "samples_in_session": int(len(session.samples)),
        "accepted_sample_count": accepted,
        "complete_training_row_count": int(complete_training),
        "target_valid_sample_count": int(target_complete_training),
        "remaining_complete_training_rows": int(remaining_complete_training),
        "dropped_post_motion_telemetry_samples": dropped,
        "dropped_pre_motion_telemetry_samples": dropped_pre_motion,
        "unrecovered_packet_error_count": int(metrics.get("unrecovered_packet_error_count", 0) or 0),
        "recovered_packet_error_count": int(metrics.get("recovered_packet_error_count", 0) or 0),
        "servo_telemetry_retry_count": int(metrics.get("servo_telemetry_retry_count", 0) or 0),
        "transport_burst_count": int(metrics.get("transport_burst_count", 0) or 0),
        "consecutive_transport_burst_failures": int(metrics.get("consecutive_transport_burst_failures", 0) or 0),
        "total_post_motion_packet_failure_events": int(metrics.get("total_post_motion_packet_failure_events", 0) or 0),
        "write_goal_packet_error_count": int(metrics.get("write_goal_packet_error_count", 0) or 0),
        "rejected_sample_count": int(metrics.get("rejected_sample_count", 0) or 0),
        "tracker_stale_count": int(tracker_stale),
        "max_abs_current_ma": max_abs_current,
        "last_command_step_index": last_command_step_index,
        "next_command_index_to_resume": int(metrics.get("next_command_index_to_resume", 0) or 0),
        "mean_sample_hz_estimate": rate,
        "estimated_commands_remaining": remaining_cmds,
        "run_status": str(metrics.get("run_status", "running")),
        "run_success": bool(metrics.get("run_success", False)),
        "recommendation": str(metrics.get("long_run_health_recommendation", "running")),
    }
    (Path(output_root) / "long_run_health.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _collect_pose_resync_telemetry(
    session: ExperimentSession,
    servo_ids: list[int],
    *,
    attempts: int,
    delay_s: float,
) -> dict[int, Any]:
    last_exc: BaseException | None = None
    for attempt in range(max(1, int(attempts))):
        if attempt > 0 and float(delay_s) > 0.0:
            session.context.sleep_fn(float(delay_s))
        try:
            return session.context.servo_service.read_telemetry([int(s) for s in servo_ids])
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"collect-pose resync telemetry failed after {attempts} attempt(s)") from last_exc


def _collect_pose_parallel_command_metadata(
    session: ExperimentSession,
    *,
    servo_ids: list[int],
    cable_command_cm: list[float],
) -> dict[str, Any]:
    parallel_command_metadata: dict[str, Any] = {}
    if _collect_pose_parallel_single_mode(session) and len(cable_command_cm) == 4:
        context = session.context.settings.robot.operating_context()
        mirror_pairs = {int(k): int(v) for k, v in dict(context.mirror_pairs or {}).items()}
        segments = dict(context.segments or {})
        segment_order = list(context.segment_order or [])
        segment_a_key = segment_order[0] if len(segment_order) >= 1 else "segment_a"
        segment_b_key = segment_order[1] if len(segment_order) >= 2 else "segment_b"
        segment_a = segments.get(segment_a_key)
        segment_b = segments.get(segment_b_key)
        shared = [float(value) for value in cable_command_cm]
        parallel_command_metadata = {
            "operating_mode": "parallel_single",
            "mirrored_parallel": True,
            "parallel_single_demo": True,
            "true_two_segment_control": False,
            "mirror_pairs": {str(k): int(v) for k, v in sorted(mirror_pairs.items())},
            "commanded_servo_ids": list(servo_ids),
            "shared_4_tendon_command_cm": list(shared),
            "segment_a_command_cm": list(shared),
            "segment_b_command_cm": list(shared),
            "segment_order": segment_order,
            "segment_a": {
                "key": str(segment_a.key if segment_a is not None else segment_a_key),
                "label": str(segment_a.label if segment_a is not None else "Segment A"),
                "segment_label": str(
                    (segment_a.segment_label if segment_a is not None else "")
                    or (segment_a.label if segment_a is not None else "Segment A")
                ),
                "segment_role": str(segment_a.segment_role if segment_a is not None else "proximal"),
                "servo_ids": [int(value) for value in (segment_a.servo_ids if segment_a is not None else [1, 2, 3, 4])],
                "pairs": {
                    str(key): [int(value) for value in values]
                    for key, values in dict(segment_a.pairs if segment_a is not None else {}).items()
                },
            },
            "segment_b": {
                "key": str(segment_b.key if segment_b is not None else segment_b_key),
                "label": str(segment_b.label if segment_b is not None else "Segment B"),
                "segment_label": str(
                    (segment_b.segment_label if segment_b is not None else "")
                    or (segment_b.label if segment_b is not None else "Segment B")
                ),
                "segment_role": str(segment_b.segment_role if segment_b is not None else "distal"),
                "servo_ids": [int(value) for value in (segment_b.servo_ids if segment_b is not None else [5, 6, 7, 8])],
                "pairs": {
                    str(key): [int(value) for value in values]
                    for key, values in dict(segment_b.pairs if segment_b is not None else {}).items()
                },
            },
        }
    return parallel_command_metadata


def _synthetic_collect_pose_command_result(
    *,
    exc: ServoTelemetryRetryError,
    step: ModelingCommandStep | None,
    parallel_command_metadata: dict[str, Any],
    resync_feedback: dict[str, Any],
    fallback_cable_command_cm: list[float] | None = None,
) -> dict[str, Any]:
    ctx = dict(exc.context or {})
    goals_raw = ctx.get("last_commanded_goal_ticks") or {}
    commanded = {str(k): int(v) for k, v in goals_raw.items()}
    if step is not None:
        requested_cm = list(step.cable_command_cm)
        requested_pair = list(step.pair_command_cm)
    else:
        requested_cm = list(fallback_cable_command_cm or [0.0, 0.0, 0.0, 0.0])
        requested_pair = _pair_command_from_cable_deltas(requested_cm)
    resolved_cm = list(ctx.get("last_resolved_cable_command_cm") or requested_cm)
    pair_cm = _pair_command_from_cable_deltas(resolved_cm)
    retry_count = int(ctx.get("retry_count", 0) or 0)
    return {
        "requested_cable_command_cm": list(requested_cm),
        "resolved_cable_command_cm": list(resolved_cm),
        "requested_pair_command_cm": list(requested_pair),
        "resolved_pair_command_cm": list(pair_cm),
        "commanded_motor_values": dict(commanded),
        "raw_goal_ticks_by_servo": dict(commanded),
        "final_goal_ticks_by_servo": dict(commanded),
        "clamp_reasons_by_servo": {},
        "servo_debug": {},
        "servo_feedback": dict(resync_feedback),
        "command_metadata": {
            **dict(parallel_command_metadata),
            "post_motion_telemetry_unrecovered": True,
            "synthetic_command_result_after_packet_error": True,
            "original_failure_category": ctx.get("failure_category"),
            "original_failure_reason": ctx.get("failure_reason"),
            "telemetry_retry_count": retry_count,
            "unrecovered_packet_error_count": 1,
        },
        "message": "Synthetic command summary after unrecovered post-motion telemetry; goals assumed from last commanded ticks.",
        "motion_profile": {
            "operating_mode_label": "unknown_after_packet_error",
            "operating_mode": None,
            "goal_current_ma": None,
            "profile_velocity": None,
            "profile_acceleration": None,
        },
        "post_motion_telemetry_unrecovered": True,
    }


class CollectPoseCommandDatasetExperiment(BaseExperiment):
    """Collect thesis-grade single-segment modeling datasets."""

    name = "collect_pose_command_dataset"
    description = "Canonical Motor Babble modeling dataset collector for the current single-segment robot."
    hardware_requirements = ExperimentHardwareRequirements(tracking_required=True, mock_compatible=True)

    def __init__(self, config: CollectPoseCommandDatasetConfig) -> None:
        super().__init__(config=config)
        self._tracking_started_here = False
        self._initial_neutral_ticks: list[int] = []
        self._servo_ids: list[int] = []

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "CollectPoseCommandDatasetExperiment":
        return cls(config=CollectPoseCommandDatasetConfig.from_dict(payload))

    def setup(self, session: ExperimentSession) -> None:
        # Mutual-exclusion guard for the two per-command sample knobs. Both
        # control how many samples a single command produces, but with different
        # semantics: samples_per_command runs the gate N times (legacy),
        # tracker_samples_per_command runs the gate once for the first frame
        # then collects N-1 more deduped fresh frames for averaging. Letting
        # both be > 1 would multiply samples in a way the operator probably
        # didn't mean.
        if int(self.config.samples_per_command) > 1 and int(self.config.tracker_samples_per_command) > 1:
            raise RuntimeError(
                "collect_pose_command_dataset: samples_per_command and "
                "tracker_samples_per_command are mutually exclusive when "
                "both > 1. Set one of them to 1. "
                f"(got samples_per_command={int(self.config.samples_per_command)}, "
                f"tracker_samples_per_command={int(self.config.tracker_samples_per_command)})"
            )
        tracking_service = session.context.tracking_service
        if tracking_service is not None and getattr(tracking_service, "_thread", None) is None:
            tracking_service.start()
            self._tracking_started_here = True
        self._servo_ids = _configured_collect_pose_servo_ids(session)
        if self.config.dry_run:
            self._initial_neutral_ticks = [0 for _ in self._servo_ids]
        else:
            self._initial_neutral_ticks = _load_collect_pose_neutral_ticks(session, servo_ids=self._servo_ids)
        # Side accumulators for the multi-frame averaging path. When
        # tracker_samples_per_command > 1, session.samples receives the first-
        # frame row (preserves today's training pipeline) and these lists
        # collect the parallel averaged-row stream + per-frame raw stream.
        # Empty unless the averaging path runs.
        self._averaged_dataset_samples: list[ExperimentTimeseriesSample] = []
        self._raw_tracker_frame_rows: list[dict[str, Any]] = []
        # Resolve averaged_label_enabled with the "None means auto" rule.
        if self.config.averaged_label_enabled is None:
            self._averaged_label_enabled = int(self.config.tracker_samples_per_command) > 1
        else:
            self._averaged_label_enabled = bool(self.config.averaged_label_enabled)
        if self.config.export_averaged_sample_label is None:
            self._export_averaged_sample_label = self._averaged_label_enabled
        else:
            self._export_averaged_sample_label = bool(self.config.export_averaged_sample_label)

    def precheck(self, session: ExperimentSession) -> None:
        _precheck_collect_pose_command_dataset(
            session=session,
            config=self.config,
            servo_ids=list(self._servo_ids),
            neutral_ticks=list(self._initial_neutral_ticks),
        )

    def _should_recover_collect_pose_post_motion_packet_error(self, exc: BaseException) -> bool:
        if not self._transport_burst_recovery_enabled():
            return False
        if str(self.config.on_unrecovered_post_motion_telemetry or "").strip().lower() != "drop_sample_and_resync":
            return False
        retry_error = _find_servo_telemetry_retry_error(exc)
        if retry_error is None:
            return False
        return str((retry_error.context or {}).get("failure_category") or "") == "servo_telemetry_packet_error"

    @staticmethod
    def _collect_pose_failure_ctx(exc: BaseException) -> tuple[ServoTelemetryRetryError | None, dict[str, Any]]:
        retry_error = _find_servo_telemetry_retry_error(exc)
        return retry_error, dict((retry_error.context or {}) if retry_error is not None else {})

    def _transport_burst_recovery_enabled(self) -> bool:
        return bool(self.config.transport_burst_recovery_enabled or self.config.long_run_recovery_enabled)

    def _should_recover_collect_pose_pre_motion_packet_error(self, exc: BaseException) -> bool:
        if not self._transport_burst_recovery_enabled():
            return False
        if str(self.config.on_unrecovered_post_motion_telemetry or "").strip().lower() != "drop_sample_and_resync":
            return False
        retry_error, ctx = self._collect_pose_failure_ctx(exc)
        if retry_error is None:
            return False
        if str(ctx.get("failure_category") or "") != "simple_experiment_motion_rejected":
            return False
        command_metadata = dict(ctx.get("command_metadata", {}) or {})
        pre_motion_profile = str(command_metadata.get("pre_motion_telemetry_profile", "") or "")
        if pre_motion_profile.strip().lower() != "minimal":
            return False
        pre_motion_source = str(command_metadata.get("pre_motion_read_source", "") or "").strip().lower()
        if pre_motion_source not in {
            "experiment_owned_minimal_read",
            "experiment_owned_minimal_read_after_configuration",
            "prevalidated_experiment_owned_health_read",
        }:
            return False
        err = str(ctx.get("telemetry_error_code") or "").lower()
        if "packet" in err or "status" in err:
            return True
        failed_servo = ctx.get("failed_servo_id")
        missing_fields = dict(ctx.get("missing_fields", {}) or {})
        if failed_servo is not None:
            missing_for_failed = list(missing_fields.get(str(int(failed_servo)), []) or [])
            if "present_position" in missing_for_failed:
                return True
        for fields in missing_fields.values():
            if "present_position" in list(fields or []):
                return True
        telemetry_by_servo = dict(ctx.get("last_valid_telemetry_by_servo", {}) or {})
        if failed_servo is not None:
            tel_row = dict(telemetry_by_servo.get(str(int(failed_servo)), {}) or {})
            tel_err = str(tel_row.get("telemetry_error_code") or "").lower()
            if "packet" in tel_err or "status" in tel_err:
                return True
        return False

    @staticmethod
    def _collect_pose_tracker_age_s(session: ExperimentSession) -> float | None:
        if session.context.tracking_service is None:
            return None
        try:
            reader = getattr(session.context.tracking_service, "peek_snapshot", None)
            snapshot = reader() if callable(reader) else session.context.tracking_service.get_snapshot()
            value = getattr(snapshot, "tracker_data_age_s", None)
        except Exception:
            value = None
        return None if value is None else float(value)

    def _append_collect_pose_failure_event(
        self,
        session: ExperimentSession,
        *,
        event: str,
        payload: dict[str, Any],
    ) -> None:
        _append_collect_pose_jsonl_event(
            _collect_pose_output_root(session),
            "sample_failure_events.jsonl",
            {"event": str(event), **dict(payload)},
        )

    def _write_collect_pose_transport_recovery_report(
        self,
        session: ExperimentSession,
        *,
        stop_reason: str,
        step_index: int,
        command_vector_cm: list[float],
        failed_servo_ids: list[int],
        cooldown_s: float,
        resync_attempts: int,
    ) -> None:
        output_root = _collect_pose_output_root(session)
        events_path = output_root / "sample_failure_events.jsonl"
        lines: list[str] = []
        try:
            if events_path.exists():
                lines = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except Exception:
            lines = []
        recent_events: list[dict[str, Any]] = []
        for line in lines[-15:]:
            try:
                recent_events.append(json.loads(line))
            except Exception:
                continue
        passive_diag: dict[str, Any] = {"attempted": False}
        if not self.config.dry_run and session.context.servo_service.is_connected:
            passive_diag["attempted"] = True
            try:
                read = session.context.servo_service.read_telemetry([int(value) for value in self._servo_ids])
                passive_diag["success"] = True
                passive_diag["servo_feedback"] = _servo_feedback_payload(read, servo_service=session.context.servo_service)
            except Exception as exc:
                passive_diag["success"] = False
                passive_diag["error"] = str(exc)
        payload = {
            "schema_version": "collect_pose_transport_recovery_report_v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "stop_reason": str(stop_reason),
            "step_index": int(step_index),
            "command_vector_cm": [float(value) for value in command_vector_cm],
            "failed_servo_ids": [int(value) for value in failed_servo_ids],
            "cooldown_s": float(cooldown_s),
            "resync_attempts_per_cycle": int(resync_attempts),
            "consecutive_transport_burst_failures": int(
                session.metrics.get("consecutive_transport_burst_failures", 0) or 0
            ),
            "transport_burst_count": int(session.metrics.get("transport_burst_count", 0) or 0),
            "total_post_motion_packet_failure_events": int(
                session.metrics.get("total_post_motion_packet_failure_events", 0) or 0
            ),
            "unrecovered_packet_error_count": int(session.metrics.get("unrecovered_packet_error_count", 0) or 0),
            "tracker_age_s": self._collect_pose_tracker_age_s(session),
            "recent_failure_events": recent_events,
            "next_action": "run servo_transport_diagnostic and inspect hardware status/power before resume",
            "passive_transport_diagnostic": passive_diag,
        }
        (output_root / "transport_recovery_report.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _stop_collect_pose_for_transport_budget(
        self,
        *,
        session: ExperimentSession,
        exc: Exception,
        requested_cable_command_cm: list[float],
        servo_ids: list[int],
        step_index_for_event: int,
        failed_servo_ids: list[int],
        stop_reason: str,
        cooldown_s: float,
        resync_attempts: int,
        skip_packet_error_metric_bump: bool = True,
    ) -> None:
        self._append_collect_pose_failure_event(
            session,
            event="run_stop_budget_exceeded",
            payload={
                "step_index": int(step_index_for_event),
                "command_vector_cm": list(requested_cable_command_cm),
                "failed_servo_ids": [int(value) for value in failed_servo_ids],
                "stop_reason": str(stop_reason),
                "consecutive_transport_burst_failures": int(
                    session.metrics.get("consecutive_transport_burst_failures", 0) or 0
                ),
                "total_post_motion_packet_failure_events": int(
                    session.metrics.get("total_post_motion_packet_failure_events", 0) or 0
                ),
                "dropped_post_motion_telemetry_samples": int(
                    session.metrics.get("dropped_post_motion_telemetry_samples", 0) or 0
                ),
                "unrecovered_packet_error_count": int(session.metrics.get("unrecovered_packet_error_count", 0) or 0),
                "tracker_age_s": self._collect_pose_tracker_age_s(session),
            },
        )
        self._write_collect_pose_transport_recovery_report(
            session,
            stop_reason=str(stop_reason),
            step_index=int(step_index_for_event),
            command_vector_cm=list(requested_cable_command_cm),
            failed_servo_ids=[int(value) for value in failed_servo_ids],
            cooldown_s=float(cooldown_s),
            resync_attempts=int(resync_attempts),
        )
        self._record_failure_context(
            session=session,
            exc=exc,
            requested_cable_command_cm=list(requested_cable_command_cm),
            servo_ids=list(servo_ids),
            skip_packet_error_metric_bump=bool(skip_packet_error_metric_bump),
        )
        raise RuntimeError(str(stop_reason)) from exc

    def _collect_pose_transition_waypoints(
        self,
        session: ExperimentSession,
        *,
        requested: list[float],
    ) -> list[list[float]]:
        target = [float(value) for value in requested]
        if not bool(self.config.command_transition_ramp_enabled):
            return [target]
        max_delta = (
            self.config.max_delta_cm_per_ramp_step
            if self.config.max_delta_cm_per_ramp_step is not None
            else self.config.max_delta_cm_per_transition
        )
        if max_delta is None or float(max_delta) <= 0.0:
            return [target]
        previous_raw = session.metrics.get("last_dispatched_cable_command_cm")
        previous = [float(value) for value in previous_raw] if isinstance(previous_raw, list) else []
        if len(previous) != len(target):
            return [target]
        max_component_delta = max(abs(float(a) - float(b)) for a, b in zip(target, previous))
        steps = max(1, int(math.ceil(max_component_delta / float(max_delta))))
        if steps <= 1:
            return [target]
        waypoints: list[list[float]] = []
        for index in range(1, steps + 1):
            alpha = float(index) / float(steps)
            waypoints.append(
                [float(start + (goal - start) * alpha) for start, goal in zip(previous, target)]
            )
        return waypoints

    @staticmethod
    def _metric_int_map(metrics: dict[str, Any], key: str) -> dict[str, int]:
        raw = dict(metrics.get(key, {}) or {})
        return {str(k): int(v) for k, v in raw.items()}

    @staticmethod
    def _set_metric_int_map(session: ExperimentSession, key: str, payload: dict[str, int]) -> None:
        session.set_metric(key, {str(k): int(v) for k, v in sorted(payload.items())})

    @staticmethod
    def _parse_post_move_overcurrent(exc: BaseException) -> tuple[int | None, int | None, int | None]:
        text = str(exc)
        if "post-move overcurrent/jam protection:" not in text:
            return None, None, None
        sid_match = re.search(r"servo\s+(\d+)", text, flags=re.IGNORECASE)
        abs_match = re.search(r"\|(-?\d+)\|=(\d+)\s*mA\s*>\s*(\d+)\s*mA", text)
        servo_id = int(sid_match.group(1)) if sid_match else None
        measured_abs = int(abs_match.group(2)) if abs_match else None
        threshold = int(abs_match.group(3)) if abs_match else None
        return servo_id, measured_abs, threshold

    def _collect_pose_bump_packet_failure_metrics(
        self,
        session: ExperimentSession,
        *,
        dropped_count: int,
        pre_motion: bool = False,
    ) -> tuple[int, int]:
        session.set_metric(
            "unrecovered_packet_error_count",
            int(session.metrics.get("unrecovered_packet_error_count", 0) or 0) + 1,
        )
        session.set_metric(
            "consecutive_post_motion_packet_failures",
            int(session.metrics.get("consecutive_post_motion_packet_failures", 0) or 0) + 1,
        )
        session.set_metric(
            "total_post_motion_packet_failure_events",
            int(session.metrics.get("total_post_motion_packet_failure_events", 0) or 0) + 1,
        )
        dropped_key = "dropped_pre_motion_telemetry_samples" if pre_motion else "dropped_post_motion_telemetry_samples"
        session.set_metric(
            dropped_key,
            int(session.metrics.get(dropped_key, 0) or 0) + int(max(1, dropped_count)),
        )
        return (
            int(session.metrics.get("consecutive_post_motion_packet_failures", 0) or 0),
            int(session.metrics.get("total_post_motion_packet_failure_events", 0) or 0),
        )

    def _recover_collect_pose_post_motion_packet_error(
        self,
        session: ExperimentSession,
        exc: ServoTelemetryRetryError,
        *,
        step: ModelingCommandStep | None,
        servo_ids: list[int],
        parallel_command_metadata: dict[str, Any],
        zero_vector: list[float] | None,
        step_index_for_event: int,
    ) -> dict[str, Any]:
        ctx = dict(exc.context or {})
        retry_count = int(ctx.get("retry_count", 0) or 0)
        session.set_metric(
            "servo_telemetry_retry_count",
            int(session.metrics.get("servo_telemetry_retry_count", 0) or 0) + retry_count,
        )
        consecutive_packet_failures, total_ev = self._collect_pose_bump_packet_failure_metrics(
            session,
            dropped_count=max(1, int(self.config.samples_per_command)),
        )
        tracker_age_s = self._collect_pose_tracker_age_s(session)
        command_vector_cm = list(ctx.get("last_resolved_cable_command_cm") or [])
        failed_servo_ids_raw = ctx.get("failed_servo_ids")
        if isinstance(failed_servo_ids_raw, list) and failed_servo_ids_raw:
            failed_servo_ids = [int(value) for value in failed_servo_ids_raw]
        else:
            failed_servo = ctx.get("failed_servo_id")
            failed_servo_ids = [int(failed_servo)] if failed_servo is not None else []
        session.set_metric("transport_burst_count", int(session.metrics.get("transport_burst_count", 0) or 0) + 1)
        transport_burst_count = int(session.metrics.get("transport_burst_count", 0) or 0)
        self._append_collect_pose_failure_event(
            session,
            event="post_motion_telemetry_packet_error",
            payload={
                "step_index": int(step_index_for_event),
                "command_vector_cm": list(command_vector_cm),
                "failed_servo_id": ctx.get("failed_servo_id"),
                "failed_servo_ids": [int(value) for value in failed_servo_ids],
                "failure_reason": ctx.get("failure_reason"),
                "failure_category": ctx.get("failure_category"),
                "retry_count": retry_count,
                "telemetry_retry_budget": int(self.config.telemetry_retry_count),
                "telemetry_error_code": ctx.get("telemetry_error_code"),
                "missing_fields": ctx.get("missing_fields"),
                "last_valid_telemetry_by_servo": ctx.get("last_valid_telemetry_by_servo"),
                "last_commanded_goal_ticks": ctx.get("last_commanded_goal_ticks"),
                "last_resolved_cable_command_cm": list(command_vector_cm),
                "tracker_age_s": tracker_age_s,
                "consecutive_post_motion_packet_failures": int(consecutive_packet_failures),
                "total_post_motion_packet_failure_events": int(total_ev),
                "transport_burst_count": int(transport_burst_count),
            },
        )
        self._append_collect_pose_failure_event(
            session,
            event="sample_quarantined",
            payload={
                "step_index": int(step_index_for_event),
                "command_vector_cm": list(command_vector_cm),
                "reason": "post_motion_telemetry_packet_error",
                "dropped_post_motion_telemetry_samples": int(
                    session.metrics.get("dropped_post_motion_telemetry_samples", 0) or 0
                ),
                "tracker_age_s": tracker_age_s,
            },
        )
        self._append_collect_pose_failure_event(
            session,
            event="transport_burst_recovery_started",
            payload={
                "step_index": int(step_index_for_event),
                "command_vector_cm": list(command_vector_cm),
                "failed_servo_ids": [int(value) for value in failed_servo_ids],
                "cooldown_s": float(self.config.transport_burst_cooldown_s),
                "resync_attempts": int(self.config.transport_burst_resync_attempts),
                "resync_delay_s": float(self.config.transport_burst_resync_delay_s),
                "retry_same_command_after_resync": bool(self.config.retry_same_command_after_resync),
                "consecutive_post_motion_packet_failures": int(consecutive_packet_failures),
                "total_post_motion_packet_failure_events": int(total_ev),
                "transport_burst_count": int(transport_burst_count),
                "tracker_age_s": tracker_age_s,
            },
        )
        immediate_stop_reasons: list[str] = []
        max_total = self.config.max_total_packet_failures
        if max_total is not None and int(total_ev) > int(max_total):
            immediate_stop_reasons.append(
                f"collect_pose_command_dataset: exceeded max_total_packet_failures={int(max_total)}."
            )
        max_drop_fraction = self.config.max_total_dropped_samples_fraction
        if max_drop_fraction is not None:
            accepted = int(session.metrics.get("accepted_sample_count", 0) or 0)
            rejected = int(session.metrics.get("rejected_sample_count", 0) or 0)
            dropped = int(session.metrics.get("dropped_post_motion_telemetry_samples", 0) or 0) + int(
                session.metrics.get("dropped_pre_motion_telemetry_samples", 0) or 0
            )
            denominator = max(1, accepted + rejected)
            if float(dropped) / float(denominator) > float(max_drop_fraction):
                immediate_stop_reasons.append(
                    "collect_pose_command_dataset: exceeded max_total_dropped_samples_fraction="
                    f"{float(max_drop_fraction):.4f}."
                )
        if immediate_stop_reasons:
            self._stop_collect_pose_for_transport_budget(
                session=session,
                exc=exc,
                requested_cable_command_cm=list(command_vector_cm),
                servo_ids=list(servo_ids),
                step_index_for_event=int(step_index_for_event),
                failed_servo_ids=list(failed_servo_ids),
                stop_reason=" ".join(immediate_stop_reasons),
                cooldown_s=float(self.config.transport_burst_cooldown_s),
                resync_attempts=int(self.config.transport_burst_resync_attempts),
            )
        cooldown_s = float(self.config.transport_burst_cooldown_s)
        per_cycle_attempts = int(self.config.transport_burst_resync_attempts)
        per_attempt_delay_s = float(self.config.transport_burst_resync_delay_s)
        last_resync_exc: Exception | None = None
        tel_map: dict[int, Any] | None = None
        max_cycles = 2
        for cycle_index in range(max_cycles):
            if cooldown_s > 0.0:
                session.context.sleep_fn(cooldown_s)
            for attempt_index in range(per_cycle_attempts):
                if attempt_index > 0 and per_attempt_delay_s > 0.0:
                    session.context.sleep_fn(per_attempt_delay_s)
                self._append_collect_pose_failure_event(
                    session,
                    event="transport_burst_resync_attempt",
                    payload={
                        "step_index": int(step_index_for_event),
                        "command_vector_cm": list(command_vector_cm),
                        "cycle_index": int(cycle_index + 1),
                        "attempt_index": int(attempt_index + 1),
                        "cooldown_s": cooldown_s,
                        "resync_delay_s": per_attempt_delay_s,
                        "failed_servo_ids": [int(value) for value in failed_servo_ids],
                        "tracker_age_s": self._collect_pose_tracker_age_s(session),
                    },
                )
                try:
                    candidate_map = session.context.servo_service.read_live_telemetry([int(value) for value in servo_ids])
                    bad_ids = [
                        int(servo_id)
                        for servo_id in [int(value) for value in servo_ids]
                        if (
                            int(servo_id) not in candidate_map
                            or getattr(candidate_map[int(servo_id)], "present_position", None) is None
                            or getattr(candidate_map[int(servo_id)], "telemetry_error", None) is not None
                            or getattr(candidate_map[int(servo_id)], "hardware_error", None) is not None
                        )
                    ]
                    if bad_ids:
                        raise RuntimeError(
                            "resync read returned packet/status error or missing present_position for servo(s): "
                            + ", ".join(str(value) for value in bad_ids)
                        )
                    tel_map = dict(candidate_map)
                    break
                except Exception as resync_exc:  # noqa: PERF203
                    last_resync_exc = resync_exc
            if tel_map is not None:
                break
        if tel_map is None:
            if bool(self.config.return_to_neutral_on_resync_failure):
                try:
                    self._issue_command(
                        session,
                        tendon_displacement_cm=list(zero_vector or [0.0, 0.0, 0.0, 0.0]),
                        servo_ids=list(servo_ids),
                        neutral_ticks=list(self._initial_neutral_ticks),
                        record_failure_context_on_error=False,
                    )
                except Exception:
                    pass
            session.set_metric(
                "consecutive_transport_burst_failures",
                int(session.metrics.get("consecutive_transport_burst_failures", 0) or 0) + 1,
            )
            session.set_metric(
                "consecutive_post_motion_packet_failures",
                int(session.metrics.get("consecutive_transport_burst_failures", 0) or 0),
            )
            self._append_collect_pose_failure_event(
                session,
                event="transport_burst_resync_failed",
                payload={
                    "step_index": int(step_index_for_event),
                    "command_vector_cm": list(command_vector_cm),
                    "failed_servo_ids": [int(value) for value in failed_servo_ids],
                    "cooldown_s": cooldown_s,
                    "resync_attempts": int(per_cycle_attempts * max_cycles),
                    "resync_delay_s": per_attempt_delay_s,
                    "error": str(last_resync_exc) if last_resync_exc is not None else "unknown resync failure",
                    "consecutive_transport_burst_failures": int(
                        session.metrics.get("consecutive_transport_burst_failures", 0) or 0
                    ),
                    "total_post_motion_packet_failure_events": int(total_ev),
                    "tracker_age_s": self._collect_pose_tracker_age_s(session),
                },
            )
            stop_reasons: list[str] = []
            if int(session.metrics.get("consecutive_transport_burst_failures", 0) or 0) >= int(
                self.config.max_consecutive_transport_bursts
            ):
                stop_reasons.append(
                    "collect_pose_command_dataset: exceeded max_consecutive_transport_bursts="
                    f"{int(self.config.max_consecutive_transport_bursts)} during post-motion transport recovery."
                )
            max_total = self.config.max_total_packet_failures
            if max_total is not None and int(total_ev) > int(max_total):
                stop_reasons.append(f"collect_pose_command_dataset: exceeded max_total_packet_failures={int(max_total)}.")
            max_drop_fraction = self.config.max_total_dropped_samples_fraction
            if max_drop_fraction is not None:
                accepted = int(session.metrics.get("accepted_sample_count", 0) or 0)
                rejected = int(session.metrics.get("rejected_sample_count", 0) or 0)
                dropped = int(session.metrics.get("dropped_post_motion_telemetry_samples", 0) or 0) + int(
                    session.metrics.get("dropped_pre_motion_telemetry_samples", 0) or 0
                )
                denominator = max(1, accepted + rejected)
                if float(dropped) / float(denominator) > float(max_drop_fraction):
                    stop_reasons.append(
                        "collect_pose_command_dataset: exceeded max_total_dropped_samples_fraction="
                        f"{float(max_drop_fraction):.4f}."
                    )
            if stop_reasons:
                self._stop_collect_pose_for_transport_budget(
                    session=session,
                    exc=exc,
                    requested_cable_command_cm=list(command_vector_cm),
                    servo_ids=list(servo_ids),
                    step_index_for_event=int(step_index_for_event),
                    failed_servo_ids=list(failed_servo_ids),
                    stop_reason=" ".join(stop_reasons),
                    cooldown_s=cooldown_s,
                    resync_attempts=int(per_cycle_attempts * max_cycles),
                )
            return _synthetic_collect_pose_command_result(
                exc=exc,
                step=step,
                parallel_command_metadata=parallel_command_metadata,
                resync_feedback={},
                fallback_cable_command_cm=list(zero_vector) if zero_vector is not None else None,
            )
        persistent_hardware_fault_servo_ids = [
            int(servo_id)
            for servo_id, telemetry in dict(tel_map).items()
            if getattr(telemetry, "hardware_error_code", None) not in (None, 0)
        ]
        if persistent_hardware_fault_servo_ids:
            self._stop_collect_pose_for_transport_budget(
                session=session,
                exc=exc,
                requested_cable_command_cm=list(command_vector_cm),
                servo_ids=list(servo_ids),
                step_index_for_event=int(step_index_for_event),
                failed_servo_ids=list(persistent_hardware_fault_servo_ids),
                stop_reason=(
                    "collect_pose_command_dataset: persistent hardware error bit after transport recovery on servo(s) "
                    + ", ".join(str(value) for value in persistent_hardware_fault_servo_ids)
                ),
                cooldown_s=cooldown_s,
                resync_attempts=int(per_cycle_attempts),
            )
        session.set_metric("consecutive_post_motion_packet_failures", 0)
        session.set_metric("consecutive_transport_burst_failures", 0)
        self._append_collect_pose_failure_event(
            session,
            event="transport_burst_resync_success",
            payload={
                "step_index": int(step_index_for_event),
                "command_vector_cm": list(command_vector_cm),
                "failed_servo_ids": [int(value) for value in failed_servo_ids],
                "cooldown_s": cooldown_s,
                "resync_attempts": int(per_cycle_attempts),
                "resync_delay_s": per_attempt_delay_s,
                "consecutive_transport_burst_failures": 0,
                "consecutive_post_motion_packet_failures": 0,
                "total_post_motion_packet_failure_events": int(total_ev),
                "tracker_age_s": self._collect_pose_tracker_age_s(session),
            },
        )
        # Note: renamed from "post_motion_telemetry_resync_success" because that name
        # misleadingly implied sample recovery. The bus was resynced after a post-motion
        # packet error, but the original sample was already dropped (synthetic_drop=True /
        # modeling_export_exclude=True). This event means the bus is healthy again for the
        # next command; it does NOT mean the sample was recovered. See
        # recovered_packet_error_count for true recoveries.
        self._append_collect_pose_failure_event(
            session,
            event="bus_resynced_after_drop",
            payload={
                "step_index": int(step_index_for_event),
                "failed_servo_id": ctx.get("failed_servo_id"),
                "failed_servo_ids": [int(value) for value in failed_servo_ids],
                "retry_count": retry_count,
                "resync_attempts": int(per_cycle_attempts),
                "resync_delay_s": float(per_attempt_delay_s),
                "consecutive_post_motion_packet_failures": 0,
                "total_post_motion_packet_failure_events": int(total_ev),
                "sample_recovered": False,
            },
        )
        command_metadata = dict(parallel_command_metadata)
        if bool(self.config.retry_same_command_after_resync) and step is not None:
            self._append_collect_pose_failure_event(
                session,
                event="command_retry_after_resync",
                payload={
                    "step_index": int(step_index_for_event),
                    "command_vector_cm": list(command_vector_cm or step.cable_command_cm),
                    "failed_servo_ids": [int(value) for value in failed_servo_ids],
                },
            )
            command_metadata["post_resync_command_retry_requested"] = True
        resync_fb = _servo_feedback_payload(tel_map, servo_service=session.context.servo_service)
        return _synthetic_collect_pose_command_result(
            exc=exc,
            step=step,
            parallel_command_metadata=command_metadata,
            resync_feedback=resync_fb,
            fallback_cable_command_cm=list(zero_vector) if zero_vector is not None else None,
        )

    def _dispatch_collect_pose_motion_command(
        self,
        session: ExperimentSession,
        *,
        tendon_displacement_cm: list[float],
        servo_ids: list[int],
        neutral_ticks: list[int],
        step: ModelingCommandStep | None,
        parallel_command_metadata: dict[str, Any],
        zero_vector: list[float] | None,
        step_index_for_event: int,
    ) -> dict[str, Any]:
        requested = list(tendon_displacement_cm)
        command_retries_attempted = 0
        initial_exception: Exception | None = None
        while True:
            active_ramp_step_index = 1
            active_ramp_step_count = 1
            try:
                result: dict[str, Any] | None = None
                waypoints = self._collect_pose_transition_waypoints(session, requested=list(requested))
                active_ramp_step_count = max(1, int(len(waypoints)))
                for waypoint_index, waypoint in enumerate(waypoints, start=1):
                    active_ramp_step_index = int(waypoint_index)
                    is_intermediate = waypoint_index < active_ramp_step_count
                    result = self._issue_command(
                        session,
                        tendon_displacement_cm=list(waypoint),
                        servo_ids=list(servo_ids),
                        neutral_ticks=list(neutral_ticks),
                        record_failure_context_on_error=False,
                        skip_post_command_telemetry=bool(
                            is_intermediate and not bool(self.config.ramp_include_telemetry_checks)
                        ),
                        profile_velocity_override=self.config.profile_velocity_ticks_per_s,
                        profile_acceleration_override=self.config.profile_acceleration,
                    )
                    if bool(self.config.ramp_log_intermediate_telemetry) and is_intermediate:
                        self._append_collect_pose_failure_event(
                            session,
                            event="ramp_intermediate_step",
                            payload={
                                "step_index": int(step_index_for_event),
                                "ramp_step_index": int(waypoint_index),
                                "ramp_step_count": int(active_ramp_step_count),
                                "requested_cable_command_cm": list(requested),
                                "waypoint_cable_command_cm": list(waypoint),
                                "servo_feedback": dict(result.get("servo_feedback", {}) or {}),
                            },
                        )
                    if is_intermediate and float(self.config.ramp_step_settle_s) > 0.0:
                        session.context.sleep_fn(float(self.config.ramp_step_settle_s))
                    session.set_metric("last_dispatched_cable_command_cm", [float(value) for value in waypoint])
                assert result is not None
                result.setdefault("command_metadata", {})
                result["command_metadata"]["ramp"] = {
                    "enabled": bool(self.config.command_transition_ramp_enabled),
                    "ramp_step_count": int(active_ramp_step_count),
                    "max_delta_cm_per_ramp_step": (
                        self.config.max_delta_cm_per_ramp_step
                        if self.config.max_delta_cm_per_ramp_step is not None
                        else self.config.max_delta_cm_per_transition
                    ),
                    "ramp_step_settle_s": float(self.config.ramp_step_settle_s),
                    "ramp_include_telemetry_checks": bool(self.config.ramp_include_telemetry_checks),
                }
                session.set_metric("consecutive_post_motion_packet_failures", 0)
                session.set_metric("consecutive_transport_burst_failures", 0)
                if command_retries_attempted > 0:
                    # A previously-failing command produced a usable sample after retry/resync; this is a real recovery.
                    session.set_metric(
                        "recovered_packet_error_count",
                        int(session.metrics.get("recovered_packet_error_count", 0) or 0) + 1,
                    )
                return result
            except Exception as exc:
                if initial_exception is None:
                    initial_exception = exc
                # Post-motion packet errors keep the existing synthetic-drop recovery path.
                if self._should_recover_collect_pose_post_motion_packet_error(exc):
                    retry_error = _find_servo_telemetry_retry_error(exc)
                    if retry_error is not None:
                        recovered = self._recover_collect_pose_post_motion_packet_error(
                            session,
                            retry_error,
                            step=step,
                            servo_ids=list(servo_ids),
                            parallel_command_metadata=dict(parallel_command_metadata),
                            zero_vector=zero_vector,
                            step_index_for_event=int(step_index_for_event),
                        )
                        retry_requested = bool(
                            recovered.get("command_metadata", {}).get("post_resync_command_retry_requested")
                        )
                        if retry_requested and bool(self.config.retry_same_command_after_resync):
                            try:
                                retried = self._issue_command(
                                    session,
                                    tendon_displacement_cm=list(requested),
                                    servo_ids=list(servo_ids),
                                    neutral_ticks=list(neutral_ticks),
                                    record_failure_context_on_error=False,
                                )
                                self._append_collect_pose_failure_event(
                                    session,
                                    event="command_retry_after_resync_success",
                                    payload={
                                        "step_index": int(step_index_for_event),
                                        "command_vector_cm": list(requested),
                                    },
                                )
                                session.set_metric("consecutive_post_motion_packet_failures", 0)
                                return retried
                            except Exception as retry_exc:
                                self._append_collect_pose_failure_event(
                                    session,
                                    event="command_retry_after_resync_failed",
                                    payload={
                                        "step_index": int(step_index_for_event),
                                        "command_vector_cm": list(requested),
                                        "error": str(retry_exc),
                                    },
                                )
                        return recovered
                overcurrent_servo_id, measured_abs_current_ma, threshold_ma = self._parse_post_move_overcurrent(exc)
                if overcurrent_servo_id is not None:
                    guard = session.context.servo_service.safety_guard
                    transient_threshold_ma = int(
                        self.config.transient_current_spike_ma
                        if self.config.transient_current_spike_ma is not None
                        else guard.transient_current_spike_ma
                    )
                    sustained_threshold_ma = int(
                        self.config.sustained_jam_current_ma
                        if self.config.sustained_jam_current_ma is not None
                        else guard.sustained_jam_current_ma
                    )
                    sustained_cycles = max(1, int(self.config.sustained_jam_cycles))
                    transient_policy = str(self.config.transient_spike_policy or "warn_drop_sample_continue").strip().lower()
                    sustained_policy = str(self.config.sustained_jam_policy or "stop_safely").strip().lower()
                    spike_budget = max(1, int(self.config.current_spike_max_events_per_servo))
                    sid = str(int(overcurrent_servo_id))
                    transient_counts = self._metric_int_map(session.metrics, "transient_current_spike_count_by_servo")
                    transient_counts[sid] = int(transient_counts.get(sid, 0)) + 1
                    self._set_metric_int_map(session, "transient_current_spike_count_by_servo", transient_counts)
                    event_counts = self._metric_int_map(session.metrics, "current_spike_event_count_by_servo")
                    event_counts[sid] = int(event_counts.get(sid, 0)) + 1
                    self._set_metric_int_map(session, "current_spike_event_count_by_servo", event_counts)
                    if int(event_counts[sid]) > int(spike_budget):
                        raise RuntimeError(
                            "collect_pose_command_dataset: repeated high-current spikes exceeded "
                            f"current_spike_max_events_per_servo={int(spike_budget)} on servo {sid}. "
                            "Likely mechanical/tendon issue."
                        ) from exc
                    if float(self.config.current_spike_cooldown_s) > 0.0:
                        session.context.sleep_fn(float(self.config.current_spike_cooldown_s))
                    resync_feedback: dict[int, Any] = {}
                    resync_current_ma: int | None = None
                    resync_error: str | None = None
                    if bool(self.config.current_spike_resync_enabled):
                        try:
                            resync_feedback = _collect_pose_resync_telemetry(
                                session,
                                list(servo_ids),
                                attempts=max(1, int(self.config.resync_read_attempts)),
                                delay_s=max(0.0, float(self.config.resync_delay_s)),
                            )
                            feedback_row = dict(_servo_feedback_payload(resync_feedback, servo_service=session.context.servo_service).get(sid, {}) or {})
                            if feedback_row.get("present_current_ma") is not None:
                                resync_current_ma = int(feedback_row.get("present_current_ma"))
                        except Exception as resync_exc:
                            resync_error = str(resync_exc)
                    consecutive = self._metric_int_map(session.metrics, "current_spike_consecutive_high_by_servo")
                    sustained_now = (
                        (measured_abs_current_ma is not None and int(measured_abs_current_ma) >= int(sustained_threshold_ma))
                        or (resync_current_ma is not None and abs(int(resync_current_ma)) >= int(sustained_threshold_ma))
                    )
                    if sustained_now:
                        consecutive[sid] = int(consecutive.get(sid, 0)) + 1
                    else:
                        consecutive[sid] = 0
                    self._set_metric_int_map(session, "current_spike_consecutive_high_by_servo", consecutive)
                    sustained_counts = self._metric_int_map(session.metrics, "sustained_current_exceedance_count_by_servo")
                    if sustained_now:
                        sustained_counts[sid] = int(sustained_counts.get(sid, 0)) + 1
                    self._set_metric_int_map(session, "sustained_current_exceedance_count_by_servo", sustained_counts)
                    self._append_collect_pose_failure_event(
                        session,
                        event="current_spike_detected",
                        payload={
                            "step_index": int(step_index_for_event),
                            "ramp_step_index": int(active_ramp_step_index),
                            "ramp_step_count": int(active_ramp_step_count),
                            "servo_id": int(overcurrent_servo_id),
                            "requested_cable_command_cm": list(requested),
                            "measured_abs_current_ma": measured_abs_current_ma,
                            "threshold_ma": threshold_ma,
                            "transient_threshold_ma": int(transient_threshold_ma),
                            "sustained_threshold_ma": int(sustained_threshold_ma),
                            "resync_current_ma": resync_current_ma,
                            "resync_error": resync_error,
                            "transient_count_for_servo": int(transient_counts[sid]),
                            "consecutive_high_for_servo": int(consecutive[sid]),
                        },
                    )
                    if sustained_now and sustained_policy == "stop_safely" and int(consecutive[sid]) >= int(sustained_cycles):
                        raise RuntimeError(
                            "collect_pose_command_dataset: sustained overcurrent/jam detected on servo "
                            f"{sid} ({int(consecutive[sid])} consecutive high-current cycle(s), "
                            f"threshold={int(sustained_threshold_ma)} mA)."
                        ) from exc
                    if transient_policy != "warn_drop_sample_continue":
                        raise
                    if bool(self.config.current_spike_return_to_previous_safe_goal):
                        previous_safe = session.metrics.get("last_dispatched_cable_command_cm")
                        if isinstance(previous_safe, list) and len(previous_safe) == len(requested):
                            try:
                                self._issue_command(
                                    session,
                                    tendon_displacement_cm=[float(value) for value in previous_safe],
                                    servo_ids=list(servo_ids),
                                    neutral_ticks=list(neutral_ticks),
                                    record_failure_context_on_error=False,
                                    skip_post_command_telemetry=True,
                                    profile_velocity_override=self.config.profile_velocity_ticks_per_s,
                                    profile_acceleration_override=self.config.profile_acceleration,
                                )
                            except Exception:
                                pass
                    session.set_metric(
                        "transient_current_spike_drop_count",
                        int(session.metrics.get("transient_current_spike_drop_count", 0) or 0)
                        + int(max(1, self.config.samples_per_command)),
                    )
                    return {
                        "command_deferred_or_dropped": True,
                        "deferred_reason": "transient_current_spike",
                        "requested_cable_command_cm": list(requested),
                        "resolved_cable_command_cm": list(requested),
                        "requested_pair_command_cm": _pair_command_from_cable_deltas(list(requested)),
                        "resolved_pair_command_cm": _pair_command_from_cable_deltas(list(requested)),
                        "command_metadata": {
                            **dict(parallel_command_metadata),
                            "transient_current_spike": True,
                            "spike_servo_id": int(overcurrent_servo_id),
                            "spike_abs_current_ma": measured_abs_current_ma,
                            "sustained_jam_current_ma": int(sustained_threshold_ma),
                            "sustained_jam_cycles": int(sustained_cycles),
                            "ramp_step_index": int(active_ramp_step_index),
                            "ramp_step_count": int(active_ramp_step_count),
                        },
                        "servo_feedback": _servo_feedback_payload(
                            resync_feedback or session.context.servo_service.last_known_telemetry([int(value) for value in servo_ids]),
                            servo_service=session.context.servo_service,
                        ),
                    }
                # Pre-motion packet/missing-position failures should not immediately kill long runs.
                if self._should_recover_collect_pose_pre_motion_packet_error(exc):
                    retry_error, ctx = self._collect_pose_failure_ctx(exc)
                    failed_servo_id = ctx.get("failed_servo_id")
                    missing_fields = dict(ctx.get("missing_fields", {}) or {})
                    telemetry_error_code = ctx.get("telemetry_error_code")
                    tracker_age_s = None
                    if session.context.tracking_service is not None:
                        try:
                            reader = getattr(session.context.tracking_service, "peek_snapshot", None)
                            snapshot = reader() if callable(reader) else session.context.tracking_service.get_snapshot()
                            tracker_age_s = getattr(snapshot, "tracker_data_age_s", None)
                        except Exception:
                            tracker_age_s = None
                    last_safe = _servo_feedback_payload(
                        session.context.servo_service.last_known_telemetry([int(value) for value in servo_ids]),
                        servo_service=session.context.servo_service,
                    )
                    _append_collect_pose_jsonl_event(
                        _collect_pose_output_root(session),
                        "sample_failure_events.jsonl",
                        {
                            "event": "pre_motion_telemetry_packet_error",
                            "step_index": int(step_index_for_event),
                            "command_vector_cm": list(requested),
                            "failed_servo_id": failed_servo_id,
                            "missing_fields": missing_fields,
                            "telemetry_error_code": telemetry_error_code,
                            "retry_count": int(command_retries_attempted),
                            "resync_attempts": 0,
                            "tracker_age_s": tracker_age_s,
                            "last_known_safe_telemetry": last_safe,
                        },
                    )
                    max_retries = max(0, int(self.config.telemetry_retry_count))
                    if command_retries_attempted < max_retries:
                        command_retries_attempted += 1
                        session.set_metric(
                            "servo_telemetry_retry_count",
                            int(session.metrics.get("servo_telemetry_retry_count", 0) or 0) + 1,
                        )
                        _append_collect_pose_jsonl_event(
                            _collect_pose_output_root(session),
                            "sample_failure_events.jsonl",
                            {
                                "event": "pre_motion_command_retry",
                                "step_index": int(step_index_for_event),
                                "command_vector_cm": list(requested),
                                "retry_count": int(command_retries_attempted),
                                "retry_delay_s": float(self.config.telemetry_retry_delay_s),
                                "failed_servo_id": failed_servo_id,
                            },
                        )
                        if float(self.config.telemetry_retry_delay_s) > 0.0:
                            session.context.sleep_fn(float(self.config.telemetry_retry_delay_s))
                        continue
                    # After command-level retries, force resync and one final command retry.
                    try:
                        _collect_pose_resync_telemetry(
                            session,
                            list(servo_ids),
                            attempts=int(self.config.resync_read_attempts),
                            delay_s=float(self.config.resync_delay_s),
                        )
                        _append_collect_pose_jsonl_event(
                            _collect_pose_output_root(session),
                            "sample_failure_events.jsonl",
                            {
                                # Pre-motion path: bus is resynced and immediately followed by a
                                # command retry. If the retry succeeds, recovered_packet_error_count
                                # is bumped at the success return and the sample is included.
                                "event": "pre_motion_telemetry_resync_success",
                                "step_index": int(step_index_for_event),
                                "command_vector_cm": list(requested),
                                "failed_servo_id": failed_servo_id,
                                "retry_count": int(command_retries_attempted),
                                "resync_attempts": int(self.config.resync_read_attempts),
                                "resync_delay_s": float(self.config.resync_delay_s),
                                "tracker_age_s": tracker_age_s,
                            },
                        )
                        result = self._issue_command(
                            session,
                            tendon_displacement_cm=list(requested),
                            servo_ids=list(servo_ids),
                            neutral_ticks=list(neutral_ticks),
                            record_failure_context_on_error=False,
                        )
                        session.set_metric("consecutive_post_motion_packet_failures", 0)
                        # Pre-motion bus resync + final command retry succeeded; count this as a real recovery.
                        session.set_metric(
                            "recovered_packet_error_count",
                            int(session.metrics.get("recovered_packet_error_count", 0) or 0) + 1,
                        )
                        return result
                    except Exception as retry_after_resync_exc:
                        # Continue below as unrecovered only for the same recoverable pre-motion pattern.
                        if self._should_recover_collect_pose_pre_motion_packet_error(retry_after_resync_exc):
                            fail_exc = retry_after_resync_exc
                        else:
                            self._record_failure_context(
                                session=session,
                                exc=retry_after_resync_exc,
                                requested_cable_command_cm=list(requested),
                                servo_ids=list(servo_ids),
                            )
                            raise
                        consecutive, total_ev = self._collect_pose_bump_packet_failure_metrics(
                            session,
                            dropped_count=max(1, int(self.config.samples_per_command)),
                            pre_motion=True,
                        )
                        _append_collect_pose_jsonl_event(
                            _collect_pose_output_root(session),
                            "sample_failure_events.jsonl",
                            {
                                "event": "command_deferred_or_dropped",
                                "step_index": int(step_index_for_event),
                                "command_vector_cm": list(requested),
                                "failed_servo_id": failed_servo_id,
                                "retry_count": int(command_retries_attempted),
                                "resync_attempts": int(self.config.resync_read_attempts),
                                "consecutive_post_motion_packet_failures": int(consecutive),
                                "total_post_motion_packet_failure_events": int(total_ev),
                                "reason": "pre_motion_telemetry_unrecovered_after_retry_and_resync",
                            },
                        )
                        if consecutive >= int(self.config.max_consecutive_packet_failures):
                            self._record_failure_context(
                                session=session,
                                exc=fail_exc,
                                requested_cable_command_cm=list(requested),
                                servo_ids=list(servo_ids),
                                skip_packet_error_metric_bump=True,
                            )
                            raise RuntimeError(
                                "collect_pose_command_dataset: exceeded max_consecutive_packet_failures="
                                f"{int(self.config.max_consecutive_packet_failures)} during pre-motion recovery."
                            ) from fail_exc
                        max_total = self.config.max_total_packet_failures
                        if max_total is not None and int(total_ev) > int(max_total):
                            self._record_failure_context(
                                session=session,
                                exc=fail_exc,
                                requested_cable_command_cm=list(requested),
                                servo_ids=list(servo_ids),
                                skip_packet_error_metric_bump=True,
                            )
                            raise RuntimeError(
                                f"collect_pose_command_dataset: exceeded max_total_packet_failures={int(max_total)}."
                            ) from fail_exc
                        return {
                            "command_deferred_or_dropped": True,
                            "deferred_reason": "pre_motion_telemetry_unrecovered",
                            "requested_cable_command_cm": list(requested),
                            "resolved_cable_command_cm": list(requested),
                            "requested_pair_command_cm": _pair_command_from_cable_deltas(list(requested)),
                            "resolved_pair_command_cm": _pair_command_from_cable_deltas(list(requested)),
                            "command_metadata": {
                                **dict(parallel_command_metadata),
                                "pre_motion_telemetry_unrecovered": True,
                                "telemetry_retry_count": int(command_retries_attempted),
                            },
                            "servo_feedback": last_safe,
                        }
                if _is_workspace_boundary_rejection(exc):
                    self._append_collect_pose_failure_event(
                        session,
                        event="command_skipped_workspace_boundary",
                        payload={
                            "step_index": int(step_index_for_event),
                            "requested_cable_command_cm": list(requested),
                            "error": str(exc),
                        },
                    )
                    session.set_metric(
                        "workspace_boundary_skip_count",
                        int(session.metrics.get("workspace_boundary_skip_count", 0) or 0) + 1,
                    )
                    return {
                        "command_deferred_or_dropped": True,
                        "deferred_reason": "workspace_boundary",
                        "requested_cable_command_cm": list(requested),
                        "resolved_cable_command_cm": list(requested),
                        "requested_pair_command_cm": _pair_command_from_cable_deltas(list(requested)),
                        "resolved_pair_command_cm": _pair_command_from_cable_deltas(list(requested)),
                        "command_metadata": {
                            **dict(parallel_command_metadata),
                            "workspace_boundary_rejection": True,
                        },
                    }
                self._record_failure_context(
                    session=session,
                    exc=exc,
                    requested_cable_command_cm=list(requested),
                    servo_ids=list(servo_ids),
                )
                raise

    def execute(self, session: ExperimentSession) -> None:
        servo_ids = list(self._servo_ids)
        neutral_ticks = list(self._initial_neutral_ticks)
        pair_limit_servo_ids = list(servo_ids)
        pair_limit_neutral_ticks = list(neutral_ticks)
        if _collect_pose_parallel_single_mode(session):
            active_ids = [int(value) for value in session.context.settings.robot.active_segment_servo_ids()]
            active_neutral_by_servo = {int(servo_id): int(tick) for servo_id, tick in zip(servo_ids, neutral_ticks)}
            pair_limit_servo_ids = list(active_ids)
            pair_limit_neutral_ticks = [
                int(active_neutral_by_servo[servo_id])
                for servo_id in active_ids
                if servo_id in active_neutral_by_servo
            ]
        pair_limits = _collect_pose_pair_limits(
            session=session,
            config=self.config,
            servo_ids=pair_limit_servo_ids,
            neutral_ticks=pair_limit_neutral_ticks,
        )
        command_steps_all = _build_collect_pose_command_steps(
            config=self.config,
            pair_limits=pair_limits,
        )
        resume_from = max(0, int(self.config.resume_from_command_index))
        command_steps = [step for step in command_steps_all if int(step.index) >= int(resume_from)]
        samples_per_command = max(1, int(self.config.samples_per_command))
        target_valid_samples = int(self.config.target_valid_sample_count or self.config.sample_count_target)
        continue_until_valid = bool(self.config.continue_until_valid_samples)
        configured_max_total_attempts = (
            int(self.config.max_total_attempts) if self.config.max_total_attempts is not None else None
        )
        max_total_attempts = configured_max_total_attempts
        if continue_until_valid and max_total_attempts is None:
            max_total_attempts = max(
                int(target_valid_samples) * 3,
                int(len(command_steps_all) * samples_per_command) + 2,
            )
        configured_max_dropped_fraction = (
            float(self.config.max_dropped_fraction)
            if self.config.max_dropped_fraction is not None
            else self.config.max_total_dropped_samples_fraction
        )
        configured_max_consecutive_failures = (
            int(self.config.max_consecutive_failures) if self.config.max_consecutive_failures is not None else None
        )
        total = max(2 + (len(command_steps) * samples_per_command), 2 + int(target_valid_samples))
        accepted_count = 0
        rejected_count = 0
        accepted_workspace_count = 0
        complete_training_row_count = 0
        incomplete_workspace_count = 0
        non_training_accepted_count = 0
        dropped_quarantined_count = 0
        total_workspace_attempts = 0
        consecutive_incomplete_or_dropped = 0
        progress = 0
        zero_vector = [0.0, 0.0, 0.0, 0.0]
        servo_only_mode = _collect_pose_servo_only_test_mode(config=self.config, tracking_service=session.context.tracking_service)
        parallel_single_demo = _collect_pose_parallel_single_mode(session)
        parallel_single_demo = _collect_pose_parallel_single_mode(session)
        run_trust_mode = "servo_only" if servo_only_mode else str(self.config.run_trust_mode or "thesis_trusted")
        session.set_metric("dataset_mode", str(self.config.dataset_mode or "workspace_coverage"))
        session.set_metric("dry_run", bool(self.config.dry_run))
        session.set_metric("run_trust_mode", run_trust_mode)
        session.set_metric("tracker_connected", bool(session.context.tracking_service is not None))
        session.set_metric("parallel_single_demo", bool(parallel_single_demo))
        session.set_metric("true_two_segment_control", False)
        session.set_metric("valid_for_model_training", not bool(servo_only_mode or parallel_single_demo))
        session.set_metric("valid_for_thesis_repeatability", not bool(servo_only_mode or parallel_single_demo))
        session.set_metric("not_model_training_ready", bool(servo_only_mode or parallel_single_demo))
        if servo_only_mode:
            session.add_warning(
                "Tracker is not connected. This run is allowed as a servo-only hardware test. "
                "Tip position, robot-frame pose, modeling labels, and thesis repeatability metrics will not be produced."
            )
        if parallel_single_demo:
            session.add_warning(
                "Parallel Single Demo mode is active: the same single-segment command is sent to Spine 1 and Spine 2. "
                "This is synchronized demo playback, not true two-segment kinematics/control."
            )
        session.set_metric("run_label", str(self.config.run_label or ""))
        session.set_metric("dataset_tag", str(self.config.dataset_tag or ""))
        session.set_metric(
            "dataset_stability_parameters",
            {
                "post_write_settle_s": float(self.config.post_write_settle_s),
                "telemetry_retry_count": int(self.config.telemetry_retry_count),
                "telemetry_retry_delay_s": float(self.config.telemetry_retry_delay_s),
                "allow_recovered_packet_errors": bool(self.config.allow_recovered_packet_errors),
                "max_recovered_packet_errors_per_run": self.config.max_recovered_packet_errors_per_run,
                "max_current_warning_ma": self.config.max_current_warning_ma,
                "current_warning_ma": self.config.current_warning_ma,
                "transient_current_spike_ma": self.config.transient_current_spike_ma,
                "sustained_jam_current_ma": self.config.sustained_jam_current_ma,
                "sustained_jam_cycles": int(self.config.sustained_jam_cycles),
                "transient_spike_policy": str(self.config.transient_spike_policy),
                "sustained_jam_policy": str(self.config.sustained_jam_policy),
                "current_spike_resync_enabled": bool(self.config.current_spike_resync_enabled),
                "current_spike_cooldown_s": float(self.config.current_spike_cooldown_s),
                "current_spike_return_to_previous_safe_goal": bool(self.config.current_spike_return_to_previous_safe_goal),
                "current_spike_max_events_per_servo": int(self.config.current_spike_max_events_per_servo),
                "command_transition_ramp_enabled": bool(self.config.command_transition_ramp_enabled),
                "max_delta_cm_per_ramp_step": (
                    self.config.max_delta_cm_per_ramp_step
                    if self.config.max_delta_cm_per_ramp_step is not None
                    else self.config.max_delta_cm_per_transition
                ),
                "ramp_step_settle_s": float(self.config.ramp_step_settle_s),
                "ramp_include_telemetry_checks": bool(self.config.ramp_include_telemetry_checks),
                "profile_velocity_ticks_per_s": self.config.profile_velocity_ticks_per_s,
                "profile_acceleration": self.config.profile_acceleration,
                "transport_burst_recovery_enabled": bool(self._transport_burst_recovery_enabled()),
                "transport_burst_cooldown_s": float(self.config.transport_burst_cooldown_s),
                "transport_burst_resync_attempts": int(self.config.transport_burst_resync_attempts),
                "transport_burst_resync_delay_s": float(self.config.transport_burst_resync_delay_s),
                "retry_same_command_after_resync": bool(self.config.retry_same_command_after_resync),
                "max_consecutive_transport_bursts": int(self.config.max_consecutive_transport_bursts),
                "max_total_packet_failures": self.config.max_total_packet_failures,
                "max_total_dropped_samples_fraction": self.config.max_total_dropped_samples_fraction,
                "max_total_attempts": max_total_attempts,
                "max_dropped_fraction": configured_max_dropped_fraction,
                "max_consecutive_failures": configured_max_consecutive_failures,
            },
        )
        session.set_metric("recovered_packet_error_count", 0)
        session.set_metric("unrecovered_packet_error_count", 0)
        session.set_metric("servo_telemetry_retry_count", 0)
        session.set_metric("dropped_post_motion_telemetry_samples", 0)
        session.set_metric("dropped_pre_motion_telemetry_samples", 0)
        session.set_metric("consecutive_post_motion_packet_failures", 0)
        session.set_metric("total_post_motion_packet_failure_events", 0)
        session.set_metric("transport_burst_count", 0)
        session.set_metric("consecutive_transport_burst_failures", 0)
        session.set_metric("max_consecutive_transport_bursts", int(self.config.max_consecutive_transport_bursts))
        session.set_metric("write_goal_packet_error_count", 0)
        session.set_metric("transient_current_spike_count_by_servo", {})
        session.set_metric("sustained_current_exceedance_count_by_servo", {})
        session.set_metric("current_spike_event_count_by_servo", {})
        session.set_metric("current_spike_consecutive_high_by_servo", {})
        session.set_metric("transient_current_spike_drop_count", 0)
        session.set_metric("resume_from_command_index", int(resume_from))
        session.set_metric("next_command_index_to_resume", int(resume_from))
        session.set_metric("target_valid_sample_count", int(target_valid_samples))
        session.set_metric("continue_until_valid_samples", bool(continue_until_valid))
        session.set_metric("max_total_attempts", max_total_attempts)
        session.set_metric("max_dropped_fraction", configured_max_dropped_fraction)
        session.set_metric("max_consecutive_failures", configured_max_consecutive_failures)
        session.set_metric("accepted_sample_count", 0)
        session.set_metric("rejected_sample_count", 0)
        session.set_metric("accepted_workspace_sample_count", 0)
        session.set_metric("complete_training_row_count", 0)
        session.set_metric("accepted_training_row_count", 0)
        session.set_metric("incomplete_accepted_workspace_row_count", 0)
        session.set_metric("non_training_accepted_row_count", 0)
        session.set_metric("dropped_quarantined_sample_count", 0)
        session.set_metric("total_workspace_attempt_count", 0)
        session.set_metric("consecutive_incomplete_or_dropped_count", 0)
        session.set_metric("remaining_complete_training_rows", int(target_valid_samples))
        session.set_metric("complete_training_target_reached", False)
        session.set_metric("last_dispatched_cable_command_cm", list(zero_vector))
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
        session.set_metric("command_step_count", int(len(command_steps_all)))
        session.set_metric("command_steps_executed_count", 0)
        session.set_metric("samples_per_command", int(samples_per_command))
        session.set_metric("pair_limits_cm", pair_limits)
        session.set_metric("dataset_mode_summary", _collect_pose_mode_summary(self.config))
        session.set_metric(
            "schedule_checksum",
            command_schedule_checksum(
                [
                    ExperimentPoint(
                        index=int(step.index),
                        tendon_displacement_cm=list(step.cable_command_cm),
                        settle_time_s=float(step.settle_time_s),
                        repeat=1,
                        label=str(step.label),
                    )
                    for step in command_steps_all
                ]
            ),
        )
        _record_collect_pose_run_provenance(
            session=session,
            config=self.config,
            servo_ids=servo_ids,
            neutral_ticks=neutral_ticks,
            pair_limits=pair_limits,
        )

        def _remaining_complete_rows() -> int:
            return max(0, int(target_valid_samples) - int(complete_training_row_count))

        def _complete_target_reached() -> bool:
            return bool(continue_until_valid) and int(complete_training_row_count) >= int(target_valid_samples)

        def _apply_runtime_row_counters(sample: ExperimentTimeseriesSample, *, workspace_attempt: bool) -> None:
            nonlocal accepted_workspace_count
            nonlocal complete_training_row_count
            nonlocal incomplete_workspace_count
            nonlocal non_training_accepted_count
            nonlocal dropped_quarantined_count
            nonlocal total_workspace_attempts
            nonlocal consecutive_incomplete_or_dropped
            phase = str(sample.phase or "")
            extra = dict(sample.extra or {})
            accepted = bool(extra.get("capture_accepted"))
            excluded = bool(extra.get("modeling_export_exclude"))
            if excluded:
                dropped_quarantined_count += 1
            is_complete_training_row = False
            if accepted:
                if phase in NON_TRAINING_PHASES:
                    non_training_accepted_count += 1
                else:
                    accepted_workspace_count += 1
                    is_complete_training_row = bool((not excluded) and sample_has_complete_command_servo_tip(sample))
                    if is_complete_training_row:
                        complete_training_row_count += 1
                    else:
                        incomplete_workspace_count += 1
            if workspace_attempt:
                total_workspace_attempts += 1
                if is_complete_training_row:
                    consecutive_incomplete_or_dropped = 0
                else:
                    consecutive_incomplete_or_dropped += 1
            session.set_metric("accepted_workspace_sample_count", int(accepted_workspace_count))
            session.set_metric("complete_training_row_count", int(complete_training_row_count))
            session.set_metric("accepted_training_row_count", int(complete_training_row_count))
            session.set_metric("incomplete_accepted_workspace_row_count", int(incomplete_workspace_count))
            session.set_metric("non_training_accepted_row_count", int(non_training_accepted_count))
            session.set_metric("dropped_quarantined_sample_count", int(dropped_quarantined_count))
            session.set_metric("total_workspace_attempt_count", int(total_workspace_attempts))
            session.set_metric("consecutive_incomplete_or_dropped_count", int(consecutive_incomplete_or_dropped))
            session.set_metric("remaining_complete_training_rows", int(_remaining_complete_rows()))
            session.set_metric("complete_training_target_reached", bool(_complete_target_reached()))

        def _record_deferred_workspace_attempts(attempt_count: int) -> None:
            nonlocal total_workspace_attempts
            nonlocal consecutive_incomplete_or_dropped
            if int(attempt_count) <= 0:
                return
            total_workspace_attempts += int(attempt_count)
            consecutive_incomplete_or_dropped += int(attempt_count)
            session.set_metric("total_workspace_attempt_count", int(total_workspace_attempts))
            session.set_metric("consecutive_incomplete_or_dropped_count", int(consecutive_incomplete_or_dropped))
            session.set_metric("remaining_complete_training_rows", int(_remaining_complete_rows()))
            session.set_metric("complete_training_target_reached", bool(_complete_target_reached()))

        def _enforce_continue_mode_budgets() -> None:
            if not continue_until_valid:
                return
            stop_reasons: list[str] = []
            if max_total_attempts is not None and int(total_workspace_attempts) >= int(max_total_attempts):
                stop_reasons.append(
                    "collect_pose_command_dataset: exceeded max_total_attempts="
                    f"{int(max_total_attempts)} before reaching target_valid_sample_count={int(target_valid_samples)} "
                    f"(complete_training_row_count={int(complete_training_row_count)})."
                )
            if configured_max_dropped_fraction is not None and int(total_workspace_attempts) > 0:
                dropped_total = int(session.metrics.get("dropped_post_motion_telemetry_samples", 0) or 0) + int(
                    session.metrics.get("dropped_pre_motion_telemetry_samples", 0) or 0
                )
                dropped_fraction = float(dropped_total) / float(max(1, int(total_workspace_attempts)))
                if dropped_fraction > float(configured_max_dropped_fraction):
                    stop_reasons.append(
                        "collect_pose_command_dataset: exceeded max_dropped_fraction="
                        f"{float(configured_max_dropped_fraction):.4f} before reaching target_valid_sample_count="
                        f"{int(target_valid_samples)} (dropped_fraction={dropped_fraction:.4f})."
                    )
            if (
                configured_max_consecutive_failures is not None
                and int(consecutive_incomplete_or_dropped) >= int(configured_max_consecutive_failures)
            ):
                stop_reasons.append(
                    "collect_pose_command_dataset: exceeded max_consecutive_failures="
                    f"{int(configured_max_consecutive_failures)} before reaching target_valid_sample_count="
                    f"{int(target_valid_samples)}."
                )
            if stop_reasons:
                raise RuntimeError(" ".join(stop_reasons))

        command_owner = contextlib.nullcontext()
        if not self.config.dry_run and getattr(session.context.servo_service, "is_connected", False):
            command_owner = session.context.servo_service.exclusive_bus_operation(
                owner=self.name,
                reason="collect-pose dataset experiment motion",
            )
        with command_owner:
            parallel_neutral_meta = _collect_pose_parallel_command_metadata(
                session, servo_ids=servo_ids, cable_command_cm=zero_vector
            )
            neutral_command = self._dispatch_collect_pose_motion_command(
                session,
                tendon_displacement_cm=zero_vector,
                servo_ids=servo_ids,
                neutral_ticks=neutral_ticks,
                step=None,
                parallel_command_metadata=parallel_neutral_meta,
                zero_vector=zero_vector,
                step_index_for_event=-1,
            )
            neutral_sample = self._capture_dataset_sample(
                session=session,
                command_result=neutral_command,
                phase="initial_neutral",
                step_index=-1,
                sample_index=0,
                servo_ids=servo_ids,
                previous_pair_command_cm=None,
                block_index=None,
                prior_family=None,
                step_metadata={"role": "initial_neutral"},
            )
            session.add_sample(neutral_sample)
            accepted_count += int(bool(neutral_sample.extra.get("capture_accepted")))
            rejected_count += int(not bool(neutral_sample.extra.get("capture_accepted")))
            _apply_runtime_row_counters(neutral_sample, workspace_attempt=False)
            session.set_metric("accepted_sample_count", int(accepted_count))
            session.set_metric("rejected_sample_count", int(rejected_count))
            progress += 1
            session.update_progress(progress, total, {"phase": "initial_neutral", "step_index": -1})

            previous_pair_command_cm = [0.0, 0.0]
            commands_completed_in_run = 0
            if not command_steps:
                raise RuntimeError(
                    "collect_pose_command_dataset has no command steps to execute after resume_from_command_index="
                    f"{int(resume_from)}."
                )
            scheduled_command_count = max(1, int(len(command_steps)))
            max_runtime_step_index = int(resume_from)
            schedule_cycle = 0
            while True:
                if _complete_target_reached():
                    break
                if (not continue_until_valid) and schedule_cycle > 0:
                    break
                for step_offset, base_step in enumerate(command_steps):
                    if _complete_target_reached():
                        break
                    runtime_step_index = int(base_step.index) + int(schedule_cycle) * int(scheduled_command_count)
                    max_runtime_step_index = max(max_runtime_step_index, runtime_step_index)
                    session.raise_if_stop_requested()
                    parallel_step_meta = _collect_pose_parallel_command_metadata(
                        session, servo_ids=servo_ids, cable_command_cm=list(base_step.cable_command_cm)
                    )
                    command_result = self._dispatch_collect_pose_motion_command(
                        session,
                        tendon_displacement_cm=list(base_step.cable_command_cm),
                        servo_ids=servo_ids,
                        neutral_ticks=neutral_ticks,
                        step=base_step,
                        parallel_command_metadata=parallel_step_meta,
                        zero_vector=zero_vector,
                        step_index_for_event=int(runtime_step_index),
                    )
                    session.set_metric("next_command_index_to_resume", int(runtime_step_index) + 1)
                    commands_completed_in_run += 1
                    if bool(command_result.get("command_deferred_or_dropped")):
                        _record_deferred_workspace_attempts(int(samples_per_command))
                        _enforce_continue_mode_budgets()
                        session.update_progress(
                            progress,
                            total,
                            {
                                "phase": str(base_step.phase),
                                "step_index": int(runtime_step_index),
                                "command_label": str(base_step.label),
                                "accepted_samples": int(accepted_count),
                                "rejected_samples": int(rejected_count),
                                "deferred": True,
                                "complete_training_row_count": int(complete_training_row_count),
                                "remaining_complete_training_rows": int(_remaining_complete_rows()),
                            },
                        )
                        continue
                    settle_time_s = float(
                        base_step.settle_time_s if base_step.settle_time_s is not None else self.config.settle_time_s
                    )
                    if settle_time_s > 0.0:
                        session.context.sleep_fn(settle_time_s)
                    tracker_samples_n = max(1, int(self.config.tracker_samples_per_command))
                    if tracker_samples_n > 1:
                        if _complete_target_reached():
                            pass
                        else:
                            self._capture_one_command_with_averaging(
                                session=session,
                                command_result=command_result,
                                base_step=base_step,
                                runtime_step_index=runtime_step_index,
                                servo_ids=servo_ids,
                                previous_pair_command_cm=previous_pair_command_cm,
                                tracker_samples_n=tracker_samples_n,
                                accepted_inc=lambda: None,
                            )
                            # Counter increments + progress are handled inside
                            # _capture_one_command_with_averaging via closures
                            # injected next; for now use post-call bookkeeping.
                            last_sample = session.samples[-1] if session.samples else None
                            if last_sample is not None:
                                accepted_count += int(bool(last_sample.extra.get("capture_accepted")))
                                rejected_count += int(not bool(last_sample.extra.get("capture_accepted")))
                                _apply_runtime_row_counters(last_sample, workspace_attempt=True)
                            _enforce_continue_mode_budgets()
                            session.set_metric("accepted_sample_count", int(accepted_count))
                            session.set_metric("rejected_sample_count", int(rejected_count))
                            progress += 1
                            session.update_progress(
                                progress,
                                total,
                                {
                                    "phase": str(base_step.phase),
                                    "step_index": int(runtime_step_index),
                                    "command_label": str(base_step.label),
                                    "accepted_samples": int(accepted_count),
                                    "rejected_samples": int(rejected_count),
                                    "complete_training_row_count": int(complete_training_row_count),
                                    "remaining_complete_training_rows": int(_remaining_complete_rows()),
                                    "tracker_averaging_active": True,
                                    "tracker_samples_per_command": int(tracker_samples_n),
                                },
                            )
                    else:
                        for sample_index in range(samples_per_command):
                            if _complete_target_reached():
                                break
                            sample = self._capture_dataset_sample(
                                session=session,
                                command_result=command_result,
                                phase=str(base_step.phase),
                                step_index=int(runtime_step_index),
                                sample_index=int(sample_index),
                                servo_ids=servo_ids,
                                previous_pair_command_cm=list(previous_pair_command_cm),
                                block_index=base_step.block_index,
                                prior_family=base_step.prior_family,
                                step_metadata={
                                    "label": base_step.label,
                                    **dict(base_step.metadata or {}),
                                },
                            )
                            session.add_sample(sample)
                            accepted_count += int(bool(sample.extra.get("capture_accepted")))
                            rejected_count += int(not bool(sample.extra.get("capture_accepted")))
                            _apply_runtime_row_counters(sample, workspace_attempt=True)
                            _enforce_continue_mode_budgets()
                            session.set_metric("accepted_sample_count", int(accepted_count))
                            session.set_metric("rejected_sample_count", int(rejected_count))
                            progress += 1
                            session.update_progress(
                                progress,
                                total,
                                {
                                    "phase": str(base_step.phase),
                                    "step_index": int(runtime_step_index),
                                    "command_label": str(base_step.label),
                                    "accepted_samples": int(accepted_count),
                                    "rejected_samples": int(rejected_count),
                                    "complete_training_row_count": int(complete_training_row_count),
                                    "remaining_complete_training_rows": int(_remaining_complete_rows()),
                                },
                            )
                    previous_pair_command_cm = list(base_step.pair_command_cm)
                    chunk_n = self.config.chunk_flush_every_n_commands
                    if chunk_n is not None and commands_completed_in_run % int(chunk_n) == 0:
                        _write_collect_pose_checkpoint(
                            output_root=_collect_pose_output_root(session),
                            last_completed_command_index=int(runtime_step_index),
                            accepted_sample_count=int(accepted_count),
                            dropped_post_motion_telemetry_samples=int(
                                session.metrics.get("dropped_post_motion_telemetry_samples", 0) or 0
                            ),
                            next_command_index_to_resume=int(runtime_step_index) + 1,
                            total_packet_failure_events=int(
                                session.metrics.get("total_post_motion_packet_failure_events", 0) or 0
                            ),
                        )
                    if float(session.elapsed_s()) > 1e-6:
                        session.set_metric("mean_sample_hz_estimate", float(accepted_count) / float(session.elapsed_s()))
                    health_n = int(self.config.long_run_health_write_interval_samples)
                    if int(accepted_count) > 0 and int(accepted_count) % health_n == 0:
                        session.set_metric("long_run_health_recommendation", "running")
                        _write_collect_pose_long_run_health(
                            output_root=_collect_pose_output_root(session),
                            session=session,
                            metrics={
                                **dict(session.metrics),
                                "accepted_sample_count": int(accepted_count),
                                "rejected_sample_count": int(rejected_count),
                            },
                            last_command_step_index=int(runtime_step_index),
                            estimated_total_commands=max(int(len(command_steps_all)), int(runtime_step_index) + 1),
                            samples=list(session.samples),
                        )
                schedule_cycle += 1

            parallel_final_meta = _collect_pose_parallel_command_metadata(
                session, servo_ids=servo_ids, cable_command_cm=zero_vector
            )
            final_command = self._dispatch_collect_pose_motion_command(
                session,
                tendon_displacement_cm=zero_vector,
                servo_ids=servo_ids,
                neutral_ticks=neutral_ticks,
                step=None,
                parallel_command_metadata=parallel_final_meta,
                zero_vector=zero_vector,
                step_index_for_event=int(max_runtime_step_index) + 1,
            )
            if not bool(final_command.get("command_deferred_or_dropped")):
                final_sample = self._capture_dataset_sample(
                    session=session,
                    command_result=final_command,
                    phase="final_neutral",
                    step_index=int(max_runtime_step_index) + 1,
                    sample_index=0,
                    servo_ids=servo_ids,
                    previous_pair_command_cm=list(previous_pair_command_cm),
                    block_index=None,
                    prior_family=None,
                    step_metadata={"role": "final_neutral"},
                )
                session.add_sample(final_sample)
                accepted_count += int(bool(final_sample.extra.get("capture_accepted")))
                rejected_count += int(not bool(final_sample.extra.get("capture_accepted")))
                _apply_runtime_row_counters(final_sample, workspace_attempt=False)
                session.set_metric("accepted_sample_count", int(accepted_count))
                session.set_metric("rejected_sample_count", int(rejected_count))
                progress += 1
                session.update_progress(
                    progress,
                    total,
                    {"phase": "final_neutral", "step_index": int(max_runtime_step_index) + 1},
                )
            session.set_metric("command_steps_executed_count", int(commands_completed_in_run))
        session.set_metric("accepted_sample_count", int(accepted_count))
        session.set_metric("rejected_sample_count", int(rejected_count))
        if continue_until_valid:
            session.set_metric("next_command_index_to_resume", int(max_runtime_step_index) + 1)
        else:
            session.set_metric("next_command_index_to_resume", int(len(command_steps_all)))
        session.set_metric("registration_loaded", session.context.registration_path.exists())

    def finalize(self, session: ExperimentSession) -> None:
        # Stash multi-frame averaging summary so it lands in summary.json
        # alongside the rest of experiment_metrics. Safe no-op when averaging
        # wasn't active (averaged_samples list will be empty).
        averaged_samples = list(getattr(self, "_averaged_dataset_samples", []) or [])
        raw_frame_rows = list(getattr(self, "_raw_tracker_frame_rows", []) or [])
        if int(self.config.tracker_samples_per_command) > 1 or averaged_samples:
            from continuum_robot.experiments.modeling_dataset_outputs import (
                _summarize_tracker_variability,
            )
            variability_summary = _summarize_tracker_variability(averaged_samples)
            variability_summary["tracker_samples_per_command"] = int(self.config.tracker_samples_per_command)
            variability_summary["averaged_label_enabled"] = bool(
                getattr(self, "_averaged_label_enabled", False)
            )
            variability_summary["raw_tracker_frame_row_count"] = int(len(raw_frame_rows))
            session.set_metric("tracker_variability", variability_summary)
        try:
            if not self.config.dry_run and session.context.servo_service.is_connected and self._initial_neutral_ticks:
                failure_context = dict(session.metrics.get("failure_context", {}) or {})
                if (
                    session.stage_pass_fail.get("execute") == "failed"
                    and not session.samples
                    and str(failure_context.get("failure_category", "")) == "simple_experiment_motion_rejected"
                ):
                    session.add_warning(
                        "Skipping collect-pose finalize neutral command because execute failed before any "
                        "sample or goal write; primary command failure is preserved in failure_context.json."
                    )
                    return
                with session.context.servo_service.exclusive_bus_operation(
                    owner=self.name,
                    reason="collect-pose finalize neutral return",
                ):
                    self._issue_command(
                        session,
                        tendon_displacement_cm=[0.0, 0.0, 0.0, 0.0],
                        servo_ids=list(self._servo_ids),
                        neutral_ticks=list(self._initial_neutral_ticks),
                    )
        finally:
            if self._tracking_started_here and session.context.tracking_service is not None:
                session.context.tracking_service.stop()
                self._tracking_started_here = False

    def _issue_command(
        self,
        session: ExperimentSession,
        *,
        tendon_displacement_cm: list[float],
        servo_ids: list[int],
        neutral_ticks: list[int],
        skip_post_command_telemetry: bool = False,
        profile_velocity_override: int | None = None,
        profile_acceleration_override: int | None = None,
        record_failure_context_on_error: bool = True,
    ) -> dict[str, Any]:
        servo_service = session.context.servo_service
        if not neutral_ticks:
            neutral_ticks = [0 for _ in servo_ids]
        requested_cable_command_cm = [float(value) for value in tendon_displacement_cm]
        requested_pair_command_cm = _pair_command_from_cable_deltas(requested_cable_command_cm)
        LOG.info(
            "Collect-pose command dispatch | requested_cable_cm=%s | requested_pair_cm=%s | servo_ids=%s | dry_run=%s",
            requested_cable_command_cm,
            requested_pair_command_cm,
            list(servo_ids),
            bool(self.config.dry_run),
        )
        command_servo_ids = list(servo_ids)
        command_neutral_ticks = list(neutral_ticks)
        parallel_command_metadata: dict[str, Any] = {}
        parallel_mirror_pairs: dict[int, int] = {}
        if _collect_pose_parallel_single_mode(session) and len(requested_cable_command_cm) == 4:
            context = session.context.settings.robot.operating_context()
            parallel_mirror_pairs = {int(k): int(v) for k, v in dict(context.mirror_pairs or {}).items()}
            parallel_command_metadata = _collect_pose_parallel_command_metadata(
                session,
                servo_ids=command_servo_ids,
                cable_command_cm=requested_cable_command_cm,
            )
            if self.config.dry_run or not servo_service.is_connected:
                requested_cable_command_cm = _mirror_parallel_single_displacements(
                    requested_cable_command_cm,
                    servo_ids=command_servo_ids,
                    context=context,
                )
                requested_pair_command_cm = _pair_command_from_cable_deltas(requested_cable_command_cm[:4])
        if self.config.dry_run or not servo_service.is_connected:
            if len(command_neutral_ticks) != len(requested_cable_command_cm):
                raise RuntimeError("Dry-run modeling command dimensions do not match the configured servo set.")
            goals = servo_service.mapper.to_goal_positions(requested_cable_command_cm, command_neutral_ticks)
            commanded_motor_values = {str(servo_id): int(goal) for servo_id, goal in zip(command_servo_ids, goals)}
            command_metadata = dict(parallel_command_metadata)
            if command_metadata.get("parallel_single_demo"):
                segment_a_ids = [int(value) for value in command_metadata.get("segment_a", {}).get("servo_ids", [])]
                segment_b_ids = [int(value) for value in command_metadata.get("segment_b", {}).get("servo_ids", [])]
                command_metadata["segment_a_goal_ticks"] = {
                    str(servo_id): int(commanded_motor_values[str(servo_id)])
                    for servo_id in segment_a_ids
                    if str(servo_id) in commanded_motor_values
                }
                command_metadata["segment_b_goal_ticks"] = {
                    str(servo_id): int(commanded_motor_values[str(servo_id)])
                    for servo_id in segment_b_ids
                    if str(servo_id) in commanded_motor_values
                }
                command_metadata["all_8_goal_ticks"] = dict(commanded_motor_values)
            return {
                "requested_cable_command_cm": list(requested_cable_command_cm),
                "resolved_cable_command_cm": list(requested_cable_command_cm),
                "requested_pair_command_cm": list(requested_pair_command_cm),
                "resolved_pair_command_cm": list(requested_pair_command_cm),
                "commanded_motor_values": commanded_motor_values,
                "raw_goal_ticks_by_servo": dict(commanded_motor_values),
                "final_goal_ticks_by_servo": dict(commanded_motor_values),
                "clamp_reasons_by_servo": {},
                "servo_debug": {},
                "servo_feedback": {},
                "command_metadata": command_metadata,
                "message": "Dry-run command resolved through the tendon-displacement mapper only.",
                "motion_profile": {
                    "operating_mode_label": "dry_run",
                    "operating_mode": None,
                    "goal_current_ma": None,
                    "profile_velocity": None,
                    "profile_acceleration": None,
                },
            }
        original_profile_velocity = getattr(servo_service.dxl_bus.config, "single_segment_experiment_default_profile_velocity", None)
        original_profile_acc = getattr(servo_service.dxl_bus.config, "single_segment_experiment_default_profile_acceleration", None)
        if profile_velocity_override is not None:
            servo_service.dxl_bus.config.single_segment_experiment_default_profile_velocity = int(profile_velocity_override)
        if profile_acceleration_override is not None:
            servo_service.dxl_bus.config.single_segment_experiment_default_profile_acceleration = int(profile_acceleration_override)
        try:
            try:
                command = servo_service.command_displacement(
                    tendon_displacements_cm=list(requested_cable_command_cm),
                    neutral_ticks=command_neutral_ticks,
                    servo_ids=command_servo_ids,
                    motion_workflow="experiment_motion",
                    parallel_mirror_pairs=parallel_mirror_pairs or None,
                    telemetry_retry_count=int(self.config.telemetry_retry_count),
                    telemetry_retry_delay_s=float(self.config.telemetry_retry_delay_s),
                    allow_recovered_packet_errors=bool(self.config.allow_recovered_packet_errors),
                    skip_post_command_telemetry=bool(skip_post_command_telemetry),
                    write_goal_attempts=self.config.goal_write_retry_attempts,
                )
                if float(self.config.post_write_settle_s) > 0.0:
                    session.context.sleep_fn(float(self.config.post_write_settle_s))
            except Exception as exc:
                if record_failure_context_on_error:
                    self._record_failure_context(
                        session=session,
                        exc=exc,
                        requested_cable_command_cm=requested_cable_command_cm,
                        servo_ids=command_servo_ids,
                    )
                LOG.exception(
                    "Collect-pose command failed | requested_cable_cm=%s | servo_ids=%s | error=%s",
                    requested_cable_command_cm,
                    list(command_servo_ids),
                    exc,
                )
                raise
        finally:
            if profile_velocity_override is not None:
                servo_service.dxl_bus.config.single_segment_experiment_default_profile_velocity = original_profile_velocity
            if profile_acceleration_override is not None:
                servo_service.dxl_bus.config.single_segment_experiment_default_profile_acceleration = original_profile_acc
        motion_profile = _servo_motion_profile_from_result(command)
        command_metadata = {**dict(parallel_command_metadata), **dict(command.command_metadata or {})}
        if command_metadata.get("parallel_single_demo"):
            segment_a_ids = [int(value) for value in command_metadata.get("segment_a", {}).get("servo_ids", [])]
            segment_b_ids = [int(value) for value in command_metadata.get("segment_b", {}).get("servo_ids", [])]
            final_goal_ticks = {
                str(servo_id): int(goal)
                for servo_id, goal in sorted(command.positions_by_id.items())
            }
            command_metadata["segment_a_goal_ticks"] = {
                str(servo_id): int(final_goal_ticks[str(servo_id)])
                for servo_id in segment_a_ids
                if str(servo_id) in final_goal_ticks
            }
            command_metadata["segment_b_goal_ticks"] = {
                str(servo_id): int(final_goal_ticks[str(servo_id)])
                for servo_id in segment_b_ids
                if str(servo_id) in final_goal_ticks
            }
            command_metadata["all_8_goal_ticks"] = dict(final_goal_ticks)
        LOG.info(
            "Collect-pose command success | resolved_cable_cm=%s | final_goals=%s | clamp_reasons=%s | motion_profile=%s",
            list(command.resolved_displacements_cm or requested_cable_command_cm),
            {str(servo_id): int(goal) for servo_id, goal in sorted(command.positions_by_id.items())},
            {str(servo_id): str(reason) for servo_id, reason in sorted(command.clamp_reasons_by_id.items())},
            motion_profile,
        )
        result = {
            "requested_cable_command_cm": list(requested_cable_command_cm),
            "resolved_cable_command_cm": list(command.resolved_displacements_cm or requested_cable_command_cm),
            "requested_pair_command_cm": list(requested_pair_command_cm),
            "resolved_pair_command_cm": _pair_command_from_cable_deltas(
                list(command.resolved_displacements_cm or requested_cable_command_cm)
            ),
            "commanded_motor_values": {
                str(servo_id): int(goal)
                for servo_id, goal in sorted(command.positions_by_id.items())
            },
            "raw_goal_ticks_by_servo": {
                str(servo_id): int(goal)
                for servo_id, goal in sorted(command.raw_positions_by_id.items())
            },
            "final_goal_ticks_by_servo": {
                str(servo_id): int(goal)
                for servo_id, goal in sorted(command.positions_by_id.items())
            },
            "clamp_reasons_by_servo": {
                str(servo_id): str(reason)
                for servo_id, reason in sorted(command.clamp_reasons_by_id.items())
            },
            "servo_debug": _servo_debug_payload(command),
            "servo_feedback": _servo_feedback_payload(command.telemetry_by_id, servo_service=servo_service),
            "command_metadata": command_metadata,
            "message": str(command.message or ""),
            "motion_profile": motion_profile,
        }
        self._record_command_retry_metrics(session, command.command_metadata or {})
        return result

    def _record_command_retry_metrics(self, session: ExperimentSession, metadata: dict[str, Any]) -> None:
        retry_count = int(metadata.get("telemetry_retry_count", 0) or 0)
        recovered = int(metadata.get("recovered_packet_error_count", 0) or 0)
        unrecovered = int(metadata.get("unrecovered_packet_error_count", 0) or 0)
        session.set_metric(
            "servo_telemetry_retry_count",
            int(session.metrics.get("servo_telemetry_retry_count", 0) or 0) + retry_count,
        )
        session.set_metric(
            "recovered_packet_error_count",
            int(session.metrics.get("recovered_packet_error_count", 0) or 0) + recovered,
        )
        session.set_metric(
            "unrecovered_packet_error_count",
            int(session.metrics.get("unrecovered_packet_error_count", 0) or 0) + unrecovered,
        )
        max_recovered = self.config.max_recovered_packet_errors_per_run
        if max_recovered is not None and int(session.metrics.get("recovered_packet_error_count", 0) or 0) > int(max_recovered):
            raise RuntimeError(
                "Collect-pose stopped: recovered packet error count exceeded "
                f"max_recovered_packet_errors_per_run={int(max_recovered)}."
            )

    def _record_failure_context(
        self,
        *,
        session: ExperimentSession,
        exc: Exception,
        requested_cable_command_cm: list[float],
        servo_ids: list[int],
        skip_packet_error_metric_bump: bool = False,
    ) -> None:
        retry_error = _find_servo_telemetry_retry_error(exc)
        base = dict(retry_error.context if retry_error is not None else {})
        if retry_error is not None:
            if not skip_packet_error_metric_bump:
                ctx0 = dict(retry_error.context or {})
                fc0 = str(ctx0.get("failure_category", "") or "")
                if fc0 == "servo_telemetry_packet_error":
                    failed_ids = ctx0.get("failed_servo_ids")
                    if not isinstance(failed_ids, list) or not failed_ids:
                        fid = ctx0.get("failed_servo_id")
                        failed_ids = [fid] if fid is not None else []
                    bump = max(1, len(failed_ids))
                    session.set_metric(
                        "unrecovered_packet_error_count",
                        int(session.metrics.get("unrecovered_packet_error_count", 0) or 0) + int(bump),
                    )
                    session.set_metric(
                        "servo_telemetry_retry_count",
                        int(session.metrics.get("servo_telemetry_retry_count", 0) or 0)
                        + int(ctx0.get("retry_count", 0) or 0),
                    )
                elif fc0 == "write_goal_packet_error":
                    session.set_metric(
                        "write_goal_packet_error_count",
                        int(session.metrics.get("write_goal_packet_error_count", 0) or 0) + 1,
                    )
        else:
            lowered = str(exc).lower()
            if "write goal" in lowered or "write servo goal" in lowered:
                base.setdefault("failure_category", "write_goal_packet_error")
                session.set_metric(
                    "write_goal_packet_error_count",
                    int(session.metrics.get("write_goal_packet_error_count", 0) or 0) + 1,
                )
        servo_service = session.context.servo_service
        tracker_age_s = None
        if session.context.tracking_service is not None:
            try:
                reader = getattr(session.context.tracking_service, "peek_snapshot", None)
                snapshot = reader() if callable(reader) else session.context.tracking_service.get_snapshot()
                tracker_age_s = getattr(snapshot, "tracker_data_age_s", None)
            except Exception:
                tracker_age_s = None
        bus_owner = {}
        try:
            owner = servo_service.bus_ownership_status()
            bus_owner = {
                "active": bool(owner.active),
                "owner": owner.owner,
                "reason": owner.reason,
                "servo_id": owner.servo_id,
                "held_by_current_thread": bool(owner.held_by_current_thread),
            }
        except Exception:
            bus_owner = {}
        if "last_valid_telemetry_by_servo" not in base:
            try:
                base["last_valid_telemetry_by_servo"] = _servo_feedback_payload(
                    servo_service.last_known_telemetry([int(value) for value in servo_ids]),
                    servo_service=servo_service,
                )
            except Exception:
                base["last_valid_telemetry_by_servo"] = {}
        context = {
            "failure_category": base.get("failure_category", "collect_pose_command_failure"),
            "failure_reason": base.get("failure_reason", str(exc)),
            "sample_index_at_failure": int(len(session.samples)),
            "progress_fraction_at_failure": (
                float(session.completed_progress_steps) / float(session.total_progress_steps)
                if int(session.total_progress_steps or 0) > 0
                else None
            ),
            "failed_servo_id": base.get("failed_servo_id"),
            "telemetry_error_code": base.get("telemetry_error_code"),
            "missing_fields": base.get("missing_fields"),
            "last_valid_telemetry_by_servo": base.get("last_valid_telemetry_by_servo", {}),
            "last_commanded_goal_ticks": base.get("last_commanded_goal_ticks", servo_service.last_goal_positions()),
            "last_resolved_cable_command_cm": base.get("last_resolved_cable_command_cm", list(requested_cable_command_cm)),
            "command_metadata": dict(base.get("command_metadata", {}) or {}),
            "telemetry_snapshot_fields": {
                "read_batch_started_monotonic_s": True,
                "read_batch_completed_monotonic_s": True,
                "read_batch_duration_ms": True,
                "snapshot_age_s": True,
                "per_servo_packet_age_s": True,
                "freshness_decision_source": True,
            },
            "tracker_age_s": None if tracker_age_s is None else float(tracker_age_s),
            "bus_owner": bus_owner,
            "retry_count": int(base.get("retry_count", 0) or 0),
            "recovered_packet_error_count": int(
                session.metrics.get("recovered_packet_error_count", base.get("recovered_packet_error_count", 0)) or 0
            ),
            "unrecovered_packet_error_count": int(session.metrics.get("unrecovered_packet_error_count", 0) or 0),
            "servo_telemetry_retry_count": int(session.metrics.get("servo_telemetry_retry_count", 0) or 0),
        }
        session.set_metric("failure_category", context["failure_category"])
        session.set_metric("failure_reason", context["failure_reason"])
        session.set_metric("failure_context", context)

    def _capture_dataset_sample(
        self,
        session: ExperimentSession,
        *,
        command_result: dict[str, Any],
        phase: str,
        step_index: int,
        sample_index: int,
        servo_ids: list[int],
        previous_pair_command_cm: list[float] | None,
        block_index: int | None,
        prior_family: str | None,
        step_metadata: dict[str, Any] | None = None,
        pre_captured: tuple[Any, dict[str, Any]] | None = None,
        extra_overrides: dict[str, Any] | None = None,
    ) -> ExperimentTimeseriesSample:
        if _collect_pose_servo_only_test_mode(config=self.config, tracking_service=session.context.tracking_service):
            return self._capture_servo_only_dataset_sample(
                session=session,
                command_result=command_result,
                phase=phase,
                step_index=step_index,
                sample_index=sample_index,
                servo_ids=servo_ids,
                previous_pair_command_cm=previous_pair_command_cm,
                block_index=block_index,
                prior_family=prior_family,
                step_metadata=step_metadata,
            )
        if pre_captured is not None:
            # Caller already has a settled snapshot in hand (e.g. multi-frame
            # post-settle averaging path); reuse it instead of waiting again.
            snapshot, gate = pre_captured
        else:
            snapshot, gate = _wait_for_collect_pose_capture(
                session=session,
                tool_id=str(self.config.tool_id or "0A"),
                max_tracker_age_s=float(self.config.max_tracker_age_s),
                timeout_s=float(self.config.capture_timeout_s),
                poll_interval_s=float(self.config.capture_poll_interval_s),
                require_robot_frame_tip=bool(self.config.require_robot_frame_tip),
                allow_mock_state=bool(self.config.dry_run),
                allow_lower_trust_runtime_tip=bool(self.config.allow_lower_trust_runtime_tip),
            )
        accepted = bool(gate.get("accepted"))
        synthetic_drop = bool(command_result.get("post_motion_telemetry_unrecovered"))
        if synthetic_drop:
            accepted = False
        status_flags = ["capture_accepted"] if accepted else ["capture_rejected"]
        if synthetic_drop:
            status_flags.append("post_motion_telemetry_unrecovered")
        if self.config.dry_run:
            status_flags.append("dry_run")
        live_servo_feedback = _read_collect_pose_live_servo_feedback(
            session=session,
            servo_ids=servo_ids,
        )
        runtime_tip_policy = evaluate_runtime_tip_trust(
            snapshot=snapshot,
            workflow=WORKFLOW_MODELING_DATASET,
            allow_lower_trust=bool(self.config.allow_lower_trust_runtime_tip),
        )
        sample = _sample_from_tracking_snapshot(
            session,
            snapshot=snapshot,
            phase=phase,
            step_index=step_index,
            sample_index=sample_index,
            commanded_cable_deltas_cm=list(command_result.get("resolved_cable_command_cm", []) or []),
            commanded_motor_values=dict(command_result.get("final_goal_ticks_by_servo", {}) or {}),
            status_flags=status_flags,
            cycle_index=block_index,
            extra={
                "record_kind": "modeling_dataset_capture",
                "dataset_mode": str(self.config.dataset_mode or "workspace_coverage"),
                "run_label": str(self.config.run_label or ""),
                "dataset_tag": str(self.config.dataset_tag or ""),
                "tool_id": str(self.config.tool_id or "0A"),
                "capture_accepted": bool(accepted),
                "capture_rejection_reason": (
                    "post_motion_telemetry_packet_error"
                    if synthetic_drop
                    else (None if accepted else str(gate.get("reason", "tracker_gate_rejected")))
                ),
                "modeling_export_exclude": bool(synthetic_drop),
                "tracker_gate": dict(gate),
                "runtime_tip_mode": runtime_tip_policy.mode,
                "runtime_tip_trust_level": runtime_tip_policy.trust_label,
                "runtime_tip_policy": runtime_tip_policy.to_dict(),
                "requested_pair_command_cm": list(command_result.get("requested_pair_command_cm", []) or []),
                "resolved_pair_command_cm": list(command_result.get("resolved_pair_command_cm", []) or []),
                "previous_pair_command_cm": list(previous_pair_command_cm or []),
                "requested_cable_command_cm": list(command_result.get("requested_cable_command_cm", []) or []),
                "resolved_cable_command_cm": list(command_result.get("resolved_cable_command_cm", []) or []),
                "raw_goal_ticks_by_servo": dict(command_result.get("raw_goal_ticks_by_servo", {}) or {}),
                "final_goal_ticks_by_servo": dict(command_result.get("final_goal_ticks_by_servo", {}) or {}),
                "clamp_reasons_by_servo": dict(command_result.get("clamp_reasons_by_servo", {}) or {}),
                "motion_profile": dict(command_result.get("motion_profile", {}) or {}),
                "command_metadata": dict(command_result.get("command_metadata", {}) or {}),
                "servo_feedback_at_command": dict(command_result.get("servo_feedback", {}) or {}),
                "servo_feedback_at_capture": dict(live_servo_feedback),
                "servo_debug": dict(command_result.get("servo_debug", {}) or {}),
                "command_message": str(command_result.get("message", "") or ""),
                "prior_family": None if prior_family in (None, "") else str(prior_family),
                "block_index": None if block_index is None else int(block_index),
                "step_metadata": dict(step_metadata or {}),
                "sequential_order_preserved": True,
                "model_io_convention": {
                    "inputs": "resolved_cable_command_cm with explicit pair commands and ordered previous_pair_command_cm context",
                    "outputs": "robot-frame tip translation_mm plus tip tangent_xyz/quaternion_wxyz when available",
                },
            },
        )
        if not accepted and "capture_rejected" not in sample.status_flags:
            sample.status_flags.append("capture_rejected")
        if snapshot.registration_state != "loaded":
            if "registration_missing" not in sample.status_flags:
                sample.status_flags.append("registration_missing")
        elif snapshot.T_robot_tip is not None and "full_pose_available" not in sample.status_flags:
            sample.status_flags.append("full_pose_available")
        if extra_overrides:
            sample.extra.update(dict(extra_overrides))
        return sample

    def _capture_one_command_with_averaging(
        self,
        session: ExperimentSession,
        *,
        command_result: dict[str, Any],
        base_step: "ModelingCommandStep",
        runtime_step_index: int,
        servo_ids: list[int],
        previous_pair_command_cm: list[float] | None,
        tracker_samples_n: int,
        accepted_inc,
    ) -> None:
        """Multi-frame post-settle averaging path for one command point.

        Always emits at least one row to ``session.samples`` (the first-frame
        label, possibly marked rejected if no valid frames arrived). When
        averaging is enabled and at least one valid frame was collected, also
        appends an averaged-label row to ``self._averaged_dataset_samples``
        and per-frame raw records to ``self._raw_tracker_frame_rows``.

        This is the canonical "one settle, N frame collection, two label
        variants" path. samples_per_command's old "one settle, N independent
        captures, N rows" path is preserved on the legacy branch.
        """
        _ = accepted_inc  # reserved for future closure-based counter wiring
        tool_id = str(self.config.tool_id or "0A")
        poll_interval_s = float(
            self.config.tracker_sample_period_s
            if self.config.tracker_sample_period_s is not None
            else self.config.capture_poll_interval_s
        )
        frames, frames_reason = _collect_post_settle_tracker_frames(
            session=session,
            tool_id=tool_id,
            max_tracker_age_s=float(self.config.max_tracker_age_s),
            per_frame_max_wait_s=float(self.config.tracker_per_frame_max_wait_s),
            poll_interval_s=poll_interval_s,
            require_robot_frame_tip=bool(self.config.require_robot_frame_tip),
            allow_mock_state=bool(self.config.dry_run),
            allow_lower_trust_runtime_tip=bool(self.config.allow_lower_trust_runtime_tip),
            tracker_samples_per_command=tracker_samples_n,
        )
        step_metadata = {
            "label": base_step.label,
            **dict(base_step.metadata or {}),
        }
        averaging_meta_base = {
            "tracker_averaging_requested_n": int(tracker_samples_n),
            "tracker_averaging_valid_n": int(len(frames)),
            "tracker_averaging_underflow": bool(frames_reason is not None),
            "tracker_averaging_underflow_reason": frames_reason,
            "tracker_averaging_window_s": (
                float(frames[-1]["monotonic_t"] - frames[0]["monotonic_t"])
                if len(frames) >= 2
                else None
            ),
        }
        if not frames:
            # Zero valid frames: synthesize a rejected first-row by going
            # through the normal capture path with no pre-captured snapshot
            # so it polls (and likely fails the gate again, producing the
            # standard "capture_rejected" row).
            rejected_sample = self._capture_dataset_sample(
                session=session,
                command_result=command_result,
                phase=str(base_step.phase),
                step_index=int(runtime_step_index),
                sample_index=0,
                servo_ids=servo_ids,
                previous_pair_command_cm=list(previous_pair_command_cm or []),
                block_index=base_step.block_index,
                prior_family=base_step.prior_family,
                step_metadata=step_metadata,
                extra_overrides={
                    "label_kind": "first",
                    **averaging_meta_base,
                },
            )
            session.add_sample(rejected_sample)
            return

        first_sample = self._capture_dataset_sample(
            session=session,
            command_result=command_result,
            phase=str(base_step.phase),
            step_index=int(runtime_step_index),
            sample_index=0,
            servo_ids=servo_ids,
            previous_pair_command_cm=list(previous_pair_command_cm or []),
            block_index=base_step.block_index,
            prior_family=base_step.prior_family,
            step_metadata=step_metadata,
            pre_captured=(frames[0]["snapshot"], frames[0]["gate"]),
            extra_overrides={
                "label_kind": "first",
                **averaging_meta_base,
            },
        )
        session.add_sample(first_sample)

        if self._averaged_label_enabled:
            averaged = _average_tracker_frames(frames=frames, tool_id=tool_id)
            averaged_sample = self._build_averaged_dataset_sample(
                session=session,
                first_sample=first_sample,
                averaged=averaged,
                tool_id=tool_id,
                extra_overrides=dict(averaging_meta_base),
            )
            self._averaged_dataset_samples.append(averaged_sample)

        for frame in frames:
            self._raw_tracker_frame_rows.append(
                _serialize_raw_tracker_frame(
                    session=session,
                    command_index=int(runtime_step_index),
                    frame=frame,
                    tool_id=tool_id,
                )
            )

    def _build_averaged_dataset_sample(
        self,
        session: ExperimentSession,
        *,
        first_sample: ExperimentTimeseriesSample,
        averaged: dict[str, Any],
        tool_id: str,
        extra_overrides: dict[str, Any] | None = None,
    ) -> ExperimentTimeseriesSample:
        """Build an averaged-label row from the already-built first-frame row.

        Mutates pose fields in a deep copy of ``first_sample`` so the returned
        sample carries identical metadata (command, telemetry, status flags)
        but with the per-axis pose replaced by the averaged values from
        :func:`_average_tracker_frames`. Honest framing: this is *averaged
        random tracker noise*, not corrected for systematic bias.
        """
        averaged_sample = copy.deepcopy(first_sample)
        tool_key = str(tool_id or "0A").upper()
        matrix = list(averaged["averaged_T_robot_tip"])
        averaged_position = list(averaged["averaged_robot_position_mm"])
        averaged_quat = list(averaged["averaged_robot_quaternion_wxyz"])
        averaged_tangent = [float(matrix[0][2]), float(matrix[1][2]), float(matrix[2][2])]
        if "tip" in (averaged_sample.pose_in_robot_frame or {}):
            averaged_sample.pose_in_robot_frame["tip"] = {
                "matrix": matrix,
                "translation_mm": averaged_position,
                "quaternion_wxyz": averaged_quat,
                "tangent_xyz": averaged_tangent,
            }
        if tool_key in (averaged_sample.pose_in_tracker_frame or {}):
            tracker_entry = dict(averaged_sample.pose_in_tracker_frame[tool_key])
            if averaged.get("averaged_tracker_translation_mm") is not None:
                tracker_entry["translation_mm"] = list(averaged["averaged_tracker_translation_mm"])
            if averaged.get("averaged_tracker_quaternion_wxyz") is not None:
                tracker_entry["quaternion_wxyz"] = list(averaged["averaged_tracker_quaternion_wxyz"])
                # Recompute tangent from the averaged quaternion using the
                # same convention the standard builder uses.
                quat = np.asarray(tracker_entry["quaternion_wxyz"], dtype=float)
                norm = float(np.linalg.norm(quat))
                if norm > 1e-12:
                    w, x, y, z = (quat / norm).tolist()
                    tracker_entry["tangent_xyz"] = [
                        2.0 * (x * z + y * w),
                        2.0 * (y * z - x * w),
                        1.0 - 2.0 * (x * x + y * y),
                    ]
            averaged_sample.pose_in_tracker_frame[tool_key] = tracker_entry
        if extra_overrides:
            averaged_sample.extra.update(dict(extra_overrides))
        averaged_sample.extra["label_kind"] = "averaged"
        averaged_sample.extra["tracker_averaging"] = dict(averaged.get("stats", {}))
        return averaged_sample

    def _capture_servo_only_dataset_sample(
        self,
        session: ExperimentSession,
        *,
        command_result: dict[str, Any],
        phase: str,
        step_index: int,
        sample_index: int,
        servo_ids: list[int],
        previous_pair_command_cm: list[float] | None,
        block_index: int | None,
        prior_family: str | None,
        step_metadata: dict[str, Any] | None = None,
    ) -> ExperimentTimeseriesSample:
        live_servo_feedback = _read_collect_pose_live_servo_feedback(
            session=session,
            servo_ids=servo_ids,
        )
        status_flags = [
            "capture_accepted",
            "servo_only",
            "no_tracker",
            "lower_trust",
            "not_thesis_trusted",
            "not_model_training_ready",
        ]
        return ExperimentTimeseriesSample(
            monotonic_time_s=session.elapsed_s(),
            wall_time_utc=datetime.now(timezone.utc).isoformat(),
            phase=phase,
            step_index=int(step_index),
            sample_index=int(sample_index),
            cycle_index=None if block_index is None else int(block_index),
            commanded_motor_values=dict(command_result.get("final_goal_ticks_by_servo", {}) or {}),
            commanded_cable_deltas_cm=list(command_result.get("resolved_cable_command_cm", []) or []),
            tracker_frame_id=None,
            tool_ids_seen=[],
            transform_validity={},
            pose_in_tracker_frame={},
            pose_in_robot_frame={},
            freshness_s=None,
            latency_s=None,
            status_flags=sorted(set(status_flags)),
            backend_health={
                "canonical_state": "disabled",
                "selected_backend_name": "none",
                "backend_identity": "servo_only_no_tracker",
            },
            extra={
                "record_kind": "servo_only_motion_test",
                "dataset_mode": str(self.config.dataset_mode or "workspace_coverage"),
                "run_label": str(self.config.run_label or ""),
                "dataset_tag": str(self.config.dataset_tag or ""),
                "tool_id": str(self.config.tool_id or "0A"),
                "capture_accepted": True,
                "capture_rejection_reason": None,
                "tracker_gate": {
                    "accepted": False,
                    "reason": "servo_only_no_tracker_test_run",
                    "tracker_connected": False,
                },
                "run_trust_mode": "servo_only",
                "valid_for_model_training": False,
                "valid_for_thesis_repeatability": False,
                "not_model_training_ready": True,
                "runtime_tip_mode": "unavailable",
                "runtime_tip_trust_level": "servo_only",
                "runtime_tip_policy": None,
                "requested_pair_command_cm": list(command_result.get("requested_pair_command_cm", []) or []),
                "resolved_pair_command_cm": list(command_result.get("resolved_pair_command_cm", []) or []),
                "previous_pair_command_cm": list(previous_pair_command_cm or []),
                "requested_cable_command_cm": list(command_result.get("requested_cable_command_cm", []) or []),
                "resolved_cable_command_cm": list(command_result.get("resolved_cable_command_cm", []) or []),
                "raw_goal_ticks_by_servo": dict(command_result.get("raw_goal_ticks_by_servo", {}) or {}),
                "final_goal_ticks_by_servo": dict(command_result.get("final_goal_ticks_by_servo", {}) or {}),
                "clamp_reasons_by_servo": dict(command_result.get("clamp_reasons_by_servo", {}) or {}),
                "motion_profile": dict(command_result.get("motion_profile", {}) or {}),
                "command_metadata": dict(command_result.get("command_metadata", {}) or {}),
                "servo_feedback_at_command": dict(command_result.get("servo_feedback", {}) or {}),
                "servo_feedback_at_capture": dict(live_servo_feedback),
                "servo_debug": dict(command_result.get("servo_debug", {}) or {}),
                "command_message": str(command_result.get("message", "") or ""),
                "prior_family": None if prior_family in (None, "") else str(prior_family),
                "block_index": None if block_index is None else int(block_index),
                "step_metadata": dict(step_metadata or {}),
                "sequential_order_preserved": True,
                "model_io_convention": {
                    "inputs": "resolved_cable_command_cm and servo telemetry only",
                    "outputs": "unavailable; no tracker or robot-frame labels were captured",
                },
            },
        )

    def summarize(self, session: ExperimentSession) -> dict[str, Any]:
        samples = list(session.samples)
        accepted_samples = [sample for sample in samples if bool(sample.extra.get("capture_accepted"))]
        rejected_samples = [sample for sample in samples if sample.extra.get("capture_accepted") is False]
        command_samples = [
            sample
            for sample in samples
            if sample.phase not in {"initial_neutral", "final_neutral"}
        ]
        robot_positions = [
            list(sample.pose_in_robot_frame.get("tip", {}).get("translation_mm", []))
            for sample in accepted_samples
            if isinstance(sample.pose_in_robot_frame.get("tip", {}).get("translation_mm"), list)
            and len(sample.pose_in_robot_frame.get("tip", {}).get("translation_mm")) == 3
        ]
        tip_tangents = [
            list(sample.pose_in_robot_frame.get("tip", {}).get("tangent_xyz", []))
            for sample in accepted_samples
            if isinstance(sample.pose_in_robot_frame.get("tip", {}).get("tangent_xyz"), list)
            and len(sample.pose_in_robot_frame.get("tip", {}).get("tangent_xyz")) == 3
        ]
        requested_pair_commands = [
            list(sample.extra.get("requested_pair_command_cm", []))
            for sample in command_samples
            if isinstance(sample.extra.get("requested_pair_command_cm"), list)
            and len(sample.extra.get("requested_pair_command_cm")) == 2
        ]
        resolved_pair_commands = [
            list(sample.extra.get("resolved_pair_command_cm", []))
            for sample in command_samples
            if isinstance(sample.extra.get("resolved_pair_command_cm"), list)
            and len(sample.extra.get("resolved_pair_command_cm")) == 2
        ]
        servo_only_mode = _collect_pose_servo_only_test_mode(config=self.config, tracking_service=session.context.tracking_service)
        parallel_single_demo = _collect_pose_parallel_single_mode(session)
        snapshot = session.context.tracking_service.get_snapshot() if session.context.tracking_service is not None else None
        runtime_tip_policy = (
            evaluate_runtime_tip_trust(
                snapshot=snapshot,
                workflow=WORKFLOW_MODELING_DATASET,
                allow_lower_trust=bool(self.config.allow_lower_trust_runtime_tip),
            )
            if snapshot is not None
            else None
        )
        runtime_tip_mode = runtime_tip_policy.mode if runtime_tip_policy is not None else "unavailable"
        pretension_source = session.context.servo_service.pretension_source_summary(list(self._servo_ids))
        strict_runtime_tip = not bool(self.config.allow_lower_trust_runtime_tip)
        strict_pretension = not bool(self.config.allow_lower_trust_pretension)
        lower_trust_active = (
            bool(self.config.dry_run)
            or bool(servo_only_mode)
            or runtime_tip_policy is None
            or not bool(runtime_tip_policy.thesis_trusted)
            or (not pretension_source.accepted or not pretension_source.usable)
            or (not strict_pretension and pretension_source.source_type not in {"manual", "algorithmic"})
        )
        base_train = bool(
            (not servo_only_mode)
            and runtime_tip_policy is not None
            and runtime_tip_policy.thesis_trusted
            and bool(robot_positions)
        )
        metrics = {
            **dict(session.metrics),
            "dataset_mode": str(self.config.dataset_mode or "workspace_coverage"),
            "dataset_mode_summary": _collect_pose_mode_summary(self.config),
            "command_step_count": int(session.metrics.get("command_step_count", 0) or 0),
            "samples_per_command": int(self.config.samples_per_command),
            "planned_capture_count": int((session.metrics.get("command_step_count", 0) or 0) * max(1, int(self.config.samples_per_command)) + 2),
            "accepted_sample_count": int(len(accepted_samples)),
            "rejected_sample_count": int(len(rejected_samples)),
            "accepted_capture_rate": (
                float(len(accepted_samples)) / float(len(accepted_samples) + len(rejected_samples))
                if (len(accepted_samples) + len(rejected_samples)) > 0
                else None
            ),
            "robot_frame_tip_sample_count": int(len(robot_positions)),
            "orientation_sample_count": int(len(tip_tangents)),
            "tracker_connected": bool(session.context.tracking_service is not None),
            "registration_available": bool(snapshot is not None and getattr(snapshot, "registration_state", None) == "loaded"),
            "run_trust_mode": "servo_only" if servo_only_mode else str(self.config.run_trust_mode or "thesis_trusted"),
            "valid_for_model_training": bool((not parallel_single_demo) and base_train),
            "valid_for_thesis_repeatability": bool((not parallel_single_demo) and base_train),
            "not_model_training_ready": bool(servo_only_mode or parallel_single_demo or not robot_positions),
            "parallel_single_demo": bool(parallel_single_demo),
            "true_two_segment_control": False,
            "runtime_tip_mode_used": runtime_tip_mode,
            "runtime_tip_trust_level": runtime_tip_policy.trust_label if runtime_tip_policy is not None else "servo_only",
            "runtime_tip_policy": runtime_tip_policy.to_dict() if runtime_tip_policy is not None else None,
            "thesis_trusted_runtime_tip": bool(runtime_tip_policy.thesis_trusted) if runtime_tip_policy is not None else False,
            "pretension_source_used": pretension_source.source_type,
            "pretension_source_message": pretension_source.message,
            "requires_robot_frame_tip": bool(self.config.require_robot_frame_tip),
            "lower_trust_active": bool(lower_trust_active),
            "sample_order_preserved": True,
            "legacy_export_enabled": bool(self.config.export_legacy_dat and not servo_only_mode),
            "position_frame": "robot" if robot_positions else ("none" if servo_only_mode else "tracker"),
            "command_pair_range_cm": _collect_pose_pair_range(resolved_pair_commands or requested_pair_commands),
            "workspace_span_mm": _collect_pose_workspace_span(robot_positions),
            "command_norm_stats_cm": _collect_pose_command_norm_stats(resolved_pair_commands or requested_pair_commands),
            "rejection_reasons": _collect_pose_rejection_counts(rejected_samples),
            "phase_counts": _collect_pose_phase_counts(samples),
            "summary_requirements": {
                "min_sample_count": 1,
                "require_registration": bool(self.config.require_robot_frame_tip and not self.config.dry_run and strict_runtime_tip and not servo_only_mode),
                "registration_available": bool(robot_positions),
                "require_tip_calibration": bool(self.config.require_robot_frame_tip and not self.config.dry_run and strict_runtime_tip and not servo_only_mode),
                "tip_calibration_available": bool(robot_positions),
                "allow_partial_missing_registration": bool(self.config.dry_run or servo_only_mode or not strict_runtime_tip),
                "allow_partial_missing_tip_cal": bool(self.config.dry_run or servo_only_mode or not strict_runtime_tip),
                "invalid_transforms_are_fatal": False,
                **(
                    {"force_status": "invalid_due_to_insufficient_samples"}
                    if not accepted_samples
                    else ({"force_status": "partial_success"} if lower_trust_active else {})
                ),
            },
        }
        return metrics

    def write_outputs(self, session: ExperimentSession, paths, summary) -> None:
        write_modeling_dataset_outputs(
            output_dir=paths.output_dir,
            metadata=session.metadata,
            summary=summary,
            samples=session.samples,
            averaged_samples=list(getattr(self, "_averaged_dataset_samples", []) or []),
            raw_tracker_frame_rows=list(getattr(self, "_raw_tracker_frame_rows", []) or []),
            tracker_samples_per_command=int(self.config.tracker_samples_per_command),
            averaged_label_enabled=bool(getattr(self, "_averaged_label_enabled", False)),
            export_first_sample_label=bool(self.config.export_first_sample_label),
            export_averaged_sample_label=bool(getattr(self, "_export_averaged_sample_label", False)),
        )
        failure_context = dict(session.metrics.get("failure_context", {}) or {})
        if failure_context:
            (paths.output_dir / "failure_context.json").write_text(
                json.dumps(failure_context, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        precheck_ctx = dict(session.metrics.get("precheck_failure_context", {}) or {})
        if precheck_ctx:
            (paths.output_dir / "precheck_failure_context.json").write_text(
                json.dumps(precheck_ctx, indent=2, sort_keys=False),
                encoding="utf-8",
            )
        quality = _collect_pose_dataset_quality_summary(
            samples=list(session.samples),
            metrics=summary.experiment_metrics if isinstance(summary.experiment_metrics, dict) else dict(session.metrics),
            max_current_warning_ma=(
                self.config.current_warning_ma
                if self.config.current_warning_ma is not None
                else self.config.max_current_warning_ma
            ),
        )
        summary_metrics = summary.experiment_metrics if isinstance(summary.experiment_metrics, dict) else dict(session.metrics)
        final_recommendation = _collect_pose_long_run_recommendation(
            success=bool(summary.success),
            metrics=dict(summary_metrics),
        )
        long_run_metrics = {
            **dict(summary_metrics),
            "accepted_sample_count": int(quality.get("accepted_sample_count", 0) or 0),
            "rejected_sample_count": int(quality.get("rejected_sample_count", 0) or 0),
            "run_status": str(summary.status),
            "run_success": bool(summary.success),
            "long_run_health_recommendation": str(final_recommendation),
        }
        _write_collect_pose_long_run_health(
            output_root=paths.output_dir,
            session=session,
            metrics=long_run_metrics,
            last_command_step_index=max(-1, int(long_run_metrics.get("next_command_index_to_resume", 0) or 0) - 1),
            estimated_total_commands=int(long_run_metrics.get("command_step_count", 0) or 0),
            samples=list(session.samples),
        )
        (paths.output_dir / "dataset_quality_summary.json").write_text(
            json.dumps(quality, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (paths.output_dir / "dataset_quality_summary.txt").write_text(
            _render_collect_pose_dataset_quality_summary(quality),
            encoding="utf-8",
        )


def _configured_collect_pose_servo_ids(session: ExperimentSession) -> list[int]:
    robot = session.context.settings.robot
    servo_ids = [int(value) for value in robot.operating_context().commanded_servo_ids]
    return list(servo_ids)


def _find_servo_telemetry_retry_error(exc: BaseException | None) -> ServoTelemetryRetryError | None:
    current = exc
    seen = 0
    while current is not None and seen < 8:
        if isinstance(current, ServoTelemetryRetryError):
            return current
        current = current.__cause__ or current.__context__
        seen += 1
    return None


def _is_workspace_boundary_rejection(exc: BaseException) -> bool:
    """True iff exc is a single-segment motion rejection caused purely by hardware/safe range bounds.

    The schedule generator may propose tendon commands at the edge of the workspace; the
    servo safety layer correctly rejects ones that would drive a servo past the raw [0, 4095]
    range or past the configured single-segment envelope. Treat these as deferrable rather
    than as run-terminating telemetry/hardware faults.
    """
    retry_error = _find_servo_telemetry_retry_error(exc)
    if retry_error is None:
        return False
    context = getattr(retry_error, "context", None) or {}
    if context.get("failure_category") != "simple_experiment_motion_rejected":
        return False
    text = (str(retry_error) + " " + str(exc)).lower()
    return "hard bound rejection" in text or "hardware-limit rejection" in text


def _collect_pose_dataset_quality_summary(
    *,
    samples: list[ExperimentTimeseriesSample],
    metrics: dict[str, Any],
    max_current_warning_ma: int | None,
) -> dict[str, Any]:
    accepted = [sample for sample in samples if bool(sample.extra.get("capture_accepted"))]
    rejected = [sample for sample in samples if sample.extra.get("capture_accepted") is False]
    failure_context = dict(metrics.get("failure_context", {}) or {})
    commands = [
        list(sample.extra.get("resolved_cable_command_cm") or sample.commanded_cable_deltas_cm or [])
        for sample in samples
    ]
    commands = [values for values in commands if values]
    xyz_values = [
        list(sample.pose_in_robot_frame.get("tip", {}).get("translation_mm", []))
        for sample in accepted
        if isinstance(sample.pose_in_robot_frame.get("tip"), dict)
    ]
    xyz_values = [values for values in xyz_values if len(values) == 3]
    deltas = [
        b - a
        for a, b in zip(
            [float(sample.monotonic_time_s) for sample in samples],
            [float(sample.monotonic_time_s) for sample in samples][1:],
        )
        if b >= a
    ]
    current_summary = _collect_pose_current_summary(samples)
    max_abs_current = current_summary.get("max_abs_current_ma")
    trainable = bool(metrics.get("valid_for_model_training"))
    validity_status = str(metrics.get("model_training_validity_status", "not_applicable") or "not_applicable")
    validity_reason = str(metrics.get("model_training_validity_reason", "") or "")
    validity_checks = dict(metrics.get("model_training_validity_checks", {}) or {})
    model_training_warnings = [str(value) for value in list(metrics.get("model_training_warnings", []) or []) if str(value)]
    hard_invalid_reasons = [
        str(value) for value in list(metrics.get("model_training_hard_invalidation_reasons", []) or []) if str(value)
    ]
    failure_category = str(failure_context.get("failure_category") or metrics.get("failure_category") or "")
    unrec = int(metrics.get("unrecovered_packet_error_count", 0) or 0)
    drops = int(metrics.get("dropped_post_motion_telemetry_samples", 0) or 0)
    drops_pre = int(metrics.get("dropped_pre_motion_telemetry_samples", 0) or 0)
    if failure_context:
        if failure_category == "write_goal_packet_error":
            recommendation = "failed_due_to_write_goal_packet_error"
        elif "telemetry" in failure_category or "packet" in failure_category:
            recommendation = "failed_due_to_telemetry"
        else:
            recommendation = "run_failed"
    elif (drops > 0 or drops_pre > 0) and unrec > 0:
        recommendation = "partial_data_telemetry_drops_exportable"
    elif trainable and len(accepted) >= max(20, int(metrics.get("command_step_count", 0) or 0)):
        recommendation = "good_for_training"
    elif len(accepted) > 0:
        recommendation = "needs_more_coverage"
    else:
        recommendation = "good_for_debug"
    high_current_warning = (
        max_abs_current is not None
        and max_current_warning_ma is not None
        and int(max_abs_current) >= int(max_current_warning_ma)
    )
    return {
        "schema_version": "collect_pose_dataset_quality_v1",
        "accepted_sample_count": int(len(accepted)),
        "rejected_sample_count": int(len(rejected)),
        "failure_count": (1 if failure_context else 0),
        "dropped_post_motion_telemetry_samples": drops,
        "dropped_pre_motion_telemetry_samples": drops_pre,
        "next_command_index_to_resume": int(metrics.get("next_command_index_to_resume", 0) or 0),
        "recovered_packet_error_count": int(metrics.get("recovered_packet_error_count", 0) or 0),
        "unrecovered_packet_error_count": int(metrics.get("unrecovered_packet_error_count", 0) or 0),
        "transport_burst_count": int(metrics.get("transport_burst_count", 0) or 0),
        "consecutive_transport_burst_failures": int(metrics.get("consecutive_transport_burst_failures", 0) or 0),
        "total_post_motion_packet_failure_events": int(metrics.get("total_post_motion_packet_failure_events", 0) or 0),
        "tracker_stale_count": sum(
            1
            for sample in rejected
            if "stale" in str(sample.extra.get("capture_rejection_reason", "")).lower()
            or "tracker_age" in str(sample.extra.get("capture_rejection_reason", "")).lower()
        ),
        "servo_telemetry_retry_count": int(metrics.get("servo_telemetry_retry_count", 0) or 0),
        "write_goal_packet_error_count": int(metrics.get("write_goal_packet_error_count", 0) or 0),
        "command_range_per_tendon_cm": _range_by_index(commands, prefix="tendon"),
        "measured_xyz_range_mm": _range_by_index(xyz_values, prefix="axis", labels=["x", "y", "z"]),
        "sample_rate_stats": _sample_rate_stats(deltas),
        "max_abs_current_ma": max_abs_current,
        "max_current_warning_ma": max_current_warning_ma,
        "high_current_warning": bool(high_current_warning),
        "max_abs_current_ma_by_servo": dict(current_summary.get("max_abs_current_ma_by_servo", {}) or {}),
        "mean_abs_current_ma_by_servo": dict(current_summary.get("mean_abs_current_ma_by_servo", {}) or {}),
        "p95_abs_current_ma_by_servo": dict(current_summary.get("p95_abs_current_ma_by_servo", {}) or {}),
        "input_voltage_min_mv_by_servo": dict(current_summary.get("input_voltage_min_mv_by_servo", {}) or {}),
        "current_voltage_correlation_notes_by_servo": dict(
            current_summary.get("current_voltage_correlation_notes_by_servo", {}) or {}
        ),
        "peak_current_sample_by_servo": dict(current_summary.get("peak_current_sample_by_servo", {}) or {}),
        "transient_current_spike_count_by_servo": dict(
            metrics.get("transient_current_spike_count_by_servo", {}) or {}
        ),
        "sustained_current_exceedance_count_by_servo": dict(
            metrics.get("sustained_current_exceedance_count_by_servo", {}) or {}
        ),
        "transient_current_spike_drop_count": int(metrics.get("transient_current_spike_drop_count", 0) or 0),
        "trainability_status": {
            "valid_for_model_training": trainable,
            "not_model_training_ready": bool(metrics.get("not_model_training_ready")),
            "run_trust_mode": str(metrics.get("run_trust_mode", "unknown")),
        },
        "model_training_validity_status": validity_status,
        "model_training_validity_reason": validity_reason,
        "model_training_validity_checks": validity_checks,
        "model_training_warnings": model_training_warnings,
        "model_training_hard_invalidation_reasons": hard_invalid_reasons,
        "dropped_samples_excluded_from_training": bool(metrics.get("dropped_samples_excluded_from_training")),
        "accepted_rows_complete": bool(metrics.get("accepted_rows_complete")),
        "accepted_workspace_sample_count": int(metrics.get("accepted_workspace_sample_count", 0) or 0),
        "complete_training_row_count": int(metrics.get("complete_training_row_count", 0) or 0),
        "target_valid_sample_count": int(metrics.get("target_valid_sample_count", 0) or 0),
        "remaining_complete_training_rows": int(metrics.get("remaining_complete_training_rows", 0) or 0),
        "non_training_accepted_row_count": int(metrics.get("non_training_accepted_row_count", 0) or 0),
        "incomplete_accepted_workspace_row_count": int(metrics.get("incomplete_accepted_workspace_row_count", 0) or 0),
        "dropped_quarantined_sample_count": int(metrics.get("dropped_quarantined_sample_count", 0) or 0),
        "modeling_export_row_count": int(metrics.get("modeling_export_row_count", 0) or 0),
        "modeling_legacy_row_count": int(metrics.get("modeling_legacy_row_count", 0) or 0),
        "accepted_training_row_count": int(metrics.get("accepted_training_row_count", 0) or 0),
        "recommendation": recommendation,
    }


def _collect_pose_long_run_recommendation(*, success: bool, metrics: dict[str, Any]) -> str:
    if bool(success):
        if int(metrics.get("dropped_post_motion_telemetry_samples", 0) or 0) > 0:
            return "completed_with_dropped_samples"
        return "completed"
    failure_context = dict(metrics.get("failure_context", {}) or {})
    failure_category = str(
        failure_context.get("failure_category", metrics.get("failure_category", "")) or ""
    ).strip()
    if failure_category in {"servo_telemetry_packet_error", "servo_telemetry_missing", "simple_experiment_motion_rejected"}:
        return "resume_after_transport_check"
    if failure_category == "write_goal_packet_error":
        return "resume_after_goal_write_path_check"
    return "investigate_failure_before_resume"


def _render_collect_pose_dataset_quality_summary(quality: dict[str, Any]) -> str:
    max_by_servo = dict(quality.get("max_abs_current_ma_by_servo", {}) or {})
    max_by_servo_text = ", ".join(f"{sid}:{int(val)}" for sid, val in sorted(max_by_servo.items())) if max_by_servo else "n/a"
    p95_by_servo = dict(quality.get("p95_abs_current_ma_by_servo", {}) or {})
    p95_by_servo_text = ", ".join(f"{sid}:{float(val):.1f}" for sid, val in sorted(p95_by_servo.items())) if p95_by_servo else "n/a"
    voltage_min = dict(quality.get("input_voltage_min_mv_by_servo", {}) or {})
    voltage_min_text = ", ".join(f"{sid}:{int(val)}" for sid, val in sorted(voltage_min.items())) if voltage_min else "n/a"
    lines = [
        "Collect-Pose Dataset Quality Summary",
        f"Accepted samples: {quality.get('accepted_sample_count')}",
        f"Rejected samples: {quality.get('rejected_sample_count')}",
        f"Failures: {quality.get('failure_count')}",
        f"Dropped post-motion samples: {quality.get('dropped_post_motion_telemetry_samples')}",
        f"Dropped pre-motion samples: {quality.get('dropped_pre_motion_telemetry_samples')}",
        f"Next command index to resume: {quality.get('next_command_index_to_resume')}",
        f"Recovered packet errors: {quality.get('recovered_packet_error_count')}",
        f"Unrecovered packet errors: {quality.get('unrecovered_packet_error_count')}",
        f"Transport bursts: {quality.get('transport_burst_count')}",
        f"Telemetry retries: {quality.get('servo_telemetry_retry_count')}",
        f"Write-goal packet errors: {quality.get('write_goal_packet_error_count')}",
        f"Tracker stale count: {quality.get('tracker_stale_count')}",
        f"Max abs current (mA): {quality.get('max_abs_current_ma')}",
        f"Max abs current by servo (mA): {max_by_servo_text}",
        f"P95 abs current by servo (mA): {p95_by_servo_text}",
        f"Input voltage minimum by servo (mV): {voltage_min_text}",
        f"Transient current spikes by servo: {quality.get('transient_current_spike_count_by_servo')}",
        f"Sustained current exceedances by servo: {quality.get('sustained_current_exceedance_count_by_servo')}",
        f"Transient current spike dropped samples: {quality.get('transient_current_spike_drop_count')}",
        f"Model training validity status: {quality.get('model_training_validity_status')}",
        f"Model training validity reason: {quality.get('model_training_validity_reason')}",
        f"Dropped rows excluded from training: {quality.get('dropped_samples_excluded_from_training')}",
        f"Accepted rows complete: {quality.get('accepted_rows_complete')}",
        (
            "Training row counts: "
            f"accepted_workspace={quality.get('accepted_workspace_sample_count')}, "
            f"export={quality.get('modeling_export_row_count')}, "
            f"legacy={quality.get('modeling_legacy_row_count')}, "
            f"accepted_training={quality.get('accepted_training_row_count')}, "
            f"complete_training={quality.get('complete_training_row_count')}, "
            f"target_complete_training={quality.get('target_valid_sample_count')}, "
            f"remaining_complete_training={quality.get('remaining_complete_training_rows')}"
        ),
        f"Accepted non-training rows: {quality.get('non_training_accepted_row_count')}",
        f"Incomplete accepted workspace rows: {quality.get('incomplete_accepted_workspace_row_count')}",
        f"Dropped/quarantined sample count: {quality.get('dropped_quarantined_sample_count')}",
        f"Recommendation: {quality.get('recommendation')}",
    ]
    warnings = [str(value) for value in list(quality.get("model_training_warnings", []) or []) if str(value)]
    if warnings:
        lines.append("Model training warnings:")
        for warning in warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines).strip() + "\n"


def _range_by_index(
    values: list[list[float]],
    *,
    prefix: str,
    labels: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    if not values:
        return {}
    width = max(len(row) for row in values)
    result: dict[str, dict[str, float]] = {}
    for index in range(width):
        series = [float(row[index]) for row in values if len(row) > index]
        if not series:
            continue
        key = labels[index] if labels is not None and index < len(labels) else f"{prefix}_{index + 1}"
        result[key] = {"min": min(series), "max": max(series), "span": max(series) - min(series)}
    return result


def _sample_rate_stats(deltas: list[float]) -> dict[str, Any]:
    if not deltas:
        return {"sample_count": 0, "mean_dt_s": None, "min_dt_s": None, "max_dt_s": None, "mean_hz": None}
    mean_dt = sum(deltas) / len(deltas)
    return {
        "sample_count": int(len(deltas) + 1),
        "mean_dt_s": float(mean_dt),
        "min_dt_s": float(min(deltas)),
        "max_dt_s": float(max(deltas)),
        "mean_hz": (1.0 / mean_dt if mean_dt > 0.0 else None),
    }


def _collect_pose_current_values(samples: list[ExperimentTimeseriesSample]) -> list[int]:
    currents: list[int] = []
    for sample in samples:
        for key in ("servo_feedback_at_command", "servo_feedback_at_capture"):
            feedback = dict(sample.extra.get(key, {}) or {})
            for item in feedback.values():
                if not isinstance(item, dict):
                    continue
                current = item.get("present_current_ma")
                if current is not None:
                    currents.append(int(current))
    return currents


def _collect_pose_current_summary(samples: list[ExperimentTimeseriesSample]) -> dict[str, Any]:
    abs_series_by_servo: dict[str, list[float]] = {}
    voltage_series_by_servo: dict[str, list[int]] = {}
    max_abs_by_servo: dict[str, int] = {}
    peak_info_by_servo: dict[str, dict[str, Any]] = {}
    current_voltage_pairs_by_servo: dict[str, list[tuple[int, int]]] = {}
    global_max_abs: int | None = None
    for sequence_index, sample in enumerate(samples):
        for source in ("servo_feedback_at_command", "servo_feedback_at_capture"):
            feedback = dict(sample.extra.get(source, {}) or {})
            for servo_id, item in feedback.items():
                if not isinstance(item, dict):
                    continue
                current = item.get("present_current_ma")
                voltage_mv = item.get("present_voltage_mv")
                if current is None:
                    continue
                sid = str(servo_id)
                abs_current = abs(int(current))
                abs_series_by_servo.setdefault(sid, []).append(float(abs_current))
                if voltage_mv is not None:
                    voltage_series_by_servo.setdefault(sid, []).append(int(voltage_mv))
                    current_voltage_pairs_by_servo.setdefault(sid, []).append((int(abs_current), int(voltage_mv)))
                if global_max_abs is None or int(abs_current) > int(global_max_abs):
                    global_max_abs = int(abs_current)
                if sid not in max_abs_by_servo or int(abs_current) > int(max_abs_by_servo[sid]):
                    max_abs_by_servo[sid] = int(abs_current)
                    peak_info_by_servo[sid] = {
                        "sequence_index": int(sequence_index),
                        "step_index": int(sample.step_index),
                        "sample_index": int(sample.sample_index),
                        "phase": str(sample.phase or ""),
                        "source": str(source),
                        "abs_current_ma": int(abs_current),
                        "command_at_peak_current_cm": list(sample.extra.get("resolved_cable_command_cm", []) or []),
                        "voltage_at_peak_mv": (int(voltage_mv) if voltage_mv is not None else None),
                        "resolved_cable_command_cm": list(sample.extra.get("resolved_cable_command_cm", []) or []),
                        "resolved_pair_command_cm": list(sample.extra.get("resolved_pair_command_cm", []) or []),
                    }
    mean_abs_by_servo: dict[str, float] = {}
    p95_abs_by_servo: dict[str, float] = {}
    voltage_min_by_servo: dict[str, int] = {}
    current_voltage_notes_by_servo: dict[str, str] = {}
    for sid, series in abs_series_by_servo.items():
        if series:
            mean_abs_by_servo[str(sid)] = float(sum(series) / len(series))
            sorted_series = sorted(float(value) for value in series)
            rank = max(0, min(len(sorted_series) - 1, int(math.ceil(0.95 * len(sorted_series))) - 1))
            p95_abs_by_servo[str(sid)] = float(sorted_series[rank])
    for sid, series in voltage_series_by_servo.items():
        if series:
            voltage_min_by_servo[str(sid)] = int(min(series))
    for sid, pairs in current_voltage_pairs_by_servo.items():
        if len(pairs) < 3:
            continue
        highs = [voltage for current, voltage in pairs if current >= 700]
        lows = [voltage for current, voltage in pairs if current <= 250]
        if highs and lows:
            high_mean = float(sum(highs) / len(highs))
            low_mean = float(sum(lows) / len(lows))
            if high_mean + 50.0 < low_mean:
                current_voltage_notes_by_servo[str(sid)] = (
                    f"voltage droop likely under high load (mean_high={high_mean:.0f} mV, mean_low={low_mean:.0f} mV)"
                )
            else:
                current_voltage_notes_by_servo[str(sid)] = (
                    f"no clear voltage droop trend (mean_high={high_mean:.0f} mV, mean_low={low_mean:.0f} mV)"
                )
    return {
        "max_abs_current_ma": int(global_max_abs) if global_max_abs is not None else None,
        "max_abs_current_ma_by_servo": {str(sid): int(value) for sid, value in sorted(max_abs_by_servo.items())},
        "mean_abs_current_ma_by_servo": {str(sid): float(value) for sid, value in sorted(mean_abs_by_servo.items())},
        "p95_abs_current_ma_by_servo": {str(sid): float(value) for sid, value in sorted(p95_abs_by_servo.items())},
        "input_voltage_min_mv_by_servo": {str(sid): int(value) for sid, value in sorted(voltage_min_by_servo.items())},
        "current_voltage_correlation_notes_by_servo": {
            str(sid): str(value) for sid, value in sorted(current_voltage_notes_by_servo.items())
        },
        "peak_current_sample_by_servo": {str(sid): dict(info) for sid, info in sorted(peak_info_by_servo.items())},
    }


def _load_collect_pose_neutral_ticks(session: ExperimentSession, *, servo_ids: list[int]) -> list[int]:
    reference = session.context.servo_service.resolve_startup_reference_ticks(list(servo_ids))
    return [int(reference.ticks_by_servo[servo_id]) for servo_id in servo_ids if servo_id in reference.ticks_by_servo]


def _record_collect_pose_servo_precheck_failure(
    session: ExperimentSession,
    *,
    failed_servo_id: int,
    assessment: Any,
    fresh_precheck_read_attempted: bool,
) -> None:
    tel = assessment.telemetry
    bus = session.context.servo_service.bus_ownership_status()
    missing: list[str] = []
    if getattr(tel, "present_position", None) is None:
        missing.append("present_position")
    if getattr(tel, "operating_mode", None) is None:
        missing.append("operating_mode")
    if getattr(tel, "hardware_error_code", None) is None and getattr(tel, "hardware_error", None):
        missing.append("hardware_error_status")
    session.set_metric(
        "precheck_failure_context",
        {
            "experiment": "collect_pose_command_dataset",
            "failed_servo_id": int(failed_servo_id),
            "missing_fields": missing,
            "telemetry_read_source": getattr(tel, "read_source", None),
            "telemetry_bus_owner_during_read": getattr(tel, "bus_owner", None),
            "fresh_precheck_read_attempted": bool(fresh_precheck_read_attempted),
            "exclusive_bus_owner_snapshot": bus.owner,
            "exclusive_bus_active": bool(bus.active),
            "assessment_reason": assessment.reason,
            "assessment_blocking_reasons": list(assessment.blocking_reasons),
        },
    )


def _precheck_collect_pose_command_dataset(
    *,
    session: ExperimentSession,
    config: CollectPoseCommandDatasetConfig,
    servo_ids: list[int],
    neutral_ticks: list[int],
) -> None:
    tracking_service = session.context.tracking_service
    servo_only_mode = _collect_pose_servo_only_test_mode(config=config, tracking_service=tracking_service)
    if tracking_service is None and not servo_only_mode:
        raise RuntimeError(
            "Tracker precheck failed: Motor Babble modeling dataset collection requires tracking_service."
        )
    parallel_single = _collect_pose_parallel_single_mode(session)
    expected_count = 8 if parallel_single else 4
    if len(servo_ids) != expected_count:
        raise RuntimeError(
            "Motor Babble modeling dataset collection requires "
            f"{'8 mirrored servos in parallel_single mode' if parallel_single else 'exactly 4 active segment servos'}; found {servo_ids}."
        )
    if not config.dry_run and not session.context.servo_service.is_connected:
        raise RuntimeError("Live Motor Babble modeling dataset collection requires a connected servo service.")
    if not config.dry_run and len(neutral_ticks) != len(servo_ids):
        raise RuntimeError("Live modeling data collection requires neutral setpoints for all 4 servos.")
    if servo_only_mode:
        session.add_warning(
            "Tracker is not connected. This run is allowed as a servo-only hardware test. "
            "Tip position, robot-frame pose, modeling labels, and thesis repeatability metrics will not be produced."
        )
    snapshot = tracking_service.get_snapshot() if tracking_service is not None else None
    if not config.dry_run and bool(session.context.settings.runtime.mock_mode) and not servo_only_mode:
        raise RuntimeError("Thesis-grade modeling datasets require live runtime mode. Disable mock mode or switch to dry_run.")
    if (not config.dry_run) and not servo_only_mode and snapshot.canonical_state != "streaming_healthy":
        raise RuntimeError(
            "Tracker precheck failed: Tracker must be streaming healthy before modeling dataset collection; "
            f"current state is {snapshot.canonical_state}."
        )
    if servo_only_mode:
        runtime_tip_policy = None
    else:
        runtime_tip_policy = evaluate_runtime_tip_trust(
            snapshot=snapshot,
            workflow=WORKFLOW_MODELING_DATASET,
            allow_lower_trust=bool(config.allow_lower_trust_runtime_tip),
        )
    if servo_only_mode and bool(config.export_legacy_dat):
        session.add_warning("Servo-only no-tracker collection will disable legacy .dat export because no valid pose labels exist.")
    if servo_only_mode:
        calibration_summary = session.context.servo_service.get_calibration_summary()
        pretension_source = session.context.servo_service.pretension_source_summary(list(servo_ids))
        if not config.dry_run:
            if not calibration_summary.exists or not calibration_summary.compatible:
                raise RuntimeError(f"Servo calibration artifact is not ready: {calibration_summary.message}")
            _check_collect_pose_selected_segment_truth(
                session=session,
                calibration_summary=calibration_summary,
                servo_ids=servo_ids,
                trusted_run=False,
            )
            if (not pretension_source.accepted or not pretension_source.usable) and not config.allow_lower_trust_pretension:
                raise RuntimeError(
                    "An accepted pretension/startup artifact is required before servo-only motion testing. "
                    + pretension_source.message
                )
            assessments = session.context.servo_service.coordinated_motion_precheck_assessments(
                list(servo_ids),
                owner="collect_pose_command_dataset_precheck",
                reason="coordinated motion readiness before modeling dataset collection (servo-only path)",
            )
            for servo_id in servo_ids:
                assessment = assessments[int(servo_id)]
                if not assessment.ready:
                    _record_collect_pose_servo_precheck_failure(
                        session,
                        failed_servo_id=int(servo_id),
                        assessment=assessment,
                        fresh_precheck_read_attempted=True,
                    )
                    raise RuntimeError(
                        f"Servo precheck failed: Servo {servo_id} is not ready for coordinated motion: {assessment.reason}"
                    )
        if config.dataset_mode not in {"workspace_coverage", "hysteresis_path_dependence", "repeatability_linked", "angular_test_mesh"}:
            if not config.command_points and not bool(config.legacy_schedule_override):
                raise RuntimeError(f"Unsupported modeling dataset mode: {config.dataset_mode}")
        return
    if config.require_robot_frame_tip and snapshot.registration_state != "loaded":
        raise RuntimeError(
            "Tracker precheck failed: Accepted base registration must be loaded before modeling dataset collection."
        )
    if config.require_robot_frame_tip:
        if not runtime_tip_policy.allowed_for_workflow:
            raise RuntimeError(
                "Tracker precheck failed: Motor Babble requires the shared runtime tip policy to allow modeling data. "
                f"Current mode={runtime_tip_policy.mode}, trust={runtime_tip_policy.trust_label}, "
                f"reasons={runtime_tip_policy.reasons or ['policy_not_allowed']}. "
                "Enable the lower-trust override only when you intend a lower-trust modeling dataset."
            )
        if snapshot.tip_pose_status not in {"ok", "coil_as_tip"} or snapshot.T_robot_tip is None:
            raise RuntimeError(
                "Tracker precheck failed: Live robot-frame tip pose must be active before modeling dataset collection; "
                f"tip pose status is {snapshot.tip_pose_status}."
            )
    gate = _modeling_tracker_gate_status(
        snapshot=snapshot,
        tool_id=str(config.tool_id or "0A"),
        max_tracker_age_s=float(config.max_tracker_age_s),
        require_robot_frame_tip=bool(config.require_robot_frame_tip),
        allow_mock_state=bool(config.dry_run),
        allow_lower_trust_runtime_tip=bool(config.allow_lower_trust_runtime_tip),
    )
    if not gate["accepted"]:
        session.add_warning(
            "Tracker capture gate is not ready at precheck and may reject captures during the run: "
            f"{gate['reason']}."
        )
    calibration_summary = session.context.servo_service.get_calibration_summary()
    pretension_source = session.context.servo_service.pretension_source_summary(list(servo_ids))
    if not config.dry_run:
        if not calibration_summary.exists or not calibration_summary.compatible:
            raise RuntimeError(f"Servo calibration artifact is not ready: {calibration_summary.message}")
        _check_collect_pose_selected_segment_truth(
            session=session,
            calibration_summary=calibration_summary,
            servo_ids=servo_ids,
            trusted_run=not bool(config.allow_lower_trust_pretension),
        )
        if (not pretension_source.accepted or not pretension_source.usable) and not config.allow_lower_trust_pretension:
            raise RuntimeError(
                "An accepted pretension/startup artifact is required before modeling dataset collection. "
                + pretension_source.message
            )
    if not config.dry_run:
        assessments = session.context.servo_service.coordinated_motion_precheck_assessments(
            list(servo_ids),
            owner="collect_pose_command_dataset_precheck",
            reason="coordinated motion readiness before modeling dataset collection",
        )
        for servo_id in servo_ids:
            assessment = assessments[int(servo_id)]
            if not assessment.ready:
                _record_collect_pose_servo_precheck_failure(
                    session,
                    failed_servo_id=int(servo_id),
                    assessment=assessment,
                    fresh_precheck_read_attempted=True,
                )
                raise RuntimeError(
                    f"Servo precheck failed: Servo {servo_id} is not ready for coordinated motion: {assessment.reason}"
                )
    if config.dataset_mode not in {"workspace_coverage", "hysteresis_path_dependence", "repeatability_linked", "angular_test_mesh"}:
        if not config.command_points and not bool(config.legacy_schedule_override):
            raise RuntimeError(f"Unsupported modeling dataset mode: {config.dataset_mode}")


def _collect_pose_servo_only_test_mode(*, config: CollectPoseCommandDatasetConfig, tracking_service) -> bool:
    explicit_mode = str(getattr(config, "run_trust_mode", "") or "").strip().lower()
    explicit_no_tracker = bool(getattr(config, "allow_no_tracker_test_run", False)) or explicit_mode == "servo_only"
    if not explicit_no_tracker:
        return False
    snapshot = _optional_tracking_snapshot(tracking_service)
    if snapshot is None:
        return True
    return str(getattr(snapshot, "canonical_state", "") or "") not in {
        "mock",
        "connected",
        "streaming_healthy",
        "streaming_degraded",
    }


def _check_collect_pose_selected_segment_truth(
    *,
    session: ExperimentSession,
    calibration_summary,
    servo_ids: list[int],
    trusted_run: bool,
) -> None:
    context = session.context.settings.robot.operating_context()
    if context.operating_mode != "single_segment":
        return
    readiness = evaluate_selected_segment_readiness(
        operating_mode=context.operating_mode,
        active_segment_key=context.active_segment_key,
        active_segment_label=context.active_segment_label,
        expected_servo_ids=[int(value) for value in servo_ids],
        calibration_summary=calibration_summary,
        mock_mode=bool(session.context.settings.runtime.mock_mode),
        servo_connected=bool(getattr(session.context.servo_service, "is_connected", False)),
    )
    if not readiness.neutral_safe_calibration.ready:
        if trusted_run:
            raise RuntimeError(readiness.neutral_safe_calibration.message)
        session.add_warning(
            readiness.neutral_safe_calibration.message
            + " This run is explicit servo-only/lower-trust debug output and is not training-ready."
        )
    if not readiness.startup_pretension.ready:
        if trusted_run:
            raise RuntimeError(readiness.startup_pretension.message)
        session.add_warning(
            readiness.startup_pretension.message
            + " This run is explicit servo-only/lower-trust debug output and is not training-ready."
        )


def _collect_pose_parallel_single_mode(session: ExperimentSession) -> bool:
    return session.context.settings.robot.operating_context().operating_mode == "parallel_single"


def _mirror_parallel_single_displacements(
    displacements_cm: list[float],
    *,
    servo_ids: list[int],
    context,
) -> list[float]:
    requested = [float(value) for value in displacements_cm]
    if len(requested) != 4:
        return list(requested)
    mirror_pairs = {int(key): int(value) for key, value in dict(context.mirror_pairs or {}).items()}
    if not mirror_pairs:
        return list(requested)
    base_ids = [int(value) for value in context.active_segment_servo_ids]
    displacement_by_servo = {int(servo_id): float(value) for servo_id, value in zip(base_ids, requested)}
    for source, mirror in mirror_pairs.items():
        if int(source) in displacement_by_servo:
            displacement_by_servo[int(mirror)] = float(displacement_by_servo[int(source)])
    return [float(displacement_by_servo.get(int(servo_id), 0.0)) for servo_id in servo_ids]


def _optional_tracking_snapshot(tracking_service):
    if tracking_service is None:
        return None
    try:
        snapshot_reader = getattr(tracking_service, "peek_snapshot", None)
        return snapshot_reader() if callable(snapshot_reader) else tracking_service.get_snapshot()
    except Exception:
        return None


def _pretension_current_only_explicit(config: PretensionValidationExperimentConfig) -> bool:
    explicit_mode = str(getattr(config, "run_trust_mode", "") or "").strip().lower()
    return bool(
        getattr(config, "allow_no_tracker_test_run", False)
        or getattr(config, "allow_current_only_when_tracker_missing", False)
        or explicit_mode in {"current_only", "servo_only", "lower_trust"}
    )


def _summarize_pretension_population(
    records: list[dict[str, Any]],
    *,
    servo_ids: list[int],
    tip_target_xy_mm: list[float],
) -> dict[str, Any]:
    """Reduce a list of pretension end-states (manual or algorithm) to a single
    population summary for the comparison report.

    Each record must expose ``positions_by_servo``, ``currents_ma_by_servo``,
    and optionally ``tip_xy_mm``. Spreads are reported as standard deviation
    across records ("repeatability") and within-record max-minus-min across
    servos ("equality")."""

    def _per_servo_values(field_name: str) -> dict[int, list[float]]:
        per_servo: dict[int, list[float]] = {int(sid): [] for sid in servo_ids}
        for record in records:
            block = record.get(field_name) or {}
            for sid in servo_ids:
                value = block.get(int(sid))
                if value is None:
                    value = block.get(str(int(sid)))
                if value is None:
                    continue
                per_servo[int(sid)].append(float(value))
        return per_servo

    def _std(values: list[float]) -> float | None:
        if not values:
            return None
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))

    positions = _per_servo_values("positions_by_servo")
    currents = _per_servo_values("currents_ma_by_servo")
    per_run_current_spread_ma: list[float] = []
    per_run_position_spread_ticks: list[float] = []
    for record in records:
        run_currents = record.get("currents_ma_by_servo") or {}
        run_positions = record.get("positions_by_servo") or {}
        c_values = [
            float(run_currents.get(int(sid), run_currents.get(str(int(sid))) or 0.0))
            for sid in servo_ids
            if (run_currents.get(int(sid)) is not None or run_currents.get(str(int(sid))) is not None)
        ]
        if len(c_values) == len(servo_ids):
            per_run_current_spread_ma.append(float(max(c_values) - min(c_values)))
        p_values = [
            float(run_positions.get(int(sid), run_positions.get(str(int(sid))) or 0.0))
            for sid in servo_ids
            if (run_positions.get(int(sid)) is not None or run_positions.get(str(int(sid))) is not None)
        ]
        if len(p_values) == len(servo_ids):
            per_run_position_spread_ticks.append(float(max(p_values) - min(p_values)))

    tip_points = [
        (float(record["tip_xy_mm"][0]), float(record["tip_xy_mm"][1]))
        for record in records
        if isinstance(record.get("tip_xy_mm"), (list, tuple)) and len(record["tip_xy_mm"]) >= 2
    ]
    target_xy = (
        float(tip_target_xy_mm[0]) if tip_target_xy_mm else 0.0,
        float(tip_target_xy_mm[1]) if len(tip_target_xy_mm) > 1 else 0.0,
    )
    tip_errors_mm = [
        math.sqrt((x - target_xy[0]) ** 2 + (y - target_xy[1]) ** 2)
        for x, y in tip_points
    ]
    tip_xy_centroid = None
    if tip_points:
        cx = sum(x for x, _ in tip_points) / len(tip_points)
        cy = sum(y for _, y in tip_points) / len(tip_points)
        tip_xy_centroid = [float(cx), float(cy)]
    tip_radial_dispersion_mm = None
    if tip_xy_centroid is not None and len(tip_points) >= 2:
        squared = [
            (x - tip_xy_centroid[0]) ** 2 + (y - tip_xy_centroid[1]) ** 2
            for x, y in tip_points
        ]
        tip_radial_dispersion_mm = float(math.sqrt(sum(squared) / len(squared)))

    return {
        "record_count": int(len(records)),
        "servo_ids": [int(sid) for sid in servo_ids],
        # Equality across servos within each record (max-min of the 4 servos).
        # The MEAN of this metric tells you how unequal the 4 servos are on a
        # typical run; the STD tells you whether that inequality is consistent.
        "per_run_current_spread_ma": {
            "mean": (sum(per_run_current_spread_ma) / len(per_run_current_spread_ma))
            if per_run_current_spread_ma
            else None,
            "std": _std(per_run_current_spread_ma),
            "min": (min(per_run_current_spread_ma) if per_run_current_spread_ma else None),
            "max": (max(per_run_current_spread_ma) if per_run_current_spread_ma else None),
            "count": len(per_run_current_spread_ma),
        },
        "per_run_position_spread_ticks": {
            "mean": (sum(per_run_position_spread_ticks) / len(per_run_position_spread_ticks))
            if per_run_position_spread_ticks
            else None,
            "std": _std(per_run_position_spread_ticks),
            "min": (min(per_run_position_spread_ticks) if per_run_position_spread_ticks else None),
            "max": (max(per_run_position_spread_ticks) if per_run_position_spread_ticks else None),
            "count": len(per_run_position_spread_ticks),
        },
        # Repeatability across records per servo. The std of (servo i current)
        # across records is the run-to-run consistency for that servo.
        "per_servo_current_std_ma": {
            int(sid): _std(values) for sid, values in currents.items()
        },
        "per_servo_position_std_ticks": {
            int(sid): _std(values) for sid, values in positions.items()
        },
        "per_servo_current_mean_ma": {
            int(sid): (sum(values) / len(values)) if values else None
            for sid, values in currents.items()
        },
        "per_servo_position_mean_ticks": {
            int(sid): (sum(values) / len(values)) if values else None
            for sid, values in positions.items()
        },
        # Tip centering / repeatability.
        "tip_xy_points_mm": [list(point) for point in tip_points],
        "tip_xy_centroid_mm": tip_xy_centroid,
        "tip_radial_dispersion_mm": tip_radial_dispersion_mm,
        "tip_xy_error_to_target_mm": {
            "mean": (sum(tip_errors_mm) / len(tip_errors_mm)) if tip_errors_mm else None,
            "std": _std(tip_errors_mm),
            "min": (min(tip_errors_mm) if tip_errors_mm else None),
            "max": (max(tip_errors_mm) if tip_errors_mm else None),
            "count": len(tip_errors_mm),
        },
    }


def _build_pretension_comparison_report(
    *,
    algorithm_run_rows: list[dict[str, Any]],
    manual_baseline_records: list[dict[str, Any]],
    servo_ids: list[int],
    tip_target_xy_mm: list[float],
    target_load_band_ma: tuple[float, float],
) -> dict[str, Any]:
    """Build the algorithm-vs-manual comparison summary.

    Each algorithm run row carries its final state under
    ``final_positions_by_servo``, ``final_currents_ma_by_servo``, and
    ``final_tip_xy_mm`` (set by the staged-pretension execute loop). We
    normalize those into the same shape that manual baseline records use,
    then summarize both populations and compute per-metric deltas."""
    normalized_algorithm = []
    for row in algorithm_run_rows:
        positions = row.get("positions_by_servo") or row.get("final_position_ticks_by_servo") or {}
        currents = row.get("currents_ma_by_servo") or row.get("final_current_ma_by_servo") or {}
        normalized_algorithm.append(
            {
                "run_label": str(row.get("run_label", "")),
                "positions_by_servo": dict(positions),
                "currents_ma_by_servo": dict(currents),
                "tip_xy_mm": row.get("final_tip_xy_mm"),
                "tip_xyz_mm": row.get("final_tip_xyz_mm"),
                "accepted": bool(row.get("accepted")),
                "stop_reason": row.get("stop_reason"),
                "tip_centering_variant": row.get("variant"),
            }
        )

    algorithm_summary = _summarize_pretension_population(
        normalized_algorithm,
        servo_ids=servo_ids,
        tip_target_xy_mm=tip_target_xy_mm,
    )
    manual_summary = _summarize_pretension_population(
        manual_baseline_records,
        servo_ids=servo_ids,
        tip_target_xy_mm=tip_target_xy_mm,
    )

    def _delta_field(
        algo_block: dict[str, Any],
        manual_block: dict[str, Any],
        field: str,
    ) -> dict[str, Any]:
        algo_value = algo_block.get(field)
        manual_value = manual_block.get(field)
        algorithm_better_when_smaller = field in {"std", "max", "mean"}
        delta = (
            float(algo_value) - float(manual_value)
            if algo_value is not None and manual_value is not None
            else None
        )
        improved = (
            (bool(delta < 0.0) if algorithm_better_when_smaller else bool(delta > 0.0))
            if delta is not None
            else None
        )
        return {
            "algorithm": algo_value,
            "manual": manual_value,
            "delta_algorithm_minus_manual": delta,
            "algorithm_better": improved,
        }

    def _compare_block(
        algo_block: dict[str, Any] | None, manual_block: dict[str, Any] | None
    ) -> dict[str, Any]:
        out = {}
        algo = algo_block or {}
        manual = manual_block or {}
        for field in ("mean", "std", "min", "max"):
            out[field] = _delta_field(algo, manual, field)
        return out

    comparison = {
        "per_run_current_spread_ma": _compare_block(
            algorithm_summary.get("per_run_current_spread_ma"),
            manual_summary.get("per_run_current_spread_ma"),
        ),
        "per_run_position_spread_ticks": _compare_block(
            algorithm_summary.get("per_run_position_spread_ticks"),
            manual_summary.get("per_run_position_spread_ticks"),
        ),
        "tip_xy_error_to_target_mm": _compare_block(
            algorithm_summary.get("tip_xy_error_to_target_mm"),
            manual_summary.get("tip_xy_error_to_target_mm"),
        ),
        "tip_radial_dispersion_mm": {
            "algorithm": algorithm_summary.get("tip_radial_dispersion_mm"),
            "manual": manual_summary.get("tip_radial_dispersion_mm"),
            "delta_algorithm_minus_manual": (
                float(algorithm_summary.get("tip_radial_dispersion_mm")) - float(manual_summary.get("tip_radial_dispersion_mm"))
                if algorithm_summary.get("tip_radial_dispersion_mm") is not None
                and manual_summary.get("tip_radial_dispersion_mm") is not None
                else None
            ),
            "algorithm_better": (
                (
                    bool(
                        float(algorithm_summary.get("tip_radial_dispersion_mm"))
                        < float(manual_summary.get("tip_radial_dispersion_mm"))
                    )
                )
                if algorithm_summary.get("tip_radial_dispersion_mm") is not None
                and manual_summary.get("tip_radial_dispersion_mm") is not None
                else None
            ),
        },
    }

    # A simple per-metric verdict: how many comparisons favored the algorithm?
    wins = 0
    losses = 0
    ties = 0
    for block in comparison.values():
        for sub in (block.values() if isinstance(block, dict) else []):
            if isinstance(sub, dict) and "algorithm_better" in sub:
                value = sub.get("algorithm_better")
                if value is True:
                    wins += 1
                elif value is False:
                    losses += 1
                else:
                    ties += 1

    return {
        "schema_version": "1.0",
        "comparison_kind": "algorithm_vs_manual_pretension",
        "tip_target_xy_mm": [float(v) for v in tip_target_xy_mm],
        "target_load_band_ma": [float(target_load_band_ma[0]), float(target_load_band_ma[1])],
        "algorithm_population_summary": algorithm_summary,
        "manual_population_summary": manual_summary,
        "comparison": comparison,
        "algorithm_wins": int(wins),
        "manual_wins": int(losses),
        "ties_or_missing": int(ties),
    }


def _collect_pose_pair_limits(
    *,
    session: ExperimentSession,
    config: CollectPoseCommandDatasetConfig,
    servo_ids: list[int],
    neutral_ticks: list[int],
) -> dict[str, list[tuple[float, float]] | str | bool | float]:
    default_bounds = [
        (-abs(float(config.workspace_amplitude_cm)), abs(float(config.workspace_amplitude_cm))),
        (-abs(float(config.workspace_amplitude_cm)), abs(float(config.workspace_amplitude_cm))),
    ]
    if config.dry_run or not session.context.servo_service.is_connected or len(neutral_ticks) != len(servo_ids):
        return {
            "available": False,
            "mode": "configured_amplitude",
            "pair_bounds_cm": default_bounds,
            "envelope_utilization": float(config.envelope_utilization),
        }
    characterization = session.context.servo_service.characterize_single_segment_motion(
        servo_ids=list(servo_ids),
        neutral_ticks_by_id={int(servo_id): int(tick) for servo_id, tick in zip(servo_ids, neutral_ticks)},
    )
    if not characterization.available:
        return {
            "available": False,
            "mode": "configured_amplitude",
            "pair_bounds_cm": default_bounds,
            "envelope_utilization": float(config.envelope_utilization),
            "message": characterization.message,
        }
    pair_bounds: list[tuple[float, float]] = []
    for pair_key in [f"{servo_ids[0]}/{servo_ids[2]}", f"{servo_ids[1]}/{servo_ids[3]}"]:
        row = dict(characterization.pair_limits.get(pair_key, {}) or {})
        negative_cm = abs(float(row.get("negative_cm", config.workspace_amplitude_cm) or config.workspace_amplitude_cm))
        positive_cm = abs(float(row.get("positive_cm", config.workspace_amplitude_cm) or config.workspace_amplitude_cm))
        usable_negative = min(negative_cm, abs(float(config.workspace_amplitude_cm))) * float(config.envelope_utilization)
        usable_positive = min(positive_cm, abs(float(config.workspace_amplitude_cm))) * float(config.envelope_utilization)
        pair_bounds.append((-usable_negative, usable_positive))
    return {
        "available": True,
        "mode": "hardware_characterized",
        "pair_bounds_cm": pair_bounds,
        "envelope_utilization": float(config.envelope_utilization),
        "message": characterization.message,
        "characterization": characterization.pair_limits,
    }


def _build_collect_pose_command_steps(
    *,
    config: CollectPoseCommandDatasetConfig,
    pair_limits: dict[str, Any],
) -> list[ModelingCommandStep]:
    if config.command_points:
        steps: list[ModelingCommandStep] = []
        for index, point in enumerate(config.command_points):
            cable_command = [float(value) for value in point.get("tendon_displacement_cm", [])]
            steps.append(
                ModelingCommandStep(
                    index=int(point.get("index", index)),
                    phase="explicit_command",
                    label=str(point.get("label", f"explicit_{index:04d}") or f"explicit_{index:04d}"),
                    pair_command_cm=_pair_command_from_cable_deltas(cable_command),
                    cable_command_cm=cable_command,
                    settle_time_s=float(point.get("settle_time_s") if point.get("settle_time_s") not in (None, "") else config.settle_time_s),
                    metadata={"source": "explicit_command_points"},
                )
            )
        return steps
    if bool(config.legacy_schedule_override):
        return _build_collect_pose_schedule_override_steps(config)
    if config.dataset_mode == "workspace_coverage":
        return _build_collect_pose_workspace_steps(config=config, pair_limits=pair_limits)
    if config.dataset_mode == "hysteresis_path_dependence":
        return _build_collect_pose_hysteresis_steps(config=config, pair_limits=pair_limits)
    if config.dataset_mode == "angular_test_mesh":
        return _build_collect_pose_angular_test_mesh_steps(config=config, pair_limits=pair_limits)
    return _build_collect_pose_repeatability_linked_steps(config=config, pair_limits=pair_limits)


def _build_collect_pose_angular_test_mesh_steps(
    *,
    config: CollectPoseCommandDatasetConfig,
    pair_limits: dict[str, Any],
) -> list[ModelingCommandStep]:
    """Wolfe-style angular test mesh: θ × φ grid of constant-curvature targets.

    Wolfe MS thesis §3.2.3 p85 evaluates models on a 12 × 24 grid where the cable
    displacements come from pure-kinematics CC inverse:

        Δℓ_i = -(d_{i,x} cos φ + d_{i,y} sin φ) · θ

    For a 4-cable rig arranged at (±r, 0), (0, ±r), the differential pair-commands
    reduce to ``pair_x = scale · sin(θ) · cos(φ)`` / ``pair_y = scale · sin(θ) · sin(φ)``
    when ``scale`` represents the maximum per-pair displacement at θ = π/2. We use
    ``sin(θ)`` rather than θ to keep the largest commands within the existing
    workspace-amplitude bounds. The schedule is then ``theta_count × phi_count``
    targets visited in order; ``samples_per_command`` (set to 5 for Wolfe's mesh)
    captures each target multiple times.
    """
    theta_count = max(2, int(config.test_mesh_theta_count or 12))
    phi_count = max(3, int(config.test_mesh_phi_count or 24))
    amplitude_cm = float(
        config.test_mesh_amplitude_cm
        if config.test_mesh_amplitude_cm is not None
        else config.workspace_amplitude_cm
    )
    bounds = list(
        pair_limits.get(
            "pair_bounds_cm",
            [(-config.workspace_amplitude_cm, config.workspace_amplitude_cm)] * 2,
        )
        or []
    )
    pair_x_max = float(min(abs(bounds[0][0]), abs(bounds[0][1]))) if bounds else amplitude_cm
    pair_y_max = float(min(abs(bounds[1][0]), abs(bounds[1][1]))) if len(bounds) > 1 else amplitude_cm
    safe_amplitude = min(amplitude_cm, pair_x_max, pair_y_max)
    steps: list[ModelingCommandStep] = []
    index = 0
    for k in range(1, theta_count + 1):
        theta = math.pi / 2.0 * (k / float(theta_count))
        radial = safe_amplitude * math.sin(theta)
        for j in range(phi_count):
            phi = 2.0 * math.pi * (j / float(phi_count))
            pair_x = radial * math.cos(phi)
            pair_y = radial * math.sin(phi)
            steps.append(
                ModelingCommandStep(
                    index=index,
                    phase="angular_test_mesh",
                    label=f"mesh_{index:04d}_t{k:02d}_p{j:02d}",
                    pair_command_cm=[pair_x, pair_y],
                    cable_command_cm=_expand_pair_command_cm([pair_x, pair_y]),
                    settle_time_s=float(config.settle_time_s),
                    metadata={
                        "mode_family": "angular_test_mesh",
                        "theta_rad": theta,
                        "phi_rad": phi,
                        "theta_index": int(k),
                        "phi_index": int(j),
                    },
                )
            )
            index += 1
    return steps


def _build_collect_pose_schedule_override_steps(config: CollectPoseCommandDatasetConfig) -> list[ModelingCommandStep]:
    steps: list[ModelingCommandStep] = []
    for point in generate_command_schedule(config.command_schedule):
        cable_command = [float(value) for value in point.tendon_displacement_cm]
        steps.append(
            ModelingCommandStep(
                index=int(point.index),
                phase="schedule_override",
                label=str(point.label or f"schedule_{point.index:04d}"),
                pair_command_cm=_pair_command_from_cable_deltas(cable_command),
                cable_command_cm=cable_command,
                settle_time_s=float(point.settle_time_s if point.settle_time_s is not None else config.settle_time_s),
                metadata={
                    "source": "legacy_schedule_override",
                    "schedule_kind": str(config.command_schedule.kind or "unknown"),
                },
            )
        )
    return steps


def _build_collect_pose_workspace_steps(
    *,
    config: CollectPoseCommandDatasetConfig,
    pair_limits: dict[str, Any],
) -> list[ModelingCommandStep]:
    planned_commands = max(1, int(math.ceil(float(config.sample_count_target) / float(max(1, int(config.samples_per_command))))))
    bounds = list(pair_limits.get("pair_bounds_cm", [(-config.workspace_amplitude_cm, config.workspace_amplitude_cm)] * 2) or [])
    pair_points = _collect_pose_pair_sequence(
        count=planned_commands,
        bounds=bounds,
        quasi_random=bool(config.quasi_random),
        seed=int(config.random_seed),
        include_center=True,
    )
    return [
        ModelingCommandStep(
            index=index,
            phase="workspace_capture",
            label=f"workspace_{index:04d}",
            pair_command_cm=list(pair_command),
            cable_command_cm=_expand_pair_command_cm(pair_command),
            settle_time_s=float(config.settle_time_s),
            metadata={"mode_family": "workspace_coverage"},
        )
        for index, pair_command in enumerate(pair_points)
    ]


def _build_collect_pose_hysteresis_steps(
    *,
    config: CollectPoseCommandDatasetConfig,
    pair_limits: dict[str, Any],
) -> list[ModelingCommandStep]:
    bounds = list(pair_limits.get("pair_bounds_cm", [(-config.workspace_amplitude_cm, config.workspace_amplitude_cm)] * 2) or [])
    target_points = _collect_pose_pair_sequence(
        count=max(1, int(config.hysteresis_target_count)),
        bounds=bounds,
        quasi_random=bool(config.quasi_random),
        seed=int(config.random_seed),
        include_center=False,
    )
    prior_states = _collect_pose_prior_family_states(bounds=bounds, family_count=max(2, int(config.hysteresis_prior_family_count)))
    steps: list[ModelingCommandStep] = []
    step_index = 0
    previous_pair = [0.0, 0.0]
    for cycle_index in range(max(1, int(config.hysteresis_cycle_count))):
        for target_index, target_pair in enumerate(target_points):
            ordered_prior_states = list(prior_states)
            rotation = (target_index + cycle_index) % len(ordered_prior_states)
            ordered_prior_states = ordered_prior_states[rotation:] + ordered_prior_states[:rotation]
            for prior_family, prior_pair in ordered_prior_states:
                steps.append(
                    ModelingCommandStep(
                        index=step_index,
                        phase="hysteresis_prior",
                        label=f"prior_{cycle_index:02d}_{target_index:02d}_{prior_family}",
                        pair_command_cm=list(prior_pair),
                        cable_command_cm=_expand_pair_command_cm(prior_pair),
                        settle_time_s=float(config.settle_time_s),
                        block_index=cycle_index,
                        prior_family=str(prior_family),
                        previous_pair_command_cm=list(previous_pair),
                        metadata={
                            "mode_family": "hysteresis_path_dependence",
                            "target_pair_command_cm": list(target_pair),
                            "target_index": int(target_index),
                        },
                    )
                )
                step_index += 1
                steps.append(
                    ModelingCommandStep(
                        index=step_index,
                        phase="hysteresis_target",
                        label=f"target_{cycle_index:02d}_{target_index:02d}_{prior_family}",
                        pair_command_cm=list(target_pair),
                        cable_command_cm=_expand_pair_command_cm(target_pair),
                        settle_time_s=float(config.settle_time_s),
                        block_index=cycle_index,
                        prior_family=str(prior_family),
                        previous_pair_command_cm=list(prior_pair),
                        metadata={
                            "mode_family": "hysteresis_path_dependence",
                            "target_index": int(target_index),
                            "target_pair_command_cm": list(target_pair),
                        },
                    )
                )
                step_index += 1
                previous_pair = list(target_pair)
    return steps


def _build_collect_pose_repeatability_linked_steps(
    *,
    config: CollectPoseCommandDatasetConfig,
    pair_limits: dict[str, Any],
) -> list[ModelingCommandStep]:
    bounds = list(pair_limits.get("pair_bounds_cm", [(-config.workspace_amplitude_cm, config.workspace_amplitude_cm)] * 2) or [])
    block_commands = max(1, int(math.ceil(float(config.sample_count_target) / float(max(1, int(config.samples_per_command))))))
    steps: list[ModelingCommandStep] = []
    step_index = 0
    for block_index in range(max(1, int(config.repeatability_block_count))):
        block_points = _collect_pose_pair_sequence(
            count=block_commands,
            bounds=bounds,
            quasi_random=bool(config.quasi_random),
            seed=int(config.random_seed) + block_index,
            include_center=True,
        )
        for point_index, pair_command in enumerate(block_points):
            steps.append(
                ModelingCommandStep(
                    index=step_index,
                    phase="repeatability_linked_capture",
                    label=f"repeatability_block_{block_index:02d}_{point_index:04d}",
                    pair_command_cm=list(pair_command),
                    cable_command_cm=_expand_pair_command_cm(pair_command),
                    settle_time_s=float(config.settle_time_s),
                    block_index=block_index,
                    metadata={
                        "mode_family": "repeatability_linked",
                        "dataset_block_index": int(block_index),
                    },
                )
            )
            step_index += 1
    return steps


def _collect_pose_pair_sequence(
    *,
    count: int,
    bounds: list[tuple[float, float]],
    quasi_random: bool,
    seed: int,
    include_center: bool,
) -> list[list[float]]:
    planned = max(1, int(count))
    points: list[list[float]] = []
    if include_center:
        points.append([0.0, 0.0])
    remaining = max(0, planned - len(points))
    if remaining <= 0:
        return points[:planned]
    if quasi_random:
        for index in range(remaining):
            pair = [
                _collect_pose_scale_unit_interval(_halton_value(index + 1, 2), bounds[0]),
                _collect_pose_scale_unit_interval(_halton_value(index + 1, 3), bounds[1]),
            ]
            points.append(pair)
        return points[:planned]
    rng = random.Random(int(seed))
    for _ in range(remaining):
        points.append(
            [
                float(rng.uniform(bounds[0][0], bounds[0][1])),
                float(rng.uniform(bounds[1][0], bounds[1][1])),
            ]
        )
    return points[:planned]


def _collect_pose_prior_family_states(
    *,
    bounds: list[tuple[float, float]],
    family_count: int,
) -> list[tuple[str, list[float]]]:
    positive_x = [float(bounds[0][1]), 0.0]
    negative_x = [float(bounds[0][0]), 0.0]
    positive_y = [0.0, float(bounds[1][1])]
    negative_y = [0.0, float(bounds[1][0])]
    families = [
        ("tighten_x", positive_x),
        ("loosen_x", negative_x),
        ("tighten_y", positive_y),
        ("loosen_y", negative_y),
        ("center", [0.0, 0.0]),
    ]
    return families[: max(2, min(int(family_count), len(families)))]


def _collect_pose_scale_unit_interval(value: float, bounds: tuple[float, float]) -> float:
    low, high = float(bounds[0]), float(bounds[1])
    return float(low + ((high - low) * float(value)))


def _halton_value(index: int, base: int) -> float:
    result = 0.0
    fraction = 1.0 / float(base)
    current = int(index)
    while current > 0:
        current, remainder = divmod(current, int(base))
        result += float(remainder) * fraction
        fraction /= float(base)
    return float(result)


def _expand_pair_command_cm(pair_command_cm: list[float]) -> list[float]:
    pair_x = float(pair_command_cm[0] if len(pair_command_cm) > 0 else 0.0)
    pair_y = float(pair_command_cm[1] if len(pair_command_cm) > 1 else 0.0)
    return [pair_x, pair_y, -pair_x, -pair_y]


def _pair_command_from_cable_deltas(cable_deltas_cm: list[float]) -> list[float]:
    if len(cable_deltas_cm) >= 4:
        return [float(cable_deltas_cm[0]), float(cable_deltas_cm[1])]
    if len(cable_deltas_cm) == 2:
        return [float(cable_deltas_cm[0]), float(cable_deltas_cm[1])]
    if len(cable_deltas_cm) == 1:
        return [float(cable_deltas_cm[0]), 0.0]
    return [0.0, 0.0]


def _serialize_raw_tracker_frame(
    *,
    session: ExperimentSession,
    command_index: int,
    frame: dict[str, Any],
    tool_id: str,
) -> dict[str, Any]:
    """Build one raw-tracker-frame JSON row for raw_tracker_samples.jsonl.

    Captures the per-frame pose in both tracker (aurora) and robot frames,
    plus timestamps and validity state, so a reader can investigate whether
    per-command spread is random noise, drift, settling, or outliers without
    having to reload the full snapshot stream.
    """
    snapshot = frame["snapshot"]
    gate = frame["gate"]
    tool_key = str(tool_id or "0A").upper()
    tool = snapshot.tools.get(tool_key)
    tracker_pose: dict[str, Any] = {
        "tracking_state": getattr(tool, "tracking_state", None) if tool is not None else None,
        "translation_mm": (
            list(tool.translation_mm) if tool is not None and tool.translation_mm is not None else None
        ),
        "quaternion_wxyz": (
            list(tool.quaternion_wxyz) if tool is not None and tool.quaternion_wxyz is not None else None
        ),
        "frame_number": getattr(tool, "frame_number", None) if tool is not None else None,
    }
    robot_pose: dict[str, Any] = {}
    if getattr(snapshot, "T_robot_tip", None) is not None:
        matrix = snapshot.T_robot_tip
        robot_pose = {
            "matrix": matrix,
            "translation_mm": [
                float(matrix[0][3]),
                float(matrix[1][3]),
                float(matrix[2][3]),
            ],
        }
    return {
        "command_index": int(command_index),
        "frame_index": int(frame["frame_index"]),
        "tool_id": tool_key,
        "monotonic_time_s": float(frame.get("monotonic_t", 0.0)),
        "wall_time_utc": datetime.now(timezone.utc).isoformat(),
        "tracker_frame_id": getattr(snapshot, "last_frame_number", None),
        "tracker_data_age_s": getattr(snapshot, "tracker_data_age_s", None),
        "tracker_canonical_state": getattr(snapshot, "canonical_state", None),
        "tip_pose_status": getattr(snapshot, "tip_pose_status", None),
        "gate_accepted": bool(gate.get("accepted")),
        "gate_reason": str(gate.get("reason", "")),
        "pose_in_tracker_frame": tracker_pose,
        "pose_in_robot_frame": robot_pose,
    }


def _collect_post_settle_tracker_frames(
    *,
    session: ExperimentSession,
    tool_id: str,
    max_tracker_age_s: float,
    per_frame_max_wait_s: float,
    poll_interval_s: float,
    require_robot_frame_tip: bool,
    allow_mock_state: bool,
    allow_lower_trust_runtime_tip: bool,
    tracker_samples_per_command: int,
) -> tuple[list[dict[str, Any]], str | None]:
    """Collect N deduplicated post-settle tracker frames.

    Returns ``(frames, reason)``:
      - ``frames``: list of dicts ``{snapshot, gate, monotonic_t, frame_index}``
        with at most ``tracker_samples_per_command`` entries. Each entry has
        passed the per-modeling-tracker gate and has a unique
        ``snapshot.last_frame_number`` (the tracker cache returns the same
        frame if polled faster than the backend produces new ones, so we
        dedupe on frame_number).
      - ``reason``: None on success, else a short string describing why we
        couldn't collect N frames (e.g. "frame_wait_timeout_at_index_3").

    No filtering or rejection of "high spread" samples is done here — every
    accepted frame is kept, per the user's explicit "do not add rejection
    thresholds" directive. This is an estimator of frame-to-frame variability,
    not a noise filter.
    """
    frames: list[dict[str, Any]] = []
    seen_frame_ids: set[int] = set()
    target = max(1, int(tracker_samples_per_command))
    while len(frames) < target:
        session.raise_if_stop_requested()
        frame_deadline = session.context.monotonic_fn() + float(per_frame_max_wait_s)
        snapshot = session.context.tracking_service.get_snapshot()
        gate = _modeling_tracker_gate_status(
            snapshot=snapshot,
            tool_id=tool_id,
            max_tracker_age_s=max_tracker_age_s,
            require_robot_frame_tip=require_robot_frame_tip,
            allow_mock_state=allow_mock_state,
            allow_lower_trust_runtime_tip=allow_lower_trust_runtime_tip,
        )
        frame_id = getattr(snapshot, "last_frame_number", None)
        accepted = bool(gate.get("accepted")) and frame_id is not None and frame_id not in seen_frame_ids
        while not accepted and session.context.monotonic_fn() < frame_deadline:
            session.raise_if_stop_requested()
            session.context.sleep_fn(float(poll_interval_s))
            snapshot = session.context.tracking_service.get_snapshot()
            gate = _modeling_tracker_gate_status(
                snapshot=snapshot,
                tool_id=tool_id,
                max_tracker_age_s=max_tracker_age_s,
                require_robot_frame_tip=require_robot_frame_tip,
                allow_mock_state=allow_mock_state,
                allow_lower_trust_runtime_tip=allow_lower_trust_runtime_tip,
            )
            frame_id = getattr(snapshot, "last_frame_number", None)
            accepted = bool(gate.get("accepted")) and frame_id is not None and frame_id not in seen_frame_ids
        if not accepted:
            return frames, f"frame_wait_timeout_at_index_{len(frames)}"
        seen_frame_ids.add(int(frame_id))
        frames.append(
            {
                "snapshot": snapshot,
                "gate": gate,
                "monotonic_t": float(session.context.monotonic_fn()),
                "frame_index": int(len(frames)),
            }
        )
    return frames, None


def _average_tracker_frames(
    *,
    frames: list[dict[str, Any]],
    tool_id: str,
) -> dict[str, Any]:
    """Compute averaged pose + spread stats from a list of valid frames.

    Returns a dict with:
      - ``averaged_T_robot_tip``: 4x4 list-of-lists, mean position + sign-aligned
        quaternion mean of per-frame robot-frame rotations.
      - ``averaged_tracker_translation_mm``, ``averaged_tracker_quaternion_wxyz``:
        averaged aurora-frame tool pose (the raw tracker reading averages).
      - ``stats``: position std per-axis / RMS / max-deviation, orientation
        spread (degrees), first-vs-mean position+orientation differences,
        valid frame count, and orientation_average_method.

    Honest framing per the user's spec: this estimates random frame-to-frame
    variability around a single command point. It does not, and cannot,
    distinguish that variability from systematic registration error, settling
    drift, or mechanical hysteresis. Raw frames are preserved separately so
    a reader can investigate.
    """
    if not frames:
        raise ValueError("No frames to average")
    missing_t_robot_tip = [
        frame["frame_index"]
        for frame in frames
        if frame["snapshot"].T_robot_tip is None
    ]
    if missing_t_robot_tip:
        raise ValueError(
            "_average_tracker_frames requires snapshot.T_robot_tip on every frame "
            f"(missing on frame_index {missing_t_robot_tip}). Enable "
            "require_robot_frame_tip on the experiment config."
        )

    # ---- robot-frame position & orientation ---------------------------------
    robot_positions = np.asarray(
        [
            [
                float(frame["snapshot"].T_robot_tip[0][3]),
                float(frame["snapshot"].T_robot_tip[1][3]),
                float(frame["snapshot"].T_robot_tip[2][3]),
            ]
            for frame in frames
        ],
        dtype=float,
    )
    robot_quats = np.asarray(
        [
            list(
                rotmat_to_quat_wxyz(
                    np.asarray(frame["snapshot"].T_robot_tip, dtype=float)[0:3, 0:3]
                )
            )
            for frame in frames
        ],
        dtype=float,
    )
    mean_robot_position = robot_positions.mean(axis=0)
    # robot_quats is guaranteed non-empty here because we already required at
    # least one frame and every frame must have T_robot_tip (asserted above).
    mean_robot_quat, quat_details = average_quaternions(
        robot_quats, method="sign_aligned_mean"
    )
    averaged_T_robot_tip = np.eye(4, dtype=float)
    averaged_T_robot_tip[0:3, 0:3] = quat_wxyz_to_rotmat(tuple(mean_robot_quat))
    averaged_T_robot_tip[0:3, 3] = mean_robot_position

    # ---- aurora-frame (tracker) pose averages (raw readings) ----------------
    tracker_translations: list[list[float]] = []
    tracker_quats: list[list[float]] = []
    for frame in frames:
        tool = frame["snapshot"].tools.get(str(tool_id or "0A").upper())
        if tool is None:
            continue
        if tool.translation_mm is not None:
            tracker_translations.append([float(value) for value in tool.translation_mm])
        if tool.quaternion_wxyz is not None:
            tracker_quats.append([float(value) for value in tool.quaternion_wxyz])
    averaged_tracker_translation_mm: list[float] | None = None
    averaged_tracker_quaternion_wxyz: list[float] | None = None
    tracker_quat_details: dict[str, Any] = {}
    if tracker_translations:
        averaged_tracker_translation_mm = (
            np.asarray(tracker_translations, dtype=float).mean(axis=0).tolist()
        )
    if tracker_quats:
        mean_tracker_quat, tracker_quat_details = average_quaternions(
            np.asarray(tracker_quats, dtype=float), method="sign_aligned_mean"
        )
        averaged_tracker_quaternion_wxyz = [float(value) for value in mean_tracker_quat]

    # ---- spread stats -------------------------------------------------------
    deviations = robot_positions - mean_robot_position
    per_axis_std_mm = robot_positions.std(axis=0, ddof=0)
    rms_std_mm = float(np.sqrt(np.mean(per_axis_std_mm ** 2)))
    deviation_norms_mm = np.linalg.norm(deviations, axis=1)
    max_deviation_mm = float(deviation_norms_mm.max()) if deviation_norms_mm.size else 0.0
    first_position = robot_positions[0]
    first_vs_mean_position_diff_mm = float(np.linalg.norm(first_position - mean_robot_position))

    # Per-frame angular deltas around the averaged orientation.
    angle_diffs_rad: list[float] = []
    for quat in robot_quats:
        dot = abs(float(np.dot(quat, mean_robot_quat)))
        dot = max(min(dot, 1.0), -1.0)
        angle_diffs_rad.append(2.0 * float(np.arccos(dot)))
    angle_diffs_deg = [float(np.degrees(value)) for value in angle_diffs_rad]
    orientation_max_spread_deg = float(max(angle_diffs_deg)) if angle_diffs_deg else 0.0
    orientation_std_deg = float(np.std(angle_diffs_deg, ddof=0)) if angle_diffs_deg else 0.0
    first_vs_mean_orientation_diff_deg = float(angle_diffs_deg[0]) if angle_diffs_deg else 0.0

    sample_window_s: float | None = None
    if len(frames) >= 2:
        sample_window_s = float(frames[-1]["monotonic_t"] - frames[0]["monotonic_t"])

    stats = {
        "valid_sample_count": int(len(frames)),
        "sample_window_s": sample_window_s,
        "position_std_per_axis_mm": [float(value) for value in per_axis_std_mm],
        "position_std_rms_mm": rms_std_mm,
        "position_max_deviation_mm": max_deviation_mm,
        "first_vs_mean_position_diff_mm": first_vs_mean_position_diff_mm,
        "orientation_average_available": bool(robot_quats.size),
        "orientation_average_method": "sign_aligned_mean_of_robot_frame_quaternions",
        "orientation_max_spread_deg": orientation_max_spread_deg,
        "orientation_std_deg": orientation_std_deg,
        "first_vs_mean_orientation_diff_deg": first_vs_mean_orientation_diff_deg,
        "robot_quat_sign_flips_applied": int(quat_details.get("sign_flips_applied", 0)),
        "tracker_quat_sign_flips_applied": int(tracker_quat_details.get("sign_flips_applied", 0)),
        "frame_ids": [int(f["snapshot"].last_frame_number) for f in frames if f["snapshot"].last_frame_number is not None],
    }
    return {
        "averaged_T_robot_tip": averaged_T_robot_tip.tolist(),
        "averaged_tracker_translation_mm": averaged_tracker_translation_mm,
        "averaged_tracker_quaternion_wxyz": averaged_tracker_quaternion_wxyz,
        "averaged_robot_position_mm": [float(value) for value in mean_robot_position],
        "averaged_robot_quaternion_wxyz": [float(value) for value in mean_robot_quat],
        "stats": stats,
    }


def _wait_for_collect_pose_capture(
    *,
    session: ExperimentSession,
    tool_id: str,
    max_tracker_age_s: float,
    timeout_s: float,
    poll_interval_s: float,
    require_robot_frame_tip: bool,
    allow_mock_state: bool,
    allow_lower_trust_runtime_tip: bool = False,
) -> tuple[Any, dict[str, Any]]:
    deadline = session.context.monotonic_fn() + float(timeout_s)
    last_snapshot = session.context.tracking_service.get_snapshot()
    last_gate = _modeling_tracker_gate_status(
        snapshot=last_snapshot,
        tool_id=tool_id,
        max_tracker_age_s=max_tracker_age_s,
        require_robot_frame_tip=require_robot_frame_tip,
        allow_mock_state=allow_mock_state,
        allow_lower_trust_runtime_tip=allow_lower_trust_runtime_tip,
    )
    if last_gate["accepted"]:
        return last_snapshot, last_gate
    while session.context.monotonic_fn() < deadline:
        session.raise_if_stop_requested()
        session.context.sleep_fn(float(poll_interval_s))
        last_snapshot = session.context.tracking_service.get_snapshot()
        last_gate = _modeling_tracker_gate_status(
            snapshot=last_snapshot,
            tool_id=tool_id,
            max_tracker_age_s=max_tracker_age_s,
            require_robot_frame_tip=require_robot_frame_tip,
            allow_mock_state=allow_mock_state,
            allow_lower_trust_runtime_tip=allow_lower_trust_runtime_tip,
        )
        if last_gate["accepted"]:
            return last_snapshot, last_gate
    return last_snapshot, last_gate


def _modeling_tracker_gate_status(
    *,
    snapshot,
    tool_id: str,
    max_tracker_age_s: float,
    require_robot_frame_tip: bool,
    allow_mock_state: bool = False,
    allow_lower_trust_runtime_tip: bool = False,
) -> dict[str, Any]:
    tool_key = str(tool_id or "0A").upper()
    tool = snapshot.tools.get(tool_key)
    reasons: list[str] = []
    age = snapshot.tracker_data_age_s
    if (
        snapshot.canonical_state not in {"streaming_healthy", "streaming_degraded"}
        and not (allow_mock_state and snapshot.canonical_state == "mock")
    ):
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
    if require_robot_frame_tip:
        if snapshot.registration_state != "loaded":
            reasons.append(f"registration_state_{snapshot.registration_state}")
        runtime_tip_policy = evaluate_runtime_tip_trust(
            snapshot=snapshot,
            workflow=WORKFLOW_MODELING_DATASET,
            allow_lower_trust=bool(allow_lower_trust_runtime_tip),
        )
        if not runtime_tip_policy.allowed_for_workflow:
            reasons.extend(runtime_tip_policy.reasons or ["runtime_tip_policy_not_allowed"])
        if snapshot.tip_pose_status not in {"ok", "coil_as_tip"}:
            reasons.append(f"tip_pose_status_{snapshot.tip_pose_status}")
        if snapshot.T_robot_tip is None:
            reasons.append("missing_T_robot_tip")
    return {
        "accepted": not reasons,
        "reason": "ok" if not reasons else "; ".join(reasons),
        "tracker_age_s": None if age is None else float(age),
        "tracker_frame_id": snapshot.last_frame_number,
        "tool_id": tool_key,
        "tool_present": bool(tool.present) if tool is not None else False,
        "tool_valid": tool.valid if tool is not None else None,
        "tip_pose_status": getattr(snapshot, "tip_pose_status", None),
        "registration_state": getattr(snapshot, "registration_state", None),
        "runtime_tip_mode": getattr(snapshot, "runtime_tip_mode", None),
        "runtime_tip_trust_level": (
            evaluate_runtime_tip_trust(
                snapshot=snapshot,
                workflow=WORKFLOW_MODELING_DATASET,
                allow_lower_trust=bool(allow_lower_trust_runtime_tip),
            ).trust_label
        ),
    }


def _record_collect_pose_run_provenance(
    *,
    session: ExperimentSession,
    config: CollectPoseCommandDatasetConfig,
    servo_ids: list[int],
    neutral_ticks: list[int],
    pair_limits: dict[str, Any],
) -> None:
    servo_only_mode = _collect_pose_servo_only_test_mode(config=config, tracking_service=session.context.tracking_service)
    parallel_single_demo = _collect_pose_parallel_single_mode(session)
    snapshot = session.context.tracking_service.get_snapshot() if session.context.tracking_service is not None else None
    runtime_tip_policy = (
        evaluate_runtime_tip_trust(
            snapshot=snapshot,
            workflow=WORKFLOW_MODELING_DATASET,
            allow_lower_trust=bool(config.allow_lower_trust_runtime_tip),
        )
        if snapshot is not None
        else None
    )
    servo_calibration_summary = session.context.servo_service.get_calibration_summary()
    pretension_source = session.context.servo_service.pretension_source_summary([int(value) for value in servo_ids])
    startup_reference_error: str | None = None
    startup_reference_source = "unavailable"
    startup_reference_ticks_by_servo: dict[str, int] = {}
    try:
        startup_reference = session.context.servo_service.resolve_startup_reference_ticks(list(servo_ids))
    except Exception as exc:
        startup_reference = None
        startup_reference_error = str(exc)
    if startup_reference is not None:
        startup_reference_source = str(startup_reference.source or "neutral")
        startup_reference_ticks_by_servo = {
            str(servo_id): int(tick)
            for servo_id, tick in sorted(startup_reference.ticks_by_servo.items())
        }
    registration_path = Path(
        getattr(snapshot, "registration_path", None) or session.context.registration_path
    )
    runtime_tip_path_raw = getattr(snapshot, "runtime_tip_calibration_path", None) if snapshot is not None else None
    runtime_tip_path = Path(runtime_tip_path_raw) if runtime_tip_path_raw else None
    gate = (
        {
            "accepted": False,
            "reason": "servo_only_no_tracker_test_run",
            "tracker_connected": False,
            "runtime_tip_trust_level": "servo_only",
        }
        if servo_only_mode
        else _modeling_tracker_gate_status(
            snapshot=snapshot,
            tool_id=str(config.tool_id or "0A"),
            max_tracker_age_s=float(config.max_tracker_age_s),
            require_robot_frame_tip=bool(config.require_robot_frame_tip),
            allow_mock_state=bool(config.dry_run),
            allow_lower_trust_runtime_tip=bool(config.allow_lower_trust_runtime_tip),
        )
    )
    provenance = {
        "dataset_mode": str(config.dataset_mode or "workspace_coverage"),
        "run_label": str(config.run_label or ""),
        "dataset_tag": str(config.dataset_tag or ""),
        "run_trust_mode": "servo_only" if servo_only_mode else str(config.run_trust_mode or "thesis_trusted"),
        "valid_for_model_training": not bool(servo_only_mode or parallel_single_demo),
        "valid_for_thesis_repeatability": not bool(servo_only_mode or parallel_single_demo),
        "parallel_single_demo": bool(parallel_single_demo),
        "true_two_segment_control": False,
        "tracker_connected": bool(snapshot is not None),
        "backend_identity": str(getattr(snapshot, "backend_identity", "") or ""),
        "selected_backend_name": str(getattr(snapshot, "selected_backend_name", "") or ""),
        "configured_backend_name": str(getattr(session.context.settings.serial, "tracker_backend", "") or ""),
        "tool_id": str(config.tool_id or "0A"),
        "require_robot_frame_tip": bool(config.require_robot_frame_tip),
        "max_tracker_age_s": float(config.max_tracker_age_s),
        "dry_run": bool(config.dry_run),
        "pair_limits": dict(pair_limits),
        "startup_reference_source": startup_reference_source,
        "startup_reference_ticks_by_servo": startup_reference_ticks_by_servo,
        "startup_reference_error": startup_reference_error,
        "neutral_ticks_by_servo": {
            str(servo_id): int(tick)
            for servo_id, tick in zip(servo_ids, neutral_ticks)
        },
        "base_registration": {
            **_collect_pose_file_provenance(registration_path),
            "state": str(getattr(snapshot, "registration_state", None) or "unavailable"),
            "stored_timestamp_utc": getattr(snapshot, "stored_registration_timestamp_utc", None),
            "stored_fre_mm": getattr(snapshot, "stored_registration_fre_mm", None),
        },
        "runtime_tip_calibration": {
            **_collect_pose_file_provenance(runtime_tip_path),
            "state": str(getattr(snapshot, "runtime_tip_calibration_state", None) or "unavailable"),
            "mode": runtime_tip_policy.mode if runtime_tip_policy is not None else "unavailable",
            "trust_level": runtime_tip_policy.trust_label if runtime_tip_policy is not None else "servo_only",
            "policy": runtime_tip_policy.to_dict() if runtime_tip_policy is not None else None,
            "thesis_trusted": bool(runtime_tip_policy.thesis_trusted) if runtime_tip_policy is not None else False,
            "mode_message": str(getattr(snapshot, "runtime_tip_mode_message", "") or ""),
            "selected_artifact_kind": getattr(snapshot, "runtime_tip_selected_artifact_kind", None),
            "selected_artifact_path": getattr(snapshot, "runtime_tip_selected_artifact_path", None),
            "stored_timestamp_utc": getattr(snapshot, "stored_runtime_tip_timestamp_utc", None),
            "identity_fallback": bool(getattr(snapshot, "runtime_tip_identity_fallback", False)),
            "tip_pose_status": str(getattr(snapshot, "tip_pose_status", "")),
        },
        "pretension_artifact": {
            **_collect_pose_file_provenance(Path(servo_calibration_summary.path)),
            "status": servo_calibration_summary.status,
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
        },
        "run_start_preflight": {
            "overall_status": "warning"
            if (
                bool(config.dry_run)
                or bool(config.allow_lower_trust_runtime_tip)
                or bool(config.allow_lower_trust_pretension)
                or bool(servo_only_mode)
            )
            else "ready",
            "tracker_gate": dict(gate),
            "tracking_state": str(getattr(snapshot, "canonical_state", None) or "disabled"),
            "runtime_tip_mode": runtime_tip_policy.mode if runtime_tip_policy is not None else "unavailable",
            "runtime_tip_trust_level": runtime_tip_policy.trust_label if runtime_tip_policy is not None else "servo_only",
            "pretension_source_type": pretension_source.source_type,
            "pretension_message": pretension_source.message,
            "operating_mode": str(session.context.settings.robot.operating_context().operating_mode),
            "active_segment": str(session.context.settings.robot.active_segment_key()),
            "commanded_servo_ids": [int(value) for value in session.context.settings.robot.commanded_servo_ids()],
            "baud": int(session.context.settings.serial.baudrate),
            "parallel_single_demo": bool(parallel_single_demo),
        },
    }
    if parallel_single_demo:
        operating_context = session.context.settings.robot.operating_context()
        segments = dict(operating_context.segments or {})
        segment_order = list(operating_context.segment_order or [])
        segment_a_key = segment_order[0] if len(segment_order) >= 1 else "segment_a"
        segment_b_key = segment_order[1] if len(segment_order) >= 2 else "segment_b"
        segment_a = segments.get(segment_a_key)
        segment_b = segments.get(segment_b_key)
        sign_mapping_info = session.metadata.backend_info.get("servo_sign_mapping_check")
        sign_mapping_path = None
        if isinstance(sign_mapping_info, dict):
            sign_mapping_path = sign_mapping_info.get("path")
        startup_artifact_path = str(servo_calibration_summary.path)
        provenance["parallel_single"] = {
            "warning": "This is not two-segment kinematics. It is synchronized single-segment command playback.",
            "segment_order": segment_order,
            "segment_a": {
                "key": str(segment_a.key if segment_a is not None else segment_a_key),
                "label": str(segment_a.label if segment_a is not None else "Segment A"),
                "segment_role": str(segment_a.segment_role if segment_a is not None else "proximal"),
                "servo_ids": [int(value) for value in (segment_a.servo_ids if segment_a is not None else [1, 2, 3, 4])],
                "startup_artifact_path": startup_artifact_path,
            },
            "segment_b": {
                "key": str(segment_b.key if segment_b is not None else segment_b_key),
                "label": str(segment_b.label if segment_b is not None else "Segment B"),
                "segment_role": str(segment_b.segment_role if segment_b is not None else "distal"),
                "servo_ids": [int(value) for value in (segment_b.servo_ids if segment_b is not None else [5, 6, 7, 8])],
                "startup_artifact_path": startup_artifact_path,
            },
            "mapping_confirmation_path": str(sign_mapping_path or ""),
            "pose_label_scope_note": "Pose label corresponds only to tracked coil/tool, not both spines.",
        }
    session.metadata.backend_info["run_provenance"] = {
        "dataset_mode": provenance["dataset_mode"],
        "run_label": provenance["run_label"],
        "dataset_tag": provenance["dataset_tag"],
        "run_trust_mode": provenance["run_trust_mode"],
        "valid_for_model_training": provenance["valid_for_model_training"],
        "valid_for_thesis_repeatability": provenance["valid_for_thesis_repeatability"],
        "parallel_single_demo": provenance.get("parallel_single_demo", False),
        "true_two_segment_control": provenance.get("true_two_segment_control", False),
        "tracker_connected": provenance["tracker_connected"],
        "backend_identity": provenance["backend_identity"],
        "selected_backend_name": provenance["selected_backend_name"],
        "pretension_artifact": provenance["pretension_artifact"],
        "run_start_preflight": provenance["run_start_preflight"],
    }
    session.metadata.registration_info["base_registration"] = provenance["base_registration"]
    session.metadata.registration_info["runtime_tip_calibration"] = provenance["runtime_tip_calibration"]
    session.metadata.registration_info["runtime_tip_policy"] = (
        runtime_tip_policy.to_dict() if runtime_tip_policy is not None else None
    )
    session.set_metric("run_provenance", provenance)


def _servo_motion_profile_from_result(command) -> dict[str, Any]:
    debug_entries = list((command.debug_entries_by_id or {}).values())
    if not debug_entries:
        return {
            "operating_mode_label": "unknown",
            "operating_mode": None,
            "goal_current_ma": None,
            "profile_velocity": None,
            "profile_acceleration": None,
        }
    entry = debug_entries[0]
    mode_label = {
        0: "Current Control",
        1: "Velocity Control",
        3: "Position Control",
        4: "Extended Position Control",
        5: "Current-based Position Control",
        16: "PWM Control",
    }.get(entry.operating_mode, f"Mode {entry.operating_mode}" if entry.operating_mode is not None else "unknown")
    return {
        "operating_mode_label": mode_label,
        "operating_mode": entry.operating_mode,
        "goal_current_ma": entry.goal_current_ma,
        "profile_velocity": entry.profile_velocity,
        "profile_acceleration": entry.profile_acceleration,
    }


def _servo_debug_payload(command) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for servo_id, entry in sorted((command.debug_entries_by_id or {}).items()):
        payload[str(servo_id)] = {
            "servo_id": int(entry.servo_id),
            "requested_displacement_cm": float(entry.requested_displacement_cm),
            "resolved_displacement_cm": float(entry.resolved_displacement_cm),
            "present_position_tick": entry.present_position_tick,
            "present_current_ma": entry.present_current_ma,
            "raw_goal_tick": entry.raw_goal_tick,
            "final_goal_tick": entry.final_goal_tick,
            "safe_min_tick": entry.safe_min_tick,
            "safe_max_tick": entry.safe_max_tick,
            "telemetry_fresh": entry.telemetry_fresh,
            "operating_mode": entry.operating_mode,
            "preferred_operating_mode": entry.preferred_operating_mode,
            "goal_current_ma": entry.goal_current_ma,
            "profile_velocity": entry.profile_velocity,
            "profile_acceleration": entry.profile_acceleration,
            "clamp_reason": entry.clamp_reason,
            "limit_source": entry.limit_source,
        }
    return payload


def _servo_feedback_payload(telemetry_by_id: dict[int, Any], *, servo_service) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for servo_id, telemetry in sorted((telemetry_by_id or {}).items()):
        payload[str(servo_id)] = {
            "present_position_ticks": telemetry.present_position,
            "present_current_raw_unit": telemetry.present_current_raw_unit,
            "present_current_ma": telemetry.present_current_ma,
            "torque_enabled": telemetry.torque_enabled,
            "telemetry_age_s": servo_service.telemetry_age_s(telemetry),
            "telemetry_fresh": servo_service.telemetry_is_fresh(telemetry),
            "operating_mode": telemetry.operating_mode,
            "hardware_error": telemetry.hardware_error,
            "present_voltage_raw_unit": telemetry.present_voltage_raw_unit,
            "present_voltage_mv": telemetry.present_voltage_mv,
            "present_temperature_c": telemetry.present_temperature_c,
            "last_valid_packet_monotonic_s": getattr(telemetry, "last_valid_packet_monotonic_s", None),
            "last_valid_packet_wall_time": getattr(telemetry, "last_valid_packet_wall_time", None),
            "last_read_attempt_monotonic_s": getattr(telemetry, "last_read_attempt_monotonic_s", None),
            "read_duration_ms": getattr(telemetry, "read_duration_ms", None),
            "packet_age_s": getattr(telemetry, "packet_age_s", None),
            "read_batch_started_monotonic_s": getattr(telemetry, "read_batch_started_monotonic_s", None),
            "read_batch_completed_monotonic_s": getattr(telemetry, "read_batch_completed_monotonic_s", None),
            "read_batch_duration_ms": getattr(telemetry, "read_batch_duration_ms", None),
            "snapshot_age_s": getattr(telemetry, "snapshot_age_s", None),
            "per_servo_packet_age_s": getattr(telemetry, "per_servo_packet_age_s", None),
            "freshness_decision_source": getattr(telemetry, "freshness_decision_source", None),
            "read_source": getattr(telemetry, "read_source", None),
            "telemetry_error_code": getattr(telemetry, "telemetry_error_code", None),
            "telemetry_error_detail": getattr(telemetry, "telemetry_error_detail", None),
            "bus_owner": getattr(telemetry, "bus_owner", None),
            "read_sequence_index": getattr(telemetry, "read_sequence_index", None),
        }
    return payload


def _read_collect_pose_live_servo_feedback(
    *,
    session: ExperimentSession,
    servo_ids: list[int],
) -> dict[str, Any]:
    if session.context.servo_service is None or not getattr(session.context.servo_service, "is_connected", False):
        return {}
    try:
        telemetry = session.context.servo_service.read_live_telemetry([int(value) for value in servo_ids])
    except Exception as exc:
        return {"read_error": str(exc)}
    return _servo_feedback_payload(telemetry, servo_service=session.context.servo_service)


def _collect_pose_pair_range(pair_commands: list[list[float]]) -> dict[str, Any]:
    if not pair_commands:
        return {}
    pair_x = [float(row[0]) for row in pair_commands]
    pair_y = [float(row[1]) for row in pair_commands]
    return {
        "pair_13_min_cm": min(pair_x),
        "pair_13_max_cm": max(pair_x),
        "pair_24_min_cm": min(pair_y),
        "pair_24_max_cm": max(pair_y),
    }


def _collect_pose_workspace_span(robot_positions: list[list[float]]) -> dict[str, Any]:
    if not robot_positions:
        return {}
    xs = [float(row[0]) for row in robot_positions]
    ys = [float(row[1]) for row in robot_positions]
    zs = [float(row[2]) for row in robot_positions]
    return {
        "x_min_mm": min(xs),
        "x_max_mm": max(xs),
        "y_min_mm": min(ys),
        "y_max_mm": max(ys),
        "z_min_mm": min(zs),
        "z_max_mm": max(zs),
        "x_span_mm": max(xs) - min(xs),
        "y_span_mm": max(ys) - min(ys),
        "z_span_mm": max(zs) - min(zs),
    }


def _collect_pose_command_norm_stats(pair_commands: list[list[float]]) -> dict[str, Any]:
    if not pair_commands:
        return {}
    norms = [math.sqrt((float(row[0]) ** 2) + (float(row[1]) ** 2)) for row in pair_commands]
    return {
        "min_cm": min(norms),
        "mean_cm": (sum(norms) / float(len(norms))) if norms else None,
        "max_cm": max(norms),
    }


def _collect_pose_rejection_counts(samples: list[ExperimentTimeseriesSample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        reason = str(sample.extra.get("capture_rejection_reason", "unknown") or "unknown")
        counts[reason] = int(counts.get(reason, 0)) + 1
    return counts


def _collect_pose_phase_counts(samples: list[ExperimentTimeseriesSample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        counts[str(sample.phase or "unknown")] = int(counts.get(str(sample.phase or "unknown"), 0)) + 1
    return counts


def _collect_pose_mode_summary(config: CollectPoseCommandDatasetConfig) -> str:
    if config.dataset_mode == "workspace_coverage":
        return "Bounded workspace-coverage collection for first-pass forward-model training."
    if config.dataset_mode == "hysteresis_path_dependence":
        return "Ordered revisit protocol for state-aware and path-dependence modeling."
    if config.dataset_mode == "angular_test_mesh":
        theta_count = int(config.test_mesh_theta_count or 12)
        phi_count = int(config.test_mesh_phi_count or 24)
        return (
            f"Wolfe-style angular test mesh ({theta_count}×{phi_count} θ×φ grid, "
            f"{int(config.samples_per_command)} reps/cell) for thesis-grade model evaluation."
        )
    return "Repeated trusted startup blocks for cross-revision modeling comparisons."


def _collect_pose_file_provenance(path: Path | None) -> dict[str, Any]:
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
    info["sha256"] = _collect_pose_file_sha256(file_path)
    return info


def _collect_pose_file_sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def register_builtin_experiments(registry) -> None:
    """Register the built-in canonical experiments."""
    registry.register(
        name=TrackerPipelineMockExperiment.name,
        title="Tracker Pipeline Mock",
        description=TrackerPipelineMockExperiment.description,
        category="diagnostic",
        tags=["Mock", "Tracking"],
        workspace_visible=False,
        factory=TrackerPipelineMockExperiment.from_dict,
    )
    registry.register(
        name=TransformChainValidationExperiment.name,
        title="Transform Chain Validation",
        description=TransformChainValidationExperiment.description,
        category="diagnostic",
        tags=["Mock", "Transforms"],
        workspace_visible=False,
        factory=TransformChainValidationExperiment.from_dict,
    )
    registry.register(
        name=CommandScheduleValidationExperiment.name,
        title="Command Schedule Validation",
        description=CommandScheduleValidationExperiment.description,
        category="validation",
        tags=["Commands", "Schedule"],
        default_config_path="config/experiment_command_schedule_validation.example.yaml",
        workspace_visible=False,
        factory=CommandScheduleValidationExperiment.from_dict,
    )
    registry.register(
        name=DatasetSchemaRoundtripExperiment.name,
        title="Dataset Schema Roundtrip",
        description=DatasetSchemaRoundtripExperiment.description,
        category="diagnostic",
        tags=["Schema", "Roundtrip"],
        workspace_visible=False,
        factory=DatasetSchemaRoundtripExperiment.from_dict,
    )
    registry.register(
        name=ReplayRunnerExperiment.name,
        title="Replay Runner",
        description=ReplayRunnerExperiment.description,
        category="analysis",
        tags=["Replay", "Offline"],
        default_config_path="config/experiment_replay_runner.example.yaml",
        workspace_visible=False,
        factory=ReplayRunnerExperiment.from_dict,
    )
    registry.register(
        name=TwoSegmentStartupValidationExperiment.name,
        title="Two-Segment Startup Validation",
        description=TwoSegmentStartupValidationExperiment.description,
        category="validation",
        tags=["Two Segment", "Startup", "Manual Pretension", "Servo"],
        factory=TwoSegmentStartupValidationExperiment.from_dict,
    )
    registry.register(
        name=TwoSegmentCollectPoseCommandDatasetExperiment.name,
        title="Two-Segment Collect-Pose Dataset",
        description=TwoSegmentCollectPoseCommandDatasetExperiment.description,
        category="validation",
        tags=["Two Segment", "Dataset", "Commands", "Pose"],
        factory=TwoSegmentCollectPoseCommandDatasetExperiment.from_dict,
    )
    registry.register(
        name=TwoSegmentRepeatabilityExperiment.name,
        title="Two-Segment Repeatability (Scaffold)",
        description=TwoSegmentRepeatabilityExperiment.description,
        category="validation",
        tags=["Two Segment", "Repeatability", "Open Loop"],
        factory=TwoSegmentRepeatabilityExperiment.from_dict,
    )
    registry.register(
        name=TrackerTimingValidationExperiment.name,
        title="Tracker Timing Validation",
        description=TrackerTimingValidationExperiment.description,
        category="diagnostic",
        tags=["Tracking", "Timing", "Aurora"],
        default_config_path="config/experiment_tracker_timing_validation.example.yaml",
        factory=TrackerTimingValidationExperiment.from_dict,
    )
    registry.register(
        name=ServoTrackerSyncValidationExperiment.name,
        title="Servo-Tracker Sync Validation",
        description=ServoTrackerSyncValidationExperiment.description,
        category="validation",
        tags=["Tracking", "Servo", "Timing", "Synchronization"],
        default_config_path="config/experiment_servo_tracker_sync_validation.example.yaml",
        factory=ServoTrackerSyncValidationExperiment.from_dict,
    )
    register_single_segment_repeatability(registry)
    register_workspace_repeatability_map(registry)
    registry.register(
        name=PretensionValidationExperiment.name,
        title="Pretension Validation",
        description=PretensionValidationExperiment.description,
        category="validation",
        tags=["Pretension", "Tracking", "Servo"],
        default_config_path="config/experiment_pretension_validation.example.yaml",
        factory=PretensionValidationExperiment.from_dict,
    )
    registry.register(
        name=PenprobeChasingDemoExperiment.name,
        title="Penprobe Chasing Demo",
        description=PenprobeChasingDemoExperiment.description,
        category="demo",
        tags=["Penprobe", "Tracking", "Servo", "Demo"],
        default_config_path="config/experiment_penprobe_chasing_demo.example.yaml",
        factory=PenprobeChasingDemoExperiment.from_dict,
    )
    registry.register(
        name=CollectPoseCommandDatasetExperiment.name,
        title="Random Data Collection",
        description=CollectPoseCommandDatasetExperiment.description,
        category="dataset",
        tags=["Random Data Collection", "Modeling", "Tracking", "Servo"],
        default_config_path="config/experiment_collect_pose_command_dataset.example.yaml",
        factory=CollectPoseCommandDatasetExperiment.from_dict,
    )
    registry.register(
        name=RegistrationSamplingStudyExperiment.name,
        title="Registration Sampling Study",
        description=RegistrationSamplingStudyExperiment.description,
        category="validation",
        tags=["Registration", "Tracking", "Validation"],
        default_config_path="config/experiment_registration_sampling_study.example.yaml",
        factory=RegistrationSamplingStudyExperiment.from_dict,
    )
    register_calibration_validation_experiments(registry)
    register_critical_experiments(registry)
    register_registration_trial_experiment(registry)


def _load_neutral_ticks(session: ExperimentSession) -> list[int]:
    neutral_map = session.context.servo_service.load_neutral_setpoints()
    return [
        int(neutral_map[servo_id])
        for servo_id in session.context.settings.robot.servo_ids
        if servo_id in neutral_map
    ]


def _sample_from_tracking_snapshot(
    session: ExperimentSession,
    *,
    snapshot,
    phase: str,
    step_index: int,
    sample_index: int,
    commanded_cable_deltas_cm: list[float],
    commanded_motor_values: dict[str, int | float | None],
    status_flags: list[str] | None = None,
    extra: dict[str, Any] | None = None,
    cycle_index: int | None = None,
    target_index: int | None = None,
    revisit_index: int | None = None,
    approach_index: int | None = None,
) -> ExperimentTimeseriesSample:
    return sample_from_tracking_snapshot(
        session,
        snapshot=snapshot,
        phase=phase,
        step_index=step_index,
        sample_index=sample_index,
        commanded_cable_deltas_cm=commanded_cable_deltas_cm,
        commanded_motor_values=commanded_motor_values,
        status_flags=status_flags,
        extra=extra,
        cycle_index=cycle_index,
        target_index=target_index,
        revisit_index=revisit_index,
        approach_index=approach_index,
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
