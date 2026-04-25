"""Built-in canonical experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import logging
import math
from pathlib import Path
import random
import threading
import time
from typing import Any

from continuum_robot.experiments.dataset_io import ExperimentDatasetLoader, ExperimentDatasetWriter
from continuum_robot.experiments.experiment_models import ExperimentPoint
from continuum_robot.experiments.framework import BaseExperiment, ExperimentHardwareRequirements, ExperimentSession
from continuum_robot.experiments.critical_experiments import register_critical_experiments
from continuum_robot.experiments.calibration_validation import register_calibration_validation_experiments
from continuum_robot.experiments.modeling_dataset_outputs import write_modeling_dataset_outputs
from continuum_robot.experiments.pretension_validation_outputs import write_pretension_validation_outputs
from continuum_robot.experiments.servo_tracker_sync_outputs import write_servo_tracker_sync_outputs
from continuum_robot.experiments.single_segment_repeatability import register_single_segment_repeatability
from continuum_robot.experiments.tracker_timing_outputs import write_tracker_timing_outputs
from continuum_robot.experiments.schedules import (
    CommandScheduleConfig,
    command_schedule_checksum,
    generate_command_schedule,
)
from continuum_robot.experiments.sample_builders import sample_from_tracking_snapshot
from continuum_robot.experiments.schemas import ExperimentMetadata, ExperimentSummary, ExperimentTimeseriesSample
from continuum_robot.servos.servo_service import PretensionParameters
from continuum_robot.tracking.timing_benchmark import (
    compute_servo_tracker_sync_summary,
    compute_servo_sync_summary,
    extract_servo_command_records,
    compute_tracker_timing_summary,
    extract_servo_timing_records,
    extract_tracker_timing_records,
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
    allow_lower_trust_runtime_tip: bool = False
    allow_lower_trust_pretension: bool = False
    export_legacy_dat: bool = True
    run_label: str = ""
    dataset_tag: str = ""
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
            allow_lower_trust_runtime_tip=bool(payload.get("allow_lower_trust_runtime_tip", False)),
            allow_lower_trust_pretension=bool(payload.get("allow_lower_trust_pretension", False)),
            export_legacy_dat=bool(payload.get("export_legacy_dat", True)),
            run_label=str(payload.get("run_label", "") or ""),
            dataset_tag=str(payload.get("dataset_tag", "") or ""),
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


@dataclass
class PretensionValidationExperimentConfig:
    """Config for one-servo pretension response validation."""

    mode: str = "single_servo_trace"
    servo_id: int = 1
    servo_ids: list[int] = field(default_factory=list)
    repeat_runs: int = 1
    move_to_reference: bool = True
    include_tracker_displacement: bool = True
    allow_current_only_when_tracker_missing: bool = True
    enable_tip_centering: bool = True
    tip_center_tolerance_mm: float = 3.0
    tip_center_max_iterations: int = 16
    tip_center_step_ticks: int = 2
    tip_center_x_sign: int = 1
    tip_center_y_sign: int = 1
    tip_target_xy_mm: list[float] = field(default_factory=lambda: [0.0, 0.0])
    max_tip_displacement_mm: float = 50.0
    equalization_step_ticks: int = 2
    equalization_max_iterations: int = 24
    load_balance_tolerance_ma: float = 120.0
    pair_balance_tolerance_ma: float = 80.0
    settle_verify_time_s: float = 0.2
    accept_max_load_balance_error_ma: float = 150.0
    accept_max_pair_balance_error_ma: float = 120.0
    accept_max_final_tip_xy_offset_mm: float = 5.0
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
        return cls(
            mode=str(payload.get("mode", "single_servo_trace") or "single_servo_trace").strip().lower(),
            servo_id=int(payload.get("servo_id", 1)),
            servo_ids=servo_ids,
            repeat_runs=max(1, int(payload.get("repeat_runs", 1))),
            move_to_reference=bool(payload.get("move_to_reference", True)),
            include_tracker_displacement=bool(payload.get("include_tracker_displacement", True)),
            allow_current_only_when_tracker_missing=bool(payload.get("allow_current_only_when_tracker_missing", True)),
            enable_tip_centering=bool(payload.get("enable_tip_centering", True)),
            tip_center_tolerance_mm=max(0.0, float(payload.get("tip_center_tolerance_mm", 3.0))),
            tip_center_max_iterations=max(0, int(payload.get("tip_center_max_iterations", 16))),
            tip_center_step_ticks=max(1, int(payload.get("tip_center_step_ticks", 2))),
            tip_center_x_sign=(1 if int(payload.get("tip_center_x_sign", 1)) >= 0 else -1),
            tip_center_y_sign=(1 if int(payload.get("tip_center_y_sign", 1)) >= 0 else -1),
            tip_target_xy_mm=[
                float(value)
                for value in (payload.get("tip_target_xy_mm") or [0.0, 0.0])
            ][:2]
            or [0.0, 0.0],
            max_tip_displacement_mm=max(0.0, float(payload.get("max_tip_displacement_mm", 50.0))),
            equalization_step_ticks=max(1, int(payload.get("equalization_step_ticks", 2))),
            equalization_max_iterations=max(0, int(payload.get("equalization_max_iterations", 24))),
            load_balance_tolerance_ma=max(0.0, float(payload.get("load_balance_tolerance_ma", 120.0))),
            pair_balance_tolerance_ma=max(0.0, float(payload.get("pair_balance_tolerance_ma", 80.0))),
            settle_verify_time_s=max(0.0, float(payload.get("settle_verify_time_s", 0.2))),
            accept_max_load_balance_error_ma=max(0.0, float(payload.get("accept_max_load_balance_error_ma", 150.0))),
            accept_max_pair_balance_error_ma=max(0.0, float(payload.get("accept_max_pair_balance_error_ma", 120.0))),
            accept_max_final_tip_xy_offset_mm=max(0.0, float(payload.get("accept_max_final_tip_xy_offset_mm", 5.0))),
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

    def _is_staged_mode(self) -> bool:
        return str(getattr(self.config, "mode", "single_servo_trace") or "").strip().lower() in {
            "single_segment_staged",
            "staged",
            "four_servo_staged",
        }

    def _execute_single_segment_staged(self, session: ExperimentSession) -> None:
        servo_service = session.context.servo_service
        tracker_service = session.context.tracking_service
        include_tracker = bool(self.config.include_tracker_displacement and tracker_service is not None)
        if bool(self.config.include_tracker_displacement) and tracker_service is None:
            if not bool(self.config.allow_current_only_when_tracker_missing):
                raise RuntimeError(
                    "Tracker displacement is required for staged pretension, but tracking_service is unavailable."
                )
            session.add_warning(
                "Tracker displacement is unavailable; staged pretension will run in current-only mode."
            )
        if include_tracker:
            snapshot = tracker_service.get_snapshot()
            if snapshot is None:
                include_tracker = False
                if not bool(self.config.allow_current_only_when_tracker_missing):
                    raise RuntimeError("Tracker snapshot is unavailable for staged pretension.")
                session.add_warning(
                    "Tracker snapshot is unavailable at staged pretension start; falling back to current-only mode."
                )

        servo_ids = self._staged_servo_ids(session)
        repeat_runs = max(1, int(self.config.repeat_runs))
        total_progress = max(1, repeat_runs * 5)
        stage_progress = 0

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

            # Stage B: slack take-up per tendon
            for servo_id in servo_ids:
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
                row = {
                    "mode": "single_segment_staged",
                    "run_index": int(run_index),
                    "stage": "takeup",
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
                    "travel_used_ticks": (
                        max(0, int(result.untensioned_reference_tick) - int(result.final_position_tick))
                        if result.untensioned_reference_tick is not None and result.final_position_tick is not None
                        else None
                    ),
                    "travel_used_mm": (
                        float(
                            servo_service.mapper.ticks_to_displacement_mm(
                                max(0, int(result.untensioned_reference_tick) - int(result.final_position_tick))
                            )
                        )
                        if result.untensioned_reference_tick is not None
                        and result.final_position_tick is not None
                        and getattr(servo_service, "mapper", None) is not None
                        else None
                    ),
                }
                run_trace_rows.append(row)
                trace_rows.append(dict(row))
                self._add_staged_sample(
                    session,
                    phase="pretension_stage_takeup",
                    run_index=run_index,
                    step_index=len(run_trace_rows) - 1,
                    payload=row,
                )
            stage_progress += 1
            session.update_progress(stage_progress, total_progress, {"phase": "takeup", "run_index": run_index})

            # Stage C: load equalization
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
                target = float(sum(load_values) / len(load_values))
                moved = 0
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

            # Stage E: settle and verify
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
                "final_tip_xyz_mm": list(final_tip_xyz_mm) if final_tip_xyz_mm is not None else None,
                "final_tip_xy_offset_mm": tip_xy_offset_mm,
                "missing_fields": sorted(set(missing_fields)),
            }
            run_rows.append(run_row)
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
            session.update_progress(stage_progress, total_progress, {"phase": "verify", "run_index": run_index})

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

        session.metrics.update(
            {
                "mode": "single_segment_staged",
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
                },
                "summary_requirements": {
                    "force_status": (
                        "success"
                        if accepted_runs == len(run_rows)
                        else "partial_success"
                    )
                },
            }
        )

    def _staged_servo_ids(self, session: ExperimentSession) -> list[int]:
        configured_ids = [int(value) for value in (self.config.servo_ids or []) if int(value) > 0]
        if not configured_ids:
            configured_ids = [int(value) for value in (session.context.settings.robot.servo_ids or []) if int(value) > 0]
        deduped: list[int] = []
        for servo_id in configured_ids:
            if servo_id not in deduped:
                deduped.append(int(servo_id))
        if len(deduped) != 4:
            raise RuntimeError(
                "Staged pretension validation requires exactly 4 servo IDs for the single-segment tendon set."
            )
        return deduped

    def _staged_parameters_for_servo(self, session: ExperimentSession, servo_id: int) -> PretensionParameters:
        defaults = session.context.servo_service.default_pretension_parameters(int(servo_id))
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
        tracking_service = session.context.tracking_service
        if tracking_service is not None and getattr(tracking_service, "_thread", None) is None:
            tracking_service.start()
            self._tracking_started_here = True
        self._servo_ids = _configured_collect_pose_servo_ids(session)
        if self.config.dry_run:
            self._initial_neutral_ticks = [0 for _ in self._servo_ids]
        else:
            self._initial_neutral_ticks = _load_collect_pose_neutral_ticks(session, servo_ids=self._servo_ids)

    def precheck(self, session: ExperimentSession) -> None:
        if session.context.tracking_service is None:
            raise RuntimeError("collect_pose_command_dataset requires tracking_service.")
        _precheck_collect_pose_command_dataset(
            session=session,
            config=self.config,
            servo_ids=list(self._servo_ids),
            neutral_ticks=list(self._initial_neutral_ticks),
        )

    def execute(self, session: ExperimentSession) -> None:
        servo_ids = list(self._servo_ids)
        neutral_ticks = list(self._initial_neutral_ticks)
        pair_limits = _collect_pose_pair_limits(
            session=session,
            config=self.config,
            servo_ids=servo_ids,
            neutral_ticks=neutral_ticks,
        )
        command_steps = _build_collect_pose_command_steps(
            config=self.config,
            pair_limits=pair_limits,
        )
        samples_per_command = max(1, int(self.config.samples_per_command))
        total = 2 + (len(command_steps) * samples_per_command)
        accepted_count = 0
        rejected_count = 0
        progress = 0
        zero_vector = [0.0, 0.0, 0.0, 0.0]
        session.set_metric("dataset_mode", str(self.config.dataset_mode or "workspace_coverage"))
        session.set_metric("dry_run", bool(self.config.dry_run))
        session.set_metric("run_label", str(self.config.run_label or ""))
        session.set_metric("dataset_tag", str(self.config.dataset_tag or ""))
        session.set_metric("command_step_count", int(len(command_steps)))
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
                    for step in command_steps
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

        neutral_command = self._issue_command(
            session,
            tendon_displacement_cm=zero_vector,
            servo_ids=servo_ids,
            neutral_ticks=neutral_ticks,
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
        progress += 1
        session.update_progress(progress, total, {"phase": "initial_neutral", "step_index": -1})

        previous_pair_command_cm = [0.0, 0.0]
        for step in command_steps:
            session.raise_if_stop_requested()
            command_result = self._issue_command(
                session,
                tendon_displacement_cm=list(step.cable_command_cm),
                servo_ids=servo_ids,
                neutral_ticks=neutral_ticks,
            )
            settle_time_s = float(step.settle_time_s if step.settle_time_s is not None else self.config.settle_time_s)
            if settle_time_s > 0.0:
                session.context.sleep_fn(settle_time_s)
            for sample_index in range(samples_per_command):
                sample = self._capture_dataset_sample(
                    session=session,
                    command_result=command_result,
                    phase=str(step.phase),
                    step_index=int(step.index),
                    sample_index=int(sample_index),
                    servo_ids=servo_ids,
                    previous_pair_command_cm=list(previous_pair_command_cm),
                    block_index=step.block_index,
                    prior_family=step.prior_family,
                    step_metadata={
                        "label": step.label,
                        **dict(step.metadata or {}),
                    },
                )
                session.add_sample(sample)
                accepted_count += int(bool(sample.extra.get("capture_accepted")))
                rejected_count += int(not bool(sample.extra.get("capture_accepted")))
                progress += 1
                session.update_progress(
                    progress,
                    total,
                    {
                        "phase": str(step.phase),
                        "step_index": int(step.index),
                        "command_label": str(step.label),
                        "accepted_samples": int(accepted_count),
                        "rejected_samples": int(rejected_count),
                    },
                )
            previous_pair_command_cm = list(step.pair_command_cm)

        final_command = self._issue_command(
            session,
            tendon_displacement_cm=zero_vector,
            servo_ids=servo_ids,
            neutral_ticks=neutral_ticks,
        )
        final_sample = self._capture_dataset_sample(
            session=session,
            command_result=final_command,
            phase="final_neutral",
            step_index=len(command_steps),
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
        progress += 1
        session.update_progress(progress, total, {"phase": "final_neutral", "step_index": len(command_steps)})
        session.set_metric("accepted_sample_count", int(accepted_count))
        session.set_metric("rejected_sample_count", int(rejected_count))
        session.set_metric("registration_loaded", session.context.registration_path.exists())

    def finalize(self, session: ExperimentSession) -> None:
        try:
            if not self.config.dry_run and session.context.servo_service.is_connected and self._initial_neutral_ticks:
                self._issue_command(
                    session,
                    tendon_displacement_cm=[0.0, 0.0, 0.0, 0.0],
                    servo_ids=list(self._servo_ids),
                    neutral_ticks=list(self._initial_neutral_ticks),
                )
        finally:
            if self._tracking_started_here:
                session.context.tracking_service.stop()
                self._tracking_started_here = False

    def _issue_command(
        self,
        session: ExperimentSession,
        *,
        tendon_displacement_cm: list[float],
        servo_ids: list[int],
        neutral_ticks: list[int],
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
        if self.config.dry_run or not servo_service.is_connected:
            if len(neutral_ticks) != len(requested_cable_command_cm):
                raise RuntimeError("Dry-run modeling command dimensions do not match the configured servo set.")
            goals = servo_service.mapper.to_goal_positions(requested_cable_command_cm, neutral_ticks)
            commanded_motor_values = {str(servo_id): int(goal) for servo_id, goal in zip(servo_ids, goals)}
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
                "message": "Dry-run command resolved through the tendon-displacement mapper only.",
                "motion_profile": {
                    "operating_mode_label": "dry_run",
                    "operating_mode": None,
                    "goal_current_ma": None,
                    "profile_velocity": None,
                    "profile_acceleration": None,
                },
            }
        try:
            command = servo_service.command_displacement(
                tendon_displacements_cm=list(requested_cable_command_cm),
                neutral_ticks=neutral_ticks,
                servo_ids=servo_ids,
                motion_workflow="experiment_motion",
            )
        except Exception as exc:
            LOG.exception(
                "Collect-pose command failed | requested_cable_cm=%s | servo_ids=%s | error=%s",
                requested_cable_command_cm,
                list(servo_ids),
                exc,
            )
            raise
        motion_profile = _servo_motion_profile_from_result(command)
        LOG.info(
            "Collect-pose command success | resolved_cable_cm=%s | final_goals=%s | clamp_reasons=%s | motion_profile=%s",
            list(command.resolved_displacements_cm or requested_cable_command_cm),
            {str(servo_id): int(goal) for servo_id, goal in sorted(command.positions_by_id.items())},
            {str(servo_id): str(reason) for servo_id, reason in sorted(command.clamp_reasons_by_id.items())},
            motion_profile,
        )
        return {
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
            "message": str(command.message or ""),
            "motion_profile": motion_profile,
        }

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
    ) -> ExperimentTimeseriesSample:
        snapshot, gate = _wait_for_collect_pose_capture(
            session=session,
            tool_id=str(self.config.tool_id or "0A"),
            max_tracker_age_s=float(self.config.max_tracker_age_s),
            timeout_s=float(self.config.capture_timeout_s),
            poll_interval_s=float(self.config.capture_poll_interval_s),
            require_robot_frame_tip=bool(self.config.require_robot_frame_tip),
            allow_mock_state=bool(self.config.dry_run),
        )
        accepted = bool(gate.get("accepted"))
        status_flags = ["capture_accepted"] if accepted else ["capture_rejected"]
        if self.config.dry_run:
            status_flags.append("dry_run")
        live_servo_feedback = _read_collect_pose_live_servo_feedback(
            session=session,
            servo_ids=servo_ids,
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
                "capture_rejection_reason": None if accepted else str(gate.get("reason", "tracker_gate_rejected")),
                "tracker_gate": dict(gate),
                "requested_pair_command_cm": list(command_result.get("requested_pair_command_cm", []) or []),
                "resolved_pair_command_cm": list(command_result.get("resolved_pair_command_cm", []) or []),
                "previous_pair_command_cm": list(previous_pair_command_cm or []),
                "requested_cable_command_cm": list(command_result.get("requested_cable_command_cm", []) or []),
                "resolved_cable_command_cm": list(command_result.get("resolved_cable_command_cm", []) or []),
                "raw_goal_ticks_by_servo": dict(command_result.get("raw_goal_ticks_by_servo", {}) or {}),
                "final_goal_ticks_by_servo": dict(command_result.get("final_goal_ticks_by_servo", {}) or {}),
                "clamp_reasons_by_servo": dict(command_result.get("clamp_reasons_by_servo", {}) or {}),
                "motion_profile": dict(command_result.get("motion_profile", {}) or {}),
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
        return sample

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
        runtime_tip_mode = str(getattr(session.context.tracking_service.get_snapshot(), "runtime_tip_mode", "latest_accepted") or "latest_accepted")
        pretension_source = session.context.servo_service.pretension_source_summary(list(self._servo_ids))
        strict_runtime_tip = not bool(self.config.allow_lower_trust_runtime_tip)
        strict_pretension = not bool(self.config.allow_lower_trust_pretension)
        lower_trust_active = (
            bool(self.config.dry_run)
            or (strict_runtime_tip and runtime_tip_mode != "latest_accepted")
            or (not strict_runtime_tip and runtime_tip_mode != "latest_accepted")
            or (not pretension_source.accepted or not pretension_source.usable)
            or (not strict_pretension and pretension_source.source_type not in {"manual", "algorithmic"})
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
            "runtime_tip_mode_used": runtime_tip_mode,
            "pretension_source_used": pretension_source.source_type,
            "pretension_source_message": pretension_source.message,
            "requires_robot_frame_tip": bool(self.config.require_robot_frame_tip),
            "lower_trust_active": bool(lower_trust_active),
            "sample_order_preserved": True,
            "legacy_export_enabled": bool(self.config.export_legacy_dat),
            "position_frame": "robot" if robot_positions else "tracker",
            "command_pair_range_cm": _collect_pose_pair_range(resolved_pair_commands or requested_pair_commands),
            "workspace_span_mm": _collect_pose_workspace_span(robot_positions),
            "command_norm_stats_cm": _collect_pose_command_norm_stats(resolved_pair_commands or requested_pair_commands),
            "rejection_reasons": _collect_pose_rejection_counts(rejected_samples),
            "phase_counts": _collect_pose_phase_counts(samples),
            "summary_requirements": {
                "min_sample_count": 1,
                "require_registration": bool(self.config.require_robot_frame_tip and not self.config.dry_run and strict_runtime_tip),
                "registration_available": bool(robot_positions),
                "require_tip_calibration": bool(self.config.require_robot_frame_tip and not self.config.dry_run and strict_runtime_tip),
                "tip_calibration_available": bool(robot_positions),
                "allow_partial_missing_registration": bool(self.config.dry_run or not strict_runtime_tip),
                "allow_partial_missing_tip_cal": bool(self.config.dry_run or not strict_runtime_tip),
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
        )


def _configured_collect_pose_servo_ids(session: ExperimentSession) -> list[int]:
    robot = session.context.settings.robot
    servo_ids = [int(value) for value in (robot.tendon_to_servo or robot.servo_ids or [])]
    return list(servo_ids)


def _load_collect_pose_neutral_ticks(session: ExperimentSession, *, servo_ids: list[int]) -> list[int]:
    reference = session.context.servo_service.resolve_startup_reference_ticks(list(servo_ids))
    return [int(reference.ticks_by_servo[servo_id]) for servo_id in servo_ids if servo_id in reference.ticks_by_servo]


def _precheck_collect_pose_command_dataset(
    *,
    session: ExperimentSession,
    config: CollectPoseCommandDatasetConfig,
    servo_ids: list[int],
    neutral_ticks: list[int],
) -> None:
    tracking_service = session.context.tracking_service
    if tracking_service is None:
        raise RuntimeError("Motor Babble modeling dataset collection requires tracking_service.")
    if len(servo_ids) != 4:
        raise RuntimeError(f"Motor Babble modeling dataset collection requires exactly 4 configured servos; found {servo_ids}.")
    if not config.dry_run and not session.context.servo_service.is_connected:
        raise RuntimeError("Live Motor Babble modeling dataset collection requires a connected servo service.")
    if not config.dry_run and len(neutral_ticks) != len(servo_ids):
        raise RuntimeError("Live modeling data collection requires neutral setpoints for all 4 servos.")
    snapshot = tracking_service.get_snapshot()
    if not config.dry_run and bool(session.context.settings.runtime.mock_mode):
        raise RuntimeError("Thesis-grade modeling datasets require live runtime mode. Disable mock mode or switch to dry_run.")
    if (not config.dry_run) and snapshot.canonical_state != "streaming_healthy":
        raise RuntimeError(f"Tracker must be streaming healthy before modeling dataset collection; current state is {snapshot.canonical_state}.")
    runtime_tip_mode = str(getattr(snapshot, "runtime_tip_mode", "latest_accepted") or "latest_accepted")
    if config.require_robot_frame_tip and snapshot.registration_state != "loaded":
        raise RuntimeError("Accepted base registration must be loaded before modeling dataset collection.")
    if config.require_robot_frame_tip:
        if runtime_tip_mode != "latest_accepted" and not config.allow_lower_trust_runtime_tip:
            raise RuntimeError(
                "Motor Babble defaults to the trusted latest_accepted runtime tip mode. "
                f"Current mode is {runtime_tip_mode}; enable the lower-trust override explicitly if you intend to use it."
            )
        if snapshot.tip_pose_status != "ok" or snapshot.T_robot_tip is None:
            raise RuntimeError(
                f"Live robot-frame tip pose must be active before modeling dataset collection; tip pose status is {snapshot.tip_pose_status}."
            )
    gate = _modeling_tracker_gate_status(
        snapshot=snapshot,
        tool_id=str(config.tool_id or "0A"),
        max_tracker_age_s=float(config.max_tracker_age_s),
        require_robot_frame_tip=bool(config.require_robot_frame_tip),
        allow_mock_state=bool(config.dry_run),
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
        if (not pretension_source.accepted or not pretension_source.usable) and not config.allow_lower_trust_pretension:
            raise RuntimeError(
                "An accepted pretension/startup artifact is required before modeling dataset collection. "
                + pretension_source.message
            )
    if not config.dry_run:
        for servo_id in servo_ids:
            assessment = session.context.servo_service.assess_experiment_motion(int(servo_id))
            if not assessment.ready:
                raise RuntimeError(f"Servo {servo_id} is not ready for coordinated motion: {assessment.reason}")
    if config.dataset_mode not in {"workspace_coverage", "hysteresis_path_dependence", "repeatability_linked"}:
        if not config.command_points and not bool(config.legacy_schedule_override):
            raise RuntimeError(f"Unsupported modeling dataset mode: {config.dataset_mode}")


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
    return _build_collect_pose_repeatability_linked_steps(config=config, pair_limits=pair_limits)


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


def _wait_for_collect_pose_capture(
    *,
    session: ExperimentSession,
    tool_id: str,
    max_tracker_age_s: float,
    timeout_s: float,
    poll_interval_s: float,
    require_robot_frame_tip: bool,
    allow_mock_state: bool,
) -> tuple[Any, dict[str, Any]]:
    deadline = session.context.monotonic_fn() + float(timeout_s)
    last_snapshot = session.context.tracking_service.get_snapshot()
    last_gate = _modeling_tracker_gate_status(
        snapshot=last_snapshot,
        tool_id=tool_id,
        max_tracker_age_s=max_tracker_age_s,
        require_robot_frame_tip=require_robot_frame_tip,
        allow_mock_state=allow_mock_state,
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
        if snapshot.runtime_tip_calibration_state != "loaded":
            reasons.append(f"runtime_tip_state_{snapshot.runtime_tip_calibration_state}")
        if snapshot.tip_pose_status != "ok":
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
    }


def _record_collect_pose_run_provenance(
    *,
    session: ExperimentSession,
    config: CollectPoseCommandDatasetConfig,
    servo_ids: list[int],
    neutral_ticks: list[int],
    pair_limits: dict[str, Any],
) -> None:
    snapshot = session.context.tracking_service.get_snapshot()
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
    runtime_tip_path_raw = getattr(snapshot, "runtime_tip_calibration_path", None)
    runtime_tip_path = Path(runtime_tip_path_raw) if runtime_tip_path_raw else None
    gate = _modeling_tracker_gate_status(
        snapshot=snapshot,
        tool_id=str(config.tool_id or "0A"),
        max_tracker_age_s=float(config.max_tracker_age_s),
        require_robot_frame_tip=bool(config.require_robot_frame_tip),
        allow_mock_state=bool(config.dry_run),
    )
    provenance = {
        "dataset_mode": str(config.dataset_mode or "workspace_coverage"),
        "run_label": str(config.run_label or ""),
        "dataset_tag": str(config.dataset_tag or ""),
        "backend_identity": str(snapshot.backend_identity or ""),
        "selected_backend_name": str(snapshot.selected_backend_name or ""),
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
            "state": str(snapshot.registration_state or ""),
            "stored_timestamp_utc": getattr(snapshot, "stored_registration_timestamp_utc", None),
            "stored_fre_mm": getattr(snapshot, "stored_registration_fre_mm", None),
        },
        "runtime_tip_calibration": {
            **_collect_pose_file_provenance(runtime_tip_path),
            "state": str(snapshot.runtime_tip_calibration_state or ""),
            "mode": str(getattr(snapshot, "runtime_tip_mode", "latest_accepted") or "latest_accepted"),
            "trust_level": str(getattr(snapshot, "runtime_tip_trust_level", "missing") or "missing"),
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
            )
            else "ready",
            "tracker_gate": dict(gate),
            "tracking_state": str(snapshot.canonical_state or ""),
            "runtime_tip_mode": str(getattr(snapshot, "runtime_tip_mode", "latest_accepted") or "latest_accepted"),
            "pretension_source_type": pretension_source.source_type,
            "pretension_message": pretension_source.message,
        },
    }
    session.metadata.backend_info["run_provenance"] = {
        "dataset_mode": provenance["dataset_mode"],
        "run_label": provenance["run_label"],
        "dataset_tag": provenance["dataset_tag"],
        "backend_identity": provenance["backend_identity"],
        "selected_backend_name": provenance["selected_backend_name"],
        "pretension_artifact": provenance["pretension_artifact"],
        "run_start_preflight": provenance["run_start_preflight"],
    }
    session.metadata.registration_info["base_registration"] = provenance["base_registration"]
    session.metadata.registration_info["runtime_tip_calibration"] = provenance["runtime_tip_calibration"]
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
        factory=ReplayRunnerExperiment.from_dict,
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
        name=CollectPoseCommandDatasetExperiment.name,
        title="Motor Babble Modeling Dataset",
        description=CollectPoseCommandDatasetExperiment.description,
        category="dataset",
        tags=["Motor Babble", "Modeling", "Tracking", "Servo"],
        default_config_path="config/experiment_collect_pose_command_dataset.example.yaml",
        factory=CollectPoseCommandDatasetExperiment.from_dict,
    )
    register_calibration_validation_experiments(registry)
    register_critical_experiments(registry)


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
