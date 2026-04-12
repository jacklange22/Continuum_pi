"""Built-in canonical experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
import threading
import time
from typing import Any

from continuum_robot.experiments.dataset_io import ExperimentDatasetLoader, ExperimentDatasetWriter
from continuum_robot.experiments.experiment_models import ExperimentPoint
from continuum_robot.experiments.framework import BaseExperiment, ExperimentHardwareRequirements, ExperimentSession
from continuum_robot.experiments.critical_experiments import register_critical_experiments
from continuum_robot.experiments.pretension_validation_outputs import write_pretension_validation_outputs
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
    compute_servo_sync_summary,
    compute_tracker_timing_summary,
    extract_servo_timing_records,
    extract_tracker_timing_records,
)

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is a declared dependency
    np = None


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
    """Config for the main command plus pose dataset collection experiment."""

    dry_run: bool = True
    sample_count_per_point: int = 1
    settle_time_s: float = 0.0
    command_points: list[dict[str, Any]] = field(default_factory=list)
    command_schedule: CommandScheduleConfig = field(
        default_factory=lambda: CommandScheduleConfig(kind="sweep", dimensions=4, amplitude_cm=0.2, steps_per_axis=3)
    )

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CollectPoseCommandDatasetConfig":
        payload = dict(payload or {})
        schedule_payload = payload.get("command_schedule") or payload.get("schedule")
        return cls(
            dry_run=bool(payload.get("dry_run", True)),
            sample_count_per_point=int(payload.get("sample_count_per_point", 1)),
            settle_time_s=float(payload.get("settle_time_s", 0.0)),
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
class PretensionValidationExperimentConfig:
    """Config for one-servo pretension response validation."""

    servo_id: int = 1
    move_to_reference: bool = True
    include_tracker_displacement: bool = True
    tracker_tool_id: str = "0A"
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
        return cls(
            servo_id=int(payload.get("servo_id", 1)),
            move_to_reference=bool(payload.get("move_to_reference", True)),
            include_tracker_displacement=bool(payload.get("include_tracker_displacement", True)),
            tracker_tool_id=str(payload.get("tracker_tool_id", "0A")),
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
                    self._build_servo_timing_sample(
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

    def _pretension_parameters(self, session: ExperimentSession) -> PretensionParameters:
        defaults = session.context.servo_service.default_pretension_parameters(int(self.config.servo_id))
        return PretensionParameters(
            untensioned_reference_tick=(
                int(self.config.untensioned_reference_tick)
                if self.config.untensioned_reference_tick is not None
                else int(defaults.untensioned_reference_tick)
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
    """Collect command plus pose datasets in dry-run/mock or live mode."""

    name = "collect_pose_command_dataset"
    description = "Canonical command plus pose dataset collection experiment."
    hardware_requirements = ExperimentHardwareRequirements(tracking_required=True, mock_compatible=True)

    def __init__(self, config: CollectPoseCommandDatasetConfig) -> None:
        super().__init__(config=config)
        self._tracking_started_here = False
        self._initial_neutral_ticks: list[int] = []

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "CollectPoseCommandDatasetExperiment":
        return cls(config=CollectPoseCommandDatasetConfig.from_dict(payload))

    def setup(self, session: ExperimentSession) -> None:
        tracking_service = session.context.tracking_service
        if tracking_service is not None and getattr(tracking_service, "_thread", None) is None:
            tracking_service.start()
            self._tracking_started_here = True
        self._initial_neutral_ticks = _load_neutral_ticks(session)

    def precheck(self, session: ExperimentSession) -> None:
        if not self.config.dry_run and not session.context.servo_service.is_connected:
            raise RuntimeError("collect_pose_command_dataset requires a connected servo service when dry_run=false.")
        if session.context.tracking_service is None:
            raise RuntimeError("collect_pose_command_dataset requires tracking_service.")

    def execute(self, session: ExperimentSession) -> None:
        points = (
            [
                ExperimentPoint(
                    index=int(point["index"]),
                    tendon_displacement_cm=[float(value) for value in point["tendon_displacement_cm"]],
                    settle_time_s=point.get("settle_time_s"),
                    repeat=int(point.get("repeat", 1)),
                    label=str(point.get("label", "")),
                )
                for point in self.config.command_points
            ]
            if self.config.command_points
            else generate_command_schedule(self.config.command_schedule)
        )
        samples_per_point = max(1, int(self.config.sample_count_per_point))
        total = 2 + len(points) * (2 + samples_per_point) + 1
        progress = 0
        dimensions = (
            len(points[0].tendon_displacement_cm)
            if points
            else int(self.config.command_schedule.dimensions)
        )
        zero_vector = [0.0] * dimensions
        neutral_payload = self._issue_command(session, zero_vector)
        session.add_sample(
            self._capture_sample(
                session,
                phase="setup",
                step_index=-1,
                sample_index=0,
                commanded_cable_deltas_cm=zero_vector,
                commanded_motor_values=neutral_payload,
                extra={"dry_run": self.config.dry_run},
            )
        )
        progress += 1
        session.update_progress(progress, total, {"phase": "setup", "step_index": -1})
        session.add_sample(
            self._capture_sample(
                session,
                phase="neutral_home",
                step_index=-1,
                sample_index=0,
                commanded_cable_deltas_cm=zero_vector,
                commanded_motor_values=neutral_payload,
            )
        )
        progress += 1
        session.update_progress(progress, total, {"phase": "neutral_home", "step_index": -1})
        for point in points:
            session.raise_if_stop_requested()
            command_payload = self._issue_command(session, point.tendon_displacement_cm)
            session.add_sample(
                self._capture_sample(
                    session,
                    phase="command_sequence",
                    step_index=point.index,
                    sample_index=0,
                    commanded_cable_deltas_cm=point.tendon_displacement_cm,
                    commanded_motor_values=command_payload,
                    extra={"label": point.label},
                )
            )
            progress += 1
            session.update_progress(progress, total, {"phase": "command_sequence", "step_index": point.index})
            settle_time_s = (
                float(point.settle_time_s)
                if point.settle_time_s is not None
                else float(self.config.settle_time_s)
            )
            if settle_time_s > 0:
                session.context.sleep_fn(settle_time_s)
            session.add_sample(
                self._capture_sample(
                    session,
                    phase="settle",
                    step_index=point.index,
                    sample_index=0,
                    commanded_cable_deltas_cm=point.tendon_displacement_cm,
                    commanded_motor_values=command_payload,
                )
            )
            progress += 1
            session.update_progress(progress, total, {"phase": "settle", "step_index": point.index})
            for sample_index in range(samples_per_point):
                session.raise_if_stop_requested()
                session.add_sample(
                    self._capture_sample(
                        session,
                        phase="sample",
                        step_index=point.index,
                        sample_index=sample_index,
                        commanded_cable_deltas_cm=point.tendon_displacement_cm,
                        commanded_motor_values=command_payload,
                    )
                )
                progress += 1
                session.update_progress(progress, total, {"phase": "sample", "step_index": point.index})
        final_payload = self._issue_command(session, zero_vector)
        session.add_sample(
            self._capture_sample(
                session,
                phase="finalize",
                step_index=len(points),
                sample_index=0,
                commanded_cable_deltas_cm=zero_vector,
                commanded_motor_values=final_payload,
            )
        )
        progress += 1
        session.update_progress(progress, total, {"phase": "finalize", "step_index": len(points)})
        session.set_metric("schedule_checksum", command_schedule_checksum(points))
        session.set_metric("schedule_point_count", len(points))
        session.set_metric("samples_per_point", samples_per_point)
        session.set_metric("registration_loaded", session.context.registration_path.exists())
        session.set_metric("dry_run", self.config.dry_run)

    def finalize(self, session: ExperimentSession) -> None:
        try:
            if not self.config.dry_run and session.context.servo_service.is_connected and self._initial_neutral_ticks:
                dimensions = int(self.config.command_schedule.dimensions)
                if self.config.command_points:
                    dimensions = len(self.config.command_points[0]["tendon_displacement_cm"])
                zero_vector = [0.0] * dimensions
                self._issue_command(session, zero_vector)
        finally:
            if self._tracking_started_here:
                session.context.tracking_service.stop()
                self._tracking_started_here = False

    def _issue_command(self, session: ExperimentSession, tendon_displacement_cm: list[float]) -> dict[str, int | float | None]:
        servo_service = session.context.servo_service
        servo_ids = list(session.context.settings.robot.servo_ids)
        neutral_ticks = list(self._initial_neutral_ticks)
        if not neutral_ticks:
            neutral_ticks = [0 for _ in servo_ids]
        if self.config.dry_run or not servo_service.is_connected:
            if len(neutral_ticks) == len(tendon_displacement_cm):
                goals = servo_service.mapper.to_goal_positions(tendon_displacement_cm, neutral_ticks)
                return {str(servo_id): int(goal) for servo_id, goal in zip(servo_ids, goals)}
            return {}
        command = servo_service.command_displacement(
            tendon_displacements_cm=[float(value) for value in tendon_displacement_cm],
            neutral_ticks=neutral_ticks,
            servo_ids=servo_ids,
        )
        return {str(servo_id): int(goal) for servo_id, goal in command.positions_by_id.items()}

    def _capture_sample(
        self,
        session: ExperimentSession,
        *,
        phase: str,
        step_index: int,
        sample_index: int,
        commanded_cable_deltas_cm: list[float],
        commanded_motor_values: dict[str, int | float | None],
        extra: dict[str, Any] | None = None,
    ) -> ExperimentTimeseriesSample:
        snapshot = session.context.tracking_service.get_snapshot()
        sample = _sample_from_tracking_snapshot(
            session,
            snapshot=snapshot,
            phase=phase,
            step_index=step_index,
            sample_index=sample_index,
            commanded_cable_deltas_cm=commanded_cable_deltas_cm,
            commanded_motor_values=commanded_motor_values,
            status_flags=["dry_run"] if self.config.dry_run else [],
            extra=extra or {},
        )
        if snapshot.registration_state != "loaded":
            if "registration_missing" not in sample.status_flags:
                sample.status_flags.append("registration_missing")
        elif snapshot.T_robot_tip is not None and "full_pose_available" not in sample.status_flags:
            sample.status_flags.append("full_pose_available")
        return sample


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
        title="Collect Pose Command Dataset",
        description=CollectPoseCommandDatasetExperiment.description,
        category="dataset",
        tags=["Commands", "Tracking", "Servo"],
        default_config_path="config/experiment_collect_pose_command_dataset.example.yaml",
        factory=CollectPoseCommandDatasetExperiment.from_dict,
    )
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
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
