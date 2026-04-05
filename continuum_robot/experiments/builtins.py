"""Built-in canonical experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any

from continuum_robot.experiments.dataset_io import ExperimentDatasetLoader, ExperimentDatasetWriter
from continuum_robot.experiments.experiment_models import ExperimentPoint
from continuum_robot.experiments.framework import BaseExperiment, ExperimentHardwareRequirements, ExperimentSession
from continuum_robot.experiments.critical_experiments import register_critical_experiments
from continuum_robot.experiments.schedules import (
    CommandScheduleConfig,
    command_schedule_checksum,
    generate_command_schedule,
)
from continuum_robot.experiments.sample_builders import sample_from_tracking_snapshot
from continuum_robot.experiments.schemas import ExperimentMetadata, ExperimentSummary, ExperimentTimeseriesSample
from continuum_robot.servos.servo_service import PretensionParameters

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
    """Config for repeated one-servo pretension validation."""

    servo_id: int = 1
    run_count: int = 3
    move_to_reference: bool = True
    accept_results: bool = False
    tracker_tool_id: str = "0A"
    validation_direction: str = "loosen"
    validation_delta_ticks: int = 6
    validation_settle_time_s: float = 0.05
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
    max_final_position_spread_ticks: int | None = None
    max_validation_displacement_spread_mm: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PretensionValidationExperimentConfig":
        payload = dict(payload or {})
        return cls(
            servo_id=int(payload.get("servo_id", 1)),
            run_count=max(1, int(payload.get("run_count", 3))),
            move_to_reference=bool(payload.get("move_to_reference", True)),
            accept_results=bool(payload.get("accept_results", False)),
            tracker_tool_id=str(payload.get("tracker_tool_id", "0A")),
            validation_direction=str(payload.get("validation_direction", "loosen")).strip().lower(),
            validation_delta_ticks=max(1, int(payload.get("validation_delta_ticks", 6))),
            validation_settle_time_s=float(payload.get("validation_settle_time_s", 0.05)),
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
            max_final_position_spread_ticks=(
                int(payload["max_final_position_spread_ticks"])
                if payload.get("max_final_position_spread_ticks") not in (None, "")
                else None
            ),
            max_validation_displacement_spread_mm=(
                float(payload["max_validation_displacement_spread_mm"])
                if payload.get("max_validation_displacement_spread_mm") not in (None, "")
                else None
            ),
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


class PretensionValidationExperiment(BaseExperiment):
    """Repeat one-servo pretension runs and compare servo + tracker-side consistency."""

    name = "pretension_validation"
    description = "Repeated one-servo pretension validation with tracker-side displacement checks."
    hardware_requirements = ExperimentHardwareRequirements(
        tracking_required=True,
        servo_required=True,
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
        if tracking_service is not None and getattr(tracking_service, "_thread", None) is None:
            tracking_service.start()
            self._tracking_started_here = True

    def precheck(self, session: ExperimentSession) -> None:
        if session.context.tracking_service is None:
            raise RuntimeError("pretension_validation requires tracking_service.")
        if not session.context.servo_service.is_connected:
            raise RuntimeError("pretension_validation requires a connected servo service.")
        if self.config.validation_direction not in {"tighten", "loosen"}:
            raise RuntimeError("pretension_validation requires validation_direction to be 'tighten' or 'loosen'.")

    def execute(self, session: ExperimentSession) -> None:
        servo_service = session.context.servo_service
        tracker_service = session.context.tracking_service
        validation_service = servo_service.pretension_validation
        parameters = self._pretension_parameters(session)
        run_records: list[dict[str, Any]] = []
        total = max(1, int(self.config.run_count))

        for run_index in range(total):
            session.raise_if_stop_requested()
            if self.config.move_to_reference:
                servo_service.move_servo_to_pretension_reference(
                    servo_id=int(self.config.servo_id),
                    parameters=parameters,
                )
            baseline = servo_service.measure_pretension_baseline(
                servo_id=int(self.config.servo_id),
                sample_count=int(parameters.baseline_sample_count),
                filter_window=int(parameters.current_filter_window),
                parameters=parameters,
            )
            pretension_result = servo_service.run_pretension_routine(
                servo_id=int(self.config.servo_id),
                parameters=parameters,
            )
            accepted = False
            if bool(self.config.accept_results) and pretension_result.success:
                servo_service.accept_pretension_result(int(self.config.servo_id))
                accepted = True

            before_snapshot = tracker_service.get_snapshot()
            validation_command = None
            validation_metric = validation_service.compute_tracker_displacement(
                before_snapshot,
                before_snapshot,
                tracker_tool_id=self.config.tracker_tool_id,
            )
            if pretension_result.final_position_tick is not None:
                direction_sign = 1 if self.config.validation_direction == "loosen" else -1
                target_tick = int(pretension_result.final_position_tick) + (
                    direction_sign * int(self.config.validation_delta_ticks)
                )
                validation_command = servo_service.move_servo_to_raw_target(
                    servo_id=int(self.config.servo_id),
                    target_tick=int(target_tick),
                    reason="pretension_validation",
                )
                if validation_command.blocked:
                    session.add_warning(validation_command.message)
                    validation_metric = validation_service.compute_tracker_displacement(
                        None,
                        None,
                        tracker_tool_id=self.config.tracker_tool_id,
                    )
                else:
                    if float(self.config.validation_settle_time_s) > 0.0:
                        session.context.sleep_fn(float(self.config.validation_settle_time_s))
                    after_snapshot = tracker_service.get_snapshot()
                    validation_metric = validation_service.compute_tracker_displacement(
                        before_snapshot,
                        after_snapshot,
                        tracker_tool_id=self.config.tracker_tool_id,
                    )

            run_record = {
                "run_index": int(run_index),
                "servo_id": int(self.config.servo_id),
                "pretension_success": bool(pretension_result.success),
                "accepted": bool(accepted),
                "baseline_current_ma": float(baseline.baseline_current_ma),
                "filtered_current_ma": float(pretension_result.filtered_current_ma or baseline.filtered_current_ma),
                "trigger_current_ma": int(pretension_result.threshold_ma),
                "final_position_tick": pretension_result.final_position_tick,
                "travel_used_ticks": (
                    None
                    if pretension_result.final_position_tick is None
                    or pretension_result.untensioned_reference_tick is None
                    else int(pretension_result.untensioned_reference_tick) - int(pretension_result.final_position_tick)
                ),
                "stop_reason": pretension_result.stop_reason,
                "validation_direction": self.config.validation_direction,
                "validation_delta_ticks": int(self.config.validation_delta_ticks),
                "validation_target_tick": (
                    validation_command.goal_tick
                    if validation_command is not None
                    else None
                ),
                "validation_displacement_mm": validation_metric.displacement_magnitude_mm,
                "validation_metric_frame": validation_metric.metric_frame,
                "parameters": {
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
            run_records.append(run_record)

            session.add_sample(
                _sample_from_tracking_snapshot(
                    session,
                    snapshot=before_snapshot,
                    phase="pretension_run",
                    step_index=int(run_index),
                    sample_index=0,
                    commanded_cable_deltas_cm=[],
                    commanded_motor_values={"servo_id": int(self.config.servo_id)},
                    status_flags=["pretension_validation"],
                    extra={
                        **run_record,
                        "baseline_samples_ma": list(baseline.samples_ma),
                    },
                )
            )
            session.add_sample(
                _sample_from_tracking_snapshot(
                    session,
                    snapshot=tracker_service.get_snapshot(),
                    phase="validation_motion",
                    step_index=int(run_index),
                    sample_index=1,
                    commanded_cable_deltas_cm=[],
                    commanded_motor_values={
                        "servo_id": int(self.config.servo_id),
                        "validation_target_tick": run_record["validation_target_tick"],
                    },
                    status_flags=["pretension_validation"],
                    extra={
                        "run_index": int(run_index),
                        "validation_metric_frame": validation_metric.metric_frame,
                        "validation_displacement_mm": validation_metric.displacement_magnitude_mm,
                        "validation_vector_mm": validation_metric.displacement_vector_mm,
                        "validation_message": validation_metric.message,
                        "pretension_stop_reason": pretension_result.stop_reason,
                    },
                )
            )
            session.update_progress(
                run_index + 1,
                total,
                {
                    "phase": "pretension_validation",
                    "run_index": int(run_index),
                    "pretension_status": pretension_result.status,
                },
            )
            if not pretension_result.success:
                session.add_warning(
                    f"Pretension validation run {run_index + 1} ended with {pretension_result.status}: {pretension_result.stop_reason or pretension_result.message}"
                )

        metrics = validation_service.summarize_validation_runs(run_records)
        metrics["runs"] = run_records
        if (
            self.config.max_final_position_spread_ticks is not None
            and metrics.get("final_position_spread_ticks") is not None
            and int(metrics["final_position_spread_ticks"]) > int(self.config.max_final_position_spread_ticks)
        ):
            session.add_warning(
                "Final pretension position spread exceeded the configured validation limit."
            )
        if (
            self.config.max_validation_displacement_spread_mm is not None
            and metrics.get("validation_displacement_spread_mm") is not None
            and float(metrics["validation_displacement_spread_mm"]) > float(self.config.max_validation_displacement_spread_mm)
        ):
            session.add_warning(
                "Tracker displacement spread exceeded the configured validation limit."
            )
        if any(not bool(record.get("pretension_success")) for record in run_records):
            metrics["summary_requirements"] = {"force_status": "partial_success"}
        session.metrics.update(metrics)

    def finalize(self, session: ExperimentSession) -> None:
        if self._tracking_started_here:
            session.context.tracking_service.stop()
            self._tracking_started_here = False

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
        description=TrackerPipelineMockExperiment.description,
        factory=TrackerPipelineMockExperiment.from_dict,
    )
    registry.register(
        name=TransformChainValidationExperiment.name,
        description=TransformChainValidationExperiment.description,
        factory=TransformChainValidationExperiment.from_dict,
    )
    registry.register(
        name=CommandScheduleValidationExperiment.name,
        description=CommandScheduleValidationExperiment.description,
        factory=CommandScheduleValidationExperiment.from_dict,
    )
    registry.register(
        name=DatasetSchemaRoundtripExperiment.name,
        description=DatasetSchemaRoundtripExperiment.description,
        factory=DatasetSchemaRoundtripExperiment.from_dict,
    )
    registry.register(
        name=ReplayRunnerExperiment.name,
        description=ReplayRunnerExperiment.description,
        factory=ReplayRunnerExperiment.from_dict,
    )
    registry.register(
        name=PretensionValidationExperiment.name,
        description=PretensionValidationExperiment.description,
        factory=PretensionValidationExperiment.from_dict,
    )
    registry.register(
        name=CollectPoseCommandDatasetExperiment.name,
        description=CollectPoseCommandDatasetExperiment.description,
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
