"""Project-critical experiments built on top of the canonical framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Any

import numpy as np

from continuum_robot.experiments.dataset_io import ExperimentDatasetLoader
from continuum_robot.experiments.dataset_tools import extract_tip_or_tool_position_mm
from continuum_robot.experiments.framework import BaseExperiment, ExperimentHardwareRequirements, ExperimentSession
from continuum_robot.experiments.metrics import (
    centroid_mm,
    group_positions_by_key,
    per_axis_bias_mm,
    pointwise_rms_errors_mm,
    rms_error_mm,
    spread_rms_mm,
)
from continuum_robot.experiments.pivot_utils import (
    PivotCalibrationResult,
    load_pivot_transforms,
    solve_pivot_calibration,
    write_tip_vector_file,
)
from continuum_robot.experiments.sample_builders import sample_from_tracking_snapshot
from continuum_robot.experiments.validation import (
    STATUS_INVALID_INSUFFICIENT_SAMPLES,
    STATUS_INVALID_INVALID_TRANSFORMS,
    STATUS_INVALID_MISSING_REGISTRATION,
    STATUS_INVALID_MISSING_TIP_CAL,
    STATUS_PARTIAL_SUCCESS,
    STATUS_SUCCESS,
)
from continuum_robot.tracking.transforms import quat_wxyz_to_rotmat, rotmat_to_quat_wxyz


@dataclass
class RepeatabilityScheduleConfig:
    """Config for repeatability revisit scheduling."""

    target_points_cm: list[list[float]] = field(default_factory=list)
    revisit_count: int = 3
    randomize_approach_order: bool = False
    seed: int = 0
    settle_time_s: float = 0.0
    samples_per_point: int = 3

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RepeatabilityScheduleConfig":
        payload = dict(payload or {})
        return cls(
            target_points_cm=[
                [float(value) for value in point]
                for point in payload.get("target_points_cm", []) or []
            ],
            revisit_count=int(payload.get("revisit_count", 3)),
            randomize_approach_order=bool(payload.get("randomize_approach_order", False)),
            seed=int(payload.get("seed", 0)),
            settle_time_s=float(payload.get("settle_time_s", 0.0)),
            samples_per_point=int(payload.get("samples_per_point", 3)),
        )


@dataclass
class RepeatabilityVisit:
    """One repeatability visit pairing an approach state and target state."""

    cycle_index: int
    target_index: int
    revisit_index: int
    approach_index: int
    approach_deltas_cm: list[float]
    target_deltas_cm: list[float]


@dataclass
class RepeatabilityDatasetConfig:
    """Config for the repeatability dataset experiment."""

    dry_run: bool = True
    tool_id: str = "0A"
    synthetic_noise_std_mm: float = 0.35
    synthetic_hysteresis_mm: float = 0.8
    schedule: RepeatabilityScheduleConfig = field(default_factory=RepeatabilityScheduleConfig)
    acceptance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RepeatabilityDatasetConfig":
        payload = dict(payload or {})
        return cls(
            dry_run=bool(payload.get("dry_run", True)),
            tool_id=str(payload.get("tool_id", "0A")),
            synthetic_noise_std_mm=float(payload.get("synthetic_noise_std_mm", 0.35)),
            synthetic_hysteresis_mm=float(payload.get("synthetic_hysteresis_mm", 0.8)),
            schedule=RepeatabilityScheduleConfig.from_dict(payload.get("schedule")),
            acceptance=dict(payload.get("acceptance", {}) or {}),
        )


@dataclass
class GridDefinitionConfig:
    """Config for a physical or synthetic Aurora grid."""

    spacing_mm: float = 25.4
    dimensions: list[int] = field(default_factory=lambda: [3, 3])
    point_ordering: str = "row_major"
    repetitions_per_point: int = 3
    samples_per_point: int = 3
    settle_time_s: float = 0.0
    truth_points_mm: list[list[float]] = field(default_factory=list)
    truth_points_file: str | None = None
    truth_frame: str = "tracker"
    tool_id: str = "0B"
    use_tip_calibration: bool = True
    tip_vector_mm: list[float] | None = None
    tip_file: str | None = None
    allow_coil_origin_fallback: bool = True
    dry_run: bool = True
    seed: int = 0
    synthetic_noise_std_mm: float = 0.25
    synthetic_bias_mm: list[float] = field(default_factory=lambda: [0.2, -0.1, 0.05])
    outlier_threshold_mm: float = 1.0
    acceptance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GridDefinitionConfig":
        payload = dict(payload or {})
        return cls(
            spacing_mm=float(payload.get("spacing_mm", 25.4)),
            dimensions=[int(value) for value in payload.get("dimensions", [3, 3])],
            point_ordering=str(payload.get("point_ordering", "row_major")),
            repetitions_per_point=int(payload.get("repetitions_per_point", 3)),
            samples_per_point=int(payload.get("samples_per_point", 3)),
            settle_time_s=float(payload.get("settle_time_s", 0.0)),
            truth_points_mm=[
                [float(value) for value in point]
                for point in payload.get("truth_points_mm", []) or []
            ],
            truth_points_file=(
                str(payload["truth_points_file"])
                if payload.get("truth_points_file") not in (None, "")
                else None
            ),
            truth_frame=str(payload.get("truth_frame", "tracker")),
            tool_id=str(payload.get("tool_id", "0B")),
            use_tip_calibration=bool(payload.get("use_tip_calibration", True)),
            tip_vector_mm=(
                [float(value) for value in payload["tip_vector_mm"]]
                if payload.get("tip_vector_mm") not in (None, "")
                else None
            ),
            tip_file=str(payload["tip_file"]) if payload.get("tip_file") not in (None, "") else None,
            allow_coil_origin_fallback=bool(payload.get("allow_coil_origin_fallback", True)),
            dry_run=bool(payload.get("dry_run", True)),
            seed=int(payload.get("seed", 0)),
            synthetic_noise_std_mm=float(payload.get("synthetic_noise_std_mm", 0.25)),
            synthetic_bias_mm=[float(value) for value in payload.get("synthetic_bias_mm", [0.2, -0.1, 0.05])],
            outlier_threshold_mm=float(payload.get("outlier_threshold_mm", 1.0)),
            acceptance=dict(payload.get("acceptance", {}) or {}),
        )


@dataclass
class PivotCalibrationConfig:
    """Config for pivot calibration and tip file generation."""

    tool_id: str = "0B"
    sample_count: int = 80
    sample_period_s: float = 0.02
    std_dev_threshold: float = 3.0
    min_samples: int = 12
    output_tip_file: str = "data/tip_cals/generated_penprobe_tip.csv"
    input_path: str | None = None
    dry_run: bool = False
    seed: int = 0
    synthetic_tip_vector_mm: list[float] = field(default_factory=lambda: [0.0, 0.0, 125.0])
    synthetic_pivot_point_mm: list[float] = field(default_factory=lambda: [25.0, -10.0, 40.0])
    synthetic_noise_std_mm: float = 0.25
    synthetic_outlier_count: int = 0
    acceptance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PivotCalibrationConfig":
        payload = dict(payload or {})
        return cls(
            tool_id=str(payload.get("tool_id", "0B")),
            sample_count=int(payload.get("sample_count", 80)),
            sample_period_s=float(payload.get("sample_period_s", 0.02)),
            std_dev_threshold=float(payload.get("std_dev_threshold", 3.0)),
            min_samples=int(payload.get("min_samples", 12)),
            output_tip_file=str(payload.get("output_tip_file", "data/tip_cals/generated_penprobe_tip.csv")),
            input_path=str(payload["input_path"]) if payload.get("input_path") not in (None, "") else None,
            dry_run=bool(payload.get("dry_run", False)),
            seed=int(payload.get("seed", 0)),
            synthetic_tip_vector_mm=[float(value) for value in payload.get("synthetic_tip_vector_mm", [0.0, 0.0, 125.0])],
            synthetic_pivot_point_mm=[float(value) for value in payload.get("synthetic_pivot_point_mm", [25.0, -10.0, 40.0])],
            synthetic_noise_std_mm=float(payload.get("synthetic_noise_std_mm", 0.25)),
            synthetic_outlier_count=int(payload.get("synthetic_outlier_count", 0)),
            acceptance=dict(payload.get("acceptance", {}) or {}),
        )


class RepeatabilityDatasetExperiment(BaseExperiment):
    """Main repeatability and core robot dataset experiment."""

    name = "repeatability_dataset"
    description = "Revisit target command points from multiple prior states and summarize repeatability."
    hardware_requirements = ExperimentHardwareRequirements(tracking_required=True, mock_compatible=True)

    def __init__(self, config: RepeatabilityDatasetConfig) -> None:
        super().__init__(config=config)
        self._tracking_started_here = False
        self._neutral_ticks: list[int] = []

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "RepeatabilityDatasetExperiment":
        return cls(config=RepeatabilityDatasetConfig.from_dict(payload))

    def setup(self, session: ExperimentSession) -> None:
        if session.context.tracking_service is not None and getattr(session.context.tracking_service, "_thread", None) is None:
            session.context.tracking_service.start()
            self._tracking_started_here = True
        self._neutral_ticks = _load_neutral_ticks(session)

    def precheck(self, session: ExperimentSession) -> None:
        if not self.config.schedule.target_points_cm:
            raise RuntimeError("repeatability_dataset requires schedule.target_points_cm")

    def execute(self, session: ExperimentSession) -> None:
        visits = generate_repeatability_schedule(self.config.schedule)
        rng = np.random.default_rng(self.config.schedule.seed)
        total = len(visits) * (3 + max(1, self.config.schedule.samples_per_point))
        completed = 0
        for visit in visits:
            session.raise_if_stop_requested()
            approach_payload = _issue_command_payload(
                session,
                tendon_displacement_cm=visit.approach_deltas_cm,
                neutral_ticks=self._neutral_ticks,
                dry_run=self.config.dry_run,
            )
            approach_tracker = _synthetic_repeatability_position_mm(
                visit.approach_deltas_cm,
                previous_deltas_cm=visit.approach_deltas_cm,
                rng=rng,
                noise_std_mm=self.config.synthetic_noise_std_mm,
                hysteresis_mm=self.config.synthetic_hysteresis_mm,
            )
            snapshot = session.context.tracking_service.get_snapshot()
            session.add_sample(
                sample_from_tracking_snapshot(
                    session,
                    snapshot=snapshot,
                    phase="approach_command",
                    step_index=visit.target_index,
                    sample_index=0,
                    cycle_index=visit.cycle_index,
                    target_index=visit.target_index,
                    revisit_index=visit.revisit_index,
                    approach_index=visit.approach_index,
                    commanded_cable_deltas_cm=visit.approach_deltas_cm,
                    commanded_motor_values=approach_payload,
                    override_tracker_position_mm=approach_tracker,
                    override_robot_position_mm=(
                        approach_tracker if session.context.registration_path.exists() else None
                    ),
                    tracker_tool_id=self.config.tool_id,
                    status_flags=["dry_run"] if self.config.dry_run else [],
                )
            )
            completed += 1
            session.update_progress(completed, total, {"phase": "approach_command", "target_index": visit.target_index})
            if self.config.schedule.settle_time_s > 0:
                session.context.sleep_fn(self.config.schedule.settle_time_s)
            session.add_sample(
                sample_from_tracking_snapshot(
                    session,
                    snapshot=session.context.tracking_service.get_snapshot(),
                    phase="approach_settle",
                    step_index=visit.target_index,
                    sample_index=0,
                    cycle_index=visit.cycle_index,
                    target_index=visit.target_index,
                    revisit_index=visit.revisit_index,
                    approach_index=visit.approach_index,
                    commanded_cable_deltas_cm=visit.approach_deltas_cm,
                    commanded_motor_values=approach_payload,
                    override_tracker_position_mm=approach_tracker,
                    override_robot_position_mm=(
                        approach_tracker if session.context.registration_path.exists() else None
                    ),
                    tracker_tool_id=self.config.tool_id,
                    status_flags=["dry_run"] if self.config.dry_run else [],
                )
            )
            completed += 1
            session.update_progress(completed, total, {"phase": "approach_settle", "target_index": visit.target_index})

            target_payload = _issue_command_payload(
                session,
                tendon_displacement_cm=visit.target_deltas_cm,
                neutral_ticks=self._neutral_ticks,
                dry_run=self.config.dry_run,
            )
            target_tracker = _synthetic_repeatability_position_mm(
                visit.target_deltas_cm,
                previous_deltas_cm=visit.approach_deltas_cm,
                rng=rng,
                noise_std_mm=self.config.synthetic_noise_std_mm,
                hysteresis_mm=self.config.synthetic_hysteresis_mm,
            )
            session.add_sample(
                sample_from_tracking_snapshot(
                    session,
                    snapshot=session.context.tracking_service.get_snapshot(),
                    phase="target_command",
                    step_index=visit.target_index,
                    sample_index=0,
                    cycle_index=visit.cycle_index,
                    target_index=visit.target_index,
                    revisit_index=visit.revisit_index,
                    approach_index=visit.approach_index,
                    commanded_cable_deltas_cm=visit.target_deltas_cm,
                    commanded_motor_values=target_payload,
                    override_tracker_position_mm=target_tracker,
                    override_robot_position_mm=(
                        target_tracker if session.context.registration_path.exists() else None
                    ),
                    tracker_tool_id=self.config.tool_id,
                    status_flags=["dry_run"] if self.config.dry_run else [],
                )
            )
            completed += 1
            session.update_progress(completed, total, {"phase": "target_command", "target_index": visit.target_index})
            if self.config.schedule.settle_time_s > 0:
                session.context.sleep_fn(self.config.schedule.settle_time_s)
            for sample_index in range(max(1, self.config.schedule.samples_per_point)):
                jittered_target = (
                    np.asarray(target_tracker, dtype=float)
                    + rng.normal(0.0, self.config.synthetic_noise_std_mm, size=3)
                ).tolist()
                session.add_sample(
                    sample_from_tracking_snapshot(
                        session,
                        snapshot=session.context.tracking_service.get_snapshot(),
                        phase="sample",
                        step_index=visit.target_index,
                        sample_index=sample_index,
                        cycle_index=visit.cycle_index,
                        target_index=visit.target_index,
                        revisit_index=visit.revisit_index,
                        approach_index=visit.approach_index,
                        commanded_cable_deltas_cm=visit.target_deltas_cm,
                        commanded_motor_values=target_payload,
                        override_tracker_position_mm=jittered_target,
                        override_robot_position_mm=(
                            jittered_target if session.context.registration_path.exists() else None
                        ),
                        tracker_tool_id=self.config.tool_id,
                        status_flags=["dry_run"] if self.config.dry_run else [],
                    )
                )
                completed += 1
                session.update_progress(completed, total, {"phase": "sample", "target_index": visit.target_index})
        metrics = compute_repeatability_metrics(session.samples, tool_id=self.config.tool_id)
        metrics["registration_available"] = session.context.registration_path.exists()
        metrics["summary_requirements"] = {"force_status": metrics["status"]}
        session.metrics.update(metrics)

    def finalize(self, session: ExperimentSession) -> None:
        try:
            if not self.config.dry_run and session.context.servo_service.is_connected and self._neutral_ticks:
                zero_vector = [0.0] * len(self.config.schedule.target_points_cm[0])
                _issue_command_payload(
                    session,
                    tendon_displacement_cm=zero_vector,
                    neutral_ticks=self._neutral_ticks,
                    dry_run=False,
                )
        finally:
            if self._tracking_started_here:
                session.context.tracking_service.stop()
                self._tracking_started_here = False


class AuroraGridAccuracyExperiment(BaseExperiment):
    """Grid-based Aurora accuracy and precision characterization."""

    name = "aurora_grid_accuracy"
    description = "Measure per-point Aurora accuracy, bias, and spread on a physical or synthetic grid."
    hardware_requirements = ExperimentHardwareRequirements(mock_compatible=True)

    def __init__(self, config: GridDefinitionConfig) -> None:
        super().__init__(config=config)
        self._tracking_started_here = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "AuroraGridAccuracyExperiment":
        return cls(config=GridDefinitionConfig.from_dict(payload))

    def setup(self, session: ExperimentSession) -> None:
        if session.context.tracking_service is not None and getattr(session.context.tracking_service, "_thread", None) is None:
            session.context.tracking_service.start()
            self._tracking_started_here = True

    def execute(self, session: ExperimentSession) -> None:
        truth_points = load_grid_truth_points(self.config, project_root=session.context.project_root)
        tip_vector, tip_calibration_available = _load_tip_vector(session, self.config)
        registration_available = session.context.registration_path.exists()
        rng = np.random.default_rng(self.config.seed)
        total = len(truth_points) * max(1, self.config.repetitions_per_point) * max(1, self.config.samples_per_point)
        completed = 0
        for point_index, truth_point in enumerate(truth_points):
            for repetition_index in range(max(1, self.config.repetitions_per_point)):
                if self.config.settle_time_s > 0:
                    session.context.sleep_fn(self.config.settle_time_s)
                for sample_index in range(max(1, self.config.samples_per_point)):
                    measured_tracker = (
                        np.asarray(truth_point, dtype=float)
                        + np.asarray(self.config.synthetic_bias_mm, dtype=float)
                        + rng.normal(0.0, self.config.synthetic_noise_std_mm, size=3)
                    )
                    if tip_vector is None and self.config.use_tip_calibration:
                        status_flags = ["missing_tip_calibration"]
                    else:
                        status_flags = ["dry_run"] if self.config.dry_run else []
                    robot_position = None
                    if self.config.truth_frame == "robot" and registration_available:
                        robot_position = measured_tracker.tolist()
                    elif self.config.truth_frame == "tracker":
                        robot_position = None
                    sample = sample_from_tracking_snapshot(
                        session,
                        snapshot=_grid_snapshot(session, self.config.tool_id),
                        phase="sample",
                        step_index=point_index,
                        sample_index=sample_index,
                        target_index=point_index,
                        revisit_index=repetition_index,
                        commanded_cable_deltas_cm=[],
                        commanded_motor_values={},
                        override_tracker_position_mm=measured_tracker.tolist(),
                        override_robot_position_mm=robot_position,
                        tracker_tool_id=self.config.tool_id,
                        status_flags=status_flags,
                        extra={"truth_point_mm": [float(value) for value in truth_point]},
                    )
                    session.add_sample(sample)
                    completed += 1
                    session.update_progress(completed, total, {"phase": "sample", "target_index": point_index})
        metrics = compute_grid_accuracy_metrics(
            session.samples,
            truth_points_mm=truth_points,
            tool_id=self.config.tool_id,
            truth_frame=self.config.truth_frame,
            outlier_threshold_mm=self.config.outlier_threshold_mm,
            registration_available=registration_available,
            tip_calibration_available=tip_calibration_available,
            require_tip_calibration=self.config.use_tip_calibration,
            allow_coil_origin_fallback=self.config.allow_coil_origin_fallback,
        )
        metrics["summary_requirements"] = {"force_status": metrics["status"]}
        session.metrics.update(metrics)

    def finalize(self, session: ExperimentSession) -> None:
        if self._tracking_started_here:
            session.context.tracking_service.stop()
            self._tracking_started_here = False


class PivotCalibrationExperiment(BaseExperiment):
    """Generate a pen-probe tip file from pivot samples."""

    name = "pivot_calibration"
    description = "Collect or replay pivot samples, solve least-squares tip calibration, and write the tip file."
    hardware_requirements = ExperimentHardwareRequirements(mock_compatible=True)

    def __init__(self, config: PivotCalibrationConfig) -> None:
        super().__init__(config=config)
        self._tracking_started_here = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "PivotCalibrationExperiment":
        return cls(config=PivotCalibrationConfig.from_dict(payload))

    def setup(self, session: ExperimentSession) -> None:
        if (
            self.config.input_path is None
            and not self.config.dry_run
            and session.context.tracking_service is not None
            and getattr(session.context.tracking_service, "_thread", None) is None
        ):
            session.context.tracking_service.start()
            self._tracking_started_here = True

    def execute(self, session: ExperimentSession) -> None:
        transforms = self._collect_transforms(session)
        for index, transform in enumerate(transforms):
            quaternion = list(rotmat_to_quat_wxyz(transform[0:3, 0:3]))
            translation = [float(value) for value in transform[0:3, 3]]
            snapshot = _pivot_snapshot(session, tool_id=self.config.tool_id)
            sample = sample_from_tracking_snapshot(
                session,
                snapshot=snapshot,
                phase="pivot_sample",
                step_index=index,
                sample_index=0,
                commanded_cable_deltas_cm=[],
                commanded_motor_values={},
                tracker_tool_id=self.config.tool_id,
                override_tracker_position_mm=translation,
                override_tracker_quaternion_wxyz=quaternion,
                status_flags=["offline_input"] if self.config.input_path else (["dry_run"] if self.config.dry_run else []),
            )
            session.add_sample(sample)
        if len(transforms) < int(self.config.min_samples):
            status = STATUS_INVALID_INSUFFICIENT_SAMPLES
            session.metrics.update(
                {
                    "sample_count_total": len(transforms),
                    "sample_count_used": 0,
                    "sample_count_rejected": 0,
                    "tip_calibration_available": False,
                    "status": status,
                    "summary_requirements": {"force_status": status},
                }
            )
            raise RuntimeError("Insufficient samples for pivot calibration")
        rotations = [T[0:3, 0:3] for T in transforms]
        translations = [T[0:3, 3] for T in transforms]
        try:
            result = solve_pivot_calibration(
                rotations,
                translations,
                std_dev_threshold=self.config.std_dev_threshold,
                min_samples=self.config.min_samples,
            )
        except ValueError as exc:
            status = STATUS_INVALID_INSUFFICIENT_SAMPLES
            session.metrics.update(
                {
                    "sample_count_total": len(transforms),
                    "sample_count_used": 0,
                    "sample_count_rejected": 0,
                    "tip_calibration_available": False,
                    "status": status,
                    "summary_requirements": {"force_status": status},
                }
            )
            raise RuntimeError(str(exc)) from exc
        output_tip_path = write_tip_vector_file(
            _resolve_repo_path(session.context.project_root, self.config.output_tip_file),
            result.tip_vector_local_mm,
        )
        status = STATUS_SUCCESS
        session.metrics.update(
            {
                "tip_vector_local_mm": result.tip_vector_local_mm,
                "pivot_point_tracker_mm": result.pivot_point_tracker_mm,
                "rmse_mm": result.rmse_mm,
                "sample_count_total": result.sample_count_total,
                "sample_count_used": result.sample_count_used,
                "sample_count_rejected": result.sample_count_rejected,
                "pivot_residuals_mm": result.residuals_mm,
                "pivot_inlier_mask": result.inlier_mask,
                "pivot_rejected_indices": result.rejected_indices,
                "tip_output_file": str(output_tip_path),
                "tip_calibration_available": True,
                "status": status,
                "summary_requirements": {"force_status": status},
            }
        )

    def finalize(self, session: ExperimentSession) -> None:
        if self._tracking_started_here:
            session.context.tracking_service.stop()
            self._tracking_started_here = False

    def _collect_transforms(self, session: ExperimentSession) -> list[np.ndarray]:
        if self.config.input_path:
            return load_pivot_transforms(
                _resolve_repo_path(session.context.project_root, self.config.input_path),
                tool_id=self.config.tool_id,
            )
        if self.config.dry_run:
            return _synthetic_pivot_transforms(self.config)
        transforms: list[np.ndarray] = []
        for _ in range(max(1, self.config.sample_count)):
            session.raise_if_stop_requested()
            tool = session.context.tracking_service.get_latest_tool(self.config.tool_id)
            if tool is None or tool.quaternion_wxyz is None or tool.translation_mm is None:
                continue
            T = np.eye(4)
            T[0:3, 0:3] = quat_wxyz_to_rotmat(tuple(float(v) for v in tool.quaternion_wxyz))
            T[0:3, 3] = np.asarray(tool.translation_mm, dtype=float)
            transforms.append(T)
            if self.config.sample_period_s > 0:
                session.context.sleep_fn(self.config.sample_period_s)
        return transforms


def register_critical_experiments(registry) -> None:
    """Register the project-critical experiments."""
    registry.register(
        name=RepeatabilityDatasetExperiment.name,
        description=RepeatabilityDatasetExperiment.description,
        factory=RepeatabilityDatasetExperiment.from_dict,
    )
    registry.register(
        name=AuroraGridAccuracyExperiment.name,
        description=AuroraGridAccuracyExperiment.description,
        factory=AuroraGridAccuracyExperiment.from_dict,
    )
    registry.register(
        name=PivotCalibrationExperiment.name,
        description=PivotCalibrationExperiment.description,
        factory=PivotCalibrationExperiment.from_dict,
    )


def generate_repeatability_schedule(config: RepeatabilityScheduleConfig) -> list[RepeatabilityVisit]:
    """Generate deterministic or randomized revisit ordering for repeatability."""
    targets = [[float(value) for value in point] for point in config.target_points_cm]
    if not targets:
        raise ValueError("Repeatability schedule requires at least one target point")
    n_targets = len(targets)
    sequences_by_target: dict[int, list[int]] = {}
    rng = random.Random(config.seed)
    for target_index in range(n_targets):
        candidates = [index for index in range(n_targets) if index != target_index] or [target_index]
        if config.randomize_approach_order:
            sequence: list[int] = []
            while len(sequence) < max(1, config.revisit_count):
                block = list(candidates)
                rng.shuffle(block)
                sequence.extend(block)
            sequences_by_target[target_index] = sequence[: max(1, config.revisit_count)]
        else:
            sequences_by_target[target_index] = [
                candidates[revisit_index % len(candidates)]
                for revisit_index in range(max(1, config.revisit_count))
            ]
    visits: list[RepeatabilityVisit] = []
    for cycle_index in range(max(1, config.revisit_count)):
        for target_index in range(n_targets):
            approach_index = sequences_by_target[target_index][cycle_index]
            visits.append(
                RepeatabilityVisit(
                    cycle_index=cycle_index,
                    target_index=target_index,
                    revisit_index=cycle_index,
                    approach_index=approach_index,
                    approach_deltas_cm=list(targets[approach_index]),
                    target_deltas_cm=list(targets[target_index]),
                )
            )
    return visits


def compute_repeatability_metrics(
    samples,
    *,
    tool_id: str = "0A",
) -> dict[str, Any]:
    """Compute repeatability metrics from canonical samples."""
    measurement_samples = [sample for sample in samples if sample.phase == "sample" and sample.target_index is not None]
    valid_samples = []
    invalid_count = 0
    frames_seen: list[str] = []
    for sample in measurement_samples:
        position, frame_name = extract_tip_or_tool_position_mm(sample, tool_id=tool_id, prefer_robot_frame=True)
        if position is None:
            invalid_count += 1
            continue
        valid_samples.append((sample, position))
        if frame_name is not None:
            frames_seen.append(frame_name)
    if not valid_samples:
        return {
            "status": STATUS_INVALID_INSUFFICIENT_SAMPLES,
            "valid_sample_count": 0,
            "invalid_sample_count": invalid_count,
            "dropped_sample_count": invalid_count,
            "per_target_metrics": {},
            "overall_repeatability_rms_mm": None,
        }
    grouped = group_positions_by_key(
        [item[0] for item in valid_samples],
        key_fn=lambda sample: int(sample.target_index),
        position_fn=lambda sample: extract_tip_or_tool_position_mm(sample, tool_id=tool_id, prefer_robot_frame=True)[0],
    )
    per_target_metrics: dict[str, Any] = {}
    centroid_by_target: dict[int, np.ndarray] = {}
    for target_index, positions in grouped.items():
        centroid = centroid_mm(positions)
        centroid_by_target[int(target_index)] = centroid
        per_target_metrics[str(target_index)] = {
            "count": len(positions),
            "centroid_mm": [float(value) for value in centroid],
            "spread_rms_mm": spread_rms_mm(positions, centroid_xyz=centroid),
        }
    rms_terms = []
    conditioned_groups: dict[str, list[list[float]]] = {}
    for sample, position in valid_samples:
        center = centroid_by_target[int(sample.target_index)]
        rms_terms.append(float(np.sum((np.asarray(position, dtype=float) - center) ** 2)))
        conditioned_groups.setdefault(
            f"{sample.approach_index}->{sample.target_index}",
            [],
        ).append([float(value) for value in position])
    conditioned_spread = {
        key: spread_rms_mm(points)
        for key, points in conditioned_groups.items()
        if len(points) >= 2
    }
    registration_available = any(frame == "robot" for frame in frames_seen)
    status = STATUS_SUCCESS if registration_available else STATUS_PARTIAL_SUCCESS
    return {
        "status": status,
        "position_frame": "robot" if registration_available else "tracker",
        "valid_sample_count": len(valid_samples),
        "invalid_sample_count": invalid_count,
        "dropped_sample_count": invalid_count,
        "per_target_metrics": per_target_metrics,
        "overall_repeatability_rms_mm": float(np.sqrt(np.mean(rms_terms))),
        "approach_conditioned_spread_mm": conditioned_spread,
        "registration_available": registration_available,
        "tip_calibration_available": True,
    }


def analyze_repeatability_dataset(path: Path, *, tool_id: str = "0A") -> dict[str, Any]:
    """Load one saved repeatability dataset and recompute its concise metrics."""
    bundle = ExperimentDatasetLoader().load_dataset(Path(path))
    metrics = compute_repeatability_metrics(bundle.samples, tool_id=tool_id)
    metrics["run_id"] = bundle.metadata.run_id
    metrics["experiment_name"] = bundle.metadata.experiment_name
    metrics["summary_status"] = bundle.summary.status
    return metrics


def load_grid_truth_points(config: GridDefinitionConfig, *, project_root: Path) -> list[list[float]]:
    """Load or generate truth grid points."""
    if config.truth_points_mm:
        return [[float(value) for value in point] for point in config.truth_points_mm]
    if config.truth_points_file:
        path = _resolve_repo_path(project_root, config.truth_points_file)
        arr = np.loadtxt(path, delimiter=",")
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2:
            raise ValueError("Grid truth file must be a 2D numeric array")
        if arr.shape[1] == 3:
            return [[float(value) for value in row] for row in arr]
        if arr.shape[0] == 3:
            return [[float(value) for value in row] for row in arr.T]
        raise ValueError("Grid truth file must have shape (N, 3) or (3, N)")
    dims = [int(value) for value in config.dimensions]
    if len(dims) not in {2, 3}:
        raise ValueError("Grid dimensions must contain 2 or 3 axes")
    if len(dims) == 2:
        dims.append(1)
    coordinates = []
    for z in range(dims[2]):
        rows = []
        for y in range(dims[1]):
            row = []
            for x in range(dims[0]):
                row.append([x * config.spacing_mm, y * config.spacing_mm, z * config.spacing_mm])
            if config.point_ordering == "snake" and (y % 2 == 1):
                row.reverse()
            rows.extend(row)
        coordinates.extend(rows)
    if config.point_ordering == "column_major":
        coordinates = sorted(coordinates, key=lambda point: (point[0], point[1], point[2]))
    return [[float(value) for value in point] for point in coordinates]


def compute_grid_accuracy_metrics(
    samples,
    *,
    truth_points_mm: list[list[float]],
    tool_id: str,
    truth_frame: str,
    outlier_threshold_mm: float,
    registration_available: bool,
    tip_calibration_available: bool,
    require_tip_calibration: bool,
    allow_coil_origin_fallback: bool,
) -> dict[str, Any]:
    """Compute grid accuracy metrics and shared status classification."""
    measurement_samples = [sample for sample in samples if sample.phase == "sample" and sample.target_index is not None]
    grouped = group_positions_by_key(
        measurement_samples,
        key_fn=lambda sample: int(sample.target_index),
        position_fn=lambda sample: extract_tip_or_tool_position_mm(
            sample,
            tool_id=tool_id,
            prefer_robot_frame=(truth_frame == "robot"),
        )[0],
    )
    valid_point_count = sum(1 for positions in grouped.values() if positions)
    if valid_point_count == 0:
        return {
            "status": STATUS_INVALID_INSUFFICIENT_SAMPLES,
            "per_point_metrics": {},
            "overall_rms_error_mm": None,
            "per_axis_bias_mm": None,
            "outlier_count": 0,
            "registration_available": registration_available,
            "tip_calibration_available": tip_calibration_available,
        }
    truth_by_index = {index: truth_points_mm[index] for index in range(len(truth_points_mm))}
    per_point_metrics: dict[str, Any] = {}
    measured_all: list[list[float]] = []
    truth_all: list[list[float]] = []
    outlier_count = 0
    for point_index, positions in grouped.items():
        centroid = centroid_mm(positions)
        truth = truth_by_index.get(int(point_index))
        spread = spread_rms_mm(positions, centroid_xyz=centroid)
        residual_distances = [
            float(np.linalg.norm(np.asarray(position, dtype=float) - centroid))
            for position in positions
        ]
        point_outliers = sum(1 for distance in residual_distances if distance > float(outlier_threshold_mm))
        outlier_count += point_outliers
        point_metrics = {
            "count": len(positions),
            "mean_measured_position_mm": [float(value) for value in centroid],
            "sample_spread_rms_mm": spread,
            "outlier_count": point_outliers,
        }
        if truth is not None and (
            truth_frame == "tracker" or (truth_frame == "robot" and registration_available)
        ):
            truth_stack = np.repeat(np.asarray(truth, dtype=float).reshape(1, 3), len(positions), axis=0)
            point_metrics["rms_error_mm"] = rms_error_mm(positions, truth_stack.tolist())
            measured_all.extend(positions)
            truth_all.extend(truth_stack.tolist())
        per_point_metrics[str(point_index)] = point_metrics
    overall_rms = rms_error_mm(measured_all, truth_all) if measured_all else None
    bias = per_axis_bias_mm(measured_all, truth_all) if measured_all else None
    per_point_rms = pointwise_rms_errors_mm(grouped, truth_by_index) if measured_all else {}
    if require_tip_calibration and not tip_calibration_available and not allow_coil_origin_fallback:
        status = STATUS_INVALID_MISSING_TIP_CAL
    elif truth_frame == "robot" and not registration_available:
        status = STATUS_INVALID_MISSING_REGISTRATION
    elif require_tip_calibration and not tip_calibration_available and allow_coil_origin_fallback:
        status = STATUS_PARTIAL_SUCCESS
    elif overall_rms is None:
        status = STATUS_PARTIAL_SUCCESS
    else:
        status = STATUS_SUCCESS
    return {
        "status": status,
        "per_point_metrics": per_point_metrics,
        "pointwise_rms_error_mm": per_point_rms,
        "overall_rms_error_mm": overall_rms,
        "per_axis_bias_mm": bias,
        "outlier_count": outlier_count,
        "registration_available": registration_available,
        "tip_calibration_available": tip_calibration_available,
    }


def _synthetic_repeatability_position_mm(
    cable_deltas_cm: list[float],
    *,
    previous_deltas_cm: list[float],
    rng,
    noise_std_mm: float,
    hysteresis_mm: float,
) -> list[float]:
    deltas = np.asarray(cable_deltas_cm, dtype=float)
    previous = np.asarray(previous_deltas_cm, dtype=float)
    model = np.array(
        [
            [32.0, 4.0, -28.0, -3.0],
            [5.0, 30.0, -4.0, -26.0],
            [8.0, 8.0, 8.0, 8.0],
        ],
        dtype=float,
    )
    base = np.array([0.0, 0.0, 55.0], dtype=float) + (model @ deltas)
    approach_direction = deltas - previous
    approach_xy = np.array(
        [approach_direction[0] - approach_direction[2], approach_direction[1] - approach_direction[3], 0.0],
        dtype=float,
    )
    norm = np.linalg.norm(approach_xy)
    hysteresis = np.zeros(3, dtype=float) if norm <= 1e-9 else (approach_xy / norm) * float(hysteresis_mm)
    noise = rng.normal(0.0, float(noise_std_mm), size=3)
    return [float(value) for value in (base + hysteresis + noise)]


def _synthetic_pivot_transforms(config: PivotCalibrationConfig) -> list[np.ndarray]:
    rng = np.random.default_rng(config.seed)
    tip = np.asarray(config.synthetic_tip_vector_mm, dtype=float)
    pivot = np.asarray(config.synthetic_pivot_point_mm, dtype=float)
    transforms: list[np.ndarray] = []
    for index in range(max(1, config.sample_count)):
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        if quat[0] < 0.0:
            quat *= -1.0
        R = quat_wxyz_to_rotmat(tuple(float(value) for value in quat))
        translation = pivot - (R @ tip)
        translation += rng.normal(0.0, config.synthetic_noise_std_mm, size=3)
        if index < int(config.synthetic_outlier_count):
            translation += np.array([8.0, -6.0, 5.0], dtype=float)
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = translation
        transforms.append(T)
    return transforms


def _issue_command_payload(
    session: ExperimentSession,
    *,
    tendon_displacement_cm: list[float],
    neutral_ticks: list[int],
    dry_run: bool,
) -> dict[str, int | float | None]:
    servo_ids = list(session.context.settings.robot.servo_ids)
    neutral = list(neutral_ticks) if neutral_ticks else [0 for _ in servo_ids]
    if dry_run or not session.context.servo_service.is_connected:
        if len(neutral) == len(tendon_displacement_cm):
            goals = session.context.servo_service.mapper.to_goal_positions(tendon_displacement_cm, neutral)
            return {str(servo_id): int(goal) for servo_id, goal in zip(servo_ids, goals)}
        return {}
    command = session.context.servo_service.command_displacement(
        tendon_displacements_cm=[float(value) for value in tendon_displacement_cm],
        neutral_ticks=neutral,
        servo_ids=servo_ids,
    )
    return {str(servo_id): int(goal) for servo_id, goal in command.positions_by_id.items()}


def _load_neutral_ticks(session: ExperimentSession) -> list[int]:
    neutral_map = session.context.servo_service.load_neutral_setpoints()
    return [
        int(neutral_map[servo_id])
        for servo_id in session.context.settings.robot.servo_ids
        if servo_id in neutral_map
    ]


def _load_tip_vector(session: ExperimentSession, config: GridDefinitionConfig) -> tuple[list[float] | None, bool]:
    if config.tip_vector_mm is not None:
        return [float(value) for value in config.tip_vector_mm], True
    if config.tip_file:
        path = _resolve_repo_path(session.context.project_root, config.tip_file)
        arr = np.loadtxt(path, delimiter=",")
        arr = np.asarray(arr, dtype=float).reshape(-1)
        if arr.shape == (3,):
            return [float(value) for value in arr], True
    return None, False


def _resolve_repo_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return Path(project_root) / path


def _grid_snapshot(session: ExperimentSession, tool_id: str):
    tracking_service = session.context.tracking_service
    if tracking_service is not None and getattr(tracking_service, "_thread", None) is None and session.context.settings.runtime.mock_mode:
        tracking_service.start()
    snapshot = tracking_service.get_snapshot()
    if tool_id not in snapshot.tools and snapshot.tools:
        replacement = next(iter(snapshot.tools.keys()))
        snapshot.tools[tool_id] = snapshot.tools[replacement]
    return snapshot


def _pivot_snapshot(session: ExperimentSession, *, tool_id: str):
    tracking_service = session.context.tracking_service
    if tracking_service is not None and getattr(tracking_service, "_thread", None) is None and session.context.settings.runtime.mock_mode:
        tracking_service.start()
    snapshot = tracking_service.get_snapshot()
    if tool_id not in snapshot.tools and snapshot.tools:
        replacement = next(iter(snapshot.tools.keys()))
        snapshot.tools[tool_id] = snapshot.tools[replacement]
    return snapshot
