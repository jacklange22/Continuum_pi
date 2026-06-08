"""Project-critical experiments built on top of the canonical framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np

from continuum_robot.experiments.dataset_io import ExperimentDatasetLoader
from continuum_robot.experiments.dataset_tools import extract_tip_or_tool_position_mm
from continuum_robot.experiments.framework import BaseExperiment, ExperimentHardwareRequirements, ExperimentSession
from continuum_robot.experiments.grid_accuracy_outputs import write_grid_accuracy_outputs
from continuum_robot.experiments.metrics import (
    as_points_n3,
    centroid_mm,
    group_positions_by_key,
    per_axis_bias_mm,
    pointwise_rms_errors_mm,
    rms_error_mm,
    spread_rms_mm,
)
from continuum_robot.experiments.pivot_utils import (
    PivotCalibrationResult,
    PivotInputParseError,
    PivotRansacFailure,
    RansacPivotCalibrationResult,
    load_pivot_transforms_with_report,
    solve_pivot_calibration,
    solve_pivot_calibration_ransac,
    write_tip_vector_file,
)
from continuum_robot.experiments.sample_builders import sample_from_tracking_snapshot
from continuum_robot.experiments.schemas import ExperimentTimeseriesSample
from continuum_robot.experiments.validation import (
    STATUS_INVALID_INSUFFICIENT_SAMPLES,
    STATUS_INVALID_MISSING_TIP_CAL,
    STATUS_PARTIAL_SUCCESS,
    STATUS_SUCCESS,
)
from continuum_robot.tracking.transforms import quat_wxyz_to_rotmat, rotmat_to_quat_wxyz


@dataclass
class RepeatabilityScheduleConfig:
    """Config for repeatability revisit scheduling."""

    target_set: str = ""
    target_labels: list[str] = field(default_factory=list)
    target_points_cm: list[list[float]] = field(default_factory=list)
    low_magnitude_cm: float = 0.10
    high_magnitude_cm: float = 0.20
    revisit_count: int = 3
    randomize_approach_order: bool = False
    seed: int = 0
    settle_time_s: float = 0.0
    samples_per_point: int = 3

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RepeatabilityScheduleConfig":
        payload = dict(payload or {})
        explicit_target_set = str(payload.get("target_set", "") or "").strip()
        target_points_payload = payload.get("target_points_cm", []) or []
        target_set = explicit_target_set or ("manual" if target_points_payload else "single_segment_ring_17")
        return cls(
            target_set=target_set,
            target_labels=[str(value) for value in payload.get("target_labels", []) or []],
            target_points_cm=[
                [float(value) for value in point]
                for point in target_points_payload
            ],
            low_magnitude_cm=float(payload.get("low_magnitude_cm", 0.10)),
            high_magnitude_cm=float(payload.get("high_magnitude_cm", 0.20)),
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
    approach_label: str = ""
    target_label: str = ""
    approach_groups: list[str] = field(default_factory=list)
    target_groups: list[str] = field(default_factory=list)
    approach_axis_class: str | None = None
    target_axis_class: str | None = None
    approach_magnitude_class: str | None = None
    target_magnitude_class: str | None = None


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
    """DEPRECATED — synthetic-mode-only multiplier on samples_per_point.

    Never reached the GUI. For real click-by-click captures the operator
    controls how many samples per point via the (visible) ``samples_per_point``
    knob. Kept for YAML/test backward compatibility; new work should not set it.
    """
    samples_per_point: int = 3
    settle_time_s: float = 0.0
    truth_points_mm: list[list[float]] = field(default_factory=list)
    truth_points_file: str | None = None
    truth_frame: str = "tracker"
    tool_id: str = "0B"
    # The tip vector is ALWAYS sourced from the latest valid pivot calibration
    # (see resolve_grid_tip_source). use_tip_calibration / tip_vector_mm /
    # tip_file / allow_coil_origin_fallback are retained for back-compat config
    # parsing only and are IGNORED by the experiment — there is no hardcoded
    # vector and no coil-origin fallback.
    use_tip_calibration: bool = True
    tip_vector_mm: list[float] | None = None
    tip_file: str | None = None
    allow_coil_origin_fallback: bool = False
    # dry_run / synthetic_* are retained for back-compat parsing only. The
    # experiment no longer synthesizes data — it requires real captured points.
    dry_run: bool = False
    seed: int | None = None
    """Synthetic-mode RNG seed.

    ``None`` (default) → a fresh seed is generated each run so successive
    "Save Grid Validation Run" presses produce *different* synthetic
    datasets. Set an explicit integer to reproduce a specific run.
    """
    synthetic_noise_std_mm: float = 0.25
    synthetic_bias_mm: list[float] = field(default_factory=lambda: [0.2, -0.1, 0.05])
    outlier_threshold_mm: float = 1.0
    # Optional acceptance threshold drawn as a horizontal cutoff on the
    # per-point residuals report and the pass/fail badge in the thesis
    # figures. None = no threshold drawn / pass-fail badge omitted.
    acceptance_threshold_mm: float | None = None
    captured_points: list[dict[str, Any]] = field(default_factory=list)
    acceptance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GridDefinitionConfig":
        payload = dict(payload or {})
        dimensions_payload = payload.get("dimensions")
        if dimensions_payload in (None, ""):
            rows = payload.get("rows")
            cols = payload.get("cols")
            if rows not in (None, "") and cols not in (None, ""):
                dimensions_payload = [int(cols), int(rows)]
            else:
                dimensions_payload = [3, 3]
        return cls(
            spacing_mm=float(payload.get("spacing_mm", 25.4)),
            dimensions=[int(value) for value in dimensions_payload],
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
            allow_coil_origin_fallback=bool(payload.get("allow_coil_origin_fallback", False)),
            dry_run=bool(payload.get("dry_run", False)),
            seed=(int(payload["seed"]) if payload.get("seed") not in (None, "") else None),
            synthetic_noise_std_mm=float(payload.get("synthetic_noise_std_mm", 0.25)),
            synthetic_bias_mm=[float(value) for value in payload.get("synthetic_bias_mm", [0.2, -0.1, 0.05])],
            outlier_threshold_mm=float(payload.get("outlier_threshold_mm", 1.0)),
            acceptance_threshold_mm=(
                float(payload["acceptance_threshold_mm"])
                if payload.get("acceptance_threshold_mm") not in (None, "")
                else None
            ),
            captured_points=[
                dict(point)
                for point in payload.get("captured_points", []) or []
                if isinstance(point, dict)
            ],
            acceptance=dict(payload.get("acceptance", {}) or {}),
        )


@dataclass
class GridAccuracyPreview:
    """Normalized capture-session preview for the aligned grid validation workflow."""

    truth_catalog: list[dict[str, Any]]
    samples: list[ExperimentTimeseriesSample]
    metrics: dict[str, Any]


@dataclass
class RepeatabilityPreview:
    """Resolved target/schedule preview for the repeatability page and preflight."""

    target_catalog: list[dict[str, Any]]
    visits: list[RepeatabilityVisit]
    summary: dict[str, Any]


@dataclass
class PivotCalibrationConfig:
    """Config for pivot calibration and tip file generation."""

    tool_id: str = "0B"
    sample_count: int = 80
    sample_period_s: float = 0.02
    std_dev_threshold: float = 3.0
    min_samples: int = 12
    output_tip_file: str = "data/pivot_calibration/generated_penprobe_tip.csv"
    input_path: str | None = None
    dry_run: bool = False
    seed: int = 0
    synthetic_tip_vector_mm: list[float] = field(default_factory=lambda: [0.0, 0.0, 125.0])
    synthetic_pivot_point_mm: list[float] = field(default_factory=lambda: [25.0, -10.0, 40.0])
    synthetic_noise_std_mm: float = 0.25
    synthetic_outlier_count: int = 0
    acceptance: dict[str, Any] = field(default_factory=dict)
    use_ransac: bool = False
    """When True, run RANSAC outlier rejection instead of the classical std-dev pass."""
    ransac_inlier_threshold_mm: float = 1.0
    """Residual-norm cutoff (mm) for a pose to count as a RANSAC inlier."""
    ransac_minimum_sample_size: int = 3
    """Poses drawn per RANSAC iteration; 3 conditions per-sample fits well."""
    ransac_min_consensus_size: int | None = None
    """Required inlier count; defaults to ``max(min_samples, ceil(0.5 N))``."""
    ransac_max_iterations: int = 1000
    """Hard iteration cap; adaptive shrinking will typically stop sooner."""
    ransac_confidence: float = 0.99
    """Target probability of sampling an all-inlier minimum set at least once."""
    ransac_seed: int | None = None
    """Optional dedicated seed for RANSAC; falls back to ``seed`` when None."""

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PivotCalibrationConfig":
        payload = dict(payload or {})
        raw_ransac_seed = payload.get("ransac_seed")
        ransac_seed: int | None
        if raw_ransac_seed is None or raw_ransac_seed == "":
            ransac_seed = None
        else:
            ransac_seed = int(raw_ransac_seed)
        raw_ransac_floor = payload.get("ransac_min_consensus_size")
        ransac_floor: int | None
        if raw_ransac_floor is None or raw_ransac_floor == "":
            ransac_floor = None
        else:
            ransac_floor = int(raw_ransac_floor)
        return cls(
            tool_id=str(payload.get("tool_id", "0B")),
            sample_count=int(payload.get("sample_count", 80)),
            sample_period_s=float(payload.get("sample_period_s", 0.02)),
            std_dev_threshold=float(payload.get("std_dev_threshold", 3.0)),
            min_samples=int(payload.get("min_samples", 12)),
            output_tip_file=str(
                payload.get("output_tip_file", "data/pivot_calibration/generated_penprobe_tip.csv")
            ),
            input_path=str(payload["input_path"]) if payload.get("input_path") not in (None, "") else None,
            dry_run=bool(payload.get("dry_run", False)),
            seed=int(payload.get("seed", 0)),
            synthetic_tip_vector_mm=[float(value) for value in payload.get("synthetic_tip_vector_mm", [0.0, 0.0, 125.0])],
            synthetic_pivot_point_mm=[float(value) for value in payload.get("synthetic_pivot_point_mm", [25.0, -10.0, 40.0])],
            synthetic_noise_std_mm=float(payload.get("synthetic_noise_std_mm", 0.25)),
            synthetic_outlier_count=int(payload.get("synthetic_outlier_count", 0)),
            acceptance=dict(payload.get("acceptance", {}) or {}),
            use_ransac=bool(payload.get("use_ransac", False)),
            ransac_inlier_threshold_mm=float(payload.get("ransac_inlier_threshold_mm", 1.0)),
            ransac_minimum_sample_size=int(payload.get("ransac_minimum_sample_size", 3)),
            ransac_min_consensus_size=ransac_floor,
            ransac_max_iterations=int(payload.get("ransac_max_iterations", 1000)),
            ransac_confidence=float(payload.get("ransac_confidence", 0.99)),
            ransac_seed=ransac_seed,
        )


class RepeatabilityDatasetExperiment(BaseExperiment):
    """Main repeatability and core robot dataset experiment."""

    name = "repeatability_dataset"
    description = (
        "Revisit the same commanded targets from different prior states, save a canonical repeatability dataset, "
        "and summarize per-target spread plus path-dependence."
    )
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
        tendon_count = len(session.context.settings.robot.tendon_to_servo or session.context.settings.robot.servo_ids)
        target_catalog = build_repeatability_target_catalog(self.config.schedule, tendon_count=tendon_count)
        if not target_catalog:
            raise RuntimeError("repeatability_dataset requires at least one repeatability target")

    def execute(self, session: ExperimentSession) -> None:
        tendon_count = len(session.context.settings.robot.tendon_to_servo or session.context.settings.robot.servo_ids)
        target_catalog = build_repeatability_target_catalog(self.config.schedule, tendon_count=tendon_count)
        resolved_target_set = str(self.config.schedule.target_set or "").strip() or (
            "manual" if self.config.schedule.target_points_cm else "single_segment_ring_17"
        )
        visits = generate_repeatability_schedule(
            self.config.schedule,
            tendon_count=tendon_count,
            target_catalog=target_catalog,
        )
        rng = np.random.default_rng(self.config.schedule.seed)
        total = len(visits) * (3 + max(1, self.config.schedule.samples_per_point))
        completed = 0
        capture_mode = "synthetic_dry_run" if self.config.dry_run else "live_tracker"
        sample_status_flags = ["dry_run", "synthetic_capture"] if self.config.dry_run else []
        session.set_metric("dry_run", bool(self.config.dry_run))
        for visit_sequence_index, visit in enumerate(visits):
            session.raise_if_stop_requested()
            visit_extra = {
                "target_label": visit.target_label,
                "approach_label": visit.approach_label,
                "target_groups": list(visit.target_groups),
                "approach_groups": list(visit.approach_groups),
                "target_axis_class": visit.target_axis_class,
                "approach_axis_class": visit.approach_axis_class,
                "target_magnitude_class": visit.target_magnitude_class,
                "approach_magnitude_class": visit.approach_magnitude_class,
                "target_set": resolved_target_set,
                "visit_sequence_index": int(visit_sequence_index),
                "capture_mode": capture_mode,
            }
            approach_payload = _issue_command_payload(
                session,
                tendon_displacement_cm=visit.approach_deltas_cm,
                neutral_ticks=self._neutral_ticks,
                dry_run=self.config.dry_run,
            )
            approach_tracker = (
                _synthetic_repeatability_position_mm(
                    visit.approach_deltas_cm,
                    previous_deltas_cm=visit.approach_deltas_cm,
                    rng=rng,
                    noise_std_mm=self.config.synthetic_noise_std_mm,
                    hysteresis_mm=self.config.synthetic_hysteresis_mm,
                )
                if self.config.dry_run
                else None
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
                        approach_tracker
                        if self.config.dry_run and session.context.registration_path.exists()
                        else None
                    ),
                    tracker_tool_id=self.config.tool_id,
                    status_flags=list(sample_status_flags),
                    extra=visit_extra,
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
                        approach_tracker
                        if self.config.dry_run and session.context.registration_path.exists()
                        else None
                    ),
                    tracker_tool_id=self.config.tool_id,
                    status_flags=list(sample_status_flags),
                    extra=visit_extra,
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
            target_tracker = (
                _synthetic_repeatability_position_mm(
                    visit.target_deltas_cm,
                    previous_deltas_cm=visit.approach_deltas_cm,
                    rng=rng,
                    noise_std_mm=self.config.synthetic_noise_std_mm,
                    hysteresis_mm=self.config.synthetic_hysteresis_mm,
                )
                if self.config.dry_run
                else None
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
                        target_tracker
                        if self.config.dry_run and session.context.registration_path.exists()
                        else None
                    ),
                    tracker_tool_id=self.config.tool_id,
                    status_flags=list(sample_status_flags),
                    extra=visit_extra,
                )
            )
            completed += 1
            session.update_progress(completed, total, {"phase": "target_command", "target_index": visit.target_index})
            if self.config.schedule.settle_time_s > 0:
                session.context.sleep_fn(self.config.schedule.settle_time_s)
            for sample_index in range(max(1, self.config.schedule.samples_per_point)):
                jittered_target = (
                    (
                        np.asarray(target_tracker, dtype=float)
                        + rng.normal(0.0, self.config.synthetic_noise_std_mm, size=3)
                    ).tolist()
                    if self.config.dry_run and target_tracker is not None
                    else None
                )
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
                            jittered_target
                            if self.config.dry_run and session.context.registration_path.exists()
                            else None
                        ),
                        tracker_tool_id=self.config.tool_id,
                        status_flags=list(sample_status_flags),
                        extra=visit_extra,
                    )
                )
                completed += 1
                session.update_progress(completed, total, {"phase": "sample", "target_index": visit.target_index})
        metrics = compute_repeatability_metrics(session.samples, tool_id=self.config.tool_id)
        metrics["target_catalog"] = [
            {
                "target_index": int(entry["target_index"]),
                "label": str(entry["label"]),
                "tendon_deltas_cm": [float(value) for value in entry["tendon_deltas_cm"]],
                "axis_class": str(entry.get("axis_class", "")),
                "magnitude_class": str(entry.get("magnitude_class", "")),
                "group_tags": [str(tag) for tag in entry.get("group_tags", []) or []],
            }
            for entry in target_catalog
        ]
        metrics["planned_visit_count"] = len(visits)
        metrics["planned_sample_count"] = len(visits) * max(1, self.config.schedule.samples_per_point)
        metrics["registration_available"] = session.context.registration_path.exists()
        metrics["summary_requirements"] = {"force_status": metrics["status"]}
        session.metrics.update(metrics)

    def finalize(self, session: ExperimentSession) -> None:
        try:
            if not self.config.dry_run and session.context.servo_service.is_connected and self._neutral_ticks:
                zero_vector = [0.0] * len(self._neutral_ticks)
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
    description = (
        "Capture labeled 0B grid points, rigidly align them to an ideal truth grid, "
        "and report residual consistency after alignment."
    )
    hardware_requirements = ExperimentHardwareRequirements(mock_compatible=True)

    def __init__(self, config: GridDefinitionConfig) -> None:
        super().__init__(config=config)
        self._tracking_started_here = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "AuroraGridAccuracyExperiment":
        return cls(config=GridDefinitionConfig.from_dict(payload))

    def config_dict(self) -> dict[str, Any]:
        payload = super().config_dict()
        captured_points = payload.pop("captured_points", []) or []
        if captured_points:
            payload["captured_point_count"] = int(len(captured_points))
            payload["captured_sample_count"] = int(
                sum(len(point.get("raw_samples", []) or []) for point in captured_points if isinstance(point, dict))
            )
        return payload

    def setup(self, session: ExperimentSession) -> None:
        if session.context.tracking_service is not None and getattr(session.context.tracking_service, "_thread", None) is None:
            session.context.tracking_service.start()
            self._tracking_started_here = True

    def execute(self, session: ExperimentSession) -> None:
        # 1) Refuse synthetic data: the run requires REAL captured grid points.
        #    There is no synthetic / dry-run data-generation path any more.
        if not self.config.captured_points:
            raise RuntimeError(
                "Aurora grid accuracy requires real captured grid points. Capture the grid "
                "points from the live Aurora tracker on the grid page first — the experiment "
                "no longer synthesizes data."
            )
        if _captured_points_contain_synthetic_samples(self.config.captured_points):
            raise RuntimeError(
                "Aurora grid accuracy refuses synthetic/dry-run captured samples. Restart the "
                "grid run and recapture every point from the live tracker."
            )
        # 2) Require a valid, current pivot calibration as the 0B tip vector.
        #    No hardcoded vector, no coil-origin fallback — the tip MUST come
        #    from the latest successful pivot calibration.
        tip_source = resolve_grid_tip_source(self.config, project_root=session.context.project_root)
        if not tip_source.available:
            raise RuntimeError(
                "Aurora grid accuracy requires a valid pivot calibration for tool "
                f"{self.config.tool_id}. No successful pivot calibration was found under "
                "data/pivot_calibration/. Run a pivot calibration first; the latest successful "
                "one is used automatically."
            )

        preview = build_grid_accuracy_preview(
            self.config,
            project_root=session.context.project_root,
        )
        truth_points = [entry["truth_point_mm"] for entry in preview.truth_catalog]

        total = len(preview.samples)
        for completed, sample in enumerate(preview.samples, start=1):
            session.raise_if_stop_requested()
            session.add_sample(sample)
            session.update_progress(
                completed,
                total,
                {"phase": sample.phase, "target_index": sample.target_index},
            )

        metrics = compute_grid_accuracy_metrics(
            session.samples,
            truth_points_mm=truth_points,
            tool_id=self.config.tool_id,
            truth_frame="grid_local",
            outlier_threshold_mm=self.config.outlier_threshold_mm,
            registration_available=session.context.registration_path.exists(),
            tip_calibration_available=True,
            require_tip_calibration=True,
            allow_coil_origin_fallback=False,
        )
        metrics.update(
            _grid_capture_progress_metrics(
                captured_points=self.config.captured_points,
                truth_catalog=preview.truth_catalog,
                expected_samples=max(1, int(self.config.samples_per_point)),
            )
        )
        # Record exactly which pivot calibration supplied the tip vector so the
        # bundle is self-documenting for the thesis.
        metrics["pivot_calibration"] = {
            "run_name": tip_source.pivot_run_name,
            "tip_vector_mm": list(tip_source.tip_vector_mm or []),
            "rmse_mm": tip_source.pivot_rmse_mm,
            "summary_path": tip_source.pivot_summary_path,
            "tool_id": self.config.tool_id,
        }
        metrics["pivot_tip_vector_mm"] = list(tip_source.tip_vector_mm or [])
        metrics["pivot_calibration_run"] = tip_source.pivot_run_name
        metrics["summary_requirements"] = {"force_status": metrics["status"]}
        session.metrics.update(metrics)

    def finalize(self, session: ExperimentSession) -> None:
        if self._tracking_started_here:
            session.context.tracking_service.stop()
            self._tracking_started_here = False

    def write_outputs(self, session: ExperimentSession, paths, summary) -> None:
        write_grid_accuracy_outputs(
            output_dir=paths.output_dir,
            metadata=session.metadata,
            summary=summary,
        )


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
        try:
            transforms, input_metrics = self._collect_transforms(session)
        except PivotInputParseError as exc:
            session.metrics.update(exc.report.to_metrics())
            raise RuntimeError(str(exc)) from exc
        session.metrics.update(input_metrics)
        session.set_metric("dry_run", bool(self.config.dry_run))
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
                status_flags=(
                    ["offline_input"]
                    if self.config.input_path
                    else (["dry_run", "synthetic_capture"] if self.config.dry_run else [])
                ),
                extra={
                    "capture_mode": (
                        "offline_input"
                        if self.config.input_path
                        else ("synthetic_dry_run" if self.config.dry_run else "live_tracker")
                    )
                },
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
        # Always run both solvers on the same input so the operator sees a
        # side-by-side comparison every Solve. Which result populates the
        # saved tip vector + output file is still driven by ``use_ransac``.
        classical_result: PivotCalibrationResult | None = None
        classical_error: Exception | None = None
        try:
            classical_result = solve_pivot_calibration(
                rotations,
                translations,
                std_dev_threshold=self.config.std_dev_threshold,
                min_samples=self.config.min_samples,
            )
        except ValueError as exc:
            classical_error = exc

        ransac_result: RansacPivotCalibrationResult | None = None
        ransac_error: Exception | None = None
        ransac_attempted = False
        if len(rotations) >= max(int(self.config.ransac_minimum_sample_size) + 1, int(self.config.min_samples)):
            ransac_attempted = True
            effective_seed = (
                self.config.ransac_seed
                if self.config.ransac_seed is not None
                else int(self.config.seed)
            )
            effective_floor = (
                int(self.config.ransac_min_consensus_size)
                if self.config.ransac_min_consensus_size is not None
                else max(int(self.config.min_samples), math.ceil(0.5 * len(rotations)))
            )
            try:
                ransac_result = solve_pivot_calibration_ransac(
                    rotations,
                    translations,
                    inlier_threshold_mm=self.config.ransac_inlier_threshold_mm,
                    minimum_sample_size=self.config.ransac_minimum_sample_size,
                    min_consensus_size=effective_floor,
                    max_iterations=self.config.ransac_max_iterations,
                    confidence=self.config.ransac_confidence,
                    seed=effective_seed,
                )
            except (ValueError, PivotRansacFailure) as exc:
                ransac_error = exc

        solver_comparison = self._build_pivot_solver_comparison(
            classical=classical_result,
            classical_error=classical_error,
            ransac=ransac_result,
            ransac_error=ransac_error,
            ransac_attempted=ransac_attempted,
            sample_count=len(rotations),
        )

        if self.config.use_ransac:
            primary_solver_label = "ransac"
            primary_result = ransac_result.to_pivot_calibration_result() if ransac_result is not None else None
            primary_error = ransac_error or classical_error
        else:
            primary_solver_label = "classical_std_dev"
            primary_result = classical_result
            primary_error = classical_error

        if primary_result is None:
            status = STATUS_INVALID_INSUFFICIENT_SAMPLES
            session.metrics.update(
                {
                    "sample_count_total": len(transforms),
                    "sample_count_used": 0,
                    "sample_count_rejected": 0,
                    "tip_calibration_available": False,
                    "status": status,
                    "summary_requirements": {"force_status": status},
                    "pivot_solver": primary_solver_label,
                    "pivot_solver_comparison": solver_comparison,
                    "ransac_failure_partial": (
                        dict(ransac_error.partial)
                        if isinstance(ransac_error, PivotRansacFailure) and ransac_error.partial
                        else None
                    ),
                }
            )
            raise RuntimeError(str(primary_error or "Pivot solve produced no result")) from primary_error

        output_tip_path = write_tip_vector_file(
            _resolve_repo_path(session.context.project_root, self.config.output_tip_file),
            primary_result.tip_vector_local_mm,
        )
        # Stage one tip file per candidate solver next to the canonical output
        # so the operator can flip the selected solver in the Tracker tab and
        # have the controller promote the right file at accept time.
        canonical_path = Path(_resolve_repo_path(session.context.project_root, self.config.output_tip_file))
        classical_tip_path: Path | None = None
        ransac_tip_path: Path | None = None
        if classical_result is not None:
            classical_tip_path = canonical_path.with_name(
                f"{canonical_path.stem}__classical{canonical_path.suffix}"
            )
            write_tip_vector_file(classical_tip_path, classical_result.tip_vector_local_mm)
            solver_comparison.setdefault("classical", {})
            if isinstance(solver_comparison.get("classical"), dict):
                solver_comparison["classical"]["tip_output_file"] = str(classical_tip_path)
        if ransac_result is not None:
            ransac_tip_path = canonical_path.with_name(
                f"{canonical_path.stem}__ransac{canonical_path.suffix}"
            )
            write_tip_vector_file(ransac_tip_path, ransac_result.tip_vector_local_mm)
            solver_comparison.setdefault("ransac", {})
            if isinstance(solver_comparison.get("ransac"), dict):
                solver_comparison["ransac"]["tip_output_file"] = str(ransac_tip_path)

        status = STATUS_SUCCESS
        metrics_update: dict[str, Any] = {
            "tip_vector_local_mm": primary_result.tip_vector_local_mm,
            "pivot_point_tracker_mm": primary_result.pivot_point_tracker_mm,
            "rmse_mm": primary_result.rmse_mm,
            "sample_count_total": primary_result.sample_count_total,
            "sample_count_used": primary_result.sample_count_used,
            "sample_count_rejected": primary_result.sample_count_rejected,
            "pivot_residuals_mm": primary_result.residuals_mm,
            "pivot_inlier_mask": primary_result.inlier_mask,
            "pivot_rejected_indices": primary_result.rejected_indices,
            "tip_output_file": str(output_tip_path),
            "tip_output_file_classical": str(classical_tip_path) if classical_tip_path is not None else None,
            "tip_output_file_ransac": str(ransac_tip_path) if ransac_tip_path is not None else None,
            "tip_calibration_available": True,
            "status": status,
            "summary_requirements": {"force_status": status},
            "pivot_solver": primary_solver_label,
            "pivot_solver_comparison": solver_comparison,
        }
        # The `pivot_ransac` metric is the "this run used RANSAC; here's the
        # full result including the inlier mask" canonical field. Only stamp
        # it when the operator opted INTO RANSAC. Even though we always run
        # both solvers for the side-by-side comparison panel
        # (`pivot_solver_comparison`), a classical run should not have
        # `pivot_ransac` set — otherwise downstream consumers cannot tell
        # whether the operator actually selected RANSAC. The comparison
        # panel still exposes `comparison["ransac"]` for visibility.
        if self.config.use_ransac and ransac_result is not None:
            metrics_update["pivot_ransac"] = ransac_result.to_dict()
        session.metrics.update(metrics_update)

    @staticmethod
    def _build_pivot_solver_comparison(
        *,
        classical: PivotCalibrationResult | None,
        classical_error: Exception | None,
        ransac: RansacPivotCalibrationResult | None,
        ransac_error: Exception | None,
        ransac_attempted: bool,
        sample_count: int,
    ) -> dict[str, Any]:
        comparison: dict[str, Any] = {"sample_count_total": int(sample_count)}
        if classical is not None:
            comparison["classical"] = {
                "tip_vector_local_mm": list(classical.tip_vector_local_mm),
                "pivot_point_tracker_mm": list(classical.pivot_point_tracker_mm),
                "rmse_mm": float(classical.rmse_mm),
                "sample_count_used": int(classical.sample_count_used),
                "sample_count_rejected": int(classical.sample_count_rejected),
            }
        elif classical_error is not None:
            comparison["classical_failure"] = {
                "message": str(classical_error),
            }
        if ransac is not None:
            comparison["ransac"] = {
                "tip_vector_local_mm": list(ransac.tip_vector_local_mm),
                "pivot_point_tracker_mm": list(ransac.pivot_point_tracker_mm),
                "rmse_mm": float(ransac.rmse_mm),
                "sample_count_used": int(ransac.sample_count_used),
                "sample_count_rejected": int(ransac.sample_count_rejected),
                "inlier_threshold_mm": float(ransac.inlier_threshold_mm),
                "converged": bool(ransac.converged),
                "iterations_run": int(ransac.iterations_run),
                "best_consensus_size": int(ransac.best_consensus_size),
                "min_consensus_size": int(ransac.min_consensus_size),
            }
        elif not ransac_attempted:
            comparison["ransac_skipped"] = (
                f"Need at least minimum_sample_size + 1 poses to run RANSAC; "
                f"received {sample_count}."
            )
        elif ransac_error is not None:
            comparison["ransac_failure"] = {
                "message": str(ransac_error),
                "partial": (
                    dict(getattr(ransac_error, "partial", {}) or {})
                    if isinstance(ransac_error, PivotRansacFailure)
                    else {}
                ),
            }
        if classical is not None and ransac is not None:
            comparison["delta"] = {
                "rmse_mm_classical_minus_ransac": float(classical.rmse_mm - ransac.rmse_mm),
                "tip_vector_mm_difference": [
                    float(c - r)
                    for c, r in zip(classical.tip_vector_local_mm, ransac.tip_vector_local_mm)
                ],
                "pivot_point_mm_difference": [
                    float(c - r)
                    for c, r in zip(classical.pivot_point_tracker_mm, ransac.pivot_point_tracker_mm)
                ],
                "tip_vector_distance_mm": float(
                    math.sqrt(
                        sum(
                            (c - r) ** 2
                            for c, r in zip(
                                classical.tip_vector_local_mm,
                                ransac.tip_vector_local_mm,
                            )
                        )
                    )
                ),
                "sample_count_difference": int(
                    classical.sample_count_used - ransac.sample_count_used
                ),
            }
        return comparison

    def finalize(self, session: ExperimentSession) -> None:
        if self._tracking_started_here:
            session.context.tracking_service.stop()
            self._tracking_started_here = False

    def _collect_transforms(self, session: ExperimentSession) -> tuple[list[np.ndarray], dict[str, object]]:
        if self.config.input_path:
            load_result = load_pivot_transforms_with_report(
                _resolve_repo_path(session.context.project_root, self.config.input_path),
                tool_id=self.config.tool_id,
            )
            return load_result.transforms, load_result.report.to_metrics()
        if self.config.dry_run:
            transforms = _synthetic_pivot_transforms(self.config)
            return transforms, {
                "pivot_input_format": "synthetic_dry_run",
                "pivot_input_tool_id": self.config.tool_id,
                "pivot_input_total_rows": len(transforms),
                "pivot_input_usable_rows": len(transforms),
                "pivot_input_filtered_other_tool_rows": 0,
                "pivot_input_rejected_row_count": 0,
                "pivot_input_rejected_rows": [],
            }
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
        return transforms, {
            "pivot_input_format": "live_tracking_stream",
            "pivot_input_tool_id": self.config.tool_id,
            "pivot_input_total_rows": len(transforms),
            "pivot_input_usable_rows": len(transforms),
            "pivot_input_filtered_other_tool_rows": 0,
            "pivot_input_rejected_row_count": 0,
            "pivot_input_rejected_rows": [],
        }


def register_critical_experiments(registry) -> None:
    """Register the project-critical experiments."""
    registry.register(
        name=RepeatabilityDatasetExperiment.name,
        title="Repeatability Dataset",
        description=RepeatabilityDatasetExperiment.description,
        category="validation",
        tags=["Repeatability", "Tracking", "Servo"],
        workspace_visible=False,
        default_config_path="config/experiment_repeatability_dataset.example.yaml",
        factory=RepeatabilityDatasetExperiment.from_dict,
    )
    registry.register(
        name=AuroraGridAccuracyExperiment.name,
        title="Aurora Grid Accuracy",
        description=AuroraGridAccuracyExperiment.description,
        category="validation",
        tags=["Tracking", "Grid", "Accuracy"],
        default_config_path="config/experiment_aurora_grid_accuracy.example.yaml",
        factory=AuroraGridAccuracyExperiment.from_dict,
    )
    registry.register(
        name=PivotCalibrationExperiment.name,
        title="Pivot Calibration",
        description=PivotCalibrationExperiment.description,
        category="calibration",
        tags=["Pivot", "Tip File"],
        workspace_visible=False,
        default_config_path="config/experiment_pivot_calibration.example.yaml",
        factory=PivotCalibrationExperiment.from_dict,
    )


def build_repeatability_target_catalog(
    config: RepeatabilityScheduleConfig,
    *,
    tendon_count: int | None = None,
) -> list[dict[str, Any]]:
    """Resolve the configured repeatability target set into labeled canonical targets."""
    target_set = str(config.target_set or "").strip().lower()
    if not target_set:
        target_set = "manual" if config.target_points_cm else "single_segment_ring_17"
    if target_set == "single_segment_ring_17":
        return _build_single_segment_ring_targets(
            low_magnitude_cm=float(config.low_magnitude_cm),
            high_magnitude_cm=float(config.high_magnitude_cm),
            tendon_count=(4 if tendon_count in (None, 0) else int(tendon_count)),
        )
    if target_set == "manual":
        targets = [[float(value) for value in point] for point in config.target_points_cm]
        if not targets:
            return []
        if tendon_count is not None:
            bad = sorted({len(point) for point in targets if len(point) != int(tendon_count)})
            if bad:
                raise ValueError(
                    f"Manual repeatability targets must contain {int(tendon_count)} tendon values, found {bad}."
                )
        labels = [str(value).strip() for value in config.target_labels if str(value).strip()]
        catalog: list[dict[str, Any]] = []
        for index, point in enumerate(targets):
            label = labels[index] if index < len(labels) else f"T{index + 1:02d}"
            catalog.append(
                {
                    "target_index": int(index),
                    "label": label,
                    "tendon_deltas_cm": [float(value) for value in point],
                    "axis_class": "custom",
                    "magnitude_class": "custom",
                    "angle_deg": None,
                    "group_tags": ["custom"],
                }
            )
        return catalog
    raise ValueError(f"Unsupported repeatability target set: {config.target_set}")


def build_repeatability_preview(
    config: RepeatabilityDatasetConfig,
    *,
    tendon_count: int | None = None,
) -> RepeatabilityPreview:
    """Build the repeatability target/schedule preview for the custom page."""
    target_catalog = build_repeatability_target_catalog(config.schedule, tendon_count=tendon_count)
    visits = generate_repeatability_schedule(
        config.schedule,
        tendon_count=tendon_count,
        target_catalog=target_catalog,
    )
    axis_counts: dict[str, int] = {}
    magnitude_counts: dict[str, int] = {}
    visits_by_target: dict[int, int] = {}
    unique_approaches_by_target: dict[int, set[int]] = {}
    for entry in target_catalog:
        axis = str(entry.get("axis_class", "unknown"))
        magnitude = str(entry.get("magnitude_class", "unknown"))
        axis_counts[axis] = int(axis_counts.get(axis, 0) + 1)
        magnitude_counts[magnitude] = int(magnitude_counts.get(magnitude, 0) + 1)
        visits_by_target[int(entry["target_index"])] = 0
        unique_approaches_by_target[int(entry["target_index"])] = set()
    for visit in visits:
        visits_by_target[int(visit.target_index)] = int(visits_by_target.get(int(visit.target_index), 0) + 1)
        unique_approaches_by_target.setdefault(int(visit.target_index), set()).add(int(visit.approach_index))
    resolved_target_set = str(config.schedule.target_set or "").strip() or (
        "manual" if config.schedule.target_points_cm else "single_segment_ring_17"
    )
    summary = {
        "target_set": resolved_target_set,
        "target_count": len(target_catalog),
        "visit_count": len(visits),
        "planned_sample_count": len(visits) * max(1, int(config.schedule.samples_per_point)),
        "axis_counts": axis_counts,
        "magnitude_counts": magnitude_counts,
        "visits_by_target": {str(key): int(value) for key, value in visits_by_target.items()},
        "unique_approach_counts": {
            str(key): len(value)
            for key, value in unique_approaches_by_target.items()
        },
    }
    return RepeatabilityPreview(target_catalog=target_catalog, visits=visits, summary=summary)


def generate_repeatability_schedule(
    config: RepeatabilityScheduleConfig,
    *,
    tendon_count: int | None = None,
    target_catalog: list[dict[str, Any]] | None = None,
) -> list[RepeatabilityVisit]:
    """Generate deterministic or randomized revisit ordering for repeatability."""
    catalog = list(target_catalog or build_repeatability_target_catalog(config, tendon_count=tendon_count))
    if not catalog:
        raise ValueError("Repeatability schedule requires at least one target point")
    targets = [
        [float(value) for value in entry["tendon_deltas_cm"]]
        for entry in catalog
    ]
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
            target_entry = catalog[target_index]
            approach_entry = catalog[approach_index]
            visits.append(
                RepeatabilityVisit(
                    cycle_index=cycle_index,
                    target_index=target_index,
                    revisit_index=cycle_index,
                    approach_index=approach_index,
                    approach_deltas_cm=list(targets[approach_index]),
                    target_deltas_cm=list(targets[target_index]),
                    approach_label=str(approach_entry.get("label", f"T{approach_index + 1:02d}")),
                    target_label=str(target_entry.get("label", f"T{target_index + 1:02d}")),
                    approach_groups=[str(value) for value in approach_entry.get("group_tags", []) or []],
                    target_groups=[str(value) for value in target_entry.get("group_tags", []) or []],
                    approach_axis_class=(
                        str(approach_entry.get("axis_class"))
                        if approach_entry.get("axis_class") is not None
                        else None
                    ),
                    target_axis_class=(
                        str(target_entry.get("axis_class"))
                        if target_entry.get("axis_class") is not None
                        else None
                    ),
                    approach_magnitude_class=(
                        str(approach_entry.get("magnitude_class"))
                        if approach_entry.get("magnitude_class") is not None
                        else None
                    ),
                    target_magnitude_class=(
                        str(target_entry.get("magnitude_class"))
                        if target_entry.get("magnitude_class") is not None
                        else None
                    ),
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
            "position_frame": "unknown",
            "registration_available": False,
            "tip_calibration_available": True,
            "target_count": 0,
            "visit_count": 0,
            "valid_sample_count": 0,
            "invalid_sample_count": invalid_count,
            "dropped_sample_count": invalid_count,
            "overall_max_deviation_mm": None,
            "path_dependence_rms_mm": None,
            "path_dependence_max_shift_mm": None,
            "per_target_metrics": {},
            "overall_repeatability_rms_mm": None,
            "per_target_repeatability_rms_mm": {},
            "per_target_max_deviation_mm": {},
            "approach_conditioned_centroid_shift_mm": {},
            "per_target_path_dependence_mm": {},
            "group_metrics": {},
        }
    grouped = group_positions_by_key(
        [item[0] for item in valid_samples],
        key_fn=lambda sample: int(sample.target_index),
        position_fn=lambda sample: extract_tip_or_tool_position_mm(sample, tool_id=tool_id, prefer_robot_frame=True)[0],
    )
    samples_by_target: dict[int, list[tuple[Any, list[float]]]] = {}
    for sample, position in valid_samples:
        samples_by_target.setdefault(int(sample.target_index), []).append((sample, [float(value) for value in position]))
    per_target_metrics: dict[str, Any] = {}
    centroid_by_target: dict[int, np.ndarray] = {}
    per_target_repeatability: dict[str, float] = {}
    per_target_max_deviation: dict[str, float] = {}
    overall_deviation_norms: list[float] = []
    for target_index in sorted(grouped):
        positions = grouped[target_index]
        centroid = centroid_mm(positions)
        centroid_by_target[int(target_index)] = centroid
        target_samples = samples_by_target.get(int(target_index), [])
        label = next(
            (
                str(sample.extra.get("target_label"))
                for sample, _position in target_samples
                if sample.extra.get("target_label")
            ),
            f"T{int(target_index) + 1:02d}",
        )
        axis_class = next(
            (
                str(sample.extra.get("target_axis_class"))
                for sample, _position in target_samples
                if sample.extra.get("target_axis_class") not in (None, "")
            ),
            None,
        )
        magnitude_class = next(
            (
                str(sample.extra.get("target_magnitude_class"))
                for sample, _position in target_samples
                if sample.extra.get("target_magnitude_class") not in (None, "")
            ),
            None,
        )
        group_tags = sorted(
            {
                str(tag)
                for sample, _position in target_samples
                for tag in (sample.extra.get("target_groups", []) or [])
                if str(tag)
            }
        )
        deviation_norms = [
            float(np.linalg.norm(np.asarray(position, dtype=float) - centroid))
            for position in positions
        ]
        overall_deviation_norms.extend(deviation_norms)
        target_key = str(target_index)
        per_target_repeatability[target_key] = spread_rms_mm(positions, centroid_xyz=centroid)
        per_target_max_deviation[target_key] = max(deviation_norms) if deviation_norms else 0.0
        per_target_metrics[target_key] = {
            "target_index": int(target_index),
            "label": label,
            "count": len(positions),
            "centroid_mm": [float(value) for value in centroid],
            "spread_rms_mm": per_target_repeatability[target_key],
            "max_deviation_mm": per_target_max_deviation[target_key],
            "mean_deviation_mm": float(np.mean(deviation_norms)) if deviation_norms else 0.0,
            "revisit_count": len({sample.revisit_index for sample, _position in target_samples}),
            "approach_state_count": len({sample.approach_index for sample, _position in target_samples}),
            "axis_class": axis_class,
            "magnitude_class": magnitude_class,
            "group_tags": group_tags,
        }
    rms_terms = []
    conditioned_groups: dict[str, list[tuple[Any, list[float]]]] = {}
    for sample, position in valid_samples:
        center = centroid_by_target[int(sample.target_index)]
        rms_terms.append(float(np.sum((np.asarray(position, dtype=float) - center) ** 2)))
        explicit_target_label = sample.extra.get("target_label")
        explicit_approach_label = sample.extra.get("approach_label")
        target_label = str(explicit_target_label or f"T{int(sample.target_index) + 1:02d}")
        approach_label = str(
            explicit_approach_label
            or (
                f"T{int(sample.approach_index) + 1:02d}"
                if sample.approach_index is not None
                else "Unknown"
            )
        )
        if explicit_target_label or explicit_approach_label:
            key = f"{approach_label}->{target_label}"
        else:
            key = f"{sample.approach_index}->{sample.target_index}"
        conditioned_groups.setdefault(key, []).append((sample, [float(value) for value in position]))
    conditioned_metrics: dict[str, Any] = {}
    conditioned_spread: dict[str, float] = {}
    conditioned_shift: dict[str, float] = {}
    per_target_path_groups: dict[str, list[float]] = {}
    for key, rows in conditioned_groups.items():
        positions = [position for _sample, position in rows]
        group_centroid = centroid_mm(positions)
        target_index = int(rows[0][0].target_index)
        group_spread = spread_rms_mm(positions, centroid_xyz=group_centroid) if len(positions) >= 2 else 0.0
        centroid_shift = float(np.linalg.norm(group_centroid - centroid_by_target[target_index]))
        conditioned_shift[key] = centroid_shift
        per_target_path_groups.setdefault(str(target_index), []).append(centroid_shift)
        if len(positions) >= 2:
            conditioned_spread[key] = group_spread
        conditioned_metrics[key] = {
            "target_index": target_index,
            "approach_index": (
                int(rows[0][0].approach_index)
                if rows[0][0].approach_index is not None
                else None
            ),
            "sample_count": len(positions),
            "centroid_mm": [float(value) for value in group_centroid],
            "spread_rms_mm": group_spread,
            "centroid_shift_mm": centroid_shift,
            "target_label": str(rows[0][0].extra.get("target_label") or per_target_metrics[str(target_index)]["label"]),
            "approach_label": str(rows[0][0].extra.get("approach_label") or key.split("->", 1)[0]),
        }
    per_target_path_dependence = {
        target_key: {
            "mean_centroid_shift_mm": float(np.mean(shifts)),
            "max_centroid_shift_mm": float(np.max(shifts)),
            "approach_pair_count": len(shifts),
        }
        for target_key, shifts in per_target_path_groups.items()
        if shifts
    }
    path_shift_values = [float(value) for values in per_target_path_groups.values() for value in values]
    group_metrics = _compute_repeatability_group_metrics(per_target_metrics)
    visit_count = len(
        {
            (
                int(sample.target_index),
                int(sample.revisit_index) if sample.revisit_index is not None else -1,
                int(sample.approach_index) if sample.approach_index is not None else -1,
            )
            for sample, _position in valid_samples
        }
    )
    registration_available = any(frame == "robot" for frame in frames_seen)
    status = STATUS_SUCCESS if registration_available else STATUS_PARTIAL_SUCCESS
    return {
        "status": status,
        "position_frame": "robot" if registration_available else "tracker",
        "registration_available": registration_available,
        "tip_calibration_available": True,
        "target_count": len(per_target_metrics),
        "visit_count": visit_count,
        "valid_sample_count": len(valid_samples),
        "invalid_sample_count": invalid_count,
        "dropped_sample_count": invalid_count,
        "overall_repeatability_rms_mm": float(np.sqrt(np.mean(rms_terms))),
        "overall_max_deviation_mm": (float(np.max(overall_deviation_norms)) if overall_deviation_norms else None),
        "path_dependence_rms_mm": (
            float(np.sqrt(np.mean(np.square(path_shift_values))))
            if path_shift_values
            else None
        ),
        "path_dependence_max_shift_mm": (float(np.max(path_shift_values)) if path_shift_values else None),
        "per_target_metrics": per_target_metrics,
        "per_target_repeatability_rms_mm": per_target_repeatability,
        "per_target_max_deviation_mm": per_target_max_deviation,
        "approach_conditioned_spread_mm": conditioned_spread,
        "approach_conditioned_centroid_shift_mm": conditioned_shift,
        "approach_conditioned_metrics": conditioned_metrics,
        "per_target_path_dependence_mm": per_target_path_dependence,
        "group_metrics": group_metrics,
        "thesis_repeatability_goal_mm": 1.0,
        "thesis_goal_pass": float(np.sqrt(np.mean(rms_terms))) <= 1.0,
        "thesis_goal_margin_mm": 1.0 - float(np.sqrt(np.mean(rms_terms))),
    }


def analyze_repeatability_dataset(path: Path, *, tool_id: str = "0A") -> dict[str, Any]:
    """Load one saved repeatability dataset and recompute its concise metrics."""
    bundle = ExperimentDatasetLoader().load_dataset(Path(path))
    metrics = compute_repeatability_metrics(bundle.samples, tool_id=tool_id)
    metrics["run_id"] = bundle.metadata.run_id
    metrics["experiment_name"] = bundle.metadata.experiment_name
    metrics["summary_status"] = bundle.summary.status
    return metrics


def _build_single_segment_ring_targets(
    *,
    low_magnitude_cm: float,
    high_magnitude_cm: float,
    tendon_count: int,
) -> list[dict[str, Any]]:
    """Legacy-inspired repeatability targets for the current 4-tendon single segment."""
    if int(tendon_count) != 4:
        raise ValueError(
            "The single_segment_ring_17 preset is currently defined for the 4-tendon single-segment robot. "
            "Use the manual target set for other tendon counts."
        )
    catalog = [
        {
            "target_index": 0,
            "label": "Home",
            "tendon_deltas_cm": [0.0, 0.0, 0.0, 0.0],
            "axis_class": "home",
            "magnitude_class": "home",
            "angle_deg": None,
            "group_tags": ["home"],
        }
    ]
    target_index = 1
    for magnitude_class, magnitude_cm in (("low", low_magnitude_cm), ("high", high_magnitude_cm)):
        for angle_deg in range(0, 360, 45):
            angle_rad = np.deg2rad(float(angle_deg))
            deltas = np.asarray(
                [
                    -magnitude_cm * np.cos(angle_rad),
                    -magnitude_cm * np.sin(angle_rad),
                    magnitude_cm * np.cos(angle_rad),
                    magnitude_cm * np.sin(angle_rad),
                ],
                dtype=float,
            )
            axis_class = "on_axis" if angle_deg % 90 == 0 else "off_axis"
            catalog.append(
                {
                    "target_index": int(target_index),
                    "label": f"{magnitude_class[0].upper()}{angle_deg:03d}",
                    "tendon_deltas_cm": [float(value) for value in deltas.tolist()],
                    "axis_class": axis_class,
                    "magnitude_class": magnitude_class,
                    "angle_deg": float(angle_deg),
                    "group_tags": [
                        magnitude_class,
                        axis_class,
                        f"{magnitude_class}_{axis_class}",
                    ],
                }
            )
            target_index += 1
    return catalog


def _compute_repeatability_group_metrics(per_target_metrics: dict[str, Any]) -> dict[str, Any]:
    """Summarize per-target spread by useful legacy-inspired target classes."""
    grouped_metrics: dict[str, Any] = {}
    for class_name in ("axis_class", "magnitude_class"):
        grouped_rows: dict[str, list[dict[str, Any]]] = {}
        for point_metrics in per_target_metrics.values():
            class_value = point_metrics.get(class_name)
            if class_value in (None, "", "home", "custom"):
                continue
            grouped_rows.setdefault(str(class_value), []).append(point_metrics)
        if not grouped_rows:
            continue
        grouped_metrics[class_name] = {}
        for class_value, rows in grouped_rows.items():
            spread_values = [float(row.get("spread_rms_mm", 0.0) or 0.0) for row in rows]
            max_values = [float(row.get("max_deviation_mm", 0.0) or 0.0) for row in rows]
            grouped_metrics[class_name][class_value] = {
                "target_count": len(rows),
                "mean_target_rms_mm": float(np.mean(spread_values)) if spread_values else 0.0,
                "max_target_rms_mm": float(np.max(spread_values)) if spread_values else 0.0,
                "mean_max_deviation_mm": float(np.mean(max_values)) if max_values else 0.0,
            }
    return grouped_metrics


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


def build_grid_truth_catalog(
    config: GridDefinitionConfig,
    *,
    project_root: Path,
) -> list[dict[str, Any]]:
    """Return labeled truth-grid points for the custom Aurora grid page and backend."""
    truth_points = load_grid_truth_points(config, project_root=project_root)
    return [
        {
            "label": f"P{index + 1:02d}",
            "target_index": index,
            "truth_point_mm": [float(value) for value in point],
        }
        for index, point in enumerate(truth_points)
    ]


def build_grid_accuracy_preview(
    config: GridDefinitionConfig,
    *,
    project_root: Path,
) -> GridAccuracyPreview:
    """Normalize captured point records into canonical samples plus aligned residual metrics."""
    truth_catalog = build_grid_truth_catalog(config, project_root=project_root)
    samples = grid_capture_records_to_samples(config.captured_points, truth_catalog=truth_catalog, tool_id=config.tool_id)
    # Tip availability is gated on a valid pivot calibration — there is no
    # hardcoded vector or coil-origin fallback for the grid accuracy run.
    tip_calibration_available = resolve_grid_tip_source(config, project_root=project_root).available
    metrics = compute_grid_accuracy_metrics(
        samples,
        truth_points_mm=[entry["truth_point_mm"] for entry in truth_catalog],
        tool_id=config.tool_id,
        truth_frame="grid_local",
        outlier_threshold_mm=config.outlier_threshold_mm,
        registration_available=False,
        tip_calibration_available=tip_calibration_available,
        require_tip_calibration=True,
        allow_coil_origin_fallback=False,
    )
    metrics.update(
        _grid_capture_progress_metrics(
            captured_points=config.captured_points,
            truth_catalog=truth_catalog,
            expected_samples=max(1, int(config.samples_per_point)),
        )
    )
    return GridAccuracyPreview(truth_catalog=truth_catalog, samples=samples, metrics=metrics)


@dataclass(frozen=True)
class GridTipSource:
    """Resolved tip vector for the grid run, with pivot-calibration provenance.

    The grid accuracy experiment REQUIRES the tip vector to come from a real,
    current pivot calibration — there is no hardcoded vector and no coil-origin
    fallback. ``available`` is False when no valid pivot calibration exists, in
    which case the run is refused.
    """

    tip_vector_mm: list[float] | None
    available: bool
    source: str  # "pivot" or "none"
    pivot_run_name: str | None = None
    pivot_rmse_mm: float | None = None
    pivot_summary_path: str | None = None


def find_latest_valid_pivot_calibration(
    project_root: Path,
    *,
    tool_id: str = "0B",
) -> GridTipSource:
    """Return the tip vector from the latest VALID pivot calibration for ``tool_id``.

    Valid = a ``data/pivot_calibration/<ts>_pivot_calibration[_review]/`` run
    whose ``summary.json`` has ``status == "success"``, a matching
    ``pivot_input_tool_id``, and a 3-vector ``tip_vector_local_mm``. Runs are
    scanned newest-first by directory name (timestamp prefix). Returns a
    GridTipSource with ``available=False`` when none qualifies.
    """
    root = Path(project_root) / "data" / "pivot_calibration"
    if not root.exists():
        return GridTipSource(None, False, "none")
    tool = str(tool_id or "").strip().upper()
    try:
        run_dirs = sorted(
            (d for d in root.iterdir() if d.is_dir() and d.name not in {"captures", "staged"}),
            key=lambda d: d.name,
            reverse=True,
        )
    except OSError:
        return GridTipSource(None, False, "none")
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if str(payload.get("status", "")).strip().lower() != "success":
            continue
        metrics = payload.get("experiment_metrics", {})
        if not isinstance(metrics, dict):
            continue
        run_tool = str(metrics.get("pivot_input_tool_id", "")).strip().upper()
        if tool and run_tool and run_tool != tool:
            continue
        tip = metrics.get("tip_vector_local_mm")
        if not isinstance(tip, (list, tuple)) or len(tip) != 3:
            continue
        try:
            vector = [float(value) for value in tip]
        except (TypeError, ValueError):
            continue
        rmse = metrics.get("rmse_mm")
        return GridTipSource(
            tip_vector_mm=vector,
            available=True,
            source="pivot",
            pivot_run_name=run_dir.name,
            pivot_rmse_mm=(float(rmse) if isinstance(rmse, (int, float)) else None),
            pivot_summary_path=str(summary_path),
        )
    return GridTipSource(None, False, "none")


def resolve_grid_tip_source(
    config: GridDefinitionConfig,
    *,
    project_root: Path,
) -> GridTipSource:
    """The grid tip vector MUST come from the latest valid pivot calibration.

    Any hardcoded ``config.tip_vector_mm`` / ``tip_file`` is intentionally
    ignored: a thesis-grade grid accuracy run requires a real, current pivot
    calibration for the probe (tool 0B).
    """
    return find_latest_valid_pivot_calibration(project_root, tool_id=config.tool_id)


def resolve_grid_tip_vector(
    config: GridDefinitionConfig,
    *,
    project_root: Path,
) -> tuple[list[float] | None, bool]:
    """Back-compat wrapper: returns ``(tip_vector_mm, available)`` sourced only
    from the latest valid pivot calibration."""
    source = resolve_grid_tip_source(config, project_root=project_root)
    return source.tip_vector_mm, source.available


def grid_capture_records_to_samples(
    captured_points: list[dict[str, Any]],
    *,
    truth_catalog: list[dict[str, Any]],
    tool_id: str,
) -> list[ExperimentTimeseriesSample]:
    """Convert page-side captured point records into canonical experiment samples."""
    truth_by_index = {
        int(entry["target_index"]): entry
        for entry in truth_catalog
    }
    output: list[ExperimentTimeseriesSample] = []
    for point_record in captured_points or []:
        if not isinstance(point_record, dict):
            continue
        try:
            target_index = int(point_record.get("target_index"))
        except Exception:
            continue
        truth_entry = truth_by_index.get(target_index)
        if truth_entry is None:
            continue
        label = str(point_record.get("label") or truth_entry["label"])
        raw_samples = point_record.get("raw_samples", []) or []
        for sample_index, raw_sample in enumerate(raw_samples):
            if not isinstance(raw_sample, dict):
                continue
            position_mm = raw_sample.get("position_mm")
            if not (isinstance(position_mm, list) and len(position_mm) == 3):
                continue
            quaternion = raw_sample.get("quaternion_wxyz")
            if not (isinstance(quaternion, list) and len(quaternion) == 4):
                quaternion = [1.0, 0.0, 0.0, 0.0]
            tracker_frame_id = raw_sample.get("tracker_frame_id")
            freshness_s = raw_sample.get("freshness_s")
            status_flags = sorted(set(str(flag) for flag in raw_sample.get("status_flags", []) or []))
            capture_mode = str(raw_sample.get("capture_mode", "legacy_manual_capture") or "legacy_manual_capture")
            if capture_mode == "synthetic_dry_run":
                status_flags = sorted(set(status_flags + ["dry_run", "synthetic_capture"]))
            if raw_sample.get("position_source") == "coil_origin":
                status_flags.append("coil_origin_fallback")
            output.append(
                ExperimentTimeseriesSample(
                    monotonic_time_s=float(raw_sample.get("monotonic_time_s", float(sample_index))),
                    wall_time_utc=str(
                        raw_sample.get("wall_time_utc")
                        or datetime.now(timezone.utc).isoformat()
                    ),
                    phase="sample",
                    step_index=target_index,
                    sample_index=sample_index,
                    cycle_index=None,
                    target_index=target_index,
                    revisit_index=int(point_record.get("capture_round", 0) or 0),
                    approach_index=None,
                    commanded_motor_values={},
                    commanded_cable_deltas_cm=[],
                    tracker_frame_id=int(tracker_frame_id) if tracker_frame_id is not None else None,
                    tool_ids_seen=[str(tool_id)],
                    transform_validity={str(tool_id): str(raw_sample.get("tracking_state", "valid"))},
                    pose_in_tracker_frame={
                        str(tool_id): {
                            "tracking_state": str(raw_sample.get("tracking_state", "valid")),
                            "translation_mm": [float(value) for value in position_mm],
                            "quaternion_wxyz": [float(value) for value in quaternion],
                            "frame_number": int(tracker_frame_id) if tracker_frame_id is not None else None,
                        }
                    },
                    pose_in_robot_frame={},
                    freshness_s=(float(freshness_s) if freshness_s is not None else None),
                    latency_s=(float(freshness_s) if freshness_s is not None else None),
                    status_flags=sorted(set(status_flags)),
                    backend_health={
                        "position_source": str(raw_sample.get("position_source", "tracker_tool")),
                        "measurement_mode": "captured_point",
                        "capture_mode": capture_mode,
                        "backend_identity": raw_sample.get("backend_identity"),
                        "selected_backend_name": raw_sample.get("selected_backend_name"),
                        "backend_frame_counter": raw_sample.get("backend_frame_counter"),
                        "last_packet_utc": raw_sample.get("last_packet_utc"),
                    },
                    extra={
                        "truth_label": label,
                        "truth_point_mm": [float(value) for value in truth_entry["truth_point_mm"]],
                        "position_source": str(raw_sample.get("position_source", "tracker_tool")),
                        "capture_mode": capture_mode,
                        "capture_wall_time_utc": raw_sample.get("wall_time_utc"),
                        "capture_monotonic_time_s": raw_sample.get("monotonic_time_s"),
                        "synthetic_seed_used": raw_sample.get("synthetic_seed_used"),
                        "tool_translation_mm": raw_sample.get("tool_translation_mm"),
                    },
                )
            )
    return output


def _captured_points_contain_synthetic_samples(captured_points: list[dict[str, Any]]) -> bool:
    """Return True if saved page-side captures include synthetic dry-run samples."""
    for point in captured_points or []:
        if not isinstance(point, dict):
            continue
        for raw_sample in point.get("raw_samples", []) or []:
            if not isinstance(raw_sample, dict):
                continue
            flags = {str(flag).lower() for flag in raw_sample.get("status_flags", []) or []}
            capture_mode = str(raw_sample.get("capture_mode", "") or "").lower()
            if capture_mode.startswith("synthetic") or "dry_run" in flags or "synthetic_capture" in flags:
                return True
    return False


def capture_grid_measurement_from_snapshot(
    snapshot,
    *,
    tool_id: str,
    tip_vector_mm: list[float] | None,
    require_tip_calibration: bool,
    allow_coil_origin_fallback: bool,
) -> dict[str, Any]:
    """Extract one tracker-frame measurement for the selected 0B capture point."""
    if snapshot.tracker_data_stale:
        raise RuntimeError("Tracker telemetry is stale. Wait for fresh frames before capturing.")
    tool = snapshot.tools.get(tool_id)
    if tool is None or not tool.present:
        raise RuntimeError(f"Tool {tool_id} is not currently visible.")
    if tool.translation_mm is None:
        raise RuntimeError(f"Tool {tool_id} translation is unavailable.")
    if tool.quaternion_wxyz is None:
        raise RuntimeError(f"Tool {tool_id} orientation is unavailable.")
    if tool.valid is False:
        raise RuntimeError(f"Tool {tool_id} is invalid for capture.")
    if str(tool.tracking_state).lower() not in {"tracked", "valid", "tracking", "enabled", "ok", "visible"}:
        raise RuntimeError(f"Tool {tool_id} is not valid for capture ({tool.tracking_state}).")

    position_source = "tip"
    position_mm = None
    if tip_vector_mm is not None:
        rotation = quat_wxyz_to_rotmat(tuple(float(value) for value in tool.quaternion_wxyz))
        translation = np.asarray(tool.translation_mm, dtype=float)
        tip_offset = np.asarray(tip_vector_mm, dtype=float)
        position_mm = (rotation @ tip_offset) + translation
    elif require_tip_calibration and not allow_coil_origin_fallback:
        raise RuntimeError("Tip calibration is required for this grid capture.")
    else:
        position_mm = np.asarray(tool.translation_mm, dtype=float)
        position_source = "coil_origin"

    status_flags: list[str] = []
    if snapshot.tracker_data_stale:
        status_flags.append("tracker_data_stale")
    status_flags.extend(str(flag) for flag in snapshot.tracker_faults)
    status_flags.extend(str(flag) for flag in snapshot.pipeline_faults)
    return {
        "position_mm": [float(value) for value in position_mm.tolist()],
        "tool_translation_mm": [float(value) for value in tool.translation_mm],
        "quaternion_wxyz": [float(value) for value in tool.quaternion_wxyz],
        "tracker_frame_id": tool.frame_number,
        "freshness_s": snapshot.tracker_data_age_s,
        "tracking_state": str(tool.tracking_state),
        "status_flags": status_flags,
        "position_source": position_source,
        "capture_mode": "live_tracker",
        "backend_identity": getattr(snapshot, "backend_identity", None),
        "selected_backend_name": getattr(snapshot, "selected_backend_name", None),
        "backend_frame_counter": getattr(snapshot, "backend_frame_counter", None),
        "last_packet_utc": getattr(snapshot, "last_packet_utc", None),
    }


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
    """Compute aligned grid-consistency metrics from canonical samples."""
    measurement_samples = [sample for sample in samples if sample.phase == "sample" and sample.target_index is not None]
    grouped = group_positions_by_key(
        measurement_samples,
        key_fn=lambda sample: int(sample.target_index),
        position_fn=lambda sample: extract_tip_or_tool_position_mm(
            sample,
            tool_id=tool_id,
            prefer_robot_frame=False,
        )[0],
    )
    valid_point_count = sum(1 for positions in grouped.values() if positions)
    if valid_point_count == 0:
        return {
            "status": STATUS_INVALID_INSUFFICIENT_SAMPLES,
            "per_point_metrics": {},
            "overall_rms_error_mm": None,
            "overall_rms_residual_mm": None,
            "max_residual_mm": None,
            "mean_within_point_spread_mm": None,
            "max_within_point_spread_mm": None,
            "per_axis_bias_mm": None,
            "outlier_count": 0,
            "residual_outlier_count": 0,
            "raw_sample_count": 0,
            "accepted_sample_count": 0,
            "point_count_aligned": 0,
            "registration_available": registration_available,
            "tip_calibration_available": tip_calibration_available,
            "tip_calibration_used": bool(tip_calibration_available),
            "coil_origin_fallback_used": False,
            "position_source_counts": {},
            "capture_mode_counts": {},
            "dry_run_sample_count": 0,
            "live_tracker_sample_count": 0,
        }
    truth_by_index = {index: [float(value) for value in truth_points_mm[index]] for index in range(len(truth_points_mm))}
    samples_by_index: dict[int, list[Any]] = {}
    for sample in measurement_samples:
        samples_by_index.setdefault(int(sample.target_index), []).append(sample)

    per_point_metrics: dict[str, Any] = {}
    measured_centroids: list[list[float]] = []
    truth_used: list[list[float]] = []
    aligned_keys: list[str] = []
    raw_sample_count = 0
    accepted_sample_count = 0
    outlier_count = 0
    per_point_spreads: list[float] = []
    position_source_counts: dict[str, int] = {}
    capture_mode_counts: dict[str, int] = {}
    for point_index in sorted(grouped):
        positions = grouped.get(point_index, [])
        if not positions:
            continue
        raw_sample_count += len(positions)
        raw_positions_arr = as_points_n3(positions, name="grid_point_samples")
        raw_centroid = centroid_mm(positions)
        robust_center = np.median(raw_positions_arr, axis=0)
        raw_spread = spread_rms_mm(positions, centroid_xyz=raw_centroid)
        distances = [
            float(np.linalg.norm(np.asarray(position, dtype=float) - robust_center))
            for position in positions
        ]
        accepted_indices = [
            index
            for index, distance in enumerate(distances)
            if distance <= float(outlier_threshold_mm)
        ]
        if not accepted_indices:
            accepted_indices = list(range(len(positions)))
        rejected_indices = [index for index in range(len(positions)) if index not in accepted_indices]
        accepted_positions = [positions[index] for index in accepted_indices]
        accepted_centroid = centroid_mm(accepted_positions)
        accepted_spread = spread_rms_mm(accepted_positions, centroid_xyz=accepted_centroid)
        accepted_sample_count += len(accepted_positions)
        outlier_count += len(rejected_indices)
        point_samples = samples_by_index.get(point_index, [])
        point_position_source_counts: dict[str, int] = {}
        point_capture_mode_counts: dict[str, int] = {}
        for sample in point_samples:
            position_source = str(sample.extra.get("position_source", "tracker_tool") or "tracker_tool")
            position_source_counts[position_source] = int(position_source_counts.get(position_source, 0) or 0) + 1
            point_position_source_counts[position_source] = int(point_position_source_counts.get(position_source, 0) or 0) + 1
            capture_mode = str(sample.extra.get("capture_mode", "unknown") or "unknown")
            capture_mode_counts[capture_mode] = int(capture_mode_counts.get(capture_mode, 0) or 0) + 1
            point_capture_mode_counts[capture_mode] = int(point_capture_mode_counts.get(capture_mode, 0) or 0) + 1
        label = next(
            (
                str(sample.extra.get("truth_label"))
                for sample in point_samples
                if sample.extra.get("truth_label")
            ),
            f"P{point_index + 1:02d}",
        )
        truth_point = truth_by_index.get(point_index)
        point_metrics = {
            "target_index": int(point_index),
            "label": label,
            "truth_point_mm": [float(value) for value in truth_point] if truth_point is not None else None,
            "raw_sample_count": len(positions),
            "accepted_sample_count": len(accepted_positions),
            "accepted_sample_indices": accepted_indices,
            "rejected_sample_indices": rejected_indices,
            "centroid_mm": [float(value) for value in accepted_centroid],
            "mean_measured_position_mm": [float(value) for value in accepted_centroid],
            "sample_spread_rms_mm": accepted_spread,
            "raw_sample_spread_rms_mm": raw_spread,
            "outlier_count": len(rejected_indices),
            "position_source_counts": point_position_source_counts,
            "position_source_summary": ", ".join(
                f"{source}={count}"
                for source, count in sorted(point_position_source_counts.items())
            )
            or "n/a",
            "capture_mode_counts": point_capture_mode_counts,
            "capture_mode_summary": ", ".join(
                f"{mode}={count}"
                for mode, count in sorted(point_capture_mode_counts.items())
            )
            or "n/a",
            "status": (
                "complete" if len(positions) > 0 else "not_captured"
            ),
        }
        per_point_metrics[label] = point_metrics
        per_point_spreads.append(float(accepted_spread))
        if truth_point is not None:
            truth_used.append([float(value) for value in truth_point])
            measured_centroids.append([float(value) for value in accepted_centroid])
            aligned_keys.append(label)

    overall_rms = None
    max_residual = None
    bias = None
    residual_outlier_count = 0
    alignment_transform = None
    mean_spread = float(np.mean(per_point_spreads)) if per_point_spreads else None
    max_spread = float(np.max(per_point_spreads)) if per_point_spreads else None
    per_point_residuals: dict[str, float] = {}
    point_count_total = len(truth_points_mm)
    point_count_captured = len(per_point_metrics)
    point_count_missing = max(0, point_count_total - point_count_captured)
    point_coverage_fraction = (
        float(point_count_captured / point_count_total)
        if point_count_total > 0
        else None
    )
    alignment_ready = len(measured_centroids) >= 3
    alignment_ready_reason = (
        "At least three labeled point centroids are available for the rigid alignment solve."
        if alignment_ready
        else (
            f"Capture {max(0, 3 - len(measured_centroids))} more labeled point(s) before aligned RMS residuals are available."
        )
    )
    if measured_centroids and len(measured_centroids) >= 3:
        truth_arr = as_points_n3(truth_used, name="truth_points_mm")
        measured_arr = as_points_n3(measured_centroids, name="measured_centroids_mm")
        try:
            alignment = _solve_rigid_fit_truth_to_measured(truth_arr, measured_arr)
        except ValueError:
            alignment = None
        if alignment is not None:
            aligned_truth = alignment["aligned_truth_mm"]
            residual_vectors = measured_arr - aligned_truth
            residual_norms = np.linalg.norm(residual_vectors, axis=1)
            overall_rms = float(np.sqrt(np.mean(residual_norms ** 2)))
            max_residual = float(np.max(residual_norms)) if residual_norms.size else None
            bias = [float(value) for value in residual_vectors.mean(axis=0)]
            residual_outlier_count = int(np.sum(residual_norms > float(outlier_threshold_mm)))
            alignment_transform = alignment["T_truth_to_measured"].tolist()
            for index, label in enumerate(aligned_keys):
                point_metrics = per_point_metrics[label]
                point_metrics["aligned_truth_point_mm"] = [float(value) for value in aligned_truth[index].tolist()]
                aligned_centroid_truth = alignment["measured_in_truth_mm"][index]
                point_metrics["aligned_centroid_truth_mm"] = [float(value) for value in aligned_centroid_truth.tolist()]
                point_metrics["residual_vector_mm"] = [float(value) for value in residual_vectors[index].tolist()]
                point_metrics["residual_mm"] = float(residual_norms[index])
                per_point_residuals[label] = float(residual_norms[index])

    coil_origin_fallback_used = bool(
        position_source_counts.get("coil_origin", 0)
        or position_source_counts.get("synthetic_coil_origin", 0)
    )
    tip_calibration_used = bool(
        position_source_counts.get("tip", 0)
        or position_source_counts.get("synthetic_tip", 0)
    ) and not coil_origin_fallback_used
    dry_run_sample_count = int(capture_mode_counts.get("synthetic_dry_run", 0) or 0)
    live_tracker_sample_count = int(capture_mode_counts.get("live_tracker", 0) or 0)

    if require_tip_calibration and (coil_origin_fallback_used or not tip_calibration_available) and not allow_coil_origin_fallback:
        status = STATUS_INVALID_MISSING_TIP_CAL
    elif overall_rms is None:
        status = STATUS_INVALID_INSUFFICIENT_SAMPLES
    elif require_tip_calibration and (coil_origin_fallback_used or not tip_calibration_available) and allow_coil_origin_fallback:
        status = STATUS_PARTIAL_SUCCESS
    else:
        status = STATUS_SUCCESS
    return {
        "status": status,
        "per_point_metrics": per_point_metrics,
        # `per_point_residual_mm` is the canonical key; `pointwise_rms_error_mm`
        # is kept as a back-compat alias for older summaries / external tools.
        # Both point to the same dict — do not branch on them differently.
        "per_point_residual_mm": per_point_residuals,
        "pointwise_rms_error_mm": per_point_residuals,
        # Same alias pattern for the overall metric.
        "overall_rms_residual_mm": overall_rms,
        "overall_rms_error_mm": overall_rms,
        "max_residual_mm": max_residual,
        "mean_within_point_spread_mm": mean_spread,
        "max_within_point_spread_mm": max_spread,
        "per_axis_bias_mm": bias,
        "outlier_count": outlier_count,
        "residual_outlier_count": residual_outlier_count,
        "raw_sample_count": raw_sample_count,
        "accepted_sample_count": accepted_sample_count,
        "valid_sample_count": accepted_sample_count,
        "rejected_sample_count": outlier_count,
        "point_count_total": point_count_total,
        "point_count_captured": point_count_captured,
        "point_count_missing": point_count_missing,
        "point_coverage_fraction": point_coverage_fraction,
        "point_count_aligned": len(per_point_residuals),
        "alignment_ready": alignment_ready,
        "alignment_ready_reason": alignment_ready_reason,
        "alignment_transform_truth_to_measured": alignment_transform,
        "position_source_counts": position_source_counts,
        "capture_mode_counts": capture_mode_counts,
        "dry_run_sample_count": dry_run_sample_count,
        "live_tracker_sample_count": live_tracker_sample_count,
        "registration_available": registration_available,
        "tip_calibration_available": tip_calibration_available,
        "tip_calibration_used": tip_calibration_used,
        "coil_origin_fallback_used": coil_origin_fallback_used,
    }


def _grid_tip_calibration_used_by_samples(
    samples,
    *,
    fallback_configured: bool,
) -> bool:
    """Return whether the current captured dataset actually used tip-based point positions."""
    position_sources = {
        str(sample.extra.get("position_source", "") or "")
        for sample in samples or []
        if getattr(sample, "extra", None)
    }
    if not position_sources:
        return bool(fallback_configured)
    return bool({"tip", "synthetic_tip"} & position_sources) and not bool(
        {"coil_origin", "synthetic_coil_origin"} & position_sources
    )


def _grid_capture_progress_metrics(
    *,
    captured_points: list[dict[str, Any]],
    truth_catalog: list[dict[str, Any]],
    expected_samples: int,
) -> dict[str, Any]:
    """Summarize page-side capture completeness for the labeled grid workflow."""
    complete = 0
    partial = 0
    for point in captured_points or []:
        if not isinstance(point, dict):
            continue
        raw_count = len(point.get("raw_samples", []) or [])
        if raw_count >= max(1, int(expected_samples)):
            complete += 1
        elif raw_count > 0:
            partial += 1
    total = len(truth_catalog)
    return {
        "point_count_complete": int(complete),
        "point_count_partial": int(partial),
        "point_count_not_started": max(0, int(total) - int(complete) - int(partial)),
    }


def _solve_rigid_fit_truth_to_measured(
    truth_points: np.ndarray,
    measured_points: np.ndarray,
) -> dict[str, np.ndarray]:
    """Solve the best-fit rigid transform from truth-grid coordinates to measured centroids."""
    truth = as_points_n3(truth_points, name="truth_points")
    measured = as_points_n3(measured_points, name="measured_points")
    if truth.shape != measured.shape:
        raise ValueError("truth_points and measured_points must have the same shape")
    if truth.shape[0] < 3:
        raise ValueError("At least three labeled grid points are required for rigid alignment")
    truth_centroid = truth.mean(axis=0)
    measured_centroid = measured.mean(axis=0)
    truth_centered = truth - truth_centroid
    measured_centered = measured - measured_centroid
    if np.linalg.matrix_rank(truth_centered) < 2 or np.linalg.matrix_rank(measured_centered) < 2:
        raise ValueError("Grid points must span a 2D plane to support aligned residual analysis")
    covariance = truth_centered.T @ measured_centered
    U, _singular_values, Vt = np.linalg.svd(covariance)
    rotation = Vt.T @ U.T
    if np.linalg.det(rotation) < 0.0:
        Vt[-1, :] *= -1.0
        rotation = Vt.T @ U.T
    translation = measured_centroid - (rotation @ truth_centroid)
    aligned_truth = (rotation @ truth.T).T + translation
    measured_in_truth = (rotation.T @ (measured - translation).T).T
    transform = np.eye(4, dtype=float)
    transform[0:3, 0:3] = rotation
    transform[0:3, 3] = translation
    return {
        "rotation": rotation,
        "translation": translation,
        "aligned_truth_mm": aligned_truth,
        "measured_in_truth_mm": measured_in_truth,
        "T_truth_to_measured": transform,
    }


def _synthetic_grid_alignment(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic truth-grid placement in tracker frame for dry-run captures."""
    rng = np.random.default_rng(seed or 0)
    angles_rad = rng.uniform(low=-0.18, high=0.18, size=3)
    cx, cy, cz = np.cos(angles_rad)
    sx, sy, sz = np.sin(angles_rad)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    rotation = Rz @ Ry @ Rx
    translation = np.asarray([18.0, -24.0, 42.0], dtype=float)
    return rotation, translation


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
        motion_workflow="experiment_motion",
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
    return resolve_grid_tip_vector(config, project_root=session.context.project_root)


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
