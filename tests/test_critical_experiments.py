from pathlib import Path
import json

import numpy as np

from continuum_robot.config.schemas import (
    CalibrationConfig,
    ExperimentConfig,
    RegistrationWorkflowConfig,
    RobotConfig,
    RuntimeConfig,
    SafetyConfig,
    SerialConfig,
)
from continuum_robot.config.settings import Settings
from continuum_robot.experiments.critical_experiments import (
    GridDefinitionConfig,
    RepeatabilityDatasetConfig,
    RepeatabilityScheduleConfig,
    analyze_repeatability_dataset,
    build_grid_accuracy_preview,
    build_repeatability_preview,
    compute_grid_accuracy_metrics,
    compute_repeatability_metrics,
    generate_repeatability_schedule,
)
from continuum_robot.experiments.experiment_runner import ExperimentRunner
from continuum_robot.experiments.pivot_utils import solve_pivot_calibration
from continuum_robot.experiments.schemas import ExperimentTimeseriesSample
from continuum_robot.hardware.mock_dxl_bus import MockDxlBus
from continuum_robot.services.tracking_service import TrackingService
from continuum_robot.servos.displacement_mapper import TendonDisplacementMapper
from continuum_robot.servos.neutral_calibration_service import NeutralCalibrationService
from continuum_robot.servos.pretension_validation_service import PretensionValidationService
from continuum_robot.servos.safety_guard import SafetyGuard
from continuum_robot.servos.servo_service import ServoService
from continuum_robot.tracking.mock_tracker_manager import MockTrackerManager
from continuum_robot.tracking.transforms import quat_wxyz_to_rotmat


def _settings() -> Settings:
    return Settings(
        runtime=RuntimeConfig(mock_mode=True, poll_rate_hz=20, robot_config="robot_4servo.yaml"),
        robot=RobotConfig(mode="4-servo", spool_diameter_cm=1.2, ticks_per_revolution=4096, servo_ids=[1, 2, 3, 4]),
        serial=SerialConfig(aurora_port="/dev/mock-aurora", openrb_port="/dev/mock-openrb", baudrate=115200),
        safety=SafetyConfig(
            position_min_offset_ticks=-600,
            position_max_offset_ticks=600,
            max_current_ma=850,
            pretension_current_balance_tolerance_ma=120,
        ),
        registration=RegistrationWorkflowConfig(capture_tool_id="0B", coil_tool_id="0A", max_fre_mm=None),
        experiment=ExperimentConfig(default_settle_time_s=0.0, sample_count_per_point=1, output_dir="data/experiments"),
        calibration=CalibrationConfig(
            neutral_setpoints_path="data/calibrations/neutral_setpoints.json",
            latest_registration_path="data/registrations/latest_registration.json",
        ),
    )


def _servo_service(tmp_path: Path) -> ServoService:
    return ServoService(
        dxl_bus=MockDxlBus([1, 2, 3, 4]),
        mapper=TendonDisplacementMapper(spool_diameter_cm=1.2),
        safety_guard=SafetyGuard(min_offset_ticks=-600, max_offset_ticks=600, max_current_ma=850),
        neutral_calibration=NeutralCalibrationService(path=tmp_path / "neutral.json"),
        pretension_validation=PretensionValidationService(),
    )


def _tracking_service(settings: Settings, tmp_path: Path, registration_path: Path | None = None) -> TrackingService:
    return TrackingService(
        live_backend=MockTrackerManager(poll_hz=30),
        port=settings.serial.aurora_port,
        registration_path=registration_path or (tmp_path / "latest_registration.json"),
        config_source="test",
        runtime_coil_tool_id=settings.registration.coil_tool_id,
        registration_tool_id=settings.registration.capture_tool_id,
    )


def _runner(
    tmp_path: Path,
    *,
    registration_path: Path | None = None,
) -> ExperimentRunner:
    settings = _settings()
    registration = registration_path or (tmp_path / "latest_registration.json")
    return ExperimentRunner(
        project_root=Path(__file__).resolve().parents[1],
        settings=settings,
        tracking_service=_tracking_service(settings, tmp_path, registration_path=registration),
        servo_service=_servo_service(tmp_path),
        output_dir=tmp_path / "data" / "experiments",
        default_settle_time_s=0.0,
        registration_path=registration,
        sleep_fn=lambda _seconds: None,
    )


def _sample(
    *,
    tool_id: str,
    tracker_position: list[float] | None,
    robot_position: list[float] | None,
    target_index: int,
    revisit_index: int = 0,
    approach_index: int = 0,
    sample_index: int = 0,
) -> ExperimentTimeseriesSample:
    pose_in_tracker_frame = {}
    if tracker_position is not None:
        pose_in_tracker_frame[tool_id] = {
            "translation_mm": [float(value) for value in tracker_position],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "frame_number": sample_index,
            "tracking_state": "valid",
        }
    pose_in_robot_frame = {}
    if robot_position is not None:
        pose_in_robot_frame["tip"] = {"translation_mm": [float(value) for value in robot_position]}
    return ExperimentTimeseriesSample(
        monotonic_time_s=float(sample_index),
        wall_time_utc="2026-01-01T00:00:00+00:00",
        phase="sample",
        step_index=target_index,
        sample_index=sample_index,
        target_index=target_index,
        revisit_index=revisit_index,
        approach_index=approach_index,
        tracker_frame_id=sample_index,
        tool_ids_seen=[tool_id],
        transform_validity={tool_id: "valid"},
        pose_in_tracker_frame=pose_in_tracker_frame,
        pose_in_robot_frame=pose_in_robot_frame,
        freshness_s=0.01,
        latency_s=0.01,
    )


def _pivot_rotations_and_translations() -> tuple[list[np.ndarray], list[np.ndarray]]:
    quaternions = [
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        (0.70710678, 0.70710678, 0.0, 0.0),
        (0.70710678, -0.70710678, 0.0, 0.0),
        (0.70710678, 0.0, 0.70710678, 0.0),
        (0.70710678, 0.0, -0.70710678, 0.0),
        (0.70710678, 0.0, 0.0, 0.70710678),
        (0.70710678, 0.0, 0.0, -0.70710678),
    ]
    tip = np.asarray([10.0, 20.0, 100.0], dtype=float)
    pivot = np.asarray([25.0, -10.0, 40.0], dtype=float)
    rotations = [quat_wxyz_to_rotmat(quat) for quat in quaternions]
    translations = [pivot - (rotation @ tip) for rotation in rotations]
    return rotations, translations


def _write_pivot_csv(path: Path) -> Path:
    rotations, translations = _pivot_rotations_and_translations()
    quaternions = [
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        (0.70710678, 0.70710678, 0.0, 0.0),
        (0.70710678, -0.70710678, 0.0, 0.0),
        (0.70710678, 0.0, 0.70710678, 0.0),
        (0.70710678, 0.0, -0.70710678, 0.0),
        (0.70710678, 0.0, 0.0, 0.70710678),
        (0.70710678, 0.0, 0.0, -0.70710678),
    ]
    lines = []
    for quaternion, translation in zip(quaternions, translations):
        fields = ["0B", *(f"{float(value):0.8f}" for value in quaternion), *(f"{float(value):0.8f}" for value in translation)]
        lines.append(",".join(fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_headered_pivot_csv(path: Path) -> Path:
    rotations, translations = _pivot_rotations_and_translations()
    quaternions = [
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        (0.70710678, 0.70710678, 0.0, 0.0),
        (0.70710678, -0.70710678, 0.0, 0.0),
        (0.70710678, 0.0, 0.70710678, 0.0),
        (0.70710678, 0.0, -0.70710678, 0.0),
        (0.70710678, 0.0, 0.0, 0.70710678),
        (0.70710678, 0.0, 0.0, -0.70710678),
    ]
    lines = ["tool_id,qw,qx,qy,qz,x,y,z"]
    for quaternion, translation in zip(quaternions, translations):
        fields = ["0B", *(f"{float(value):0.8f}" for value in quaternion), *(f"{float(value):0.8f}" for value in translation)]
        lines.append(",".join(fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_repeatability_schedule_generation_is_deterministic_and_cycles_approaches() -> None:
    config = RepeatabilityScheduleConfig(
        target_points_cm=[
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0, 0.0],
        ],
        revisit_count=4,
        randomize_approach_order=False,
    )
    visits = generate_repeatability_schedule(config)

    by_target = {
        target_index: [visit.approach_index for visit in visits if visit.target_index == target_index]
        for target_index in range(3)
    }
    assert by_target == {
        0: [1, 2, 1, 2],
        1: [0, 2, 0, 2],
        2: [0, 1, 0, 1],
    }

    randomized = RepeatabilityScheduleConfig(
        target_points_cm=config.target_points_cm,
        revisit_count=4,
        randomize_approach_order=True,
        seed=7,
    )
    first = generate_repeatability_schedule(randomized)
    second = generate_repeatability_schedule(randomized)
    assert [(visit.target_index, visit.approach_index) for visit in first] == [
        (visit.target_index, visit.approach_index) for visit in second
    ]


def test_repeatability_preview_builds_legacy_ring_target_catalog() -> None:
    preview = build_repeatability_preview(
        RepeatabilityDatasetConfig.from_dict(
            {
                "schedule": {
                    "target_set": "single_segment_ring_17",
                    "low_magnitude_cm": 0.10,
                    "high_magnitude_cm": 0.20,
                    "revisit_count": 4,
                    "samples_per_point": 3,
                }
            }
        ),
        tendon_count=4,
    )

    assert preview.summary["target_count"] == 17
    assert preview.summary["visit_count"] == 68
    assert preview.summary["planned_sample_count"] == 204
    assert preview.summary["axis_counts"]["on_axis"] == 8
    assert preview.summary["axis_counts"]["off_axis"] == 8
    assert preview.summary["magnitude_counts"]["low"] == 8
    assert preview.summary["magnitude_counts"]["high"] == 8
    assert preview.target_catalog[0]["label"] == "Home"
    assert preview.target_catalog[1]["label"] == "L000"
    assert preview.target_catalog[9]["label"] == "H000"


def test_repeatability_metrics_on_synthetic_samples() -> None:
    samples = [
        _sample(tool_id="0A", tracker_position=[0.1, 0.0, 0.0], robot_position=[0.0, 0.0, 0.0], target_index=0, revisit_index=0, approach_index=1, sample_index=0),
        _sample(tool_id="0A", tracker_position=[1.1, 0.0, 0.0], robot_position=[1.0, 0.0, 0.0], target_index=0, revisit_index=1, approach_index=1, sample_index=1),
        _sample(tool_id="0A", tracker_position=[0.1, 1.0, 0.0], robot_position=[0.0, 1.0, 0.0], target_index=0, revisit_index=2, approach_index=2, sample_index=2),
        _sample(tool_id="0A", tracker_position=[10.1, 0.0, 0.0], robot_position=[10.0, 0.0, 0.0], target_index=1, revisit_index=0, approach_index=0, sample_index=3),
        _sample(tool_id="0A", tracker_position=[10.1, 1.0, 0.0], robot_position=[10.0, 1.0, 0.0], target_index=1, revisit_index=1, approach_index=0, sample_index=4),
        _sample(tool_id="0A", tracker_position=[9.1, 0.0, 0.0], robot_position=[9.0, 0.0, 0.0], target_index=1, revisit_index=2, approach_index=2, sample_index=5),
    ]

    metrics = compute_repeatability_metrics(samples, tool_id="0A")

    assert metrics["status"] == "success"
    assert metrics["valid_sample_count"] == 6
    assert np.allclose(metrics["per_target_metrics"]["0"]["centroid_mm"], [1.0 / 3.0, 1.0 / 3.0, 0.0])
    assert np.allclose(metrics["per_target_metrics"]["1"]["centroid_mm"], [29.0 / 3.0, 1.0 / 3.0, 0.0])
    assert np.isclose(metrics["per_target_metrics"]["0"]["spread_rms_mm"], np.sqrt(4.0 / 9.0))
    assert np.isclose(metrics["overall_repeatability_rms_mm"], np.sqrt(4.0 / 9.0))
    assert "1->0" in metrics["approach_conditioned_spread_mm"]
    assert np.isclose(metrics["overall_max_deviation_mm"], np.sqrt(5.0) / 3.0)
    assert metrics["visit_count"] == 6
    assert metrics["target_count"] == 2


def test_repeatability_metrics_compute_grouped_and_path_dependence_metrics() -> None:
    samples = [
        _sample(tool_id="0A", tracker_position=[0.0, 0.0, 0.0], robot_position=[0.0, 0.0, 0.0], target_index=0, revisit_index=0, approach_index=1, sample_index=0),
        _sample(tool_id="0A", tracker_position=[0.2, 0.0, 0.0], robot_position=[0.2, 0.0, 0.0], target_index=0, revisit_index=1, approach_index=2, sample_index=1),
        _sample(tool_id="0A", tracker_position=[10.0, 0.0, 0.0], robot_position=[10.0, 0.0, 0.0], target_index=1, revisit_index=0, approach_index=0, sample_index=2),
        _sample(tool_id="0A", tracker_position=[10.2, 0.0, 0.0], robot_position=[10.2, 0.0, 0.0], target_index=1, revisit_index=1, approach_index=2, sample_index=3),
        _sample(tool_id="0A", tracker_position=[0.0, 10.0, 0.0], robot_position=[0.0, 10.0, 0.0], target_index=2, revisit_index=0, approach_index=0, sample_index=4),
        _sample(tool_id="0A", tracker_position=[0.3, 10.0, 0.0], robot_position=[0.3, 10.0, 0.0], target_index=2, revisit_index=1, approach_index=1, sample_index=5),
    ]
    metadata = {
        0: {"target_label": "L000", "target_axis_class": "on_axis", "target_magnitude_class": "low", "target_groups": ["low", "on_axis", "low_on_axis"]},
        1: {"target_label": "L045", "target_axis_class": "off_axis", "target_magnitude_class": "low", "target_groups": ["low", "off_axis", "low_off_axis"]},
        2: {"target_label": "H000", "target_axis_class": "on_axis", "target_magnitude_class": "high", "target_groups": ["high", "on_axis", "high_on_axis"]},
    }
    approach_labels = {0: "L000", 1: "L045", 2: "H000"}
    for sample in samples:
        sample.extra.update(metadata[int(sample.target_index)])
        sample.extra["approach_label"] = approach_labels[int(sample.approach_index)]

    metrics = compute_repeatability_metrics(samples, tool_id="0A")

    assert metrics["per_target_metrics"]["0"]["label"] == "L000"
    assert metrics["group_metrics"]["axis_class"]["on_axis"]["target_count"] == 2
    assert metrics["group_metrics"]["axis_class"]["off_axis"]["target_count"] == 1
    assert metrics["group_metrics"]["magnitude_class"]["low"]["target_count"] == 2
    assert metrics["group_metrics"]["magnitude_class"]["high"]["target_count"] == 1
    assert metrics["path_dependence_rms_mm"] is not None
    assert "L045->H000" in metrics["approach_conditioned_centroid_shift_mm"]


def test_aurora_grid_metrics_fit_truth_grid_and_report_residuals() -> None:
    samples = [
        _sample(tool_id="0B", tracker_position=[5.5, 2.0, 1.0], robot_position=None, target_index=0, revisit_index=0, sample_index=0),
        _sample(tool_id="0B", tracker_position=[4.5, 2.0, 1.0], robot_position=None, target_index=0, revisit_index=1, sample_index=1),
        _sample(tool_id="0B", tracker_position=[15.5, 2.0, 1.0], robot_position=None, target_index=1, revisit_index=0, sample_index=2),
        _sample(tool_id="0B", tracker_position=[14.5, 2.0, 1.0], robot_position=None, target_index=1, revisit_index=1, sample_index=3),
        _sample(tool_id="0B", tracker_position=[5.0, 12.5, 1.0], robot_position=None, target_index=2, revisit_index=0, sample_index=4),
        _sample(tool_id="0B", tracker_position=[5.0, 11.5, 1.0], robot_position=None, target_index=2, revisit_index=1, sample_index=5),
    ]
    for sample, label in zip(samples, ["P01", "P01", "P02", "P02", "P03", "P03"]):
        sample.extra["truth_label"] = label
        sample.extra["truth_point_mm"] = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]][int(label[-2:]) - 1]

    metrics = compute_grid_accuracy_metrics(
        samples,
        truth_points_mm=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
        tool_id="0B",
        truth_frame="grid_local",
        outlier_threshold_mm=1.0,
        registration_available=False,
        tip_calibration_available=True,
        require_tip_calibration=False,
        allow_coil_origin_fallback=True,
    )

    assert metrics["status"] == "success"
    assert np.isclose(metrics["overall_rms_error_mm"], 0.0, atol=1e-9)
    assert np.isclose(metrics["mean_within_point_spread_mm"], 0.5)
    assert np.isclose(metrics["per_point_metrics"]["P01"]["sample_spread_rms_mm"], 0.5)
    assert np.isclose(metrics["per_point_metrics"]["P02"]["residual_mm"], 0.0, atol=1e-9)
    assert metrics["point_count_aligned"] == 3
    assert metrics["alignment_ready"] is True
    assert metrics["point_count_captured"] == 3
    assert np.isclose(metrics["point_coverage_fraction"], 1.0)


def test_aurora_grid_preview_normalizes_captured_points_and_rejects_outliers(tmp_path: Path) -> None:
    preview = build_grid_accuracy_preview(
        config=GridDefinitionConfig.from_dict({
            "rows": 2,
            "cols": 2,
            "spacing_mm": 25.4,
            "samples_per_point": 3,
            "tool_id": "0B",
            "use_tip_calibration": True,
            "tip_vector_mm": [0.0, 0.0, 125.0],
            "outlier_threshold_mm": 1.0,
            "captured_points": [
                {
                    "label": "P01",
                    "target_index": 0,
                    "raw_samples": [
                        {"position_mm": [10.0, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0], "tracking_state": "valid", "position_source": "tip"},
                        {"position_mm": [10.2, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0], "tracking_state": "valid", "position_source": "tip"},
                        {"position_mm": [14.5, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0], "tracking_state": "valid", "position_source": "tip"},
                    ],
                },
                {
                    "label": "P02",
                    "target_index": 1,
                    "raw_samples": [
                        {"position_mm": [35.4, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0], "tracking_state": "valid", "position_source": "tip"},
                        {"position_mm": [35.6, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0], "tracking_state": "valid", "position_source": "tip"},
                        {"position_mm": [35.5, 0.1, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0], "tracking_state": "valid", "position_source": "tip"},
                    ],
                },
                {
                    "label": "P03",
                    "target_index": 2,
                    "raw_samples": [
                        {"position_mm": [10.0, 25.4, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0], "tracking_state": "valid", "position_source": "tip"},
                        {"position_mm": [10.1, 25.5, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0], "tracking_state": "valid", "position_source": "tip"},
                        {"position_mm": [9.9, 25.3, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0], "tracking_state": "valid", "position_source": "tip"},
                    ],
                },
            ],
        }),
        project_root=tmp_path,
    )

    assert len(preview.samples) == 9
    assert preview.metrics["outlier_count"] == 1
    assert preview.metrics["rejected_sample_count"] == 1
    assert preview.metrics["per_point_metrics"]["P01"]["accepted_sample_count"] == 2
    assert preview.metrics["point_count_aligned"] == 3
    assert preview.metrics["point_count_total"] == 4
    assert preview.metrics["point_count_captured"] == 3
    assert preview.metrics["alignment_ready"] is True
    assert preview.metrics["position_source_counts"]["tip"] == 9


def test_aurora_grid_metrics_reports_within_point_spread_before_alignment_ready() -> None:
    samples = [
        _sample(tool_id="0B", tracker_position=[5.2, 2.0, 1.0], robot_position=None, target_index=0, sample_index=0),
        _sample(tool_id="0B", tracker_position=[4.8, 2.0, 1.0], robot_position=None, target_index=0, sample_index=1),
        _sample(tool_id="0B", tracker_position=[15.1, 2.0, 1.0], robot_position=None, target_index=1, sample_index=2),
        _sample(tool_id="0B", tracker_position=[14.9, 2.0, 1.0], robot_position=None, target_index=1, sample_index=3),
    ]
    for sample, label in zip(samples, ["P01", "P01", "P02", "P02"]):
        sample.extra["truth_label"] = label
        sample.extra["truth_point_mm"] = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]][int(label[-2:]) - 1]
        sample.extra["position_source"] = "tip"

    metrics = compute_grid_accuracy_metrics(
        samples,
        truth_points_mm=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
        tool_id="0B",
        truth_frame="grid_local",
        outlier_threshold_mm=1.0,
        registration_available=False,
        tip_calibration_available=True,
        require_tip_calibration=False,
        allow_coil_origin_fallback=True,
    )

    assert metrics["alignment_ready"] is False
    assert metrics["overall_rms_residual_mm"] is None
    assert metrics["mean_within_point_spread_mm"] is not None
    assert metrics["point_count_captured"] == 2
    assert "Capture 1 more labeled point" in metrics["alignment_ready_reason"]


def test_pivot_calibration_least_squares_recovers_tip_on_synthetic_data() -> None:
    rotations, translations = _pivot_rotations_and_translations()
    result = solve_pivot_calibration(rotations, translations, std_dev_threshold=3.0, min_samples=8)

    assert np.allclose(result.tip_vector_local_mm, [10.0, 20.0, 100.0], atol=1e-6)
    assert np.allclose(result.pivot_point_tracker_mm, [25.0, -10.0, 40.0], atol=1e-6)
    assert result.sample_count_used == len(rotations)
    assert result.sample_count_rejected == 0
    assert result.rmse_mm < 1e-6


def test_pivot_calibration_outlier_rejection_rejects_large_outliers() -> None:
    rotations, translations = _pivot_rotations_and_translations()
    corrupted = list(translations)
    corrupted[0] = np.asarray(corrupted[0], dtype=float) + np.asarray([25.0, -30.0, 40.0], dtype=float)
    result = solve_pivot_calibration(rotations, corrupted, std_dev_threshold=2.0, min_samples=8)

    assert result.sample_count_rejected >= 1
    assert np.allclose(result.tip_vector_local_mm, [10.0, 20.0, 100.0], atol=1e-3)
    assert np.allclose(result.pivot_point_tracker_mm, [25.0, -10.0, 40.0], atol=1e-3)


def test_pivot_calibration_offline_from_recorded_file(tmp_path: Path) -> None:
    csv_path = _write_pivot_csv(tmp_path / "pivot_samples.csv")
    runner = _runner(tmp_path)
    result = runner.run_experiment(
        "pivot_calibration",
        config={
            "tool_id": "0B",
            "input_path": str(csv_path),
            "output_tip_file": str(tmp_path / "tip.csv"),
            "std_dev_threshold": 3.0,
            "min_samples": 8,
        },
    )

    assert result.success is True
    assert result.summary.status == "success"
    assert Path(result.summary.experiment_metrics["tip_output_file"]).exists()
    assert np.allclose(result.summary.experiment_metrics["tip_vector_local_mm"], [10.0, 20.0, 100.0], atol=5e-3)


def test_pivot_calibration_offline_from_headered_recorded_file_reports_parse_metrics(tmp_path: Path) -> None:
    csv_path = _write_headered_pivot_csv(tmp_path / "pivot_headered_samples.csv")
    runner = _runner(tmp_path)
    staged_tip = tmp_path / "staged_tip.csv"
    result = runner.run_experiment(
        "pivot_calibration",
        config={
            "tool_id": "0B",
            "input_path": str(csv_path),
            "output_tip_file": str(staged_tip),
            "std_dev_threshold": 3.0,
            "min_samples": 8,
        },
    )

    assert result.success is True
    assert result.summary.experiment_metrics["pivot_input_format"] == "canonical_headered_csv"
    assert result.summary.experiment_metrics["pivot_input_usable_rows"] == 10
    assert result.summary.experiment_metrics["pivot_input_rejected_row_count"] == 0
    assert Path(result.summary.experiment_metrics["tip_output_file"]).exists()


def test_repeatability_dataset_reports_partial_success_without_registration(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    result = runner.run_experiment(
        "repeatability_dataset",
        config={
            "dry_run": True,
            "tool_id": "0A",
            "schedule": {
                "target_points_cm": [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0, 0.0],
                ],
                "revisit_count": 2,
                "randomize_approach_order": False,
                "samples_per_point": 2,
            },
        },
    )

    assert result.success is True
    assert result.summary.status == "partial_success"
    analysis = analyze_repeatability_dataset(result.paths.output_dir)
    assert analysis["experiment_name"] == "repeatability_dataset"
    assert analysis["summary_status"] == "partial_success"
    assert analysis["valid_sample_count"] > 0


def test_repeatability_dataset_run_saves_prior_state_identity_and_target_catalog(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    result = runner.run_experiment(
        "repeatability_dataset",
        config={
            "dry_run": True,
            "tool_id": "0A",
            "schedule": {
                "target_set": "single_segment_ring_17",
                "low_magnitude_cm": 0.10,
                "high_magnitude_cm": 0.20,
                "revisit_count": 2,
                "samples_per_point": 1,
                "randomize_approach_order": False,
            },
        },
    )

    assert result.success is True
    assert result.paths.output_dir.parent == tmp_path / "data" / "experiments"
    assert result.summary.experiment_metrics["target_catalog"]
    assert result.summary.experiment_metrics["planned_visit_count"] == 34
    bundle = runner.load_dataset(result.paths.output_dir)
    measurement_samples = [sample for sample in bundle.samples if sample.phase == "sample"]
    assert measurement_samples
    first = measurement_samples[0]
    assert first.extra["target_label"]
    assert first.extra["approach_label"]
    assert first.extra["target_set"] == "single_segment_ring_17"


def test_aurora_grid_accuracy_missing_tip_calibration_is_classified(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    result = runner.run_experiment(
        "aurora_grid_accuracy",
        config={
            "dry_run": True,
            "dimensions": [2, 2],
            "repetitions_per_point": 1,
            "samples_per_point": 1,
            "tool_id": "0B",
            "truth_frame": "tracker",
            "use_tip_calibration": True,
            "allow_coil_origin_fallback": False,
        },
    )

    assert result.success is False
    assert result.summary.status == "invalid_due_to_missing_tip_cal"


def test_aurora_grid_accuracy_no_longer_requires_registration(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    result = runner.run_experiment(
        "aurora_grid_accuracy",
        config={
            "dry_run": True,
            "dimensions": [2, 2],
            "repetitions_per_point": 1,
            "samples_per_point": 1,
            "tool_id": "0B",
            "use_tip_calibration": True,
            "tip_vector_mm": [0.0, 0.0, 125.0],
            "allow_coil_origin_fallback": False,
        },
    )

    assert result.success is True
    assert result.summary.status == "success"


def test_aurora_grid_accuracy_run_saves_captured_point_dataset(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    result = runner.run_experiment(
        "aurora_grid_accuracy",
        config={
            "rows": 2,
            "cols": 2,
            "spacing_mm": 25.4,
            "samples_per_point": 2,
            "tool_id": "0B",
            "use_tip_calibration": True,
            "tip_vector_mm": [0.0, 0.0, 125.0],
            "outlier_threshold_mm": 1.0,
            "captured_points": [
                {
                    "label": "P01",
                    "target_index": 0,
                    "raw_samples": [
                        {"position_mm": [5.5, 2.0, 1.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0], "tracking_state": "valid", "position_source": "tip"},
                        {"position_mm": [4.5, 2.0, 1.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0], "tracking_state": "valid", "position_source": "tip"},
                    ],
                },
                {
                    "label": "P02",
                    "target_index": 1,
                    "raw_samples": [
                        {"position_mm": [30.9, 2.0, 1.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0], "tracking_state": "valid", "position_source": "tip"},
                        {"position_mm": [29.9, 2.0, 1.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0], "tracking_state": "valid", "position_source": "tip"},
                    ],
                },
                {
                    "label": "P03",
                    "target_index": 2,
                    "raw_samples": [
                        {"position_mm": [5.4, 27.4, 1.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0], "tracking_state": "valid", "position_source": "tip"},
                        {"position_mm": [4.4, 27.4, 1.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0], "tracking_state": "valid", "position_source": "tip"},
                    ],
                },
            ],
        },
    )

    assert result.success is True
    assert result.paths.output_dir.parent == tmp_path / "data" / "experiments"
    assert result.summary.experiment_metrics["point_count_aligned"] == 3
    assert result.summary.experiment_metrics["point_count_captured"] == 3
    assert result.summary.experiment_metrics["point_count_total"] == 4
    assert result.summary.experiment_metrics["alignment_ready"] is True
    assert result.summary.experiment_metrics["position_source_counts"]["tip"] == 6
    assert result.summary.experiment_metrics["alignment_transform_truth_to_measured"] is not None


def test_repeatability_dataset_records_robot_frame_when_registration_exists(tmp_path: Path) -> None:
    registration_path = tmp_path / "latest_registration.json"
    registration_path.write_text(
        json.dumps({"T_robot_aurora": np.eye(4).tolist(), "T_coil_tip": np.eye(4).tolist()}),
        encoding="utf-8",
    )
    runner = _runner(tmp_path, registration_path=registration_path)
    result = runner.run_experiment(
        "repeatability_dataset",
        config={
            "dry_run": True,
            "tool_id": "0A",
            "schedule": {
                "target_points_cm": [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0, 0.0],
                ],
                "revisit_count": 1,
                "samples_per_point": 1,
            },
        },
    )

    assert result.success is True
    assert result.summary.status == "success"
