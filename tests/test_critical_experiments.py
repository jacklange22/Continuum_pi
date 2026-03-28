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
    RepeatabilityScheduleConfig,
    analyze_repeatability_dataset,
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
        output_dir=tmp_path / "runs",
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


def test_aurora_grid_metrics_compute_rms_bias_and_spread() -> None:
    samples = [
        _sample(tool_id="0B", tracker_position=[1.0, 0.0, 0.0], robot_position=None, target_index=0, revisit_index=0, sample_index=0),
        _sample(tool_id="0B", tracker_position=[0.0, 1.0, 0.0], robot_position=None, target_index=0, revisit_index=1, sample_index=1),
        _sample(tool_id="0B", tracker_position=[11.0, 0.0, 0.0], robot_position=None, target_index=1, revisit_index=0, sample_index=2),
        _sample(tool_id="0B", tracker_position=[10.0, 1.0, 0.0], robot_position=None, target_index=1, revisit_index=1, sample_index=3),
    ]

    metrics = compute_grid_accuracy_metrics(
        samples,
        truth_points_mm=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        tool_id="0B",
        truth_frame="tracker",
        outlier_threshold_mm=2.0,
        registration_available=False,
        tip_calibration_available=True,
        require_tip_calibration=False,
        allow_coil_origin_fallback=True,
    )

    assert metrics["status"] == "success"
    assert np.isclose(metrics["overall_rms_error_mm"], 1.0)
    assert np.allclose(metrics["per_axis_bias_mm"], [0.5, 0.5, 0.0])
    assert np.isclose(metrics["per_point_metrics"]["0"]["sample_spread_rms_mm"], np.sqrt(0.5))
    assert np.isclose(metrics["pointwise_rms_error_mm"]["1"], 1.0)


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


def test_aurora_grid_accuracy_missing_registration_is_classified(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    result = runner.run_experiment(
        "aurora_grid_accuracy",
        config={
            "dry_run": True,
            "dimensions": [2, 2],
            "repetitions_per_point": 1,
            "samples_per_point": 1,
            "tool_id": "0B",
            "truth_frame": "robot",
            "use_tip_calibration": True,
            "tip_vector_mm": [0.0, 0.0, 125.0],
            "allow_coil_origin_fallback": False,
        },
    )

    assert result.success is False
    assert result.summary.status == "invalid_due_to_missing_registration"


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
