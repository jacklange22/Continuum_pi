from pathlib import Path

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
from continuum_robot.experiments.experiment_models import ExperimentPoint
from continuum_robot.experiments.experiment_runner import ExperimentRunner
from continuum_robot.hardware.mock_dxl_bus import MockDxlBus
from continuum_robot.services.tracking_service import TrackingService
from continuum_robot.servos.displacement_mapper import TendonDisplacementMapper
from continuum_robot.servos.neutral_calibration_service import NeutralCalibrationService
from continuum_robot.servos.pretension_validation_service import PretensionValidationService
from continuum_robot.servos.safety_guard import SafetyGuard
from continuum_robot.servos.servo_service import ServoService
from continuum_robot.tracking.mock_tracker_manager import MockTrackerManager


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


def test_experiment_runner_routes_csv_points_through_canonical_dataset(tmp_path: Path) -> None:
    settings = _settings()
    servo_service = ServoService(
        dxl_bus=MockDxlBus([1, 2, 3, 4]),
        mapper=TendonDisplacementMapper(spool_diameter_cm=1.2),
        safety_guard=SafetyGuard(min_offset_ticks=-600, max_offset_ticks=600, max_current_ma=850),
        neutral_calibration=NeutralCalibrationService(path=tmp_path / "neutral.json"),
        pretension_validation=PretensionValidationService(),
    )
    tracking_service = TrackingService(
        live_backend=MockTrackerManager(poll_hz=30),
        port=settings.serial.aurora_port,
        registration_path=tmp_path / "latest_registration.json",
        config_source="test",
        runtime_coil_tool_id=settings.registration.coil_tool_id,
        registration_tool_id=settings.registration.capture_tool_id,
    )
    runner = ExperimentRunner(
        project_root=Path(__file__).resolve().parents[1],
        settings=settings,
        tracking_service=tracking_service,
        servo_service=servo_service,
        output_dir=tmp_path / "data" / "experiments",
        default_settle_time_s=0.0,
        registration_path=tmp_path / "latest_registration.json",
        sleep_fn=lambda _seconds: None,
    )

    summary = runner.run(
        [
            ExperimentPoint(index=0, tendon_displacement_cm=[0.0, 0.1, -0.1, 0.0]),
            ExperimentPoint(index=1, tendon_displacement_cm=[0.1, 0.0, 0.0, 0.0], repeat=2),
        ]
    )

    assert summary.rows_written == 12
    assert summary.output_path.exists()
    assert (summary.output_path / "metadata.json").exists()
    assert (summary.output_path / "samples.jsonl").exists()
    assert (summary.output_path / "summary.json").exists()


def test_experiment_runner_load_dataset_resolves_legacy_runs_path(tmp_path: Path) -> None:
    settings = _settings()
    servo_service = ServoService(
        dxl_bus=MockDxlBus([1, 2, 3, 4]),
        mapper=TendonDisplacementMapper(spool_diameter_cm=1.2),
        safety_guard=SafetyGuard(min_offset_ticks=-600, max_offset_ticks=600, max_current_ma=850),
        neutral_calibration=NeutralCalibrationService(path=tmp_path / "neutral.json"),
        pretension_validation=PretensionValidationService(),
    )
    tracking_service = TrackingService(
        live_backend=MockTrackerManager(poll_hz=30),
        port=settings.serial.aurora_port,
        registration_path=tmp_path / "latest_registration.json",
        config_source="test",
        runtime_coil_tool_id=settings.registration.coil_tool_id,
        registration_tool_id=settings.registration.capture_tool_id,
    )
    runner = ExperimentRunner(
        project_root=tmp_path,
        settings=settings,
        tracking_service=tracking_service,
        servo_service=servo_service,
        output_dir=tmp_path / "data" / "experiments",
        default_settle_time_s=0.0,
        registration_path=tmp_path / "latest_registration.json",
        sleep_fn=lambda _seconds: None,
    )
    result = runner.run_experiment("dataset_schema_roundtrip", config={"sample_count": 1})
    migrated_root = tmp_path / "data" / "experiments" / "pivot" / "runs"
    migrated_root.mkdir(parents=True, exist_ok=True)
    migrated_path = migrated_root / "legacy_bundle"
    result.paths.output_dir.rename(migrated_path)

    bundle = runner.load_dataset(tmp_path / "runs" / "legacy_bundle")

    assert bundle.metadata.experiment_name == "dataset_schema_roundtrip"
    assert bundle.paths.output_dir == migrated_path
