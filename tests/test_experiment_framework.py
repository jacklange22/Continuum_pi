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
from continuum_robot.experiments.dataset_io import ExperimentDatasetLoader, ExperimentDatasetWriter
from continuum_robot.experiments.experiment_runner import ExperimentRunner
from continuum_robot.experiments.framework import BaseExperiment, ExperimentHardwareRequirements, ExperimentSession
from continuum_robot.experiments.registry import ExperimentRegistry
from continuum_robot.experiments.schedules import CommandScheduleConfig, command_schedule_checksum, generate_command_schedule
from continuum_robot.experiments.schemas import ExperimentMetadata, ExperimentSummary, ExperimentTimeseriesSample
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
    settings: Settings | None = None,
    servo_service: ServoService | None = None,
    tracking_service: TrackingService | None = None,
    registration_path: Path | None = None,
    registry: ExperimentRegistry | None = None,
) -> ExperimentRunner:
    settings = settings or _settings()
    registration_path = registration_path or (tmp_path / "latest_registration.json")
    return ExperimentRunner(
        project_root=Path(__file__).resolve().parents[1],
        settings=settings,
        tracking_service=tracking_service or _tracking_service(settings, tmp_path, registration_path),
        servo_service=servo_service or _servo_service(tmp_path),
        output_dir=tmp_path / "runs",
        default_settle_time_s=0.0,
        registration_path=registration_path,
        sleep_fn=lambda _seconds: None,
        experiment_registry=registry,
    )


class LifecycleProbeExperiment(BaseExperiment):
    name = "lifecycle_probe"
    description = "Exercise lifecycle hooks for tests."
    hardware_requirements = ExperimentHardwareRequirements(mock_compatible=True)

    @classmethod
    def from_dict(cls, payload=None):
        return cls(config=dict(payload or {}))

    def setup(self, session: ExperimentSession) -> None:
        session.set_metric("setup_called", True)

    def precheck(self, session: ExperimentSession) -> None:
        session.set_metric("precheck_called", True)

    def execute(self, session: ExperimentSession) -> None:
        session.add_sample(
            ExperimentTimeseriesSample(
                monotonic_time_s=0.0,
                wall_time_utc="2026-01-01T00:00:00+00:00",
                phase="execute",
                step_index=0,
                sample_index=0,
                status_flags=["ok"],
            )
        )

    def finalize(self, session: ExperimentSession) -> None:
        session.set_metric("finalize_called", True)

    def summarize(self, session: ExperimentSession) -> dict:
        return {"summary_called": True}


def test_experiment_lifecycle_records_stage_results(tmp_path: Path) -> None:
    registry = ExperimentRegistry()
    registry.register(
        name=LifecycleProbeExperiment.name,
        description=LifecycleProbeExperiment.description,
        factory=LifecycleProbeExperiment.from_dict,
    )
    runner = _runner(tmp_path, registry=registry)
    result = runner.run_experiment("lifecycle_probe")

    assert result.success is True
    assert result.summary.stage_pass_fail == {
        "setup": "passed",
        "precheck": "passed",
        "execute": "passed",
        "finalize": "passed",
    }
    assert result.summary.experiment_metrics["setup_called"] is True
    assert result.summary.experiment_metrics["precheck_called"] is True
    assert result.summary.experiment_metrics["finalize_called"] is True
    assert result.summary.experiment_metrics["summary_called"] is True


def test_dataset_writer_roundtrip_loads_canonical_bundle(tmp_path: Path) -> None:
    writer = ExperimentDatasetWriter(tmp_path / "datasets")
    metadata = ExperimentMetadata(
        schema_version="1.0",
        experiment_name="roundtrip_test",
        run_id="abc123",
        timestamp_utc="2026-01-01T00:00:00+00:00",
        git_commit=None,
        backend_info={"mock_mode": True},
        registration_info={"exists": False},
        config_used={"alpha": 1},
        operator_notes="note",
    )
    samples = [
        ExperimentTimeseriesSample(
            monotonic_time_s=0.0,
            wall_time_utc="2026-01-01T00:00:00+00:00",
            phase="sample",
            step_index=0,
            sample_index=0,
            status_flags=["ok"],
        )
    ]
    summary = ExperimentSummary(
        schema_version="1.0",
        experiment_name="roundtrip_test",
        run_id="abc123",
        success=True,
        sample_counts={"total": 1},
        dropped_frames=0,
        invalid_transforms=0,
        stage_pass_fail={"execute": "passed"},
    )

    paths = writer.write_dataset(metadata, samples, summary)
    bundle = ExperimentDatasetLoader().load_dataset(paths.output_dir)

    assert bundle.metadata.experiment_name == "roundtrip_test"
    assert len(bundle.samples) == 1
    assert bundle.summary.success is True


def test_tracker_pipeline_mock_runs_and_logs_samples(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    result = runner.run_experiment(
        "tracker_pipeline_mock",
        config={"sample_count": 4, "sample_period_s": 0.0},
    )

    assert result.success is True
    assert result.sample_count == 4
    bundle = runner.load_dataset(result.paths.output_dir)
    assert all(sample.phase == "sample" for sample in bundle.samples)
    assert bundle.summary.sample_counts["total"] == 4


def test_transform_chain_validation_reports_zero_error(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    result = runner.run_experiment("transform_chain_validation")

    assert result.success is True
    assert result.summary.experiment_metrics["translation_error_mm"] < 1e-9


def test_command_schedule_generation_is_deterministic_and_bounded() -> None:
    config = CommandScheduleConfig(kind="babble", dimensions=4, amplitude_cm=0.2, babble_count=8, seed=7)
    first = generate_command_schedule(config)
    second = generate_command_schedule(config)

    assert command_schedule_checksum(first) == command_schedule_checksum(second)
    for point in first:
        assert all(-0.2 <= value <= 0.2 for value in point.tendon_displacement_cm)


def test_replay_runner_loads_existing_dataset(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    source = runner.run_experiment("dataset_schema_roundtrip", config={"sample_count": 2})
    replay = runner.run_experiment("replay_runner", config={"dataset_path": str(source.paths.output_dir)})

    assert replay.success is True
    assert replay.summary.experiment_metrics["source_sample_count"] == 2
    assert replay.sample_count == 2


def test_collect_pose_command_dataset_marks_registration_missing(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    result = runner.run_experiment(
        "collect_pose_command_dataset",
        config={
            "dry_run": True,
            "sample_count_per_point": 2,
            "command_schedule": {
                "kind": "trajectory",
                "dimensions": 4,
                "trajectory_points_cm": [[0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]],
            },
        },
    )

    assert result.success is True
    bundle = runner.load_dataset(result.paths.output_dir)
    assert any("registration_missing" in sample.status_flags for sample in bundle.samples)
    assert not any("full_pose_available" in sample.status_flags for sample in bundle.samples)


def test_collect_pose_command_dataset_records_full_pose_when_registration_exists(tmp_path: Path) -> None:
    registration_path = tmp_path / "latest_registration.json"
    registration_path.write_text(
        json.dumps({"T_robot_aurora": np.eye(4).tolist(), "T_coil_tip": np.eye(4).tolist()}),
        encoding="utf-8",
    )
    settings = _settings()
    tracking_service = _tracking_service(settings, tmp_path, registration_path=registration_path)
    runner = _runner(tmp_path, settings=settings, tracking_service=tracking_service, registration_path=registration_path)

    result = runner.run_experiment(
        "collect_pose_command_dataset",
        config={
            "dry_run": True,
            "sample_count_per_point": 1,
            "command_schedule": {
                "kind": "trajectory",
                "dimensions": 4,
                "trajectory_points_cm": [[0.0, 0.0, 0.0, 0.0]],
            },
        },
    )

    assert result.success is True
    bundle = runner.load_dataset(result.paths.output_dir)
    assert any("full_pose_available" in sample.status_flags for sample in bundle.samples)
