from __future__ import annotations

import json
from pathlib import Path

from continuum_robot.config.schemas import (
    CalibrationConfig,
    ExperimentConfig,
    RegistrationWorkflowConfig,
    RobotConfig,
    RobotSegmentConfig,
    RuntimeConfig,
    SafetyConfig,
    SerialConfig,
    TwoSegmentTrackingRoleConfig,
)
from continuum_robot.config.config_loader import ConfigLoader
from continuum_robot.config.settings import Settings
from continuum_robot.data.export_run_bundle import export_run_bundle
from continuum_robot.data.validate_run_bundle import validate_run_folder
from continuum_robot.experiments.builtins import register_builtin_experiments
from continuum_robot.experiments.experiment_runner import ExperimentRunner
from continuum_robot.experiments.registry import ExperimentRegistry
from continuum_robot.experiments.two_segment_collect_pose_dataset import (
    BLOCK_MESSAGE,
    EXPERIMENT_NAME,
    TwoSegmentCollectPoseDatasetConfig,
    build_two_segment_command_schedule,
)
from continuum_robot.hardware.mock_dxl_bus import MockDxlBus
from continuum_robot.servos.displacement_mapper import TendonDisplacementMapper
from continuum_robot.servos.neutral_calibration_service import NeutralCalibrationService, ServoCalibrationContext
from continuum_robot.servos.pretension_validation_service import PretensionValidationService
from continuum_robot.servos.safety_guard import SafetyGuard
from continuum_robot.servos.servo_service import ServoService
from continuum_robot.services.models import ServiceHealthSnapshot, ToolTrackingSnapshot, TrackingSnapshot
from continuum_robot.tracking.two_segment_roles import resolve_two_segment_tracking_roles


def _segments() -> dict[str, RobotSegmentConfig]:
    return {
        "segment_a": RobotSegmentConfig(
            key="segment_a",
            label="Spine 1",
            segment_label="Segment A",
            segment_role="proximal",
            segment_order_index=0,
            servo_ids=[1, 2, 3, 4],
            pairs={"axis_a": [1, 3], "axis_b": [2, 4]},
        ),
        "segment_b": RobotSegmentConfig(
            key="segment_b",
            label="Spine 2",
            segment_label="Segment B",
            segment_role="distal",
            segment_order_index=1,
            servo_ids=[5, 6, 7, 8],
            pairs={"axis_a": [5, 7], "axis_b": [6, 8]},
        ),
    }


def _settings(*, mode: str = "dual_segment") -> Settings:
    return Settings(
        runtime=RuntimeConfig(mock_mode=True, poll_rate_hz=20, robot_config="robot_8servo.yaml"),
        robot=RobotConfig(
            mode=mode,
            spool_diameter_cm=1.2,
            ticks_per_revolution=4096,
            servo_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            tendon_to_servo=[1, 2, 3, 4, 5, 6, 7, 8],
            segments=_segments(),
        ),
        serial=SerialConfig(aurora_port="/dev/mock-aurora", openrb_port="/dev/mock-openrb", baudrate=115200),
        safety=SafetyConfig(position_min_offset_ticks=-600, position_max_offset_ticks=600, max_current_ma=850),
        registration=RegistrationWorkflowConfig(
            capture_tool_id="0B",
            coil_tool_id="0A",
            max_fre_mm=None,
            two_segment_tracking_roles=_role_configs(),
        ),
        experiment=ExperimentConfig(default_settle_time_s=0.0, sample_count_per_point=1, output_dir="data/experiments"),
        calibration=CalibrationConfig(
            neutral_setpoints_path="config/neutral_setpoints.json",
            latest_registration_path="data/registrations/latest_registration.json",
        ),
    )


def _role_configs(*, include_intermediate: bool = False) -> dict[str, TwoSegmentTrackingRoleConfig]:
    return {
        "registration_probe": TwoSegmentTrackingRoleConfig(
            role_name="registration_probe",
            tool_id="0B",
            physical_meaning="registration probe",
            pose_kind="calibrated_tip",
        ),
        "distal_tip": TwoSegmentTrackingRoleConfig(
            role_name="distal_tip",
            tool_id="0A",
            physical_meaning="distal robot tip label",
            pose_kind="calibrated_tip",
            required_for_two_segment_model_training=True,
            trust_level="runtime_tip_policy",
            max_age_s=0.15,
        ),
        "intermediate_segment": TwoSegmentTrackingRoleConfig(
            role_name="intermediate_segment",
            tool_id="0C" if include_intermediate else "",
            physical_meaning="intermediate segment label",
            pose_kind="coil_origin",
            enabled=include_intermediate,
            max_age_s=0.15,
        ),
    }


class _FakeTrackingService:
    def __init__(self, snapshot) -> None:
        self._snapshot = snapshot

    def peek_snapshot(self):
        return self._snapshot

    def get_snapshot(self):
        return self._snapshot


def _tracking_snapshot(*, include_distal: bool = True, include_intermediate: bool = False, stale: bool = False) -> TrackingSnapshot:
    tools = {}
    if include_distal:
        tools["0A"] = ToolTrackingSnapshot(
            tool_id="0A",
            present=True,
            valid=True,
            tracking_state="tracked",
            translation_mm=(1.0, 2.0, 3.0),
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            quality=0.9,
        )
    if include_intermediate:
        tools["0C"] = ToolTrackingSnapshot(
            tool_id="0C",
            present=True,
            valid=True,
            tracking_state="tracked",
            translation_mm=(0.5, 1.0, 1.5),
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            quality=0.8,
        )
    return TrackingSnapshot(
        health=ServiceHealthSnapshot(name="tracking", health="healthy", state="mock", status="mock"),
        connection_state="connected",
        canonical_state="streaming_healthy",
        backend_identity="mock",
        selected_backend_name="mock",
        runtime_coil_tool_id="0A",
        registration_tool_id="0B",
        tracker_data_age_s=0.4 if stale else 0.01,
        tracker_data_stale=bool(stale),
        tools=tools,
        registration_state="loaded",
        runtime_tip_mode="latest_accepted",
        runtime_tip_trust_level="trusted",
        tip_pose_status="ok",
        T_robot_tip=[
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )


def _calibration_context(settings: Settings) -> ServoCalibrationContext:
    context = settings.robot.operating_context()
    return ServoCalibrationContext(
        robot_mode=context.operating_mode,
        robot_config_name=settings.runtime.robot_config,
        servo_ids=list(settings.robot.servo_ids),
        tendon_to_servo=list(settings.robot.tendon_to_servo),
        expected_servo_ids=list(context.expected_servo_ids),
        commanded_servo_ids=list(context.commanded_servo_ids),
        segments=settings.robot.segment_metadata(),
        segment_order=settings.robot.segment_order(),
        mode_profile=context.mode_profile,
        mode_capabilities=dict(context.mode_capabilities),
        mode_notes=list(context.mode_notes),
    )


def _servo_service(tmp_path: Path, *, settings: Settings | None = None) -> ServoService:
    settings = settings or _settings()
    service = ServoService(
        dxl_bus=MockDxlBus([1, 2, 3, 4, 5, 6, 7, 8]),
        mapper=TendonDisplacementMapper(spool_diameter_cm=1.2),
        safety_guard=SafetyGuard(min_offset_ticks=-600, max_offset_ticks=600, max_current_ma=850),
        neutral_calibration=NeutralCalibrationService(
            path=tmp_path / "neutral.json",
            context=_calibration_context(settings),
        ),
        pretension_validation=PretensionValidationService(),
    )
    service.connect(settings.serial.openrb_port, settings.serial.baudrate)
    return service


def _runner(tmp_path: Path, *, settings: Settings | None = None, service: ServoService | None = None, tracking_service=None) -> ExperimentRunner:
    settings = settings or _settings()
    return ExperimentRunner(
        project_root=tmp_path,
        settings=settings,
        tracking_service=tracking_service,
        servo_service=service or _servo_service(tmp_path, settings=settings),
        output_dir=tmp_path / "data" / "experiments",
        default_settle_time_s=0.0,
        registration_path=tmp_path / "latest_registration.json",
        sleep_fn=lambda _seconds: None,
    )


def _save_all8_startup(service: ServoService) -> None:
    service.neutral_calibration.save_manual_pretension_state(
        states_by_servo={
            servo_id: {
                "servo_id": servo_id,
                "measured_position_tick": 2048 + servo_id,
                "measured_current_ma": 100 + servo_id,
            }
            for servo_id in range(1, 9)
        },
        note="test all-8 startup",
        accepted=True,
    )
    artifact = service.neutral_calibration.load_calibration_artifact()
    artifact.robot.update(
        {
            "startup_type": "manual_two_segment_startup",
            "segment_order": ["segment_a", "segment_b"],
            "segments": service.neutral_calibration.context.segments,
            "expected_servo_ids": list(range(1, 9)),
            "commanded_servo_ids": list(range(1, 9)),
        }
    )
    service.neutral_calibration.save_calibration_artifact(artifact)


def test_two_segment_collect_pose_dataset_experiment_is_registered() -> None:
    registry = ExperimentRegistry()
    register_builtin_experiments(registry)

    descriptor = registry.get(EXPERIMENT_NAME)

    assert descriptor.title == "Two-Segment Collect-Pose Dataset"
    assert "Two Segment" in descriptor.tags


def test_two_segment_tracking_role_config_loads_from_registration_yaml() -> None:
    settings = ConfigLoader().load_settings()

    roles = settings.registration.two_segment_tracking_roles

    assert roles["registration_probe"].tool_id == "0B"
    assert roles["distal_tip"].tool_id == "0A"
    assert roles["distal_tip"].required_for_two_segment_model_training is True
    assert roles["intermediate_segment"].required_for_two_segment_model_training is False


def test_two_segment_role_resolver_maps_missing_and_distal_only_roles() -> None:
    resolved = resolve_two_segment_tracking_roles(
        snapshot=_tracking_snapshot(include_distal=True, include_intermediate=False),
        role_configs=_role_configs(include_intermediate=True),
        require_model_training_roles=True,
    )

    assert "distal_tip" in resolved["available_roles"]
    assert "intermediate_segment" in resolved["missing_roles"]
    assert resolved["missing_required_roles"] == []
    assert resolved["dataset_validity"]["distal_only"] is True
    assert resolved["dataset_validity"]["includes_intermediate_pose"] is False
    assert resolved["two_segment_pose"]["distal_tip_pose"]


def test_two_segment_role_resolver_reports_missing_required_distal_without_fabricating_pose() -> None:
    resolved = resolve_two_segment_tracking_roles(
        snapshot=_tracking_snapshot(include_distal=False),
        role_configs=_role_configs(),
        require_model_training_roles=True,
    )

    assert "distal_tip" in resolved["missing_required_roles"]
    assert resolved["two_segment_pose"]["distal_tip_pose"] == {}
    assert resolved["dataset_validity"]["required_model_training_roles_present"] is False


def test_two_segment_role_resolver_reports_stale_distal_as_not_training_ready() -> None:
    resolved = resolve_two_segment_tracking_roles(
        snapshot=_tracking_snapshot(include_distal=True, stale=True),
        role_configs=_role_configs(),
        require_model_training_roles=True,
    )

    assert "distal_tip" in resolved["stale_roles"]
    assert "distal_tip" in resolved["missing_required_roles"]
    assert resolved["role_observations"]["distal_tip"]["stale"] is True


def test_two_segment_collect_pose_dataset_blocks_outside_dual_segment(tmp_path: Path) -> None:
    settings = _settings(mode="single_segment")
    runner = _runner(tmp_path, settings=settings)

    result = runner.run_experiment(EXPERIMENT_NAME, config={})

    assert result.success is False
    assert BLOCK_MESSAGE in result.message


def test_two_segment_command_schedule_uses_canonical_command_and_flattening_order() -> None:
    context = _settings().robot.operating_context()
    config = TwoSegmentCollectPoseDatasetConfig.from_dict({"schedule_type": "single_axis_micro"})

    steps = build_two_segment_command_schedule(config, context=context)
    first_motion = steps[1].command

    assert first_motion.to_segment_mapping()["segment_a"] == [0.01, 0.0, -0.01, 0.0]
    assert first_motion.to_segment_mapping()["segment_b"] == [0.0, 0.0, 0.0, 0.0]
    assert first_motion.to_flat(context=context) == [0.01, 0.0, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_two_segment_collect_pose_dataset_allows_explicit_servo_only_mock_run_and_writes_schema(tmp_path: Path) -> None:
    runner = _runner(tmp_path)

    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={
            "dry_run": True,
            "allow_servo_only_test_run": True,
            "run_trust_mode": "servo_only",
            "schedule_type": "segment_isolation",
            "samples_per_pattern": 1,
        },
    )

    assert result.success is True
    metrics = result.summary.experiment_metrics
    assert metrics["valid_for_two_segment_model_training"] is False
    assert metrics["valid_for_model_training"] is False
    assert metrics["valid_for_thesis_repeatability"] is False
    assert "pose_labels_missing" in metrics["data_quality_warnings"]
    assert metrics["pose_label_summary"]["pose_observation_sample_count"] == 0

    sample_payload = json.loads((result.paths.samples_path).read_text(encoding="utf-8").splitlines()[0])
    assert sample_payload["two_segment_command"]["segments"]["segment_a"] == [0.0, 0.0, 0.0, 0.0]
    assert sample_payload["two_segment_command"]["flat_command_cm"] == [0.0] * 8
    assert sample_payload["extra"]["ordered_8_displacements_cm"] == [0.0] * 8
    assert sample_payload["extra"]["commanded_servo_ids"] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert sample_payload["two_segment_pose"]["status"] == "tracker_unavailable"
    assert sample_payload["pose_in_robot_frame"] == {}
    assert "distal_tip" in sample_payload["extra"]["missing_required_pose_roles"]
    assert sample_payload["extra"]["role_observations"]["distal_tip"]["status"] in {"missing_tool", "unconfigured"}

    for filename in [
        "two_segment_dataset_summary.txt",
        "two_segment_tracking_role_provenance.json",
        "metrics.csv",
        "two_segment_command_coverage_report.png",
        "two_segment_servo_position_coverage_report.png",
        "two_segment_pose_coverage_report.png",
        "two_segment_dataset_quality_report.png",
    ]:
        assert (result.paths.output_dir / filename).exists()

    validation = validate_run_folder(result.paths.output_dir)
    assert validation.status == "PASS"

    export = export_run_bundle(
        run_dir=result.paths.output_dir,
        output_root=tmp_path / "exports",
        project_root=tmp_path,
        include_samples=True,
    )
    exported = {entry.bundle_path for entry in export.entries}
    assert "two_segment_dataset_summary.txt" in exported
    assert "two_segment_tracking_role_provenance.json" in exported
    assert "samples.jsonl" in exported
    assert "two_segment_command_coverage_report.png" in exported


def test_two_segment_collect_pose_dataset_sample_records_distal_pose_role_and_validity(tmp_path: Path) -> None:
    service = _servo_service(tmp_path)
    _save_all8_startup(service)
    runner = _runner(
        tmp_path,
        service=service,
        tracking_service=_FakeTrackingService(_tracking_snapshot(include_distal=True, include_intermediate=False)),
    )

    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={
            "dry_run": False,
            "allow_servo_only_test_run": False,
            "run_trust_mode": "thesis_trusted",
            "schedule_type": "zero",
        },
    )

    assert result.success is True
    metrics = result.summary.experiment_metrics
    assert metrics["valid_for_two_segment_model_training"] is True
    assert metrics["distal_only"] is True
    assert metrics["includes_intermediate_pose"] is False
    assert metrics["pose_label_summary"]["available_roles"] == ["distal_tip"]
    sample_payload = json.loads(result.paths.samples_path.read_text(encoding="utf-8").splitlines()[0])
    assert sample_payload["extra"]["available_pose_roles"] == ["distal_tip"]
    assert sample_payload["extra"]["missing_required_pose_roles"] == []
    assert sample_payload["two_segment_pose"]["distal_tip_pose"]
    assert sample_payload["pose_in_robot_frame"]["roles"]["distal_tip"]["T_robot_tip"]


def test_two_segment_collect_pose_dataset_requires_startup_for_trusted_live_run(tmp_path: Path) -> None:
    service = _servo_service(tmp_path)
    runner = _runner(tmp_path, service=service)

    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={
            "dry_run": False,
            "allow_servo_only_test_run": False,
            "run_trust_mode": "thesis_trusted",
        },
    )

    assert result.success is False
    assert "requires an accepted all-8 manual startup artifact" in result.message


def test_two_segment_collect_pose_dataset_uses_startup_artifact_for_live_servo_only_run(tmp_path: Path) -> None:
    service = _servo_service(tmp_path)
    _save_all8_startup(service)
    runner = _runner(tmp_path, service=service)

    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={
            "dry_run": False,
            "allow_servo_only_test_run": True,
            "run_trust_mode": "servo_only",
            "schedule_type": "zero",
        },
    )

    assert result.success is True
    metrics = result.summary.experiment_metrics
    assert metrics["startup_artifact_provenance"]["accepted_all_8_startup"] is True
    assert metrics["valid_for_two_segment_model_training"] is False
    sample_payload = json.loads(result.paths.samples_path.read_text(encoding="utf-8").splitlines()[0])
    assert sample_payload["extra"]["startup_artifact_provenance"]["source"] == "manual"
    assert sample_payload["commanded_motor_values"] == {str(servo_id): 2048 + servo_id for servo_id in range(1, 9)}
