from __future__ import annotations

import json
from pathlib import Path

import pytest

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
from continuum_robot.gui.experiment_preflight import RUN_BLOCKED, evaluate_preflight
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


def _servo_service(tmp_path: Path, *, settings: Settings | None = None, dxl_bus: MockDxlBus | None = None) -> ServoService:
    settings = settings or _settings()
    service = ServoService(
        dxl_bus=dxl_bus or MockDxlBus([1, 2, 3, 4, 5, 6, 7, 8]),
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


def test_two_segment_collect_pose_dataset_blocks_invalid_physical_assembly(tmp_path: Path) -> None:
    bad = _settings()
    bad.robot.bottom_segment_key = "segment_a"
    bad.robot.top_segment_key = "segment_a"  # duplicate role assignment
    runner = _runner(tmp_path, settings=bad)

    result = runner.run_experiment(EXPERIMENT_NAME, config={})

    assert result.success is False
    assert "valid bottom/top physical assembly" in result.message


def test_two_segment_collect_pose_dataset_blocks_outside_dual_segment(tmp_path: Path) -> None:
    settings = _settings(mode="single_segment")
    runner = _runner(tmp_path, settings=settings)

    result = runner.run_experiment(EXPERIMENT_NAME, config={})

    assert result.success is False
    assert BLOCK_MESSAGE in result.message


def test_two_segment_command_schedule_uses_canonical_command_and_flattening_order() -> None:
    context = _settings().robot.operating_context()
    # Legacy MVP-style amplitude (0.1 mm = 0.01 cm) accepted via legacy field.
    config = TwoSegmentCollectPoseDatasetConfig.from_dict(
        {"schedule_type": "single_axis_micro", "max_segment_displacement_mm": 0.1}
    )

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
    settings = _settings()
    settings.runtime.mock_mode = False
    service = _servo_service(tmp_path, settings=settings)
    _save_all8_startup(service)
    runner = _runner(
        tmp_path,
        settings=settings,
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
            "physical_assembly_confirmed_by_operator": True,
        },
    )

    assert result.success is True
    metrics = result.summary.experiment_metrics
    assert metrics["valid_for_two_segment_model_training"] is True
    assert metrics["valid_for_two_segment_ann_training"] is True
    assert metrics["label_mode_used"] == "distal_xyz"
    assert metrics["distal_only"] is True
    assert metrics["includes_intermediate_pose"] is False
    assert metrics["pose_label_summary"]["available_roles"] == ["distal_tip"]
    sample_payload = json.loads(result.paths.samples_path.read_text(encoding="utf-8").splitlines()[0])
    assert sample_payload["extra"]["available_pose_roles"] == ["distal_tip"]
    assert sample_payload["extra"]["missing_required_pose_roles"] == []
    assert sample_payload["extra"]["valid_for_two_segment_ann_training"] is True
    assert sample_payload["extra"]["label_mode_used"] == "distal_xyz"
    assert sample_payload["two_segment_pose"]["distal_tip_pose"]
    assert sample_payload["pose_in_robot_frame"]["roles"]["distal_tip"]["T_robot_tip"]


def test_two_segment_collect_pose_dataset_blocks_trusted_run_without_robot_frame_distal_pose(tmp_path: Path) -> None:
    settings = _settings()
    settings.runtime.mock_mode = False
    service = _servo_service(tmp_path, settings=settings)
    _save_all8_startup(service)
    snapshot = _tracking_snapshot(include_distal=True, include_intermediate=False)
    snapshot.T_robot_tip = None
    snapshot.registration_state = "missing_registration"
    runner = _runner(
        tmp_path,
        settings=settings,
        service=service,
        tracking_service=_FakeTrackingService(snapshot),
    )

    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={
            "dry_run": False,
            "allow_servo_only_test_run": False,
            "run_trust_mode": "thesis_trusted",
            "schedule_type": "zero",
            "physical_assembly_confirmed_by_operator": True,
        },
    )

    assert result.success is False
    assert "requires a live robot-frame distal_tip pose" in result.message


def test_two_segment_collect_pose_preflight_blocks_trusted_run_without_robot_frame_distal_pose(tmp_path: Path) -> None:
    settings = _settings()
    settings.runtime.mock_mode = False
    settings.robot.physical_assembly_confirmed_by_operator = True
    service = _servo_service(tmp_path, settings=settings)
    _save_all8_startup(service)
    snapshot = _tracking_snapshot(include_distal=True, include_intermediate=False)
    snapshot.T_robot_tip = None
    snapshot.registration_state = "missing_registration"

    report = evaluate_preflight(
        experiment_name=EXPERIMENT_NAME,
        config_payload={
            "dry_run": False,
            "allow_servo_only_test_run": False,
            "run_trust_mode": "thesis_trusted",
            "schedule_type": "zero",
        },
        config_error=None,
        settings=settings,
        tracking_snapshot=snapshot,
        servo_connected=True,
        neutral_setpoints={},
        registration_path=tmp_path / "latest_registration.json",
        output_root=tmp_path / "data" / "experiments",
        planned_output_dir=tmp_path / "data" / "experiments" / EXPERIMENT_NAME / "run",
        project_root=tmp_path,
        servo_calibration_summary=service.get_calibration_summary(),
    )

    assert report.overall_status == RUN_BLOCKED
    assert any("no live robot-frame distal_tip pose" in check.message for check in report.checks)


def test_two_segment_collect_pose_dataset_requires_startup_for_trusted_live_run(tmp_path: Path) -> None:
    service = _servo_service(tmp_path)
    runner = _runner(tmp_path, service=service)

    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={
            "dry_run": False,
            "allow_servo_only_test_run": False,
            "run_trust_mode": "thesis_trusted",
            "physical_assembly_confirmed_by_operator": True,
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


def test_two_segment_schedule_honors_configured_amplitude_no_silent_cap() -> None:
    context = _settings().robot.operating_context()
    config = TwoSegmentCollectPoseDatasetConfig.from_dict(
        {"schedule_type": "single_axis_micro", "max_segment_displacement_cm": 0.5}
    )

    steps = build_two_segment_command_schedule(config, context=context)
    first_motion = steps[1].command

    # The legacy MVP capped amp_cm at 0.01. The new path honors the configured
    # 0.5 cm value (still safety-gated downstream by max_tick_delta_from_startup).
    assert first_motion.to_segment_mapping()["segment_a"] == [0.5, 0.0, -0.5, 0.0]


def test_two_segment_dataset_config_has_range_ramp_and_current_defaults() -> None:
    config = TwoSegmentCollectPoseDatasetConfig.from_dict(
        {
            "schedule_type": "workspace_coverage",
            "max_segment_displacement_cm": 1.0,
            "continue_until_valid_samples": True,
            "target_valid_sample_count": 10000,
        }
    )

    assert config.max_segment_displacement_cm == 1.0
    assert config.max_segment_displacement_mm == 10.0
    assert config.continue_until_valid_samples is True
    assert config.target_valid_sample_count == 10000
    assert config.current_warning_ma == 800
    assert config.sustained_overcurrent_ma == 1200


def test_two_segment_dataset_config_default_settle_time_is_two_seconds() -> None:
    """Policy 2026-05-19: settle_time_s default is 2.0 s (safer bench-up default)."""
    config = TwoSegmentCollectPoseDatasetConfig.from_dict({})
    assert config.settle_time_s == 2.0
    explicit = TwoSegmentCollectPoseDatasetConfig.from_dict({"settle_time_s": 0.5})
    assert explicit.settle_time_s == 0.5  # explicit override still wins


def test_two_segment_dataset_config_top_routing_compensation_defaults_on() -> None:
    """Policy 2026-05-19: top tendons route through bottom -> compensation default ON."""
    config = TwoSegmentCollectPoseDatasetConfig.from_dict({})
    assert config.top_segment_tendon_routing_compensation is True
    disabled = TwoSegmentCollectPoseDatasetConfig.from_dict(
        {"top_segment_tendon_routing_compensation": False}
    )
    assert disabled.top_segment_tendon_routing_compensation is False


def test_top_routing_compensation_adds_bottom_to_top_per_tendon_index() -> None:
    """When B=bottom, the top servos (1..4) receive top_intended + bottom_intended.

    Convention is per-tendon-index across bottom/top. Tendon order within each
    segment is [+X, +Y, -X, -Y]; index 0 of bottom maps to index 0 of top.
    """
    from continuum_robot.experiments.two_segment_collect_pose_dataset import (
        _apply_top_routing_compensation,
    )

    settings = _settings()
    # Match the bench default: B=bottom (servos 5..8), A=top (servos 1..4).
    settings.robot.bottom_segment_key = "segment_b"
    settings.robot.top_segment_key = "segment_a"
    context = settings.robot.operating_context()

    # commanded_servo_ids = [1..8]; index 0..3 = segment_a values, 4..7 = segment_b.
    # With B=bottom: segment_b is bottom and gets pulled +0.5 on tendon[0].
    # segment_a is top and intends +0.3 on tendon[0].
    # Expected compensated top tendon[0] (servo 1) = 0.3 + 0.5 = 0.8.
    intended_flat = [0.3, 0.0, 0.0, 0.0,  # segment_a (top) -- user intent
                     0.5, 0.0, 0.0, 0.0]  # segment_b (bottom) -- user intent
    compensated, info = _apply_top_routing_compensation(intended_flat, context=context)

    assert info["applied"] is True
    # Bottom servos (5..8) unchanged.
    assert compensated[4:] == [0.5, 0.0, 0.0, 0.0]
    # Top servos (1..4) get bottom contribution added per tendon index.
    assert compensated[:4] == [0.8, 0.0, 0.0, 0.0]
    # Provenance records what the bottom contributed to each top servo.
    contrib = info["bottom_contribution_added_to_top_servo_cm"]
    assert contrib == {"1": 0.5, "2": 0.0, "3": 0.0, "4": 0.0}
    assert info["bottom_servo_ids"] == [5, 6, 7, 8]
    assert info["top_servo_ids"] == [1, 2, 3, 4]


def test_top_routing_compensation_respects_a_equals_bottom_assembly_swap() -> None:
    """When A=bottom (legacy), top servos (5..8) get bottom (1..4) added per index."""
    from continuum_robot.experiments.two_segment_collect_pose_dataset import (
        _apply_top_routing_compensation,
    )

    settings = _settings()
    settings.robot.bottom_segment_key = "segment_a"
    settings.robot.top_segment_key = "segment_b"
    context = settings.robot.operating_context()

    # Bottom (segment_a, servos 1..4) intends +0.4 on tendon[1] (+Y).
    # Top (segment_b, servos 5..8) intends +0.2 on tendon[1] (+Y).
    # Expected compensated top servo 6 = 0.2 + 0.4 = 0.6.
    intended_flat = [0.0, 0.4, 0.0, 0.0,  # segment_a (bottom)
                     0.0, 0.2, 0.0, 0.0]  # segment_b (top)
    compensated, info = _apply_top_routing_compensation(intended_flat, context=context)

    assert info["applied"] is True
    assert compensated[:4] == pytest.approx([0.0, 0.4, 0.0, 0.0])  # bottom unchanged
    assert compensated[4:] == pytest.approx([0.0, 0.6, 0.0, 0.0])  # top got bottom +0.4 added at tendon[1]


def test_top_routing_compensation_records_bus_command_in_sample_extra(tmp_path: Path) -> None:
    """Live run records both intended and compensated commands so the audit trail is complete."""
    settings = _settings()
    settings.robot.bottom_segment_key = "segment_b"
    settings.robot.top_segment_key = "segment_a"
    settings.robot.physical_assembly_confirmed_by_operator = True
    service = _servo_service(tmp_path, settings=settings)
    _save_all8_startup(service)
    runner = _runner(
        tmp_path,
        settings=settings,
        service=service,
        tracking_service=_FakeTrackingService(_tracking_snapshot(include_distal=True)),
    )

    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={
            "dry_run": False,
            "allow_servo_only_test_run": True,
            "run_trust_mode": "servo_only",
            "schedule_type": "bottom_only_sweep",
            "max_segment_displacement_cm": 0.05,
            "samples_per_pattern": 1,
            "capture_repeats": 1,
            "settle_time_s": 0.0,  # keep the test fast
            "top_segment_tendon_routing_compensation": True,
        },
    )

    assert result.success is True
    # Find a sample where bottom is non-zero (so compensation should fire).
    samples = [
        json.loads(line)
        for line in result.paths.samples_path.read_text(encoding="utf-8").splitlines()
    ]
    nonzero = [
        s for s in samples
        if any(abs(v) > 1e-9 for v in s["extra"]["ordered_8_displacements_cm"][4:])
    ]
    assert nonzero, "bottom_only_sweep with B=bottom must produce non-zero segment_b values"
    sample = nonzero[0]
    intended = sample["extra"]["ordered_8_displacements_cm"]
    compensated = sample["extra"]["servo_flat_command_cm"]
    info = sample["extra"]["top_routing_compensation"]
    # User-intended top (servos 1..4) is all zeros for bottom_only_sweep.
    assert intended[:4] == [0.0, 0.0, 0.0, 0.0]
    # But the bus saw compensated values equal to the bottom values per index.
    assert compensated[:4] == intended[4:]
    # Bottom (servos 5..8) is unchanged.
    assert compensated[4:] == intended[4:]
    assert info["applied"] is True
    assert info["bottom_servo_ids"] == [5, 6, 7, 8]
    assert info["top_servo_ids"] == [1, 2, 3, 4]


def test_top_routing_compensation_disabled_by_config_passes_through(tmp_path: Path) -> None:
    """Operator can disable compensation for independently-routed-tendon benches."""
    from continuum_robot.experiments.two_segment_collect_pose_dataset import (
        _apply_top_routing_compensation,
    )

    settings = _settings()
    context = settings.robot.operating_context()
    intended_flat = [0.3, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
    # When called directly with normal assembly, helper always applies; the
    # disable lives at the experiment config layer.
    compensated, info = _apply_top_routing_compensation(intended_flat, context=context)
    assert info["applied"] is True
    # Now simulate the disable path by setting the config flag false and
    # checking the sample's compensated == intended (no change).
    settings.robot.physical_assembly_confirmed_by_operator = True
    service = _servo_service(tmp_path, settings=settings)
    _save_all8_startup(service)
    runner = _runner(
        tmp_path,
        settings=settings,
        service=service,
        tracking_service=_FakeTrackingService(_tracking_snapshot(include_distal=True)),
    )
    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={
            "dry_run": False,
            "allow_servo_only_test_run": True,
            "run_trust_mode": "servo_only",
            "schedule_type": "bottom_only_sweep",
            "max_segment_displacement_cm": 0.05,
            "samples_per_pattern": 1,
            "capture_repeats": 1,
            "settle_time_s": 0.0,
            "top_segment_tendon_routing_compensation": False,  # OPT OUT
        },
    )
    assert result.success is True
    samples = [
        json.loads(line)
        for line in result.paths.samples_path.read_text(encoding="utf-8").splitlines()
    ]
    # Without compensation, servo_flat_command_cm matches ordered_8_displacements_cm.
    for sample in samples:
        intended = sample["extra"]["ordered_8_displacements_cm"]
        compensated = sample["extra"]["servo_flat_command_cm"]
        assert compensated == intended
        info = sample["extra"]["top_routing_compensation"]
        assert info["applied"] is False
        assert info["reason"] == "disabled_by_config"


def test_two_segment_schedule_supports_bottom_only_sweep_with_role_swap() -> None:
    settings = _settings()
    settings.robot.bottom_segment_key = "segment_b"
    settings.robot.top_segment_key = "segment_a"
    context = settings.robot.operating_context()
    config = TwoSegmentCollectPoseDatasetConfig.from_dict(
        {"schedule_type": "bottom_only_sweep", "max_segment_displacement_cm": 0.25}
    )

    steps = build_two_segment_command_schedule(config, context=context)
    motion_steps = [step for step in steps if step.phase != "zero"]

    assert motion_steps, "bottom_only_sweep should generate motion steps"
    for step in motion_steps:
        mapping = step.command.to_segment_mapping()
        # Bottom is physically segment_b, so command lives in segment_b values;
        # segment_a (top) stays zero.
        assert mapping["segment_a"] == [0.0, 0.0, 0.0, 0.0]
        assert any(abs(v) > 0.0 for v in mapping["segment_b"])


def test_two_segment_schedule_top_only_sweep_routes_to_top_segment() -> None:
    context = _settings().robot.operating_context()
    config = TwoSegmentCollectPoseDatasetConfig.from_dict(
        {"schedule_type": "top_only_sweep", "max_segment_displacement_cm": 0.2}
    )

    steps = build_two_segment_command_schedule(config, context=context)
    motion = [step for step in steps if step.phase != "zero"]

    for step in motion:
        mapping = step.command.to_segment_mapping()
        # Default assembly: top is segment_b
        assert mapping["segment_a"] == [0.0, 0.0, 0.0, 0.0]
        assert any(abs(v) > 0.0 for v in mapping["segment_b"])


def test_two_segment_schedule_random_babble_is_reproducible() -> None:
    context = _settings().robot.operating_context()
    config_a = TwoSegmentCollectPoseDatasetConfig.from_dict(
        {"schedule_type": "random_babble", "max_segment_displacement_cm": 0.3,
         "pattern_count_budget": 8, "random_seed": 42}
    )
    config_b = TwoSegmentCollectPoseDatasetConfig.from_dict(
        {"schedule_type": "random_babble", "max_segment_displacement_cm": 0.3,
         "pattern_count_budget": 8, "random_seed": 42}
    )

    steps_a = build_two_segment_command_schedule(config_a, context=context)
    steps_b = build_two_segment_command_schedule(config_b, context=context)

    assert [step.command.to_segment_mapping() for step in steps_a] == [
        step.command.to_segment_mapping() for step in steps_b
    ]
    # Different seed produces different samples.
    config_c = TwoSegmentCollectPoseDatasetConfig.from_dict(
        {"schedule_type": "random_babble", "max_segment_displacement_cm": 0.3,
         "pattern_count_budget": 8, "random_seed": 7}
    )
    steps_c = build_two_segment_command_schedule(config_c, context=context)
    assert [step.command.to_segment_mapping() for step in steps_c] != [
        step.command.to_segment_mapping() for step in steps_a
    ]


def test_two_segment_schedule_supports_workspace_coverage() -> None:
    context = _settings().robot.operating_context()
    config = TwoSegmentCollectPoseDatasetConfig.from_dict(
        {"schedule_type": "workspace_coverage", "max_segment_displacement_cm": 0.2}
    )

    steps = build_two_segment_command_schedule(config, context=context)
    phases = {step.phase for step in steps[1:]}

    assert "workspace_coverage" in phases
    assert len(steps) > 10


def test_two_segment_collect_pose_dataset_writes_long_run_health_and_failure_files(tmp_path: Path) -> None:
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
            "allow_servo_only_test_run": True,
            "run_trust_mode": "servo_only",
            "schedule_type": "zero",
            "long_run_recovery_enabled": True,
        },
    )

    assert result.success is True
    metrics = result.summary.experiment_metrics
    long_run_health = metrics.get("long_run_health") or {}
    assert long_run_health.get("schema_version") == "two_segment_long_run_health_v1"
    assert long_run_health.get("stop_reason") in {"scheduled_repeat_count_reached", "target_valid_sample_count_reached"}
    assert "transport_recovery_report" in metrics
    assert (result.paths.output_dir / "long_run_health.json").exists()
    assert (result.paths.output_dir / "transport_recovery_report.json").exists()
    assert (result.paths.output_dir / "sample_failure_events.jsonl").exists()
    # The long-run files must also be exportable as part of advisor handoff
    # bundles. Confirm they show up in the bundle entries.
    export = export_run_bundle(
        run_dir=result.paths.output_dir,
        output_root=tmp_path / "exports",
        project_root=tmp_path,
        include_samples=True,
    )
    exported = {entry.bundle_path for entry in export.entries}
    assert "long_run_health.json" in exported
    assert "transport_recovery_report.json" in exported
    assert "sample_failure_events.jsonl" in exported


def test_two_segment_collect_pose_dataset_records_bottom_top_assembly_metadata(tmp_path: Path) -> None:
    swapped = _settings()
    swapped.robot.bottom_segment_key = "segment_b"
    swapped.robot.top_segment_key = "segment_a"
    service = _servo_service(tmp_path, settings=swapped)
    _save_all8_startup(service)
    runner = _runner(
        tmp_path,
        settings=swapped,
        service=service,
        tracking_service=_FakeTrackingService(_tracking_snapshot(include_distal=True, include_intermediate=False)),
    )

    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={"dry_run": False, "allow_servo_only_test_run": True, "run_trust_mode": "servo_only", "schedule_type": "zero"},
    )

    assert result.success is True
    metrics = result.summary.experiment_metrics
    assert metrics["bottom_segment_key"] == "segment_b"
    assert metrics["top_segment_key"] == "segment_a"
    assert metrics["bottom_servo_ids"] == [5, 6, 7, 8]
    role_prov = json.loads((result.paths.output_dir / "two_segment_tracking_role_provenance.json").read_text())
    assert role_prov["bottom_segment_key"] == "segment_b"


class _DropOneServoBus(MockDxlBus):
    """Mock bus that starts dropping servo 5's present_position after a warm-up.

    Precheck and startup read all 8 servos cleanly (so the experiment passes
    its hard prereqs); once in the capture loop the dropper trips and the
    operator-facing rejection path is exercised.
    """

    def __init__(self, *args, warmup_reads: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._read_call_count = 0
        self._warmup_reads = int(warmup_reads)

    def read_live_telemetry(self, servo_ids):  # type: ignore[override]
        result = super().read_live_telemetry(servo_ids)
        self._read_call_count += 1
        if self._read_call_count > self._warmup_reads:
            if 5 in result and result[5] is not None:
                result[5].present_position = None
        return result


def test_two_segment_collect_pose_dataset_rejects_sample_when_measured_position_missing_in_trusted_run(tmp_path: Path) -> None:
    service = _servo_service(tmp_path, dxl_bus=_DropOneServoBus([1, 2, 3, 4, 5, 6, 7, 8]))
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
            "physical_assembly_confirmed_by_operator": True,
        },
    )

    assert result.success is True
    metrics = result.summary.experiment_metrics
    sample_payload = json.loads(result.paths.samples_path.read_text(encoding="utf-8").splitlines()[0])
    extra = sample_payload["extra"]
    # The capture is rejected at collection time, not later in modeling.
    assert extra["capture_accepted"] is False
    assert extra["capture_rejection_reason"].startswith("measured_servo_position_missing")
    assert extra["missing_measured_servo_ids"] == [5]
    assert "measured_servo_position_missing" in sample_payload["status_flags"]
    # Counter sanity: rejected reflects the live decision.
    assert metrics["rejected_sample_count"] >= 1


def test_two_segment_collect_pose_dataset_records_missing_servo_ids_but_keeps_accepted_in_servo_only_run(tmp_path: Path) -> None:
    service = _servo_service(tmp_path, dxl_bus=_DropOneServoBus([1, 2, 3, 4, 5, 6, 7, 8]))
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
            "allow_servo_only_test_run": True,
            "run_trust_mode": "servo_only",
            "schedule_type": "zero",
        },
    )

    assert result.success is True
    sample_payload = json.loads(result.paths.samples_path.read_text(encoding="utf-8").splitlines()[0])
    extra = sample_payload["extra"]
    # In servo-only mode the missing ID is recorded but the sample stays accepted.
    assert extra["missing_measured_servo_ids"] == [5]
    assert extra["capture_accepted"] is True
    assert "measured_servo_position_missing" in sample_payload["status_flags"]
    # Modeling loader still rejects this sample later via the same field.
    assert extra["measured_servo_feedback"]["5"]["position_tick"] is None


def test_two_segment_tracker_freshness_summary_flags_stale_samples_only_above_threshold() -> None:
    from continuum_robot.experiments.schemas import ExperimentTimeseriesSample
    from continuum_robot.experiments.two_segment_collect_pose_dataset import _two_segment_tracker_freshness_summary

    def _sample(freshness: float | None, stale_roles: list[str] | None = None) -> ExperimentTimeseriesSample:
        return ExperimentTimeseriesSample(
            monotonic_time_s=0.0,
            wall_time_utc="2026-01-01T00:00:00Z",
            phase="zero",
            step_index=0,
            sample_index=0,
            freshness_s=freshness,
            extra={"stale_pose_roles": list(stale_roles or [])},
        )

    samples = [
        _sample(0.01),
        _sample(0.05),
        _sample(0.30),  # stale (above 0.25)
        _sample(0.10),
        _sample(0.40, ["distal_tip"]),  # stale + role-stale
        _sample(None),  # missing tracker timestamp (don't count)
    ]

    summary = _two_segment_tracker_freshness_summary(samples=samples, stale_threshold_s=0.25)

    assert summary["samples_with_freshness"] == 5
    assert summary["samples_stale"] == 2
    assert summary["samples_with_stale_role_observation"] == 1
    assert summary["max_freshness_s"] == 0.40
    assert summary["any_stale_observed"] is True


def test_two_segment_tracker_freshness_summary_empty_samples_returns_unobserved_block() -> None:
    from continuum_robot.experiments.two_segment_collect_pose_dataset import _two_segment_tracker_freshness_summary

    summary = _two_segment_tracker_freshness_summary(samples=[], stale_threshold_s=0.25)

    assert summary["samples_with_freshness"] == 0
    assert summary["samples_stale"] == 0
    assert summary["max_freshness_s"] is None
    assert summary["any_stale_observed"] is False


def test_two_segment_collect_pose_dataset_command_ramping_splits_large_jumps(tmp_path: Path) -> None:
    """Configuring command_ramp_step_cm splits each pattern transition into intermediate writes.

    The schedule's command_ramp_substeps_total should increase relative to a
    no-ramp run; per-sample counters (accepted/rejected) should NOT change
    because intermediate writes do not generate samples.
    """
    service = _servo_service(tmp_path)
    _save_all8_startup(service)
    runner = _runner(
        tmp_path,
        service=service,
        tracking_service=_FakeTrackingService(_tracking_snapshot(include_distal=True, include_intermediate=False)),
    )
    base_config = {
        "dry_run": False,
        "allow_servo_only_test_run": True,
        "run_trust_mode": "servo_only",
        "schedule_type": "single_axis_micro",
        "max_segment_displacement_cm": 0.05,
        "max_tick_delta_from_startup": 320,
        "samples_per_pattern": 1,
        "capture_repeats": 1,
        "long_run_recovery_enabled": True,
    }

    # Baseline: no ramping.
    no_ramp = runner.run_experiment(EXPERIMENT_NAME, config=base_config)
    assert no_ramp.success is True
    no_ramp_metrics = no_ramp.summary.experiment_metrics
    no_ramp_substeps = (no_ramp_metrics.get("long_run_health") or {}).get("command_ramp_substeps_total", 0)
    no_ramp_accepted = no_ramp_metrics["accepted_sample_count"]
    no_ramp_rejected = no_ramp_metrics["rejected_sample_count"]
    assert no_ramp_substeps == 0

    # Ramping enabled with a very small step. Same schedule should still
    # produce the same number of accepted samples, but with non-zero substeps.
    ramped_config = dict(base_config)
    ramped_config["command_ramp_step_cm"] = 0.005  # 5x smaller than amplitude
    ramped_config["command_ramp_settle_time_s"] = 0.0  # no sleep in test
    ramped = runner.run_experiment(EXPERIMENT_NAME, config=ramped_config)
    assert ramped.success is True
    ramped_metrics = ramped.summary.experiment_metrics
    ramped_substeps = (ramped_metrics.get("long_run_health") or {}).get("command_ramp_substeps_total", 0)
    # At least one intermediate sub-command should have been written.
    assert ramped_substeps > 0
    # Sample counters unchanged: ramping must not pollute the dataset.
    assert ramped_metrics["accepted_sample_count"] == no_ramp_accepted
    assert ramped_metrics["rejected_sample_count"] == no_ramp_rejected


def test_two_segment_collect_pose_dataset_blocks_runs_that_exceed_max_tick_delta(tmp_path: Path) -> None:
    """Aggressive amplitude with a tight tick-delta cap must block the run with a clear message."""
    service = _servo_service(tmp_path)
    _save_all8_startup(service)
    runner = _runner(
        tmp_path,
        service=service,
        tracking_service=_FakeTrackingService(_tracking_snapshot(include_distal=True, include_intermediate=False)),
    )

    # Large amplitude (1.0 cm) but absurdly tight tick-delta cap (5 ticks).
    # The mapper converts cm to ticks via spool circumference, and 1 cm = many
    # hundreds of ticks. The precheck must list violations and refuse to run.
    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={
            "dry_run": False,
            "allow_servo_only_test_run": True,
            "run_trust_mode": "servo_only",
            "schedule_type": "single_axis_micro",
            "max_segment_displacement_cm": 1.0,
            "max_tick_delta_from_startup": 5,  # tight cap; expect precheck failure
            "samples_per_pattern": 1,
        },
    )

    assert result.success is False
    assert "Two-segment command schedule exceeds configured tick limits" in result.message


def test_two_segment_collect_pose_dataset_emits_registration_summary_metric(tmp_path: Path) -> None:
    """Every collect-pose run captures the loaded registration's state."""
    service = _servo_service(tmp_path)
    _save_all8_startup(service)
    runner = _runner(
        tmp_path,
        service=service,
        tracking_service=_FakeTrackingService(_tracking_snapshot(include_distal=True, include_intermediate=False)),
    )

    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={"dry_run": False, "allow_servo_only_test_run": True, "run_trust_mode": "servo_only", "schedule_type": "zero"},
    )

    assert result.success is True
    registration = result.summary.experiment_metrics.get("registration_summary") or {}
    assert registration.get("schema_version") == "two_segment_registration_summary_v1"
    assert registration.get("registration_state") == "loaded"
    summary_text = (result.paths.output_dir / "two_segment_dataset_summary.txt").read_text(encoding="utf-8")
    assert "Registration Provenance:" in summary_text


def test_two_segment_collect_pose_dataset_warns_when_registration_not_loaded(tmp_path: Path) -> None:
    service = _servo_service(tmp_path)
    _save_all8_startup(service)
    runner = _runner(tmp_path, service=service, tracking_service=None)  # no tracker

    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={"dry_run": False, "allow_servo_only_test_run": True, "run_trust_mode": "servo_only", "schedule_type": "zero"},
    )

    assert result.success is True
    registration = result.summary.experiment_metrics.get("registration_summary") or {}
    assert registration.get("registration_state") == "tracker_unavailable"
    assert any("Registration not fully loaded" in w for w in result.summary.warning_messages)


def test_two_segment_collect_pose_dataset_summary_text_includes_health_blocks(tmp_path: Path) -> None:
    """two_segment_dataset_summary.txt must surface long-run, current/load, and tracker-freshness sections."""
    service = _servo_service(tmp_path)
    _save_all8_startup(service)
    runner = _runner(
        tmp_path,
        service=service,
        tracking_service=_FakeTrackingService(_tracking_snapshot(include_distal=True, include_intermediate=False)),
    )

    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={"dry_run": False, "allow_servo_only_test_run": True, "run_trust_mode": "servo_only", "schedule_type": "zero"},
    )

    assert result.success is True
    summary_text = (result.paths.output_dir / "two_segment_dataset_summary.txt").read_text(encoding="utf-8")
    assert "Long-Run Health:" in summary_text
    assert "stop_reason:" in summary_text
    assert "Current/Load Summary:" in summary_text
    assert "warning_threshold_ma:" in summary_text
    assert "Tracker Freshness:" in summary_text
    assert "samples_stale:" in summary_text
    # Bottom/top assignment must also be visible.
    assert "bottom_segment_key: segment_a" in summary_text
    assert "top_segment_key: segment_b" in summary_text


def test_two_segment_collect_pose_dataset_emits_tracker_freshness_summary_metric(tmp_path: Path) -> None:
    service = _servo_service(tmp_path)
    _save_all8_startup(service)
    runner = _runner(
        tmp_path,
        service=service,
        tracking_service=_FakeTrackingService(_tracking_snapshot(include_distal=True, include_intermediate=False)),
    )

    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={"dry_run": False, "allow_servo_only_test_run": True, "run_trust_mode": "servo_only", "schedule_type": "zero"},
    )

    assert result.success is True
    summary = result.summary.experiment_metrics.get("tracker_freshness_summary") or {}
    assert summary.get("schema_version") == "two_segment_tracker_freshness_summary_v1"


def test_two_segment_current_load_summary_flags_sustained_jam_but_not_transient_spike() -> None:
    """Unit-test the pure helper so threshold/sustained logic stays deterministic."""

    from continuum_robot.experiments.schemas import ExperimentTimeseriesSample
    from continuum_robot.experiments.two_segment_collect_pose_dataset import _two_segment_current_load_summary

    def _sample(currents_by_servo: dict[int, int | None]) -> ExperimentTimeseriesSample:
        return ExperimentTimeseriesSample(
            monotonic_time_s=0.0,
            wall_time_utc="2026-01-01T00:00:00Z",
            phase="zero",
            step_index=0,
            sample_index=0,
            extra={
                "measured_servo_feedback": {
                    str(servo_id): {
                        "servo_id": int(servo_id),
                        "load_proxy_ma": current if current is None else abs(int(current)),
                        "signed_raw_current_ma": current,
                    }
                    for servo_id, current in currents_by_servo.items()
                },
            },
        )

    samples = [
        _sample({1: 100, 2: 200, 5: 100, 6: 100}),     # normal
        _sample({1: 850, 2: 200, 5: 100, 6: 100}),     # transient spike on servo 1
        _sample({1: 100, 2: 200, 5: 100, 6: 100}),     # back to normal — servo 1 transient
        _sample({1: 100, 2: 200, 5: 820, 6: 100}),     # servo 5 warning, sample 1/3
        _sample({1: 100, 2: 200, 5: 830, 6: 100}),     # servo 5 warning, sample 2/3
        _sample({1: 100, 2: 200, 5: 840, 6: 100}),     # servo 5 warning, sample 3/3 → SUSTAINED
        _sample({1: 100, 2: 200, 5: 100, 6: 100}),     # back to normal
    ]

    summary = _two_segment_current_load_summary(
        samples=samples,
        warning_ma=800,
        hard_ma=850,
        sustained_sample_count=3,
    )

    assert summary["warning_threshold_ma"] == 800
    assert summary["hard_threshold_ma"] == 850
    # Servo 5 hit the warning threshold 3 times in a row → sustained.
    assert 5 in summary["sustained_jam_servo_ids"]
    # Servo 1 had only a single warning-class sample → transient, NOT sustained.
    assert 1 not in summary["sustained_jam_servo_ids"]
    # Per-servo stats are populated.
    assert summary["per_servo"]["5"]["sample_count"] == 7
    assert summary["per_servo"]["5"]["load_proxy"]["max_ma"] == 840.0
    assert summary["per_servo"]["5"]["warning_threshold_exceeded_count"] == 3
    # Servo 1 hit hard threshold once (850).
    assert summary["per_servo"]["1"]["hard_threshold_exceeded_count"] == 1
    assert summary["any_warning_observed"] is True
    assert summary["any_hard_observed"] is True


def test_two_segment_current_load_summary_handles_missing_telemetry_without_resetting_consecutive_counter() -> None:
    from continuum_robot.experiments.schemas import ExperimentTimeseriesSample
    from continuum_robot.experiments.two_segment_collect_pose_dataset import _two_segment_current_load_summary

    def _sample(load: int | None) -> ExperimentTimeseriesSample:
        return ExperimentTimeseriesSample(
            monotonic_time_s=0.0,
            wall_time_utc="2026-01-01T00:00:00Z",
            phase="zero",
            step_index=0,
            sample_index=0,
            extra={
                "measured_servo_feedback": {
                    "1": {"servo_id": 1, "load_proxy_ma": load, "signed_raw_current_ma": load},
                },
            },
        )

    # Warning, warning, missing (count stays at 2), warning → 3 consecutive
    samples = [_sample(820), _sample(830), _sample(None), _sample(840)]
    summary = _two_segment_current_load_summary(samples=samples, warning_ma=800, hard_ma=850, sustained_sample_count=3)
    assert 1 in summary["sustained_jam_servo_ids"]


def test_two_segment_collect_pose_dataset_emits_current_load_summary_metric(tmp_path: Path) -> None:
    service = _servo_service(tmp_path)
    _save_all8_startup(service)
    runner = _runner(
        tmp_path,
        service=service,
        tracking_service=_FakeTrackingService(_tracking_snapshot(include_distal=True, include_intermediate=False)),
    )

    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={"dry_run": False, "allow_servo_only_test_run": True, "run_trust_mode": "servo_only", "schedule_type": "zero"},
    )

    assert result.success is True
    summary = result.summary.experiment_metrics.get("current_load_summary") or {}
    assert summary.get("schema_version") == "two_segment_current_load_summary_v1"
    assert "per_servo" in summary
    # Default MockDxlBus reports nominal current; nothing sustained.
    assert summary.get("sustained_jam_servo_ids") == []


def test_two_segment_collect_pose_dataset_command_failure_produces_single_event_not_duplicate(tmp_path: Path) -> None:
    service = _servo_service(tmp_path)
    _save_all8_startup(service)

    # Force the writer to fail every time so we have a known number of command_failed events.
    def _broken_writer(positions_by_servo: dict[int, int]) -> None:  # noqa: ARG001
        raise RuntimeError("simulated write failure")

    service._write_goal_positions = _broken_writer  # type: ignore[assignment]

    runner = _runner(
        tmp_path,
        service=service,
        tracking_service=_FakeTrackingService(_tracking_snapshot(include_distal=True, include_intermediate=False)),
    )

    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={
            "dry_run": False,
            "allow_servo_only_test_run": True,
            "run_trust_mode": "servo_only",
            "schedule_type": "single_axis_micro",
            "max_segment_displacement_cm": 0.02,
            "max_tick_delta_from_startup": 320,
            "samples_per_pattern": 1,
            "capture_repeats": 1,
            "long_run_recovery_enabled": True,
        },
    )

    assert result.success is True
    metrics = result.summary.experiment_metrics
    long_run = metrics.get("long_run_health") or {}
    failure_events_path = result.paths.output_dir / "sample_failure_events.jsonl"
    raw_events = [json.loads(line) for line in failure_events_path.read_text().splitlines() if line.strip()]
    command_failed_events = [event for event in raw_events if event["failure_kind"] == "command_failed"]
    capture_rejected_events = [event for event in raw_events if event["failure_kind"] == "capture_rejected"]

    # Exactly one command_failed event per failed pattern. No duplicate capture_rejected.
    assert long_run["command_failures"] == len(command_failed_events)
    assert capture_rejected_events == []
    # Rejected counter still increments per sample, independent of the events de-dup.
    assert long_run["rejected_samples"] >= long_run["command_failures"]


def test_two_segment_collect_pose_dataset_continue_until_valid_samples_stops_when_target_reached(tmp_path: Path) -> None:
    service = _servo_service(tmp_path)
    _save_all8_startup(service)
    runner = _runner(
        tmp_path,
        service=service,
        tracking_service=_FakeTrackingService(_tracking_snapshot(include_distal=True, include_intermediate=False)),
    )

    target = 3
    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={
            "dry_run": False,
            "allow_servo_only_test_run": True,
            "run_trust_mode": "servo_only",
            "schedule_type": "single_axis_micro",
            "max_segment_displacement_cm": 0.02,
            "max_tick_delta_from_startup": 320,
            "continue_until_valid_samples": True,
            "target_valid_sample_count": target,
            "long_run_recovery_enabled": True,
        },
    )

    assert result.success is True
    metrics = result.summary.experiment_metrics
    long_run = metrics.get("long_run_health") or {}
    assert long_run["stop_reason"] == "target_valid_sample_count_reached"
    assert long_run["accepted_samples"] >= target
    assert long_run["continue_until_valid_samples"] is True
    assert long_run["target_valid_sample_count"] == target
