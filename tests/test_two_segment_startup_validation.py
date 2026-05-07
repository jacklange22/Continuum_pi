from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from continuum_robot.config.schemas import (
    CalibrationConfig,
    ExperimentConfig,
    RegistrationWorkflowConfig,
    RobotConfig,
    RobotSegmentConfig,
    RuntimeConfig,
    SafetyConfig,
    SerialConfig,
)
from continuum_robot.config.settings import Settings
from continuum_robot.data.export_run_bundle import export_run_bundle
from continuum_robot.data.validate_run_bundle import validate_run_folder
from continuum_robot.experiments.experiment_runner import ExperimentRunner
from continuum_robot.experiments.registry import ExperimentRegistry
from continuum_robot.experiments.builtins import register_builtin_experiments
from continuum_robot.experiments.two_segment_startup_validation import EXPERIMENT_NAME, STAGE_ORDER
from continuum_robot.gui.experiment_preflight import RUN_BLOCKED, evaluate_preflight
from continuum_robot.hardware.dxl_bus import ServoTelemetry
from continuum_robot.hardware.mock_dxl_bus import MockDxlBus
from continuum_robot.servos.displacement_mapper import TendonDisplacementMapper
from continuum_robot.servos.neutral_calibration_service import NeutralCalibrationService, ServoCalibrationContext
from continuum_robot.servos.pretension_validation_service import PretensionValidationService
from continuum_robot.servos.safety_guard import SafetyGuard
from continuum_robot.servos.servo_service import ServoService


class _MissingPositionBus(MockDxlBus):
    def read_live_telemetry(self, servo_ids: list[int]) -> dict[int, ServoTelemetry]:
        result = super().read_live_telemetry(servo_ids)
        result[8].present_position = None
        return result


class _MissingCurrentBus(MockDxlBus):
    def read_live_telemetry(self, servo_ids: list[int]) -> dict[int, ServoTelemetry]:
        result = super().read_live_telemetry(servo_ids)
        result[8].present_current_ma = None
        return result


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
        registration=RegistrationWorkflowConfig(capture_tool_id="0B", coil_tool_id="0A", max_fre_mm=None),
        experiment=ExperimentConfig(default_settle_time_s=0.0, sample_count_per_point=1, output_dir="data/experiments"),
        calibration=CalibrationConfig(
            neutral_setpoints_path="config/neutral_setpoints.json",
            latest_registration_path="data/registrations/latest_registration.json",
        ),
    )


def _servo_service(tmp_path: Path, *, bus: MockDxlBus | None = None, settings: Settings | None = None) -> ServoService:
    settings = settings or _settings()
    context = settings.robot.operating_context()
    service = ServoService(
        dxl_bus=bus or MockDxlBus([1, 2, 3, 4, 5, 6, 7, 8]),
        mapper=TendonDisplacementMapper(spool_diameter_cm=1.2),
        safety_guard=SafetyGuard(min_offset_ticks=-600, max_offset_ticks=600, max_current_ma=850),
        neutral_calibration=NeutralCalibrationService(
            path=tmp_path / "neutral.json",
            context=ServoCalibrationContext(
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
            ),
        ),
        pretension_validation=PretensionValidationService(),
    )
    service.connect(settings.serial.openrb_port, settings.serial.baudrate)
    return service


def _runner(tmp_path: Path, *, settings: Settings | None = None, bus: MockDxlBus | None = None) -> ExperimentRunner:
    settings = settings or _settings()
    return ExperimentRunner(
        project_root=tmp_path,
        settings=settings,
        tracking_service=None,
        servo_service=_servo_service(tmp_path, bus=bus, settings=settings),
        output_dir=tmp_path / "data" / "experiments",
        default_settle_time_s=0.0,
        registration_path=tmp_path / "latest_registration.json",
        sleep_fn=lambda _seconds: None,
    )


def test_two_segment_startup_validation_experiment_is_registered() -> None:
    registry = ExperimentRegistry()
    register_builtin_experiments(registry)

    descriptor = registry.get(EXPERIMENT_NAME)

    assert descriptor.title == "Two-Segment Startup Validation"
    assert "Two Segment" in descriptor.tags


def test_two_segment_startup_validation_blocks_outside_dual_segment(tmp_path: Path) -> None:
    runner = _runner(tmp_path, settings=_settings(mode="single_segment"))

    result = runner.run_experiment(EXPERIMENT_NAME, config={})

    assert result.success is False
    assert "requires operating_mode=dual_segment" in result.message


def test_two_segment_startup_validation_preflight_allows_missing_tracker_but_blocks_wrong_mode(tmp_path: Path) -> None:
    snapshot = SimpleNamespace(
        selected_backend_name="mock",
        backend_identity="mock",
        canonical_state="disconnected",
    )
    dual_report = evaluate_preflight(
        experiment_name=EXPERIMENT_NAME,
        config_payload={},
        config_error=None,
        settings=_settings(),
        tracking_snapshot=snapshot,
        servo_connected=True,
        neutral_setpoints={},
        registration_path=tmp_path / "registration.json",
        output_root=tmp_path / "data" / "experiments",
        planned_output_dir=tmp_path / "data" / "experiments" / EXPERIMENT_NAME / "run",
        project_root=tmp_path,
    )
    single_report = evaluate_preflight(
        experiment_name=EXPERIMENT_NAME,
        config_payload={},
        config_error=None,
        settings=_settings(mode="single_segment"),
        tracking_snapshot=snapshot,
        servo_connected=True,
        neutral_setpoints={},
        registration_path=tmp_path / "registration.json",
        output_root=tmp_path / "data" / "experiments",
        planned_output_dir=tmp_path / "data" / "experiments" / EXPERIMENT_NAME / "single_run",
        project_root=tmp_path,
    )

    assert dual_report.overall_status != RUN_BLOCKED
    assert any("Tracker is optional" in warning for warning in dual_report.warning_messages)
    assert single_report.overall_status == RUN_BLOCKED
    assert any("requires operating_mode=dual_segment" in message for message in single_report.blocking_messages)


def test_two_segment_startup_validation_writes_stages_artifact_reports_and_validates(tmp_path: Path) -> None:
    runner = _runner(tmp_path)

    result = runner.run_experiment(EXPERIMENT_NAME, config={"workflow_state_path": "data/startup_state.json"})

    assert result.success is True
    metrics = result.summary.experiment_metrics
    assert metrics["stage_order"] == STAGE_ORDER
    assert metrics["stage_completion"] == {stage: True for stage in STAGE_ORDER}
    assert metrics["final_accepted"] is True
    assert metrics["tracker_available"] is False
    assert metrics["automatic_two_segment_pretension_validated"] is False
    assert metrics["two_segment_control_validated"] is False
    assert len(metrics["stage_snapshots"]) == 5
    assert len(metrics["stage_snapshots"][0]["servos"]) == 8
    assert metrics["stage_snapshots"][0]["segments"]["segment_a"]["segment_role"] == "proximal"
    assert metrics["stage_snapshots"][0]["segments"]["segment_b"]["segment_role"] == "distal"

    artifact_path = Path(metrics["final_startup_artifact_path"])
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["robot"]["startup_type"] == "manual_two_segment_startup"
    assert artifact["robot"]["stage_order"] == STAGE_ORDER
    assert artifact["robot"]["automatic_two_segment_pretension_validated"] is False
    assert sorted(int(key) for key in artifact["servos"]) == [1, 2, 3, 4, 5, 6, 7, 8]

    single_context = _settings(mode="single_segment").robot.operating_context()
    single_calibration = NeutralCalibrationService(
        path=artifact_path,
        context=ServoCalibrationContext(
            robot_mode=single_context.operating_mode,
            servo_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            expected_servo_ids=list(single_context.expected_servo_ids),
            commanded_servo_ids=list(single_context.commanded_servo_ids),
            segments=_settings(mode="single_segment").robot.segment_metadata(),
            segment_order=_settings(mode="single_segment").robot.segment_order(),
        ),
    )
    assert single_calibration.get_calibration_summary().compatible is False

    for filename in [
        "two_segment_startup_summary.txt",
        "metrics.csv",
        "two_segment_startup_artifact_metadata.json",
        "two_segment_startup_positions_report.png",
        "two_segment_startup_load_proxy_report.png",
        "two_segment_startup_stage_drift_report.png",
    ]:
        assert (result.paths.output_dir / filename).exists()

    validation = validate_run_folder(result.paths.output_dir)
    assert validation.status == "PASS"

    export = export_run_bundle(run_dir=result.paths.output_dir, output_root=tmp_path / "exports", project_root=tmp_path)
    exported_names = {entry.bundle_path for entry in export.entries}
    assert "two_segment_startup_summary.txt" in exported_names
    assert "two_segment_startup_artifact_metadata.json" in exported_names


def test_two_segment_startup_validation_supports_one_stage_manual_checkpoint_sequence(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    config = {"workflow_state_path": "data/startup_state.json"}

    baseline = runner.run_experiment(EXPERIMENT_NAME, config={**config, "capture_stage": "baseline", "final_accept": False})
    segment_a = runner.run_experiment(EXPERIMENT_NAME, config={**config, "capture_stage": "segment_a_pretensioned", "final_accept": False})
    segment_b = runner.run_experiment(EXPERIMENT_NAME, config={**config, "capture_stage": "segment_b_pretensioned", "final_accept": False})
    recheck = runner.run_experiment(EXPERIMENT_NAME, config={**config, "capture_stage": "segment_a_recheck", "final_accept": False})
    final = runner.run_experiment(EXPERIMENT_NAME, config={**config, "capture_stage": "final_accept", "final_accept": True})

    assert all(result.success for result in [baseline, segment_a, segment_b, recheck, final])
    assert final.summary.experiment_metrics["stage_completion"] == {stage: True for stage in STAGE_ORDER}
    assert final.summary.experiment_metrics["final_accepted"] is True


def test_two_segment_startup_validation_blocks_missing_positions(tmp_path: Path) -> None:
    runner = _runner(tmp_path, bus=_MissingPositionBus([1, 2, 3, 4, 5, 6, 7, 8]))

    result = runner.run_experiment(EXPERIMENT_NAME, config={})

    assert result.success is False
    assert "position ticks are missing for servo IDs [8]" in result.message


def test_two_segment_startup_validation_warns_for_missing_current_but_captures_positions(tmp_path: Path) -> None:
    runner = _runner(tmp_path, bus=_MissingCurrentBus([1, 2, 3, 4, 5, 6, 7, 8]))

    result = runner.run_experiment(EXPERIMENT_NAME, config={"allow_missing_current": True})

    assert result.success is True
    assert any("without current estimates for servo IDs [8]" in warning for warning in result.summary.warning_messages)
    first_stage = result.summary.experiment_metrics["stage_snapshots"][0]
    assert first_stage["servos"]["8"]["signed_raw_current_ma"] is None
