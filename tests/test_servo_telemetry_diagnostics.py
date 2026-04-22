from __future__ import annotations

from pathlib import Path

from continuum_robot.hardware.mock_dxl_bus import MockDxlBus
from continuum_robot.servos.displacement_mapper import TendonDisplacementMapper
from continuum_robot.servos.neutral_calibration_service import (
    NeutralCalibrationService,
    ServoCalibrationContext,
)
from continuum_robot.servos.pretension_validation_service import PretensionValidationService
from continuum_robot.servos.safety_guard import SafetyGuard
from continuum_robot.servos.servo_service import ServoService
from continuum_robot.servos.telemetry_diagnostics import (
    benchmark_telemetry_profile,
    build_telemetry_gui_policy,
    full_telemetry_profile,
    live_telemetry_profile,
    write_benchmark_outputs,
)


def _build_service(tmp_path: Path) -> ServoService:
    servo_ids = [1, 2, 3, 4]
    return ServoService(
        dxl_bus=MockDxlBus(servo_ids=servo_ids),
        mapper=TendonDisplacementMapper(spool_diameter_cm=1.2),
        safety_guard=SafetyGuard(
            min_offset_ticks=-600,
            max_offset_ticks=600,
            max_current_ma=850,
            telemetry_stale_after_s=0.25,
            time_fn=lambda: 0.0,
        ),
        neutral_calibration=NeutralCalibrationService(
            path=tmp_path / "neutral.json",
            context=ServoCalibrationContext(
                robot_mode="4-servo",
                robot_config_name="robot_4servo.yaml",
                servo_ids=list(servo_ids),
                tendon_to_servo=list(servo_ids),
                position_min_offset_ticks=-600,
                position_max_offset_ticks=600,
                default_pretension_current_threshold_ma=220,
                tightening_rotation_by_servo={servo_id: "cw" for servo_id in servo_ids},
            ),
        ),
        pretension_validation=PretensionValidationService(),
        sleep_fn=lambda _seconds: None,
        time_fn=lambda: 0.0,
    )


def test_telemetry_profiles_match_current_bus_contract() -> None:
    live = live_telemetry_profile()
    full = full_telemetry_profile()

    assert live.name == "live"
    assert live.register_transaction_count_per_servo == 7
    assert "present_position" in live.register_keys
    assert "hardware_error_status" in live.register_keys

    assert full.name == "full"
    assert full.register_transaction_count_per_servo == 14
    assert "servo_id" in full.register_keys
    assert "bus_watchdog" in full.register_keys


def test_build_telemetry_gui_policy_reports_current_refresh_contract() -> None:
    policy = build_telemetry_gui_policy(
        baudrate=57600,
        poll_rate_hz=20,
        telemetry_stale_after_s=0.25,
    )

    assert policy.servos_selected_refresh_target_hz == 20.0
    assert policy.servos_full_refresh_target_hz == 5.0
    assert policy.system_summary_refresh_target_hz == 5.0
    assert policy.primary_limiter == "baudrate"
    assert "System auto servo summary 5.0 Hz" in policy.cadence_summary
    assert "present position" in policy.field_summary
    assert "57600-style DYNAMIXEL baudrate" in policy.bottleneck_summary


def test_benchmark_telemetry_profile_measures_mock_bus_and_writes_outputs(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    service.connect("/dev/mock-openrb", 57600)

    live_result = benchmark_telemetry_profile(service, [1, 2, 3, 4], profile="live", iterations=8)
    full_result = benchmark_telemetry_profile(service, [1, 2], profile="full", iterations=6)
    outputs = write_benchmark_outputs(
        output_dir=tmp_path / "telemetry_benchmark",
        results=[live_result, full_result],
        metadata={"baudrate": 57600, "servo_ids": [1, 2, 3, 4]},
    )

    assert live_result.profile_name == "live"
    assert live_result.register_transaction_count_per_servo == 7
    assert live_result.effective_loop_hz > 0.0
    assert live_result.aggregate_servo_samples_hz > live_result.effective_loop_hz

    assert full_result.profile_name == "full"
    assert full_result.register_transaction_count_per_servo == 14
    assert full_result.effective_loop_hz > 0.0

    summary_json = outputs["summary_json"]
    summary_text = outputs["summary_text"]
    assert summary_json.exists()
    assert summary_text.exists()
    assert '"profile_name": "live"' in summary_json.read_text(encoding="utf-8")
    assert "register_transaction_count_per_servo=14" in summary_text.read_text(encoding="utf-8")
