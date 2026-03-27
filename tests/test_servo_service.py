from pathlib import Path

from continuum_robot.hardware.mock_dxl_bus import MockDxlBus
from continuum_robot.servos.displacement_mapper import TendonDisplacementMapper
from continuum_robot.servos.neutral_calibration_service import NeutralCalibrationService
from continuum_robot.servos.pretension_validation_service import PretensionValidationService
from continuum_robot.servos.safety_guard import SafetyGuard
from continuum_robot.servos.servo_service import ServoService


def _build_service(tmp_path: Path) -> ServoService:
    return ServoService(
        dxl_bus=MockDxlBus([1, 2, 3, 4]),
        mapper=TendonDisplacementMapper(spool_diameter_cm=1.2),
        safety_guard=SafetyGuard(min_offset_ticks=-600, max_offset_ticks=600, max_current_ma=850),
        neutral_calibration=NeutralCalibrationService(path=tmp_path / "neutral.json"),
        pretension_validation=PretensionValidationService(),
    )


def test_servo_service_capture_save_load_and_command(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    service.connect("/dev/mock-openrb", 115200)
    neutral = service.capture_neutral_setpoints([1, 2, 3, 4])
    service.save_neutral_setpoints(neutral)

    loaded = service.load_neutral_setpoints()
    result = service.command_displacement(
        tendon_displacements_cm=[0.0, 0.1, -0.1, 0.0],
        neutral_ticks=[loaded[1], loaded[2], loaded[3], loaded[4]],
        servo_ids=[1, 2, 3, 4],
    )

    assert loaded == neutral
    assert set(result.positions_by_id) == {1, 2, 3, 4}
    assert result.telemetry_by_id[2].present_position == result.positions_by_id[2]


def test_servo_service_pretension_validation_returns_message(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    service.connect("/dev/mock-openrb", 115200)
    result = service.validate_pretension([1, 2, 3, 4], tolerance_ma=80)

    assert result.spread_ma is not None
    assert "spread" in result.message
