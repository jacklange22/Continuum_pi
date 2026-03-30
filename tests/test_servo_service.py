from pathlib import Path

import pytest

from continuum_robot.hardware.mock_dxl_bus import MockDxlBus
from continuum_robot.servos.displacement_mapper import TendonDisplacementMapper
from continuum_robot.servos.neutral_calibration_service import (
    NeutralCalibrationService,
    ServoCalibrationContext,
)
from continuum_robot.servos.pretension_validation_service import PretensionValidationService
from continuum_robot.servos.safety_guard import SafetyGuard
from continuum_robot.servos.servo_service import ServoService


class _PretensionBus(MockDxlBus):
    def __init__(self, *, current_sequence: list[int | None], max_limit: int = 4095) -> None:
        super().__init__([1])
        self._current_sequence = list(current_sequence)
        self._state[1].max_position_limit = max_limit

    def write_goal_positions(self, positions_by_id: dict[int, int]) -> None:
        super().write_goal_positions(positions_by_id)
        if self._current_sequence:
            self._state[1].present_current_ma = self._current_sequence.pop(0)
        self._state[1].last_read_monotonic_s = self._state[1].last_read_monotonic_s


def _build_service(
    tmp_path: Path,
    *,
    dxl_bus=None,
    time_fn=None,
    context_servo_ids: list[int] | None = None,
) -> ServoService:
    bus = dxl_bus or MockDxlBus([1, 2, 3, 4])
    if context_servo_ids is None:
        state = getattr(bus, "_state", {})
        context_servo_ids = sorted(state) if state else [1, 2, 3, 4]
    robot_mode = "1-servo" if len(context_servo_ids) == 1 else "4-servo"
    return ServoService(
        dxl_bus=bus,
        mapper=TendonDisplacementMapper(spool_diameter_cm=1.2),
        safety_guard=SafetyGuard(
            min_offset_ticks=-600,
            max_offset_ticks=600,
            max_current_ma=850,
            default_pretension_current_threshold_ma=220,
            fine_jog_step_ticks=5,
            coarse_jog_step_ticks=25,
            software_position_margin_ticks=64,
            pretension_step_ticks=2,
            pretension_timeout_s=2.0,
            pretension_settle_time_s=0.0,
            time_fn=time_fn or (lambda: 0.0),
        ),
        neutral_calibration=NeutralCalibrationService(
            path=tmp_path / "neutral.json",
            context=ServoCalibrationContext(
                robot_mode=robot_mode,
                servo_ids=list(context_servo_ids),
                tendon_to_servo=list(context_servo_ids),
                position_min_offset_ticks=-600,
                position_max_offset_ticks=600,
                default_pretension_current_threshold_ma=220,
                tightening_rotation_by_servo={servo_id: "cw" for servo_id in context_servo_ids},
            ),
        ),
        pretension_validation=PretensionValidationService(),
        sleep_fn=lambda _seconds: None,
        time_fn=time_fn or (lambda: 0.0),
    )


def test_servo_service_capture_save_load_and_command(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    service.connect("/dev/mock-openrb", 115200)
    neutral = service.capture_neutral_setpoints([1, 2, 3, 4])
    service.save_neutral_setpoints(neutral)

    loaded = service.load_neutral_setpoints()
    result = service.command_displacement(
        tendon_displacements_cm=[0.0, 0.05, -0.05, 0.0],
        neutral_ticks=[loaded[1], loaded[2], loaded[3], loaded[4]],
        servo_ids=[1, 2, 3, 4],
    )

    assert loaded == neutral
    assert set(result.positions_by_id) == {1, 2, 3, 4}
    assert result.telemetry_by_id[2].present_position == result.positions_by_id[2]
    summary = service.get_calibration_summary()
    assert summary.compatible is True
    assert summary.servo_entries[1].safe_min_tick == neutral[1] - 600
    assert summary.servo_entries[1].safe_max_tick == neutral[1] + 600
    assert summary.servo_entries[1].pretension_current_threshold_ma == 220


def test_servo_service_startup_calibration_persists_bounds_and_threshold(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    service.connect("/dev/mock-openrb", 115200)

    entry = service.save_startup_calibration(
        servo_id=1,
        min_offset_ticks=-100,
        max_offset_ticks=150,
        pretension_current_threshold_ma=240,
    )

    assert entry.neutral_setpoint == 2048
    assert entry.safe_min_tick == 1948
    assert entry.safe_max_tick == 2198
    summary = service.get_calibration_summary()
    assert summary.servo_entries[1].pretension_current_threshold_ma == 240
    assert summary.servo_entries[1].tightening_rotation == "cw"


def test_servo_service_blocks_jog_when_operating_mode_is_wrong(tmp_path: Path) -> None:
    bus = MockDxlBus([1])
    bus._state[1].operating_mode = 5
    service = _build_service(tmp_path, dxl_bus=bus)
    service.connect("/dev/mock-openrb", 115200)

    with pytest.raises(RuntimeError, match="Operating Mode 5 is not allowed"):
        service.jog_servo(1, 5)


def test_servo_service_blocks_jog_when_telemetry_is_missing(tmp_path: Path) -> None:
    bus = MockDxlBus([1])
    bus._state[1].present_current_ma = None
    service = _build_service(tmp_path, dxl_bus=bus)
    service.connect("/dev/mock-openrb", 115200)

    with pytest.raises(RuntimeError, match="Current telemetry is unavailable"):
        service.jog_servo(1, 5)


def test_servo_service_enforces_coarse_jog_limit(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    service.connect("/dev/mock-openrb", 115200)

    with pytest.raises(ValueError, match="coarse jog limit"):
        service.jog_servo(1, 40)


def test_servo_service_pretension_validation_returns_message(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    service.connect("/dev/mock-openrb", 115200)
    result = service.validate_pretension([1, 2, 3, 4], tolerance_ma=80)

    assert result.spread_ma is not None
    assert "spread" in result.message


def test_servo_service_pretension_stops_on_threshold_and_can_be_accepted(tmp_path: Path) -> None:
    bus = _PretensionBus(current_sequence=[180, 230])
    service = _build_service(tmp_path, dxl_bus=bus)
    service.connect("/dev/mock-openrb", 115200)
    service.save_startup_calibration(
        servo_id=1,
        min_offset_ticks=-40,
        max_offset_ticks=40,
        pretension_current_threshold_ma=220,
    )

    result = service.run_pretension_routine(servo_id=1)

    assert result.success is True
    assert result.status == "threshold_reached"
    accepted = service.accept_pretension_result(1)
    assert accepted.pretension_result_status == "accepted"


def test_servo_service_pretension_fails_on_overcurrent(tmp_path: Path) -> None:
    bus = _PretensionBus(current_sequence=[900])
    service = _build_service(tmp_path, dxl_bus=bus)
    service.connect("/dev/mock-openrb", 115200)
    service.save_startup_calibration(
        servo_id=1,
        min_offset_ticks=-20,
        max_offset_ticks=20,
        pretension_current_threshold_ma=220,
    )

    result = service.run_pretension_routine(servo_id=1)

    assert result.success is False
    assert result.status == "overcurrent"


def test_servo_service_pretension_fails_on_travel_limit(tmp_path: Path) -> None:
    bus = _PretensionBus(current_sequence=[150], max_limit=2114)
    service = _build_service(tmp_path, dxl_bus=bus)
    service.connect("/dev/mock-openrb", 115200)
    service.save_startup_calibration(
        servo_id=1,
        min_offset_ticks=-5,
        max_offset_ticks=2,
        pretension_current_threshold_ma=220,
    )

    result = service.run_pretension_routine(servo_id=1)

    assert result.success is False
    assert result.status == "travel_limit"


def test_servo_service_pretension_fails_on_timeout(tmp_path: Path) -> None:
    timeline = iter([0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0])
    service = _build_service(
        tmp_path,
        dxl_bus=_PretensionBus(current_sequence=[180, 190]),
        time_fn=lambda: next(timeline),
    )
    service.connect("/dev/mock-openrb", 115200)
    service.save_startup_calibration(
        servo_id=1,
        min_offset_ticks=-40,
        max_offset_ticks=40,
        pretension_current_threshold_ma=220,
    )

    result = service.run_pretension_routine(servo_id=1)

    assert result.success is False
    assert result.status == "timeout"


def test_servo_service_pretension_fails_when_current_disappears(tmp_path: Path) -> None:
    bus = _PretensionBus(current_sequence=[None])
    service = _build_service(tmp_path, dxl_bus=bus)
    service.connect("/dev/mock-openrb", 115200)
    service.save_startup_calibration(
        servo_id=1,
        min_offset_ticks=-40,
        max_offset_ticks=40,
        pretension_current_threshold_ma=220,
    )

    result = service.run_pretension_routine(servo_id=1)

    assert result.success is False
    assert result.status == "invalid_telemetry"
