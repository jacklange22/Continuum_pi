from __future__ import annotations

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
from continuum_robot.servos.servo_service import PretensionParameters, ServoService


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


class _RecordingIdBus(MockDxlBus):
    def __init__(self) -> None:
        super().__init__([1])
        self.write_id_calls: list[tuple[int, int, bool | None]] = []

    def write_servo_id(self, current_id: int, new_id: int) -> None:
        self.write_id_calls.append((int(current_id), int(new_id), self._state[current_id].torque_enabled))
        super().write_servo_id(current_id, new_id)


class _StaleTelemetryBus(MockDxlBus):
    def read_telemetry(self, servo_ids: list[int], **kwargs) -> dict[int, object]:
        result = super().read_telemetry(servo_ids, **kwargs)
        for servo_id in servo_ids:
            result[int(servo_id)].last_read_monotonic_s = self._state[int(servo_id)].last_read_monotonic_s
        return result


class _StalePretensionBus(_PretensionBus):
    def read_telemetry(self, servo_ids: list[int], **kwargs) -> dict[int, object]:
        result = super().read_telemetry(servo_ids, **kwargs)
        for servo_id in servo_ids:
            result[int(servo_id)].last_read_monotonic_s = self._state[int(servo_id)].last_read_monotonic_s
        return result


class _BaselineSequenceBus(MockDxlBus):
    def __init__(self, *, baseline_sequence: list[int]) -> None:
        super().__init__([1])
        self._baseline_sequence = list(baseline_sequence)
        self._state[1].torque_enabled = True

    def read_telemetry(self, servo_ids: list[int], **kwargs) -> dict[int, object]:
        result = super().read_telemetry(servo_ids, **kwargs)
        if self._baseline_sequence:
            current = self._baseline_sequence.pop(0)
            result[1].present_current_ma = current
            self._state[1].present_current_ma = current
        return result


def _build_service(
    tmp_path: Path,
    *,
    dxl_bus=None,
    time_fn=None,
    context_servo_ids: list[int] | None = None,
    telemetry_stale_after_s: float = 0.25,
    min_input_voltage_mv: int = 4000,
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
            telemetry_stale_after_s=telemetry_stale_after_s,
            pretension_step_ticks=2,
            pretension_timeout_s=2.0,
            pretension_settle_time_s=0.0,
            min_input_voltage_mv=min_input_voltage_mv,
            time_fn=time_fn or (lambda: 0.0),
        ),
        neutral_calibration=NeutralCalibrationService(
            path=tmp_path / "neutral.json",
            context=ServoCalibrationContext(
                robot_mode=robot_mode,
                robot_config_name=f"robot_{robot_mode}.yaml",
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
    assert summary.servo_entries[1].capture_source == "live_present_position"


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
    service = _build_service(tmp_path, dxl_bus=bus, context_servo_ids=[1])
    service.connect("/dev/mock-openrb", 115200)
    service.save_startup_calibration(
        servo_id=1,
        min_offset_ticks=-20,
        max_offset_ticks=20,
        pretension_current_threshold_ma=220,
    )
    bus._state[1].operating_mode = 5

    with pytest.raises(RuntimeError, match="Operating Mode 5 is not allowed"):
        service.jog_servo(1, 5)


def test_servo_service_blocks_jog_when_telemetry_is_missing(tmp_path: Path) -> None:
    bus = MockDxlBus([1])
    service = _build_service(tmp_path, dxl_bus=bus, context_servo_ids=[1])
    service.connect("/dev/mock-openrb", 115200)
    service.save_startup_calibration(
        servo_id=1,
        min_offset_ticks=-20,
        max_offset_ticks=20,
        pretension_current_threshold_ma=220,
    )
    bus._state[1].present_current_ma = None

    with pytest.raises(RuntimeError, match="Current telemetry is unavailable"):
        service.jog_servo(1, 5)


def test_servo_service_enforces_coarse_jog_limit(tmp_path: Path) -> None:
    service = _build_service(tmp_path, dxl_bus=MockDxlBus([1]), context_servo_ids=[1])
    service.connect("/dev/mock-openrb", 115200)
    service.save_startup_calibration(
        servo_id=1,
        min_offset_ticks=-20,
        max_offset_ticks=20,
        pretension_current_threshold_ma=220,
    )

    with pytest.raises(ValueError, match="coarse jog limit"):
        service.jog_servo(1, 40)


def test_servo_service_blocks_jog_when_telemetry_is_stale(tmp_path: Path) -> None:
    bus = _StaleTelemetryBus([1])
    bus._state[1].last_read_monotonic_s = 0.0
    service = _build_service(
        tmp_path,
        dxl_bus=bus,
        context_servo_ids=[1],
        time_fn=lambda: 0.0,
        telemetry_stale_after_s=0.25,
    )
    service.connect("/dev/mock-openrb", 115200)
    service.save_startup_calibration(
        servo_id=1,
        min_offset_ticks=-20,
        max_offset_ticks=20,
        pretension_current_threshold_ma=220,
    )
    bus._state[1].last_read_monotonic_s = 0.0
    service.safety_guard._time_fn = lambda: 1.0

    with pytest.raises(RuntimeError, match="Telemetry is stale"):
        service.jog_servo(1, 5)


def test_servo_service_blocks_jog_when_hardware_error_is_present(tmp_path: Path) -> None:
    bus = MockDxlBus([1])
    service = _build_service(tmp_path, dxl_bus=bus, context_servo_ids=[1])
    service.connect("/dev/mock-openrb", 115200)
    service.save_startup_calibration(
        servo_id=1,
        min_offset_ticks=-20,
        max_offset_ticks=20,
        pretension_current_threshold_ma=220,
    )
    bus._state[1].hardware_error_code = 4

    with pytest.raises(RuntimeError, match="Hardware Error Status is 0x04"):
        service.jog_servo(1, 5)


def test_servo_service_blocks_jog_on_unsafe_temperature_and_voltage(tmp_path: Path) -> None:
    bus = MockDxlBus([1])
    service = _build_service(tmp_path, dxl_bus=bus, context_servo_ids=[1], min_input_voltage_mv=6000)
    service.connect("/dev/mock-openrb", 115200)
    service.save_startup_calibration(
        servo_id=1,
        min_offset_ticks=-20,
        max_offset_ticks=20,
        pretension_current_threshold_ma=220,
    )
    bus._state[1].present_temperature_c = 71
    with pytest.raises(RuntimeError, match="Temperature threshold exceeded"):
        service.jog_servo(1, 5)

    bus._state[1].present_temperature_c = 33
    bus._state[1].present_voltage_mv = 5000
    with pytest.raises(RuntimeError, match="Input voltage is below the configured motion minimum"):
        service.jog_servo(1, 5)


def test_servo_service_manual_jog_rejects_targets_outside_raw_bench_range(tmp_path: Path) -> None:
    service = _build_service(tmp_path, dxl_bus=MockDxlBus([1]), context_servo_ids=[1])
    service.connect("/dev/mock-openrb", 115200)
    service.dxl_bus._state[1].present_position = 4092

    with pytest.raises(ValueError, match="outside the active motion range"):
        service.jog_servo(1, 6)


def test_servo_service_directional_jog_uses_canonical_raw_position_convention(tmp_path: Path) -> None:
    bus = MockDxlBus([1])
    bus.config.positive_tick_rotation = "cw"
    service = _build_service(tmp_path, dxl_bus=bus, context_servo_ids=[1])
    service.connect("/dev/mock-openrb", 115200)
    service.save_startup_calibration(
        servo_id=1,
        min_offset_ticks=-10,
        max_offset_ticks=10,
        pretension_current_threshold_ma=220,
    )

    tighten = service.jog_servo_directional(servo_id=1, command_direction="tighten", step_ticks=5)
    loosen = service.jog_servo_directional(servo_id=1, command_direction="loosen", step_ticks=5)

    assert tighten.success is True
    assert tighten.delta_ticks == -5
    assert loosen.success is True
    assert loosen.delta_ticks == 5


def test_servo_service_jog_action_clamps_to_full_raw_range_in_bench_mode(tmp_path: Path) -> None:
    bus = MockDxlBus([1])
    bus._state[1].present_position = 4090
    service = _build_service(tmp_path, dxl_bus=bus, context_servo_ids=[1])
    service.connect("/dev/mock-openrb", 115200)

    result = service.jog_servo_action(servo_id=1, action="loosen_coarse")

    assert result.success is True
    assert result.clamped is True
    assert result.goal_tick == 4095
    assert result.unclamped_goal_tick == 4115
    assert result.safe_min_tick == 0
    assert result.safe_max_tick == 4095


def test_servo_service_jog_action_blocks_when_already_at_raw_limit(tmp_path: Path) -> None:
    bus = MockDxlBus([1])
    bus._state[1].present_position = 0
    service = _build_service(tmp_path, dxl_bus=bus, context_servo_ids=[1])
    service.connect("/dev/mock-openrb", 115200)

    result = service.jog_servo_action(servo_id=1, action="tighten_fine")

    assert result.success is False
    assert result.blocked is True
    assert "active minimum raw position" in result.message


def test_servo_service_capture_neutral_persists_hardware_clamped_raw_bounds(tmp_path: Path) -> None:
    bus = MockDxlBus([1])
    bus._state[1].present_position = 4020
    service = _build_service(tmp_path, dxl_bus=bus, context_servo_ids=[1])
    service.connect("/dev/mock-openrb", 115200)

    result = service.capture_and_save_neutral_setpoints([1])

    assert result.safe_bounds_by_id[1] == (3420, 4095)
    summary = service.get_calibration_summary()
    assert summary.servo_entries[1].safe_min_tick == 3420
    assert summary.servo_entries[1].safe_max_tick == 4095


def test_servo_service_ignores_application_bounds_metadata_for_single_servo_bench_jog(tmp_path: Path) -> None:
    bus = MockDxlBus([1])
    service = _build_service(tmp_path, dxl_bus=bus, context_servo_ids=[1])
    service.connect("/dev/mock-openrb", 115200)
    service.save_startup_calibration(
        servo_id=1,
        min_offset_ticks=-10,
        max_offset_ticks=5,
        pretension_current_threshold_ma=220,
    )
    bus._state[1].present_position = 2100

    result = service.jog_servo_action(servo_id=1, action="tighten_fine")

    assert result.success is True
    assert result.blocked is False
    assert result.goal_tick == 2095
    assert result.safe_min_tick == 0
    assert result.safe_max_tick == 4095


def test_servo_service_safe_id_assignment_requires_one_discovered_servo(tmp_path: Path) -> None:
    bus = _RecordingIdBus()
    bus._state[1].torque_enabled = True
    service = _build_service(tmp_path, dxl_bus=bus, context_servo_ids=[1])
    service.connect("/dev/mock-openrb", 115200)

    result = service.assign_servo_id_safely(1, 3)

    assert result.success is True
    assert result.selected_ids == [3]
    assert bus.write_id_calls == [(1, 3, True)]
    assert 3 in bus._state


def test_servo_service_discovery_prefers_expected_servo_id(tmp_path: Path) -> None:
    service = _build_service(tmp_path, dxl_bus=MockDxlBus([7]), context_servo_ids=[7])
    service.connect("/dev/mock-openrb", 115200)

    discovery = service.discover_one_servo(expected_servo_id=7, allow_scan=False)

    assert discovery.status == "expected_id_read_ok"
    assert discovery.selected_servo_id == 7
    assert discovery.bus_reachable is True
    assert discovery.telemetry is not None


def test_servo_service_configured_bringup_snapshot_reports_all_expected_servos(tmp_path: Path) -> None:
    service = _build_service(tmp_path, dxl_bus=MockDxlBus([1, 2, 3, 4]), context_servo_ids=[1, 2, 3, 4])
    service.connect("/dev/mock-openrb", 115200)

    snapshot = service.build_configured_servo_bringup_snapshot([1, 2, 3, 4], allow_scan=True)

    assert snapshot.status == "ready"
    assert snapshot.discovered_ids == [1, 2, 3, 4]
    assert snapshot.missing_servo_ids == []
    assert snapshot.unexpected_servo_ids == []
    assert snapshot.all_expected_telemetry_ok is True
    assert snapshot.all_motion_ready is True
    assert all(entry.telemetry_read_ok for entry in snapshot.servo_entries.values())


def test_servo_service_configured_bringup_snapshot_reports_missing_and_unexpected_ids(tmp_path: Path) -> None:
    service = _build_service(tmp_path, dxl_bus=MockDxlBus([1, 2, 4, 9]), context_servo_ids=[1, 2, 3, 4])
    service.connect("/dev/mock-openrb", 115200)

    snapshot = service.build_configured_servo_bringup_snapshot([1, 2, 3, 4], allow_scan=True)

    assert snapshot.status == "expected_missing"
    assert snapshot.missing_servo_ids == [3]
    assert snapshot.unexpected_servo_ids == [9]
    assert snapshot.servo_entries[3].status == "expected_missing"


def test_servo_service_four_servo_jog_moves_only_selected_servo(tmp_path: Path) -> None:
    bus = MockDxlBus([1, 2, 3, 4])
    service = _build_service(tmp_path, dxl_bus=bus, context_servo_ids=[1, 2, 3, 4])
    service.connect("/dev/mock-openrb", 115200)
    before = {servo_id: telemetry.present_position for servo_id, telemetry in bus._state.items()}

    result = service.jog_servo_action(servo_id=2, action="tighten_fine")

    assert result.success is True
    assert bus._state[2].present_position == before[2] - 5
    assert bus._state[1].present_position == before[1]
    assert bus._state[3].present_position == before[3]
    assert bus._state[4].present_position == before[4]


def test_servo_service_pretension_validation_returns_message(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    service.connect("/dev/mock-openrb", 115200)
    result = service.validate_pretension([1, 2, 3, 4], tolerance_ma=80)

    assert result.spread_ma is not None
    assert "spread" in result.message


def test_servo_service_pretension_stops_on_threshold_and_can_be_accepted(tmp_path: Path) -> None:
    bus = _PretensionBus(current_sequence=[180, 230])
    bus._state[1].torque_enabled = True
    service = _build_service(tmp_path, dxl_bus=bus)
    service.connect("/dev/mock-openrb", 115200)
    service.save_startup_calibration(
        servo_id=1,
        min_offset_ticks=-40,
        max_offset_ticks=40,
        pretension_current_threshold_ma=220,
    )

    result = service.run_pretension_routine(
        servo_id=1,
        parameters=PretensionParameters(
            untensioned_reference_tick=4095,
            step_ticks=2,
            settle_time_s=0.0,
            baseline_sample_count=3,
            current_filter_window=1,
            current_delta_threshold_ma=60,
            absolute_trigger_current_ma=220,
            hard_current_stop_ma=850,
            max_travel_ticks=320,
            timeout_s=2.0,
        ),
    )

    assert result.success is True
    assert result.status == "threshold_reached"
    assert result.steps_taken >= 1
    accepted = service.accept_pretension_result(1)
    assert accepted.pretension_result_status == "accepted"


def test_servo_service_pretension_uses_decreasing_raw_position_for_tightening(tmp_path: Path) -> None:
    bus = _PretensionBus(current_sequence=[180, 230])
    bus.config.positive_tick_rotation = "cw"
    bus._state[1].torque_enabled = True
    service = _build_service(tmp_path, dxl_bus=bus, context_servo_ids=[1])
    service.connect("/dev/mock-openrb", 115200)
    service.save_startup_calibration(
        servo_id=1,
        min_offset_ticks=-40,
        max_offset_ticks=40,
        pretension_current_threshold_ma=220,
    )

    result = service.run_pretension_routine(
        servo_id=1,
        parameters=PretensionParameters(
            untensioned_reference_tick=4095,
            step_ticks=2,
            settle_time_s=0.0,
            baseline_sample_count=3,
            current_filter_window=1,
            current_delta_threshold_ma=60,
            absolute_trigger_current_ma=220,
            hard_current_stop_ma=850,
            max_travel_ticks=320,
            timeout_s=2.0,
        ),
    )

    assert result.success is True
    assert bus._state[1].present_position < 2048


def test_servo_service_pretension_fails_on_overcurrent(tmp_path: Path) -> None:
    bus = _PretensionBus(current_sequence=[900])
    bus._state[1].torque_enabled = True
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
    bus = _PretensionBus(current_sequence=[], max_limit=4095)
    bus._state[1].present_position = 64
    bus._state[1].torque_enabled = True
    service = _build_service(tmp_path, dxl_bus=bus)
    service.connect("/dev/mock-openrb", 115200)

    result = service.run_pretension_routine(servo_id=1)

    assert result.success is False
    assert result.status == "travel_limit"


def test_servo_service_pretension_fails_on_timeout(tmp_path: Path) -> None:
    timeline = iter([0.0] * 20 + [3.0] * 20)
    bus = _PretensionBus(current_sequence=[180, 190])
    bus._state[1].torque_enabled = True
    service = _build_service(
        tmp_path,
        dxl_bus=bus,
        time_fn=lambda: next(timeline),
        telemetry_stale_after_s=10.0,
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
    bus._state[1].torque_enabled = True
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


def test_servo_service_measures_filtered_pretension_baseline(tmp_path: Path) -> None:
    bus = _BaselineSequenceBus(baseline_sequence=[110, 130, 150, 170])
    service = _build_service(tmp_path, dxl_bus=bus, context_servo_ids=[1])
    service.connect("/dev/mock-openrb", 115200)

    baseline = service.measure_pretension_baseline(servo_id=1, sample_count=4, filter_window=2)

    assert baseline.samples_ma == [110, 130, 150, 170]
    assert baseline.baseline_current_ma == pytest.approx(140.0)
    assert baseline.filtered_current_ma == pytest.approx(160.0)


def test_servo_service_pretension_uses_baseline_delta_trigger_for_mvp(tmp_path: Path) -> None:
    bus = _PretensionBus(current_sequence=[180, 205])
    bus._state[1].torque_enabled = True
    service = _build_service(tmp_path, dxl_bus=bus, context_servo_ids=[1])
    service.connect("/dev/mock-openrb", 115200)

    result = service.run_pretension_routine(
        servo_id=1,
        parameters=PretensionParameters(
            untensioned_reference_tick=4095,
            step_ticks=2,
            settle_time_s=0.0,
            baseline_sample_count=3,
            current_filter_window=1,
            current_delta_threshold_ma=60,
            absolute_trigger_current_ma=500,
            hard_current_stop_ma=850,
            max_travel_ticks=320,
            timeout_s=2.0,
        ),
    )

    assert result.success is True
    assert result.status == "threshold_reached"
    assert result.stop_reason == "baseline_delta_trigger"
    summary = service.get_calibration_summary()
    assert summary.servo_entries[1].latest_pretension_run is not None
    assert summary.servo_entries[1].latest_pretension_run["stop_reason"] == "baseline_delta_trigger"


def test_servo_service_pretension_fails_on_stale_telemetry(tmp_path: Path) -> None:
    bus = _StalePretensionBus(current_sequence=[180, 180])
    bus._state[1].last_read_monotonic_s = 0.0
    service = _build_service(
        tmp_path,
        dxl_bus=bus,
        context_servo_ids=[1],
        time_fn=lambda: 0.0,
        telemetry_stale_after_s=0.25,
    )
    service.connect("/dev/mock-openrb", 115200)
    service.save_startup_calibration(
        servo_id=1,
        min_offset_ticks=-40,
        max_offset_ticks=40,
        pretension_current_threshold_ma=220,
    )
    bus._state[1].last_read_monotonic_s = 0.0
    service.safety_guard._time_fn = lambda: 1.0

    with pytest.raises(RuntimeError, match="Telemetry is stale"):
        service.run_pretension_routine(servo_id=1)


def test_servo_service_pretension_fails_on_unsafe_voltage_temperature_and_fault(tmp_path: Path) -> None:
    bus = _PretensionBus(current_sequence=[180])
    service = _build_service(tmp_path, dxl_bus=bus, context_servo_ids=[1], min_input_voltage_mv=6000)
    service.connect("/dev/mock-openrb", 115200)
    service.save_startup_calibration(
        servo_id=1,
        min_offset_ticks=-40,
        max_offset_ticks=40,
        pretension_current_threshold_ma=220,
    )

    bus._state[1].present_voltage_mv = 5000
    with pytest.raises(RuntimeError, match="Input voltage is below the configured motion minimum"):
        service.run_pretension_routine(servo_id=1)

    bus._state[1].present_voltage_mv = 12000
    bus._state[1].present_temperature_c = 71
    with pytest.raises(RuntimeError, match="Temperature threshold exceeded"):
        service.run_pretension_routine(servo_id=1)

    bus._state[1].present_temperature_c = 33
    bus._state[1].hardware_error_code = 8
    with pytest.raises(RuntimeError, match="Hardware Error Status is 0x08"):
        service.run_pretension_routine(servo_id=1)
