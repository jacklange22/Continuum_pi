from __future__ import annotations

import logging
import json
from pathlib import Path
import threading

import pytest

from continuum_robot.hardware.mock_dxl_bus import MockDxlBus
from continuum_robot.servos.displacement_mapper import TendonDisplacementMapper
from continuum_robot.servos.neutral_calibration_service import (
    NeutralCalibrationService,
    ServoCalibrationContext,
)
from continuum_robot.servos.pretension_validation_service import PretensionValidationService
from continuum_robot.servos.safety_guard import SafetyGuard
from continuum_robot.servos.servo_service import (
    PretensionParameters,
    ServoBusBusyError,
    ServoService,
    is_wrap_risk,
)


def test_safety_guard_current_limit_uses_absolute_magnitude_positive_and_negative() -> None:
    guard = SafetyGuard(
        min_offset_ticks=-600,
        max_offset_ticks=600,
        max_current_ma=850,
        servo_model="XC330-M288-T",
        servo_reported_current_hard_limit_ma=850,
        current_safety_basis="unit-test basis",
    )

    guard.validate_currents([849, -849], require_present=True)
    with pytest.raises(ValueError, match="\\|851\\|=851 mA > 850 mA"):
        guard.validate_currents([851], require_present=True)
    with pytest.raises(ValueError, match="\\|-851\\|=851 mA > 850 mA"):
        guard.validate_currents([-851], require_present=True)


def test_safety_guard_missing_current_policy_is_unchanged() -> None:
    guard = SafetyGuard(min_offset_ticks=-600, max_offset_ticks=600, max_current_ma=850)

    guard.validate_currents([None], require_present=False)
    with pytest.raises(ValueError, match="Current telemetry is unavailable"):
        guard.validate_currents([None], require_present=True)


def test_pretension_current_balance_uses_absolute_current_proxy() -> None:
    result = PretensionValidationService().validate_current_balance([-30, 20, -25, 24], tolerance_ma=12)

    assert result.currents_ma == [30, 20, 25, 24]
    assert result.spread_ma == 10
    assert result.passed is True
    assert "Load-proxy balance" in result.message


class _PretensionBus(MockDxlBus):
    def __init__(self, *, current_sequence: list[int | None], max_limit: int = 4095) -> None:
        super().__init__([1])
        self._current_sequence = list(current_sequence)
        self._state[1].max_position_limit = max_limit
        self._state[1].present_position = 4031
        self._state[1].present_current_ma = 150

    def write_goal_positions(self, positions_by_id: dict[int, int]) -> None:
        super().write_goal_positions(positions_by_id)
        if self._current_sequence:
            self._state[1].present_current_ma = self._current_sequence.pop(0)
        self._state[1].last_read_monotonic_s = self._state[1].last_read_monotonic_s


class _TorqueEnableFailurePretensionBus(_PretensionBus):
    def __init__(self, *, current_sequence: list[int | None], fail_message: str = "mock torque enable failure") -> None:
        super().__init__(current_sequence=current_sequence)
        self.fail_message = fail_message
        self.torque_enable_calls: list[tuple[int, bool]] = []

    def write_torque_enable(self, servo_id: int, enabled: bool) -> None:
        self.torque_enable_calls.append((int(servo_id), bool(enabled)))
        raise RuntimeError(self.fail_message)


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
        self._state[1].present_position = 4031

    def read_telemetry(self, servo_ids: list[int], **kwargs) -> dict[int, object]:
        result = super().read_telemetry(servo_ids, **kwargs)
        if self._baseline_sequence:
            current = self._baseline_sequence.pop(0)
            result[1].present_current_ma = current
            self._state[1].present_current_ma = current
        return result


class _SteppingReadDropPretensionBus(_PretensionBus):
    def __init__(self, *, drop_read_count: int, recovered_current_ma: int = 230) -> None:
        super().__init__(current_sequence=[int(recovered_current_ma)])
        self._drop_read_count = max(0, int(drop_read_count))
        self._step_started = False
        self._state[1].torque_enabled = True

    def write_goal_positions(self, positions_by_id: dict[int, int]) -> None:
        self._step_started = True
        super().write_goal_positions(positions_by_id)

    def read_telemetry(self, servo_ids: list[int], **kwargs) -> dict[int, object]:
        result = super().read_telemetry(servo_ids, **kwargs)
        if self._step_started and self._drop_read_count > 0:
            self._drop_read_count -= 1
            result[1].present_current_ma = None
            result[1].present_current_raw_unit = None
        return result


class _GoalWriteFailureBus(MockDxlBus):
    def write_goal_positions(self, positions_by_id: dict[int, int]) -> None:
        raise RuntimeError("mock write timeout")


class _RecordingMotionConfigBus(MockDxlBus):
    def __init__(self, servo_ids: list[int] | None = None) -> None:
        super().__init__(servo_ids or [1, 2, 3, 4])
        self.operating_mode_writes: list[tuple[int, int]] = []
        self.goal_current_writes: list[tuple[int, int]] = []
        self.profile_velocity_writes: list[tuple[int, int]] = []
        self.profile_acceleration_writes: list[tuple[int, int]] = []

    def write_operating_mode(self, servo_id: int, operating_mode: int) -> None:
        self.operating_mode_writes.append((int(servo_id), int(operating_mode)))
        super().write_operating_mode(servo_id, operating_mode)

    def write_goal_current_ma(self, servo_id: int, current_ma: int) -> None:
        self.goal_current_writes.append((int(servo_id), int(current_ma)))
        super().write_goal_current_ma(servo_id, current_ma)

    def write_profile_velocity(self, servo_id: int, profile_velocity: int) -> None:
        self.profile_velocity_writes.append((int(servo_id), int(profile_velocity)))
        super().write_profile_velocity(servo_id, profile_velocity)

    def write_profile_acceleration(self, servo_id: int, profile_acceleration: int) -> None:
        self.profile_acceleration_writes.append((int(servo_id), int(profile_acceleration)))
        super().write_profile_acceleration(servo_id, profile_acceleration)


class _GoalCurrentFailureBus(_RecordingMotionConfigBus):
    def write_goal_current_ma(self, servo_id: int, current_ma: int) -> None:
        raise RuntimeError("[RxPacketError] The data value exceeds the limit value!")


class _DisconnectOrderBus(MockDxlBus):
    def __init__(self, *, fail_servo_id: int | None = None) -> None:
        super().__init__([1, 2])
        self.fail_servo_id = fail_servo_id
        self.events: list[tuple[str, int | None, bool | None]] = []

    def write_torque_enable(self, servo_id: int, enabled: bool) -> None:
        self.events.append(("torque", int(servo_id), bool(enabled)))
        if self.fail_servo_id is not None and int(servo_id) == int(self.fail_servo_id):
            raise RuntimeError("mock torque disable failure")
        super().write_torque_enable(servo_id, enabled)

    def disconnect(self) -> None:
        self.events.append(("disconnect", None, None))
        super().disconnect()


class _TorqueRecordingBus(MockDxlBus):
    def __init__(self, servo_ids: list[int]) -> None:
        super().__init__(servo_ids)
        self.torque_disable_calls: list[int] = []

    def write_torque_enable(self, servo_id: int, enabled: bool) -> None:
        if not enabled:
            self.torque_disable_calls.append(int(servo_id))
        super().write_torque_enable(servo_id, enabled)


class _PretensionGoalWriteCrashBus(_PretensionBus):
    def write_goal_positions(self, positions_by_id: dict[int, int]) -> None:
        raise RuntimeError("mock unexpected write failure")


def _build_service(
    tmp_path: Path,
    *,
    dxl_bus=None,
    time_fn=None,
    context_servo_ids: list[int] | None = None,
    telemetry_stale_after_s: float = 0.25,
    min_input_voltage_mv: int = 4000,
    mock_mode: bool = False,
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
        mock_mode=mock_mode,
        mock_neutral_calibration_path=tmp_path / "data" / "mock_calibration" / "latest_mock_neutral_setpoints.json",
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


def test_mock_neutral_capture_writes_mock_calibration_not_real_config(tmp_path: Path) -> None:
    service = _build_service(tmp_path, mock_mode=True)
    service.connect("/dev/mock-openrb", 115200)

    result = service.capture_and_save_neutral_setpoints([1, 2, 3, 4])

    real_path = tmp_path / "neutral.json"
    mock_path = tmp_path / "data" / "mock_calibration" / "latest_mock_neutral_setpoints.json"
    assert result.artifact_path == str(mock_path)
    assert not real_path.exists()
    payload = json.loads(mock_path.read_text(encoding="utf-8"))
    assert payload["robot"]["mock_mode"] is True
    assert payload["robot"]["calibration_trust"] == "mock"
    assert payload["robot"]["valid_for_hardware_startup"] is False


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


def test_servo_service_single_segment_displacement_uses_hardware_informed_range(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    service.connect("/dev/mock-openrb", 115200)
    neutral = service.capture_neutral_setpoints([1, 2, 3, 4])
    loaded = service.load_neutral_setpoints()

    result = service.command_displacement(
        tendon_displacements_cm=[1.0, 0.0, -1.0, 0.0],
        neutral_ticks=[loaded[1], loaded[2], loaded[3], loaded[4]],
        servo_ids=[1, 2, 3, 4],
    )

    assert result.positions_by_id[1] > neutral[1] + 600
    assert result.positions_by_id[3] < neutral[3] - 600
    assert "hardware-informed bounds" in result.message
    assert "Position Control" in result.message
    assert result.debug_entries_by_id[1].limit_source == "single_segment_hardware_envelope"


def test_servo_service_single_segment_displacement_projects_antagonistic_pairs(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    service.connect("/dev/mock-openrb", 115200)
    neutral = service.capture_neutral_setpoints([1, 2, 3, 4])
    loaded = service.load_neutral_setpoints()

    result = service.command_displacement(
        tendon_displacements_cm=[1.0, 0.0, 0.0, 0.0],
        neutral_ticks=[loaded[1], loaded[2], loaded[3], loaded[4]],
        servo_ids=[1, 2, 3, 4],
    )

    projected_ticks = service.mapper.displacement_cm_to_ticks(0.5)
    assert result.resolved_displacements_cm == pytest.approx([0.5, 0.0, -0.5, 0.0])
    assert result.positions_by_id[1] == neutral[1] + projected_ticks
    assert result.positions_by_id[3] == neutral[3] - projected_ticks


def test_parallel_single_displacement_expands_to_all_8_servo_goals(tmp_path: Path) -> None:
    bus = MockDxlBus([1, 2, 3, 4, 5, 6, 7, 8])
    service = _build_service(tmp_path, dxl_bus=bus, context_servo_ids=[1, 2, 3, 4, 5, 6, 7, 8])
    service.connect("/dev/mock-openrb", 115200)
    neutral = service.capture_neutral_setpoints([1, 2, 3, 4, 5, 6, 7, 8])
    servo_ids = [1, 2, 3, 4, 5, 6, 7, 8]

    result = service.command_displacement(
        tendon_displacements_cm=[0.2, -0.1, -0.2, 0.1],
        neutral_ticks=[neutral[servo_id] for servo_id in servo_ids],
        servo_ids=servo_ids,
        parallel_mirror_pairs={1: 5, 2: 6, 3: 7, 4: 8},
    )

    assert sorted(result.positions_by_id) == servo_ids
    for source, mirror in {1: 5, 2: 6, 3: 7, 4: 8}.items():
        assert result.positions_by_id[source] - neutral[source] == result.positions_by_id[mirror] - neutral[mirror]
    assert result.command_metadata["mirrored_parallel"] is True
    assert result.command_metadata["mirror_pairs"] == {"1": 5, "2": 6, "3": 7, "4": 8}
    assert "Antagonistic-pair projection applied" in result.message
    assert "parallel_single mirrored command expanded" in result.message


def test_parallel_single_displacement_missing_mirrored_servo_fails_clearly(tmp_path: Path) -> None:
    bus = MockDxlBus([1, 2, 3, 4, 6, 7, 8])
    service = _build_service(tmp_path, dxl_bus=bus, context_servo_ids=[1, 2, 3, 4, 6, 7, 8])
    service.connect("/dev/mock-openrb", 115200)
    neutral = service.capture_neutral_setpoints([1, 2, 3, 4, 6, 7, 8])
    servo_ids = [1, 2, 3, 4, 6, 7, 8]

    with pytest.raises(ValueError, match="missing \\[5\\]"):
        service.command_displacement(
            tendon_displacements_cm=[0.2, -0.1, -0.2, 0.1],
            neutral_ticks=[neutral[servo_id] for servo_id in servo_ids],
            servo_ids=servo_ids,
            parallel_mirror_pairs={1: 5, 2: 6, 3: 7, 4: 8},
        )


def test_servo_service_single_segment_displacement_uses_position_mode_for_experiment_motion(tmp_path: Path) -> None:
    bus = _RecordingMotionConfigBus([1, 2, 3, 4])
    for telemetry in bus._state.values():
        telemetry.operating_mode = 5
    service = _build_service(tmp_path, dxl_bus=bus)
    service.connect("/dev/mock-openrb", 115200)
    neutral = service.capture_neutral_setpoints([1, 2, 3, 4])

    result = service.command_displacement(
        tendon_displacements_cm=[1.0, 0.0, -1.0, 0.0],
        neutral_ticks=[neutral[1], neutral[2], neutral[3], neutral[4]],
        servo_ids=[1, 2, 3, 4],
    )

    assert all(bus._state[servo_id].operating_mode == 3 for servo_id in [1, 2, 3, 4])
    assert bus.goal_current_writes == []
    assert bus.profile_velocity_writes == []
    assert bus.profile_acceleration_writes == []
    assert "Applied Position Control mode" in result.message
    assert "experiment motion config: Position Control" in result.message
    assert result.debug_entries_by_id[1].operating_mode == 3
    assert result.debug_entries_by_id[1].preferred_operating_mode == 3
    assert result.debug_entries_by_id[1].goal_current_ma is None
    assert result.debug_entries_by_id[1].profile_velocity is None
    assert result.debug_entries_by_id[1].profile_acceleration is None


def test_servo_service_current_aware_single_segment_motion_can_still_write_goal_current(tmp_path: Path) -> None:
    bus = _RecordingMotionConfigBus([1, 2, 3, 4])
    for telemetry in bus._state.values():
        telemetry.operating_mode = 3
        telemetry.current_limit_ma = 2352
    service = _build_service(tmp_path, dxl_bus=bus)
    service.connect("/dev/mock-openrb", 115200)
    neutral = service.capture_neutral_setpoints([1, 2, 3, 4])

    result = service.command_displacement(
        tendon_displacements_cm=[1.0, 0.0, -1.0, 0.0],
        neutral_ticks=[neutral[1], neutral[2], neutral[3], neutral[4]],
        servo_ids=[1, 2, 3, 4],
        motion_workflow="current_aware_validation",
    )

    assert all(bus._state[servo_id].operating_mode == 5 for servo_id in [1, 2, 3, 4])
    assert all(getattr(bus._state[servo_id], "goal_current_ma", None) == 850 for servo_id in [1, 2, 3, 4])
    assert "current aware validation config: Current-based Position Control" in result.message
    assert result.debug_entries_by_id[1].preferred_operating_mode == 5
    assert result.debug_entries_by_id[1].goal_current_ma == 850


def test_servo_service_single_segment_characterization_reports_pairwise_travel(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    service.connect("/dev/mock-openrb", 115200)
    service.capture_neutral_setpoints([1, 2, 3, 4])

    characterization = service.characterize_single_segment_motion()

    assert characterization.available is True
    assert "pair 1/3" in characterization.message
    assert "pair 2/4" in characterization.message
    assert characterization.pair_limits["1/3"]["positive_cm"] is not None
    assert characterization.pair_limits["1/3"]["positive_cm"] > 1.5
    assert characterization.pair_limits["1/3"]["negative_cm"] > 1.5


def test_servo_service_single_segment_displacement_rejects_hard_raw_bounds(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    service.connect("/dev/mock-openrb", 115200)
    neutral = service.capture_neutral_setpoints([1, 2, 3, 4])

    with pytest.raises(RuntimeError, match="raw hardware range"):
        service.command_displacement(
            tendon_displacements_cm=[3.0, 0.0, -3.0, 0.0],
            neutral_ticks=[neutral[1], neutral[2], neutral[3], neutral[4]],
            servo_ids=[1, 2, 3, 4],
        )


def test_servo_service_single_segment_displacement_rejects_narrow_hardware_soft_limit(tmp_path: Path) -> None:
    bus = MockDxlBus([1, 2, 3, 4])
    bus._state[1].max_position_limit = 2600
    service = _build_service(tmp_path, dxl_bus=bus)
    service.connect("/dev/mock-openrb", 115200)
    neutral = service.capture_neutral_setpoints([1, 2, 3, 4])

    with pytest.raises(RuntimeError, match="hardware-informed single-segment envelope"):
        service.command_displacement(
            tendon_displacements_cm=[1.0, 0.0, -1.0, 0.0],
            neutral_ticks=[neutral[1], neutral[2], neutral[3], neutral[4]],
            servo_ids=[1, 2, 3, 4],
        )


def test_servo_service_single_segment_displacement_blocks_on_overcurrent(tmp_path: Path) -> None:
    bus = MockDxlBus([1, 2, 3, 4])
    bus._state[1].present_current_ma = 900
    service = _build_service(tmp_path, dxl_bus=bus)
    service.connect("/dev/mock-openrb", 115200)
    neutral = service.capture_neutral_setpoints([1, 2, 3, 4])

    with pytest.raises(RuntimeError, match="overcurrent/jam protection"):
        service.command_displacement(
            tendon_displacements_cm=[1.0, 0.0, -1.0, 0.0],
            neutral_ticks=[neutral[1], neutral[2], neutral[3], neutral[4]],
            servo_ids=[1, 2, 3, 4],
        )


def test_servo_service_simple_experiment_motion_does_not_require_current_voltage_or_temperature(tmp_path: Path) -> None:
    bus = MockDxlBus([1, 2, 3, 4])
    for telemetry in bus._state.values():
        telemetry.present_position = 2048
        telemetry.present_current_ma = None
        telemetry.present_voltage_mv = None
        telemetry.present_temperature_c = None
    service = _build_service(tmp_path, dxl_bus=bus)
    service.connect("/dev/mock-openrb", 115200)

    result = service.command_displacement(
        tendon_displacements_cm=[1.0, 0.0, -1.0, 0.0],
        neutral_ticks=[2048, 2048, 2048, 2048],
        servo_ids=[1, 2, 3, 4],
    )

    assert result.positions_by_id[1] > 2048
    assert result.positions_by_id[3] < 2048
    assert "simple Position-control path" in result.message
    assert result.debug_entries_by_id[1].present_current_ma is None


def test_servo_service_simple_experiment_motion_is_not_limited_by_startup_artifact_bounds(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    service.connect("/dev/mock-openrb", 115200)
    for servo_id in [1, 2, 3, 4]:
        service.set_servo_torque_enabled(servo_id, True)
        service.save_startup_calibration(
            servo_id=servo_id,
            neutral_setpoint=3600,
            min_offset_ticks=-200,
            max_offset_ticks=200,
            pretension_current_threshold_ma=220,
        )
        service.dxl_bus._state[servo_id].present_position = 3010 + servo_id
    service.capture_manual_pretension_state(note="manual startup")
    service.accept_manual_pretension_state()

    resolution = service.resolve_startup_reference_ticks([1, 2, 3, 4])
    result = service.command_displacement(
        tendon_displacements_cm=[0.5, 0.0, -0.5, 0.0],
        neutral_ticks=[
            resolution.ticks_by_servo[1],
            resolution.ticks_by_servo[2],
            resolution.ticks_by_servo[3],
            resolution.ticks_by_servo[4],
        ],
        servo_ids=[1, 2, 3, 4],
    )

    expected_delta = service.mapper.displacement_cm_to_ticks(0.5)
    assert result.positions_by_id[1] == resolution.ticks_by_servo[1] + expected_delta
    assert result.positions_by_id[3] == resolution.ticks_by_servo[3] - expected_delta
    assert result.debug_entries_by_id[1].safe_min_tick == 64
    assert result.debug_entries_by_id[1].safe_max_tick == 4031
    assert result.debug_entries_by_id[1].limit_source == "single_segment_hardware_envelope"


def test_servo_service_simple_experiment_motion_reports_stale_telemetry_explicitly(tmp_path: Path) -> None:
    bus = _StaleTelemetryBus([1, 2, 3, 4])
    for telemetry in bus._state.values():
        telemetry.present_position = 2048
        telemetry.last_read_monotonic_s = 0.0
    service = _build_service(
        tmp_path,
        dxl_bus=bus,
        time_fn=lambda: 1.0,
        telemetry_stale_after_s=0.25,
    )
    service.connect("/dev/mock-openrb", 115200)
    service.safety_guard._time_fn = lambda: 1.0

    with pytest.raises(RuntimeError, match="stale/missing telemetry"):
        service.command_displacement(
            tendon_displacements_cm=[1.0, 0.0, -1.0, 0.0],
            neutral_ticks=[2048, 2048, 2048, 2048],
            servo_ids=[1, 2, 3, 4],
        )


def test_servo_service_simple_experiment_motion_reports_position_mode_issue_when_auto_config_is_off(tmp_path: Path) -> None:
    bus = _RecordingMotionConfigBus([1, 2, 3, 4])
    bus.config.single_segment_auto_configure_motion_defaults = False
    for telemetry in bus._state.values():
        telemetry.operating_mode = 5
        telemetry.present_position = 2048
    service = _build_service(tmp_path, dxl_bus=bus)
    service.connect("/dev/mock-openrb", 115200)

    with pytest.raises(RuntimeError, match="is not in Position Control Mode"):
        service.command_displacement(
            tendon_displacements_cm=[1.0, 0.0, -1.0, 0.0],
            neutral_ticks=[2048, 2048, 2048, 2048],
            servo_ids=[1, 2, 3, 4],
        )

    assert bus.operating_mode_writes == []


def test_servo_service_simple_experiment_motion_emits_diagnostic_logs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    service = _build_service(tmp_path)
    service.connect("/dev/mock-openrb", 115200)
    neutral = service.capture_neutral_setpoints([1, 2, 3, 4])
    caplog.set_level(logging.INFO)

    service.command_displacement(
        tendon_displacements_cm=[0.4, 0.0, -0.4, 0.0],
        neutral_ticks=[neutral[1], neutral[2], neutral[3], neutral[4]],
        servo_ids=[1, 2, 3, 4],
        motion_workflow="experiment_motion",
    )

    joined = "\n".join(record.getMessage() for record in caplog.records)
    assert "Simple experiment motion start" in joined
    assert "Simple experiment motion success" in joined
    assert "workflow=experiment_motion" in joined
    assert "raw_goals" in joined
    assert "telemetry" in joined


def test_servo_service_single_segment_displacement_reports_goal_write_failures_separately(tmp_path: Path) -> None:
    service = _build_service(tmp_path, dxl_bus=_GoalWriteFailureBus([1, 2, 3, 4]))
    service.connect("/dev/mock-openrb", 115200)
    neutral = service.capture_neutral_setpoints([1, 2, 3, 4])

    with pytest.raises(RuntimeError, match="communication failure during write goal positions for simple experiment motion: mock write timeout"):
        service.command_displacement(
            tendon_displacements_cm=[1.0, 0.0, -1.0, 0.0],
            neutral_ticks=[neutral[1], neutral[2], neutral[3], neutral[4]],
            servo_ids=[1, 2, 3, 4],
        )


def test_servo_service_current_aware_profile_reports_goal_current_failure_explicitly(tmp_path: Path) -> None:
    service = _build_service(tmp_path, dxl_bus=_GoalCurrentFailureBus([1, 2, 3, 4]))
    service.connect("/dev/mock-openrb", 115200)
    neutral = service.capture_neutral_setpoints([1, 2, 3, 4])

    with pytest.raises(RuntimeError, match="Failed to write goal current for servo 1"):
        service.command_displacement(
            tendon_displacements_cm=[1.0, 0.0, -1.0, 0.0],
            neutral_ticks=[neutral[1], neutral[2], neutral[3], neutral[4]],
            servo_ids=[1, 2, 3, 4],
            motion_workflow="current_aware_validation",
        )


def test_servo_service_build_runtime_snapshot_reports_consistent_counts(tmp_path: Path) -> None:
    bus = MockDxlBus([1, 2, 3, 4])
    for telemetry in bus._state.values():
        telemetry.torque_enabled = True
        telemetry.present_position = 4031
        telemetry.present_current_ma = 160
    service = _build_service(tmp_path, dxl_bus=bus)
    service.connect("/dev/mock-openrb", 115200)

    snapshot = service.build_runtime_servo_snapshot([1, 2, 3, 4], selected_servo_id=2)

    assert snapshot.connected is True
    assert snapshot.detected_servo_ids == [1, 2, 3, 4]
    assert snapshot.telemetry_ready_count == 4
    assert snapshot.motion_ready_count == 4
    assert snapshot.pretension_ready_count == 4
    assert snapshot.all_motion_ready is True
    assert snapshot.entries[2].telemetry_status == "Live"
    assert snapshot.entries[2].pretension_assessment is not None
    assert snapshot.entries[2].pretension_assessment.ready is True


def test_servo_service_blocks_non_owner_bus_reads_during_exclusive_pretension(tmp_path: Path) -> None:
    service = _build_service(tmp_path, dxl_bus=MockDxlBus([1]), context_servo_ids=[1])
    service.connect("/dev/mock-openrb", 115200)
    owner_ready = threading.Event()
    release_owner = threading.Event()

    def _owner() -> None:
        with service.exclusive_bus_operation(
            owner="pretension run",
            servo_id=1,
            reason="selected-servo pretension",
        ):
            owner_ready.set()
            release_owner.wait(timeout=1.0)

    thread = threading.Thread(target=_owner, daemon=True)
    thread.start()
    assert owner_ready.wait(timeout=1.0)
    try:
        with pytest.raises(ServoBusBusyError, match="owned by active pretension run on servo 1"):
            service.read_telemetry([1])
    finally:
        release_owner.set()
        thread.join(timeout=1.0)


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
    bus._state[1].operating_mode = 1

    with pytest.raises(RuntimeError, match="Operating Mode 1 is not allowed"):
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


def test_servo_service_move_servo_to_raw_target_refreshes_limits_when_live_snapshot_is_partial(tmp_path: Path) -> None:
    bus = MockDxlBus([1])
    bus._state[1].torque_enabled = True
    service = _build_service(tmp_path, dxl_bus=bus, context_servo_ids=[1])
    service.connect("/dev/mock-openrb", 115200)

    result = service.move_servo_to_raw_target(
        servo_id=1,
        target_tick=2055,
        reason="pretension_validation",
    )

    assert result.success is True
    assert result.blocked is False
    assert result.goal_tick == 2055
    assert result.safe_min_tick is not None
    assert result.safe_max_tick is not None


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
    assert accepted.pretension_source == "algorithmic"


def test_servo_service_can_capture_and_accept_manual_pretension_for_single_segment_set(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    service.connect("/dev/mock-openrb", 115200)
    for servo_id in [1, 2, 3, 4]:
        service.set_servo_torque_enabled(servo_id, True)
        service.save_startup_calibration(
            servo_id=servo_id,
            min_offset_ticks=-100,
            max_offset_ticks=120,
            pretension_current_threshold_ma=220,
        )
        service.dxl_bus._state[servo_id].present_position = 2020 - servo_id
        service.dxl_bus._state[servo_id].present_current_ma = 225 + servo_id

    saved = service.capture_manual_pretension_state(note="bench manual state")
    pending_summary = service.get_calibration_summary()
    accepted = service.accept_manual_pretension_state()

    assert sorted(saved) == [1, 2, 3, 4]
    assert pending_summary.servo_entries[1].pretension_result_status == "manual_captured"
    assert pending_summary.servo_entries[1].pretension_source == "manual"
    assert accepted.accepted is True
    assert accepted.usable is True
    assert accepted.source_type == "manual"
    assert accepted.note == "bench manual state"


def test_servo_service_resolve_startup_reference_ticks_prefers_accepted_manual_pretension(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    service.connect("/dev/mock-openrb", 115200)
    for servo_id in [1, 2, 3, 4]:
        service.set_servo_torque_enabled(servo_id, True)
        service.save_startup_calibration(
            servo_id=servo_id,
            min_offset_ticks=-100,
            max_offset_ticks=120,
            pretension_current_threshold_ma=220,
        )
        service.dxl_bus._state[servo_id].present_position = 3010 + servo_id
        service.dxl_bus._state[servo_id].present_current_ma = 220 + servo_id

    service.capture_manual_pretension_state(note="startup reference")
    pending = service.resolve_startup_reference_ticks([1, 2, 3, 4])
    assert pending.source == "neutral"

    accepted = service.accept_manual_pretension_state()
    resolution = service.resolve_startup_reference_ticks([1, 2, 3, 4])

    assert accepted.source_type == "manual"
    assert resolution.source == "manual"
    assert resolution.ticks_by_servo == {1: 3011, 2: 3012, 3: 3013, 4: 3014}
    assert "accepted manual pretension/startup reference positions" in resolution.message


def test_servo_service_pretension_source_summary_distinguishes_algorithmic_from_manual(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    service.connect("/dev/mock-openrb", 115200)
    for servo_id in [1, 2, 3, 4]:
        service.set_servo_torque_enabled(servo_id, True)
        service.save_startup_calibration(
            servo_id=servo_id,
            min_offset_ticks=-100,
            max_offset_ticks=120,
            pretension_current_threshold_ma=220,
        )

    for servo_id in [1, 2, 3, 4]:
        service.neutral_calibration.save_pretension_result(
            servo_id=servo_id,
            final_position_tick=2020,
            final_current_ma=230,
            threshold_ma=220,
            result_status="completed",
            pretension_source="algorithmic",
        )
        service.neutral_calibration.mark_pretension_accepted(servo_id)

    algorithmic = service.pretension_source_summary([1, 2, 3, 4])
    assert algorithmic.source_type == "algorithmic"
    assert algorithmic.usable is True

    service.capture_manual_pretension_state(note="override")
    pending = service.get_calibration_summary()
    assert pending.servo_entries[1].pretension_source == "manual"
    assert pending.servo_entries[1].pretension_result_status == "manual_captured"


def test_servo_service_pretension_readiness_allows_safe_torque_arming(tmp_path: Path) -> None:
    bus = _PretensionBus(current_sequence=[180, 230])
    bus._state[1].torque_enabled = False
    service = _build_service(tmp_path, dxl_bus=bus, context_servo_ids=[1])
    service.connect("/dev/mock-openrb", 115200)
    service.save_startup_calibration(
        servo_id=1,
        min_offset_ticks=-40,
        max_offset_ticks=40,
        pretension_current_threshold_ma=220,
    )

    assessment = service.assess_pretension_readiness(servo_id=1)

    assert assessment.ready is True
    assert assessment.torque_arm_required is True
    assert "Torque will be enabled during arming" in assessment.reason


def test_servo_service_pretension_auto_enables_torque_when_safe(tmp_path: Path) -> None:
    bus = _PretensionBus(current_sequence=[180, 230])
    bus._state[1].torque_enabled = False
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
    assert result.status == "threshold_reached"
    assert bus._state[1].torque_enabled is True
    assert result.torque_cleanup_action == "left_on_after_success"
    assert result.torque_cleanup_success is True


def test_servo_service_pretension_blocks_when_torque_enable_fails(tmp_path: Path) -> None:
    bus = _TorqueEnableFailurePretensionBus(current_sequence=[180, 230])
    bus._state[1].torque_enabled = False
    service = _build_service(tmp_path, dxl_bus=bus, context_servo_ids=[1])
    service.connect("/dev/mock-openrb", 115200)
    service.save_startup_calibration(
        servo_id=1,
        min_offset_ticks=-40,
        max_offset_ticks=40,
        pretension_current_threshold_ma=220,
    )

    result = service.run_pretension_routine(servo_id=1)

    assert result.success is False
    assert result.status == "arming_failed"
    assert result.failure_phase == "arming"
    assert result.primary_reason == "Failed to enable torque for pretension."
    assert "mock torque enable failure" in (result.detail_reason or "")
    assert bus.torque_enable_calls == [(1, True), (1, False)]


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
    assert bus._state[1].present_position < 4031


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
    assert result.torque_cleanup_action == "disarm_after_terminal_state"
    assert result.torque_cleanup_attempted is True
    assert result.torque_cleanup_success is True
    assert bus._state[1].torque_enabled is False


def test_servo_service_pretension_fails_on_travel_limit(tmp_path: Path) -> None:
    bus = _PretensionBus(current_sequence=[], max_limit=4095)
    bus._state[1].torque_enabled = True
    service = _build_service(tmp_path, dxl_bus=bus)
    service.connect("/dev/mock-openrb", 115200)

    result = service.run_pretension_routine(servo_id=1)

    assert result.success is False
    assert result.status == "travel_limit"


def test_servo_service_pretension_current_position_start_mode_does_not_block_on_legacy_window(tmp_path: Path) -> None:
    bus = _PretensionBus(current_sequence=[180, 230])
    bus._state[1].present_position = 64
    bus._state[1].torque_enabled = True
    service = _build_service(tmp_path, dxl_bus=bus)
    service.connect("/dev/mock-openrb", 115200)

    result = service.run_pretension_routine(
        servo_id=1,
        parameters=PretensionParameters(
            untensioned_reference_tick=4095,
            step_ticks=2,
            settle_time_s=0.0,
            baseline_sample_count=1,
            current_filter_window=1,
            current_delta_threshold_ma=40,
            absolute_trigger_current_ma=220,
            hard_current_stop_ma=850,
            max_travel_ticks=320,
            timeout_s=2.0,
            start_mode="current_position",
        ),
    )

    assert result.success is True
    assert result.status == "threshold_reached"
    assert result.failure_phase is None


def test_servo_service_pretension_full_release_start_mode_uses_4095_reference(tmp_path: Path) -> None:
    bus = _PretensionBus(current_sequence=[])
    bus._state[1].present_position = 4031
    bus._state[1].torque_enabled = True
    service = _build_service(tmp_path, dxl_bus=bus)
    service.connect("/dev/mock-openrb", 115200)

    window = service.pretension_window_for_servo(
        servo_id=1,
        parameters=PretensionParameters(
            untensioned_reference_tick=4031,
            step_ticks=2,
            settle_time_s=0.0,
            baseline_sample_count=1,
            current_filter_window=1,
            current_delta_threshold_ma=60,
            absolute_trigger_current_ma=220,
            hard_current_stop_ma=850,
            max_travel_ticks=320,
            timeout_s=2.0,
            start_mode="full_release_4095",
        ),
    )

    assert window.untensioned_reference_tick == 4095
    assert window.start_mode == "full_release_4095"


def test_servo_service_pretension_release_200_start_mode_uses_current_bias(tmp_path: Path) -> None:
    bus = _PretensionBus(current_sequence=[])
    bus._state[1].present_position = 3000
    bus._state[1].torque_enabled = True
    service = _build_service(tmp_path, dxl_bus=bus)
    service.connect("/dev/mock-openrb", 115200)

    window = service.pretension_window_for_servo(
        servo_id=1,
        parameters=PretensionParameters(
            untensioned_reference_tick=4095,
            step_ticks=2,
            settle_time_s=0.0,
            baseline_sample_count=1,
            current_filter_window=1,
            current_delta_threshold_ma=60,
            absolute_trigger_current_ma=220,
            hard_current_stop_ma=850,
            max_travel_ticks=320,
            timeout_s=2.0,
            start_mode="release_200_from_current",
        ),
    )

    assert window.untensioned_reference_tick == 3200
    assert window.effective_min_target_tick == 2880
    assert window.start_mode == "release_200_from_current"
    assert "200 tick release" in str(window.start_mode_detail)


def test_servo_service_pretension_release_200_start_mode_respects_hard_position_limit(tmp_path: Path) -> None:
    bus = _PretensionBus(current_sequence=[], max_limit=4050)
    bus._state[1].present_position = 4000
    bus._state[1].torque_enabled = True
    service = _build_service(tmp_path, dxl_bus=bus)
    service.connect("/dev/mock-openrb", 115200)

    window = service.pretension_window_for_servo(
        servo_id=1,
        parameters=PretensionParameters(
            untensioned_reference_tick=4095,
            step_ticks=2,
            settle_time_s=0.0,
            baseline_sample_count=1,
            current_filter_window=1,
            current_delta_threshold_ma=60,
            absolute_trigger_current_ma=220,
            hard_current_stop_ma=850,
            max_travel_ticks=320,
            timeout_s=2.0,
            start_mode="release_200_from_current",
        ),
    )

    assert window.untensioned_reference_tick == 4050
    assert window.hardware_safe_max_tick == 4050


def test_servo_service_pretension_manual_artifact_start_mode_uses_saved_manual_reference(tmp_path: Path) -> None:
    bus = _PretensionBus(current_sequence=[230])
    bus._state[1].present_position = 4010
    bus._state[1].torque_enabled = True
    service = _build_service(tmp_path, dxl_bus=bus)
    service.connect("/dev/mock-openrb", 115200)
    service.save_startup_calibration(
        servo_id=1,
        min_offset_ticks=-40,
        max_offset_ticks=40,
        pretension_current_threshold_ma=220,
    )
    service.neutral_calibration.save_pretension_result(
        servo_id=1,
        final_position_tick=3990,
        final_current_ma=180,
        threshold_ma=220,
        result_status="accepted",
        pretension_source="manual",
    )

    window = service.pretension_window_for_servo(
        servo_id=1,
        parameters=PretensionParameters(
            untensioned_reference_tick=4095,
            step_ticks=2,
            settle_time_s=0.0,
            baseline_sample_count=1,
            current_filter_window=1,
            current_delta_threshold_ma=40,
            absolute_trigger_current_ma=220,
            hard_current_stop_ma=850,
            max_travel_ticks=320,
            timeout_s=2.0,
            start_mode="manual_startup_artifact",
        ),
    )

    assert window.untensioned_reference_tick == 3990
    assert window.start_mode == "manual_startup_artifact"


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
    assert result.torque_cleanup_action == "disarm_after_terminal_state"
    assert result.torque_cleanup_success is True
    assert bus._state[1].torque_enabled is False


def test_wrap_risk_helper_blocks_raw_discontinuity_commands() -> None:
    assert is_wrap_risk(50, 3500, (0, 4095)) is True
    assert is_wrap_risk(100, 4095, (0, 4095)) is True
    assert is_wrap_risk(4050, 3500, (0, 4095)) is False
    assert is_wrap_risk(3500, 3480, (0, 4095)) is False


def test_move_to_raw_target_blocks_wrap_risk(tmp_path: Path) -> None:
    bus = MockDxlBus([1])
    bus._state[1].present_position = 50
    service = _build_service(tmp_path, dxl_bus=bus, context_servo_ids=[1])
    service.connect("/dev/mock-openrb", 115200)

    result = service.move_servo_to_raw_target(servo_id=1, target_tick=3500)

    assert result.success is False
    assert result.blocked is True
    assert "Target crosses raw tick discontinuity".lower() in result.message.lower()
    assert bus._state[1].present_position == 50


def test_servo_service_disconnect_preserves_torque_by_default(tmp_path: Path) -> None:
    bus = _DisconnectOrderBus(fail_servo_id=1)
    service = _build_service(tmp_path, dxl_bus=bus, context_servo_ids=[1, 2])
    service.connect("/dev/mock-openrb", 115200)

    service.disconnect()

    assert bus.events == [("disconnect", None, None)]


def test_servo_service_explicit_disconnect_torque_off_disarms_before_bus_close(tmp_path: Path) -> None:
    bus = _DisconnectOrderBus(fail_servo_id=1)
    service = _build_service(tmp_path, dxl_bus=bus, context_servo_ids=[1, 2])
    service.connect("/dev/mock-openrb", 115200)

    service.disconnect(torque_off=True, requested_by_operator=True)

    assert bus.events[0] == ("torque", 1, False)
    assert bus.events[1] == ("torque", 2, False)
    assert bus.events[-1][0] == "disconnect"


def test_explicit_torque_off_targets_active_commanded_scope(tmp_path: Path) -> None:
    bus = _TorqueRecordingBus([1, 2, 3, 4, 5, 6, 7, 8])
    service = _build_service(tmp_path, dxl_bus=bus, context_servo_ids=[1, 2, 3, 4, 5, 6, 7, 8])
    service.neutral_calibration.context.expected_servo_ids = [5, 6, 7, 8]
    service.neutral_calibration.context.commanded_servo_ids = [5, 6, 7, 8]
    service._last_goal_positions_by_id = {1: 2048, 5: 2048}
    service.connect("/dev/mock-openrb", 115200)

    service.disconnect(torque_off=True, requested_by_operator=True)

    assert bus.torque_disable_calls == [5, 6, 7, 8]


def test_servo_service_pretension_cancel_disarms_servo_when_routine_armed_torque(tmp_path: Path) -> None:
    bus = _PretensionBus(current_sequence=[180, 190])
    bus._state[1].torque_enabled = False
    service = _build_service(tmp_path, dxl_bus=bus, context_servo_ids=[1])
    service.connect("/dev/mock-openrb", 115200)
    service.save_startup_calibration(
        servo_id=1,
        min_offset_ticks=-40,
        max_offset_ticks=40,
        pretension_current_threshold_ma=220,
    )

    result = service.run_pretension_routine(servo_id=1, stop_requested=lambda: True)

    assert result.success is False
    assert result.status == "canceled"
    assert result.torque_cleanup_action == "disarm_after_terminal_state"
    assert result.torque_cleanup_success is True
    assert bus._state[1].torque_enabled is False


def test_servo_service_pretension_timeout_disarms_servo_when_routine_armed_torque(tmp_path: Path) -> None:
    timeline = iter([0.0] * 20 + [3.0] * 20)
    bus = _PretensionBus(current_sequence=[180, 190])
    bus._state[1].torque_enabled = False
    service = _build_service(
        tmp_path,
        dxl_bus=bus,
        context_servo_ids=[1],
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
    assert result.torque_cleanup_action == "disarm_after_terminal_state"
    assert result.torque_cleanup_success is True
    assert bus._state[1].torque_enabled is False


def test_servo_service_pretension_cancel_disarms_servo_when_torque_was_already_on(tmp_path: Path) -> None:
    bus = _PretensionBus(current_sequence=[180, 190])
    bus._state[1].torque_enabled = True
    service = _build_service(tmp_path, dxl_bus=bus, context_servo_ids=[1])
    service.connect("/dev/mock-openrb", 115200)
    service.save_startup_calibration(
        servo_id=1,
        min_offset_ticks=-40,
        max_offset_ticks=40,
        pretension_current_threshold_ma=220,
    )

    result = service.run_pretension_routine(servo_id=1, stop_requested=lambda: True)

    assert result.success is False
    assert result.status == "canceled"
    assert result.torque_cleanup_action == "disarm_after_terminal_state"
    assert result.torque_cleanup_success is True
    assert bus._state[1].torque_enabled is False


def test_servo_service_pretension_unexpected_exception_still_disarms_and_reports_cleanup(tmp_path: Path) -> None:
    bus = _PretensionGoalWriteCrashBus(current_sequence=[180])
    bus._state[1].torque_enabled = True
    service = _build_service(tmp_path, dxl_bus=bus, context_servo_ids=[1])
    service.connect("/dev/mock-openrb", 115200)
    service.save_startup_calibration(
        servo_id=1,
        min_offset_ticks=-40,
        max_offset_ticks=40,
        pretension_current_threshold_ma=220,
    )

    result = service.run_pretension_routine(servo_id=1)

    assert result.success is False
    assert result.status == "exception"
    assert result.torque_cleanup_action == "disarm_after_terminal_state"
    assert result.torque_cleanup_attempted is True
    assert result.torque_cleanup_success is True
    assert bus._state[1].torque_enabled is False


def test_servo_service_jog_behavior_still_uses_auto_torque_on_goal_write(tmp_path: Path) -> None:
    bus = MockDxlBus([1])
    bus._state[1].torque_enabled = False
    service = _build_service(tmp_path, dxl_bus=bus, context_servo_ids=[1])
    service.connect("/dev/mock-openrb", 115200)
    bus._state[1].present_position = 2048

    result = service.jog_servo_action(servo_id=1, action="tighten_fine")

    assert result.success is True
    assert bus._state[1].torque_enabled is True


def test_servo_service_pretension_fails_when_current_disappears(tmp_path: Path) -> None:
    bus = _PretensionBus(current_sequence=[None])
    bus._state[1].torque_enabled = True
    bus._state[1].telemetry_error = "[TxRxResult] Incorrect status packet!"
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
    assert result.failure_phase == "stepping"
    assert result.primary_reason == "The DYNAMIXEL bus returned an incorrect status packet."
    assert result.stop_reason == "incorrect_status_packet"
    assert "Incorrect status packet" in (result.detail_reason or "")
    assert result.torque_cleanup_action == "disarm_after_terminal_state"
    assert result.torque_cleanup_attempted is True
    assert result.torque_cleanup_success is True
    assert bus._state[1].torque_enabled is False


def test_servo_service_pretension_classifies_no_status_packet(tmp_path: Path) -> None:
    bus = _PretensionBus(current_sequence=[None])
    bus._state[1].torque_enabled = True
    bus._state[1].telemetry_error = "[TxRxResult] There is no status packet!"
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
    assert result.stop_reason == "no_status_packet"
    assert result.primary_reason == "The DYNAMIXEL bus did not return a status packet."


def test_servo_service_pretension_tolerates_one_transient_missing_current_sample(tmp_path: Path) -> None:
    bus = _SteppingReadDropPretensionBus(drop_read_count=1, recovered_current_ma=230)
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
            baseline_sample_count=1,
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


def test_servo_service_pretension_fails_after_transient_missing_current_budget_is_exhausted(tmp_path: Path) -> None:
    bus = _SteppingReadDropPretensionBus(drop_read_count=4, recovered_current_ma=230)
    bus._state[1].telemetry_error = "[TxRxResult] Incorrect status packet!"
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
            baseline_sample_count=1,
            current_filter_window=1,
            current_delta_threshold_ma=60,
            absolute_trigger_current_ma=500,
            hard_current_stop_ma=850,
            max_travel_ticks=320,
            timeout_s=2.0,
        ),
    )

    assert result.success is False
    assert result.status == "invalid_telemetry"
    assert result.stop_reason == "missing_current"
    assert result.primary_reason == "Current telemetry is unavailable."


def test_servo_service_measures_filtered_pretension_baseline(tmp_path: Path) -> None:
    bus = _BaselineSequenceBus(baseline_sequence=[110, 130, 150, 170])
    service = _build_service(tmp_path, dxl_bus=bus, context_servo_ids=[1])
    service.connect("/dev/mock-openrb", 115200)

    baseline = service.measure_pretension_baseline(servo_id=1, sample_count=4, filter_window=2)

    assert baseline.samples_ma == [110, 130, 150, 170]
    assert baseline.baseline_current_ma == pytest.approx(140.0)
    assert baseline.filtered_current_ma == pytest.approx(160.0)


def test_servo_service_pretension_uses_baseline_delta_trigger_for_mvp(tmp_path: Path) -> None:
    bus = _PretensionBus(current_sequence=[180, 215])
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

    result = service.run_pretension_routine(servo_id=1)

    assert result.success is False
    assert result.status == "arming_failed"
    assert result.failure_phase == "arming"
    assert result.primary_reason == "Telemetry is stale."


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
    result = service.run_pretension_routine(servo_id=1)
    assert result.success is False
    assert result.status == "arming_failed"
    assert result.primary_reason == "Input voltage is unsafe."

    bus._state[1].present_voltage_mv = 12000
    bus._state[1].present_temperature_c = 71
    result = service.run_pretension_routine(servo_id=1)
    assert result.success is False
    assert result.status == "arming_failed"
    assert result.primary_reason == "Temperature is unsafe."

    bus._state[1].present_temperature_c = 33
    bus._state[1].hardware_error_code = 8
    result = service.run_pretension_routine(servo_id=1)
    assert result.success is False
    assert result.status == "arming_failed"
    assert result.primary_reason == "Servo hardware error is active."
