"""Mock DYNAMIXEL bus for non-hardware test and GUI development."""

from __future__ import annotations

from datetime import datetime, timezone
import time

from continuum_robot.hardware.dxl_bus import DxlBus, ServoPingResult, ServoTelemetry


class MockDxlBus(DxlBus):
    """Stateful mock implementation of low-level bus operations."""

    def __init__(self, servo_ids: list[int] | None = None) -> None:
        super().__init__()
        ids = servo_ids or [1, 2, 3, 4]
        self._state: dict[int, ServoTelemetry] = {
            sid: ServoTelemetry(
                servo_id=sid,
                reported_servo_id=sid,
                model_number=1240,
                firmware_version=53,
                operating_mode=3,
                torque_enabled=False,
                current_limit_ma=2352,
                min_position_limit=0,
                max_position_limit=4095,
                bus_watchdog_value=0,
                present_position=2048 + 25 * idx,
                present_current_raw_unit=140 + 10 * idx,
                present_current_ma=140 + 10 * idx,
                present_voltage_raw_unit=120,
                present_voltage_mv=12000,
                present_temperature_c=33 + idx,
                hardware_error_code=0,
                hardware_error=None,
            )
            for idx, sid in enumerate(ids)
        }

    @property
    def is_connected(self) -> bool:
        return self._port is not None

    def connect(self, port: str, baudrate: int) -> None:
        if not port:
            raise RuntimeError("Mock DYNAMIXEL port is empty. Pick a mock or real port first.")
        self._port = port
        self._baudrate = baudrate

    def scan_ids(self, min_id: int = 1, max_id: int = 20) -> list[int]:
        return [sid for sid in sorted(self._state) if min_id <= sid <= max_id]

    def ping_servo(self, servo_id: int) -> bool:
        return int(servo_id) in self._state

    def ping_servo_snapshot(self, servo_id: int) -> ServoPingResult:
        telemetry = self._state.get(int(servo_id))
        if telemetry is None:
            return ServoPingResult(
                servo_id=int(servo_id),
                responded=False,
                error=f"Servo {servo_id} did not respond on the mock bus.",
            )
        return ServoPingResult(
            servo_id=int(servo_id),
            responded=True,
            model_number=telemetry.model_number,
            error=None,
        )

    def write_goal_positions(self, positions_by_id: dict[int, int]) -> None:
        for servo_id, goal in positions_by_id.items():
            telemetry = self._state.setdefault(servo_id, ServoTelemetry(servo_id=servo_id))
            current = 120 + min(600, abs(goal - (telemetry.present_position or goal)) // 4)
            telemetry.torque_enabled = True
            telemetry.present_position = int(goal)
            telemetry.present_current_ma = current
            telemetry.present_current_raw_unit = int(current)
            telemetry.present_voltage_mv = 12000 - min(500, current // 10)
            telemetry.present_voltage_raw_unit = int(round(float(telemetry.present_voltage_mv) / 100.0))
            telemetry.present_temperature_c = min(65, 30 + current // 20)
            telemetry.last_read_monotonic_s = time.monotonic()

    def write_operating_mode(self, servo_id: int, operating_mode: int) -> None:
        telemetry = self._state.setdefault(int(servo_id), ServoTelemetry(servo_id=int(servo_id)))
        telemetry.operating_mode = int(operating_mode)
        telemetry.last_read_monotonic_s = time.monotonic()

    def write_goal_current_ma(self, servo_id: int, current_ma: int) -> None:
        telemetry = self._state.setdefault(int(servo_id), ServoTelemetry(servo_id=int(servo_id)))
        telemetry.goal_current_ma = int(current_ma)
        telemetry.last_read_monotonic_s = time.monotonic()

    def write_profile_velocity(self, servo_id: int, profile_velocity: int) -> None:
        telemetry = self._state.setdefault(int(servo_id), ServoTelemetry(servo_id=int(servo_id)))
        telemetry.profile_velocity = int(profile_velocity)
        telemetry.last_read_monotonic_s = time.monotonic()

    def write_profile_acceleration(self, servo_id: int, profile_acceleration: int) -> None:
        telemetry = self._state.setdefault(int(servo_id), ServoTelemetry(servo_id=int(servo_id)))
        telemetry.profile_acceleration = int(profile_acceleration)
        telemetry.last_read_monotonic_s = time.monotonic()

    def write_torque_enable(self, servo_id: int, enabled: bool) -> None:
        telemetry = self._state.setdefault(int(servo_id), ServoTelemetry(servo_id=int(servo_id)))
        telemetry.torque_enabled = bool(enabled)
        telemetry.last_read_monotonic_s = time.monotonic()

    def write_servo_id(self, current_id: int, new_id: int) -> None:
        if current_id not in self._state:
            raise ValueError(f"Servo {current_id} not found")
        if new_id in self._state:
            raise ValueError(f"Servo {new_id} already exists")
        if self._state[current_id].torque_enabled:
            self._state[current_id].torque_enabled = False
        if self._state[current_id].torque_enabled not in (False, None):
            raise RuntimeError(f"Servo {current_id} torque must be disabled before ID assignment")
        telemetry = self._state.pop(current_id)
        telemetry.servo_id = new_id
        telemetry.reported_servo_id = new_id
        self._state[new_id] = telemetry

    def read_live_telemetry(self, servo_ids: list[int]) -> dict[int, ServoTelemetry]:
        return self.read_telemetry(
            servo_ids,
            include_reported_id=False,
            include_identity=False,
            include_limits=False,
        )

    def read_minimal_telemetry(self, servo_ids: list[int]) -> dict[int, ServoTelemetry]:
        telemetry = self.read_telemetry(
            servo_ids,
            include_reported_id=False,
            include_identity=False,
            include_limits=False,
        )
        for item in telemetry.values():
            item.present_voltage_raw_unit = None
            item.present_voltage_mv = None
            item.present_temperature_c = None
            item.read_source = "live_read"
        return telemetry

    def read_telemetry(
        self,
        servo_ids: list[int],
        *,
        include_reported_id: bool = True,
        include_identity: bool = True,
        include_limits: bool = True,
    ) -> dict[int, ServoTelemetry]:
        result: dict[int, ServoTelemetry] = {}
        for servo_id in servo_ids:
            read_started_at = time.monotonic()
            telemetry = self._state.get(servo_id)
            completed_at = time.monotonic()
            wall_time = datetime.now(timezone.utc).isoformat()
            if telemetry is None:
                result[servo_id] = ServoTelemetry(
                    servo_id=servo_id,
                    reported_servo_id=None,
                    present_position=None,
                    present_current_ma=None,
                    present_voltage_mv=None,
                    present_temperature_c=None,
                    hardware_error="missing",
                    identity_error="missing servo",
                    telemetry_error="missing servo",
                    last_read_monotonic_s=completed_at,
                    last_valid_packet_monotonic_s=None,
                    last_valid_packet_wall_time=None,
                    last_read_attempt_monotonic_s=read_started_at,
                    read_duration_ms=max(0.0, (completed_at - read_started_at) * 1000.0),
                    packet_age_s=None,
                    read_source="live_read",
                    telemetry_error_code="servo_missing",
                    telemetry_error_detail="missing servo",
                )
            else:
                valid_packet_time = (
                    telemetry.last_valid_packet_monotonic_s
                    if telemetry.last_valid_packet_monotonic_s is not None
                    else (
                        telemetry.last_read_monotonic_s
                        if telemetry.last_read_monotonic_s is not None
                        else completed_at
                    )
                )
                result[servo_id] = ServoTelemetry(
                    servo_id=servo_id,
                    reported_servo_id=(
                        telemetry.reported_servo_id
                        if include_reported_id and telemetry.reported_servo_id is not None
                        else None
                    ),
                    model_number=telemetry.model_number if include_identity else None,
                    firmware_version=telemetry.firmware_version if include_identity else None,
                    operating_mode=telemetry.operating_mode,
                    torque_enabled=telemetry.torque_enabled,
                    current_limit_ma=telemetry.current_limit_ma if include_limits else None,
                    min_position_limit=telemetry.min_position_limit if include_limits else None,
                    max_position_limit=telemetry.max_position_limit if include_limits else None,
                    bus_watchdog_value=telemetry.bus_watchdog_value if include_limits else None,
                    present_position=telemetry.present_position,
                    present_current_raw_unit=telemetry.present_current_raw_unit,
                    present_current_ma=telemetry.present_current_ma,
                    present_voltage_raw_unit=telemetry.present_voltage_raw_unit,
                    present_voltage_mv=telemetry.present_voltage_mv,
                    present_temperature_c=telemetry.present_temperature_c,
                    hardware_error=telemetry.hardware_error,
                    hardware_error_code=telemetry.hardware_error_code,
                    identity_error=telemetry.identity_error,
                    telemetry_error=telemetry.telemetry_error,
                    last_read_monotonic_s=(
                        telemetry.last_read_monotonic_s
                        if telemetry.last_read_monotonic_s is not None
                        else completed_at
                    ),
                    last_valid_packet_monotonic_s=valid_packet_time,
                    last_valid_packet_wall_time=telemetry.last_valid_packet_wall_time or wall_time,
                    last_read_attempt_monotonic_s=read_started_at,
                    read_duration_ms=max(0.0, (completed_at - read_started_at) * 1000.0),
                    packet_age_s=max(0.0, completed_at - float(valid_packet_time)),
                    read_source="live_read",
                    telemetry_error_code=telemetry.telemetry_error_code,
                    telemetry_error_detail=telemetry.telemetry_error_detail or telemetry.telemetry_error,
                    bus_owner=telemetry.bus_owner,
                    read_sequence_index=telemetry.read_sequence_index,
                )
        return result
