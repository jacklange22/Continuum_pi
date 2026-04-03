"""Mock DYNAMIXEL bus for non-hardware test and GUI development."""

from __future__ import annotations

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
                present_current_ma=140 + 10 * idx,
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
            telemetry.present_voltage_mv = 12000 - min(500, current // 10)
            telemetry.present_temperature_c = min(65, 30 + current // 20)
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
        return self.read_telemetry(servo_ids, include_identity=False, include_limits=False)

    def read_telemetry(
        self,
        servo_ids: list[int],
        *,
        include_identity: bool = True,
        include_limits: bool = True,
    ) -> dict[int, ServoTelemetry]:
        result: dict[int, ServoTelemetry] = {}
        for servo_id in servo_ids:
            telemetry = self._state.get(servo_id)
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
                    last_read_monotonic_s=time.monotonic(),
                )
            else:
                result[servo_id] = ServoTelemetry(
                    servo_id=servo_id,
                    reported_servo_id=telemetry.reported_servo_id if telemetry.reported_servo_id is not None else servo_id,
                    model_number=telemetry.model_number if include_identity else None,
                    firmware_version=telemetry.firmware_version if include_identity else None,
                    operating_mode=telemetry.operating_mode,
                    torque_enabled=telemetry.torque_enabled,
                    current_limit_ma=telemetry.current_limit_ma if include_limits else None,
                    min_position_limit=telemetry.min_position_limit if include_limits else None,
                    max_position_limit=telemetry.max_position_limit if include_limits else None,
                    bus_watchdog_value=telemetry.bus_watchdog_value if include_limits else None,
                    present_position=telemetry.present_position,
                    present_current_ma=telemetry.present_current_ma,
                    present_voltage_mv=telemetry.present_voltage_mv,
                    present_temperature_c=telemetry.present_temperature_c,
                    hardware_error=telemetry.hardware_error,
                    hardware_error_code=telemetry.hardware_error_code,
                    identity_error=telemetry.identity_error,
                    telemetry_error=telemetry.telemetry_error,
                    last_read_monotonic_s=time.monotonic(),
                )
        return result
