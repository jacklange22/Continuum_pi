"""Mock DYNAMIXEL bus for non-hardware test and GUI development."""

from __future__ import annotations

from continuum_robot.hardware.dxl_bus import DxlBus, ServoTelemetry


class MockDxlBus(DxlBus):
    """Stateful mock implementation of low-level bus operations."""

    def __init__(self, servo_ids: list[int] | None = None) -> None:
        super().__init__()
        ids = servo_ids or [1, 2, 3, 4]
        self._state: dict[int, ServoTelemetry] = {
            sid: ServoTelemetry(
                servo_id=sid,
                present_position=2048 + 25 * idx,
                present_current_ma=140 + 10 * idx,
                present_voltage_mv=12000,
                hardware_error=None,
            )
            for idx, sid in enumerate(ids)
        }

    def connect(self, port: str, baudrate: int) -> None:
        if not port:
            raise RuntimeError("Mock DYNAMIXEL port is empty. Pick a mock or real port first.")
        self._port = port
        self._baudrate = baudrate

    def scan_ids(self, min_id: int = 1, max_id: int = 20) -> list[int]:
        return [sid for sid in sorted(self._state) if min_id <= sid <= max_id]

    def write_goal_positions(self, positions_by_id: dict[int, int]) -> None:
        for servo_id, goal in positions_by_id.items():
            telemetry = self._state.setdefault(servo_id, ServoTelemetry(servo_id=servo_id))
            current = 120 + min(600, abs(goal - (telemetry.present_position or goal)) // 4)
            telemetry.present_position = int(goal)
            telemetry.present_current_ma = current
            telemetry.present_voltage_mv = 12000 - min(500, current // 10)

    def write_servo_id(self, current_id: int, new_id: int) -> None:
        if current_id not in self._state:
            raise ValueError(f"Servo {current_id} not found")
        if new_id in self._state:
            raise ValueError(f"Servo {new_id} already exists")
        telemetry = self._state.pop(current_id)
        telemetry.servo_id = new_id
        self._state[new_id] = telemetry

    def read_telemetry(self, servo_ids: list[int]) -> dict[int, ServoTelemetry]:
        result: dict[int, ServoTelemetry] = {}
        for servo_id in servo_ids:
            telemetry = self._state.get(servo_id)
            if telemetry is None:
                result[servo_id] = ServoTelemetry(
                    servo_id=servo_id,
                    present_position=None,
                    present_current_ma=None,
                    present_voltage_mv=None,
                    hardware_error="missing",
                )
            else:
                result[servo_id] = ServoTelemetry(
                    servo_id=servo_id,
                    present_position=telemetry.present_position,
                    present_current_ma=telemetry.present_current_ma,
                    present_voltage_mv=telemetry.present_voltage_mv,
                    hardware_error=telemetry.hardware_error,
                )
        return result
