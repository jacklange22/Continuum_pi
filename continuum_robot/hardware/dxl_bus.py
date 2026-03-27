"""Low-level DYNAMIXEL bus abstraction.

This module owns raw DYNAMIXEL protocol communication.
OpenRB board concerns are intentionally excluded.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ServoTelemetry:
    """Readback values for one servo."""

    servo_id: int
    present_position: int | None = None
    present_current_ma: int | None = None
    present_voltage_mv: int | None = None
    hardware_error: str | None = None


class DxlBus:
    """Low-level DYNAMIXEL communication interface."""

    def __init__(self) -> None:
        self._port: str | None = None
        self._baudrate: int | None = None

    @property
    def is_connected(self) -> bool:
        return self._port is not None

    def connect(self, port: str, baudrate: int) -> None:
        """Refuse to fake a real hardware connection."""
        _ = baudrate
        if not port:
            raise RuntimeError("OpenRB/DYNAMIXEL port is empty. Configure openrb_port before connecting.")
        raise RuntimeError(
            "Real DYNAMIXEL/OpenRB transport is not implemented yet. "
            "Use mock mode for validated servo workflows or complete continuum_robot/hardware/dxl_bus.py."
        )

    def disconnect(self) -> None:
        """Close the DYNAMIXEL serial bus."""
        self._port = None
        self._baudrate = None

    def scan_ids(self, min_id: int = 1, max_id: int = 20) -> list[int]:
        """Return discovered servo IDs."""
        _ = (min_id, max_id)
        if not self.is_connected:
            raise RuntimeError("DYNAMIXEL bus is not connected")
        raise RuntimeError("Real servo ID scanning is not implemented yet")

    def write_goal_positions(self, positions_by_id: dict[int, int]) -> None:
        """Send goal positions in ticks."""
        _ = positions_by_id
        if not self.is_connected:
            raise RuntimeError("DYNAMIXEL bus is not connected")
        raise RuntimeError("Real goal-position writes are not implemented yet")

    def write_servo_id(self, current_id: int, new_id: int) -> None:
        """Assign a new servo ID.

        The concrete hardware implementation will override this.
        """
        _ = (current_id, new_id)
        if not self.is_connected:
            raise RuntimeError("DYNAMIXEL bus is not connected")
        raise RuntimeError("Real servo ID assignment is not implemented yet")

    def read_telemetry(self, servo_ids: list[int]) -> dict[int, ServoTelemetry]:
        """Return telemetry map for requested IDs."""
        if not self.is_connected:
            return {
                sid: ServoTelemetry(servo_id=sid, hardware_error="disconnected")
                for sid in servo_ids
            }
        raise RuntimeError("Real DYNAMIXEL telemetry reads are not implemented yet")
