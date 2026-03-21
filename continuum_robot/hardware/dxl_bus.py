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


class DxlBus:
    """Low-level DYNAMIXEL communication interface.

    Future implementation will wrap SDK/group sync operations.
    """

    def connect(self, port: str, baudrate: int) -> None:
        """Open the DYNAMIXEL serial bus."""
        self._port = port
        self._baudrate = baudrate

    def disconnect(self) -> None:
        """Close the DYNAMIXEL serial bus."""
        self._port = None

    def scan_ids(self, min_id: int = 1, max_id: int = 20) -> list[int]:
        """Return discovered servo IDs.

        Scaffold returns an empty list.
        """
        return []

    def write_goal_positions(self, positions_by_id: dict[int, int]) -> None:
        """Send goal positions in ticks."""
        _ = positions_by_id

    def read_telemetry(self, servo_ids: list[int]) -> dict[int, ServoTelemetry]:
        """Return telemetry map for requested IDs."""
        return {sid: ServoTelemetry(servo_id=sid) for sid in servo_ids}
