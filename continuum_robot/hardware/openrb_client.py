"""OpenRB-150 board-specific preparation and utility actions.

This module does not own low-level DYNAMIXEL protocol operations.
"""


class OpenRbClient:
    """Handles board-specific setup/prep and status actions."""

    def connect(self, port: str, baudrate: int = 115200) -> None:
        """Connect to board control interface for prep/status operations."""
        self._port = port
        self._baudrate = baudrate

    def disconnect(self) -> None:
        """Disconnect board control interface."""
        self._port = None

    def prepare_for_dynamixel_use(self) -> bool:
        """Run safe board prep workflow required before servo operations.

        Returns True in scaffold mode.
        """
        return True
