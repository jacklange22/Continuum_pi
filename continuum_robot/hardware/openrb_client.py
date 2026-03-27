"""OpenRB-150 board-specific preparation and utility actions."""

from __future__ import annotations


class OpenRbClient:
    """Hardware-facing OpenRB-150 seam.

    The real OpenRB serial/status path is not implemented yet. This class
    intentionally fails clearly in hardware mode so the operator is not told
    that the board is connected when the repo is still using a mock-only path.
    """

    def __init__(self) -> None:
        self._port: str | None = None
        self._baudrate: int | None = None
        self._last_status = "disconnected"

    @property
    def is_connected(self) -> bool:
        return self._port is not None

    @property
    def last_status(self) -> str:
        return self._last_status

    def connect(self, port: str, baudrate: int = 115200) -> None:
        """Refuse to fake a hardware connection."""
        _ = baudrate
        if not port:
            raise RuntimeError("OpenRB port is empty. Configure openrb_port before connecting.")
        self._last_status = "hardware transport not implemented"
        raise RuntimeError(
            "Real OpenRB-150 transport/status handling is not implemented yet. "
            "Use mock mode for validated GUI testing or complete the hardware integration seam first."
        )

    def disconnect(self) -> None:
        """Clear any cached status."""
        self._port = None
        self._baudrate = None
        self._last_status = "disconnected"

    def prepare_for_dynamixel_use(self) -> bool:
        """Refuse to claim successful hardware preparation without an implementation."""
        self._last_status = "hardware transport not implemented"
        raise RuntimeError(
            "OpenRB-150 prepare action is not implemented for real hardware yet. "
            "This button is validated in mock mode only."
        )
