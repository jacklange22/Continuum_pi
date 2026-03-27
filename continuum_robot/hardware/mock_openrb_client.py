"""Mock OpenRB-150 client used for GUI and test workflows."""

from __future__ import annotations

from continuum_robot.hardware.openrb_client import OpenRbClient


class MockOpenRbClient(OpenRbClient):
    """Stateful mock OpenRB client for operator-flow validation."""

    def connect(self, port: str, baudrate: int = 115200) -> None:
        if not port:
            raise RuntimeError("Mock OpenRB port is empty. Pick a mock or real port first.")
        self._port = port
        self._baudrate = baudrate
        self._last_status = f"connected to {port} @ {baudrate}"

    def prepare_for_dynamixel_use(self) -> bool:
        if not self.is_connected:
            raise RuntimeError("OpenRB-150 is not connected")
        self._last_status = "prepared for DYNAMIXEL use"
        return True
