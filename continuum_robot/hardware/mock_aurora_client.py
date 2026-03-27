"""Mock Aurora client for GUI and pipeline testing without hardware."""

from __future__ import annotations

from continuum_robot.hardware.aurora_client import AuroraClient


class MockAuroraClient(AuroraClient):
    """Aurora mock with an optional byte buffer for tests and replay."""

    def __init__(self, initial_bytes: bytes = b"") -> None:
        super().__init__()
        self._connected = False
        self._buffer = bytearray(initial_bytes)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, port: str, baudrate: int = 115200, timeout_s: float = 0.1) -> None:
        _ = (port, baudrate, timeout_s)
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def read_bytes(self, nbytes: int = 1) -> bytes:
        if not self._connected:
            raise RuntimeError("Aurora mock connection is not open")
        if nbytes <= 0 or not self._buffer:
            return b""
        chunk = bytes(self._buffer[:nbytes])
        del self._buffer[:nbytes]
        return chunk

    def write_bytes(self, payload: bytes) -> int:
        _ = payload
        if not self._connected:
            raise RuntimeError("Aurora mock connection is not open")
        return 0

    def flush_input(self) -> None:
        self._buffer.clear()

    def wait_for_data(self, timeout_s: float) -> bool:
        _ = timeout_s
        return bool(self._buffer)

    def append_bytes(self, payload: bytes) -> None:
        self._buffer.extend(payload)
