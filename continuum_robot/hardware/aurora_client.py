"""Aurora tracker serial client.

This module intentionally only owns serial I/O (open/close/read/write).
Protocol framing/parsing is implemented in tracking modules.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import serial


@dataclass
class AuroraConnectionConfig:
    """Serial settings for Aurora connection."""

    port: str
    baudrate: int = 115200
    timeout_s: float = 0.1


class AuroraClient:
    """Serial interface for Aurora byte stream access."""

    def __init__(self) -> None:
        self._serial: serial.Serial | None = None

    @property
    def is_connected(self) -> bool:
        """Return True when the serial port is open."""
        return self._serial is not None and self._serial.is_open

    def connect(self, port: str, baudrate: int = 115200, timeout_s: float = 0.1) -> None:
        """Open Aurora serial port.

        Raises serial.SerialException on connection failures.
        """
        self.disconnect()
        self._serial = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout_s,
            xonxoff=False,
        )

    def disconnect(self) -> None:
        """Close Aurora serial connection if open."""
        if self._serial is not None:
            try:
                if self._serial.is_open:
                    self._serial.close()
            finally:
                self._serial = None

    def set_timeout(self, timeout_s: float) -> None:
        """Update serial read timeout."""
        if self._serial is None:
            raise RuntimeError("Aurora serial connection is not open")
        self._serial.timeout = timeout_s

    def read_bytes(self, nbytes: int = 1) -> bytes:
        """Read up to ``nbytes`` from Aurora stream."""
        if self._serial is None:
            raise RuntimeError("Aurora serial connection is not open")
        if nbytes <= 0:
            return b""
        return bytes(self._serial.read(nbytes))

    def write_bytes(self, payload: bytes) -> int:
        """Write raw bytes to Aurora stream and return bytes written."""
        if self._serial is None:
            raise RuntimeError("Aurora serial connection is not open")
        return int(self._serial.write(payload))

    def flush_input(self) -> None:
        """Flush input buffer if connected."""
        if self._serial is not None:
            self._serial.reset_input_buffer()

    def wait_for_data(self, timeout_s: float) -> bool:
        """Poll whether bytes are available before timeout."""
        if self._serial is None:
            raise RuntimeError("Aurora serial connection is not open")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._serial.in_waiting > 0:
                return True
            time.sleep(0.001)
        return False
