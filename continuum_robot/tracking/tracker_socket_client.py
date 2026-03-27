"""Unix-domain socket client for tracker_bridge stream."""

from __future__ import annotations

from pathlib import Path
import socket
import time


class TrackerSocketClient:
    """Simple line-oriented Unix socket client for tracker_bridge."""

    def __init__(self, socket_path: Path) -> None:
        self.socket_path = socket_path
        self._sock: socket.socket | None = None
        self._file = None

    @property
    def is_connected(self) -> bool:
        return self._sock is not None

    def connect(self, timeout_s: float = 5.0) -> None:
        self.close()
        deadline = time.monotonic() + timeout_s
        last_exc: Exception | None = None

        while time.monotonic() < deadline:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(str(self.socket_path))
                sock.settimeout(0.5)
                self._sock = sock
                self._file = sock.makefile("r", encoding="utf-8")
                return
            except OSError as exc:
                last_exc = exc
                sock.close()
                time.sleep(0.1)

        raise ConnectionError(f"Could not connect to tracker socket {self.socket_path}: {last_exc}")

    def read_line(self, timeout_s: float = 0.5) -> str | None:
        if self._sock is None or self._file is None:
            raise RuntimeError("tracker socket is not connected")
        self._sock.settimeout(timeout_s)
        try:
            line = self._file.readline()
        except socket.timeout:
            return None
        if line == "":
            raise ConnectionError("tracker socket closed")
        return line.rstrip("\n")

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
