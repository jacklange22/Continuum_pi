import io
from pathlib import Path
from unittest.mock import patch

from continuum_robot.tracking.tracker_socket_client import TrackerSocketClient


class _FakeSocket:
    def __init__(self) -> None:
        self._file = io.StringIO('{"type":"status","state":"ok","details":{}}\n')
        self.closed = False

    def connect(self, _path: str) -> None:
        return None

    def settimeout(self, _timeout: float) -> None:
        return None

    def makefile(self, _mode: str, encoding: str = "utf-8"):
        _ = encoding
        return self._file

    def close(self) -> None:
        self.closed = True


def test_tracker_socket_client_reads_line_with_mock_socket() -> None:
    fake = _FakeSocket()
    with patch("socket.socket", return_value=fake):
        client = TrackerSocketClient(Path("/tmp/tracker_bridge.sock"))
        client.connect(timeout_s=0.1)
        line = client.read_line(timeout_s=0.1)
        client.close()

    assert line is not None
    assert '"type":"status"' in line
    assert fake.closed is True
