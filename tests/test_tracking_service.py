import json
from pathlib import Path
import time

import numpy as np

from continuum_robot.hardware.mock_aurora_client import MockAuroraClient
from continuum_robot.services.packet_capture import PacketCaptureWriter
from continuum_robot.services.tracking_service import TrackingService
from tests.fixtures.aurora_samples import (
    build_tool_0A_record,
    build_tool_0B_record,
    build_transform_frame_from_records,
    build_valid_transform_frame,
)


class _ReconnectClient(MockAuroraClient):
    def __init__(self, first_session: bytes, second_session: bytes) -> None:
        super().__init__()
        self._sessions = [bytearray(first_session), bytearray(second_session)]
        self._connect_count = 0
        self._raised = False

    def connect(self, port: str, baudrate: int = 115200, timeout_s: float = 0.1) -> None:
        super().connect(port, baudrate=baudrate, timeout_s=timeout_s)
        self._connect_count += 1
        if self._connect_count <= len(self._sessions):
            self._buffer = bytearray(self._sessions[self._connect_count - 1])

    def read_bytes(self, nbytes: int = 1) -> bytes:
        if self._connect_count == 1 and not self._raised and not self._buffer:
            self._raised = True
            raise OSError("simulated serial read failure")
        return super().read_bytes(nbytes)


def _write_registration_file(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "T_robot_aurora": np.eye(4).tolist(),
                "T_coil_tip": np.eye(4).tolist(),
                "config_used": {"tip_calibration_source": "test_identity"},
            }
        ),
        encoding="utf-8",
    )


def test_tracking_service_reports_missing_registration_and_missing_0b(tmp_path: Path) -> None:
    service = TrackingService(
        MockAuroraClient(),
        port="/dev/null",
        registration_path=tmp_path / "missing_registration.json",
        config_source="test",
    )
    frame = build_transform_frame_from_records(frame_number=1, records=[build_tool_0A_record()])
    service.ingest_frame(frame, source="test")
    snapshot = service.get_snapshot()

    assert snapshot.packets_received_count == 1
    assert "missing_registration" in snapshot.faults
    assert "missing_0B" in snapshot.faults
    assert snapshot.tip_pose_status == "missing_registration"


def test_tracking_service_replay_updates_tip_pose_when_registration_exists(tmp_path: Path) -> None:
    registration_path = tmp_path / "latest_registration.json"
    _write_registration_file(registration_path)

    service = TrackingService(
        MockAuroraClient(),
        port="/dev/null",
        registration_path=registration_path,
        config_source="test",
    )

    capture_path = tmp_path / "capture.jsonl"
    writer = PacketCaptureWriter(capture_path)
    writer.write_frame(
        build_valid_transform_frame(frame_number=10),
        captured_at_utc="2026-01-01T00:00:00.000Z",
        source="serial",
    )
    writer.close()

    snapshot = service.replay_capture(capture_path)
    assert snapshot.packets_received_count == 1
    assert snapshot.tip_pose_status == "ok"
    assert snapshot.T_robot_tip is not None
    assert snapshot.last_frame_number == 10


def test_tracking_service_reconnects_after_serial_failure(tmp_path: Path) -> None:
    first = build_transform_frame_from_records(frame_number=1, records=[build_tool_0A_record()])
    second = build_transform_frame_from_records(frame_number=2, records=[build_tool_0A_record(), build_tool_0B_record()])
    client = _ReconnectClient(first_session=first, second_session=second)
    service = TrackingService(
        client,
        port="/dev/ttyUSB0",
        registration_path=tmp_path / "missing_registration.json",
        config_source="test",
        frame_timeout_s=0.05,
        reconnect_delay_s=0.01,
    )

    service.start()
    deadline = time.monotonic() + 1.0
    try:
        while time.monotonic() < deadline:
            snapshot = service.get_snapshot()
            if snapshot.reconnect_count >= 1 and snapshot.last_frame_number == 2:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("Tracking service did not reconnect in time")
    finally:
        service.stop()

    snapshot = service.get_snapshot()
    assert snapshot.reconnect_count >= 1
    assert snapshot.last_frame_number == 2
