import json
from pathlib import Path
import time

import numpy as np

from continuum_robot.hardware.mock_aurora_client import MockAuroraClient
from continuum_robot.services.packet_capture import PacketCaptureWriter
from continuum_robot.services.tracking_service import TrackingService
from continuum_robot.tracking.tracker_service_manager import TrackerRuntimeState, TrackerToolState
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


class _FakeLiveBackend:
    def __init__(self, state: TrackerRuntimeState) -> None:
        self._state = state
        self._alive = False
        self._timing_listeners = []

    def start(self) -> None:
        self._alive = True

    def stop(self) -> None:
        self._alive = False

    def is_alive(self) -> bool:
        return self._alive

    def get_state_snapshot(self) -> TrackerRuntimeState:
        return self._state

    def set_state(self, state: TrackerRuntimeState) -> None:
        self._state = state

    def register_timing_listener(self, listener) -> None:
        self._timing_listeners.append(listener)

    def unregister_timing_listener(self, listener) -> None:
        self._timing_listeners = [item for item in self._timing_listeners if item is not listener]


def _write_registration_file(path: Path, *, measurement_tool_id: str = "0B", coil_tool_id: str = "0A") -> None:
    path.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-01-01T00:00:00Z",
                "fre_mm": 0.25,
                "T_robot_aurora": np.eye(4).tolist(),
                "T_coil_tip": np.eye(4).tolist(),
                "config_used": {
                    "tip_calibration_source": "test_identity",
                    "measurement_tool_id": measurement_tool_id,
                    "coil_tool_id": coil_tool_id,
                },
            }
        ),
        encoding="utf-8",
    )


def _tracked_tool(tool_id: str, *, frame_number: int, xyz=(1.0, 2.0, 3.0), quat=(1.0, 0.0, 0.0, 0.0), status="tracked") -> TrackerToolState:
    return TrackerToolState(
        tool_id=tool_id,
        frame_number=frame_number,
        valid=True,
        status=status,
        quaternion=quat,
        translation_mm=tuple(float(v) for v in xyz),
        quality=0.1,
        timestamp="2026-01-01T00:00:00Z",
    )


def _live_state(*, frame_number: int | None, tools: dict[str, TrackerToolState], connection_state: str = "tracking") -> TrackerRuntimeState:
    return TrackerRuntimeState(
        connection_state=connection_state,
        socket_connected=True,
        bridge_running=True,
        latest_frame_number=frame_number,
        latest_timestamp="2026-01-01T00:00:00Z" if frame_number is not None else None,
        last_status_message="tracker ok",
        last_error=None,
        tools=tools,
    )


def test_tracking_service_live_backend_reports_tracked_tools_with_unknown_validity(tmp_path: Path) -> None:
    registration_path = tmp_path / "latest_registration.json"
    _write_registration_file(registration_path)
    backend = _FakeLiveBackend(
        _live_state(
            frame_number=7,
            tools={
                "0A": _tracked_tool("0A", frame_number=7, xyz=(10.0, 20.0, 30.0)),
                "0B": _tracked_tool("0B", frame_number=7, xyz=(1.0, 2.0, 3.0)),
            },
        )
    )
    service = TrackingService(
        live_backend=backend,
        port="/dev/ttyUSB0",
        registration_path=registration_path,
        config_source="test",
    )

    service.start()
    try:
        snapshot = service.get_snapshot()
    finally:
        service.stop()

    assert snapshot.backend_identity == "fake_live_backend"
    assert snapshot.packets_received_count == 1
    assert snapshot.tools["0A"].tracking_state == "tracked"
    assert snapshot.tools["0A"].valid is None
    assert snapshot.tools["0A"].validity_known is False
    assert snapshot.tip_pose_status == "ok"
    assert snapshot.T_robot_tip is not None
    assert snapshot.stored_registration_timestamp_utc == "2026-01-01T00:00:00Z"
    assert snapshot.stored_registration_fre_mm == 0.25


def test_tracking_service_forwards_timing_listener_registration_to_live_backend(tmp_path: Path) -> None:
    backend = _FakeLiveBackend(_live_state(frame_number=1, tools={"0A": _tracked_tool("0A", frame_number=1)}))
    service = TrackingService(
        live_backend=backend,
        port="/dev/ttyUSB0",
        registration_path=tmp_path / "missing_registration.json",
        config_source="test",
    )

    listener = lambda _record: None
    service.register_timing_listener(listener)
    service.unregister_timing_listener(listener)

    assert backend._timing_listeners == []


def test_tracking_service_live_backend_detects_role_mismatch(tmp_path: Path) -> None:
    registration_path = tmp_path / "latest_registration.json"
    _write_registration_file(registration_path, measurement_tool_id="0A", coil_tool_id="0B")
    backend = _FakeLiveBackend(_live_state(frame_number=5, tools={"0A": _tracked_tool("0A", frame_number=5)}))
    service = TrackingService(
        live_backend=backend,
        port="/dev/ttyUSB0",
        registration_path=registration_path,
        config_source="test",
        runtime_coil_tool_id="0A",
        registration_tool_id="0B",
    )

    service.start()
    try:
        snapshot = service.get_snapshot()
    finally:
        service.stop()

    assert snapshot.tip_pose_status == "role_mismatch"
    assert "registration_role_mismatch" in snapshot.faults
    assert snapshot.health.health == "failed"
    assert "pose pipeline registration_role_mismatch" in snapshot.health.status
    assert snapshot.stored_registration_coil_tool_id == "0B"
    assert snapshot.stored_registration_measurement_tool_id == "0A"


def test_tracking_service_invalid_registration_escalates_health_to_failed(tmp_path: Path) -> None:
    registration_path = tmp_path / "latest_registration.json"
    registration_path.write_text(
        json.dumps(
            {
                "T_robot_aurora": np.eye(4).tolist(),
                "T_coil_tip": [[1.0, 0.0], [0.0, 1.0]],
            }
        ),
        encoding="utf-8",
    )
    service = TrackingService(
        live_backend=_FakeLiveBackend(_live_state(frame_number=None, tools={}, connection_state="connecting")),
        port="/dev/ttyUSB0",
        registration_path=registration_path,
        config_source="test",
    )

    snapshot = service.get_snapshot()

    assert snapshot.registration_state == "invalid_registration"
    assert snapshot.tip_pose_status == "invalid_registration"
    assert "invalid_registration" in snapshot.pipeline_faults
    assert snapshot.health.health == "failed"


def test_tracking_service_live_backend_marks_invalid_transform(tmp_path: Path) -> None:
    backend = _FakeLiveBackend(
        _live_state(
            frame_number=3,
            tools={"0A": _tracked_tool("0A", frame_number=3, quat=(0.0, 0.0, 0.0, 0.0))},
        )
    )
    service = TrackingService(
        live_backend=backend,
        port="/dev/ttyUSB0",
        registration_path=tmp_path / "missing_registration.json",
        config_source="test",
    )

    service.start()
    try:
        snapshot = service.get_snapshot()
    finally:
        service.stop()

    assert snapshot.tools["0A"].tracking_state == "invalid"
    assert snapshot.tools["0A"].valid is False
    assert snapshot.tools["0A"].validity_known is True
    assert "invalid_0A" in snapshot.faults


def test_tracking_service_preserves_backend_invalid_reason_when_pose_missing(tmp_path: Path) -> None:
    backend = _FakeLiveBackend(
        _live_state(
            frame_number=9,
            tools={
                "0A": TrackerToolState(
                    tool_id="0A",
                    frame_number=9,
                    valid=False,
                    validity_known=True,
                    status="invalid_transform: ndarray(2, 8): unsupported payload",
                    quaternion=None,
                    translation_mm=None,
                    quality=0.1,
                    timestamp="2026-01-01T00:00:00Z",
                )
            },
        )
    )
    service = TrackingService(
        live_backend=backend,
        port="/dev/ttyUSB0",
        registration_path=tmp_path / "missing_registration.json",
        config_source="test",
    )

    service.start()
    try:
        snapshot = service.get_snapshot()
    finally:
        service.stop()

    assert snapshot.tools["0A"].tracking_state == "invalid"


def test_tracking_service_uses_backend_reported_port_as_runtime_source_of_truth(tmp_path: Path) -> None:
    registration_path = tmp_path / "latest_registration.json"
    _write_registration_file(registration_path)
    backend_state = _live_state(
        frame_number=4,
        tools={
            "0A": _tracked_tool("0A", frame_number=4),
            "0B": _tracked_tool("0B", frame_number=4, xyz=(4.0, 5.0, 6.0)),
        },
    )
    backend_state.selected_backend_name = "ndi"
    backend_state.backend_details = {"aurora_port": "/dev/ttyUSB0"}
    backend_state.capability_report = {
        "ndi": {
            "available": True,
            "code": "ok",
            "details": {"aurora_port": "/dev/ttyUSB0"},
        }
    }
    backend = _FakeLiveBackend(backend_state)
    service = TrackingService(
        live_backend=backend,
        port="/dev/ttyAMA10",
        registration_path=registration_path,
        config_source="test",
    )

    service.start()
    try:
        snapshot = service.get_snapshot()
    finally:
        service.stop()

    assert service.port == "/dev/ttyUSB0"
    assert snapshot.port == "/dev/ttyUSB0"


def test_tracking_service_reports_unknown_before_live_frames_arrive(tmp_path: Path) -> None:
    backend = _FakeLiveBackend(_live_state(frame_number=None, tools={}, connection_state="connecting"))
    service = TrackingService(
        live_backend=backend,
        port="/dev/ttyUSB0",
        registration_path=tmp_path / "missing_registration.json",
        config_source="test",
    )

    snapshot = service.get_snapshot()
    assert snapshot.tools["0A"].tracking_state == "unknown"
    assert snapshot.tools["0B"].tracking_state == "unknown"


def test_tracking_service_compatibility_parser_reports_missing_registration_and_missing_0b(tmp_path: Path) -> None:
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
    assert snapshot.tools["0A"].tracking_state == "tracked"
    assert snapshot.tools["0A"].valid is None
    assert snapshot.tools["0A"].validity_known is False


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
