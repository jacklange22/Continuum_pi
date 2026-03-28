from pathlib import Path

import pytest

import continuum_robot.tracking.backend_router as backend_router_module
from continuum_robot.tracking.backend_router import TrackingBackendRouter
from continuum_robot.tracking.runtime_models import TrackerRuntimeState


class _FakeBackend:
    def __init__(self, *, start_error: Exception | None = None) -> None:
        self._alive = False
        self._start_error = start_error

    def start(self) -> None:
        if self._start_error is not None:
            raise self._start_error
        self._alive = True

    def stop(self, timeout_s: float = 3.0) -> None:
        _ = timeout_s
        self._alive = False

    def is_alive(self) -> bool:
        return self._alive

    def get_state_snapshot(self) -> TrackerRuntimeState:
        return TrackerRuntimeState(
            connection_state="tracking" if self._alive else "disconnected",
            backend_running=self._alive,
            backend_connected=self._alive,
            backend_frame_counter=1 if self._alive else 0,
            latest_frame_number=1 if self._alive else None,
            latest_timestamp="2026-01-01T00:00:00Z" if self._alive else None,
            last_status_message="ok" if self._alive else "stopped",
        )


def _make_port(path: Path) -> Path:
    path.write_text("", encoding="utf-8")
    path.chmod(0o666)
    return path


def _make_executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_backend_router_falls_back_when_preferred_backend_fails_to_start(tmp_path: Path, monkeypatch) -> None:
    aurora_port = _make_port(tmp_path / "ttyUSB0")
    bridge_exec = _make_executable(tmp_path / "tracker_bridge")
    monkeypatch.setattr(backend_router_module, "load_ndi_tracker_class", lambda: object())

    router = TrackingBackendRouter(
        mock_mode=False,
        preferred_backend="ndi",
        fallback_backend="bridge",
        fallback_enabled=True,
        aurora_port=str(aurora_port),
        tracker_type="aurora",
        poll_interval_ms=20,
        reconnect_delay_s=0.1,
        ports_to_probe=[],
        settings_overrides={},
        tool_id_aliases={"10": "0A", "11": "0B"},
        bridge_executable=bridge_exec,
        socket_path=tmp_path / "tracker.sock",
        backend_factories={
            "ndi": lambda: _FakeBackend(start_error=RuntimeError("ndi start failed")),
            "bridge": lambda: _FakeBackend(),
        },
    )

    router.start()
    try:
        state = router.get_state_snapshot()
    finally:
        router.stop()

    assert state.selected_backend_name == "bridge"
    assert state.fallback_used is True
    assert any("ndi start failed" in message for message in state.startup_messages)
    assert any("Falling back from ndi to bridge." == message for message in state.startup_messages)


def test_backend_router_capability_report_surfaces_import_failure_code(tmp_path: Path, monkeypatch) -> None:
    aurora_port = _make_port(tmp_path / "ttyUSB0")
    monkeypatch.setattr(
        backend_router_module,
        "load_ndi_tracker_class",
        lambda: (_ for _ in ()).throw(RuntimeError("missing ndi package")),
    )

    router = TrackingBackendRouter(
        mock_mode=False,
        preferred_backend="ndi",
        fallback_backend="bridge",
        fallback_enabled=True,
        aurora_port=str(aurora_port),
        tracker_type="aurora",
        poll_interval_ms=20,
        reconnect_delay_s=0.1,
        ports_to_probe=[],
        settings_overrides={},
        tool_id_aliases={},
        bridge_executable=tmp_path / "tracker_bridge",
        socket_path=tmp_path / "tracker.sock",
    )

    capability = router.probe_capabilities()["ndi"]
    assert capability["available"] is False
    assert capability["code"] == "backend_import_failure"


def test_backend_router_rejects_missing_serial_port_with_structured_code(tmp_path: Path) -> None:
    router = TrackingBackendRouter(
        mock_mode=False,
        preferred_backend="ndi",
        fallback_backend="bridge",
        fallback_enabled=True,
        aurora_port=str(tmp_path / "missing-ttyUSB0"),
        tracker_type="aurora",
        poll_interval_ms=20,
        reconnect_delay_s=0.1,
        ports_to_probe=[],
        settings_overrides={},
        tool_id_aliases={},
        bridge_executable=tmp_path / "tracker_bridge",
        socket_path=tmp_path / "tracker.sock",
    )

    capability = router.probe_capabilities()["ndi"]
    assert capability["available"] is False
    assert capability["code"] == "serial_port_missing"
