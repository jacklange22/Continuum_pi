from pathlib import Path

import pytest

from continuum_robot.tracking.tracker_service_manager import TrackerServiceManager


def test_tracker_service_manager_rejects_empty_aurora_port(tmp_path: Path) -> None:
    manager = TrackerServiceManager(
        bridge_executable=tmp_path / "tracker_bridge",
        socket_path=tmp_path / "tracker.sock",
        aurora_port="",
    )

    with pytest.raises(RuntimeError, match="Aurora port is empty"):
        manager.start()

    state = manager.get_state_snapshot()
    assert state.connection_state == "error"
    assert state.last_error is not None


def test_tracker_service_manager_reports_missing_bridge_executable(tmp_path: Path) -> None:
    manager = TrackerServiceManager(
        bridge_executable=tmp_path / "missing_tracker_bridge",
        socket_path=tmp_path / "tracker.sock",
        aurora_port="/dev/ttyUSB0",
    )

    with pytest.raises(FileNotFoundError, match="tracker_bridge executable not found"):
        manager.start()

    state = manager.get_state_snapshot()
    assert state.connection_state == "error"
    assert "tracker_bridge executable not found" in (state.last_error or "")


def test_tracker_service_manager_reports_non_executable_bridge(tmp_path: Path) -> None:
    bridge = tmp_path / "tracker_bridge"
    bridge.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    bridge.chmod(0o644)
    manager = TrackerServiceManager(
        bridge_executable=bridge,
        socket_path=tmp_path / "tracker.sock",
        aurora_port="/dev/ttyUSB0",
    )

    with pytest.raises(PermissionError, match="not executable"):
        manager.start()

    state = manager.get_state_snapshot()
    assert state.connection_state == "error"
    assert "not executable" in (state.last_error or "")
