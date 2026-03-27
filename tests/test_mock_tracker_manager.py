import time

from continuum_robot.tracking.mock_tracker_manager import MockTrackerManager


def test_mock_tracker_manager_streams_synthetic_tools() -> None:
    manager = MockTrackerManager(poll_hz=20)
    manager.start()
    try:
        time.sleep(0.05)
        snapshot = manager.get_state_snapshot()
        tool_0a = manager.get_latest_tool("0A")
        tool_0b = manager.get_latest_tool("0B")
    finally:
        manager.stop()

    assert snapshot.connection_state == "tracking"
    assert snapshot.bridge_running is True
    assert tool_0a is not None and tool_0a.valid is True
    assert tool_0b is not None and tool_0b.valid is True
    assert tool_0a.frame_number >= 0
    assert tool_0b.translation_mm != tool_0a.translation_mm
