"""Synthetic tracker manager for mock-mode GUI and integration tests."""

from __future__ import annotations

from dataclasses import replace
import math
import threading
import time

from continuum_robot.tracking.tracker_service_manager import TrackerRuntimeState, TrackerToolState


class MockTrackerManager:
    """Small in-process tracker manager that mimics the live service interface."""

    backend_identity = "mock_tracker_manager"

    def __init__(self, poll_hz: int = 10) -> None:
        self.poll_hz = max(1, int(poll_hz))
        self._started_at: float | None = None
        self._lock = threading.Lock()
        self._state = TrackerRuntimeState(
            connection_state="disconnected",
            backend_connected=False,
            backend_running=False,
            socket_connected=False,
            bridge_running=False,
            backend_frame_counter=0,
            latest_frame_number=None,
            latest_timestamp=None,
            last_status_message="Mock tracker idle",
        )

    def start(self) -> None:
        with self._lock:
            self._started_at = time.monotonic()
            self._state.connection_state = "tracking"
            self._state.backend_connected = True
            self._state.backend_running = True
            self._state.socket_connected = True
            self._state.bridge_running = True
            self._state.last_error = None
            self._state.last_status_message = "Mock tracker streaming synthetic tool poses"
        self._refresh_state()

    def stop(self, timeout_s: float = 3.0) -> None:
        _ = timeout_s
        with self._lock:
            self._started_at = None
            self._state.connection_state = "disconnected"
            self._state.backend_connected = False
            self._state.backend_running = False
            self._state.socket_connected = False
            self._state.bridge_running = False
            self._state.tools = {}
            self._state.backend_frame_counter = 0
            self._state.latest_frame_number = None
            self._state.latest_timestamp = None
            self._state.raw_tool_ids = []
            self._state.normalized_tool_ids = []
            self._state.tool_id_mapping = {}
            self._state.runtime_role_mappings = {}
            self._state.unmapped_tool_ids = []
            self._state.last_status_message = "Mock tracker disconnected"

    def is_alive(self) -> bool:
        with self._lock:
            return self._started_at is not None

    def get_state_snapshot(self) -> TrackerRuntimeState:
        self._refresh_state()
        with self._lock:
            tools = {tool_id: replace(tool) for tool_id, tool in self._state.tools.items()}
            return TrackerRuntimeState(
                connection_state=self._state.connection_state,
                backend_running=self._state.backend_running,
                backend_connected=self._state.backend_connected,
                socket_connected=self._state.socket_connected,
                bridge_running=self._state.bridge_running,
                latest_frame_number=self._state.latest_frame_number,
                latest_timestamp=self._state.latest_timestamp,
                last_status_message=self._state.last_status_message,
                last_error=self._state.last_error,
                tools=tools,
            )

    def get_latest_tool(self, tool_id: str) -> TrackerToolState | None:
        self._refresh_state()
        with self._lock:
            tool = self._state.tools.get(tool_id)
            return replace(tool) if tool is not None else None

    def _refresh_state(self) -> None:
        with self._lock:
            if self._started_at is None:
                return
            elapsed = max(0.0, time.monotonic() - self._started_at)
            frame_number = int(elapsed * self.poll_hz)
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            phase = elapsed
            tool_0a = TrackerToolState(
                tool_id="0A",
                frame_number=frame_number,
                valid=True,
                validity_known=True,
                status="tracked",
                quaternion=(1.0, 0.0, 0.0, 0.0),
                translation_mm=(
                    25.0 + 8.0 * math.cos(phase),
                    12.0 + 6.0 * math.sin(phase),
                    60.0 + 2.0 * math.sin(phase * 0.5),
                ),
                quality=0.15,
                timestamp=timestamp,
            )
            tool_0b = TrackerToolState(
                tool_id="0B",
                frame_number=frame_number,
                valid=True,
                validity_known=True,
                status="tracked",
                quaternion=(1.0, 0.0, 0.0, 0.0),
                translation_mm=(
                    5.0 + 18.0 * math.cos(phase * 0.3),
                    5.0 + 18.0 * math.sin(phase * 0.3),
                    0.5 * math.sin(phase),
                ),
                quality=0.1,
                timestamp=timestamp,
            )
            self._state.tools = {"0A": tool_0a, "0B": tool_0b}
            self._state.latest_frame_number = frame_number
            self._state.backend_frame_counter = frame_number
            self._state.latest_timestamp = timestamp
            self._state.raw_tool_ids = ["0A", "0B"]
            self._state.normalized_tool_ids = ["0A", "0B"]
            self._state.tool_id_mapping = {"0A": "0A", "0B": "0B"}
            self._state.runtime_role_mappings = {"0A": "0A", "0B": "0B"}
            self._state.unmapped_tool_ids = []
