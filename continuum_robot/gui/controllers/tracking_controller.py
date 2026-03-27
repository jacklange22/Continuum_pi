"""Tracking tab controller wiring for tracker_bridge state."""

from __future__ import annotations

from dataclasses import dataclass, field

from continuum_robot.config.settings import Settings
from continuum_robot.tracking.tracker_service_manager import TrackerServiceManager


@dataclass
class TrackingViewState:
    """UI-facing snapshot of live tracker state."""

    device_path: str
    socket_path: str
    connection_state: str = "disconnected"
    bridge_running: bool = False
    socket_connected: bool = False
    latest_frame_number: int | None = None
    latest_timestamp: str | None = None
    last_status_message: str = ""
    last_error: str | None = None
    tools: dict[str, dict] = field(default_factory=dict)


class TrackingController:
    """Owns live tool status and tip pose display updates."""

    def __init__(self, tracker_manager: TrackerServiceManager, settings: Settings) -> None:
        self.tracker_manager = tracker_manager
        self.state = TrackingViewState(
            device_path=settings.serial.aurora_port,
            socket_path=settings.serial.tracker_socket_path,
        )

    def set_device_path(self, device_path: str) -> None:
        self.state.device_path = device_path
        self.tracker_manager.aurora_port = device_path

    def connect(self) -> None:
        try:
            self.tracker_manager.start()
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.connection_state = "error"
            return
        self.refresh()

    def disconnect(self) -> None:
        try:
            self.tracker_manager.stop()
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.connection_state = "error"
            return
        self.refresh()

    def refresh(self) -> TrackingViewState:
        snapshot = self.tracker_manager.get_state_snapshot()
        self.state.connection_state = snapshot.connection_state
        self.state.bridge_running = snapshot.bridge_running
        self.state.socket_connected = snapshot.socket_connected
        self.state.latest_frame_number = snapshot.latest_frame_number
        self.state.latest_timestamp = snapshot.latest_timestamp
        self.state.last_status_message = snapshot.last_status_message
        self.state.last_error = snapshot.last_error
        self.state.tools = {
            tool_id: {
                "valid": tool.valid,
                "status": tool.status,
                "frame_number": tool.frame_number,
                "translation_mm": tool.translation_mm,
                "quaternion": tool.quaternion,
                "quality": tool.quality,
                "timestamp": tool.timestamp,
            }
            for tool_id, tool in snapshot.tools.items()
        }
        return self.state

    def shutdown(self) -> None:
        self.tracker_manager.stop()
