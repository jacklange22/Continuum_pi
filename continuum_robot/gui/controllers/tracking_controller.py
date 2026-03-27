"""Tracking tab controller backed by the shared tracking service."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from continuum_robot.config.settings import Settings


@dataclass
class TrackingViewState:
    """UI-facing snapshot of live tracker state."""

    device_path: str
    backend_identity: str = ""
    connection_state: str = "disconnected"
    backend_running: bool = False
    backend_connected: bool = False
    latest_frame_number: int | None = None
    latest_timestamp: str | None = None
    tracker_data_age_s: float | None = None
    tracker_data_stale: bool = False
    first_frame_latency_s: float | None = None
    last_status_message: str = ""
    last_error: str | None = None
    tools: dict[str, dict] = field(default_factory=dict)
    registration_path: str = ""
    tip_position_mm: tuple[float, float, float] | None = None
    tip_direction_xyz: tuple[float, float, float] | None = None
    tip_status: str = "Registration not loaded"
    runtime_coil_tool_id: str = "0A"
    registration_tool_id: str = "0B"


class TrackingController:
    """Owns live tool status and tip pose display updates."""

    def __init__(self, tracking_service, settings: Settings, registration_path: Path | None = None) -> None:
        self.tracking_service = tracking_service
        self.registration_path = registration_path or Path(settings.calibration.latest_registration_path)
        self.state = TrackingViewState(
            device_path=str(settings.serial.aurora_port),
            registration_path=str(self.registration_path),
        )

    def set_device_path(self, device_path: str) -> None:
        self.state.device_path = device_path
        self.tracking_service.set_port(device_path)

    def connect(self) -> None:
        try:
            self.tracking_service.start(self.state.device_path)
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.connection_state = "error"
            return
        self.refresh()

    def disconnect(self) -> None:
        try:
            self.tracking_service.stop()
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.connection_state = "error"
            return
        self.refresh()

    def refresh(self) -> TrackingViewState:
        snapshot = self.tracking_service.get_snapshot()
        self.state.backend_identity = snapshot.backend_identity
        self.state.connection_state = snapshot.connection_state
        self.state.backend_running = bool(
            snapshot.backend_running if snapshot.backend_running is not None else snapshot.bridge_running
        )
        self.state.backend_connected = bool(
            snapshot.backend_connected if snapshot.backend_connected is not None else snapshot.socket_connected
        )
        self.state.latest_frame_number = snapshot.last_frame_number
        self.state.latest_timestamp = snapshot.last_packet_utc
        self.state.tracker_data_age_s = snapshot.tracker_data_age_s
        self.state.tracker_data_stale = snapshot.tracker_data_stale
        self.state.first_frame_latency_s = snapshot.first_frame_latency_s
        self.state.last_status_message = snapshot.backend_status_message or snapshot.health.status
        self.state.last_error = snapshot.last_error
        self.state.runtime_coil_tool_id = snapshot.runtime_coil_tool_id
        self.state.registration_tool_id = snapshot.registration_tool_id
        self.state.tools = {
            tool_id: {
                "present": tool.present,
                "valid": tool.valid,
                "validity_known": tool.validity_known,
                "tracking_state": tool.tracking_state,
                "status": tool.status,
                "frame_number": tool.frame_number,
                "translation_mm": tool.translation_mm,
                "quaternion_wxyz": tool.quaternion_wxyz,
                "quality": tool.quality,
                "timestamp": tool.last_update_utc,
            }
            for tool_id, tool in snapshot.tools.items()
        }
        self._refresh_tip_pose(snapshot)
        return self.state

    def shutdown(self) -> None:
        self.tracking_service.stop()

    def _refresh_tip_pose(self, snapshot) -> None:
        self.state.tip_position_mm = None
        self.state.tip_direction_xyz = None
        self.state.tip_status = snapshot.tip_pose_status

        if snapshot.T_robot_tip is None:
            return

        self.state.tip_position_mm = tuple(float(v) for v in (snapshot.T_robot_tip[0][3], snapshot.T_robot_tip[1][3], snapshot.T_robot_tip[2][3]))
        self.state.tip_direction_xyz = tuple(
            float(v) for v in (snapshot.T_robot_tip[0][2], snapshot.T_robot_tip[1][2], snapshot.T_robot_tip[2][2])
        )
