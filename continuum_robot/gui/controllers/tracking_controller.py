"""Tracking tab controller wiring for tracker state and tip pose display."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from continuum_robot.config.settings import Settings
from continuum_robot.tracking.tip_pose_service import TipPoseService
from continuum_robot.tracking.transforms import make_transform_A_B


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
    registration_path: str = ""
    tip_position_mm: tuple[float, float, float] | None = None
    tip_direction_xyz: tuple[float, float, float] | None = None
    tip_status: str = "Registration not loaded"


class TrackingController:
    """Owns live tool status and tip pose display updates."""

    def __init__(self, tracker_manager, settings: Settings, registration_path: Path | None = None) -> None:
        self.tracker_manager = tracker_manager
        self.registration_path = registration_path or Path(settings.calibration.latest_registration_path)
        self._tip_pose_service: TipPoseService | None = None
        self._tip_pose_mtime_ns: int | None = None
        device_path = getattr(self.tracker_manager, "aurora_port", settings.serial.aurora_port)
        socket_path = getattr(self.tracker_manager, "socket_path", settings.serial.tracker_socket_path)
        self.state = TrackingViewState(
            device_path=str(device_path),
            socket_path=str(socket_path),
            registration_path=str(self.registration_path),
        )

    def set_device_path(self, device_path: str) -> None:
        self.state.device_path = device_path
        if hasattr(self.tracker_manager, "aurora_port"):
            self.tracker_manager.aurora_port = device_path

    def connect(self) -> None:
        try:
            if hasattr(self.tracker_manager, "aurora_port"):
                self.tracker_manager.aurora_port = self.state.device_path
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
        self._refresh_tip_pose(snapshot.tools.get("0A"))
        return self.state

    def shutdown(self) -> None:
        self.tracker_manager.stop()

    def _refresh_tip_pose(self, tool_0a) -> None:
        self.state.tip_position_mm = None
        self.state.tip_direction_xyz = None
        self.state.tip_status = "Registration not loaded"

        tip_service = self._load_tip_pose_service()
        if tip_service is None:
            return
        if tool_0a is None:
            self.state.tip_status = "Tool 0A is unavailable."
            return
        if not tool_0a.valid:
            self.state.tip_status = f"Tool 0A is invalid: {tool_0a.status}"
            return
        try:
            T_robot_tip = tip_service.compute_T_robot_tip(
                T_robot_aurora=tip_service.inputs.T_robot_aurora,
                T_aurora_coil=make_transform_A_B(tool_0a.quaternion, tool_0a.translation_mm),
                T_coil_tip=tip_service.inputs.T_coil_tip,
            )
            self.state.tip_position_mm = tuple(float(v) for v in T_robot_tip[0:3, 3])
            self.state.tip_direction_xyz = tuple(float(v) for v in T_robot_tip[0:3, 2])
            self.state.tip_status = "T_robot_tip is valid."
        except Exception as exc:
            self.state.tip_status = f"T_robot_tip unavailable: {exc}"

    def _load_tip_pose_service(self) -> TipPoseService | None:
        if not self.registration_path.exists():
            return None
        stat = self.registration_path.stat()
        if self._tip_pose_service is not None and self._tip_pose_mtime_ns == stat.st_mtime_ns:
            return self._tip_pose_service
        try:
            self._tip_pose_service = TipPoseService.from_registration_file(self.registration_path)
            self._tip_pose_mtime_ns = stat.st_mtime_ns
            return self._tip_pose_service
        except Exception as exc:
            self.state.tip_status = f"Registration invalid: {exc}"
            self._tip_pose_service = None
            self._tip_pose_mtime_ns = None
            return None
