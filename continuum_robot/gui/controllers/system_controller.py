"""System tab controller for top-level connectivity and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

from continuum_robot.config.settings import Settings
from continuum_robot.hardware.serial_ports import SerialPortInfo, discover_serial_ports


@dataclass
class SystemViewState:
    """UI-facing system status."""

    mock_mode: bool
    aurora_port: str
    openrb_port: str
    baudrate: int
    tracker_connection_state: str = "disconnected"
    tracker_bridge_running: bool = False
    tracker_socket_connected: bool = False
    openrb_connected: bool = False
    dynamixel_connected: bool = False
    openrb_status: str = "OpenRB disconnected."
    status_message: str = "System idle."
    last_error: str | None = None
    available_ports: list[SerialPortInfo] = field(default_factory=list)
    config_summary: str = ""


class SystemController:
    """Owns system-level connect/disconnect and setup actions."""

    def __init__(self, tracker_manager, openrb_client, servo_service, settings: Settings) -> None:
        self.tracker_manager = tracker_manager
        self.openrb_client = openrb_client
        self.servo_service = servo_service
        self.settings = settings
        self.state = SystemViewState(
            mock_mode=settings.runtime.mock_mode,
            aurora_port=settings.serial.aurora_port,
            openrb_port=settings.serial.openrb_port,
            baudrate=settings.serial.baudrate,
            config_summary=self._build_config_summary(settings),
        )
        self.rescan_ports()

    def rescan_ports(self) -> SystemViewState:
        ports = discover_serial_ports()
        if self.settings.runtime.mock_mode:
            ports.extend(
                [
                    SerialPortInfo(device="/dev/mock-aurora", description="Mock Aurora port"),
                    SerialPortInfo(device="/dev/mock-openrb", description="Mock OpenRB port"),
                ]
            )
        deduped = {port.device: port for port in ports}
        self.state.available_ports = sorted(deduped.values(), key=lambda port: port.device)
        return self.refresh()

    def set_aurora_port(self, port: str) -> None:
        self.state.aurora_port = port
        if hasattr(self.tracker_manager, "aurora_port"):
            self.tracker_manager.aurora_port = port

    def set_openrb_port(self, port: str) -> None:
        self.state.openrb_port = port

    def connect_tracker(self) -> None:
        try:
            if not self.settings.runtime.mock_mode and not self.state.aurora_port:
                raise RuntimeError("Aurora port is empty. Set the tracker port before connecting.")
            if hasattr(self.tracker_manager, "aurora_port"):
                self.tracker_manager.aurora_port = self.state.aurora_port
            self.tracker_manager.start()
            self.state.status_message = "Tracker connection requested."
            self.state.last_error = None
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Tracker connect failed: {exc}"
        self.refresh()

    def disconnect_tracker(self) -> None:
        try:
            self.tracker_manager.stop()
            self.state.status_message = "Tracker disconnected."
            self.state.last_error = None
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Tracker disconnect failed: {exc}"
        self.refresh()

    def connect_openrb(self) -> None:
        try:
            if not self.state.openrb_port:
                raise RuntimeError("OpenRB port is empty. Set the board port before connecting.")
            self.openrb_client.connect(self.state.openrb_port, self.state.baudrate)
            try:
                self.servo_service.connect(self.state.openrb_port, self.state.baudrate)
            except Exception:
                self.openrb_client.disconnect()
                raise
            self.state.status_message = "OpenRB and DYNAMIXEL bus connected."
            self.state.last_error = None
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"OpenRB connect failed: {exc}"
        self.refresh()

    def disconnect_openrb(self) -> None:
        try:
            self.openrb_client.disconnect()
            self.servo_service.disconnect()
            self.state.status_message = "OpenRB and DYNAMIXEL bus disconnected."
            self.state.last_error = None
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"OpenRB disconnect failed: {exc}"
        self.refresh()

    def prepare_openrb(self) -> None:
        try:
            self.openrb_client.prepare_for_dynamixel_use()
            self.state.status_message = "OpenRB prepared for DYNAMIXEL use."
            self.state.last_error = None
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"OpenRB prepare failed: {exc}"
        self.refresh()

    def refresh(self) -> SystemViewState:
        tracker_state = self.tracker_manager.get_state_snapshot()
        self.state.tracker_connection_state = tracker_state.connection_state
        self.state.tracker_bridge_running = tracker_state.bridge_running
        self.state.tracker_socket_connected = tracker_state.socket_connected
        self.state.openrb_connected = self.openrb_client.is_connected
        self.state.dynamixel_connected = self.servo_service.is_connected
        self.state.openrb_status = self.openrb_client.last_status
        if tracker_state.last_error:
            self.state.last_error = tracker_state.last_error
        return self.state

    @staticmethod
    def _build_config_summary(settings: Settings) -> str:
        return (
            f"Mode: {settings.robot.mode}\n"
            f"Servo IDs: {settings.robot.servo_ids}\n"
            f"Mock mode: {settings.runtime.mock_mode}\n"
            f"Tracker socket: {settings.serial.tracker_socket_path}\n"
            f"Robot config: {settings.runtime.robot_config}"
        )
