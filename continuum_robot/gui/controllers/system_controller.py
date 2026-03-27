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
    tracker_backend_identity: str = ""
    tracker_backend_running: bool = False
    tracker_backend_connected: bool = False
    openrb_connected: bool = False
    dynamixel_connected: bool = False
    openrb_status: str = "OpenRB disconnected."
    status_message: str = "System idle."
    last_error: str | None = None
    available_ports: list[SerialPortInfo] = field(default_factory=list)
    config_summary: str = ""


class SystemController:
    """Owns system-level connect/disconnect and setup actions."""

    def __init__(self, tracking_service, openrb_client, servo_service, settings: Settings) -> None:
        self.tracking_service = tracking_service
        self.openrb_client = openrb_client
        self.servo_service = servo_service
        self.settings = settings
        self.state = SystemViewState(
            mock_mode=settings.runtime.mock_mode,
            aurora_port=settings.serial.aurora_port,
            openrb_port=settings.serial.openrb_port,
            baudrate=settings.serial.baudrate,
            config_summary=self._build_config_summary(settings),
            status_message=self._initial_status_message(settings),
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
        self.tracking_service.set_port(port)

    def set_openrb_port(self, port: str) -> None:
        self.state.openrb_port = port

    def connect_tracker(self) -> None:
        try:
            if not self.settings.runtime.mock_mode and not self.state.aurora_port:
                raise RuntimeError("Aurora port is empty. Set the tracker port before connecting.")
            self.tracking_service.start(self.state.aurora_port)
            self.state.status_message = "Tracker connection requested."
            self.state.last_error = None
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Tracker connect failed: {exc}"
        self.refresh()

    def disconnect_tracker(self) -> None:
        try:
            self.tracking_service.stop()
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
        tracker_state = self.tracking_service.get_snapshot()
        self.state.tracker_connection_state = tracker_state.connection_state
        self.state.tracker_backend_identity = tracker_state.backend_identity
        self.state.tracker_backend_running = bool(
            tracker_state.backend_running
            if tracker_state.backend_running is not None
            else tracker_state.bridge_running
        )
        self.state.tracker_backend_connected = bool(
            tracker_state.backend_connected
            if tracker_state.backend_connected is not None
            else tracker_state.socket_connected
        )
        self.state.openrb_connected = self.openrb_client.is_connected
        self.state.dynamixel_connected = self.servo_service.is_connected
        self.state.openrb_status = self.openrb_client.last_status
        if tracker_state.last_error:
            self.state.last_error = tracker_state.last_error
        return self.state

    @staticmethod
    def _build_config_summary(settings: Settings) -> str:
        hardware_note = (
            "Servo hardware path: mock backend validated."
            if settings.runtime.mock_mode
            else "Servo hardware path: real OpenRB/DYNAMIXEL transport still pending."
        )
        return (
            f"Mode: {settings.robot.mode}\n"
            f"Servo IDs: {settings.robot.servo_ids}\n"
            f"Mock mode: {settings.runtime.mock_mode}\n"
            f"Tracker backend: {settings.serial.tracker_backend}\n"
            f"Tracker type: {settings.serial.tracker_type}\n"
            f"Tracker freshness timeout: {settings.serial.tracker_freshness_timeout_s}s\n"
            f"Runtime coil tool: {settings.registration.coil_tool_id}\n"
            f"Registration tool: {settings.registration.capture_tool_id}\n"
            f"Robot config: {settings.runtime.robot_config}\n"
            f"{hardware_note}"
        )

    @staticmethod
    def _initial_status_message(settings: Settings) -> str:
        if settings.runtime.mock_mode:
            return "Mock mode ready. Tracker, registration, and servo workflows can be exercised without hardware."
        return (
            "Hardware mode ready for Aurora tracker bring-up. "
            "OpenRB/DYNAMIXEL transport is still blocked pending implementation."
        )
