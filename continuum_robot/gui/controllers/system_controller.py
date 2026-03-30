"""System tab controller for top-level connectivity and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

from continuum_robot.config.config_loader import ConfigLoader
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
    openrb_prepared: bool = False
    dynamixel_connected: bool = False
    openrb_status: str = "OpenRB disconnected."
    status_message: str = "System idle."
    last_error: str | None = None
    available_ports: list[SerialPortInfo] = field(default_factory=list)
    available_robot_configs: list[str] = field(default_factory=list)
    robot_config: str = "robot_4servo.yaml"
    robot_mode: str = ""
    fine_jog_step_ticks: int = 5
    coarse_jog_step_ticks: int = 25
    pretension_threshold_ma: int = 220
    tightening_direction_default: str = "cw"
    saved_overrides_path: str = ""
    config_summary: str = ""


class SystemController:
    """Owns system-level connect/disconnect and setup actions."""

    def __init__(
        self,
        tracking_service,
        openrb_client,
        servo_service,
        settings: Settings,
        *,
        config_loader: ConfigLoader | None = None,
    ) -> None:
        self.tracking_service = tracking_service
        self.openrb_client = openrb_client
        self.servo_service = servo_service
        self.settings = settings
        self.config_loader = config_loader
        self.state = SystemViewState(
            mock_mode=settings.runtime.mock_mode,
            aurora_port=settings.serial.aurora_port,
            openrb_port=settings.serial.openrb_port,
            baudrate=settings.serial.baudrate,
            available_robot_configs=(
                config_loader.list_robot_configs() if config_loader is not None else [settings.runtime.robot_config]
            ),
            robot_config=settings.runtime.robot_config,
            robot_mode=settings.robot.mode,
            fine_jog_step_ticks=settings.safety.fine_jog_step_ticks,
            coarse_jog_step_ticks=settings.safety.coarse_jog_step_ticks,
            pretension_threshold_ma=settings.safety.default_pretension_current_threshold_ma,
            tightening_direction_default=self._default_tightening_direction(settings),
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
            # Keep board validation and bus ownership on the canonical path so
            # GUI status always reflects the real OpenRB + DYNAMIXEL state.
            self.openrb_client.connect(self.state.openrb_port, self.state.baudrate)
            self.openrb_client.prepare_for_dynamixel_use()
            try:
                self.servo_service.connect(self.state.openrb_port, self.state.baudrate)
            except Exception:
                self.openrb_client.disconnect()
                raise
            self.state.status_message = (
                "OpenRB validated and prepared; DYNAMIXEL bus connected. "
                "Use one-servo bring-up before attempting full robot motion."
            )
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
        self.state.openrb_prepared = bool(getattr(self.openrb_client, "is_prepared", False))
        self.state.dynamixel_connected = self.servo_service.is_connected
        self.state.openrb_status = self.openrb_client.last_status
        if tracker_state.last_error:
            self.state.last_error = tracker_state.last_error
        self.state.config_summary = self._build_live_config_summary()
        return self.state

    def save_runtime_parameters(
        self,
        *,
        mock_mode: bool,
        robot_config: str,
        openrb_port: str,
        baudrate: int,
        fine_jog_step_ticks: int,
        coarse_jog_step_ticks: int,
        pretension_threshold_ma: int,
        tightening_direction: str,
    ) -> None:
        try:
            if self.config_loader is None:
                raise RuntimeError("Config loader is unavailable; runtime parameter editing is disabled.")
            if fine_jog_step_ticks <= 0 or coarse_jog_step_ticks <= 0:
                raise ValueError("Jog increments must be positive.")
            if fine_jog_step_ticks > coarse_jog_step_ticks:
                raise ValueError("Fine jog increment must be less than or equal to coarse jog increment.")
            if pretension_threshold_ma <= 0:
                raise ValueError("Pretension threshold must be positive.")
            robot = self.config_loader.load_robot_config(robot_config)
            overrides = {
                "mock_mode": bool(mock_mode),
                "robot_config": str(robot_config),
                "openrb_port": str(openrb_port).strip(),
                "baudrate": int(baudrate),
                "safety_overrides": {
                    "fine_jog_step_ticks": int(fine_jog_step_ticks),
                    "coarse_jog_step_ticks": int(coarse_jog_step_ticks),
                    "default_pretension_current_threshold_ma": int(pretension_threshold_ma),
                },
                "robot_overrides": {
                    "tightening_rotation_by_servo": {
                        str(servo_id): str(tightening_direction).strip().lower()
                        for servo_id in robot.servo_ids
                    }
                },
            }
            path = self.config_loader.save_system_local_overrides(overrides)
            self.state.mock_mode = bool(mock_mode)
            self.state.robot_config = str(robot_config)
            self.state.robot_mode = robot.mode
            self.state.openrb_port = str(openrb_port).strip()
            self.state.baudrate = int(baudrate)
            self.state.fine_jog_step_ticks = int(fine_jog_step_ticks)
            self.state.coarse_jog_step_ticks = int(coarse_jog_step_ticks)
            self.state.pretension_threshold_ma = int(pretension_threshold_ma)
            self.state.tightening_direction_default = str(tightening_direction).strip().lower()
            self.state.saved_overrides_path = str(path)
            self.state.status_message = (
                f"Saved runtime parameters to {path}. Restart the app or reconnect services before relying on the new values."
            )
            self.state.last_error = None
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Save runtime parameters failed: {exc}"
            self.refresh()
            raise
        self.refresh()

    @staticmethod
    def _build_config_summary(settings: Settings) -> str:
        hardware_note = (
            "Servo hardware path: mock backend validated."
            if settings.runtime.mock_mode
            else "Servo hardware path: real OpenRB serial validation and DYNAMIXEL SDK transport enabled."
        )
        return (
            f"Mode: {settings.robot.mode}\n"
            f"Servo IDs: {settings.robot.servo_ids}\n"
            f"Mock mode: {settings.runtime.mock_mode}\n"
            f"Tracker backend: {settings.serial.tracker_backend}\n"
            f"Tracker fallback backend: {settings.serial.tracker_fallback_backend}\n"
            f"Tracker fallback enabled: {settings.serial.tracker_fallback_enabled}\n"
            f"Tracker type: {settings.serial.tracker_type}\n"
            f"Tracker freshness timeout: {settings.serial.tracker_freshness_timeout_s}s\n"
            f"Runtime coil tool: {settings.registration.coil_tool_id}\n"
            f"Registration tool: {settings.registration.capture_tool_id}\n"
            f"Robot config: {settings.runtime.robot_config}\n"
            f"Robot mode: {settings.robot.mode}\n"
            f"Fine/coarse jog: {settings.safety.fine_jog_step_ticks}/{settings.safety.coarse_jog_step_ticks} ticks\n"
            f"Default pretension threshold: {settings.safety.default_pretension_current_threshold_ma} mA\n"
            f"DYNAMIXEL protocol version: {settings.serial.dynamixel_settings.get('protocol_version', 2.0)}\n"
            f"{hardware_note}"
        )

    @staticmethod
    def _initial_status_message(settings: Settings) -> str:
        if settings.runtime.mock_mode:
            return "Mock mode ready. Tracker, registration, and servo workflows can be exercised without hardware."
        return (
            "Hardware mode ready. Connect OpenRB, prepare the board for DYNAMIXEL pass-through, "
            "then scan and validate servo telemetry."
        )

    def _build_live_config_summary(self) -> str:
        return (
            f"Robot config: {self.state.robot_config}\n"
            f"Robot mode: {self.state.robot_mode}\n"
            f"Mock mode: {self.state.mock_mode}\n"
            f"OpenRB port: {self.state.openrb_port}\n"
            f"Baudrate: {self.state.baudrate}\n"
            f"Fine/coarse jog: {self.state.fine_jog_step_ticks}/{self.state.coarse_jog_step_ticks} ticks\n"
            f"Default pretension threshold: {self.state.pretension_threshold_ma} mA\n"
            f"Tightening direction default: {self.state.tightening_direction_default}\n"
            f"OpenRB prepared: {self.state.openrb_prepared}\n"
            f"Saved overrides: {self.state.saved_overrides_path or 'none'}"
        )

    @staticmethod
    def _default_tightening_direction(settings: Settings) -> str:
        if settings.robot.tightening_rotation_by_servo:
            first_key = sorted(settings.robot.tightening_rotation_by_servo)[0]
            return str(settings.robot.tightening_rotation_by_servo[first_key]).strip().lower()
        return "cw"
