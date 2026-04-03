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
    bus_reachable: bool = False
    motion_ready: bool = False
    external_power_ready: bool | None = None
    openrb_status: str = "OpenRB disconnected."
    readiness_message: str = "OpenRB readiness not checked."
    status_message: str = "System idle."
    last_error: str | None = None
    available_ports: list[SerialPortInfo] = field(default_factory=list)
    available_robot_configs: list[str] = field(default_factory=list)
    robot_config: str = "robot_4servo.yaml"
    robot_mode: str = ""
    expected_servo_ids: list[int] = field(default_factory=list)
    fine_jog_step_ticks: int = 5
    coarse_jog_step_ticks: int = 25
    position_min_offset_ticks: int = -600
    position_max_offset_ticks: int = 600
    software_position_margin_ticks: int = 64
    telemetry_freshness_timeout_s: float = 0.25
    pretension_threshold_ma: int = 220
    tightening_direction_default: str = "cw"
    bench_debug_text: str = ""
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
            expected_servo_ids=list(settings.robot.servo_ids),
            fine_jog_step_ticks=settings.safety.fine_jog_step_ticks,
            coarse_jog_step_ticks=settings.safety.coarse_jog_step_ticks,
            position_min_offset_ticks=settings.safety.position_min_offset_ticks,
            position_max_offset_ticks=settings.safety.position_max_offset_ticks,
            software_position_margin_ticks=settings.safety.software_position_margin_ticks,
            telemetry_freshness_timeout_s=settings.safety.telemetry_stale_after_s,
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
                "Use configured-servo bring-up next and jog one selected servo at a time."
            )
            self.state.last_error = None
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"OpenRB connect failed: {exc}"
        self.refresh_readiness()

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
        self.refresh_readiness()

    def refresh_readiness(self) -> SystemViewState:
        if not self.servo_service.is_connected:
            self.state.readiness_message = "DYNAMIXEL bus disconnected."
            self.state.bench_debug_text = self._build_disconnected_bench_debug_text()
            return self.refresh()
        try:
            if self.settings.robot.mode == "1-servo" or len(self.settings.robot.servo_ids) == 1:
                debug_snapshot = self.servo_service.build_bench_debug_snapshot(self._expected_servo_id())
                self.state.bus_reachable = bool(debug_snapshot.bus_reachable)
                self.state.motion_ready = bool(debug_snapshot.motion_ready)
                self.state.external_power_ready = (
                    debug_snapshot.motion_assessment.external_power_ready
                    if debug_snapshot.motion_assessment is not None
                    else None
                )
                self.state.readiness_message = debug_snapshot.message
                self.state.bench_debug_text = self._build_bench_debug_text(debug_snapshot)
                self.state.last_error = (
                    debug_snapshot.message
                    if debug_snapshot.ping_ok is False
                    or debug_snapshot.identity_read_ok is False
                    or debug_snapshot.telemetry_read_ok is False
                    else None
                )
            else:
                snapshot = self.servo_service.build_configured_servo_bringup_snapshot(
                    list(self.settings.robot.servo_ids),
                    allow_scan=True,
                )
                self.state.bus_reachable = bool(snapshot.bus_reachable)
                self.state.motion_ready = bool(snapshot.all_motion_ready)
                external_power_flags = [
                    entry.motion_assessment.external_power_ready
                    for entry in snapshot.servo_entries.values()
                    if entry.motion_assessment is not None
                    and entry.motion_assessment.external_power_ready is not None
                ]
                if not external_power_flags:
                    self.state.external_power_ready = None
                else:
                    self.state.external_power_ready = all(bool(flag) for flag in external_power_flags)
                self.state.readiness_message = snapshot.message
                self.state.bench_debug_text = self._build_configured_servo_bringup_debug_text(snapshot)
                self.state.last_error = None if snapshot.status == "ready" else snapshot.message
        except Exception as exc:
            self.state.bus_reachable = False
            self.state.motion_ready = False
            self.state.external_power_ready = None
            self.state.last_error = str(exc)
            self.state.readiness_message = f"Readiness refresh failed: {exc}"
            self.state.bench_debug_text = self._build_disconnected_bench_debug_text(extra_error=str(exc))
        return self.refresh()

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
        openrb_snapshot = self.openrb_client.get_status_snapshot()
        self.state.openrb_connected = openrb_snapshot.connected
        self.state.openrb_prepared = openrb_snapshot.prepared
        self.state.dynamixel_connected = self.servo_service.is_connected
        self.state.openrb_status = openrb_snapshot.message
        self.state.expected_servo_ids = list(self.settings.robot.servo_ids)
        if not self.state.dynamixel_connected:
            self.state.bus_reachable = False
            self.state.motion_ready = False
            self.state.external_power_ready = None
            self.state.readiness_message = "Connect OpenRB and refresh readiness."
            self.state.bench_debug_text = self._build_disconnected_bench_debug_text()
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
        position_min_offset_ticks: int,
        position_max_offset_ticks: int,
        software_position_margin_ticks: int,
        telemetry_freshness_timeout_s: float,
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
            if position_min_offset_ticks > 0 or position_max_offset_ticks < 0:
                raise ValueError("Neutral-centered bounds must straddle zero offset.")
            if position_min_offset_ticks >= position_max_offset_ticks:
                raise ValueError("Minimum offset must be less than maximum offset.")
            if software_position_margin_ticks < 0:
                raise ValueError("Software position margin must be non-negative.")
            if telemetry_freshness_timeout_s <= 0:
                raise ValueError("Telemetry freshness timeout must be positive.")
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
                    "position_min_offset_ticks": int(position_min_offset_ticks),
                    "position_max_offset_ticks": int(position_max_offset_ticks),
                    "software_position_margin_ticks": int(software_position_margin_ticks),
                    "telemetry_stale_after_s": float(telemetry_freshness_timeout_s),
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
            self.state.expected_servo_ids = list(robot.servo_ids)
            self.state.fine_jog_step_ticks = int(fine_jog_step_ticks)
            self.state.coarse_jog_step_ticks = int(coarse_jog_step_ticks)
            self.state.position_min_offset_ticks = int(position_min_offset_ticks)
            self.state.position_max_offset_ticks = int(position_max_offset_ticks)
            self.state.software_position_margin_ticks = int(software_position_margin_ticks)
            self.state.telemetry_freshness_timeout_s = float(telemetry_freshness_timeout_s)
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
            f"Application bounds metadata: {settings.safety.position_min_offset_ticks}..{settings.safety.position_max_offset_ticks} ticks\n"
            f"Software margin: {settings.safety.software_position_margin_ticks} ticks\n"
            f"Telemetry freshness timeout: {settings.safety.telemetry_stale_after_s}s\n"
            f"Minimum motion voltage: {settings.safety.min_input_voltage_mv} mV\n"
            f"Default pretension threshold: {settings.safety.default_pretension_current_threshold_ma} mA\n"
            f"DYNAMIXEL protocol version: {settings.serial.dynamixel_settings.get('protocol_version', 2.0)}\n"
            "Bring-up raw position convention: 0 = more tensioned, 4095 = untensioned, tighten -> smaller counts\n"
            f"{hardware_note}"
        )

    @staticmethod
    def _initial_status_message(settings: Settings) -> str:
        if settings.runtime.mock_mode:
            return "Mock mode ready. Tracker, registration, and servo workflows can be exercised without hardware."
        return (
            "Hardware mode ready. Connect OpenRB, prepare the board for DYNAMIXEL pass-through, "
            "refresh readiness, verify the configured servos, then jog one selected servo at a time."
        )

    def _build_live_config_summary(self) -> str:
        return (
            f"Robot config: {self.state.robot_config}\n"
            f"Robot mode: {self.state.robot_mode}\n"
            f"Expected servo IDs: {self.state.expected_servo_ids}\n"
            f"Mock mode: {self.state.mock_mode}\n"
            f"OpenRB port: {self.state.openrb_port}\n"
            f"Baudrate: {self.state.baudrate}\n"
            f"Fine/coarse jog: {self.state.fine_jog_step_ticks}/{self.state.coarse_jog_step_ticks} ticks\n"
            f"Application bounds metadata: {self.state.position_min_offset_ticks}/{self.state.position_max_offset_ticks} ticks\n"
            f"Software margin: {self.state.software_position_margin_ticks} ticks\n"
            f"Telemetry freshness timeout: {self.state.telemetry_freshness_timeout_s}s\n"
            f"Default pretension threshold: {self.state.pretension_threshold_ma} mA\n"
            f"Wrap direction metadata: {self.state.tightening_direction_default}\n"
            "Bring-up raw position convention: tighten -> smaller counts, loosen -> larger counts\n"
            f"OpenRB prepared: {self.state.openrb_prepared}\n"
            f"Bus reachable: {self.state.bus_reachable}\n"
            f"Motion ready: {self.state.motion_ready}\n"
            f"Saved overrides: {self.state.saved_overrides_path or 'none'}"
        )

    @staticmethod
    def _default_tightening_direction(settings: Settings) -> str:
        if settings.robot.tightening_rotation_by_servo:
            first_key = sorted(settings.robot.tightening_rotation_by_servo)[0]
            return str(settings.robot.tightening_rotation_by_servo[first_key]).strip().lower()
        return "cw"

    def _expected_servo_id(self) -> int | None:
        if self.settings.robot.servo_ids:
            return int(self.settings.robot.servo_ids[0])
        return None

    def _build_bench_debug_text(self, snapshot) -> str:
        openrb_connected = self.openrb_client.get_status_snapshot().connected
        telemetry = snapshot.telemetry
        last_hw_error = None
        if telemetry is not None:
            last_hw_error = telemetry.hardware_error or (
                f"0x{telemetry.hardware_error_code:02X}"
                if telemetry.hardware_error_code not in (None, 0)
                else "0"
            )
        return "\n".join(
            [
                "Bench debug:",
                f"openrb_connected={openrb_connected}",
                f"selected_port={snapshot.selected_port or self.state.openrb_port or 'unset'}",
                f"selected_baud={snapshot.selected_baud or self.state.baudrate}",
                f"expected_servo_id={snapshot.expected_servo_id}",
                f"ping_ok={snapshot.ping_ok}",
                f"identity_read_ok={snapshot.identity_read_ok}",
                f"telemetry_read_ok={snapshot.telemetry_read_ok}",
                f"last_position={telemetry.present_position if telemetry is not None else None}",
                f"last_current={telemetry.present_current_ma if telemetry is not None else None}",
                f"last_voltage={telemetry.present_voltage_mv if telemetry is not None else None}",
                f"last_temperature={telemetry.present_temperature_c if telemetry is not None else None}",
                f"last_hw_error={last_hw_error}",
                f"calibration_entries_loaded={snapshot.calibration_entries_loaded}",
                f"one_servo_mode_ok={snapshot.one_servo_mode_ok}",
                f"active_range={self.servo_service.raw_position_range()[0]}..{self.servo_service.raw_position_range()[1]}",
                f"motion_ready={snapshot.motion_ready}",
                "position_convention=tighten->smaller_counts; loosen->larger_counts",
                f"motion_block_reason={snapshot.motion_block_reason or 'none'}",
            ]
        )

    def _build_configured_servo_bringup_debug_text(self, snapshot) -> str:
        openrb_connected = self.openrb_client.get_status_snapshot().connected
        motion_ready_ids = [
            entry.servo_id
            for entry in snapshot.servo_entries.values()
            if entry.motion_assessment is not None and entry.motion_assessment.ready
        ]
        telemetry_ok_ids = [
            entry.servo_id for entry in snapshot.servo_entries.values() if entry.telemetry_read_ok
        ]
        return "\n".join(
            [
                "Bench debug:",
                f"openrb_connected={openrb_connected}",
                f"selected_port={snapshot.selected_port or self.state.openrb_port or 'unset'}",
                f"selected_baud={snapshot.selected_baud or self.state.baudrate}",
                f"expected_servo_ids={snapshot.expected_servo_ids}",
                f"discovered_ids={snapshot.discovered_ids}",
                f"missing_servo_ids={snapshot.missing_servo_ids}",
                f"unexpected_servo_ids={snapshot.unexpected_servo_ids}",
                f"telemetry_ok_ids={telemetry_ok_ids}",
                f"motion_ready_ids={motion_ready_ids}",
                f"all_expected_present={snapshot.all_expected_present}",
                f"all_expected_telemetry_ok={snapshot.all_expected_telemetry_ok}",
                f"all_motion_ready={snapshot.all_motion_ready}",
                f"active_range={self.servo_service.raw_position_range()[0]}..{self.servo_service.raw_position_range()[1]}",
                "position_convention=tighten->smaller_counts; loosen->larger_counts",
            ]
        )

    def _build_disconnected_bench_debug_text(self, *, extra_error: str | None = None) -> str:
        summary = self.servo_service.get_calibration_summary()
        openrb_connected = self.openrb_client.get_status_snapshot().connected
        lines = [
            "Bench debug:",
            f"openrb_connected={openrb_connected}",
            f"selected_port={self.state.openrb_port or 'unset'}",
            f"selected_baud={self.state.baudrate}",
            f"expected_servo_ids={self.state.expected_servo_ids}",
            "ping_ok=None",
            "identity_read_ok=None",
            "telemetry_read_ok=None",
            "last_position=None",
            "last_current=None",
            "last_voltage=None",
            "last_temperature=None",
            "last_hw_error=None",
            f"calibration_entries_loaded={sorted(summary.servo_entries)}",
            f"one_servo_mode_ok={self.settings.robot.mode == '1-servo' and len(self.settings.robot.servo_ids) == 1}",
            f"active_range={self.servo_service.raw_position_range()[0]}..{self.servo_service.raw_position_range()[1]}",
            "motion_ready=False",
            "position_convention=tighten->smaller_counts; loosen->larger_counts",
            f"motion_block_reason={extra_error or 'OpenRB/DYNAMIXEL not ready'}",
        ]
        return "\n".join(lines)
