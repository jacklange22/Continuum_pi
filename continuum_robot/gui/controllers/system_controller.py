"""System tab controller for top-level connectivity and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from continuum_robot.config.config_loader import ConfigLoader
from continuum_robot.config.settings import Settings
from continuum_robot.hardware.serial_ports import SerialPortInfo, discover_serial_ports
from continuum_robot.servos.servo_service import ServoBusBusyError


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
    registration_summary: str = "Not loaded."
    runtime_tip_summary: str = "Not loaded."
    live_tip_summary: str = "Blocked."
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
    detected_servo_ids: list[int] = field(default_factory=list)
    telemetry_ready_count: int = 0
    motion_ready_count: int = 0
    poll_rate_hz: int = 10
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
            poll_rate_hz=settings.runtime.poll_rate_hz,
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
            saved_overrides_path=self._existing_overrides_path(config_loader),
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
            self.servo_service.disconnect()
            self.openrb_client.disconnect()
            self.state.status_message = "OpenRB and DYNAMIXEL bus disconnected."
            self.state.last_error = None
        except Exception as exc:
            try:
                self.openrb_client.disconnect()
            except Exception:
                pass
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
                self.state.detected_servo_ids = (
                    [int(debug_snapshot.selected_servo_id)]
                    if debug_snapshot.ping_ok and debug_snapshot.selected_servo_id is not None
                    else []
                )
                self.state.telemetry_ready_count = 1 if debug_snapshot.telemetry_read_ok else 0
                self.state.motion_ready_count = 1 if debug_snapshot.motion_ready else 0
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
                snapshot = self.servo_service.build_runtime_servo_snapshot(
                    list(self.settings.robot.servo_ids),
                )
                self.state.detected_servo_ids = list(snapshot.detected_servo_ids)
                self.state.telemetry_ready_count = int(snapshot.telemetry_ready_count)
                self.state.motion_ready_count = int(snapshot.motion_ready_count)
                self.state.bus_reachable = bool(snapshot.detected_servo_ids)
                self.state.motion_ready = bool(snapshot.all_motion_ready)
                external_power_flags = [
                    entry.motion_assessment.external_power_ready
                    for entry in snapshot.entries.values()
                    if entry.motion_assessment is not None and entry.motion_assessment.external_power_ready is not None
                ]
                if not external_power_flags:
                    self.state.external_power_ready = None
                else:
                    self.state.external_power_ready = all(bool(flag) for flag in external_power_flags)
                self.state.readiness_message = snapshot.message
                self.state.bench_debug_text = self._build_runtime_servo_debug_text(snapshot)
                self.state.last_error = None if snapshot.all_motion_ready else snapshot.message
        except ServoBusBusyError as exc:
            self.state.readiness_message = str(exc)
            self.state.status_message = str(exc)
            self.state.last_error = None
        except Exception as exc:
            self.state.bus_reachable = False
            self.state.motion_ready = False
            self.state.external_power_ready = None
            self.state.detected_servo_ids = []
            self.state.telemetry_ready_count = 0
            self.state.motion_ready_count = 0
            self.state.last_error = str(exc)
            self.state.readiness_message = f"Readiness refresh failed: {exc}"
            self.state.bench_debug_text = self._build_disconnected_bench_debug_text(extra_error=str(exc))
        return self.refresh()

    def sync_servo_runtime_snapshot(self, snapshot) -> SystemViewState:
        """Keep System-tab servo readiness aligned with the canonical live servo snapshot."""
        self.state.expected_servo_ids = [int(servo_id) for servo_id in getattr(snapshot, "expected_servo_ids", [])]
        self.state.detected_servo_ids = [int(servo_id) for servo_id in getattr(snapshot, "detected_servo_ids", [])]
        self.state.bus_reachable = bool(self.state.detected_servo_ids)
        self.state.telemetry_ready_count = int(getattr(snapshot, "telemetry_ready_count", 0))
        self.state.motion_ready_count = int(getattr(snapshot, "motion_ready_count", 0))
        self.state.motion_ready = bool(getattr(snapshot, "all_motion_ready", False))
        external_power_flags = [
            entry.motion_assessment.external_power_ready
            for entry in getattr(snapshot, "entries", {}).values()
            if entry.motion_assessment is not None and entry.motion_assessment.external_power_ready is not None
        ]
        self.state.external_power_ready = (
            all(bool(flag) for flag in external_power_flags) if external_power_flags else None
        )
        self.state.readiness_message = str(getattr(snapshot, "message", self.state.readiness_message))
        self.state.bench_debug_text = self._build_runtime_servo_debug_text(snapshot)
        if self.state.motion_ready:
            self.state.last_error = None
        return self.state

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
        self.state.registration_summary = self._registration_summary(tracker_state)
        self.state.runtime_tip_summary = self._runtime_tip_summary(tracker_state)
        self.state.live_tip_summary = self._live_tip_summary(tracker_state)
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
            self.state.detected_servo_ids = []
            self.state.telemetry_ready_count = 0
            self.state.motion_ready_count = 0
            self.state.readiness_message = "Connect OpenRB and refresh readiness."
            self.state.bench_debug_text = self._build_disconnected_bench_debug_text()
        if tracker_state.last_error:
            self.state.last_error = tracker_state.last_error
        self.state.config_summary = self._build_live_config_summary()
        return self.state

    @staticmethod
    def _registration_summary(tracker_state) -> str:
        state = str(getattr(tracker_state, "registration_state", "missing_registration"))
        timestamp = getattr(tracker_state, "stored_registration_timestamp_utc", None)
        fre = getattr(tracker_state, "stored_registration_fre_mm", None)
        if state == "loaded":
            parts = ["Loaded"]
            if fre is not None:
                parts.append(f"FRE {float(fre):.3f} mm")
            if timestamp:
                parts.append(str(timestamp))
            return " | ".join(parts)
        if state == "invalid_registration":
            return "Warning | invalid registration artifact"
        if state == "missing_registration":
            return "Not loaded"
        return str(state).replace("_", " ")

    @staticmethod
    def _runtime_tip_summary(tracker_state) -> str:
        state = str(getattr(tracker_state, "runtime_tip_calibration_state", "missing_runtime_tip_calibration"))
        mode = str(getattr(tracker_state, "runtime_tip_mode", "latest_accepted"))
        trust = str(getattr(tracker_state, "runtime_tip_trust_level", "missing"))
        timestamp = getattr(tracker_state, "stored_runtime_tip_timestamp_utc", None)
        if state == "loaded":
            detail = f" | {timestamp}" if timestamp else ""
            return f"{mode.replace('_', ' ')} | {trust.replace('_', ' ')}{detail}"
        if state == "quick_4_point_loaded":
            detail = f" | {timestamp}" if timestamp else ""
            return f"quick 4 point | {trust.replace('_', ' ')}{detail}"
        if state == "coil_as_tip":
            return "coil as tip | fallback debug"
        if state == "identity_tip_fallback":
            return "latest accepted | fallback debug"
        if state in {"missing_runtime_tip_calibration", "missing_quick_4_point_runtime_tip"}:
            return f"{mode.replace('_', ' ')} | not loaded"
        if state == "invalid_runtime_tip_calibration":
            return f"{mode.replace('_', ' ')} | invalid"
        return f"{mode.replace('_', ' ')} | {str(state).replace('_', ' ')}"

    @staticmethod
    def _live_tip_summary(tracker_state) -> str:
        tip_pose_status = str(getattr(tracker_state, "tip_pose_status", "missing_registration"))
        if tip_pose_status == "ok":
            age_s = getattr(tracker_state, "tracker_data_age_s", None)
            if getattr(tracker_state, "tracker_data_stale", False):
                return f"Stale | {float(age_s):.3f} s" if age_s is not None else "Stale"
            return "Ready"
        if tip_pose_status == "missing_registration":
            return "Blocked | registration not loaded"
        if tip_pose_status == "identity_tip_fallback":
            return "Warning | runtime tip fallback"
        if tip_pose_status == "invalid_runtime_tip_calibration":
            return "Blocked | invalid runtime tip artifact"
        return str(tip_pose_status).replace("_", " ")

    def sync_servo_bringup_state(self, servo_state) -> SystemViewState:
        """Keep System-tab servo readiness summary aligned with the canonical servo controller state."""
        snapshot = getattr(servo_state, "latest_runtime_snapshot", None)
        if snapshot is not None:
            return self.sync_servo_runtime_snapshot(snapshot)
        expected_ids = [int(servo_id) for servo_id in getattr(servo_state, "expected_servo_ids", [])]
        detected_ids = [int(servo_id) for servo_id in getattr(servo_state, "detected_servo_ids", [])]
        telemetry = dict(getattr(servo_state, "telemetry", {}))
        total = len(expected_ids)
        telemetry_ready_count = sum(
            1
            for servo_id in expected_ids
            if str(telemetry.get(int(servo_id), {}).get("telemetry_status", "")).strip().lower()
            not in {"", "unknown", "unreadable"}
        )
        motion_ready_count = sum(
            1
            for servo_id in expected_ids
            if bool(telemetry.get(int(servo_id), {}).get("motion_ready"))
        )
        external_power_flags = [
            row.get("external_power_ready")
            for row in telemetry.values()
            if row.get("external_power_ready") is not None
        ]
        if external_power_flags:
            self.state.external_power_ready = all(bool(flag) for flag in external_power_flags)
        else:
            self.state.external_power_ready = None
        if getattr(servo_state, "bench_debug_text", ""):
            self.state.bench_debug_text = str(servo_state.bench_debug_text)
        if getattr(servo_state, "last_error", None):
            self.state.last_error = str(servo_state.last_error)
        self.state.expected_servo_ids = expected_ids
        self.state.detected_servo_ids = detected_ids
        self.state.bus_reachable = bool(detected_ids)
        self.state.telemetry_ready_count = int(telemetry_ready_count)
        self.state.motion_ready_count = int(motion_ready_count)
        self.state.motion_ready = bool(total > 0 and motion_ready_count == total)
        issues: list[str] = []
        missing_ids = [int(servo_id) for servo_id in getattr(servo_state, "missing_servo_ids", [])]
        unexpected_ids = [int(servo_id) for servo_id in getattr(servo_state, "unexpected_servo_ids", [])]
        if missing_ids:
            issues.append(f"missing={missing_ids}")
        if unexpected_ids:
            issues.append(f"unexpected={unexpected_ids}")
        if total:
            summary = (
                f"Detected {len(detected_ids)}/{total} | "
                f"Telemetry {telemetry_ready_count}/{total} | "
                f"Motion ready {motion_ready_count}/{total}"
            )
        else:
            summary = "No expected servo IDs are configured."
        if issues:
            summary += " | " + " | ".join(issues)
        self.state.readiness_message = summary
        return self.state

    def save_runtime_parameters(
        self,
        *,
        mock_mode: bool,
        robot_config: str,
        openrb_port: str,
        baudrate: int,
        poll_rate_hz: int,
        fine_jog_step_ticks: int,
        coarse_jog_step_ticks: int,
        position_min_offset_ticks: int | None = None,
        position_max_offset_ticks: int | None = None,
        software_position_margin_ticks: int | None = None,
        telemetry_freshness_timeout_s: float,
        pretension_threshold_ma: int | None = None,
        tightening_direction: str | None = None,
    ) -> str:
        try:
            if self.config_loader is None:
                raise RuntimeError("Config loader is unavailable; runtime parameter editing is disabled.")
            if poll_rate_hz <= 0:
                raise ValueError("GUI refresh rate must be positive.")
            if fine_jog_step_ticks <= 0 or coarse_jog_step_ticks <= 0:
                raise ValueError("Jog increments must be positive.")
            if fine_jog_step_ticks > coarse_jog_step_ticks:
                raise ValueError("Fine jog increment must be less than or equal to coarse jog increment.")
            resolved_min_offset = (
                self.state.position_min_offset_ticks
                if position_min_offset_ticks is None
                else int(position_min_offset_ticks)
            )
            resolved_max_offset = (
                self.state.position_max_offset_ticks
                if position_max_offset_ticks is None
                else int(position_max_offset_ticks)
            )
            resolved_margin = (
                self.state.software_position_margin_ticks
                if software_position_margin_ticks is None
                else int(software_position_margin_ticks)
            )
            resolved_threshold = (
                self.state.pretension_threshold_ma
                if pretension_threshold_ma is None
                else int(pretension_threshold_ma)
            )
            resolved_direction = (
                self.state.tightening_direction_default
                if tightening_direction is None
                else str(tightening_direction).strip().lower()
            )
            if resolved_min_offset > 0 or resolved_max_offset < 0:
                raise ValueError("Neutral-centered bounds must straddle zero offset.")
            if resolved_min_offset >= resolved_max_offset:
                raise ValueError("Minimum offset must be less than maximum offset.")
            if resolved_margin < 0:
                raise ValueError("Software position margin must be non-negative.")
            if telemetry_freshness_timeout_s <= 0:
                raise ValueError("Telemetry freshness timeout must be positive.")
            if resolved_threshold <= 0:
                raise ValueError("Pretension threshold must be positive.")
            robot = self.config_loader.load_robot_config(robot_config)
            overrides = {
                "mock_mode": bool(mock_mode),
                "robot_config": str(robot_config),
                "openrb_port": str(openrb_port).strip(),
                "baudrate": int(baudrate),
                "poll_rate_hz": int(poll_rate_hz),
                "safety_overrides": {
                    "fine_jog_step_ticks": int(fine_jog_step_ticks),
                    "coarse_jog_step_ticks": int(coarse_jog_step_ticks),
                    "position_min_offset_ticks": int(resolved_min_offset),
                    "position_max_offset_ticks": int(resolved_max_offset),
                    "software_position_margin_ticks": int(resolved_margin),
                    "telemetry_stale_after_s": float(telemetry_freshness_timeout_s),
                    "default_pretension_current_threshold_ma": int(resolved_threshold),
                },
                "robot_overrides": {
                    "tightening_rotation_by_servo": {
                        str(servo_id): str(resolved_direction).strip().lower()
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
            self.state.poll_rate_hz = int(poll_rate_hz)
            self.state.fine_jog_step_ticks = int(fine_jog_step_ticks)
            self.state.coarse_jog_step_ticks = int(coarse_jog_step_ticks)
            self.state.position_min_offset_ticks = int(resolved_min_offset)
            self.state.position_max_offset_ticks = int(resolved_max_offset)
            self.state.software_position_margin_ticks = int(resolved_margin)
            self.state.telemetry_freshness_timeout_s = float(telemetry_freshness_timeout_s)
            self.state.pretension_threshold_ma = int(resolved_threshold)
            self.state.tightening_direction_default = str(resolved_direction).strip().lower()
            self.state.saved_overrides_path = str(path)
            self.state.status_message = f"Saved runtime parameters to {path}."
            self.state.last_error = None
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Save runtime parameters failed: {exc}"
            self.refresh()
            raise
        self.refresh()
        return str(path)

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
            f"Legacy bridge fallback backend: {settings.serial.tracker_fallback_backend}\n"
            f"Legacy bridge fallback enabled: {settings.serial.tracker_fallback_enabled}\n"
            f"Tracker type: {settings.serial.tracker_type}\n"
            f"Tracker freshness timeout: {settings.serial.tracker_freshness_timeout_s}s\n"
            f"GUI refresh rate: {settings.runtime.poll_rate_hz} Hz\n"
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
            f"GUI refresh rate: {self.state.poll_rate_hz} Hz\n"
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

    @staticmethod
    def _existing_overrides_path(config_loader: ConfigLoader | None) -> str:
        if config_loader is None:
            return ""
        path = Path(config_loader.base_dir) / "system.local.yaml"
        return str(path) if path.exists() else ""

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
                f"freshness_threshold_s={self.state.telemetry_freshness_timeout_s:.3f}",
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
                f"freshness_threshold_s={self.state.telemetry_freshness_timeout_s:.3f}",
                f"all_expected_present={snapshot.all_expected_present}",
                f"all_expected_telemetry_ok={snapshot.all_expected_telemetry_ok}",
                f"all_motion_ready={snapshot.all_motion_ready}",
                f"active_range={self.servo_service.raw_position_range()[0]}..{self.servo_service.raw_position_range()[1]}",
                "position_convention=tighten->smaller_counts; loosen->larger_counts",
            ]
        )

    def _build_runtime_servo_debug_text(self, snapshot) -> str:
        openrb_connected = self.openrb_client.get_status_snapshot().connected
        motion_ready_ids = [
            int(entry.servo_id)
            for entry in getattr(snapshot, "entries", {}).values()
            if entry.motion_assessment is not None and entry.motion_assessment.ready
        ]
        telemetry_ok_ids = [
            int(entry.servo_id)
            for entry in getattr(snapshot, "entries", {}).values()
            if entry.telemetry_status == "Live"
        ]
        return "\n".join(
            [
                "Bench debug:",
                f"openrb_connected={openrb_connected}",
                f"selected_port={self.state.openrb_port or 'unset'}",
                f"selected_baud={self.state.baudrate}",
                f"expected_servo_ids={getattr(snapshot, 'expected_servo_ids', [])}",
                f"discovered_ids={getattr(snapshot, 'detected_servo_ids', [])}",
                f"detected_servo_ids={getattr(snapshot, 'detected_servo_ids', [])}",
                f"missing_servo_ids={getattr(snapshot, 'missing_servo_ids', [])}",
                f"unexpected_servo_ids={getattr(snapshot, 'unexpected_servo_ids', [])}",
                f"telemetry_ok_ids={telemetry_ok_ids}",
                f"motion_ready_ids={motion_ready_ids}",
                f"freshness_threshold_s={self.state.telemetry_freshness_timeout_s:.3f}",
                f"all_motion_ready={getattr(snapshot, 'all_motion_ready', False)}",
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
            f"freshness_threshold_s={self.state.telemetry_freshness_timeout_s:.3f}",
            f"active_range={self.servo_service.raw_position_range()[0]}..{self.servo_service.raw_position_range()[1]}",
            "motion_ready=False",
            "position_convention=tighten->smaller_counts; loosen->larger_counts",
            f"motion_block_reason={extra_error or 'OpenRB/DYNAMIXEL not ready'}",
        ]
        return "\n".join(lines)
