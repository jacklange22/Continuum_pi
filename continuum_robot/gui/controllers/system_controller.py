"""System tab controller for top-level connectivity and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import time

from continuum_robot.config.config_loader import ConfigLoader
from continuum_robot.config.settings import Settings
from continuum_robot.hardware.serial_ports import SerialPortInfo, discover_serial_ports
from continuum_robot.servos.servo_service import ServoBusBusyError
from continuum_robot.servos.telemetry_diagnostics import (
    DEFAULT_SERVO_FULL_REFRESH_DIVISOR,
    DEFAULT_SYSTEM_SUMMARY_REFRESH_DIVISOR,
    build_telemetry_gui_policy,
)


LOG = logging.getLogger(__name__)


@dataclass
class SystemViewState:
    """UI-facing system status."""

    mock_mode: bool
    aurora_port: str
    openrb_port: str
    baudrate: int
    mode_display: str = "Hardware"
    robot_layout_display: str = "Unknown"
    tracker_status_label: str = "Not Connected"
    tracker_status_kind: str = "blocked"
    openrb_status_label: str = "Not Connected"
    openrb_status_kind: str = "blocked"
    overall_status_label: str = "Blocked"
    overall_status_kind: str = "blocked"
    tracker_connection_state: str = "disconnected"
    tracker_backend_identity: str = ""
    tracker_backend_running: bool = False
    tracker_backend_connected: bool = False
    tracker_truth_summary: str = "Tracker not connected."
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
    openrb_truth_summary: str = "OpenRB not connected."
    readiness_message: str = "OpenRB readiness not checked."
    overall_status_summary: str = "Blocked"
    primary_blocker: str = "Tracker is not connected."
    status_message: str = "System idle."
    last_error: str | None = None
    available_ports: list[SerialPortInfo] = field(default_factory=list)
    available_robot_configs: list[str] = field(default_factory=list)
    robot_config: str = "robot_8servo.yaml"
    robot_mode: str = ""
    operating_mode: str = "single_segment"
    selected_servo_id: int = 1
    active_segment_key: str = "segment_a"
    active_segment_label: str = "Spine 1"
    available_segments: list[dict[str, object]] = field(default_factory=list)
    active_segment_servo_ids: list[int] = field(default_factory=list)
    active_segment_pairs: dict[str, list[int]] = field(default_factory=dict)
    expected_servo_ids: list[int] = field(default_factory=list)
    detected_servo_ids: list[int] = field(default_factory=list)
    telemetry_ready_count: int = 0
    motion_ready_count: int = 0
    poll_rate_hz: int = 10
    servo_telemetry_cadence_summary: str = ""
    servo_telemetry_field_summary: str = ""
    servo_telemetry_bottleneck_summary: str = ""
    fine_jog_step_ticks: int = 5
    coarse_jog_step_ticks: int = 25
    position_min_offset_ticks: int = -600
    position_max_offset_ticks: int = 600
    software_position_margin_ticks: int = 64
    telemetry_freshness_timeout_s: float = 0.25
    figure_output_quality: str = "production"
    pretension_threshold_ma: int = 220
    tightening_direction_default: str = "cw"
    bench_debug_text: str = ""
    saved_overrides_path: str = ""
    config_summary: str = ""
    session_log_path: str = ""
    session_log_summary: str = ""
    diagnostics_preview: str = ""


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
        session_log_path: str | None = None,
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
            robot_mode=settings.robot.operating_mode(),
            operating_mode=settings.robot.operating_context().operating_mode,
            selected_servo_id=int(settings.robot.operating_context().selected_servo_id or settings.robot.selected_servo_id or 1),
            active_segment_key=settings.robot.active_segment_key(),
            active_segment_label=settings.robot.active_segment_label(),
            available_segments=self._segment_options(settings.robot),
            active_segment_servo_ids=settings.robot.active_segment_servo_ids(),
            active_segment_pairs=settings.robot.active_segment_pairs(),
            expected_servo_ids=list(settings.robot.expected_servo_ids()),
            poll_rate_hz=settings.runtime.poll_rate_hz,
            servo_telemetry_cadence_summary="",
            servo_telemetry_field_summary="",
            servo_telemetry_bottleneck_summary="",
            fine_jog_step_ticks=settings.safety.fine_jog_step_ticks,
            coarse_jog_step_ticks=settings.safety.coarse_jog_step_ticks,
            position_min_offset_ticks=settings.safety.position_min_offset_ticks,
            position_max_offset_ticks=settings.safety.position_max_offset_ticks,
            software_position_margin_ticks=settings.safety.software_position_margin_ticks,
            telemetry_freshness_timeout_s=settings.safety.telemetry_stale_after_s,
            figure_output_quality=str(settings.runtime.figure_output_quality),
            pretension_threshold_ma=settings.safety.default_pretension_current_threshold_ma,
            tightening_direction_default=self._default_tightening_direction(settings),
            config_summary=self._build_config_summary(settings),
            status_message=self._initial_status_message(settings),
            saved_overrides_path=self._existing_overrides_path(config_loader),
            session_log_path=str(session_log_path or ""),
        )
        self._last_truth_snapshot: dict[str, str] = {}
        self._readiness_cache: tuple[tuple[int, ...], bool, object] | None = None
        self._readiness_cache_monotonic_s: float = 0.0
        self._readiness_cache_ttl_s: float = 2.0
        self._last_slow_readiness_warning_s: float = 0.0
        self._refresh_telemetry_policy()
        self.rescan_ports()

    def rescan_ports(self) -> SystemViewState:
        self._refresh_available_ports_snapshot()
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
            LOG.info("Tracker connect requested | port=%s", self.state.aurora_port)
            self.state.status_message = "Tracker connection requested."
            self.state.last_error = None
        except Exception as exc:
            LOG.exception("Tracker connect failed | port=%s | error=%s", self.state.aurora_port, exc)
            self.state.last_error = str(exc)
            self.state.status_message = f"Tracker connect failed: {exc}"
        self.refresh()

    def disconnect_tracker(self) -> None:
        try:
            self.tracking_service.stop()
            LOG.info("Tracker disconnected.")
            self.state.status_message = "Tracker disconnected."
            self.state.last_error = None
        except Exception as exc:
            LOG.exception("Tracker disconnect failed | error=%s", exc)
            self.state.last_error = str(exc)
            self.state.status_message = f"Tracker disconnect failed: {exc}"
        self.refresh()

    def connect_openrb(self) -> None:
        attempted: list[dict[str, str]] = []
        try:
            candidates, skipped = self._openrb_port_candidates()
            for skipped_row in skipped:
                LOG.info(
                    "OpenRB candidate skipped | port=%s | reason=%s | detail=%s",
                    skipped_row["port"],
                    skipped_row["reason"],
                    skipped_row.get("detail", ""),
                )
            if not candidates:
                raise RuntimeError("OpenRB port is empty. Set or rescan the board port before connecting.")

            connected_port: str | None = None
            selected_reason: str | None = None
            configured_port = str(self.state.openrb_port or "").strip()
            for candidate_port, candidate_reason in candidates:
                LOG.info(
                    "OpenRB candidate selected for attempt | port=%s | reason=%s",
                    candidate_port,
                    candidate_reason,
                )
                try:
                    self.openrb_client.connect(candidate_port, self.state.baudrate)
                except Exception as exc:
                    attempted.append(
                        {
                            "port": str(candidate_port),
                            "reason": str(candidate_reason),
                            "stage": "serial_open",
                            "error": str(exc),
                        }
                    )
                    self._best_effort_disconnect_after_failed_openrb_candidate()
                    LOG.warning(
                        "OpenRB candidate failed | port=%s | reason=%s | stage=serial_open | error=%s",
                        candidate_port,
                        candidate_reason,
                        exc,
                    )
                    continue
                try:
                    self.openrb_client.prepare_for_dynamixel_use()
                except Exception as exc:
                    attempted.append(
                        {
                            "port": str(candidate_port),
                            "reason": str(candidate_reason),
                            "stage": "pass_through_prepare",
                            "error": str(exc),
                        }
                    )
                    self._best_effort_disconnect_after_failed_openrb_candidate()
                    LOG.warning(
                        "OpenRB candidate failed | port=%s | reason=%s | stage=pass_through_prepare | error=%s",
                        candidate_port,
                        candidate_reason,
                        exc,
                    )
                    continue
                try:
                    self.servo_service.connect(candidate_port, self.state.baudrate)
                except Exception as exc:
                    attempted.append(
                        {
                            "port": str(candidate_port),
                            "reason": str(candidate_reason),
                            "stage": "dynamixel_bus_connect",
                            "error": str(exc),
                        }
                    )
                    self._best_effort_disconnect_after_failed_openrb_candidate()
                    LOG.warning(
                        "OpenRB candidate failed | port=%s | reason=%s | stage=dynamixel_bus_connect | error=%s",
                        candidate_port,
                        candidate_reason,
                        exc,
                    )
                    continue
                try:
                    connected_port = str(candidate_port)
                    selected_reason = str(candidate_reason)
                    break
                finally:
                    # The loop exits on success, but keep this branch explicit so
                    # later stages can append instrumentation without changing flow.
                    pass
            if connected_port is None:
                attempts_text = "; ".join(
                    f"{row['port']} ({row['reason']}, {row['stage']}): {row['error']}"
                    for row in attempted
                )
                raise RuntimeError(
                    "Could not connect OpenRB on any candidate serial port. "
                    f"Tried: {attempts_text or 'none'}."
                )
            self.state.openrb_port = str(connected_port)
            LOG.info(
                "OpenRB connected | port=%s | baud=%s | expected_servo_ids=%s | selection_reason=%s",
                connected_port,
                self.state.baudrate,
                self.settings.robot.operating_context().expected_servo_ids,
                selected_reason or "configured_port",
            )
            replacement_message = ""
            if configured_port and configured_port != str(connected_port):
                replacement_message = (
                    f" Replaced configured port {configured_port} with detected fallback {connected_port}."
                )
            if attempted:
                attempts_text = ", ".join(f"{row['port']} ({row['reason']})" for row in attempted)
                self.state.status_message = (
                    f"OpenRB serial connected on {connected_port} after fallback scan."
                    f"{replacement_message} Failed candidates: {attempts_text}."
                )
            else:
                self.state.status_message = (
                    f"OpenRB serial connected on {connected_port} and pass-through prepared."
                    " Verifying DYNAMIXEL bus readiness next."
                )
            self.state.last_error = None
        except Exception as exc:
            LOG.exception(
                "OpenRB connect failed | port=%s | baud=%s | error=%s",
                self.state.openrb_port,
                self.state.baudrate,
                exc,
            )
            self.state.last_error = str(exc)
            self.state.status_message = f"OpenRB connect failed: {exc}"
        self.refresh_readiness()

    def _best_effort_disconnect_after_failed_openrb_candidate(self) -> None:
        try:
            self.servo_service.disconnect()
        except Exception:
            pass
        try:
            self.openrb_client.disconnect()
        except Exception:
            pass

    def _refresh_available_ports_snapshot(self) -> list[SerialPortInfo]:
        ports = discover_serial_ports()
        if self.settings.runtime.mock_mode:
            ports.extend(
                [
                    SerialPortInfo(device="/dev/mock-aurora", description="Mock Aurora port"),
                    SerialPortInfo(device="/dev/mock-openrb", description="Mock OpenRB port"),
                ]
            )
        deduped = {str(port.device): port for port in ports if str(port.device).strip()}
        self.state.available_ports = sorted(deduped.values(), key=lambda port: port.device)
        return list(self.state.available_ports)

    def _openrb_port_candidates(self) -> tuple[list[tuple[str, str]], list[dict[str, str]]]:
        available_ports = self._refresh_available_ports_snapshot()
        available_by_device = {
            str(port.device).strip(): port
            for port in available_ports
            if str(port.device).strip()
        }
        available_devices = set(available_by_device)
        selected = str(self.state.openrb_port or "").strip()
        tracker_port = str(self.state.aurora_port or "").strip()
        candidates: list[tuple[str, str]] = []
        skipped: list[dict[str, str]] = []
        seen: set[str] = set()
        allow_onboard_uart_fallback = self._allow_onboard_uart_fallback()
        has_non_onboard_candidates = any(
            self._is_usb_serial_candidate(port.device) or self._is_openrb_hint(port)
            for port in available_ports
            if str(port.device).strip() and str(port.device).strip() != tracker_port
        )

        def _skip(port: str, reason: str, detail: str) -> None:
            candidate = str(port or "").strip()
            if not candidate:
                return
            skipped.append({"port": candidate, "reason": reason, "detail": detail})

        def _allow_candidate(port_info: SerialPortInfo, *, reason: str) -> bool:
            device = str(port_info.device or "").strip()
            if not device:
                return False
            if device == tracker_port:
                _skip(device, reason, "matches tracker_port")
                return False
            if (
                self._is_onboard_uart_candidate(device)
                and reason != "configured_port"
                and not self._is_openrb_hint(port_info)
                and not allow_onboard_uart_fallback
            ):
                _skip(device, reason, "onboard_uart_fallback_disabled")
                return False
            return True

        def _add(port: str, reason: str) -> None:
            candidate = str(port or "").strip()
            if not candidate or candidate in seen:
                return
            seen.add(candidate)
            candidates.append((candidate, reason))

        if selected:
            if selected == tracker_port:
                _skip(selected, "configured_port", "configured_port_matches_tracker_port")
            elif selected in available_devices:
                selected_info = available_by_device[selected]
                if _allow_candidate(selected_info, reason="configured_port"):
                    _add(selected, "configured_port")
            elif has_non_onboard_candidates:
                _skip(selected, "configured_port", "configured_port_not_present")
            else:
                _add(selected, "configured_port_unlisted")
        for port_info in available_ports:
            device = str(port_info.device or "").strip()
            if not _allow_candidate(port_info, reason="openrb_hint"):
                continue
            if self._is_openrb_hint(port_info):
                _add(device, "openrb_hint")
        for port_info in available_ports:
            device = str(port_info.device or "").strip()
            if not _allow_candidate(port_info, reason="tty_usb_fallback"):
                continue
            if self._is_usb_serial_candidate(device):
                _add(device, "tty_usb_fallback")
        return candidates, skipped

    def _allow_onboard_uart_fallback(self) -> bool:
        openrb_settings = dict(getattr(self.settings.serial, "openrb_settings", {}) or {})
        return bool(openrb_settings.get("allow_onboard_uart_fallback", False))

    @staticmethod
    def _is_onboard_uart_candidate(device: str) -> bool:
        normalized = str(device or "").strip().lower()
        return normalized.startswith("/dev/ttyama") or normalized.startswith("/dev/ttys")

    @staticmethod
    def _is_usb_serial_candidate(device: str) -> bool:
        normalized = str(device or "").strip().lower()
        return (
            normalized.startswith("/dev/ttyacm")
            or normalized.startswith("/dev/ttyusb")
            or normalized.startswith("/dev/cu.usb")
            or normalized.startswith("/dev/tty.usb")
            or normalized.startswith("/dev/cu.usbserial")
            or normalized.startswith("/dev/tty.usbserial")
        )

    @staticmethod
    def _is_openrb_hint(port_info: SerialPortInfo) -> bool:
        text = " ".join(
            [
                str(port_info.description or ""),
                str(port_info.hwid or ""),
                str(port_info.manufacturer or ""),
                str(port_info.product or ""),
                str(port_info.interface or ""),
            ]
        ).lower()
        return any(token in text for token in ("openrb", "dynamixel", "robotis", "u2d2"))

    def disconnect_openrb(self) -> None:
        try:
            self.servo_service.disconnect()
            self.openrb_client.disconnect()
            LOG.info("OpenRB and DYNAMIXEL disconnected; torque state preserved unless explicitly configured otherwise.")
            self.state.status_message = "OpenRB and DYNAMIXEL bus disconnected; torque state was preserved."
            self.state.last_error = None
        except Exception as exc:
            try:
                self.openrb_client.disconnect()
            except Exception:
                pass
            if self._is_expected_disconnect_cleanup_issue(exc):
                LOG.info("OpenRB disconnect cleanup warning | error=%s", exc)
                self.state.last_error = None
                self.state.status_message = (
                    "OpenRB disconnected; torque state was preserved, but the bus was already unavailable during cleanup."
                )
            else:
                LOG.exception("OpenRB disconnect failed | error=%s", exc)
                self.state.last_error = str(exc)
                self.state.status_message = f"OpenRB disconnect failed: {exc}"
        self.refresh()

    def prepare_openrb(self) -> None:
        try:
            self.openrb_client.prepare_for_dynamixel_use()
            LOG.info("OpenRB prepare-for-dynamixel requested.")
            self.state.status_message = "OpenRB prepared for DYNAMIXEL use."
            self.state.last_error = None
        except Exception as exc:
            LOG.exception("OpenRB prepare failed | error=%s", exc)
            self.state.last_error = str(exc)
            self.state.status_message = f"OpenRB prepare failed: {exc}"
        self.refresh_readiness()

    def refresh_readiness(self, *, include_scan: bool = True) -> SystemViewState:
        if not self.servo_service.is_connected:
            self.state.readiness_message = "DYNAMIXEL bus disconnected."
            self.state.bench_debug_text = self._build_disconnected_bench_debug_text()
            return self.refresh()
        try:
            started = time.monotonic()
            context = self.settings.robot.operating_context()
            expected_ids = [int(value) for value in context.expected_servo_ids]
            ownership = self.servo_service.bus_ownership_status()
            if ownership.active and not ownership.held_by_current_thread:
                snapshot = self.servo_service.build_cached_runtime_servo_snapshot(
                    expected_ids,
                    selected_servo_id=int(context.selected_servo_id) if context.selected_servo_id is not None else None,
                )
                self.sync_servo_runtime_snapshot(snapshot)
                self.state.status_message = self.servo_service.bus_busy_message(action="system readiness refresh")
                self.state.last_error = None
                return self.refresh()
            if context.operating_mode == "one_servo":
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
                telemetry_profile = "minimal"
                cache_key = (tuple(expected_ids), bool(include_scan))
                cached = self._readiness_cache
                if (
                    not include_scan
                    and cached is not None
                    and cached[0] == cache_key[0]
                    and cached[1] == cache_key[1]
                    and (time.monotonic() - self._readiness_cache_monotonic_s) <= self._readiness_cache_ttl_s
                ):
                    snapshot = cached[2]
                else:
                    snapshot = self.servo_service.build_runtime_servo_snapshot(
                        expected_ids,
                        include_scan=bool(include_scan),
                        telemetry_profile=telemetry_profile,
                    )
                    self._readiness_cache = (cache_key[0], cache_key[1], snapshot)
                    self._readiness_cache_monotonic_s = time.monotonic()
                self.sync_servo_runtime_snapshot(snapshot)
                self.state.last_error = None if snapshot.all_motion_ready else snapshot.message
            elapsed_ms = (time.monotonic() - started) * 1000.0
            now_for_warning = time.monotonic()
            if elapsed_ms > 250.0 and (now_for_warning - self._last_slow_readiness_warning_s) > 10.0:
                self._last_slow_readiness_warning_s = now_for_warning
                LOG.warning(
                    "System readiness refresh slow | elapsed_ms=%.1f | expected_servo_ids=%s | include_scan=%s",
                    elapsed_ms,
                    expected_ids,
                    bool(include_scan),
                )
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
        context = self.settings.robot.operating_context()
        self.state.robot_mode = context.operating_mode
        self.state.operating_mode = context.operating_mode
        self.state.selected_servo_id = int(context.selected_servo_id or self.settings.robot.selected_servo_id or 1)
        self.state.expected_servo_ids = list(context.expected_servo_ids)
        self.state.active_segment_key = self.settings.robot.active_segment_key()
        self.state.active_segment_label = self.settings.robot.active_segment_label()
        self.state.available_segments = self._segment_options(self.settings.robot)
        self.state.active_segment_servo_ids = self.settings.robot.active_segment_servo_ids()
        self.state.active_segment_pairs = self.settings.robot.active_segment_pairs()
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
        self._refresh_telemetry_policy()
        self.state.config_summary = self._build_live_config_summary()
        self._refresh_operator_truth(tracker_state)
        self._refresh_session_diagnostics()
        self._log_truth_transitions()
        return self.state

    def _refresh_telemetry_policy(self) -> None:
        policy = build_telemetry_gui_policy(
            baudrate=self.state.baudrate,
            poll_rate_hz=self.state.poll_rate_hz,
            telemetry_stale_after_s=self.state.telemetry_freshness_timeout_s,
            servo_full_refresh_divisor=DEFAULT_SERVO_FULL_REFRESH_DIVISOR,
            system_summary_refresh_divisor=DEFAULT_SYSTEM_SUMMARY_REFRESH_DIVISOR,
        )
        self.state.servo_telemetry_cadence_summary = policy.cadence_summary
        self.state.servo_telemetry_field_summary = policy.field_summary
        self.state.servo_telemetry_bottleneck_summary = policy.bottleneck_summary

    def _refresh_operator_truth(self, tracker_state) -> None:
        self.state.mode_display = "Mock" if self.state.mock_mode else "Hardware"
        self.state.robot_layout_display = self._robot_layout_display(
            self.state.robot_mode,
            self.state.expected_servo_ids,
        )
        tracker_connected = bool(
            self.state.tracker_backend_connected
            or self.state.tracker_connection_state in {"starting", "connecting"}
        )
        tracker_healthy = bool(
            tracker_connected
            and self.state.tracker_connection_state not in {"disconnected", "error"}
            and not getattr(tracker_state, "tracker_data_stale", False)
            and not getattr(tracker_state, "last_error", None)
        )
        if tracker_healthy:
            backend = self.state.tracker_backend_identity or "backend unknown"
            self.state.tracker_status_label = "Connected"
            self.state.tracker_status_kind = "ready"
            self.state.tracker_truth_summary = f"Healthy on {backend}."
        elif tracker_connected:
            backend = self.state.tracker_backend_identity or "backend unknown"
            self.state.tracker_status_label = "Degraded"
            self.state.tracker_status_kind = "warning"
            state_label = self.state.tracker_connection_state.replace("_", " ")
            self.state.tracker_truth_summary = f"{state_label.capitalize()} on {backend}."
        else:
            self.state.tracker_status_label = "Not Connected"
            self.state.tracker_status_kind = "blocked"
            self.state.tracker_truth_summary = "Tracker not connected."

        openrb_connected = bool(self.state.openrb_connected or self.state.dynamixel_connected)
        expected_ids = [int(value) for value in self.state.expected_servo_ids]
        detected_ids = [int(value) for value in self.state.detected_servo_ids]
        missing_ids = [servo_id for servo_id in expected_ids if servo_id not in detected_ids]
        detected_configured_count = len([servo_id for servo_id in detected_ids if servo_id in expected_ids])

        if self.state.motion_ready and openrb_connected:
            self.state.openrb_status_label = "Connected"
            self.state.openrb_status_kind = "ready"
            self.state.openrb_truth_summary = "OpenRB and the DYNAMIXEL bus are ready."
        elif openrb_connected and not self.state.openrb_prepared:
            self.state.openrb_status_label = "Degraded"
            self.state.openrb_status_kind = "warning"
            self.state.openrb_truth_summary = "OpenRB serial link is up, but pass-through is not prepared."
        elif openrb_connected and not self.state.dynamixel_connected:
            self.state.openrb_status_label = "Degraded"
            self.state.openrb_status_kind = "warning"
            self.state.openrb_truth_summary = "OpenRB pass-through is prepared, but the DYNAMIXEL bus is not connected."
        elif openrb_connected and not self.state.bus_reachable:
            self.state.openrb_status_label = "Degraded"
            self.state.openrb_status_kind = "warning"
            self.state.openrb_truth_summary = (
                "OpenRB is connected, but no configured servos responded on the DYNAMIXEL bus."
            )
        elif openrb_connected and missing_ids:
            self.state.openrb_status_label = "Degraded"
            self.state.openrb_status_kind = "warning"
            self.state.openrb_truth_summary = (
                f"DYNAMIXEL bus responding ({detected_configured_count}/{len(expected_ids)} configured servos). "
                f"Missing: {missing_ids}."
            )
        elif openrb_connected and self.state.bus_reachable:
            self.state.openrb_status_label = "Degraded"
            self.state.openrb_status_kind = "warning"
            self.state.openrb_truth_summary = self.state.readiness_message or "OpenRB is connected with warnings."
        else:
            self.state.openrb_status_label = "Not Connected"
            self.state.openrb_status_kind = "blocked"
            self.state.openrb_truth_summary = "OpenRB not connected."

        if tracker_healthy and (self.state.mock_mode or self.state.motion_ready):
            self.state.overall_status_label = "Ready"
            self.state.overall_status_kind = "ready"
            self.state.overall_status_summary = "Ready"
            self.state.primary_blocker = ""
            return
        self.state.overall_status_label = "Blocked"
        self.state.overall_status_kind = "blocked"
        self.state.overall_status_summary = "Blocked"
        self.state.primary_blocker = self._primary_blocker_message(
            tracker_connected=tracker_connected,
            tracker_healthy=tracker_healthy,
            openrb_connected=openrb_connected,
        )

    def _primary_blocker_message(
        self,
        *,
        tracker_connected: bool,
        tracker_healthy: bool,
        openrb_connected: bool,
    ) -> str:
        if self.state.last_error:
            return str(self.state.last_error)
        if not tracker_connected:
            return "Tracker is not connected."
        if not tracker_healthy:
            return self.state.tracker_connection_state.replace("_", " ")
        if self.state.mock_mode:
            return self.state.status_message
        if not openrb_connected:
            return "OpenRB / DYNAMIXEL is not connected."
        if not self.state.openrb_prepared:
            return "OpenRB is connected, but pass-through preparation is incomplete."
        if not self.state.dynamixel_connected:
            return "OpenRB serial link is ready, but the DYNAMIXEL session is not connected."
        if not self.state.bus_reachable:
            return "Configured servos are not responding on the DYNAMIXEL bus."
        missing_ids = [
            int(servo_id)
            for servo_id in self.state.expected_servo_ids
            if int(servo_id) not in {int(value) for value in self.state.detected_servo_ids}
        ]
        if missing_ids:
            return f"Configured servos missing on DYNAMIXEL bus: {missing_ids}."
        if not self.state.motion_ready:
            return self.state.readiness_message or "Servo readiness is blocked."
        return self.state.status_message

    @staticmethod
    def _is_expected_disconnect_cleanup_issue(exc: Exception) -> bool:
        text = str(exc or "").strip().lower()
        if not text:
            return False
        expected_tokens = (
            "not connected",
            "disconnected",
            "port is not open",
            "port not open",
            "incorrect status packet",
            "txrxresult",
            "no status packet",
            "bus is disconnected",
            "communication failure",
        )
        return any(token in text for token in expected_tokens)

    def _refresh_session_diagnostics(self) -> None:
        session_log_path = Path(self.state.session_log_path) if self.state.session_log_path else None
        if session_log_path is None:
            self.state.session_log_summary = "Current session log unavailable."
            self.state.diagnostics_preview = "No session log is active for this launch."
            return
        self.state.session_log_summary = str(session_log_path)
        log_lines = self._read_session_log_lines(session_log_path)
        tail = "\n".join(log_lines[-12:]).strip()
        if tail:
            self.state.diagnostics_preview = tail
        else:
            self.state.diagnostics_preview = "No session log lines yet."

    def build_session_diagnostics_document(self) -> str:
        session_log_path = Path(self.state.session_log_path) if self.state.session_log_path else None
        log_text = ""
        if session_log_path is not None and session_log_path.exists():
            log_text = session_log_path.read_text(encoding="utf-8")
        sections = [
            "System Session Diagnostics",
            "",
            f"Overall: {self.state.overall_status_label}",
            f"Primary blocker: {self.state.primary_blocker or 'none'}",
            f"Mode: {self.state.mode_display}",
            f"Robot layout: {self.state.robot_layout_display}",
            f"Robot profile: {self.state.robot_config}",
            f"Tracker: {self.state.tracker_truth_summary}",
            f"OpenRB: {self.state.openrb_truth_summary}",
            f"Status message: {self.state.status_message}",
            f"Last error: {self.state.last_error or 'none'}",
            f"Saved overrides: {self.state.saved_overrides_path or 'none'}",
            f"Session log: {self.state.session_log_path or 'unset'}",
            "",
            "Measurement chain:",
            f"Registration: {self.state.registration_summary}",
            f"Runtime tip: {self.state.runtime_tip_summary}",
            f"Live tip: {self.state.live_tip_summary}",
            "",
            "Effective config:",
            self.state.config_summary,
            "",
            "Session log contents:",
            log_text or "(session log is empty)",
        ]
        return "\n".join(sections).rstrip() + "\n"

    @staticmethod
    def _read_session_log_lines(path: Path) -> list[str]:
        if not path.exists():
            return []
        return path.read_text(encoding="utf-8").splitlines()

    def _log_truth_transitions(self) -> None:
        current = {
            "overall": self.state.overall_status_summary,
            "blocker": self.state.primary_blocker,
            "tracker": self.state.tracker_truth_summary,
            "openrb": self.state.openrb_truth_summary,
            "status": self.state.status_message,
            "error": self.state.last_error or "",
        }
        if not self._last_truth_snapshot:
            self._last_truth_snapshot = dict(current)
            return
        for key, value in current.items():
            previous = self._last_truth_snapshot.get(key, "")
            if value == previous:
                continue
            if key == "error" and value:
                LOG.warning("System %s changed | %s", key, value)
            else:
                LOG.info("System %s changed | %s", key, value)
        self._last_truth_snapshot = dict(current)

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
        mode_message = str(getattr(tracker_state, "runtime_tip_mode_message", "") or "")
        if state == "loaded":
            detail = f" | {timestamp}" if timestamp else ""
            return f"{mode.replace('_', ' ')} | {trust.replace('_', ' ')}{detail}"
        if state == "quick_4_point_loaded":
            detail = f" | {timestamp}" if timestamp else ""
            return f"quick 4 point | {trust.replace('_', ' ')}{detail}"
        if state == "coil_as_tip":
            return "coil as tip | lower trust | 0A coil pose is shown directly as tip"
        if state == "identity_tip_fallback":
            return "identity fallback | lower trust | accepted runtime tip is not active"
        if state in {"missing_runtime_tip_calibration", "missing_quick_4_point_runtime_tip"}:
            return f"{mode.replace('_', ' ')} | not loaded"
        if state == "invalid_runtime_tip_calibration":
            return f"{mode.replace('_', ' ')} | invalid"
        if mode_message:
            return mode_message
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
        if tip_pose_status == "coil_as_tip":
            return "Ready | 0A coil pose is shown directly as tip"
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
        packet_read_count = sum(
            1
            for servo_id in expected_ids
            if bool(telemetry.get(int(servo_id), {}).get("packet_read_ok"))
            or str(telemetry.get(int(servo_id), {}).get("telemetry_status", "")).strip().lower()
            not in {"", "unknown", "unreadable", "missing"}
        )
        stale_display_count = sum(
            1
            for servo_id in expected_ids
            if bool(telemetry.get(int(servo_id), {}).get("stale_display_warning"))
            or str(telemetry.get(int(servo_id), {}).get("telemetry_status", "")).strip().lower()
            in {"stale display", "cached stale display"}
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
        self.state.telemetry_ready_count = int(packet_read_count)
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
                f"Packet read {packet_read_count}/{total} | "
                f"GUI cache age warning {stale_display_count}/{total} | "
                "Experiments use fresh pre-motion read"
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
        operating_mode: str | None = None,
        selected_servo_id: int | None = None,
        active_segment: str | None = None,
        openrb_port: str,
        baudrate: int,
        poll_rate_hz: int,
        fine_jog_step_ticks: int | None = None,
        coarse_jog_step_ticks: int | None = None,
        position_min_offset_ticks: int | None = None,
        position_max_offset_ticks: int | None = None,
        software_position_margin_ticks: int | None = None,
        telemetry_freshness_timeout_s: float,
        figure_output_quality: str | None = None,
        pretension_threshold_ma: int | None = None,
        tightening_direction: str | None = None,
    ) -> str:
        try:
            if self.config_loader is None:
                raise RuntimeError("Config loader is unavailable; runtime parameter editing is disabled.")
            if poll_rate_hz <= 0:
                raise ValueError("GUI refresh rate must be positive.")
            resolved_fine_jog = self.state.fine_jog_step_ticks if fine_jog_step_ticks is None else int(fine_jog_step_ticks)
            resolved_coarse_jog = (
                self.state.coarse_jog_step_ticks if coarse_jog_step_ticks is None else int(coarse_jog_step_ticks)
            )
            if resolved_fine_jog <= 0 or resolved_coarse_jog <= 0:
                raise ValueError("Jog increments must be positive.")
            if resolved_fine_jog > resolved_coarse_jog:
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
            resolved_figure_quality = str(figure_output_quality or self.state.figure_output_quality or "production").strip().lower()
            if resolved_figure_quality not in {"low", "medium", "production"}:
                raise ValueError("Figure output quality must be low, medium, or production.")
            if resolved_threshold <= 0:
                raise ValueError("Pretension threshold must be positive.")
            robot = self.config_loader.load_robot_config(robot_config)
            resolved_operating_mode = (
                str(operating_mode).strip().lower()
                if operating_mode not in (None, "")
                else str(robot.operating_mode())
            )
            robot.mode = resolved_operating_mode
            if selected_servo_id not in (None, ""):
                robot.selected_servo_id = int(selected_servo_id)
            resolved_active_segment = (
                str(active_segment).strip()
                if active_segment not in (None, "")
                else str(robot.active_segment_key())
            )
            if resolved_active_segment not in robot.segment_map():
                raise ValueError(f"Active segment {resolved_active_segment!r} is not available in {robot_config}.")
            robot.active_segment = resolved_active_segment
            context = robot.operating_context()
            if context.operating_mode == "one_servo" and int(robot.selected_servo_id) not in context.all_configured_servo_ids:
                raise ValueError(
                    f"Selected servo {robot.selected_servo_id} is not available in configured IDs {context.all_configured_servo_ids}."
                )
            if context.operating_mode in {"dual_segment", "parallel_single"} and len(context.expected_servo_ids) != 8:
                raise ValueError(
                    f"{context.operating_mode} requires an 8-servo robot profile; resolved expected IDs {context.expected_servo_ids}."
                )
            if context.operating_mode == "parallel_single" and len(context.mirror_pairs) != 4:
                raise ValueError(
                    f"parallel_single requires four mirror pairs; resolved {context.mirror_pairs}."
                )
            overrides = {
                "mock_mode": bool(mock_mode),
                "robot_config": str(robot_config),
                "openrb_port": str(openrb_port).strip(),
                "baudrate": int(baudrate),
                "poll_rate_hz": int(poll_rate_hz),
                "figure_output_quality": resolved_figure_quality,
                "safety_overrides": {
                    "fine_jog_step_ticks": int(resolved_fine_jog),
                    "coarse_jog_step_ticks": int(resolved_coarse_jog),
                    "position_min_offset_ticks": int(resolved_min_offset),
                    "position_max_offset_ticks": int(resolved_max_offset),
                    "software_position_margin_ticks": int(resolved_margin),
                    "telemetry_stale_after_s": float(telemetry_freshness_timeout_s),
                    "default_pretension_current_threshold_ma": int(resolved_threshold),
                },
                "robot_overrides": {
                    "mode": context.operating_mode,
                    "selected_servo_id": int(robot.selected_servo_id),
                    "active_segment": resolved_active_segment,
                    "tightening_rotation_by_servo": {
                        str(servo_id): str(resolved_direction).strip().lower()
                        for servo_id in robot.servo_ids
                    },
                },
            }
            path = self.config_loader.save_system_local_overrides(overrides)
            self.state.mock_mode = bool(mock_mode)
            self.state.robot_config = str(robot_config)
            self.state.robot_mode = context.operating_mode
            self.state.operating_mode = context.operating_mode
            self.state.selected_servo_id = int(context.selected_servo_id or robot.selected_servo_id or 1)
            self.state.active_segment_key = robot.active_segment_key()
            self.state.active_segment_label = robot.active_segment_label()
            self.state.available_segments = self._segment_options(robot)
            self.state.active_segment_servo_ids = robot.active_segment_servo_ids()
            self.state.active_segment_pairs = robot.active_segment_pairs()
            self.state.openrb_port = str(openrb_port).strip()
            self.state.baudrate = int(baudrate)
            self.state.expected_servo_ids = list(context.expected_servo_ids)
            self.state.poll_rate_hz = int(poll_rate_hz)
            self.state.figure_output_quality = resolved_figure_quality
            self.state.fine_jog_step_ticks = int(resolved_fine_jog)
            self.state.coarse_jog_step_ticks = int(resolved_coarse_jog)
            self.state.position_min_offset_ticks = int(resolved_min_offset)
            self.state.position_max_offset_ticks = int(resolved_max_offset)
            self.state.software_position_margin_ticks = int(resolved_margin)
            self.state.telemetry_freshness_timeout_s = float(telemetry_freshness_timeout_s)
            self.state.pretension_threshold_ma = int(resolved_threshold)
            self.state.tightening_direction_default = str(resolved_direction).strip().lower()
            self.state.saved_overrides_path = str(path)
            self.state.status_message = f"Saved runtime parameters to {path}."
            self.state.last_error = None
            self.settings.runtime.robot_config = str(robot_config)
            self.settings.runtime.figure_output_quality = resolved_figure_quality
            self.settings.robot = robot
            self._apply_robot_config_to_servo_context(robot)
            LOG.info(
                "Runtime parameters saved | hardware_profile=%s | operating_mode=%s | "
                "expected_servo_ids=%s | commanded_servo_ids=%s | active_segment=%s",
                robot_config,
                context.operating_mode,
                context.expected_servo_ids,
                context.commanded_servo_ids,
                resolved_active_segment if context.active_segment_key else "none",
            )
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Save runtime parameters failed: {exc}"
            self.refresh()
            raise
        self.refresh()
        return str(path)

    def _apply_robot_config_to_servo_context(self, robot) -> None:
        calibration = getattr(self.servo_service, "neutral_calibration", None)
        context = getattr(calibration, "context", None)
        if context is None:
            return
        resolved = robot.operating_context()
        context.robot_mode = str(resolved.operating_mode)
        context.servo_ids = list(robot.servo_ids)
        context.tendon_to_servo = list(robot.tendon_to_servo)
        context.tightening_rotation_by_servo = dict(robot.tightening_rotation_by_servo)
        context.active_segment_key = robot.active_segment_key()
        context.active_segment_label = robot.active_segment_label()
        context.active_segment_servo_ids = robot.active_segment_servo_ids()
        context.active_segment_pairs = robot.active_segment_pairs()
        context.segments = robot.segment_metadata()
        context.segment_order = robot.segment_order()
        context.selected_servo_id = robot.selected_servo_id
        context.expected_servo_ids = list(resolved.expected_servo_ids)
        context.commanded_servo_ids = list(resolved.commanded_servo_ids)
        context.mirror_pairs = dict(resolved.mirror_pairs)
        context.mode_profile = resolved.mode_profile
        context.mode_capabilities = dict(resolved.mode_capabilities)
        context.mode_notes = list(resolved.mode_notes)

    @staticmethod
    def _segment_options(robot) -> list[dict[str, object]]:
        options: list[dict[str, object]] = []
        active_key = str(robot.active_segment_key())
        for key, segment in robot.segment_map().items():
            servo_ids = [int(value) for value in segment.servo_ids]
            label = str(segment.label or key)
            options.append(
                {
                    "key": str(key),
                    "label": label,
                    "servo_ids": servo_ids,
                    "pairs": {str(k): [int(v) for v in values] for k, values in dict(segment.pairs or {}).items()},
                    "display": f"{label} ({', '.join(str(value) for value in servo_ids)})",
                    "active": str(key) == active_key,
                }
            )
        return options

    @staticmethod
    def _build_config_summary(settings: Settings) -> str:
        context = settings.robot.operating_context()
        hardware_note = (
            "Servo hardware path: mock backend validated."
            if settings.runtime.mock_mode
            else "Servo hardware path: real OpenRB serial validation and DYNAMIXEL SDK transport enabled."
        )
        if context.operating_mode == "parallel_single":
            operating_scope_note = (
                "Parallel Single Demo: all 8 servos, same 4-tendon command to both spines."
            )
        elif context.operating_mode == "single_segment":
            operating_scope_note = (
                f"Single Segment: only {settings.robot.active_segment_key()} "
                f"{settings.robot.active_segment_servo_ids()}."
            )
        else:
            operating_scope_note = f"Operating scope: {context.operating_mode}."
        return (
            f"Operating mode: {context.operating_mode}\n"
            f"{operating_scope_note}\n"
            f"Expected servo IDs: {context.expected_servo_ids}\n"
            f"All configured servo IDs: {context.all_configured_servo_ids}\n"
            f"Active segment: {settings.robot.active_segment_label()} ({settings.robot.active_segment_key()}) "
            f"{settings.robot.active_segment_servo_ids()}\n"
            f"Mirror pairs: {context.mirror_pairs}\n"
            f"Mock mode: {settings.runtime.mock_mode}\n"
            f"Tracker backend: {settings.serial.tracker_backend}\n"
            f"Legacy bridge fallback backend: {settings.serial.tracker_fallback_backend}\n"
            f"Legacy bridge fallback enabled: {settings.serial.tracker_fallback_enabled}\n"
            f"Tracker type: {settings.serial.tracker_type}\n"
            f"Tracker freshness timeout: {settings.serial.tracker_freshness_timeout_s}s\n"
            f"GUI refresh rate: {settings.runtime.poll_rate_hz} Hz\n"
            f"Servo telemetry cadence: {build_telemetry_gui_policy(baudrate=settings.serial.baudrate, poll_rate_hz=settings.runtime.poll_rate_hz, telemetry_stale_after_s=settings.safety.telemetry_stale_after_s, servo_full_refresh_divisor=DEFAULT_SERVO_FULL_REFRESH_DIVISOR, system_summary_refresh_divisor=DEFAULT_SYSTEM_SUMMARY_REFRESH_DIVISOR).cadence_summary}\n"
            f"Runtime coil tool: {settings.registration.coil_tool_id}\n"
            f"Registration tool: {settings.registration.capture_tool_id}\n"
            f"Hardware profile: {settings.runtime.robot_config}\n"
            f"Robot mode: {context.operating_mode}\n"
            f"Fine/coarse jog: {settings.safety.fine_jog_step_ticks}/{settings.safety.coarse_jog_step_ticks} ticks\n"
            f"Application bounds metadata: {settings.safety.position_min_offset_ticks}..{settings.safety.position_max_offset_ticks} ticks\n"
            f"Software margin: {settings.safety.software_position_margin_ticks} ticks\n"
            f"Telemetry freshness timeout: {settings.safety.telemetry_stale_after_s}s\n"
            "Servo telemetry fields: Live = operating mode, torque enable, present position/current/voltage/temperature, hardware error. "
            "Full = live plus ID, model, firmware, current limit, min/max limits, bus watchdog.\n"
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
        if self.state.operating_mode == "parallel_single":
            operating_scope_note = "Parallel Single Demo: all 8 servos, same 4-tendon command to both spines."
        elif self.state.operating_mode == "single_segment":
            operating_scope_note = (
                f"Single Segment: only {self.state.active_segment_key} {self.state.active_segment_servo_ids}."
            )
        else:
            operating_scope_note = f"Operating scope: {self.state.operating_mode}."
        return (
            f"Hardware profile: {self.state.robot_config}\n"
            f"Operating mode: {self.state.operating_mode}\n"
            f"{operating_scope_note}\n"
            f"Selected servo: {self.state.selected_servo_id}\n"
            f"Expected servo IDs: {self.state.expected_servo_ids}\n"
            f"Active segment: {self.state.active_segment_label} ({self.state.active_segment_key}) "
            f"{self.state.active_segment_servo_ids}; pairs={self.state.active_segment_pairs}\n"
            f"Mock mode: {self.state.mock_mode}\n"
            f"OpenRB port: {self.state.openrb_port}\n"
            f"Baudrate: {self.state.baudrate}\n"
            f"GUI refresh rate: {self.state.poll_rate_hz} Hz\n"
            f"Servo telemetry cadence: {self.state.servo_telemetry_cadence_summary}\n"
            f"Fine/coarse jog: {self.state.fine_jog_step_ticks}/{self.state.coarse_jog_step_ticks} ticks\n"
            f"Application bounds metadata: {self.state.position_min_offset_ticks}/{self.state.position_max_offset_ticks} ticks\n"
            f"Software margin: {self.state.software_position_margin_ticks} ticks\n"
            f"Telemetry freshness timeout: {self.state.telemetry_freshness_timeout_s}s\n"
            f"Servo telemetry fields: {self.state.servo_telemetry_field_summary}\n"
            f"Servo telemetry bottleneck: {self.state.servo_telemetry_bottleneck_summary}\n"
            f"Default pretension threshold: {self.state.pretension_threshold_ma} mA\n"
            f"Wrap direction metadata: {self.state.tightening_direction_default}\n"
            "Bring-up raw position convention: tighten -> smaller counts, loosen -> larger counts\n"
            f"OpenRB prepared: {self.state.openrb_prepared}\n"
            f"Bus reachable: {self.state.bus_reachable}\n"
            f"Motion ready: {self.state.motion_ready}\n"
            f"Saved overrides: {self.state.saved_overrides_path or 'none'}"
            f"\nSession log: {self.state.session_log_path or 'unset'}"
        )

    @staticmethod
    def _robot_layout_display(robot_mode: str, expected_servo_ids: list[int]) -> str:
        mode = str(robot_mode or "").strip().lower().replace("-", "_")
        if mode in {"1_servo", "one_servo"}:
            return "1 Servo"
        if mode in {"4_servo", "single_segment"}:
            return "1 Segment"
        if mode == "dual_segment":
            return "2 Segments"
        if mode == "parallel_single":
            return "Parallel Single"
        servo_count = len(expected_servo_ids)
        if servo_count == 1:
            return "1 Servo"
        if servo_count == 4:
            return "1 Segment"
        if servo_count == 8:
            return "2 Segments"
        return str(robot_mode or "Unknown").replace("_", " ")

    @staticmethod
    def _default_tightening_direction(settings: Settings) -> str:
        if settings.robot.tightening_rotation_by_servo:
            first_key = sorted(settings.robot.tightening_rotation_by_servo)[0]
            return str(settings.robot.tightening_rotation_by_servo[first_key]).strip().lower()
        return "cw"

    def _expected_servo_id(self) -> int | None:
        expected = self.settings.robot.expected_servo_ids()
        if expected:
            return int(expected[0])
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
                f"session_log={self.state.session_log_path or 'unset'}",
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
                f"session_log={self.state.session_log_path or 'unset'}",
                "position_convention=tighten->smaller_counts; loosen->larger_counts",
            ]
        )

    def _build_runtime_servo_debug_text(self, snapshot) -> str:
        openrb_connected = self.openrb_client.get_status_snapshot().connected
        motion_ready_ids = [
            int(entry.servo_id)
            for entry in getattr(snapshot, "entries", {}).values()
            if bool(getattr(entry, "experiment_motion_ready", False))
            or (entry.motion_assessment is not None and entry.motion_assessment.ready)
        ]
        packet_read_ok_ids = [
            int(entry.servo_id)
            for entry in getattr(snapshot, "entries", {}).values()
            if bool(getattr(entry, "packet_read_ok", False))
        ]
        stale_display_ids = [
            int(entry.servo_id)
            for entry in getattr(snapshot, "entries", {}).values()
            if bool(getattr(entry, "stale_display_warning", False))
        ]
        return "\n".join(
            [
                "Bench debug:",
                f"openrb_connected={openrb_connected}",
                f"selected_port={self.state.openrb_port or 'unset'}",
                f"selected_baud={self.state.baudrate}",
                f"telemetry_cadence={self.state.servo_telemetry_cadence_summary}",
                f"telemetry_fields={self.state.servo_telemetry_field_summary}",
                f"telemetry_bottleneck={self.state.servo_telemetry_bottleneck_summary}",
                f"expected_servo_ids={getattr(snapshot, 'expected_servo_ids', [])}",
                f"discovered_ids={getattr(snapshot, 'detected_servo_ids', [])}",
                f"detected_servo_ids={getattr(snapshot, 'detected_servo_ids', [])}",
                f"missing_servo_ids={getattr(snapshot, 'missing_servo_ids', [])}",
                f"unexpected_servo_ids={getattr(snapshot, 'unexpected_servo_ids', [])}",
                f"packet_read_ok_ids={packet_read_ok_ids}",
                f"gui_cache_stale_ids={stale_display_ids}",
                f"motion_ready_ids={motion_ready_ids}",
                f"freshness_threshold_s={self.state.telemetry_freshness_timeout_s:.3f}",
                f"all_motion_ready={getattr(snapshot, 'all_motion_ready', False)}",
                f"active_range={self.servo_service.raw_position_range()[0]}..{self.servo_service.raw_position_range()[1]}",
                f"session_log={self.state.session_log_path or 'unset'}",
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
            f"telemetry_cadence={self.state.servo_telemetry_cadence_summary}",
            f"telemetry_fields={self.state.servo_telemetry_field_summary}",
            f"telemetry_bottleneck={self.state.servo_telemetry_bottleneck_summary}",
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
            f"one_servo_mode_ok={self.settings.robot.operating_context().operating_mode == 'one_servo'}",
            f"freshness_threshold_s={self.state.telemetry_freshness_timeout_s:.3f}",
            f"active_range={self.servo_service.raw_position_range()[0]}..{self.servo_service.raw_position_range()[1]}",
            f"session_log={self.state.session_log_path or 'unset'}",
            "motion_ready=False",
            "position_convention=tighten->smaller_counts; loosen->larger_counts",
            f"motion_block_reason={extra_error or 'OpenRB/DYNAMIXEL not ready'}",
        ]
        return "\n".join(lines)
