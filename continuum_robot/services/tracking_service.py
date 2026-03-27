"""Unified Aurora tracking service.

Production live path:
    NDITracker/scikit-surgerynditracker -> backend manager -> TrackingService

Compatibility path retained for replay and parser regression tests:
    AuroraClient -> AuroraFramer -> AuroraParser -> TrackingService
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

try:
    from serial import SerialException
except Exception:  # pragma: no cover - pyserial is a declared dependency
    class SerialException(Exception):
        """Fallback serial exception when pyserial import is unavailable."""


from continuum_robot.hardware.aurora_client import AuroraClient
from continuum_robot.services.models import (
    HEALTH_DEGRADED,
    HEALTH_FAILED,
    HEALTH_HEALTHY,
    ServiceHealthSnapshot,
    ToolTrackingSnapshot,
    TrackingSnapshot,
)
from continuum_robot.services.packet_capture import PacketCaptureWriter, iter_packet_capture_records
from continuum_robot.tracking.aurora_framer import AuroraFramer
from continuum_robot.tracking.aurora_parser import AuroraParser
from continuum_robot.tracking.tip_pose_service import TipPoseService
from continuum_robot.tracking.tool_models import AuroraToolMeasurement
from continuum_robot.tracking.transforms import assert_rigid_transform_matrix, make_transform_A_B
from continuum_robot.utils.time_utils import utc_now_iso


class TrackingService:
    """Owns the app-visible tracking state cache and runtime tip-pose chain."""

    SUPPORTED_TOOL_IDS = ("0A", "0B")
    _FAILED_FAULTS = {
        "no_serial_connection",
        "invalid_registration",
        "registration_role_mismatch",
        "invalid_transform_chain",
    }

    def __init__(
        self,
        aurora_client: AuroraClient | None = None,
        *,
        live_backend=None,
        port: str,
        baudrate: int = 115200,
        read_timeout_s: float = 0.05,
        frame_timeout_s: float = 0.5,
        reconnect_delay_s: float = 1.0,
        live_poll_hz: float = 20.0,
        stale_after_s: float = 0.5,
        registration_path: Path | None = None,
        runtime_coil_tool_id: str = "0A",
        registration_tool_id: str = "0B",
        config_source: str = "config/system.yaml",
        backend_identity: str | None = None,
        parser: AuroraParser | None = None,
        framer_factory=AuroraFramer,
    ) -> None:
        if live_backend is None and aurora_client is None:
            raise ValueError("TrackingService requires either a live_backend or an aurora_client")
        if runtime_coil_tool_id not in self.SUPPORTED_TOOL_IDS:
            raise ValueError(f"Unsupported runtime coil tool id: {runtime_coil_tool_id}")
        if registration_tool_id not in self.SUPPORTED_TOOL_IDS:
            raise ValueError(f"Unsupported registration tool id: {registration_tool_id}")

        self.aurora_client = aurora_client
        self.live_backend = live_backend
        self.port = port
        self.baudrate = baudrate
        self.read_timeout_s = read_timeout_s
        self.frame_timeout_s = frame_timeout_s
        self.reconnect_delay_s = reconnect_delay_s
        self.live_poll_hz = max(1.0, float(live_poll_hz))
        self.stale_after_s = max(0.05, float(stale_after_s))
        self.runtime_coil_tool_id = runtime_coil_tool_id
        self.registration_tool_id = registration_tool_id
        self.config_source = config_source
        self.parser = parser or AuroraParser()
        self.framer_factory = framer_factory
        self._uses_live_backend = live_backend is not None
        self.backend_identity = backend_identity or self._default_backend_identity(live_backend, aurora_client)

        project_root = Path(__file__).resolve().parents[2]
        self.registration_path = registration_path or (project_root / "data" / "registrations" / "latest_registration.json")

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture_writer: PacketCaptureWriter | None = None
        self._tip_service: TipPoseService | None = None
        self._tip_calibration_source: str | None = None
        self._registration_role_error: str | None = None
        self._registration_measurement_tool_id: str | None = None
        self._registration_coil_tool_id: str | None = None
        self._backend_started_here = False
        self._started_monotonic: float | None = None
        self._last_frame_monotonic: float | None = None

        initial_connection_state = "disconnected" if self._uses_live_backend else "stopped"
        self._state = TrackingSnapshot(
            health=ServiceHealthSnapshot(
                name="tracking_service",
                health=HEALTH_DEGRADED,
                state=initial_connection_state,
                status="Tracking service stopped",
                current_config_source=config_source,
            ),
            connection_state=initial_connection_state,
            backend_identity=self.backend_identity,
            port=port,
            baudrate=baudrate,
            runtime_coil_tool_id=runtime_coil_tool_id,
            registration_tool_id=registration_tool_id,
            backend_running=False if self._uses_live_backend else None,
            backend_connected=False if self._uses_live_backend else None,
            tools={tool_id: self._blank_tool(tool_id) for tool_id in self.SUPPORTED_TOOL_IDS},
        )
        self.refresh_registration()
        with self._lock:
            self._recompute_health_locked()

    def set_port(self, port: str) -> None:
        """Update the configured Aurora device path for the live backend."""
        self.port = port
        if self.live_backend is not None and hasattr(self.live_backend, "aurora_port"):
            self.live_backend.aurora_port = port
        with self._lock:
            self._state.port = port

    def start(self, port: str | None = None) -> None:
        """Start the live backend poll loop or the compatibility serial loop."""
        if port is not None:
            self.set_port(port)
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._state.port = self.port
            self._state.backend_identity = self.backend_identity
            self._state.connection_state = "starting" if self._uses_live_backend else "connecting"
            self._state.health.state = self._state.connection_state
            self._state.health.health = HEALTH_DEGRADED
            self._state.health.status = f"Starting tracking backend {self.backend_identity}"
            self._state.health.last_error = None
            self._state.last_error = None
        self._stop_event.clear()
        self._started_monotonic = time.monotonic()
        self._last_frame_monotonic = None
        self.refresh_registration()

        if self._uses_live_backend:
            try:
                self._start_live_backend()
            except Exception as exc:
                with self._lock:
                    self._state.connection_state = "error"
                    self._state.health.state = "error"
                    self._state.health.last_error = str(exc)
                    self._state.last_error = str(exc)
                    self._recompute_health_locked()
                raise
            self._thread = threading.Thread(target=self._run_live_loop, daemon=True)
            self._thread.start()
            return

        self._thread = threading.Thread(target=self._run_compatibility_loop, daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        """Stop the active runtime path."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout_s)
            self._thread = None

        if self._uses_live_backend:
            if self._backend_started_here and hasattr(self.live_backend, "stop"):
                self.live_backend.stop()
            self._backend_started_here = False
        else:
            self._safe_disconnect()

        self.disable_packet_capture()
        self._started_monotonic = None
        self._last_frame_monotonic = None
        with self._lock:
            self._state.connection_state = "stopped"
            if self._uses_live_backend:
                self._state.backend_running = False
                self._state.backend_connected = False
                self._state.bridge_running = False
                self._state.socket_connected = False
            self._state.health.state = "stopped"
            self._state.health.health = HEALTH_DEGRADED
            self._state.health.status = "Tracking service stopped"
            self._recompute_health_locked()

    def reconnect(self) -> None:
        """Restart the configured runtime path."""
        self.stop()
        with self._lock:
            self._state.reconnect_count += 1
        self.start()

    def enable_packet_capture(self, path: Path, include_summary: bool = True) -> None:
        """Capture compatibility/replay frames to an append-only JSONL file."""
        self.disable_packet_capture()
        self._capture_writer = PacketCaptureWriter(path, include_summary=include_summary)
        with self._lock:
            self._state.packet_capture_enabled = True
            self._state.packet_capture_path = str(path)

    def disable_packet_capture(self) -> None:
        """Stop capture output."""
        if self._capture_writer is not None:
            self._capture_writer.close()
            self._capture_writer = None
        with self._lock:
            self._state.packet_capture_enabled = False

    def refresh_registration(self) -> None:
        """Load the current accepted registration and validate runtime tool roles."""
        with self._lock:
            self._tip_service = None
            self._tip_calibration_source = None
            self._registration_role_error = None
            self._registration_measurement_tool_id = None
            self._registration_coil_tool_id = None
            self._state.registration_path = str(self.registration_path)
            self._state.stored_registration_measurement_tool_id = None
            self._state.stored_registration_coil_tool_id = None
            self._state.registration_state = "missing_registration"
            self._state.tip_calibration_source = None
            self._state.tip_pose_status = "missing_registration"

        if not self.registration_path.exists():
            with self._lock:
                self._recompute_health_locked()
            return

        try:
            payload = json.loads(self.registration_path.read_text(encoding="utf-8"))
            tip_service = TipPoseService.from_registration_file(self.registration_path)
            config_used = payload.get("config_used", {}) if isinstance(payload.get("config_used"), dict) else {}
            tip_calibration_source = config_used.get("tip_calibration_source") or "unknown"
            measurement_tool_id = (
                config_used.get("measurement_tool_id")
                or config_used.get("capture_tool_id")
                or payload.get("measurement_tool_id")
            )
            coil_tool_id = config_used.get("coil_tool_id") or payload.get("coil_tool_id")
            role_error = self._build_registration_role_error(
                measurement_tool_id=str(measurement_tool_id) if measurement_tool_id is not None else None,
                coil_tool_id=str(coil_tool_id) if coil_tool_id is not None else None,
            )
        except Exception as exc:
            with self._lock:
                self._state.registration_state = "invalid_registration"
                self._state.health.last_error = str(exc)
                self._state.last_error = str(exc)
                self._state.tip_pose_status = "invalid_registration"
                self._recompute_health_locked()
            return

        with self._lock:
            self._tip_service = tip_service
            self._tip_calibration_source = tip_calibration_source
            self._registration_role_error = role_error
            self._registration_measurement_tool_id = (
                str(measurement_tool_id) if measurement_tool_id is not None else None
            )
            self._registration_coil_tool_id = str(coil_tool_id) if coil_tool_id is not None else None
            self._state.registration_state = "loaded"
            self._state.stored_registration_measurement_tool_id = self._registration_measurement_tool_id
            self._state.stored_registration_coil_tool_id = self._registration_coil_tool_id
            self._state.tip_calibration_source = tip_calibration_source
            self._state.health.last_error = None
            self._state.last_error = None
            if role_error is not None:
                self._state.tip_pose_status = "role_mismatch"
                self._state.health.last_error = role_error
                self._state.last_error = role_error
            self._recompute_health_locked()

    def ingest_frame(
        self,
        frame: bytes,
        *,
        source: str,
        received_at_utc: str | None = None,
        record_capture: bool = True,
    ) -> None:
        """Parse one compatibility frame into the shared cache."""
        packet_timestamp = received_at_utc or utc_now_iso()
        summary: dict[str, Any] | None = None
        parse_error: str | None = None

        try:
            payload = self.parser.parse_payload(frame)
            measurements = self.parser.parse_transform_packet(frame)
            summary = {
                "frame_number": payload.header.frame_number,
                "tool_count": payload.header.tool_count,
                "tool_ids": sorted(measurements.keys()),
                "crc_received": payload.crc_received,
                "crc_computed": payload.crc_computed,
            }
            with self._lock:
                self._state.packets_received_count += 1
                self._state.last_frame_number = payload.header.frame_number
                self._state.last_packet_utc = packet_timestamp
                self._state.health.last_successful_update_utc = packet_timestamp
                self._apply_compatibility_measurements_locked(
                    measurements,
                    frame_number=payload.header.frame_number,
                    packet_timestamp=packet_timestamp,
                )
                self._update_tip_pose_locked(packet_timestamp)
                self._state.connection_state = "tracking"
                self._state.health.state = "tracking"
                self._state.health.last_error = None
                self._state.last_error = None
                self._recompute_health_locked()
        except ValueError as exc:
            parse_error = str(exc)
            with self._lock:
                self._state.bad_packets_count += 1
                if "CRC mismatch" in parse_error:
                    self._state.crc_failures_count += 1
                self._state.health.last_error = parse_error
                self._state.last_error = parse_error
                if self._state.connection_state != "stopped":
                    self._state.connection_state = "degraded"
                    self._state.health.state = "degraded"
                self._recompute_health_locked()
        finally:
            if record_capture and self._capture_writer is not None:
                self._capture_writer.write_frame(
                    frame,
                    captured_at_utc=packet_timestamp,
                    source=source,
                    summary=summary,
                    parse_error=parse_error,
                )

    def replay_capture(
        self,
        capture_path: Path,
        *,
        reset_state: bool = True,
        delay_s: float = 0.0,
    ) -> TrackingSnapshot:
        """Replay a captured compatibility JSONL stream through the parser path."""
        if reset_state:
            self._reset_runtime_state()
        self.refresh_registration()
        with self._lock:
            self._state.connection_state = "replay"
            self._state.health.state = "replay"
            self._state.health.status = f"Replaying capture from {capture_path}"
            self._recompute_health_locked()

        for record in iter_packet_capture_records(capture_path):
            self.ingest_frame(
                record.frame_bytes,
                source="replay",
                received_at_utc=record.captured_at_utc,
                record_capture=False,
            )
            if delay_s > 0:
                time.sleep(delay_s)
        return self.get_snapshot()

    def get_snapshot(self) -> TrackingSnapshot:
        """Return a deep copy of the current tracking state."""
        with self._lock:
            return copy.deepcopy(self._state)

    def get_latest_tool(self, tool_id: str) -> ToolTrackingSnapshot | None:
        """Return a deep copy of one tool snapshot."""
        with self._lock:
            tool = self._state.tools.get(tool_id)
            return copy.deepcopy(tool) if tool is not None else None

    def _start_live_backend(self) -> None:
        already_alive = bool(getattr(self.live_backend, "is_alive", lambda: False)())
        self._backend_started_here = not already_alive
        if not already_alive and hasattr(self.live_backend, "start"):
            self.live_backend.start()
        self._sync_live_backend_snapshot()

    def _run_live_loop(self) -> None:
        poll_interval_s = 1.0 / self.live_poll_hz
        while not self._stop_event.is_set():
            try:
                self._sync_live_backend_snapshot()
            except Exception as exc:
                with self._lock:
                    self._state.connection_state = "error"
                    self._state.health.state = "error"
                    self._state.health.last_error = str(exc)
                    self._state.last_error = str(exc)
                    self._recompute_health_locked()
            time.sleep(poll_interval_s)

    def _run_compatibility_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._connect_client()
                framer = self.framer_factory()
                while not self._stop_event.is_set():
                    try:
                        frame = framer.read_next_frame(self.aurora_client.read_bytes, timeout_s=self.frame_timeout_s)
                    except TimeoutError:
                        with self._lock:
                            if self._state.packets_received_count == 0:
                                self._state.connection_state = "waiting_for_packets"
                                self._state.health.state = "waiting_for_packets"
                            self._recompute_health_locked()
                        continue
                    self.ingest_frame(frame, source="compatibility_serial")
            except (OSError, RuntimeError, SerialException) as exc:
                if self._stop_event.is_set():
                    break
                with self._lock:
                    self._state.reconnect_count += 1
                    self._state.connection_state = "reconnecting"
                    self._state.health.state = "reconnecting"
                    self._state.health.last_error = str(exc)
                    self._state.last_error = str(exc)
                    self._recompute_health_locked()
                self._safe_disconnect()
                time.sleep(self.reconnect_delay_s)

    def _connect_client(self) -> None:
        with self._lock:
            self._state.connection_state = "connecting"
            self._state.health.state = "connecting"
            self._recompute_health_locked()
        self.aurora_client.connect(self.port, baudrate=self.baudrate, timeout_s=self.read_timeout_s)
        with self._lock:
            self._state.connection_state = "waiting_for_packets"
            self._state.health.state = "waiting_for_packets"
            self._recompute_health_locked()

    def _safe_disconnect(self) -> None:
        try:
            if self.aurora_client is not None:
                self.aurora_client.disconnect()
        except Exception:
            return

    def _sync_live_backend_snapshot(self) -> None:
        snapshot = self.live_backend.get_state_snapshot()
        synced_at_utc = utc_now_iso()
        synced_at_monotonic = time.monotonic()
        with self._lock:
            self._apply_live_backend_snapshot_locked(snapshot, synced_at_utc, synced_at_monotonic)
            self._update_tip_pose_locked(snapshot.latest_timestamp or synced_at_utc)
            self._recompute_health_locked()

    def _apply_live_backend_snapshot_locked(self, snapshot, synced_at_utc: str, synced_at_monotonic: float) -> None:
        latest_timestamp = snapshot.latest_timestamp or synced_at_utc
        previous_frame = self._state.last_frame_number
        current_frame = snapshot.latest_frame_number

        self._state.connection_state = snapshot.connection_state
        backend_running = getattr(snapshot, "backend_running", None)
        if not backend_running and getattr(snapshot, "bridge_running", False):
            backend_running = True
        backend_connected = getattr(snapshot, "backend_connected", None)
        if not backend_connected and getattr(snapshot, "socket_connected", False):
            backend_connected = True
        self._state.backend_running = backend_running
        self._state.backend_connected = backend_connected
        self._state.bridge_running = getattr(snapshot, "bridge_running", self._state.backend_running)
        self._state.socket_connected = getattr(snapshot, "socket_connected", self._state.backend_connected)
        self._state.backend_status_message = snapshot.last_status_message or None
        self._state.health.state = snapshot.connection_state
        if snapshot.last_error:
            self._state.health.last_error = snapshot.last_error
            self._state.last_error = snapshot.last_error

        if current_frame is not None and current_frame != previous_frame:
            self._state.packets_received_count += 1
            self._state.last_frame_number = current_frame
            self._state.last_packet_utc = latest_timestamp
            self._state.health.last_successful_update_utc = latest_timestamp
            self._last_frame_monotonic = synced_at_monotonic
            if self._started_monotonic is not None and self._state.first_frame_latency_s is None:
                self._state.first_frame_latency_s = max(0.0, synced_at_monotonic - self._started_monotonic)

        for tool_id in self.SUPPORTED_TOOL_IDS:
            self._apply_live_tool_snapshot_locked(
                tool_id=tool_id,
                raw_tool=snapshot.tools.get(tool_id),
                default_frame_number=current_frame,
                default_timestamp=latest_timestamp,
            )

        if self._last_frame_monotonic is None:
            self._state.tracker_data_age_s = None
            self._state.tracker_data_stale = False
        else:
            self._state.tracker_data_age_s = max(0.0, synced_at_monotonic - self._last_frame_monotonic)
            self._state.tracker_data_stale = self._state.tracker_data_age_s > self.stale_after_s

    def _apply_live_tool_snapshot_locked(
        self,
        *,
        tool_id: str,
        raw_tool,
        default_frame_number: int | None,
        default_timestamp: str,
    ) -> None:
        tool_state = self._state.tools.setdefault(tool_id, self._blank_tool(tool_id))
        if raw_tool is None:
            inferred_state = "missing" if default_frame_number is not None else "unknown"
            self._set_tool_state_locked(
                tool_state,
                frame_number=default_frame_number,
                packet_timestamp=default_timestamp,
                tracking_state=inferred_state,
                status="missing_from_live_backend" if inferred_state == "missing" else "no_sample",
                valid=None,
                validity_known=False,
                quat=None,
                translation=None,
                quality=None,
            )
            return

        tracking_state = self._normalize_tracking_state(raw_tool.status, raw_tool.valid)
        raw_frame_number = raw_tool.frame_number if raw_tool.frame_number is not None else default_frame_number
        raw_timestamp = raw_tool.timestamp or default_timestamp
        raw_validity_known = bool(getattr(raw_tool, "validity_known", raw_tool.valid is not None))
        raw_valid = raw_tool.valid if raw_validity_known else None

        if tracking_state in {"missing", "unknown"}:
            self._set_tool_state_locked(
                tool_state,
                frame_number=raw_frame_number,
                packet_timestamp=raw_timestamp,
                tracking_state=tracking_state,
                status=raw_tool.status,
                valid=raw_valid,
                validity_known=raw_validity_known,
                quat=None,
                translation=None,
                quality=raw_tool.quality,
            )
            return

        quat = tuple(float(v) for v in (raw_tool.quaternion or ()))
        translation = tuple(float(v) for v in (raw_tool.translation_mm or ()))
        if len(quat) != 4 or len(translation) != 3:
            self._set_tool_state_locked(
                tool_state,
                frame_number=raw_frame_number,
                packet_timestamp=raw_timestamp,
                tracking_state="invalid",
                status="invalid_transform: missing quaternion/translation payload",
                valid=False,
                validity_known=True,
                quat=None,
                translation=None,
                quality=raw_tool.quality,
            )
            return
        try:
            T_aurora_tool = make_transform_A_B(quat, translation)
            assert_rigid_transform_matrix(T_aurora_tool, f"T_aurora_{tool_id}")
        except Exception as exc:
            self._set_tool_state_locked(
                tool_state,
                frame_number=raw_frame_number,
                packet_timestamp=raw_timestamp,
                tracking_state="invalid",
                status=f"invalid_transform: {exc}",
                valid=False,
                validity_known=True,
                quat=quat,
                translation=translation,
                quality=raw_tool.quality,
            )
            return

        self._set_tool_state_locked(
            tool_state,
            frame_number=raw_frame_number,
            packet_timestamp=raw_timestamp,
            tracking_state=tracking_state,
            status=raw_tool.status,
            valid=raw_valid,
            validity_known=raw_validity_known,
            quat=quat,
            translation=translation,
            quality=raw_tool.quality,
            T_aurora_tool=T_aurora_tool,
        )

    def _apply_compatibility_measurements_locked(
        self,
        measurements: dict[str, AuroraToolMeasurement],
        *,
        frame_number: int,
        packet_timestamp: str,
    ) -> None:
        observed = set(measurements.keys())
        for tool_id in self.SUPPORTED_TOOL_IDS:
            tool_state = self._state.tools.setdefault(tool_id, self._blank_tool(tool_id))
            measurement = measurements.get(tool_id)
            if measurement is None:
                self._set_tool_state_locked(
                    tool_state,
                    frame_number=frame_number,
                    packet_timestamp=packet_timestamp,
                    tracking_state="missing",
                    status="missing_from_packet",
                    valid=None,
                    validity_known=False,
                    quat=None,
                    translation=None,
                    quality=None,
                )
                continue

            tracking_state = "tracked"
            if measurement.valid is False:
                tracking_state = "invalid"

            self._apply_measurement_tool_locked(
                tool_state=tool_state,
                frame_number=frame_number,
                packet_timestamp=packet_timestamp,
                measurement=measurement,
                tracking_state=tracking_state,
            )

        for tool_id in self.SUPPORTED_TOOL_IDS:
            if tool_id not in observed and tool_id in self._state.latest_measurements:
                self._state.latest_measurements[tool_id]["tracking_state"] = self._state.tools[tool_id].tracking_state

    def _apply_measurement_tool_locked(
        self,
        *,
        tool_state: ToolTrackingSnapshot,
        frame_number: int,
        packet_timestamp: str,
        measurement: AuroraToolMeasurement,
        tracking_state: str,
    ) -> None:
        quat = tuple(float(v) for v in measurement.quat_wxyz)
        translation = tuple(float(v) for v in measurement.translation_xyz)
        T_aurora_tool = None
        status = measurement.status_text
        valid = measurement.valid
        validity_known = measurement.valid is not None
        if tracking_state == "tracked":
            try:
                T_aurora_tool = make_transform_A_B(quat, translation)
                assert_rigid_transform_matrix(T_aurora_tool, f"T_aurora_{measurement.tool_id}")
            except Exception as exc:
                tracking_state = "invalid"
                status = f"invalid_transform: {exc}"
                valid = False
                validity_known = True

        self._set_tool_state_locked(
            tool_state,
            frame_number=frame_number,
            packet_timestamp=packet_timestamp,
            tracking_state=tracking_state,
            status=status,
            valid=valid,
            validity_known=validity_known,
            quat=quat if tracking_state != "missing" else None,
            translation=translation if tracking_state != "missing" else None,
            quality=measurement.quality,
            T_aurora_tool=T_aurora_tool,
        )

    def _set_tool_state_locked(
        self,
        tool_state: ToolTrackingSnapshot,
        *,
        frame_number: int | None,
        packet_timestamp: str | None,
        tracking_state: str,
        status: str,
        valid: bool | None,
        validity_known: bool,
        quat: tuple[float, float, float, float] | None,
        translation: tuple[float, float, float] | None,
        quality: float | None,
        T_aurora_tool: np.ndarray | None = None,
    ) -> None:
        tool_state.tracking_state = tracking_state
        tool_state.present = tracking_state in {"tracked", "invalid"}
        tool_state.valid = valid
        tool_state.validity_known = validity_known
        tool_state.status = status
        tool_state.frame_number = frame_number
        tool_state.last_update_utc = packet_timestamp
        tool_state.quaternion_wxyz = quat
        tool_state.translation_mm = translation
        tool_state.quality = quality
        tool_state.T_aurora_tool = T_aurora_tool.tolist() if T_aurora_tool is not None else None

        if tracking_state == "tracked" and T_aurora_tool is not None:
            tool_state.last_good_frame_number = frame_number
            tool_state.last_good_update_utc = packet_timestamp
            tool_state.last_good_quaternion_wxyz = quat
            tool_state.last_good_translation_mm = translation
            tool_state.last_good_T_aurora_tool = T_aurora_tool.tolist()

        self._state.latest_measurements[tool_state.tool_id] = {
            "frame_number": frame_number,
            "timestamp_utc": packet_timestamp,
            "tracking_state": tracking_state,
            "valid": valid,
            "validity_known": validity_known,
            "quaternion_wxyz": list(quat) if quat is not None else None,
            "translation_mm": list(translation) if translation is not None else None,
            "quality": quality,
            "status": status,
        }

    def _update_tip_pose_locked(self, packet_timestamp: str) -> None:
        self._state.T_robot_tip = None
        self._state.tip_pose_timestamp_utc = None

        if self._tip_service is None:
            self._state.tip_pose_status = self._state.registration_state
            return

        if self._registration_role_error is not None:
            self._state.tip_pose_status = "role_mismatch"
            return

        coil_tool = self._state.tools[self.runtime_coil_tool_id]
        if coil_tool.tracking_state != "tracked":
            self._state.tip_pose_status = f"missing_{self.runtime_coil_tool_id}"
            return
        if coil_tool.T_aurora_tool is None:
            self._state.tip_pose_status = "invalid_transform_chain"
            return

        try:
            T_aurora_coil = np.asarray(coil_tool.T_aurora_tool, dtype=float)
            assert_rigid_transform_matrix(T_aurora_coil, f"T_aurora_{self.runtime_coil_tool_id}")
            T_robot_tip = self._tip_service.compute_T_robot_tip(
                T_robot_aurora=self._tip_service.inputs.T_robot_aurora,
                T_aurora_coil=T_aurora_coil,
                T_coil_tip=self._tip_service.inputs.T_coil_tip,
            )
        except Exception as exc:
            self._state.tip_pose_status = "invalid_transform_chain"
            self._state.health.last_error = str(exc)
            self._state.last_error = str(exc)
            return

        self._state.tip_pose_status = "ok"
        self._state.T_robot_tip = T_robot_tip.tolist()
        self._state.tip_pose_timestamp_utc = packet_timestamp
        self._state.last_good_T_robot_tip = T_robot_tip.tolist()
        self._state.last_good_tip_pose_utc = packet_timestamp

    def _reset_runtime_state(self) -> None:
        with self._lock:
            self._state.packets_received_count = 0
            self._state.bad_packets_count = 0
            self._state.crc_failures_count = 0
            self._state.last_frame_number = None
            self._state.last_packet_utc = None
            self._state.T_robot_tip = None
            self._state.tip_pose_timestamp_utc = None
            self._state.last_good_T_robot_tip = None
            self._state.last_good_tip_pose_utc = None
            self._state.latest_measurements = {}
            self._state.tools = {tool_id: self._blank_tool(tool_id) for tool_id in self.SUPPORTED_TOOL_IDS}
            self._state.health.last_error = None
            self._state.last_error = None
            self._recompute_health_locked()

    def _recompute_health_locked(self) -> None:
        faults: list[str] = []

        if self._uses_live_backend:
            if self._state.connection_state == "stopped":
                faults.append("service_stopped")
            elif self._state.connection_state == "error":
                faults.append("no_serial_connection")
            elif self._state.connection_state in {"starting", "connecting", "reconnecting"}:
                faults.append("no_serial_connection")
            elif self._state.backend_running is False:
                faults.append("backend_not_running")
            elif self._state.backend_connected is False:
                faults.append("backend_not_connected")
            if self._state.connection_state == "tracking" and self._state.packets_received_count == 0:
                faults.append("no_packets")
            if self._state.tracker_data_stale:
                faults.append("stale_tracker_data")
        else:
            if self._state.connection_state in {"connecting", "reconnecting"}:
                faults.append("no_serial_connection")
            elif self._state.connection_state == "waiting_for_packets" and self._state.packets_received_count == 0:
                faults.append("no_packets")
            elif self._state.connection_state == "stopped":
                faults.append("service_stopped")

        if self._state.bad_packets_count > 0:
            faults.append("bad_packets")

        for tool_id in self.SUPPORTED_TOOL_IDS:
            tool = self._state.tools[tool_id]
            if self._state.packets_received_count <= 0:
                continue
            if tool.tracking_state == "missing":
                faults.append(f"missing_{tool_id}")
            elif tool.tracking_state == "invalid":
                faults.append(f"invalid_{tool_id}")

        if self._state.registration_state == "missing_registration":
            faults.append("missing_registration")
        elif self._state.registration_state == "invalid_registration":
            faults.append("invalid_registration")
        if self._registration_role_error is not None:
            faults.append("registration_role_mismatch")
        if self._state.tip_pose_status == "invalid_transform_chain":
            faults.append("invalid_transform_chain")

        self._state.faults = faults
        self._state.health.current_config_source = self.config_source
        self._state.last_error = self._state.health.last_error
        self._state.health.details = {
            "backend_identity": self.backend_identity,
            "faults": list(faults),
            "runtime_coil_tool_id": self.runtime_coil_tool_id,
            "registration_tool_id": self.registration_tool_id,
            "stored_registration_measurement_tool_id": self._registration_measurement_tool_id,
            "stored_registration_coil_tool_id": self._registration_coil_tool_id,
            "packets_received_count": self._state.packets_received_count,
            "bad_packets_count": self._state.bad_packets_count,
            "crc_failures_count": self._state.crc_failures_count,
            "registration_state": self._state.registration_state,
            "tip_pose_status": self._state.tip_pose_status,
            "backend_running": self._state.backend_running,
            "backend_connected": self._state.backend_connected,
            "bridge_running": self._state.bridge_running,
            "socket_connected": self._state.socket_connected,
            "backend_status_message": self._state.backend_status_message,
            "tracker_data_age_s": self._state.tracker_data_age_s,
            "tracker_data_stale": self._state.tracker_data_stale,
            "first_frame_latency_s": self._state.first_frame_latency_s,
        }

        if any(fault in self._FAILED_FAULTS for fault in faults):
            health = HEALTH_FAILED
        elif faults:
            health = HEALTH_DEGRADED
        else:
            health = HEALTH_HEALTHY

        self._state.health.health = health
        if health == HEALTH_HEALTHY:
            self._state.health.status = (
                f"Tracking healthy via {self.backend_identity}: "
                f"tools 0A/0B tracked and registration loaded"
            )
            return

        if self._state.connection_state == "stopped":
            self._state.health.status = "Tracking service stopped"
            return

        if faults:
            self._state.health.status = (
                f"Tracking {health} via {self.backend_identity}: " + ", ".join(faults)
            )
            return

        self._state.health.status = f"Tracking state unavailable via {self.backend_identity}"

    def _build_registration_role_error(
        self,
        *,
        measurement_tool_id: str | None,
        coil_tool_id: str | None,
    ) -> str | None:
        issues: list[str] = []
        if measurement_tool_id is not None and measurement_tool_id != self.registration_tool_id:
            issues.append(
                f"registration file expects measurement tool {measurement_tool_id} but runtime registration tool is {self.registration_tool_id}"
            )
        if coil_tool_id is not None and coil_tool_id != self.runtime_coil_tool_id:
            issues.append(
                f"registration file expects coil tool {coil_tool_id} but runtime coil tool is {self.runtime_coil_tool_id}"
            )
        if not issues:
            return None
        return "; ".join(issues)

    @classmethod
    def _default_backend_identity(cls, live_backend, aurora_client: AuroraClient | None) -> str:
        if live_backend is not None:
            identity = getattr(live_backend, "backend_identity", None)
            if identity:
                return str(identity)
            name = type(live_backend).__name__
            if name == "MockTrackerManager":
                return "mock_tracker_manager"
            if name == "TrackerBackendNDI":
                return "ndi_tracker_python"
            return "tracker_bridge_json"
        if aurora_client is not None:
            return "legacy_client_packet_compat"
        return "unknown_backend"

    @staticmethod
    def _normalize_tracking_state(status: str, raw_valid: bool | None) -> str:
        normalized = (status or "").strip().lower()
        if normalized.startswith("invalid"):
            return "invalid"
        if normalized in {"tracked", "visible", "ok"}:
            return "tracked"
        if normalized in {"missing", "missing_from_packet", "absent", "not_tracked"}:
            return "missing"
        if normalized in {"invalid", "bad_transform"}:
            return "invalid"
        if raw_valid is True:
            return "tracked"
        if raw_valid is False:
            return "invalid" if "invalid" in normalized else "missing"
        return "unknown"

    @staticmethod
    def _blank_tool(tool_id: str) -> ToolTrackingSnapshot:
        return ToolTrackingSnapshot(tool_id=tool_id)
