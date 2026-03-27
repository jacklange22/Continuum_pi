"""Direct Aurora tracking service with health reporting and replay support."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import threading
import time

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
from continuum_robot.tracking.transforms import make_transform_A_B
from continuum_robot.utils.time_utils import utc_now_iso


class TrackingService:
    """Read, parse, cache, and diagnose Aurora tracking data from the serial stream."""

    SUPPORTED_TOOL_IDS = ("0A", "0B")

    def __init__(
        self,
        aurora_client: AuroraClient,
        *,
        port: str,
        baudrate: int = 115200,
        read_timeout_s: float = 0.05,
        frame_timeout_s: float = 0.5,
        reconnect_delay_s: float = 1.0,
        registration_path: Path | None = None,
        config_source: str = "config/system.yaml",
        parser: AuroraParser | None = None,
        framer_factory=AuroraFramer,
    ) -> None:
        self.aurora_client = aurora_client
        self.port = port
        self.baudrate = baudrate
        self.read_timeout_s = read_timeout_s
        self.frame_timeout_s = frame_timeout_s
        self.reconnect_delay_s = reconnect_delay_s
        self.config_source = config_source
        self.parser = parser or AuroraParser()
        self.framer_factory = framer_factory

        project_root = Path(__file__).resolve().parents[2]
        self.registration_path = registration_path or (project_root / "data" / "registrations" / "latest_registration.json")

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture_writer: PacketCaptureWriter | None = None
        self._tip_service: TipPoseService | None = None
        self._tip_calibration_source: str | None = None

        self._state = TrackingSnapshot(
            health=ServiceHealthSnapshot(
                name="tracking_service",
                health=HEALTH_DEGRADED,
                state="stopped",
                status="Tracking service stopped",
                current_config_source=config_source,
            ),
            connection_state="stopped",
            port=port,
            baudrate=baudrate,
            tools={tool_id: self._blank_tool(tool_id) for tool_id in self.SUPPORTED_TOOL_IDS},
        )
        self.refresh_registration()
        with self._lock:
            self._recompute_health_locked()

    def start(self, port: str | None = None) -> None:
        """Start the background Aurora read loop."""
        if port is not None:
            self.port = port
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._state.port = self.port
            self._state.connection_state = "connecting"
            self._state.health.state = "connecting"
            self._state.health.health = HEALTH_DEGRADED
            self._state.health.status = f"Connecting to Aurora on {self.port or '<unset>'}"
            self._state.health.last_error = None
        self._stop_event.clear()
        self.refresh_registration()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        """Stop the background read loop and close the serial connection."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self._thread = None
        self._safe_disconnect()
        self.disable_packet_capture()
        with self._lock:
            self._state.connection_state = "stopped"
            self._state.health.state = "stopped"
            self._state.health.health = HEALTH_DEGRADED
            self._state.health.status = "Tracking service stopped"
            self._recompute_health_locked()

    def reconnect(self) -> None:
        """Restart the serial connection and read loop."""
        self.stop()
        self.start()

    def enable_packet_capture(self, path: Path, include_summary: bool = True) -> None:
        """Capture raw Aurora frames to an append-only JSONL file."""
        self.disable_packet_capture()
        self._capture_writer = PacketCaptureWriter(path, include_summary=include_summary)
        with self._lock:
            self._state.packet_capture_enabled = True
            self._state.packet_capture_path = str(path)

    def disable_packet_capture(self) -> None:
        """Stop capturing Aurora frames."""
        if self._capture_writer is not None:
            self._capture_writer.close()
            self._capture_writer = None
        with self._lock:
            self._state.packet_capture_enabled = False

    def refresh_registration(self) -> None:
        """Load the latest accepted registration for runtime tip pose computation."""
        with self._lock:
            self._tip_service = None
            self._tip_calibration_source = None
            self._state.registration_path = str(self.registration_path)
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
            tip_calibration_source = (
                payload.get("config_used", {}).get("tip_calibration_source")
                if isinstance(payload.get("config_used"), dict)
                else None
            ) or "unknown"
        except Exception as exc:
            with self._lock:
                self._state.registration_state = "invalid_registration"
                self._state.health.last_error = str(exc)
                self._state.tip_pose_status = "missing_registration"
                self._recompute_health_locked()
            return

        with self._lock:
            self._tip_service = tip_service
            self._tip_calibration_source = tip_calibration_source
            self._state.registration_state = "loaded"
            self._state.tip_calibration_source = tip_calibration_source
            self._recompute_health_locked()

    def ingest_frame(
        self,
        frame: bytes,
        *,
        source: str,
        received_at_utc: str | None = None,
        record_capture: bool = True,
    ) -> None:
        """Parse one Aurora frame into the shared state cache."""
        packet_timestamp = received_at_utc or utc_now_iso()
        summary: dict | None = None
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
                self._apply_measurements_locked(measurements, payload.header.frame_number, packet_timestamp)
                self._update_tip_pose_locked(packet_timestamp)
                self._state.connection_state = "tracking"
                self._state.health.state = "tracking"
                self._state.health.last_error = None
                self._recompute_health_locked()
        except ValueError as exc:
            parse_error = str(exc)
            with self._lock:
                self._state.bad_packets_count += 1
                if "CRC mismatch" in parse_error:
                    self._state.crc_failures_count += 1
                self._state.health.last_error = parse_error
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
        """Replay a captured Aurora JSONL file through the parser and state cache."""
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

    def _run_loop(self) -> None:
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
                    self.ingest_frame(frame, source="serial")
            except (OSError, RuntimeError, SerialException) as exc:
                if self._stop_event.is_set():
                    break
                with self._lock:
                    self._state.reconnect_count += 1
                    self._state.connection_state = "reconnecting"
                    self._state.health.state = "reconnecting"
                    self._state.health.last_error = str(exc)
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
            self.aurora_client.disconnect()
        except Exception:
            return

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
            self._recompute_health_locked()

    def _apply_measurements_locked(
        self,
        measurements: dict[str, AuroraToolMeasurement],
        frame_number: int,
        packet_timestamp: str,
    ) -> None:
        observed = set(measurements.keys())
        for tool_id in self.SUPPORTED_TOOL_IDS:
            tool_state = self._state.tools.setdefault(tool_id, self._blank_tool(tool_id))
            measurement = measurements.get(tool_id)
            if measurement is None or not measurement.valid:
                tool_state.present = False
                tool_state.valid = False
                tool_state.status = "missing_from_packet" if measurement is None else measurement.status_text
                tool_state.frame_number = frame_number
                tool_state.last_update_utc = packet_timestamp
                tool_state.quaternion_wxyz = None
                tool_state.translation_mm = None
                tool_state.quality = measurement.quality if measurement is not None else None
                tool_state.T_aurora_tool = None
                continue

            T_aurora_tool = make_transform_A_B(measurement.quat_wxyz, measurement.translation_xyz)
            tool_state.present = True
            tool_state.valid = True
            tool_state.status = "tracked"
            tool_state.frame_number = frame_number
            tool_state.last_update_utc = packet_timestamp
            tool_state.quaternion_wxyz = measurement.quat_wxyz
            tool_state.translation_mm = measurement.translation_xyz
            tool_state.quality = measurement.quality
            tool_state.T_aurora_tool = T_aurora_tool.tolist()
            tool_state.last_good_frame_number = frame_number
            tool_state.last_good_update_utc = packet_timestamp
            tool_state.last_good_quaternion_wxyz = measurement.quat_wxyz
            tool_state.last_good_translation_mm = measurement.translation_xyz
            tool_state.last_good_T_aurora_tool = T_aurora_tool.tolist()
            self._state.latest_measurements[tool_id] = {
                "frame_number": frame_number,
                "timestamp_utc": packet_timestamp,
                "quaternion_wxyz": list(measurement.quat_wxyz),
                "translation_mm": list(measurement.translation_xyz),
                "quality": measurement.quality,
                "status": "tracked",
            }

        for tool_id in self.SUPPORTED_TOOL_IDS:
            if tool_id not in observed and tool_id in self._state.latest_measurements:
                self._state.latest_measurements[tool_id]["status"] = self._state.tools[tool_id].status

    def _update_tip_pose_locked(self, packet_timestamp: str) -> None:
        self._state.T_robot_tip = None
        self._state.tip_pose_timestamp_utc = None

        if self._tip_service is None:
            self._state.tip_pose_status = "missing_registration"
            return

        tool_0A = self._state.tools["0A"]
        if not tool_0A.present or not tool_0A.valid or tool_0A.quaternion_wxyz is None or tool_0A.translation_mm is None:
            self._state.tip_pose_status = "missing_0A"
            return

        try:
            T_aurora_coil = make_transform_A_B(tool_0A.quaternion_wxyz, tool_0A.translation_mm)
            T_robot_tip = self._tip_service.compute_T_robot_tip(
                T_robot_aurora=self._tip_service.inputs.T_robot_aurora,
                T_aurora_coil=T_aurora_coil,
                T_coil_tip=self._tip_service.inputs.T_coil_tip,
            )
        except Exception as exc:
            self._state.tip_pose_status = "invalid_transform_chain"
            self._state.health.last_error = str(exc)
            return

        self._state.tip_pose_status = "ok"
        self._state.T_robot_tip = T_robot_tip.tolist()
        self._state.tip_pose_timestamp_utc = packet_timestamp
        self._state.last_good_T_robot_tip = T_robot_tip.tolist()
        self._state.last_good_tip_pose_utc = packet_timestamp

    def _recompute_health_locked(self) -> None:
        faults: list[str] = []
        if self._state.connection_state in {"connecting", "reconnecting"}:
            faults.append("no_serial_connection")
        elif self._state.connection_state == "waiting_for_packets" and self._state.packets_received_count == 0:
            faults.append("no_packets")
        elif self._state.connection_state == "stopped":
            faults.append("service_stopped")

        if self._state.bad_packets_count > 0:
            faults.append("bad_packets")
        if self._state.packets_received_count > 0 and not self._state.tools["0A"].present:
            faults.append("missing_0A")
        if self._state.packets_received_count > 0 and not self._state.tools["0B"].present:
            faults.append("missing_0B")
        if self._state.registration_state == "missing_registration":
            faults.append("missing_registration")
        elif self._state.registration_state == "invalid_registration":
            faults.append("invalid_registration")
        if self._state.tip_pose_status == "invalid_transform_chain":
            faults.append("invalid_transform_chain")

        self._state.faults = faults
        if faults:
            health = HEALTH_DEGRADED
        else:
            health = HEALTH_HEALTHY

        self._state.health.health = health
        self._state.health.current_config_source = self.config_source
        self._state.health.details = {
            "faults": list(faults),
            "packets_received_count": self._state.packets_received_count,
            "bad_packets_count": self._state.bad_packets_count,
            "crc_failures_count": self._state.crc_failures_count,
            "registration_state": self._state.registration_state,
            "tip_pose_status": self._state.tip_pose_status,
        }

        if health == HEALTH_HEALTHY:
            self._state.health.status = "Tracking healthy: packets flowing, tools 0A/0B present, registration loaded"
            return

        if self._state.connection_state == "stopped":
            self._state.health.status = "Tracking service stopped"
            return

        if faults:
            self._state.health.status = "Tracking degraded: " + ", ".join(faults)
            return

        self._state.health.status = "Tracking state unavailable"

    @staticmethod
    def _blank_tool(tool_id: str) -> ToolTrackingSnapshot:
        return ToolTrackingSnapshot(tool_id=tool_id)
