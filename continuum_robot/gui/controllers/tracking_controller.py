"""Tracking tab controller backed by the shared tracking service."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from continuum_robot.tracking.benchmarking import TrackerBenchmarkThresholds
from continuum_robot.tracking.diagnostics import build_live_stage_results, render_live_tracking_lines
from continuum_robot.config.settings import Settings


@dataclass
class TrackingViewState:
    """UI-facing snapshot of live tracker state."""

    device_path: str
    canonical_state: str = "disconnected"
    configured_backend_name: str = ""
    selected_backend_name: str = ""
    fallback_used: bool = False
    backend_identity: str = ""
    connection_state: str = "disconnected"
    backend_running: bool = False
    backend_connected: bool = False
    latest_frame_number: int | None = None
    latest_timestamp: str | None = None
    tracker_data_age_s: float | None = None
    tracker_data_stale: bool = False
    first_frame_latency_s: float | None = None
    unique_frames_observed: int = 0
    effective_frame_rate_hz: float | None = None
    last_status_message: str = ""
    last_error: str | None = None
    raw_live_tool_ids: list[str] = field(default_factory=list)
    normalized_live_tool_ids: list[str] = field(default_factory=list)
    runtime_role_mappings: dict[str, str] = field(default_factory=dict)
    backend_startup_messages: list[str] = field(default_factory=list)
    backend_capability_report: dict[str, dict] = field(default_factory=dict)
    backend_details: dict[str, object] = field(default_factory=dict)
    tracker_faults: list[str] = field(default_factory=list)
    pipeline_faults: list[str] = field(default_factory=list)
    warning_messages: list[str] = field(default_factory=list)
    error_messages: list[str] = field(default_factory=list)
    tracker_ready: bool = False
    full_pose_pipeline_ready: bool = False
    diagnostic_lines: list[str] = field(default_factory=list)
    tools: dict[str, dict] = field(default_factory=dict)
    registration_path: str = ""
    registration_state: str = "missing_registration"
    runtime_tip_calibration_state: str = "missing_runtime_tip_calibration"
    runtime_tip_mode: str = "latest_accepted"
    runtime_tip_trust_level: str = "missing"
    runtime_tip_mode_message: str = ""
    runtime_tip_identity_fallback: bool = False
    tip_calibration_source: str | None = None
    T_robot_aurora: list[list[float]] | None = None
    T_robot_tip_matrix: list[list[float]] | None = None
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
        self.thresholds = TrackerBenchmarkThresholds(
            min_effective_fps=float(settings.serial.tracker_min_effective_fps),
            max_stale_interval_s=float(settings.serial.tracker_max_stale_interval_s),
            max_consecutive_missing_frames=int(settings.serial.tracker_max_consecutive_missing_frames),
            require_valid_transforms=bool(settings.serial.tracker_require_valid_transforms),
        )
        self.required_tool_ids = (
            settings.registration.coil_tool_id,
            settings.registration.capture_tool_id,
        )
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
        self.state.canonical_state = snapshot.canonical_state
        self.state.configured_backend_name = snapshot.configured_backend_name
        self.state.selected_backend_name = snapshot.selected_backend_name
        self.state.fallback_used = snapshot.fallback_used
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
        self.state.unique_frames_observed = snapshot.unique_frames_observed
        self.state.effective_frame_rate_hz = snapshot.effective_frame_rate_hz
        self.state.last_status_message = snapshot.backend_status_message or snapshot.health.status
        self.state.last_error = snapshot.last_error
        self.state.raw_live_tool_ids = list(snapshot.raw_live_tool_ids)
        self.state.normalized_live_tool_ids = list(snapshot.normalized_live_tool_ids)
        self.state.runtime_role_mappings = dict(snapshot.runtime_role_mappings)
        self.state.backend_startup_messages = list(snapshot.backend_startup_messages)
        self.state.backend_capability_report = dict(snapshot.backend_capability_report)
        self.state.backend_details = dict(snapshot.backend_details)
        self.state.tracker_faults = list(snapshot.tracker_faults)
        self.state.pipeline_faults = list(snapshot.pipeline_faults)
        self.state.warning_messages = list(snapshot.warning_messages)
        self.state.error_messages = list(snapshot.error_messages)
        self.state.runtime_coil_tool_id = snapshot.runtime_coil_tool_id
        self.state.registration_tool_id = snapshot.registration_tool_id
        self.state.registration_state = snapshot.registration_state
        self.state.runtime_tip_calibration_state = snapshot.runtime_tip_calibration_state
        self.state.runtime_tip_mode = snapshot.runtime_tip_mode
        self.state.runtime_tip_trust_level = snapshot.runtime_tip_trust_level
        self.state.runtime_tip_mode_message = snapshot.runtime_tip_mode_message
        self.state.runtime_tip_identity_fallback = bool(snapshot.runtime_tip_identity_fallback)
        self.state.tip_calibration_source = snapshot.tip_calibration_source
        self.state.T_robot_aurora = snapshot.T_robot_aurora
        self.state.T_robot_tip_matrix = snapshot.T_robot_tip
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
        self.state.diagnostic_lines = render_live_tracking_lines(
            snapshot,
            thresholds=self.thresholds,
            required_tool_ids=self.required_tool_ids,
        )
        stages = build_live_stage_results(
            snapshot,
            thresholds=self.thresholds,
            required_tool_ids=self.required_tool_ids,
        )
        self.state.tracker_ready = all(stage.status == "passed" for stage in stages[:4])
        self.state.full_pose_pipeline_ready = self.state.tracker_ready and stages[4].status == "passed"
        self._refresh_tip_pose(snapshot)
        return self.state

    def shutdown(self) -> None:
        self.tracking_service.stop()

    def _refresh_tip_pose(self, snapshot) -> None:
        self.state.tip_position_mm = None
        self.state.tip_direction_xyz = None
        self.state.tip_status = self._render_tip_status(snapshot)

        if snapshot.T_robot_tip is None:
            return

        self.state.tip_position_mm = tuple(float(v) for v in (snapshot.T_robot_tip[0][3], snapshot.T_robot_tip[1][3], snapshot.T_robot_tip[2][3]))
        self.state.tip_direction_xyz = tuple(
            float(v) for v in (snapshot.T_robot_tip[0][2], snapshot.T_robot_tip[1][2], snapshot.T_robot_tip[2][2])
        )

    @staticmethod
    def _render_tip_status(snapshot) -> str:
        if snapshot.tip_pose_status == "ok" and snapshot.tracker_data_stale:
            if snapshot.tracker_data_age_s is not None:
                return f"stale_tracker_data ({snapshot.tracker_data_age_s:.3f} s)"
            return "stale_tracker_data"
        return snapshot.tip_pose_status
