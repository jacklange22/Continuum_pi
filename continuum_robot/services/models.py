"""Shared runtime models for service-oriented subsystem state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


HEALTH_HEALTHY = "healthy"
HEALTH_DEGRADED = "degraded"
HEALTH_FAILED = "failed"


@dataclass
class ServiceHealthSnapshot:
    """Health view common to all runtime services."""

    name: str
    health: str
    state: str
    status: str
    last_error: str | None = None
    last_successful_update_utc: str | None = None
    current_config_source: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolTrackingSnapshot:
    """Latest and last-good state for one Aurora tool."""

    tool_id: str
    present: bool = False
    valid: bool = False
    status: str = "no_sample"
    frame_number: int | None = None
    last_update_utc: str | None = None
    quaternion_wxyz: tuple[float, float, float, float] | None = None
    translation_mm: tuple[float, float, float] | None = None
    quality: float | None = None
    T_aurora_tool: list[list[float]] | None = None
    last_good_frame_number: int | None = None
    last_good_update_utc: str | None = None
    last_good_quaternion_wxyz: tuple[float, float, float, float] | None = None
    last_good_translation_mm: tuple[float, float, float] | None = None
    last_good_T_aurora_tool: list[list[float]] | None = None


@dataclass
class TrackingSnapshot:
    """Tracking service snapshot shared by CLI diagnostics and controllers."""

    health: ServiceHealthSnapshot
    connection_state: str
    port: str
    baudrate: int
    packets_received_count: int = 0
    bad_packets_count: int = 0
    crc_failures_count: int = 0
    reconnect_count: int = 0
    last_frame_number: int | None = None
    last_packet_utc: str | None = None
    packet_capture_path: str | None = None
    packet_capture_enabled: bool = False
    tools: dict[str, ToolTrackingSnapshot] = field(default_factory=dict)
    faults: list[str] = field(default_factory=list)
    registration_state: str = "missing_registration"
    registration_path: str | None = None
    tip_calibration_source: str | None = None
    tip_pose_status: str = "missing_registration"
    T_robot_tip: list[list[float]] | None = None
    tip_pose_timestamp_utc: str | None = None
    last_good_T_robot_tip: list[list[float]] | None = None
    last_good_tip_pose_utc: str | None = None
    latest_measurements: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class RegistrationSnapshot:
    """Registration service snapshot for CLI and GUI workflows."""

    health: ServiceHealthSnapshot
    active: bool = False
    capture_tool_id: str = "0B"
    labels: list[str] = field(default_factory=list)
    captures_per_landmark: int = 0
    current_landmark_index: int = 0
    current_label: str | None = None
    raw_points_by_label: dict[str, list[list[float]]] = field(default_factory=dict)
    averaged_points_by_label: dict[str, list[float]] = field(default_factory=dict)
    captured_counts: dict[str, int] = field(default_factory=dict)
    residuals_by_label: dict[str, list[float]] = field(default_factory=dict)
    fre_mm: float | None = None
    last_sample_xyz_mm: list[float] | None = None
    pending_accept: bool = False
    accepted_output_path: str | None = None
    latest_accepted_path: str | None = None
    config_path: str | None = None
    nominal_landmarks_robot_xyz_mm: dict[str, list[float]] = field(default_factory=dict)
    pending_record: dict[str, Any] | None = None


@dataclass
class SystemHealthSnapshot:
    """Aggregated subsystem health for diagnostics surfaces."""

    generated_at_utc: str
    health: ServiceHealthSnapshot
    tracking: ServiceHealthSnapshot
    registration: ServiceHealthSnapshot
    summary: dict[str, Any] = field(default_factory=dict)
