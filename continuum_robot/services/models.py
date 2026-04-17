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
    valid: bool | None = None
    validity_known: bool = False
    tracking_state: str = "unknown"
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
    canonical_state: str = "disconnected"
    backend_identity: str = ""
    configured_backend_name: str = ""
    selected_backend_name: str = ""
    port: str = ""
    baudrate: int = 115200
    runtime_coil_tool_id: str = "0A"
    registration_tool_id: str = "0B"
    fallback_backend_name: str | None = None
    fallback_used: bool = False
    backend_running: bool | None = None
    backend_connected: bool | None = None
    bridge_running: bool | None = None
    socket_connected: bool | None = None
    backend_frame_counter: int = 0
    backend_status_message: str | None = None
    last_error: str | None = None
    packets_received_count: int = 0
    bad_packets_count: int = 0
    crc_failures_count: int = 0
    reconnect_count: int = 0
    last_frame_number: int | None = None
    last_packet_utc: str | None = None
    tracker_data_age_s: float | None = None
    tracker_data_stale: bool = False
    first_frame_latency_s: float | None = None
    raw_live_tool_ids: list[str] = field(default_factory=list)
    normalized_live_tool_ids: list[str] = field(default_factory=list)
    backend_tool_mappings: dict[str, str] = field(default_factory=dict)
    runtime_role_mappings: dict[str, str] = field(default_factory=dict)
    unmapped_live_tool_ids: list[str] = field(default_factory=list)
    packet_capture_path: str | None = None
    packet_capture_enabled: bool = False
    backend_startup_messages: list[str] = field(default_factory=list)
    backend_capability_report: dict[str, Any] = field(default_factory=dict)
    backend_details: dict[str, Any] = field(default_factory=dict)
    unique_frames_observed: int = 0
    effective_frame_rate_hz: float | None = None
    frame_interval_min_s: float | None = None
    frame_interval_max_s: float | None = None
    frame_interval_mean_s: float | None = None
    max_observed_data_age_s: float | None = None
    tools: dict[str, ToolTrackingSnapshot] = field(default_factory=dict)
    faults: list[str] = field(default_factory=list)
    tracker_faults: list[str] = field(default_factory=list)
    pipeline_faults: list[str] = field(default_factory=list)
    warning_messages: list[str] = field(default_factory=list)
    error_messages: list[str] = field(default_factory=list)
    registration_state: str = "missing_registration"
    registration_path: str | None = None
    stored_registration_measurement_tool_id: str | None = None
    stored_registration_coil_tool_id: str | None = None
    stored_registration_timestamp_utc: str | None = None
    stored_registration_fre_mm: float | None = None
    T_robot_aurora: list[list[float]] | None = None
    runtime_tip_calibration_state: str = "missing_runtime_tip_calibration"
    runtime_tip_calibration_path: str | None = None
    runtime_tip_mode: str = "latest_accepted"
    runtime_tip_trust_level: str = "missing"
    runtime_tip_mode_message: str = "No runtime tip mode selected."
    runtime_tip_selected_artifact_kind: str | None = None
    runtime_tip_selected_artifact_path: str | None = None
    stored_runtime_tip_measurement_tool_id: str | None = None
    stored_runtime_tip_coil_tool_id: str | None = None
    stored_runtime_tip_timestamp_utc: str | None = None
    runtime_tip_identity_fallback: bool = False
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
    T_robot_aurora: list[list[float]] | None = None
    last_sample_xyz_mm: list[float] | None = None
    pending_accept: bool = False
    accepted_output_path: str | None = None
    latest_accepted_path: str | None = None
    config_path: str | None = None
    nominal_landmarks_robot_xyz_mm: dict[str, list[float]] = field(default_factory=dict)
    truth_points_in_sw_by_label: dict[str, list[float]] = field(default_factory=dict)
    group_by_label: dict[str, str] = field(default_factory=dict)
    raw_measurement_tool_samples_by_label: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    raw_coil_samples_by_label: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    latest_validation_summary: dict[str, Any] = field(default_factory=dict)
    pending_record: dict[str, Any] | None = None


@dataclass
class RuntimeTipCalibrationSnapshot:
    """Runtime 0A hat-based tip calibration snapshot for the advanced workflow."""

    health: ServiceHealthSnapshot
    active: bool = False
    measurement_tool_id: str = "0B"
    coil_tool_id: str = "0A"
    session_mode: str = "full_hat"
    session_mode_summary: str = "Full accepted hat calibration"
    selected_labels: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    captures_per_landmark: int = 0
    current_label_index: int = 0
    current_label: str | None = None
    truth_points_in_tip_by_label: dict[str, list[float]] = field(default_factory=dict)
    raw_points_by_label: dict[str, list[list[float]]] = field(default_factory=dict)
    captured_counts: dict[str, int] = field(default_factory=dict)
    averaged_points_by_label: dict[str, list[float]] = field(default_factory=dict)
    fit_residuals_by_label: dict[str, list[float]] = field(default_factory=dict)
    fit_rmse_mm: float | None = None
    max_residual_mm: float | None = None
    translation_spread_summary_mm: dict[str, Any] = field(default_factory=dict)
    rotation_spread_summary_deg: dict[str, Any] = field(default_factory=dict)
    coil_sample_count: int = 0
    coil_samples_captured: int = 0
    averaged_coil_translation_mm: list[float] | None = None
    averaged_coil_quaternion_wxyz: list[float] | None = None
    T_tip_aurora: list[list[float]] | None = None
    T_aurora_tip: list[list[float]] | None = None
    T_aurora_coil_avg: list[list[float]] | None = None
    T_coil_tip: list[list[float]] | None = None
    pending_accept: bool = False
    accepted_output_path: str | None = None
    latest_accepted_path: str | None = None
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    pending_record: dict[str, Any] | None = None
    hat_geometry_path: str | None = None
    hat_geometry_status: str = "unknown"
    measurement_point_status: str = "unknown"
    runtime_chain_state: str = "missing_runtime_tip_calibration"
    runtime_chain_message: str = "Runtime tip calibration is not loaded."
    status_message: str = "Runtime tip calibration idle."


@dataclass
class SystemHealthSnapshot:
    """Aggregated subsystem health for diagnostics surfaces."""

    generated_at_utc: str
    health: ServiceHealthSnapshot
    tracking: ServiceHealthSnapshot
    registration: ServiceHealthSnapshot
    summary: dict[str, Any] = field(default_factory=dict)
