"""Shared runtime models for live tracking backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrackerToolState:
    """Latest known state for one tracked tool."""

    tool_id: str
    frame_number: int | None = None
    valid: bool | None = None
    validity_known: bool = False
    status: str = "unknown"
    quaternion: tuple[float, float, float, float] | None = None
    translation_mm: tuple[float, float, float] | None = None
    quality: float | None = None
    timestamp: str | None = None


@dataclass
class TrackerRuntimeState:
    """Normalized live-backend state used by TrackingService."""

    connection_state: str = "disconnected"
    canonical_state: str = "disconnected"
    backend_identity: str = ""
    configured_backend_name: str = ""
    selected_backend_name: str = ""
    fallback_backend_name: str | None = None
    fallback_used: bool = False
    backend_running: bool = False
    backend_connected: bool = False
    socket_connected: bool = False
    bridge_running: bool = False
    backend_frame_counter: int = 0
    latest_frame_number: int | None = None
    latest_timestamp: str | None = None
    last_status_message: str = ""
    last_error: str | None = None
    raw_tool_ids: list[str] = field(default_factory=list)
    normalized_tool_ids: list[str] = field(default_factory=list)
    tool_id_mapping: dict[str, str] = field(default_factory=dict)
    runtime_role_mappings: dict[str, str] = field(default_factory=dict)
    unmapped_tool_ids: list[str] = field(default_factory=list)
    startup_messages: list[str] = field(default_factory=list)
    capability_report: dict[str, dict[str, Any]] = field(default_factory=dict)
    warning_messages: list[str] = field(default_factory=list)
    error_messages: list[str] = field(default_factory=list)
    backend_details: dict[str, Any] = field(default_factory=dict)
    tools: dict[str, TrackerToolState] = field(default_factory=dict)
