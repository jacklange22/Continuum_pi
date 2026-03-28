"""Shared CLI helpers for the canonical tracking validation commands."""

from __future__ import annotations

from pathlib import Path

from continuum_robot.config.settings import Settings
from continuum_robot.tracking.benchmarking import TrackerBenchmarkThresholds


def apply_tracking_runtime_overrides(
    tracking_service,
    settings: Settings,
    *,
    tracker_port: str = "",
    poll_ms: int | None = None,
    socket_path: Path | None = None,
    bridge_executable: Path | None = None,
    registration_file: Path | None = None,
) -> str:
    """Apply CLI overrides through TrackingService and return the effective port."""
    normalized_port = tracker_port.strip() or None
    tracking_service.configure_live_backend(
        tracker_port=normalized_port,
        poll_ms=poll_ms,
        socket_path=socket_path,
        bridge_executable=bridge_executable,
    )
    if registration_file is not None:
        tracking_service.registration_path = registration_file
        tracking_service.refresh_registration()
    return normalized_port or tracking_service.port or settings.serial.aurora_port


def build_tracker_thresholds(
    settings: Settings,
    *,
    min_effective_fps: float | None = None,
    max_stale_interval_s: float | None = None,
    max_consecutive_missing_frames: int | None = None,
    require_valid_transforms: bool | None = None,
) -> TrackerBenchmarkThresholds:
    """Build threshold settings from config with optional CLI overrides."""
    return TrackerBenchmarkThresholds(
        min_effective_fps=(
            float(min_effective_fps)
            if min_effective_fps is not None
            else float(settings.serial.tracker_min_effective_fps)
        ),
        max_stale_interval_s=(
            float(max_stale_interval_s)
            if max_stale_interval_s is not None
            else float(settings.serial.tracker_max_stale_interval_s)
        ),
        max_consecutive_missing_frames=(
            int(max_consecutive_missing_frames)
            if max_consecutive_missing_frames is not None
            else int(settings.serial.tracker_max_consecutive_missing_frames)
        ),
        require_valid_transforms=(
            bool(require_valid_transforms)
            if require_valid_transforms is not None
            else bool(settings.serial.tracker_require_valid_transforms)
        ),
    )


def required_tool_ids_from_settings(settings: Settings) -> tuple[str, str]:
    """Return the canonical runtime tool roles used by the app."""
    return (
        settings.registration.coil_tool_id,
        settings.registration.capture_tool_id,
    )
