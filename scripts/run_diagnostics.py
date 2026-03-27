"""Tracking diagnostics through the shared tracking service."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from continuum_robot.app.bootstrap import build_app_context


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuum robot tracker diagnostics")
    parser.add_argument("--tracker-port", type=str, default="", help="Aurora device path (for example /dev/ttyUSB0)")
    parser.add_argument(
        "--socket-path",
        type=Path,
        default=None,
        help="Override tracker socket path when using legacy bridge compatibility mode",
    )
    parser.add_argument(
        "--bridge-exec",
        type=Path,
        default=None,
        help="Override tracker bridge executable when using legacy bridge compatibility mode",
    )
    parser.add_argument("--poll-ms", type=int, default=None, help="Override tracker poll period in milliseconds")
    parser.add_argument("--frames", type=int, default=10, help="Number of frame updates to inspect")
    parser.add_argument(
        "--registration-file",
        type=Path,
        default=None,
        help="Optional registration JSON override for computing T_robot_tip",
    )
    return parser.parse_args()


def _render_tool_summary(snapshot, tool_id: str) -> str:
    tool = snapshot.tools[tool_id]
    translation = tuple(round(float(v), 3) for v in tool.translation_mm) if tool.translation_mm is not None else None
    validity = (
        str(tool.valid)
        if tool.validity_known
        else "unknown"
    )
    return (
        f"{tool_id}: state={tool.tracking_state} present={tool.present} valid={validity} "
        f"status={tool.status} frame={tool.frame_number} t_mm={translation}"
    )


def _render_failure_causes(snapshot) -> list[str]:
    causes: list[str] = []
    if snapshot.backend_frame_counter <= 0:
        causes.append("no_live_frames_returned")
    if not snapshot.raw_live_tool_ids:
        causes.append("no_live_tool_returned")
    elif snapshot.unmapped_live_tool_ids:
        causes.append("mapping_mismatch")
    for tool_id in ("0A", "0B"):
        tool = snapshot.tools[tool_id]
        if tool.tracking_state == "invalid":
            causes.append(f"{tool_id}_invalid_transform")
    if snapshot.registration_state == "missing_registration":
        causes.append("registration_missing")
    return causes


def main() -> int:
    args = _parse_args()
    ctx = build_app_context()
    settings = ctx.settings
    tracking_service = ctx.services.get("tracking_service")
    tracker_backend = ctx.services.get("tracker_backend")

    if args.tracker_port:
        tracking_service.set_port(args.tracker_port)
    configured_tracker_port = args.tracker_port or tracking_service.port or settings.serial.aurora_port
    if not settings.runtime.mock_mode and not configured_tracker_port:
        print("ERROR: no Aurora port is configured. Set config/system.local.yaml or pass --tracker-port.")
        return 2

    if args.socket_path is not None and hasattr(tracker_backend, "socket_path"):
        tracker_backend.socket_path = args.socket_path
    if args.bridge_exec is not None and hasattr(tracker_backend, "bridge_executable"):
        tracker_backend.bridge_executable = args.bridge_exec
    if args.poll_ms is not None:
        if hasattr(tracker_backend, "poll_interval_ms"):
            tracker_backend.poll_interval_ms = args.poll_ms
        elif hasattr(tracker_backend, "poll_ms"):
            tracker_backend.poll_ms = args.poll_ms
    if args.registration_file is not None:
        tracking_service.registration_path = args.registration_file
        tracking_service.refresh_registration()

    registration_path = tracking_service.registration_path

    print(f"Mock mode: {settings.runtime.mock_mode}")
    print(f"Tracker backend: {tracking_service.backend_identity}")
    print(f"Tracker port: {configured_tracker_port or '/dev/mock-aurora'}")
    if args.socket_path is not None:
        print(f"Socket override: {args.socket_path}")
    if args.bridge_exec is not None:
        print(f"Bridge override: {args.bridge_exec}")
    print(f"Registration file: {registration_path}")
    print(
        f"Runtime roles: coil={tracking_service.runtime_coil_tool_id} "
        f"registration={tracking_service.registration_tool_id}"
    )

    try:
        tracking_service.start(configured_tracker_port)
    except Exception as exc:
        print(f"ERROR: failed to start tracking service: {exc}")
        return 2

    printed = 0
    last_frame_key = None
    last_connection_state = None
    try:
        while printed < args.frames:
            snapshot = tracking_service.get_snapshot()
            if snapshot.connection_state != last_connection_state:
                print(
                    f"State: {snapshot.connection_state} "
                    f"backend={snapshot.backend_identity} backend_running={snapshot.backend_running} "
                    f"backend_connected={snapshot.backend_connected}"
                )
                if snapshot.backend_status_message:
                    print(f"  backend_status: {snapshot.backend_status_message}")
                if snapshot.last_error:
                    print(f"  error: {snapshot.last_error}")
                last_connection_state = snapshot.connection_state

            frame_key = snapshot.backend_frame_counter if snapshot.backend_frame_counter > 0 else snapshot.last_frame_number
            if frame_key is None or frame_key == last_frame_key:
                time.sleep(0.05)
                continue

            last_frame_key = frame_key
            printed += 1
            print(
                f"Frame #{printed} frame_number={snapshot.last_frame_number} "
                f"backend_frames={snapshot.backend_frame_counter} packets={snapshot.packets_received_count} faults={snapshot.faults} "
                f"stale={snapshot.tracker_data_stale} age_s={snapshot.tracker_data_age_s}"
            )
            print(f"  raw_live_tool_ids={snapshot.raw_live_tool_ids}")
            print(f"  normalized_live_tool_ids={snapshot.normalized_live_tool_ids}")
            print(f"  backend_tool_mappings={snapshot.backend_tool_mappings}")
            print(f"  runtime_role_mappings={snapshot.runtime_role_mappings}")
            print(
                "  required_role_mapping="
                f"{{'0A': {bool(snapshot.runtime_role_mappings.get('0A'))}, "
                f"'0B': {bool(snapshot.runtime_role_mappings.get('0B'))}}}"
            )
            if snapshot.unmapped_live_tool_ids:
                print(f"  unmapped_live_tool_ids={snapshot.unmapped_live_tool_ids}")
            print(f"  {_render_tool_summary(snapshot, '0A')}")
            print(f"  {_render_tool_summary(snapshot, '0B')}")
            print(
                f"  registration={snapshot.registration_state} "
                f"stored_roles=(measurement={snapshot.stored_registration_measurement_tool_id}, "
                f"coil={snapshot.stored_registration_coil_tool_id})"
            )
            print(f"  tip_pose_status={snapshot.tip_pose_status}")
            print(f"  likely_causes={_render_failure_causes(snapshot)}")
            if snapshot.T_robot_tip is None:
                print("  T_robot_tip: unavailable")
            else:
                translation = tuple(round(float(snapshot.T_robot_tip[row][3]), 3) for row in range(3))
                print(f"  T_robot_tip translation: {translation}")
            if snapshot.registration_state == "missing_registration":
                print("  registration_hint=run the registration workflow to create data/registrations/latest_registration.json")

        return 0
    finally:
        tracking_service.stop()
        print("Tracker diagnostics stopped")


if __name__ == "__main__":
    raise SystemExit(main())
