"""Tracker diagnostics using the configured tracker manager."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from continuum_robot.app.bootstrap import build_app_context
from continuum_robot.tracking.tip_pose_service import TipPoseService
from continuum_robot.tracking.transforms import make_transform_A_B


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuum robot tracker diagnostics")
    parser.add_argument("--tracker-port", type=str, default="", help="Aurora device path (for example /dev/ttyUSB0)")
    parser.add_argument("--socket-path", type=Path, default=None, help="Override tracker socket path")
    parser.add_argument("--bridge-exec", type=Path, default=None, help="Override tracker bridge executable")
    parser.add_argument("--poll-ms", type=int, default=None, help="Override tracker poll period in milliseconds")
    parser.add_argument("--packets", type=int, default=10, help="Number of tool samples to inspect")
    parser.add_argument("--tool-id", type=str, default="0A", help="Tool id to report")
    parser.add_argument(
        "--registration-file",
        type=Path,
        default=None,
        help="Optional registration JSON for computing T_robot_tip",
    )
    return parser.parse_args()


def _resolve_registration_path(project_root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return project_root / path


def main() -> int:
    args = _parse_args()
    ctx = build_app_context()
    settings = ctx.settings
    project_root = ctx.project_root
    tracker_manager = ctx.services.get("tracker_manager")

    if args.tracker_port and hasattr(tracker_manager, "aurora_port"):
        tracker_manager.aurora_port = args.tracker_port
    if args.socket_path is not None and hasattr(tracker_manager, "socket_path"):
        tracker_manager.socket_path = args.socket_path
    if args.bridge_exec is not None and hasattr(tracker_manager, "bridge_executable"):
        tracker_manager.bridge_executable = args.bridge_exec
    if args.poll_ms is not None and hasattr(tracker_manager, "poll_ms"):
        tracker_manager.poll_ms = args.poll_ms

    configured_tracker_port = args.tracker_port or getattr(tracker_manager, "aurora_port", "") or settings.serial.aurora_port
    if not settings.runtime.mock_mode and not configured_tracker_port:
        print("ERROR: no Aurora port is configured. Set config/system.local.yaml or pass --tracker-port.")
        return 2
    tracker_port = configured_tracker_port or "/dev/mock-aurora"
    if hasattr(tracker_manager, "aurora_port"):
        tracker_manager.aurora_port = tracker_port
    registration_path = args.registration_file or _resolve_registration_path(
        project_root,
        settings.calibration.latest_registration_path,
    )

    tip_service = None
    if registration_path.exists():
        try:
            tip_service = TipPoseService.from_registration_file(registration_path)
        except Exception as exc:
            print(f"WARNING: registration unavailable: {exc}")
    else:
        print("WARNING: registration file missing; T_robot_tip is unavailable.")

    print(f"Mock mode: {settings.runtime.mock_mode}")
    print(f"Tracker port: {tracker_port}")
    print(f"Tracker backend: {tracker_manager.__class__.__name__}")
    if args.socket_path is not None:
        print(f"Socket override: {args.socket_path}")
    if args.bridge_exec is not None:
        print(f"Bridge override: {args.bridge_exec}")
    print(f"Registration file: {registration_path}")

    try:
        tracker_manager.start()
    except Exception as exc:
        print(f"ERROR: failed to start tracker backend: {exc}")
        return 2
    printed = 0
    seen_frames: set[int] = set()
    last_state = None

    try:
        while printed < args.packets:
            state = tracker_manager.get_state_snapshot()
            if state.connection_state != last_state:
                print(f"State: {state.connection_state}")
                if state.last_status_message:
                    print(f"  status: {state.last_status_message}")
                if state.last_error:
                    print(f"  error: {state.last_error}")
                last_state = state.connection_state

            tool = tracker_manager.get_latest_tool(args.tool_id)
            if tool is None or tool.frame_number in seen_frames:
                time.sleep(0.05)
                continue

            seen_frames.add(tool.frame_number)
            printed += 1
            print(
                f"Frame #{printed} frame_number={tool.frame_number} "
                f"tool={tool.tool_id} valid={tool.valid} status={tool.status} "
                f"t_mm={tuple(round(v, 3) for v in tool.translation_mm)} quality={tool.quality}"
            )

            if not tool.valid or tip_service is None:
                print("  T_robot_tip: unavailable")
                continue

            try:
                T_aurora_coil = make_transform_A_B(tool.quaternion, tool.translation_mm)
                T_robot_tip = tip_service.compute_T_robot_tip(
                    T_robot_aurora=tip_service.inputs.T_robot_aurora,
                    T_aurora_coil=T_aurora_coil,
                    T_coil_tip=tip_service.inputs.T_coil_tip,
                )
                print(f"  T_robot_tip translation: {tuple(round(float(v), 3) for v in T_robot_tip[0:3, 3])}")
            except Exception as exc:
                print(f"  T_robot_tip invalid: {exc}")

        return 0
    finally:
        tracker_manager.stop()
        print("Tracker diagnostics stopped")


if __name__ == "__main__":
    raise SystemExit(main())
