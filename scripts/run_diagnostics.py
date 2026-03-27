"""Tracker diagnostics using tracker_bridge Unix socket stream."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from continuum_robot.app.bootstrap import build_app_context
from continuum_robot.tracking.tip_pose_service import TipPoseService
from continuum_robot.tracking.tracker_service_manager import TrackerServiceManager
from continuum_robot.tracking.transforms import make_transform_A_B


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="tracker_bridge diagnostics")
    parser.add_argument("--tracker-port", type=str, default="", help="Aurora device path (for example /dev/ttyUSB0)")
    parser.add_argument("--socket-path", type=Path, default=None, help="Unix socket path used by tracker_bridge")
    parser.add_argument("--bridge-exec", type=Path, default=None, help="Path to tracker_bridge executable")
    parser.add_argument("--poll-ms", type=int, default=None, help="tracker_bridge poll period in milliseconds")
    parser.add_argument("--packets", type=int, default=10, help="Number of transform samples to inspect")
    parser.add_argument("--tool-id", type=str, default="0A", help="Tool id to report")
    parser.add_argument(
        "--registration-file",
        type=Path,
        default=None,
        help="Optional registration JSON for computing T_robot_tip",
    )
    return parser.parse_args()


def _resolve_registration_default(project_root: Path) -> Path:
    return project_root / "data" / "registrations" / "latest_registration.json"


def main() -> int:
    args = _parse_args()
    ctx = build_app_context()
    settings = ctx.config_loader.load_settings()

    project_root = Path(__file__).resolve().parents[1]
    bridge_exec = args.bridge_exec or Path(settings.serial.tracker_bridge_executable)
    if not bridge_exec.is_absolute():
        bridge_exec = project_root / bridge_exec

    socket_path = args.socket_path or Path(settings.serial.tracker_socket_path)
    tracker_port = args.tracker_port or settings.serial.aurora_port
    poll_ms = args.poll_ms if args.poll_ms is not None else settings.serial.tracker_poll_ms

    if not tracker_port:
        print("ERROR: tracker port is not set. Pass --tracker-port or set config/system.yaml aurora_port.")
        return 2

    registration_path = args.registration_file or _resolve_registration_default(project_root)
    tip_service = None
    if registration_path.exists():
        try:
            tip_service = TipPoseService.from_registration_file(registration_path)
        except Exception as exc:
            print(f"WARNING: registration unavailable: {exc}")
            print("         Diagnostics will continue without T_robot_tip.")
    else:
        print("WARNING: registration file missing; T_robot_tip is unavailable.")

    manager = TrackerServiceManager(
        bridge_executable=bridge_exec,
        socket_path=socket_path,
        aurora_port=tracker_port,
        poll_ms=poll_ms,
    )

    print(f"Starting tracker_bridge: {bridge_exec}")
    print(f"Aurora port: {tracker_port}")
    print(f"Socket: {socket_path}")

    manager.start()

    seen_frames: set[int] = set()
    printed = 0
    last_state = None

    try:
        while printed < args.packets:
            state = manager.get_state_snapshot()
            if state.connection_state != last_state:
                print(f"State: {state.connection_state}")
                if state.last_error:
                    print(f"  error: {state.last_error}")
                if state.last_status_message:
                    print(f"  status: {state.last_status_message}")
                last_state = state.connection_state

            tool = manager.get_latest_tool(args.tool_id)
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

            if not tool.valid:
                print("  T_robot_tip: unavailable (tool not valid)")
                continue

            if tip_service is None:
                print("  T_robot_tip: unavailable (missing registration)")
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
        manager.stop()
        print("Tracker bridge stopped")


if __name__ == "__main__":
    raise SystemExit(main())
