"""Tracker diagnostics mode for Aurora parsing and tip-pose computation."""

from __future__ import annotations

import argparse
from pathlib import Path

from continuum_robot.app.bootstrap import build_app_context
from continuum_robot.hardware.aurora_client import AuroraClient
from continuum_robot.tracking.aurora_framer import AuroraFramer
from continuum_robot.tracking.aurora_parser import AuroraParser
from continuum_robot.tracking.tip_pose_service import TipPoseService
from continuum_robot.tracking.transforms import make_transform_A_B


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aurora tracker diagnostics")
    parser.add_argument("--tracker-port", type=str, default="", help="Aurora serial port (for example /dev/ttyUSB0)")
    parser.add_argument("--baudrate", type=int, default=115200, help="Aurora serial baudrate")
    parser.add_argument("--timeout", type=float, default=1.0, help="Frame read timeout in seconds")
    parser.add_argument("--packets", type=int, default=10, help="Number of packets to inspect")
    parser.add_argument(
        "--registration-file",
        type=Path,
        default=Path("/Users/jacklange/Continuum/pi_code/data/registrations/latest_registration.json"),
        help="Path to registration/calibration JSON containing T_robot_aurora and T_coil_tip",
    )
    return parser.parse_args()


def _print_tool_status(tool_id: str, measurements: dict) -> None:
    m = measurements.get(tool_id)
    if m is None:
        print(f"  {tool_id}: missing_from_packet")
        return

    print(
        f"  {tool_id}: valid={m.valid} status={m.status_text} "
        f"t={tuple(round(v, 3) for v in m.translation_xyz)} quality={m.quality:.5f}"
    )


def main() -> int:
    args = _parse_args()

    ctx = build_app_context()
    settings = ctx.config_loader.load_settings()
    port = args.tracker_port or settings.serial.aurora_port
    if not port:
        print("ERROR: Aurora port is not set. Pass --tracker-port or set config/system.yaml aurora_port.")
        return 2

    tip_service = None
    config_error = False
    try:
        tip_service = TipPoseService.from_registration_file(args.registration_file)
    except Exception as exc:
        config_error = True
        print(f"WARNING: Could not load registration file: {exc}")
        print("         Diagnostics will print T_aurora_coil only; T_robot_tip is unavailable.")

    client = AuroraClient()
    framer = AuroraFramer()
    parser = AuroraParser()

    try:
        client.connect(port=port, baudrate=args.baudrate, timeout_s=min(args.timeout, 0.1))
        print(f"Connected to Aurora on {port} @ {args.baudrate}")
        print("Inspecting packets...")

        processed = 0
        while processed < args.packets:
            try:
                frame = framer.read_next_frame(client.read_bytes, timeout_s=args.timeout)
                payload = parser.parse_payload(frame)
                measurements = parser.parse_transform_packet(frame)
            except TimeoutError as exc:
                print(f"Timeout: {exc}")
                continue
            except Exception as exc:
                print(f"Invalid packet: {exc}")
                continue

            processed += 1
            print(
                f"Packet #{processed} frame={payload.header.frame_number} "
                f"tools={payload.header.tool_count} crc=0x{payload.crc_received:02X}"
            )
            _print_tool_status("0A", measurements)
            _print_tool_status("0B", measurements)

            tool_0A = measurements.get("0A")
            if tool_0A is None:
                print("  T_aurora_coil: unavailable (0A missing)")
                print("  T_robot_tip: unavailable (0A missing)")
                continue

            try:
                T_aurora_coil = make_transform_A_B(tool_0A.quat_wxyz, tool_0A.translation_xyz)
                print(f"  T_aurora_coil translation: {tuple(round(float(v), 3) for v in T_aurora_coil[0:3, 3])}")
            except Exception as exc:
                print(f"  T_aurora_coil invalid: {exc}")
                print("  T_robot_tip: unavailable")
                continue

            if tip_service is None:
                print("  T_robot_tip: unavailable (missing registration/calibration file)")
                continue

            try:
                T_robot_tip = tip_service.compute_T_robot_tip_from_0A(tool_0A)
                print(f"  T_robot_tip translation: {tuple(round(float(v), 3) for v in T_robot_tip[0:3, 3])}")
            except Exception as exc:
                print(f"  T_robot_tip invalid: {exc}")

        return 2 if config_error else 0
    finally:
        client.disconnect()
        print("Disconnected Aurora")


if __name__ == "__main__":
    raise SystemExit(main())
