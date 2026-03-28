"""Legacy bridge-only socket reader for tracker_bridge debugging.

This bypasses the canonical TrackingService path and should only be used when
debugging the fallback bridge backend itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from continuum_robot.tracking.tracker_protocol import parse_tracker_json_line
from continuum_robot.tracking.tracker_socket_client import TrackerSocketClient


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print tracker_bridge socket stream")
    parser.add_argument("--socket-path", type=Path, default=Path("/tmp/tracker_bridge.sock"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    client = TrackerSocketClient(args.socket_path)
    client.connect(timeout_s=10.0)
    print(f"Connected to {args.socket_path}")

    try:
        while True:
            line = client.read_line(timeout_s=1.0)
            if line is None:
                continue
            msg = parse_tracker_json_line(line)
            print(msg)
    except KeyboardInterrupt:
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
