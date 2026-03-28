"""Quick smoke test for tracker bring-up before registration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from continuum_robot.app.bootstrap import build_app_context
from continuum_robot.tracking.cli_tools import (
    apply_tracking_runtime_overrides,
    build_tracker_thresholds,
    required_tool_ids_from_settings,
)
from continuum_robot.tracking.diagnostics import build_tracking_diagnostics_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick tracker smoke test")
    parser.add_argument("--tracker-port", type=str, default="", help="Aurora device path, for example /dev/ttyUSB0")
    parser.add_argument("--duration-s", type=float, default=1.5, help="Sampling window in seconds")
    parser.add_argument("--sample-period-s", type=float, default=0.02, help="Snapshot sample period in seconds")
    parser.add_argument(
        "--wait-for-first-frame-s",
        type=float,
        default=2.0,
        help="How long to wait for the first live frame before timing the smoke window",
    )
    parser.add_argument("--poll-ms", type=int, default=None, help="Override backend poll period in milliseconds")
    parser.add_argument("--socket-path", type=Path, default=None, help="Override bridge socket path")
    parser.add_argument("--bridge-exec", type=Path, default=None, help="Override bridge executable path")
    parser.add_argument("--registration-file", type=Path, default=None, help="Optional registration JSON override")
    parser.add_argument(
        "--require-registration",
        action="store_true",
        help="Fail unless the full pose pipeline, including T_robot_tip, is ready",
    )
    parser.add_argument("--save-report", type=Path, default=None, help="Optional path for the JSON smoke report")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    ctx = build_app_context()
    settings = ctx.settings
    tracking_service = ctx.services.get("tracking_service")

    configured_tracker_port = apply_tracking_runtime_overrides(
        tracking_service,
        settings,
        tracker_port=args.tracker_port,
        poll_ms=args.poll_ms,
        socket_path=args.socket_path,
        bridge_executable=args.bridge_exec,
        registration_file=args.registration_file,
    )
    if not settings.runtime.mock_mode and not configured_tracker_port:
        print("ERROR: no Aurora port is configured. Set config/system.local.yaml or pass --tracker-port.")
        return 2

    thresholds = build_tracker_thresholds(settings)
    required_tool_ids = required_tool_ids_from_settings(settings)

    try:
        tracking_service.start(configured_tracker_port)
        report = build_tracking_diagnostics_report(
            tracking_service,
            duration_s=args.duration_s,
            sample_period_s=args.sample_period_s,
            wait_for_first_frame_s=args.wait_for_first_frame_s,
            thresholds=thresholds,
            required_tool_ids=required_tool_ids,
        )
    except Exception as exc:
        print(f"ERROR: tracker smoke failed: {exc}")
        return 2
    finally:
        tracking_service.stop()

    print(f"selected_backend={report.selected_backend_name}")
    print(f"backend_identity={report.backend_identity}")
    print(f"configured_port={report.configured_port or '/dev/mock-aurora'}")
    print(f"tracker_ready={report.tracker_ready}")
    print(f"full_pose_pipeline_ready={report.full_pose_pipeline_ready}")
    for stage in report.stage_results:
        print(f"{stage.stage}: {stage.status} | {stage.message}")
    if report.failure_codes:
        print(f"failure_codes={report.failure_codes}")

    if args.save_report is not None:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"saved_report={args.save_report}")

    if args.require_registration:
        return 0 if report.full_pose_pipeline_ready else 1
    return 0 if report.tracker_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
