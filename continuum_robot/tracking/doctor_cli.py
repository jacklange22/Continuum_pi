"""CLI implementation for the canonical tracker doctor command."""

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
from continuum_robot.tracking.diagnostics import (
    build_tracking_diagnostics_report,
    render_tracking_diagnostics_report_lines,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the tracker doctor CLI parser."""
    parser = argparse.ArgumentParser(description="Detailed diagnostics for the canonical tracking path")
    parser.add_argument("--tracker-port", type=str, default="", help="Aurora device path, for example /dev/ttyUSB0")
    parser.add_argument("--duration-s", type=float, default=3.0, help="Sampling window in seconds")
    parser.add_argument("--sample-period-s", type=float, default=0.02, help="Snapshot sample period in seconds")
    parser.add_argument(
        "--wait-for-first-frame-s",
        type=float,
        default=2.0,
        help="How long to wait for the first live frame before timing the doctor window",
    )
    parser.add_argument("--poll-ms", type=int, default=None, help="Override backend poll period in milliseconds")
    parser.add_argument(
        "--socket-path",
        type=Path,
        default=None,
        help="Override tracker socket path when bridge fallback/debug mode is used",
    )
    parser.add_argument(
        "--bridge-exec",
        type=Path,
        default=None,
        help="Override tracker bridge executable when bridge fallback/debug mode is used",
    )
    parser.add_argument(
        "--registration-file",
        type=Path,
        default=None,
        help="Optional registration JSON override for tip-pose readiness checks",
    )
    parser.add_argument("--min-effective-fps", type=float, default=None, help="Override minimum acceptable FPS")
    parser.add_argument("--max-stale-interval-s", type=float, default=None, help="Override max acceptable data age")
    parser.add_argument(
        "--max-consecutive-missing-frames",
        type=int,
        default=None,
        help="Override missing-frame streak threshold",
    )
    parser.add_argument(
        "--allow-invalid-transforms",
        action="store_true",
        help="Do not require every tracked transform to be rigid-valid",
    )
    parser.add_argument(
        "--save-report",
        type=Path,
        default=None,
        help="Optional path for the JSON doctor report",
    )
    parser.add_argument(
        "--debug-transforms",
        action="store_true",
        help="Print per-tool backend/conversion/validation details for the latest frame",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the tracker doctor CLI."""
    args = build_arg_parser().parse_args(argv)
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

    thresholds = build_tracker_thresholds(
        settings,
        min_effective_fps=args.min_effective_fps,
        max_stale_interval_s=args.max_stale_interval_s,
        max_consecutive_missing_frames=args.max_consecutive_missing_frames,
        require_valid_transforms=not args.allow_invalid_transforms,
    )
    required_tool_ids = required_tool_ids_from_settings(settings)

    print(f"mock_mode={settings.runtime.mock_mode}")
    print(f"configured_backend={settings.serial.tracker_backend}")
    print(f"configured_fallback_backend={settings.serial.tracker_fallback_backend}")
    print(f"fallback_enabled={settings.serial.tracker_fallback_enabled}")
    print(f"configured_port={configured_tracker_port or '/dev/mock-aurora'}")
    print(f"required_tool_ids={required_tool_ids}")
    print(
        "thresholds="
        f"min_fps={thresholds.min_effective_fps:.2f}, "
        f"max_stale_s={thresholds.max_stale_interval_s:.3f}, "
        f"max_missing={thresholds.max_consecutive_missing_frames}, "
        f"require_valid_transforms={thresholds.require_valid_transforms}"
    )

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
        print(f"ERROR: tracker doctor failed: {exc}")
        return 2
    finally:
        tracking_service.stop()

    for line in render_tracking_diagnostics_report_lines(report):
        print(line)
    if args.debug_transforms:
        transform_debug = dict(report.backend_details or {}).get("ndi_transform_debug", {})
        print("transform_debug=")
        print(json.dumps(transform_debug, indent=2, sort_keys=True))

    if args.save_report is not None:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"saved_report={args.save_report}")

    return 0 if report.tracker_ready else 1
