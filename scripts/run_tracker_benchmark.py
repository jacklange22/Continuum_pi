"""Benchmark the live tracker path through TrackingService."""

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
from continuum_robot.tracking.benchmarking import (
    collect_tracking_snapshots,
    compute_tracker_benchmark_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark live Aurora tracking performance")
    parser.add_argument("--tracker-port", type=str, default="", help="Aurora device path (for example /dev/ttyUSB0)")
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
        "--duration-s",
        type=float,
        default=5.0,
        help="Benchmark window in seconds",
    )
    parser.add_argument(
        "--sample-period-s",
        type=float,
        default=0.02,
        help="Tracking snapshot sample period in seconds",
    )
    parser.add_argument(
        "--registration-file",
        type=Path,
        default=None,
        help="Optional registration JSON override for tip-pose availability checks",
    )
    parser.add_argument("--poll-ms", type=int, default=None, help="Override backend poll period in milliseconds")
    parser.add_argument(
        "--min-effective-fps",
        type=float,
        default=None,
        help="Fail if measured effective frame rate is lower than this",
    )
    parser.add_argument(
        "--max-stale-interval-s",
        type=float,
        default=None,
        help="Fail if tracker data age exceeds this threshold",
    )
    parser.add_argument(
        "--max-consecutive-missing-frames",
        type=int,
        default=None,
        help="Fail if 0A or 0B exceeds this missing-frame streak",
    )
    parser.add_argument(
        "--allow-invalid-transforms",
        action="store_true",
        help="Do not fail on invalid transform frames",
    )
    parser.add_argument(
        "--save-report",
        type=Path,
        default=None,
        help="Optional path for the JSON benchmark report",
    )
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

    thresholds = build_tracker_thresholds(
        settings,
        min_effective_fps=args.min_effective_fps,
        max_stale_interval_s=args.max_stale_interval_s,
        max_consecutive_missing_frames=args.max_consecutive_missing_frames,
        require_valid_transforms=not args.allow_invalid_transforms,
    )
    required_tool_ids = required_tool_ids_from_settings(settings)

    print(f"Tracker backend: {settings.serial.tracker_backend}")
    print(f"Configured legacy fallback backend: {settings.serial.tracker_fallback_backend}")
    print(f"Legacy fallback enabled: {settings.serial.tracker_fallback_enabled}")
    print(f"Tracker port: {configured_tracker_port or '/dev/mock-aurora'}")
    print(f"Benchmark duration: {args.duration_s:.2f}s")
    print(f"Startup wait: {max(2.0, float(args.duration_s)):.2f}s")
    print(
        "Thresholds: "
        f"min_fps={thresholds.min_effective_fps:.2f}, "
        f"max_stale_s={thresholds.max_stale_interval_s:.3f}, "
        f"max_missing={thresholds.max_consecutive_missing_frames}, "
        f"require_valid_transforms={thresholds.require_valid_transforms}"
    )

    try:
        tracking_service.start(configured_tracker_port)
        samples = collect_tracking_snapshots(
            tracking_service,
            duration_s=args.duration_s,
            sample_period_s=args.sample_period_s,
            wait_for_first_frame_s=max(2.0, float(args.duration_s)),
        )
        report = compute_tracker_benchmark_report(
            samples,
            thresholds=thresholds,
            required_tool_ids=required_tool_ids,
        )
    except Exception as exc:
        print(f"ERROR: benchmark failed: {exc}")
        return 2
    finally:
        tracking_service.stop()

    print(f"Connection state: {report.final_connection_state}")
    print(f"Configured backend: {report.configured_backend_name}")
    print(f"Selected backend: {report.selected_backend_name}")
    print(f"Canonical state: {report.canonical_state_final}")
    print(f"Unique frames observed: {report.unique_frames_observed}")
    print(f"Backend frame counter: {report.backend_frame_counter_final}")
    print(f"Effective FPS: {report.effective_frame_rate_hz}")
    print(f"Frame interval stats: {report.frame_interval_s}")
    print(f"Max data age (s): {report.max_data_age_s}")
    print(f"First frame latency (s): {report.first_frame_latency_s}")
    print(f"Raw live tool ids: {report.raw_live_tool_ids_final}")
    print(f"Normalized live tool ids: {report.normalized_live_tool_ids_final}")
    print(f"Runtime role mappings: {report.runtime_role_mappings_final}")
    print(f"Warnings: {report.warning_messages_final}")
    print(f"Errors: {report.error_messages_final}")
    if report.unmapped_live_tool_ids_final:
        print(f"Unmapped live tool ids: {report.unmapped_live_tool_ids_final}")
    print(f"Registration loaded: {report.registration_loaded}")
    print(f"Registration state: {report.registration_state_final}")
    print(f"Tip pose status: {report.tip_pose_status_final}")
    print(f"T_robot_tip computable: {report.tip_pose_computable}")
    for tool_id in sorted(report.tool_metrics):
        metrics = report.tool_metrics[tool_id]
        print(
            f"{tool_id}: tracked={metrics.tracked_frames} missing={metrics.missing_frames} "
            f"invalid={metrics.invalid_frames} max_missing={metrics.max_consecutive_missing_frames} "
            f"ttfv={metrics.time_to_first_tracked_frame_s} quality={metrics.quality}"
        )
    if report.failures:
        print("Failures:")
        for failure in report.failures:
            print(f"  - {failure}")

    if args.save_report is not None:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"Saved report: {args.save_report}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
