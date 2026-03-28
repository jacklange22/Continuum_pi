"""Sanity-check runtime tip-pose computation from saved registration plus live or replayed 0A data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from continuum_robot.app.bootstrap import build_app_context
from continuum_robot.registration.validation_tools import (
    evaluate_runtime_sanity_from_capture,
    evaluate_runtime_sanity_live,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check runtime T_robot_tip from saved registration plus 0A data")
    parser.add_argument("--registration-file", type=Path, default=None, help="Registration JSON to use")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--capture-jsonl", type=Path, help="Replay packet capture JSONL")
    source.add_argument("--live", action="store_true", help="Use live TrackingService data")
    parser.add_argument("--expected-runtime-coil-tool-id", type=str, default="0A", help="Expected runtime coil tool id")
    parser.add_argument("--tracker-port", type=str, default="", help="Aurora device path for live mode")
    parser.add_argument("--poll-ms", type=int, default=None, help="Override live tracker poll period in milliseconds")
    parser.add_argument("--frames", type=int, default=3, help="Number of live frames to observe before computing tip pose")
    parser.add_argument("--timeout-s", type=float, default=5.0, help="Timeout for live mode")
    parser.add_argument("--save-report", type=Path, default=None, help="Optional path for the sanity JSON report")
    return parser.parse_args()


def _resolve_registration_path(project_root: Path, configured_path: str, override: Path | None) -> Path:
    if override is not None:
        return override
    path = Path(configured_path)
    return path if path.is_absolute() else project_root / path


def main() -> int:
    args = _parse_args()
    ctx = build_app_context()
    registration_path = _resolve_registration_path(
        ctx.project_root,
        ctx.settings.calibration.latest_registration_path,
        args.registration_file,
    )

    if args.capture_jsonl is not None:
        report = evaluate_runtime_sanity_from_capture(
            registration_path=registration_path,
            capture_path=args.capture_jsonl,
            expected_runtime_coil_tool_id=args.expected_runtime_coil_tool_id,
        )
    else:
        tracking_service = ctx.services.get("tracking_service")
        tracking_service.configure_live_backend(
            tracker_port=args.tracker_port or None,
            poll_ms=args.poll_ms,
        )
        report = evaluate_runtime_sanity_live(
            tracking_service=tracking_service,
            registration_path=registration_path,
            expected_runtime_coil_tool_id=args.expected_runtime_coil_tool_id,
            frames=args.frames,
            timeout_s=args.timeout_s,
        )

    print(f"source_kind={report.source_kind}")
    print(f"registration_path={report.registration_path}")
    print(f"expected_runtime_coil_tool_id={report.expected_runtime_coil_tool_id}")
    print(f"stored_coil_tool_id={report.stored_coil_tool_id}")
    print(f"registration_state={report.registration_state}")
    print(f"tip_pose_status={report.tip_pose_status}")
    print(f"connection_state={report.connection_state}")
    print(f"packets_received_count={report.packets_received_count}")
    print(f"last_frame_number={report.last_frame_number}")
    print(f"tracking_faults={report.tracking_faults}")
    print(f"tip_translation_mm={report.tip_translation_mm}")
    print(f"status={report.status}")
    if report.last_error:
        print(f"last_error={report.last_error}")
    print(f"passed={report.passed}")

    if args.save_report is not None:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(json.dumps(report, default=lambda obj: obj.__dict__, indent=2), encoding="utf-8")
        print(f"saved_report={args.save_report}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
