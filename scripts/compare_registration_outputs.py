"""Compare legacy and new registration outputs numerically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from continuum_robot.registration.validation_tools import compare_registration_outputs, load_registration_output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare registration outputs")
    parser.add_argument("left", type=Path, help="Legacy transform dir or registration/validation JSON")
    parser.add_argument("right", type=Path, help="Registration/validation JSON or legacy transform dir")
    parser.add_argument("--translation-tol-mm", type=float, default=0.25, help="Translation tolerance in mm")
    parser.add_argument("--rotation-tol-deg", type=float, default=0.25, help="Rotation tolerance in degrees")
    parser.add_argument("--fre-tol-mm", type=float, default=0.05, help="FRE metric tolerance in mm")
    parser.add_argument("--save-report", type=Path, default=None, help="Optional path for the comparison JSON report")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    left = load_registration_output(args.left)
    right = load_registration_output(args.right)
    report = compare_registration_outputs(
        left,
        right,
        translation_tolerance_mm=args.translation_tol_mm,
        rotation_tolerance_deg=args.rotation_tol_deg,
        fre_tolerance_mm=args.fre_tol_mm,
    )

    print(f"left_source={report.left_source}")
    print(f"right_source={report.right_source}")
    print(f"translation_tolerance_mm={report.translation_tolerance_mm}")
    print(f"rotation_tolerance_deg={report.rotation_tolerance_deg}")
    print(f"fre_tolerance_mm={report.fre_tolerance_mm}")
    print(f"tool_role_match={report.tool_role_match}")
    print(f"tool_role_note={report.tool_role_note}")
    for item in report.transform_comparisons:
        print(
            f"{item.name}: compared={item.compared} required={item.required} "
            f"within_tolerance={item.within_tolerance} "
            f"translation_difference_mm={item.translation_difference_mm} "
            f"rotation_difference_deg={item.rotation_difference_deg} "
            f"note={item.note}"
        )
    print(f"metric_differences_mm={json.dumps(report.metric_differences_mm, sort_keys=True)}")
    print(f"passed={report.passed}")

    if args.save_report is not None:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(json.dumps(report, default=lambda obj: obj.__dict__, indent=2), encoding="utf-8")
        print(f"saved_report={args.save_report}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
