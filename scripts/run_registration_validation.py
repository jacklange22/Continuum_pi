"""Validate registration outputs from a saved CSV or session artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from continuum_robot.app.bootstrap import build_app_context
from continuum_robot.registration.validation_tools import (
    RegistrationValidationReport,
    run_registration_validation_from_csv,
    run_registration_validation_from_session_json,
    save_validation_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate registration from saved CSV or session JSON")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--registration-csv", type=Path, help="Saved Aurora registration CSV")
    source.add_argument("--session-json", type=Path, help="Saved registration/session JSON with raw tool samples")
    parser.add_argument("--measurement-tool-id", type=str, default="", help="Measurement tool id override")
    parser.add_argument("--coil-tool-id", type=str, default="", help="Coil tool id override")
    parser.add_argument("--save-report", type=Path, default=None, help="Optional path for the validation JSON report")
    return parser.parse_args()


def _print_report(report: RegistrationValidationReport) -> None:
    print(f"source_kind={report.source_kind}")
    print(f"source_path={report.source_path}")
    print(f"measurement_tool_id={report.measurement_tool_id}")
    print(f"coil_tool_id={report.coil_tool_id}")
    print(f"repetition_count={report.repetition_count}")
    print(f"ordered_labels={report.ordered_labels}")
    print(f"point_counts={json.dumps(report.point_counts, sort_keys=True)}")
    print(f"captured_counts_by_label={json.dumps(report.captured_counts_by_label, sort_keys=True)}")
    for name in ("T_aurora_2_model", "T_aurora_2_tip", "T_tip_2_coil", "T_coil_tip"):
        matrix = np.asarray(report.transforms[name], dtype=float)
        print(f"{name}=")
        print(np.array2string(matrix, precision=6, suppress_small=False))
    print("validation_metrics=")
    print(json.dumps(report.validation_metrics, indent=2, sort_keys=True))


def main() -> int:
    args = _parse_args()
    ctx = build_app_context()
    service = ctx.services.get("live_registration")
    if service.assets is None:
        print("ERROR: rigorous registration validation requires configured protected asset files.")
        return 2

    if args.registration_csv is not None:
        report = run_registration_validation_from_csv(
            registration_csv=args.registration_csv,
            assets=service.assets,
            solver=service.solver,
            measurement_tool_id=args.measurement_tool_id or service.capture_tool_id,
            coil_tool_id=args.coil_tool_id or service.coil_tool_id,
            quaternion_average_method=service.quaternion_average_method,
            model_tre_reference_radius_mm=service.model_tre_reference_radius_mm,
            tip_tre_reference_radius_mm=service.tip_tre_reference_radius_mm,
        )
    else:
        report = run_registration_validation_from_session_json(
            session_json=args.session_json,
            assets=service.assets,
            solver=service.solver,
            measurement_tool_id=args.measurement_tool_id or None,
            coil_tool_id=args.coil_tool_id or None,
            quaternion_average_method=service.quaternion_average_method,
            model_tre_reference_radius_mm=service.model_tre_reference_radius_mm,
            tip_tre_reference_radius_mm=service.tip_tre_reference_radius_mm,
        )

    _print_report(report)
    if args.save_report is not None:
        path = save_validation_report(report, args.save_report)
        print(f"saved_report={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
