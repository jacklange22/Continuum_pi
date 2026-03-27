"""Run the legacy-compatible registration pipeline from a saved Aurora CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from continuum_robot.app.bootstrap import build_app_context


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run registration from a saved Aurora CSV")
    parser.add_argument("registration_csv", type=Path, help="Path to the Aurora registration CSV")
    parser.add_argument("--measurement-tool-id", type=str, default="", help="Tool id used for measured point captures")
    parser.add_argument("--coil-tool-id", type=str, default="", help="Tool id used for coil averaging")
    parser.add_argument(
        "--max-fre-mm",
        type=float,
        default=None,
        help="Optional FRE limit to enforce across model, tip, and overall metrics",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    ctx = build_app_context()
    service = ctx.services.get("live_registration")

    result = service.complete_registration_from_csv(
        args.registration_csv,
        config_used={"invoked_by": "scripts/run_registration_from_csv.py"},
        max_fre_mm=args.max_fre_mm,
        measurement_tool_id=args.measurement_tool_id or None,
        coil_tool_id=args.coil_tool_id or None,
    )

    metrics = result.record.validation_metrics
    print(f"Saved registration to {result.output_path}")
    print(f"measurement_tool_id={result.record.measurement_tool_id}")
    print(f"coil_tool_id={result.record.coil_tool_id}")
    print(f"overall_fre_mm={metrics.get('overall_fre_mm')}")
    print(f"model_fre_mm={metrics.get('model_fre_mm')}")
    print(f"tip_fre_mm={metrics.get('tip_fre_mm')}")
    print(f"T_robot_aurora={'present' if result.record.T_robot_aurora else 'missing'}")
    print(f"T_tip_2_coil={'present' if result.record.T_coil_tip else 'missing'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
