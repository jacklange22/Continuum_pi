#!/usr/bin/env python3
"""Write current transform-chain validation artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from continuum_robot.app.bootstrap import build_app_context
from continuum_robot.experiments.dataset_io import canonical_timestamped_name
from continuum_robot.experiments.transform_chain_outputs import write_transform_chain_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root. Defaults to data/diagnostics/transform_chain_validation.",
    )
    parser.add_argument(
        "--output-dir-name",
        default=None,
        help="Optional run directory name under the output root.",
    )
    args = parser.parse_args()

    context = build_app_context()
    tracking_service = context.services.get("tracking_service")
    snapshot = tracking_service.get_snapshot()
    output_root = args.output_root or (
        context.project_root / "data" / "diagnostics" / "transform_chain_validation"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / (
        args.output_dir_name
        or canonical_timestamped_name(output_root, "transform_chain_validation")
    )
    paths = write_transform_chain_outputs(
        output_dir=output_dir,
        snapshot=snapshot,
        workflow="transform_chain_validation",
        provenance_note="current_accepted_state_cli",
    )
    print(f"Transform-chain outputs written to: {output_dir}")
    for label, path in paths.items():
        print(f"- {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
