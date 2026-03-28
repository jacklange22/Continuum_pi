"""Canonical CLI helpers for experiment execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from continuum_robot.app.bootstrap import build_app_context


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the canonical experiment CLI parser."""
    parser = argparse.ArgumentParser(description="Run canonical continuum robot experiments")
    parser.add_argument("--list", action="store_true", help="List available experiments and exit")
    parser.add_argument("--experiment", type=str, default="", help="Registered experiment name to run")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML or JSON experiment config file")
    parser.add_argument("--notes", type=str, default="", help="Optional operator notes stored in metadata")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory override")
    parser.add_argument("--save-result", type=Path, default=None, help="Optional path to save a compact JSON result")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the canonical experiment CLI."""
    args = build_arg_parser().parse_args(argv)
    ctx = build_app_context()
    runner = ctx.services.get("experiment_runner")
    if args.list:
        for descriptor in runner.available_experiments():
            print(f"{descriptor.name}: {descriptor.description}")
        return 0
    if not args.experiment:
        print("ERROR: --experiment is required unless --list is used.")
        return 2
    config = _load_config(args.config) if args.config is not None else {}
    result = runner.run_experiment(
        args.experiment,
        config=config,
        operator_notes=args.notes,
        output_dir=args.output_dir,
    )
    print(f"experiment_name={result.experiment_name}")
    print(f"run_id={result.run_id}")
    print(f"success={result.success}")
    print(f"status={result.summary.status}")
    print(f"message={result.message}")
    print(f"output_dir={result.paths.output_dir}")
    print(f"metadata_path={result.paths.metadata_path}")
    print(f"samples_path={result.paths.samples_path}")
    print(f"summary_path={result.paths.summary_path}")
    print(f"sample_count={result.sample_count}")
    print(f"stage_pass_fail={result.summary.stage_pass_fail}")
    print(f"error_messages={result.summary.error_messages}")
    if args.save_result is not None:
        args.save_result.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "experiment_name": result.experiment_name,
            "run_id": result.run_id,
            "success": result.success,
            "status": result.summary.status,
            "message": result.message,
            "output_dir": str(result.paths.output_dir),
            "metadata_path": str(result.paths.metadata_path),
            "samples_path": str(result.paths.samples_path),
            "summary_path": str(result.paths.summary_path),
            "sample_count": result.sample_count,
            "stage_pass_fail": dict(result.summary.stage_pass_fail),
            "error_messages": list(result.summary.error_messages),
        }
        args.save_result.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"saved_result={args.save_result}")
    return 0 if result.success else 1


def _load_config(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(raw)
    else:
        payload = yaml.safe_load(raw) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Experiment config must be a mapping: {path}")
    return dict(payload)
