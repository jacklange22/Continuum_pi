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

    # Surface thesis-eligibility verdict + generated figure paths for any
    # experiment that emits them (currently tracker_timing_validation and
    # servo_tracker_sync_validation). Other experiments emit empty / absent
    # blocks and these lines silently skip.
    eligibility = _extract_thesis_eligibility(result)
    if eligibility is not None:
        print(f"thesis_eligibility_label={eligibility.get('label', '')}")
        print(f"thesis_eligibility_eligible={eligibility.get('eligible', '')}")
        reasons = eligibility.get("reasons") or []
        for reason in reasons:
            print(f"thesis_eligibility_reason={reason}")
    generated_figures = _extract_generated_figures(result)
    for figure_path in generated_figures:
        print(f"figure={figure_path}")

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
            "thesis_eligibility": eligibility,
            "generated_figures": [str(path) for path in generated_figures],
        }
        args.save_result.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"saved_result={args.save_result}")
    return 0 if result.success else 1


def _extract_thesis_eligibility(result: Any) -> dict[str, Any] | None:
    """Pull the thesis_eligibility verdict out of summary.experiment_metrics.

    Returns ``None`` (not a placeholder) when the experiment didn't stamp
    one — keeps the CLI output clean for experiments that don't use the
    eligibility system yet.
    """
    metrics = getattr(result.summary, "experiment_metrics", None) or {}
    eligibility = metrics.get("thesis_eligibility")
    if isinstance(eligibility, dict) and eligibility:
        return dict(eligibility)
    return None


def _extract_generated_figures(result: Any) -> list[Path]:
    """Return the PNG figure paths the run wrote next to its output dir.

    We scan the output dir on disk rather than relying on the writer's
    return value (the writer's dict isn't carried on the result struct).
    Sorted for deterministic CLI output.
    """
    output_dir = getattr(result.paths, "output_dir", None)
    if output_dir is None:
        return []
    output_dir = Path(output_dir)
    if not output_dir.exists() or not output_dir.is_dir():
        return []
    return sorted(output_dir.glob("*.png"))


def _load_config(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(raw)
    else:
        payload = yaml.safe_load(raw) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Experiment config must be a mapping: {path}")
    return dict(payload)
