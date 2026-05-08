"""CLI for offline two-segment modeling scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

from continuum_robot.data.export_run_bundle import find_latest_run
from continuum_robot.modeling.two_segment.dataset import load_two_segment_modeling_dataset
from continuum_robot.modeling.two_segment.train import TwoSegmentModelingConfig, run_two_segment_modeling


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run offline two-segment modeling analysis on collected dataset runs.")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--runs", nargs="+", help="One or more two_segment_collect_pose_command_dataset run directories.")
    selector.add_argument("--latest", action="store_true", help="Use the latest two_segment_collect_pose_command_dataset run.")
    parser.add_argument("--project-root", default=".", help="Project root. Defaults to current directory.")
    parser.add_argument("--config", help="YAML config path, for example config/modeling_two_segment.example.yaml.")
    parser.add_argument("--output-root", help="Output root. Defaults to data/experiments/two_segment_modeling.")
    parser.add_argument("--output-dir", help="Exact output directory to create. Existing paths get a numeric suffix.")
    parser.add_argument("--allow-lower-trust", action="store_true", help="Allow lower-trust labeled samples and mark outputs lower-trust.")
    parser.add_argument("--models", nargs="+", help="Model keys to attempt: linear_baseline camarillo mike_constant_curvature ann.")
    parser.add_argument("--test-fraction", type=float, help="Test split fraction.")
    parser.add_argument("--seed", type=int, help="Random seed for split/training.")
    parser.add_argument("--by-run-split", action="store_true", help="Force a by-run split when multiple runs are supplied.")
    parser.add_argument("--random-split", action="store_true", help="Force a random sample split.")
    parser.add_argument("--figure-quality", default="production", choices=["low", "medium", "production"], help="Report PNG quality.")
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve()
    try:
        run_dirs = _resolve_run_dirs(args=args, project_root=project_root)
    except Exception as exc:
        print(f"Two-segment modeling setup failed: {exc}", file=sys.stderr)
        return 2
    config = _config_from_args(args)
    try:
        result = run_two_segment_modeling(
            run_dirs=run_dirs,
            project_root=project_root,
            config=config,
        )
    except ValueError as exc:
        _print_no_trainable_data_report(run_dirs=run_dirs, config=config, error=str(exc))
        return 2
    print(f"Two-segment modeling output: {result.output_dir}")
    print(f"Accepted samples: {result.dataset.accepted_count}")
    print(f"Rejected samples: {result.dataset.rejected_count}")
    for model_result in result.model_results:
        rmse = model_result.metrics.get("xyz_rmse_mm") if model_result.metrics else None
        suffix = f", xyz_rmse_mm={rmse:.4g}" if isinstance(rmse, float) else f", reason={model_result.reason}" if model_result.reason else ""
        print(f"- {model_result.model_key}: {model_result.status}{suffix}")
    return 0


def _resolve_run_dirs(*, args: argparse.Namespace, project_root: Path) -> list[Path]:
    if bool(args.latest):
        return [
            find_latest_run(
                project_root=project_root,
                experiment_name="two_segment_collect_pose_command_dataset",
            )
        ]
    return [Path(value).resolve() for value in list(args.runs or [])]


def _config_from_args(args: argparse.Namespace) -> TwoSegmentModelingConfig:
    payload = _read_yaml_config(Path(args.config)) if args.config else {}
    split = dict(payload.get("split", {}) or {})
    models = dict(payload.get("models", {}) or {})
    model_config = _model_config(payload)
    split_method = str(split.get("method", "auto") or "auto").lower()
    by_run_split = None
    if split_method == "by_run":
        by_run_split = True
    elif split_method in {"random", "single_run_random"}:
        by_run_split = False
    if args.by_run_split:
        by_run_split = True
    if args.random_split:
        by_run_split = False
    return TwoSegmentModelingConfig(
        allow_lower_trust=bool(args.allow_lower_trust or payload.get("allow_lower_trust", False)),
        test_fraction=float(args.test_fraction if args.test_fraction is not None else split.get("test_fraction", 0.25)),
        random_seed=int(args.seed if args.seed is not None else split.get("seed", payload.get("seed", 0))),
        by_run_split=by_run_split,
        model_keys=[str(value) for value in (args.models or models.get("enabled") or ["linear_baseline", "camarillo", "mike_constant_curvature", "ann"])],
        model_config=model_config,
        output_root=args.output_root or dict(payload.get("output", {}) or {}).get("output_root"),
        output_dir=args.output_dir or dict(payload.get("output", {}) or {}).get("output_dir"),
        figure_quality=str(args.figure_quality or dict(payload.get("output", {}) or {}).get("figure_quality", "production")),
    )


def _read_yaml_config(path: Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return dict(payload or {}) if isinstance(payload, dict) else {}


def _model_config(payload: dict) -> dict:
    linear = dict(payload.get("linear_baseline", {}) or {})
    ann = dict(payload.get("ann", {}) or {})
    physics = dict(payload.get("physics_models", {}) or {})
    result = {
        "ridge_alpha": float(linear.get("ridge_alpha", 1e-6)),
        "ann": ann,
    }
    if physics:
        result["physics_models"] = physics
        result["camarillo"] = physics
        result["mike_constant_curvature"] = physics
    return result


def _print_no_trainable_data_report(*, run_dirs: list[Path], config: TwoSegmentModelingConfig, error: str) -> None:
    dataset = load_two_segment_modeling_dataset(run_dirs, allow_lower_trust=bool(config.allow_lower_trust))
    print("Two-segment modeling could not start: no trainable dataset is available.", file=sys.stderr)
    print(f"Reason: {error}", file=sys.stderr)
    print(f"Runs scanned: {len(run_dirs)}", file=sys.stderr)
    print(f"Samples scanned: {dataset.accepted_count + dataset.rejected_count}", file=sys.stderr)
    print(f"Samples accepted: {dataset.accepted_count}", file=sys.stderr)
    print(f"Samples rejected: {dataset.rejected_count}", file=sys.stderr)
    print(f"Rejection reasons: {dataset.rejection_counts()}", file=sys.stderr)
    print("Trainability requires:", file=sys.stderr)
    print("- two_segment_collect_pose_command_dataset samples", file=sys.stderr)
    print("- accepted all-8 startup provenance", file=sys.stderr)
    print("- non-servo-only trusted run mode unless --allow-lower-trust is explicit", file=sys.stderr)
    print("- successful accepted commands", file=sys.stderr)
    print("- 8 tendon displacements with known units", file=sys.stderr)
    print("- distal_tip pose label in robot frame", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
