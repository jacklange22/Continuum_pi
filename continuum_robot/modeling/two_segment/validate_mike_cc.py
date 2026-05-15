"""CLI for the Mike constant-curvature convention probe.

The operator collects a small two-segment dataset (e.g. ``bottom_only_sweep``
or ``workspace_coverage`` at conservative amplitude) and runs this probe to
check whether the configured sign/frame conventions match the bench. The probe
NEVER auto-flips the convention flag in config — it only reports evidence and
a recommendation; the operator decides.

Example:

    .venv/bin/python -m continuum_robot.modeling.two_segment.validate_mike_cc \
        --runs data/experiments/two_segment_collect_pose_command_dataset/<run> \
        --config config/modeling_two_segment.example.yaml \
        --output-dir data/experiments/two_segment_mike_convention_probe/<probe>

The report is written to ``mike_cc_convention_report.json`` and
``mike_cc_convention_report.txt`` in the output directory. Exit code is 0 when
the recommendation is "conventions_consistent_with_evidence_safe_to_confirm";
2 otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

from continuum_robot.data.export_run_bundle import find_latest_run
from continuum_robot.modeling.two_segment.dataset import load_two_segment_modeling_dataset
from continuum_robot.modeling.two_segment.physics import (
    MikeCCConventionReport,
    validate_mike_cc_conventions,
)


def _read_yaml_config(path: Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return dict(payload or {}) if isinstance(payload, dict) else {}


def _render_text(report: MikeCCConventionReport) -> str:
    lines = [
        "Mike Constant-Curvature Convention Probe",
        f"sample_count: {report.sample_count}",
        f"distal_residual_norm_mean_mm: {report.distal_residual_norm_mean_mm:.3f}",
        f"distal_residual_norm_median_mm: {report.distal_residual_norm_median_mm:.3f}",
        f"distal_residual_norm_p95_mm: {report.distal_residual_norm_p95_mm:.3f}",
        f"distal_residual_norm_max_mm: {report.distal_residual_norm_max_mm:.3f}",
        f"intermediate_residual_norm_mean_mm: "
        f"{report.intermediate_residual_norm_mean_mm if report.intermediate_residual_norm_mean_mm is not None else 'n/a'}",
        f"intermediate_residual_norm_max_mm: "
        f"{report.intermediate_residual_norm_max_mm if report.intermediate_residual_norm_max_mm is not None else 'n/a'}",
        f"per_axis_distal_signs_match: {report.per_axis_distal_signs_match}",
        f"largest_distal_sign_disagreement_axis: {report.largest_distal_sign_disagreement_axis or 'none'}",
        f"intermediate_sample_count: {report.intermediate_sample_count}",
        f"per_axis_intermediate_signs_match: {report.per_axis_intermediate_signs_match or 'n/a (no intermediate samples)'}",
        f"largest_intermediate_sign_disagreement_axis: {report.largest_intermediate_sign_disagreement_axis or 'none'}",
        f"recommendation: {report.recommendation}",
        "",
        "Notes:",
    ]
    for note in report.notes or ["(no additional notes)"]:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "This probe is evidence-only. It does NOT change config or modeling artifacts.",
            "Set physics_models.mike_constant_curvature.required_conventions_confirmed only after",
            "reviewing the residuals and signs above.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe Mike constant-curvature conventions against measured samples.")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--runs", nargs="+", help="One or more two_segment_collect_pose_command_dataset run directories.")
    selector.add_argument("--latest", action="store_true", help="Use the latest two_segment_collect_pose_command_dataset run.")
    parser.add_argument("--project-root", default=".", help="Project root. Defaults to current directory.")
    parser.add_argument("--config", help="YAML config with physics_models.* block (e.g. config/modeling_two_segment.example.yaml).")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write mike_cc_convention_report.{json,txt} into. Created if it does not exist.",
    )
    parser.add_argument(
        "--allow-lower-trust",
        action="store_true",
        help="Allow lower-trust samples (servo-only, dry-run) into the probe. Use with caution; the probe is descriptive either way.",
    )
    parser.add_argument(
        "--distal-mean-threshold-mm",
        type=float,
        default=5.0,
        help="Distal residual mean tolerance for the 'safe-to-confirm' recommendation.",
    )
    parser.add_argument(
        "--distal-max-threshold-mm",
        type=float,
        default=15.0,
        help="Distal residual max tolerance for the 'safe-to-confirm' recommendation.",
    )
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve()
    if args.latest:
        try:
            run_dirs = [find_latest_run(project_root=project_root, experiment_name="two_segment_collect_pose_command_dataset")]
        except Exception as exc:
            print(f"Mike convention probe setup failed: {exc}", file=sys.stderr)
            print(
                "Hint: collect a `two_segment_collect_pose_command_dataset` run first, "
                "then re-run the probe with --latest or pass --runs <path>.",
                file=sys.stderr,
            )
            return 2
    else:
        run_dirs = [Path(value).resolve() for value in list(args.runs or [])]
        # Friendly first-time error when --runs points at a typo or missing dir.
        missing = [
            run_dir
            for run_dir in run_dirs
            if not run_dir.exists() or not (run_dir / "samples.jsonl").exists()
        ]
        if missing:
            print(
                "Mike convention probe setup failed: the following run directories are missing or do not contain samples.jsonl:",
                file=sys.stderr,
            )
            for run_dir in missing:
                print(f"  - {run_dir}", file=sys.stderr)
            print(
                "Hint: check the path and confirm the directory was produced by `two_segment_collect_pose_command_dataset`.",
                file=sys.stderr,
            )
            return 2
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Mike convention probe setup failed: config file not found: {config_path}", file=sys.stderr)
            return 2
        config = _read_yaml_config(config_path)
    else:
        config = {}

    dataset = load_two_segment_modeling_dataset(run_dirs, allow_lower_trust=bool(args.allow_lower_trust))
    report = validate_mike_cc_conventions(
        samples=list(dataset.samples),
        config=config,
        distal_mean_threshold_mm=float(args.distal_mean_threshold_mm),
        distal_max_threshold_mm=float(args.distal_max_threshold_mm),
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mike_cc_convention_report.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    (output_dir / "mike_cc_convention_report.txt").write_text(_render_text(report), encoding="utf-8")

    print(f"Mike CC convention probe wrote {output_dir / 'mike_cc_convention_report.json'}")
    print(f"sample_count={report.sample_count}")
    print(f"distal_residual_mean_mm={report.distal_residual_norm_mean_mm:.3f}")
    print(f"distal_residual_max_mm={report.distal_residual_norm_max_mm:.3f}")
    print(f"recommendation: {report.recommendation}")
    return 0 if report.recommendation == "conventions_consistent_with_evidence_safe_to_confirm" else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
