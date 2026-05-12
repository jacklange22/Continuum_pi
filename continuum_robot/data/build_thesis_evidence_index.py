"""Build a lightweight thesis evidence index from reviewed run folders."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from continuum_robot.data.run_management import ManagedRunSummary, discover_experiment_run_dirs, summarize_run
from continuum_robot.experiments.dataset_io import canonical_timestamp_token


def build_thesis_evidence_index(
    *,
    project_root: Path,
    output_root: Path | None = None,
    experiment_name: str | None = None,
    include_debug: bool = False,
    include_mock: bool = False,
    include_unreviewed: bool = False,
) -> Path:
    """Write thesis_evidence_index.md/json and return the output directory."""
    project_root = Path(project_root).resolve()
    output_root = Path(output_root or (project_root / "data" / "exports")).resolve()
    output_dir = _collision_safe_dir(output_root / f"thesis_evidence_index_{canonical_timestamp_token()}")
    output_dir.mkdir(parents=True, exist_ok=False)
    runs = [
        summarize_run(run_dir)
        for run_dir in discover_experiment_run_dirs(project_root, experiment_name=experiment_name)
    ]
    included = [
        run
        for run in runs
        if _include_run(
            run,
            include_debug=include_debug,
            include_mock=include_mock,
            include_unreviewed=include_unreviewed,
        )
    ]
    payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "experiment_name_filter": experiment_name,
        "include_debug": bool(include_debug),
        "include_mock": bool(include_mock),
        "include_unreviewed": bool(include_unreviewed),
        "run_count_scanned": len(runs),
        "run_count_included": len(included),
        "experiments": _group_runs(included),
    }
    (output_dir / "thesis_evidence_index.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "thesis_evidence_index.md").write_text(_render_markdown(payload), encoding="utf-8")
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a reviewed thesis evidence index from experiment runs.")
    parser.add_argument("--project-root", default=".", help="Project root. Defaults to current directory.")
    parser.add_argument("--output-dir", help="Destination root. Defaults to data/exports.")
    parser.add_argument("--experiment", help="Limit index to one experiment name.")
    parser.add_argument("--include-debug", action="store_true", help="Explicitly include debug-reviewed runs.")
    parser.add_argument("--include-mock", action="store_true", help="Explicitly allow mock runs into the index.")
    parser.add_argument("--include-unreviewed", action="store_true", help="Explicitly include old runs missing run_review.json.")
    args = parser.parse_args(argv)
    output_dir = build_thesis_evidence_index(
        project_root=Path(args.project_root),
        output_root=Path(args.output_dir) if args.output_dir else None,
        experiment_name=args.experiment,
        include_debug=bool(args.include_debug),
        include_mock=bool(args.include_mock),
        include_unreviewed=bool(args.include_unreviewed),
    )
    print(f"Thesis evidence index: {output_dir}")
    print(f"- {output_dir / 'thesis_evidence_index.md'}")
    print(f"- {output_dir / 'thesis_evidence_index.json'}")
    return 0


def _include_run(
    run: ManagedRunSummary,
    *,
    include_debug: bool = False,
    include_mock: bool = False,
    include_unreviewed: bool = False,
) -> bool:
    if run.mock_mode is True and not include_mock:
        return False
    if run.run_trust_mode in {"mock", "servo_only", "current_only", "lower_trust", "debug", "debug_only"}:
        if run.run_trust_mode == "mock" and not include_mock:
            return False
        if run.run_trust_mode != "mock" and not include_debug:
            return False
    if not run.has_run_review and not include_unreviewed:
        return False
    if run.review.review_status in {"debug", "garbage", "archived"}:
        return bool(include_debug)
    if run.review.include_in_evidence_index:
        return True
    if run.review.review_status in {"thesis_candidate", "advisor_share"}:
        return True
    return False


def _group_runs(runs: list[ManagedRunSummary]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(run.experiment_name, []).append(_run_payload(run))
    for entries in grouped.values():
        entries.sort(key=lambda item: str(item.get("timestamp_label") or ""), reverse=True)
    return dict(sorted(grouped.items()))


def _run_payload(run: ManagedRunSummary) -> dict[str, Any]:
    payload = asdict(run)
    payload["run_dir"] = str(run.run_dir)
    payload["review"] = run.review.to_dict()
    payload["key_metrics"] = _extract_key_metrics(run.run_dir)
    return payload


def _extract_key_metrics(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    metrics = summary.get("experiment_metrics")
    if not isinstance(metrics, dict):
        return {}
    useful_keys = [
        "rmse_mm",
        "rms_error_mm",
        "fre_mm",
        "mean_fre_mm",
        "final_xy_error_mm",
        "quality_score",
        "position_rmse_mm",
        "angular_rmse_deg",
        "sample_count",
        "valid_sample_count",
        "accepted_sample_count",
        "rejected_sample_count",
        "source_dataset_run_ids",
        "best_model_by_xyz_rmse",
        "orientation_available",
        "distal_only",
        "includes_intermediate_pose",
        "includes_intermediate_label",
        "label_mode",
        "physics_model_status",
    ]
    extracted = {key: metrics[key] for key in useful_keys if key in metrics}
    if str(metrics.get("dataset_type") or "") == "two_segment_modeling":
        best = metrics.get("best_model_by_xyz_rmse") if isinstance(metrics.get("best_model_by_xyz_rmse"), dict) else {}
        extracted["two_segment_modeling"] = {
            "input_dataset_run_ids": metrics.get("source_dataset_run_ids", []),
            "accepted_sample_count": metrics.get("accepted_sample_count"),
            "rejected_sample_count": metrics.get("rejected_sample_count"),
            "best_model": best.get("model_key"),
            "best_xyz_rmse_mm": best.get("xyz_rmse_mm"),
            "best_orientation_mean_error_deg": best.get("orientation_mean_error_deg"),
            "label_mode": metrics.get("label_mode"),
            "includes_intermediate_label": metrics.get("includes_intermediate_label"),
            "physics_model_status": metrics.get("physics_model_status", {}),
            "lower_trust_warning": "allow_lower_trust_used_outputs_not_thesis_trusted"
            in list(metrics.get("data_quality_warnings", []) or []),
        }
    return extracted


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Thesis Evidence Index",
        "",
        f"Created: {payload.get('created_at_utc')}",
        f"Runs scanned: {payload.get('run_count_scanned')}",
        f"Runs included: {payload.get('run_count_included')}",
        "",
        "This index lists reviewed thesis/advisor evidence only by default. It does not include raw samples.",
        "",
    ]
    experiments = payload.get("experiments")
    if not isinstance(experiments, dict) or not experiments:
        lines.append("No candidate evidence runs found.")
        return "\n".join(lines) + "\n"
    for experiment, runs in experiments.items():
        lines.extend([f"## {experiment}", ""])
        for run in runs:
            metrics = run.get("key_metrics") if isinstance(run.get("key_metrics"), dict) else {}
            figures = run.get("report_figures") if isinstance(run.get("report_figures"), list) else []
            warnings = run.get("data_quality_warnings") if isinstance(run.get("data_quality_warnings"), list) else []
            validation = run.get("validation_issues") if isinstance(run.get("validation_issues"), list) else []
            lines.extend(
                [
                    f"### {run.get('run_id') or Path(str(run.get('run_dir'))).name}",
                    "",
                    f"- Path: `{run.get('run_dir')}`",
                    f"- Validation: `{run.get('validation_status')}`",
                    f"- Trust mode: `{run.get('run_trust_mode')}`",
                    f"- Review status: `{(run.get('review') or {}).get('review_status', 'debug')}`",
                    f"- Model training valid: `{run.get('valid_for_model_training')}`",
                    f"- Thesis repeatability valid: `{run.get('valid_for_thesis_repeatability')}`",
                    f"- Operating mode: `{run.get('operating_mode')}`",
                    f"- Active segment: `{run.get('active_segment') or 'n/a'}`",
                    f"- Stop/failure reason: `{run.get('stop_or_failure_reason') or 'n/a'}`",
                ]
            )
            if metrics:
                lines.append(f"- Key metrics: `{json.dumps(metrics, sort_keys=True)}`")
            if figures:
                lines.append(f"- Report figures: {', '.join(f'`{figure}`' for figure in figures[:8])}")
            if warnings:
                lines.append(f"- Data quality warnings: {'; '.join(str(item) for item in warnings)}")
            if validation:
                lines.append(f"- Validation issues: {'; '.join(str(item) for item in validation[:6])}")
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def _collision_safe_dir(path: Path) -> Path:
    candidate = Path(path)
    if not candidate.exists():
        return candidate
    suffix = 1
    while True:
        next_candidate = candidate.with_name(f"{candidate.name}_{suffix:02d}")
        if not next_candidate.exists():
            return next_candidate
        suffix += 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
