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
    grouped_experiments, top_level_warnings, lower_trust_count = _group_runs(included)
    payload = {
        "schema_version": "1.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "experiment_name_filter": experiment_name,
        "include_debug": bool(include_debug),
        "include_mock": bool(include_mock),
        "include_unreviewed": bool(include_unreviewed),
        "run_count_scanned": len(runs),
        "run_count_included": len(included),
        "run_count_lower_trust": int(lower_trust_count),
        "warnings": list(top_level_warnings),
        "experiments": grouped_experiments,
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


def _group_runs(
    runs: list[ManagedRunSummary],
) -> tuple[dict[str, list[dict[str, Any]]], list[str], int]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    top_level_warnings: list[str] = []
    lower_trust_count = 0
    for run in runs:
        payload = _run_payload(run)
        if payload.get("trust_level") == "lower_trust":
            lower_trust_count += 1
            for warning in payload.get("trust_warnings", []) or []:
                top_level_warnings.append(f"{run.experiment_name}/{run.run_id or run.run_dir.name}: {warning}")
        grouped.setdefault(run.experiment_name, []).append(payload)
    for entries in grouped.values():
        entries.sort(key=lambda item: str(item.get("timestamp_label") or ""), reverse=True)
    return dict(sorted(grouped.items())), top_level_warnings, lower_trust_count


def _run_payload(run: ManagedRunSummary) -> dict[str, Any]:
    payload = asdict(run)
    payload["run_dir"] = str(run.run_dir)
    payload["review"] = run.review.to_dict()
    payload["key_metrics"] = _extract_key_metrics(run.run_dir)
    trust_level, trust_warnings = _assess_trust_level(run)
    payload["trust_level"] = trust_level
    payload["trust_warnings"] = trust_warnings
    return payload


def _assess_trust_level(run: ManagedRunSummary) -> tuple[str, list[str]]:
    """Derive a trust label for an included run.

    A run that is curated as ``thesis_candidate``/``advisor_share`` but whose own
    summary reports the relevant validity flag as ``False`` is still included
    (per operator preference), but tagged ``lower_trust`` with a specific reason
    so downstream readers and the index header surface the inconsistency.
    """
    review_status = (run.review.review_status or "").strip().lower()
    if review_status not in {"thesis_candidate", "advisor_share"}:
        return "thesis_trusted", []
    warnings: list[str] = []
    experiment_name = (run.experiment_name or "").lower()
    is_modeling_experiment = (
        "modeling" in experiment_name
        or "collect_pose" in experiment_name
        or "two_segment" in experiment_name
    )
    is_registration_protocol_study = experiment_name == "registration_sampling_study"
    if is_registration_protocol_study:
        # This experiment is never thesis_repeatability- or model-training-valid by design;
        # the appropriate trust gate is whether a registration protocol recommendation
        # was produced. Look it up directly from the run's summary.json.
        recommendation_valid = _registration_protocol_recommendation_valid(run.run_dir)
        if recommendation_valid is False:
            warnings.append(
                f"review_status={review_status!r} but valid_for_registration_protocol_recommendation=False"
            )
    else:
        if run.valid_for_thesis_repeatability is False:
            warnings.append(
                f"review_status={review_status!r} but summary.valid_for_thesis_repeatability=False"
            )
        if is_modeling_experiment and run.valid_for_model_training is False:
            warnings.append(
                f"review_status={review_status!r} but summary.valid_for_model_training=False"
            )
    if (run.run_trust_mode or "").strip().lower() in {"servo_only", "current_only", "lower_trust", "debug", "debug_only"}:
        warnings.append(
            f"review_status={review_status!r} but run_trust_mode={run.run_trust_mode!r}"
        )
    return ("lower_trust" if warnings else "thesis_trusted", warnings)


def _registration_protocol_recommendation_valid(run_dir: Path) -> bool | None:
    """Return the run's `valid_for_registration_protocol_recommendation` flag or None."""
    summary_path = Path(run_dir) / "summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    metrics = summary.get("experiment_metrics") if isinstance(summary.get("experiment_metrics"), dict) else {}
    value = metrics.get("valid_for_registration_protocol_recommendation")
    if isinstance(value, bool):
        return value
    return None


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
        # registration_sampling_study
        "captured_label_count",
        "captured_sample_count_total",
        "candidate_registration_fre_mm",
        "candidate_registration_max_residual_mm",
        "candidate_registration_label_count",
        "recommended_protocol",
        "valid_for_registration_protocol_recommendation",
    ]
    extracted = {key: metrics[key] for key in useful_keys if key in metrics}
    dataset_type = str(metrics.get("dataset_type") or "")
    if dataset_type == "two_segment_modeling":
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
    elif dataset_type == "two_segment_collect_pose_command_dataset":
        long_run = dict(metrics.get("long_run_health") or {})
        current_load = dict(metrics.get("current_load_summary") or {})
        extracted["two_segment_collect_pose"] = {
            "schedule_type": metrics.get("schedule_type"),
            "bottom_segment_key": metrics.get("bottom_segment_key"),
            "top_segment_key": metrics.get("top_segment_key"),
            "accepted_sample_count": metrics.get("accepted_sample_count"),
            "rejected_sample_count": metrics.get("rejected_sample_count"),
            "command_failure_count": metrics.get("command_failure_count"),
            "stop_reason": long_run.get("stop_reason"),
            "transport_failures": long_run.get("transport_failures"),
            "continue_until_valid_samples": long_run.get("continue_until_valid_samples"),
            "target_valid_sample_count": long_run.get("target_valid_sample_count"),
            "valid_for_two_segment_model_training": metrics.get("valid_for_two_segment_model_training"),
            "run_trust_mode": metrics.get("run_trust_mode"),
            "current_load_any_warning_observed": current_load.get("any_warning_observed"),
            "current_load_any_hard_observed": current_load.get("any_hard_observed"),
            "current_load_sustained_jam_servo_ids": current_load.get("sustained_jam_servo_ids"),
            "tracker_any_stale_observed": (
                bool(dict(metrics.get("tracker_freshness_summary") or {}).get("any_stale_observed"))
                if isinstance(metrics.get("tracker_freshness_summary"), dict)
                else None
            ),
            "tracker_max_freshness_s": (
                dict(metrics.get("tracker_freshness_summary") or {}).get("max_freshness_s")
                if isinstance(metrics.get("tracker_freshness_summary"), dict)
                else None
            ),
        }
    elif dataset_type == "two_segment_repeatability":
        scatter = dict(metrics.get("scatter_metrics") or {})
        extracted["two_segment_repeatability"] = {
            "bottom_segment_key": metrics.get("bottom_segment_key"),
            "top_segment_key": metrics.get("top_segment_key"),
            "target_count": metrics.get("target_count"),
            "repeat_visits": metrics.get("repeat_visits"),
            "aggregate_distal_rms_mm": scatter.get("aggregate_distal_rms_mm"),
            "aggregate_intermediate_rms_mm": scatter.get("aggregate_intermediate_rms_mm"),
            "target_distal_rms_mm": metrics.get("target_distal_rms_mm"),
            "target_intermediate_rms_mm": metrics.get("target_intermediate_rms_mm"),
            "run_trust_mode": metrics.get("run_trust_mode"),
        }
    elif str(metrics.get("startup_type") or "") == "manual_two_segment_startup":
        extracted["two_segment_startup"] = {
            "bottom_segment_key": metrics.get("bottom_segment_key"),
            "top_segment_key": metrics.get("top_segment_key"),
            "final_accepted": metrics.get("final_accepted"),
            "tracker_available": metrics.get("tracker_available"),
            "stage_order": metrics.get("stage_order"),
        }
    return extracted


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Thesis Evidence Index",
        "",
        f"Created: {payload.get('created_at_utc')}",
        f"Runs scanned: {payload.get('run_count_scanned')}",
        f"Runs included: {payload.get('run_count_included')}",
        f"Lower-trust included: {payload.get('run_count_lower_trust', 0)}",
        "",
        "This index lists reviewed thesis/advisor evidence only by default. It does not include raw samples.",
        "",
    ]
    top_warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    if top_warnings:
        lines.extend(["## Lower-trust warnings", ""])
        for warning in top_warnings:
            lines.append(f"- {warning}")
        lines.append("")
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
            trust_warnings = run.get("trust_warnings") if isinstance(run.get("trust_warnings"), list) else []
            lines.extend(
                [
                    f"### {run.get('run_id') or Path(str(run.get('run_dir'))).name}",
                    "",
                    f"- Path: `{run.get('run_dir')}`",
                    f"- Validation: `{run.get('validation_status')}`",
                    f"- Trust level: `{run.get('trust_level', 'thesis_trusted')}`",
                    f"- Trust mode: `{run.get('run_trust_mode')}`",
                    f"- Review status: `{(run.get('review') or {}).get('review_status', 'debug')}`",
                    f"- Model training valid: `{run.get('valid_for_model_training')}`",
                    f"- Thesis repeatability valid: `{run.get('valid_for_thesis_repeatability')}`",
                    f"- Operating mode: `{run.get('operating_mode')}`",
                    f"- Active segment: `{run.get('active_segment') or 'n/a'}`",
                    f"- Stop/failure reason: `{run.get('stop_or_failure_reason') or 'n/a'}`",
                ]
            )
            if trust_warnings:
                lines.append(f"- Trust warnings: {'; '.join(str(item) for item in trust_warnings)}")
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
