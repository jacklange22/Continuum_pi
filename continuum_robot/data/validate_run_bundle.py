"""Validate run-folder trust/provenance completeness without changing metrics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Any


EXPECTED_REPORT_FIGURES = {
    "aurora_grid_accuracy": [
        "aurora_grid_alignment_report.png",
        "aurora_grid_residuals_report.png",
    ],
    "pivot_validation": [
        "thesis_01_tip_vectors_3d.png",
        "thesis_02_sample_count_vs_quality.png",
    ],
    "registration_validation": [
        "thesis_01_robot_origins_3d.png",
        "thesis_02_within_vs_cross_run_quality.png",
    ],
    "single_segment_repeatability": [
        "repeatability_clusters_report.png",
        "repeatability_error_by_target_report.png",
    ],
    "pretension_validation": [
        "pretension_tip_xy_path_report.png",
        "pretension_load_proxy_by_servo_report.png",
        "pretension_tendon_displacement_vs_load_proxy_report.png",
        "pretension_final_state_report.png",
    ],
    "collect_pose_command_dataset": [
        "modeling_workspace_coverage_report.png",
        "commanded_tendon_space_report.png",
    ],
    "two_segment_startup_validation": [
        "two_segment_startup_positions_report.png",
        "two_segment_startup_load_proxy_report.png",
        "two_segment_startup_stage_drift_report.png",
    ],
    "two_segment_collect_pose_command_dataset": [
        "two_segment_command_coverage_report.png",
        "two_segment_servo_position_coverage_report.png",
        "two_segment_pose_coverage_report.png",
        "two_segment_dataset_quality_report.png",
    ],
    "two_segment_modeling": [
        "two_segment_model_comparison_report.png",
        "two_segment_distal_measured_vs_predicted_xy_report.png",
        "two_segment_position_error_distribution_report.png",
        "two_segment_axis_error_report.png",
        "two_segment_two_coil_error_report.png",
    ],
    "two_segment_repeatability": [
        "two_segment_repeatability_distal_scatter.png",
        "two_segment_repeatability_per_target_rms.png",
    ],
}

TWO_SEGMENT_STARTUP_REQUIRED_STAGES = [
    "baseline",
    "bottom_pretensioned",
    "top_pretensioned",
    "bottom_recheck",
    "final_accept",
]
# Legacy stage names (pre-bottom/top abstraction) accepted as equivalent during
# validation so older runs continue to pass.
TWO_SEGMENT_STARTUP_STAGE_ALIASES = {
    "segment_a_pretensioned": "bottom_pretensioned",
    "segment_b_pretensioned": "top_pretensioned",
    "segment_a_recheck": "bottom_recheck",
}


@dataclass(frozen=True)
class RunValidationIssue:
    level: str
    message: str


@dataclass(frozen=True)
class RunValidationReport:
    run_dir: Path
    experiment_name: str
    status: str
    trust_interpretation: str
    issues: list[RunValidationIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


def validate_run_folder(run_dir: Path) -> RunValidationReport:
    """Return a PASS/WARN/FAIL report for output completeness and trust metadata."""
    run_dir = Path(run_dir)
    issues: list[RunValidationIssue] = []
    summary_path = run_dir / "summary.json"
    metadata_path = run_dir / "metadata.json"
    summary = _read_json(summary_path, issues, required=True)
    metadata = _read_json(metadata_path, issues, required=False)
    experiment_name = str(
        metadata.get("experiment_name")
        or summary.get("experiment_name")
        or _infer_experiment_name(run_dir)
        or "unknown"
    )
    metrics = summary.get("experiment_metrics") if isinstance(summary.get("experiment_metrics"), dict) else {}
    metadata_trust = metadata.get("trust_info") if isinstance(metadata.get("trust_info"), dict) else {}
    metadata_provenance = metadata.get("provenance_info") if isinstance(metadata.get("provenance_info"), dict) else {}
    run_trust = metrics.get("run_trust") if isinstance(metrics.get("run_trust"), dict) else {}
    run_provenance = metrics.get("run_provenance") if isinstance(metrics.get("run_provenance"), dict) else {}

    if summary_path.exists() and not summary:
        issues.append(RunValidationIssue("FAIL", "summary.json exists but could not be read as an object."))
    if not metadata_path.exists():
        issues.append(RunValidationIssue("WARN", "metadata.json is missing; provenance may be incomplete."))

    _check_any_field(
        issues,
        "run_trust_mode",
        metrics.get("run_trust_mode"),
        run_trust.get("run_trust_mode"),
        metadata_trust.get("run_trust_mode"),
    )
    _check_any_field(issues, "operating_mode", run_provenance.get("operating_mode"), metadata_provenance.get("operating_mode"))
    _check_any_field(issues, "hardware_profile", run_provenance.get("hardware_profile"), metadata_provenance.get("hardware_profile"))
    operating_mode = run_provenance.get("operating_mode") or metadata_provenance.get("operating_mode")
    two_segment_foundation = (
        run_provenance.get("two_segment_foundation")
        if isinstance(run_provenance.get("two_segment_foundation"), dict)
        else metadata_provenance.get("two_segment_foundation")
    )
    if operating_mode == "dual_segment":
        _check_any_field(issues, "two_segment_foundation", two_segment_foundation)
    if isinstance(two_segment_foundation, dict) and two_segment_foundation:
        _check_any_field(issues, "two_segment command schema", _nested(two_segment_foundation, "command_schema", "schema_version"))
        _check_any_field(issues, "two_segment pose schema", _nested(two_segment_foundation, "pose_schema", "schema_version"))
    if experiment_name == "two_segment_startup_validation":
        _check_two_segment_startup_validation(issues, run_dir=run_dir, metrics=metrics)
        _check_physical_assembly_metadata(issues, metrics)
    if experiment_name == "two_segment_collect_pose_command_dataset":
        _check_two_segment_collect_pose_dataset(issues, run_dir=run_dir, metrics=metrics)
        _check_physical_assembly_metadata(issues, metrics)
    if experiment_name == "two_segment_modeling":
        _check_two_segment_modeling(issues, run_dir=run_dir, metrics=metrics)
    if experiment_name == "two_segment_repeatability":
        _check_physical_assembly_metadata(issues, metrics)
    _check_any_field(
        issues,
        "valid_for_model_training",
        metrics.get("valid_for_model_training"),
        run_trust.get("valid_for_model_training"),
        metadata_trust.get("valid_for_model_training"),
    )
    _check_any_field(
        issues,
        "valid_for_thesis_repeatability",
        metrics.get("valid_for_thesis_repeatability"),
        run_trust.get("valid_for_thesis_repeatability"),
        metadata_trust.get("valid_for_thesis_repeatability"),
    )
    if experiment_name in {"single_segment_repeatability", "collect_pose_command_dataset", "pretension_validation"}:
        _check_any_field(
            issues,
            "runtime_tip info",
            _nested(run_provenance, "runtime_tip_calibration", "mode"),
            _nested(metadata_provenance, "runtime_tip_calibration", "mode"),
            metrics.get("runtime_tip_mode_used"),
        )
    if operating_mode == "single_segment":
        _check_any_field(
            issues,
            "active_segment",
            run_provenance.get("active_segment"),
            metadata_provenance.get("active_segment"),
            _nested(run_provenance, "operating_context", "active_segment"),
        )
    if str(summary.get("status", "")).lower() in {"failed", "error", "partial_success"} or not bool(summary.get("success", True)):
        _check_any_field(issues, "stop/failure reason", metrics.get("stop_reason"), metrics.get("failure_reason"), summary.get("failure_reason"))

    expected = EXPECTED_REPORT_FIGURES.get(experiment_name, [])
    for filename in expected:
        if not (run_dir / filename).exists():
            issues.append(RunValidationIssue("WARN", f"Expected report figure is missing: {filename}"))

    trust_mode = (
        metrics.get("run_trust_mode")
        or run_trust.get("run_trust_mode")
        or metadata_trust.get("run_trust_mode")
        or "unknown"
    )
    model_valid = (
        metrics.get("valid_for_model_training")
        if metrics.get("valid_for_model_training") is not None
        else run_trust.get("valid_for_model_training", metadata_trust.get("valid_for_model_training"))
    )
    thesis_valid = (
        metrics.get("valid_for_thesis_repeatability")
        if metrics.get("valid_for_thesis_repeatability") is not None
        else run_trust.get("valid_for_thesis_repeatability", metadata_trust.get("valid_for_thesis_repeatability"))
    )
    trust_interpretation = (
        f"run_trust_mode={trust_mode}; valid_for_model_training={model_valid}; "
        f"valid_for_thesis_repeatability={thesis_valid}"
    )
    if any(issue.level == "FAIL" for issue in issues):
        status = "FAIL"
    elif issues:
        status = "WARN"
    else:
        status = "PASS"
    return RunValidationReport(
        run_dir=run_dir,
        experiment_name=experiment_name,
        status=status,
        trust_interpretation=trust_interpretation,
        issues=issues,
    )


def render_validation_report(report: RunValidationReport) -> str:
    lines = [
        f"{report.status}: {report.run_dir}",
        f"Experiment: {report.experiment_name}",
        f"Trust: {report.trust_interpretation}",
    ]
    if report.issues:
        lines.append("Issues:")
        for issue in report.issues:
            lines.append(f"- {issue.level}: {issue.message}")
    else:
        lines.append("No completeness issues found.")
    return "\n".join(lines)


def validation_report_to_dict(report: RunValidationReport) -> dict[str, Any]:
    """Machine-readable view of a validation report (for CI gating / scripted handoff)."""
    return {
        "schema_version": "run_validation_report_v1",
        "run_dir": str(report.run_dir),
        "experiment_name": str(report.experiment_name),
        "status": str(report.status),
        "trust_interpretation": str(report.trust_interpretation),
        "issues": [{"level": str(issue.level), "message": str(issue.message)} for issue in report.issues],
        "fail_count": int(sum(1 for issue in report.issues if issue.level == "FAIL")),
        "warn_count": int(sum(1 for issue in report.issues if issue.level == "WARN")),
        "info_count": int(sum(1 for issue in report.issues if issue.level == "INFO")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate one run folder for trust/provenance completeness.")
    parser.add_argument("run_dir", help="Experiment run directory to validate.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON report to stdout instead of the human-readable text.",
    )
    args = parser.parse_args(argv)
    report = validate_run_folder(Path(args.run_dir))
    if args.json:
        print(json.dumps(validation_report_to_dict(report), indent=2))
    else:
        print(render_validation_report(report))
    return 1 if report.status == "FAIL" else 0


def _read_json(path: Path, issues: list[RunValidationIssue], *, required: bool) -> dict[str, Any]:
    if not path.exists():
        if required:
            issues.append(RunValidationIssue("FAIL", f"Required file is missing: {path.name}"))
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        issues.append(RunValidationIssue("FAIL", f"Could not read {path.name}: {exc}"))
        return {}
    if not isinstance(payload, dict):
        issues.append(RunValidationIssue("FAIL", f"{path.name} is not a JSON object."))
        return {}
    return payload


def _check_any_field(issues: list[RunValidationIssue], label: str, *values: Any) -> None:
    if not any(value not in (None, "", {}) for value in values):
        issues.append(RunValidationIssue("WARN", f"Missing expected field: {label}"))


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _check_physical_assembly_metadata(issues: list[RunValidationIssue], metrics: dict[str, Any]) -> None:
    """Two-segment runs must record the bottom/top physical-role assignment."""
    assembly = metrics.get("physical_assembly") if isinstance(metrics.get("physical_assembly"), dict) else {}
    bottom_key = str(metrics.get("bottom_segment_key") or assembly.get("bottom_segment_key") or "")
    top_key = str(metrics.get("top_segment_key") or assembly.get("top_segment_key") or "")
    if not bottom_key or not top_key:
        issues.append(RunValidationIssue("WARN", "Two-segment run is missing bottom/top physical assembly metadata."))
        return
    if bottom_key == top_key:
        issues.append(RunValidationIssue("WARN", f"Two-segment bottom/top assembly is invalid: bottom={bottom_key} top={top_key}."))


def _check_two_segment_startup_validation(
    issues: list[RunValidationIssue],
    *,
    run_dir: Path,
    metrics: dict[str, Any],
) -> None:
    snapshots = metrics.get("stage_snapshots") if isinstance(metrics.get("stage_snapshots"), list) else []
    captured = {
        TWO_SEGMENT_STARTUP_STAGE_ALIASES.get(str(dict(stage or {}).get("stage")), str(dict(stage or {}).get("stage")))
        for stage in snapshots
        if isinstance(stage, dict)
    }
    missing_stages = [stage for stage in TWO_SEGMENT_STARTUP_REQUIRED_STAGES if stage not in captured]
    if missing_stages:
        issues.append(RunValidationIssue("WARN", f"Missing two-segment startup stage snapshot(s): {missing_stages}"))
    raw_order = [str(value) for value in metrics.get("stage_order", [])] if isinstance(metrics.get("stage_order"), list) else []
    normalized_order = [TWO_SEGMENT_STARTUP_STAGE_ALIASES.get(stage, stage) for stage in raw_order]
    if normalized_order != TWO_SEGMENT_STARTUP_REQUIRED_STAGES:
        issues.append(RunValidationIssue("WARN", f"Unexpected two-segment startup stage order: {raw_order}"))
    if metrics.get("final_accepted") is True:
        artifact_path = str(metrics.get("final_startup_artifact_path") or "")
        if not artifact_path:
            issues.append(RunValidationIssue("WARN", "Final startup was accepted but final_startup_artifact_path is missing."))
        elif not Path(artifact_path).exists():
            issues.append(RunValidationIssue("WARN", f"Final startup artifact path does not exist: {artifact_path}"))
    if metrics.get("automatic_two_segment_pretension_validated") is not False:
        issues.append(RunValidationIssue("WARN", "automatic_two_segment_pretension_validated should be false for this scaffold."))
    if metrics.get("two_segment_control_validated") is not False:
        issues.append(RunValidationIssue("WARN", "two_segment_control_validated should be false for this scaffold."))
    artifact_metadata = run_dir / "two_segment_startup_artifact_metadata.json"
    summary_text = run_dir / "two_segment_startup_summary.txt"
    _check_any_field(issues, "two-segment startup artifact metadata", str(artifact_metadata) if artifact_metadata.exists() else None)
    _check_any_field(issues, "two-segment startup summary text", str(summary_text) if summary_text.exists() else None)


def _check_two_segment_collect_pose_dataset(
    issues: list[RunValidationIssue],
    *,
    run_dir: Path,
    metrics: dict[str, Any],
) -> None:
    _check_any_field(issues, "two-segment dataset command schema", _nested(metrics, "two_segment_command_schema", "schema_version"))
    _check_any_field(issues, "two-segment startup artifact provenance", metrics.get("startup_artifact_provenance"))
    _check_any_field(issues, "two-segment tracking role config", metrics.get("two_segment_tracking_role_config"))
    if "valid_for_two_segment_model_training" not in metrics:
        issues.append(RunValidationIssue("WARN", "Missing expected field: valid_for_two_segment_model_training"))
    pose_summary = metrics.get("pose_label_summary") if isinstance(metrics.get("pose_label_summary"), dict) else {}
    missing_required = list(pose_summary.get("missing_required_roles", []) or [])
    if metrics.get("valid_for_two_segment_model_training") is True and missing_required:
        issues.append(RunValidationIssue("FAIL", f"Run is marked training-valid but required pose roles are missing: {missing_required}"))
    if metrics.get("valid_for_model_training") is not False:
        issues.append(RunValidationIssue("WARN", "MVP two-segment dataset should not set generic valid_for_model_training=true."))
    if metrics.get("valid_for_thesis_repeatability") is not False:
        issues.append(RunValidationIssue("WARN", "MVP two-segment dataset should not set valid_for_thesis_repeatability=true."))
    if not (run_dir / "two_segment_dataset_summary.txt").exists():
        issues.append(RunValidationIssue("WARN", "two_segment_dataset_summary.txt is missing."))
    if not (run_dir / "two_segment_tracking_role_provenance.json").exists():
        issues.append(RunValidationIssue("WARN", "two_segment_tracking_role_provenance.json is missing."))
    # The long-run observability artifacts are emitted by every run since the
    # cycle-10/11/23 improvements. Older runs lack them; warn rather than fail.
    for filename, label in [
        ("long_run_health.json", "long-run health"),
        ("transport_recovery_report.json", "transport recovery report"),
        ("sample_failure_events.jsonl", "sample failure events log"),
    ]:
        if not (run_dir / filename).exists():
            issues.append(RunValidationIssue("WARN", f"{filename} is missing ({label})."))
    if not isinstance(metrics.get("current_load_summary"), dict):
        issues.append(RunValidationIssue("WARN", "Missing expected field: current_load_summary"))
    if not isinstance(metrics.get("tracker_freshness_summary"), dict):
        issues.append(RunValidationIssue("WARN", "Missing expected field: tracker_freshness_summary"))
    samples_path = run_dir / "samples.jsonl"
    if not samples_path.exists():
        issues.append(RunValidationIssue("FAIL", "samples.jsonl is missing."))
        return
    try:
        first_line = next((line for line in samples_path.read_text(encoding="utf-8").splitlines() if line.strip()), "")
        first = json.loads(first_line) if first_line else {}
    except Exception as exc:
        issues.append(RunValidationIssue("FAIL", f"Could not inspect samples.jsonl: {exc}"))
        return
    command = first.get("two_segment_command") if isinstance(first.get("two_segment_command"), dict) else {}
    if not command.get("segments") or not command.get("flat_command_cm"):
        issues.append(RunValidationIssue("WARN", "First sample does not include structured two_segment_command with segment and ordered vector fields."))
    extra = first.get("extra") if isinstance(first.get("extra"), dict) else {}
    if "role_observations" not in extra or "missing_required_pose_roles" not in extra:
        issues.append(RunValidationIssue("WARN", "First sample is missing two-segment pose role observation metadata."))


def _check_two_segment_modeling(
    issues: list[RunValidationIssue],
    *,
    run_dir: Path,
    metrics: dict[str, Any],
) -> None:
    if str(metrics.get("dataset_type") or "") != "two_segment_modeling":
        issues.append(RunValidationIssue("WARN", "Two-segment modeling summary should set dataset_type=two_segment_modeling."))
    if int(metrics.get("accepted_sample_count", 0) or 0) <= 0:
        issues.append(RunValidationIssue("FAIL", "Two-segment modeling run has no accepted modeling samples."))
    models = metrics.get("models") if isinstance(metrics.get("models"), dict) else {}
    if not models:
        issues.append(RunValidationIssue("WARN", "Two-segment modeling run has no model result summaries."))
    if "linear_baseline" not in models:
        issues.append(RunValidationIssue("WARN", "Two-segment modeling run is missing the linear_baseline result."))
    for model_key, model_payload in models.items():
        model_data = model_payload if isinstance(model_payload, dict) else {}
        if model_data.get("status") != "completed":
            continue
        model_dir = run_dir / "models" / str(model_key)
        if not model_dir.exists() or not any(path.is_file() for path in model_dir.iterdir()):
            issues.append(RunValidationIssue("WARN", f"Completed model is missing artifact files: {model_key}"))
    for filename in [
        "predictions.csv",
        "feature_metadata.json",
        "label_metadata.json",
        "train_test_split.json",
        "rejected_samples.jsonl",
        "two_segment_modeling_summary.txt",
        "run_provenance.json",
        "model_status.json",
        "physics_model_parameter_report.json",
        "physics_model_parameter_report.txt",
    ]:
        if not (run_dir / filename).exists():
            issues.append(RunValidationIssue("WARN", f"Expected two-segment modeling artifact is missing: {filename}"))
    warnings = list(metrics.get("data_quality_warnings", []) or [])
    if metrics.get("valid_for_two_segment_model_training") is True and "allow_lower_trust_used_outputs_not_thesis_trusted" in warnings:
        issues.append(RunValidationIssue("FAIL", "Lower-trust modeling output cannot be marked valid_for_two_segment_model_training=true."))


def _infer_experiment_name(run_dir: Path) -> str:
    if run_dir.parent.name and run_dir.parent.name != "experiments":
        return run_dir.parent.name
    parts = run_dir.name.split("_", 2)
    return parts[2] if len(parts) >= 3 else run_dir.name


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
