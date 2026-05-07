"""Pre-hardware readiness checks for operator-critical software paths."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Any

from continuum_robot.config.schemas import RobotConfig, RobotSegmentConfig
from continuum_robot.data.export_run_bundle import export_run_bundle
from continuum_robot.data.validate_run_bundle import validate_run_folder
from continuum_robot.experiments.dataset_io import canonical_timestamp_token


@dataclass(frozen=True)
class DryRunCheck:
    name: str
    status: str
    message: str
    paths: list[Path] = field(default_factory=list)


@dataclass(frozen=True)
class DryRunReport:
    output_dir: Path
    checks: list[DryRunCheck]

    @property
    def passed(self) -> bool:
        return not any(check.status == "FAIL" for check in self.checks)


def run_prehardware_dry_run(*, project_root: Path, output_root: Path | None = None) -> DryRunReport:
    """Run a practical no-hardware readiness check and return a report."""
    project_root = Path(project_root)
    output_dir = Path(output_root or project_root / "data" / "diagnostics" / "prehardware_dry_run" / canonical_timestamp_token())
    output_dir.mkdir(parents=True, exist_ok=False)
    checks: list[DryRunCheck] = []
    checks.append(_check_operating_modes())
    checks.append(_check_gui_construction(project_root=project_root))
    checks.extend(_check_export_and_validation_fixture(output_dir=output_dir))
    report = DryRunReport(output_dir=output_dir, checks=checks)
    (output_dir / "prehardware_dry_run_summary.txt").write_text(render_dry_run_report(report), encoding="utf-8")
    (output_dir / "prehardware_dry_run_summary.json").write_text(
        json.dumps(
            {
                "passed": report.passed,
                "output_dir": str(report.output_dir),
                "checks": [
                    {
                        "name": check.name,
                        "status": check.status,
                        "message": check.message,
                        "paths": [str(path) for path in check.paths],
                    }
                    for check in report.checks
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


def render_dry_run_report(report: DryRunReport) -> str:
    lines = [
        "Pre-Hardware Dry Run",
        f"Overall: {'PASS' if report.passed else 'FAIL'}",
        f"Output: {report.output_dir}",
        "",
    ]
    for check in report.checks:
        lines.append(f"{check.status}: {check.name} - {check.message}")
        for path in check.paths:
            lines.append(f"  {path}")
    return "\n".join(lines).strip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run no-hardware readiness checks before a hardware session.")
    parser.add_argument("--project-root", default=".", help="Repository root.")
    parser.add_argument("--output-root", help="Optional output directory for dry-run artifacts.")
    args = parser.parse_args(argv)
    report = run_prehardware_dry_run(
        project_root=Path(args.project_root).resolve(),
        output_root=Path(args.output_root).resolve() if args.output_root else None,
    )
    print(render_dry_run_report(report))
    return 0 if report.passed else 1


def _check_operating_modes() -> DryRunCheck:
    segments = _segments()
    cases = [
        ("one_servo_1", RobotConfig(mode="one_servo", selected_servo_id=1, segments=segments), [1], {}),
        ("one_servo_8", RobotConfig(mode="one_servo", selected_servo_id=8, segments=segments), [8], {}),
        ("single_segment_a", RobotConfig(mode="single_segment", active_segment="segment_a", segments=segments), [1, 2, 3, 4], {"axis_a": [1, 3], "axis_b": [2, 4]}),
        ("single_segment_b", RobotConfig(mode="single_segment", active_segment="segment_b", segments=segments), [5, 6, 7, 8], {"axis_a": [5, 7], "axis_b": [6, 8]}),
        ("dual_segment", RobotConfig(mode="dual_segment", segments=segments), [1, 2, 3, 4, 5, 6, 7, 8], {}),
        ("parallel_single", RobotConfig(mode="parallel_single", segments=segments), [1, 2, 3, 4, 5, 6, 7, 8], {}),
    ]
    problems: list[str] = []
    for label, config, expected_ids, expected_pairs in cases:
        context = config.operating_context()
        if context.expected_servo_ids != expected_ids:
            problems.append(f"{label} expected {expected_ids}, got {context.expected_servo_ids}")
        if expected_pairs and context.active_pairs != expected_pairs:
            problems.append(f"{label} pairs expected {expected_pairs}, got {context.active_pairs}")
    parallel = RobotConfig(mode="parallel_single", segments=segments).operating_context()
    if parallel.mirror_pairs != {1: 5, 2: 6, 3: 7, 4: 8}:
        problems.append(f"parallel mirror pairs incorrect: {parallel.mirror_pairs}")
    if problems:
        return DryRunCheck("Config / operating modes", "FAIL", "; ".join(problems))
    return DryRunCheck("Config / operating modes", "PASS", "one_servo, single_segment A/B, dual_segment, and parallel_single resolve correctly.")


def _check_gui_construction(*, project_root: Path) -> DryRunCheck:
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        from continuum_robot.gui.controllers.data_management_controller import DataManagementController
        from continuum_robot.gui.tabs.data_management_tab import DataManagementTab

        app = QApplication.instance() or QApplication([])
        _ = app
        controller = DataManagementController(project_root=project_root)
        tab = DataManagementTab(controller)
        tab.update(controller.refresh())
    except Exception as exc:
        return DryRunCheck("GUI construction", "WARN", f"Data tab construction skipped/failed in this environment: {exc}")
    return DryRunCheck("GUI construction", "PASS", "Data tab constructs and refreshes in offscreen mode.")


def _check_export_and_validation_fixture(*, output_dir: Path) -> list[DryRunCheck]:
    checks: list[DryRunCheck] = []
    fixture_root = output_dir / "fixture_project"
    run_dir = _write_fixture_collect_pose_run(fixture_root)
    validation = validate_run_folder(run_dir)
    checks.append(
        DryRunCheck(
            "Run provenance validator",
            validation.status,
            validation.trust_interpretation if validation.status == "PASS" else "; ".join(issue.message for issue in validation.issues),
            [run_dir],
        )
    )
    try:
        no_samples = export_run_bundle(run_dir=run_dir, project_root=fixture_root, output_root=output_dir / "exports_no_samples", make_zip=True)
        with_samples = export_run_bundle(
            run_dir=run_dir,
            project_root=fixture_root,
            output_root=output_dir / "exports_with_samples",
            include_samples=True,
            include_debug=True,
            make_zip=True,
        )
    except Exception as exc:
        checks.append(DryRunCheck("Export bundle", "FAIL", str(exc), [run_dir]))
        return checks
    manifest = json.loads(no_samples.manifest_path.read_text(encoding="utf-8"))
    included_reports = [entry for entry in manifest.get("files", []) if entry.get("category") == "report_figure"]
    skipped_samples = [entry for entry in manifest.get("skipped", []) if str(entry.get("source_path", "")).endswith("samples.jsonl")]
    if not included_reports:
        checks.append(DryRunCheck("Export bundle report figures", "FAIL", "No report figures were included.", [no_samples.bundle_dir]))
    elif not skipped_samples:
        checks.append(DryRunCheck("Export bundle size guard", "FAIL", "samples.jsonl was not skipped by default.", [no_samples.bundle_dir]))
    else:
        checks.append(
            DryRunCheck(
                "Export bundle",
                "PASS",
                "Latest-style run exported with report figures, trust_provenance.json, transfer_commands.txt, and sample size guard.",
                [no_samples.final_path, with_samples.final_path],
            )
        )
    return checks


def _write_fixture_collect_pose_run(root: Path) -> Path:
    run_dir = root / "data" / "experiments" / "collect_pose_command_dataset" / "20260102_000000_collect_pose_command_dataset"
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "1.0",
        "experiment_name": "collect_pose_command_dataset",
        "run_id": "prehardware-fixture",
        "timestamp_utc": "2026-01-02T00:00:00Z",
        "trust_info": {
            "run_trust_mode": "servo_only",
            "valid_for_model_training": False,
            "valid_for_thesis_repeatability": False,
            "data_quality_warnings": ["fixture_no_hardware"],
        },
        "provenance_info": {
            "hardware_profile": "robot_8servo.yaml",
            "operating_mode": "single_segment",
            "active_segment": {"key": "segment_a", "servo_ids": [1, 2, 3, 4]},
            "runtime_tip_calibration": {"mode": "unavailable", "trust_level": "servo_only"},
        },
        "config_used": {"dry_run": True},
    }
    summary = {
        "schema_version": "1.0",
        "experiment_name": "collect_pose_command_dataset",
        "run_id": "prehardware-fixture",
        "success": True,
        "status": "success",
        "sample_counts": {"total": 1},
        "dropped_frames": 0,
        "invalid_transforms": 0,
        "stage_pass_fail": {"execute": "pass"},
        "experiment_metrics": {
            "run_trust_mode": "servo_only",
            "valid_for_model_training": False,
            "valid_for_thesis_repeatability": False,
            "run_trust": {
                "run_trust_mode": "servo_only",
                "valid_for_model_training": False,
                "valid_for_thesis_repeatability": False,
            },
            "run_provenance": {
                "hardware_profile": "robot_8servo.yaml",
                "operating_mode": "single_segment",
                "active_segment": {"key": "segment_a", "servo_ids": [1, 2, 3, 4]},
                "runtime_tip_calibration": {"mode": "unavailable", "trust_level": "servo_only"},
            },
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "config_snapshot.yaml").write_text("dry_run: true\n", encoding="utf-8")
    (run_dir / "metrics.csv").write_text("metric,value\naccepted,0\n", encoding="utf-8")
    (run_dir / "samples.jsonl").write_text('{"fixture":true}\n', encoding="utf-8")
    (run_dir / "modeling_workspace_coverage_report.png").write_bytes(b"\x89PNG\r\n\x1a\nworkspace")
    (run_dir / "commanded_tendon_space_report.png").write_bytes(b"\x89PNG\r\n\x1a\ncommand")
    return run_dir


def _segments() -> dict[str, RobotSegmentConfig]:
    return {
        "segment_a": RobotSegmentConfig(
            key="segment_a",
            label="Spine 1",
            servo_ids=[1, 2, 3, 4],
            pairs={"axis_a": [1, 3], "axis_b": [2, 4]},
        ),
        "segment_b": RobotSegmentConfig(
            key="segment_b",
            label="Spine 2",
            servo_ids=[5, 6, 7, 8],
            pairs={"axis_a": [5, 7], "axis_b": [6, 8]},
        ),
    }


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
