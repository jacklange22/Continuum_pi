"""Validator + export coverage for the two-segment penprobe lookup demo artifacts.

Ensures the demo-only labelling is enforced at the data-plumbing layer and
that the map + demo-run artifacts are exportable via the standard bundle.
"""

from __future__ import annotations

import json
from pathlib import Path


from continuum_robot.data.export_run_bundle import CORE_FILENAMES, export_run_bundle
from continuum_robot.data.validate_run_bundle import validate_run_folder


def _write_demo_run(root: Path) -> Path:
    run_dir = (
        root
        / "data"
        / "experiments"
        / "two_segment_penprobe_lookup_demo"
        / "20260601_120000_two_segment_penprobe_lookup_demo"
    )
    run_dir.mkdir(parents=True)
    metadata = {
        "experiment_name": "two_segment_penprobe_lookup_demo",
        "run_id": run_dir.name,
        "trust_info": {"run_trust_mode": "demo_only"},
        "provenance_info": {"operating_mode": "dual_segment"},
    }
    summary = {
        "experiment_name": "two_segment_penprobe_lookup_demo",
        "run_id": run_dir.name,
        "success": True,
        "status": "success",
        "experiment_metrics": {
            "demo_only": True,
            "not_closed_loop_validated": True,
            "valid_for_model_training": False,
            "valid_for_thesis_repeatability": False,
            "run_trust_mode": "demo_only",
            "map_metadata": {"demo_only_artifact": True},
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "two_segment_penprobe_lookup_demo_summary.txt").write_text("DEMO ONLY\n", encoding="utf-8")
    (run_dir / "demo_trace.csv").write_text("iteration\n0\n", encoding="utf-8")
    (run_dir / "demo_trace.jsonl").write_text('{"iteration": 0}\n', encoding="utf-8")
    (run_dir / "map_used.json").write_text('{"demo_only_artifact": true}', encoding="utf-8")
    return run_dir


def test_validator_accepts_well_formed_demo_run(tmp_path: Path) -> None:
    run_dir = _write_demo_run(tmp_path)
    report = validate_run_folder(run_dir)
    # PASS or WARN are acceptable for the demo; FAIL means the trust labels
    # drifted (e.g. someone tagged it model-training-valid by mistake).
    assert report.status in {"PASS", "WARN"}
    fail_issues = [i for i in report.issues if i.level == "FAIL"]
    assert not fail_issues, [i.message for i in fail_issues]


def test_validator_fails_when_demo_run_claims_model_training_valid(tmp_path: Path) -> None:
    run_dir = _write_demo_run(tmp_path)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    summary["experiment_metrics"]["valid_for_model_training"] = True  # forbidden
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    report = validate_run_folder(run_dir)
    assert report.status == "FAIL"
    fails = [i.message for i in report.issues if i.level == "FAIL"]
    assert any("valid_for_model_training" in m for m in fails)


def test_validator_warns_when_demo_summary_missing(tmp_path: Path) -> None:
    run_dir = _write_demo_run(tmp_path)
    (run_dir / "two_segment_penprobe_lookup_demo_summary.txt").unlink()
    report = validate_run_folder(run_dir)
    warns = [i.message for i in report.issues if i.level == "WARN"]
    assert any("two_segment_penprobe_lookup_demo_summary.txt" in m for m in warns)


def test_export_bundle_includes_demo_artifacts(tmp_path: Path) -> None:
    run_dir = _write_demo_run(tmp_path)
    export = export_run_bundle(
        run_dir=run_dir, output_root=tmp_path / "exports", project_root=tmp_path
    )
    exported = {entry.bundle_path for entry in export.entries}
    assert "two_segment_penprobe_lookup_demo_summary.txt" in exported
    assert "demo_trace.csv" in exported
    assert "demo_trace.jsonl" in exported
    assert "map_used.json" in exported


def test_export_core_filenames_includes_workspace_lookup_artifacts() -> None:
    expected = {
        "two_segment_workspace_lookup_map.json",
        "two_segment_workspace_lookup_points.csv",
        "two_segment_workspace_lookup_summary.txt",
        "two_segment_workspace_lookup_quality.json",
        "two_segment_penprobe_lookup_demo_summary.txt",
        "demo_trace.csv",
        "demo_trace.jsonl",
        "map_used.json",
        "penprobe_lookup_demo_path_report.png",
        "penprobe_lookup_demo_distance_report.png",
        "penprobe_lookup_demo_command_report.png",
    }
    assert expected.issubset(CORE_FILENAMES)


def test_validator_warns_when_target_tool_id_not_0b(tmp_path: Path) -> None:
    """Polished default: target=0B, tip=0A. Drift is a WARN-level issue."""
    run_dir = _write_demo_run(tmp_path)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    summary["experiment_metrics"]["target_tool_id"] = "0C"
    summary["experiment_metrics"]["tip_tool_id"] = "0A"
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    report = validate_run_folder(run_dir)
    warns = [i.message for i in report.issues if i.level == "WARN"]
    assert any("target_tool_id='0C'" in m for m in warns)


def test_validator_fails_when_map_distal_tool_disagrees_with_tip_tool_id(tmp_path: Path) -> None:
    """Map distal != tip tool is a FAIL-level mismatch."""
    run_dir = _write_demo_run(tmp_path)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    summary["experiment_metrics"]["target_tool_id"] = "0B"
    summary["experiment_metrics"]["tip_tool_id"] = "0A"
    summary["experiment_metrics"]["map_distal_tool_id"] = "0C"
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    report = validate_run_folder(run_dir)
    assert report.status == "FAIL"
    fails = [i.message for i in report.issues if i.level == "FAIL"]
    assert any("map_distal_tool_id='0C'" in m and "tip_tool_id='0A'" in m for m in fails)
