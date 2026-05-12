from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_data_for_git.py"
SPEC = importlib.util.spec_from_file_location("check_data_for_git", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
check_data_for_git = importlib.util.module_from_spec(SPEC)
sys.modules["check_data_for_git"] = check_data_for_git
SPEC.loader.exec_module(check_data_for_git)


def _write_run(root: Path, *, review_status: str | None = None, mock: bool = False) -> Path:
    run_dir = root / "data" / "experiments" / "single_segment_repeatability" / "20260102_000000_single_segment_repeatability"
    run_dir.mkdir(parents=True)
    metadata = {
        "experiment_name": "single_segment_repeatability",
        "run_id": "repeatability-1",
        "trust_info": {
            "run_trust_mode": "mock" if mock else "thesis_trusted",
            "valid_for_model_training": False,
            "valid_for_thesis_repeatability": False if mock else True,
        },
        "provenance_info": {
            "mock_mode": bool(mock),
            "hardware_profile": "robot_8servo.yaml",
            "operating_mode": "single_segment",
            "active_segment": {"key": "segment_a"},
            "runtime_tip_calibration": {"mode": "coil_as_tip"},
        },
    }
    summary = {
        "experiment_name": "single_segment_repeatability",
        "run_id": "repeatability-1",
        "success": True,
        "status": "success",
        "experiment_metrics": {
            "run_trust_mode": "mock" if mock else "thesis_trusted",
            "valid_for_model_training": False,
            "valid_for_thesis_repeatability": False if mock else True,
            "run_provenance": metadata["provenance_info"],
            "mock_mode": bool(mock),
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "repeatability_clusters_report.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (run_dir / "repeatability_error_by_target_report.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    if review_status:
        (run_dir / "run_review.json").write_text(
            json.dumps({"review_status": review_status, "include_in_evidence_index": review_status == "thesis_candidate"}),
            encoding="utf-8",
        )
    return run_dir


def test_data_hygiene_handles_empty_data_folder(tmp_path: Path) -> None:
    report = check_data_for_git.run_data_hygiene_check(project_root=tmp_path)

    assert report.status == "PASS"
    assert report.experiment_run_count == 0


def test_data_hygiene_reports_run_review_status(tmp_path: Path) -> None:
    _write_run(tmp_path, review_status="thesis_candidate")

    report = check_data_for_git.run_data_hygiene_check(project_root=tmp_path)
    text = check_data_for_git.render_report(report)

    assert report.run_status_counts["thesis_candidate"] == 1
    assert "thesis_candidate=1" in text


def test_data_hygiene_warns_for_missing_run_review(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)

    report = check_data_for_git.run_data_hygiene_check(project_root=tmp_path)

    assert report.status == "WARN"
    assert any(str(run_dir.name) in finding.path and "run_review" in finding.message for finding in report.findings)


def test_data_hygiene_detects_export_bundle_and_large_files(tmp_path: Path) -> None:
    export_path = tmp_path / "data" / "exports" / "bundle.zip"
    export_path.parent.mkdir(parents=True)
    export_path.write_bytes(b"x" * 16)
    large_path = tmp_path / "data" / "experiments" / "demo" / "big.bin"
    large_path.parent.mkdir(parents=True)
    large_path.write_bytes(b"x" * 32)

    report = check_data_for_git.run_data_hygiene_check(project_root=tmp_path, warn_bytes=12, fail_bytes=24)

    assert report.status == "FAIL"
    assert any("data/exports/bundle.zip" in finding.path for finding in report.findings)
    assert any("big.bin" in finding.path and finding.level == "FAIL" for finding in report.findings)


def test_data_hygiene_detects_mock_run_in_real_experiment_root(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, review_status="debug", mock=True)

    report = check_data_for_git.run_data_hygiene_check(project_root=tmp_path)

    assert report.status == "FAIL"
    assert any(str(run_dir.name) in finding.path and "Mock/debug run is in data/experiments" in finding.message for finding in report.findings)


def test_data_hygiene_detects_mock_calibration_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "data" / "mock_calibration" / "latest_mock_neutral_setpoints.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps({"robot": {"mock_mode": True, "calibration_trust": "mock", "valid_for_hardware_startup": False}}),
        encoding="utf-8",
    )

    report = check_data_for_git.run_data_hygiene_check(project_root=tmp_path)

    assert report.status == "FAIL"
    assert any("mock_calibration" in finding.path and "Mock calibration" in finding.message for finding in report.findings)


def test_gitignore_does_not_hide_experiment_or_archive_runs() -> None:
    gitignore = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8")

    assert "data/experiments/**" not in gitignore
    assert "data/experiments_archived/**" not in gitignore
    assert "data/trash/**" in gitignore
    assert "data/exports/**" in gitignore
    assert "data/mock_experiments/**" in gitignore
    assert "data/mock_calibration/**" in gitignore
    assert "data/diagnostics/**" in gitignore
