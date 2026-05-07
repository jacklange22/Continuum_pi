from __future__ import annotations

import json
from pathlib import Path

from continuum_robot.data.validate_run_bundle import render_validation_report, validate_run_folder


def _write_valid_collect_pose_run(root: Path) -> Path:
    run_dir = root / "data" / "experiments" / "collect_pose_command_dataset" / "20260102_000000_collect_pose_command_dataset"
    run_dir.mkdir(parents=True)
    metadata = {
        "experiment_name": "collect_pose_command_dataset",
        "trust_info": {
            "run_trust_mode": "thesis_trusted",
            "valid_for_model_training": True,
            "valid_for_thesis_repeatability": False,
        },
        "provenance_info": {
            "hardware_profile": "robot_8servo.yaml",
            "operating_mode": "single_segment",
            "active_segment": {"key": "segment_a"},
            "runtime_tip_calibration": {"mode": "coil_as_tip"},
        },
    }
    summary = {
        "experiment_name": "collect_pose_command_dataset",
        "success": True,
        "status": "success",
        "experiment_metrics": {
            "run_trust_mode": "thesis_trusted",
            "valid_for_model_training": True,
            "valid_for_thesis_repeatability": False,
            "run_provenance": {
                "hardware_profile": "robot_8servo.yaml",
                "operating_mode": "single_segment",
                "active_segment": {"key": "segment_a"},
                "runtime_tip_calibration": {"mode": "coil_as_tip"},
            },
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "modeling_workspace_coverage_report.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (run_dir / "commanded_tendon_space_report.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return run_dir


def test_validate_run_folder_passes_complete_run(tmp_path: Path) -> None:
    run_dir = _write_valid_collect_pose_run(tmp_path)

    report = validate_run_folder(run_dir)

    assert report.status == "PASS"
    assert report.experiment_name == "collect_pose_command_dataset"
    assert "valid_for_model_training=True" in report.trust_interpretation


def test_validate_run_folder_warns_for_missing_trust_and_report_figure(tmp_path: Path) -> None:
    run_dir = tmp_path / "data" / "experiments" / "pretension_validation" / "20260102_000000_pretension_validation"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps({"experiment_name": "pretension_validation", "success": True, "status": "success"}), encoding="utf-8")

    report = validate_run_folder(run_dir)
    text = render_validation_report(report)

    assert report.status == "WARN"
    assert "run_trust_mode" in text
    assert "pretension_tip_xy_path_report.png" in text


def test_validate_run_folder_fails_without_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "data" / "experiments" / "registration_validation" / "20260102_000000_registration_validation"
    run_dir.mkdir(parents=True)

    report = validate_run_folder(run_dir)

    assert report.status == "FAIL"
