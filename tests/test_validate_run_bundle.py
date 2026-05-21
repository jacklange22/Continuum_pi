from __future__ import annotations

import json
from pathlib import Path

from continuum_robot.data.validate_run_bundle import (
    main as validate_main,
    render_validation_report,
    validate_run_folder,
    validation_report_to_dict,
)


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
    (run_dir / "thesis_01_workspace_coverage_3d.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (run_dir / "thesis_02_command_and_workspace_2d.png").write_bytes(b"\x89PNG\r\n\x1a\n")
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
    assert "pretension_telemetry_timeline_report.png" in text


def test_validate_run_folder_fails_without_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "data" / "experiments" / "registration_validation" / "20260102_000000_registration_validation"
    run_dir.mkdir(parents=True)

    report = validate_run_folder(run_dir)

    assert report.status == "FAIL"


def test_validation_report_to_dict_emits_machine_readable_view(tmp_path: Path) -> None:
    """The JSON view exposes status, issue counts, and per-issue details."""
    run_dir = _write_valid_collect_pose_run(tmp_path)

    report = validate_run_folder(run_dir)
    payload = validation_report_to_dict(report)

    assert payload["schema_version"] == "run_validation_report_v1"
    assert payload["status"] == report.status
    assert payload["experiment_name"] == "collect_pose_command_dataset"
    assert isinstance(payload["issues"], list)
    assert (
        payload["fail_count"]
        + payload["warn_count"]
        + payload["info_count"]
    ) == len(report.issues) - sum(
        1 for issue in report.issues if issue.level not in {"FAIL", "WARN", "INFO"}
    )


def test_validate_main_emits_json_when_flag_set(tmp_path: Path, capsys) -> None:
    """CLI --json prints a parseable JSON document and returns the same exit code as the text path."""
    run_dir = _write_valid_collect_pose_run(tmp_path)

    rc = validate_main([str(run_dir), "--json"])

    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["schema_version"] == "run_validation_report_v1"
    assert payload["status"] == "PASS"


def test_validate_main_returns_nonzero_on_failure_even_in_json_mode(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "data" / "experiments" / "registration_validation" / "20260102_000000_registration_validation"
    run_dir.mkdir(parents=True)

    rc = validate_main([str(run_dir), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 1
    assert payload["status"] == "FAIL"


def _write_registration_sampling_study_run(
    root: Path,
    *,
    name: str = "20260514_220000_registration_sampling_study",
    valid_for_model_training: bool = False,
    valid_for_thesis_repeatability: bool = False,
    valid_for_registration_protocol_recommendation: bool = True,
    include_all_artifacts: bool = True,
) -> Path:
    run_dir = root / "data" / "experiments" / "registration_sampling_study" / name
    run_dir.mkdir(parents=True)
    metadata = {
        "experiment_name": "registration_sampling_study",
        "trust_info": {
            "run_trust_mode": "thesis_trusted",
            "valid_for_model_training": valid_for_model_training,
            "valid_for_thesis_repeatability": valid_for_thesis_repeatability,
        },
        "provenance_info": {
            "operating_mode": "single_segment",
            "hardware_profile": "robot_8servo.yaml",
        },
    }
    summary = {
        "experiment_name": "registration_sampling_study",
        "success": True,
        "status": "success",
        "experiment_metrics": {
            "run_trust_mode": "thesis_trusted",
            "valid_for_model_training": valid_for_model_training,
            "valid_for_thesis_repeatability": valid_for_thesis_repeatability,
            "valid_for_registration_protocol_recommendation": valid_for_registration_protocol_recommendation,
            "recommended_protocol": {
                "recommended_subset_size": 12,
                "recommended_samples_per_point": 20,
                "recommended_averaging_method": "mean",
                "rationale": "test",
            },
            "candidate_registration_fre_mm": 0.42,
            "captured_label_count": 12,
            "captured_sample_count_total": 240,
            "run_provenance": {
                "operating_mode": "single_segment",
                "hardware_profile": "robot_8servo.yaml",
            },
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if include_all_artifacts:
        for filename in (
            "registration_candidate.json",
            "registration_sampling_study_summary.txt",
            "metrics.csv",
            "point_centers.csv",
            "subset_results.csv",
            "leave_one_out_results.csv",
            "samples_per_point_results.csv",
            "raw_point_samples.jsonl",
        ):
            (run_dir / filename).write_text("placeholder", encoding="utf-8")
        for filename in (
            "registration_point_spread_report.png",
            "registration_subset_rms_report.png",
            "registration_samples_per_point_report.png",
            "registration_transform_consistency_report.png",
        ):
            (run_dir / filename).write_bytes(b"\x89PNG\r\n\x1a\n")
    return run_dir


def test_validate_run_folder_passes_registration_sampling_study(tmp_path: Path) -> None:
    run_dir = _write_registration_sampling_study_run(tmp_path)

    report = validate_run_folder(run_dir)

    assert report.status == "PASS", render_validation_report(report)
    assert report.experiment_name == "registration_sampling_study"


def test_validate_run_folder_warns_when_registration_sampling_study_artifacts_missing(tmp_path: Path) -> None:
    run_dir = _write_registration_sampling_study_run(tmp_path, include_all_artifacts=False)

    report = validate_run_folder(run_dir)
    text = render_validation_report(report)

    assert report.status == "WARN"
    assert "registration_candidate.json" in text
    assert "registration_point_spread_report.png" in text


def test_validate_run_folder_warns_when_registration_sampling_study_overclaims(tmp_path: Path) -> None:
    run_dir = _write_registration_sampling_study_run(
        tmp_path,
        valid_for_model_training=True,
        valid_for_thesis_repeatability=True,
    )

    report = validate_run_folder(run_dir)
    text = render_validation_report(report)

    assert report.status == "WARN"
    assert "must not set valid_for_model_training=true" in text
    assert "must not set valid_for_thesis_repeatability=true" in text
