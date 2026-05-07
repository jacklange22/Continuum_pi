from __future__ import annotations

import json
from pathlib import Path

from continuum_robot.data.build_thesis_evidence_index import build_thesis_evidence_index
from continuum_robot.data.run_management import (
    archive_run,
    discover_experiment_run_dirs,
    load_run_review,
    summarize_run,
    trash_run,
    write_run_review,
)


def _write_run(root: Path, experiment: str, name: str, *, trust: str = "thesis_trusted") -> Path:
    run_dir = root / "data" / "experiments" / experiment / name
    run_dir.mkdir(parents=True)
    metadata = {
        "experiment_name": experiment,
        "run_id": name,
        "timestamp_utc": "2026-01-02T00:00:00Z",
        "trust_info": {
            "run_trust_mode": trust,
            "valid_for_model_training": experiment == "collect_pose_command_dataset" and trust == "thesis_trusted",
            "valid_for_thesis_repeatability": experiment == "single_segment_repeatability" and trust == "thesis_trusted",
        },
        "provenance_info": {
            "hardware_profile": "robot_8servo.yaml",
            "operating_mode": "single_segment",
            "active_segment": {"key": "segment_a", "label": "Spine 1", "servo_ids": [1, 2, 3, 4]},
            "runtime_tip_calibration": {"mode": "coil_as_tip", "trust_status": "thesis_trusted"},
        },
    }
    summary = {
        "experiment_name": experiment,
        "run_id": name,
        "success": True,
        "status": "success",
        "experiment_metrics": {
            "run_trust_mode": trust,
            "valid_for_model_training": metadata["trust_info"]["valid_for_model_training"],
            "valid_for_thesis_repeatability": metadata["trust_info"]["valid_for_thesis_repeatability"],
            "run_provenance": metadata["provenance_info"],
            "rmse_mm": 0.42,
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if experiment == "collect_pose_command_dataset":
        (run_dir / "modeling_workspace_coverage_report.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (run_dir / "commanded_tendon_space_report.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    elif experiment == "single_segment_repeatability":
        (run_dir / "repeatability_clusters_report.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (run_dir / "repeatability_error_by_target_report.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (run_dir / "metrics.csv").write_text("metric,value\nrmse,0.42\n", encoding="utf-8")
    return run_dir


def test_discover_and_summarize_runs_with_trust_and_segment(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "collect_pose_command_dataset", "20260102_000000_collect_pose_command_dataset")

    runs = discover_experiment_run_dirs(tmp_path, experiment_name="collect_pose_command_dataset")
    summary = summarize_run(run_dir)

    assert runs == [run_dir]
    assert summary.validation_status == "PASS"
    assert summary.run_trust_mode == "thesis_trusted"
    assert summary.valid_for_model_training is True
    assert summary.operating_mode == "single_segment"
    assert "segment_a" in summary.active_segment
    assert "modeling_workspace_coverage_report.png" in summary.report_figures


def test_write_run_review_marks_candidate_sidecar(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "single_segment_repeatability", "20260102_000000_single_segment_repeatability")

    review = write_run_review(run_dir, status="thesis_candidate", notes="good repeatability")
    loaded = load_run_review(run_dir)

    assert (run_dir / "run_review.json").exists()
    assert review.review_status == "thesis_candidate"
    assert loaded.include_in_evidence_index is True
    assert loaded.notes == "good repeatability"


def test_archive_and_trash_move_runs_without_permanent_delete(tmp_path: Path) -> None:
    archive_source = _write_run(tmp_path, "collect_pose_command_dataset", "20260102_000000_collect_pose_command_dataset")
    trash_source = _write_run(tmp_path, "single_segment_repeatability", "20260103_000000_single_segment_repeatability")

    archive_result = archive_run(archive_source, project_root=tmp_path)
    trash_result = trash_run(trash_source, project_root=tmp_path)

    assert archive_source.exists() is False
    assert trash_source.exists() is False
    assert archive_result.destination_path.exists()
    assert trash_result.destination_path.exists()
    assert "data/experiments_archived/collect_pose_command_dataset" in str(archive_result.destination_path)
    assert "data/trash/single_segment_repeatability" in str(trash_result.destination_path)
    assert load_run_review(archive_result.destination_path).review_status == "archived"
    assert load_run_review(trash_result.destination_path).review_status == "garbage"


def test_trash_blocks_protected_review_unless_forced(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "single_segment_repeatability", "20260102_000000_single_segment_repeatability")
    write_run_review(run_dir, status="thesis_candidate")

    try:
        trash_run(run_dir, project_root=tmp_path)
    except ValueError as exc:
        assert "thesis_candidate" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("protected run should have blocked trash")

    result = trash_run(run_dir, project_root=tmp_path, force=True)
    assert result.destination_path.exists()


def test_build_thesis_evidence_index_includes_reviewed_candidates(tmp_path: Path) -> None:
    candidate = _write_run(tmp_path, "single_segment_repeatability", "20260102_000000_single_segment_repeatability")
    _write_run(tmp_path, "collect_pose_command_dataset", "20260103_000000_collect_pose_command_dataset", trust="servo_only")
    write_run_review(candidate, status="thesis_candidate", notes="candidate run")

    output_dir = build_thesis_evidence_index(project_root=tmp_path)

    index_json = json.loads((output_dir / "thesis_evidence_index.json").read_text(encoding="utf-8"))
    index_md = (output_dir / "thesis_evidence_index.md").read_text(encoding="utf-8")
    assert index_json["run_count_scanned"] == 2
    assert "single_segment_repeatability" in index_json["experiments"]
    assert "thesis_candidate" in index_md
    assert "samples.jsonl" not in index_md

