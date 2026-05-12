from __future__ import annotations

import json
from pathlib import Path

from continuum_robot.data.export_run_bundle import build_transfer_commands, export_run_bundle, find_latest_run


def _write_run(root: Path, experiment: str, name: str, *, sample_bytes: int = 0) -> Path:
    run_dir = root / "data" / "experiments" / experiment / name
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": name,
                "experiment_name": experiment,
                "trust_info": {
                    "run_trust_mode": "thesis_trusted",
                    "valid_for_model_training": True,
                    "valid_for_thesis_repeatability": False,
                },
                "provenance_info": {
                    "hardware_profile": "robot_8servo.yaml",
                    "operating_mode": "single_segment",
                    "selected_segment_readiness": {
                        "active_segment_key": "segment_b",
                        "expected_servo_ids": [5, 6, 7, 8],
                    },
                    "startup_pretension_artifact": {
                        "source_type": "manual_startup",
                        "accepted": True,
                    },
                    "servo_sign_mapping_check": {
                        "path": "data/calibration/servo_mapping_checks/example.json",
                        "confirmed": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "success",
                "experiment_metrics": {
                    "run_trust_mode": "thesis_trusted",
                    "valid_for_model_training": True,
                    "valid_for_thesis_repeatability": False,
                    "run_provenance": {"operating_mode": "single_segment"},
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "config_snapshot.yaml").write_text("runtime:\n  mock_mode: true\n", encoding="utf-8")
    (run_dir / "metrics.csv").write_text("metric,value\nrmse,1.0\n", encoding="utf-8")
    (run_dir / "run_review.json").write_text(
        json.dumps({"review_status": "debug", "include_in_evidence_index": False}),
        encoding="utf-8",
    )
    (run_dir / "modeling_workspace_coverage_report.png").write_bytes(b"\x89PNG\r\n\x1a\nreport")
    (run_dir / "dashboard.png").write_bytes(b"\x89PNG\r\n\x1a\ndashboard")
    debug_dir = run_dir / "debug"
    debug_dir.mkdir()
    (debug_dir / "debug_manifest.json").write_text("{}", encoding="utf-8")
    if sample_bytes:
        (run_dir / "samples.jsonl").write_bytes(b"x" * int(sample_bytes))
    return run_dir


def test_find_latest_run_uses_canonical_experiment_root(tmp_path: Path) -> None:
    _write_run(tmp_path, "single_segment_repeatability", "20260101_000000_single_segment_repeatability")
    latest = _write_run(tmp_path, "single_segment_repeatability", "20260102_000000_single_segment_repeatability")

    assert find_latest_run(project_root=tmp_path, experiment_name="single_segment_repeatability") == latest


def test_export_run_bundle_writes_manifest_report_figures_and_trust_block(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "collect_pose_command_dataset", "20260102_000000_collect_pose_command_dataset")

    result = export_run_bundle(run_dir=run_dir, project_root=tmp_path)

    assert result.bundle_dir.exists()
    assert (result.bundle_dir / "manifest.json").exists()
    assert (result.bundle_dir / "trust_provenance.json").exists()
    assert (result.bundle_dir / "README.txt").exists()
    assert (result.bundle_dir / "run_review.json").exists()
    assert (result.bundle_dir / "modeling_workspace_coverage_report.png").exists()
    assert not (result.bundle_dir / "dashboard.png").exists()
    manifest = json.loads((result.bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_name"] == "collect_pose_command_dataset"
    assert any(file["category"] == "report_figure" for file in manifest["files"])
    trust = json.loads((result.bundle_dir / "trust_provenance.json").read_text(encoding="utf-8"))
    assert trust["run_trust_mode"] == "thesis_trusted"
    assert trust["valid_for_model_training"] is True
    assert trust["selected_segment_readiness"]["active_segment_key"] == "segment_b"
    assert trust["startup_pretension_artifact"]["source_type"] == "manual_startup"
    assert trust["servo_sign_mapping_check"]["confirmed"] is True


def test_export_run_bundle_skips_large_samples_unless_enabled(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "pretension_validation", "20260102_000000_pretension_validation", sample_bytes=128)

    result = export_run_bundle(run_dir=run_dir, project_root=tmp_path, max_sample_bytes=64)

    assert not (result.bundle_dir / "samples.jsonl").exists()
    manifest = json.loads((result.bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert any(item["source_path"].endswith("samples.jsonl") for item in manifest["skipped"])


def test_export_run_bundle_can_include_samples_and_debug_outputs(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "pretension_validation", "20260102_000000_pretension_validation", sample_bytes=16)

    result = export_run_bundle(
        run_dir=run_dir,
        project_root=tmp_path,
        include_samples=True,
        include_debug=True,
        max_sample_bytes=64,
    )

    assert (result.bundle_dir / "samples.jsonl").exists()
    assert (result.bundle_dir / "debug" / "debug_manifest.json").exists()


def test_ai_debug_profile_includes_small_samples_without_include_samples_flag(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "pretension_validation", "20260102_000000_pretension_validation", sample_bytes=16)

    result = export_run_bundle(
        run_dir=run_dir,
        project_root=tmp_path,
        profile="ai_debug",
        max_sample_bytes=64,
    )

    assert (result.bundle_dir / "samples.jsonl").exists()
    assert (result.bundle_dir / "debug" / "debug_manifest.json").exists()
    manifest = json.loads((result.bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["profile"] == "ai_debug"


def test_export_run_bundle_writes_transfer_commands(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "single_segment_repeatability", "20260102_000000_single_segment_repeatability")

    result = export_run_bundle(run_dir=run_dir, project_root=tmp_path, make_zip=True)

    transfer_path = result.bundle_dir / "transfer_commands.txt"
    assert transfer_path.exists()
    text = transfer_path.read_text(encoding="utf-8")
    assert "rsync -av" in text
    assert str(result.zip_path) in text
    assert "<pi-host>" in build_transfer_commands(Path("/tmp/run_bundle.zip"))
