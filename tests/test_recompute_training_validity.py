from __future__ import annotations

import json
from pathlib import Path

from continuum_robot.data.recompute_training_validity import main as recompute_main


def _sample_payload(*, accepted: bool, excluded: bool, missing_servo_position: bool = False) -> dict:
    servo_value = None if missing_servo_position else 1234
    return {
        "monotonic_time_s": 0.0,
        "wall_time_utc": "2026-01-01T00:00:00Z",
        "phase": "workspace_capture",
        "step_index": 0,
        "sample_index": 0,
        "commanded_motor_values": {},
        "commanded_cable_deltas_cm": [0.1, 0.0, -0.1, 0.0],
        "tracker_frame_id": 1,
        "tool_ids_seen": ["0A"],
        "transform_validity": {"T_robot_tip": "valid"},
        "pose_in_tracker_frame": {},
        "pose_in_robot_frame": {
            "tip": {
                "translation_mm": [1.0, 2.0, 3.0],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "tangent_xyz": [0.0, 0.0, 1.0],
            }
        },
        "freshness_s": 0.0,
        "latency_s": 0.0,
        "status_flags": ["capture_accepted"] if accepted else ["capture_rejected"],
        "backend_health": {},
        "extra": {
            "capture_accepted": accepted,
            "modeling_export_exclude": excluded,
            "resolved_cable_command_cm": [0.1, 0.0, -0.1, 0.0],
            "final_goal_ticks_by_servo": {"5": 2100, "6": 2200, "7": 2300, "8": 2400},
            "servo_feedback_at_capture": {
                "5": {"present_position_ticks": 2101},
                "6": {"present_position_ticks": 2201},
                "7": {"present_position_ticks": 2301},
                "8": {"present_position_ticks": servo_value},
            },
        },
    }


def _write_run_fixture(tmp_path: Path, *, missing_servo_position: bool = False) -> Path:
    run_dir = tmp_path / "data" / "experiments" / "collect_pose_command_dataset" / "20260101_000000_collect_pose_command_dataset"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "1.0",
        "experiment_name": "collect_pose_command_dataset",
        "run_id": "run123",
        "success": True,
        "status": "success",
        "sample_counts": {"total": 2},
        "dropped_frames": 0,
        "invalid_transforms": 0,
        "stage_pass_fail": {"setup": "passed", "precheck": "passed", "execute": "passed", "finalize": "passed"},
        "experiment_metrics": {
            "run_trust_mode": "thesis_trusted",
            "dry_run": False,
            "mock_mode": False,
            "tracker_connected": True,
            "parallel_single_demo": False,
            "accepted_sample_count": 1,
            "rejected_sample_count": 1,
            "dropped_post_motion_telemetry_samples": 1,
            "dropped_pre_motion_telemetry_samples": 0,
            "unrecovered_packet_error_count": 1,
            "target_valid_sample_count": 1,
            "legacy_export_enabled": True,
            "runtime_tip_mode_used": "coil_as_tip",
            "runtime_tip_trust_level": "thesis_trusted",
            "runtime_tip_policy": {"allowed_for_workflow": True},
            "run_provenance": {
                "pretension_artifact": {"accepted": True, "usable": True},
            },
        },
    }
    metadata = {
        "schema_version": "1.0",
        "experiment_name": "collect_pose_command_dataset",
        "run_id": "run123",
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "trust_info": {"run_trust_mode": "thesis_trusted", "valid_for_model_training": False},
        "config_used": {"max_current_warning_ma": 800},
    }
    quality = {"schema_version": "collect_pose_dataset_quality_v1"}
    samples = [
        _sample_payload(accepted=True, excluded=False, missing_servo_position=missing_servo_position),
        _sample_payload(accepted=False, excluded=True, missing_servo_position=False),
    ]
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (run_dir / "dataset_quality_summary.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    (run_dir / "dataset_quality_summary.txt").write_text("placeholder\n", encoding="utf-8")
    (run_dir / "modeling_dataset_summary.txt").write_text("placeholder\n", encoding="utf-8")
    (run_dir / "samples.jsonl").write_text(
        "\n".join(json.dumps(sample, separators=(",", ":")) for sample in samples) + "\n",
        encoding="utf-8",
    )
    (run_dir / "modeling_dataset_legacy_compat.dat").write_text(
        "DATE: 2026-1-1\nTIME: 00-00-00\nNUM_CABLES: 4\nnum_coils: 1\nNUM_MEASUREMENTS: 1\n---\n,0,0\n",
        encoding="utf-8",
    )
    return run_dir


def test_recompute_training_validity_marks_warning_valid_with_quarantined_drops(tmp_path: Path) -> None:
    run_dir = _write_run_fixture(tmp_path, missing_servo_position=False)
    rc = recompute_main([str(run_dir), "--apply"])
    assert rc == 0
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    metrics = dict(summary.get("experiment_metrics", {}) or {})
    assert metrics["valid_for_model_training"] is True
    assert metrics["not_model_training_ready"] is False
    assert metrics["model_training_validity_status"] == "warning_valid"
    assert metrics["model_training_validity_reason"] == "warnings_present_but_training_ready"
    assert bool(metrics["dropped_samples_excluded_from_training"]) is True
    quality = json.loads((run_dir / "dataset_quality_summary.json").read_text(encoding="utf-8"))
    assert quality["model_training_validity_status"] == "warning_valid"
    assert quality["trainability_status"]["valid_for_model_training"] is True
    export_rows = [json.loads(line) for line in (run_dir / "modeling_dataset_export.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(export_rows) == 1
    assert export_rows[0]["phase"] == "workspace_capture"
    assert (run_dir / "dataset_quality_summary.txt").read_text(encoding="utf-8").find("Model training validity status") >= 0
    assert (run_dir / "modeling_dataset_summary.txt").read_text(encoding="utf-8").find("Trainability:") >= 0


def test_recompute_training_validity_marks_invalid_for_incomplete_accepted_rows(tmp_path: Path) -> None:
    run_dir = _write_run_fixture(tmp_path, missing_servo_position=True)
    rc = recompute_main([str(run_dir), "--apply"])
    assert rc == 0
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    metrics = dict(summary.get("experiment_metrics", {}) or {})
    assert metrics["valid_for_model_training"] is False
    assert metrics["not_model_training_ready"] is True
    assert metrics["model_training_validity_status"] == "invalid"
    checks = dict(metrics.get("model_training_validity_checks", {}) or {})
    assert checks.get("accepted_rows_complete") is False
    export_rows = [line for line in (run_dir / "modeling_dataset_export.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert export_rows == []
