from __future__ import annotations

import json
from pathlib import Path

import pytest

from continuum_robot.modeling import analysis as analysis_module
from continuum_robot.modeling.analysis import (
    ModelingEvaluationConfig,
    ModelingGeometryConfig,
    build_artifact_summary_pairs,
    build_dataset_summary_pairs,
    evaluate_models,
    load_trained_artifact_details,
)
from continuum_robot.modeling.ann_training import load_modeling_dataset_summary


def _write_modeling_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "data" / "experiments" / "collect_pose_command_dataset" / "20260419_120000_collect_pose_command_dataset"
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "1.0",
        "experiment_name": "collect_pose_command_dataset",
        "run_id": "dataset-run-123",
        "timestamp_utc": "2026-04-19T12:00:00+00:00",
        "git_commit": None,
        "backend_info": {},
        "registration_info": {},
        "config_used": {"dataset_mode": "repeatability_linked"},
        "operator_notes": "",
    }
    summary = {
        "schema_version": "1.0",
        "experiment_name": "collect_pose_command_dataset",
        "run_id": "dataset-run-123",
        "success": True,
        "sample_counts": {"total": 4},
        "dropped_frames": 0,
        "invalid_transforms": 0,
        "stage_pass_fail": {},
        "status": "success",
        "experiment_metrics": {
            "dataset_mode": "repeatability_linked",
            "dataset_mode_summary": "Repeated trusted startup blocks for cross-revision modeling comparisons.",
            "accepted_sample_count": 4,
            "rejected_sample_count": 0,
            "accepted_capture_rate": 1.0,
            "run_provenance": {
                "runtime_tip_calibration": {"mode": "latest_accepted", "trust_level": "trusted"},
                "pretension_artifact": {"active_source_type": "accepted_artifact", "status": "ready"},
            },
        },
        "warning_messages": [],
        "error_messages": [],
    }
    rows = []
    commands = [
        [0.0, 0.0, 0.0, 0.0],
        [0.2, 0.1, 0.0, 0.0],
        [0.1, 0.25, 0.0, 0.0],
        [0.15, 0.15, 0.0, 0.0],
    ]
    phases = ["startup_block_a", "startup_block_a", "startup_block_b", "startup_block_b"]
    geometry = ModelingGeometryConfig()
    for index, (command, phase) in enumerate(zip(commands, phases)):
        webster = analysis_module._mike_forward_webster(command=command, geometry=geometry)
        transform = analysis_module._calculate_transform(webster)
        rows.append(
            {
                "sequence_index": index,
                "step_index": index,
                "sample_index": index,
                "phase": phase,
                "accepted": True,
                "resolved_cable_command_cm": command,
                "tip_position_xyz_mm": transform[:3, 3].tolist(),
                "tip_tangent_xyz": transform[:3, 2].tolist(),
                "previous_pair_command_cm": ([0.0, 0.0] if index > 0 else []),
            }
        )
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "modeling_dataset_summary.txt").write_text("summary\n", encoding="utf-8")
    (run_dir / "modeling_dataset_legacy_compat.dat").write_text("legacy\n", encoding="utf-8")
    with (run_dir / "modeling_dataset_export.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return run_dir


def _write_artifact(tmp_path: Path, dataset_path: Path) -> Path:
    artifact_dir = tmp_path / "data" / "models" / "ann" / "20260419_120100_legacy_ann"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at_utc": "2026-04-19T12:01:00+00:00",
        "status": "completed",
        "dataset": {
            "path": str(dataset_path),
            "run_name": dataset_path.name,
        },
        "backend": {"selected_backend": "cpu", "torch_version": "2.test"},
        "model": {"input_dim": 4, "output_dim": 6, "hidden_layers": [32, 32], "dtype": "float64"},
        "training": {
            "epochs_completed": 8,
            "batch_size": 64,
            "learning_rate": 0.001,
            "best_validation_loss": 0.123,
        },
        "files": {"model_path": str(artifact_dir / "model.pt")},
    }
    training_config = {"epochs": 8, "batch_size": 64, "learning_rate": 0.001}
    split_manifest = {"test_indices": [2, 3], "train_indices": [0, 1], "validation_indices": [], "group_ids": [0, 1, 2, 3]}
    (artifact_dir / "training_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (artifact_dir / "training_config.json").write_text(json.dumps(training_config), encoding="utf-8")
    (artifact_dir / "split_manifest.json").write_text(json.dumps(split_manifest), encoding="utf-8")
    (artifact_dir / "loss_history.csv").write_text(
        "epoch,train_loss,validation_loss,elapsed_s\n1,1.0,1.1,0.1\n2,0.8,0.9,0.2\n",
        encoding="utf-8",
    )
    (artifact_dir / "model.pt").write_bytes(b"fake")
    return artifact_dir


def test_modeling_dataset_and_artifact_browser_pairs(tmp_path: Path) -> None:
    run_dir = _write_modeling_run(tmp_path)
    artifact_dir = _write_artifact(tmp_path, run_dir)

    dataset_summary = load_modeling_dataset_summary(run_dir)
    artifact_details = load_trained_artifact_details(artifact_dir)

    dataset_pairs = dict(build_dataset_summary_pairs(dataset_summary))
    artifact_pairs = dict(build_artifact_summary_pairs(artifact_details))

    assert dataset_pairs["Dataset Mode"] == "repeatability_linked"
    assert dataset_pairs["Sequential Context"] == "yes"
    assert artifact_pairs["Linked Dataset"] == run_dir.name
    assert "best val" in artifact_pairs["Loss Summary"]


def test_evaluate_models_writes_outputs_and_comparison_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = _write_modeling_run(tmp_path)
    artifact_dir = _write_artifact(tmp_path, run_dir)

    def _fake_ann(*, inputs, truths, phases, artifact_details):
        return analysis_module._complete_model_evaluation(
            model_key="ann",
            label="ANN",
            predictions=truths.copy(),
            truths=truths,
            phases=phases,
        )

    monkeypatch.setattr(analysis_module, "_evaluate_ann", _fake_ann)

    result = evaluate_models(
        project_root=tmp_path,
        dataset_path=run_dir,
        artifact_path=artifact_dir,
        config=ModelingEvaluationConfig(
            include_mike=True,
            include_camarillo=False,
            include_ann=True,
            evaluation_scope="artifact_test_split",
            results_root="data/modeling_results",
        ),
    )

    summary_payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert result.evaluation_scope_used == "artifact_test_split"
    assert result.selected_sample_count == 2
    assert summary_payload["models"]["ann"]["position_rmse_mm"] == 0.0
    assert result.comparison_csv_path.exists()
    assert result.plot_paths["workspace_xy"].exists()
    assert result.plot_paths["comparison_summary"].exists()
    assert result.plot_paths["model_workspace_prediction_report"].exists()
    assert result.plot_paths["model_comparison_summary_report"].exists()


def test_evaluate_models_handles_missing_camarillo_reference_inputs_cleanly(tmp_path: Path) -> None:
    run_dir = _write_modeling_run(tmp_path)

    result = evaluate_models(
        project_root=tmp_path,
        dataset_path=run_dir,
        artifact_path=None,
        config=ModelingEvaluationConfig(
            include_mike=False,
            include_camarillo=True,
            include_ann=False,
            evaluation_scope="full_dataset",
            geometry=ModelingGeometryConfig(camarillo_stiffness_path="tools/missing_stiffness"),
        ),
    )

    assert result.model_evaluations["camarillo"].metrics.status == "unavailable"
    assert "Missing stiffness file" in result.model_evaluations["camarillo"].metrics.reason


def test_load_trained_artifact_details_preserves_split_manifest_and_loss_history(tmp_path: Path) -> None:
    run_dir = _write_modeling_run(tmp_path)
    artifact_dir = _write_artifact(tmp_path, run_dir)

    details = load_trained_artifact_details(artifact_dir)

    assert details.split_manifest["test_indices"] == [2, 3]
    assert details.train_losses == [1.0, 0.8]
    assert details.validation_losses == [1.1, 0.9]
