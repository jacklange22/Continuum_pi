from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import yaml

from continuum_robot.data.build_thesis_evidence_index import build_thesis_evidence_index
from continuum_robot.data.export_run_bundle import export_run_bundle
from continuum_robot.data.run_management import detail_pairs_for_run, summarize_run, write_run_review
from continuum_robot.data.validate_run_bundle import validate_run_folder
from continuum_robot.modeling.two_segment import (
    TwoSegmentModelingConfig,
    build_feature_label_bundle,
    load_two_segment_modeling_dataset,
    run_two_segment_modeling,
)
from continuum_robot.modeling.two_segment.cli import main as modeling_cli_main
from continuum_robot.modeling.two_segment.models import TorchANNModel, default_model_suite


FIXTURE_RUN = Path(__file__).parent / "fixtures" / "two_segment_modeling_trainable"


def _write_two_segment_dataset_run(root: Path, *, name: str = "20260508_120000_two_segment_collect_pose_command_dataset", servo_only: bool = False) -> Path:
    run_dir = root / "data" / "experiments" / "two_segment_collect_pose_command_dataset" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "dataset_type": "two_segment_collect_pose_command_dataset",
        "run_trust_mode": "servo_only" if servo_only else "thesis_trusted",
        "valid_for_model_training": False,
        "valid_for_two_segment_model_training": False if servo_only else True,
        "valid_for_thesis_repeatability": False,
        "startup_artifact_provenance": {
            "accepted_all_8_startup": not servo_only,
            "artifact_path": str(run_dir / "all8_startup.json"),
            "artifact_sha256": "abc123",
        },
        "pose_label_summary": {
            "available_roles": [] if servo_only else ["distal_tip"],
            "missing_required_roles": ["distal_tip"] if servo_only else [],
            "distal_pose_sample_count": 0 if servo_only else 8,
        },
        "run_provenance": {
            "operating_mode": "dual_segment",
            "hardware_profile": "robot_8servo.yaml",
            "two_segment_foundation": {
                "command_schema": {"schema_version": "two_segment_command_v1"},
                "pose_schema": {"schema_version": "two_segment_pose_observation_v1"},
            },
        },
    }
    metadata = {
        "experiment_name": "two_segment_collect_pose_command_dataset",
        "trust_info": {
            "run_trust_mode": metrics["run_trust_mode"],
            "valid_for_model_training": False,
            "valid_for_two_segment_model_training": metrics["valid_for_two_segment_model_training"],
            "valid_for_thesis_repeatability": False,
        },
        "provenance_info": {
            "operating_mode": "dual_segment",
            "hardware_profile": "robot_8servo.yaml",
            "two_segment_foundation": metrics["run_provenance"]["two_segment_foundation"],
        },
    }
    summary = {
        "experiment_name": "two_segment_collect_pose_command_dataset",
        "success": True,
        "status": "success",
        "sample_counts": {"total": 8},
        "experiment_metrics": metrics,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "config_snapshot.yaml").write_text("robot:\n  mode: dual_segment\n", encoding="utf-8")
    (run_dir / "two_segment_tracking_role_provenance.json").write_text(json.dumps({"pose_label_summary": metrics["pose_label_summary"]}), encoding="utf-8")
    with (run_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(8):
            command_mm = [
                float(index),
                float(index % 3),
                -float(index) / 2.0,
                0.25 * index,
                0.1 * index,
                -0.2 * index,
                0.3 * index,
                -0.1 * index,
            ]
            position = [
                10.0 + 0.5 * command_mm[0] - 0.2 * command_mm[2],
                20.0 + 0.3 * command_mm[1] + 0.1 * command_mm[6],
                30.0 + 0.05 * sum(command_mm),
            ]
            matrix = [
                [1.0, 0.0, 0.0, position[0]],
                [0.0, 1.0, 0.0, position[1]],
                [0.0, 0.0, 1.0, position[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
            pose_payload = {"frame": "robot", "distal_tip_pose": {"T_robot_tip": matrix}}
            if servo_only:
                pose_payload = {"frame": "tracker", "distal_tip_pose": {}}
            sample = {
                "wall_time_utc": f"2026-05-08T12:00:{index:02d}+00:00",
                "phase": "synthetic_test",
                "step_index": index,
                "sample_index": index,
                "two_segment_command": {
                    "schema_version": "two_segment_command_v1",
                    "units": "cm",
                    "segments": {
                        "segment_a": [value / 10.0 for value in command_mm[:4]],
                        "segment_b": [value / 10.0 for value in command_mm[4:]],
                    },
                    "commanded_servo_ids": [1, 2, 3, 4, 5, 6, 7, 8],
                    "flat_command_cm": [value / 10.0 for value in command_mm],
                },
                "pose_in_robot_frame": {} if servo_only else {"roles": {"distal_tip": {"T_robot_tip": matrix}}},
                "two_segment_pose": pose_payload,
                "extra": {
                    "record_kind": "two_segment_dataset_capture",
                    "run_trust_mode": "servo_only" if servo_only else "thesis_trusted",
                    "capture_accepted": True,
                    "command_success": True,
                    "valid_for_two_segment_model_training": False if servo_only else True,
                    "command_units": "cm",
                    "commanded_servo_ids": [1, 2, 3, 4, 5, 6, 7, 8],
                    "startup_artifact_provenance": metrics["startup_artifact_provenance"],
                    "available_pose_roles": [] if servo_only else ["distal_tip"],
                    "missing_required_pose_roles": ["distal_tip"] if servo_only else [],
                    "distal_only": True,
                    "includes_intermediate_pose": False,
                },
            }
            handle.write(json.dumps(sample) + "\n")
    return run_dir


def test_two_segment_modeling_loader_is_strict_and_builds_mm_features(tmp_path: Path) -> None:
    trusted = _write_two_segment_dataset_run(tmp_path)
    servo_only = _write_two_segment_dataset_run(tmp_path, name="20260508_120100_two_segment_collect_pose_command_dataset", servo_only=True)

    dataset = load_two_segment_modeling_dataset([trusted, servo_only])
    bundle = build_feature_label_bundle(dataset)

    assert dataset.accepted_count == 8
    assert dataset.rejection_counts()["sample_not_two_segment_model_training_valid"] == 8
    assert bundle.X.shape == (8, 8)
    assert bundle.X[1, 0] == 1.0
    assert bundle.feature_metadata["feature_names"][0] == "segment_a_servo_1_displacement_mm"
    assert bundle.label_metadata["orientation_available"] is False


def test_two_segment_modeling_config_example_loads() -> None:
    payload = yaml.safe_load(Path("config/modeling_two_segment.example.yaml").read_text(encoding="utf-8"))

    assert payload["strict_trainability"] is True
    assert payload["input_units"] == "tendon_displacement_mm"
    assert payload["output_target"] == "distal_tip_xyz"
    assert payload["ann"]["hidden_layers"] == [128, 128]
    assert "linear_baseline" in payload["models"]["enabled"]


def test_two_segment_modeling_writes_outputs_validator_export_and_data_summary(tmp_path: Path) -> None:
    run_dir = _write_two_segment_dataset_run(tmp_path)

    result = run_two_segment_modeling(
        run_dirs=[run_dir],
        project_root=tmp_path,
        config=TwoSegmentModelingConfig(
            model_keys=["linear_baseline", "camarillo", "mike_constant_curvature"],
            output_root=str(tmp_path / "data" / "experiments"),
            random_seed=4,
        ),
    )

    assert result.output_dir.parent.name == "two_segment_modeling"
    for filename in [
        "summary.json",
        "metadata.json",
        "metrics.csv",
        "predictions.csv",
        "two_segment_modeling_summary.txt",
        "model_config.yaml",
        "feature_metadata.json",
        "label_metadata.json",
        "train_test_split.json",
        "rejected_samples.jsonl",
        "two_segment_model_comparison_report.png",
        "two_segment_measured_vs_predicted_xy_report.png",
        "two_segment_position_error_distribution_report.png",
        "two_segment_axis_error_report.png",
    ]:
        assert (result.output_dir / filename).exists()
    summary = json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["experiment_name"] == "two_segment_modeling"
    assert summary["experiment_metrics"]["models"]["linear_baseline"]["status"] == "completed"
    assert summary["experiment_metrics"]["models"]["camarillo"]["status"] == "unavailable"
    assert "single_run_random_split_can_overestimate_generalization" in summary["experiment_metrics"]["data_quality_warnings"]

    validation = validate_run_folder(result.output_dir)
    assert validation.status == "PASS"
    details = dict(detail_pairs_for_run(summarize_run(result.output_dir), project_root=tmp_path))
    assert "linear_baseline" in details["Two-Segment Modeling"]

    export = export_run_bundle(run_dir=result.output_dir, output_root=tmp_path / "exports", project_root=tmp_path)
    exported = {entry.bundle_path for entry in export.entries}
    assert "two_segment_modeling_summary.txt" in exported
    assert "predictions.csv" in exported
    assert "models/linear_baseline/linear_baseline_weights.json" in exported


def test_two_segment_modeling_fixture_supports_orientation_outputs_and_evidence_index(tmp_path: Path) -> None:
    run_dir = tmp_path / "data" / "experiments" / "two_segment_collect_pose_command_dataset" / "20260508_120000_two_segment_collect_pose_command_dataset"
    shutil.copytree(FIXTURE_RUN, run_dir)

    dataset = load_two_segment_modeling_dataset([run_dir])
    bundle = build_feature_label_bundle(dataset)
    assert bundle.X.shape == (6, 8)
    assert bundle.y.shape == (6, 6)
    assert bundle.feature_metadata["feature_units"] == ["mm"] * 8
    assert bundle.y_position[0].tolist() == [10.0, 20.0, 30.0]
    assert bundle.label_metadata["orientation_available"] is True

    result = run_two_segment_modeling(
        run_dirs=[run_dir],
        project_root=tmp_path,
        config=TwoSegmentModelingConfig(
            model_keys=["linear_baseline", "camarillo"],
            output_root=str(tmp_path / "data" / "experiments"),
            random_seed=1,
        ),
    )
    summary = json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8"))
    linear = summary["experiment_metrics"]["models"]["linear_baseline"]
    assert "orientation_mean_error_deg" in linear["metrics"]
    assert (result.output_dir / "two_segment_orientation_error_report.png").exists()
    assert summary["experiment_metrics"]["best_model_by_xyz_rmse"]["model_key"] == "linear_baseline"

    write_run_review(result.output_dir, status="thesis_candidate")
    index_dir = build_thesis_evidence_index(project_root=tmp_path)
    index_json = json.loads((index_dir / "thesis_evidence_index.json").read_text(encoding="utf-8"))
    entry = index_json["experiments"]["two_segment_modeling"][0]
    modeling_metrics = entry["key_metrics"]["two_segment_modeling"]
    assert modeling_metrics["input_dataset_run_ids"] == ["fixture-two-segment-trainable"]
    assert modeling_metrics["best_model"] == "linear_baseline"
    assert "two_segment_orientation_error_report.png" in "\n".join(entry["report_figures"])


def test_two_segment_modeling_cli_help_and_no_trainable_data_message(tmp_path: Path, capsys) -> None:
    try:
        modeling_cli_main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    help_text = capsys.readouterr().out
    assert "--latest" in help_text
    assert "--config" in help_text
    assert "--output-dir" in help_text

    servo_only = _write_two_segment_dataset_run(tmp_path, servo_only=True)
    status = modeling_cli_main(["--runs", str(servo_only), "--project-root", str(tmp_path), "--models", "linear_baseline"])

    captured = capsys.readouterr()
    assert status == 2
    assert "Runs scanned: 1" in captured.err
    assert "Samples accepted: 0" in captured.err
    assert "distal_tip pose label in robot frame" in captured.err


def test_two_segment_modeling_cli_latest_and_config_succeed(tmp_path: Path) -> None:
    run_dir = tmp_path / "data" / "experiments" / "two_segment_collect_pose_command_dataset" / "20260508_120000_two_segment_collect_pose_command_dataset"
    shutil.copytree(FIXTURE_RUN, run_dir)

    status = modeling_cli_main(
        [
            "--latest",
            "--project-root",
            str(tmp_path),
            "--config",
            "config/modeling_two_segment.example.yaml",
            "--models",
            "linear_baseline",
            "--output-dir",
            str(tmp_path / "custom_two_segment_modeling_output"),
        ]
    )

    assert status == 0
    assert (tmp_path / "custom_two_segment_modeling_output" / "summary.json").exists()


def test_two_segment_ann_reports_unavailable_when_torch_missing(monkeypatch) -> None:
    original_import = __import__

    def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise ImportError("torch deliberately unavailable in test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", _blocked_import)
    model = TorchANNModel(epochs=1)
    result = model.fit_predict(
        X_train=np.zeros((2, 8)),
        y_train=np.zeros((2, 3)),
        X_test=np.zeros((1, 8)),
        y_test=np.zeros((1, 3)),
        model_dir=Path("/tmp/not_written_ann_missing"),
    )

    assert result.status == "unavailable"
    assert "PyTorch unavailable" in result.reason


def test_two_segment_physics_adapters_remain_honestly_unavailable(tmp_path: Path) -> None:
    models = default_model_suite(model_keys=["camarillo", "mike_constant_curvature"], config={})
    results = [
        model.fit_predict(
            X_train=np.zeros((2, 8)),
            y_train=np.zeros((2, 3)),
            X_test=np.zeros((1, 8)),
            y_test=np.zeros((1, 3)),
            model_dir=tmp_path / model.model_key,
        )
        for model in models
    ]

    assert all(result.status == "unavailable" for result in results)
    assert all("fake" in result.reason or "No active" in result.reason or "not yet wired" in result.reason for result in results)


def test_two_segment_modeling_cli_runs_linear_only(tmp_path: Path) -> None:
    run_dir = _write_two_segment_dataset_run(tmp_path)

    status = modeling_cli_main(
        [
            "--runs",
            str(run_dir),
            "--project-root",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "data" / "experiments"),
            "--models",
            "linear_baseline",
            "--seed",
            "2",
        ]
    )

    assert status == 0
    output_root = tmp_path / "data" / "experiments" / "two_segment_modeling"
    assert any(path.joinpath("summary.json").exists() for path in output_root.iterdir())
