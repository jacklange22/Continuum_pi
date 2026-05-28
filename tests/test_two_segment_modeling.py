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
from continuum_robot.modeling.two_segment.physics import two_segment_constant_curvature_prediction


FIXTURE_RUN = Path(__file__).parent / "fixtures" / "two_segment_modeling_trainable"


def _write_two_segment_dataset_run(
    root: Path,
    *,
    name: str = "20260508_120000_two_segment_collect_pose_command_dataset",
    servo_only: bool = False,
    include_intermediate: bool = False,
) -> Path:
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
            intermediate = [
                5.0 + 0.2 * command_mm[0],
                8.0 + 0.15 * command_mm[1],
                15.0 + 0.02 * sum(command_mm[:4]),
            ]
            intermediate_matrix = [
                [1.0, 0.0, 0.0, intermediate[0]],
                [0.0, 1.0, 0.0, intermediate[1]],
                [0.0, 0.0, 1.0, intermediate[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
            pose_payload = {"frame": "robot", "distal_tip_pose": {"T_robot_tip": matrix}}
            if servo_only:
                pose_payload = {"frame": "tracker", "distal_tip_pose": {}}
            robot_roles = {} if servo_only else {"distal_tip": {"T_robot_tip": matrix}}
            available_roles = [] if servo_only else ["distal_tip"]
            if include_intermediate and not servo_only:
                robot_roles["intermediate_segment"] = {"T_robot_intermediate": intermediate_matrix}
                available_roles.append("intermediate_segment")
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
                "pose_in_robot_frame": {} if servo_only else {"roles": robot_roles},
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
                    "available_pose_roles": available_roles,
                    "missing_required_pose_roles": ["distal_tip"] if servo_only else [],
                    "distal_only": not bool(include_intermediate and not servo_only),
                    "includes_intermediate_pose": bool(include_intermediate and not servo_only),
                    "segment_order": ["segment_a", "segment_b"],
                    "segments": {
                        "segment_a": {"label": "Segment A", "role": "proximal", "servo_ids": [1, 2, 3, 4], "pair_mapping": [[1, 3], [2, 4]], "segment_order_index": 0},
                        "segment_b": {"label": "Segment B", "role": "distal", "servo_ids": [5, 6, 7, 8], "pair_mapping": [[5, 7], [6, 8]], "segment_order_index": 1},
                    },
                    "measured_servo_feedback": {
                        str(servo_id): {
                            "servo_id": int(servo_id),
                            "position_tick": 2048 + index + int(servo_id),
                            "signed_raw_current_ma": 50 + int(servo_id),
                            "load_proxy_ma": 50 + int(servo_id),
                        }
                        for servo_id in range(1, 9)
                    },
                },
            }
            handle.write(json.dumps(sample) + "\n")
    return run_dir


def test_two_segment_modeling_rejects_samples_missing_measured_servo_positions(tmp_path: Path) -> None:
    run_dir = _write_two_segment_dataset_run(tmp_path)
    # Corrupt one sample's measured feedback to drop servo 5's position.
    samples_path = run_dir / "samples.jsonl"
    lines = samples_path.read_text(encoding="utf-8").strip().split("\n")
    corrupt = json.loads(lines[0])
    corrupt["extra"]["measured_servo_feedback"]["5"]["position_tick"] = None
    lines[0] = json.dumps(corrupt)
    samples_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    dataset = load_two_segment_modeling_dataset([run_dir])

    assert dataset.accepted_count == 7
    assert dataset.rejected_count == 1
    reasons = dataset.rejection_counts()
    assert any("measured_servo_position_missing" in reason for reason in reasons)
    # And the same dataset loaded with allow_lower_trust accepts all 8 again.
    relaxed = load_two_segment_modeling_dataset([run_dir], allow_lower_trust=True)
    assert relaxed.accepted_count == 8


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
    assert bundle.feature_metadata["tendon_displacement_positive_direction"] == "shortening"
    assert bundle.feature_metadata["encoder_tick_direction_for_shortening"] == "decreasing_ticks"
    assert bundle.label_metadata["orientation_available"] is False


def test_two_segment_modeling_config_example_loads() -> None:
    payload = yaml.safe_load(Path("config/modeling_two_segment.example.yaml").read_text(encoding="utf-8"))

    assert payload["strict_trainability"] is True
    assert payload["input_units"] == "tendon_displacement_mm"
    assert payload["output_target"] == "auto"
    assert "two_coil_xyz" in payload["labeling"]["allowed_modes"]
    assert payload["ann"]["hidden_layers"] == [128, 128]
    assert payload["ann"]["hidden_layer_options"] == [[32, 32], [64, 64], [128, 128]]
    assert payload["physics_models"]["global"]["output_pose_frame"]["value"] == "robot"
    assert payload["physics_models"]["global"]["tendon_displacement_positive_direction"]["value"] == "shortening"
    assert payload["physics_models"]["global"]["encoder_tick_direction_for_shortening"]["value"] == "decreasing_ticks"
    assert payload["physics_models"]["segments"]["segment_a"]["segment_length_mm"]["value"] == 66.0
    assert payload["physics_models"]["segments"]["segment_b"]["tendon_center_radius_mm"]["value"] == 4.12
    assert payload["physics_models"]["segments"]["segment_a"]["hole_radius_or_diameter_mm"]["value"] == 1.65
    assert "confirm whether" in payload["physics_models"]["segments"]["segment_a"]["hole_radius_or_diameter_mm"]["notes"].lower()
    assert "required_conventions_confirmed" in payload["physics_models"]["mike_constant_curvature"]
    assert payload["physics_models"]["mike_constant_curvature"]["geometry_complete"] is True
    assert payload["physics_models"]["mike_constant_curvature"]["sign_convention_known"] is True
    assert "segment_stiffness_values" in payload["physics_models"]["camarillo"]
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
        "model_status.json",
        "physics_model_parameter_report.json",
        "physics_model_parameter_report.txt",
        "model_sweep_summary.json",
        "model_sweep_summary.csv",
        "model_sweep_summary.txt",
        "two_segment_model_comparison_report.png",
        "two_segment_distal_measured_vs_predicted_xy_report.png",
        "two_segment_measured_vs_predicted_xy_report.png",
        "two_segment_position_error_distribution_report.png",
        "two_segment_axis_error_report.png",
        "two_segment_train_test_workspace_coverage_report.png",
        "two_segment_two_coil_error_report.png",
    ]:
        assert (result.output_dir / filename).exists()
    summary = json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["experiment_name"] == "two_segment_modeling"
    assert summary["experiment_metrics"]["models"]["linear_baseline"]["status"] == "completed"
    assert summary["experiment_metrics"]["models"]["camarillo"]["status"].startswith("unavailable")
    assert summary["experiment_metrics"]["models"]["mike_constant_curvature"]["status"].startswith("unavailable")
    assert summary["experiment_metrics"]["models"]["mike_constant_curvature"]["artifact_paths"]
    assert summary["experiment_metrics"]["physics_model_status"]["camarillo"]["predictions_generated"] is False
    assert "single_run_random_split_can_overestimate_generalization" in summary["experiment_metrics"]["data_quality_warnings"]
    # Completed/unavailable bookkeeping is surfaced at the summary level.
    assert summary["experiment_metrics"]["completed_model_keys"] == ["linear_baseline"]
    assert sorted(summary["experiment_metrics"]["unavailable_model_keys"]) == ["camarillo", "mike_constant_curvature"]
    assert summary["experiment_metrics"]["completed_model_count"] == 1
    summary_text = (result.output_dir / "two_segment_modeling_summary.txt").read_text(encoding="utf-8")
    assert "models_completed: ['linear_baseline']" in summary_text
    assert "models_unavailable" in summary_text
    assert "best_model: linear_baseline" in summary_text
    sweep_summary = json.loads((result.output_dir / "model_sweep_summary.json").read_text(encoding="utf-8"))
    assert sweep_summary["primary_success_metric"] == "distal_xyz_rmse_mm"
    assert sweep_summary["best_model_by_distal_xyz_rmse"]["model_key"] == "linear_baseline"

    validation = validate_run_folder(result.output_dir)
    assert validation.status == "PASS"
    details = dict(detail_pairs_for_run(summarize_run(result.output_dir), project_root=tmp_path))
    assert "linear_baseline" in details["Two-Segment Modeling"]

    export = export_run_bundle(run_dir=result.output_dir, output_root=tmp_path / "exports", project_root=tmp_path)
    exported = {entry.bundle_path for entry in export.entries}
    assert "two_segment_modeling_summary.txt" in exported
    assert "predictions.csv" in exported
    assert "model_status.json" in exported
    assert "model_sweep_summary.json" in exported
    assert "physics_model_parameter_report.json" in exported
    assert "models/linear_baseline/linear_baseline_weights.json" in exported


def test_two_segment_modeling_fixture_supports_orientation_outputs_and_evidence_index(tmp_path: Path) -> None:
    run_dir = tmp_path / "data" / "experiments" / "two_segment_collect_pose_command_dataset" / "20260508_120000_two_segment_collect_pose_command_dataset"
    shutil.copytree(FIXTURE_RUN, run_dir)

    dataset = load_two_segment_modeling_dataset([run_dir])
    bundle = build_feature_label_bundle(dataset, include_orientation_if_available=True)
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
            include_orientation_if_available=True,
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
    assert "camarillo" in modeling_metrics["physics_model_status"]
    assert "two_segment_orientation_error_report.png" in "\n".join(entry["report_figures"])


def test_two_segment_modeling_two_coil_label_mode_and_segment_metadata(tmp_path: Path) -> None:
    run_dir = _write_two_segment_dataset_run(tmp_path, include_intermediate=True)
    dataset = load_two_segment_modeling_dataset([run_dir])
    bundle = build_feature_label_bundle(dataset)

    assert bundle.label_metadata["label_mode"] == "two_coil_xyz"
    assert bundle.y.shape == (8, 6)
    assert bundle.label_metadata["label_slices"]["intermediate_segment"]["position"] == [0, 1, 2]
    assert bundle.label_metadata["label_slices"]["distal_tip"]["position"] == [3, 4, 5]
    assert bundle.label_metadata["includes_intermediate_label"] is True
    assert bundle.feature_metadata["segment_grouping"]["lower_segment"]["segment"] == "segment_a"
    assert bundle.feature_metadata["segment_grouping"]["upper_segment"]["segment"] == "segment_b"

    result = run_two_segment_modeling(
        run_dirs=[run_dir],
        project_root=tmp_path,
        config=TwoSegmentModelingConfig(
            model_keys=["linear_baseline"],
            output_root=str(tmp_path / "data" / "experiments"),
            random_seed=3,
        ),
    )
    summary = json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8"))
    linear = summary["experiment_metrics"]["models"]["linear_baseline"]["metrics"]
    assert summary["experiment_metrics"]["label_mode"] == "two_coil_xyz"
    assert summary["experiment_metrics"]["includes_intermediate_label"] is True
    assert "intermediate_xyz_rmse_mm" in linear
    assert "two_coil_combined_xyz_rmse_mm" in linear
    assert (result.output_dir / "two_segment_intermediate_measured_vs_predicted_xy_report.png").exists()


def test_two_segment_modeling_auto_label_mode_degrades_to_distal_xyz_on_mixed_intermediate_dataset(tmp_path: Path) -> None:
    """A dataset with SOME samples missing intermediate poses auto-resolves to distal_xyz.

    The auto resolver uses an all-or-nothing rule (every sample must have
    intermediate). Mixed datasets gracefully degrade rather than failing,
    so the operator doesn't lose the run because one sample dropped the
    intermediate label.
    """
    # Build a dataset where the *fixture* has all intermediates, then corrupt
    # one sample's intermediate role observation so it's effectively missing.
    run_dir = _write_two_segment_dataset_run(tmp_path, include_intermediate=True)
    samples_path = run_dir / "samples.jsonl"
    lines = samples_path.read_text(encoding="utf-8").strip().split("\n")
    corrupt = json.loads(lines[0])
    # Strip the intermediate role from this sample's robot-frame pose.
    pose_roles = (corrupt.get("pose_in_robot_frame") or {}).get("roles") or {}
    pose_roles.pop("intermediate_segment", None)
    corrupt.setdefault("pose_in_robot_frame", {})["roles"] = pose_roles
    lines[0] = json.dumps(corrupt)
    samples_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    dataset = load_two_segment_modeling_dataset([run_dir])
    bundle = build_feature_label_bundle(dataset)  # auto mode

    # Mixed → auto picks distal_xyz, not two_coil_xyz.
    assert bundle.label_metadata["label_mode"] == "distal_xyz"
    assert "intermediate_segment" not in bundle.label_metadata["label_slices"]


def test_two_segment_modeling_strict_two_coil_mode_raises_clearly_on_mixed_dataset(tmp_path: Path) -> None:
    """Operator explicitly requesting two_coil_xyz on a mixed dataset gets a clear error, not silent degradation."""
    from continuum_robot.modeling.two_segment.features import LabelBuildError

    run_dir = _write_two_segment_dataset_run(tmp_path, include_intermediate=True)
    samples_path = run_dir / "samples.jsonl"
    lines = samples_path.read_text(encoding="utf-8").strip().split("\n")
    corrupt = json.loads(lines[0])
    pose_roles = (corrupt.get("pose_in_robot_frame") or {}).get("roles") or {}
    pose_roles.pop("intermediate_segment", None)
    corrupt.setdefault("pose_in_robot_frame", {})["roles"] = pose_roles
    lines[0] = json.dumps(corrupt)
    samples_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    dataset = load_two_segment_modeling_dataset([run_dir])
    try:
        build_feature_label_bundle(dataset, label_mode="two_coil_xyz")
    except LabelBuildError as exc:
        assert "intermediate_segment" in str(exc)
    else:
        raise AssertionError("Expected LabelBuildError when explicit two_coil_xyz is requested on a mixed dataset.")


def test_two_segment_modeling_distal_xyz_fallback_does_not_fabricate_intermediate(tmp_path: Path) -> None:
    run_dir = _write_two_segment_dataset_run(tmp_path, include_intermediate=False)
    dataset = load_two_segment_modeling_dataset([run_dir])
    bundle = build_feature_label_bundle(dataset)

    assert bundle.label_metadata["label_mode"] == "distal_xyz"
    assert bundle.y.shape == (8, 3)
    assert "intermediate_segment" not in bundle.label_metadata["label_slices"]


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
        X_val=np.zeros((1, 8)),
        y_val=np.zeros((1, 3)),
        X_test=np.zeros((1, 8)),
        y_test=np.zeros((1, 3)),
        model_dir=Path("/tmp/not_written_ann_missing"),
    )

    assert result.status == "unavailable"
    assert "PyTorch unavailable" in result.reason


def test_two_segment_ann_sweep_skips_cleanly_when_torch_missing(monkeypatch, tmp_path: Path) -> None:
    original_import = __import__

    def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise ImportError("torch deliberately unavailable in test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", _blocked_import)
    model = default_model_suite(
        model_keys=["ann"],
        config={"ann": {"sweep_enabled": True, "hidden_layer_options": [[32, 32]], "epochs": 1}},
    )[0]
    result = model.fit_predict(
        X_train=np.zeros((2, 8)),
        y_train=np.zeros((2, 3)),
        X_val=np.zeros((1, 8)),
        y_val=np.zeros((1, 3)),
        X_test=np.zeros((1, 8)),
        y_test=np.zeros((1, 3)),
        model_dir=tmp_path / "ann_sweep_missing",
    )

    assert result.status == "unavailable"
    assert "PyTorch unavailable" in result.reason
    assert (tmp_path / "ann_sweep_missing" / "ann_sweep_summary.json").exists()


def test_two_segment_physics_adapters_remain_honestly_unavailable(tmp_path: Path) -> None:
    models = default_model_suite(model_keys=["camarillo", "mike_constant_curvature"], config={})
    results = [
        model.fit_predict(
            X_train=np.zeros((2, 8)),
            y_train=np.zeros((2, 3)),
            X_val=np.zeros((1, 8)),
            y_val=np.zeros((1, 3)),
            X_test=np.zeros((1, 8)),
            y_test=np.zeros((1, 3)),
            model_dir=tmp_path / model.model_key,
        )
        for model in models
    ]

    assert all(result.status.startswith("unavailable") for result in results)
    assert all(result.predictions is None for result in results)
    assert all((tmp_path / result.model_key / "model_status.json").exists() for result in results)
    assert any("Missing required" in result.reason for result in results)


def test_design_geometry_makes_mike_ready_but_still_convention_gated(tmp_path: Path) -> None:
    run_dir = _write_two_segment_dataset_run(tmp_path, include_intermediate=True)
    payload = yaml.safe_load(Path("config/modeling_two_segment.example.yaml").read_text(encoding="utf-8"))
    result = run_two_segment_modeling(
        run_dirs=[run_dir],
        project_root=tmp_path,
        config=TwoSegmentModelingConfig(
            model_keys=["mike_constant_curvature", "camarillo"],
            model_config={"physics_models": payload["physics_models"]},
            output_root=str(tmp_path / "data" / "experiments"),
            random_seed=4,
        ),
    )
    summary = json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8"))
    report = json.loads((result.output_dir / "physics_model_parameter_report.json").read_text(encoding="utf-8"))
    mike = summary["experiment_metrics"]["models"]["mike_constant_curvature"]
    camarillo = summary["experiment_metrics"]["models"]["camarillo"]

    assert mike["status"] == "unavailable_unvalidated_convention"
    assert "segment_length_mm" not in "\n".join(mike["status_details"]["missing_parameters"])
    assert "tendon_positions_mm" not in "\n".join(mike["status_details"]["missing_parameters"])
    assert any("segment_a.segment_length_mm" in item for item in report["known_design_parameters"])
    assert any("hole_radius_or_diameter_mm" in item for item in report["nonblocking_uncertain_parameters"])
    assert "required_conventions_confirmed" in "\n".join(mike["status_details"]["hardware_validation_required"])
    assert camarillo["status"].startswith("unavailable")
    assert "segment_stiffness_values" in "\n".join(camarillo["status_details"]["blocking_missing_parameters"])
    assert "servo_current_is_load_proxy_not_stiffness" in "\n".join(payload["physics_models"]["camarillo"]["unavailable_reason_notes"])


def _complete_cc_physics_config() -> dict:
    positions = [[4.0, 0.0], [0.0, 4.0], [-4.0, 0.0], [0.0, -4.0]]
    return {
        "physics_models": {
            "global": {
                "model_frame_convention": "base_frame_z_along_segment_then_compose_segment_a_segment_b",
                "tendon_displacement_sign_convention": "positive_shortens_tendon",
                "output_pose_frame": "robot",
                "tangent_representation": "transform_z_axis",
                "segment_order_source": "robot_config",
            },
            "segments": {
                "segment_a": {"segment_length_mm": 80.0, "tendon_positions_mm": positions},
                "segment_b": {"segment_length_mm": 70.0, "tendon_positions_mm": positions},
            },
            "mike_constant_curvature": {
                "required_conventions_confirmed": True,
                "validated_against_hardware": False,
                "curvature_from_tendon_displacement": "least_squares_tendon_plane",
            },
        }
    }


def test_mike_constant_curvature_composes_two_segments_without_fake_outputs(tmp_path: Path) -> None:
    run_dir = _write_two_segment_dataset_run(tmp_path, include_intermediate=True)
    config = _complete_cc_physics_config()
    result = run_two_segment_modeling(
        run_dirs=[run_dir],
        project_root=tmp_path,
        config=TwoSegmentModelingConfig(
            model_keys=["mike_constant_curvature"],
            model_config=config,
            output_root=str(tmp_path / "data" / "experiments"),
            random_seed=3,
        ),
    )
    model_result = result.model_results[0]

    assert model_result.status == "completed"
    assert model_result.predictions is not None
    assert model_result.status_details["hardware_validated"] is False
    assert model_result.status_details["predictions_generated"] is True
    assert "distal_xyz_rmse_mm" in model_result.metrics
    assert "intermediate_xyz_rmse_mm" in model_result.metrics


def test_constant_curvature_composition_semantics() -> None:
    config = _complete_cc_physics_config()
    zero = np.zeros(8, dtype=float)
    segment_a_only = np.asarray([1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    segment_b_only = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 1.0], dtype=float)
    combined = segment_a_only + segment_b_only

    pose_zero = two_segment_constant_curvature_prediction(zero, config)
    pose_a = two_segment_constant_curvature_prediction(segment_a_only, config)
    pose_b = two_segment_constant_curvature_prediction(segment_b_only, config)
    pose_ab = two_segment_constant_curvature_prediction(combined, config)

    assert not np.allclose(pose_a["distal_xyz"], pose_zero["distal_xyz"])
    assert not np.allclose(pose_b["distal_xyz"], pose_zero["distal_xyz"])
    assert not np.allclose(pose_ab["distal_xyz"], pose_a["distal_xyz"])
    assert not np.allclose(pose_ab["distal_xyz"], pose_b["distal_xyz"])
    assert np.allclose(pose_b["intermediate_xyz"], pose_zero["intermediate_xyz"])
    assert not np.allclose(pose_a["intermediate_xyz"], pose_zero["intermediate_xyz"])


def test_constant_curvature_sign_convention_is_explicit() -> None:
    config = _complete_cc_physics_config()
    command = np.asarray([1.0, 0.0, -1.0, 0.0, 0.5, 0.0, -0.5, 0.0], dtype=float)
    shortened = two_segment_constant_curvature_prediction(command, config)
    lengthen_config = json.loads(json.dumps(config))
    lengthen_config["physics_models"]["global"]["tendon_displacement_sign_convention"] = "positive_lengthens_tendon"
    lengthened = two_segment_constant_curvature_prediction(command, lengthen_config)

    assert not np.allclose(shortened["distal_xyz"], lengthened["distal_xyz"])
    assert np.sign(shortened["distal_xyz"][0]) == -np.sign(lengthened["distal_xyz"][0])


def test_segment_order_and_roles_are_configurable_in_feature_metadata(tmp_path: Path) -> None:
    run_dir = _write_two_segment_dataset_run(tmp_path)
    samples_path = run_dir / "samples.jsonl"
    rows = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        row["extra"]["segment_order"] = ["segment_b", "segment_a"]
        row["extra"]["segments"]["segment_a"]["role"] = "distal"
        row["extra"]["segments"]["segment_a"]["segment_order_index"] = 1
        row["extra"]["segments"]["segment_b"]["role"] = "proximal"
        row["extra"]["segments"]["segment_b"]["segment_order_index"] = 0
    samples_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    dataset = load_two_segment_modeling_dataset([run_dir])
    bundle = build_feature_label_bundle(dataset)

    assert bundle.feature_metadata["segment_order"] == ["segment_b", "segment_a"]
    assert bundle.feature_metadata["segment_grouping"]["lower_segment"]["segment"] == "segment_b"
    assert bundle.feature_metadata["segment_grouping"]["upper_segment"]["segment"] == "segment_a"


def test_two_segment_modeling_predictions_csv_includes_bottom_top_and_command_pattern(tmp_path: Path) -> None:
    """predictions.csv must carry bottom/top metadata + command pattern per row."""
    import csv

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
            "11",
        ]
    )

    assert status == 0
    output_root = tmp_path / "data" / "experiments" / "two_segment_modeling"
    run_dir_modeling = next(path for path in output_root.iterdir() if path.is_dir())
    predictions_path = run_dir_modeling / "predictions.csv"
    rows = list(csv.DictReader(predictions_path.open(encoding="utf-8")))
    assert rows, "predictions.csv must not be empty"
    # All rows must carry the bottom/top + command pattern columns.
    for row in rows:
        assert row["bottom_segment_key"] in {"segment_a", "segment_b"}
        assert row["top_segment_key"] in {"segment_a", "segment_b"}
        assert row["bottom_segment_key"] != row["top_segment_key"]
        assert row["command_pattern"] in {"both", "bottom_only", "top_only", "zero"}
        # Magnitudes should be parseable floats.
        float(row["bottom_command_magnitude_mm"])
        float(row["top_command_magnitude_mm"])


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


def test_two_segment_modeling_cli_minimum_accepted_samples_gate(tmp_path: Path, capsys) -> None:
    """Returns exit 2 when accepted samples are below the configured minimum."""
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
            "3",
            "--minimum-accepted-samples",
            "999",  # we only ever produce 8 fixture samples
        ]
    )

    assert status == 2
    captured = capsys.readouterr()
    assert "GATE FAIL: accepted samples" in captured.err


def test_two_segment_modeling_cli_maximum_best_rmse_gate(tmp_path: Path, capsys) -> None:
    """Returns exit 3 when the best completed model's RMSE exceeds the configured cap."""
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
            "5",
            "--maximum-best-rmse-mm",
            "0.0",  # nothing real is exactly zero RMSE
        ]
    )

    assert status == 3
    captured = capsys.readouterr()
    assert "GATE FAIL: best model" in captured.err


def test_two_segment_modeling_cli_gates_succeed_with_realistic_thresholds(tmp_path: Path) -> None:
    """Both gates pass when their thresholds are loose enough for the fixture."""
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
            "7",
            "--minimum-accepted-samples",
            "1",
            "--maximum-best-rmse-mm",
            "1000000.0",
        ]
    )

    assert status == 0


def _write_servo_only_run_with_real_poses(tmp_path: Path) -> Path:
    """A servo_only run that DOES carry real distal T_robot_tip poses.

    Models the real bench case (run flagged servo_only for trust, but the
    Aurora tracker WAS live and produced robot-frame distal poses). The old
    fixtures tied servo_only to no-pose, so this case was never covered.
    """
    run_dir = (
        tmp_path
        / "data"
        / "experiments"
        / "two_segment_collect_pose_command_dataset"
        / "20260526_235950_two_segment_collect_pose_command_dataset"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "experiment_name": "two_segment_collect_pose_command_dataset",
                "run_id": run_dir.name,
                "success": True,
                "status": "success",
                "experiment_metrics": {
                    "run_trust_mode": "servo_only",
                    "valid_for_two_segment_model_training": False,
                    "startup_artifact_provenance": {"accepted_all_8_startup": True},
                },
            }
        ),
        encoding="utf-8",
    )
    with (run_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(8):
            command_mm = [float((index + s) % 5) for s in range(8)]
            matrix = [
                [1.0, 0.0, 0.0, 10.0 + command_mm[0]],
                [0.0, 1.0, 0.0, 20.0 + command_mm[1]],
                [0.0, 0.0, 1.0, 30.0 + command_mm[4]],
                [0.0, 0.0, 0.0, 1.0],
            ]
            sample = {
                "wall_time_utc": f"2026-05-26T23:59:{index:02d}+00:00",
                "phase": "synthetic_servo_only_with_poses",
                "step_index": index,
                "sample_index": index,
                "two_segment_command": {
                    "schema_version": "two_segment_command_v1",
                    "units": "cm",
                    "segments": {
                        "segment_a": [v / 10.0 for v in command_mm[:4]],
                        "segment_b": [v / 10.0 for v in command_mm[4:]],
                    },
                    "commanded_servo_ids": [1, 2, 3, 4, 5, 6, 7, 8],
                    "flat_command_cm": [v / 10.0 for v in command_mm],
                },
                # Real robot-frame distal pose stored as a T_robot_tip matrix,
                # exactly like the bench dataset.
                "pose_in_robot_frame": {"roles": {"distal_tip": {"T_robot_tip": matrix}}},
                "two_segment_pose": {"frame": "robot", "distal_tip_pose": {"T_robot_tip": matrix}},
                "extra": {
                    "record_kind": "two_segment_dataset_capture",
                    "run_trust_mode": "servo_only",
                    "capture_accepted": True,
                    "command_success": True,
                    "valid_for_two_segment_model_training": False,
                    "command_units": "cm",
                    "commanded_servo_ids": [1, 2, 3, 4, 5, 6, 7, 8],
                    "ordered_8_displacements_cm": [v / 10.0 for v in command_mm],
                    "startup_artifact_provenance": {"accepted_all_8_startup": True},
                    "missing_required_pose_roles": [],
                    "measured_servo_feedback": {
                        str(servo_id): {
                            "servo_id": int(servo_id),
                            "position_tick": 2048 + index + int(servo_id),
                            "load_proxy_ma": 50 + int(servo_id),
                        }
                        for servo_id in range(1, 9)
                    },
                },
            }
            handle.write(json.dumps(sample) + "\n")
    return run_dir


def test_servo_only_run_with_real_poses_is_strict_rejected_but_lower_trust_trainable(tmp_path: Path) -> None:
    """Regression: a complete servo_only dataset (real poses) must be:

    - rejected under strict (servo_only is not thesis-valid), and
    - ACCEPTED under allow_lower_trust (the data is genuinely complete).

    This is the exact bench scenario that previously rejected 100% of
    samples even with allow_lower_trust via the old servo_only hard-block.
    """
    run_dir = _write_servo_only_run_with_real_poses(tmp_path)

    strict = load_two_segment_modeling_dataset([run_dir])
    assert strict.accepted_count == 0
    assert any(
        "sample_not_two_segment_model_training_valid" in reason for reason in strict.rejection_counts()
    )

    relaxed = load_two_segment_modeling_dataset([run_dir], allow_lower_trust=True)
    assert relaxed.accepted_count == 8, relaxed.rejection_counts()


def test_dry_run_dataset_stays_blocked_even_under_lower_trust(tmp_path: Path) -> None:
    """dry_run has no real hardware, so it must stay blocked under allow_lower_trust."""
    run_dir = _write_servo_only_run_with_real_poses(tmp_path)
    # Flip the trust mode to dry_run on every sample + summary.
    samples_path = run_dir / "samples.jsonl"
    lines = samples_path.read_text(encoding="utf-8").strip().split("\n")
    rewritten = []
    for line in lines:
        row = json.loads(line)
        row["extra"]["run_trust_mode"] = "dry_run"
        rewritten.append(json.dumps(row))
    samples_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

    relaxed = load_two_segment_modeling_dataset([run_dir], allow_lower_trust=True)
    assert relaxed.accepted_count == 0
    assert any("dry_run_samples_not_trainable" in reason for reason in relaxed.rejection_counts())
