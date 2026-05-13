from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from pathlib import Path
import time

import pytest

from continuum_robot.modeling import ann_training as training_module
from continuum_robot.modeling.ann_training import (
    AnnTrainingConfig,
    build_grouped_split,
    detect_training_backends,
    discover_modeling_datasets,
    discover_trained_artifacts,
    effective_legacy_ann_training_allowed,
    format_hidden_layers_for_ann_ui,
    load_loss_history,
    load_modeling_dataset_summary,
    parse_hidden_layers_text,
    prepare_legacy_ann_dataset,
    validate_training_config,
)


def _write_modeling_run(tmp_path: Path) -> Path:
    output_root = tmp_path / "data" / "experiments" / "collect_pose_command_dataset" / "20260419_120000_collect_pose_command_dataset"
    output_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "1.0",
        "experiment_name": "collect_pose_command_dataset",
        "run_id": "abc123",
        "timestamp_utc": "2026-04-19T12:00:00+00:00",
        "git_commit": None,
        "backend_info": {},
        "registration_info": {},
        "config_used": {"dataset_mode": "workspace_coverage"},
        "operator_notes": "",
    }
    summary = {
        "schema_version": "1.0",
        "experiment_name": "collect_pose_command_dataset",
        "run_id": "abc123",
        "success": True,
        "sample_counts": {"total": 4},
        "dropped_frames": 0,
        "invalid_transforms": 0,
        "stage_pass_fail": {},
        "status": "success",
        "experiment_metrics": {
            "dataset_mode": "workspace_coverage",
            "dataset_mode_summary": "Bounded workspace exploration for first-pass forward-model training.",
            "accepted_sample_count": 3,
            "rejected_sample_count": 1,
            "accepted_capture_rate": 0.75,
            "run_trust_mode": "thesis_trusted",
            "valid_for_model_training": True,
            "run_provenance": {
                "runtime_tip_calibration": {"mode": "latest_accepted", "trust_level": "trusted"},
                "pretension_artifact": {"active_source_type": "accepted_artifact", "status": "ready"},
            },
        },
        "warning_messages": [],
        "error_messages": [],
    }
    rows = [
        {
            "sequence_index": 0,
            "step_index": 0,
            "sample_index": 0,
            "accepted": True,
            "resolved_cable_command_cm": [0.1, 0.2, 0.3, 0.4],
            "tip_position_xyz_mm": [1.0, 2.0, 3.0],
            "tip_tangent_xyz": [0.01, 0.02, 0.03],
            "previous_pair_command_cm": [],
        },
        {
            "sequence_index": 1,
            "step_index": 0,
            "sample_index": 1,
            "accepted": True,
            "resolved_cable_command_cm": [0.2, 0.3, 0.4, 0.5],
            "tip_position_xyz_mm": [4.0, 5.0, 6.0],
            "tip_tangent_xyz": [0.04, 0.05, 0.06],
            "previous_pair_command_cm": [0.0, 0.0],
        },
        {
            "sequence_index": 2,
            "step_index": 1,
            "sample_index": 2,
            "accepted": True,
            "resolved_cable_command_cm": [0.3, 0.4, 0.5, 0.6],
            "tip_position_xyz_mm": [7.0, 8.0, 9.0],
            "tip_tangent_xyz": [10.0, 0.01, 0.02],
            "previous_pair_command_cm": [0.1, 0.1],
        },
        {
            "sequence_index": 3,
            "step_index": 2,
            "sample_index": 3,
            "accepted": False,
            "resolved_cable_command_cm": [0.4, 0.5, 0.6, 0.7],
            "tip_position_xyz_mm": [10.0, 11.0, 12.0],
            "tip_tangent_xyz": [0.07, 0.08, 0.09],
            "previous_pair_command_cm": [0.2, 0.2],
        },
    ]
    (output_root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (output_root / "modeling_dataset_summary.txt").write_text("summary\n", encoding="utf-8")
    (output_root / "modeling_dataset_legacy_compat.dat").write_text("legacy\n", encoding="utf-8")
    with (output_root / "modeling_dataset_export.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return output_root


def test_discover_modeling_datasets_finds_real_collect_pose_runs(tmp_path: Path) -> None:
    run_dir = _write_modeling_run(tmp_path)
    catalog = discover_modeling_datasets(project_root=tmp_path, output_root=tmp_path / "data" / "experiments")
    assert len(catalog) == 1
    assert catalog[0].path.resolve() == run_dir.resolve()
    assert catalog[0].dataset_scan_root == "experiments"
    assert catalog[0].trainable_for_legacy_ann is True


def test_discover_modeling_datasets_marks_mock_root_non_trainable(tmp_path: Path) -> None:
    mock_run = tmp_path / "data" / "mock_experiments" / "collect_pose_command_dataset" / "20260420_mock_pose"
    mock_run.mkdir(parents=True)
    metadata = {
        "schema_version": "1.0",
        "experiment_name": "collect_pose_command_dataset",
        "run_id": "mock1",
        "timestamp_utc": "2026-04-20T12:00:00+00:00",
    }
    summary = {
        "schema_version": "1.0",
        "experiment_name": "collect_pose_command_dataset",
        "run_id": "mock1",
        "success": True,
        "sample_counts": {"total": 1},
        "status": "success",
        "experiment_metrics": {
            "dataset_mode": "workspace_coverage",
            "accepted_sample_count": 1,
            "rejected_sample_count": 0,
            "run_trust_mode": "thesis_trusted",
            "valid_for_model_training": True,
        },
    }
    row = {
        "accepted": True,
        "resolved_cable_command_cm": [0.1, 0.2, 0.3, 0.4],
        "tip_position_xyz_mm": [1.0, 2.0, 3.0],
        "tip_tangent_xyz": [0.0, 0.0, 1.0],
    }
    (mock_run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (mock_run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with (mock_run / "modeling_dataset_export.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")

    catalog = discover_modeling_datasets(
        project_root=tmp_path,
        output_root=tmp_path / "data" / "experiments",
        include_mock_experiments=True,
    )
    mock_entry = next(entry for entry in catalog if "mock_pose" in entry.run_name)
    assert mock_entry.dataset_scan_root == "mock"
    assert mock_entry.trainable_for_legacy_ann is False
    assert not effective_legacy_ann_training_allowed(mock_entry, allow_mock_training=False)
    assert effective_legacy_ann_training_allowed(mock_entry, allow_mock_training=True)


def test_discover_modeling_datasets_lists_incomplete_run_with_reason(tmp_path: Path) -> None:
    run_dir = tmp_path / "data" / "experiments" / "collect_pose_command_dataset" / "incomplete_run"
    run_dir.mkdir(parents=True)
    metadata = {"experiment_name": "collect_pose_command_dataset", "run_id": "x", "timestamp_utc": "2026-01-01T00:00:00Z"}
    summary = {"status": "success", "sample_counts": {"total": 0}, "experiment_metrics": {}}
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    catalog = discover_modeling_datasets(project_root=tmp_path)
    entry = next(e for e in catalog if e.run_name == "incomplete_run")
    assert entry.export_jsonl_path is None
    assert any("No modeling_dataset_export.jsonl" in reason for reason in entry.legacy_ann_rejection_reasons)
    assert entry.trainable_for_legacy_ann is False


def test_prepare_legacy_ann_dataset_filters_invalid_rows(tmp_path: Path) -> None:
    run_dir = _write_modeling_run(tmp_path)

    summary = load_modeling_dataset_summary(run_dir)
    prepared = prepare_legacy_ann_dataset(run_dir)

    assert summary.accepted_count == 3
    assert summary.rejected_count == 1
    assert summary.trainable_for_legacy_ann is True
    assert summary.accepted_legacy_trainable_count >= 1
    assert summary.full_pose_available is True
    assert summary.sequential_context_available is True
    assert prepared.inputs.shape == (2, 4)
    assert prepared.outputs.shape == (2, 6)
    assert prepared.filtered_reason_counts == {"legacy_tangent_threshold": 1}


def test_build_grouped_split_uses_ordered_step_groups(tmp_path: Path) -> None:
    run_dir = _write_modeling_run(tmp_path)
    prepared = prepare_legacy_ann_dataset(run_dir)
    config = AnnTrainingConfig(train_ratio=0.5, validation_ratio=0.25, test_ratio=0.25)

    split = build_grouped_split(prepared, config)

    assert split.strategy == "ordered_step_group"
    assert split.train_indices == [0, 1]
    assert split.validation_indices == []
    assert split.test_indices == []


def test_detect_training_backends_without_torch(monkeypatch) -> None:
    def _fail():
        raise training_module.TorchUnavailableError("torch missing")

    monkeypatch.setattr(training_module, "_require_torch", _fail)

    report = detect_training_backends()

    assert report.torch_available is False
    assert report.selected_backend == "cpu"
    assert any(option.name == "cpu" for option in report.backend_options)


def test_detect_training_backends_falls_back_to_cpu_when_accelerators_unavailable(monkeypatch) -> None:
    class _FakeMps:
        @staticmethod
        def is_available() -> bool:
            return False

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _FakeTorch:
        __version__ = "2.fake"
        cuda = _FakeCuda()

        class backends:
            mps = _FakeMps()

    monkeypatch.setattr(training_module, "_require_torch", lambda: _FakeTorch())

    report = detect_training_backends(preferred_backend="mps")

    assert report.torch_available is True
    assert report.selected_backend == "cpu"
    assert report.selected_dtype == "float64"
    assert next(option for option in report.backend_options if option.name == "mps").available is False


def test_validate_training_config_rejects_invalid_split() -> None:
    config = AnnTrainingConfig(train_ratio=0.6, validation_ratio=0.3, test_ratio=0.3)

    try:
        validate_training_config(config)
    except ValueError as exc:
        assert "sum to 1.0" in str(exc)
    else:
        raise AssertionError("Expected invalid split configuration to raise.")


def test_discover_trained_artifacts_and_loss_history_round_trip(tmp_path: Path) -> None:
    artifact_root = tmp_path / "data" / "models" / "ann"
    artifact_dir = artifact_root / "20260419_120100_legacy_ann"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at_utc": "2026-04-19T12:01:00+00:00",
        "status": "completed",
        "dataset": {"run_name": "dataset_a"},
        "backend": {"selected_backend": "mps"},
        "training": {"epochs_completed": 8, "best_validation_loss": 0.123},
        "files": {"model_path": str(artifact_dir / "model.pt")},
    }
    (artifact_dir / "training_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (artifact_dir / "model.pt").write_bytes(b"fake")
    (artifact_dir / "loss_history.csv").write_text(
        "epoch,train_loss,validation_loss,elapsed_s\n1,1.0,1.1,0.1\n2,0.8,0.9,0.2\n",
        encoding="utf-8",
    )

    artifacts = discover_trained_artifacts(artifact_root=artifact_root)
    train_losses, validation_losses = load_loss_history(artifact_dir)

    assert len(artifacts) == 1
    assert artifacts[0].backend_name == "mps"
    assert train_losses == [1.0, 0.8]
    assert validation_losses == [1.1, 0.9]


def test_ann_training_config_default_hidden_layers() -> None:
    assert AnnTrainingConfig().hidden_layers == [32, 32]


def test_parse_hidden_layers_text_accepts_semicolon_and_spacing() -> None:
    got, err = parse_hidden_layers_text("64; 64 , 32")
    assert err is None
    assert got == [64, 64, 32]


def test_parse_hidden_layers_text_rejects_non_positive() -> None:
    _got, err = parse_hidden_layers_text("32, 0")
    assert err is not None and "positive" in str(err).lower()


def test_parse_hidden_layers_text_rejects_garbage() -> None:
    _got, err = parse_hidden_layers_text("32, x")
    assert err is not None


def test_format_hidden_layers_for_ann_ui() -> None:
    assert format_hidden_layers_for_ann_ui([32, 32]) == "32, 32"


def test_render_summary_includes_hidden_layers() -> None:
    text = training_module._render_summary_text(
        {
            "status": "completed",
            "created_at_utc": "2026-01-01T00:00:00Z",
            "dataset": {"run_name": "r", "dataset_mode": "m", "prepared_sample_count": 1},
            "backend": {"selected_backend": "cpu", "platform_summary": "test"},
            "model": {"hidden_layers": [128, 128]},
            "training": {"epochs_completed": 1, "best_epoch": 1, "best_validation_loss": 0.1, "test_loss": 0.2},
        }
    )
    assert "[128, 128]" in text


def test_build_legacy_ann_model_respects_128_hidden_layers() -> None:
    torch = pytest.importorskip("torch")
    model = training_module._build_legacy_ann_model(
        torch=torch,
        input_dim=4,
        output_dim=6,
        hidden_layers=[128, 128],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert model.input.in_features == 4
    assert model.input.out_features == 128
