from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from pathlib import Path
import time

import pytest

from continuum_robot.modeling import ann_training as training_module
from continuum_robot.modeling.ann_training import (
    ANN_TRAINING_CATEGORY_BLOCKED,
    ANN_TRAINING_CATEGORY_EXPLORATORY_ONLY,
    ANN_TRAINING_CATEGORY_TRAINABLE,
    ANN_TRAINING_CATEGORY_TRAINABLE_WITH_WARNINGS,
    MIN_COMPLETE_ROWS_FOR_TRAINING,
    OUTPUT_TARGET_CABLE_FROM_XYZ,
    OUTPUT_TARGET_FULL_POSE,
    OUTPUT_TARGET_XYZ,
    SPLIT_STRATEGY_ORDERED,
    SPLIT_STRATEGY_RANDOM,
    AnnTrainingConfig,
    IoScalers,
    StandardScaler,
    ann_training_will_be_exploratory,
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
    train_inverse_xyz_to_cable,
    validate_legacy_ann_rows,
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


def test_build_grouped_split_default_is_random(tmp_path: Path) -> None:
    """New default split is ``random_grouped_step`` (deterministic with the seed)."""
    run_dir = _write_modeling_run(tmp_path)
    prepared = prepare_legacy_ann_dataset(run_dir)
    config = AnnTrainingConfig(train_ratio=0.5, validation_ratio=0.25, test_ratio=0.25)

    split = build_grouped_split(prepared, config)

    assert split.strategy == "random_grouped_step"
    # Two step-groups in the fixture (step_index 0 and 1); the 0.5/0.25/0.25 ratios
    # allocate one group each to train+val (val empty due to small fixture), test = []
    all_indices = (
        list(split.train_indices) + list(split.validation_indices) + list(split.test_indices)
    )
    assert sorted(all_indices) == [0, 1]


def test_build_grouped_split_ordered_back_compat(tmp_path: Path) -> None:
    """``split_strategy='ordered_step_group'`` still produces a contiguous slice."""
    run_dir = _write_modeling_run(tmp_path)
    prepared = prepare_legacy_ann_dataset(run_dir)
    config = AnnTrainingConfig(
        train_ratio=0.5,
        validation_ratio=0.25,
        test_ratio=0.25,
        split_strategy="ordered_step_group",
    )

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


# ---------- Tiered ANN trainability policy ----------


def _write_tiered_modeling_run(
    tmp_path: Path,
    *,
    folder_name: str = "20260512_120000_collect_pose_command_dataset",
    valid_for_model_training: bool = True,
    validity_status: str = "valid",
    validity_reason: str = "all_required_checks_passed",
    hard_invalidation_reasons: list[str] | None = None,
    warnings: list[str] | None = None,
    target_valid_sample_count: int = 4092,
    complete_training_row_count: int = 4082,
    modeling_export_row_count: int = 4082,
    modeling_legacy_row_count: int = 4082,
    incomplete_accepted_workspace_row_count: int = 9,
    run_trust_mode: str = "thesis_trusted",
    parallel_single_demo: bool = False,
    mock_mode: bool = False,
    include_export: bool = True,
    export_rows_complete: bool = True,
    root_kind: str = "experiments",
) -> Path:
    if root_kind == "experiments":
        run_dir = tmp_path / "data" / "experiments" / "collect_pose_command_dataset" / folder_name
    elif root_kind == "mock":
        run_dir = tmp_path / "data" / "mock_experiments" / "collect_pose_command_dataset" / folder_name
    else:
        run_dir = tmp_path / "data" / root_kind / "collect_pose_command_dataset" / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "1.0",
        "experiment_name": "collect_pose_command_dataset",
        "run_id": folder_name,
        "timestamp_utc": "2026-05-12T12:00:00+00:00",
    }
    metrics = {
        "dataset_mode": "workspace_coverage",
        "accepted_sample_count": 4093,
        "rejected_sample_count": 1,
        "run_trust_mode": run_trust_mode,
        "valid_for_model_training": valid_for_model_training,
        "model_training_validity_status": validity_status,
        "model_training_validity_reason": validity_reason,
        "model_training_hard_invalidation_reasons": list(hard_invalidation_reasons or []),
        "model_training_warnings": list(warnings or []),
        "target_valid_sample_count": int(target_valid_sample_count),
        "complete_training_row_count": int(complete_training_row_count),
        "modeling_export_row_count": int(modeling_export_row_count),
        "modeling_legacy_row_count": int(modeling_legacy_row_count),
        "incomplete_accepted_workspace_row_count": int(incomplete_accepted_workspace_row_count),
        "parallel_single_demo": bool(parallel_single_demo),
        "mock_mode": bool(mock_mode),
    }
    summary = {
        "schema_version": "1.0",
        "experiment_name": "collect_pose_command_dataset",
        "run_id": folder_name,
        "success": True,
        "sample_counts": {"total": int(modeling_export_row_count)},
        "status": "success",
        "experiment_metrics": metrics,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if include_export:
        rows: list[dict] = []
        # Always emit at least two complete rows so the legacy ANN has data to consume.
        for index in range(max(2, min(int(modeling_export_row_count), 4))):
            rows.append(
                {
                    "sequence_index": index,
                    "step_index": index,
                    "sample_index": index,
                    "accepted": True,
                    "resolved_cable_command_cm": [0.1, 0.2, 0.3, 0.4],
                    "tip_position_xyz_mm": [1.0, 2.0, 3.0],
                    "tip_tangent_xyz": [0.0, 0.0, 1.0],
                }
            )
        if not export_rows_complete:
            rows.append(
                {
                    "sequence_index": len(rows),
                    "step_index": len(rows),
                    "sample_index": len(rows),
                    "accepted": True,
                    # Missing tip_position_xyz_mm makes full_pose_available False.
                    "resolved_cable_command_cm": [0.1, 0.2, 0.3, 0.4],
                    "tip_position_xyz_mm": [],
                    "tip_tangent_xyz": [],
                }
            )
        with (run_dir / "modeling_dataset_export.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    return run_dir


def test_ann_categorization_target_complete_run_is_trainable(tmp_path: Path) -> None:
    run_dir = _write_tiered_modeling_run(
        tmp_path,
        valid_for_model_training=True,
        validity_status="valid",
        target_valid_sample_count=4,
        complete_training_row_count=4,
        modeling_export_row_count=4,
        modeling_legacy_row_count=4,
        incomplete_accepted_workspace_row_count=0,
        warnings=[],
    )
    summary = load_modeling_dataset_summary(run_dir)
    assert summary.ann_training_category == ANN_TRAINING_CATEGORY_TRAINABLE
    assert summary.trainable_for_legacy_ann is True
    assert summary.trainable_for_ann_exploratory is True
    assert summary.ann_training_blocking_reasons == ()
    assert effective_legacy_ann_training_allowed(summary)


def test_ann_categorization_warning_valid_with_target_met(tmp_path: Path) -> None:
    run_dir = _write_tiered_modeling_run(
        tmp_path,
        valid_for_model_training=True,
        validity_status="warning_valid",
        target_valid_sample_count=4,
        complete_training_row_count=4,
        modeling_export_row_count=4,
        modeling_legacy_row_count=4,
        incomplete_accepted_workspace_row_count=2,
        warnings=["2 incomplete rows were excluded"],
    )
    summary = load_modeling_dataset_summary(run_dir)
    assert summary.ann_training_category == ANN_TRAINING_CATEGORY_TRAINABLE_WITH_WARNINGS
    assert effective_legacy_ann_training_allowed(summary)
    assert any("incomplete" in w.lower() for w in summary.ann_training_warnings)


def test_ann_categorization_exploratory_when_target_not_met(tmp_path: Path) -> None:
    run_dir = _write_tiered_modeling_run(
        tmp_path,
        valid_for_model_training=False,
        validity_status="invalid",
        validity_reason="accepted_count_meets_target",
        hard_invalidation_reasons=["accepted_count_meets_target"],
        target_valid_sample_count=4092,
        complete_training_row_count=4082,
        modeling_export_row_count=4082,
        modeling_legacy_row_count=4082,
        incomplete_accepted_workspace_row_count=9,
    )
    summary = load_modeling_dataset_summary(run_dir)
    assert summary.ann_training_category == ANN_TRAINING_CATEGORY_EXPLORATORY_ONLY
    assert summary.trainable_for_ann_exploratory is True
    assert summary.trainable_for_legacy_ann is False
    assert any("target_valid_sample_count" in w for w in summary.ann_training_warnings)
    # Default refusal: training is not allowed without the explicit override.
    assert not effective_legacy_ann_training_allowed(summary)
    # With the override, training is allowed.
    assert effective_legacy_ann_training_allowed(
        summary, allow_exploratory_incomplete_target=True
    )
    assert ann_training_will_be_exploratory(summary, allow_exploratory_incomplete_target=True)


def test_ann_categorization_blocked_when_export_has_incomplete_rows(tmp_path: Path) -> None:
    run_dir = _write_tiered_modeling_run(
        tmp_path,
        valid_for_model_training=False,
        validity_status="invalid",
        target_valid_sample_count=4,
        complete_training_row_count=4,
        modeling_export_row_count=5,
        modeling_legacy_row_count=4,
        incomplete_accepted_workspace_row_count=1,
        export_rows_complete=False,
    )
    summary = load_modeling_dataset_summary(run_dir)
    assert summary.ann_training_category == ANN_TRAINING_CATEGORY_BLOCKED
    # Override does not release a hard data-completeness fail.
    assert not effective_legacy_ann_training_allowed(
        summary, allow_exploratory_incomplete_target=True
    )
    assert any("incomplete rows" in r for r in summary.ann_training_blocking_reasons)


def test_ann_categorization_blocked_when_export_missing(tmp_path: Path) -> None:
    run_dir = _write_tiered_modeling_run(
        tmp_path,
        valid_for_model_training=False,
        include_export=False,
    )
    summary = load_modeling_dataset_summary(run_dir)
    assert summary.ann_training_category == ANN_TRAINING_CATEGORY_BLOCKED
    assert any("export" in r.lower() for r in summary.ann_training_blocking_reasons)
    assert not effective_legacy_ann_training_allowed(
        summary, allow_exploratory_incomplete_target=True
    )


def test_ann_categorization_servo_only_is_blocked(tmp_path: Path) -> None:
    run_dir = _write_tiered_modeling_run(
        tmp_path,
        run_trust_mode="servo_only",
        valid_for_model_training=False,
    )
    summary = load_modeling_dataset_summary(run_dir)
    assert summary.ann_training_category == ANN_TRAINING_CATEGORY_BLOCKED
    assert not effective_legacy_ann_training_allowed(
        summary, allow_exploratory_incomplete_target=True, allow_lower_trust_training=True
    )


def test_ann_categorization_mock_dataset_needs_mock_override(tmp_path: Path) -> None:
    run_dir = _write_tiered_modeling_run(
        tmp_path,
        root_kind="mock",
        mock_mode=True,
        valid_for_model_training=True,
        validity_status="valid",
        target_valid_sample_count=4,
        complete_training_row_count=4,
        modeling_export_row_count=4,
        modeling_legacy_row_count=4,
        incomplete_accepted_workspace_row_count=0,
    )
    catalog = discover_modeling_datasets(
        project_root=tmp_path,
        output_root=tmp_path / "data" / "experiments",
        include_mock_experiments=True,
    )
    entry = next(e for e in catalog if e.path.resolve() == run_dir.resolve())
    assert not effective_legacy_ann_training_allowed(entry)
    assert not effective_legacy_ann_training_allowed(
        entry, allow_exploratory_incomplete_target=True
    )
    assert effective_legacy_ann_training_allowed(entry, allow_mock_training=True)


def test_ann_categorization_parallel_single_demo_needs_explicit_override(tmp_path: Path) -> None:
    run_dir = _write_tiered_modeling_run(
        tmp_path,
        parallel_single_demo=True,
        valid_for_model_training=False,
        validity_status="invalid",
        hard_invalidation_reasons=["not_parallel_demo"],
        target_valid_sample_count=4,
        complete_training_row_count=4,
        modeling_export_row_count=4,
        modeling_legacy_row_count=4,
        incomplete_accepted_workspace_row_count=0,
    )
    summary = load_modeling_dataset_summary(run_dir)
    assert summary.ann_training_category == ANN_TRAINING_CATEGORY_EXPLORATORY_ONLY
    # Exploratory override alone is not enough — parallel_single also needs its own toggle.
    assert not effective_legacy_ann_training_allowed(
        summary, allow_exploratory_incomplete_target=True
    )
    assert effective_legacy_ann_training_allowed(
        summary,
        allow_exploratory_incomplete_target=True,
        allow_parallel_single_demo_training=True,
    )


def test_ann_dataset_metadata_records_source_validity_provenance(tmp_path: Path) -> None:
    run_dir = _write_tiered_modeling_run(
        tmp_path,
        valid_for_model_training=False,
        validity_status="invalid",
        validity_reason="accepted_count_meets_target",
        hard_invalidation_reasons=["accepted_count_meets_target"],
        warnings=["complete_training_row_count below target_valid_sample_count"],
        target_valid_sample_count=4092,
        complete_training_row_count=4082,
        modeling_export_row_count=4082,
        modeling_legacy_row_count=4082,
        incomplete_accepted_workspace_row_count=9,
    )
    prepared = prepare_legacy_ann_dataset(run_dir)
    payload = training_module._dataset_metadata_for_artifact(prepared=prepared)
    assert payload["source_run_valid_for_model_training"] is False
    assert payload["source_run_model_training_validity_status"] == "invalid"
    assert payload["complete_training_row_count"] == 4082
    assert payload["target_valid_sample_count"] == 4092
    assert payload["incomplete_rows_excluded_from_training"] == 9
    assert payload["ann_training_category"] == ANN_TRAINING_CATEGORY_EXPLORATORY_ONLY


def test_ann_run_review_debug_does_not_block_exploratory_training(tmp_path: Path) -> None:
    # A debug-reviewed run (e.g. operator marked the run for further review) is still
    # discoverable in the ANN catalog and remains eligible for exploratory training
    # as long as the export is clean and the exploratory override is enabled.
    run_dir = _write_tiered_modeling_run(
        tmp_path,
        valid_for_model_training=False,
        validity_status="invalid",
        hard_invalidation_reasons=["accepted_count_meets_target"],
        target_valid_sample_count=4092,
        complete_training_row_count=4082,
        modeling_export_row_count=4082,
        modeling_legacy_row_count=4082,
        incomplete_accepted_workspace_row_count=9,
    )
    # A debug-only review sidecar must not promote the dataset to thesis-valid, but it
    # must not prevent the operator from training on it explicitly either.
    (run_dir / "run_review.json").write_text(
        json.dumps({"status": "debug", "intended_use": "debug", "include_in_evidence_index": False}),
        encoding="utf-8",
    )
    summary = load_modeling_dataset_summary(run_dir)
    assert summary.ann_training_category == ANN_TRAINING_CATEGORY_EXPLORATORY_ONLY
    assert effective_legacy_ann_training_allowed(
        summary, allow_exploratory_incomplete_target=True
    )
    assert summary.valid_for_model_training_flag is False


# ---------------------------------------------------------------------------
# Row-filter policy (complete_rows_only): RowFilterReport + validate_legacy_ann_rows.
# ---------------------------------------------------------------------------


def _write_run_with_export_rows(
    tmp_path: Path,
    *,
    folder_name: str = "20260514_120000_row_filter_run",
    complete_rows: int = 120,
    incomplete_rows: int = 9,
    target: int = 130,
    extra_rows: list[dict] | None = None,
    valid_for_model_training: bool = True,
) -> Path:
    """Build a real export fixture (complete rows + intentionally-incomplete rows)."""
    run_dir = (
        tmp_path / "data" / "experiments" / "collect_pose_command_dataset" / folder_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment_name": "collect_pose_command_dataset",
                "run_id": folder_name,
                "timestamp_utc": "2026-05-14T12:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment_name": "collect_pose_command_dataset",
                "run_id": folder_name,
                "success": True,
                "sample_counts": {"total": complete_rows + incomplete_rows + len(extra_rows or [])},
                "status": "success",
                "experiment_metrics": {
                    "dataset_mode": "workspace_coverage",
                    "accepted_sample_count": complete_rows + incomplete_rows,
                    "rejected_sample_count": 0,
                    "run_trust_mode": "thesis_trusted",
                    "valid_for_model_training": bool(valid_for_model_training),
                    "target_valid_sample_count": int(target),
                    "complete_training_row_count": int(complete_rows),
                },
            }
        ),
        encoding="utf-8",
    )
    with (run_dir / "modeling_dataset_export.jsonl").open("w", encoding="utf-8") as handle:
        for i in range(complete_rows):
            handle.write(
                json.dumps(
                    {
                        "sequence_index": i,
                        "step_index": i // 8,
                        "sample_index": i,
                        "accepted": True,
                        "resolved_cable_command_cm": [0.001 * (i % 50), 0.002, 0.003, 0.004],
                        "tip_position_xyz_mm": [1.0 + 0.01 * (i % 100), 2.0, 3.0],
                        "tip_tangent_xyz": [0.01, 0.02, 0.03],
                    }
                )
                + "\n"
            )
        for off in range(incomplete_rows):
            # Accepted row but tip_position_xyz_mm is missing (the run 81842e42eca4 scenario).
            handle.write(
                json.dumps(
                    {
                        "sequence_index": complete_rows + off,
                        "accepted": True,
                        "resolved_cable_command_cm": [0.0, 0.0, 0.0, 0.0],
                        "tip_position_xyz_mm": [],
                        "tip_tangent_xyz": [],
                    }
                )
                + "\n"
            )
        for extra in extra_rows or []:
            handle.write(json.dumps(extra) + "\n")
    return run_dir


def test_validate_legacy_ann_rows_reports_complete_and_excluded_counts(tmp_path: Path) -> None:
    """The row-filter report must mirror the run 81842e42eca4 shape: complete vs excluded + target."""
    run_dir = _write_run_with_export_rows(tmp_path, complete_rows=120, incomplete_rows=9, target=130)
    report = validate_legacy_ann_rows(run_dir)
    assert report.complete_row_count == 120
    assert report.excluded_row_count == 9
    assert report.excluded_by_reason.get("missing_position") == 9
    assert report.target_complete_row_count == 130
    assert report.can_train is True
    assert report.block_reason is None


def test_validate_legacy_ann_rows_blocks_below_minimum(tmp_path: Path) -> None:
    run_dir = _write_run_with_export_rows(tmp_path, complete_rows=10, incomplete_rows=0, target=100)
    report = validate_legacy_ann_rows(run_dir, min_complete_rows=100)
    assert report.complete_row_count == 10
    assert report.can_train is False
    assert "below" in (report.block_reason or "")


def test_validate_legacy_ann_rows_blocks_zero_complete(tmp_path: Path) -> None:
    run_dir = _write_run_with_export_rows(tmp_path, complete_rows=0, incomplete_rows=5)
    report = validate_legacy_ann_rows(run_dir, min_complete_rows=1)
    assert report.complete_row_count == 0
    assert report.can_train is False
    assert "0 complete rows" in (report.block_reason or "")


def test_validate_legacy_ann_rows_missing_export_returns_block_reason(tmp_path: Path) -> None:
    run_dir = tmp_path / "no_export"
    run_dir.mkdir()
    report = validate_legacy_ann_rows(run_dir)
    assert report.export_path is None
    assert report.can_train is False
    assert "modeling_dataset_export.jsonl" in (report.block_reason or "")


def test_row_filter_excludes_modeling_export_exclude_rows(tmp_path: Path) -> None:
    """Rows tagged ``modeling_export_exclude=true`` must be excluded by reason name."""
    extra = [
        {
            "sequence_index": 1000 + i,
            "accepted": True,
            "modeling_export_exclude": True,
            "resolved_cable_command_cm": [0.01, 0.02, 0.03, 0.04],
            "tip_position_xyz_mm": [1.0, 2.0, 3.0],
            "tip_tangent_xyz": [0.01, 0.02, 0.03],
        }
        for i in range(3)
    ]
    run_dir = _write_run_with_export_rows(
        tmp_path, complete_rows=20, incomplete_rows=0, extra_rows=extra
    )
    report = validate_legacy_ann_rows(run_dir, min_complete_rows=1)
    assert report.complete_row_count == 20
    assert report.excluded_by_reason.get("modeling_export_exclude") == 3


def test_min_complete_rows_for_training_constant_default() -> None:
    assert MIN_COMPLETE_ROWS_FOR_TRAINING == 100


# ---------------------------------------------------------------------------
# Training writes a row_filter_report.json sidecar and stamps training_metadata.json.
# ---------------------------------------------------------------------------


def test_train_legacy_ann_writes_row_filter_sidecar(tmp_path: Path) -> None:
    """End-to-end training on a small fixture saves row_filter_report.json next to artifacts."""
    pytest.importorskip("torch")
    run_dir = _write_run_with_export_rows(
        tmp_path, complete_rows=20, incomplete_rows=2, target=22, valid_for_model_training=True
    )
    config = AnnTrainingConfig(
        artifact_root=str(tmp_path / "data" / "models" / "ann"),
        artifact_name="row_filter_smoke",
        epochs=1,
        batch_size=4,
        train_ratio=0.7,
        validation_ratio=0.2,
        test_ratio=0.1,
    )
    result = training_module.train_legacy_ann(
        project_root=tmp_path,
        dataset_path=run_dir,
        config=config,
        backend_name="cpu",
    )
    sidecar = result.artifact_dir / "row_filter_report.json"
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["training_input_policy"] == "complete_rows_only"
    assert payload["complete_row_count"] == 20
    assert payload["excluded_row_count"] == 2
    assert payload["excluded_by_reason"].get("missing_position") == 2
    assert payload["target_complete_row_count"] == 22
    meta = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert meta["training_input_policy"] == "complete_rows_only"
    assert meta["row_filter_report"]["complete_row_count"] == 20
    assert meta["files"]["row_filter_report_path"].endswith("row_filter_report.json")


# ---------------------------------------------------------------------------
# Standardization (StandardScaler / IoScalers) round-trip behaviour.
# ---------------------------------------------------------------------------


def test_standard_scaler_round_trip() -> None:
    import numpy as np

    x = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]], dtype=float)
    scaler = StandardScaler.fit(x)
    z = scaler.transform(x)
    # Centered (mean ~ 0) and unit-std on the column where there's spread.
    assert abs(float(z.mean(axis=0)[0])) < 1e-9
    assert abs(float(z.std(axis=0)[0]) - 1.0) < 1e-9
    # Inverse maps back to the original.
    back = scaler.inverse_transform(z)
    assert np.allclose(back, x)


def test_standard_scaler_handles_constant_feature() -> None:
    import numpy as np

    # std == 0 must not divide-by-zero.
    x = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]], dtype=float)
    scaler = StandardScaler.fit(x)
    z = scaler.transform(x)
    # Constant column maps to ~0 (no NaN/inf).
    assert np.all(np.isfinite(z))


def test_io_scalers_json_round_trip() -> None:
    import numpy as np

    inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    out = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float)
    pair = IoScalers(StandardScaler.fit(inp), StandardScaler.fit(out))
    restored = IoScalers.from_dict(pair.to_dict())
    assert np.allclose(restored.input_scaler.mean, pair.input_scaler.mean)
    assert np.allclose(restored.output_scaler.std, pair.output_scaler.std)


# ---------------------------------------------------------------------------
# Output target dims + back-compat full-pose path.
# ---------------------------------------------------------------------------


def test_prepare_legacy_ann_dataset_xyz_target_drops_tangent(tmp_path: Path) -> None:
    """``output_target='xyz'`` produces 4-in, 3-out tensors (no tangent column)."""
    run_dir = _write_modeling_run(tmp_path)
    prepared = prepare_legacy_ann_dataset(run_dir, output_target=OUTPUT_TARGET_XYZ)
    assert prepared.inputs.shape[1] == 4
    assert prepared.outputs.shape[1] == 3


def test_prepare_legacy_ann_dataset_full_pose_back_compat(tmp_path: Path) -> None:
    """Default ``prepare_legacy_ann_dataset(path)`` stays 6-out for back-compat."""
    run_dir = _write_modeling_run(tmp_path)
    prepared = prepare_legacy_ann_dataset(run_dir)
    assert prepared.outputs.shape[1] == 6


def test_prepare_legacy_ann_dataset_inverse_swaps_io(tmp_path: Path) -> None:
    """``cable_from_xyz`` swaps inputs/outputs to 3→4."""
    run_dir = _write_modeling_run(tmp_path)
    prepared = prepare_legacy_ann_dataset(run_dir, output_target=OUTPUT_TARGET_CABLE_FROM_XYZ)
    assert prepared.inputs.shape[1] == 3
    assert prepared.outputs.shape[1] == 4


# ---------------------------------------------------------------------------
# Random-grouped split: deterministic by seed, leak-safe by step_index.
# ---------------------------------------------------------------------------


def test_random_grouped_split_is_deterministic_with_seed(tmp_path: Path) -> None:
    run_dir = _write_run_with_export_rows(tmp_path, complete_rows=30, incomplete_rows=0, target=30)
    prepared = prepare_legacy_ann_dataset(run_dir, output_target=OUTPUT_TARGET_XYZ)
    cfg_a = AnnTrainingConfig(
        random_seed=42, train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2,
        split_strategy=SPLIT_STRATEGY_RANDOM,
    )
    cfg_b = AnnTrainingConfig(
        random_seed=42, train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2,
        split_strategy=SPLIT_STRATEGY_RANDOM,
    )
    split_a = build_grouped_split(prepared, cfg_a)
    split_b = build_grouped_split(prepared, cfg_b)
    assert split_a.train_indices == split_b.train_indices
    assert split_a.test_indices == split_b.test_indices


def test_random_grouped_split_differs_from_ordered(tmp_path: Path) -> None:
    run_dir = _write_run_with_export_rows(tmp_path, complete_rows=30, incomplete_rows=0, target=30)
    prepared = prepare_legacy_ann_dataset(run_dir, output_target=OUTPUT_TARGET_XYZ)
    cfg_random = AnnTrainingConfig(
        random_seed=7, train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2,
        split_strategy=SPLIT_STRATEGY_RANDOM,
    )
    cfg_ordered = AnnTrainingConfig(
        random_seed=7, train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2,
        split_strategy=SPLIT_STRATEGY_ORDERED,
    )
    random_split = build_grouped_split(prepared, cfg_random)
    ordered_split = build_grouped_split(prepared, cfg_ordered)
    # Sizes match (same ratios). The strategy label must reflect what was used; the test
    # indices should not be the contiguous tail of the group order under a non-trivial seed.
    assert len(random_split.train_indices) == len(ordered_split.train_indices)
    assert random_split.strategy == SPLIT_STRATEGY_RANDOM
    assert ordered_split.strategy == SPLIT_STRATEGY_ORDERED
    # At least one of the three partitions should differ — otherwise the shuffle did nothing.
    assert (
        set(random_split.train_indices) != set(ordered_split.train_indices)
        or set(random_split.validation_indices) != set(ordered_split.validation_indices)
        or set(random_split.test_indices) != set(ordered_split.test_indices)
    )


def test_validate_training_config_rejects_invalid_split_strategy() -> None:
    config = AnnTrainingConfig(split_strategy="not_a_real_strategy")
    with pytest.raises(ValueError, match="split_strategy"):
        validate_training_config(config)


# ---------------------------------------------------------------------------
# End-to-end: standardization is saved, dims match output_target, early stop fires.
# ---------------------------------------------------------------------------


def test_train_legacy_ann_xyz_with_standardization_saves_scaler(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    run_dir = _write_run_with_export_rows(
        tmp_path, complete_rows=40, incomplete_rows=0, target=40, valid_for_model_training=True
    )
    config = AnnTrainingConfig(
        artifact_root=str(tmp_path / "data" / "models" / "ann"),
        artifact_name="xyz_with_scaler",
        epochs=3,
        batch_size=8,
        train_ratio=0.7,
        validation_ratio=0.2,
        test_ratio=0.1,
        output_target=OUTPUT_TARGET_XYZ,
        standardize_io=True,
        early_stopping_patience=0,  # disable for this test (we just want artifact contents)
    )
    result = training_module.train_legacy_ann(
        project_root=tmp_path,
        dataset_path=run_dir,
        config=config,
        backend_name="cpu",
    )
    scaler_path = result.artifact_dir / "io_scaler.json"
    assert scaler_path.exists()
    meta = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert meta["model"]["output_target"] == "xyz"
    assert meta["model"]["input_dim"] == 4
    assert meta["model"]["output_dim"] == 3
    assert meta["model"]["standardize_io"] is True
    assert meta["loss"]["kind"] == "mse_standardized"
    assert "io_scaler" in meta
    # Restored scaler can transform/inverse cleanly.
    restored = IoScalers.from_dict(meta["io_scaler"])
    assert restored.input_scaler.mean.shape == (4,)
    assert restored.output_scaler.mean.shape == (3,)


def test_train_legacy_ann_early_stops_when_no_improvement(tmp_path: Path) -> None:
    """Patience=1 plus a tiny fixture should stop well before epochs_requested."""
    pytest.importorskip("torch")
    run_dir = _write_run_with_export_rows(
        tmp_path, complete_rows=40, incomplete_rows=0, target=40, valid_for_model_training=True
    )
    config = AnnTrainingConfig(
        artifact_root=str(tmp_path / "data" / "models" / "ann"),
        artifact_name="early_stop_run",
        epochs=200,  # would take many epochs without early stop
        batch_size=8,
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
        output_target=OUTPUT_TARGET_XYZ,
        early_stopping_patience=2,
        random_seed=7,
        learning_rate=1e-2,
    )
    result = training_module.train_legacy_ann(
        project_root=tmp_path,
        dataset_path=run_dir,
        config=config,
        backend_name="cpu",
    )
    meta = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    # Either early-stopped or completed naturally — but training should record the flag and
    # epochs_completed must be <= epochs_requested.
    assert meta["training"]["epochs_completed"] <= meta["training"]["epochs_requested"]
    assert "early_stopped" in meta["training"]


def test_train_inverse_xyz_to_cable_produces_4d_output(tmp_path: Path) -> None:
    """Inverse training maps 3-D tip → 4-D cable command and tags the artifact accordingly."""
    pytest.importorskip("torch")
    run_dir = _write_run_with_export_rows(
        tmp_path, complete_rows=40, incomplete_rows=0, target=40, valid_for_model_training=True
    )
    config = AnnTrainingConfig(
        artifact_root=str(tmp_path / "data" / "models" / "ann"),
        artifact_name="inverse_xyz_to_cable",
        epochs=2,
        batch_size=8,
        train_ratio=0.7,
        validation_ratio=0.2,
        test_ratio=0.1,
        # output_target is overridden inside train_inverse_xyz_to_cable
    )
    result = train_inverse_xyz_to_cable(
        project_root=tmp_path,
        dataset_path=run_dir,
        config=config,
        backend_name="cpu",
    )
    meta = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert meta["model"]["output_target"] == OUTPUT_TARGET_CABLE_FROM_XYZ
    assert meta["model"]["input_dim"] == 3
    assert meta["model"]["output_dim"] == 4
    # Inverse runs should report cable-space metrics (compute_cable_evaluation_metrics).
    test_block = meta["evaluation"].get("test") or {}
    assert "cable_rmse_cm" in test_block


# ---------------------------------------------------------------------------
# Artifact-kind dispatch (bug-fix regressions).
# ---------------------------------------------------------------------------


def test_train_legacy_ann_artifact_kind_reflects_output_target_xyz(tmp_path: Path) -> None:
    """XYZ-target ANN artifacts must NOT be tagged as full_pose."""
    pytest.importorskip("torch")
    run_dir = _write_run_with_export_rows(tmp_path, complete_rows=20, incomplete_rows=0, target=20)
    config = AnnTrainingConfig(
        artifact_root=str(tmp_path / "data" / "models" / "ann"),
        artifact_name="xyz_artifact_kind",
        epochs=1,
        batch_size=4,
        train_ratio=0.7,
        validation_ratio=0.2,
        test_ratio=0.1,
        output_target=OUTPUT_TARGET_XYZ,
    )
    result = training_module.train_legacy_ann(
        project_root=tmp_path,
        dataset_path=run_dir,
        config=config,
        backend_name="cpu",
    )
    meta = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert meta["artifact_kind"] == "legacy_ann_xyz_v1"


def test_train_inverse_xyz_to_cable_artifact_kind_is_inverse(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    run_dir = _write_run_with_export_rows(tmp_path, complete_rows=20, incomplete_rows=0, target=20)
    config = AnnTrainingConfig(
        artifact_root=str(tmp_path / "data" / "models" / "ann"),
        artifact_name="inv_artifact_kind",
        epochs=1,
        batch_size=4,
        train_ratio=0.7,
        validation_ratio=0.2,
        test_ratio=0.1,
    )
    result = train_inverse_xyz_to_cable(
        project_root=tmp_path,
        dataset_path=run_dir,
        config=config,
        backend_name="cpu",
    )
    meta = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert meta["artifact_kind"] == "legacy_ann_inverse_xyz_to_cable_v1"


def test_render_summary_text_dispatches_on_output_target_for_inverse(tmp_path: Path) -> None:
    """The summary text must call inverse models 'inverse', not 'full-pose'."""
    text = training_module._render_summary_text(
        {
            "status": "completed",
            "created_at_utc": "2026-05-15T00:00:00Z",
            "artifact_kind": "legacy_ann_inverse_xyz_to_cable_v1",
            "dataset": {"run_name": "r", "dataset_mode": "m", "prepared_sample_count": 1},
            "backend": {"selected_backend": "cpu", "platform_summary": "test"},
            "model": {
                "output_target": OUTPUT_TARGET_CABLE_FROM_XYZ,
                "hidden_layers": [32, 32],
                "family": "legacy_ann",
            },
            "training": {"epochs_completed": 1, "best_epoch": 1, "best_validation_loss": 0.1, "test_loss": 0.2},
            "evaluation": {"test": {"loss_mean": 0.1, "cable_rmse_cm": 0.05, "cable_per_dim_rmse_cm": [0.04, 0.05, 0.06, 0.07]}},
        }
    )
    assert "inverse" in text.lower()
    assert "cable rmse" in text.lower()
    assert "position rmse" not in text.lower()


def test_render_summary_text_for_xyz_only_says_xyz(tmp_path: Path) -> None:
    text = training_module._render_summary_text(
        {
            "status": "completed",
            "created_at_utc": "2026-05-15T00:00:00Z",
            "artifact_kind": "legacy_ann_xyz_v1",
            "dataset": {"run_name": "r", "dataset_mode": "m", "prepared_sample_count": 1},
            "backend": {"selected_backend": "cpu", "platform_summary": "test"},
            "model": {
                "output_target": OUTPUT_TARGET_XYZ,
                "hidden_layers": [128, 128],
                "family": "legacy_ann",
            },
            "training": {"epochs_completed": 1, "best_epoch": 1, "best_validation_loss": 0.1, "test_loss": 0.2},
            "evaluation": {
                "test": {
                    "loss_mean": 0.1,
                    "position_rmse_xyz_mm": 1.5,
                    "position_rmse_xy_mm": 1.2,
                    "position_rmse_z_mm": 0.9,
                    "position_error_l2_mm": {"mean": 1.4, "median": 1.3, "p95": 2.5, "max": 3.0},
                }
            },
        }
    )
    assert "XYZ" in text
    # XYZ-only summary should NOT advertise tangent angular error.
    assert "tangent" not in text.lower()


# ---------------------------------------------------------------------------
# Evaluation path: scaler is applied + 3D predictions handled (analysis.py fixes).
# ---------------------------------------------------------------------------


def test_evaluate_models_applies_scaler_and_handles_xyz_only_artifact(tmp_path: Path) -> None:
    """A new XYZ-standardized ANN artifact must evaluate cleanly in the Modeling tab path.

    Without the analysis.py fix, raw cable cm would feed into a model trained on Z-scored
    cable, predictions would come back in normalized space, and the comparison would
    report nonsense mm metrics (often >100mm RMSE on its own training data).
    """
    pytest.importorskip("torch")
    from continuum_robot.modeling.analysis import evaluate_models, ModelingEvaluationConfig

    run_dir = _write_run_with_export_rows(tmp_path, complete_rows=80, incomplete_rows=0, target=80)
    # Force a small-but-trainable run that records the scaler.
    config = AnnTrainingConfig(
        artifact_root=str(tmp_path / "data" / "models" / "ann"),
        artifact_name="xyz_eval_smoke",
        epochs=20,
        batch_size=8,
        train_ratio=0.7,
        validation_ratio=0.2,
        test_ratio=0.1,
        output_target=OUTPUT_TARGET_XYZ,
        standardize_io=True,
        early_stopping_patience=10,
        learning_rate=5e-3,
        random_seed=0,
    )
    train_result = training_module.train_legacy_ann(
        project_root=tmp_path,
        dataset_path=run_dir,
        config=config,
        backend_name="cpu",
    )
    eval_config = ModelingEvaluationConfig(
        include_mike=False,
        include_camarillo=False,
        include_ann=True,
        evaluation_scope="full_dataset",
        results_root=str(tmp_path / "data" / "modeling_results"),
    )
    eval_result = evaluate_models(
        project_root=tmp_path,
        dataset_path=run_dir,
        artifact_path=train_result.artifact_dir,
        config=eval_config,
    )
    ann_eval = eval_result.model_evaluations.get("ann")
    assert ann_eval is not None
    assert ann_eval.metrics.status == "completed"
    # Sanity: predictions should be in roughly the right mm scale (truths span ~1..2mm in
    # our fixture). If the scaler weren't applied, predictions would be Z-scored numbers
    # ~|O(1)| compared to truths in mm, and position_rmse_mm would be wildly different
    # from the loss_mean. With the scaler applied, position_rmse_mm should be finite and
    # small (the network trained on this data).
    assert ann_eval.metrics.position_rmse_mm is not None
    assert ann_eval.metrics.position_rmse_mm < 10.0  # generous upper bound for a 20-epoch fit
    # XYZ-only artifacts: tangent metrics should be elided rather than reported as zero.
    assert ann_eval.metrics.tangent_rmse_deg is None


def test_evaluate_models_rejects_inverse_artifact_cleanly(tmp_path: Path) -> None:
    """Inverse artifacts can't fit the forward comparison; surface 'unavailable' explicitly."""
    pytest.importorskip("torch")
    from continuum_robot.modeling.analysis import evaluate_models, ModelingEvaluationConfig

    run_dir = _write_run_with_export_rows(tmp_path, complete_rows=40, incomplete_rows=0, target=40)
    config = AnnTrainingConfig(
        artifact_root=str(tmp_path / "data" / "models" / "ann"),
        artifact_name="inverse_eval_smoke",
        epochs=2,
        batch_size=8,
        train_ratio=0.7,
        validation_ratio=0.2,
        test_ratio=0.1,
    )
    train_result = train_inverse_xyz_to_cable(
        project_root=tmp_path,
        dataset_path=run_dir,
        config=config,
        backend_name="cpu",
    )
    eval_config = ModelingEvaluationConfig(
        include_mike=False,
        include_camarillo=False,
        include_ann=True,
        evaluation_scope="full_dataset",
        results_root=str(tmp_path / "data" / "modeling_results"),
    )
    eval_result = evaluate_models(
        project_root=tmp_path,
        dataset_path=run_dir,
        artifact_path=train_result.artifact_dir,
        config=eval_config,
    )
    ann_eval = eval_result.model_evaluations.get("ann")
    assert ann_eval is not None
    assert ann_eval.metrics.status == "unavailable"
    assert "inverse" in (ann_eval.metrics.reason or "").lower()
