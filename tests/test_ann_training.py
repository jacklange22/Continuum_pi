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
    AnnTrainingConfig,
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
