from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")

from continuum_robot.modeling import ann_training as m
from continuum_robot.modeling.ann_training import (
    AnnTrainingConfig,
    TrainingEstimate,
    TrainingResult,
    compute_pose_evaluation_metrics,
    merge_ann_sweep_architectures,
    parse_sweep_extra_hidden_layers_groups,
    prepare_legacy_ann_dataset,
    run_model_sweep,
    select_best_sweep_row_by_test_position_rmse,
)


def test_parse_sweep_extra_hidden_layers_groups() -> None:
    assert parse_sweep_extra_hidden_layers_groups("")[0] == []
    g, err = parse_sweep_extra_hidden_layers_groups("48,48 | 96")
    assert err is None and g == [[48, 48], [96]]
    g2, err2 = parse_sweep_extra_hidden_layers_groups("64,64\n32")
    assert err2 is None and g2 == [[64, 64], [32]]


def test_ann_model_sweep_cli_no_linear_excludes_ridge_baseline(monkeypatch, tmp_path: Path) -> None:
    import scripts.ann_model_sweep as cli

    captured: dict[str, object] = {}

    def _fake_run_model_sweep(**kwargs):
        captured.update(kwargs)
        root = tmp_path / "sweep"
        root.mkdir()
        summary = root / "model_sweep_summary.json"
        summary.write_text("{}", encoding="utf-8")
        return SimpleNamespace(sweep_root=root, summary_json_path=summary, best_model={})

    monkeypatch.setattr(cli, "run_model_sweep", _fake_run_model_sweep)
    assert cli.main(
        [
            "--project-root",
            str(tmp_path),
            "--dataset-path",
            str(tmp_path),
            "--backend",
            "cpu",
            "--no-linear",
        ]
    ) == 0

    assert captured["include_linear_baseline"] is False
    assert parse_sweep_extra_hidden_layers_groups("bad")[0] is None


def test_ann_model_sweep_cli_passes_exploratory_override(monkeypatch, tmp_path: Path) -> None:
    """CLI flag must thread through to ``run_model_sweep.training_provenance``."""
    import scripts.ann_model_sweep as cli

    captured: dict[str, object] = {}

    def _fake_run_model_sweep(**kwargs):
        captured.update(kwargs)
        root = tmp_path / "sweep_explo"
        root.mkdir()
        summary = root / "model_sweep_summary.json"
        summary.write_text("{}", encoding="utf-8")
        return SimpleNamespace(sweep_root=root, summary_json_path=summary, best_model={})

    monkeypatch.setattr(cli, "run_model_sweep", _fake_run_model_sweep)
    assert cli.main(
        [
            "--project-root",
            str(tmp_path),
            "--dataset-path",
            str(tmp_path),
            "--backend",
            "cpu",
            "--allow-exploratory-incomplete-target",
        ]
    ) == 0
    provenance = captured.get("training_provenance")
    assert isinstance(provenance, dict)
    assert provenance.get("exploratory_training_override") is True


def test_merge_ann_sweep_architectures_dedupes() -> None:
    out = merge_ann_sweep_architectures(defaults=[[32, 32], [64, 64]], extras=[[64, 64], [48, 48]])
    assert out == [[32, 32], [64, 64], [48, 48]]


def test_compute_pose_evaluation_metrics_basic() -> None:
    pred = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0, 1.0, 0.0]], dtype=float)
    targ = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=float)
    metrics = compute_pose_evaluation_metrics(pred, targ)
    assert metrics["position_rmse_xyz_mm"] == pytest.approx(np.sqrt(np.mean([0.0, 25.0])))
    assert "mean" in metrics["position_error_l2_mm"]


def test_select_best_sweep_row_by_test_position_rmse() -> None:
    rows = [
        {"model_key": "a", "test_position_rmse_xyz_mm": 2.0, "test_loss": 0.5},
        {"model_key": "b", "test_position_rmse_xyz_mm": 1.0, "test_loss": 0.9},
    ]
    best = select_best_sweep_row_by_test_position_rmse(rows)
    assert best is not None and best["model_key"] == "b"


def _estimate_stub(**_kwargs: object) -> TrainingEstimate:
    return TrainingEstimate(
        estimated_total_s=1.0,
        estimated_epoch_s=1.0,
        train_batch_time_s=0.1,
        validation_batch_time_s=0.1,
        benchmark_train_batches=1,
        benchmark_validation_batches=1,
        train_batch_count=1,
        validation_batch_count=1,
    )


def test_run_model_sweep_shared_split_and_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "data" / "experiments" / "collect_pose_command_dataset" / "sweep_fixture"
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "1.0",
        "experiment_name": "collect_pose_command_dataset",
        "run_id": "s1",
        "timestamp_utc": "2026-05-12T12:00:00+00:00",
        "git_commit": None,
        "backend_info": {},
        "registration_info": {},
        "config_used": {"dataset_mode": "workspace_coverage"},
        "operator_notes": "",
    }
    summary = {
        "schema_version": "1.0",
        "experiment_name": "collect_pose_command_dataset",
        "run_id": "s1",
        "success": True,
        "sample_counts": {"total": 6},
        "dropped_frames": 0,
        "invalid_transforms": 0,
        "stage_pass_fail": {},
        "status": "success",
        "experiment_metrics": {
            "dataset_mode": "workspace_coverage",
            "dataset_mode_summary": "x",
            "accepted_sample_count": 6,
            "rejected_sample_count": 0,
            "accepted_capture_rate": 1.0,
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
    rows_jsonl = []
    for i in range(6):
        rows_jsonl.append(
            {
                "sequence_index": i,
                "step_index": i // 2,
                "sample_index": i,
                "accepted": True,
                "resolved_cable_command_cm": [0.1 * i, 0.2, 0.3, 0.4],
                "tip_position_xyz_mm": [1.0 + i, 2.0, 3.0],
                "tip_tangent_xyz": [0.01, 0.02, 0.03],
                "previous_pair_command_cm": [],
            }
        )
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with (run_dir / "modeling_dataset_export.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows_jsonl:
            handle.write(json.dumps(row) + "\n")

    splits_seen: list[object] = []

    def _fake_linear(*, artifact_dir, split, config, **kwargs: object) -> TrainingResult:
        splits_seen.append(split)
        ad = Path(artifact_dir)
        ad.mkdir(parents=True, exist_ok=True)
        meta = {
            "training": {
                "test_loss": 0.4,
                "training_wall_time_s": 0.01,
                "epochs_completed": 1,
                "best_validation_loss": 0.35,
                "best_epoch": 1,
            },
            "evaluation": {
                "validation": {"loss_mean": 0.33},
                "test": {
                    "loss_mean": 0.31,
                    "position_rmse_xyz_mm": 2.5,
                    "position_rmse_xy_mm": 2.0,
                    "position_rmse_z_mm": 1.0,
                    "position_error_l2_mm": {"mean": 2.5, "median": 2.5, "p95": 2.5, "max": 2.5},
                    "tangent_angular_error_rad": {"mean": 0.1, "median": 0.1, "p95": 0.2, "max": 0.3},
                },
            },
            "model": {"family": "linear_ridge_full_pose", "hidden_layers": []},
        }
        (ad / "training_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        (ad / "training_config.json").write_text(json.dumps({"linear": True}), encoding="utf-8")
        (ad / "loss_history.csv").write_text("epoch,train_loss,validation_loss,elapsed_s\n1,0.3,0.33,0.01\n", encoding="utf-8")
        (ad / "loss_curve.png").write_bytes(b"")
        (ad / "split_manifest.json").write_text("{}", encoding="utf-8")
        (ad / "training_summary.txt").write_text("ok\n", encoding="utf-8")
        (ad / "model.pt").write_bytes(b"")
        return TrainingResult(
            artifact_dir=ad,
            model_path=ad / "model.pt",
            metadata_path=ad / "training_metadata.json",
            loss_history_path=ad / "loss_history.csv",
            loss_plot_path=ad / "loss_curve.png",
            split_manifest_path=ad / "split_manifest.json",
            summary_text_path=ad / "training_summary.txt",
            status="completed",
            best_epoch=1,
            best_validation_loss=0.33,
            test_loss=0.4,
            epochs_completed=1,
            train_losses=[0.3],
            validation_losses=[0.33],
            estimate=_estimate_stub(),
        )

    def _fake_ann(
        *,
        artifact_dir,
        split,
        config,
        **kwargs: object,
    ) -> TrainingResult:
        splits_seen.append(split)
        ad = Path(artifact_dir)
        ad.mkdir(parents=True, exist_ok=True)
        first = int(config.hidden_layers[0])
        rmse = {32: 3.0, 64: 2.0, 128: 1.0}.get(first, 9.0)
        meta = {
            "training": {
                "test_loss": float(0.1 * rmse),
                "training_wall_time_s": 0.02,
                "epochs_completed": 2,
                "best_validation_loss": 0.2,
                "best_epoch": 2,
            },
            "evaluation": {
                "validation": {"loss_mean": 0.25},
                "test": {
                    "loss_mean": 0.22,
                    "position_rmse_xyz_mm": rmse,
                    "position_rmse_xy_mm": rmse * 0.9,
                    "position_rmse_z_mm": rmse * 0.5,
                    "position_error_l2_mm": {"mean": rmse, "median": rmse, "p95": rmse, "max": rmse},
                    "tangent_angular_error_rad": {"mean": 0.05, "median": 0.05, "p95": 0.06, "max": 0.07},
                },
            },
            "model": {"family": "legacy_ann", "hidden_layers": list(config.hidden_layers)},
        }
        (ad / "training_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        (ad / "training_config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
        (ad / "loss_history.csv").write_text("epoch,train_loss,validation_loss,elapsed_s\n1,0.5,0.6,0.1\n", encoding="utf-8")
        (ad / "loss_curve.png").write_bytes(b"")
        (ad / "split_manifest.json").write_text("{}", encoding="utf-8")
        (ad / "training_summary.txt").write_text("ok\n", encoding="utf-8")
        (ad / "model.pt").write_bytes(b"")
        return TrainingResult(
            artifact_dir=ad,
            model_path=ad / "model.pt",
            metadata_path=ad / "training_metadata.json",
            loss_history_path=ad / "loss_history.csv",
            loss_plot_path=ad / "loss_curve.png",
            split_manifest_path=ad / "split_manifest.json",
            summary_text_path=ad / "training_summary.txt",
            status="completed",
            best_epoch=2,
            best_validation_loss=0.2,
            test_loss=float(0.1 * rmse),
            epochs_completed=2,
            train_losses=[0.5],
            validation_losses=[0.6],
            estimate=_estimate_stub(),
        )

    monkeypatch.setattr(m, "train_linear_ridge_full_pose", _fake_linear)
    monkeypatch.setattr(m, "train_legacy_ann", _fake_ann)

    cfg = AnnTrainingConfig(
        artifact_root=str(tmp_path / "data" / "models" / "ann"),
        artifact_name="fixture_sweep",
        epochs=1,
        batch_size=2,
        train_ratio=0.5,
        validation_ratio=0.25,
        test_ratio=0.25,
    )
    result = run_model_sweep(
        project_root=tmp_path,
        dataset_path=run_dir,
        base_config=cfg,
        backend_name="cpu",
        include_linear_baseline=True,
        ann_hidden_layers_list=[[32, 32], [64, 64], [128, 128]],
    )
    assert len(splits_seen) == 4
    ref_train = list(splits_seen[0].train_indices)
    for sp in splits_seen[1:]:
        assert list(sp.train_indices) == ref_train

    summary_payload = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    assert len(summary_payload["rows"]) == 4
    assert summary_payload["best_model"]["artifact_subdir"] == "ann_128_128"
    assert summary_payload["best_model"]["test_position_rmse_xyz_mm"] == pytest.approx(1.0)
    if result.comparison_png_path is not None:
        assert result.comparison_png_path.exists()

    for sub in ("linear_ridge_full_pose", "ann_32_32", "ann_64_64", "ann_128_128"):
        meta_path = result.sweep_root / sub / "training_metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["model"]["hidden_layers"] == (
            [] if sub == "linear_ridge_full_pose" else [int(x) for x in sub.split("_")[1:]]
        )

    prepared = prepare_legacy_ann_dataset(run_dir)
    assert prepared.inputs.shape[0] >= 1


def test_run_model_sweep_best_falls_back_to_test_loss() -> None:
    rows = [
        {"model_key": "a", "test_position_rmse_xyz_mm": None, "test_loss": 0.5},
        {"model_key": "b", "test_position_rmse_xyz_mm": None, "test_loss": 0.2},
    ]
    best = select_best_sweep_row_by_test_position_rmse(rows)
    assert best is not None and best["model_key"] == "b"
