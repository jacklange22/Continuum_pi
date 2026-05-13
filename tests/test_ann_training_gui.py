from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from pathlib import Path
import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from continuum_robot.gui.controllers import ann_training_controller as controller_module
from continuum_robot.gui.controllers.ann_training_controller import AnnTrainingController
from continuum_robot.gui.widgets.ann_training_window import AnnTrainingWindow
from continuum_robot.modeling.ann_training import BackendOption, BackendReport, TrainingEstimate, TrainingResult


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


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
        "sample_counts": {"total": 2},
        "dropped_frames": 0,
        "invalid_transforms": 0,
        "stage_pass_fail": {},
        "status": "success",
        "experiment_metrics": {
            "dataset_mode": "workspace_coverage",
            "dataset_mode_summary": "Bounded workspace exploration for first-pass forward-model training.",
            "accepted_sample_count": 2,
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
            "step_index": 1,
            "sample_index": 1,
            "accepted": True,
            "resolved_cable_command_cm": [0.2, 0.3, 0.4, 0.5],
            "tip_position_xyz_mm": [4.0, 5.0, 6.0],
            "tip_tangent_xyz": [0.04, 0.05, 0.06],
            "previous_pair_command_cm": [0.0, 0.0],
        },
    ]
    (output_root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with (output_root / "modeling_dataset_export.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return output_root


def test_ann_training_window_launches_and_training_stays_async(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _app()
    run_dir = _write_modeling_run(tmp_path)
    dataset_output_root = tmp_path / "data" / "experiments"
    artifact_root = tmp_path / "data" / "models" / "ann"

    fake_report = BackendReport(
        python_version="3.11.0",
        platform_summary="Darwin test",
        torch_available=True,
        torch_version="2.test",
        selected_backend="mps",
        recommended_backend="mps",
        selected_dtype="float32",
        backend_options=[
            BackendOption(name="cpu", label="CPU", available=True, recommended=False, dtype="float64"),
            BackendOption(name="mps", label="MPS", available=True, recommended=True, dtype="float32"),
            BackendOption(name="cuda", label="CUDA", available=False, recommended=False, dtype="float64"),
        ],
    )

    def _fake_train_legacy_ann(*, project_root, dataset_path, config, backend_name, progress_callback, stop_requested, **_kwargs):
        artifact_dir = Path(project_root) / "data" / "models" / "ann" / "20260419_120200_fake_run"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "training_config.json").write_text(
            json.dumps(config.to_dict(), indent=2),
            encoding="utf-8",
        )
        loss_history_path = artifact_dir / "loss_history.csv"
        loss_history_path.write_text(
            "epoch,train_loss,validation_loss,elapsed_s\n1,0.5,0.6,0.1\n",
            encoding="utf-8",
        )
        metadata = {
            "status": "completed",
            "training": {"epochs_completed": 1, "best_epoch": 1, "best_validation_loss": 0.6},
            "model": {
                "family": "legacy_ann",
                "variant": "full_pose",
                "hidden_layers": list(config.hidden_layers),
                "input_dim": 4,
                "output_dim": 6,
            },
            "files": {"loss_history_path": str(loss_history_path), "model_path": str(artifact_dir / "model.pt")},
            "dataset": {"run_name": Path(dataset_path).name},
            "backend": {"selected_backend": backend_name},
        }
        (artifact_dir / "training_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (artifact_dir / "model.pt").write_bytes(b"fake")
        progress_callback(
            controller_module.TrainingProgress(
                epoch=1,
                total_epochs=1,
                train_loss=0.5,
                validation_loss=0.6,
                elapsed_s=0.1,
                remaining_s=0.0,
                status="running",
            )
        )
        time.sleep(0.05)
        return TrainingResult(
            artifact_dir=artifact_dir,
            model_path=artifact_dir / "model.pt",
            metadata_path=artifact_dir / "training_metadata.json",
            loss_history_path=loss_history_path,
            loss_plot_path=artifact_dir / "loss_curve.png",
            split_manifest_path=artifact_dir / "split_manifest.json",
            summary_text_path=artifact_dir / "training_summary.txt",
            status="completed",
            best_epoch=1,
            best_validation_loss=0.6,
            test_loss=0.7,
            epochs_completed=1,
            train_losses=[0.5],
            validation_losses=[0.6],
            estimate=TrainingEstimate(
                estimated_total_s=1.0,
                estimated_epoch_s=1.0,
                train_batch_time_s=0.1,
                validation_batch_time_s=0.1,
                benchmark_train_batches=1,
                benchmark_validation_batches=1,
                train_batch_count=1,
                validation_batch_count=1,
            ),
        )

    monkeypatch.setattr(controller_module, "detect_training_backends", lambda preferred_backend=None: fake_report)
    monkeypatch.setattr(
        controller_module,
        "estimate_runtime",
        lambda **_kwargs: TrainingEstimate(
            estimated_total_s=1.0,
            estimated_epoch_s=1.0,
            train_batch_time_s=0.1,
            validation_batch_time_s=0.1,
            benchmark_train_batches=1,
            benchmark_validation_batches=1,
            train_batch_count=1,
            validation_batch_count=1,
        ),
    )
    monkeypatch.setattr(controller_module, "train_legacy_ann", _fake_train_legacy_ann)

    controller = AnnTrainingController(
        project_root=tmp_path,
        dataset_output_root=dataset_output_root,
        artifact_root=artifact_root,
    )
    controller.select_dataset(str(run_dir))
    window = AnnTrainingWindow(controller)
    try:
        window.show()
        window._refresh_state()
        assert window.dataset_list.count() == 1
        controller.train()
        assert controller.refresh().training_active is True
        QApplication.processEvents()
        assert window.status_label.text()
        time.sleep(0.12)
        window._refresh_state()
        assert controller.refresh().training_active is False
        assert controller.refresh().last_output_path is not None
        artifact_path = Path(controller.refresh().last_output_path)
        training_cfg = json.loads((artifact_path / "training_config.json").read_text(encoding="utf-8"))
        assert training_cfg["hidden_layers"] == [32, 32]
        meta = json.loads((artifact_path / "training_metadata.json").read_text(encoding="utf-8"))
        assert meta["model"]["hidden_layers"] == [32, 32]
    finally:
        window.close()
        controller.shutdown()


def test_training_config_reflects_custom_hidden_layers_and_epochs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _app()
    run_dir = _write_modeling_run(tmp_path)
    dataset_output_root = tmp_path / "data" / "experiments"
    artifact_root = tmp_path / "data" / "models" / "ann"

    fake_report = BackendReport(
        python_version="3.11.0",
        platform_summary="Darwin test",
        torch_available=True,
        torch_version="2.test",
        selected_backend="mps",
        recommended_backend="mps",
        selected_dtype="float32",
        backend_options=[
            BackendOption(name="cpu", label="CPU", available=True, recommended=False, dtype="float64"),
            BackendOption(name="mps", label="MPS", available=True, recommended=True, dtype="float32"),
            BackendOption(name="cuda", label="CUDA", available=False, recommended=False, dtype="float64"),
        ],
    )

    def _fake_train_legacy_ann(*, project_root, dataset_path, config, backend_name, progress_callback, stop_requested, **_kwargs):
        artifact_dir = Path(project_root) / "data" / "models" / "ann" / "20260419_120201_custom_hl"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "training_config.json").write_text(
            json.dumps(config.to_dict(), indent=2),
            encoding="utf-8",
        )
        loss_history_path = artifact_dir / "loss_history.csv"
        loss_history_path.write_text(
            "epoch,train_loss,validation_loss,elapsed_s\n1,0.5,0.6,0.1\n",
            encoding="utf-8",
        )
        metadata = {
            "status": "completed",
            "training": {"epochs_completed": int(config.epochs), "best_epoch": 1, "best_validation_loss": 0.6},
            "model": {
                "family": "legacy_ann",
                "variant": "full_pose",
                "hidden_layers": list(config.hidden_layers),
                "input_dim": 4,
                "output_dim": 6,
            },
            "files": {"loss_history_path": str(loss_history_path), "model_path": str(artifact_dir / "model.pt")},
            "dataset": {"run_name": Path(dataset_path).name},
            "backend": {"selected_backend": backend_name},
        }
        (artifact_dir / "training_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (artifact_dir / "model.pt").write_bytes(b"fake")
        progress_callback(
            controller_module.TrainingProgress(
                epoch=1,
                total_epochs=int(config.epochs),
                train_loss=0.5,
                validation_loss=0.6,
                elapsed_s=0.1,
                remaining_s=0.0,
                status="running",
            )
        )
        time.sleep(0.05)
        return TrainingResult(
            artifact_dir=artifact_dir,
            model_path=artifact_dir / "model.pt",
            metadata_path=artifact_dir / "training_metadata.json",
            loss_history_path=loss_history_path,
            loss_plot_path=artifact_dir / "loss_curve.png",
            split_manifest_path=artifact_dir / "split_manifest.json",
            summary_text_path=artifact_dir / "training_summary.txt",
            status="completed",
            best_epoch=1,
            best_validation_loss=0.6,
            test_loss=0.7,
            epochs_completed=int(config.epochs),
            train_losses=[0.5],
            validation_losses=[0.6],
            estimate=TrainingEstimate(
                estimated_total_s=1.0,
                estimated_epoch_s=1.0,
                train_batch_time_s=0.1,
                validation_batch_time_s=0.1,
                benchmark_train_batches=1,
                benchmark_validation_batches=1,
                train_batch_count=1,
                validation_batch_count=1,
            ),
        )

    monkeypatch.setattr(controller_module, "detect_training_backends", lambda preferred_backend=None: fake_report)
    monkeypatch.setattr(
        controller_module,
        "estimate_runtime",
        lambda **_kwargs: TrainingEstimate(
            estimated_total_s=1.0,
            estimated_epoch_s=1.0,
            train_batch_time_s=0.1,
            validation_batch_time_s=0.1,
            benchmark_train_batches=1,
            benchmark_validation_batches=1,
            train_batch_count=1,
            validation_batch_count=1,
        ),
    )
    monkeypatch.setattr(controller_module, "train_legacy_ann", _fake_train_legacy_ann)

    controller = AnnTrainingController(
        project_root=tmp_path,
        dataset_output_root=dataset_output_root,
        artifact_root=artifact_root,
    )
    controller.select_dataset(str(run_dir))
    controller.set_hidden_layers_text("64, 64")
    controller.set_epochs(7)
    window = AnnTrainingWindow(controller)
    try:
        window.show()
        window._refresh_state()
        controller.train()
        QApplication.processEvents()
        time.sleep(0.15)
        window._refresh_state()
        assert controller.refresh().training_active is False
        artifact_path = Path(controller.refresh().last_output_path)
        training_cfg = json.loads((artifact_path / "training_config.json").read_text(encoding="utf-8"))
        assert training_cfg["hidden_layers"] == [64, 64]
        assert training_cfg["epochs"] == 7
        meta = json.loads((artifact_path / "training_metadata.json").read_text(encoding="utf-8"))
        assert meta["model"]["hidden_layers"] == [64, 64]
    finally:
        window.close()
        controller.shutdown()


def test_hidden_layers_line_not_overwritten_while_focused_during_poll(tmp_path: Path) -> None:
    _app()
    (tmp_path / "data" / "experiments").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "models" / "ann").mkdir(parents=True, exist_ok=True)
    controller = AnnTrainingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
    )
    window = AnnTrainingWindow(controller)
    try:
        window.show()
        controller.set_hidden_layers_text("32, 32")
        window._refresh_state()
        window.hidden_layers_edit.setText("128, 128")
        for _ in range(25):
            window._sync_parameters(controller.refresh())
        assert window.hidden_layers_edit.text() == "128, 128"
        window._on_hidden_layers_editing_finished()
        assert controller.config_snapshot().hidden_layers == [128, 128]
    finally:
        window.close()
        controller.shutdown()


def test_hidden_layers_preset_combo_updates_controller(tmp_path: Path) -> None:
    _app()
    (tmp_path / "data" / "experiments").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "models" / "ann").mkdir(parents=True, exist_ok=True)
    controller = AnnTrainingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
    )
    window = AnnTrainingWindow(controller)
    try:
        window.show()
        window._sync_parameters(controller.refresh())
        window.hidden_layers_preset_combo.setCurrentIndex(3)
        assert controller.config_snapshot().hidden_layers == [128, 128]
        assert controller.hidden_layers_text() == "128, 128"
    finally:
        window.close()
        controller.shutdown()

    _app()
    bad_run = tmp_path / "data" / "experiments" / "collect_pose_command_dataset" / "broken_run"
    bad_run.mkdir(parents=True)
    metadata = {"experiment_name": "collect_pose_command_dataset", "run_id": "b", "timestamp_utc": "2026-01-02T00:00:00Z"}
    summary = {"status": "success", "sample_counts": {"total": 0}, "experiment_metrics": {}}
    (bad_run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (bad_run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    controller = AnnTrainingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
    )
    window = AnnTrainingWindow(controller)
    try:
        window.show()
        window._refresh_state()
        assert window.dataset_list.count() == 0
        controller.set_show_non_trainable_datasets(True)
        window._refresh_state()
        assert window.dataset_list.count() == 1
    finally:
        window.close()
        controller.shutdown()
