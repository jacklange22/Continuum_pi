from __future__ import annotations

import json
import os
from pathlib import Path
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QScrollArea

from continuum_robot.gui.controllers.ann_training_controller import AnnTrainingController
from continuum_robot.gui.controllers import modeling_controller as controller_module
from continuum_robot.gui.controllers.modeling_controller import ModelingController
from continuum_robot.gui.experiment_visualization import ChartModel, VisualizationModel
from continuum_robot.gui.theme import grouped_workspace_stylesheet
from continuum_robot.gui.tabs.modeling_tab import ModelingTab
from continuum_robot.gui.widgets.ann_training_window import AnnTrainingWindow
from continuum_robot.modeling import analysis as analysis_module
from continuum_robot.modeling.analysis import ModelMetrics, ModelEvaluation, ModelingEvaluationResult


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


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
        "config_used": {"dataset_mode": "workspace_coverage"},
        "operator_notes": "",
    }
    summary = {
        "schema_version": "1.0",
        "experiment_name": "collect_pose_command_dataset",
        "run_id": "dataset-run-123",
        "success": True,
        "sample_counts": {"total": 2},
        "dropped_frames": 0,
        "invalid_transforms": 0,
        "stage_pass_fail": {},
        "status": "success",
        "experiment_metrics": {
            "dataset_mode": "workspace_coverage",
            "dataset_mode_summary": "Bounded workspace-coverage collection for first-pass forward-model training.",
            "accepted_sample_count": 2,
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
    rows = [
        {
            "sequence_index": 0,
            "step_index": 0,
            "sample_index": 0,
            "phase": "workspace",
            "accepted": True,
            "resolved_cable_command_cm": [0.0, 0.0, 0.0, 0.0],
            "tip_position_xyz_mm": [0.0, 0.0, 64.0],
            "tip_tangent_xyz": [0.0, 0.0, 1.0],
            "previous_pair_command_cm": [],
        },
        {
            "sequence_index": 1,
            "step_index": 1,
            "sample_index": 1,
            "phase": "workspace",
            "accepted": True,
            "resolved_cable_command_cm": [0.1, 0.1, 0.0, 0.0],
            "tip_position_xyz_mm": [0.0, 0.0, 64.0],
            "tip_tangent_xyz": [0.0, 0.0, 1.0],
            "previous_pair_command_cm": [0.0, 0.0],
        },
    ]
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
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
        "dataset": {"path": str(dataset_path), "run_name": dataset_path.name},
        "backend": {"selected_backend": "cpu", "torch_version": "2.test"},
        "model": {"input_dim": 4, "output_dim": 6, "hidden_layers": [32, 32], "dtype": "float64"},
        "training": {"epochs_completed": 1, "batch_size": 64, "learning_rate": 0.001, "best_validation_loss": 0.1},
        "files": {"model_path": str(artifact_dir / "model.pt")},
    }
    (artifact_dir / "training_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (artifact_dir / "training_config.json").write_text(json.dumps({"epochs": 1}), encoding="utf-8")
    (artifact_dir / "split_manifest.json").write_text(json.dumps({"test_indices": [1]}), encoding="utf-8")
    (artifact_dir / "loss_history.csv").write_text(
        "epoch,train_loss,validation_loss,elapsed_s\n1,0.5,0.6,0.1\n",
        encoding="utf-8",
    )
    (artifact_dir / "model.pt").write_bytes(b"fake")
    return artifact_dir


def test_modeling_tab_launches_and_evaluates_async(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _app()
    run_dir = _write_modeling_run(tmp_path)
    artifact_dir = _write_artifact(tmp_path, run_dir)

    def _fake_evaluate_models(*, project_root, dataset_path, artifact_path, config):
        output_dir = Path(project_root) / "data" / "modeling_results" / "20260419_130000_eval"
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "summary.json"
        metadata_path = output_dir / "evaluation_metadata.json"
        comparison_csv_path = output_dir / "comparison_metrics.csv"
        plot_path = output_dir / "workspace_xy.png"
        summary_path.write_text("{}", encoding="utf-8")
        metadata_path.write_text("{}", encoding="utf-8")
        comparison_csv_path.write_text("model_key,label,status\nann,ANN,completed\n", encoding="utf-8")
        plot_path.write_bytes(b"png")
        time.sleep(0.05)
        return ModelingEvaluationResult(
            dataset_summary=controller_module.discover_modeling_datasets(output_root=tmp_path / "data" / "experiments")[0],
            artifact_details=controller_module.load_trained_artifact_details(artifact_path),
            evaluation_scope_requested=config.evaluation_scope,
            evaluation_scope_used="full_dataset",
            evaluation_scope_note="test",
            selected_sample_count=2,
            output_dir=output_dir,
            summary_path=summary_path,
            metadata_path=metadata_path,
            comparison_csv_path=comparison_csv_path,
            phase_csv_path=None,
            plot_paths={"workspace_xy": plot_path},
            model_evaluations={
                "ann": ModelEvaluation(
                    metrics=ModelMetrics(
                        model_key="ann",
                        label="ANN",
                        status="completed",
                        sample_count=2,
                        position_rmse_mm=0.1,
                        mean_position_error_mm=0.1,
                        max_position_error_mm=0.1,
                        tangent_mean_error_deg=0.1,
                        tangent_rmse_deg=0.1,
                        tangent_max_error_deg=0.1,
                    ),
                    predictions=None,
                    position_errors_mm=[0.1, 0.1],
                    tangent_errors_deg=[0.1, 0.1],
                )
            },
            visualization_model=VisualizationModel(
                summary_lines=["done"],
                charts=[ChartModel(kind="bar", title="RMSE", x_title="Model", y_title="mm", categories=["ANN"], values=[0.1])],
            ),
        )

    monkeypatch.setattr(controller_module, "evaluate_models", _fake_evaluate_models)

    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    state = controller.refresh()
    controller.select_dataset(str(run_dir))
    controller.select_artifact(str(artifact_dir))
    tab = ModelingTab(controller)
    try:
        tab.show()
        tab.update(controller.refresh())
        assert tab.dataset_list.count() == 1
        controller.evaluate()
        assert controller.refresh().evaluation_active is True
        QApplication.processEvents()
        time.sleep(0.12)
        tab.update(controller.refresh())
        assert controller.refresh().evaluation_active is False
        assert controller.refresh().last_output_path is not None
    finally:
        tab.close()
        controller.shutdown()


def test_modeling_tab_constructs_with_grouped_workspace_stylesheet_contract(tmp_path: Path) -> None:
    _app()
    run_dir = _write_modeling_run(tmp_path)
    artifact_dir = _write_artifact(tmp_path, run_dir)
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    controller.select_dataset(str(run_dir))
    controller.select_artifact(str(artifact_dir))
    tab = ModelingTab(controller)
    try:
        assert tab.objectName() == "modelingWorkspace"
        assert "QWidget#modelingWorkspace QComboBox" in tab.styleSheet()
        assert "QWidget#modelingWorkspace QListWidget" in tab.styleSheet()
    finally:
        tab.close()
        controller.shutdown()


def test_grouped_workspace_stylesheet_allows_missing_input_selectors() -> None:
    stylesheet = grouped_workspace_stylesheet(object_name="testWorkspace")

    assert "QWidget#testWorkspace" in stylesheet


def test_ann_training_window_uses_main_scroll_area(tmp_path: Path) -> None:
    _app()
    run_dir = _write_modeling_run(tmp_path)
    artifact_dir = _write_artifact(tmp_path, run_dir)
    controller = AnnTrainingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
    )
    controller.select_dataset(str(run_dir))
    controller.select_artifact(str(artifact_dir))
    window = AnnTrainingWindow(controller)
    try:
        assert isinstance(window.main_scroll_area, QScrollArea)
        assert window.main_scroll_area.widget() is not None
    finally:
        window.close()
        controller.shutdown()
