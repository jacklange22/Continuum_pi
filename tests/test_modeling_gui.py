from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

pytestmark = pytest.mark.gui

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


TWO_SEGMENT_FIXTURE = Path(__file__).parent / "fixtures" / "two_segment_modeling_trainable"


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

    def _fake_evaluate_models(*, project_root, dataset_path, artifact_path, config, test_dataset_path=None):
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
            dataset_summary=controller_module.discover_modeling_datasets(
                project_root=tmp_path,
                output_root=tmp_path / "data" / "experiments",
            )[0],
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


def test_modeling_tab_train_ann_button_invokes_opener_callback(tmp_path: Path) -> None:
    """The 'Train ANN' button hands the selected dataset path to the supplied opener."""
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
    captured: dict[str, str] = {}

    def _opener(path: str) -> None:
        captured["path"] = str(path)

    tab = ModelingTab(controller, open_in_ann_training=_opener)
    try:
        tab.update(controller.refresh())
        assert tab.train_ann_button.isEnabled()
        tab.train_ann_button.click()
        assert captured.get("path") == str(run_dir)
    finally:
        tab.close()
        controller.shutdown()


def test_modeling_tab_train_ann_button_without_opener_does_not_crash(tmp_path: Path) -> None:
    """Without an opener callback, the click should be a no-op (status text only)."""
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
    tab = ModelingTab(controller)  # no opener passed
    try:
        tab.update(controller.refresh())
        tab.train_ann_button.click()
        # Status label should reflect the missing opener; should not raise.
        assert "not available" in tab.status_label.text().lower()
    finally:
        tab.close()
        controller.shutdown()


def test_two_segment_controller_and_tab_expose_dataset_discovery(tmp_path: Path) -> None:
    """The new dedicated TwoSegmentModelingController + TwoSegmentModelingTab handle the
    two-segment workflow. The single-segment ModelingTab no longer carries that UI."""
    from continuum_robot.gui.controllers.two_segment_modeling_controller import (
        TwoSegmentModelingController,
    )
    from continuum_robot.gui.tabs.two_segment_modeling_tab import TwoSegmentModelingTab

    _app()
    run_dir = (
        tmp_path
        / "data"
        / "experiments"
        / "two_segment_collect_pose_command_dataset"
        / "20260508_120000_two_segment_collect_pose_command_dataset"
    )
    shutil.copytree(TWO_SEGMENT_FIXTURE, run_dir)
    controller = TwoSegmentModelingController(project_root=tmp_path)
    state = controller.refresh()

    assert str(run_dir) in state.dataset_runs
    assert state.can_run is True
    trainability = controller.validate_trainability([run_dir])
    assert trainability["samples_accepted"] == 6
    assert trainability["samples_rejected"] == 0
    assert "two_coil_xyz_available" in trainability

    tab = TwoSegmentModelingTab(controller)
    try:
        tab.update(state)
        assert tab.run_list.count() == 1
        assert tab.run_button.isEnabled() is True
        assert tab.strict_check.isChecked() is True
        assert tab.label_mode_combo.currentData() == "auto"
        assert tab.hidden_combo.currentData() == "128,128"
    finally:
        tab.close()
        controller.shutdown()


def test_modeling_tab_headline_tile_colors(tmp_path: Path) -> None:
    """Per-model RMSE tiles must color-code by surgical-accuracy threshold."""
    from continuum_robot.gui.tabs.modeling_tab import _headline_color_for_rmse

    # Sub-millimeter — green
    green_border, _ = _headline_color_for_rmse(0.8)
    assert green_border == "#16a34a"
    # Wolfe's 2.24 mm — amber
    amber_border, _ = _headline_color_for_rmse(2.24)
    assert amber_border == "#d97706"
    # 9.5 mm constant-curvature — red
    red_border, _ = _headline_color_for_rmse(9.5)
    assert red_border == "#dc2626"
    # No value — muted/grey
    muted_border, _ = _headline_color_for_rmse(None)
    assert muted_border  # any non-empty color


def test_modeling_tab_headline_widget_renders_tiles(tmp_path: Path) -> None:
    """Setting headline metrics on the widget produces one tile per model."""
    from continuum_robot.gui.controllers.modeling_controller import HeadlineMetric
    from continuum_robot.gui.tabs.modeling_tab import _HeadlineMetricsWidget

    _app()
    widget = _HeadlineMetricsWidget()
    widget.show()
    widget.set_metrics(
        [
            HeadlineMetric(label="Mike", model_key="mike", rmse_mm=9.5, status="completed"),
            HeadlineMetric(label="Camarillo", model_key="camarillo", rmse_mm=9.46, status="completed"),
            HeadlineMetric(label="ANN", model_key="ann", rmse_mm=2.24, status="completed"),
        ]
    )
    # One tile per metric — _layout.count() reflects the tile count exactly.
    assert widget._layout.count() == 3
    # Idempotent on identical input.
    widget.set_metrics(
        [
            HeadlineMetric(label="Mike", model_key="mike", rmse_mm=9.5, status="completed"),
            HeadlineMetric(label="Camarillo", model_key="camarillo", rmse_mm=9.46, status="completed"),
            HeadlineMetric(label="ANN", model_key="ann", rmse_mm=2.24, status="completed"),
        ]
    )
    assert widget._layout.count() == 3
    # Unavailable tiles render too, with a reason.
    widget.set_metrics(
        [
            HeadlineMetric(
                label="ANN",
                model_key="ann",
                rmse_mm=None,
                status="unavailable",
                reason="No ANN artifact selected.",
            ),
        ]
    )
    assert widget._layout.count() == 1
    widget.close()


def test_build_worst_predictions_table_returns_top_k_rows(tmp_path: Path) -> None:
    """Top-K worst-predictions table emits one row per worst sample with cable + xyz + err."""
    import numpy as np
    from continuum_robot.modeling.analysis import _build_worst_predictions_table

    truths = np.array([[i, 0, 0, 0, 0, 0] for i in range(5)], dtype=float)
    predictions = truths.copy()
    # Inflate errors on samples 1 and 3 so they should be at the top.
    predictions[1, 0] = 10.0
    predictions[3, 0] = 5.0
    fake_eval = ModelEvaluation(
        metrics=ModelMetrics(
            model_key="ann",
            label="ANN",
            status="completed",
            sample_count=5,
            position_rmse_mm=4.0,
        ),
        predictions=predictions,
        position_errors_mm=[
            float(np.linalg.norm(predictions[i, :3] - truths[i, :3])) for i in range(5)
        ],
    )
    inputs = np.arange(5 * 4, dtype=float).reshape(5, 4) / 10.0
    chart = _build_worst_predictions_table(
        fake_eval,
        truths=truths,
        inputs=inputs,
        sample_indices=[100, 101, 102, 103, 104],
        top_k=3,
    )
    assert chart is not None
    assert chart.kind == "table"
    assert "ANN" in chart.title
    # Rank 1 is sample with the biggest error — sample 1, mapped to original index 101.
    first_row = chart.table_rows[0]
    assert first_row[0] == "1"
    assert first_row[1] == "101"
    assert "[" in first_row[2]  # cable command formatted as list
    # Top-K limited correctly.
    assert len(chart.table_rows) == 3


def test_modeling_controller_exposes_past_evaluations(tmp_path: Path) -> None:
    """Past evaluation history is discovered from results_root and surfaced on state."""
    _app()
    run_dir = _write_modeling_run(tmp_path)
    results_root = tmp_path / "data" / "modeling_results"
    # Hand-craft a past evaluation folder matching the dataset.
    past_dir = results_root / "20260419_120100_collect_pose_command_dataset"
    past_dir.mkdir(parents=True)
    (past_dir / "summary.json").write_text(
        json.dumps(
            {
                "dataset_run_name": run_dir.name,
                "dataset_mode": "workspace_coverage",
                "selected_sample_count": 42,
                "evaluation_scope_used": "artifact_test_split",
                "models": {
                    "mike": {"label": "Mike", "position_rmse_mm": 9.5},
                    "ann": {"label": "ANN", "position_rmse_mm": 1.87},
                },
            }
        ),
        encoding="utf-8",
    )
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=results_root,
    )
    controller.select_dataset(str(run_dir))
    state = controller.refresh()
    assert len(state.past_evaluations) == 1
    past = state.past_evaluations[0]
    assert past.best_label == "ANN"
    assert past.best_rmse_mm == pytest.approx(1.87)
    assert past.selected_sample_count == 42


def test_modeling_tab_history_list_double_click_opens_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Double-clicking a history row triggers a folder open via QDesktopServices."""
    _app()
    run_dir = _write_modeling_run(tmp_path)
    results_root = tmp_path / "data" / "modeling_results"
    past_dir = results_root / "20260419_120100_collect_pose_command_dataset"
    past_dir.mkdir(parents=True)
    (past_dir / "summary.json").write_text(
        json.dumps(
            {
                "dataset_run_name": run_dir.name,
                "selected_sample_count": 5,
                "models": {"ann": {"label": "ANN", "position_rmse_mm": 1.2}},
            }
        ),
        encoding="utf-8",
    )
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=results_root,
    )
    controller.select_dataset(str(run_dir))
    captured: dict[str, str] = {}

    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices

    def _fake_open(url: QUrl) -> bool:
        captured["url"] = url.toLocalFile()
        return True

    monkeypatch.setattr(QDesktopServices, "openUrl", _fake_open)
    tab = ModelingTab(controller)
    try:
        tab.update(controller.refresh())
        assert tab.history_list.count() == 1
        # Simulate double-click on the first row.
        tab._on_history_item_double_clicked(tab.history_list.item(0))
        assert captured.get("url") == str(past_dir)
    finally:
        tab.close()
        controller.shutdown()


def test_modeling_tab_headline_status_label_reflects_state(tmp_path: Path) -> None:
    """Status label echoes the dataset name pre-eval and the best RMSE post-eval."""
    _app()
    run_dir = _write_modeling_run(tmp_path)
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    controller.select_dataset(str(run_dir))
    tab = ModelingTab(controller)
    try:
        tab.show()
        tab.update(controller.refresh())
        # Pre-eval: status label mentions the run name.
        assert run_dir.name in tab.status_label.text()
        # Inject a fake result and re-render — post-eval label changes.
        fake_metrics = ModelMetrics(
            model_key="ann",
            label="ANN",
            status="completed",
            sample_count=10,
            position_rmse_mm=0.87,
        )
        fake_result = ModelingEvaluationResult(
            dataset_summary=controller._selected_dataset_summary,
            artifact_details=controller._selected_artifact_details,
            evaluation_scope_requested="full_dataset",
            evaluation_scope_used="full_dataset",
            evaluation_scope_note="",
            selected_sample_count=10,
            output_dir=tmp_path / "data" / "modeling_results" / "label_fake",
            summary_path=tmp_path / "summary.json",
            metadata_path=tmp_path / "metadata.json",
            comparison_csv_path=tmp_path / "comp.csv",
            phase_csv_path=None,
            plot_paths={},
            model_evaluations={"ann": ModelEvaluation(metrics=fake_metrics, predictions=None)},
            visualization_model=VisualizationModel(summary_lines=["ok"]),
        )
        controller._last_result = fake_result  # type: ignore[attr-defined]
        tab.update(controller.refresh())
        text = tab.status_label.text().lower()
        assert "ann" in text
        assert "0.87" in text or "0.870" in text
        assert "full dataset" in text or "10 samples" in text
    finally:
        tab.close()
        controller.shutdown()


def test_modeling_tab_validate_rows_button_runs_row_filter_on_eval_dataset(tmp_path: Path) -> None:
    """Click 'Validate Rows' → row filter runs on the effective eval dataset, status echoes."""
    _app()
    run_dir = _write_modeling_run(tmp_path)
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    controller.select_dataset(str(run_dir))
    tab = ModelingTab(controller)
    try:
        tab.show()
        tab.update(controller.refresh())
        # Button enabled when a dataset is selected.
        assert tab.validate_rows_button.isEnabled() is True
        # Status label hidden before first click.
        assert tab.row_filter_status_label.isVisible() is False
        tab._on_validate_rows_clicked()
        state = controller.refresh()
        assert state.row_filter_status_text  # populated by the validator
        assert tab.row_filter_status_label.isVisible() is True
        assert "Row filter" in state.row_filter_status_text
    finally:
        tab.close()
        controller.shutdown()


def test_modeling_tab_empty_state_hint_when_no_datasets(tmp_path: Path) -> None:
    """When zero datasets are discovered, dataset card shows a clear next-action hint."""
    _app()
    (tmp_path / "data" / "experiments").mkdir(parents=True, exist_ok=True)
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    tab = ModelingTab(controller)
    try:
        tab.show()
        tab.update(controller.refresh())
        assert tab.dataset_empty_hint.isVisible() is True
        assert "Experiment tab" in tab.dataset_empty_hint.text()
    finally:
        tab.close()
        controller.shutdown()


def test_modeling_tab_artifact_dataset_mismatch_chip(tmp_path: Path) -> None:
    """Selecting an artifact trained on a different dataset triggers the mismatch chip."""
    _app()
    run_dir = _write_modeling_run(tmp_path)
    # Hand-craft an artifact whose metadata says it was trained on a different dataset.
    artifact_dir = tmp_path / "data" / "models" / "ann" / "mismatch_artifact"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "created_at_utc": "2026-05-15T00:00:00Z",
                "status": "completed",
                "model": {"output_target": "xyz", "hidden_layers": [32, 32]},
                "training": {"epochs_completed": 1, "best_validation_loss": 0.5},
                # Mismatch: artifact says it was trained on a different run name.
                "dataset": {"run_name": "some_other_dataset_run", "path": str(tmp_path / "other")},
                "backend": {"selected_backend": "cpu"},
                "files": {"loss_history_path": str(artifact_dir / "loss_history.csv"),
                          "model_path": str(artifact_dir / "model.pt")},
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "loss_history.csv").write_text(
        "epoch,train_loss,validation_loss,elapsed_s\n1,0.5,0.6,0.1\n", encoding="utf-8"
    )
    (artifact_dir / "model.pt").write_bytes(b"x")
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
        tab.show()
        tab.update(controller.refresh())
        assert tab.artifact_mismatch_chip.isVisible() is True
        text = tab.artifact_mismatch_chip.text().lower()
        assert "different dataset" in text or "some_other_dataset_run" in text.lower()
    finally:
        tab.close()
        controller.shutdown()


def test_modeling_tab_headline_tile_tooltip_has_per_axis_breakdown(tmp_path: Path) -> None:
    """The headline tile's tooltip exposes per-axis RMSE + mean/max for the operator."""
    from continuum_robot.gui.controllers.modeling_controller import HeadlineMetric
    from continuum_robot.gui.tabs.modeling_tab import _HeadlineMetricsWidget

    _app()
    widget = _HeadlineMetricsWidget()
    widget.show()
    widget.set_metrics(
        [
            HeadlineMetric(
                label="ANN",
                model_key="ann",
                rmse_mm=1.234,
                status="completed",
                per_axis_rmse_mm=(0.5, 0.6, 0.9),
                mean_position_error_mm=1.1,
                max_position_error_mm=2.5,
            )
        ]
    )
    # Find the only tile via its layout.
    tile = widget._layout.itemAt(0).widget()
    tooltip = tile.toolTip()
    assert "0.500" in tooltip and "0.600" in tooltip and "0.900" in tooltip
    assert "1.234" in tooltip
    assert "mean position error" in tooltip.lower()
    widget.close()


def test_modeling_tab_copy_summary_button_pushes_text_to_clipboard(tmp_path: Path) -> None:
    """Clicking Copy Summary on a populated tab puts a plain-text summary on the clipboard."""
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
    fake_metrics = ModelMetrics(
        model_key="ann",
        label="ANN",
        status="completed",
        sample_count=10,
        position_rmse_mm=1.234,
        axis_position_rmse_mm=[0.5, 0.5, 0.5],
    )
    fake_result = ModelingEvaluationResult(
        dataset_summary=controller._selected_dataset_summary,
        artifact_details=controller._selected_artifact_details,
        evaluation_scope_requested="full_dataset",
        evaluation_scope_used="full_dataset",
        evaluation_scope_note="",
        selected_sample_count=10,
        output_dir=tmp_path / "data" / "modeling_results" / "clipfake",
        summary_path=tmp_path / "summary.json",
        metadata_path=tmp_path / "metadata.json",
        comparison_csv_path=tmp_path / "comp.csv",
        phase_csv_path=None,
        plot_paths={},
        model_evaluations={"ann": ModelEvaluation(metrics=fake_metrics, predictions=None)},
        visualization_model=VisualizationModel(summary_lines=["ok"]),
    )
    controller._last_result = fake_result  # type: ignore[attr-defined]
    try:
        tab.show()
        tab.update(controller.refresh())
        assert tab.copy_summary_button.isEnabled()
        tab._copy_summary_to_clipboard()
        from PySide6.QtGui import QGuiApplication

        clipboard = QGuiApplication.clipboard()
        assert clipboard is not None
        text = clipboard.text()
        assert "1.234" in text
        assert "ANN" in text
    finally:
        tab.close()
        controller.shutdown()


def test_modeling_tab_workflow_header_renders_4_steps(tmp_path: Path) -> None:
    """Quick-start workflow header surfaces the 4-step journey without scrolling."""
    _app()
    run_dir = _write_modeling_run(tmp_path)
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    controller.select_dataset(str(run_dir))
    tab = ModelingTab(controller)
    try:
        tab.show()
        tab.update(controller.refresh())
        # Walk the tab's children for labels matching the workflow step texts.
        from PySide6.QtWidgets import QLabel as _QLabel

        labels = [w.text() for w in tab.findChildren(_QLabel)]
        # The 4 numbered steps each have a unique text fragment.
        assert any("training dataset" in t.lower() for t in labels)
        assert any("train ann" in t.lower() for t in labels)
        assert any("separate test dataset" in t.lower() for t in labels)
        assert any("rmse per model" in t.lower() for t in labels)
    finally:
        tab.close()
        controller.shutdown()


def test_modeling_tab_small_eval_dataset_warning_chip(tmp_path: Path) -> None:
    """When the selected dataset has fewer than 100 accepted samples, the warning chip fires."""
    _app()
    run_dir = _write_modeling_run(tmp_path)  # _write_modeling_run fixture has 3 accepted samples
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    controller.select_dataset(str(run_dir))
    tab = ModelingTab(controller)
    try:
        tab.show()
        tab.update(controller.refresh())
        # 3 accepted samples is well under the 100-row threshold.
        assert tab.eval_sample_count_chip.isVisible() is True
        text = tab.eval_sample_count_chip.text().lower()
        assert "noisy" in text
        assert "100" in text
    finally:
        tab.close()
        controller.shutdown()


def _write_real_hardware_test_run(
    tmp_path: Path,
    *,
    name: str,
    run_id: str,
    complete_rows: int = 150,
    mock: bool = False,
    trust_mode: str = "thesis_trusted",
) -> Path:
    """Build a fixture that satisfies the strict 'real hardware' thesis-grade rule
    when all flags are at defaults. Flip any flag/count to violate one rule at a time."""
    run_dir = tmp_path / "data" / "experiments" / "collect_pose_command_dataset" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "1.0",
        "experiment_name": "collect_pose_command_dataset",
        "run_id": run_id,
        "timestamp_utc": "2026-05-16T12:00:00+00:00",
    }
    summary = {
        "schema_version": "1.0",
        "experiment_name": "collect_pose_command_dataset",
        "run_id": run_id,
        "success": True,
        "sample_counts": {"total": complete_rows},
        "status": "success",
        "experiment_metrics": {
            "dataset_mode": "angular_test_mesh",
            "accepted_sample_count": complete_rows,
            "rejected_sample_count": 0,
            "run_trust_mode": trust_mode,
            "valid_for_model_training": True,
            "target_valid_sample_count": complete_rows,
            "complete_training_row_count": complete_rows,
            "mock_mode": mock,
            "run_provenance": {
                "runtime_tip_calibration": {"mode": "latest_accepted", "trust_level": "trusted"},
                "pretension_artifact": {"active_source_type": "accepted_artifact", "status": "ready"},
                "mock_mode": mock,
            },
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
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
    return run_dir


def _make_thesis_grade_evaluation(
    *,
    tmp_path: Path,
    controller: ModelingController,
    train_run: Path,
    test_run: Path,
    exploratory: bool = False,
) -> ModelingEvaluationResult:
    """Build a populated ModelingEvaluationResult that should pass _eval_is_thesis_grade."""
    from continuum_robot.modeling.ann_training import load_modeling_dataset_summary
    from continuum_robot.modeling.analysis import (
        ArtifactDetails as _ArtifactDetails,
        ModelingEvaluationResult as _ModelingEvaluationResult,
    )

    fake_metrics = ModelMetrics(
        model_key="ann",
        label="ANN",
        status="completed",
        sample_count=10,
        position_rmse_mm=1.0,
        axis_position_rmse_mm=[0.5, 0.5, 0.5],
    )
    train_summary = load_modeling_dataset_summary(train_run)
    test_summary = load_modeling_dataset_summary(test_run)
    artifact_metadata = {
        "training_provenance": {
            "exploratory_training_override": bool(exploratory),
        },
    }
    # We can't construct an ArtifactDetails without all fields; build a minimal stand-in
    # using the existing _write_artifact + load helper.
    artifact_dir = _write_artifact(tmp_path, train_run)
    # Overwrite its metadata with our provenance fixture.
    (artifact_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "created_at_utc": "2026-05-16T00:00:00Z",
                "status": "completed",
                "model": {"output_target": "xyz", "hidden_layers": [32, 32]},
                "training": {"epochs_completed": 1, "best_validation_loss": 0.5},
                "dataset": {"run_name": train_summary.run_name, "path": str(train_run)},
                "backend": {"selected_backend": "cpu"},
                "files": {
                    "loss_history_path": str(artifact_dir / "loss_history.csv"),
                    "model_path": str(artifact_dir / "model.pt"),
                },
                **artifact_metadata,
            }
        ),
        encoding="utf-8",
    )
    # Force the controller to re-discover artifacts now that we've just written one;
    # otherwise refresh() short-circuits because _catalog_dirty is False and the new
    # artifact won't appear in state.artifact_details.
    controller.set_artifact_root(controller.artifact_root)
    controller.select_artifact(str(artifact_dir))
    state = controller.refresh()
    assert state.artifact_details is not None, (
        "fixture invariant: artifact must be loadable; got None"
    )
    return _ModelingEvaluationResult(
        dataset_summary=train_summary,
        artifact_details=state.artifact_details,
        evaluation_scope_requested="full_dataset",
        evaluation_scope_used="separate_test_dataset",
        evaluation_scope_note="",
        selected_sample_count=150,
        output_dir=tmp_path / "data" / "modeling_results" / "audit_fake",
        summary_path=tmp_path / "summary.json",
        metadata_path=tmp_path / "metadata.json",
        comparison_csv_path=tmp_path / "comp.csv",
        phase_csv_path=None,
        plot_paths={},
        model_evaluations={"ann": ModelEvaluation(metrics=fake_metrics, predictions=None)},
        visualization_model=VisualizationModel(summary_lines=["ok"]),
        test_dataset_summary=test_summary,
    )


def test_thesis_grade_chip_fires_when_all_conditions_met(tmp_path: Path) -> None:
    """All 5 audit conditions satisfied ⇒ thesis-grade chip visible, caveat hidden."""
    _app()
    train_run = _write_real_hardware_test_run(tmp_path, name="train_run_real", run_id="train_001")
    test_run = _write_real_hardware_test_run(tmp_path, name="test_run_real", run_id="test_002")
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    controller.select_dataset(str(train_run))
    tab = ModelingTab(controller)
    try:
        tab.show()
        tab.update(controller.refresh())
        fake_result = _make_thesis_grade_evaluation(
            tmp_path=tmp_path, controller=controller, train_run=train_run, test_run=test_run
        )
        controller._last_result = fake_result  # type: ignore[attr-defined]
        tab.update(controller.refresh())
        assert tab.thesis_grade_chip.isVisible() is True
        assert tab.same_session_chip.isVisible() is False
    finally:
        tab.close()
        controller.shutdown()


def test_thesis_grade_blocked_when_train_test_same_run_id(tmp_path: Path) -> None:
    """Gate 2: same run_id between train and test ⇒ no thesis-grade chip."""
    _app()
    # Both runs carry the same run_id "shared_001" — even though paths differ.
    train_run = _write_real_hardware_test_run(tmp_path, name="run_a", run_id="shared_001")
    test_run = _write_real_hardware_test_run(tmp_path, name="run_b", run_id="shared_001")
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    controller.select_dataset(str(train_run))
    tab = ModelingTab(controller)
    try:
        tab.show()
        tab.update(controller.refresh())
        fake_result = _make_thesis_grade_evaluation(
            tmp_path=tmp_path, controller=controller, train_run=train_run, test_run=test_run
        )
        controller._last_result = fake_result  # type: ignore[attr-defined]
        tab.update(controller.refresh())
        assert tab.thesis_grade_chip.isVisible() is False
    finally:
        tab.close()
        controller.shutdown()


def test_thesis_grade_blocked_when_train_test_same_path(tmp_path: Path) -> None:
    """Gate 2: same resolved path between train and test (even if run_id strings differ)
    ⇒ no thesis-grade chip. evaluate_models's same-path collapse also handles this at
    the source — but the controller-side gate must defend it too."""
    _app()
    train_run = _write_real_hardware_test_run(tmp_path, name="single_run", run_id="single_001")
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    controller.select_dataset(str(train_run))
    tab = ModelingTab(controller)
    try:
        tab.show()
        tab.update(controller.refresh())
        # Build a result whose test_dataset_summary points at the SAME run.
        fake_result = _make_thesis_grade_evaluation(
            tmp_path=tmp_path, controller=controller, train_run=train_run, test_run=train_run
        )
        controller._last_result = fake_result  # type: ignore[attr-defined]
        tab.update(controller.refresh())
        assert tab.thesis_grade_chip.isVisible() is False
    finally:
        tab.close()
        controller.shutdown()


def test_thesis_grade_blocked_when_test_dataset_is_mock(tmp_path: Path) -> None:
    """Gate 3: mock_mode test dataset ⇒ no thesis-grade chip."""
    _app()
    train_run = _write_real_hardware_test_run(tmp_path, name="real_train", run_id="t01")
    test_run = _write_real_hardware_test_run(tmp_path, name="mock_test", run_id="t02", mock=True)
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    controller.select_dataset(str(train_run))
    tab = ModelingTab(controller)
    try:
        tab.show()
        tab.update(controller.refresh())
        fake_result = _make_thesis_grade_evaluation(
            tmp_path=tmp_path, controller=controller, train_run=train_run, test_run=test_run
        )
        controller._last_result = fake_result  # type: ignore[attr-defined]
        tab.update(controller.refresh())
        assert tab.thesis_grade_chip.isVisible() is False
    finally:
        tab.close()
        controller.shutdown()


def test_thesis_grade_blocked_when_test_dataset_servo_only(tmp_path: Path) -> None:
    """Gate 3: servo_only / lower_trust test dataset ⇒ no thesis-grade chip."""
    _app()
    train_run = _write_real_hardware_test_run(tmp_path, name="real_train", run_id="t10")
    test_run = _write_real_hardware_test_run(
        tmp_path, name="servo_only_test", run_id="t11", trust_mode="servo_only"
    )
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    controller.select_dataset(str(train_run))
    tab = ModelingTab(controller)
    try:
        tab.show()
        tab.update(controller.refresh())
        fake_result = _make_thesis_grade_evaluation(
            tmp_path=tmp_path, controller=controller, train_run=train_run, test_run=test_run
        )
        controller._last_result = fake_result  # type: ignore[attr-defined]
        tab.update(controller.refresh())
        assert tab.thesis_grade_chip.isVisible() is False
    finally:
        tab.close()
        controller.shutdown()


def test_thesis_grade_blocked_when_test_dataset_too_few_complete_rows(tmp_path: Path) -> None:
    """Gate 4: test dataset with <100 complete rows ⇒ no thesis-grade chip."""
    _app()
    train_run = _write_real_hardware_test_run(tmp_path, name="real_train", run_id="t20")
    test_run = _write_real_hardware_test_run(
        tmp_path, name="tiny_test", run_id="t21", complete_rows=50
    )
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    controller.select_dataset(str(train_run))
    tab = ModelingTab(controller)
    try:
        tab.show()
        tab.update(controller.refresh())
        fake_result = _make_thesis_grade_evaluation(
            tmp_path=tmp_path, controller=controller, train_run=train_run, test_run=test_run
        )
        controller._last_result = fake_result  # type: ignore[attr-defined]
        tab.update(controller.refresh())
        assert tab.thesis_grade_chip.isVisible() is False
    finally:
        tab.close()
        controller.shutdown()


def test_thesis_grade_blocked_when_artifact_is_exploratory(tmp_path: Path) -> None:
    """Gate 5: ANN trained with exploratory_training_override ⇒ no thesis-grade chip."""
    _app()
    train_run = _write_real_hardware_test_run(tmp_path, name="real_train", run_id="t30")
    test_run = _write_real_hardware_test_run(tmp_path, name="real_test", run_id="t31")
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    controller.select_dataset(str(train_run))
    tab = ModelingTab(controller)
    try:
        tab.show()
        tab.update(controller.refresh())
        # Same as the all-good fixture, but with exploratory=True on the artifact.
        fake_result = _make_thesis_grade_evaluation(
            tmp_path=tmp_path,
            controller=controller,
            train_run=train_run,
            test_run=test_run,
            exploratory=True,
        )
        controller._last_result = fake_result  # type: ignore[attr-defined]
        tab.update(controller.refresh())
        assert tab.thesis_grade_chip.isVisible() is False
    finally:
        tab.close()
        controller.shutdown()


def test_evaluate_models_collapses_test_override_when_paths_match(tmp_path: Path) -> None:
    """Bug 1 fix at the analysis layer: if test_dataset_path == dataset_path, evaluate_models
    must NOT set scope_used='separate_test_dataset'."""
    from continuum_robot.modeling.analysis import evaluate_models, ModelingEvaluationConfig

    pytest.importorskip("torch")
    run_dir = _write_modeling_run(tmp_path)
    config = ModelingEvaluationConfig(
        include_mike=False,
        include_camarillo=False,
        include_ann=False,
        evaluation_scope="full_dataset",
        results_root=str(tmp_path / "data" / "modeling_results"),
    )
    result = evaluate_models(
        project_root=tmp_path,
        dataset_path=run_dir,
        artifact_path=None,
        config=config,
        test_dataset_path=run_dir,  # ← intentionally the same as training
    )
    assert result.evaluation_scope_used != "separate_test_dataset"
    assert result.test_dataset_summary is None


def test_modeling_tab_dataset_mode_chip_highlights_angular_test_mesh(tmp_path: Path) -> None:
    """When the selected dataset is an angular_test_mesh run, the chip is green."""
    _app()
    run_dir = tmp_path / "data" / "experiments" / "collect_pose_command_dataset" / "20260515_999999_mesh"
    run_dir.mkdir(parents=True)
    metadata = {
        "schema_version": "1.0",
        "experiment_name": "collect_pose_command_dataset",
        "run_id": "mesh",
        "timestamp_utc": "2026-05-15T12:00:00+00:00",
    }
    summary = {
        "schema_version": "1.0",
        "experiment_name": "collect_pose_command_dataset",
        "run_id": "mesh",
        "success": True,
        "sample_counts": {"total": 4},
        "status": "success",
        "experiment_metrics": {
            "dataset_mode": "angular_test_mesh",
            "accepted_sample_count": 4,
            "rejected_sample_count": 0,
            "run_trust_mode": "thesis_trusted",
            "valid_for_model_training": True,
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with (run_dir / "modeling_dataset_export.jsonl").open("w", encoding="utf-8") as handle:
        for i in range(4):
            handle.write(
                json.dumps(
                    {
                        "sequence_index": i,
                        "accepted": True,
                        "resolved_cable_command_cm": [0.1, 0.2, 0.3, 0.4],
                        "tip_position_xyz_mm": [1.0, 2.0, 3.0],
                        "tip_tangent_xyz": [0.0, 0.0, 1.0],
                    }
                )
                + "\n"
            )
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    controller.select_dataset(str(run_dir))
    tab = ModelingTab(controller)
    try:
        tab.show()
        tab.update(controller.refresh())
        assert tab.dataset_mode_chip.isVisible()
        text = tab.dataset_mode_chip.text().lower()
        assert "angular test mesh" in text
        assert "wolfe" in text
    finally:
        tab.close()
        controller.shutdown()


def test_modeling_controller_test_dataset_path_threads_through_evaluate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A test-dataset override on the controller must reach evaluate_models()."""
    _app()
    train_run = _write_modeling_run(tmp_path)
    test_run = _write_modeling_run(tmp_path)
    artifact_dir = _write_artifact(tmp_path, train_run)
    captured: dict = {}

    def _fake_evaluate(*, project_root, dataset_path, artifact_path, config, test_dataset_path=None):
        captured["dataset_path"] = dataset_path
        captured["test_dataset_path"] = test_dataset_path
        # Minimal result so the worker can finish cleanly.
        out_dir = tmp_path / "data" / "modeling_results" / "fake_test_dataset_eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(
            json.dumps({"dataset_run_name": train_run.name, "selected_sample_count": 1, "models": {}}),
            encoding="utf-8",
        )
        return ModelingEvaluationResult(
            dataset_summary=__import__(
                "continuum_robot.modeling.ann_training", fromlist=["load_modeling_dataset_summary"]
            ).load_modeling_dataset_summary(dataset_path),
            artifact_details=None,
            evaluation_scope_requested="full_dataset",
            evaluation_scope_used="separate_test_dataset" if test_dataset_path else "full_dataset",
            evaluation_scope_note="fake",
            selected_sample_count=1,
            output_dir=out_dir,
            summary_path=out_dir / "summary.json",
            metadata_path=out_dir / "metadata.json",
            comparison_csv_path=out_dir / "comp.csv",
            phase_csv_path=None,
            plot_paths={},
            model_evaluations={},
            visualization_model=VisualizationModel(summary_lines=["ok"]),
        )

    monkeypatch.setattr(controller_module, "evaluate_models", _fake_evaluate)
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    controller.select_dataset(str(train_run))
    controller.select_artifact(str(artifact_dir))
    controller.set_test_dataset_path(str(test_run))
    state = controller.refresh()
    assert state.selected_test_dataset_path == str(test_run)
    controller.evaluate()
    QApplication.processEvents()
    time.sleep(0.12)
    state = controller.refresh()
    assert captured.get("test_dataset_path") is not None
    assert str(captured["test_dataset_path"]) == str(test_run)
    assert state.last_eval_same_session is False, (
        "scope_used == 'separate_test_dataset' must clear the same-session flag"
    )


def test_modeling_tab_no_longer_has_two_segment_widgets(tmp_path: Path) -> None:
    """Sanity check: ModelingTab is single-segment only after the split."""
    _app()
    run_dir = _write_modeling_run(tmp_path)
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    controller.select_dataset(str(run_dir))
    tab = ModelingTab(controller)
    try:
        tab.update(controller.refresh())
        # None of the two-segment widgets exist on the single-segment Modeling tab.
        for attr in (
            "two_segment_run_list",
            "two_segment_run_button",
            "two_segment_strict_check",
            "two_segment_label_mode_combo",
            "two_segment_hidden_combo",
            "two_segment_toggle_button",
            "_two_segment_card",
        ):
            assert not hasattr(tab, attr), f"ModelingTab should not have {attr!r}"
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




def test_modeling_tab_same_session_chip_visible_after_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When an evaluation reuses the training run for its test split, the caveat chip lights up."""
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
        tab.show()  # isVisible() only reflects shown state
        tab.update(controller.refresh())
        # Before any evaluation: chip hidden.
        assert tab.same_session_chip.isVisible() is False
        # Inject a fake evaluation result so we don't have to run torch.
        fake_metrics = ModelMetrics(
            model_key="ann",
            label="ANN",
            status="completed",
            sample_count=2,
            position_rmse_mm=1.23,
            mean_position_error_mm=1.1,
            max_position_error_mm=2.0,
            axis_position_rmse_mm=[0.5, 0.7, 0.9],
        )
        fake_result = ModelingEvaluationResult(
            dataset_summary=controller._selected_dataset_summary,
            artifact_details=controller._selected_artifact_details,
            evaluation_scope_requested="artifact_test_split",
            evaluation_scope_used="artifact_test_split",
            evaluation_scope_note="",
            selected_sample_count=2,
            output_dir=tmp_path / "data" / "modeling_results" / "fake",
            summary_path=tmp_path / "summary.json",
            metadata_path=tmp_path / "metadata.json",
            comparison_csv_path=tmp_path / "comp.csv",
            phase_csv_path=None,
            plot_paths={},
            model_evaluations={"ann": ModelEvaluation(metrics=fake_metrics, predictions=None)},
            visualization_model=VisualizationModel(summary_lines=["ok"]),
        )
        controller._last_result = fake_result  # type: ignore[attr-defined]
        tab.update(controller.refresh())
        assert tab.same_session_chip.isVisible() is True
        # Headline pairs include the per-model RMSE row.
        labels = [pair[0] for pair in controller.refresh().headline_rmse_pairs]
        assert "ANN" in labels
    finally:
        tab.close()
        controller.shutdown()


def test_modeling_controller_filters_inverse_artifacts(tmp_path: Path) -> None:
    """Inverse-model artifacts must not appear in the Modeling-tab artifact list."""
    _app()
    forward_dir = _write_artifact(tmp_path, _write_modeling_run(tmp_path))
    # Stamp a second artifact as inverse via its metadata.
    inverse_dir = tmp_path / "data" / "models" / "ann" / "20260515_999999_inverse"
    inverse_dir.mkdir(parents=True, exist_ok=True)
    (inverse_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "artifact_kind": "legacy_ann_inverse_xyz_to_cable_v1",
                "created_at_utc": "2026-05-15T12:00:00+00:00",
                "status": "completed",
                "model": {"output_target": "cable_from_xyz", "hidden_layers": [32, 32]},
                "training": {"epochs_completed": 1, "best_validation_loss": 0.5},
                "dataset": {"run_name": "inverse_src"},
                "backend": {"selected_backend": "cpu"},
                "files": {"model_path": str(inverse_dir / "model.pt")},
            }
        ),
        encoding="utf-8",
    )
    (inverse_dir / "model.pt").write_bytes(b"x")
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    state = controller.refresh()
    artifact_names = [a.artifact_name for a in state.artifacts]
    assert forward_dir.name in artifact_names
    assert inverse_dir.name not in artifact_names  # inverse hidden from the comparison list
    controller.shutdown()


def test_modeling_refresh_caches_eval_warn_disk_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: repeated refresh() ticks must not re-read the eval summary.json from
    disk. The 5 Hz refresh timer was hammering this read on every fire — caching by
    (path, mtime) is what unblocks scroll-thread paint."""
    _app()
    run_dir = _write_modeling_run(tmp_path)
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    controller.select_dataset(str(run_dir))
    # Prime: do an initial refresh so the cache fills.
    controller.refresh()

    # Count Path.read_text calls during 10 subsequent refresh ticks. With the cache
    # working, the eval warn path reads zero times (cache hit on each tick); without
    # it, it would read 10 times.
    read_count = {"n": 0}
    real_read_text = Path.read_text

    def _counting_read_text(self, *args, **kwargs):
        if self.name == "summary.json" and str(self).startswith(str(run_dir)):
            read_count["n"] += 1
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _counting_read_text)
    for _ in range(10):
        controller.refresh()
    assert read_count["n"] == 0, (
        f"eval-warn cache miss on steady-state refresh — {read_count['n']} disk reads "
        "across 10 ticks (expected 0 once primed). The 5 Hz refresh timer would block "
        "the GUI paint thread."
    )

    # Sanity: bumping the file's mtime invalidates the cache → next refresh re-reads.
    (run_dir / "summary.json").touch()
    controller.refresh()
    assert read_count["n"] >= 1, "cache should re-read after mtime change"
    controller.shutdown()


def test_modeling_refresh_caches_trainability_jsonl_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the two-segment trainability pairs builder must cache its result
    across refresh() ticks instead of re-parsing samples.jsonl every 200ms.
    """
    _app()
    run_dir = _write_modeling_run(tmp_path)
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    controller.select_dataset(str(run_dir))
    controller.refresh()  # prime

    # Spy on the validator that does the heavy JSONL parse.
    call_count = {"n": 0}
    real_validate = controller.validate_two_segment_modeling_trainability

    def _counting_validate(*args, **kwargs):
        call_count["n"] += 1
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(
        controller, "validate_two_segment_modeling_trainability", _counting_validate
    )
    for _ in range(10):
        controller.refresh()
    # No selected two-segment runs in this fixture, so the validator shouldn't fire at
    # all — but if it did, the cache should keep the count at most 1.
    assert call_count["n"] <= 1, (
        f"trainability cache miss — validator called {call_count['n']} times across "
        "10 steady-state ticks (expected ≤1)."
    )
    controller.shutdown()


# ---------------------------------------------------------------------------
# External Model Comparison card (Upload .pt + side-by-side 3D error plot)
# ---------------------------------------------------------------------------


def _write_xyz_artifact(tmp_path: Path, *, name: str, hidden_layers: list[int]) -> Path:
    """Build a legal XYZ-target artifact with a real-shape state_dict.

    Distinct from _write_artifact above (which uses fake bytes for model.pt) —
    here we need PyTorch to actually load+run the model, so we save a real
    state_dict matching the declared architecture.
    """
    torch = pytest.importorskip("torch")
    from continuum_robot.modeling import ann_training as training_module

    artifact_dir = tmp_path / "data" / "models" / "ann" / name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model = training_module._build_legacy_ann_model(
        torch=torch,
        input_dim=4,
        output_dim=3,
        hidden_layers=hidden_layers,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    torch.save(model.state_dict(), artifact_dir / "model.pt")
    (artifact_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "artifact_kind": "legacy_ann_xyz_v1",
                "created_at_utc": "2026-05-18T00:00:00+00:00",
                "status": "completed",
                "model": {
                    "input_dim": 4,
                    "output_dim": 3,
                    "hidden_layers": hidden_layers,
                    "dtype": "float32",
                    "output_target": "xyz",
                },
                "training": {"epochs_completed": 1, "best_validation_loss": 0.1},
                "dataset": {"run_name": "fake", "path": str(tmp_path)},
                "backend": {"selected_backend": "cpu"},
                "files": {"model_path": str(artifact_dir / "model.pt")},
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "training_config.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "split_manifest.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "loss_history.csv").write_text("epoch,train_loss,validation_loss,elapsed_s\n", encoding="utf-8")
    return artifact_dir


def _write_xyz_artifact_at(root: Path, *, name: str, hidden_layers: list[int]) -> Path:
    """Like :func:`_write_xyz_artifact` but writes under an arbitrary ``root``.

    Used by tests that want the artifact to be loadable + functional BUT not
    discoverable by the controller (e.g., for testing slot gating without the
    auto-pick fallback firing). Put the artifact at ``<root>/<name>/`` instead
    of ``<tmp_path>/data/models/ann/<name>/``.
    """
    torch = pytest.importorskip("torch")
    from continuum_robot.modeling import ann_training as training_module

    artifact_dir = Path(root) / name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model = training_module._build_legacy_ann_model(
        torch=torch,
        input_dim=4,
        output_dim=3,
        hidden_layers=hidden_layers,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    torch.save(model.state_dict(), artifact_dir / "model.pt")
    (artifact_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "artifact_kind": "legacy_ann_xyz_v1",
                "created_at_utc": "2026-05-18T00:00:00+00:00",
                "status": "completed",
                "model": {
                    "input_dim": 4,
                    "output_dim": 3,
                    "hidden_layers": hidden_layers,
                    "dtype": "float32",
                    "output_target": "xyz",
                },
                "training": {"epochs_completed": 1, "best_validation_loss": 0.1},
                "dataset": {"run_name": "fake", "path": str(root)},
                "backend": {"selected_backend": "cpu"},
                "files": {"model_path": str(artifact_dir / "model.pt")},
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "training_config.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "split_manifest.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "loss_history.csv").write_text(
        "epoch,train_loss,validation_loss,elapsed_s\n", encoding="utf-8"
    )
    return artifact_dir


def _write_workspace_dataset(tmp_path: Path, *, run_name: str, n: int = 64) -> Path:
    """Build a dataset run with enough samples for the comparison's 10-sample floor."""
    import numpy as _np

    run_dir = tmp_path / "data" / "experiments" / "collect_pose_command_dataset" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment_name": "collect_pose_command_dataset",
                "run_id": run_name,
                "timestamp_utc": "2026-05-18T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment_name": "collect_pose_command_dataset",
                "run_id": run_name,
                "success": True,
                "sample_counts": {"total": n},
                "status": "success",
                "experiment_metrics": {
                    "dataset_mode": "angular_test_mesh",
                    "accepted_sample_count": n,
                    "rejected_sample_count": 0,
                    "run_trust_mode": "thesis_trusted",
                    "valid_for_model_training": True,
                    "target_valid_sample_count": n,
                    "complete_training_row_count": n,
                    "mock_mode": False,
                },
            }
        ),
        encoding="utf-8",
    )
    with (run_dir / "modeling_dataset_export.jsonl").open("w", encoding="utf-8") as handle:
        for i in range(n):
            angle = (i / max(n - 1, 1)) * 2.0 * _np.pi
            handle.write(
                json.dumps(
                    {
                        "sequence_index": i,
                        "step_index": i // 4,
                        "sample_index": i,
                        "accepted": True,
                        "resolved_cable_command_cm": [0.5 * _np.cos(angle), 0.5 * _np.sin(angle), 0.4, 0.3],
                        "tip_position_xyz_mm": [10.0 * _np.cos(angle), 10.0 * _np.sin(angle), 5.0 + 0.1 * i],
                        "tip_tangent_xyz": [0.0, 0.0, 1.0],
                    }
                )
                + "\n"
            )
    return run_dir


def test_comparison_card_renders_without_uploaded_model(tmp_path: Path) -> None:
    """Tab loads cleanly with the new comparison card; the Generate button stays
    disabled until all three inputs (dataset + artifact + .pt) are present."""
    _app()
    run_dir = _write_modeling_run(tmp_path)
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    controller.select_dataset(str(run_dir))
    tab = ModelingTab(controller)
    try:
        tab.show()
        tab.update(controller.refresh())
        assert tab.comparison_generate_button is not None
        assert tab.comparison_generate_button.isEnabled() is False  # no Model B yet
        assert tab.comparison_save_button.isEnabled() is False
        # Status hint should mention picking a Model B.
        assert "Model B" in tab.comparison_status_label.text() or "pt" in tab.comparison_status_label.text().lower()
    finally:
        tab.close()
        controller.shutdown()


def test_comparison_card_generate_enables_after_all_inputs_set(tmp_path: Path) -> None:
    """Once a dataset, an ANN artifact for Slot A (via back-compat dropdown
    fallback), AND an uploaded .pt for Slot B are all selected, the Generate
    button becomes enabled.

    Also asserts the new two-slot combos surface the chosen models: Model A
    appears in the Slot A combo (from the back-compat fallback to the main
    artifact selection), and Model B's archived upload appears in the Slot B
    combo with the ``uploaded_*`` prefix.
    """
    pytest.importorskip("torch")
    _app()
    run_dir = _write_workspace_dataset(tmp_path, run_name="compdata_a", n=24)
    artifact_a = _write_xyz_artifact(tmp_path, name="a_artifact", hidden_layers=[8, 8])
    bare_b = tmp_path / "uploads" / "b_model.pt"
    bare_b.parent.mkdir(parents=True, exist_ok=True)
    # Reuse the a-artifact's model.pt as a stand-in for the bare upload — same shape.
    shutil.copy(artifact_a / "model.pt", bare_b)
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    controller.select_dataset(str(run_dir))
    controller.select_artifact(str(artifact_a))
    controller.set_comparison_external_model_path(str(bare_b))
    tab = ModelingTab(controller)
    try:
        tab.show()
        tab.update(controller.refresh())
        assert tab.comparison_generate_button.isEnabled() is True
        # Slot B's combo should carry the archived upload's display name.
        b_combo = tab.comparison_model_b_combo
        b_label = b_combo.itemText(b_combo.currentIndex())
        assert "uploaded_" in b_label or "b_model" in b_label, (
            f"Slot B combo doesn't show the archived upload; got {b_label!r}"
        )
        # The dropdown ANN artifact (Model A back-compat path) is still
        # discoverable by name in the Slot A combo's items.
        a_items = [
            tab.comparison_model_a_combo.itemText(i)
            for i in range(tab.comparison_model_a_combo.count())
        ]
        assert any("a_artifact" in t for t in a_items), (
            f"Slot A combo missing the discovered artifact; items={a_items}"
        )
    finally:
        tab.close()
        controller.shutdown()


def test_comparison_card_runs_and_renders_canvas(tmp_path: Path) -> None:
    """End-to-end synchronous run: drive the controller through dataset →
    artifact → upload → run, wait for the worker, and assert the canvas
    re-rendered (comparison_result_id incremented from 0)."""
    pytest.importorskip("torch")
    _app()
    run_dir = _write_workspace_dataset(tmp_path, run_name="compdata_b", n=32)
    artifact_a = _write_xyz_artifact(tmp_path, name="cmp_a", hidden_layers=[8, 8])
    artifact_b_dir = _write_xyz_artifact(tmp_path, name="cmp_b", hidden_layers=[16, 16])
    bare_b = artifact_b_dir / "model.pt"
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    controller.select_dataset(str(run_dir))
    controller.select_artifact(str(artifact_a))
    controller.set_comparison_external_model_path(str(bare_b))
    tab = ModelingTab(controller)
    try:
        tab.show()
        tab.update(controller.refresh())
        starting_id = controller.refresh().comparison_result_id
        controller.run_external_comparison()
        # Worker is a thread; wait for it to finish.
        thread = getattr(controller, "_worker_thread", None)
        if thread is not None:
            thread.join(timeout=30.0)
        final_state = controller.refresh()
        assert final_state.comparison_active is False
        assert final_state.comparison_result_id > starting_id
        assert final_state.comparison_error_message == ""
        # Update the tab so the canvas swaps to the new figure.
        tab.update(final_state)
        # Save button should now be enabled.
        assert tab.comparison_save_button.isEnabled() is True
        # The controller's last comparison result is fetchable.
        result = controller.get_last_comparison_result()
        assert result is not None
        assert result.a_errors_mm.shape == (32,)
        assert result.b_errors_mm.shape == (32,)
    finally:
        tab.close()
        controller.shutdown()


def test_comparison_card_surfaces_error_for_bad_inputs(tmp_path: Path) -> None:
    """Clicking Generate without a Model B should populate
    comparison_error_message — not crash."""
    _app()
    run_dir = _write_modeling_run(tmp_path)
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    controller.select_dataset(str(run_dir))
    tab = ModelingTab(controller)
    try:
        tab.show()
        tab.update(controller.refresh())
        # No Model B chosen; calling run anyway should surface a useful error.
        controller.run_external_comparison()
        state = controller.refresh()
        assert state.comparison_active is False
        assert state.comparison_error_message  # non-empty
        tab.update(state)
        assert tab.comparison_error_label.isVisible() is True
    finally:
        tab.close()
        controller.shutdown()


def test_uploaded_pt_is_archived_into_artifact_root(tmp_path: Path) -> None:
    """When the operator uploads a .pt for Model B, the controller copies it
    into ``<artifact_root>/uploaded_<timestamp>_<basename>/`` with an inferred
    training_metadata.json so the file shows up in the artifact dropdown next
    refresh — the operator never has to re-browse for the same .pt twice.
    """
    pytest.importorskip("torch")
    _app()
    # Save a real bare state_dict at an external location.
    bare_pt = tmp_path / "downloads" / "old_model.pt"
    bare_pt.parent.mkdir(parents=True, exist_ok=True)
    import torch as _torch
    from continuum_robot.modeling import ann_training as training_module

    model = training_module._build_legacy_ann_model(
        torch=_torch,
        input_dim=4,
        output_dim=3,
        hidden_layers=[8, 8],
        device=_torch.device("cpu"),
        dtype=_torch.float32,
    )
    _torch.save(model.state_dict(), bare_pt)

    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    # No artifacts yet, no error.
    starting = controller.refresh()
    assert all(not str(a.path).startswith(str(tmp_path / "downloads")) for a in starting.artifacts)

    controller.set_comparison_external_model_path(str(bare_pt))

    # Refresh — the new uploaded artifact should appear in the dropdown.
    state = controller.refresh()
    upload_artifacts = [a for a in state.artifacts if a.artifact_name.startswith("uploaded_")]
    assert len(upload_artifacts) == 1
    archived = upload_artifacts[0]
    # The archived path should live under artifact_root, not under downloads.
    assert str(archived.path).startswith(str(tmp_path / "data" / "models" / "ann"))
    assert "old_model" in archived.artifact_name
    # Status message tells the operator where it landed.
    assert "uploaded_" in state.comparison_status_message
    # The model.pt file actually got copied (not just a symlink to /downloads).
    archived_pt = archived.path / "model.pt"
    assert archived_pt.exists()
    assert archived_pt.stat().st_size == bare_pt.stat().st_size
    # The controller's stored path now points at the ARCHIVED .pt, not the original.
    assert state.comparison_external_model_path == str(archived_pt)
    controller.shutdown()


def test_re_selecting_already_archived_upload_is_idempotent(tmp_path: Path) -> None:
    """If the operator picks a .pt that already lives under artifact_root (e.g.
    a previously-archived upload), the controller should NOT make a duplicate
    copy — just use the existing path."""
    pytest.importorskip("torch")
    _app()
    import torch as _torch
    from continuum_robot.modeling import ann_training as training_module

    artifact_root = tmp_path / "data" / "models" / "ann"
    # Pre-existing archived upload.
    existing_dir = artifact_root / "uploaded_20260518_010101_old_model"
    existing_dir.mkdir(parents=True)
    model = training_module._build_legacy_ann_model(
        torch=_torch,
        input_dim=4,
        output_dim=3,
        hidden_layers=[8, 8],
        device=_torch.device("cpu"),
        dtype=_torch.float32,
    )
    existing_pt = existing_dir / "model.pt"
    _torch.save(model.state_dict(), existing_pt)
    (existing_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "artifact_kind": "legacy_ann_xyz_v1",
                "created_at_utc": "2026-05-18T01:01:01+00:00",
                "status": "completed",
                "model": {
                    "input_dim": 4, "output_dim": 3, "hidden_layers": [8, 8],
                    "dtype": "float32", "output_target": "xyz",
                },
                "training": {"epochs_completed": 1, "best_validation_loss": 0.1},
                "dataset": {"run_name": "uploaded", "path": ""},
                "backend": {"selected_backend": "cpu"},
                "files": {"model_path": str(existing_pt)},
            }
        ),
        encoding="utf-8",
    )
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=artifact_root,
        results_root=tmp_path / "data" / "modeling_results",
    )
    # Pick the already-archived .pt as if browsing back to a previous upload.
    controller.set_comparison_external_model_path(str(existing_pt))
    state = controller.refresh()
    # No NEW uploaded_* directory should have been created.
    upload_dirs = list(artifact_root.glob("uploaded_*"))
    assert len(upload_dirs) == 1
    # The stored path points at the same file (no duplicate).
    assert state.comparison_external_model_path == str(existing_pt)
    controller.shutdown()


def test_dataset_list_is_newest_first(tmp_path: Path) -> None:
    """Regression: the dataset picker must show newest entries at the TOP of
    the trainable group, not at the bottom. Operators were missing their just-
    collected runs because they were sorted to position 20+ of 30+."""
    _app()
    output_root = tmp_path / "data" / "experiments"
    # Three runs with monotonic timestamps.
    older = output_root / "collect_pose_command_dataset" / "20260101_120000_older_run"
    middle = output_root / "collect_pose_command_dataset" / "20260301_120000_middle_run"
    newer = output_root / "collect_pose_command_dataset" / "20260518_164344_newer_run"
    for run_dir, stamp in [
        (older, "2026-01-01T12:00:00+00:00"),
        (middle, "2026-03-01T12:00:00+00:00"),
        (newer, "2026-05-18T16:43:44+00:00"),
    ]:
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "experiment_name": "collect_pose_command_dataset",
                    "run_id": run_dir.name,
                    "timestamp_utc": stamp,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "experiment_name": "collect_pose_command_dataset",
                    "run_id": run_dir.name,
                    "success": True,
                    "sample_counts": {"total": 4000},
                    "status": "success",
                    "experiment_metrics": {
                        "dataset_mode": "workspace_coverage",
                        "accepted_sample_count": 4000,
                        "rejected_sample_count": 0,
                        "run_trust_mode": "thesis_trusted",
                        "valid_for_model_training": True,
                        "target_valid_sample_count": 4000,
                        "complete_training_row_count": 4000,
                        "mock_mode": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        # Write enough rows so the trainability check passes (MIN_COMPLETE_ROWS_FOR_TRAINING).
        with (run_dir / "modeling_dataset_export.jsonl").open("w", encoding="utf-8") as h:
            for i in range(200):
                h.write(
                    json.dumps(
                        {
                            "sequence_index": i,
                            "step_index": i // 4,
                            "sample_index": i,
                            "accepted": True,
                            "resolved_cable_command_cm": [0.1, 0.2, 0.3, 0.4],
                            "tip_position_xyz_mm": [1.0 + 0.1 * i, 2.0, 3.0],
                            "tip_tangent_xyz": [0.0, 0.0, 1.0],
                        }
                    )
                    + "\n"
                )
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=output_root,
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    state = controller.refresh()
    names = [d.run_name for d in state.datasets]
    # Newest run must be at index 0.
    assert names[0] == "20260518_164344_newer_run", (
        f"newest run should be first; got order: {names}"
    )
    # Older runs appear after.
    assert names.index("20260518_164344_newer_run") < names.index("20260301_120000_middle_run")
    assert names.index("20260301_120000_middle_run") < names.index("20260101_120000_older_run")
    controller.shutdown()


def test_comparison_warnings_appear_in_status_not_on_figure(tmp_path: Path) -> None:
    """Operator asked for a clean thesis-bound PNG: warnings must NOT be drawn on
    the figure. They should still surface in the GUI status message so the
    auto-detect diagnostics remain visible during interaction."""
    pytest.importorskip("torch")
    _app()
    run_dir = _write_workspace_dataset(tmp_path, run_name="warn_ds", n=32)
    # Model A: discovered artifact. Model B: bare .pt that triggers a warning
    # (no sidecar metadata ⇒ bare-.pt warning fires unconditionally).
    artifact_a = _write_xyz_artifact(tmp_path, name="cmp_a", hidden_layers=[8, 8])
    bare_b = tmp_path / "uploads" / "bare.pt"
    bare_b.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(artifact_a / "model.pt", bare_b)
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    controller.select_dataset(str(run_dir))
    controller.select_artifact(str(artifact_a))
    controller.set_comparison_external_model_path(str(bare_b))
    controller.refresh()  # resolve _selected_artifact_details before kicking off the worker
    controller.run_external_comparison()
    thread = getattr(controller, "_worker_thread", None)
    if thread is not None:
        thread.join(timeout=30.0)
    state = controller.refresh()
    # Status carries the warning prefix.
    assert "⚠" in state.comparison_status_message
    # And the figure doesn't have an extra free-floating text artist for warnings.
    from continuum_robot.modeling.model_comparison import build_comparison_figure

    result = controller.get_last_comparison_result()
    assert result is not None
    figure = build_comparison_figure(result)
    try:
        # Three axes total: two 3D + colorbar. Confirm no extra text artists
        # were planted at the figure level (would indicate a footer regression).
        figure_level_texts = [
            t for t in figure.texts if t is not figure._suptitle  # noqa: SLF001
        ]
        assert figure_level_texts == [], (
            f"figure has unexpected footer text(s): "
            f"{[t.get_text() for t in figure_level_texts]}"
        )
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)
    controller.shutdown()


def test_comparison_canvas_links_3d_axis_rotation(tmp_path: Path) -> None:
    """Linked rotation: dragging on one 3D panel updates the other's view_init.

    Verifies the motion-notify hook the tab installs on its FigureCanvas.
    The hook keys off ``ax.button_pressed`` (matplotlib's own drag flag) rather
    than ``event.button``, because motion_notify_event always reports
    ``button=None`` while a drag is in progress — the button-press event has
    already fired separately. Simulate that by setting ``button_pressed``
    directly before synthesizing the motion event."""
    pytest.importorskip("torch")
    _app()
    run_dir = _write_workspace_dataset(tmp_path, run_name="rot_ds", n=32)
    artifact_a = _write_xyz_artifact(tmp_path, name="rot_a", hidden_layers=[8, 8])
    artifact_b = _write_xyz_artifact(tmp_path, name="rot_b", hidden_layers=[8, 8])
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    controller.select_dataset(str(run_dir))
    controller.select_artifact(str(artifact_a))
    controller.set_comparison_external_model_path(str(artifact_b / "model.pt"))
    controller.refresh()
    controller.run_external_comparison()
    thread = getattr(controller, "_worker_thread", None)
    if thread is not None:
        thread.join(timeout=30.0)
    tab = ModelingTab(controller)
    try:
        tab.show()
        tab.update(controller.refresh())
        three_d = [ax for ax in tab.comparison_canvas.figure.axes if ax.name == "3d"]
        assert len(three_d) == 2
        ax_a, ax_b = three_d
        # Set up a mismatch + flag ax_a as the one being dragged (matches what
        # matplotlib does in real interaction after a button_press_event).
        ax_a.view_init(elev=45.0, azim=130.0)
        ax_b.view_init(elev=10.0, azim=20.0)
        ax_a.button_pressed = 1  # noqa: SLF001 — matches matplotlib's own attribute
        ax_b.button_pressed = None

        from matplotlib.backend_bases import MouseEvent

        bbox = ax_a.bbox
        event = MouseEvent(
            "motion_notify_event",
            tab.comparison_canvas,
            (bbox.x0 + bbox.x1) / 2.0,
            (bbox.y0 + bbox.y1) / 2.0,
            button=None,  # real motion events report button=None during drag
        )
        event.inaxes = ax_a
        tab.comparison_canvas.callbacks.process("motion_notify_event", event)
        assert ax_b.elev == pytest.approx(ax_a.elev)
        assert ax_b.azim == pytest.approx(ax_a.azim)

        # Now drag the OTHER panel — ax_b → ax_a should mirror.
        ax_a.button_pressed = None
        ax_b.button_pressed = 1
        ax_b.view_init(elev=-20.0, azim=200.0)
        event2 = MouseEvent(
            "motion_notify_event",
            tab.comparison_canvas,
            (bbox.x0 + bbox.x1) / 2.0,
            (bbox.y0 + bbox.y1) / 2.0,
            button=None,
        )
        event2.inaxes = ax_b
        tab.comparison_canvas.callbacks.process("motion_notify_event", event2)
        assert ax_a.elev == pytest.approx(ax_b.elev)
        assert ax_a.azim == pytest.approx(ax_b.azim)
    finally:
        tab.close()
        controller.shutdown()


def test_comparison_canvas_links_3d_axis_zoom(tmp_path: Path) -> None:
    """Scroll-wheel zoom on one panel mirrors xlim/ylim/zlim to the other."""
    pytest.importorskip("torch")
    _app()
    run_dir = _write_workspace_dataset(tmp_path, run_name="zoom_ds", n=32)
    artifact_a = _write_xyz_artifact(tmp_path, name="zoom_a", hidden_layers=[8, 8])
    artifact_b = _write_xyz_artifact(tmp_path, name="zoom_b", hidden_layers=[8, 8])
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    controller.select_dataset(str(run_dir))
    controller.select_artifact(str(artifact_a))
    controller.set_comparison_external_model_path(str(artifact_b / "model.pt"))
    controller.refresh()
    controller.run_external_comparison()
    thread = getattr(controller, "_worker_thread", None)
    if thread is not None:
        thread.join(timeout=30.0)
    tab = ModelingTab(controller)
    try:
        tab.show()
        tab.update(controller.refresh())
        three_d = [ax for ax in tab.comparison_canvas.figure.axes if ax.name == "3d"]
        ax_a, ax_b = three_d
        # Simulate scroll-zoom changing ax_a's limits, then fire scroll_event.
        ax_a.set_xlim(-5.0, 5.0)
        ax_a.set_ylim(-5.0, 5.0)
        ax_a.set_zlim(-5.0, 5.0)
        ax_b.set_xlim(-100.0, 100.0)
        ax_b.set_ylim(-100.0, 100.0)
        ax_b.set_zlim(-100.0, 100.0)

        from matplotlib.backend_bases import MouseEvent

        bbox = ax_a.bbox
        event = MouseEvent(
            "scroll_event",
            tab.comparison_canvas,
            (bbox.x0 + bbox.x1) / 2.0,
            (bbox.y0 + bbox.y1) / 2.0,
            button="up",
        )
        event.inaxes = ax_a
        tab.comparison_canvas.callbacks.process("scroll_event", event)
        assert ax_b.get_xlim() == pytest.approx(ax_a.get_xlim())
        assert ax_b.get_ylim() == pytest.approx(ax_a.get_ylim())
        assert ax_b.get_zlim() == pytest.approx(ax_a.get_zlim())
    finally:
        tab.close()
        controller.shutdown()


def test_comparison_canvas_does_not_sync_on_idle_hover(tmp_path: Path) -> None:
    """Defensive: when no panel is being dragged (button_pressed is None on both),
    mouse motion across either panel must NOT clobber the other's view. This
    guards against the original bug where ``event.button`` was the key — that
    check fired on idle hover too once we removed the strict button==1 filter."""
    pytest.importorskip("torch")
    _app()
    run_dir = _write_workspace_dataset(tmp_path, run_name="hover_ds", n=32)
    artifact_a = _write_xyz_artifact(tmp_path, name="hov_a", hidden_layers=[8, 8])
    artifact_b = _write_xyz_artifact(tmp_path, name="hov_b", hidden_layers=[8, 8])
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    controller.select_dataset(str(run_dir))
    controller.select_artifact(str(artifact_a))
    controller.set_comparison_external_model_path(str(artifact_b / "model.pt"))
    controller.refresh()
    controller.run_external_comparison()
    thread = getattr(controller, "_worker_thread", None)
    if thread is not None:
        thread.join(timeout=30.0)
    tab = ModelingTab(controller)
    try:
        tab.show()
        tab.update(controller.refresh())
        three_d = [ax for ax in tab.comparison_canvas.figure.axes if ax.name == "3d"]
        ax_a, ax_b = three_d
        # Deliberate mismatch. Neither axis is being dragged.
        ax_a.view_init(elev=15.0, azim=45.0)
        ax_b.view_init(elev=85.0, azim=-100.0)
        ax_a.button_pressed = None
        ax_b.button_pressed = None
        before_b = (ax_b.elev, ax_b.azim)

        from matplotlib.backend_bases import MouseEvent

        bbox = ax_a.bbox
        event = MouseEvent(
            "motion_notify_event",
            tab.comparison_canvas,
            (bbox.x0 + bbox.x1) / 2.0,
            (bbox.y0 + bbox.y1) / 2.0,
            button=None,
        )
        event.inaxes = ax_a
        tab.comparison_canvas.callbacks.process("motion_notify_event", event)
        # ax_b must NOT have changed.
        assert (ax_b.elev, ax_b.azim) == before_b
    finally:
        tab.close()
        controller.shutdown()


# ---------------------------------------------------------------------------
# Multi-slot comparison: pick two artifacts from the catalog without uploading
# ---------------------------------------------------------------------------


def test_comparison_runs_with_two_dropdown_artifacts(tmp_path: Path) -> None:
    """Operator-driven flow: pick TWO discovered artifacts (no upload) and run
    the side-by-side comparison. The previous single-upload-required UX
    blocked this; the multi-slot refactor must allow it.
    """
    pytest.importorskip("torch")
    _app()
    run_dir = _write_workspace_dataset(tmp_path, run_name="twoartifacts_ds", n=32)
    artifact_a = _write_xyz_artifact(tmp_path, name="model_alpha", hidden_layers=[8, 8])
    artifact_b = _write_xyz_artifact(tmp_path, name="model_beta", hidden_layers=[16, 16])
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    controller.select_dataset(str(run_dir))
    # No upload at all. Both slots filled from the discovered catalog.
    controller.set_comparison_model_a_path(str(artifact_a))
    controller.set_comparison_model_b_path(str(artifact_b))
    controller.refresh()
    controller.run_external_comparison()
    thread = getattr(controller, "_worker_thread", None)
    if thread is not None:
        thread.join(timeout=30.0)
    result = controller.get_last_comparison_result()
    assert result is not None, "comparison didn't produce a result"
    # Both panels rendered with the right names.
    assert "model_alpha" in result.model_a.label
    assert "model_beta" in result.model_b.label
    controller.shutdown()


def test_comparison_slot_a_accepts_upload(tmp_path: Path) -> None:
    """The Model A slot now accepts uploads — not just Model B. Confirms
    set_comparison_model_a_path archives the .pt under data/models/ann/uploaded_*
    and the controller's stored path points at the archived copy."""
    pytest.importorskip("torch")
    _app()
    artifact_a = _write_xyz_artifact(tmp_path, name="src_xyz", hidden_layers=[8, 8])
    bare_for_a = tmp_path / "uploads" / "for_slot_a.pt"
    bare_for_a.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(artifact_a / "model.pt", bare_for_a)
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    controller.set_comparison_model_a_path(str(bare_for_a))
    state = controller.refresh()
    # Archived under artifact_root, not at the original /uploads/ location.
    assert state.comparison_model_a_path.startswith(
        str(tmp_path / "data" / "models" / "ann")
    ), state.comparison_model_a_path
    assert "uploaded_" in state.comparison_model_a_path
    assert "for_slot_a" in state.comparison_model_a_path
    # The archived file actually exists on disk.
    assert Path(state.comparison_model_a_path).exists()
    controller.shutdown()


def test_comparison_combo_picks_archived_upload_for_either_slot(tmp_path: Path) -> None:
    """A previously-archived upload appears in BOTH slot combos and is
    selectable for either one. This is the long-running benefit: pick an old
    upload from a prior session as Model A, a brand-new training run as Model
    B, no re-browsing needed."""
    pytest.importorskip("torch")
    _app()
    artifact_a = _write_xyz_artifact(tmp_path, name="real_train", hidden_layers=[8, 8])
    bare = tmp_path / "uploads" / "vintage.pt"
    bare.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(artifact_a / "model.pt", bare)

    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",
        results_root=tmp_path / "data" / "modeling_results",
    )
    # Archive the upload (any slot will do — same archiver).
    controller.set_comparison_model_b_path(str(bare))
    state = controller.refresh()
    archived_path = state.comparison_model_b_path
    archived_dir = str(Path(archived_path).parent)
    # The archived artifact should appear in the catalog's discovered list.
    discovered_paths = [str(a.path) for a in state.artifacts]
    assert archived_dir in discovered_paths, (
        f"archived upload {archived_dir} should be in artifacts list; "
        f"discovered: {discovered_paths}"
    )
    # Now drive the tab and confirm both combos list it.
    from PySide6.QtCore import Qt

    tab = ModelingTab(controller)
    try:
        tab.show()
        tab.update(state)
        a_paths = [
            tab.comparison_model_a_combo.itemData(i, Qt.UserRole)
            for i in range(tab.comparison_model_a_combo.count())
        ]
        b_paths = [
            tab.comparison_model_b_combo.itemData(i, Qt.UserRole)
            for i in range(tab.comparison_model_b_combo.count())
        ]
        assert archived_dir in a_paths, f"Slot A combo missing archived upload; items={a_paths}"
        assert archived_dir in b_paths or archived_path in b_paths, (
            f"Slot B combo missing archived upload; items={b_paths}"
        )
    finally:
        tab.close()
        controller.shutdown()


def test_comparison_generate_gated_on_dataset_plus_both_slots(tmp_path: Path) -> None:
    """The Generate button enables when dataset + both slots are populated.

    Notable subtlety: refresh() auto-selects the first discovered ANN artifact
    as the "main" dropdown selection (used by the regular Mike/Camarillo/ANN
    flow). The comparison gate intentionally treats that as a back-compat
    fallback for Slot A, so once any catalog exists, Slot A is considered set.
    This test verifies the gate's three flips by using an EMPTY artifact_root
    (no catalog ⇒ no back-compat fallback) so we exercise the explicit-slot
    path cleanly.
    """
    pytest.importorskip("torch")
    _app()
    run_dir = _write_workspace_dataset(tmp_path, run_name="gate_ds", n=24)
    # Put artifacts in a NON-discovered location so the catalog stays empty
    # and the back-compat fallback can't fire.
    sandbox_root = tmp_path / "sandbox_artifacts"
    artifact_a = _write_xyz_artifact_at(sandbox_root, name="art_a", hidden_layers=[8, 8])
    artifact_b = _write_xyz_artifact_at(sandbox_root, name="art_b", hidden_layers=[8, 8])
    controller = ModelingController(
        project_root=tmp_path,
        dataset_output_root=tmp_path / "data" / "experiments",
        artifact_root=tmp_path / "data" / "models" / "ann",  # empty
        results_root=tmp_path / "data" / "modeling_results",
    )
    controller.select_dataset(str(run_dir))
    tab = ModelingTab(controller)
    try:
        tab.show()
        # Dataset selected but no slots set → disabled.
        tab.update(controller.refresh())
        assert tab.comparison_generate_button.isEnabled() is False, (
            "Generate should be disabled with no slots set; artifact_details was "
            f"{controller.state.artifact_details!r}"
        )
        # Only slot B → still disabled.
        controller.set_comparison_model_b_path(str(artifact_b))
        tab.update(controller.refresh())
        assert tab.comparison_generate_button.isEnabled() is False
        # Add slot A → enabled.
        controller.set_comparison_model_a_path(str(artifact_a))
        tab.update(controller.refresh())
        assert tab.comparison_generate_button.isEnabled() is True
    finally:
        tab.close()
        controller.shutdown()
