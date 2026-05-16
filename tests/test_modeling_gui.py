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
