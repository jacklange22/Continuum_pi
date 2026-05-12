import os
import json
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QScrollArea

from continuum_robot.gui.controllers.data_management_controller import DataManagementController
from continuum_robot.gui.tabs.data_management_tab import DataManagementTab


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_data_management_tab_wraps_workspace_in_scroll_area(tmp_path) -> None:
    _app()
    controller = DataManagementController(project_root=tmp_path)
    tab = DataManagementTab(controller)

    assert isinstance(tab.scroll_area, QScrollArea)
    assert tab.scroll_area.widget() is not None


def test_data_management_tab_updates_from_controller_state(tmp_path) -> None:
    _app()
    controller = DataManagementController(project_root=tmp_path)
    tab = DataManagementTab(controller)

    tab.update(controller.refresh())

    assert tab.table.columnCount() == 8
    assert tab.delete_button.text() == "Delete Selected File/Bundle"
    assert tab.trash_run_button.text() == "Move Selected Run to Trash"
    assert tab.delete_button.isEnabled() is False
    assert tab.preview_migration_button.isEnabled() is False
    assert tab.export_selected_button.isEnabled() is False
    assert tab.export_latest_button.isEnabled() is False
    assert tab.validate_selected_button.isEnabled() is False
    assert tab.run_two_segment_modeling_button.isEnabled() is False
    assert tab.open_modeling_summary_button.isEnabled() is False
    assert tab.export_modeling_bundle_button.isEnabled() is False
    assert tab.save_review_button.isEnabled() is False
    assert tab.archive_run_button.isEnabled() is False
    assert tab.trash_run_button.isEnabled() is False
    assert tab.build_evidence_index_button.isEnabled() is True
    assert tab.zip_export_checkbox.isChecked() is True


def test_data_management_tab_preset_filters_segment_b(tmp_path) -> None:
    _app()
    run_dir = tmp_path / "data" / "experiments" / "single_segment_repeatability" / "20260102_000000_single_segment_repeatability"
    run_dir.mkdir(parents=True)
    provenance = {
        "operating_mode": "single_segment",
        "active_segment": {"key": "segment_b", "label": "Spine 2", "servo_ids": [5, 6, 7, 8]},
        "mock_mode": False,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps({"experiment_name": "single_segment_repeatability", "provenance_info": provenance}),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "experiment_name": "single_segment_repeatability",
                "success": True,
                "status": "success",
                "experiment_metrics": {
                    "run_trust_mode": "servo_only",
                    "run_provenance": provenance,
                    "valid_for_thesis_repeatability": False,
                },
            }
        ),
        encoding="utf-8",
    )
    controller = DataManagementController(project_root=tmp_path)
    tab = DataManagementTab(controller)

    index = tab.preset_combo.findData("segment_b")
    tab.preset_combo.setCurrentIndex(index)

    assert tab.table.rowCount() == 1
    assert controller.state.operating_mode_filter == "single_segment"
    assert controller.state.segment_filter == "segment_b"


def test_data_management_table_refresh_does_not_read_samples_jsonl(tmp_path, monkeypatch) -> None:
    _app()
    run_dir = tmp_path / "data" / "experiments" / "collect_pose_command_dataset" / "20260102_000000_collect_pose_command_dataset"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps({"experiment_name": "collect_pose_command_dataset"}), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({"experiment_name": "collect_pose_command_dataset"}), encoding="utf-8")
    (run_dir / "samples.jsonl").write_text("{bad sample payload is intentionally never parsed", encoding="utf-8")
    original_read_text = Path.read_text

    def guarded_read_text(self, *args, **kwargs):
        if self.name == "samples.jsonl":
            raise AssertionError("Data table refresh should not read samples.jsonl")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    controller = DataManagementController(project_root=tmp_path)
    tab = DataManagementTab(controller)

    tab.update(controller.refresh())

    assert tab.table.rowCount() == 1
    assert controller.state.performance_pairs
