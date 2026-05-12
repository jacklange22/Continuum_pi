import os

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
