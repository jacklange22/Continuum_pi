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

    assert tab.table.columnCount() == 6
    assert tab.delete_button.isEnabled() is False
    assert tab.preview_migration_button.isEnabled() is False
