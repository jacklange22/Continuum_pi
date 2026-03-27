import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from continuum_robot.app.bootstrap import build_app_context
from continuum_robot.gui.app_window import AppWindow


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_app_window_bootstraps_with_real_tabs() -> None:
    _app()
    window = AppWindow(build_app_context())
    try:
        assert window.windowTitle() == "Continuum Robot Operator Console"
        assert window.tab_widget.count() == 5
        assert window.tab_widget.tabText(0) == "System"
        assert window.tab_widget.tabText(4) == "Experiment"
    finally:
        window.shutdown()
