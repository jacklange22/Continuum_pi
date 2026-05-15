import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytestmark = [pytest.mark.gui]

pytest.importorskip("PySide6")

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
    context = build_app_context()
    assert context.services.get("live_registration") is not None
    window = AppWindow(context)
    try:
        assert window.windowTitle() == "Continuum Robot Operator Console"
        assert window.tab_widget.count() == 8
        assert window.tab_widget.tabText(0) == "System"
        assert window.tab_widget.tabText(1) == "Tracking"
        assert window.tab_widget.tabText(2) == "Registration"
        assert window.tab_widget.tabText(3) == "Servos"
        assert window.tab_widget.tabText(4) == "Pretension"
        assert window.tab_widget.tabText(5) == "Experiment"
        assert window.tab_widget.tabText(6) == "Modeling"
        assert window.tab_widget.tabText(7) == "Data"
    finally:
        window.shutdown()


def test_app_window_shutdown_calls_system_disconnect() -> None:
    _app()
    context = build_app_context()
    window = AppWindow(context)
    calls: list[str] = []
    original_disconnect = window.system_controller.disconnect_openrb

    def _recording_disconnect() -> None:
        calls.append("disconnect_openrb")
        original_disconnect()

    window.system_controller.disconnect_openrb = _recording_disconnect
    try:
        window.shutdown()
    finally:
        if window.isVisible():
            window.close()
    assert "disconnect_openrb" in calls
