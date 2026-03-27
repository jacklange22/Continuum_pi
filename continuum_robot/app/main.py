"""GUI entrypoint wiring."""

from __future__ import annotations

from PySide6.QtWidgets import QApplication

from continuum_robot.app.bootstrap import build_app_context
from continuum_robot.gui.app_window import AppWindow


def main() -> int:
    """Start the application in GUI mode."""
    app = QApplication.instance() or QApplication([])
    context = build_app_context()
    window = AppWindow(context)
    window.show()
    exit_code = app.exec()
    window.shutdown()
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
