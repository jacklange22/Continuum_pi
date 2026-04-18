"""GUI entrypoint wiring."""

from __future__ import annotations

import logging

from PySide6.QtWidgets import QApplication

from continuum_robot.app.bootstrap import build_app_context
from continuum_robot.gui.app_window import AppWindow
from continuum_robot.gui.theme import apply_dark_theme
from continuum_robot.utils.logging_setup import configure_session_logging


LOG = logging.getLogger(__name__)


def main() -> int:
    """Start the application in GUI mode."""
    configure_session_logging()
    app = QApplication.instance() or QApplication([])
    apply_dark_theme(app)
    context = build_app_context()
    LOG.info("Operator GUI launch complete.")
    window = AppWindow(context)
    window.show()
    try:
        exit_code = app.exec()
    except Exception:
        LOG.exception("Unhandled exception while running the GUI event loop.")
        raise
    finally:
        window.shutdown()
    LOG.info("Operator GUI closed with exit code %s.", int(exit_code))
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
