"""GUI entrypoint wiring.

This file keeps startup shallow and delegates setup details to bootstrap.
"""

from continuum_robot.app.bootstrap import build_app_context
from continuum_robot.gui.app_window import AppWindow


def main() -> int:
    """Start the application in GUI mode.

    Returns a process-style status code.
    """
    context = build_app_context()
    window = AppWindow(context)
    window.show()
    # Real PySide event loop integration can call refresh() on a timer and
    # invoke window.shutdown() on application exit.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
