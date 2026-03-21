"""GUI entrypoint wiring.

This file keeps startup shallow and delegates setup details to bootstrap.
"""

from continuum_robot.app.bootstrap import build_app_context


def main() -> int:
    """Start the application in GUI mode.

    Returns a process-style status code.
    """
    _context = build_app_context()
    print("GUI scaffold bootstrapped. Integrate PySide6 window launch here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
