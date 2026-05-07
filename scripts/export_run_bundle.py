#!/usr/bin/env python3
"""Thin wrapper for `python -m continuum_robot.data.export_run_bundle`."""

from __future__ import annotations

import sys
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from continuum_robot.data.export_run_bundle import main


if __name__ == "__main__":
    sys.exit(main())
