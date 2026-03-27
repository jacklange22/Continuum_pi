#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GUI_PYTHON_BIN="${GUI_PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

if [[ ! -x "$GUI_PYTHON_BIN" ]]; then
  echo "Python environment not found at $GUI_PYTHON_BIN" >&2
  echo "Run scripts/bootstrap.sh first." >&2
  exit 1
fi

exec "$GUI_PYTHON_BIN" "$ROOT_DIR/scripts/run_gui.py"
