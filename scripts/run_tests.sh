#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
MODE="${1:-quick}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found at $PYTHON_BIN" >&2
  echo "Run scripts/bootstrap.sh first, or set PYTHON_BIN to the intended venv Python." >&2
  exit 2
fi

PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PY_OK="$("$PYTHON_BIN" -c 'import sys; print(int(sys.version_info >= (3, 10)))')"
if [[ "$PY_OK" != "1" ]]; then
  echo "Python 3.10+ is required for the canonical test runner; $PYTHON_BIN is Python $PY_VERSION." >&2
  echo "Recommended: Python 3.11+ in .venv." >&2
  exit 2
fi

echo "Using test Python: $PYTHON_BIN ($PY_VERSION)"

case "$MODE" in
  --check|check)
    exit 0
    ;;
  quick)
    exec "$PYTHON_BIN" -m pytest -q \
      tests/test_servo_service.py \
      tests/test_active_segment_config.py \
      tests/test_config_loader.py \
      tests/test_runtime_tip_policy.py
    ;;
  hardware-safe)
    exec "$PYTHON_BIN" -m pytest -q \
      tests/test_servo_service.py \
      tests/test_pretension_validation_experiment.py \
      tests/test_active_segment_config.py \
      tests/test_experiment_framework.py \
      tests/test_calibration_validation.py \
      tests/test_runtime_tip_policy.py \
      tests/test_penprobe_chasing_demo.py
    ;;
  full-nongui)
    exec "$PYTHON_BIN" -m pytest -q tests -m "not gui"
    ;;
  gui)
    exec "$PYTHON_BIN" -m pytest -q tests/test_gui_controllers.py tests/test_gui_bootstrap.py
    ;;
  *)
    exec "$PYTHON_BIN" -m pytest "$@"
    ;;
esac
