#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
INSTALL_DEV_DEPS="${INSTALL_DEV_DEPS:-1}"
BUILD_TRACKER_BRIDGE="${BUILD_TRACKER_BRIDGE:-0}"

echo "Project root: $ROOT_DIR"
echo "Using Python: $PYTHON_BIN"
echo "Virtualenv: $VENV_DIR"

REQUESTED_PYTHON_MM="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "Python 3.10 or newer is required. $PYTHON_BIN resolved to Python $REQUESTED_PYTHON_MM." >&2
  echo "On Raspberry Pi OS, install python3 and python3-venv, then rerun with PYTHON_BIN=python3." >&2
  exit 1
fi
if [[ -x "$VENV_DIR/bin/python" ]]; then
  EXISTING_PYTHON_MM="$("$VENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
  if [[ "$EXISTING_PYTHON_MM" != "$REQUESTED_PYTHON_MM" ]]; then
    echo "Existing virtualenv uses Python $EXISTING_PYTHON_MM but $PYTHON_BIN is Python $REQUESTED_PYTHON_MM."
    echo "Recreating virtualenv at $VENV_DIR to match the requested interpreter."
    rm -rf "$VENV_DIR"
  fi
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip

cd "$ROOT_DIR"
if [[ "$INSTALL_DEV_DEPS" == "1" ]]; then
  "$VENV_DIR/bin/python" -m pip install -e ".[dev]"
else
  "$VENV_DIR/bin/python" -m pip install -e "."
fi

mkdir -p \
  "$ROOT_DIR/bin" \
  "$ROOT_DIR/data/calibrations" \
  "$ROOT_DIR/data/logs" \
  "$ROOT_DIR/data/registrations" \
  "$ROOT_DIR/data/tracker_captures" \
  "$ROOT_DIR/data/runs"

if [[ "$BUILD_TRACKER_BRIDGE" == "1" ]]; then
  : "${NDI_SDK_INCLUDE_DIR:?Set NDI_SDK_INCLUDE_DIR to the NDI SDK include directory}"
  : "${NDI_SDK_LIB_DIR:?Set NDI_SDK_LIB_DIR to the NDI SDK library directory}"
  "$ROOT_DIR/scripts/build_tracker_bridge.sh"
else
  echo "Skipping tracker_bridge build."
  echo "The default live tracker path is Python-native via scikit-surgerynditracker."
  echo "Set BUILD_TRACKER_BRIDGE=1 only if you need the legacy bridge compatibility backend."
fi

echo
echo "Bootstrap complete."
echo "If needed, copy config/system.local.example.yaml to config/system.local.yaml and edit local serial ports."
echo "Start the GUI with: $ROOT_DIR/scripts/run_gui.sh"
