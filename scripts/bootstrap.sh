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
  "$ROOT_DIR/data/runs"

if [[ "$BUILD_TRACKER_BRIDGE" == "1" ]]; then
  : "${NDI_SDK_INCLUDE_DIR:?Set NDI_SDK_INCLUDE_DIR to the NDI SDK include directory}"
  : "${NDI_SDK_LIB_DIR:?Set NDI_SDK_LIB_DIR to the NDI SDK library directory}"
  "$ROOT_DIR/scripts/build_tracker_bridge.sh"
else
  echo "Skipping tracker_bridge build."
  echo "Set BUILD_TRACKER_BRIDGE=1 after installing the NDI SDK to build bin/tracker_bridge."
fi

echo
echo "Bootstrap complete."
echo "If needed, copy config/system.local.example.yaml to config/system.local.yaml and edit local serial ports."
echo "Start the GUI with: $ROOT_DIR/scripts/run_gui.sh"
