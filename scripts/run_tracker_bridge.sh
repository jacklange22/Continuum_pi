#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${TRACKER_BRIDGE_BIN:-$ROOT_DIR/bin/tracker_bridge}"
AURORA_PORT="${AURORA_PORT:-/dev/ttyUSB0}"
SOCKET_PATH="${TRACKER_SOCKET_PATH:-/tmp/tracker_bridge.sock}"
POLL_MS="${TRACKER_POLL_MS:-20}"

if [[ ! -x "$BIN" ]]; then
  echo "tracker_bridge binary not found or not executable: $BIN" >&2
  echo "Build it first with scripts/build_tracker_bridge.sh" >&2
  exit 1
fi

exec "$BIN" --aurora-port "$AURORA_PORT" --socket-path "$SOCKET_PATH" --poll-ms "$POLL_MS"
