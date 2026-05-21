#!/usr/bin/env bash
# claude-bg-stop — stop a background Claude run by tag, or list active ones.
#
# Usage:
#   scripts/claude-bg-stop.sh                 # list active background runs
#   scripts/claude-bg-stop.sh <tag>           # stop one
#   scripts/claude-bg-stop.sh --all           # stop all
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/.claude/bg-logs"

list() {
  shopt -s nullglob
  local found=0
  for pidfile in "$LOG_DIR"/*.pid; do
    found=1
    local tag pid
    tag="$(basename "$pidfile" .pid)"
    pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then
      printf "%-40s pid=%-6s RUNNING  log=%s\n" "$tag" "$pid" "$LOG_DIR/${tag}.log"
    else
      printf "%-40s pid=%-6s done     log=%s\n" "$tag" "$pid" "$LOG_DIR/${tag}.log"
    fi
  done
  [[ "$found" -eq 0 ]] && echo "no background runs found in $LOG_DIR"
}

stop_tag() {
  local tag="$1"
  local pidfile="$LOG_DIR/${tag}.pid"
  if [[ ! -f "$pidfile" ]]; then
    echo "no pid file for tag: $tag" >&2; return 1
  fi
  local pid; pid="$(cat "$pidfile")"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" && echo "killed $tag pid=$pid"
  else
    echo "$tag already finished (pid=$pid)"
  fi
}

case "${1:-}" in
  ""|-l|--list) list ;;
  --all)
    shopt -s nullglob
    for pidfile in "$LOG_DIR"/*.pid; do stop_tag "$(basename "$pidfile" .pid)" || true; done ;;
  *) stop_tag "$1" ;;
esac
