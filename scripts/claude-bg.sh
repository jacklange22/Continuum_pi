#!/usr/bin/env bash
# claude-bg — fire-and-forget background runner for headless Claude Code.
#
# Usage:
#   scripts/claude-bg.sh "fix the failing test in tests/test_registration.py"
#   scripts/claude-bg.sh -m opus "refactor continuum_robot/tracking to use the new NDI path"
#   scripts/claude-bg.sh -t "nightly-test-fix" "run scripts/run_tests.sh and fix anything failing"
#
# Flags:
#   -m MODEL    pick model (e.g. sonnet, opus). default: claude default
#   -t TAG      log/tag name. default: timestamp
#   -d DIR      run in this dir (defaults to current). useful with worktrees.
#   -p PERM     permission mode. default: bypassPermissions
#   -q          quiet — don't print the tail-tip line at the end
#
# Logs go to .claude/bg-logs/<tag>.log so you can `tail -f` whenever you want.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL=""
TAG=""
RUN_DIR="$PWD"
PERM_MODE="bypassPermissions"
QUIET=0

while getopts ":m:t:d:p:q" opt; do
  case "$opt" in
    m) MODEL="$OPTARG" ;;
    t) TAG="$OPTARG" ;;
    d) RUN_DIR="$OPTARG" ;;
    p) PERM_MODE="$OPTARG" ;;
    q) QUIET=1 ;;
    \?) echo "Unknown flag: -$OPTARG" >&2; exit 2 ;;
  esac
done
shift $((OPTIND-1))

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 [-m MODEL] [-t TAG] [-d DIR] [-p PERM_MODE] [-q] \"<prompt>\"" >&2
  exit 2
fi
PROMPT="$*"

if [[ -z "$TAG" ]]; then
  TAG="bg-$(date +%Y%m%d-%H%M%S)-$$"
fi

LOG_DIR="$ROOT/.claude/bg-logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${TAG}.log"

if ! command -v claude >/dev/null 2>&1; then
  echo "claude CLI not found on PATH. Install: npm install -g @anthropic-ai/claude-code" >&2
  exit 127
fi

# Build argv for claude headless mode.
ARGS=( -p --permission-mode "$PERM_MODE" --output-format stream-json --verbose )
[[ -n "$MODEL" ]] && ARGS+=( --model "$MODEL" )

{
  echo "===== claude-bg ====="
  echo "tag:    $TAG"
  echo "dir:    $RUN_DIR"
  echo "model:  ${MODEL:-default}"
  echo "perm:   $PERM_MODE"
  echo "prompt: $PROMPT"
  echo "started: $(date -Iseconds)"
  echo "====================="
} >"$LOG"

# Launch detached. nohup + & + disown survives terminal close.
( cd "$RUN_DIR" && nohup claude "${ARGS[@]}" "$PROMPT" >>"$LOG" 2>&1 ) &
PID=$!
disown "$PID" 2>/dev/null || true

echo "$PID" > "$LOG_DIR/${TAG}.pid"

if [[ "$QUIET" -eq 0 ]]; then
  echo "started claude-bg [$TAG] pid=$PID"
  echo "  tail:  tail -f $LOG"
  echo "  stop:  kill $PID  # or: scripts/claude-bg-stop.sh $TAG"
fi
