#!/usr/bin/env bash
# claude-parallel — dispatch one Claude headless run per git worktree, in parallel.
#
# You already use worktrees in .claude/worktrees/. This script fans out the same
# prompt across multiple of them so several attempts run concurrently and you can
# pick the best diff at the end.
#
# Usage:
#   scripts/claude-parallel.sh -n 3 "implement registration export in continuum_robot/registration/exporters.py"
#       → spawn 3 fresh worktrees and run the prompt in each
#
#   scripts/claude-parallel.sh -w funny-tharp-199d99 -w great-ride-96aa7c "..."
#       → run in two named existing worktrees
#
#   scripts/claude-parallel.sh -f prompts.txt
#       → one line per prompt, each gets its own worktree
#
# Each run is launched via scripts/claude-bg.sh so logs land in .claude/bg-logs/.
# After they all start: `scripts/claude-bg-stop.sh --list` to see status.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKTREE_ROOT="$ROOT/.claude/worktrees"
mkdir -p "$WORKTREE_ROOT"

N_FRESH=0
WORKTREES=()
PROMPT_FILE=""
MODEL=""

while getopts ":n:w:f:m:" opt; do
  case "$opt" in
    n) N_FRESH="$OPTARG" ;;
    w) WORKTREES+=("$OPTARG") ;;
    f) PROMPT_FILE="$OPTARG" ;;
    m) MODEL="$OPTARG" ;;
    \?) echo "Unknown flag: -$OPTARG" >&2; exit 2 ;;
  esac
done
shift $((OPTIND-1))

PROMPTS=()
if [[ -n "$PROMPT_FILE" ]]; then
  while IFS= read -r line; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    PROMPTS+=("$line")
  done < "$PROMPT_FILE"
elif [[ $# -gt 0 ]]; then
  PROMPTS+=("$*")
else
  echo "Need a prompt: positional arg, or -f prompts.txt" >&2
  exit 2
fi

# Make fresh worktrees if requested.
gen_name() {
  # short random suffix, e.g. par-7f3a91
  echo "par-$(LC_ALL=C tr -dc 'a-f0-9' < /dev/urandom | head -c 6)"
}

if [[ "$N_FRESH" -gt 0 ]]; then
  cd "$ROOT"
  CURRENT_BRANCH="$(git symbolic-ref --quiet --short HEAD 2>/dev/null || echo main)"
  for i in $(seq 1 "$N_FRESH"); do
    NAME="$(gen_name)"
    BR="claude/${NAME}"
    git worktree add -b "$BR" "$WORKTREE_ROOT/$NAME" "$CURRENT_BRANCH" >/dev/null
    WORKTREES+=("$NAME")
    echo "+ worktree $NAME (branch $BR)"
  done
fi

if [[ "${#WORKTREES[@]}" -eq 0 ]]; then
  echo "No worktrees specified. Use -n N or -w name." >&2
  exit 2
fi

# Pair prompts and worktrees. If one prompt + multiple worktrees, replicate it.
if [[ "${#PROMPTS[@]}" -eq 1 && "${#WORKTREES[@]}" -gt 1 ]]; then
  P="${PROMPTS[0]}"
  while [[ "${#PROMPTS[@]}" -lt "${#WORKTREES[@]}" ]]; do PROMPTS+=("$P"); done
fi

if [[ "${#PROMPTS[@]}" -ne "${#WORKTREES[@]}" ]]; then
  echo "Got ${#PROMPTS[@]} prompts but ${#WORKTREES[@]} worktrees. Pair them or use a single prompt." >&2
  exit 2
fi

LAUNCHER="$ROOT/scripts/claude-bg.sh"
[[ -x "$LAUNCHER" ]] || { echo "missing $LAUNCHER" >&2; exit 1; }

for i in "${!WORKTREES[@]}"; do
  WT="${WORKTREES[$i]}"
  WT_PATH="$WORKTREE_ROOT/$WT"
  [[ -d "$WT_PATH" ]] || { echo "skip $WT — no dir at $WT_PATH" >&2; continue; }
  PROMPT="${PROMPTS[$i]}"
  TAG="parallel-${WT}-$(date +%H%M%S)"
  ARGS=( -t "$TAG" -d "$WT_PATH" -q )
  [[ -n "$MODEL" ]] && ARGS+=( -m "$MODEL" )
  "$LAUNCHER" "${ARGS[@]}" "$PROMPT"
  echo "dispatched: $WT  tag=$TAG"
done

echo
echo "All started. Check status:   scripts/claude-bg-stop.sh"
echo "Diff a worktree later with:  git -C .claude/worktrees/<name> diff main"
