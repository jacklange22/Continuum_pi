#!/usr/bin/env bash
# One-command Pi -> Mac dataset pull.
#
# Usage:
#   scripts/pull_pi_data.sh
#   scripts/pull_pi_data.sh all
#   scripts/pull_pi_data.sh workspace_repeatability_map
#   scripts/pull_pi_data.sh collect_pose_command_dataset 20260518_220000_collect_pose_command_dataset
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "all" || "${1:-}" == "--all" ]]; then
  shift
  exec python3 "$ROOT/scripts/sync_pi_dataset.py" pull-all \
    --pi "continuum-pi@10.28.63.49" \
    --remote-project-root "/home/continuum-pi/Continuum_pi" \
    --local-mirror-root "$ROOT" \
    "$@"
fi

EXPERIMENT="${1:-collect_pose_command_dataset}"
RUN_ID="${2:-latest}"

pull_dataset() {
  python3 "$ROOT/scripts/sync_pi_dataset.py" pull \
    --pi "continuum-pi@10.28.63.49" \
    --remote-project-root "/home/continuum-pi/Continuum_pi" \
    --local-mirror-root "$ROOT" \
    --experiment "$EXPERIMENT" \
    --run "$RUN_ID" \
    "$@"
}

if [[ "$#" -gt 2 ]]; then
  pull_dataset "${@:3}"
else
  pull_dataset
fi
