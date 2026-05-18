#!/usr/bin/env bash
# One-command Pi -> Mac dataset pull.
#
# Usage:
#   scripts/pull_pi_data.sh
#   scripts/pull_pi_data.sh workspace_repeatability_map
#   scripts/pull_pi_data.sh collect_pose_command_dataset 20260518_220000_collect_pose_command_dataset
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT="${1:-collect_pose_command_dataset}"
RUN_ID="${2:-latest}"
EXTRA_ARGS=()
if [[ "$#" -gt 2 ]]; then
  EXTRA_ARGS=("${@:3}")
fi

exec python3 "$ROOT/scripts/sync_pi_dataset.py" pull \
  --pi "continuum-pi@10.28.63.49" \
  --remote-project-root "/home/continuum-pi/Continuum_pi" \
  --local-mirror-root "$ROOT" \
  --experiment "$EXPERIMENT" \
  --run "$RUN_ID" \
  "${EXTRA_ARGS[@]}"
