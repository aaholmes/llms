#!/usr/bin/env bash
# Heal the adaptive 4x point AFTER the uniform frontier queue finishes.
# They can't share the 16 GB GPU, so this idle-waits for run_heal_frontier.sh
# to exit, then heals r256_drope32_adaptive with the identical recipe — the
# direct "adaptive vs uniform at the same cache size" comparison.
set -uo pipefail
cd "$(dirname "$0")/../.."

echo "[adaptive-after] $(date -Is) waiting for the frontier queue to finish..."
while pgrep -f run_heal_frontier.sh >/dev/null 2>&1; do
  sleep 120
done
echo "[adaptive-after] $(date -Is) frontier queue done; starting adaptive heal"

ARTIFACT="experiments/stage_b/r256_drope32_adaptive.pt" \
OUT="experiments/stage_b/heal_r256_drope32_adaptive" \
MAX_HOURS=5 \
  bash experiments/stage_b/run_heal_overnight.sh

echo "[adaptive-after] $(date -Is) adaptive heal complete"
