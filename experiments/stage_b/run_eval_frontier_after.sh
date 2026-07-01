#!/usr/bin/env bash
# Run the batched frontier eval once ALL heals (frontier queue + chained
# adaptive) have finished and the GPU is free. Idle-waits, then evaluates
# every healed point on the 250-chunk slice in sequence.
set -uo pipefail
cd "$(dirname "$0")/../.."

echo "[eval-after] $(date -Is) waiting for all heal processes to finish..."
while pgrep -f "run_heal_frontier.sh|run_heal_adaptive_after.sh|mla.heal" >/dev/null 2>&1; do
  sleep 120
done
echo "[eval-after] $(date -Is) all heals done; running frontier eval"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
uv run python experiments/stage_b/eval_frontier.py
echo "[eval-after] $(date -Is) frontier eval complete"
