#!/usr/bin/env bash
# Replacement tail of the pipeline after the r512 stale-artifact fix.
# Waits for the running adaptive heal to finish, then heals the freshly
# re-converted r512_drope128 (V-only), then runs the batched frontier eval
# over ALL healed points.
set -uo pipefail
cd "$(dirname "$0")/../.."

echo "[r512-then-eval] $(date -Is) waiting for adaptive heal to finish..."
while pgrep -f "run_heal_adaptive_after.sh|mla.heal" >/dev/null 2>&1; do
  sleep 120
done
echo "[r512-then-eval] $(date -Is) adaptive heal done; healing r512_drope128"

ARTIFACT="experiments/stage_b/r512_drope128.pt" \
OUT="experiments/stage_b/heal_r512_drope128" \
MAX_HOURS=5 \
  bash experiments/stage_b/run_heal_overnight.sh \
  || echo "[r512-then-eval] r512 heal exited non-zero; running eval anyway" >&2

echo "[r512-then-eval] $(date -Is) running batched frontier eval"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
uv run python experiments/stage_b/eval_frontier.py
echo "[r512-then-eval] $(date -Is) complete"
