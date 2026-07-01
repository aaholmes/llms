#!/usr/bin/env bash
# Heal the adaptive min_rank=128 variant (early layers protected), then re-run
# the batched frontier eval so all points sit on the same 250-chunk slice.
# Tests whether protecting early layers recovers adaptive rank toward/below
# uniform's +16% (the propagation hypothesis).
set -uo pipefail
cd "$(dirname "$0")/../.."

echo "[mr128] $(date -Is) healing r256_drope32_adaptive_mr128"
ARTIFACT="experiments/stage_b/r256_drope32_adaptive_mr128.pt" \
OUT="experiments/stage_b/heal_r256_drope32_adaptive_mr128" \
MAX_HOURS=5 \
  bash experiments/stage_b/run_heal_overnight.sh \
  || echo "[mr128] heal exited non-zero; running eval anyway" >&2

echo "[mr128] $(date -Is) running batched frontier eval"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
uv run python experiments/stage_b/eval_frontier.py
echo "[mr128] $(date -Is) complete"
