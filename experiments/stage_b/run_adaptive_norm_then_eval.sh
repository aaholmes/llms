#!/usr/bin/env bash
# Heal the normalized-score adaptive variant (CARE-style σ²/tail-energy
# allocation), then re-run the batched frontier eval so all points sit on the
# same 250-chunk slice. Tests whether CARE's normalized allocation lifts the
# adaptive result from +98% (raw) toward/below uniform's +16%.
set -uo pipefail
cd "$(dirname "$0")/../.."

echo "[norm] $(date -Is) healing r256_drope32_adaptive_norm"
ARTIFACT="experiments/stage_b/r256_drope32_adaptive_norm.pt" \
OUT="experiments/stage_b/heal_r256_drope32_adaptive_norm" \
MAX_HOURS=5 \
  bash experiments/stage_b/run_heal_overnight.sh \
  || echo "[norm] heal exited non-zero; running eval anyway" >&2

echo "[norm] $(date -Is) running batched frontier eval"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
uv run python experiments/stage_b/eval_frontier.py
echo "[norm] $(date -Is) complete"
