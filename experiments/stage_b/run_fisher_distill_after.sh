#!/usr/bin/env bash
# Queue the Fisher-init distill run to fire after the covariance-init distill
# heal finishes. Both distill runs are identical except the SVD init, so this
# is a clean A/B on "does two-sided Fisher-weighted init help?".
#
# Chain: wait for the running distill heal → collect output-Fisher (GPU) →
# convert r256_drope32 with --fisher (CPU) → distill-heal the Fisher artifact
# (GPU), reusing the existing teacher top-K cache.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TEACHER="experiments/stage_b/teacher_topk_wt103_2k.pt"
FISHER="experiments/stage_b/fisher_wt103.pt"
FISHER_ART="experiments/stage_b/r256_drope32_fisher.pt"

echo "[fisher-chain] $(date -Is) waiting for the running distill heal to finish..."
while pgrep -f "mla.heal" >/dev/null 2>&1; do
  sleep 120
done
echo "[fisher-chain] $(date -Is) GPU free; collecting output-Fisher"

# 1. Output-gradient Fisher for the original model (GPU, ~10-15 min).
uv run python experiments/stage_b/collect_fisher.py \
    --n-chunks 512 --chunk-tokens 256 --seed 42 --out "$FISHER" \
    || { echo "[fisher-chain] fisher collection failed" >&2; exit 1; }

# 2. Fisher-weighted conversion at the 4x point (CPU, ~1 min). Same rank/d_rope/
#    calib as r256_drope32.pt — only the init differs.
uv run python -m mla.convert Qwen/Qwen3-4B \
    --rank 256 --d-rope 32 --calib wt103-1k --fisher "$FISHER" \
    --out "$FISHER_ART" \
    || { echo "[fisher-chain] fisher conversion failed" >&2; exit 1; }

# 3. Distill-to-original heal on the Fisher-init artifact (GPU, ~3.3h),
#    reusing the same teacher top-K as the covariance-init run.
echo "[fisher-chain] $(date -Is) distilling the Fisher-init 4x point"
uv run python -m mla.heal \
    --mla-artifact "$FISHER_ART" \
    --out experiments/stage_b/heal_r256_drope32_fisher_distill \
    --teacher-topk "$TEACHER" \
    --distill-temp 1.0 --distill-ce-weight 0.0 \
    --max-hours 5 --steps 400 --seq-len 1024 --grad-accum 32 \
    --lora-rank 16 --lora-alpha 32 --lr-mla 5e-5 --lr-lora 1e-4 --warmup-frac 0.03 \
    --val-every 30 --val-chunks 50 --log-every 10

echo "[fisher-chain] $(date -Is) complete"
