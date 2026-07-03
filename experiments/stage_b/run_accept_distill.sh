#!/usr/bin/env bash
# Acceptance-objective A/B: heal the covariance-init 4x point to maximize
# speculative-decode acceptance (1−TV) instead of forward KL. Two runs, same
# init/budget/teacher as the KL-distill run — isolating only the loss:
#   1. direct  : loss = 1 − Σ min(p_t, p_s)         (hard 0/1 gradient)
#   2. softmin : soft-min smoothed (τ=0.05)          (graded sigmoid gradient)
# Baseline to beat: KL-distill covariance = 73.8% acceptance.
# Runs after any in-flight heal frees the GPU; evals all points at the end.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TEACHER="experiments/stage_b/teacher_topk_wt103_2k.pt"
ART="experiments/stage_b/r256_drope32.pt"

echo "[accept] $(date -Is) waiting for any running heal to finish..."
while pgrep -f "mla.heal" >/dev/null 2>&1; do sleep 120; done

common=(--mla-artifact "$ART" --teacher-topk "$TEACHER" --distill-objective accept
        --distill-temp 1.0 --max-hours 5 --steps 400 --seq-len 1024 --grad-accum 32
        --lora-rank 16 --lora-alpha 32 --lr-mla 5e-5 --lr-lora 1e-4 --warmup-frac 0.03
        --val-every 30 --val-chunks 50 --log-every 10)

echo "[accept] $(date -Is) run 1/2: direct acceptance (softmin_tau=0)"
uv run python -m mla.heal "${common[@]}" --softmin-tau 0.0 \
    --out experiments/stage_b/heal_r256_drope32_accept_direct \
    || echo "[accept] direct run exited non-zero; continuing" >&2

echo "[accept] $(date -Is) run 2/2: soft-min acceptance (softmin_tau=0.05)"
uv run python -m mla.heal "${common[@]}" --softmin-tau 0.05 \
    --out experiments/stage_b/heal_r256_drope32_accept_softmin \
    || echo "[accept] softmin run exited non-zero; continuing" >&2

echo "[accept] $(date -Is) evaluating (acceptance/KL + PPL)"
uv run python experiments/stage_b/eval_frontier_kl.py
uv run python experiments/stage_b/eval_frontier.py
echo "[accept] $(date -Is) complete"
