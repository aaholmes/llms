#!/usr/bin/env bash
# TVD-tune a speculative-decode DRAFT (Qwen3-0.6B) to the TARGET (Qwen3-4B) and
# measure the acceptance gain (Stage C R4). Add LoRA to the draft, minimize
# TVD(target, draft) on the workload = maximize acceptance directly, reusing the
# SAME 4B teacher top-K cache (the teacher for draft-tuning IS the target).
# Runs after the current chain frees the GPU.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TEACHER="experiments/stage_b/teacher_topk_wt103_2k.pt"   # Qwen3-4B top-K = the target

echo "[draft-tvd] $(date -Is) waiting for the GPU to be free (heals + evals + chain)..."
# Poll for ANY GPU-using job — the current chain ends with eval phases where no
# mla.heal is running, so wait for the evals and the chain orchestrator too.
while pgrep -f "mla.heal|eval_frontier|run_accept_distill" >/dev/null 2>&1; do sleep 120; done

echo "[draft-tvd] $(date -Is) TVD-tuning Qwen3-0.6B draft to the 4B target"
uv run python -m mla.heal \
    --model-id Qwen/Qwen3-0.6B \
    --out experiments/stage_b/heal_draft_0_6b_tvd \
    --teacher-topk "$TEACHER" \
    --distill-objective accept --softmin-tau 0.0 \
    --max-hours 4 --steps 400 --seq-len 1024 --grad-accum 32 \
    --lora-rank 16 --lora-alpha 32 --lr-lora 1e-4 --lr-mla 1e-4 --warmup-frac 0.03 \
    --val-every 30 --val-chunks 50 --log-every 10 \
    || echo "[draft-tvd] heal exited non-zero; running eval anyway" >&2

echo "[draft-tvd] $(date -Is) evaluating base vs TVD-tuned draft acceptance"
uv run python experiments/stage_b/eval_draft_accept.py
echo "[draft-tvd] $(date -Is) complete"
