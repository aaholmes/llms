#!/usr/bin/env bash
# Narrow-workload draft-optimizer (Thread 1): does TVD-tuning a Qwen3-0.6B draft
# on a NARROW workload (Python code) beat WikiText's 1.48× — via higher acceptance
# → higher optimal K? Precompute the 4B target's top-K on code, TVD-tune the draft,
# bench base-vs-tuned on held-out code. Sequential; GPU is free so run directly.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

TEACHER="experiments/stage_b/teacher_topk_code_2k.pt"

echo "[code] $(date -Is) precompute 4B teacher top-K on code"
uv run python experiments/stage_b/precompute_teacher_topk.py \
    --corpus code --n-chunks 2048 --seq-len 1024 --k 64 \
    --out "$TEACHER" || { echo "[code] precompute failed" >&2; exit 1; }

echo "[code] $(date -Is) TVD-tune 0.6B draft on code (accept objective)"
uv run python -m mla.heal \
    --model-id Qwen/Qwen3-0.6B \
    --out experiments/stage_b/heal_draft_0_6b_code \
    --teacher-topk "$TEACHER" \
    --distill-objective accept --softmin-tau 0.0 \
    --max-hours 4 --steps 400 --seq-len 1024 --grad-accum 32 \
    --lora-rank 16 --lora-alpha 32 --lr-lora 1e-4 --lr-mla 1e-4 --warmup-frac 0.03 \
    --val-corpus code --val-every 30 --val-chunks 50 --log-every 10 \
    || echo "[code] heal exited non-zero; benching anyway" >&2

echo "[code] $(date -Is) bench base vs code-tuned draft on held-out code"
uv run python experiments/stage_b/bench_draft_speedup.py \
    --corpus code --tuned-dir heal_draft_0_6b_code --tuned-label code-tuned \
    --out-suffix _code
echo "[code] $(date -Is) complete"
