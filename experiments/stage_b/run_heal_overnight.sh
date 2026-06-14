#!/usr/bin/env bash
# Overnight healing-FT run for the 4x-compression operating point (r256_drope32).
#
# Budget: 9 wall-clock hours (set MAX_HOURS to override). The run stops itself
# cleanly at the first step boundary past the budget, after a final validation,
# and writes summary.{md,json} either way. best.pt holds the val-best trainable
# state throughout, so a crash or power loss costs at most one val interval.
#
# Sizing: one step is grad_accum=32 x seq_len=1024 = 32,768 tokens; at the
# ~300-450 tok/s training throughput implied by the calibration run that is
# ~75-110 s/step, so --steps 400 (~13M tokens) roughly fills 9 h and the
# cosine schedule completes if throughput lands mid-range; --max-hours cuts
# it off cleanly if slower. val every 30 steps ~= every 1M tokens, so
# training_log.jsonl yields a recovery curve, ~4% val overhead.
#
# Usage:
#   nohup bash experiments/stage_b/run_heal_overnight.sh > /dev/null 2>&1 &
#   tail -f experiments/stage_b/heal_r256_drope32/run.log
set -euo pipefail
cd "$(dirname "$0")/../.."

ARTIFACT=${ARTIFACT:-experiments/stage_b/r256_drope32.pt}
OUT=${OUT:-experiments/stage_b/heal_r256_drope32}
MAX_HOURS=${MAX_HOURS:-9}

[ -f "$ARTIFACT" ] || { echo "missing artifact: $ARTIFACT" >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader >&2

mkdir -p "$OUT"
uv run python -m mla.heal \
    --mla-artifact "$ARTIFACT" \
    --out "$OUT" \
    --max-hours "$MAX_HOURS" \
    --steps "${STEPS:-400}" \
    --seq-len 1024 --grad-accum 32 \
    --lora-rank 16 --lora-alpha 32 \
    --lr-mla 5e-5 --lr-lora 1e-4 --warmup-frac 0.03 \
    --val-every 30 --val-chunks 50 \
    --train-chunks 20000 \
    --log-every 10 \
    2>&1 | tee "$OUT/run.log"
