#!/usr/bin/env bash
# Recalibrate the SVD on chat-formatted prompts, convert at 4× and 1.33×, and
# run the retrieval eval on both un-healed models at 1K and 4K.
set -euo pipefail
cd "$(dirname "$0")/../.."
CAL=experiments/stage_b/calib/dolly_256k.pt
[ -f "$CAL" ] || uv run python -m experiments.stage_b.calibrate_chat --out "$CAL"
[ -f experiments/stage_b/r256_drope32_chatcal.pt ] || uv run python -m mla.convert Qwen/Qwen3-4B \
  --rank 256 --d-rope 32 --calib "$CAL" --out experiments/stage_b/r256_drope32_chatcal.pt
[ -f experiments/stage_b/r512_drope128_chatcal.pt ] || uv run python -m mla.convert Qwen/Qwen3-4B \
  --rank 512 --d-rope 128 --calib "$CAL" --out experiments/stage_b/r512_drope128_chatcal.pt
uv run python -m experiments.stage_b.niah_eval --lengths 1024 4096 \
  --arms mla-4x-unhealed-chatcal mla-1.33x-unhealed-chatcal
uv run python -m experiments.stage_b.niah_analyze
