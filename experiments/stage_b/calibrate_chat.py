"""Collect per-layer input covariances on chat-formatted prompts (Dolly-15K).

Same output format as the WikiText calibration (``calib/wt103_1k.pt``), so
``mla.convert --calib`` takes it unchanged. The default token budget matches
that file's 256,000 tokens; only the calibration text differs.

Run (GPU):
    uv run python -m experiments.stage_b.calibrate_chat --out experiments/stage_b/calib/dolly_256k.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from engine.model import Qwen3Model
from engine.weights import load_weights
from mla.calibrate import collect_covariances, load_chat_prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--tokens", type=int, default=256_000)
    ap.add_argument("--max-tokens", type=int, default=1024, help="per-prompt truncation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompts = load_chat_prompts(args.tokens, max_tokens=args.max_tokens,
                                tokenizer_id=args.model, seed=args.seed)
    token_count = sum(p.shape[1] for p in prompts)
    print(f"[calib-chat] {len(prompts)} prompts, {token_count} tokens", flush=True)

    loaded = load_weights(args.model, dtype=torch.bfloat16, device="cpu")
    model = Qwen3Model.from_loaded(loaded).to(dtype=torch.bfloat16, device="cuda").eval()
    del loaded
    t0 = time.time()
    covs = collect_covariances(model, [p.to("cuda") for p in prompts],
                               accumulator_device="cuda", progress_every=200)
    meta = {
        "model_id": args.model, "num_layers": len(covs), "hidden_size": covs[0].shape[0],
        "dataset": "databricks-dolly-15k", "format": "chat template, thinking off",
        "n_prompts": len(prompts), "max_tokens": args.max_tokens,
        "token_count": token_count, "seed": args.seed,
        "wall_seconds_forward": time.time() - t0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"covariances": [c.cpu() for c in covs], "meta": meta}, out)
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[calib-chat] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
