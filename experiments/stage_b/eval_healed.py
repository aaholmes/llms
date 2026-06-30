"""Clean apples-to-apples PPL eval of the healed r256_drope32 model.

Reconstructs the healed model exactly as the heal harness built it
(base weights -> apply_mla -> wrap_lora -> load_trainable best.pt) and
evaluates it over the SAME held-out WikiText-103 validation slice used by
the no-FT sweep (256 x 1024 tokens, seed 42). Re-confirms the baseline on
that identical slice so the delta is exact.

Emits experiments/stage_b/eval_summary_healed.{md,json}.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from bench.eval_ppl import evaluate_ppl
from engine.model import Qwen3Model
from engine.weights import load_weights
from mla.calibrate import load_wikitext103_chunks
from mla.heal import load_trainable, wrap_lora
from mla.swap import apply_mla

MODEL_ID = "Qwen/Qwen3-4B"
ARTIFACT = "experiments/stage_b/r256_drope32.pt"
BEST = "experiments/stage_b/heal_r256_drope32/best.pt"
LORA_RANK = 16
LORA_ALPHA = 32.0
N_EVAL_CHUNKS = 256
EVAL_CHUNK_TOKENS = 1024
DEVICE = "cuda"
DTYPE = torch.bfloat16


def main() -> int:
    print(f"[eval] loading {N_EVAL_CHUNKS} x {EVAL_CHUNK_TOKENS}-tok validation chunks", flush=True)
    eval_chunks = load_wikitext103_chunks(
        n_samples=N_EVAL_CHUNKS,
        chunk_tokens=EVAL_CHUNK_TOKENS,
        tokenizer_id=MODEL_ID,
        split="validation",
    )
    print(f"[eval]   {len(eval_chunks)} chunks ready", flush=True)

    results = {}

    # --- Baseline (unconverted), same slice ---
    print("[eval] === baseline (no MLA) ===", flush=True)
    loaded = load_weights(MODEL_ID, dtype=DTYPE, device="cpu")
    model = Qwen3Model.from_loaded(loaded).to(dtype=DTYPE, device=DEVICE).eval()
    del loaded
    torch.cuda.empty_cache()
    m = evaluate_ppl(model, eval_chunks, progress_every=64)
    results["baseline"] = {"ppl": m["ppl"], "avg_nll": m["avg_nll"],
                           "tokens": m["token_count"], "wall_s": m["wall_seconds"]}
    print(f"[eval]   baseline PPL = {m['ppl']:.4f}", flush=True)
    del model
    torch.cuda.empty_cache()

    # --- Healed r256_drope32 ---
    print("[eval] === healed r256_drope32 ===", flush=True)
    loaded = load_weights(MODEL_ID, dtype=DTYPE, device="cpu")
    model = Qwen3Model.from_loaded(loaded).to(dtype=DTYPE, device=DEVICE)
    del loaded
    torch.cuda.empty_cache()
    artifact = torch.load(ARTIFACT, map_location="cpu", weights_only=False)
    apply_mla(model, artifact)
    wrap_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA)
    missing = load_trainable(model, BEST)
    if missing:
        print(f"[eval]   WARNING unmatched trainable names: {missing[:5]}...", flush=True)
    model.eval()
    m = evaluate_ppl(model, eval_chunks, progress_every=64)
    results["healed_r256_drope32"] = {"ppl": m["ppl"], "avg_nll": m["avg_nll"],
                                      "tokens": m["token_count"], "wall_s": m["wall_seconds"]}
    print(f"[eval]   healed PPL = {m['ppl']:.4f}", flush=True)

    base_ppl = results["baseline"]["ppl"]
    heal_ppl = results["healed_r256_drope32"]["ppl"]
    delta_pct = (heal_ppl - base_ppl) / base_ppl * 100.0
    results["delta_pct_vs_baseline"] = delta_pct
    results["compression"] = 4.0
    results["meta"] = {"n_chunks": len(eval_chunks), "chunk_tokens": EVAL_CHUNK_TOKENS,
                       "no_ft_ppl": 200.33, "no_ft_delta_pct": 826.0}

    out = Path("experiments/stage_b/eval_summary_healed.json")
    out.write_text(json.dumps(results, indent=2))
    md = Path("experiments/stage_b/eval_summary_healed.md")
    md.write_text(
        "# Healed r256_drope32 PPL (4x compression), 256x1024 WT103-val slice\n\n"
        f"| Config | Compression | PPL | Δ% vs baseline |\n"
        f"|---|---:|---:|---:|\n"
        f"| baseline | 1.00× | {base_ppl:.4f} | — |\n"
        f"| no-FT r256_drope32 | 4.00× | 200.33 | +826% |\n"
        f"| **healed r256_drope32** | 4.00× | **{heal_ppl:.4f}** | **{delta_pct:+.2f}%** |\n"
    )
    print(f"[eval] baseline={base_ppl:.4f}  healed={heal_ppl:.4f}  Δ={delta_pct:+.2f}%", flush=True)
    print(f"[eval] wrote {md} and {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
