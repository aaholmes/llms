"""Collect per-layer output-gradient Fisher diagonals for the original model.

Supplies the output-sensitivity Kronecker factor missing from input-covariance
whitening: diag(G) where G = E[g gᵀ], g = ∂L/∂(K or V projection output),
empirical (true-next-token) Fisher. Feeds `mla.convert --fisher` for a
two-sided (C ⊗ G) Fisher-weighted SVD init.

Output (.pt): {"fisher_k": [ (num_kv*head_dim,) ]×L, "fisher_v": [...]×L,
               "model_id", "token_count"}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from engine.model import Qwen3Model
from engine.weights import load_weights
from mla.calibrate import collect_output_fisher, load_wikitext103_chunks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--model-id", default="Qwen/Qwen3-4B")
    ap.add_argument("--n-chunks", type=int, default=512, help="calibration chunks (fwd+bwd)")
    ap.add_argument("--chunk-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42, help="match the covariance calib seed")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="experiments/stage_b/fisher_wt103.pt")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    dtype = getattr(torch, args.dtype)
    print(f"[fisher] loading {args.n_chunks} × {args.chunk_tokens}-tok chunks "
          f"(seed={args.seed}, split={args.split})", flush=True)
    chunks = load_wikitext103_chunks(
        n_samples=args.n_chunks, chunk_tokens=args.chunk_tokens,
        tokenizer_id=args.model_id, seed=args.seed, split=args.split,
    )
    print(f"[fisher]   got {len(chunks)} chunks", flush=True)

    print(f"[fisher] loading {args.model_id}", flush=True)
    loaded = load_weights(args.model_id, dtype=dtype, device="cpu")
    model = Qwen3Model.from_loaded(loaded).to(dtype=dtype, device=args.device)
    del loaded
    torch.cuda.empty_cache()

    # Accumulate output Fisher on CPU (fp64) to avoid VRAM pressure under backward.
    fk, fv = collect_output_fisher(
        model, chunks, accumulator_device="cpu", progress_every=32,
    )
    fk = [x.cpu() for x in fk]
    fv = [x.cpu() for x in fv]
    print(f"[fisher] done: {len(fk)} layers; "
          f"k[0] mean={fk[0].mean():.3e} min={fk[0].min():.3e} max={fk[0].max():.3e}",
          flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"fisher_k": fk, "fisher_v": fv, "model_id": args.model_id}, out)
    print(f"[fisher] wrote {out} ({out.stat().st_size/1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
