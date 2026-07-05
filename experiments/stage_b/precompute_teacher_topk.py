"""Precompute the original (teacher) model's top-K logits for distillation.

Distill-to-original healing needs the frozen original's next-token distribution
for every training token. Two 4B models don't co-fit in 16 GB, so we cannot run
the teacher inline with the student — instead we run the teacher once, offline,
and cache its top-K logits per position. A peaked LM puts almost all mass in the
top ~64 tokens, so top-K is a tight approximation of the full distribution at a
tiny fraction of the storage.

Output (.pt): {
  "chunk_ids":   list[N] of (1, T) long   — the exact training chunks,
  "topk_ids":    (N, T-1, K) long          — teacher top-K token indices,
  "topk_logits": (N, T-1, K) float16       — teacher top-K raw logits,
  "model_id", "k", "seq_len", "split_seed"
}
Aligned by index: chunk i's labels are topk_*[i]. heal.py --teacher-topk consumes this.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from engine.model import Qwen3Model
from engine.weights import load_weights
from mla.calibrate import load_wikitext103_chunks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--model-id", default="Qwen/Qwen3-4B")
    ap.add_argument("--n-chunks", type=int, default=2048, help="teacher pool size")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--k", type=int, default=64, help="top-K to store per position")
    ap.add_argument("--split-seed", type=int, default=17, help="match heal --split-seed")
    ap.add_argument("--split", default="train")
    ap.add_argument("--corpus", default="wikitext", choices=["wikitext", "code"],
                    help="teacher workload: wikitext (broad) or code (narrow)")
    ap.add_argument("--out", default="experiments/stage_b/teacher_topk_wt103_2k.pt")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    dtype = getattr(torch, args.dtype)
    print(f"[teacher] loading {args.n_chunks} {args.corpus} chunks (seq_len={args.seq_len}, "
          f"seed={args.split_seed})", flush=True)
    if args.corpus == "code":
        from mla.calibrate import load_code_chunks
        chunks = load_code_chunks(n_samples=args.n_chunks, chunk_tokens=args.seq_len,
                                  tokenizer_id=args.model_id, seed=args.split_seed, skip=0)
    else:
        chunks = load_wikitext103_chunks(
            n_samples=args.n_chunks, chunk_tokens=args.seq_len,
            tokenizer_id=args.model_id, seed=args.split_seed, split=args.split,
        )
    print(f"[teacher]   got {len(chunks)} chunks", flush=True)

    print(f"[teacher] loading {args.model_id}", flush=True)
    loaded = load_weights(args.model_id, dtype=dtype, device="cpu")
    model = Qwen3Model.from_loaded(loaded).to(dtype=dtype, device=args.device).eval()
    del loaded
    torch.cuda.empty_cache()

    K = args.k
    N = len(chunks)
    Tm1 = args.seq_len - 1
    topk_ids = torch.empty((N, Tm1, K), dtype=torch.long)
    topk_logits = torch.empty((N, Tm1, K), dtype=torch.float16)

    import time
    t0 = time.time()
    with torch.inference_mode():
        for i, ids in enumerate(chunks):
            cache = model.alloc_cache(ids.shape[1])
            logits = model(ids.to(args.device), cache, start_pos=0)  # (1, T, V)
            vals, idx = logits[0, :-1].float().topk(K, dim=-1)        # (T-1, K)
            topk_ids[i] = idx.cpu()
            topk_logits[i] = vals.half().cpu()
            del cache, logits, vals, idx
            if (i + 1) % 64 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (N - i - 1) / rate
                print(f"[teacher]   {i+1}/{N}  ({rate:.1f} chunk/s, ETA {eta/60:.1f} min)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "chunk_ids": [c.cpu() for c in chunks],
        "topk_ids": topk_ids,
        "topk_logits": topk_logits,
        "model_id": args.model_id,
        "k": K,
        "seq_len": args.seq_len,
        "split_seed": args.split_seed,
    }, out)
    size_gb = out.stat().st_size / 1e9
    print(f"[teacher] wrote {out} ({size_gb:.2f} GB) in {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
