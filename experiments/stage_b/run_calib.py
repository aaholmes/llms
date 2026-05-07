"""Run the activation-covariance calibration for Qwen3-4B.

Streams a deterministic WikiText-103 slice through the engine and persists
per-layer covariances C_ℓ = (1/N) Σ x_t x_tᵀ for the post-norm-1 attention
input. Output feeds the activation-aware SVD that produces the MLA factors.

Usage (defaults match the calibration described in DESIGN.md):
    uv run python -m experiments.stage_b.run_calib

Override slice or model:
    uv run python -m experiments.stage_b.run_calib \\
        --model-id Qwen/Qwen3-0.6B --n-samples 100 \\
        --out experiments/stage_b/calib/wt103_smoke_0_6b.pt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from engine.model import Qwen3Model
from engine.weights import load_weights
from mla.calibrate import CovarianceCollector, load_wikitext103_chunks


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent
        ).decode().strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Qwen/Qwen3-4B")
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--chunk-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        default="experiments/stage_b/calib/wt103_1k.pt",
        help="output path for the covariance artifact (.pt)",
    )
    ap.add_argument(
        "--device", default="cuda", help="device for the model forward pass"
    )
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="model dtype during the forward pass",
    )
    ap.add_argument(
        "--accumulator-device",
        default=None,
        help=(
            "device for the fp64 covariance accumulators; defaults to the "
            "model device (keeps the per-iter reduction on-GPU)"
        ),
    )
    ap.add_argument("--progress-every", type=int, default=50)
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = getattr(torch, args.dtype)

    print(f"[calib] loading {args.model_id} on {args.device} ({args.dtype})")
    t0 = time.time()
    loaded = load_weights(args.model_id, dtype=dtype, device=args.device)
    num_layers = loaded.config.num_hidden_layers
    hidden_size = loaded.config.hidden_size
    model = Qwen3Model.from_loaded(loaded).to(dtype=dtype, device=args.device).eval()
    del loaded
    if args.device == "cuda":
        torch.cuda.empty_cache()
    load_s = time.time() - t0
    print(f"[calib] model ready in {load_s:.1f}s; {num_layers} layers, hidden={hidden_size}")

    print(f"[calib] tokenizing {args.n_samples} × {args.chunk_tokens} WikiText-103 chunks "
          f"(seed={args.seed})")
    t0 = time.time()
    chunks = load_wikitext103_chunks(
        n_samples=args.n_samples,
        chunk_tokens=args.chunk_tokens,
        tokenizer_id=args.model_id,
        seed=args.seed,
    )
    tok_s = time.time() - t0
    if len(chunks) < args.n_samples:
        print(f"[calib] WARNING: requested {args.n_samples} chunks, got {len(chunks)} "
              f"(insufficient text in WikiText-103 split)")
    print(f"[calib] {len(chunks)} chunks ready in {tok_s:.1f}s")

    accumulator_device = args.accumulator_device or args.device
    print(f"[calib] running forward passes; covariances accumulate on {accumulator_device}")
    t0 = time.time()
    chunks_on_device = [c.to(args.device) for c in chunks]
    with CovarianceCollector(model, accumulator_device=accumulator_device) as col:
        for i, ids in enumerate(chunks_on_device):
            cache = model.alloc_cache(ids.shape[1])
            with torch.inference_mode():
                model(ids, cache, start_pos=0)
            del cache
            if (i + 1) % args.progress_every == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(chunks_on_device) - i - 1) / max(rate, 1e-9)
                print(
                    f"[calib]   {i+1}/{len(chunks_on_device)} chunks "
                    f"({col.token_count} tokens accumulated, "
                    f"{rate:.1f} chunks/s, ETA {eta:.0f}s)"
                )
        covs = col.covariances()
        token_count = col.token_count
    fwd_s = time.time() - t0
    print(f"[calib] {token_count} tokens accumulated in {fwd_s:.1f}s "
          f"({token_count / fwd_s:.0f} tok/s)")

    covs = [c.cpu() for c in covs]

    artifact = {
        "covariances": covs,
        "meta": {
            "model_id": args.model_id,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "n_samples": args.n_samples,
            "chunk_tokens": args.chunk_tokens,
            "actual_chunks": len(chunks),
            "token_count": token_count,
            "seed": args.seed,
            "dataset": "wikitext-103-v1",
            "dataset_split": "train",
            "tokenizer_id": args.model_id,
            "model_dtype": args.dtype,
            "accumulator_dtype": "float64",
            "accumulator_device": str(accumulator_device),
            "torch_version": torch.__version__,
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "git_rev": _git_rev(),
            "wall_seconds": {"load": load_s, "tokenize": tok_s, "forward": fwd_s},
        },
    }

    print(f"[calib] saving artifact to {out_path}")
    torch.save(artifact, out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"[calib] wrote {size_mb:.1f} MB")

    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(artifact["meta"], f, indent=2)
    print(f"[calib] wrote sidecar meta to {meta_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
