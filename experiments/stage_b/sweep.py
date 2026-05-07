"""Stage B perplexity sweep over (rank, d_rope) for Qwen3-4B.

Pipeline per grid point:
  1. Joint-SVD conversion against the wt103_1k calibration (CPU).
  2. apply_mla into a fresh Qwen3Model on cuda.
  3. evaluate_ppl over a held-out 256-chunk × 1024-token slice from the
     WikiText-103 validation split (disjoint from the train-split
     calibration by construction).

Emits ``experiments/stage_b/eval_summary.md`` (markdown table + interpretation)
and ``experiments/stage_b/eval_summary.json`` (machine-readable).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from bench.eval_ppl import evaluate_ppl
from engine.model import Qwen3Model
from engine.weights import load_weights
from mla.calibrate import load_wikitext103_chunks
from mla.convert import convert_loaded_to_mla
from mla.swap import apply_mla


GRIDS: dict[str, list[tuple[int, int]]] = {
    # Partial-RoPE: K and V both compressed via shared latent; small RoPE-only
    # subspace of K kept full-rank. The headline MLA design.
    "partial-rope": [
        (256, 32),
        (192, 32),
        (128, 32),
        (128, 64),
        (96, 32),
        (64, 32),
    ],
    # V-only: d_rope = head_dim, so the entire K head stays in the rope branch
    # (uncompressed, full RoPE preserved). Only V is factored through the
    # shared latent. Lower compression ceiling but no structural change to
    # what the trained model expects on the K side.
    "v-only": [
        (1024, 128),  # rank == num_kv * head_dim → V uncompressed
        (768, 128),
        (512, 128),
        (384, 128),
        (256, 128),
        (128, 128),
    ],
}


def kv_bytes_per_token(
    num_kv_heads: int, head_dim: int,
    *, rank: int | None = None, d_rope: int | None = None,
    dtype_bytes: int = 2,
) -> int:
    """Bytes per token in the KV cache (per layer is implicit; report per-token-per-layer)."""
    if rank is None:
        # Baseline GQA: K + V per head per layer.
        return 2 * num_kv_heads * head_dim * dtype_bytes
    # MLA: shared latent + uncompressed rope-K per layer.
    return (rank + num_kv_heads * d_rope) * dtype_bytes


def _format_md_table(results: list[dict]) -> str:
    lines = [
        "| Config | rank | d_rope | KV B/tok | Compression | PPL | Δ PPL % | Avg discarded energy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        rank = r["rank"] if r["rank"] is not None else "—"
        drope = r["d_rope"] if r["d_rope"] is not None else "—"
        lines.append(
            f"| {r['label']} | {rank} | {drope} | "
            f"{r['kv_bytes_per_token']} | "
            f"{r['compression_ratio']:.2f}× | "
            f"{r['ppl']:.4f} | "
            f"{r['ppl_delta_pct']:+.2f}% | "
            f"{r['discarded_avg']:.2e} |"
        )
    return "\n".join(lines)


def _interpretation(results: list[dict]) -> str:
    """Human-readable summary: best config under the 2% PPL Δ gate, knee, etc."""
    baseline = results[0]
    others = results[1:]
    qualifying = [r for r in others if r["ppl_delta_pct"] <= 2.0]
    if qualifying:
        best = max(qualifying, key=lambda r: r["compression_ratio"])
        gate_status = (
            f"**Hard gate met.** Highest-compression config under PPL Δ ≤ 2 % is "
            f"`{best['label']}`: **{best['compression_ratio']:.2f}× compression** "
            f"at **+{best['ppl_delta_pct']:.2f} % PPL** "
            f"({best['ppl']:.4f} vs baseline {baseline['ppl']:.4f})."
        )
    else:
        best = min(others, key=lambda r: r["ppl_delta_pct"])
        gate_status = (
            f"**Hard gate missed at this calibration.** Lowest-Δ config is "
            f"`{best['label']}`: {best['ppl_delta_pct']:+.2f} % PPL at "
            f"{best['compression_ratio']:.2f}×. "
            f"Contingency: scale calibration to 4k×256, switch to per-layer "
            f"adaptive rank, or promote LoRA healing-FT."
        )
    return gate_status


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--model-id", default="Qwen/Qwen3-4B")
    ap.add_argument("--calib", default="experiments/stage_b/calib/wt103_1k.pt")
    ap.add_argument("--n-eval-chunks", type=int, default=256)
    ap.add_argument("--eval-chunk-tokens", type=int, default=1024)
    ap.add_argument("--eval-split", default="validation")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument(
        "--variant", default="partial-rope", choices=sorted(GRIDS.keys()),
        help="which (rank, d_rope) grid to sweep",
    )
    ap.add_argument(
        "--out", default=None,
        help="markdown summary path (default: experiments/stage_b/eval_summary_<variant>.md)",
    )
    ap.add_argument("--artifact-dir", default="experiments/stage_b")
    args = ap.parse_args(argv)
    grid = GRIDS[args.variant]
    if args.out is None:
        args.out = f"experiments/stage_b/eval_summary_{args.variant.replace('-', '_')}.md"

    dtype = getattr(torch, args.dtype)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Held-out evaluation slice from the validation split.
    print(f"[sweep] loading {args.n_eval_chunks} × {args.eval_chunk_tokens}-token "
          f"chunks from WikiText-103 ({args.eval_split} split)", flush=True)
    t0 = time.time()
    eval_chunks = load_wikitext103_chunks(
        n_samples=args.n_eval_chunks,
        chunk_tokens=args.eval_chunk_tokens,
        tokenizer_id=args.model_id,
        split=args.eval_split,
    )
    print(f"[sweep]   {len(eval_chunks)} chunks ready in {time.time()-t0:.1f}s "
          f"(if fewer than requested, the split was exhausted)", flush=True)

    # Load LoadedModel once (CPU) and the calibration covariances.
    print(f"[sweep] loading {args.model_id} weights", flush=True)
    t0 = time.time()
    loaded = load_weights(args.model_id, dtype=dtype, device="cpu")
    cfg = loaded.config
    print(f"[sweep]   loaded in {time.time()-t0:.1f}s; "
          f"{cfg.num_hidden_layers} layers, hidden={cfg.hidden_size}, "
          f"num_kv_heads={cfg.num_key_value_heads}, head_dim={cfg.head_dim}",
          flush=True)

    print(f"[sweep] loading calibration {args.calib}", flush=True)
    calib = torch.load(args.calib, map_location="cpu", weights_only=False)
    covariances = calib["covariances"]
    calib_meta = dict(calib.get("meta", {}))
    print(f"[sweep]   {len(covariances)} layers, "
          f"token_count={calib_meta.get('token_count', '?')}", flush=True)

    baseline_kv_bytes = kv_bytes_per_token(cfg.num_key_value_heads, cfg.head_dim)
    results: list[dict] = []

    # --- Baseline ---
    print(f"[sweep] === baseline (no MLA) ===", flush=True)
    model = Qwen3Model.from_loaded(loaded).to(dtype=dtype, device=args.device).eval()
    metrics = evaluate_ppl(model, eval_chunks, progress_every=32)
    results.append({
        "label": "baseline",
        "rank": None, "d_rope": None,
        "kv_bytes_per_token": baseline_kv_bytes,
        "compression_ratio": 1.0,
        "ppl": metrics["ppl"], "avg_nll": metrics["avg_nll"],
        "ppl_delta_pct": 0.0,
        "discarded_avg": 0.0, "discarded_max": 0.0,
        "wall_seconds": metrics["wall_seconds"],
    })
    baseline_ppl = metrics["ppl"]
    print(f"[sweep]   baseline PPL = {baseline_ppl:.4f} "
          f"({metrics['wall_seconds']:.1f}s, {metrics['token_count']} positions)",
          flush=True)
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # --- Sweep ---
    for rank, d_rope in grid:
        label = f"r{rank}_drope{d_rope}"
        print(f"[sweep] === {label} ===", flush=True)
        t0_total = time.time()

        artifact = convert_loaded_to_mla(
            loaded, covariances=covariances,
            rank=rank, d_rope=d_rope,
            factor_dtype=dtype,
            target_model_id=args.model_id,
            calibration_meta=calib_meta,
        )
        t_convert = time.time() - t0_total
        print(f"[sweep]   conversion: {t_convert:.1f}s", flush=True)

        artifact_path = artifact_dir / f"{label}.pt"
        torch.save(artifact, artifact_path)

        model = Qwen3Model.from_loaded(loaded).to(dtype=dtype, device=args.device).eval()
        apply_mla(model, artifact)

        metrics = evaluate_ppl(model, eval_chunks, progress_every=32)
        kv_bytes = kv_bytes_per_token(
            cfg.num_key_value_heads, cfg.head_dim, rank=rank, d_rope=d_rope,
        )
        compression = baseline_kv_bytes / kv_bytes
        ppl_delta = (metrics["ppl"] - baseline_ppl) / baseline_ppl * 100

        diags = artifact["meta"]["per_layer_diagnostics"]
        avg_disc = sum(d["discarded_energy"] for d in diags) / len(diags)
        max_disc = max(d["discarded_energy"] for d in diags)

        results.append({
            "label": label,
            "rank": rank, "d_rope": d_rope,
            "kv_bytes_per_token": kv_bytes,
            "compression_ratio": compression,
            "ppl": metrics["ppl"], "avg_nll": metrics["avg_nll"],
            "ppl_delta_pct": ppl_delta,
            "discarded_avg": avg_disc, "discarded_max": max_disc,
            "wall_seconds_convert": t_convert,
            "wall_seconds_eval": metrics["wall_seconds"],
        })

        print(
            f"[sweep]   {label}: PPL={metrics['ppl']:.4f} "
            f"(Δ {ppl_delta:+.2f} %), {compression:.2f}× compression, "
            f"discarded avg={avg_disc:.2e} max={max_disc:.2e}, "
            f"total {time.time()-t0_total:.1f}s",
            flush=True,
        )

        del model, artifact
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # --- Write summary ---
    body_lines = [
        f"# Stage B perplexity sweep — {args.variant}",
        "",
        f"Variant: `{args.variant}` "
        f"({'partial-RoPE compresses both K-nope and V via shared latent' if args.variant == 'partial-rope' else 'V-only: d_rope=head_dim so K stays uncompressed and full-RoPE'})",
        f"Model: `{args.model_id}` ({cfg.num_hidden_layers} layers, "
        f"hidden={cfg.hidden_size}, num_kv_heads={cfg.num_key_value_heads}, "
        f"head_dim={cfg.head_dim})",
        f"Calibration: `{args.calib}` "
        f"({calib_meta.get('token_count', '?')} tokens, train split, "
        f"seed={calib_meta.get('seed', '?')})",
        f"Eval slice: {len(eval_chunks)} chunks × {args.eval_chunk_tokens} tokens "
        f"from WikiText-103 `{args.eval_split}` split (disjoint from calibration)",
        f"Dtype: {args.dtype}; device: {args.device}",
        "",
        _format_md_table(results),
        "",
        _interpretation(results),
        "",
        "**KV bytes/token** is per-layer; full-context cache size scales linearly with "
        "context length and number of layers.",
        "**Compression ratio** is baseline KV-bytes-per-token / MLA KV-bytes-per-token.",
        "**Δ PPL %** is `(ppl_mla - ppl_baseline) / ppl_baseline * 100`.",
        "**Discarded energy** is the analytic C-weighted reconstruction error "
        "averaged across layers (rises with depth, as deeper layers carry more diverse activations).",
    ]
    out_path.write_text("\n".join(body_lines) + "\n")
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps({
        "results": results,
        "meta": {
            "model_id": args.model_id,
            "calibration_path": args.calib,
            "calibration_meta": calib_meta,
            "n_eval_chunks": len(eval_chunks),
            "eval_chunk_tokens": args.eval_chunk_tokens,
            "eval_split": args.eval_split,
            "dtype": args.dtype,
            "device": args.device,
            "grid": GRID,
        },
    }, indent=2))
    print(f"[sweep] wrote {out_path} and {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
