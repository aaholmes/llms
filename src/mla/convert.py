"""MHA→MLA conversion: turn a Qwen3 LoadedModel + per-layer covariances into an MLA artifact.

Per layer the conversion does **one** activation-aware SVD over the stacked
nope rows of K plus all of V, producing a shared down-projection
``W_dkv = B`` and per-head up-projections ``W_uk_nope``/``W_uv``. The rope
rows of K are kept full-rank in ``W_kr``. Q, O, q_norm, k_norm are copied
through unchanged. Forward-equivalent to baseline GQA at rank=hidden when
``C = I``; lossy at rank<hidden, with the residual quantified by the
returned ``discarded_energy``.

Public API:
    convert_loaded_to_mla(loaded, covariances, rank, d_rope, ...) -> artifact dict
    main()                                                          # CLI entry point

The artifact format is documented in ``DESIGN.md`` / the plan file. In short,
``artifact["state"]`` keys are ``layers.{i}.attn.<name>.weight`` mirroring
the engine's name-map convention; ``artifact["meta"]`` records every shape
and provenance field needed by ``apply_mla`` and downstream evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

from engine.weights import LoadedModel, load_weights
from mla.svd import activation_aware_factor, discarded_singular_energy


def _row_indices(num_kv_heads: int, head_dim: int, d_rope: int) -> tuple[list[int], list[int]]:
    """Return (nope_rows, rope_rows) into a (num_kv_heads * head_dim,) row axis."""
    d_nope = head_dim - d_rope
    nope_rows = [
        h * head_dim + i
        for h in range(num_kv_heads)
        for i in range(d_nope)
    ]
    rope_rows = [
        h * head_dim + d_nope + j
        for h in range(num_kv_heads)
        for j in range(d_rope)
    ]
    return nope_rows, rope_rows


def _git_rev() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def convert_loaded_to_mla(
    loaded: LoadedModel,
    *,
    covariances: list[torch.Tensor],
    rank: int,
    d_rope: int,
    factor_dtype: torch.dtype = torch.bfloat16,
    ridge_lambda: float = 1e-6,
    ridge_max_iters: int = 6,
    calibration_meta: dict | None = None,
    target_model_id: str | None = None,
    progress: bool = False,
) -> dict:
    """Run per-layer joint SVD; return the MLA artifact dict (no I/O).

    Parameters
    ----------
    loaded
        Output of ``engine.weights.load_weights``. Read-only.
    covariances
        One symmetric PSD ``(hidden_size, hidden_size)`` matrix per layer.
        Order must match ``loaded.config.num_hidden_layers``.
    rank
        Target latent rank ``r``; ``W_dkv`` will be ``(r, hidden_size)``.
    d_rope
        Per-head RoPE-subspace width. ``0 ≤ d_rope ≤ head_dim``.
    factor_dtype
        Output dtype for factor weights. ``apply_mla`` will cast to the
        target model's dtype regardless, so this controls the on-disk
        precision (bf16 by default keeps artifacts compact).
    calibration_meta
        Provenance metadata from the calibration artifact. Stored in the
        output artifact's meta for traceability.
    target_model_id
        If both this and ``calibration_meta["model_id"]`` are provided, the
        two must match — guards against converting Qwen3-4B with a
        Qwen3-0.6B calibration by accident.
    """
    cfg = loaded.config
    num_layers = cfg.num_hidden_layers
    hidden_size = cfg.hidden_size
    head_dim = cfg.head_dim
    num_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads

    if not (1 <= rank <= hidden_size):
        raise ValueError(f"rank must be in [1, {hidden_size}]; got rank={rank}")
    if not (0 <= d_rope <= head_dim):
        raise ValueError(f"d_rope must be in [0, {head_dim}]; got d_rope={d_rope}")
    if len(covariances) != num_layers:
        raise ValueError(
            f"got {len(covariances)} covariance matrices for a {num_layers}-layer model"
        )
    if calibration_meta is not None and target_model_id is not None:
        calib_id = calibration_meta.get("model_id")
        if calib_id is not None and calib_id != target_model_id:
            raise ValueError(
                f"calibration model_id={calib_id!r} does not match target "
                f"model_id={target_model_id!r}"
            )

    d_nope = head_dim - d_rope
    nope_rows, rope_rows = _row_indices(num_kv_heads, head_dim, d_rope)

    rope_theta = float(
        getattr(cfg, "rope_parameters", None) and cfg.rope_parameters.get("rope_theta")
        or getattr(cfg, "rope_theta", 10000.0)
    )

    out_state: dict[str, torch.Tensor] = {}
    diagnostics: list[dict] = []

    t0_total = time.time()
    for i in range(num_layers):
        prefix = f"layers.{i}.attn."
        W_K = loaded.state[prefix + "k.weight"]
        W_V = loaded.state[prefix + "v.weight"]
        C = covariances[i]
        if C.shape != (hidden_size, hidden_size):
            raise ValueError(
                f"layer {i}: covariance shape {tuple(C.shape)} != ({hidden_size}, {hidden_size})"
            )

        if d_nope > 0 and d_rope > 0:
            W_K_nope = W_K[nope_rows]
            W_kr = W_K[rope_rows]
        elif d_nope > 0:  # d_rope == 0
            W_K_nope = W_K
            W_kr = None
        else:  # d_nope == 0, d_rope == head_dim
            W_K_nope = None
            W_kr = W_K

        # Joint SVD over the stacked nope-K + V rows so they share W_dkv.
        if W_K_nope is not None:
            W_stack = torch.cat([W_K_nope, W_V], dim=0)
        else:
            W_stack = W_V

        A, B = activation_aware_factor(
            W_stack, C, rank=rank,
            ridge_lambda=ridge_lambda, ridge_max_iters=ridge_max_iters,
        )
        # Diagnostic: how much C-weighted error is being thrown away?
        discarded = discarded_singular_energy(
            W_stack, C, rank=rank,
            ridge_lambda=ridge_lambda, ridge_max_iters=ridge_max_iters,
        ).item()

        # Split A back into its K-nope and V slices.
        if W_K_nope is not None:
            split = num_kv_heads * d_nope
            A_K_nope = A[:split]
            A_V = A[split:]
        else:
            A_K_nope = None
            A_V = A

        # Cast factors and copy-throughs to factor_dtype.
        out_state[prefix + "q.weight"] = loaded.state[prefix + "q.weight"].to(factor_dtype).clone()
        out_state[prefix + "o.weight"] = loaded.state[prefix + "o.weight"].to(factor_dtype).clone()
        out_state[prefix + "q_norm.weight"] = loaded.state[prefix + "q_norm.weight"].to(factor_dtype).clone()
        out_state[prefix + "k_norm.weight"] = loaded.state[prefix + "k_norm.weight"].to(factor_dtype).clone()
        out_state[prefix + "dkv.weight"] = B.to(factor_dtype).contiguous()
        out_state[prefix + "uv.weight"] = A_V.to(factor_dtype).contiguous()
        if A_K_nope is not None:
            out_state[prefix + "uk_nope.weight"] = A_K_nope.to(factor_dtype).contiguous()
        if W_kr is not None:
            out_state[prefix + "kr.weight"] = W_kr.to(factor_dtype).clone().contiguous()

        diagnostics.append({
            "layer": i,
            "discarded_energy": float(discarded),
            "ridge_lambda": ridge_lambda,
        })
        if progress:
            elapsed = time.time() - t0_total
            print(f"[convert]   layer {i+1}/{num_layers} done "
                  f"(discarded_energy={discarded:.3e}, t+{elapsed:.1f}s)", flush=True)

    wall_seconds = time.time() - t0_total

    meta: dict = {
        "model_id": target_model_id,
        "rank": rank,
        "d_rope": d_rope,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "max_position_embeddings": cfg.max_position_embeddings,
        "rope_theta": rope_theta,
        "rms_eps": float(cfg.rms_norm_eps),
        "qk_norm_mode": "single",
        "factor_dtype": str(factor_dtype).replace("torch.", ""),
        "ridge_lambda": ridge_lambda,
        "torch_version": torch.__version__,
        "git_rev": _git_rev(),
        "wall_seconds_factor": wall_seconds,
        "per_layer_diagnostics": diagnostics,
    }
    if calibration_meta is not None:
        meta["calibration_meta"] = dict(calibration_meta)

    return {"state": out_state, "meta": meta}


def _resolve_calib_path(arg: str) -> Path:
    """Accept either a real path or a shorthand like 'wt103-1k'."""
    p = Path(arg)
    if p.exists():
        return p
    # Shorthand: lookup under experiments/stage_b/calib/{name}.pt with hyphens→underscores.
    candidate = Path("experiments/stage_b/calib") / f"{arg.replace('-', '_')}.pt"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"calibration not found: {arg!r} (also tried {candidate})")


def _default_out_path(model_id: str, rank: int, d_rope: int) -> Path:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_id).strip("_").lower()
    return Path("experiments/stage_b") / f"{slug}_r{rank}_drope{d_rope}.pt"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("model_id", help="HF model id, e.g. Qwen/Qwen3-4B")
    ap.add_argument("--rank", type=int, required=True, help="target latent rank")
    ap.add_argument("--d-rope", type=int, required=True, help="per-head RoPE-subspace width")
    ap.add_argument(
        "--calib", required=True,
        help="calibration artifact (.pt). Shorthand like 'wt103-1k' resolves "
             "to experiments/stage_b/calib/wt103_1k.pt.",
    )
    ap.add_argument("--out", default=None, help="output artifact (.pt)")
    ap.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
        help="output factor dtype",
    )
    ap.add_argument(
        "--device", default="cpu",
        help="device for SVD compute (default cpu; only escalate if wall-clock budget is missed)",
    )
    ap.add_argument("--ridge-lambda", type=float, default=1e-6)
    args = ap.parse_args(argv)

    factor_dtype = getattr(torch, args.dtype)
    out_path = Path(args.out) if args.out else _default_out_path(args.model_id, args.rank, args.d_rope)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[convert] loading {args.model_id}", flush=True)
    t0 = time.time()
    loaded = load_weights(args.model_id, dtype=factor_dtype, device=args.device)
    print(f"[convert]   loaded in {time.time() - t0:.1f}s; "
          f"{loaded.config.num_hidden_layers} layers, hidden={loaded.config.hidden_size}",
          flush=True)

    calib_path = _resolve_calib_path(args.calib)
    print(f"[convert] loading calibration {calib_path}", flush=True)
    t0 = time.time()
    calib = torch.load(calib_path, map_location="cpu", weights_only=False)
    covariances = calib["covariances"]
    calib_meta_in = dict(calib.get("meta", {}))
    calib_meta_in["calibration_path"] = str(calib_path)
    print(f"[convert]   calib loaded in {time.time() - t0:.1f}s; "
          f"{len(covariances)} layers; "
          f"token_count={calib_meta_in.get('token_count', '?')}",
          flush=True)

    print(f"[convert] running per-layer joint SVD (rank={args.rank}, d_rope={args.d_rope})",
          flush=True)
    artifact = convert_loaded_to_mla(
        loaded,
        covariances=covariances,
        rank=args.rank,
        d_rope=args.d_rope,
        factor_dtype=factor_dtype,
        ridge_lambda=args.ridge_lambda,
        calibration_meta=calib_meta_in,
        target_model_id=args.model_id,
        progress=True,
    )

    print(f"[convert] saving artifact to {out_path}", flush=True)
    torch.save(artifact, out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"[convert]   wrote {size_mb:.1f} MB", flush=True)

    meta_path = out_path.with_suffix(".meta.json")
    with meta_path.open("w") as f:
        json.dump(artifact["meta"], f, indent=2)
    print(f"[convert]   sidecar meta to {meta_path}", flush=True)

    avg_disc = sum(d["discarded_energy"] for d in artifact["meta"]["per_layer_diagnostics"]) / len(
        artifact["meta"]["per_layer_diagnostics"]
    )
    max_disc = max(d["discarded_energy"] for d in artifact["meta"]["per_layer_diagnostics"])
    print(f"[convert] per-layer discarded energy: avg={avg_disc:.3e}, max={max_disc:.3e}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
