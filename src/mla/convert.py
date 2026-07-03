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
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

from engine.weights import LoadedModel, load_weights
from mla.svd import (
    activation_aware_factor,
    discarded_singular_energy,
    weighted_singular_values,
)

logger = logging.getLogger(__name__)


def per_head_rope_pair_perm(head_dim: int, d_rope: int) -> list[int]:
    """Per-head permutation that brings the rope subspace to the end of head_dim.

    The rope subspace is the first ``d_rope/2`` *original* Qwen3 RoPE pairs
    ``(i, i + head_dim/2)`` for ``i ∈ [0, d_rope/2)``. After this permutation:

      * positions ``[0, d_nope)`` hold the no-rope dims
        ``[d_rope/2, head_dim/2) ∪ [head_dim/2 + d_rope/2, head_dim)``
      * positions ``[d_nope, head_dim)`` hold the rope dims, laid out as
        ``[0, d_rope/2) ++ [head_dim/2, head_dim/2 + d_rope/2)`` — so that
        ``_rotate_half`` over the trailing ``d_rope`` pairs original-dim
        ``i`` with ``i + head_dim/2``, exactly matching Qwen3 training.

    Returns a length-``head_dim`` list ``perm`` such that
    ``permuted_head[i] = original_head[perm[i]]``.
    """
    half = head_dim // 2
    h_rope = d_rope // 2
    rope_dims = list(range(h_rope)) + list(range(half, half + h_rope))
    nope_dims = list(range(h_rope, half)) + list(range(half + h_rope, head_dim))
    return nope_dims + rope_dims


def _expand_row_perm(perm: list[int], num_heads: int, head_dim: int) -> list[int]:
    """Expand a per-head perm to a row permutation for a (num_heads*head_dim, *) matrix."""
    return [h * head_dim + perm[i] for h in range(num_heads) for i in range(head_dim)]


def _rope_perms(head_dim: int, num_heads: int, num_kv_heads: int, d_rope: int) -> dict:
    """Per-head + per-row permutation/index sets shared by spectrum and factor passes.

    Factored out so the activation-weighted spectrum used for adaptive-rank
    allocation is built from *exactly* the same stacked ``[K_nope; V]`` rows
    that the factorization later decomposes — they must not diverge.
    """
    d_nope = head_dim - d_rope
    head_perm = per_head_rope_pair_perm(head_dim, d_rope)
    return {
        "d_nope": d_nope,
        "head_perm": head_perm,
        "k_row_perm": _expand_row_perm(head_perm, num_kv_heads, head_dim),
        "q_row_perm": _expand_row_perm(head_perm, num_heads, head_dim),
        "nope_rows": [h * head_dim + i for h in range(num_kv_heads) for i in range(d_nope)],
        "rope_rows": [
            h * head_dim + d_nope + j for h in range(num_kv_heads) for j in range(d_rope)
        ],
    }


def _w_stack_for_layer(loaded: LoadedModel, i: int, perms: dict, d_rope: int):
    """Build the stacked ``[K_nope; V]`` weight (and the K-nope/rope/V pieces) for layer ``i``."""
    prefix = f"layers.{i}.attn."
    W_K_perm = loaded.state[prefix + "k.weight"][perms["k_row_perm"], :]
    W_V = loaded.state[prefix + "v.weight"]  # *not* permuted
    d_nope = perms["d_nope"]
    if d_nope > 0 and d_rope > 0:
        W_K_nope = W_K_perm[perms["nope_rows"]]
        W_kr = W_K_perm[perms["rope_rows"]]
    elif d_nope > 0:  # d_rope == 0
        W_K_nope = W_K_perm
        W_kr = None
    else:  # d_nope == 0, d_rope == head_dim
        W_K_nope = None
        W_kr = W_K_perm
    W_stack = torch.cat([W_K_nope, W_V], dim=0) if W_K_nope is not None else W_V
    return W_stack, W_K_nope, W_kr, W_V


def _fisher_row_weight(fisher_k_l, fisher_v_l, perms: dict, d_rope: int) -> torch.Tensor:
    """Per-output-row weight ``√diag(G)`` for the stacked ``[K_nope; V]``, matching
    the exact row order ``_w_stack_for_layer`` produces (K permuted, V as-is)."""
    d_nope = perms["d_nope"]
    fk_perm = fisher_k_l[perms["k_row_perm"]]
    if d_nope > 0 and d_rope > 0:
        parts = [fk_perm[perms["nope_rows"]], fisher_v_l]
    elif d_nope > 0:  # d_rope == 0 → all K rows are nope
        parts = [fk_perm, fisher_v_l]
    else:              # d_nope == 0 → V-only stack
        parts = [fisher_v_l]
    return torch.sqrt(torch.cat([p.to(torch.float64) for p in parts]).clamp_min(1e-12))


def allocate_adaptive_ranks(
    spectra: list[torch.Tensor],
    total_budget: int,
    *,
    min_rank: int = 1,
    score: str = "raw",
) -> list[int]:
    """Allocate a total rank budget across layers by greedy marginal value.

    Given each layer's C-weighted singular spectrum (descending), assigns
    per-layer ranks ``r_i`` with ``Σ r_i = total_budget``. Greedy: repeatedly
    give one rank to the layer with the largest marginal score for its *next*
    singular value, until the budget is spent. The ``score`` selects the
    marginal value:

    - ``"raw"`` — marginal ``= σ²_{r+1}`` (the next squared singular value).
      Equivalent to a static global top-``budget`` sort of pooled σ²; provably
      maximizes retained Σσ² (minimizes total activation-weighted reconstruction
      error). **But** because late layers have larger-scale spectra, this
      starves early layers → empirically ≈6× worse PPL than uniform. See the
      adaptive-rank negative result in DESIGN.md.
    - ``"normalized"`` — marginal ``= σ²_{r+1} / Σ_{m>r} σ²_m`` (next value over
      the layer's *remaining* tail energy at its current rank). This is CARE's
      (arXiv:2603.17946) score: it makes layers of different spectral scale
      comparable and prevents early-layer starvation. Path-dependent (the
      denominator shrinks as a layer accrues rank), so it needs the true greedy
      loop, not a static sort.

    Every layer is floored at ``min_rank`` and capped at its spectrum length
    (``min(W.shape)`` — full rank, zero discarded). Returns ``list[int]``.
    """
    import heapq

    if score not in ("raw", "normalized"):
        raise ValueError(f"score must be 'raw' or 'normalized'; got {score!r}")
    n = len(spectra)
    caps = [int(s.numel()) for s in spectra]
    floor_total = min_rank * n
    cap_total = sum(caps)
    if total_budget < floor_total:
        raise ValueError(
            f"total_budget={total_budget} < min_rank*num_layers={floor_total}"
        )
    if any(min_rank > c for c in caps):
        raise ValueError(f"min_rank={min_rank} exceeds a layer's full rank {min(caps)}")
    if total_budget >= cap_total:
        return list(caps)  # budget covers full rank everywhere

    sq = [s.to(torch.float64) ** 2 for s in spectra]
    ranks = [min_rank] * n
    remaining = total_budget - floor_total

    def marginal(li: int, r: int) -> float:
        """Score for giving layer ``li`` its (r+1)-th rank (0-indexed sv at r)."""
        v = float(sq[li][r])
        if score == "raw":
            return v
        tail = float(sq[li][r:].sum())  # Σ_{m≥r} σ²_m (remaining, incl. current)
        return v / tail if tail > 0 else 0.0

    # Max-heap (via negated score) keyed on each layer's next-rank marginal.
    heap = [(-marginal(li, ranks[li]), li) for li in range(n) if ranks[li] < caps[li]]
    heapq.heapify(heap)
    for _ in range(remaining):
        if not heap:
            break
        _, li = heapq.heappop(heap)
        ranks[li] += 1
        if ranks[li] < caps[li]:
            heapq.heappush(heap, (-marginal(li, ranks[li]), li))
    return ranks


def compute_adaptive_ranks(
    loaded: LoadedModel,
    *,
    covariances: list[torch.Tensor],
    d_rope: int,
    total_budget: int,
    min_rank: int = 1,
    score: str = "raw",
    ridge_lambda: float = 1e-6,
    ridge_max_iters: int = 6,
    progress: bool = False,
) -> list[int]:
    """Per-layer ranks from a total budget, via activation-weighted spectra.

    Runs one whitened-SVD spectrum pass per layer (same ridge schedule as the
    factorization) and water-fills ``total_budget`` across them. Convenience
    wrapper over :func:`allocate_adaptive_ranks`; ``score`` selects the greedy
    marginal-value metric ('raw' or 'normalized').
    """
    cfg = loaded.config
    perms = _rope_perms(cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads, d_rope)
    spectra: list[torch.Tensor] = []
    for i in range(cfg.num_hidden_layers):
        W_stack, *_ = _w_stack_for_layer(loaded, i, perms, d_rope)
        s = weighted_singular_values(
            W_stack, covariances[i],
            ridge_lambda=ridge_lambda, ridge_max_iters=ridge_max_iters,
        )
        spectra.append(s)
        if progress:
            print(f"[convert]   spectrum {i+1}/{cfg.num_hidden_layers}", flush=True)
    return allocate_adaptive_ranks(spectra, total_budget, min_rank=min_rank, score=score)


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
    ranks: list[int] | None = None,
    fisher: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None,
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
        Used for every layer unless ``ranks`` is given. Also stored in meta
        as the nominal/average rank for reporting.
    d_rope
        Per-head RoPE-subspace width. ``0 ≤ d_rope ≤ head_dim``.
    ranks
        Optional per-layer ranks (length ``num_layers``). When provided,
        layer ``i`` is factored to ``ranks[i]`` and the artifact is a
        per-layer adaptive-rank conversion; ``rank`` is then only metadata.
        See :func:`compute_adaptive_ranks`.
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
    if ranks is not None:
        if len(ranks) != num_layers:
            raise ValueError(
                f"got {len(ranks)} per-layer ranks for a {num_layers}-layer model"
            )
        if any(not (1 <= r <= hidden_size) for r in ranks):
            raise ValueError(f"every per-layer rank must be in [1, {hidden_size}]")
    if len(covariances) != num_layers:
        raise ValueError(
            f"got {len(covariances)} covariance matrices for a {num_layers}-layer model"
        )
    if calibration_meta is not None and target_model_id is not None:
        calib_id = calibration_meta.get("model_id")
        if calib_id is None:
            logger.warning(
                "calibration artifact has no model_id; skipping match check "
                "against target model_id=%r",
                target_model_id,
            )
        elif calib_id != target_model_id:
            raise ValueError(
                f"calibration model_id={calib_id!r} does not match target "
                f"model_id={target_model_id!r}"
            )

    d_nope = head_dim - d_rope

    rope_theta = float(
        getattr(cfg, "rope_parameters", None) and cfg.rope_parameters.get("rope_theta")
        or getattr(cfg, "rope_theta", 10000.0)
    )

    # Permutation that re-orders each head so the rope subspace (first
    # d_rope/2 original RoPE pairs) lives at the *end*. We apply it to W_q,
    # W_K, q_norm.weight, k_norm.weight at conversion time so the runtime
    # MLAttention sees a contiguous "rope = last d_rope dims" layout. V and
    # o_proj keep their original ordering — V isn't RoPE'd, and the
    # attention output (softmax(Q·Kᵀ)·V) is in V's ordering, so o_proj
    # consumes it unchanged.
    # After permutation, the first d_nope rows per head are nope, last
    # d_rope rows per head are rope — so the existing contiguous slicing
    # is correct *on the permuted matrix*. Shared with the spectrum pass.
    perms = _rope_perms(head_dim, num_heads, num_kv_heads, d_rope)
    q_row_perm = perms["q_row_perm"]
    head_perm = perms["head_perm"]

    out_state: dict[str, torch.Tensor] = {}
    diagnostics: list[dict] = []

    t0_total = time.time()
    for i in range(num_layers):
        prefix = f"layers.{i}.attn."
        layer_rank = ranks[i] if ranks is not None else rank
        C = covariances[i]
        if C.shape != (hidden_size, hidden_size):
            raise ValueError(
                f"layer {i}: covariance shape {tuple(C.shape)} != ({hidden_size}, {hidden_size})"
            )

        # Stacked nope-K + V rows (shared construction with the spectrum pass).
        W_stack, W_K_nope, W_kr, W_V = _w_stack_for_layer(loaded, i, perms, d_rope)

        # Optional two-sided Fisher weighting on the output rows.
        row_weight = None
        if fisher is not None:
            row_weight = _fisher_row_weight(fisher[0][i], fisher[1][i], perms, d_rope)

        # Joint SVD over the stack so K-nope and V share W_dkv.
        A, B = activation_aware_factor(
            W_stack, C, rank=layer_rank, row_weight=row_weight,
            ridge_lambda=ridge_lambda, ridge_max_iters=ridge_max_iters,
        )
        # Diagnostic: how much C-weighted error is being thrown away?
        discarded = discarded_singular_energy(
            W_stack, C, rank=layer_rank,
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

        # Q, q_norm, k_norm get the head permutation; o.weight stays as-is.
        W_q_perm = loaded.state[prefix + "q.weight"][q_row_perm, :]
        q_norm_perm = loaded.state[prefix + "q_norm.weight"][head_perm]
        k_norm_perm = loaded.state[prefix + "k_norm.weight"][head_perm]

        out_state[prefix + "q.weight"] = W_q_perm.to(factor_dtype).contiguous()
        out_state[prefix + "o.weight"] = loaded.state[prefix + "o.weight"].to(factor_dtype).clone()
        out_state[prefix + "q_norm.weight"] = q_norm_perm.to(factor_dtype).contiguous()
        out_state[prefix + "k_norm.weight"] = k_norm_perm.to(factor_dtype).contiguous()
        out_state[prefix + "dkv.weight"] = B.to(factor_dtype).contiguous()
        out_state[prefix + "uv.weight"] = A_V.to(factor_dtype).contiguous()
        if A_K_nope is not None:
            out_state[prefix + "uk_nope.weight"] = A_K_nope.to(factor_dtype).contiguous()
        if W_kr is not None:
            out_state[prefix + "kr.weight"] = W_kr.to(factor_dtype).clone().contiguous()

        diagnostics.append({
            "layer": i,
            "rank": layer_rank,
            "discarded_energy": float(discarded),
            "ridge_lambda": ridge_lambda,
        })
        if progress:
            elapsed = time.time() - t0_total
            print(f"[convert]   layer {i+1}/{num_layers} done "
                  f"(discarded_energy={discarded:.3e}, t+{elapsed:.1f}s)", flush=True)

    wall_seconds = time.time() - t0_total

    per_layer_ranks = list(ranks) if ranks is not None else [rank] * num_layers
    meta: dict = {
        "model_id": target_model_id,
        "rank": rank,
        "ranks": per_layer_ranks,
        "adaptive_rank": ranks is not None,
        "fisher_weighted": fisher is not None,
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
        "rope_convention": "original-pairs",
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
    ap.add_argument(
        "--rank", type=int, required=True,
        help="target latent rank. With --adaptive-rank this is the per-layer "
             "*average* (total budget = rank × num_layers).",
    )
    ap.add_argument("--d-rope", type=int, required=True, help="per-head RoPE-subspace width")
    ap.add_argument(
        "--adaptive-rank", action="store_true",
        help="allocate the rank budget per layer by activation-weighted singular "
             "spectra instead of a uniform rank (same total cache size).",
    )
    ap.add_argument(
        "--min-rank", type=int, default=8,
        help="floor on per-layer rank under --adaptive-rank",
    )
    ap.add_argument(
        "--rank-score", default="raw", choices=["raw", "normalized"],
        help="adaptive-rank marginal-value metric: 'raw' (σ², pooled) or "
             "'normalized' (σ²/tail-energy, CARE-style; prevents early-layer starvation)",
    )
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
    ap.add_argument(
        "--fisher", default=None,
        help="path to a per-layer output-Fisher file (keys fisher_k, fisher_v) "
             "from collect_fisher.py. Adds two-sided (input-covariance × "
             "output-gradient) Fisher weighting to the SVD init.",
    )
    args = ap.parse_args(argv)

    factor_dtype = getattr(torch, args.dtype)
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = _default_out_path(args.model_id, args.rank, args.d_rope)
        if args.adaptive_rank:
            out_path = out_path.with_name(out_path.stem + "_adaptive.pt")
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

    fisher = None
    if args.fisher:
        print(f"[convert] loading output-Fisher {args.fisher}", flush=True)
        fdat = torch.load(args.fisher, map_location="cpu", weights_only=False)
        fisher = (fdat["fisher_k"], fdat["fisher_v"])
        print(f"[convert]   fisher: {len(fisher[0])} layers (two-sided weighting on)", flush=True)

    ranks = None
    if args.adaptive_rank:
        budget = args.rank * loaded.config.num_hidden_layers
        print(f"[convert] computing adaptive ranks (budget={budget} = {args.rank}×"
              f"{loaded.config.num_hidden_layers} layers, min_rank={args.min_rank}, "
              f"score={args.rank_score})", flush=True)
        ranks = compute_adaptive_ranks(
            loaded, covariances=covariances, d_rope=args.d_rope,
            total_budget=budget, min_rank=args.min_rank, score=args.rank_score,
            ridge_lambda=args.ridge_lambda, progress=True,
        )
        print(f"[convert]   ranks: min={min(ranks)} max={max(ranks)} "
              f"mean={sum(ranks)/len(ranks):.1f}", flush=True)

    print(f"[convert] running per-layer joint SVD (rank={args.rank}, d_rope={args.d_rope}, "
          f"adaptive={args.adaptive_rank})", flush=True)
    artifact = convert_loaded_to_mla(
        loaded,
        covariances=covariances,
        rank=args.rank,
        d_rope=args.d_rope,
        ranks=ranks,
        fisher=fisher,
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
