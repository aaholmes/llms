"""Post-hoc swap of GQA Attention → MLAttention from a conversion artifact.

Mirror of ``src/kernels/swap.py``: walk ``model.layers``, replace each
``block.attn`` with an ``MLAttention`` configured per the artifact's meta
and weighted from the artifact's per-layer state-dict slice. Leaves
``src/engine/model.py`` untouched.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from engine.mla import MLAKVCache, MLAttention
from engine.model import Qwen3Model
from engine.weights import LoadedModel, load_weights


def apply_mla(model: Qwen3Model, artifact: dict) -> Qwen3Model:
    """Replace every ``block.attn`` with an ``MLAttention`` from ``artifact``.

    Mutates ``model`` in place; returns it for chaining (same convention as
    ``apply_triton_kernels``). Idempotent: a second call on an already-swapped
    model is a no-op per layer.
    """
    meta = artifact["meta"]
    state = artifact["state"]
    convention = meta.get("rope_convention")
    if convention != "original-pairs":
        raise ValueError(
            f"artifact rope_convention={convention!r} is not supported; only "
            f"'original-pairs' is accepted. Re-convert the model with the "
            f"current `python -m mla.convert ...` to upgrade."
        )
    for i, block in enumerate(model.layers):
        if isinstance(block.attn, MLAttention):
            continue
        mla = MLAttention(
            hidden_size=meta["hidden_size"],
            num_heads=meta["num_heads"],
            num_kv_heads=meta["num_kv_heads"],
            head_dim=meta["head_dim"],
            rms_eps=meta["rms_eps"],
            rank=meta["rank"],
            d_rope=meta["d_rope"],
            max_position_embeddings=meta["max_position_embeddings"],
            rope_theta=meta["rope_theta"],
            qk_norm_mode=meta.get("qk_norm_mode", "single"),
        )
        prefix = f"layers.{i}.attn."
        with torch.no_grad():
            mla.q.weight.copy_(state[prefix + "q.weight"])
            mla.dkv.weight.copy_(state[prefix + "dkv.weight"])
            if mla.uk_nope is not None:
                mla.uk_nope.weight.copy_(state[prefix + "uk_nope.weight"])
            mla.uv.weight.copy_(state[prefix + "uv.weight"])
            if mla.kr is not None:
                mla.kr.weight.copy_(state[prefix + "kr.weight"])
            mla.o.weight.copy_(state[prefix + "o.weight"])
            mla.q_norm.weight.copy_(state[prefix + "q_norm.weight"])
            mla.k_norm.weight.copy_(state[prefix + "k_norm.weight"])
        # Match the surrounding model's dtype/device.
        ref_dtype = next(model.parameters()).dtype
        ref_device = next(model.parameters()).device
        mla = mla.to(dtype=ref_dtype, device=ref_device)
        block.attn = mla
    return model


def alloc_mla_cache(
    model: Qwen3Model,
    max_seq_len: int,
    *,
    max_batch: int = 1,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> MLAKVCache:
    """Allocate an MLAKVCache shaped for a model that's already had MLAttention swapped in."""
    first_attn: MLAttention = next(
        m for m in model.modules() if isinstance(m, MLAttention)
    )
    if dtype is None:
        dtype = next(model.parameters()).dtype
    if device is None:
        device = next(model.parameters()).device
    return MLAKVCache.alloc(
        num_layers=len(model.layers),
        max_batch=max_batch,
        num_kv_heads=first_attn.num_kv_heads,
        max_seq_len=max_seq_len,
        rank=first_attn.rank,
        d_rope=first_attn.d_rope,
        dtype=dtype,
        device=device,
    )


def load_mla_model(
    model_id: str,
    artifact_path: str | Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
) -> Qwen3Model:
    """End-to-end loader: HF weights → Qwen3Model → apply_mla(artifact)."""
    loaded = load_weights(model_id, dtype=dtype, device=device)
    model = Qwen3Model.from_loaded(loaded).to(dtype=dtype, device=device).eval()
    artifact = torch.load(Path(artifact_path), map_location=device, weights_only=False)
    apply_mla(model, artifact)
    return model
