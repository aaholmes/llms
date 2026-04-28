"""In-place swap of engine modules with their Triton-kernel counterparts.

Keeps ``src/engine/model.py`` untouched — same hygiene as the planned
``MLAttention`` for Stage B, where the engine stays agnostic and the kernel
package owns module replacement.

Usage:
    model = Qwen3Model.from_loaded(loaded).to(device).eval()
    apply_triton_kernels(model)              # in-place; also returned
    prewarm_triton_kernels(model)            # populate autotune cache
"""

from __future__ import annotations

import torch
from torch import nn

from .fused_ffn import TritonFFN
from .fused_qkv import TritonAttention


def apply_triton_kernels(
    model: nn.Module,
    *,
    fused_ffn: bool = True,
    fused_qkv: bool = True,
) -> nn.Module:
    """Replace ``DecoderBlock.ffn`` / ``DecoderBlock.attn`` with the
    Triton-backed counterparts on every layer.

    Args:
        model: a ``Qwen3Model`` (must expose ``model.layers`` of decoder
            blocks each with ``ffn`` and ``attn`` attributes).
        fused_ffn: if False, the FFN swap is skipped.
        fused_qkv: if False, the attention swap is skipped.

    Returns:
        The same model, mutated in place. Returning is a convenience for
        chaining.
    """
    for block in model.layers:
        if fused_ffn and not isinstance(block.ffn, TritonFFN):
            block.ffn = TritonFFN.from_eager(block.ffn)
        if fused_qkv and not isinstance(block.attn, TritonAttention):
            block.attn = TritonAttention.from_eager(block.attn)
    return model


def prewarm_triton_kernels(
    model: nn.Module,
    *,
    m_values: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8),
) -> None:
    """Fire each Triton-backed module once at every M in ``m_values`` so the
    autotune cache is populated *before* any timed inference path enters it.

    Without this, demo / profile / benchmark scripts pay first-call autotune
    compile cost inside their timed regions — visible as a regression on
    spec-decode (which produces M ∈ {1..K+1} variants) even when the kernel
    itself is faster on a microbench. Standalone helper so tests can verify
    prewarm doesn't perturb forward outputs.

    Args:
        model: a model with ``.layers``, after ``apply_triton_kernels`` has
            run on it.
        m_values: M values to prewarm. Default covers spec-decode K up to 7
            with one pending token; extend if running larger K.
    """
    # FFN modules: bucket by (N, K) and fire each unique shape once per M.
    triton_ffns: list[TritonFFN] = [
        m for m in model.modules() if isinstance(m, TritonFFN)
    ]
    ffn_by_shape: dict[tuple[int, int], TritonFFN] = {}
    for ffn in triton_ffns:
        N, K = ffn.gate.weight.shape
        ffn_by_shape.setdefault((N, K), ffn)

    # Attention modules: only the QKV matmul autotunes; bucket by
    # (n_total, K) and call ``triton_fused_qkv`` directly so we don't pay
    # the rest of the attention forward (RoPE / SDPA / KV-cache / o-proj).
    from .fused_qkv import triton_fused_qkv

    triton_attns: list[TritonAttention] = [
        m for m in model.modules() if isinstance(m, TritonAttention)
    ]
    attn_by_shape: dict[tuple[int, int], TritonAttention] = {}
    for attn in triton_attns:
        n_total, K = attn.W_qkv.shape
        attn_by_shape.setdefault((n_total, K), attn)

    if not triton_ffns and not triton_attns:
        return

    with torch.inference_mode():
        for (N, K), ffn in ffn_by_shape.items():
            device = ffn.gate.weight.device
            dtype = ffn.gate.weight.dtype
            for M in m_values:
                x = torch.zeros((1, M, K), dtype=dtype, device=device)
                _ = ffn(x)
        for (n_total, K), attn in attn_by_shape.items():
            device = attn.W_qkv.device
            dtype = attn.W_qkv.dtype
            for M in m_values:
                x = torch.zeros((1, M, K), dtype=dtype, device=device)
                _ = triton_fused_qkv(
                    x, attn.W_qkv, n_q=attn.n_q, n_k=attn.n_k
                )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
