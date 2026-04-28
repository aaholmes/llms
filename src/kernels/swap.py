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


def apply_triton_kernels(model: nn.Module, *, fused_ffn: bool = True) -> nn.Module:
    """Replace ``DecoderBlock.ffn`` with ``TritonFFN`` on every layer.

    Args:
        model: a ``Qwen3Model`` (must expose ``model.layers`` of decoder
            blocks each with an ``ffn`` attribute).
        fused_ffn: if False, the FFN swap is skipped (kept for symmetry
            with future ``fused_qkv`` / ``fused_attn`` flags).

    Returns:
        The same model, mutated in place. Returning is a convenience for
        chaining.
    """
    if fused_ffn:
        for block in model.layers:
            if isinstance(block.ffn, TritonFFN):
                continue  # idempotent
            block.ffn = TritonFFN.from_eager(block.ffn)
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
    triton_ffns: list[TritonFFN] = [
        m for m in model.modules() if isinstance(m, TritonFFN)
    ]
    if not triton_ffns:
        return

    # One representative module per unique (N, K) — autotune keys on shape,
    # not on identity, so layers sharing dims (the typical case) only need
    # to fire once per M.
    by_shape: dict[tuple[int, int], TritonFFN] = {}
    for ffn in triton_ffns:
        N, K = ffn.gate.weight.shape
        by_shape.setdefault((N, K), ffn)

    with torch.inference_mode():
        for (N, K), ffn in by_shape.items():
            device = ffn.gate.weight.device
            dtype = ffn.gate.weight.dtype
            for M in m_values:
                # Zeros are fine — autotune keys on shape; values don't matter.
                x = torch.zeros((1, M, K), dtype=dtype, device=device)
                _ = ffn(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
