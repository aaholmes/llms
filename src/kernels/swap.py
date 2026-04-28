"""In-place swap of engine modules with their Triton-kernel counterparts.

Keeps ``src/engine/model.py`` untouched — same hygiene as the planned
``MLAttention`` for Stage B, where the engine stays agnostic and the kernel
package owns module replacement.

Usage:
    model = Qwen3Model.from_loaded(loaded).to(device).eval()
    apply_triton_kernels(model)              # in-place; also returned
"""

from __future__ import annotations

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
