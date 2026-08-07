"""Symmetric weight-only quantization for the engine's ``nn.Linear`` layers.

Scope note, stated up front so the numbers aren't over-read: this is a
**quality and footprint** instrument, not a speed one. Weights are stored
quantized and dequantized back to the activation dtype before the matmul, so
there is no int8 GEMM here and no throughput win on any backend. That is the
right trade for the question this repo asks — *does the TVD corrector repair a
quantized block the way it repairs MLA and a small draft?* — which is answered
by the output distribution, not the clock. A real low-precision GEMM is a
separate, later, GPU-side concern.

Granularity follows the usual ladder: symmetric, per-output-channel by
default, optionally per-group along the input dimension (``group_size``),
which is what makes 4-bit tolerable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# int8 storage for every width. 4-bit values live in an int8 tensor unpacked;
# see ``quant_memory_report`` for how the packed-equivalent footprint is
# reported without pretending the storage here is packed.
SUPPORTED_BITS = (4, 8)


def _qmax(n_bits: int) -> int:
    if n_bits not in SUPPORTED_BITS:
        raise ValueError(f"n_bits must be one of {SUPPORTED_BITS}, got {n_bits}")
    return 2 ** (n_bits - 1) - 1


def quantize_weight(
    weight: torch.Tensor,
    *,
    n_bits: int = 8,
    group_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetrically quantize ``weight`` of shape ``(out_features, in_features)``.

    Args:
        weight: the dense weight to quantize.
        n_bits: 8 or 4.
        group_size: if given, quantize each contiguous block of ``group_size``
            input columns with its own scale; ``in_features`` must be
            divisible by it. If None, one scale per output channel.

    Returns:
        ``(q, scale)`` where ``q`` is int8-stored codes with ``weight``'s
        shape and ``scale`` is float32 of shape ``(out_features, n_groups)``.
        Dequantization is exact-in-form: ``dequantize(q, scale)  ≈ weight``.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(weight.shape)}")
    qmax = _qmax(n_bits)
    out_features, in_features = weight.shape

    if group_size is None:
        n_groups, gs = 1, in_features
    else:
        if group_size <= 0 or in_features % group_size != 0:
            raise ValueError(
                f"group_size={group_size} does not divide in_features={in_features}"
            )
        gs = group_size
        n_groups = in_features // gs

    # Compute in fp32 regardless of the incoming dtype: a bf16 amax/round loses
    # enough mantissa to visibly bias the codes.
    w = weight.detach().to(torch.float32).reshape(out_features, n_groups, gs)
    amax = w.abs().amax(dim=-1)
    # A dead (all-zero) group would divide by zero; clamp to tiny and let the
    # codes come out zero, which round-trips to the correct zero weight.
    scale = (amax / qmax).clamp(min=torch.finfo(torch.float32).tiny)
    q = torch.round(w / scale.unsqueeze(-1)).clamp(-qmax, qmax)
    return q.reshape(out_features, in_features).to(torch.int8), scale


def dequantize(
    q: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Inverse of :func:`quantize_weight`, back to a dense ``dtype`` weight."""
    out_features, in_features = q.shape
    n_groups = scale.shape[-1]
    gs = in_features // n_groups
    w = q.reshape(out_features, n_groups, gs).to(torch.float32) * scale.unsqueeze(-1)
    return w.reshape(out_features, in_features).to(dtype)


class QuantLinear(nn.Module):
    """Drop-in replacement for a bias-free ``nn.Linear`` with quantized weights.

    Exposes ``.weight`` as a *property* returning the dequantized dense weight,
    so the introspection the rest of the repo does (``block.attn.q.weight``,
    kernel prewarm shape probing, MLA calibration hooks) keeps working against
    a quantized model without special-casing.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        n_bits: int = 8,
        group_size: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_bits = n_bits
        self.group_size = group_size
        self.compute_dtype = dtype
        n_groups = 1 if group_size is None else in_features // group_size
        # Buffers, not Parameters: these are frozen. A healing finetune adds a
        # LoRA *alongside* this layer rather than training the codes.
        self.register_buffer("qweight", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(out_features, n_groups, dtype=torch.float32))

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        n_bits: int = 8,
        group_size: int | None = None,
    ) -> "QuantLinear":
        """Quantize an existing dense ``nn.Linear`` (bias-free, as Qwen3 is)."""
        if linear.bias is not None:
            raise ValueError("QuantLinear only supports bias-free Linear layers")
        out_features, in_features = linear.weight.shape
        mod = cls(
            in_features,
            out_features,
            n_bits=n_bits,
            group_size=group_size,
            dtype=linear.weight.dtype,
        )
        q, scale = quantize_weight(linear.weight, n_bits=n_bits, group_size=group_size)
        mod.qweight.copy_(q)
        mod.scale.copy_(scale)
        return mod.to(device=linear.weight.device)

    @property
    def weight(self) -> torch.Tensor:
        """Dequantized dense weight — recomputed per access, never cached."""
        return dequantize(self.qweight, self.scale, dtype=self.compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"n_bits={self.n_bits}, group_size={self.group_size}, bias=False"
        )
