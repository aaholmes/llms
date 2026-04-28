"""Custom Triton kernels for Stage A.5 hot-path optimization."""

from .fused_ffn import TritonFFN, triton_fused_gate_up_silu
from .fused_qkv import TritonAttention, triton_fused_qkv
from .swap import apply_triton_kernels, prewarm_triton_kernels

__all__ = [
    "TritonAttention",
    "TritonFFN",
    "apply_triton_kernels",
    "prewarm_triton_kernels",
    "triton_fused_gate_up_silu",
    "triton_fused_qkv",
]
