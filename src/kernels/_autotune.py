"""Shared Triton autotune config list for the Stage A.5 GEMV-shaped kernels.

Both ``fused_ffn`` and ``fused_qkv`` solve a (M small, N large, K large)
matmul tile and have the same shared-memory budget on SM 12.0. Keeping the
config list here avoids drift if we tune one kernel and forget the other.
"""

from __future__ import annotations

import triton

# Sized for BF16 production within the 100 KB shared-memory budget on SM 12.0.
# Approximate per-config shared-mem cost is
# ``num_stages * (BLOCK_M*BLOCK_K + 2*BLOCK_K*BLOCK_N) * bf16_bytes`` for FFN
# (which uses two weight tiles); QKV uses one weight tile so its budget is
# even more comfortable. All entries below stay under ~70 KB on the FFN path,
# under ~35 KB on the QKV path.
#
# All use BLOCK_M=16 because Triton's tl.dot needs M-tile >= 16; for decode
# (M=1..8) the row mask drops the wasted rows.
MATMUL_AUTOTUNE_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64},
        num_warps=4, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32},
        num_warps=4, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64},
        num_warps=4, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 32},
        num_warps=8, num_stages=2,
    ),
]
