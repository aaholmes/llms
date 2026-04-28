"""Fused gate-up-silu Triton kernel for the FFN's SwiGLU activation.

Computes ``silu(x @ Wg.T) * (x @ Wu.T)`` in one launch — replaces the two
separate ``nn.Linear`` calls inside ``FFN.forward`` (``src/engine/model.py``).

The third FFN matmul (``down``) stays as a regular ``nn.Linear``: fusing it
would force ugly block-shape constraints across two different N axes.

TDD ladder + design notes live in
``~/.claude/plans/ok-make-a-detailed-keen-clover.md``.
"""

from __future__ import annotations

import torch
from torch import nn
import triton
import triton.language as tl


# Autotune configs. Sized for BF16 production (the deployment dtype) within
# the 100 KB shared-memory budget on SM 12.0. The shared-mem cost per config
# is approximately ``num_stages * (BLOCK_M*BLOCK_K + 2*BLOCK_K*BLOCK_N) *
# bf16_bytes``; values below all stay under ~70 KB.
#
# All use BLOCK_M=16 because Triton's tl.dot needs M-tile >= 16; for decode
# (M=1..8) the row mask drops the wasted rows.
_AUTOTUNE_CONFIGS = [
    triton.Config(  # ~70 KB
        {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64},
        num_warps=4, num_stages=2,
    ),
    triton.Config(  # ~52 KB — more pipeline stages, smaller K
        {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32},
        num_warps=4, num_stages=3,
    ),
    triton.Config(  # ~55 KB — more N parallelism per program
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64},
        num_warps=4, num_stages=3,
    ),
    triton.Config(  # ~67 KB — wider N tile for cases where SM count limits
        {"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 32},
        num_warps=8, num_stages=2,
    ),
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _fused_gate_up_silu_kernel(
    # Pointers to inputs / output
    X_ptr,      # (M, K)
    Wg_ptr,     # (N, K)  — nn.Linear weight layout (out, in)
    Wu_ptr,     # (N, K)
    Y_ptr,      # (M, N)
    # Sizes
    M, N, K,
    # Strides
    stride_xm, stride_xk,
    stride_wg_n, stride_wg_k,
    stride_wu_n, stride_wu_k,
    stride_ym, stride_yn,
    # Block sizes (filled in by autotune)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """One program computes Y[m_tile, n_tile].

    For each output tile we stream X[m_tile, :K] once and reuse it across the
    two matmuls. Both Wg and Wu loads are issued before either dot so the
    compiler is free to pipeline them — that's the launch-and-bandwidth point
    of the fusion (cuBLAS would issue two separate kernels and not share x).
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # X tile read as (BLOCK_M, BLOCK_K).
    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # Wg/Wu read as (BLOCK_K, BLOCK_N) — i.e. transposed view of the (N, K)
    # weight, since tl.dot wants a (K, N) right-hand side.
    wg_ptrs = Wg_ptr + offs_k[:, None] * stride_wg_k + offs_n[None, :] * stride_wg_n
    wu_ptrs = Wu_ptr + offs_k[:, None] * stride_wu_k + offs_n[None, :] * stride_wu_n

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        mask_k = (k0 + offs_k) < K

        x = tl.load(
            x_ptrs + k0 * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        wg = tl.load(
            wg_ptrs + k0 * stride_wg_k,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        wu = tl.load(
            wu_ptrs + k0 * stride_wu_k,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )

        acc_g = tl.dot(x, wg, acc_g, allow_tf32=ALLOW_TF32)
        acc_u = tl.dot(x, wu, acc_u, allow_tf32=ALLOW_TF32)

    # silu(g) * u, where silu(z) = z * sigmoid(z)
    g = acc_g * tl.sigmoid(acc_g)
    y = g * acc_u

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, y, mask=mask_m[:, None] & mask_n[None, :])


def triton_fused_gate_up_silu(
    x: torch.Tensor, Wg: torch.Tensor, Wu: torch.Tensor
) -> torch.Tensor:
    """Compute ``silu(x @ Wg.T) * (x @ Wu.T)`` via a single Triton kernel.

    Args:
        x: shape ``(..., K)``.
        Wg, Wu: shape ``(N, K)``, matching ``nn.Linear.weight`` layout.

    Returns:
        Tensor of shape ``(..., N)`` and the same dtype/device as ``x``.
    """
    if Wg.shape != Wu.shape:
        raise ValueError(
            f"Wg shape {tuple(Wg.shape)} != Wu shape {tuple(Wu.shape)}"
        )
    if Wg.dim() != 2:
        raise ValueError(f"Wg must be 2D (N, K), got shape {tuple(Wg.shape)}")
    if x.shape[-1] != Wg.shape[1]:
        raise ValueError(
            f"x last dim {x.shape[-1]} != Wg.shape[1] {Wg.shape[1]}"
        )
    if x.device != Wg.device or x.device != Wu.device:
        raise ValueError(
            f"x ({x.device}), Wg ({Wg.device}), Wu ({Wu.device}) "
            "must be on the same device"
        )

    N, K = Wg.shape
    leading_shape = x.shape[:-1]
    x2 = x.reshape(-1, K)
    M = x2.shape[0]

    y = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # Grid is a callable because BLOCK_M/BLOCK_N are picked by autotune.
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    # allow_tf32=False keeps FP32 inputs honest for the unit-test path; on
    # BF16 inputs this flag has no effect (the dot already accumulates fp32
    # from bf16 tensor cores).
    _fused_gate_up_silu_kernel[grid](
        x2, Wg, Wu, y,
        M, N, K,
        x2.stride(0), x2.stride(1),
        Wg.stride(0), Wg.stride(1),
        Wu.stride(0), Wu.stride(1),
        y.stride(0), y.stride(1),
        ALLOW_TF32=False,
    )

    return y.reshape(*leading_shape, N)


class TritonFFN(nn.Module):
    """Drop-in replacement for ``engine.model.FFN`` that uses the fused
    Triton kernel for the gate-up-silu stage. The down projection stays as
    a plain ``nn.Linear``.

    Parameter names (``gate.weight``, ``up.weight``, ``down.weight``) match
    ``FFN`` exactly so checkpoints load unchanged via ``load_state_dict``.
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(
            triton_fused_gate_up_silu(x, self.gate.weight, self.up.weight)
        )

    @classmethod
    def from_eager(cls, eager: nn.Module) -> "TritonFFN":
        """Wrap an existing ``engine.model.FFN`` (or any module with
        ``gate``/``up``/``down`` ``nn.Linear`` submodules), reusing the
        same parameter tensors — no copy, no extra memory."""
        out = cls.__new__(cls)
        nn.Module.__init__(out)
        out.gate = eager.gate
        out.up = eager.up
        out.down = eager.down
        return out
