"""Fused QKV-projection Triton kernel for the attention path.

Replaces the three separate ``self.q(x), self.k(x), self.v(x)`` calls at the
top of ``Attention.forward`` (``src/engine/attention.py:114-116``) with a
single Triton matmul against a pre-concatenated ``W_qkv`` weight, then
splits the output back into Q, K, V.

The kernel itself is a generic GEMV-shaped matmul. The "fusion" lives at
the module level: ``TritonAttention.from_eager`` ``torch.cat``'s the three
``nn.Linear`` weights at construction time and drops the originals, so the
runtime forward sees one launch instead of three.

QK-norm and RoPE stay downstream — *not* fused into the kernel — to keep
the math contract a clean pure function and the TDD ladder simple.

TDD ladder + design notes live in
``~/.claude/plans/ok-make-a-detailed-keen-clover.md``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
import triton
import triton.language as tl

from ._autotune import MATMUL_AUTOTUNE_CONFIGS


@triton.autotune(configs=MATMUL_AUTOTUNE_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _qkv_matmul_kernel(
    # Pointers
    X_ptr,      # (M, K)
    W_ptr,      # (N, K) — pre-concatenated [Wq; Wk; Wv]
    Y_ptr,      # (M, N)
    # Sizes
    M, N, K,
    # Strides
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    # Block sizes (filled in by autotune)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """One program computes Y[m_tile, n_tile] = X[m_tile, :] @ W[n_tile, :].T.

    Structurally the same K-loop as the FFN kernel but with a single weight
    pointer / accumulator and no silu / elementwise mul on the way out.
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
    # W read as (BLOCK_K, BLOCK_N) — transposed view of the (N, K) weight,
    # since tl.dot wants a (K, N) right-hand side.
    w_ptrs = W_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        mask_k = (k0 + offs_k) < K
        x = tl.load(
            x_ptrs + k0 * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ptrs + k0 * stride_wk,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        acc = tl.dot(x, w, acc, allow_tf32=ALLOW_TF32)

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_fused_qkv(
    x: torch.Tensor,
    W_qkv: torch.Tensor,
    *,
    n_q: int,
    n_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute ``(q, k, v) = split(x @ W_qkv.T)`` via a single Triton kernel.

    Args:
        x: shape ``(..., K)``.
        W_qkv: shape ``(n_q + 2*n_k, K)``, the row-stacked Q/K/V weights
            (n_v == n_k for GQA).
        n_q: number of Q-projection rows in W_qkv.
        n_k: number of K-projection rows (also equals V-projection rows).

    Returns:
        ``(q, k, v)`` of shapes ``(..., n_q)``, ``(..., n_k)``, ``(..., n_k)``,
        same dtype/device as ``x``. The split is a zero-copy view.
    """
    if W_qkv.dim() != 2:
        raise ValueError(
            f"W_qkv must be 2D (n_q+2*n_k, K), got shape {tuple(W_qkv.shape)}"
        )
    n_total, K = W_qkv.shape
    if n_q + 2 * n_k != n_total:
        raise ValueError(
            f"n_q + 2*n_k ({n_q} + 2*{n_k} = {n_q + 2 * n_k}) "
            f"!= W_qkv.shape[0] ({n_total})"
        )
    if x.shape[-1] != K:
        raise ValueError(
            f"x last dim {x.shape[-1]} != W_qkv.shape[1] {K}"
        )
    if x.device != W_qkv.device:
        raise ValueError(
            f"x ({x.device}) and W_qkv ({W_qkv.device}) must be on the same device"
        )

    leading_shape = x.shape[:-1]
    x2 = x.reshape(-1, K)
    M = x2.shape[0]

    y = torch.empty((M, n_total), dtype=x.dtype, device=x.device)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(n_total, meta["BLOCK_N"]),
    )

    _qkv_matmul_kernel[grid](
        x2, W_qkv, y,
        M, n_total, K,
        x2.stride(0), x2.stride(1),
        W_qkv.stride(0), W_qkv.stride(1),
        y.stride(0), y.stride(1),
        ALLOW_TF32=False,
    )

    y = y.reshape(*leading_shape, n_total)
    q, k, v = y.split([n_q, n_k, n_k], dim=-1)
    return q, k, v


class TritonAttention(nn.Module):
    """Drop-in replacement for ``engine.attention.Attention`` that uses the
    fused QKV Triton kernel.

    Differences from ``Attention``:
    - Stores ``self.W_qkv`` (a single concatenated parameter) instead of
      separate ``self.q``, ``self.k``, ``self.v`` ``nn.Linear`` modules.
      This means HF state-dict keys (``q_proj.weight`` etc.) cannot be
      loaded directly into a ``TritonAttention``; load into an eager
      ``Attention`` first, then call ``TritonAttention.from_eager`` to
      build the kerneled version.
    - QK-norm, RoPE, KV-cache write, and SDPA are unchanged — those stay on
      eager paths because they're not the bottleneck.

    The forward body mirrors lines 118-149 of ``src/engine/attention.py``;
    if that file evolves, run ``test_attention_module_logits_match`` to
    detect drift.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_eps: float,
    ) -> None:
        super().__init__()
        from engine.attention import RMSNorm

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.n_q = num_heads * head_dim
        self.n_k = num_kv_heads * head_dim
        n_total = self.n_q + 2 * self.n_k
        self.W_qkv = nn.Parameter(
            torch.empty(n_total, hidden_size), requires_grad=False
        )
        self.o = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=rms_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_eps)

    @classmethod
    def from_eager(cls, eager: nn.Module) -> "TritonAttention":
        """Build a ``TritonAttention`` from an eager ``Attention`` whose
        weights are already loaded.

        Pre-concatenates ``[Wq; Wk; Wv]`` into a single parameter and reuses
        the eager module's ``o``, ``q_norm``, ``k_norm`` directly (no copy).
        After this call the eager q/k/v ``nn.Linear`` modules are
        unreferenced; their GPU memory is freed by the next allocator pass.
        """
        out = cls.__new__(cls)
        nn.Module.__init__(out)

        out.hidden_size = eager.q.in_features
        out.num_heads = eager.num_heads
        out.num_kv_heads = eager.num_kv_heads
        out.head_dim = eager.head_dim
        out.n_q = eager.num_heads * eager.head_dim
        out.n_k = eager.num_kv_heads * eager.head_dim

        # Concat along output (row) axis. Source weights are (n_*, K); result
        # is (n_q + 2*n_k, K).
        W_qkv = torch.cat(
            [eager.q.weight, eager.k.weight, eager.v.weight], dim=0
        )
        out.W_qkv = nn.Parameter(W_qkv, requires_grad=False)
        out.o = eager.o
        out.q_norm = eager.q_norm
        out.k_norm = eager.k_norm
        return out

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv_cache,
        layer_idx: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int,
    ) -> torch.Tensor:
        # Body mirrors src/engine/attention.py:112-149 with the first three
        # lines replaced by the fused kernel call.
        from engine.attention import apply_rope

        B, T, _ = x.shape

        q_flat, k_flat, v_flat = triton_fused_qkv(
            x, self.W_qkv, n_q=self.n_q, n_k=self.n_k
        )
        q = q_flat.view(B, T, self.num_heads, self.head_dim)
        k = k_flat.view(B, T, self.num_kv_heads, self.head_dim)
        v = v_flat.view(B, T, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)  # (B, num_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        end_pos = start_pos + T
        kv_cache.k[layer_idx][:B, :, start_pos:end_pos, :] = k
        kv_cache.v[layer_idx][:B, :, start_pos:end_pos, :] = v
        full_k = kv_cache.k[layer_idx][:B, :, :end_pos, :]
        full_v = kv_cache.v[layer_idx][:B, :, :end_pos, :]

        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            full_k = full_k.repeat_interleave(n_rep, dim=1)
            full_v = full_v.repeat_interleave(n_rep, dim=1)

        if T == 1:
            attn = F.scaled_dot_product_attention(q, full_k, full_v, is_causal=False)
        elif start_pos == 0:
            attn = F.scaled_dot_product_attention(q, full_k, full_v, is_causal=True)
        else:
            q_pos = torch.arange(start_pos, end_pos, device=q.device).unsqueeze(1)
            k_pos = torch.arange(0, end_pos, device=q.device).unsqueeze(0)
            mask = q_pos >= k_pos
            attn = F.scaled_dot_product_attention(q, full_k, full_v, attn_mask=mask)

        attn = attn.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.o(attn)
