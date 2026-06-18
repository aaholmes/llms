"""GQA + RoPE + KV-cache attention for Qwen3.

Qwen3 quirks vs vanilla Llama:
  - Q/K/V/O projections are non-square: ``head_dim * num_heads != hidden_size``.
  - QK-norm: RMSNorm applied to Q and K per-head along ``head_dim`` before
    attention. Stabilises training and is part of the model's behaviour, so
    correctness depends on applying it.
  - No biases on Q/K/V/O projections (``attention_bias: false``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .kv_cache import KVCache


class RMSNorm(nn.Module):
    """RMSNorm with float32 accumulation, matching HF Llama/Qwen reference."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x32 = x.float()
        var = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        return self.weight * x32.to(orig_dtype)


def build_rope_tables(
    head_dim: int,
    max_pos: int,
    base: float,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute (cos, sin) tables of shape (max_pos, head_dim) for RoPE.

    Uses the HF Llama/Qwen "concat" convention: the second half of head_dim
    repeats the first half's frequencies, matched by ``rotate_half`` below.
    """
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)  # (max_pos, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (max_pos, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q and K.

    q, k: (B, num_heads, T, head_dim)
    cos, sin: (T, head_dim)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


class Attention(nn.Module):
    """Grouped-query attention with QK-norm and RoPE, against a KV cache."""

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
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=rms_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_eps)
        # Optional swappable decode attention op (strategy hook). When set, it
        # replaces the single-query (T==1) attention computation; default None
        # leaves the SDPA path byte-for-byte unchanged. The callback receives the
        # pre-GQA-expansion K/V and returns attn of shape (B, num_heads, 1, head_dim).
        self.decode_attn_op = None

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv_cache: KVCache,
        layer_idx: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_kv_heads, self.head_dim)

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

        # Swappable decode op: replaces SDPA for the single-query path, fed the
        # pre-expansion (GQA) K/V so it can group/sample as it sees fit.
        if T == 1 and self.decode_attn_op is not None:
            attn = self.decode_attn_op(
                q, full_k, full_v, scale=self.head_dim ** -0.5, layer_idx=layer_idx
            )
            attn = attn.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
            return self.o(attn)

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
