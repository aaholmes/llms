"""Multi-head Latent Attention runtime.

This module is the *runtime* half of post-hoc MHA→MLA conversion. Given
already-computed factors (``W_dkv``, ``W_uk_nope``, ``W_uv``, ``W_kr``,
plus the standard ``W_q``, ``W_o``, ``q_norm``, ``k_norm``), it serves
attention against a compressed KV cache. It does **not** know how those
factors were produced — that's the job of the conversion pipeline.

Per-token cache footprint:
    rank + (num_kv_heads · d_rope)  values
vs the original GQA cache:
    2 · num_kv_heads · head_dim     values

For Qwen3-4B (``num_kv_heads=8, head_dim=128``) at ``rank=128, d_rope=32``
this is 384 vs 2048 — a 5.3× compression.

Key design point: ``qk_norm_mode='single'``. Qwen3 applies per-head
RMSNorm to K (over the full ``head_dim``) before RoPE. To keep that
exactly, MLA must reconstruct ``k_nope`` at attention time, concat with
``k_rope_pre``, apply RMSNorm across the concatenated head, then split
and RoPE the rope half. This is what ``forward`` does. The cache stores
``k_rope_pre`` (un-normed, un-rotated) precisely so this is possible.

A future ``qk_norm_mode='split'`` (separate norms on the two halves) is
required to make the Q-absorb optimization possible — it lets the
``W_q^nope · W_uk_nope`` fusion happen at load time. Not implemented in
this revision; raises ``NotImplementedError``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .attention import RMSNorm, _rotate_half, build_rope_tables


@dataclass
class MLAKVCache:
    """Compressed KV cache: shared latent ``c_kv`` + uncompressed ``k_rope_pre``.

    Layout:
      - ``c_kv[layer]``        : (max_batch, max_seq_len, rank)
      - ``k_rope_pre[layer]``  : (max_batch, max_seq_len, num_kv_heads, d_rope)

    ``cur_len`` advances at the model level (shared across layers), same
    semantics as the GQA ``KVCache``.
    """

    c_kv: list[torch.Tensor]
    k_rope_pre: list[torch.Tensor]
    cur_len: int = 0

    @classmethod
    def alloc(
        cls,
        *,
        num_layers: int,
        max_batch: int,
        num_kv_heads: int,
        max_seq_len: int,
        rank: int,
        d_rope: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> "MLAKVCache":
        c_shape = (max_batch, max_seq_len, rank)
        r_shape = (max_batch, max_seq_len, num_kv_heads, d_rope)
        return cls(
            c_kv=[torch.empty(c_shape, dtype=dtype, device=device) for _ in range(num_layers)],
            k_rope_pre=[torch.empty(r_shape, dtype=dtype, device=device) for _ in range(num_layers)],
        )

    @property
    def max_seq_len(self) -> int:
        return self.c_kv[0].shape[1]

    def reset(self) -> None:
        self.cur_len = 0

    def truncate(self, new_len: int) -> None:
        if new_len < 0 or new_len > self.cur_len:
            raise ValueError(f"truncate {new_len=} outside [0, {self.cur_len}]")
        self.cur_len = new_len


def _rope_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to ``x``'s last dim. ``x`` is (B, H, T, R), cos/sin (T, R)."""
    c = cos.unsqueeze(0).unsqueeze(0)
    s = sin.unsqueeze(0).unsqueeze(0)
    return x * c + _rotate_half(x) * s


class MLAttention(nn.Module):
    """Partial-RoPE GQA over a compressed latent cache.

    Constructor takes the full set of MLA module shapes; weights are
    expected to be loaded by the caller (the conversion pipeline writes
    them; tests inject them directly).
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_eps: float,
        rank: int,
        d_rope: int,
        max_position_embeddings: int,
        rope_theta: float = 10000.0,
        qk_norm_mode: str = "single",
    ) -> None:
        super().__init__()
        if d_rope < 0 or d_rope > head_dim:
            raise ValueError(f"d_rope={d_rope} not in [0, head_dim={head_dim}]")
        if qk_norm_mode != "single":
            raise NotImplementedError(
                f"qk_norm_mode={qk_norm_mode!r}: only 'single' is implemented"
            )

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rank = rank
        self.d_rope = d_rope
        self.d_nope = head_dim - d_rope
        self.qk_norm_mode = qk_norm_mode

        self.q = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.dkv = nn.Linear(hidden_size, rank, bias=False)
        if self.d_nope > 0:
            self.uk_nope = nn.Linear(rank, num_kv_heads * self.d_nope, bias=False)
        else:
            self.uk_nope = None
        self.uv = nn.Linear(rank, num_kv_heads * head_dim, bias=False)
        if d_rope > 0:
            self.kr = nn.Linear(hidden_size, num_kv_heads * d_rope, bias=False)
        else:
            self.kr = None
        self.o = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.q_norm = RMSNorm(head_dim, eps=rms_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_eps)

        if d_rope > 0:
            # The rope subspace is the first d_rope/2 *original* RoPE pairs of
            # the head, brought to the end of head_dim by the conversion's
            # weight permutation. So the cos/sin table here uses a slice of
            # the original head_dim-sized inv_freq, not a fresh d_rope-sized
            # one. This is what makes the partial-RoPE rotation faithful to
            # what the trained model expects on those dims.
            full_inv_freq = 1.0 / (
                rope_theta ** (
                    torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
                )
            )
            inv_freq = full_inv_freq[: d_rope // 2]
            t = torch.arange(max_position_embeddings, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self.register_buffer("rope_cos", emb.cos(), persistent=False)
            self.register_buffer("rope_sin", emb.sin(), persistent=False)
        else:
            self.rope_cos = None
            self.rope_sin = None

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv_cache: MLAKVCache,
        layer_idx: int,
        start_pos: int,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ``cos``/``sin`` from caller are accepted for drop-in compatibility
        # with engine.attention.Attention.forward but ignored — RoPE here is
        # over a different subspace (d_rope, not head_dim) so we use our
        # own tables.
        del cos, sin

        B, T, _ = x.shape
        H, KV, D = self.num_heads, self.num_kv_heads, self.head_dim
        N, R = self.d_nope, self.d_rope
        end_pos = start_pos + T

        # Q: project, single QK-norm over full head_dim, transpose for SDPA
        q = self.q(x).view(B, T, H, D)
        q = self.q_norm(q)
        q = q.transpose(1, 2)  # (B, H, T, D)

        # KV: down-projection (latent) + uncompressed RoPE-K projection
        c_kv = self.dkv(x)  # (B, T, rank)
        kv_cache.c_kv[layer_idx][:B, start_pos:end_pos, :] = c_kv
        if R > 0:
            k_rope_pre_new = self.kr(x).view(B, T, KV, R)
            kv_cache.k_rope_pre[layer_idx][:B, start_pos:end_pos, :, :] = k_rope_pre_new

        full_c_kv = kv_cache.c_kv[layer_idx][:B, :end_pos, :]  # (B, T_full, rank)

        # Reconstruct V (no split, no RoPE on V)
        full_v = self.uv(full_c_kv).view(B, end_pos, KV, D)

        # Reconstruct K from latent + uncompressed rope half, then norm
        if N > 0 and R > 0:
            full_k_nope = self.uk_nope(full_c_kv).view(B, end_pos, KV, N)
            full_k_rope_pre = kv_cache.k_rope_pre[layer_idx][:B, :end_pos, :, :]
            full_k = torch.cat([full_k_nope, full_k_rope_pre], dim=-1)
        elif N > 0:
            full_k = self.uk_nope(full_c_kv).view(B, end_pos, KV, N)
        else:
            full_k = kv_cache.k_rope_pre[layer_idx][:B, :end_pos, :, :]

        full_k = self.k_norm(full_k)  # single norm over full head_dim

        # Now apply partial RoPE: only the last R dims, only on Q rope half
        # (positions [start_pos, end_pos)) and the K rope half (positions
        # [0, end_pos)).
        if R > 0:
            q_nope, q_rope = q.split([N, R], dim=-1)
            full_k_nope_post, full_k_rope_post = full_k.split([N, R], dim=-1)

            q_pos = torch.arange(start_pos, end_pos, device=x.device)
            k_pos = torch.arange(0, end_pos, device=x.device)
            cos_q = self.rope_cos[q_pos].to(q.dtype)
            sin_q = self.rope_sin[q_pos].to(q.dtype)
            cos_k = self.rope_cos[k_pos].to(full_k.dtype)
            sin_k = self.rope_sin[k_pos].to(full_k.dtype)

            q_rope = _rope_apply(q_rope, cos_q, sin_q)
            # full_k is (B, T_full, KV, D); transpose for the rope helper that
            # expects (B, H_or_KV, T, R)
            full_k_rope_post = full_k_rope_post.transpose(1, 2)  # (B, KV, T_full, R)
            full_k_rope_post = _rope_apply(full_k_rope_post, cos_k, sin_k)
            full_k_rope_post = full_k_rope_post.transpose(1, 2)  # (B, T_full, KV, R)

            q = torch.cat([q_nope, q_rope], dim=-1) if N > 0 else q_rope
            if N > 0:
                full_k = torch.cat([full_k_nope_post, full_k_rope_post], dim=-1)
            else:
                full_k = full_k_rope_post

        # Layout for SDPA: (B, H, T, D)
        full_k = full_k.transpose(1, 2)  # (B, KV, T_full, D)
        full_v = full_v.transpose(1, 2)

        n_rep = H // KV
        if n_rep > 1:
            full_k = full_k.repeat_interleave(n_rep, dim=1)
            full_v = full_v.repeat_interleave(n_rep, dim=1)

        q_pos = torch.arange(start_pos, end_pos, device=q.device).unsqueeze(1)
        k_pos = torch.arange(0, end_pos, device=q.device).unsqueeze(0)
        mask = q_pos >= k_pos
        attn = F.scaled_dot_product_attention(q, full_k, full_v, attn_mask=mask)

        attn = attn.transpose(1, 2).reshape(B, T, H * D)
        return self.o(attn)
