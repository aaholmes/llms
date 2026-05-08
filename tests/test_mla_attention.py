"""Correctness tests for the MLA runtime module.

The load-bearing claim of this module is that ``MLAttention`` (with
``qk_norm_mode='single'`` — RMSNorm applied across the full reconstructed
K head, then split for partial-RoPE) is *algebraically equivalent* to a
plain partial-RoPE attention given full-rank factors. The reference here
is **not** vanilla Qwen3 ``Attention`` because Qwen3 RoPE-rotates the
full head_dim while MLA only rotates the d_rope subspace; the proper
baseline is "Qwen3 attention with RoPE restricted to the d_rope subspace
that comprises the *first ``d_rope/2`` original Qwen3 RoPE pairs*". We
inline that reference (``PartialRoPEAttention``) in this file so the
comparison is unambiguous.

The "rope subspace" is the set of head dimensions
``[0, 1, …, d_rope/2 − 1, head_dim/2, …, head_dim/2 + d_rope/2 − 1]`` —
i.e. both halves of the first ``d_rope/2`` original RoPE pairs. Realised
in the runtime via a ``head_dim`` permutation that puts those dims at the
end so the runtime keeps a contiguous-last-``d_rope`` layout.

Tests:
  - third-party math check: partial-RoPE on the rope subspace equals a
    slice of original full-RoPE applied with the original frequencies
  - prefill bit-equivalence between MLAttention and PartialRoPEAttention
    at full rank, varied d_rope
  - decode-step bit-equivalence following a prefill
  - exact-low-rank construction (W_K/W_V built from rank-r₀ factors)
  - MLAKVCache shape + truncate semantics
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from engine.attention import RMSNorm, _rotate_half, build_rope_tables
from engine.mla import MLAKVCache, MLAttention


@dataclass(frozen=True)
class _Cfg:
    hidden_size: int = 64
    num_heads: int = 4
    num_kv_heads: int = 2
    head_dim: int = 16
    rms_eps: float = 1e-6
    max_pos: int = 32
    rope_theta: float = 10000.0


CFG = _Cfg()


# --- rope-subspace permutation helpers ------------------------------------

def rope_pair_permutation(head_dim: int, d_rope: int) -> list[int]:
    """Per-head permutation that puts the first d_rope/2 original RoPE pairs
    at the *end* of the head, contiguously.

    Returns a length-``head_dim`` list ``perm`` such that
    ``permuted_head[i] = original_head[perm[i]]``. The first ``d_nope`` slots
    hold the no-rope dims; the last ``d_rope`` slots hold the rope subspace
    laid out as [pair_0_half_1, …, pair_{d_rope/2-1}_half_1,
    pair_0_half_2, …, pair_{d_rope/2-1}_half_2] so that the runtime's
    ``_rotate_half`` over those last d_rope dims pairs the right halves.
    """
    half = head_dim // 2
    h_rope = d_rope // 2
    rope_dims = list(range(h_rope)) + list(range(half, half + h_rope))
    nope_dims = list(range(h_rope, half)) + list(range(half + h_rope, head_dim))
    return nope_dims + rope_dims


def inverse_permutation(perm: list[int]) -> list[int]:
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


# --- third-party math check -----------------------------------------------

def test_partial_rope_matches_original_qwen3_on_rope_subspace() -> None:
    """Partial-RoPE on the permuted-rope subspace = slice of original full-RoPE.

    Independent check that the math we're going to implement in MLAttention
    and PartialRoPEAttention is faithful to Qwen3's original full-RoPE on
    the dimensions that survive the partial-RoPE selection.
    """
    head_dim = 128
    base = 1_000_000.0  # Qwen3-4B's rope_theta
    max_pos = 64
    T = 5

    for d_rope in (32, 64):
        d_nope = head_dim - d_rope
        perm = rope_pair_permutation(head_dim, d_rope)
        inv_perm = inverse_permutation(perm)

        rope_dims_in_orig = perm[d_nope:]  # last d_rope of perm = rope subspace
        nope_dims_in_orig = perm[:d_nope]

        torch.manual_seed(d_rope)
        x = torch.randn(1, 1, T, head_dim)

        # Original full-RoPE
        cos_full, sin_full = build_rope_tables(head_dim, max_pos, base=base)
        positions = torch.arange(T)
        cos_p = cos_full[positions].unsqueeze(0).unsqueeze(0)
        sin_p = sin_full[positions].unsqueeze(0).unsqueeze(0)
        full_rotated = x * cos_p + _rotate_half(x) * sin_p

        # Partial-RoPE on the permuted x using the sliced original inv_freq
        x_perm = x[..., perm]
        full_inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        inv_freq = full_inv_freq[: d_rope // 2]
        t = torch.arange(max_pos, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_partial = emb.cos()[positions].unsqueeze(0).unsqueeze(0)
        sin_partial = emb.sin()[positions].unsqueeze(0).unsqueeze(0)

        x_nope, x_rope = x_perm.split([d_nope, d_rope], dim=-1)
        x_rope_rot = x_rope * cos_partial + _rotate_half(x_rope) * sin_partial
        partial_perm = torch.cat([x_nope, x_rope_rot], dim=-1)
        partial_unperm = partial_perm[..., inv_perm]

        # Rope subspace dims must match original full-RoPE on those same dims.
        for d in rope_dims_in_orig:
            diff = (partial_unperm[..., d] - full_rotated[..., d]).abs().max().item()
            assert diff < 1e-5, (
                f"d_rope={d_rope}, dim {d}: partial-RoPE diverges from original "
                f"full-RoPE on the rope subspace ({diff:.2e})"
            )
        # Nope-subspace dims must be unchanged from input (no rotation applied).
        for d in nope_dims_in_orig:
            diff = (partial_unperm[..., d] - x[..., d]).abs().max().item()
            assert diff < 1e-7, (
                f"d_rope={d_rope}, dim {d}: nope subspace was modified ({diff:.2e})"
            )


# --- partial-RoPE reference attention -------------------------------------

class PartialRoPEAttention(nn.Module):
    """GQA + QK-norm + RoPE *restricted to the last d_rope dims of each head*.

    Identical to ``engine.attention.Attention`` except that ``apply_rope``
    is run on a slice of width ``d_rope`` instead of the full head. This
    is the apples-to-apples reference for ``MLAttention`` correctness.
    """

    def __init__(self, cfg: _Cfg, *, d_rope: int) -> None:
        super().__init__()
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.d_rope = d_rope
        self.d_nope = cfg.head_dim - d_rope
        self.q = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.k = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.v = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.o = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, eps=cfg.rms_eps)
        self.k_norm = RMSNorm(cfg.head_dim, eps=cfg.rms_eps)
        # Match MLAttention's rope-table construction: sliced original-pair
        # frequencies, not a fresh d_rope-sized formula.
        if d_rope > 0:
            full_inv_freq = 1.0 / (
                cfg.rope_theta ** (
                    torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim
                )
            )
            inv_freq = full_inv_freq[: d_rope // 2]
            t = torch.arange(cfg.max_pos, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self.register_buffer("rope_cos", emb.cos(), persistent=False)
            self.register_buffer("rope_sin", emb.sin(), persistent=False)
        else:
            self.rope_cos = None
            self.rope_sin = None

    def _apply_rope_slice(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        if self.d_rope == 0:
            return x
        cos = self.rope_cos[positions]  # (T, d_rope)
        sin = self.rope_sin[positions]
        x_nope, x_rope = x.split([self.d_nope, self.d_rope], dim=-1)
        c = cos.unsqueeze(0).unsqueeze(0)
        s = sin.unsqueeze(0).unsqueeze(0)
        x_rope_rot = x_rope * c + _rotate_half(x_rope) * s
        return torch.cat([x_nope, x_rope_rot], dim=-1)

    def forward(self, x: torch.Tensor, *, start_pos: int, k_cache: torch.Tensor,
                v_cache: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)  # (B, H, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        positions = torch.arange(start_pos, start_pos + T, device=x.device)
        q = self._apply_rope_slice(q, positions)
        k = self._apply_rope_slice(k, positions)

        end_pos = start_pos + T
        k_cache[:B, :, start_pos:end_pos, :] = k
        v_cache[:B, :, start_pos:end_pos, :] = v
        full_k = k_cache[:B, :, :end_pos, :]
        full_v = v_cache[:B, :, :end_pos, :]

        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            full_k = full_k.repeat_interleave(n_rep, dim=1)
            full_v = full_v.repeat_interleave(n_rep, dim=1)

        q_pos = torch.arange(start_pos, end_pos, device=q.device).unsqueeze(1)
        k_pos = torch.arange(0, end_pos, device=q.device).unsqueeze(0)
        mask = q_pos >= k_pos
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, full_k, full_v, attn_mask=mask
        )
        attn = attn.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.o(attn)


def _alloc_baseline_kv(cfg: _Cfg, *, dtype=torch.float32, device="cpu"):
    shape = (1, cfg.num_kv_heads, cfg.max_pos, cfg.head_dim)
    return (
        torch.zeros(shape, dtype=dtype, device=device),
        torch.zeros(shape, dtype=dtype, device=device),
    )


# --- factor construction --------------------------------------------------

def _nope_row_indices(cfg: _Cfg, d_rope: int) -> list[int]:
    d_nope = cfg.head_dim - d_rope
    return [
        h * cfg.head_dim + i
        for h in range(cfg.num_kv_heads)
        for i in range(d_nope)
    ]


def _rope_row_indices(cfg: _Cfg, d_rope: int) -> list[int]:
    d_nope = cfg.head_dim - d_rope
    return [
        h * cfg.head_dim + d_nope + j
        for h in range(cfg.num_kv_heads)
        for j in range(d_rope)
    ]


def _build_pair_full_rank(cfg: _Cfg, *, d_rope: int, seed: int) -> tuple[
    PartialRoPEAttention, MLAttention
]:
    """Reference and MLAttention with full-rank (= hidden_size) factors.

    ``W_dkv = I`` so c_kv = x. ``W_uk_nope`` and ``W_kr`` are slices of
    the reference's ``W_K`` along the K head axis. ``W_uv = W_V``.
    """
    torch.manual_seed(seed)
    ref = PartialRoPEAttention(cfg, d_rope=d_rope)
    rank = cfg.hidden_size
    d_nope = cfg.head_dim - d_rope

    mla = MLAttention(
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        rms_eps=cfg.rms_eps,
        rank=rank,
        d_rope=d_rope,
        max_position_embeddings=cfg.max_pos,
        rope_theta=cfg.rope_theta,
        qk_norm_mode="single",
    )
    with torch.no_grad():
        mla.q.weight.copy_(ref.q.weight)
        mla.o.weight.copy_(ref.o.weight)
        mla.q_norm.weight.copy_(ref.q_norm.weight)
        mla.k_norm.weight.copy_(ref.k_norm.weight)
        mla.dkv.weight.copy_(torch.eye(cfg.hidden_size))
        nope_rows = _nope_row_indices(cfg, d_rope)
        rope_rows = _rope_row_indices(cfg, d_rope)
        if nope_rows:
            mla.uk_nope.weight.copy_(ref.k.weight[nope_rows])
        if rope_rows:
            mla.kr.weight.copy_(ref.k.weight[rope_rows])
        mla.uv.weight.copy_(ref.v.weight)
    return ref, mla


def _build_pair_exact_low_rank(
    cfg: _Cfg, *, d_rope: int, rank: int, seed: int, dtype: torch.dtype = torch.float64,
) -> tuple[PartialRoPEAttention, MLAttention]:
    """Construct W_K, W_V to factor exactly through rank ``rank``.

    Random ``W_dkv (rank, hidden)``, ``W_uk_nope (kv*d_nope, rank)``,
    ``W_uv (kv*head_dim, rank)``, ``W_kr (kv*d_rope, hidden)``. The
    reference's ``W_K`` is assembled by interleaving ``W_uk_nope @ W_dkv``
    rows (nope) with ``W_kr`` rows (rope) per head, and ``W_V = W_uv @ W_dkv``.

    All factor matrices and the assembled ``W_K``/``W_V`` are built in
    ``dtype`` (default fp64) so the rounding precision used to compute
    the reference's effective ``W_K`` matches the precision MLA will use
    when it computes the same product implicitly at runtime.
    """
    torch.manual_seed(seed)
    d_nope = cfg.head_dim - d_rope

    W_dkv = torch.randn(rank, cfg.hidden_size, dtype=dtype)
    W_uk_nope = (
        torch.randn(cfg.num_kv_heads * d_nope, rank, dtype=dtype)
        if d_nope > 0
        else torch.zeros(0, rank, dtype=dtype)
    )
    W_uv = torch.randn(cfg.num_kv_heads * cfg.head_dim, rank, dtype=dtype)
    W_kr = (
        torch.randn(cfg.num_kv_heads * d_rope, cfg.hidden_size, dtype=dtype)
        if d_rope > 0
        else torch.zeros(0, cfg.hidden_size, dtype=dtype)
    )

    K_nope_full = W_uk_nope @ W_dkv  # (kv*d_nope, hidden)
    W_K = torch.empty(cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size, dtype=dtype)
    for h in range(cfg.num_kv_heads):
        if d_nope > 0:
            W_K[h * cfg.head_dim : h * cfg.head_dim + d_nope] = (
                K_nope_full[h * d_nope : (h + 1) * d_nope]
            )
        if d_rope > 0:
            W_K[h * cfg.head_dim + d_nope : (h + 1) * cfg.head_dim] = (
                W_kr[h * d_rope : (h + 1) * d_rope]
            )
    W_V = W_uv @ W_dkv

    ref = PartialRoPEAttention(cfg, d_rope=d_rope).to(dtype=dtype)
    with torch.no_grad():
        ref.k.weight.copy_(W_K)
        ref.v.weight.copy_(W_V)

    mla = MLAttention(
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        rms_eps=cfg.rms_eps,
        rank=rank,
        d_rope=d_rope,
        max_position_embeddings=cfg.max_pos,
        rope_theta=cfg.rope_theta,
        qk_norm_mode="single",
    ).to(dtype=dtype)
    with torch.no_grad():
        mla.q.weight.copy_(ref.q.weight)
        mla.o.weight.copy_(ref.o.weight)
        mla.q_norm.weight.copy_(ref.q_norm.weight)
        mla.k_norm.weight.copy_(ref.k_norm.weight)
        mla.dkv.weight.copy_(W_dkv)
        if d_nope > 0:
            mla.uk_nope.weight.copy_(W_uk_nope)
        if d_rope > 0:
            mla.kr.weight.copy_(W_kr)
        mla.uv.weight.copy_(W_uv)
    return ref, mla


# --- tests ----------------------------------------------------------------

@pytest.mark.parametrize("d_rope", [0, 4, 8, 12, 16])
def test_full_rank_prefill_matches_partial_rope_baseline(d_rope: int) -> None:
    """At rank=hidden_size with single-norm, MLA equals partial-RoPE attention."""
    ref, mla = _build_pair_full_rank(CFG, d_rope=d_rope, seed=0)
    ref.eval(); mla.eval()

    torch.manual_seed(7)
    T = 6
    x = torch.randn(1, T, CFG.hidden_size)

    k_cache, v_cache = _alloc_baseline_kv(CFG)
    mla_cache = MLAKVCache.alloc(
        num_layers=1, max_batch=1, num_kv_heads=CFG.num_kv_heads,
        max_seq_len=CFG.max_pos, rank=CFG.hidden_size, d_rope=d_rope,
        dtype=torch.float32, device="cpu",
    )

    with torch.no_grad():
        out_ref = ref(x, start_pos=0, k_cache=k_cache, v_cache=v_cache)
        out_mla = mla(x, kv_cache=mla_cache, layer_idx=0, start_pos=0)

    diff = (out_ref - out_mla).abs().max().item()
    assert diff < 1e-5, f"d_rope={d_rope}: max abs diff {diff:.2e}"


def test_decode_step_matches_after_prefill() -> None:
    """Prefill then decode: MLA matches reference at every step."""
    d_rope = 8
    ref, mla = _build_pair_full_rank(CFG, d_rope=d_rope, seed=1)
    ref.eval(); mla.eval()

    torch.manual_seed(11)
    T_pre = 5
    x_pre = torch.randn(1, T_pre, CFG.hidden_size)
    x_dec = torch.randn(1, 1, CFG.hidden_size)

    k_cache, v_cache = _alloc_baseline_kv(CFG)
    mla_cache = MLAKVCache.alloc(
        num_layers=1, max_batch=1, num_kv_heads=CFG.num_kv_heads,
        max_seq_len=CFG.max_pos, rank=CFG.hidden_size, d_rope=d_rope,
        dtype=torch.float32, device="cpu",
    )

    with torch.no_grad():
        out_pre_ref = ref(x_pre, start_pos=0, k_cache=k_cache, v_cache=v_cache)
        out_pre_mla = mla(x_pre, kv_cache=mla_cache, layer_idx=0, start_pos=0)
        mla_cache.cur_len = T_pre
        out_dec_ref = ref(x_dec, start_pos=T_pre, k_cache=k_cache, v_cache=v_cache)
        out_dec_mla = mla(x_dec, kv_cache=mla_cache, layer_idx=0, start_pos=T_pre)

    pre_diff = (out_pre_ref - out_pre_mla).abs().max().item()
    dec_diff = (out_dec_ref - out_dec_mla).abs().max().item()
    assert pre_diff < 1e-5, f"prefill diff {pre_diff:.2e}"
    assert dec_diff < 1e-5, f"decode diff {dec_diff:.2e}"


@pytest.mark.parametrize("rank", [4, 8, 16, 32])
def test_exact_low_rank_construction(rank: int) -> None:
    """When W_K/W_V factor exactly through rank r₀, MLA at rank=r₀ matches.

    This is an *algebraic* equivalence — both paths compute the same matrix
    product, just with different parenthesization. fp32 rounding makes the
    two paths drift by ``O(rank · eps)``; fp64 narrows that to ~1e-12 so we
    can actually assert bit-equivalence.
    """
    d_rope = 8
    ref, mla = _build_pair_exact_low_rank(CFG, d_rope=d_rope, rank=rank, seed=rank)
    ref.eval()
    mla.eval()

    torch.manual_seed(rank + 100)
    T = 4
    x = torch.randn(1, T, CFG.hidden_size, dtype=torch.float64)

    k_shape = (1, CFG.num_kv_heads, CFG.max_pos, CFG.head_dim)
    k_cache = torch.zeros(k_shape, dtype=torch.float64)
    v_cache = torch.zeros(k_shape, dtype=torch.float64)
    mla_cache = MLAKVCache.alloc(
        num_layers=1, max_batch=1, num_kv_heads=CFG.num_kv_heads,
        max_seq_len=CFG.max_pos, rank=rank, d_rope=d_rope,
        dtype=torch.float64, device="cpu",
    )

    with torch.no_grad():
        out_ref = ref(x, start_pos=0, k_cache=k_cache, v_cache=v_cache)
        out_mla = mla(x, kv_cache=mla_cache, layer_idx=0, start_pos=0)

    diff = (out_ref - out_mla).abs().max().item()
    assert diff < 1e-10, f"rank={rank}: max abs diff {diff:.2e}"


def test_mla_kv_cache_shape() -> None:
    cache = MLAKVCache.alloc(
        num_layers=3, max_batch=1, num_kv_heads=2, max_seq_len=16,
        rank=8, d_rope=4, dtype=torch.float32, device="cpu",
    )
    assert len(cache.c_kv) == 3
    assert len(cache.k_rope_pre) == 3
    assert cache.c_kv[0].shape == (1, 16, 8)
    assert cache.k_rope_pre[0].shape == (1, 16, 2, 4)
    assert cache.cur_len == 0
    assert cache.max_seq_len == 16


def test_mla_kv_cache_truncate() -> None:
    cache = MLAKVCache.alloc(
        num_layers=1, max_batch=1, num_kv_heads=2, max_seq_len=16,
        rank=8, d_rope=4, dtype=torch.float32, device="cpu",
    )
    cache.cur_len = 10
    cache.truncate(7)
    assert cache.cur_len == 7
    cache.truncate(0)
    assert cache.cur_len == 0
    with pytest.raises(ValueError):
        cache.truncate(-1)
    cache.cur_len = 5
    with pytest.raises(ValueError):
        cache.truncate(6)
