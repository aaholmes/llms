"""The swappable decode attention seam (``Attention.decode_attn_op``).

A research hook: when unset (default) decode behaviour is byte-for-byte the
existing SDPA path; when set, the callback replaces the single-query (T==1)
attention op and receives the *pre-GQA-expansion* K/V so it can group as it likes.
Prefill (T>1) never calls the hook.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.attention import Attention, build_rope_tables
from engine.kv_cache import KVCache

DIMS = dict(hidden_size=32, num_heads=4, num_kv_heads=2, head_dim=8, rms_eps=1e-6)
N_LAYERS = 1


def _setup(seed: int = 0):
    torch.manual_seed(seed)
    attn = Attention(**DIMS)
    cache = KVCache.alloc(
        num_layers=N_LAYERS, max_batch=1, num_kv_heads=DIMS["num_kv_heads"],
        max_seq_len=16, head_dim=DIMS["head_dim"], dtype=torch.float32, device="cpu",
    )
    cos, sin = build_rope_tables(DIMS["head_dim"], 16, base=10000.0, dtype=torch.float32)
    return attn, cache, cos, sin


def _prefill(attn, cache, cos, sin, n: int):
    x = torch.randn(1, n, DIMS["hidden_size"])
    attn(x, kv_cache=cache, layer_idx=0, cos=cos[:n], sin=sin[:n], start_pos=0)
    cache.cur_len = n


def _decode_step(attn, cache, cos, sin, pos: int):
    x = torch.randn(1, 1, DIMS["hidden_size"])
    return attn(x, kv_cache=cache, layer_idx=0, cos=cos[pos:pos + 1], sin=sin[pos:pos + 1], start_pos=pos)


def test_default_hook_is_none_and_unchanged():
    attn, cache, cos, sin = _setup()
    assert attn.decode_attn_op is None
    _prefill(attn, cache, cos, sin, 5)
    torch.manual_seed(99)
    out = _decode_step(attn, cache, cos, sin, 5)
    assert out.shape == (1, 1, DIMS["hidden_size"])


def test_sdpa_equivalent_hook_reproduces_default():
    # Run a decode step with the default path, then identical inputs through a
    # hook that reproduces SDPA — outputs must match.
    def sdpa_hook(q, full_k, full_v, *, scale, layer_idx):
        n_rep = q.shape[1] // full_k.shape[1]
        k = full_k.repeat_interleave(n_rep, dim=1)
        v = full_v.repeat_interleave(n_rep, dim=1)
        return F.scaled_dot_product_attention(q, k, v, is_causal=False)

    attn, cache, cos, sin = _setup()
    _prefill(attn, cache, cos, sin, 5)

    torch.manual_seed(99)
    ref = _decode_step(attn, cache, cos, sin, 5)
    cache.cur_len = 5  # roll back the position written by the decode step

    attn.decode_attn_op = sdpa_hook
    torch.manual_seed(99)
    hooked = _decode_step(attn, cache, cos, sin, 5)

    torch.testing.assert_close(hooked, ref, rtol=1e-5, atol=1e-5)


def test_hook_fires_only_on_decode():
    calls = []

    def spy(q, full_k, full_v, *, scale, layer_idx):
        calls.append(q.shape)
        n_rep = q.shape[1] // full_k.shape[1]
        k = full_k.repeat_interleave(n_rep, dim=1)
        v = full_v.repeat_interleave(n_rep, dim=1)
        return F.scaled_dot_product_attention(q, k, v, is_causal=False)

    attn, cache, cos, sin = _setup()
    attn.decode_attn_op = spy
    _prefill(attn, cache, cos, sin, 5)  # T>1: must NOT call the hook
    assert calls == []
    _decode_step(attn, cache, cos, sin, 5)  # T==1: calls hook once
    assert len(calls) == 1


def test_hook_receives_pre_expansion_kv():
    seen = {}

    def grab(q, full_k, full_v, *, scale, layer_idx):
        seen["q"] = q.shape
        seen["k"] = full_k.shape
        return torch.zeros_like(q)

    attn, cache, cos, sin = _setup()
    attn.decode_attn_op = grab
    _prefill(attn, cache, cos, sin, 5)
    _decode_step(attn, cache, cos, sin, 5)
    # q has num_heads; K/V keep num_kv_heads (pre-expansion); 6 positions cached
    assert seen["q"] == (1, DIMS["num_heads"], 1, DIMS["head_dim"])
    assert seen["k"] == (1, DIMS["num_kv_heads"], 6, DIMS["head_dim"])
