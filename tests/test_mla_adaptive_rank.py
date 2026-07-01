"""Tests for per-layer adaptive-rank MLA conversion.

Covers the allocator (budget/floor/optimality), the convert→apply→cache path
with non-uniform per-layer ranks, and that an adaptive conversion at full rank
per layer is still algebraically lossless.
"""

from __future__ import annotations

import pytest
import torch

from engine.mla import MLAttention
from mla.convert import (
    allocate_adaptive_ranks,
    compute_adaptive_ranks,
    convert_loaded_to_mla,
)
from mla.swap import alloc_mla_cache, apply_mla

from ._tiny import Cfg, calib_meta, identity_covariances, tiny_loaded
from .test_mla_convert import _baseline_logits, _max_joint_rank, _mla_logits


# --- allocator ------------------------------------------------------------

def test_allocate_respects_budget_and_floor():
    spectra = [torch.tensor([10.0, 1.0, 0.1, 0.01]), torch.tensor([5.0, 4.0, 3.0, 2.0])]
    ranks = allocate_adaptive_ranks(spectra, total_budget=5, min_rank=1)
    assert sum(ranks) == 5
    assert all(r >= 1 for r in ranks)


def test_allocate_is_optimal_favours_heavier_tail():
    # layer 0 spectrum collapses fast; layer 1 has a heavy tail. With 2 units
    # of budget above the floor, both should go to layer 1 (16 and 9 > 1).
    spectra = [torch.tensor([10.0, 1.0, 0.1, 0.01]), torch.tensor([5.0, 4.0, 3.0, 2.0])]
    ranks = allocate_adaptive_ranks(spectra, total_budget=4, min_rank=1)
    assert ranks == [1, 3]


def test_allocate_full_budget_returns_caps():
    spectra = [torch.randn(4).abs(), torch.randn(6).abs()]
    ranks = allocate_adaptive_ranks(spectra, total_budget=10, min_rank=1)
    assert ranks == [4, 6]


def test_allocate_budget_below_floor_raises():
    spectra = [torch.randn(4).abs(), torch.randn(4).abs()]
    with pytest.raises(ValueError):
        allocate_adaptive_ranks(spectra, total_budget=1, min_rank=1)


def test_compute_adaptive_ranks_end_to_end():
    c = Cfg()
    loaded = tiny_loaded(c)
    covs = identity_covariances(c)
    budget = 40  # avg 20 across 2 layers
    ranks = compute_adaptive_ranks(
        loaded, covariances=covs, d_rope=0, total_budget=budget, min_rank=4
    )
    assert len(ranks) == c.num_hidden_layers
    assert sum(ranks) == budget
    assert all(r >= 4 for r in ranks)


# --- convert → apply → cache ---------------------------------------------

def test_convert_adaptive_records_per_layer_ranks():
    c = Cfg()
    loaded = tiny_loaded(c)
    covs = identity_covariances(c)
    ranks = [40, 24]
    art = convert_loaded_to_mla(
        loaded, covariances=covs, rank=32, d_rope=0, ranks=ranks,
        factor_dtype=torch.float32, calibration_meta=calib_meta(c),
    )
    assert art["meta"]["adaptive_rank"] is True
    assert art["meta"]["ranks"] == ranks
    for i, r in enumerate(ranks):
        assert art["state"][f"layers.{i}.attn.dkv.weight"].shape[0] == r


def test_apply_mla_builds_per_layer_rank_and_cache():
    c = Cfg()
    loaded = tiny_loaded(c)
    covs = identity_covariances(c)
    ranks = [40, 16]
    art = convert_loaded_to_mla(
        loaded, covariances=covs, rank=32, d_rope=0, ranks=ranks,
        factor_dtype=torch.float32, calibration_meta=calib_meta(c),
    )
    from engine.model import Qwen3Model

    model = Qwen3Model.from_loaded(loaded).eval()
    apply_mla(model, art)
    attn_ranks = [b.attn.rank for b in model.layers if isinstance(b.attn, MLAttention)]
    assert attn_ranks == ranks

    cache = alloc_mla_cache(model, max_seq_len=8)
    assert [t.shape[-1] for t in cache.c_kv] == ranks

    ids = torch.randint(0, c.vocab_size, (1, 5))
    with torch.inference_mode():
        logits = model(ids, cache, start_pos=0)
    assert logits.shape == (1, 5, c.vocab_size)
    assert torch.isfinite(logits).all()


def test_adaptive_full_rank_matches_baseline_end_to_end():
    # The only baseline-comparable config is d_rope=head_dim (full RoPE on the
    # whole head). At full joint rank per layer the adaptive path must be just
    # as lossless as the uniform path — exercised here through ranks=.
    c = Cfg()
    loaded = tiny_loaded(c, seed=1)
    covs = identity_covariances(c)
    d_rope = c.head_dim
    full = _max_joint_rank(c, d_rope)
    ranks = [full] * c.num_hidden_layers
    art = convert_loaded_to_mla(
        loaded, covariances=covs, rank=full, d_rope=d_rope, ranks=ranks,
        factor_dtype=torch.float32, calibration_meta=calib_meta(c),
    )
    prompt = torch.randint(0, c.vocab_size, (1, 8), dtype=torch.long)
    diff = (_baseline_logits(loaded, prompt) - _mla_logits(loaded, art, prompt)).abs().max().item()
    assert diff < 1e-4, f"max abs diff {diff:.2e}"
