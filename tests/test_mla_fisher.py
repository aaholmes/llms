"""Tests for two-sided Fisher-weighted SVD init.

Covers the output-gradient Fisher collector, the row-weighted SVD primitive,
and end-to-end Fisher-weighted conversion.
"""

from __future__ import annotations

import torch

from engine.model import Qwen3Model
from mla.calibrate import collect_output_fisher
from mla.convert import convert_loaded_to_mla
from mla.svd import activation_aware_factor
from mla.swap import alloc_mla_cache, apply_mla

from ._tiny import Cfg, calib_meta, identity_covariances, tiny_loaded
from .test_mla_convert import _baseline_logits, _mla_logits, _max_joint_rank


# --- row-weighted SVD primitive ------------------------------------------

def test_row_weight_full_rank_lossless():
    """Row weighting must not break full-rank reconstruction (weights cancel)."""
    torch.manual_seed(0)
    W = torch.randn(20, 16, dtype=torch.float64)
    C = torch.eye(16, dtype=torch.float64)
    w = torch.rand(20, dtype=torch.float64) + 0.1
    A, B = activation_aware_factor(W, C, rank=16, row_weight=w)
    assert (A @ B - W).abs().max().item() < 1e-10


def test_row_weight_changes_low_rank_solution():
    """At low rank, a non-uniform row weight changes which rows are preserved."""
    torch.manual_seed(1)
    W = torch.randn(12, 10, dtype=torch.float64)
    C = torch.eye(10, dtype=torch.float64)
    A0, B0 = activation_aware_factor(W, C, rank=3)
    w = torch.linspace(0.1, 5.0, 12, dtype=torch.float64)
    Aw, Bw = activation_aware_factor(W, C, rank=3, row_weight=w)
    assert not torch.allclose(A0 @ B0, Aw @ Bw, atol=1e-3)
    # heavily-weighted rows should be reconstructed more faithfully than unweighted
    err_w = ((Aw @ Bw - W)[-3:]).abs().mean()   # high-weight rows
    err_0 = ((A0 @ B0 - W)[-3:]).abs().mean()
    assert err_w <= err_0 + 1e-9


# --- Fisher collector -----------------------------------------------------

def test_collect_output_fisher_shapes_positive_and_restores_grad():
    c = Cfg()
    loaded = tiny_loaded(c, seed=1)
    model = Qwen3Model.from_loaded(loaded)
    grad_before = {n: p.requires_grad for n, p in model.named_parameters()}
    chunks = [torch.randint(0, c.vocab_size, (1, 12)) for _ in range(4)]
    fk, fv = collect_output_fisher(model, chunks, accumulator_device="cpu")

    kv_dim = c.num_key_value_heads * c.head_dim
    assert len(fk) == len(fv) == c.num_hidden_layers
    for t in fk + fv:
        assert t.shape == (kv_dim,)
        assert (t > 0).all() and torch.isfinite(t).all()
    # requires_grad state must be restored exactly (the collector toggles it).
    for n, p in model.named_parameters():
        assert p.requires_grad == grad_before[n]


# --- end-to-end Fisher-weighted conversion --------------------------------

def test_fisher_convert_sets_meta_and_matches_baseline_full_rank():
    c = Cfg()
    loaded = tiny_loaded(c, seed=1)
    model = Qwen3Model.from_loaded(loaded)
    chunks = [torch.randint(0, c.vocab_size, (1, 12)) for _ in range(4)]
    fk, fv = collect_output_fisher(model, chunks, accumulator_device="cpu")

    covs = identity_covariances(c)
    d_rope = c.head_dim  # V-only stack → the only baseline-comparable config
    full = _max_joint_rank(c, d_rope)
    art = convert_loaded_to_mla(
        loaded, covariances=covs, rank=full, d_rope=d_rope, fisher=(fk, fv),
        factor_dtype=torch.float32, calibration_meta=calib_meta(c),
        target_model_id="synthetic/tiny",
    )
    assert art["meta"]["fisher_weighted"] is True
    prompt = torch.randint(0, c.vocab_size, (1, 8), dtype=torch.long)
    diff = (_baseline_logits(loaded, prompt) - _mla_logits(loaded, art, prompt)).abs().max().item()
    assert diff < 1e-4, f"full-rank Fisher convert should be lossless, got {diff:.2e}"
