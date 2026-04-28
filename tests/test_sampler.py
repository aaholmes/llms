"""Sampler unit tests."""

from __future__ import annotations

import pytest
import torch

from engine.sampler import _apply_top_k, _apply_top_p, greedy, sample


def test_greedy_basic():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
    assert greedy(logits).tolist() == [1]


def test_sample_temperature_zero_is_greedy():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
    assert sample(logits, temperature=0.0).tolist() == [1]


def test_top_k_zeroes_out_below_kth():
    logits = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
    out = _apply_top_k(logits, 2)
    # Top-2 is [5, 4] at indices 1 and 4. Others should be -inf.
    finite = torch.isfinite(out)
    assert finite.tolist() == [False, True, False, False, True]


def test_top_p_keeps_minimal_set():
    # logits chosen so softmax gives roughly [0.6, 0.3, 0.05, 0.05]
    logits = torch.tensor([3.0, 2.3, 0.5, 0.5])
    out = _apply_top_p(logits, 0.7)
    # cumulative after sort: 0.6 (keep), 0.9 (keep), 0.95 (drop), 1.0 (drop)
    # The "drop" mask uses (cum - p) > top_p, so first token at 0.0 stays,
    # second at 0.6 stays (0.6 < 0.7), third at 0.9 drops (0.9 > 0.7), fourth drops.
    finite = torch.isfinite(out)
    assert finite.sum().item() == 2


def test_sample_deterministic_with_generator():
    g = torch.Generator().manual_seed(42)
    logits = torch.randn(4, 100)
    a = sample(logits.clone(), temperature=1.0, generator=g)

    g = torch.Generator().manual_seed(42)
    b = sample(logits.clone(), temperature=1.0, generator=g)

    assert torch.equal(a, b)


def test_sample_shapes():
    # 2D input
    out = sample(torch.randn(3, 100), temperature=0.0)
    assert out.shape == (3,)

    # 3D input (batch, seq) — for verifying multiple positions
    out = sample(torch.randn(2, 5, 100), temperature=0.0)
    assert out.shape == (2, 5)


def test_top_k_with_temperature():
    # With high temperature and tight top_k, sampling should be confined.
    g = torch.Generator().manual_seed(0)
    logits = torch.zeros(1, 1000)
    logits[0, [10, 20, 30]] = 100.0
    samples = [sample(logits, temperature=10.0, top_k=3, generator=g).item() for _ in range(50)]
    assert set(samples) <= {10, 20, 30}
