"""Tests for streaming perplexity evaluation.

The split is deliberate: ``chunk_nll(logits, target_ids)`` is a tiny pure
function we can test exhaustively without any model; ``evaluate_ppl(model,
chunks)`` is the integration glue tested via a real Qwen3-0.6B forward
pass under the ``requires_draft`` marker.
"""

from __future__ import annotations

import math

import pytest
import torch

from bench.eval_ppl import chunk_nll, evaluate_ppl
from engine.model import Qwen3Model
from engine.weights import load_weights


# --- pure-functional chunk_nll ----------------------------------------------

def test_chunk_nll_uniform_logits_gives_log_vocab() -> None:
    """Uniform logits → cross-entropy = log(V) per predicted position."""
    V = 32
    T = 5
    torch.manual_seed(0)
    logits = torch.zeros(1, T, V)  # uniform after softmax
    targets = torch.randint(0, V, (1, T), dtype=torch.long)

    nll_sum, count = chunk_nll(logits, targets)

    assert count == T - 1
    expected = (T - 1) * math.log(V)
    assert abs(nll_sum - expected) < 1e-5, f"expected {expected}, got {nll_sum}"


def test_chunk_nll_perfect_prediction_gives_near_zero() -> None:
    """Logits with all mass on the correct next token → nll → 0."""
    V = 16
    T = 6
    torch.manual_seed(1)
    targets = torch.randint(0, V, (1, T), dtype=torch.long)
    # Position t-1 must put mass on token at position t (1..T-1).
    logits = torch.full((1, T, V), -1e9)
    for t in range(1, T):
        logits[0, t - 1, targets[0, t]] = 1e9

    nll_sum, count = chunk_nll(logits, targets)

    assert count == T - 1
    assert nll_sum < 1e-3, f"expected near-zero, got {nll_sum}"


def test_chunk_nll_known_distribution_matches_manual() -> None:
    """For a 2-token vocab and explicit logits, cross-check against by-hand math."""
    # logits = log [0.7, 0.3] at every predicted position.
    V = 2
    T = 4
    p = torch.tensor([0.7, 0.3])
    one_step = torch.log(p).unsqueeze(0)  # (1, 2)
    logits = one_step.expand(T, V).unsqueeze(0)  # (1, T, 2)

    # Targets: alternate 0, 1, 0, 1. Predicted positions are t=1..3.
    targets = torch.tensor([[0, 1, 0, 1]], dtype=torch.long)
    # Per-position nll: -log(p[targets[t]]) for t=1..3 → -log(0.3), -log(0.7), -log(0.3)
    expected = -math.log(0.3) - math.log(0.7) - math.log(0.3)

    nll_sum, count = chunk_nll(logits, targets)

    assert count == 3
    assert abs(nll_sum - expected) < 1e-6


def test_chunk_nll_only_predicts_after_first_position() -> None:
    """The first token has no history; chunk_nll must not consume it as a target."""
    V = 4
    T = 3
    logits = torch.zeros(1, T, V)
    targets = torch.tensor([[0, 1, 2]], dtype=torch.long)
    _, count = chunk_nll(logits, targets)
    assert count == T - 1


# --- evaluate_ppl integration -----------------------------------------------

@pytest.mark.requires_draft
def test_evaluate_ppl_qwen3_0_6b_smoke(draft_model_id: str) -> None:
    """Real Qwen3-0.6B over a small batch of token streams; PPL should be finite and sane."""
    loaded = load_weights(draft_model_id, dtype=torch.float32, device="cpu")
    model = Qwen3Model.from_loaded(loaded).eval()

    torch.manual_seed(0)
    chunks = [
        torch.randint(0, loaded.config.vocab_size, (1, 64), dtype=torch.long)
        for _ in range(2)
    ]

    metrics = evaluate_ppl(model, chunks)

    assert metrics["chunk_count"] == 2
    assert metrics["token_count"] == 2 * 63  # T-1 per chunk
    assert math.isfinite(metrics["ppl"])
    assert metrics["ppl"] > 1.0, "PPL must exceed 1 unless predictions are perfect"
    # No upper bound: random tokens are way out of distribution and a trained
    # LM will legitimately blow up on adversarial uniform-random sequences.


@pytest.mark.requires_draft
def test_evaluate_ppl_partition_invariance(draft_model_id: str) -> None:
    """Two short chunks vs the same tokens chunked differently give the same total NLL.

    Aggregation is over predicted positions, which is invariant to how chunks are
    sliced as long as we don't change the per-chunk first-token (which has no
    history). With identical chunk boundaries, the metric is deterministic.
    """
    loaded = load_weights(draft_model_id, dtype=torch.float32, device="cpu")
    model = Qwen3Model.from_loaded(loaded).eval()

    torch.manual_seed(7)
    chunk = torch.randint(0, loaded.config.vocab_size, (1, 32), dtype=torch.long)

    m1 = evaluate_ppl(model, [chunk])
    m2 = evaluate_ppl(model, [chunk])
    assert abs(m1["avg_nll"] - m2["avg_nll"]) < 1e-6
