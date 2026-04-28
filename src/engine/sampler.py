"""Token sampling: greedy / top-k / top-p / temperature.

Single entry point ``sample(logits, ...)`` returns next-token ids of shape
(batch,). Shortcut helpers ``greedy(logits)`` and ``sample_from_dist(probs)``
are provided for spec-decode use.

Convention: ``temperature <= 0`` is interpreted as greedy. Top-k is applied
before top-p (cheap filter first).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def greedy(logits: torch.Tensor) -> torch.Tensor:
    """Argmax along last dim. ``logits`` shape: (..., vocab) -> (...,)."""
    return logits.argmax(dim=-1)


def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    kth = logits.topk(top_k, dim=-1).values[..., -1:]
    return torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative = sorted_probs.cumsum(dim=-1)
    # Keep the smallest set whose cumulative prob >= top_p; the gating uses
    # (cumulative - sorted_probs) so the first token above the threshold is
    # always retained.
    drop = (cumulative - sorted_probs) > top_p
    sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
    return torch.empty_like(sorted_logits).scatter(-1, sorted_idx, sorted_logits)


def sample(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample next tokens from logits.

    logits: (..., vocab) — typically (batch, vocab) for the last decoded step.
    Returns: (...,) int64 token ids.
    """
    if temperature <= 0.0:
        return greedy(logits)

    logits = logits.float() / temperature
    logits = _apply_top_k(logits, top_k)
    logits = _apply_top_p(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    flat_probs = probs.reshape(-1, probs.shape[-1])
    flat_tokens = torch.multinomial(flat_probs, num_samples=1, generator=generator)
    return flat_tokens.reshape(probs.shape[:-1])


def sample_from_dist(
    probs: torch.Tensor, *, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Sample from a precomputed probability distribution. Used by spec-decode
    to sample corrected tokens from the (target - draft) residual."""
    flat_probs = probs.reshape(-1, probs.shape[-1])
    flat_tokens = torch.multinomial(flat_probs, num_samples=1, generator=generator)
    return flat_tokens.reshape(probs.shape[:-1])
