"""Streaming token-level perplexity evaluation against the engine.

Two pieces deliberately split:
  - ``chunk_nll(logits, target_ids)`` — pure-functional NLL accumulator for
    one chunk's predictions; testable without any model.
  - ``evaluate_ppl(model, chunks)`` — runs the model forward on each chunk,
    auto-detects MLA vs GQA cache, returns aggregated PPL.

Each chunk's first token contributes no NLL (no history); for a chunk of
length T we count ``T-1`` predicted positions.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from engine.mla import MLAttention
from engine.model import Qwen3Model
from mla.swap import alloc_mla_cache


def chunk_nll(
    logits: torch.Tensor, target_ids: torch.Tensor
) -> tuple[float, int]:
    """Sum of cross-entropy losses over predicted positions and the count.

    Parameters
    ----------
    logits : (1, T, vocab)
    target_ids : (1, T) long

    Returns
    -------
    nll_sum : float
        Σ_t -log P(target[t] | logits[t-1]) for t in [1, T).
    count : int
        ``T - 1`` (predicted positions; the first token has no history).
    """
    log_probs = F.log_softmax(logits[0, :-1].float(), dim=-1)
    targets = target_ids[0, 1:]
    nll = -log_probs.gather(-1, targets.to(log_probs.device).unsqueeze(-1)).squeeze(-1)
    return float(nll.sum().item()), int(nll.numel())


def _alloc_cache(model: Qwen3Model, max_seq_len: int):
    """Allocate the right cache type based on whether MLA has been swapped in."""
    if any(isinstance(m, MLAttention) for m in model.modules()):
        return alloc_mla_cache(model, max_seq_len)
    return model.alloc_cache(max_seq_len)


@torch.inference_mode()
def evaluate_ppl(
    model: Qwen3Model,
    chunks: list[torch.Tensor],
    *,
    progress_every: int | None = None,
) -> dict:
    """Run ``model`` forward on each chunk; aggregate per-token cross-entropy → PPL.

    Each ``chunks[i]`` is ``(1, T)`` long. Positions [1..T) contribute to the
    NLL. A throwaway cache is allocated per chunk and freed on completion.

    Returns ``{"ppl", "avg_nll", "token_count", "chunk_count", "wall_seconds"}``.
    """
    device = next(model.parameters()).device
    total_nll = 0.0
    total_count = 0
    t0 = time.time()
    for i, ids in enumerate(chunks):
        T = ids.shape[1]
        ids_dev = ids.to(device)
        cache = _alloc_cache(model, T)
        logits = model(ids_dev, cache, start_pos=0)
        nll_sum, count = chunk_nll(logits, ids_dev)
        total_nll += nll_sum
        total_count += count
        del cache, logits
        if progress_every and (i + 1) % progress_every == 0:
            running_ppl = float(torch.tensor(total_nll / total_count).exp().item())
            print(
                f"[ppl]   {i+1}/{len(chunks)} chunks "
                f"(running ppl={running_ppl:.4f}, t+{time.time()-t0:.1f}s)",
                flush=True,
            )

    avg_nll = total_nll / total_count
    return {
        "ppl": float(torch.tensor(avg_nll).exp().item()),
        "avg_nll": avg_nll,
        "token_count": total_count,
        "chunk_count": len(chunks),
        "wall_seconds": time.time() - t0,
    }
