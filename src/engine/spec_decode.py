"""Sequential speculative decoding (Stage A — no async).

Round structure:
  1. Draft K tokens autoregressively from a known "bridge" prefix.
  2. Target verifies all K in a single batched forward pass.
  3. Accept the longest matching prefix; sample a corrected (or bonus) token
     from the target distribution.

State invariant between rounds: both KV caches at the same ``cur_len``, with
``pending`` holding committed output-tail tokens whose KV is not yet in either
cache. ``pending`` is usually length 1 (the corrected token). After an
all-accepted round it's length 2: the last drafted token's KV is in target
but not in draft, so we drop it from target and re-feed both via pending.

Stage A uses **greedy verify**: a drafted token is accepted iff
``argmax(target_logits[i]) == drafted[i]``. Probabilistic verify per Leviathan
et al. is a follow-on; greedy verify is sufficient for the correctness gate
(its outputs match plain greedy generation token-for-token).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .kv_cache import KVCache
from .model import Qwen3Model
from .sampler import greedy


@dataclass
class SpecDecodeStats:
    rounds: int = 0
    drafted_tokens: int = 0
    accepted_drafted: int = 0
    bonus_rounds: int = 0  # rounds where all K drafted were accepted

    @property
    def acceptance_rate(self) -> float:
        if self.drafted_tokens == 0:
            return 0.0
        return self.accepted_drafted / self.drafted_tokens


def prefill(
    model: Qwen3Model, cache: KVCache, prompt_ids: torch.Tensor
) -> torch.Tensor:
    """Run the prompt through ``model``; return the last logit (predicts the
    first generated position). ``cache.cur_len`` advances to ``prompt_ids.shape[1]``.
    """
    logits = model(prompt_ids, cache, start_pos=0)
    return logits[:, -1, :]


def spec_decode_round(
    *,
    target: Qwen3Model,
    draft: Qwen3Model,
    target_cache: KVCache,
    draft_cache: KVCache,
    pending: list[int],
    K: int,
    device: torch.device | str,
) -> tuple[list[int], list[int], int]:
    """One spec-decode round (greedy verify).

    Returns (new_tokens, new_pending, num_accepted_drafted) where ``new_tokens``
    are the tokens to append to the output sequence (drafted-accepted +
    corrected-or-bonus). Caches are mutated in place.

    Both caches must enter with equal ``cur_len``. ``pending`` is the tail of
    the output sequence that has not yet been written to either cache.
    """
    if target_cache.cur_len != draft_cache.cur_len:
        raise RuntimeError(
            f"Caches desynced: target={target_cache.cur_len}, draft={draft_cache.cur_len}"
        )
    L = target_cache.cur_len
    P = len(pending)
    assert P >= 1

    # --- Draft phase: feed pending then sample K tokens autoregressively. ---
    pending_t = torch.tensor([pending], dtype=torch.long, device=device)
    draft_logits = draft(pending_t, draft_cache)  # (1, P, V)
    next_tok = greedy(draft_logits[:, -1, :])     # (1,)
    drafted: list[int] = [int(next_tok.item())]
    for _ in range(K - 1):
        draft_logits = draft(next_tok.unsqueeze(0), draft_cache)
        next_tok = greedy(draft_logits[:, -1, :])
        drafted.append(int(next_tok.item()))
    # draft_cache.cur_len is now L + P + (K - 1)

    # --- Target phase: feed pending + drafted in a single batched pass. ---
    target_input = torch.tensor([pending + drafted], dtype=torch.long, device=device)
    target_logits = target(target_input, target_cache)  # (1, P + K, V)
    # target_cache.cur_len is now L + P + K

    # logits[i] (output index) predicts the token at sequence position L + i + 1.
    # drafted[j] sits at sequence position L + P + j and is predicted by
    # target_logits at output index P - 1 + j.
    verify_logits = target_logits[0, P - 1 : P - 1 + K, :]  # (K, V)
    bonus_logit = target_logits[0, P - 1 + K, :]            # (V,)

    target_pred = greedy(verify_logits).tolist()  # length K

    # Longest matching prefix
    A = K
    for j in range(K):
        if target_pred[j] != drafted[j]:
            A = j
            break

    if A < K:
        c = int(greedy(verify_logits[A]).item())
        new_tokens = drafted[:A] + [c]
        new_pending = [c]
        target_truncate_to = L + P + A
        draft_truncate_to = L + P + A
    else:
        c = int(greedy(bonus_logit).item())
        new_tokens = drafted + [c]
        # Caches are out of sync after all-accepted: target has d_K's KV at
        # position L+P+K-1, draft does not (draft only fed up through d_{K-1}).
        # Drop d_K from target so both are at L+P+K-1, then re-feed [d_K, c]
        # via pending in the next round.
        new_pending = [drafted[K - 1], c]
        target_truncate_to = L + P + K - 1
        draft_truncate_to = L + P + K - 1

    target_cache.truncate(target_truncate_to)
    draft_cache.truncate(draft_truncate_to)

    return new_tokens, new_pending, A


def speculative_generate(
    *,
    target: Qwen3Model,
    draft: Qwen3Model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    K: int = 4,
    max_seq_len: int | None = None,
    device: torch.device | str | None = None,
) -> tuple[list[int], SpecDecodeStats]:
    """Sequential speculative decoding from ``prompt_ids`` for up to
    ``max_new_tokens`` new tokens. Greedy verify, greedy correction.

    Returns (newly_generated_token_ids, stats).
    """
    if device is None:
        device = prompt_ids.device

    if max_seq_len is None:
        max_seq_len = prompt_ids.shape[1] + max_new_tokens + K + 4

    target_cache = target.alloc_cache(max_seq_len, device=device)
    draft_cache = draft.alloc_cache(max_seq_len, device=device)

    last_target_logit = prefill(target, target_cache, prompt_ids)
    _ = prefill(draft, draft_cache, prompt_ids)

    bridge = int(greedy(last_target_logit).item())
    output: list[int] = [bridge]
    pending: list[int] = [bridge]

    stats = SpecDecodeStats()

    while len(output) < max_new_tokens:
        new_tokens, pending, A = spec_decode_round(
            target=target,
            draft=draft,
            target_cache=target_cache,
            draft_cache=draft_cache,
            pending=pending,
            K=K,
            device=device,
        )
        output.extend(new_tokens)
        stats.rounds += 1
        stats.drafted_tokens += K
        stats.accepted_drafted += A
        if A == K:
            stats.bonus_rounds += 1

    return output[:max_new_tokens], stats
