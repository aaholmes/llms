"""End-to-end correctness gate for the manual Qwen3 forward pass.

The engine must match HF ``AutoModelForCausalLM.generate(do_sample=False)``
token-for-token. We test this on Qwen3-0.6B (the draft model) because it fits
comfortably in M2 RAM. The Qwen3-4B target is exercised on the GPU desktop.

These tests are slow (load weights twice: once via our loader, once via HF) so
they're marked ``requires_draft`` and skipped if the model isn't cached.
"""

from __future__ import annotations

import pytest
import torch

from engine.model import Qwen3Model
from engine.weights import load_weights

PROMPTS = [
    [100, 200, 300, 400, 500, 600, 700, 800],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    [9999, 8888, 7777, 6666],
    [1234, 5678, 9012, 3456, 7890],
    [100],
    [42, 42, 42, 42, 42],
]


@pytest.fixture(scope="module")
def loaded(draft_model_id):
    return load_weights(draft_model_id, dtype=torch.bfloat16, device="cpu")


@pytest.fixture(scope="module")
def ours(loaded):
    return Qwen3Model.from_loaded(loaded).to(dtype=torch.bfloat16).eval()


@pytest.fixture(scope="module")
def hf_ref(draft_model_id):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        draft_model_id, dtype=torch.bfloat16
    ).eval()


@pytest.mark.requires_draft
def test_prefill_logits_match_hf(ours, hf_ref):
    """Prefill logits must be bit-equal (or near-bit-equal) to HF on the same input."""
    input_ids = torch.tensor([PROMPTS[0]])
    with torch.inference_mode():
        cache = ours.alloc_cache(max_seq_len=64)
        ours_logits = ours(input_ids, cache)
        hf_logits = hf_ref(input_ids, use_cache=False).logits

    diff = (ours_logits.float() - hf_logits.float()).abs()
    assert diff.max().item() < 1e-2, f"max abs diff = {diff.max().item():.6f}"
    assert ours_logits[0, -1].argmax() == hf_logits[0, -1].argmax()


@pytest.mark.requires_draft
@pytest.mark.parametrize("prompt", PROMPTS)
def test_greedy_generate_token_match(ours, hf_ref, prompt):
    """Greedy generation must match HF generate(do_sample=False) token-for-token."""
    n_new = 64
    input_ids = torch.tensor([prompt])
    max_seq = len(prompt) + n_new + 4

    with torch.inference_mode():
        cache = ours.alloc_cache(max_seq_len=max_seq)
        logits = ours(input_ids, cache)
        next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
        ours_new = [next_tok.item()]
        for _ in range(n_new - 1):
            logits = ours(next_tok, cache)
            next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
            ours_new.append(next_tok.item())

        hf_out = hf_ref.generate(
            input_ids, max_new_tokens=n_new, do_sample=False, use_cache=True
        )
        hf_new = hf_out[0, len(prompt) :].tolist()

    assert ours_new == hf_new, (
        f"\n  prompt={prompt}"
        f"\n  ours={ours_new}"
        f"\n  hf  ={hf_new}"
        f"\n  first divergence at idx={next((i for i, (a, b) in enumerate(zip(ours_new, hf_new)) if a != b), None)}"
    )
