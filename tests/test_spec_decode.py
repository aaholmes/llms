"""Spec-decode correctness: with greedy verify and greedy correction, the
output sequence must match plain greedy decoding from the target model
token-for-token. This is the core invariant of speculative decoding.
"""

from __future__ import annotations

import pytest
import torch
from transformers import PretrainedConfig

from engine.model import Qwen3Model
from engine.sampler import greedy
from engine.spec_decode import spec_decode_round, speculative_generate
from engine.weights import LoadedModel, load_weights


PROMPTS = [
    [100, 200, 300, 400, 500, 600, 700, 800],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [1234, 5678, 9012],
    [42, 42, 42, 42],
]


@pytest.fixture(scope="module")
def draft_loaded(draft_model_id):
    return load_weights(draft_model_id, dtype=torch.bfloat16, device="cpu")


@pytest.fixture(scope="module")
def draft_model(draft_loaded):
    # FP32 for the bit-exact test: BF16 batched-vs-sequential matmul reductions
    # are non-associative and occasionally flip argmax, causing spec output to
    # diverge from plain greedy by a few tokens even when the algorithm is
    # correct. FP32 makes the comparison stable.
    return Qwen3Model.from_loaded(draft_loaded).to(dtype=torch.float32).eval()


def _greedy_generate(model: Qwen3Model, prompt_ids: torch.Tensor, n_new: int) -> list[int]:
    """Reference: plain greedy generation using only the model and our cache."""
    cache = model.alloc_cache(prompt_ids.shape[1] + n_new + 4)
    with torch.inference_mode():
        logits = model(prompt_ids, cache, start_pos=0)
        next_tok = greedy(logits[:, -1, :]).unsqueeze(0)
        out = [int(next_tok.item())]
        for _ in range(n_new - 1):
            logits = model(next_tok, cache)
            next_tok = greedy(logits[:, -1, :]).unsqueeze(0)
            out.append(int(next_tok.item()))
    return out


def _tiny_model(seed: int = 0) -> Qwen3Model:
    """Tiny synthetic Qwen3-shaped model (mirrors tests/test_mla_convert.py)."""
    torch.manual_seed(seed)
    vocab, hidden, layers, n_heads, n_kv, head_dim, inter = 32, 64, 2, 4, 2, 16, 128
    cfg = PretrainedConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        head_dim=head_dim,
        intermediate_size=inter,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        attention_bias=False,
    )
    state: dict[str, torch.Tensor] = {
        "embed.weight": torch.randn(vocab, hidden),
        "final_norm.weight": torch.ones(hidden),
    }
    H, KV = n_heads * head_dim, n_kv * head_dim
    for i in range(layers):
        state[f"layers.{i}.norm1.weight"] = torch.ones(hidden)
        state[f"layers.{i}.norm2.weight"] = torch.ones(hidden)
        state[f"layers.{i}.attn.q.weight"] = torch.randn(H, hidden) * 0.1
        state[f"layers.{i}.attn.k.weight"] = torch.randn(KV, hidden) * 0.1
        state[f"layers.{i}.attn.v.weight"] = torch.randn(KV, hidden) * 0.1
        state[f"layers.{i}.attn.o.weight"] = torch.randn(hidden, H) * 0.1
        state[f"layers.{i}.attn.q_norm.weight"] = torch.ones(head_dim)
        state[f"layers.{i}.attn.k_norm.weight"] = torch.ones(head_dim)
        state[f"layers.{i}.ffn.gate.weight"] = torch.randn(inter, hidden) * 0.1
        state[f"layers.{i}.ffn.up.weight"] = torch.randn(inter, hidden) * 0.1
        state[f"layers.{i}.ffn.down.weight"] = torch.randn(hidden, inter) * 0.1
    return Qwen3Model.from_loaded(LoadedModel(state=state, config=cfg)).eval()


def test_spec_decode_round_rejects_empty_pending():
    """Empty pending violates the round invariant: pending must hold at least
    the bridge token (last-known token) for the draft to condition on.
    """
    model = _tiny_model()
    target_cache = model.alloc_cache(32)
    draft_cache = model.alloc_cache(32)
    with pytest.raises(ValueError, match="bridge token"):
        spec_decode_round(
            target=model,
            draft=model,
            target_cache=target_cache,
            draft_cache=draft_cache,
            pending=[],
            K=4,
            device="cpu",
        )


@pytest.mark.requires_draft
@pytest.mark.parametrize("prompt", PROMPTS)
@pytest.mark.parametrize("K", [1, 3, 4, 5, 7])
def test_spec_decode_matches_greedy_self(draft_model, prompt, K):
    """Self-spec-decode (target == draft) must produce the same output as
    plain greedy decoding. This isolates the round structure / cache
    bookkeeping from any draft/target divergence — every drafted token
    should be accepted, exercising the all-accepted (bonus) code path.
    """
    n_new = 32
    prompt_t = torch.tensor([prompt], dtype=torch.long)

    with torch.inference_mode():
        ref = _greedy_generate(draft_model, prompt_t, n_new)
        spec, stats = speculative_generate(
            target=draft_model,
            draft=draft_model,
            prompt_ids=prompt_t,
            max_new_tokens=n_new,
            K=K,
        )

    assert spec == ref, (
        f"\n  prompt={prompt} K={K}"
        f"\n  spec={spec}"
        f"\n  ref ={ref}"
        f"\n  divergence at idx={next((i for i, (a, b) in enumerate(zip(spec, ref)) if a != b), None)}"
        f"\n  stats: {stats}"
    )
    # When target == draft, every drafted token should be accepted.
    assert stats.acceptance_rate == 1.0, f"self-spec-decode should accept all drafts, got {stats}"
