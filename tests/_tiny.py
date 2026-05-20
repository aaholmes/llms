"""Tiny synthetic Qwen3-shaped fixture, shared across MLA tests.

Importers: ``tests/test_mla_convert.py``, ``tests/test_mla_heal.py``.

Lives under a leading-underscore module name so pytest doesn't try to collect
it as a test file.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from engine.weights import LoadedModel


@dataclass(frozen=True)
class Cfg:
    """Qwen3-shaped but tiny — fast enough that synthetic SVD tests run in <5 s."""

    vocab_size: int = 32
    hidden_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 16
    intermediate_size: int = 128
    max_position_embeddings: int = 64
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0


def tiny_config(c: Cfg) -> PretrainedConfig:
    return PretrainedConfig(
        vocab_size=c.vocab_size,
        hidden_size=c.hidden_size,
        num_hidden_layers=c.num_hidden_layers,
        num_attention_heads=c.num_attention_heads,
        num_key_value_heads=c.num_key_value_heads,
        head_dim=c.head_dim,
        intermediate_size=c.intermediate_size,
        max_position_embeddings=c.max_position_embeddings,
        rms_norm_eps=c.rms_norm_eps,
        tie_word_embeddings=True,
        rope_theta=c.rope_theta,
        attention_bias=False,
    )


def tiny_loaded(c: Cfg, *, seed: int = 0, dtype: torch.dtype = torch.float32) -> LoadedModel:
    """Random-weights ``LoadedModel`` matching ``c``. Q/K/V/O are non-square (Qwen3-style)."""
    torch.manual_seed(seed)
    cfg = tiny_config(c)
    state: dict[str, torch.Tensor] = {
        "embed.weight": torch.randn(c.vocab_size, c.hidden_size, dtype=dtype),
        "final_norm.weight": torch.ones(c.hidden_size, dtype=dtype),
    }
    H = c.num_attention_heads * c.head_dim
    KV = c.num_key_value_heads * c.head_dim
    for i in range(c.num_hidden_layers):
        state[f"layers.{i}.norm1.weight"] = torch.ones(c.hidden_size, dtype=dtype)
        state[f"layers.{i}.norm2.weight"] = torch.ones(c.hidden_size, dtype=dtype)
        state[f"layers.{i}.attn.q.weight"] = torch.randn(H, c.hidden_size, dtype=dtype) * 0.1
        state[f"layers.{i}.attn.k.weight"] = torch.randn(KV, c.hidden_size, dtype=dtype) * 0.1
        state[f"layers.{i}.attn.v.weight"] = torch.randn(KV, c.hidden_size, dtype=dtype) * 0.1
        state[f"layers.{i}.attn.o.weight"] = torch.randn(c.hidden_size, H, dtype=dtype) * 0.1
        state[f"layers.{i}.attn.q_norm.weight"] = torch.ones(c.head_dim, dtype=dtype)
        state[f"layers.{i}.attn.k_norm.weight"] = torch.ones(c.head_dim, dtype=dtype)
        state[f"layers.{i}.ffn.gate.weight"] = torch.randn(c.intermediate_size, c.hidden_size, dtype=dtype) * 0.1
        state[f"layers.{i}.ffn.up.weight"] = torch.randn(c.intermediate_size, c.hidden_size, dtype=dtype) * 0.1
        state[f"layers.{i}.ffn.down.weight"] = torch.randn(c.hidden_size, c.intermediate_size, dtype=dtype) * 0.1
    return LoadedModel(state=state, config=cfg)


def identity_covariances(c: Cfg, dtype: torch.dtype = torch.float64) -> list[torch.Tensor]:
    return [torch.eye(c.hidden_size, dtype=dtype) for _ in range(c.num_hidden_layers)]


def calib_meta(c: Cfg, *, model_id: str = "synthetic/tiny", token_count: int = 1024) -> dict:
    return {
        "model_id": model_id,
        "num_layers": c.num_hidden_layers,
        "hidden_size": c.hidden_size,
        "token_count": token_count,
        "dataset": "synthetic",
        "seed": 0,
    }
