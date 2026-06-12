"""Manual Qwen3 forward pass (no transformers.Qwen3ForCausalLM in the hot path).

Architecture: pre-norm decoder-only transformer with GQA, RoPE, RMSNorm,
SwiGLU FFN. Same shape as Llama-3 except for QK-norm (handled in attention.py)
and non-square Q/K/V/O projections.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from .attention import Attention, RMSNorm, build_rope_tables
from .kv_cache import KVCache
from .weights import LoadedModel


def _rope_theta(cfg: PretrainedConfig) -> float:
    rp = getattr(cfg, "rope_parameters", None)
    if isinstance(rp, dict) and "rope_theta" in rp:
        return float(rp["rope_theta"])
    return float(getattr(cfg, "rope_theta", 10000.0))


class FFN(nn.Module):
    """SwiGLU feed-forward: down(silu(gate(x)) * up(x))."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DecoderBlock(nn.Module):
    def __init__(self, cfg: PretrainedConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.attn = Attention(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            rms_eps=cfg.rms_norm_eps,
        )
        self.norm2 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.ffn = FFN(cfg.hidden_size, cfg.intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv_cache: KVCache,
        layer_idx: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int,
    ) -> torch.Tensor:
        h = x + self.attn(
            self.norm1(x),
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            cos=cos,
            sin=sin,
            start_pos=start_pos,
        )
        h = h + self.ffn(self.norm2(h))
        return h


class Qwen3Model(nn.Module):
    # When True, each DecoderBlock's forward is wrapped in
    # torch.utils.checkpoint.checkpoint so activations are recomputed during
    # backward — trades ~30% throughput for ~5× lower peak activation memory.
    # Off by default; the inference path is unchanged.
    gradient_checkpointing: bool = False

    def __init__(self, cfg: PretrainedConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderBlock(cfg) for _ in range(cfg.num_hidden_layers)]
        )
        self.final_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        if not cfg.tie_word_embeddings:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        else:
            self.lm_head = None

        cos, sin = build_rope_tables(
            cfg.head_dim,
            cfg.max_position_embeddings,
            base=_rope_theta(cfg),
            dtype=torch.float32,
        )
        # Computed in float32 for precision; as registered buffers they follow
        # module-level .to(dtype=...), so forward needs no per-call cast.
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache,
        start_pos: int | None = None,
    ) -> torch.Tensor:
        if start_pos is None:
            start_pos = kv_cache.cur_len
        T = input_ids.shape[1]

        h = self.embed(input_ids)
        cos = self.rope_cos[start_pos : start_pos + T]
        sin = self.rope_sin[start_pos : start_pos + T]

        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint(
                    layer,
                    h,
                    kv_cache=kv_cache,
                    layer_idx=i,
                    cos=cos,
                    sin=sin,
                    start_pos=start_pos,
                    use_reentrant=False,
                )
                continue
            h = layer(
                h,
                kv_cache=kv_cache,
                layer_idx=i,
                cos=cos,
                sin=sin,
                start_pos=start_pos,
            )

        h = self.final_norm(h)
        if self.lm_head is None:
            logits = h @ self.embed.weight.T
        else:
            logits = self.lm_head(h)

        kv_cache.cur_len = start_pos + T
        return logits

    @classmethod
    def from_loaded(cls, loaded: LoadedModel) -> "Qwen3Model":
        model = cls(loaded.config)
        state = dict(loaded.state)
        if loaded.config.tie_word_embeddings:
            state.pop("lm_head.weight", None)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"State dict mismatch: missing={list(missing)}, unexpected={list(unexpected)}"
            )
        return model

    def alloc_cache(
        self,
        max_seq_len: int,
        *,
        max_batch: int = 1,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> KVCache:
        if dtype is None:
            dtype = self.embed.weight.dtype
        if device is None:
            device = self.embed.weight.device
        return KVCache.alloc(
            num_layers=self.cfg.num_hidden_layers,
            max_batch=max_batch,
            num_kv_heads=self.cfg.num_key_value_heads,
            max_seq_len=max_seq_len,
            head_dim=self.cfg.head_dim,
            dtype=dtype,
            device=device,
        )
