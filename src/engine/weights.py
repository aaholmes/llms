"""Load Qwen3.5 weights from HuggingFace into our own state dict.

We map HF parameter names to a compact scheme that matches the modules in
``src/engine/model.py``:

    model.embed_tokens.weight                       -> embed.weight
    model.norm.weight                               -> final_norm.weight
    lm_head.weight                                  -> lm_head.weight
    model.layers.{i}.input_layernorm.weight         -> layers.{i}.norm1.weight
    model.layers.{i}.post_attention_layernorm.weight-> layers.{i}.norm2.weight
    model.layers.{i}.self_attn.{q,k,v,o}_proj.*     -> layers.{i}.attn.{q,k,v,o}.*
    model.layers.{i}.self_attn.{q,k}_norm.weight    -> layers.{i}.attn.{q,k}_norm.weight
    model.layers.{i}.mlp.{gate,up,down}_proj.weight -> layers.{i}.ffn.{gate,up,down}.weight

Strict by default: any HF tensor we don't recognize raises so the mapping table
gets extended explicitly rather than silently dropping weights.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, PretrainedConfig

_NAME_MAP: list[tuple[str, str]] = [
    (r"^model\.embed_tokens\.weight$", r"embed.weight"),
    (r"^model\.norm\.weight$", r"final_norm.weight"),
    (r"^lm_head\.weight$", r"lm_head.weight"),
    (r"^model\.layers\.(\d+)\.input_layernorm\.weight$", r"layers.\1.norm1.weight"),
    (r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$", r"layers.\1.norm2.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$", r"layers.\1.attn.q.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$", r"layers.\1.attn.k.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$", r"layers.\1.attn.v.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$", r"layers.\1.attn.o.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.q_norm\.weight$", r"layers.\1.attn.q_norm.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.k_norm\.weight$", r"layers.\1.attn.k_norm.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$", r"layers.\1.ffn.gate.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$", r"layers.\1.ffn.up.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$", r"layers.\1.ffn.down.weight"),
]

_COMPILED = [(re.compile(p), r) for p, r in _NAME_MAP]


@dataclass
class LoadedModel:
    state: dict[str, torch.Tensor]
    config: PretrainedConfig


def _remap(hf_name: str) -> str | None:
    for pat, repl in _COMPILED:
        if pat.match(hf_name):
            return pat.sub(repl, hf_name)
    return None


def _resolve_safetensor_files(local_dir: Path) -> list[Path]:
    index_path = local_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as f:
            index = json.load(f)
        return sorted({local_dir / fname for fname in index["weight_map"].values()})
    single = local_dir / "model.safetensors"
    if single.exists():
        return [single]
    raise FileNotFoundError(f"No safetensors found under {local_dir}")


def load_weights(
    model_id: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
) -> LoadedModel:
    """Download (if needed) and load safetensors weights for a Qwen3.5 model.

    Tensors are cast to ``dtype`` and placed on ``device``. The default keeps
    weights on CPU so callers can stage them onto an accelerator deliberately.
    """
    config = AutoConfig.from_pretrained(model_id)

    local_dir = Path(
        snapshot_download(
            model_id,
            allow_patterns=["*.safetensors", "*.safetensors.index.json", "*.json"],
        )
    )

    state: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []

    for path in _resolve_safetensor_files(local_dir):
        with safe_open(path, framework="pt", device="cpu") as f:
            for hf_name in f.keys():
                ours = _remap(hf_name)
                if ours is None:
                    unmapped.append(hf_name)
                    continue
                tensor = f.get_tensor(hf_name).to(dtype=dtype, device=device)
                state[ours] = tensor

    if unmapped:
        raise ValueError(
            "Unmapped HF parameter names (extend _NAME_MAP in weights.py): "
            f"{sorted(unmapped)}"
        )

    if "lm_head.weight" not in state and getattr(config, "tie_word_embeddings", False):
        state["lm_head.weight"] = state["embed.weight"]

    return LoadedModel(state=state, config=config)
