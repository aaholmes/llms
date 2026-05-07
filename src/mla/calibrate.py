"""Calibration pipeline for activation-aware SVD.

Streams a calibration corpus through the engine and accumulates per-layer
input covariance C_ℓ = (1/N) Σ_t x_t x_tᵀ, where x_t is the post-norm-1
activation feeding the attention's Q/K/V projections (the same x feeds all
three since they share the input).

Hooks attach to ``block.attn`` as forward pre-hooks; ``args[0]`` is the
post-norm-1 tensor of shape (B, T, hidden). Accumulation is fp64 on a chosen
device — by default the same device as the model, so the per-iter
``x_flatᵀ @ x_flat`` reduction stays on-GPU and avoids ~5 MB-per-layer
transfers on every prompt.

Usage:
    with CovarianceCollector(model) as col:
        for ids in prompt_ids:
            cache = model.alloc_cache(ids.shape[1])
            with torch.inference_mode():
                model(ids, cache, start_pos=0)
        covs = col.covariances()
"""

from __future__ import annotations

import torch
from torch import nn

from engine.model import Qwen3Model


class CovarianceCollector:
    """Accumulates per-layer pre-attention input covariance via forward hooks.

    Parameters
    ----------
    model : Qwen3Model
        The engine model whose attention inputs will be captured.
    accumulator_device : str | torch.device | None
        Where the fp64 accumulators live. ``None`` (the default) resolves to
        the model's device, which keeps the per-iter ``x_flatᵀ @ x_flat``
        reduction on the same device as the activations and avoids a
        round-trip per layer per prompt. Pass ``"cpu"`` explicitly to
        offload accumulators from the model device — only needed when VRAM
        is tight (~5 MB per layer × 36 layers ≈ 200 MB on Qwen3-4B).
    """

    def __init__(
        self,
        model: Qwen3Model,
        *,
        accumulator_device: str | torch.device | None = None,
    ) -> None:
        self.num_layers = len(model.layers)
        self.hidden = model.cfg.hidden_size
        if accumulator_device is None:
            accumulator_device = next(model.parameters()).device
        self._device = torch.device(accumulator_device)
        self._accumulators: list[torch.Tensor] = [
            torch.zeros(self.hidden, self.hidden, dtype=torch.float64, device=self._device)
            for _ in range(self.num_layers)
        ]
        self._token_count: int = 0
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

        for i, layer in enumerate(model.layers):
            handle = layer.attn.register_forward_pre_hook(
                self._make_hook(i),
                with_kwargs=True,
            )
            self._hooks.append(handle)

    def _make_hook(self, layer_idx: int):
        def hook(module: nn.Module, args: tuple, kwargs: dict) -> None:
            x = args[0] if args else kwargs["x"]
            x_flat = (
                x.reshape(-1, x.shape[-1])
                .to(dtype=torch.float64, device=self._device)
            )
            self._accumulators[layer_idx].add_(x_flat.T @ x_flat)
            if layer_idx == 0:
                self._token_count += x_flat.shape[0]

        return hook

    @property
    def token_count(self) -> int:
        return self._token_count

    def covariances(self) -> list[torch.Tensor]:
        """Return [C_ℓ] = [accumulator / N], normalized per layer."""
        if self._token_count == 0:
            raise RuntimeError(
                "no tokens accumulated — run at least one forward pass under hooks"
            )
        return [acc / self._token_count for acc in self._accumulators]

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self) -> "CovarianceCollector":
        return self

    def __exit__(self, *exc) -> None:
        self.remove_hooks()


def collect_covariances(
    model: Qwen3Model,
    prompt_ids: list[torch.Tensor],
    *,
    accumulator_device: str | torch.device | None = None,
    progress_every: int | None = None,
) -> list[torch.Tensor]:
    """Run prompts through ``model`` under hooks; return per-layer covariances.

    Each prompt is a (1, T) token tensor. A throwaway KV cache is allocated
    per prompt and freed immediately. Output is fp64 on ``accumulator_device``
    (defaulting to the model's device).
    """
    with CovarianceCollector(model, accumulator_device=accumulator_device) as col:
        for i, ids in enumerate(prompt_ids):
            cache = model.alloc_cache(ids.shape[1])
            with torch.inference_mode():
                model(ids, cache, start_pos=0)
            del cache
            if progress_every and (i + 1) % progress_every == 0:
                print(
                    f"[calibrate] {i + 1}/{len(prompt_ids)} prompts; "
                    f"tokens accumulated: {col.token_count}"
                )
        return col.covariances()


def load_wikitext103_chunks(
    n_samples: int,
    *,
    chunk_tokens: int = 256,
    tokenizer_id: str = "Qwen/Qwen3-4B",
    seed: int = 42,
    split: str = "train",
) -> list[torch.Tensor]:
    """Sample ``n_samples`` chunks of ``chunk_tokens`` tokens from WikiText-103.

    Concatenates non-empty articles (in a deterministic shuffled order with
    ``seed``) and slices the stream into fixed-length chunks. First-time use
    triggers a ~180 MB dataset download via ``datasets``.

    Returns a list of (1, chunk_tokens) long tensors.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset("wikitext", "wikitext-103-v1", split=split)
    tok = AutoTokenizer.from_pretrained(tokenizer_id)

    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(ds), generator=rng).tolist()

    chunks: list[torch.Tensor] = []
    buffer: list[int] = []
    for idx in indices:
        text = ds[idx]["text"]
        if not text or not text.strip():
            continue
        # WikiText-103 article headings start with " = ..."; skip them so the
        # calibration distribution matches body prose.
        stripped = text.strip()
        if stripped.startswith("=") and stripped.endswith("="):
            continue
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        buffer.extend(ids.tolist())
        while len(buffer) >= chunk_tokens:
            chunk = torch.tensor(buffer[:chunk_tokens], dtype=torch.long).unsqueeze(0)
            chunks.append(chunk)
            buffer = buffer[chunk_tokens:]
            if len(chunks) >= n_samples:
                return chunks
    return chunks
