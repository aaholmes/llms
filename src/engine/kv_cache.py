"""Contiguous per-layer KV cache for a single batch element.

Layout per layer: (max_batch, num_kv_heads, max_seq_len, head_dim).
The cache stores K and V projections (already RoPE'd in our forward pass).
``cur_len`` advances after each forward pass and is shared across layers.

Stage A keeps things simple: one cache instance per model, single batch
element. Paged variants and multi-batch scheduling are deferred per DESIGN.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class KVCache:
    k: list[torch.Tensor]  # per-layer (max_batch, num_kv_heads, max_seq_len, head_dim)
    v: list[torch.Tensor]
    cur_len: int = 0

    @classmethod
    def alloc(
        cls,
        *,
        num_layers: int,
        max_batch: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> "KVCache":
        shape = (max_batch, num_kv_heads, max_seq_len, head_dim)
        return cls(
            k=[torch.empty(shape, dtype=dtype, device=device) for _ in range(num_layers)],
            v=[torch.empty(shape, dtype=dtype, device=device) for _ in range(num_layers)],
        )

    @property
    def max_seq_len(self) -> int:
        return self.k[0].shape[2]

    def reset(self) -> None:
        self.cur_len = 0

    def truncate(self, new_len: int) -> None:
        """Roll back to ``new_len`` tokens (for spec-decode rejection)."""
        if new_len < 0 or new_len > self.cur_len:
            raise ValueError(f"truncate {new_len=} outside [0, {self.cur_len}]")
        self.cur_len = new_len
