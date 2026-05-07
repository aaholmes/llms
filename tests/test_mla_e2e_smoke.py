"""End-to-end smoke tests for the MLA conversion CLI on real Qwen3-0.6B.

Two cases:
  - **d_rope = head_dim, max joint rank**: the only configuration where MLA
    is algebraically equivalent to baseline GQA (RoPE on the whole head, no
    nope subspace), so we can assert max abs logit diff is small. This
    exercises every wire in the pipeline against real weights.
  - **d_rope = head_dim/2, half rank**: the configuration we actually
    intend to ship. Math diverges from baseline by design (partial RoPE
    + V compression), so we only check that logits are finite and
    plausible — runtime correctness of the partial-RoPE math is covered
    by ``test_mla_attention.py``.

Both share a tiny ad-hoc calibration (random token IDs, ~30 prompts × 32
tokens) — sufficient for the SVD's Cholesky to succeed with the default
ridge. We avoid the WikiText-103 path here so the smoke test doesn't need
a dataset download.
"""

from __future__ import annotations

import pytest
import torch

from engine.model import Qwen3Model
from engine.weights import LoadedModel, load_weights
from mla.calibrate import collect_covariances
from mla.convert import convert_loaded_to_mla
from mla.swap import alloc_mla_cache, apply_mla


@pytest.fixture(scope="module")
def small_loaded(draft_model_id: str) -> LoadedModel:
    """Load Qwen3-0.6B weights once for the module."""
    return load_weights(draft_model_id, dtype=torch.float32, device="cpu")


@pytest.fixture(scope="module")
def small_covariances(small_loaded: LoadedModel) -> list[torch.Tensor]:
    """Cheap calibration: 24 prompts × 32 random token IDs through Qwen3-0.6B.

    Random tokens give a non-singular covariance under the embedding +
    norms, which is all the SVD needs. Real PPL evaluation uses the
    proper WikiText-103 calibration; this is just enough to get past
    Cholesky.
    """
    cfg = small_loaded.config
    model = Qwen3Model.from_loaded(small_loaded).eval()
    torch.manual_seed(0)
    prompts = [
        torch.randint(0, cfg.vocab_size, (1, 32), dtype=torch.long)
        for _ in range(24)
    ]
    return collect_covariances(model, prompts, accumulator_device="cpu")


def _baseline_logits(loaded: LoadedModel, prompt_ids: torch.Tensor) -> torch.Tensor:
    model = Qwen3Model.from_loaded(loaded).eval()
    cache = model.alloc_cache(prompt_ids.shape[1])
    with torch.inference_mode():
        return model(prompt_ids, cache, start_pos=0)


def _mla_logits(loaded: LoadedModel, artifact: dict, prompt_ids: torch.Tensor) -> torch.Tensor:
    model = Qwen3Model.from_loaded(loaded).eval()
    apply_mla(model, artifact)
    cache = alloc_mla_cache(model, max_seq_len=prompt_ids.shape[1])
    with torch.inference_mode():
        return model(prompt_ids, cache, start_pos=0)


@pytest.mark.requires_draft
def test_full_rope_full_rank_matches_baseline(
    small_loaded: LoadedModel,
    small_covariances: list[torch.Tensor],
    draft_model_id: str,
) -> None:
    """At d_rope=head_dim and max joint rank, MLA == baseline through Qwen3Model."""
    cfg = small_loaded.config
    d_rope = cfg.head_dim
    d_nope = 0
    rank = min(cfg.num_key_value_heads * (d_nope + cfg.head_dim), cfg.hidden_size)

    artifact = convert_loaded_to_mla(
        small_loaded,
        covariances=small_covariances,
        rank=rank,
        d_rope=d_rope,
        factor_dtype=torch.float32,
        target_model_id=draft_model_id,
        calibration_meta={"model_id": draft_model_id, "num_layers": cfg.num_hidden_layers,
                          "hidden_size": cfg.hidden_size, "token_count": 24 * 32},
    )

    torch.manual_seed(7)
    prompt = torch.randint(0, cfg.vocab_size, (1, 16), dtype=torch.long)

    out_baseline = _baseline_logits(small_loaded, prompt)
    out_mla = _mla_logits(small_loaded, artifact, prompt)

    diff = (out_baseline - out_mla).abs().max().item()
    assert diff < 1e-3, (
        f"MLA at full rank with d_rope=head_dim should match baseline; "
        f"got max abs logit diff {diff:.2e}"
    )


@pytest.mark.requires_draft
def test_partial_rope_half_rank_runs_and_outputs_are_finite(
    small_loaded: LoadedModel,
    small_covariances: list[torch.Tensor],
    draft_model_id: str,
) -> None:
    """Realistic configuration: partial RoPE, half rank.

    Doesn't compare to baseline (math diverges by design — that's what we want).
    Just verifies the pipeline runs end-to-end and produces sane logits.
    """
    cfg = small_loaded.config
    d_rope = cfg.head_dim // 2
    d_nope = cfg.head_dim - d_rope
    max_rank = min(cfg.num_key_value_heads * (d_nope + cfg.head_dim), cfg.hidden_size)
    rank = max_rank // 2

    artifact = convert_loaded_to_mla(
        small_loaded,
        covariances=small_covariances,
        rank=rank,
        d_rope=d_rope,
        factor_dtype=torch.float32,
        target_model_id=draft_model_id,
        calibration_meta={"model_id": draft_model_id, "num_layers": cfg.num_hidden_layers,
                          "hidden_size": cfg.hidden_size, "token_count": 24 * 32},
    )

    torch.manual_seed(11)
    prompt = torch.randint(0, cfg.vocab_size, (1, 16), dtype=torch.long)

    out_mla = _mla_logits(small_loaded, artifact, prompt)
    assert out_mla.shape == (1, 16, cfg.vocab_size)
    assert torch.isfinite(out_mla).all(), "MLA logits contain non-finite values"
    # Sanity on magnitudes — Qwen3 logits are usually within a few hundred,
    # certainly under 1e4. Catches catastrophic numerical blowup.
    assert out_mla.abs().max().item() < 1e4

    # And per-layer diagnostics should be present (sweep driver consumes them).
    diags = artifact["meta"]["per_layer_diagnostics"]
    assert len(diags) == cfg.num_hidden_layers
    assert all(d["discarded_energy"] >= 0 for d in diags)
