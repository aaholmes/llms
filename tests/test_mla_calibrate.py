"""Tests for the calibration pipeline.

The calibration pipeline registers forward pre-hooks on each ``block.attn``
to capture the post-norm-1 activation x (the input to Q/K/V projections),
accumulates Σ xᵀ x in fp64, and divides by the total token count to produce
per-layer covariances C_ℓ used by the activation-aware SVD.

Verifies the pipeline on the small (Qwen3-0.6B) draft model:
  - covariance shape (hidden, hidden) per layer; fp64 dtype
  - covariance is symmetric and PSD
  - token count matches the sum of prompt lengths
  - re-running the same prompts produces the same covariances
  - hook removal stops further accumulation
  - calling covariances() with no tokens raises
"""

from __future__ import annotations

import pytest
import torch

from engine.model import Qwen3Model
from engine.weights import load_weights
from mla.calibrate import CovarianceCollector, collect_covariances


@pytest.fixture(scope="module")
def small_model(draft_model_id: str):
    """Load Qwen3-0.6B once on CPU at fp32 for cheap, deterministic calibration."""
    loaded = load_weights(draft_model_id, dtype=torch.float32, device="cpu")
    return Qwen3Model.from_loaded(loaded).eval()


@pytest.fixture
def small_prompts() -> list[torch.Tensor]:
    return [
        torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long),
        torch.tensor([[9, 10, 11, 12]], dtype=torch.long),
        torch.tensor([[13, 14, 15, 16, 17, 18]], dtype=torch.long),
    ]


@pytest.mark.requires_draft
def test_covariance_shape(small_model, small_prompts):
    covs = collect_covariances(small_model, small_prompts)
    assert len(covs) == small_model.cfg.num_hidden_layers
    for c in covs:
        assert c.shape == (small_model.cfg.hidden_size, small_model.cfg.hidden_size)
        assert c.dtype == torch.float64


@pytest.mark.requires_draft
def test_covariance_symmetric(small_model, small_prompts):
    covs = collect_covariances(small_model, small_prompts)
    for i, c in enumerate(covs):
        asym = (c - c.T).abs().max().item()
        assert asym < 1e-10, f"layer {i} covariance asymmetry {asym:.2e}"


@pytest.mark.requires_draft
def test_covariance_psd(small_model, small_prompts):
    """Eigenvalues are non-negative (allow tiny negative values from round-off)."""
    covs = collect_covariances(small_model, small_prompts)
    for i, c in enumerate(covs):
        min_eig = torch.linalg.eigvalsh(c).min().item()
        assert min_eig > -1e-10, f"layer {i} negative eigval {min_eig:.2e}"


@pytest.mark.requires_draft
def test_token_count_correct(small_model, small_prompts):
    expected = sum(p.shape[1] for p in small_prompts)
    with CovarianceCollector(small_model) as col:
        for ids in small_prompts:
            cache = small_model.alloc_cache(ids.shape[1])
            with torch.inference_mode():
                small_model(ids, cache, start_pos=0)
        assert col.token_count == expected


@pytest.mark.requires_draft
def test_determinism(small_model, small_prompts):
    """Same prompts → same covariance to fp64 precision."""
    covs1 = collect_covariances(small_model, small_prompts)
    covs2 = collect_covariances(small_model, small_prompts)
    for i, (c1, c2) in enumerate(zip(covs1, covs2)):
        diff = (c1 - c2).abs().max().item()
        assert diff < 1e-10, f"layer {i} non-deterministic: max diff {diff:.2e}"


@pytest.mark.requires_draft
def test_remove_hooks_stops_accumulation(small_model, small_prompts):
    col = CovarianceCollector(small_model)

    ids = small_prompts[0]
    cache = small_model.alloc_cache(ids.shape[1])
    with torch.inference_mode():
        small_model(ids, cache, start_pos=0)
    count_before = col.token_count

    col.remove_hooks()

    cache = small_model.alloc_cache(ids.shape[1])
    with torch.inference_mode():
        small_model(ids, cache, start_pos=0)
    assert col.token_count == count_before


@pytest.mark.requires_draft
def test_empty_raises(small_model):
    with CovarianceCollector(small_model) as col:
        with pytest.raises(RuntimeError):
            col.covariances()


@pytest.mark.requires_draft
def test_default_device_follows_model(small_model, small_prompts):
    """Default ``accumulator_device=None`` resolves to the model's device.

    Keeps the per-iter ``x_flatᵀ @ x_flat`` reduction co-located with the
    activations rather than round-tripping over PCIe.
    """
    model_device = next(small_model.parameters()).device
    with CovarianceCollector(small_model) as col:
        for ids in small_prompts:
            cache = small_model.alloc_cache(ids.shape[1])
            with torch.inference_mode():
                small_model(ids, cache, start_pos=0)
        covs = col.covariances()
    for c in covs:
        assert c.device == model_device


@pytest.mark.requires_draft
def test_explicit_cpu_override_still_works(small_model, small_prompts):
    """Explicit ``accumulator_device='cpu'`` keeps accumulators on CPU
    even when the model would otherwise dictate the device."""
    with CovarianceCollector(small_model, accumulator_device="cpu") as col:
        for ids in small_prompts:
            cache = small_model.alloc_cache(ids.shape[1])
            with torch.inference_mode():
                small_model(ids, cache, start_pos=0)
        covs = col.covariances()
    for c in covs:
        assert c.device == torch.device("cpu")


# --- chat-formatted calibration prompts ---------------------------------------

def test_pack_prompts_truncates_and_stops_at_the_token_budget() -> None:
    from mla.calibrate import pack_prompts

    prompts = pack_prompts(
        [[1] * 3, [2] * 10, [3] * 4, [4] * 4], total_tokens=12, max_tokens=6
    )

    assert [p.shape for p in prompts] == [(1, 3), (1, 6), (1, 3)]
    assert sum(p.shape[1] for p in prompts) == 12
    assert prompts[1].dtype == torch.long and prompts[1][0, 0] == 2


def test_pack_prompts_returns_everything_when_the_budget_is_not_reached() -> None:
    from mla.calibrate import pack_prompts

    prompts = pack_prompts([[1, 2], [3]], total_tokens=100, max_tokens=8)
    assert [p.tolist() for p in prompts] == [[[1, 2]], [[3]]]
