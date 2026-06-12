"""Tests for the MLA conversion CLI and the apply_mla swap.

Goal: prove that ``convert_loaded_to_mla`` + ``apply_mla`` together produce
attention outputs algebraically equivalent to the original GQA attention,
given full-rank factors. Pure-synthetic tests using a tiny Qwen3-shaped
config — no HF download, no real models, fast (<5 s).
"""

from __future__ import annotations

import json
import logging

import pytest
import torch

from engine.model import Qwen3Model
from engine.weights import LoadedModel
from mla.convert import convert_loaded_to_mla, per_head_rope_pair_perm
from mla.swap import alloc_mla_cache, apply_mla

from ._tiny import Cfg, calib_meta, identity_covariances, tiny_loaded


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


# --- tests ----------------------------------------------------------------

def _max_joint_rank(c: Cfg, d_rope: int) -> int:
    """Highest rank at which joint-SVD reconstruction of [W_K_nope; W_V] is lossless."""
    d_nope = c.head_dim - d_rope
    return min(c.num_key_value_heads * (d_nope + c.head_dim), c.hidden_size)


@pytest.mark.parametrize("d_rope", [0, 4, 8, 16])
def test_factor_level_full_rank_lossless(d_rope: int) -> None:
    """At max joint rank with well-conditioned C, factors reconstruct W_K_nope, W_V losslessly.

    This is the math of the conversion in isolation — it doesn't require the
    runtime to behave any particular way. Partial-RoPE end-to-end equivalence
    against ``PartialRoPEAttention`` is covered by ``test_mla_attention.py``;
    here we check that the SVD primitive correctly factors the per-layer K/V.
    """
    c = Cfg()
    rank = _max_joint_rank(c, d_rope)
    d_nope = c.head_dim - d_rope
    loaded = tiny_loaded(c, seed=1, dtype=torch.float64)

    torch.manual_seed(3)
    covs = []
    for _ in range(c.num_hidden_layers):
        A = torch.randn(c.hidden_size, c.hidden_size, dtype=torch.float64)
        covs.append(A @ A.T + torch.eye(c.hidden_size, dtype=torch.float64))

    artifact = convert_loaded_to_mla(
        loaded,
        covariances=covs,
        rank=rank,
        d_rope=d_rope,
        factor_dtype=torch.float64,
        calibration_meta=calib_meta(c),
    )

    perm = per_head_rope_pair_perm(c.head_dim, d_rope)
    # Permuted rows of W_K (in original-coord terms): nope = first d_nope
    # of the permuted head, rope = last d_rope.
    nope_rows = [
        h * c.head_dim + perm[i]
        for h in range(c.num_key_value_heads)
        for i in range(d_nope)
    ]
    rope_rows = [
        h * c.head_dim + perm[d_nope + j]
        for h in range(c.num_key_value_heads)
        for j in range(d_rope)
    ]
    q_row_perm = [
        h * c.head_dim + perm[i]
        for h in range(c.num_attention_heads)
        for i in range(c.head_dim)
    ]

    assert artifact["meta"]["rope_convention"] == "original-pairs"

    for i in range(c.num_hidden_layers):
        W_K = loaded.state[f"layers.{i}.attn.k.weight"]
        W_V = loaded.state[f"layers.{i}.attn.v.weight"]
        prefix = f"layers.{i}.attn."
        W_dkv = artifact["state"][prefix + "dkv.weight"]
        W_uv = artifact["state"][prefix + "uv.weight"]

        # V round-trip: W_uv @ W_dkv ≈ W_V (V is *not* permuted).
        V_recon = W_uv @ W_dkv
        v_err = (V_recon - W_V).abs().max().item()
        assert v_err < 1e-9, f"layer {i}, d_rope={d_rope}: V residual {v_err:.2e}"

        if d_nope > 0:
            W_uk_nope = artifact["state"][prefix + "uk_nope.weight"]
            K_nope_recon = W_uk_nope @ W_dkv
            k_err = (K_nope_recon - W_K[nope_rows]).abs().max().item()
            assert k_err < 1e-9, f"layer {i}, d_rope={d_rope}: K-nope residual {k_err:.2e}"
        else:
            assert prefix + "uk_nope.weight" not in artifact["state"]

        if d_rope > 0:
            W_kr = artifact["state"][prefix + "kr.weight"]
            assert torch.equal(W_kr, W_K[rope_rows].contiguous()), (
                f"layer {i}: kr should be a bit-exact copy of the permuted rope rows"
            )
        else:
            assert prefix + "kr.weight" not in artifact["state"]

        # O is NOT permuted; q, q_norm, k_norm ARE permuted.
        assert torch.equal(
            artifact["state"][prefix + "o.weight"],
            loaded.state[prefix + "o.weight"],
        ), f"layer {i}: o.weight should be untouched"
        assert torch.equal(
            artifact["state"][prefix + "q.weight"],
            loaded.state[prefix + "q.weight"][q_row_perm],
        ), f"layer {i}: q.weight should be row-permuted"
        for name in ("q_norm.weight", "k_norm.weight"):
            assert torch.equal(
                artifact["state"][prefix + name],
                loaded.state[prefix + name][perm],
            ), f"layer {i}: {name} should be permuted"


@pytest.mark.parametrize("d_rope", [0, 8])
def test_factor_level_exact_low_rank_recovery(d_rope: int) -> None:
    """W_K_nope and W_V constructed from rank-r₀ factors → recovery at rank=r₀ is exact in fp64."""
    c = Cfg()
    rank = 16  # exact construction rank
    d_nope = c.head_dim - d_rope

    loaded = tiny_loaded(c, seed=7, dtype=torch.float64)

    perm = per_head_rope_pair_perm(c.head_dim, d_rope)
    nope_rows = [
        h * c.head_dim + perm[i]
        for h in range(c.num_key_value_heads)
        for i in range(d_nope)
    ]
    rope_rows = [
        h * c.head_dim + perm[d_nope + j]
        for h in range(c.num_key_value_heads)
        for j in range(d_rope)
    ]

    torch.manual_seed(7)
    for i in range(c.num_hidden_layers):
        W_dkv = torch.randn(rank, c.hidden_size, dtype=torch.float64)
        W_uk_nope = (
            torch.randn(c.num_key_value_heads * d_nope, rank, dtype=torch.float64)
            if d_nope > 0 else torch.zeros(0, rank, dtype=torch.float64)
        )
        W_uv = torch.randn(c.num_key_value_heads * c.head_dim, rank, dtype=torch.float64)
        W_kr = (
            torch.randn(c.num_key_value_heads * d_rope, c.hidden_size, dtype=torch.float64)
            if d_rope > 0 else torch.zeros(0, c.hidden_size, dtype=torch.float64)
        )

        # Nope rows at perm[:d_nope] (non-contiguous original-pair-respecting
        # positions); rope rows at perm[d_nope:].
        K_nope_full = W_uk_nope @ W_dkv
        W_K = torch.empty(c.num_key_value_heads * c.head_dim, c.hidden_size, dtype=torch.float64)
        for h in range(c.num_key_value_heads):
            for ii in range(d_nope):
                W_K[h * c.head_dim + perm[ii]] = K_nope_full[h * d_nope + ii]
            for jj in range(d_rope):
                W_K[h * c.head_dim + perm[d_nope + jj]] = W_kr[h * d_rope + jj]
        W_V = W_uv @ W_dkv

        loaded.state[f"layers.{i}.attn.k.weight"] = W_K
        loaded.state[f"layers.{i}.attn.v.weight"] = W_V

    torch.manual_seed(3)
    covs = []
    for _ in range(c.num_hidden_layers):
        A = torch.randn(c.hidden_size, c.hidden_size, dtype=torch.float64)
        covs.append(A @ A.T + torch.eye(c.hidden_size, dtype=torch.float64))

    artifact = convert_loaded_to_mla(
        loaded,
        covariances=covs,
        rank=rank,
        d_rope=d_rope,
        factor_dtype=torch.float64,
        calibration_meta=calib_meta(c),
    )

    for i in range(c.num_hidden_layers):
        prefix = f"layers.{i}.attn."
        W_dkv_out = artifact["state"][prefix + "dkv.weight"]
        W_uv_out = artifact["state"][prefix + "uv.weight"]

        W_V = loaded.state[f"layers.{i}.attn.v.weight"]
        v_err = (W_uv_out @ W_dkv_out - W_V).abs().max().item()
        assert v_err < 1e-9, f"layer {i}, d_rope={d_rope}: V residual {v_err:.2e}"

        if d_nope > 0:
            W_uk_nope_out = artifact["state"][prefix + "uk_nope.weight"]
            W_K = loaded.state[f"layers.{i}.attn.k.weight"]
            k_err = (W_uk_nope_out @ W_dkv_out - W_K[nope_rows]).abs().max().item()
            assert k_err < 1e-9, f"layer {i}, d_rope={d_rope}: K-nope residual {k_err:.2e}"


def test_full_rank_full_rope_end_to_end_matches_baseline() -> None:
    """At d_rope=head_dim and max joint rank, MLA matches Qwen3 Attention end-to-end.

    This is the only configuration where MLA's partial-RoPE collapses to
    full-RoPE-on-the-whole-head, so the comparison against vanilla Qwen3
    ``Attention`` is apples-to-apples. For d_rope<head_dim the two attentions
    differ by design (that's the point), so end-to-end equivalence is not
    expected and is not tested here.
    """
    c = Cfg()
    loaded = tiny_loaded(c, seed=1)
    covs = identity_covariances(c)
    d_rope = c.head_dim
    rank = _max_joint_rank(c, d_rope)

    artifact = convert_loaded_to_mla(
        loaded, covariances=covs, rank=rank, d_rope=d_rope,
        factor_dtype=torch.float32, calibration_meta=calib_meta(c),
    )

    torch.manual_seed(11)
    prompt = torch.randint(0, c.vocab_size, (1, 8), dtype=torch.long)

    out_baseline = _baseline_logits(loaded, prompt)
    out_mla = _mla_logits(loaded, artifact, prompt)

    diff = (out_baseline - out_mla).abs().max().item()
    assert diff < 1e-4, f"max abs diff {diff:.2e}"


def test_artifact_serialization_bit_exact(tmp_path) -> None:
    """Convert → torch.save → torch.load → every weight bit-exact; sidecar json valid."""
    c = Cfg()
    loaded = tiny_loaded(c, seed=2)
    covs = identity_covariances(c)

    artifact = convert_loaded_to_mla(
        loaded,
        covariances=covs,
        rank=c.hidden_size // 2,
        d_rope=8,
        factor_dtype=torch.float32,
        calibration_meta=calib_meta(c),
    )

    out_path = tmp_path / "artifact.pt"
    meta_path = tmp_path / "artifact.meta.json"
    torch.save(artifact, out_path)
    with meta_path.open("w") as f:
        json.dump(artifact["meta"], f)

    reloaded = torch.load(out_path, map_location="cpu", weights_only=False)
    assert set(reloaded["state"].keys()) == set(artifact["state"].keys())
    for k, v in artifact["state"].items():
        assert torch.equal(reloaded["state"][k], v), f"key {k} differs after roundtrip"

    with meta_path.open() as f:
        meta_loaded = json.load(f)
    assert meta_loaded["rank"] == artifact["meta"]["rank"]
    assert meta_loaded["d_rope"] == artifact["meta"]["d_rope"]


def test_calibration_model_id_mismatch_raises() -> None:
    """A calibration artifact with the wrong model_id must be rejected up front."""
    c = Cfg()
    loaded = tiny_loaded(c, seed=3)
    covs = identity_covariances(c)

    bad_meta = calib_meta(c, model_id="other/model")

    with pytest.raises(ValueError, match="model_id"):
        convert_loaded_to_mla(
            loaded,
            covariances=covs,
            rank=c.hidden_size,
            d_rope=8,
            factor_dtype=torch.float32,
            calibration_meta=bad_meta,
            target_model_id="synthetic/tiny",
        )


def test_calibration_layer_count_mismatch_raises() -> None:
    """Number of covariance matrices must match the model's layer count."""
    c = Cfg()
    loaded = tiny_loaded(c, seed=4)
    short_covs = identity_covariances(c)[:-1]

    with pytest.raises(ValueError, match="layer"):
        convert_loaded_to_mla(
            loaded,
            covariances=short_covs,
            rank=c.hidden_size,
            d_rope=8,
            factor_dtype=torch.float32,
            calibration_meta=calib_meta(c),
        )


def test_invalid_rank_raises() -> None:
    c = Cfg()
    loaded = tiny_loaded(c, seed=5)
    covs = identity_covariances(c)

    with pytest.raises(ValueError, match="rank"):
        convert_loaded_to_mla(
            loaded, covariances=covs, rank=c.hidden_size + 1, d_rope=8,
            factor_dtype=torch.float32, calibration_meta=calib_meta(c),
        )


def test_invalid_d_rope_raises() -> None:
    c = Cfg()
    loaded = tiny_loaded(c, seed=6)
    covs = identity_covariances(c)

    with pytest.raises(ValueError, match="d_rope"):
        convert_loaded_to_mla(
            loaded, covariances=covs, rank=c.hidden_size, d_rope=c.head_dim + 1,
            factor_dtype=torch.float32, calibration_meta=calib_meta(c),
        )


def test_apply_mla_idempotent() -> None:
    """Calling apply_mla twice on the same model is a no-op the second time."""
    c = Cfg()
    loaded = tiny_loaded(c, seed=8)
    covs = identity_covariances(c)

    artifact = convert_loaded_to_mla(
        loaded, covariances=covs, rank=_max_joint_rank(c, 8), d_rope=8,
        factor_dtype=torch.float32, calibration_meta=calib_meta(c),
    )
    model = Qwen3Model.from_loaded(loaded).eval()
    apply_mla(model, artifact)
    # Second call should be a no-op (already swapped).
    apply_mla(model, artifact)
    # Still works
    cache = alloc_mla_cache(model, max_seq_len=8)
    torch.manual_seed(0)
    prompt = torch.randint(0, c.vocab_size, (1, 4), dtype=torch.long)
    with torch.inference_mode():
        out = model(prompt, cache, start_pos=0)
    assert out.shape == (1, 4, c.vocab_size)


def test_artifact_metadata_records_diagnostics() -> None:
    """Per-layer diagnostics (discarded energy, ridge used) appear in meta."""
    c = Cfg()
    loaded = tiny_loaded(c, seed=9)
    covs = identity_covariances(c)

    artifact = convert_loaded_to_mla(
        loaded, covariances=covs, rank=c.hidden_size // 2, d_rope=8,
        factor_dtype=torch.float32, calibration_meta=calib_meta(c),
    )
    diag = artifact["meta"]["per_layer_diagnostics"]
    assert len(diag) == c.num_hidden_layers
    for entry in diag:
        assert "discarded_energy" in entry
        assert entry["discarded_energy"] >= 0.0


def test_calibration_missing_model_id_warns(caplog) -> None:
    """A calibration artifact without a model_id skips the match check loudly."""
    c = _Cfg()
    loaded = _tiny_loaded(c, seed=10)
    covs = _identity_covariances(c)

    meta = _calib_meta(c)
    del meta["model_id"]

    with caplog.at_level(logging.WARNING, logger="mla.convert"):
        convert_loaded_to_mla(
            loaded, covariances=covs, rank=_max_joint_rank(c, 8), d_rope=8,
            factor_dtype=torch.float32, calibration_meta=meta,
            target_model_id="synthetic/tiny",
        )
    assert any("no model_id" in r.message for r in caplog.records)
