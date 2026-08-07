"""Weight-only quantization: round-trip accuracy, swap hygiene, composition with MLA."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from engine.kv_cache import KVCache
from engine.model import Qwen3Model
from quant import QuantLinear, apply_weight_quant, dequantize, quant_memory_report
from quant.int8 import quantize_weight

from ._tiny import Cfg, tiny_loaded


@pytest.fixture
def tiny_model() -> Qwen3Model:
    c = Cfg()
    return Qwen3Model.from_loaded(tiny_loaded(c, seed=0)).eval()


def _fresh_cache(c: Cfg) -> KVCache:
    return KVCache.alloc(
        num_layers=c.num_hidden_layers,
        max_batch=1,
        num_kv_heads=c.num_key_value_heads,
        max_seq_len=c.max_position_embeddings,
        head_dim=c.head_dim,
        dtype=torch.float32,
        device="cpu",
    )


# --- quantize / dequantize ------------------------------------------------


@pytest.mark.parametrize("n_bits,group_size", [(8, None), (8, 16), (4, None), (4, 16)])
def test_round_trip_error_within_quantization_step(n_bits: int, group_size: int | None) -> None:
    """Dequantized weights land within half a step of the original, per group."""
    torch.manual_seed(0)
    W = torch.randn(32, 64)
    q, scale = quantize_weight(W, n_bits=n_bits, group_size=group_size)
    W_hat = dequantize(q, scale)

    # Broadcast each group's step back over its columns to bound elementwise.
    n_groups = scale.shape[-1]
    gs = W.shape[1] // n_groups
    step = scale.repeat_interleave(gs, dim=-1)
    assert torch.all((W - W_hat).abs() <= step / 2 + 1e-6)


def test_codes_stay_in_range() -> None:
    torch.manual_seed(0)
    W = torch.randn(16, 32) * 100
    for n_bits, qmax in ((8, 127), (4, 7)):
        q, _ = quantize_weight(W, n_bits=n_bits)
        assert q.dtype is torch.int8
        assert int(q.abs().max()) <= qmax


def test_finer_granularity_is_never_worse() -> None:
    """Grouped scales beat per-channel, and 8-bit beats 4-bit — the ladder holds."""
    torch.manual_seed(0)
    W = torch.randn(32, 64)

    def err(**kw) -> float:
        q, s = quantize_weight(W, **kw)
        return (W - dequantize(q, s)).abs().mean().item()

    assert err(n_bits=8, group_size=16) <= err(n_bits=8, group_size=None)
    assert err(n_bits=4, group_size=16) <= err(n_bits=4, group_size=None)
    assert err(n_bits=8, group_size=None) < err(n_bits=4, group_size=None)


def test_zero_group_round_trips_exactly() -> None:
    """An all-zero group must not produce NaN from a zero scale."""
    W = torch.zeros(4, 16)
    W[0, :8] = 1.0
    q, scale = quantize_weight(W, n_bits=8, group_size=8)
    W_hat = dequantize(q, scale)
    assert torch.isfinite(W_hat).all()
    assert torch.allclose(W_hat[1:], torch.zeros_like(W_hat[1:]))


def test_group_size_must_divide_in_features() -> None:
    with pytest.raises(ValueError, match="does not divide"):
        quantize_weight(torch.randn(4, 10), group_size=3)


def test_unsupported_bit_width_rejected() -> None:
    with pytest.raises(ValueError, match="n_bits"):
        quantize_weight(torch.randn(4, 8), n_bits=3)


# --- QuantLinear ----------------------------------------------------------


def test_quant_linear_matches_dequantized_reference() -> None:
    """forward() is exactly F.linear against the dequantized weight."""
    torch.manual_seed(0)
    lin = nn.Linear(64, 32, bias=False)
    ql = QuantLinear.from_linear(lin, n_bits=8)
    x = torch.randn(2, 5, 64)
    assert torch.allclose(ql(x), torch.nn.functional.linear(x, ql.weight), atol=1e-6)


def test_quant_linear_weight_property_preserves_introspection() -> None:
    """Downstream code reads ``.weight``/shape off layers; that must still work."""
    lin = nn.Linear(64, 32, bias=False)
    ql = QuantLinear.from_linear(lin)
    assert ql.weight.shape == lin.weight.shape
    assert ql.in_features == 64 and ql.out_features == 32
    # 8-bit per-channel on random weights stays close to the original.
    assert (ql.weight - lin.weight).abs().max() < 0.02


def test_quant_linear_rejects_bias() -> None:
    with pytest.raises(ValueError, match="bias-free"):
        QuantLinear.from_linear(nn.Linear(8, 4, bias=True))


# --- swap -----------------------------------------------------------------


def test_swap_replaces_projections_but_not_head(tiny_model: Qwen3Model) -> None:
    apply_weight_quant(tiny_model, n_bits=8)
    for block in tiny_model.layers:
        for name in ("q", "k", "v", "o"):
            assert isinstance(getattr(block.attn, name), QuantLinear)
        for name in ("gate", "up", "down"):
            assert isinstance(getattr(block.ffn, name), QuantLinear)
    if tiny_model.lm_head is not None:
        assert not isinstance(tiny_model.lm_head, QuantLinear)


def test_swap_is_idempotent(tiny_model: Qwen3Model) -> None:
    apply_weight_quant(tiny_model, n_bits=8)
    before = tiny_model.layers[0].attn.q.qweight.clone()
    apply_weight_quant(tiny_model, n_bits=8)
    assert torch.equal(tiny_model.layers[0].attn.q.qweight, before)


def test_include_filter_hits_ffn_only(tiny_model: Qwen3Model) -> None:
    apply_weight_quant(tiny_model, include=("gate", "up", "down"))
    block = tiny_model.layers[0]
    assert isinstance(block.ffn.gate, QuantLinear)
    assert not isinstance(block.attn.q, QuantLinear)


def test_8bit_swap_barely_perturbs_logits(tiny_model: Qwen3Model) -> None:
    """The headline sanity check: 8-bit weight-only is a small perturbation."""
    torch.manual_seed(0)
    c = Cfg()
    ids = torch.randint(0, c.vocab_size, (1, 8))
    with torch.no_grad():
        ref = tiny_model(ids, _fresh_cache(c))
        apply_weight_quant(tiny_model, n_bits=8)
        got = tiny_model(ids, _fresh_cache(c))
    rel = (got - ref).norm() / ref.norm()
    assert rel < 0.05, f"8-bit perturbed logits by {rel:.3%}"


def test_4bit_degrades_more_than_8bit() -> None:
    """Directional check — 4-bit is a strictly lossier block, which is the point."""
    torch.manual_seed(0)
    c = Cfg()
    ids = torch.randint(0, c.vocab_size, (1, 8))

    def logits_for(**kw) -> torch.Tensor:
        m = Qwen3Model.from_loaded(tiny_loaded(c, seed=0)).eval()
        with torch.no_grad():
            if kw:
                apply_weight_quant(m, **kw)
            return m(ids, _fresh_cache(c))

    ref = logits_for()
    d8 = (logits_for(n_bits=8) - ref).norm()
    d4 = (logits_for(n_bits=4) - ref).norm()
    assert d8 < d4


def test_memory_report_tracks_compression(tiny_model: Qwen3Model) -> None:
    apply_weight_quant(tiny_model, n_bits=8)
    rep = quant_memory_report(tiny_model)
    assert rep["n_quant_layers"] == len(tiny_model.layers) * 7  # q,k,v,o,gate,up,down
    # 8-bit against a bf16 baseline is ~2x, minus scale overhead.
    assert 1.8 < rep["compression"] < 2.0
    # Unpacked 4-bit storage must not be misreported as the packed size.
    m4 = Qwen3Model.from_loaded(tiny_loaded(Cfg(), seed=0)).eval()
    apply_weight_quant(m4, n_bits=4)
    rep4 = quant_memory_report(m4)
    assert rep4["compression"] > rep["compression"]
    assert rep4["allocated_bytes"] > rep4["quantized_bytes"]


# --- composition with MLA -------------------------------------------------


def test_quantizes_mla_projections_without_extra_code() -> None:
    """Stage C composes both lossy blocks; the generic walk must cover MLA."""
    from engine.mla import MLAttention

    c = Cfg()
    model = Qwen3Model.from_loaded(tiny_loaded(c, seed=0)).eval()
    mla = MLAttention(
        hidden_size=c.hidden_size,
        num_heads=c.num_attention_heads,
        num_kv_heads=c.num_key_value_heads,
        head_dim=c.head_dim,
        rms_eps=c.rms_norm_eps,
        rank=16,
        d_rope=8,
        max_position_embeddings=c.max_position_embeddings,
        rope_theta=c.rope_theta,
    )
    model.layers[0].attn = mla
    apply_weight_quant(model, n_bits=8)
    assert isinstance(model.layers[0].attn.dkv, QuantLinear)
    assert isinstance(model.layers[0].attn.uv, QuantLinear)
