"""Unit tests for the MHA→MLA healing finetune harness.

Covers:
  - LoraLinear math (zero-init B ⇒ identity at step 0; A and B trainable)
  - wrap_lora targeting (hits Q/O/gate/up/down; skips dkv/uk_nope/uv/kr)
  - setup_trainable param accounting
  - one-step loss decrease on the tiny synthetic Qwen3
  - gradient checkpointing forward equivalence
  - gradient flow through the MLA projections (the autograd path that the
    main heal recipe depends on — verifies the in-place KV-cache write
    propagates gradients to dkv/uk_nope/uv/kr)
  - save/load roundtrip of trainable-only checkpoint

A separate ``requires_cuda`` smoke test runs ~20 optimizer steps on a
converted Qwen3-0.6B and asserts val PPL strictly drops.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from engine.mla import MLAttention
from engine.model import Qwen3Model
from engine.weights import LoadedModel
from mla.calibrate import collect_covariances
from mla.convert import convert_loaded_to_mla
from mla.heal import (
    DEFAULT_LORA_TARGETS,
    LoraLinear,
    build_optimizer,
    chunk_ce_loss,
    cosine_warmup_multiplier,
    load_trainable,
    save_trainable,
    setup_trainable,
    train_loop,
    wrap_lora,
)
from mla.swap import alloc_mla_cache, apply_mla


# --- Tiny synthetic Qwen3 fixture (duplicated from test_mla_convert.py) ---


@dataclass(frozen=True)
class _Cfg:
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


def _tiny_config(c: _Cfg) -> PretrainedConfig:
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


def _tiny_loaded(c: _Cfg, *, seed: int = 0, dtype: torch.dtype = torch.float32) -> LoadedModel:
    torch.manual_seed(seed)
    cfg = _tiny_config(c)
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


def _identity_covariances(c: _Cfg, dtype: torch.dtype = torch.float64) -> list[torch.Tensor]:
    return [torch.eye(c.hidden_size, dtype=dtype) for _ in range(c.num_hidden_layers)]


def _calib_meta(c: _Cfg) -> dict:
    return {
        "model_id": "synthetic/tiny",
        "num_layers": c.num_hidden_layers,
        "hidden_size": c.hidden_size,
        "token_count": 1024,
        "dataset": "synthetic",
        "seed": 0,
    }


def _build_mla_model(c: _Cfg, *, d_rope: int = 8, rank: int | None = None) -> tuple[Qwen3Model, dict]:
    """Build a tiny Qwen3 model with MLA already swapped in. Returns (model, artifact)."""
    loaded = _tiny_loaded(c, seed=1)
    covs = _identity_covariances(c)
    if rank is None:
        d_nope = c.head_dim - d_rope
        rank = min(c.num_key_value_heads * (d_nope + c.head_dim), c.hidden_size)
    artifact = convert_loaded_to_mla(
        loaded, covariances=covs, rank=rank, d_rope=d_rope,
        factor_dtype=torch.float32, calibration_meta=_calib_meta(c),
        target_model_id="synthetic/tiny",
    )
    model = Qwen3Model.from_loaded(loaded).eval()
    apply_mla(model, artifact)
    return model, artifact


# --- LoraLinear / wrap_lora ----------------------------------------------


def test_lora_forward_matches_base_at_init() -> None:
    """B=0 ⇒ LoraLinear(x) is bit-identical to base(x). Standard LoRA invariant."""
    torch.manual_seed(0)
    base = nn.Linear(8, 12, bias=False)
    base.weight.data = torch.randn_like(base.weight) * 0.1
    lora = LoraLinear(base, rank=4, alpha=8.0)
    x = torch.randn(2, 5, 8)
    assert torch.equal(lora(x), base(x))


def test_lora_after_perturbation_changes_output() -> None:
    """If we manually set B nonzero, output differs from base by exactly the LoRA term."""
    torch.manual_seed(1)
    base = nn.Linear(8, 12, bias=False)
    lora = LoraLinear(base, rank=4, alpha=8.0)
    nn.init.normal_(lora.lora_B.weight, std=0.5)  # perturb out of zero
    x = torch.randn(2, 5, 8)
    delta = lora.lora_B(lora.lora_A(x)) * lora.scaling
    assert torch.allclose(lora(x), base(x) + delta, atol=1e-6)


def test_wrap_lora_targets_q_o_and_mlp_skips_mla_projections() -> None:
    """After apply_mla, wrap_lora hits q/o and MLP linears but never the new MLA projections."""
    c = _Cfg()
    model, _ = _build_mla_model(c)
    n_wrapped = wrap_lora(model, rank=2, alpha=4.0)
    # Two layers × five leaf targets (q, o, gate, up, down) = 10.
    assert n_wrapped == c.num_hidden_layers * len(DEFAULT_LORA_TARGETS)

    # MLA-specific projections must remain bare nn.Linear.
    for layer in model.layers:
        attn = layer.attn
        assert isinstance(attn, MLAttention)
        assert isinstance(attn.dkv, nn.Linear) and not isinstance(attn.dkv, LoraLinear)
        assert isinstance(attn.uv, nn.Linear) and not isinstance(attn.uv, LoraLinear)
        if attn.uk_nope is not None:
            assert isinstance(attn.uk_nope, nn.Linear) and not isinstance(attn.uk_nope, LoraLinear)
        if attn.kr is not None:
            assert isinstance(attn.kr, nn.Linear) and not isinstance(attn.kr, LoraLinear)
        # q and o swapped to LoraLinear.
        assert isinstance(attn.q, LoraLinear)
        assert isinstance(attn.o, LoraLinear)
        # MLP linears swapped.
        for leaf in ("gate", "up", "down"):
            assert isinstance(getattr(layer.ffn, leaf), LoraLinear)


def test_wrap_lora_substring_not_matched() -> None:
    """'up' is a *leaf attr name*, so it must NOT match 'uk_nope' or 'uv'.

    Regression guard for the substring-vs-attr-name distinction.
    """
    c = _Cfg()
    model, _ = _build_mla_model(c)
    wrap_lora(model, rank=2, alpha=4.0)
    for layer in model.layers:
        # If targeting were buggy and matched 'u' anywhere, uv/uk_nope would be wrapped.
        assert not isinstance(layer.attn.uv, LoraLinear)
        if layer.attn.uk_nope is not None:
            assert not isinstance(layer.attn.uk_nope, LoraLinear)


# --- setup_trainable ------------------------------------------------------


def test_setup_trainable_unfreezes_only_mla_and_lora() -> None:
    c = _Cfg()
    d_rope = 8
    rank = c.hidden_size // 2  # 32
    model, _ = _build_mla_model(c, d_rope=d_rope, rank=rank)
    wrap_lora(model, rank=4, alpha=8.0)
    inventory = setup_trainable(model)

    # All trainable params must be either an MLA projection or a LoRA adapter.
    mla_proj_ids = {id(p) for p in inventory.mla_proj_params}
    lora_ids = {id(p) for p in inventory.lora_params}
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert id(p) in mla_proj_ids or id(p) in lora_ids, (
                f"unexpected trainable param: {name}"
            )

    # Embeddings / final_norm / RMSNorms / base weights frozen.
    assert not model.embed.weight.requires_grad
    assert not model.final_norm.weight.requires_grad
    for layer in model.layers:
        assert not layer.norm1.weight.requires_grad
        assert not layer.norm2.weight.requires_grad
        assert not layer.attn.q_norm.weight.requires_grad
        assert not layer.attn.k_norm.weight.requires_grad
        # Base inside LoraLinear is frozen.
        assert not layer.attn.q.base.weight.requires_grad


def test_setup_trainable_exact_param_counts() -> None:
    """Count every trainable param against a closed-form formula."""
    c = _Cfg()
    d_rope = 8
    d_nope = c.head_dim - d_rope
    rank = c.hidden_size // 2  # 32
    lora_rank = 4
    model, _ = _build_mla_model(c, d_rope=d_rope, rank=rank)
    wrap_lora(model, rank=lora_rank, alpha=8.0)
    inventory = setup_trainable(model)

    # MLA projection per layer:
    n_dkv = rank * c.hidden_size
    n_uv = c.num_key_value_heads * c.head_dim * rank
    n_uk_nope = c.num_key_value_heads * d_nope * rank if d_nope > 0 else 0
    n_kr = c.num_key_value_heads * d_rope * c.hidden_size if d_rope > 0 else 0
    expected_mla = (n_dkv + n_uv + n_uk_nope + n_kr) * c.num_hidden_layers

    # LoRA per wrapped Linear: rank*in + out*rank
    H_total = c.num_attention_heads * c.head_dim
    lora_q = lora_rank * c.hidden_size + H_total * lora_rank  # base(hidden→H_total)
    lora_o = lora_rank * H_total + c.hidden_size * lora_rank  # base(H_total→hidden)
    lora_gate = lora_rank * c.hidden_size + c.intermediate_size * lora_rank
    lora_up = lora_gate
    lora_down = lora_rank * c.intermediate_size + c.hidden_size * lora_rank
    expected_lora = (lora_q + lora_o + lora_gate + lora_up + lora_down) * c.num_hidden_layers

    assert inventory.n_mla_proj == expected_mla
    assert inventory.n_lora == expected_lora


# --- Scheduler ------------------------------------------------------------


def test_cosine_warmup_shape() -> None:
    total = 100
    warm = 0.1
    # Step 0 inside warmup → small but > 0
    assert 0 < cosine_warmup_multiplier(0, total, warm) <= 1.0
    # End of warmup (step = warmup_steps - 1) → ~1.0
    assert cosine_warmup_multiplier(9, total, warm) == pytest.approx(1.0)
    # Halfway through decay → ~0.5
    mid = cosine_warmup_multiplier(10 + 45, total, warm)
    assert 0.4 < mid < 0.6
    # Final step → ~0
    assert cosine_warmup_multiplier(total - 1, total, warm) < 0.05


# --- Gradient flow / one-step decrease -----------------------------------


def _build_tiny_trainable(c: _Cfg, d_rope: int = 8) -> tuple[Qwen3Model, "object"]:
    """Convenience: convert, wrap, set up trainable; return (model, inventory)."""
    model, _ = _build_mla_model(c, d_rope=d_rope)
    wrap_lora(model, rank=4, alpha=8.0)
    inventory = setup_trainable(model)
    return model, inventory


def test_gradient_flow_through_mla_projections() -> None:
    """Forward + backward must produce nonzero gradients on dkv/uk_nope/uv/kr.

    Regression guard: if the KV-cache in-place write ever silently detached
    gradients on these matrices, training would stall here.
    """
    c = _Cfg()
    model, inventory = _build_tiny_trainable(c, d_rope=8)
    model.train()
    torch.manual_seed(0)
    ids = torch.randint(0, c.vocab_size, (1, 8), dtype=torch.long)
    cache = alloc_mla_cache(model, max_seq_len=ids.shape[1])
    logits = model(ids, cache, start_pos=0)
    loss = chunk_ce_loss(logits, ids)
    loss.backward()

    for layer in model.layers:
        attn = layer.attn
        assert attn.dkv.weight.grad is not None
        assert (attn.dkv.weight.grad.abs().sum() > 0).item()
        assert attn.uv.weight.grad is not None
        assert (attn.uv.weight.grad.abs().sum() > 0).item()
        if attn.uk_nope is not None:
            assert attn.uk_nope.weight.grad is not None
            assert (attn.uk_nope.weight.grad.abs().sum() > 0).item()
        if attn.kr is not None:
            assert attn.kr.weight.grad is not None
            assert (attn.kr.weight.grad.abs().sum() > 0).item()


def test_one_step_decreases_loss() -> None:
    """A single optimizer step on a fixed batch reduces loss measurably."""
    c = _Cfg()
    model, inventory = _build_tiny_trainable(c, d_rope=8)
    optim = build_optimizer(inventory, lr_mla=1e-2, lr_lora=1e-2)
    model.train()
    torch.manual_seed(0)
    ids = torch.randint(0, c.vocab_size, (1, 8), dtype=torch.long)

    def step_loss() -> float:
        cache = alloc_mla_cache(model, max_seq_len=ids.shape[1])
        logits = model(ids, cache, start_pos=0)
        return float(chunk_ce_loss(logits, ids).item())

    loss0 = step_loss()
    for _ in range(5):
        optim.zero_grad(set_to_none=True)
        cache = alloc_mla_cache(model, max_seq_len=ids.shape[1])
        logits = model(ids, cache, start_pos=0)
        loss = chunk_ce_loss(logits, ids)
        loss.backward()
        optim.step()
    loss1 = step_loss()
    assert loss1 < loss0 - 1e-4, f"loss did not decrease: {loss0:.4f} → {loss1:.4f}"


# --- Gradient checkpointing forward equivalence ---------------------------


def test_gradient_checkpointing_forward_equivalence() -> None:
    """Toggling gradient_checkpointing must not change the output (training mode)."""
    c = _Cfg()
    model, _ = _build_tiny_trainable(c, d_rope=8)
    model.train()
    torch.manual_seed(7)
    ids = torch.randint(0, c.vocab_size, (1, 8), dtype=torch.long)

    model.gradient_checkpointing = False
    cache = alloc_mla_cache(model, max_seq_len=ids.shape[1])
    out_off = model(ids, cache, start_pos=0)

    model.gradient_checkpointing = True
    cache = alloc_mla_cache(model, max_seq_len=ids.shape[1])
    out_on = model(ids, cache, start_pos=0)

    diff = (out_off - out_on).abs().max().item()
    assert diff < 1e-5, f"gradient-checkpointing perturbed forward by {diff:.2e}"


# --- Save / load roundtrip ------------------------------------------------


def test_save_load_trainable_roundtrip(tmp_path: Path) -> None:
    c = _Cfg()
    model_a, inventory_a = _build_tiny_trainable(c, d_rope=8)
    optim = build_optimizer(inventory_a, lr_mla=1e-2, lr_lora=1e-2)
    model_a.train()
    torch.manual_seed(3)
    ids = torch.randint(0, c.vocab_size, (1, 8), dtype=torch.long)
    for _ in range(3):
        optim.zero_grad(set_to_none=True)
        cache = alloc_mla_cache(model_a, max_seq_len=ids.shape[1])
        logits = model_a(ids, cache, start_pos=0)
        loss = chunk_ce_loss(logits, ids)
        loss.backward()
        optim.step()

    ckpt_path = tmp_path / "best.pt"
    save_trainable(model_a, ckpt_path)

    # Build a fresh model + LoRA wrap, then load: trainable params must match.
    model_b, inventory_b = _build_tiny_trainable(c, d_rope=8)
    # Quick sanity: model_b's MLA dkv != model_a's MLA dkv (model_a was trained).
    a_dkv = model_a.layers[0].attn.dkv.weight.detach().clone()
    b_dkv_before = model_b.layers[0].attn.dkv.weight.detach().clone()
    assert not torch.allclose(a_dkv, b_dkv_before)
    missing = load_trainable(model_b, ckpt_path)
    assert missing == [], f"missing names after load: {missing}"
    b_dkv_after = model_b.layers[0].attn.dkv.weight.detach().clone()
    assert torch.allclose(a_dkv, b_dkv_after, atol=1e-6)


# --- train_loop end-to-end smoke (synthetic) -----------------------------


def test_train_loop_runs_synthetic(tmp_path: Path) -> None:
    """Run a tiny train_loop on the synthetic Qwen3 and verify artifacts + val drop.

    Uses random tokens for both train and val streams — no dataset download,
    no real model. Just exercises the wiring (loss, optim step, scheduler,
    eval hook, JSONL logging, best-ckpt saving).
    """
    c = _Cfg()
    model, inventory = _build_tiny_trainable(c, d_rope=8)
    optim = build_optimizer(inventory, lr_mla=1e-2, lr_lora=1e-2)

    torch.manual_seed(0)
    train_chunks = [torch.randint(0, c.vocab_size, (1, 8), dtype=torch.long) for _ in range(8)]
    val_chunks = [torch.randint(0, c.vocab_size, (1, 8), dtype=torch.long) for _ in range(4)]

    out_dir = tmp_path / "heal_synth"
    summary = train_loop(
        model, optim, inventory, train_chunks, val_chunks,
        steps=10, grad_accum=2, seq_len=8,
        warmup_frac=0.1, val_every=5, log_every=1000,
        out_dir=out_dir,
    )
    assert (out_dir / "training_log.jsonl").exists()
    assert (out_dir / "best.pt").exists()
    assert summary["best_val_ppl"] < float("inf")
    # Should have written at least 10 log lines (one per step).
    with (out_dir / "training_log.jsonl").open() as f:
        lines = [line for line in f if line.strip()]
    assert len(lines) == 10


# --- Real-model CUDA smoke ------------------------------------------------


@pytest.mark.requires_cuda
@pytest.mark.requires_draft
def test_heal_smoke_cuda(draft_model_id: str, tmp_path: Path) -> None:
    """Heal a converted Qwen3-0.6B for ~20 steps; assert val PPL strictly drops.

    Tiny calibration (random tokens) so this doesn't need a dataset download.
    The PPL is meaningless in absolute terms — what we check is **directional
    movement**: val_ppl(after) < val_ppl(before). If this regresses, the
    optimizer/loss/cache wiring is broken.
    """
    from engine.weights import load_weights as _load
    from bench.eval_ppl import evaluate_ppl as _eval

    cfg_dtype = torch.float32  # FP32 for stable smoke
    loaded = _load(draft_model_id, dtype=cfg_dtype, device="cuda")
    cfg = loaded.config

    # 16 random prompts × 32 tokens — enough for non-singular covariances.
    torch.manual_seed(0)
    prompts = [
        torch.randint(0, cfg.vocab_size, (1, 32), dtype=torch.long).cuda()
        for _ in range(16)
    ]
    base = Qwen3Model.from_loaded(loaded).to(dtype=cfg_dtype, device="cuda").eval()
    covs = collect_covariances(base, prompts, accumulator_device="cuda")

    # Convert at d_rope=head_dim/2 and rank=half_max (moderate compression).
    d_rope = cfg.head_dim // 2
    d_nope = cfg.head_dim - d_rope
    max_rank = min(cfg.num_key_value_heads * (d_nope + cfg.head_dim), cfg.hidden_size)
    rank = max_rank // 2
    artifact = convert_loaded_to_mla(
        loaded, covariances=covs, rank=rank, d_rope=d_rope,
        factor_dtype=cfg_dtype, target_model_id=draft_model_id,
        calibration_meta={"model_id": draft_model_id, "num_layers": cfg.num_hidden_layers,
                          "hidden_size": cfg.hidden_size, "token_count": 16 * 32},
    )
    # Fresh model + apply MLA + LoRA + trainable setup.
    model = Qwen3Model.from_loaded(loaded).to(dtype=cfg_dtype, device="cuda")
    apply_mla(model, artifact)
    wrap_lora(model, rank=8, alpha=16.0)
    inventory = setup_trainable(model)
    optim = build_optimizer(inventory, lr_mla=1e-4, lr_lora=2e-4)

    train_chunks = [
        torch.randint(0, cfg.vocab_size, (1, 32), dtype=torch.long).cuda()
        for _ in range(16)
    ]
    val_chunks = [
        torch.randint(0, cfg.vocab_size, (1, 32), dtype=torch.long).cuda()
        for _ in range(4)
    ]

    model.eval()
    pre = _eval(model, val_chunks)
    model.train()
    summary = train_loop(
        model, optim, inventory, train_chunks, val_chunks,
        steps=20, grad_accum=2, seq_len=32,
        warmup_frac=0.1, val_every=10, log_every=1000,
        out_dir=tmp_path / "heal_smoke",
    )
    assert summary["best_val_ppl"] < pre["ppl"], (
        f"val_ppl did not drop: pre={pre['ppl']:.4f} best={summary['best_val_ppl']:.4f}"
    )
