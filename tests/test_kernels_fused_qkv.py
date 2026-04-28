"""Tests for the fused QKV-projection Triton kernel (Stage A.5b).

Math contract (pinned in step 1 below): the kernel computes the partial
expression that lives at the top of ``Attention.forward``
(``src/engine/attention.py:114-116``), without QK-norm or RoPE:

    q = x @ Wq.T  (..., n_q)
    k = x @ Wk.T  (..., n_k)
    v = x @ Wv.T  (..., n_k)

where ``x`` has shape ``(..., K)`` and the three weights have shape
``(n_q, K)``, ``(n_k, K)``, ``(n_k, K)`` (n_v == n_k for GQA).

The TDD ladder for Stage A.5b is documented in
``~/.claude/plans/ok-make-a-detailed-keen-clover.md``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


def _eager_qkv(
    x: torch.Tensor,
    Wq: torch.Tensor,
    Wk: torch.Tensor,
    Wv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference implementation of the kernel's math contract.

    x: (..., K)
    Wq: (n_q, K), Wk: (n_k, K), Wv: (n_k, K) — nn.Linear weight layout (out, in)
    Returns: (q, k, v) of shapes (..., n_q), (..., n_k), (..., n_k).
    """
    return x @ Wq.T, x @ Wk.T, x @ Wv.T


# ----- Step 1: pin the math contract (no Triton, no GPU) --------------------


@pytest.mark.parametrize(
    "shape",
    [
        # (B, T, K, n_q, n_k) — Qwen3-4B-shaped projections at small dims
        (1, 1, 64, 32, 8),    # decode-style
        (1, 8, 96, 48, 12),   # prefill-style
        (2, 4, 80, 40, 10),   # batched, n_q != 4*n_k (loose GQA factor)
        (3, 5, 72, 36, 9),    # odd dims
    ],
)
def test_eager_qkv_reference_matches_attention(shape):
    """Our pure-function reference equals what ``Attention.forward`` computes
    for q/k/v before QK-norm and the head-axis view.

    Uses ``torch.nn.Linear`` with the same (out, in) weight layout the engine
    relies on; if this drifts the kernel will silently miscompute.
    """
    B, T, K, n_q, n_k = shape
    torch.manual_seed(0)

    x = torch.randn(B, T, K)
    q_proj = torch.nn.Linear(K, n_q, bias=False)
    k_proj = torch.nn.Linear(K, n_k, bias=False)
    v_proj = torch.nn.Linear(K, n_k, bias=False)

    expected_q = q_proj(x)
    expected_k = k_proj(x)
    expected_v = v_proj(x)

    actual_q, actual_k, actual_v = _eager_qkv(
        x, q_proj.weight, k_proj.weight, v_proj.weight
    )

    torch.testing.assert_close(actual_q, expected_q, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_k, expected_k, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_v, expected_v, rtol=0.0, atol=0.0)


# Qwen3-4B attention dims.
QWEN3_4B_K = 2560        # hidden_size
QWEN3_4B_NUM_Q = 4096    # num_heads * head_dim = 32 * 128
QWEN3_4B_NUM_K = 1024    # num_kv_heads * head_dim = 8 * 128


def _bf16_close(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs().max().item()
    ref_max = expected.float().abs().max().item()
    return diff, ref_max


# ----- Step 2: Triton kernel, BF16, small shapes ----------------------------


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
@pytest.mark.parametrize(
    "MNK",
    [
        # (M, n_q, n_k, K) — small dims with the same GQA shape pattern
        (1, 128, 64, 192),
        (2, 128, 64, 192),
        (3, 160, 32, 192),
    ],
)
def test_triton_qkv_bf16_small(MNK):
    """Triton kernel matches the eager reference within BF16 envelope on
    small shapes."""
    from kernels.fused_qkv import triton_fused_qkv

    M, n_q, n_k, K = MNK
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    x = torch.randn(M, K, device=device, dtype=dtype)
    Wq = torch.randn(n_q, K, device=device, dtype=dtype)
    Wk = torch.randn(n_k, K, device=device, dtype=dtype)
    Wv = torch.randn(n_k, K, device=device, dtype=dtype)
    W_qkv = torch.cat([Wq, Wk, Wv], dim=0)

    eq, ek, ev = _eager_qkv(x, Wq, Wk, Wv)
    aq, ak, av = triton_fused_qkv(x, W_qkv, n_q=n_q, n_k=n_k)

    assert aq.shape == eq.shape
    assert ak.shape == ek.shape
    assert av.shape == ev.shape

    for label, (a, e) in zip(["q", "k", "v"], [(aq, eq), (ak, ek), (av, ev)]):
        diff, ref_max = _bf16_close(a, e)
        assert diff < 1e-2 * ref_max + 1e-2, (
            f"MNK={MNK} {label}: diff {diff:.4f}, ref_max {ref_max:.4f}"
        )


# ----- Step 3: BF16, Qwen3-4B QKV shape, masking, error cases ---------------


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
@pytest.mark.parametrize("M", [1, 5, 8])
def test_triton_qkv_bf16_qwen_shapes(M):
    """Triton kernel matches the eager reference within BF16 envelope on the
    Qwen3-4B QKV shape ``(M, n_q=4096, n_k=1024, K=2560)`` across the
    production M values (greedy decode, spec K=4 verify with P=1, spec K=7
    verify with P=1).
    """
    from kernels.fused_qkv import triton_fused_qkv

    n_q = QWEN3_4B_NUM_Q
    n_k = QWEN3_4B_NUM_K
    K = QWEN3_4B_K
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    x = torch.randn(M, K, device=device, dtype=dtype)
    Wq = torch.randn(n_q, K, device=device, dtype=dtype)
    Wk = torch.randn(n_k, K, device=device, dtype=dtype)
    Wv = torch.randn(n_k, K, device=device, dtype=dtype)
    W_qkv = torch.cat([Wq, Wk, Wv], dim=0)

    eq, ek, ev = _eager_qkv(x, Wq, Wk, Wv)
    aq, ak, av = triton_fused_qkv(x, W_qkv, n_q=n_q, n_k=n_k)

    for label, (a, e) in zip(["q", "k", "v"], [(aq, eq), (ak, ek), (av, ev)]):
        diff, ref_max = _bf16_close(a, e)
        assert diff < 1e-2 * ref_max + 1e-2, (
            f"M={M} {label}: diff {diff:.4f}, ref_max {ref_max:.4f}"
        )


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
def test_triton_qkv_bf16_non_pow2_K():
    """Exercises the K-tail mask: K=2541 is not a multiple of typical BLOCK_K."""
    from kernels.fused_qkv import triton_fused_qkv

    M, n_q, n_k, K = 1, 4096, 1024, 2541
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(1)

    x = torch.randn(M, K, device=device, dtype=dtype)
    Wq = torch.randn(n_q, K, device=device, dtype=dtype)
    Wk = torch.randn(n_k, K, device=device, dtype=dtype)
    Wv = torch.randn(n_k, K, device=device, dtype=dtype)
    W_qkv = torch.cat([Wq, Wk, Wv], dim=0)

    eq, ek, ev = _eager_qkv(x, Wq, Wk, Wv)
    aq, ak, av = triton_fused_qkv(x, W_qkv, n_q=n_q, n_k=n_k)

    for label, (a, e) in zip(["q", "k", "v"], [(aq, eq), (ak, ek), (av, ev)]):
        diff, ref_max = _bf16_close(a, e)
        assert diff < 1e-2 * ref_max + 1e-2, (
            f"non-pow2 K={K} {label}: diff {diff:.4f}, ref_max {ref_max:.4f}"
        )


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
def test_triton_qkv_preserves_leading_dims():
    """Wrapper accepts ``(B, T, K)`` and returns ``(B, T, n_*)`` outputs."""
    from kernels.fused_qkv import triton_fused_qkv

    B, T, K = 2, 3, 64
    n_q, n_k = 32, 16
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(2)

    x = torch.randn(B, T, K, device=device, dtype=dtype)
    Wq = torch.randn(n_q, K, device=device, dtype=dtype)
    Wk = torch.randn(n_k, K, device=device, dtype=dtype)
    Wv = torch.randn(n_k, K, device=device, dtype=dtype)
    W_qkv = torch.cat([Wq, Wk, Wv], dim=0)

    aq, ak, av = triton_fused_qkv(x, W_qkv, n_q=n_q, n_k=n_k)
    assert aq.shape == (B, T, n_q)
    assert ak.shape == (B, T, n_k)
    assert av.shape == (B, T, n_k)


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
def test_qkv_split_views_into_heads():
    """Outputs of ``triton_fused_qkv`` reshape cleanly into the
    ``(B, T, num_heads, head_dim)`` layout the engine relies on.
    """
    from kernels.fused_qkv import triton_fused_qkv

    B, T = 1, 5
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    n_q = num_heads * head_dim       # 4096
    n_k = num_kv_heads * head_dim    # 1024
    K = 2560
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(3)

    x = torch.randn(B, T, K, device=device, dtype=dtype)
    W_qkv = torch.randn(n_q + 2 * n_k, K, device=device, dtype=dtype)

    q_flat, k_flat, v_flat = triton_fused_qkv(x, W_qkv, n_q=n_q, n_k=n_k)
    q = q_flat.view(B, T, num_heads, head_dim)
    k = k_flat.view(B, T, num_kv_heads, head_dim)
    v = v_flat.view(B, T, num_kv_heads, head_dim)
    assert q.shape == (B, T, num_heads, head_dim)
    assert k.shape == (B, T, num_kv_heads, head_dim)
    assert v.shape == (B, T, num_kv_heads, head_dim)


def test_qkv_shape_mismatch_raises():
    """Wrapper rejects inputs with mismatched dims, with clear errors.

    CPU-only — no CUDA needed for arg validation.
    """
    from kernels.fused_qkv import triton_fused_qkv

    x = torch.randn(2, 64)
    W_qkv = torch.randn(64 + 16 + 16, 64)  # n_q=64, n_k=16 — total 96

    # Wrong n_q + 2*n_k != n_total
    with pytest.raises(ValueError, match="n_q .* 2.n_k"):
        triton_fused_qkv(x, W_qkv, n_q=64, n_k=15)

    # Bad K
    x_bad_k = torch.randn(2, 63)
    with pytest.raises(ValueError, match="x last dim .* W_qkv"):
        triton_fused_qkv(x_bad_k, W_qkv, n_q=64, n_k=16)

    # Non-2D weight
    W_3d = torch.randn(2, 96, 64)
    with pytest.raises(ValueError, match="W_qkv must be 2D"):
        triton_fused_qkv(x, W_3d, n_q=64, n_k=16)


# ----- Step 4: microbench gate (Stage A.5b ship gate) -----------------------


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
@pytest.mark.parametrize("M", [1, 5, 8])
def test_microbench_qkv_beats_eager(M):
    """Stage A.5b ship gate: median Triton wall <= median eager-3-Linears
    wall on the Qwen3-4B QKV shape across the production M values.

    Also reports the cuBLAS-fused baseline (single ``nn.Linear(K, n_total)``)
    so we can see whether the win is from Triton itself or just from the
    fusion-via-pre-concat — both routes are valid; the formal gate is vs
    eager-3-Linears (the literal engine code).
    """
    from bench.microbench_fused_qkv import bench_one_shape

    res = bench_one_shape(M=M, dtype=torch.bfloat16, warmup=50, iters=200)
    e3 = res["eager_3_linears"].median_us
    ec = res["eager_concat_linear"].median_us
    tr = res["triton"].median_us
    speedup_3L = e3 / tr
    speedup_concat = ec / tr
    print(
        f"\nM={M}: eager_3L {e3:.1f} µs, eager_concat {ec:.1f} µs, "
        f"triton {tr:.1f} µs; speedup vs 3L={speedup_3L:.2f}x, "
        f"vs concat={speedup_concat:.2f}x"
    )
    assert tr <= e3, (
        f"M={M}: triton {tr:.1f} µs > eager_3_linears {e3:.1f} µs "
        f"(speedup {speedup_3L:.2f}x)"
    )


# ----- Step 5: TritonAttention engine integration ---------------------------


@pytest.mark.requires_draft
@pytest.mark.requires_cuda
@pytest.mark.requires_triton
@pytest.mark.parametrize(
    "dtype,logit_atol",
    [
        (torch.float32, 1e-3),
        # BF16 envelope at the model level is wider than at the kernel level
        # because per-layer ULP noise compounds across 28 residual additions.
        (torch.bfloat16, 2.0),
    ],
    ids=["fp32", "bf16"],
)
def test_attention_module_logits_match(draft_model_id, dtype, logit_atol):
    """``apply_triton_kernels(model, fused_qkv=True, fused_ffn=False)`` produces
    a model whose prefill logits agree with the eager Attention path within
    envelope, with the last-token argmax matching exactly.
    """
    from engine.model import Qwen3Model
    from engine.weights import load_weights
    from kernels.fused_qkv import TritonAttention
    from kernels.swap import apply_triton_kernels

    device = "cuda"
    loaded = load_weights(draft_model_id, dtype=dtype, device=device)

    eager = Qwen3Model.from_loaded(loaded).to(dtype=dtype, device=device).eval()
    kerneled = (
        Qwen3Model.from_loaded(loaded).to(dtype=dtype, device=device).eval()
    )
    apply_triton_kernels(kerneled, fused_qkv=True, fused_ffn=False)

    # Sanity: walker actually replaced attention but not FFN
    assert all(isinstance(b.attn, TritonAttention) for b in kerneled.layers)
    assert not any(isinstance(b.attn, TritonAttention) for b in eager.layers)

    input_ids = torch.tensor(
        [[100, 200, 300, 400, 500, 600, 700, 800]], device=device
    )
    with torch.inference_mode():
        eager_logits = eager(input_ids, eager.alloc_cache(max_seq_len=64))
        kerneled_logits = kerneled(input_ids, kerneled.alloc_cache(max_seq_len=64))

    diff = (eager_logits.float() - kerneled_logits.float()).abs().max().item()
    assert diff < logit_atol, (
        f"dtype={dtype} max abs logit diff {diff:.6f} (atol {logit_atol})"
    )
    assert (
        eager_logits[0, -1].argmax().item()
        == kerneled_logits[0, -1].argmax().item()
    ), f"dtype={dtype} last-token argmax disagreement"


@pytest.mark.requires_draft
@pytest.mark.requires_cuda
@pytest.mark.requires_triton
def test_apply_qkv_idempotent(draft_model_id):
    """Calling ``apply_triton_kernels(fused_qkv=True)`` twice does not
    re-wrap or re-allocate W_qkv."""
    from engine.model import Qwen3Model
    from engine.weights import load_weights
    from kernels.fused_qkv import TritonAttention
    from kernels.swap import apply_triton_kernels

    loaded = load_weights(draft_model_id, dtype=torch.bfloat16, device="cuda")
    model = (
        Qwen3Model.from_loaded(loaded)
        .to(dtype=torch.bfloat16, device="cuda")
        .eval()
    )
    apply_triton_kernels(model, fused_qkv=True, fused_ffn=False)
    first_attns = [b.attn for b in model.layers]
    apply_triton_kernels(model, fused_qkv=True, fused_ffn=False)
    second_attns = [b.attn for b in model.layers]
    assert all(a is b for a, b in zip(first_attns, second_attns, strict=True))
    assert all(isinstance(b.attn, TritonAttention) for b in model.layers)


# ----- Step 6: greedy E2E parity --------------------------------------------


_E2E_PROMPTS = [
    [100, 200, 300, 400, 500, 600, 700, 800],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    [42, 42, 42, 42, 42],
]


def _greedy_generate(model, prompt_ids: torch.Tensor, n_new: int) -> list[int]:
    from engine.sampler import greedy

    cache = model.alloc_cache(prompt_ids.shape[1] + n_new + 4)
    out: list[int] = []
    with torch.inference_mode():
        logits = model(prompt_ids, cache, start_pos=0)
        nt = greedy(logits[:, -1, :]).unsqueeze(0)
        out.append(int(nt.item()))
        for _ in range(n_new - 1):
            logits = model(nt, cache)
            nt = greedy(logits[:, -1, :]).unsqueeze(0)
            out.append(int(nt.item()))
    return out


@pytest.mark.requires_draft
@pytest.mark.requires_cuda
@pytest.mark.requires_triton
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, id="fp32"),
        pytest.param(
            torch.bfloat16,
            id="bf16",
            marks=pytest.mark.xfail(
                strict=False,
                reason=(
                    "BF16 batched-vs-sequential reductions are non-associative; "
                    "two kernels' worth of per-layer ULP noise on top of A.5a "
                    "may flip an argmax even when the math is correct."
                ),
            ),
        ),
    ],
)
@pytest.mark.parametrize("prompt", _E2E_PROMPTS)
def test_greedy_e2e_token_match_qkv(draft_model_id, dtype, prompt):
    """Greedy generation with **both** kernels enabled (fused_ffn AND
    fused_qkv) matches the eager engine token-for-token in FP32. BF16 is
    best-effort.
    """
    from engine.model import Qwen3Model
    from engine.weights import load_weights
    from kernels.swap import apply_triton_kernels

    device = "cuda"
    n_new = 32

    loaded = load_weights(draft_model_id, dtype=dtype, device=device)

    eager = Qwen3Model.from_loaded(loaded).to(dtype=dtype, device=device).eval()
    kerneled = Qwen3Model.from_loaded(loaded).to(dtype=dtype, device=device).eval()
    apply_triton_kernels(kerneled, fused_ffn=True, fused_qkv=True)

    prompt_t = torch.tensor([prompt], dtype=torch.long, device=device)

    eager_out = _greedy_generate(eager, prompt_t, n_new)
    kerneled_out = _greedy_generate(kerneled, prompt_t, n_new)

    if eager_out != kerneled_out:
        first = next(
            (i for i, (a, b) in enumerate(zip(eager_out, kerneled_out)) if a != b),
            None,
        )
        raise AssertionError(
            f"\n  prompt={prompt} dtype={dtype}"
            f"\n  eager   ={eager_out}"
            f"\n  kerneled={kerneled_out}"
            f"\n  first divergence idx={first}"
        )
