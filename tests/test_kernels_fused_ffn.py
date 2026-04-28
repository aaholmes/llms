"""Tests for the fused gate-up-silu Triton kernel (Stage A.5a).

Math contract (pinned in step 1 below): the kernel computes the partial
expression that lives inside ``FFN.forward`` at ``src/engine/model.py:36-37``,
without the down projection:

    silu(x @ Wg.T) * (x @ Wu.T)

where ``x`` has shape ``(..., K)`` and ``Wg, Wu`` have shape ``(N, K)`` —
matching ``torch.nn.Linear.weight`` layout (out, in).

The TDD ladder for Stage A.5a is documented in
``~/.claude/plans/ok-make-a-detailed-keen-clover.md``. This file grows step
by step; each step adds tests then unblocks the implementation.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn


def _eager_gate_up_silu(
    x: torch.Tensor, Wg: torch.Tensor, Wu: torch.Tensor
) -> torch.Tensor:
    """Reference implementation of the kernel's math contract.

    x: (..., K)
    Wg, Wu: (N, K)  — nn.Linear weight layout
    Returns: (..., N)
    """
    return F.silu(x @ Wg.T) * (x @ Wu.T)


# ----- Step 1: pin the math contract (no Triton, no GPU) --------------------


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 128),  # decode-style: B=1, T=1
        (1, 8, 256),  # prefill-style
        (2, 4, 64),   # batched
        (3, 5, 96),   # odd dims
    ],
)
def test_eager_reference_matches_ffn(shape):
    """Our pure-function reference equals what the production ``FFN.forward``
    computes pre-down (i.e. ``silu(self.gate(x)) * self.up(x)``).

    Uses ``torch.nn.Linear`` to ensure our (N, K) weight convention matches
    PyTorch's; if this drifts the kernel will silently miscompute.
    """
    B, T, K = shape
    N = 32
    torch.manual_seed(0)

    x = torch.randn(B, T, K)
    gate = torch.nn.Linear(K, N, bias=False)
    up = torch.nn.Linear(K, N, bias=False)

    expected = F.silu(gate(x)) * up(x)
    actual = _eager_gate_up_silu(x, gate.weight, up.weight)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


# Qwen3-4B FFN dims: hidden=2560 (=K), intermediate=9728 (=N). M sweep:
#   M=1 — greedy decode step
#   M=5 — one spec-decode round at K=4 with P=1 (pending of size 1)
#   M=8 — max-K spec-decode round (K=7, P=1)
QWEN3_4B_K = 2560
QWEN3_4B_N = 9728


def _bf16_close(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    """Return (max_abs_diff, ref_max) for a BF16 tolerance check."""
    diff = (actual.float() - expected.float()).abs().max().item()
    ref_max = expected.float().abs().max().item()
    return diff, ref_max


# ----- Step 2: Triton kernel, BF16, small shapes ----------------------------
# BF16 is the production dtype. FP32-on-Triton would force shared-memory-
# conservative autotune configs that hurt BF16 perf; the math contract is
# already pinned in FP32 on CPU by ``test_eager_reference_matches_ffn`` above.


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
@pytest.mark.parametrize(
    "MNK",
    [
        (1, 128, 128),  # smallest workable tile (BLOCK_M=16 will mask 15)
        (2, 128, 128),
        (3, 160, 192),  # non-multiple-of-block to exercise N-tail mask
    ],
)
def test_triton_bf16_small(MNK):
    """Triton kernel matches the eager reference within BF16 envelope on
    small shapes.
    """
    M, N, K = MNK
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    x = torch.randn(M, K, device=device, dtype=dtype)
    Wg = torch.randn(N, K, device=device, dtype=dtype)
    Wu = torch.randn(N, K, device=device, dtype=dtype)

    expected = _eager_gate_up_silu(x, Wg, Wu)

    from kernels.fused_ffn import triton_fused_gate_up_silu

    actual = triton_fused_gate_up_silu(x, Wg, Wu)

    diff, ref_max = _bf16_close(actual, expected)
    assert diff < 1e-2 * ref_max + 1e-2, (
        f"MNK={MNK}: max abs diff {diff:.4f}, ref_max {ref_max:.4f}"
    )


# ----- Step 3: BF16, Qwen3-4B shapes, masking, error cases ------------------


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
@pytest.mark.parametrize("M", [1, 5, 8])
def test_triton_bf16_qwen_shapes(M):
    """Triton kernel matches the eager reference within BF16 envelope on the
    actual Qwen3-4B FFN shapes ``(M, N=9728, K=2560)``.

    Uses the same envelope as ``test_correctness.py``:
    ``diff < 1e-2 * ref_max + 1e-2`` — i.e. 1% relative + 0.01 absolute floor.
    """
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    x = torch.randn(M, QWEN3_4B_K, device=device, dtype=dtype)
    Wg = torch.randn(QWEN3_4B_N, QWEN3_4B_K, device=device, dtype=dtype)
    Wu = torch.randn(QWEN3_4B_N, QWEN3_4B_K, device=device, dtype=dtype)

    from kernels.fused_ffn import triton_fused_gate_up_silu

    expected = _eager_gate_up_silu(x, Wg, Wu)
    actual = triton_fused_gate_up_silu(x, Wg, Wu)

    diff, ref_max = _bf16_close(actual, expected)
    assert diff < 1e-2 * ref_max + 1e-2, (
        f"M={M}: max abs diff {diff:.4f}, ref_max {ref_max:.4f}, "
        f"rel {diff / max(ref_max, 1e-9):.2%}"
    )


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
def test_triton_bf16_non_pow2_N():
    """Exercises the N-tail mask: N=9700 is not a multiple of typical BLOCK_N."""
    M, N, K = 1, 9700, 2560
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(1)

    x = torch.randn(M, K, device=device, dtype=dtype)
    Wg = torch.randn(N, K, device=device, dtype=dtype)
    Wu = torch.randn(N, K, device=device, dtype=dtype)

    from kernels.fused_ffn import triton_fused_gate_up_silu

    expected = _eager_gate_up_silu(x, Wg, Wu)
    actual = triton_fused_gate_up_silu(x, Wg, Wu)
    assert actual.shape == expected.shape

    diff, ref_max = _bf16_close(actual, expected)
    assert diff < 1e-2 * ref_max + 1e-2, (
        f"non-pow2 N={N}: diff {diff:.4f}, ref_max {ref_max:.4f}"
    )


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
def test_triton_preserves_leading_dims():
    """Wrapper accepts ``(B, T, K)`` and returns ``(B, T, N)`` (BF16 path)."""
    B, T, K, N = 2, 3, 64, 128
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(2)

    x = torch.randn(B, T, K, device=device, dtype=dtype)
    Wg = torch.randn(N, K, device=device, dtype=dtype)
    Wu = torch.randn(N, K, device=device, dtype=dtype)

    from kernels.fused_ffn import triton_fused_gate_up_silu

    expected = _eager_gate_up_silu(x, Wg, Wu)
    actual = triton_fused_gate_up_silu(x, Wg, Wu)

    assert actual.shape == (B, T, N)
    diff, ref_max = _bf16_close(actual, expected)
    assert diff < 1e-2 * ref_max + 1e-2


def test_shape_mismatch_raises():
    """Wrapper rejects inputs with mismatched K or N dims, with clear errors.

    CPU-only — no CUDA needed for arg validation.
    """
    from kernels.fused_ffn import triton_fused_gate_up_silu

    x = torch.randn(2, 64)
    Wg = torch.randn(128, 64)
    Wu_bad_n = torch.randn(127, 64)
    Wu_bad_k = torch.randn(128, 63)

    with pytest.raises(ValueError, match="Wg shape.*Wu shape"):
        triton_fused_gate_up_silu(x, Wg, Wu_bad_n)

    with pytest.raises(ValueError, match="Wg shape.*Wu shape"):
        triton_fused_gate_up_silu(x, Wg, Wu_bad_k)

    Wg_3d = torch.randn(2, 128, 64)
    with pytest.raises(ValueError, match="Wg must be 2D"):
        triton_fused_gate_up_silu(x, Wg_3d, Wg_3d)

    x_bad_k = torch.randn(2, 63)
    with pytest.raises(ValueError, match="x last dim.*Wg.shape"):
        triton_fused_gate_up_silu(x_bad_k, Wg, Wg)


# ----- Step 4: microbench gate (Stage A.5a ship gate) -----------------------


@pytest.mark.requires_cuda
@pytest.mark.requires_triton
@pytest.mark.parametrize("M", [1, 5, 8])
def test_microbench_beats_eager(M):
    """Stage A.5a ship gate: median Triton wall <= median eager wall on the
    Qwen3-4B FFN shape across the production M values.

    Reads as the literal DESIGN.md gate ("kernel >= PyTorch reference"). On
    miss, see the Risks section of the plan for fallbacks.
    """
    from bench.microbench_fused_ffn import bench_one_shape

    res = bench_one_shape(
        M=M,
        N=QWEN3_4B_N,
        K=QWEN3_4B_K,
        dtype=torch.bfloat16,
        warmup=50,
        iters=200,
    )
    eager_us = res["eager"].median_us
    triton_us = res["triton"].median_us
    speedup = eager_us / triton_us
    print(
        f"\nM={M}: eager {eager_us:.1f} µs, triton {triton_us:.1f} µs, "
        f"speedup {speedup:.2f}x"
    )
    assert triton_us <= eager_us, (
        f"M={M}: triton {triton_us:.1f} µs > eager {eager_us:.1f} µs "
        f"(speedup {speedup:.2f}x)"
    )


# ----- Step 5: engine integration via subclass + walker ---------------------


@pytest.fixture(scope="module")
def _draft_loaded_bf16(draft_model_id):
    """Load Qwen3-0.6B once for all integration tests in this file."""
    from engine.weights import load_weights

    return load_weights(draft_model_id, dtype=torch.bfloat16, device="cuda")


@pytest.mark.requires_draft
@pytest.mark.requires_cuda
@pytest.mark.requires_triton
@pytest.mark.parametrize(
    "dtype,logit_atol",
    [
        # FP32: kernel and eager use FP32 accumulation; differences are at the
        # ULP level and survive 28 layers of residual accumulation tightly.
        (torch.float32, 1e-3),
        # BF16: per-FFN reduction order differs between cuBLAS and the Triton
        # kernel, and small per-layer ULP noise accumulates over 28 residual
        # additions. The argmax is the load-bearing assert; the absolute
        # logit envelope is noise-floor sized (~1.0 in practice for Qwen3-0.6B).
        (torch.bfloat16, 2.0),
    ],
    ids=["fp32", "bf16"],
)
def test_ffn_module_logits_match(draft_model_id, dtype, logit_atol):
    """``apply_triton_kernels`` produces a model whose prefill logits agree
    with the eager FFN, with the last-token argmax matching exactly.

    FP32 holds tight; BF16 is noise-floor sized after 28 layers.
    """
    from engine.model import Qwen3Model
    from engine.weights import load_weights
    from kernels.fused_ffn import TritonFFN
    from kernels.swap import apply_triton_kernels

    device = "cuda"
    loaded = load_weights(draft_model_id, dtype=dtype, device=device)

    eager = Qwen3Model.from_loaded(loaded).to(dtype=dtype, device=device).eval()
    kerneled = Qwen3Model.from_loaded(loaded).to(dtype=dtype, device=device).eval()
    apply_triton_kernels(kerneled)

    # Sanity: walker actually replaced the FFNs
    assert all(isinstance(b.ffn, TritonFFN) for b in kerneled.layers)
    assert not any(isinstance(b.ffn, TritonFFN) for b in eager.layers)

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


def test_apply_triton_kernels_idempotent(_draft_loaded_bf16):
    """Calling ``apply_triton_kernels`` twice should not stack wrappers."""
    from engine.model import Qwen3Model
    from kernels.fused_ffn import TritonFFN
    from kernels.swap import apply_triton_kernels

    model = (
        Qwen3Model.from_loaded(_draft_loaded_bf16)
        .to(dtype=torch.bfloat16, device="cuda")
        .eval()
    )
    apply_triton_kernels(model)
    first = [b.ffn for b in model.layers]
    apply_triton_kernels(model)
    second = [b.ffn for b in model.layers]
    assert all(a is b for a, b in zip(first, second, strict=True))
    assert all(isinstance(b.ffn, TritonFFN) for b in model.layers)


@pytest.mark.requires_draft
@pytest.mark.requires_cuda
@pytest.mark.requires_triton
def test_prewarm_runs_clean(_draft_loaded_bf16):
    """``prewarm_triton_kernels`` runs without error and does not perturb
    forward outputs (autotune just picks a config; outputs are deterministic
    once a config is selected).
    """
    from engine.model import Qwen3Model
    from kernels.swap import apply_triton_kernels, prewarm_triton_kernels

    model = (
        Qwen3Model.from_loaded(_draft_loaded_bf16)
        .to(dtype=torch.bfloat16, device="cuda")
        .eval()
    )
    apply_triton_kernels(model)

    input_ids = torch.tensor(
        [[100, 200, 300, 400, 500, 600, 700, 800]], device="cuda"
    )

    with torch.inference_mode():
        before = model(input_ids, model.alloc_cache(max_seq_len=32))

    prewarm_triton_kernels(model)

    with torch.inference_mode():
        after = model(input_ids, model.alloc_cache(max_seq_len=32))

    diff = (before.float() - after.float()).abs().max().item()
    assert diff == 0.0, (
        f"prewarm changed forward output by {diff} — autotune is supposed to be "
        f"deterministic once a config is chosen"
    )


def test_prewarm_no_op_without_triton_modules():
    """``prewarm_triton_kernels`` on a model with no Triton modules is a no-op
    (covers the M2 dev path where the kernel package isn't wired in)."""
    from kernels.swap import prewarm_triton_kernels

    plain = nn.Sequential(nn.Linear(4, 4))  # no .layers, no TritonFFN
    # Should not raise even though `plain.layers` doesn't exist — function
    # walks `model.modules()`.
    prewarm_triton_kernels(plain)


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
                    "greedy generation may diverge by a token or two even when the "
                    "kernel math is correct. Mirrors test_spec_decode.py's FP32 "
                    "rationale."
                ),
            ),
        ),
    ],
)
@pytest.mark.parametrize("prompt", _E2E_PROMPTS)
def test_greedy_e2e_token_match(draft_model_id, dtype, prompt):
    """Greedy generation with kernels enabled matches the eager engine
    token-for-token (FP32) — the cleanest correctness gate for the kernel
    when wired into the full forward pass.
    """
    from engine.model import Qwen3Model
    from engine.weights import load_weights
    from kernels.swap import apply_triton_kernels

    device = "cuda"
    n_new = 32

    loaded = load_weights(draft_model_id, dtype=dtype, device=device)

    eager = Qwen3Model.from_loaded(loaded).to(dtype=dtype, device=device).eval()
    kerneled = Qwen3Model.from_loaded(loaded).to(dtype=dtype, device=device).eval()
    apply_triton_kernels(kerneled)

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
