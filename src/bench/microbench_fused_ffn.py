"""Microbenchmark for the fused gate-up-silu Triton kernel.

Times the kernel against the eager PyTorch reference
``F.silu(x @ Wg.T) * (x @ Wu.T)`` on the actual Qwen3-4B FFN shapes
(M ∈ {1, 5, 8}, N=9728, K=2560), reporting median wall and the
triton-vs-eager ratio.

The Stage A.5a ship gate (DESIGN.md) is ``triton_median <= eager_median``
on each shape — checked by ``tests/test_kernels_fused_ffn.py::test_microbench_beats_eager``.

This file is also runnable directly for hands-on tuning:

  uv run python -m bench.microbench_fused_ffn
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class BenchResult:
    label: str
    median_us: float
    p10_us: float
    p90_us: float
    iters: int


def _time_fn(fn, *, warmup: int = 50, iters: int = 200) -> BenchResult:
    """Median / P10 / P90 wall time for ``fn`` in microseconds.

    Uses ``torch.cuda.Event`` pairs per iteration; only one ``synchronize``
    after issuing all events to avoid serialising the timing region.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("microbench requires CUDA")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    times_us = sorted(
        s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends, strict=True)
    )
    return BenchResult(
        label="",
        median_us=times_us[iters // 2],
        p10_us=times_us[iters // 10],
        p90_us=times_us[(iters * 9) // 10],
        iters=iters,
    )


def bench_one_shape(
    *,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    warmup: int = 50,
    iters: int = 200,
    seed: int = 0,
) -> dict[str, BenchResult]:
    """Bench Triton fused kernel vs eager on a single ``(M, N, K)`` shape."""
    from kernels.fused_ffn import triton_fused_gate_up_silu

    torch.manual_seed(seed)
    x = torch.randn(M, K, device=device, dtype=dtype)
    Wg = torch.randn(N, K, device=device, dtype=dtype)
    Wu = torch.randn(N, K, device=device, dtype=dtype)

    def eager_fn() -> torch.Tensor:
        return F.silu(x @ Wg.T) * (x @ Wu.T)

    def triton_fn() -> torch.Tensor:
        return triton_fused_gate_up_silu(x, Wg, Wu)

    # Pre-fire Triton so autotune compilation cost lands outside the timed
    # region (the warmup loop in ``_time_fn`` would handle it too, but this
    # is more explicit and easier to reason about).
    triton_fn()
    torch.cuda.synchronize()

    eager = _time_fn(eager_fn, warmup=warmup, iters=iters)
    eager.label = "eager"
    triton = _time_fn(triton_fn, warmup=warmup, iters=iters)
    triton.label = "triton"

    return {"eager": eager, "triton": triton}


# Qwen3-4B FFN shape (matches src/engine/model.py via the HF config).
QWEN3_4B_K = 2560
QWEN3_4B_N = 9728


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--M", type=int, nargs="+", default=[1, 5, 8])
    p.add_argument("--N", type=int, default=QWEN3_4B_N)
    p.add_argument("--K", type=int, default=QWEN3_4B_K)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = p.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    print(f"# bench fused gate-up-silu — shape (M, N={args.N}, K={args.K}), dtype={args.dtype}")
    print(f"{'M':>4}  {'eager_us':>10}  {'triton_us':>10}  {'speedup':>8}")
    for M in args.M:
        res = bench_one_shape(
            M=M, N=args.N, K=args.K, dtype=dtype,
            warmup=args.warmup, iters=args.iters,
        )
        speedup = res["eager"].median_us / res["triton"].median_us
        print(
            f"{M:>4}  {res['eager'].median_us:>10.2f}  "
            f"{res['triton'].median_us:>10.2f}  {speedup:>7.2f}x"
        )


if __name__ == "__main__":
    main()
