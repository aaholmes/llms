"""Microbenchmark for the fused QKV-projection Triton kernel.

Times three implementations on the Qwen3-4B QKV shape (M ∈ {1, 5, 8},
n_q=4096, n_k=1024, K=2560):

  - ``eager_3_linears``: ``q(x), k(x), v(x)`` as three separate ``nn.Linear``
    calls — the literal engine code, the formal Stage A.5b reference.
  - ``eager_concat_linear``: a single ``nn.Linear(K, n_q+2*n_k)`` followed
    by ``torch.split`` — informational, shows whether the win is from
    fusion-via-pre-concat or from Triton itself.
  - ``triton_fused``: ``triton_fused_qkv`` against the same pre-concatenated
    weight.

The Stage A.5b ship gate (DESIGN.md) is
``triton_median <= eager_3_linears_median``. Checked by
``tests/test_kernels_fused_qkv.py::test_microbench_qkv_beats_eager``.

Standalone runnable:

  uv run python -m bench.microbench_fused_qkv
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
    """Median / P10 / P90 wall time for ``fn`` in microseconds."""
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


# Qwen3-4B attention dims (matches engine config).
QWEN3_4B_K = 2560
QWEN3_4B_NUM_Q = 4096
QWEN3_4B_NUM_K = 1024


def bench_one_shape(
    *,
    M: int,
    n_q: int = QWEN3_4B_NUM_Q,
    n_k: int = QWEN3_4B_NUM_K,
    K: int = QWEN3_4B_K,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    warmup: int = 50,
    iters: int = 200,
    seed: int = 0,
) -> dict[str, BenchResult]:
    """Bench three implementations of QKV projection on one shape."""
    from kernels.fused_qkv import triton_fused_qkv

    torch.manual_seed(seed)
    x = torch.randn(M, K, device=device, dtype=dtype)
    Wq = torch.randn(n_q, K, device=device, dtype=dtype)
    Wk = torch.randn(n_k, K, device=device, dtype=dtype)
    Wv = torch.randn(n_k, K, device=device, dtype=dtype)
    W_qkv = torch.cat([Wq, Wk, Wv], dim=0).contiguous()

    def eager_3_linears() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return F.linear(x, Wq), F.linear(x, Wk), F.linear(x, Wv)

    def eager_concat_linear() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = F.linear(x, W_qkv)
        return out.split([n_q, n_k, n_k], dim=-1)

    def triton_fn() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return triton_fused_qkv(x, W_qkv, n_q=n_q, n_k=n_k)

    # Pre-fire Triton so autotune compile cost lands outside the timed region.
    triton_fn()
    torch.cuda.synchronize()

    e3 = _time_fn(eager_3_linears, warmup=warmup, iters=iters)
    e3.label = "eager_3_linears"
    ec = _time_fn(eager_concat_linear, warmup=warmup, iters=iters)
    ec.label = "eager_concat_linear"
    tr = _time_fn(triton_fn, warmup=warmup, iters=iters)
    tr.label = "triton"

    return {"eager_3_linears": e3, "eager_concat_linear": ec, "triton": tr}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--M", type=int, nargs="+", default=[1, 5, 8])
    p.add_argument("--n_q", type=int, default=QWEN3_4B_NUM_Q)
    p.add_argument("--n_k", type=int, default=QWEN3_4B_NUM_K)
    p.add_argument("--K", type=int, default=QWEN3_4B_K)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = p.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    print(
        f"# bench fused QKV — n_q={args.n_q}, n_k={args.n_k}, K={args.K}, "
        f"dtype={args.dtype}"
    )
    print(
        f"{'M':>4}  {'eager_3L_us':>12}  {'eager_concat_us':>16}  "
        f"{'triton_us':>10}  {'speedup_vs_3L':>14}  {'speedup_vs_concat':>18}"
    )
    for M in args.M:
        res = bench_one_shape(
            M=M, n_q=args.n_q, n_k=args.n_k, K=args.K, dtype=dtype,
            warmup=args.warmup, iters=args.iters,
        )
        e3 = res["eager_3_linears"].median_us
        ec = res["eager_concat_linear"].median_us
        tr = res["triton"].median_us
        print(
            f"{M:>4}  {e3:>12.2f}  {ec:>16.2f}  {tr:>10.2f}  "
            f"{e3 / tr:>13.2f}x  {ec / tr:>17.2f}x"
        )


if __name__ == "__main__":
    main()
