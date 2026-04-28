# Stage A.5 — Profile summary

- GPU: **NVIDIA GeForce RTX 5060 Ti** (cap 12.0, 15.5 GB)
- torch: 2.11.0+cu130
- target: `Qwen/Qwen3-4B`
- draft:  `Qwen/Qwen3-0.6B`
- prompt tokens: 36
- decode steps (greedy): 200; spec-decode K=4, max_new=200

## Headline

  - **greedy**: 38.6 tok/s (200 tokens in 5.19s)
  - **spec**: 28.4 tok/s (200 tokens in 7.04s)

## Run: greedy

- wall time: 5.187 s
- framework-op self-CUDA total: 2.226 s ( 42.9% of wall)
- device-kernel self-CUDA total: 4.606 s ( 88.8% of wall — cross-check)
- framework-op self-CPU total: 3.624 s
- mean wall per step: 25937.1 µs
- mean kernel launches per step: 1941

### Category breakdown over framework ops (self-CUDA, % of wall)

| category | self-CUDA µs | % of wall | events |
|---|---:|---:|---:|
| matmul | 1637612 |  31.6% | 43600 |
| elementwise | 196517 |   3.8% | 159000 |
| kv_io | 176271 |   3.4% | 86800 |
| norm_act | 98158 |   1.9% | 87000 |
| attention_core | 78387 |   1.5% | 21600 |
| memory_layout | 25862 |   0.5% | 564800 |
| other | 12317 |   0.2% | 785930 |
| sampling | 1081 |   0.0% | 200 |
| embedding | 0 |   0.0% | 200 |
| gqa_expand | 0 |   0.0% | 14400 |

### Top 15 framework ops by self-CUDA time

| op | category | self-CUDA µs | % wall | self-CPU µs | count |
|---|---|---:|---:|---:|---:|
| `aten::mm` | matmul | 1637612 |  31.6% | 165638 | 14600 |
| `aten::copy_` | kv_io | 176271 |   3.4% | 210786 | 86800 |
| `aten::mul` | elementwise | 118536 |   2.3% | 259921 | 86800 |
| `aten::_flash_attention_forward` | attention_core | 78387 |   1.5% | 48221 | 7200 |
| `aten::add` | elementwise | 56175 |   1.1% | 204106 | 57800 |
| `aten::mean` | norm_act | 44803 |   0.9% | 140313 | 29000 |
| `aten::rsqrt` | norm_act | 28491 |   0.5% | 82229 | 29000 |
| `aten::cat` | memory_layout | 25862 |   0.5% | 57512 | 14400 |
| `aten::pow` | norm_act | 24865 |   0.5% | 127329 | 29000 |
| `aten::neg` | elementwise | 21806 |   0.4% | 43884 | 14400 |
| `Command Buffer Full` | other | 6153 |   0.1% | 492317 | 5780 |
| `cuLaunchKernelEx` | other | 5609 |   0.1% | 43402 | 14400 |
| `aten::argmax` | sampling | 1081 |   0.0% | 1695 | 200 |
| `aten::index_select` | other | 255 |   0.0% | 1393 | 200 |
| `Buffer Flush` | other | 173 |   0.0% | 3151 | 49 |

### Top 10 device kernels (cross-check)

| kernel | self-CUDA µs | % wall | launches |
|---|---:|---:|---:|
| `_fused_gate_up_silu_kernel` | 1828063 |  35.2% | 7200 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_b...` | 884819 |  17.1% | 7200 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_b...` | 752794 |  14.5% | 7400 |
| `_qkv_matmul_kernel` | 563901 |  10.9% | 7200 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kern...` | 98590 |   1.9% | 28800 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c1...` | 67902 |   1.3% | 43200 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::...` | 53169 |   1.0% | 29000 |
| `void pytorch_flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass...` | 45588 |   0.9% | 4248 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float,...` | 44803 |   0.9% | 29000 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<fl...` | 30906 |   0.6% | 29000 |

## Run: spec

- rounds=68, accepted=133/272 (48.9%), bonus_rounds=20
- wall time: 7.038 s
- framework-op self-CUDA total: 1.787 s ( 25.4% of wall)
- device-kernel self-CUDA total: 3.015 s ( 42.8% of wall — cross-check)
- framework-op self-CPU total: 4.838 s
- mean wall per step: 103501.0 µs
- mean kernel launches per step: 8361
- **launch-bound signal**: framework CPU 4.838s > device kernels 3.015s. Host-side Python/dispatcher overhead is on the critical path; CUDA Graphs or a single fused decode-step kernel would address this in addition to any per-op speedup.

### Category breakdown over framework ops (self-CUDA, % of wall)

| category | self-CUDA µs | % of wall | events |
|---|---:|---:|---:|
| matmul | 999512 |  14.2% | 61452 |
| elementwise | 277563 |   3.9% | 223842 |
| kv_io | 219839 |   3.1% | 125104 |
| norm_act | 134463 |   1.9% | 122562 |
| attention_core | 96022 |   1.4% | 30384 |
| memory_layout | 36346 |   0.5% | 818559 |
| other | 20337 |   0.3% | 1199742 |
| sampling | 2471 |   0.0% | 409 |
| embedding | 0 |   0.0% | 342 |
| gqa_expand | 0 |   0.0% | 20256 |

### Top 15 framework ops by self-CUDA time

| op | category | self-CUDA µs | % wall | self-CPU µs | count |
|---|---|---:|---:|---:|---:|
| `aten::mm` | matmul | 999512 |  14.2% | 246424 | 20598 |
| `aten::copy_` | kv_io | 219839 |   3.1% | 306093 | 125104 |
| `aten::mul` | elementwise | 167262 |   2.4% | 371504 | 122220 |
| `aten::add` | elementwise | 79823 |   1.1% | 290855 | 81366 |
| `aten::mean` | norm_act | 61460 |   0.9% | 200283 | 40854 |
| `aten::_flash_attention_forward` | attention_core | 54957 |   0.8% | 46173 | 7148 |
| `aten::_efficient_attention_forward` | attention_core | 41065 |   0.6% | 15268 | 2980 |
| `aten::rsqrt` | norm_act | 38953 |   0.6% | 115724 | 40854 |
| `aten::cat` | memory_layout | 36346 |   0.5% | 80659 | 20256 |
| `aten::pow` | norm_act | 34050 |   0.5% | 179197 | 40854 |
| `aten::neg` | elementwise | 30478 |   0.4% | 61984 | 20256 |
| `aten::fill_` | other | 6525 |   0.1% | 14766 | 8640 |
| `aten::where` | other | 4570 |   0.1% | 19598 | 5960 |
| `aten::arange` | other | 4120 |   0.1% | 18571 | 11920 |
| `aten::ge` | other | 4120 |   0.1% | 12535 | 2980 |

### Top 10 device kernels (cross-check)

| kernel | self-CUDA µs | % wall | launches |
|---|---:|---:|---:|
| `_fused_gate_up_silu_kernel` | 873529 |  12.4% | 10128 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(cutlass_80_wmma...` | 470574 |   6.7% | 5979 |
| `_qkv_matmul_kernel` | 355031 |   5.0% | 10128 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_b...` | 205603 |   2.9% | 14168 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x1_tn_align8>(cutlass_80_wmma...` | 186250 |   2.6% | 253 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8>(cutlass_80_wmma...` | 126309 |   1.8% | 68 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kern...` | 109361 |   1.6% | 43192 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c1...` | 107853 |   1.5% | 66945 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::...` | 64343 |   0.9% | 34766 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float,...` | 61460 |   0.9% | 40854 |

## Bottleneck and recommendation

On the **greedy** decode, the dominant framework category is **matmul** at **31.6%** of wall time.

That clears the >30% bar in DESIGN.md, so the Stage A.5 Triton kernel target is in the **matmul** family.

### GEMV vs attention
- linear projections (`matmul`): ** 31.6%** of wall
- attention core (`flash/sdpa`): **  1.5%** of wall

Linear projections dominate by >5×. At batch=1 / T=1 every linear is a GEMV (matrix-vector), and is HBM-bandwidth bound: the weight matrix is streamed once per token with ~no reuse. Attention core is cheap because the context is short. The right Triton target is a **fused linear/GEMV kernel** (e.g., fused QKV or fused gate+up+silu+down for the FFN), not a fused attention kernel. Revisit attention only at long context (Stage C sweeps).

---

## Reproduce / extend

Re-run this profile (regenerates traces and summary):

```bash
uv run python -m bench.profile --target Qwen/Qwen3-4B --draft Qwen/Qwen3-0.6B --decode-steps 200 --K 4 --max-new 200
```

Kernel-level metrics with Nsight Compute (HBM throughput, L2 hit, occupancy):

```bash
ncu --set full --target-processes all --launch-skip 200 --launch-count 50 \
    -o experiments/stage_a/ncu_decode -f \
    uv run python -m bench.profile --target Qwen/Qwen3-4B --draft Qwen/Qwen3-0.6B --decode-steps 80 --skip-spec
```

View Chrome traces: open `chrome://tracing` and load `torch_trace_*.json`.