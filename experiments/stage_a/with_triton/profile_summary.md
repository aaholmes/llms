# Stage A.5 — Profile summary

- GPU: **NVIDIA GeForce RTX 5060 Ti** (cap 12.0, 15.5 GB)
- torch: 2.11.0+cu130
- target: `Qwen/Qwen3-4B`
- draft:  `Qwen/Qwen3-0.6B`
- prompt tokens: 36
- decode steps (greedy): 200; spec-decode K=4, max_new=200

## Headline

  - **greedy**: 38.1 tok/s (200 tokens in 5.25s)
  - **spec**: 25.9 tok/s (200 tokens in 7.73s)

## Run: greedy

- wall time: 5.248 s
- framework-op self-CUDA total: 2.844 s ( 54.2% of wall)
- device-kernel self-CUDA total: 4.652 s ( 88.6% of wall — cross-check)
- framework-op self-CPU total: 3.931 s
- mean wall per step: 26238.8 µs
- mean kernel launches per step: 2013

### Category breakdown over framework ops (self-CUDA, % of wall)

| category | self-CUDA µs | % of wall | events |
|---|---:|---:|---:|
| matmul | 2246646 |  42.8% | 108400 |
| elementwise | 195340 |   3.7% | 159000 |
| kv_io | 175943 |   3.4% | 86800 |
| norm_act | 98962 |   1.9% | 87000 |
| attention_core | 79711 |   1.5% | 21600 |
| memory_layout | 25807 |   0.5% | 600800 |
| other | 20268 |   0.4% | 834627 |
| sampling | 1058 |   0.0% | 200 |
| embedding | 0 |   0.0% | 200 |
| gqa_expand | 0 |   0.0% | 14400 |

### Top 15 framework ops by self-CUDA time

| op | category | self-CUDA µs | % wall | self-CPU µs | count |
|---|---|---:|---:|---:|---:|
| `aten::mm` | matmul | 2246646 |  42.8% | 312669 | 36200 |
| `aten::copy_` | kv_io | 175943 |   3.4% | 198975 | 86800 |
| `aten::mul` | elementwise | 118796 |   2.3% | 253502 | 86800 |
| `aten::_flash_attention_forward` | attention_core | 79711 |   1.5% | 48498 | 7200 |
| `aten::add` | elementwise | 54752 |   1.0% | 195879 | 57800 |
| `aten::mean` | norm_act | 45538 |   0.9% | 136363 | 29000 |
| `aten::rsqrt` | norm_act | 28700 |   0.5% | 77948 | 29000 |
| `aten::cat` | memory_layout | 25807 |   0.5% | 56509 | 14400 |
| `aten::pow` | norm_act | 24724 |   0.5% | 123393 | 29000 |
| `aten::neg` | elementwise | 21792 |   0.4% | 43339 | 14400 |
| `Command Buffer Full` | other | 16103 |   0.3% | 632125 | 11270 |
| `cuLaunchKernelEx` | other | 3490 |   0.1% | 22216 | 7200 |
| `aten::argmax` | sampling | 1058 |   0.0% | 1701 | 200 |
| `aten::index_select` | other | 255 |   0.0% | 1319 | 200 |
| `Buffer Flush` | other | 250 |   0.0% | 3177 | 53 |

### Top 10 device kernels (cross-check)

| kernel | self-CUDA µs | % wall | launches |
|---|---:|---:|---:|
| `_fused_gate_up_silu_kernel` | 1827857 |  34.8% | 7200 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_b...` | 1361847 |  26.0% | 29000 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_b...` | 884799 |  16.9% | 7200 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kern...` | 98330 |   1.9% | 28800 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c1...` | 67828 |   1.3% | 43200 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::...` | 53040 |   1.0% | 29000 |
| `void pytorch_flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass...` | 46139 |   0.9% | 4248 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float,...` | 45538 |   0.9% | 29000 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<fl...` | 31285 |   0.6% | 29000 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::rsqrt_kernel_cuda(at::TensorIteratorBase&)::{...` | 28700 |   0.5% | 29000 |

## Run: spec

- rounds=67, accepted=134/268 (50.0%), bonus_rounds=21
- wall time: 7.734 s
- framework-op self-CUDA total: 2.868 s ( 37.1% of wall)
- device-kernel self-CUDA total: 3.904 s ( 50.5% of wall — cross-check)
- framework-op self-CPU total: 5.885 s
- mean wall per step: 115430.1 µs
- mean kernel launches per step: 8689
- **launch-bound signal**: framework CPU 5.885s > device kernels 3.904s. Host-side Python/dispatcher overhead is on the critical path; CUDA Graphs or a single fused decode-step kernel would address this in addition to any per-op speedup.

### Category breakdown over framework ops (self-CUDA, % of wall)

| category | self-CUDA µs | % of wall | events |
|---|---:|---:|---:|
| matmul | 1393237 |  18.0% | 150374 |
| other | 716848 |   9.3% | 1267075 |
| elementwise | 273832 |   3.5% | 220571 |
| kv_io | 215796 |   2.8% | 123307 |
| norm_act | 133795 |   1.7% | 120771 |
| attention_core | 96138 |   1.2% | 29940 |
| memory_layout | 35856 |   0.5% | 856763 |
| sampling | 2534 |   0.0% | 403 |
| embedding | 0 |   0.0% | 337 |
| gqa_expand | 0 |   0.0% | 19960 |

### Top 15 framework ops by self-CUDA time

| op | category | self-CUDA µs | % wall | self-CPU µs | count |
|---|---|---:|---:|---:|---:|
| `aten::mm` | matmul | 1393237 |  18.0% | 467316 | 50237 |
| `aten::fill_` | other | 701530 |   9.1% | 15447 | 9311 |
| `aten::copy_` | kv_io | 215796 |   2.8% | 282154 | 123307 |
| `aten::mul` | elementwise | 165039 |   2.1% | 355823 | 120434 |
| `aten::add` | elementwise | 78701 |   1.0% | 276302 | 80177 |
| `aten::mean` | norm_act | 61903 |   0.8% | 190562 | 40257 |
| `aten::_flash_attention_forward` | attention_core | 55169 |   0.7% | 44675 | 7008 |
| `aten::_efficient_attention_forward` | attention_core | 40969 |   0.5% | 14890 | 2972 |
| `aten::rsqrt` | norm_act | 38299 |   0.5% | 109452 | 40257 |
| `aten::cat` | memory_layout | 35856 |   0.5% | 77173 | 19960 |
| `aten::pow` | norm_act | 33593 |   0.4% | 169342 | 40257 |
| `aten::neg` | elementwise | 30092 |   0.4% | 60728 | 19960 |
| `aten::where` | other | 4973 |   0.1% | 19048 | 5944 |
| `aten::ge` | other | 4190 |   0.1% | 12433 | 2972 |
| `aten::arange` | other | 4111 |   0.1% | 18337 | 11888 |

### Top 10 device kernels (cross-check)

| kernel | self-CUDA µs | % wall | launches |
|---|---:|---:|---:|
| `_fused_gate_up_silu_kernel` | 1036807 |  13.4% | 10842 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(in...` | 695020 |   9.0% | 695 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(cutlass_80_wmma...` | 564186 |   7.3% | 12468 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8>(cutlass_80_wmma...` | 254848 |   3.3% | 2479 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_b...` | 201775 |   2.6% | 13888 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x1_tn_align8>(cutlass_80_wmma...` | 182665 |   2.4% | 248 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kern...` | 108275 |   1.4% | 42592 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c1...` | 106360 |   1.4% | 66041 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_b...` | 91831 |   1.2% | 13888 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8>(cutlass_80_wmma...` | 83609 |   1.1% | 7000 |

## Bottleneck and recommendation

On the **greedy** decode, the dominant framework category is **matmul** at **42.8%** of wall time.

That clears the >30% bar in DESIGN.md, so the Stage A.5 Triton kernel target is in the **matmul** family.

### GEMV vs attention
- linear projections (`matmul`): ** 42.8%** of wall
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