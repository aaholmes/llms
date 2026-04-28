# Stage A.5 — Profile summary

- GPU: **NVIDIA GeForce RTX 5060 Ti** (cap 12.0, 15.5 GB)
- torch: 2.11.0+cu130
- target: `Qwen/Qwen3-4B`
- draft:  `Qwen/Qwen3-0.6B`
- prompt tokens: 36
- decode steps (greedy): 200; spec-decode K=4, max_new=200

## Headline

  - **greedy**: 38.1 tok/s (200 tokens in 5.24s)
  - **spec**: 29.2 tok/s (200 tokens in 6.85s)

## Run: greedy

- wall time: 5.245 s
- framework-op self-CUDA total: 2.842 s ( 54.2% of wall)
- device-kernel self-CUDA total: 4.651 s ( 88.7% of wall — cross-check)
- framework-op self-CPU total: 3.907 s
- mean wall per step: 26222.6 µs
- mean kernel launches per step: 2013

### Category breakdown over framework ops (self-CUDA, % of wall)

| category | self-CUDA µs | % of wall | events |
|---|---:|---:|---:|
| matmul | 2246177 |  42.8% | 108400 |
| elementwise | 195225 |   3.7% | 159000 |
| kv_io | 176260 |   3.4% | 86800 |
| norm_act | 98198 |   1.9% | 87000 |
| attention_core | 79737 |   1.5% | 21600 |
| memory_layout | 25790 |   0.5% | 600800 |
| other | 19468 |   0.4% | 833969 |
| sampling | 1060 |   0.0% | 200 |
| embedding | 0 |   0.0% | 200 |
| gqa_expand | 0 |   0.0% | 14400 |

### Top 15 framework ops by self-CUDA time

| op | category | self-CUDA µs | % wall | self-CPU µs | count |
|---|---|---:|---:|---:|---:|
| `aten::mm` | matmul | 2246177 |  42.8% | 298658 | 36200 |
| `aten::copy_` | kv_io | 176260 |   3.4% | 201662 | 86800 |
| `aten::mul` | elementwise | 118808 |   2.3% | 257586 | 86800 |
| `aten::_flash_attention_forward` | attention_core | 79737 |   1.5% | 48432 | 7200 |
| `aten::add` | elementwise | 54613 |   1.0% | 198309 | 57800 |
| `aten::mean` | norm_act | 44886 |   0.9% | 137556 | 29000 |
| `aten::rsqrt` | norm_act | 28626 |   0.5% | 79356 | 29000 |
| `aten::cat` | memory_layout | 25790 |   0.5% | 57393 | 14400 |
| `aten::pow` | norm_act | 24686 |   0.5% | 125429 | 29000 |
| `aten::neg` | elementwise | 21804 |   0.4% | 43639 | 14400 |
| `Command Buffer Full` | other | 15288 |   0.3% | 586154 | 10612 |
| `cuLaunchKernelEx` | other | 3482 |   0.1% | 23282 | 7200 |
| `aten::argmax` | sampling | 1060 |   0.0% | 1713 | 200 |
| `aten::index_select` | other | 255 |   0.0% | 1380 | 200 |
| `Buffer Flush` | other | 225 |   0.0% | 3231 | 53 |

### Top 10 device kernels (cross-check)

| kernel | self-CUDA µs | % wall | launches |
|---|---:|---:|---:|
| `_fused_gate_up_silu_kernel` | 1828287 |  34.9% | 7200 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_b...` | 1361641 |  26.0% | 29000 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_b...` | 884536 |  16.9% | 7200 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kern...` | 98361 |   1.9% | 28800 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c1...` | 67970 |   1.3% | 43200 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::...` | 53487 |   1.0% | 29000 |
| `void pytorch_flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass...` | 46193 |   0.9% | 4248 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float,...` | 44886 |   0.9% | 29000 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<fl...` | 31156 |   0.6% | 29000 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::rsqrt_kernel_cuda(at::TensorIteratorBase&)::{...` | 28626 |   0.5% | 29000 |

## Run: spec

- rounds=67, accepted=134/268 (50.0%), bonus_rounds=21
- wall time: 6.848 s
- framework-op self-CUDA total: 2.169 s ( 31.7% of wall)
- device-kernel self-CUDA total: 3.027 s ( 44.2% of wall — cross-check)
- framework-op self-CPU total: 5.006 s
- mean wall per step: 102213.6 µs
- mean kernel launches per step: 8665
- **launch-bound signal**: framework CPU 5.006s > device kernels 3.027s. Host-side Python/dispatcher overhead is on the critical path; CUDA Graphs or a single fused decode-step kernel would address this in addition to any per-op speedup.

### Category breakdown over framework ops (self-CUDA, % of wall)

| category | self-CUDA µs | % of wall | events |
|---|---:|---:|---:|
| matmul | 1391057 |  20.3% | 150374 |
| elementwise | 274219 |   4.0% | 220571 |
| kv_io | 215380 |   3.1% | 123307 |
| norm_act | 132696 |   1.9% | 120771 |
| attention_core | 96163 |   1.4% | 29940 |
| memory_layout | 35859 |   0.5% | 856763 |
| other | 20851 |   0.3% | 1259455 |
| sampling | 2428 |   0.0% | 403 |
| embedding | 0 |   0.0% | 337 |
| gqa_expand | 0 |   0.0% | 19960 |

### Top 15 framework ops by self-CUDA time

| op | category | self-CUDA µs | % wall | self-CPU µs | count |
|---|---|---:|---:|---:|---:|
| `aten::mm` | matmul | 1391057 |  20.3% | 442151 | 50237 |
| `aten::copy_` | kv_io | 215380 |   3.1% | 283334 | 123307 |
| `aten::mul` | elementwise | 165372 |   2.4% | 357795 | 120434 |
| `aten::add` | elementwise | 78775 |   1.2% | 278747 | 80177 |
| `aten::mean` | norm_act | 60915 |   0.9% | 191255 | 40257 |
| `aten::_flash_attention_forward` | attention_core | 55202 |   0.8% | 45031 | 7008 |
| `aten::_efficient_attention_forward` | attention_core | 40961 |   0.6% | 15058 | 2972 |
| `aten::rsqrt` | norm_act | 38259 |   0.6% | 110605 | 40257 |
| `aten::cat` | memory_layout | 35859 |   0.5% | 78341 | 19960 |
| `aten::pow` | norm_act | 33522 |   0.5% | 171024 | 40257 |
| `aten::neg` | elementwise | 30072 |   0.4% | 60513 | 19960 |
| `aten::fill_` | other | 6513 |   0.1% | 14477 | 8616 |
| `aten::where` | other | 4964 |   0.1% | 18767 | 5944 |
| `aten::ge` | other | 4187 |   0.1% | 12067 | 2972 |
| `aten::arange` | other | 4107 |   0.1% | 18297 | 11888 |

### Top 10 device kernels (cross-check)

| kernel | self-CUDA µs | % wall | launches |
|---|---:|---:|---:|
| `_fused_gate_up_silu_kernel` | 858898 |  12.5% | 9980 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(cutlass_80_wmma...` | 562091 |   8.2% | 12468 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8>(cutlass_80_wmma...` | 254853 |   3.7% | 2479 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_b...` | 202103 |   3.0% | 13888 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x1_tn_align8>(cutlass_80_wmma...` | 182524 |   2.7% | 248 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kern...` | 108376 |   1.6% | 42592 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c1...` | 106804 |   1.6% | 66041 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_b...` | 91885 |   1.3% | 13888 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8>(cutlass_80_wmma...` | 83490 |   1.2% | 7000 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::...` | 73999 |   1.1% | 40257 |

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