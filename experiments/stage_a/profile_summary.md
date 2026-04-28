# Stage A.5 — Profile summary

- GPU: **NVIDIA GeForce RTX 5060 Ti** (cap 12.0, 15.5 GB)
- torch: 2.11.0+cu130
- target: `Qwen/Qwen3-4B`
- draft:  `Qwen/Qwen3-0.6B`
- prompt tokens: 36
- decode steps (greedy): 200; spec-decode K=4, max_new=200

## Headline

  - **greedy**: 38.3 tok/s (200 tokens in 5.22s)
  - **spec**: 31.8 tok/s (200 tokens in 6.30s)

## Run: greedy

- wall time: 5.221 s
- framework-op self-CUDA total: 4.630 s ( 88.7% of wall)
- device-kernel self-CUDA total: 4.615 s ( 88.4% of wall — cross-check)
- framework-op self-CPU total: 4.196 s
- mean wall per step: 26103.0 µs
- mean kernel launches per step: 2121

### Category breakdown over framework ops (self-CUDA, % of wall)

| category | self-CUDA µs | % of wall | events |
|---|---:|---:|---:|
| matmul | 4039558 |  77.4% | 151600 |
| elementwise | 192227 |   3.7% | 166200 |
| kv_io | 171056 |   3.3% | 86800 |
| norm_act | 105659 |   2.0% | 94200 |
| attention_core | 79566 |   1.5% | 21600 |
| memory_layout | 25807 |   0.5% | 629600 |
| other | 15550 |   0.3% | 877304 |
| sampling | 1072 |   0.0% | 200 |
| embedding | 0 |   0.0% | 200 |
| gqa_expand | 0 |   0.0% | 14400 |

### Top 15 framework ops by self-CUDA time

| op | category | self-CUDA µs | % wall | self-CPU µs | count |
|---|---|---:|---:|---:|---:|
| `aten::mm` | matmul | 4039558 |  77.4% | 368904 | 50600 |
| `aten::copy_` | kv_io | 171056 |   3.3% | 191034 | 86800 |
| `aten::mul` | elementwise | 117960 |   2.3% | 270608 | 94000 |
| `aten::_flash_attention_forward` | attention_core | 79566 |   1.5% | 47850 | 7200 |
| `aten::add` | elementwise | 52439 |   1.0% | 183546 | 57800 |
| `aten::mean` | norm_act | 44752 |   0.9% | 131544 | 29000 |
| `aten::rsqrt` | norm_act | 27614 |   0.5% | 76558 | 29000 |
| `aten::cat` | memory_layout | 25807 |   0.5% | 56389 | 14400 |
| `aten::pow` | norm_act | 24782 |   0.5% | 118841 | 29000 |
| `aten::neg` | elementwise | 21828 |   0.4% | 42788 | 14400 |
| `Command Buffer Full` | other | 14542 |   0.3% | 778696 | 10743 |
| `aten::silu` | norm_act | 8510 |   0.2% | 24573 | 7200 |
| `aten::argmax` | sampling | 1072 |   0.0% | 1631 | 200 |
| `Activity Buffer Request` | other | 560 |   0.0% | 66871 | 53 |
| `aten::index_select` | other | 259 |   0.0% | 1223 | 200 |

### Top 10 device kernels (cross-check)

| kernel | self-CUDA µs | % wall | launches |
|---|---:|---:|---:|
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_b...` | 3151819 |  60.4% | 43400 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_b...` | 887738 |  17.0% | 7200 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kern...` | 93272 |   1.8% | 28800 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c1...` | 64155 |   1.2% | 43200 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::...` | 53049 |   1.0% | 29000 |
| `void pytorch_flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass...` | 45911 |   0.9% | 4248 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float,...` | 44752 |   0.9% | 29000 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<fl...` | 30984 |   0.6% | 29000 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::rsqrt_kernel_cuda(at::TensorIteratorBase&)::{...` | 27614 |   0.5% | 29000 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<float>, std::array<char...` | 27283 |   0.5% | 29000 |

## Run: spec

- rounds=63, accepted=139/252 (55.2%), bonus_rounds=22
- wall time: 6.299 s
- framework-op self-CUDA total: 2.877 s ( 45.7% of wall)
- device-kernel self-CUDA total: 2.876 s ( 45.7% of wall — cross-check)
- framework-op self-CPU total: 4.956 s
- mean wall per step: 99980.5 µs
- mean kernel launches per step: 9115
- **launch-bound signal**: framework CPU 4.956s > device kernels 2.876s. Host-side Python/dispatcher overhead is on the critical path; CUDA Graphs or a single fused decode-step kernel would address this in addition to any per-op speedup.

### Category breakdown over framework ops (self-CUDA, % of wall)

| category | self-CUDA µs | % of wall | events |
|---|---:|---:|---:|
| matmul | 2127057 |  33.8% | 197782 |
| elementwise | 268277 |   4.3% | 216875 |
| kv_io | 199577 |   3.2% | 115987 |
| norm_act | 135995 |   2.2% | 122995 |
| attention_core | 89524 |   1.4% | 28164 |
| memory_layout | 33717 |   0.5% | 843710 |
| other | 20964 |   0.3% | 1248276 |
| sampling | 2292 |   0.0% | 379 |
| embedding | 0 |   0.0% | 317 |
| gqa_expand | 0 |   0.0% | 18776 |

### Top 15 framework ops by self-CUDA time

| op | category | self-CUDA µs | % wall | self-CPU µs | count |
|---|---|---:|---:|---:|---:|
| `aten::mm` | matmul | 2127057 |  33.8% | 516135 | 66033 |
| `aten::copy_` | kv_io | 199577 |   3.2% | 257567 | 115987 |
| `aten::mul` | elementwise | 164997 |   2.6% | 358479 | 122678 |
| `aten::add` | elementwise | 74852 |   1.2% | 245326 | 75421 |
| `aten::mean` | norm_act | 57395 |   0.9% | 173807 | 37869 |
| `aten::_flash_attention_forward` | attention_core | 50925 |   0.8% | 40830 | 6532 |
| `aten::_efficient_attention_forward` | attention_core | 38599 |   0.6% | 14062 | 2856 |
| `aten::rsqrt` | norm_act | 35924 |   0.6% | 101697 | 37869 |
| `aten::cat` | memory_layout | 33717 |   0.5% | 72756 | 18776 |
| `aten::pow` | norm_act | 31506 |   0.5% | 154524 | 37869 |
| `aten::neg` | elementwise | 28427 |   0.5% | 56316 | 18776 |
| `aten::silu` | norm_act | 11170 |   0.2% | 32724 | 9388 |
| `aten::fill_` | other | 6213 |   0.1% | 13497 | 8220 |
| `aten::where` | other | 4790 |   0.1% | 17682 | 5712 |
| `aten::ge` | other | 4045 |   0.1% | 11856 | 2856 |

### Top 10 device kernels (cross-check)

| kernel | self-CUDA µs | % wall | launches |
|---|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8>(cutlass_80_wmma...` | 835993 |  13.3% | 8043 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(cutlass_80_wmma...` | 535603 |   8.5% | 12033 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_b...` | 396237 |   6.3% | 25872 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x1_tn_align8>(cutlass_80_wmma...` | 170114 |   2.7% | 231 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c1...` | 100587 |   1.6% | 62254 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kern...` | 99374 |   1.6% | 40060 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_b...` | 86306 |   1.4% | 12936 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8>(cutlass_80_wmma...` | 77942 |   1.2% | 6524 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::...` | 69186 |   1.1% | 37869 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float,...` | 57398 |   0.9% | 37869 |

## Bottleneck and recommendation

On the **greedy** decode, the dominant framework category is **matmul** at **77.4%** of wall time.

That clears the >30% bar in DESIGN.md, so the Stage A.5 Triton kernel target is in the **matmul** family.

### GEMV vs attention
- linear projections (`matmul`): ** 77.4%** of wall
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