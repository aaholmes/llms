# Async Speculative Decoding on a Single GPU

**A characterization of HBM-bandwidth contention with CUDA Green Contexts**

---

## One-line pitch

A from-scratch single-GPU asynchronous speculative-decoding inference engine, designed to characterize when explicit SM partitioning (via CUDA Green Contexts) beats letting the CUDA scheduler interleave streams of draft and target model execution under HBM-bandwidth-bound autoregressive decoding.

---

## Problem statement

Speculative decoding accelerates autoregressive LLM inference by having a small **draft model** propose K future tokens that a larger **target model** verifies in a single batched forward pass. Production deployments typically take one of three paths:

1. **Multi-GPU**: draft on GPU 0, target on GPU 1. Eliminates resource contention; requires expensive hardware.
2. **Single-GPU sequential**: draft for K tokens, then target verifies. Trivial to implement; leaves significant wall-clock idle time on the GPU.
3. **Tiny draft heads** (Medusa, Eagle): replace the draft model with output heads grafted onto the target. Avoids contention because the "draft" is essentially free.

The fourth path — **single-GPU async**, where draft and target execute concurrently — is comparatively under-studied. The reason is non-obvious: in autoregressive decoding with small batch size, models are not compute-bound but **memory-bandwidth-bound**. Both draft and target read weights from the same High-Bandwidth Memory (HBM) pool. Naively running them on two CUDA streams therefore shares scarce HBM bandwidth, and the result can match, slightly improve on, or even regress versus sequential execution.

This project asks: **under what conditions does explicit Streaming Multiprocessor (SM) partitioning between draft and target outperform the default kernel scheduler?**

---

## Hypothesis

On a single consumer GPU running autoregressive decoding, explicit SM partitioning via CUDA Green Contexts can outperform default scheduling under regimes determined by:

- **Relative draft/target model size** — at extreme size ratios, the smaller model gets starved or the larger model gets bandwidth-saturated regardless of scheduling.
- **L2 cache fit** — if the target model's working set fits within an L2 partition implied by the SM split, partitioning may unlock cache reuse that interleaving cannot.
- **Compute-to-bandwidth ratio per kernel** — kernels that are more compute-bound (e.g., long-prompt prefill) tolerate concurrent execution differently than those that are bandwidth-bound (decode steps).
- **Cross-stream synchronization overhead** — naive two-stream execution still requires inter-stream sync points (verify before sample, sample before next-K draft) that may amortize differently with partitioning.

The expected phase diagram has at least two regimes:

- **Regime 1**: small/medium draft size, decode-only workload → naive two-stream async ≈ sequential (HBM bound). Partitioning helps modestly via L2 isolation.
- **Regime 2**: very small draft (heads) or large prefill component → naive async > sequential. Partitioning marginal.

Identifying the boundary is the contribution.

---

## Approach

- **Host language**: PyTorch. Manual forward pass (no `huggingface/transformers` modeling code in the hot path), HF weights loaded directly.
- **Hot-path kernels**: Triton (one custom kernel in Stage A; possibly two in Stage C if profiling indicates).
- **SM partitioning**: CUDA Green Contexts via the CUDA driver API. Accessed from Python via [`cuda-python`](https://nvidia.github.io/cuda-python/).
- **Profiling**: NVIDIA Nsight Compute for kernel-level metrics (HBM throughput, L2 hit rate, SM occupancy). `nvidia-smi dmon` for sanity checks. PyTorch profiler for Python-side overhead.
- **Empirical method**: factorial sweep over SM partition ratios × model size pairs × batch sizes × acceptance rates, with each cell yielding tok/s and a profile vector.

---

## Stages

### Stage A — Sequential baseline (~2 weeks, 30–40 hrs)

Working artifact: a PyTorch implementation of sequential speculative decoding for **Llama-3.2-1B** (target) with **Qwen-2.5-0.5B** as draft.

Required pieces:

1. **Model loading**: Pull HF weights, instantiate manual forward pass. No `model.generate()`, no `transformers.LlamaForCausalLM` in the hot path.
2. **Decoder block**: RoPE, Grouped-Query Attention (GQA), RMSNorm, SwiGLU FFN.
3. **KV cache**: contiguous (per-layer `[batch, num_kv_heads, max_seq_len, head_dim]`). Paged variant deferred to optional stretch.
4. **Sampling**: greedy, top-k, top-p (nucleus), temperature.
5. **Speculative decode loop**:
   - Draft proposes K tokens autoregressively (K ∈ {3, 4, 5, 7}).
   - Target verifies all K + 1 in a single forward pass.
   - Accept the longest agreeing prefix; sample the corrected token from the target distribution.
6. **One Triton kernel** for the identified hot path. Candidates:
   - Fused scaled-dot-product attention with KV cache (likely choice).
   - Fused logit-sample-decode step (top-p in-kernel).
   - Decision deferred until Stage A profiling identifies the actual bottleneck.

**Stage A success criteria**:

- Llama-3.2-1B greedy generation matches HF `generate()` token-for-token on a fixed prompt set.
- Speculative speedup ≥ 1.5× over greedy on a representative prompt distribution.
- Triton kernel matches or beats `torch.nn.functional.scaled_dot_product_attention` on the target model's shapes.

### Stage B — Validation spike (~2 days)

**Before** committing to Stage C, run a controlled experiment to validate the central premise.

Setup: draft and target on two CUDA streams, no partitioning. Same problem instance as Stage A baseline. Measure:

- End-to-end tok/s, sequential vs naive async.
- HBM bandwidth utilization (Nsight Compute).
- L2 cache hit rate per stream.
- Achieved SM occupancy.

Three possible outcomes — each pivots the project:

| Outcome | Interpretation | Pivot |
|---|---|---|
| Naive async ≈ sequential | HBM-bound. Streams compete for the same finite resource. | **Proceed to Stage C.** SM partitioning has well-defined upside via L2 isolation. |
| Naive async > sequential by ≥10% | Compute headroom exists; streams successfully parallelize. | Stage C value reduced. Pivot to characterizing *why* it works (cache effects? compute overlap?), or pivot to a different research question (e.g., L2-aware scheduling). |
| Naive async < sequential | Scheduler thrash dominates concurrent execution. | Stage C upside is largest. Partitioning should clearly win. |

Stage B is the cheapest insurance against committing four weeks to a question with a boring answer.

### Stage C — Async with SM partitioning + characterization (~3–4 weeks)

Build the partitioned engine and run the characterization sweep.

1. **Plumbing**: integrate `cuda-python` for Green Context creation and per-stream binding.
2. **Engine**: launch draft kernels on context A (K SMs) and target kernels on context B (N − K SMs). Manage stream synchronization at speculative-decode boundaries (verify → sample → next-K draft).
3. **Characterization sweep** — factorial design over:
   - SM split ratios: {25/75, 33/67, 50/50, 67/33, 75/25} (constrained by Blackwell SM granularity).
   - Draft K: {3, 5, 7}.
   - Target/draft model size pairs: {1B/0.5B, 3B/0.5B, 8B/1B if VRAM permits}.
   - Batch size: {1, 2, 4}.
   - Prefill vs decode-only.
4. **Outputs**:
   - Phase diagram (heatmap) of partitioning speedup over naive async, across the 5+ axes.
   - Per-cell Nsight Compute report capturing HBM utilization, L2 hit rate, SM occupancy.
   - 5–10 page writeup with charts. Target venue: workshop paper at MLSys / EuroSys / MLArchSys, or a substantive blog post.

**Stage C success criteria**:

- A characterization that an external reviewer could reproduce with the published code.
- At least one regime where SM partitioning produces ≥10% speedup over naive async (otherwise the writeup is "negative result, but here's what we learned" — still publishable, less interesting).
- A predictive model (even informal) for when partitioning is expected to win.

---

## Hardware

- **Primary target**: RTX 5060 Ti 16GB (Blackwell, SM 10.0+).
- **VRAM budget**: 16 GB. Comfortably fits Llama-3.2-1B (FP16 ≈ 2.5 GB) + Qwen-0.5B (≈1 GB) + KV caches + activations. Larger pairs (3B/0.5B, 8B/1B) require quantization or offloading.

**Pre-flight checks before committing to Stage C**:

- CUDA driver ≥ 12.4 installed (Green Contexts requirement).
- `cuda-python` package installs and imports cleanly.
- Verify Green Context feature support on the specific SKU. **Some Green Context features are SM 9.0+ datacenter-only** (H100/B100 class). Confirm via NVIDIA developer docs and a minimal "create-and-bind" smoke test before designing around the API.
- Nsight Compute installed and functional with the consumer driver.

If any pre-flight check fails, the project pivots to a non-Green-Context partitioning mechanism (CUDA streams + `cudaStreamAttrPriority`, or MPS) — the research question survives, the implementation differs.

---

## Repo layout (proposed)

```
llms/
├── DESIGN.md                  # this file (public)
├── MOTIVATION.md              # private
├── README.md                  # short summary, links to DESIGN
├── pyproject.toml
├── src/
│   ├── engine/
│   │   ├── model.py           # manual Llama/Qwen forward pass
│   │   ├── attention.py       # GQA + RoPE + KV cache
│   │   ├── kv_cache.py        # contiguous (Stage A) and paged (stretch)
│   │   ├── sampler.py         # greedy / top-k / top-p / temperature
│   │   └── spec_decode.py     # speculative decoding loop
│   ├── kernels/
│   │   └── attention_triton.py # the hot-path Triton kernel
│   ├── partitioning/
│   │   ├── green_ctx.py       # cuda-python Green Context wrapper
│   │   └── stream_mgr.py      # stream + context lifecycle
│   └── bench/
│       ├── microbench.py      # individual kernel benches
│       ├── e2e.py             # end-to-end tok/s
│       └── sweep.py           # Stage C factorial sweep driver
├── experiments/
│   ├── stage_a/               # baseline numbers + plots
│   ├── stage_b/               # validation spike data
│   └── stage_c/               # characterization sweep + phase diagrams
├── writeup/
│   ├── paper.tex              # workshop paper draft
│   └── figures/
└── tests/
```

---

## Open questions

These need answers during implementation; flagged here to prevent surprise mid-stage.

- **Blackwell consumer Green Context support**: which features are available on RTX 5060 Ti? (Pre-flight check above.)
- **L2 partitioning controllability**: is L2 cache split a user-tunable knob alongside SM split, or implicit / driver-managed?
- **Triton-Green-Context interaction**: does Triton respect a Green Context binding, or does it launch on the parent context regardless?
- **Hot-path kernel choice**: attention vs sample-decode? Resolve via Stage A profiling, not upfront speculation.
- **KV cache pressure**: at what target-model size does KV cache start crowding HBM enough that partitioning gains evaporate?

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Scope explosion (research arcs always overrun) | Strict stage gates. Stage A must ship a working artifact before Stage B begins. Stage B's pivot rules are written down; do not skip the spike. |
| Green Contexts unavailable on consumer Blackwell | Pre-flight smoke test before Stage C. Fallback: stream priority + MPS. Research question survives. |
| Stage B finds the contention question is uninteresting | Stage A engine alone is still a tellable artifact. Pivot Stage C to a related question (paged KV cache contention, MoE routing scheduling). |
| `cuda-python` driver-API friction | Budget 1 day of plumbing work in Stage C. Encapsulate in `partitioning/green_ctx.py` so the rest of the engine doesn't need to care. |
| Negative result in characterization | Negative results are publishable in systems venues if the methodology is rigorous. The phase diagram itself is the contribution, regardless of which regions favor partitioning. |

---

## Success criteria (overall)

- **Stage A**: working speculative decode, ≥1.5× greedy speedup, one custom Triton kernel.
- **Stage B**: clear pivot decision based on profile data.
- **Stage C**: phase diagram + writeup + reproducible code.
- **Artifact quality**: an external systems researcher should be able to reproduce the central plots from the published code in <1 day.

---

## References

- Leviathan, Kalman, Matias. "Fast Inference from Transformers via Speculative Decoding." ICML 2023.
- Cai et al. "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads." 2024.
- Li et al. "EAGLE / EAGLE-2: Speculative Sampling Requires Rethinking Feature Uncertainty." 2024.
- Fu et al. "Lookahead Decoding." 2024.
- Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023.
- NVIDIA. CUDA Green Contexts documentation. CUDA Toolkit 12.4+.
- Dao. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." 2023.

---

## License

TBD before public release.
