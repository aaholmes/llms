# Post-hoc MHA→MLA Conversion on a Single GPU

**Activation-aware SVD with partial-RoPE for KV-cache compression of an existing transformer, validated on a from-scratch inference engine**

---

## One-line pitch

Convert an existing MHA/GQA-architecture LLM (Qwen3-4B) to Multi-head Latent Attention (MLA) post-hoc, without retraining, via activation-aware SVD with partial-RoPE; validate on a from-scratch single-GPU inference engine; characterize how compression ratio, speculative-decode acceptance rate, and perplexity interact at fixed 16 GB VRAM.

---

## Problem statement

LLM inference on a single consumer GPU is bottlenecked by KV-cache memory and HBM bandwidth. Multi-head Latent Attention (MLA), introduced by DeepSeek-V2 and used by every 2026 frontier model (DeepSeek V3/V4, Kimi K2.6, Nemotron 3), compresses the KV cache by 4–8× by absorbing K and V projections into a low-rank latent. The vast existing population of MHA/GQA models — Llama-3, Qwen3, Mistral — predates MLA and ships with full-size KV caches.

**Post-hoc conversion of MHA → MLA without retraining** is an emerging research line (MHA2MLA, Liu et al. 2025; SVD-LLM family; activation-aware variants). The technique uses calibration-data-weighted SVD to find the best low-rank approximation of the K/V projections.

The catch is RoPE. Position-dependent rotation prevents the "absorb" trick that fuses Q-projection with attention into a single matmul on the compressed latent. The standard workaround, **partial-RoPE**, splits each head's dimension into a NoPE latent-absorbed subspace and a small RoPE-only concatenated subspace — the design used by DeepSeek-V3.

This project asks: **how cleanly can a public MHA/GQA model be converted to MLA without retraining, and how does the resulting KV compression interact with speculative decoding at a fixed 16 GB VRAM budget?**

---

## Hypothesis

A post-hoc MHA→MLA conversion of Qwen3-4B via activation-aware SVD with partial-RoPE achieves at least **4× KV-cache compression** with **≤2% perplexity degradation** on standard benchmarks (HellaSwag, WikiText-103, MMLU subset), without healing finetune.

KV compression and speculative decoding interact in two opposing ways:

- **Pro:** smaller KV footprint shifts more of the decode-step memory budget to weights, improving HBM cache locality and potentially raising tok/s for both target and draft.
- **Con:** lossy K/V reconstruction can subtly shift the target's logit distribution, lowering draft-target agreement and hence the spec-decode acceptance rate.

The phase diagram across compression ratio × draft-K × VRAM ceiling is the contribution.

---

## Approach

- **Host language:** PyTorch with manual forward pass (no `transformers.AutoModelForCausalLM` in the hot path).
- **Hot-path kernel:** one custom Triton kernel, target chosen by Stage A.5 profiling (likely fused MLA-style attention).
- **MLA conversion:** activation-aware SVD with calibration-data-weighted truncation; partial-RoPE split following DeepSeek-V3's NoPE-latent + RoPE-concat design.
- **Calibration data:** ~1k samples from an instruction-tuned set (e.g., Dolly-15k); per-layer input covariances captured during forward passes through the original model.
- **No healing finetune in v1.** Healing finetune (LoRA-style) is a stretch milestone.
- **Profiling:** `torch.profiler` for Python-side, NVIDIA Nsight Compute for kernel-level (HBM throughput, L2 hit rate, SM occupancy).

---

## Stages

### Stage A — Inference engine (DONE)

A working from-scratch sequential-speculative-decoding engine for Qwen3-4B (target) + Qwen3-0.6B (draft). Manual forward pass, contiguous KV cache, greedy verification.

Status (April 2026):
- Manual Qwen3 forward pass — RMSNorm, RoPE, GQA + QK-norm, SwiGLU, decoder block. Bit-exact with `transformers.AutoModelForCausalLM` on Qwen3-0.6B.
- Sequential speculative decode with K ∈ {1,3,4,5,7}; self-spec produces token-for-token plain greedy in FP32.
- 34 passing tests on M2 Mac (CPU/MPS, BF16 + FP32).
- Demo CLI (`bench/demo.py`) for end-to-end runs.

Stage A is the validation harness for everything that follows — every conversion or compression variant is checked by running it end-to-end through this engine.

### Stage A.5 — One Triton kernel (~1 week, GPU desktop)

Profile the engine on a CUDA GPU; identify the actual decode-step hot path (likely fused SDPA-with-KV-cache, but **decision deferred to profile data**); implement and microbenchmark a single Triton kernel.

**Stage A.5 success criteria:**
- Triton kernel ≥ `torch.nn.functional.scaled_dot_product_attention` on target shapes (microbenchmark).
- E2E speedup of ≥1.5× over the un-Triton'd Stage A baseline on Qwen3-4B greedy.

### Stage B — MLA conversion (~3–4 weeks)

Build a `MLAttention` module that drops into the existing decoder block in place of `Attention`, parameterised by (compression ratio, NoPE/RoPE split).

1. **Calibration pipeline:** forward pass Qwen3-4B over ~1k calibration samples; capture per-layer input activations and compute covariances XXᵀ.
2. **Activation-aware SVD:** for each layer's K and V projections, compute the rank-r approximation that minimises reconstruction error weighted by the activation covariance (the SVD-LLM-family formulation).
3. **Partial-RoPE split:** decompose each head's `head_dim` into `d_nope + d_rope`. NoPE subspace projects through the compressed latent; RoPE subspace runs through a small uncompressed K projection that is then RoPE-rotated and concatenated for attention.
4. **Eval:** perplexity on WikiText-103, accuracy on HellaSwag and an MMLU subset, KV memory in bytes/token, throughput. Compare original Qwen3-4B vs MLA-converted variants at compression ratios {2×, 4×, 6×, 8×}.
5. **Stretch:** healing finetune via LoRA on the calibration set, ~1k–10k steps. Re-eval.

**Stage B success criteria:**
- ≥4× KV-cache compression with ≤2% perplexity degradation on Qwen3-4B (no FT).
- Reproducible conversion script: `python -m specd.mla.convert Qwen/Qwen3-4B --rank 128 --d_rope 32 --calib dolly-1k`.
- Token-for-token greedy match between converted MLA model and a reference HF implementation of MLA at the same compression rank (sanity check on the math).

### Stage C — Interaction characterization (~2 weeks)

Run the joint sweep at fixed 16 GB VRAM ceiling.

Axes:
- KV compression ratio: {1× (baseline MHA), 2×, 4×, 6×, 8×}.
- Speculative-decode K: {3, 5, 7}.
- (Optional) INT4 quantization of weights via `bitsandbytes` or hand-rolled.

Metrics per cell: tok/s, acceptance rate, perplexity, max usable sequence length, peak VRAM.

Output:
- Pareto frontier chart (throughput vs perplexity, parameterised by compression).
- 5–10 page writeup. Target: workshop / blog post.
- Reproducible code + cached calibration data + per-cell logs.

**Stage C success criteria:**
- Phase diagram showing the regime where MLA compression is *complementary* to speculative decoding versus where compression's KV-locality benefit is offset by acceptance-rate degradation.
- One headline number for the resume / interview pitch (e.g. "4× KV compression on Qwen3-4B, 30% throughput improvement at fixed VRAM, ≤2% perplexity loss").

---

## Hardware

- **Primary target:** RTX 5060 Ti 16 GB (Blackwell, SM 10.0+).
- **VRAM budget:** 16 GB. Qwen3-4B in BF16 ≈ 8 GB; baseline KV at long context dominates the rest. The 16 GB ceiling is the *forcing function* for the whole study — it makes "compress to fit" a real engineering constraint, not an academic one.
- Development on a 16 GB MacBook M2 (MPS / CPU for primitives, no Triton/CUDA). Stage A.5 onward requires the desktop.

---

## Repo layout

```
llms/
├── DESIGN.md                       # this file (public)
├── README.md                       # short summary
├── pyproject.toml
├── src/
│   ├── engine/
│   │   ├── model.py                # manual Qwen3 forward pass
│   │   ├── attention.py            # GQA + RoPE + KV cache (Stage A)
│   │   ├── mla.py                  # Stage B: MLAttention module
│   │   ├── kv_cache.py
│   │   ├── sampler.py
│   │   ├── spec_decode.py
│   │   └── weights.py
│   ├── kernels/
│   │   └── attention_triton.py     # Stage A.5 hot-path kernel
│   ├── mla/                        # Stage B conversion pipeline
│   │   ├── calibrate.py            # collect activation covariances
│   │   ├── svd.py                  # activation-aware SVD
│   │   └── convert.py              # CLI: produce MLA-converted state dict
│   └── bench/
│       ├── demo.py
│       ├── microbench.py
│       ├── profile.py
│       └── e2e.py                  # Stage C sweep driver
├── experiments/
│   ├── stage_a/
│   ├── stage_b/                    # conversion runs + eval results
│   └── stage_c/                    # joint-sweep data + figures
├── writeup/
└── tests/
```

---

## Open questions

- **Calibration sensitivity:** how much does calibration-set composition (instruction-tuned vs base, ~100 vs ~10k samples) move the no-FT perplexity?
- **Optimal partial-RoPE split:** what `(d_nope, d_rope)` minimizes perplexity at a fixed total head-dim budget?
- **Compression × spec-decode interaction:** does smaller KV improve acceptance via better cache locality, hurt it via lossy K/V mismatch with the draft, or both depending on regime?
- **Rank choice per layer:** uniform rank vs adaptive (some layers need more)? SVD-LLM literature is split.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Stage B no-FT perplexity is too high (>5% loss at 4×) | Bring forward the LoRA healing-finetune stretch milestone; it's the standard remedy and adds ~1 week. |
| Activation-aware SVD turns out to require a finetune harness we don't have | Use a published recipe (e.g. MHA2MLA reference code) rather than reinventing; budget 3 days of plumbing. |
| Stage C interaction story is uninteresting (compression and spec decode are independent) | The Pareto chart itself is still the deliverable — a "they're independent" finding is publishable. Pivot writeup accordingly. |
| 16 GB VRAM proves too tight even after compression | Drop to Qwen3-1.7B target or cap sequence length. The technique generalizes; the headline number changes. |

---

## Future extensions

The following directions are deferred — each would extend the project past the current 6–10 week budget but is a natural follow-on:

- **SM partitioning study** *(the original direction of this design doc, archived in [`DESIGN-sm-partitioning.md`](./DESIGN-sm-partitioning.md))* — characterize when explicit SM partitioning via CUDA Green Contexts beats default kernel scheduling for two-stream async speculative decoding under HBM contention. Independent research question; uses the same engine substrate.
- **Probabilistic spec-decode verify** (Leviathan et al.) — replace greedy verification with proper rejection sampling. Required for non-greedy (sampled) generation.
- **INT4/FP8 weight quantization** stacked on MLA compression. SVD-LLM literature suggests quantization wins below ~4× compression but stacks usefully above.
- **Healing finetune** as a first-class stage rather than a stretch — small LoRA training run on calibration data to recover the last 1–2% perplexity.
- **Paged KV cache** (vLLM-style) for multi-batch serving.
- **Hybrid-architecture extension** — adapt the conversion pipeline for hybrid linear/full attention models (Qwen3.5, LFM2, Nemotron 3).

---

## References

- DeepSeek-AI. "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model." 2024.
- DeepSeek-AI. "DeepSeek-V3 Technical Report." 2024.
- Liu et al. "MHA2MLA: Post-Hoc Conversion of Multi-Head Attention to Multi-head Latent Attention." 2025.
- Wang et al. "SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression." 2024.
- Lin et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." MLSys 2024.
- Leviathan, Kalman, Matias. "Fast Inference from Transformers via Speculative Decoding." ICML 2023.
- Dao. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." 2023.
- NVIDIA. CUDA Green Contexts documentation. CUDA Toolkit 12.4+.

---

## License

TBD before public release.
