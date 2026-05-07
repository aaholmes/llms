# Post-hoc MHA→MLA Conversion on a Single GPU

## What this is, why it matters

Generating text from a large language model token-by-token is bottlenecked by **KV-cache memory bandwidth**: at every new token, the GPU has to re-read a full set of cached "keys" and "values" — one pair per attention head, per past position — from main memory before it can compute the next token. On a 16 GB consumer GPU, that cache is the single biggest constraint on context length, and its memory traffic is what keeps decode rate well below peak compute.

**Multi-head Latent Attention (MLA)**, introduced in DeepSeek-V2 and adopted by every 2026 frontier model (DeepSeek V3/V4, Kimi K2.6, Nemotron 3), shrinks that cache by 4–8× by storing a single low-rank "latent" vector per token instead of one K and V vector per attention head, and reconstructing K/V on the fly. But the existing population of MHA/GQA models — Llama-3, Qwen3, Mistral — was trained without MLA and is stuck with the original full-size cache.

This project converts an existing model (**Qwen3-4B**) to MLA *after the fact*, **without retraining**. The recipe: collect per-layer activation statistics on a calibration corpus, run an SVD weighted by those statistics to choose a latent subspace that minimizes reconstruction error *on the actual activations the model sees*, and replace the original K/V projections with a small shared down-projection plus per-head up-projections. Validation runs on a from-scratch single-GPU inference engine that's bit-exact with HuggingFace's reference. The headline study is how the resulting KV compression interacts with **speculative decoding** at a fixed 16 GB VRAM budget — i.e. once you can fit either a longer context or a faster draft model in the freed memory, what's the throughput-vs-quality Pareto frontier?

**The wrinkle MLA solves with a "partial-RoPE split":** rotary position embedding (RoPE) is a position-dependent rotation that sits algebraically *between* the Q and K projections at attention time. That breaks the trick MLA otherwise uses to never reconstruct K at runtime — the rotation prevents you from fusing the K up-projection into Q at load time. The DeepSeek-V3 fix, which this project adopts as the conversion target, is to split every attention head's dimension into a no-RoPE subspace that gets compressed (and absorbed into Q) and a small RoPE-only subspace that stays full-rank but rotated.

## Status

Hard-gated stages; see `DESIGN.md` for the plan, hardware, and the future-extensions list.

- **Stage A — inference engine** — done. From-scratch sequential speculative-decoding engine for Qwen3-4B + Qwen3-0.6B. Manual forward pass (no `transformers.AutoModelForCausalLM` in the hot path), bit-exact with HuggingFace on Qwen3-0.6B; **1.29× speculative-decoding speedup** at K=2–3 on the formal e2e bench (see below).
- **Stage A.5 — Triton kernel** — done. Two fused Triton kernels (gate-up-silu, QKV projection) shipped with microbench wins of 1.05× and 1.50–2.98×; both clear isolation gates but do not improve the spec-decode hot path end-to-end. The 1.5× e2e ship gate was missed: at batch=1 in BF16 the target model is HBM-bandwidth-bound, and per-call kernel-launch overhead × ~14k spec calls per generation erodes the per-call gain.
- **Stage B — MLA conversion** — in progress.
  - ✅ **Activation-aware SVD primitive.** Pure-CPU fp64 routine that, given a weight matrix `W` and an activation covariance `C`, returns rank-`r` factors `(A, B)` minimizing `‖(W − AB) X‖_F` over the calibration distribution (not the unweighted Frobenius distance — this is the difference between the math working and not). 8 tests, including ridge backoff for ill-conditioned `C`.
  - ✅ **Per-layer covariance collector.** Forward-hook pipeline that streams a calibration corpus (1k × 256-token chunks of WikiText-103) through the engine and accumulates `C_ℓ = (1/N) Σ x_t x_tᵀ` per attention layer. **256k tokens accumulated through Qwen3-4B in 9.4 min on the 5060 Ti**; all 36 layers symmetric and positive semidefinite, conditioning ~10⁴–10⁵ — well inside the default ridge regime. 9 tests.
  - ✅ **MLAttention runtime + compressed KV cache.** Drop-in replacement for the engine's standard attention module that consumes already-computed MLA factors, applies partial-RoPE per head, and serves attention against a cache storing `(c_kv, k_rope_pre)` per token. Single-norm-on-concatenated-K mode preserves *algebraic* equivalence to baseline GQA attention given full-rank factors — verified to ≤1e-5 in fp32 and ≤1e-10 in fp64. 12 tests covering a `d_rope` sweep, prefill+decode, and exact-low-rank construction.
  - ⏳ **Conversion CLI + post-hoc swap.** Wire the calibration artifact and the SVD primitive into a one-shot `python -m mla.convert ...` that produces an MLA-converted Qwen3-4B checkpoint loadable by the existing engine.
  - ⏳ **Perplexity sweep.** The headline gate: ≥4× KV-cache compression at ≤2% perplexity Δ on Qwen3-4B over a held-out WikiText-103 slice, swept over `(rank, d_rope)`.
- **Stage C — joint sweep at fixed VRAM** — pending. How target-compression × draft-compression × spec-decode-K trade off at a fixed 16 GB VRAM budget. Three named regimes — both uncompressed, target only, both compressed at matched ratios — directly test whether coupling SVD distortion across target and draft preserves spec-decode acceptance relative to compressing the target alone.

## Engine speedups (Stage A end-to-end)

100 Dolly-15k prompts × 200 generated tokens, Qwen3-4B target in BF16 on RTX 5060 Ti 16 GB. Greedy baseline: **38.9 tok/s** (matches HuggingFace's `AutoModelForCausalLM` greedy at `use_cache=True` to one decimal place).

| draft | K=1 | K=2 | K=3 | K=4 | K=5 | K=7 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 46.6 — 1.21× (acc 78.9%) | 49.7 — 1.29× (69.8%) | **50.3 — 1.29×** (64.5%) | 48.7 — 1.25× (58.3%) | 47.5 — 1.22× (53.2%) | 41.4 — 1.06× (42.7%) |
| Qwen3-1.7B | 43.7 — 1.13× (acc 82.6%) | 46.2 — 1.20× (75.8%) | **46.3 — 1.19×** (70.3%) | 44.7 — 1.15× (63.7%) | 42.1 — 1.08× (57.5%) | 39.0 — 1.00× (50.5%) |

**Optimal K = 2–3 for both drafts** (K=2 and K=3 are tied within run-to-run noise). The curve is sharply peaked: K=1 leaves performance on the table because per-round overhead dominates one drafted token, and K≥4 wastes draft work that the target rejects. Acceptance climbs monotonically as K drops (peaking at 78.9% / 82.6% at K=1), so the bottleneck above K=3 is acceptance, not draft cost; below K=2 it flips and the bottleneck is round overhead. The 0.6B draft beats the 1.7B at every K despite a lower acceptance rate — the 1.7B's larger draft-step compute cost outruns its acceptance-rate gain. K=7 is the breakeven point for the 1.7B draft; beyond that, drafting more tokens than the target will accept loses to a plain greedy run.

## Hardware

Target: RTX 5060 Ti 16 GB (Blackwell). Engine and conversion development on a MacBook M2 (CPU/MPS); Triton, profiling, calibration runs, and the sweeps on the CUDA desktop.

## License

TBD before public release.
