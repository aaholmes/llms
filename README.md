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
  - ✅ **Conversion CLI + post-hoc swap.** `python -m mla.convert <hf-id> --rank ... --d-rope ... --calib ...` reads the calibration artifact and runs one *joint* SVD per layer over the stacked `[W_K_nope; W_V]` rows — joint factoring is what makes the down-projection `W_dkv` shared across K-nope and V (the whole point of MLA's compressed cache). Output is a self-contained `.pt` artifact swapped into a fresh `Qwen3Model` via `apply_mla(...)`. 14 synthetic tests + 2 real-Qwen3-0.6B end-to-end smokes. **Qwen3-4B converts in 55 s on CPU** at `rank=128, d_rope=32`, giving a structural KV-cache compression of **5.33×** before perplexity is even measured (target was ≥4×).
  - ✅ **Perplexity sweep on Qwen3-4B.** Streaming PPL over a held-out 250-chunk × 1024-token slice of WikiText-103 validation (disjoint from the train-split calibration). Two grids run: the headline `partial-rope` design and a `v-only` fallback (`d_rope = head_dim`, no nope subspace, K stays full-RoPE). **The 2% PPL Δ at ≥4× compression hard gate was *not* met at this calibration with no finetuning** — see results below.
- **Stage C — joint sweep at fixed VRAM** — pending. How target-compression × draft-compression × spec-decode-K trade off at a fixed 16 GB VRAM budget. Three named regimes — both uncompressed, target only, both compressed at matched ratios — directly test whether coupling SVD distortion across target and draft preserves spec-decode acceptance relative to compressing the target alone.

## Stage B perplexity results

The conversion pipeline is correct end-to-end (the algebraic round-trip and exact-low-rank tests are tight to fp64). The empirical finding is about *the model*, not the math: post-hoc partial-RoPE on a checkpoint trained with full-RoPE catastrophically degrades quality without a healing finetune.

**Setup.** Eval slice is 250 chunks × 1024 tokens from the WikiText-103 *validation* split (disjoint from the 1k × 256-token train-split calibration). PPL is computed in BF16 on cuda using fixed-length chunks with cache reset per chunk; the absolute number is therefore higher than published sliding-window PPLs (≈21.6 here vs ~9 in the literature for similar Qwen3-class models) but the *ratio* between baseline and MLA is what matters and is consistent across both runs.

**Variant A — partial-RoPE** (the headline MLA design: K and V both compressed via shared latent, small RoPE-only K subspace kept full-rank).

| Config | rank | d_rope | KV B/tok | Compression | PPL | Δ % vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| baseline | — | — | 4096 | 1.00× | 21.64 | — |
| r256_drope32 | 256 | 32 | 1024 | 4.00× | 1986.71 | **+9079.6%** |
| r192_drope32 | 192 | 32 | 896 | 4.57× | 3586.73 | **+16472.6%** |
| r128_drope32 | 128 | 32 | 768 | 5.33× | 2970.20 | **+13623.9%** |
| r128_drope64 | 128 | 64 | 1280 | 3.20× | 1191.47 | **+5405.2%** |
| r96_drope32 | 96 | 32 | 704 | 5.82× | 3699.57 | **+16994.0%** |
| r64_drope32 | 64 | 32 | 640 | 6.40× | 5572.51 | **+25648.0%** |

**Variant B — V-only** (`d_rope = head_dim`, so no nope subspace; K stays full-rank with full RoPE preserved, only V gets factored through the latent).

| Config | rank | d_rope | KV B/tok | Compression | PPL | Δ % vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| baseline | — | — | 4096 | 1.00× | 21.64 | — |
| r1024_drope128 | 1024 | 128 | 4096 | 1.00× | 21.64 | −0.01% (noise) |
| r768_drope128 | 768 | 128 | 3584 | 1.14× | 25.68 | +18.7% |
| r512_drope128 | 512 | 128 | 3072 | 1.33× | 27.12 | +25.3% |
| r384_drope128 | 384 | 128 | 2816 | 1.45× | 29.87 | +38.0% |
| r256_drope128 | 256 | 128 | 2560 | 1.60× | 35.45 | +63.8% |
| r128_drope128 | 128 | 128 | 2304 | 1.78× | 89.86 | +315.2% |

**Diagnosis.** Three observations make the picture tight:

1. *The conversion's per-layer reconstruction error is small.* The discarded singular energy (analytic C-weighted residual) at rank=128, d_rope=32 averages 1.3e3 across layers — a few orders of magnitude below the activation magnitudes the layers see. The SVD is doing its job.
2. *V-only at maximum rank reproduces baseline to numerical noise* (Δ = −0.01% at rank=1024, d_rope=128). The runtime, factor assembly, and post-hoc swap pipeline are end-to-end correct.
3. *V-only degrades gracefully; partial-RoPE doesn't.* At V-only `rank=128` (1.78× compression) the PPL Δ is +315%; at partial-RoPE `rank=128, d_rope=32` (5.33× compression) it's +13624%. The gap between these — 40× more PPL damage at higher compression — is overwhelmingly attributable to the partial-RoPE structural change, not to the lower rank.

The mechanism: Qwen3-4B was trained to apply RoPE rotation to the *full* `head_dim` of K. Partial-RoPE attention applies the rotation only to a `d_rope`-wide subspace — algebraically a different operation. Even when the shared latent reconstructs `K_nope` faithfully, the model's downstream layers have never seen the resulting attention pattern. Every published post-hoc MHA→MLA paper resolves this by running a healing finetune (LoRA on the K/V factors over a few thousand steps), which is exactly the contingency the project plan flags as `B.stretch-2 (LoRA healing-FT)`.

**What this means for Stage C.** The 4–8× KV-cache compression promised by MLA is conditional on healing FT. Without it, the achievable compression at acceptable PPL is ≤1.45× via V-only (≈1.45× compression at +38% PPL, or ≈1.33× at +25%). The Stage C joint sweep can either (a) operate on these V-only points as the "no-FT working zone", or (b) wait on the +1-week LoRA healing-FT stretch milestone to unlock the full headline regime. That's the open decision at the Stage B → Stage C handoff.

Raw artifacts: `experiments/stage_b/eval_summary_partial_rope.md` and `eval_summary_v_only.md` (committed); per-config converted checkpoints under `experiments/stage_b/r*.pt` (gitignored).

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
