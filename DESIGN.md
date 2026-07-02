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

The headline question is how MLA compression interacts with speculative decoding under three distinct regimes:

1. **Both uncompressed** — baseline MHA spec decode.
2. **Target only compressed** — the naive application. The draft was calibrated against the *original* target's distribution; any post-conversion shift in target's logits potentially lowers draft-target agreement, partially offsetting the per-round verification speedup from cheaper KV reads.
3. **Both target and draft compressed (coupled distortion)** — the non-obvious hypothesis. If the same activation-aware SVD pipeline is applied to both models on the same calibration data, the resulting distortions in target and draft are correlated: both models drift in similar directions, potentially preserving (or even improving) draft-target alignment relative to (2).

A priori, the sign of the interaction is unclear in any of these regimes. At 4× compression with ≤2% perplexity loss, ~98% of token-level argmaxes are unchanged — acceptance might barely move. The Pareto frontier across (target-compression × draft-compression × draft-K × VRAM ceiling) is the contribution; a finding that coupled distortion preserves acceptance materially better than target-only is the cleanest headline outcome, but a "no significant interaction" result is also informative and publishable.

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

Status (May 2026):
- Manual Qwen3 forward pass — RMSNorm, RoPE, GQA + QK-norm, SwiGLU, decoder block. Bit-exact with `transformers.AutoModelForCausalLM` on Qwen3-0.6B.
- Sequential speculative decode characterized over K ∈ {1,2,3,4,5,7} on RTX 5060 Ti 16 GB BF16, 100 Dolly-15k prompts × 200 generated tokens. **Headline: 1.29× spec/greedy at K=2–3 with Qwen3-0.6B draft (50.3 vs 38.9 tok/s).** Acceptance rate peaks at 78.9% at K=1 but per-round overhead wins; falls to 42.7% at K=7. Self-spec produces token-for-token plain greedy in FP32.
- Eager greedy matches HuggingFace `AutoModelForCausalLM` with `use_cache=True` to one decimal (38.9 tok/s, 1.00× HF) — confirms the manual forward pass is not pathologically slow despite no `torch.compile`.
- 79 passing tests on M2 Mac and CUDA desktop (CPU/MPS, BF16 + FP32).
- E2E bench CLI (`bench.e2e`) and demo CLI (`bench.demo`) for reproducible runs.

Stage A is the validation harness for everything that follows — every conversion or compression variant is checked by running it end-to-end through this engine.

### Stage A.5 — One Triton kernel (DONE)

Profile the engine on a CUDA GPU; identify the actual decode-step hot path; implement and microbenchmark a Triton kernel.

**Stage A.5 success criteria (original):**
- Triton kernel ≥ eager equivalent on target shapes (microbenchmark): **MET**.
- E2E speedup of ≥1.5× over the un-Triton'd Stage A baseline on Qwen3-4B greedy: **NOT MET** (best result 1.04× greedy, 0.97× on the spec hot path).

#### Stage A.5 retrospective

Profile data on Qwen3-4B BF16 decode (`experiments/stage_a/profile_summary.md`) showed matmul = 77.4% of wall time, attention = 1.5%, RMSNorm = 5%. Hot path is *not* attention but the per-step linear projections — at batch=1 these are GEMV (M=1) operations that are HBM-bandwidth-bound rather than compute-bound. Two fused Triton kernels were shipped:

- **A.5a — fused gate-up-silu** (`07fe2d6`, [src/kernels/fused_ffn.py](./src/kernels/fused_ffn.py)). Computes `silu(x @ Wg.T) * (x @ Wu.T)` in one launch, fusing the SwiGLU activation into the GEMV. Microbench: 1.05× over eager 2-linear+silu on Qwen3-4B FFN shapes.
- **A.5b — fused QKV projection** (`7af0978`, [src/kernels/fused_qkv.py](./src/kernels/fused_qkv.py)). Pre-concatenates Wq/Wk/Wv into a single buffer and runs one matmul. Microbench: 1.50–2.98× over eager-3-separate-Linears, but **ties** an eager-concat-Linear (single `nn.Linear` with split) at all M ∈ {1..8} — the win is from concatenation, not from Triton.

Both kernels cleared their microbench gates but did **not** translate to e2e wins on the spec-decode hot path. At K=4: eager spec = 49.8 tok/s, kerneled spec = 43.7 tok/s (kernel **hurts** by 0.88×). At K=2 and K=3 the same pattern holds.

**Mechanism inventory** (in descending order of estimated contribution):
1. **Per-call Python/Triton dispatch overhead × ~14k spec calls per 200-token generation.** Each spec round makes K+1 forward passes through 28 decoder layers × 5 ops = ~140 ops/round; at K=4 that's ~700 dispatches per round × 60 rounds ≈ 42k dispatches. Triton's per-call launch + autotune-cache lookup is small but accumulates; cuBLAS via PyTorch's caching dispatcher is faster on the per-call axis.
2. **BF16 reduction-order non-associativity drops draft/target argmax agreement.** Acceptance rate at K=4 fell from 58.3% (eager) to 54.5% (Triton); at K=3 from 64.5% to 60.8%. Triton's tiling produces a different reduction order than cuBLAS, occasionally flipping argmax under BF16. Greedy verify is unforgiving — a single argmax flip rejects the rest of the round.
3. **The microbench wins were small in absolute terms (~6-12 µs/call).** At ~120 dispatches per token × 200 tokens × 6 µs = ~144 ms, vs ~5 s of decode wall time per prompt. The maximum theoretical e2e win was ~3% even before factoring in items (1) and (2).
4. **cuBLAS gemvx on M=1 BF16 GEMV is already near peak HBM bandwidth.** RTX 5060 Ti peak HBM ≈ 448 GB/s; greedy decode at 38.9 tok/s × 8 GB model ≈ 311 GB/s = ~70% of peak. Custom Triton can shave a few percent off; it cannot shave 50%.

**Falsified hypotheses (worth recording as warnings to future-self):**
- *"A bigger draft model will lift acceptance enough to clear 1.5×."* Tried Qwen3-1.7B as draft; acceptance climbed (64.5% → 70.3% at K=3) but per-step compute cost grew faster (1.29× → 1.19× speedup). The 0.6B draft beats the 1.7B at every K tested.
- *"Triton kernels that win in microbench will help spec decode if we reduce overhead."* Adding an autotune-cache prewarm (`8566854`) recovered some of the gap (25.9 → 29.2 tok/s on a 64-token demo), but the underlying mechanisms above kept Triton spec ≤ eager spec on a 100-prompt × 200-token formal bench.
- *"K=4 is roughly optimal for 0.6B/4B at greedy verify."* (DESIGN.md Stage C originally specified K ∈ {3,5,7}, not K ∈ {1,2,3,4,5,7}.) Actual optimum is K=2–3; K=4 is suboptimal by ~3-4%; K=1 is worse than K=2 because per-round overhead dominates one drafted token.

**Implications for future stages:**
- Stage C's K axis should be {2, 3, 5} not {3, 5, 7} — K=2 and K=3 are tied within run-to-run noise and either is a defensible choice; K=7 is below the per-round-overhead breakeven.
- Stage B / C's optimization budget should not be spent on more Triton kernels for batch=1 GEMV; the regime is HBM-bound and cuBLAS is already near peak. Real follow-on candidates if more spec speedup is wanted: (a) **CUDA Graphs** to amortize the ~14k dispatch overhead; (b) **probabilistic verify** (Leviathan et al.) to recover the BF16-induced acceptance loss; (c) **fused MLA-style attention** in Stage B, which is a different kernel target (the K/V latent absorb fuses Q and attention into a single op — a different shape and a different bottleneck).

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
- **Target compression ratio:** {1× (baseline MHA), 2×, 4×, 6×, 8×}.
- **Draft compression ratio:** {1× (uncompressed), matched-to-target, independently-swept}.
- **Speculative-decode K:** {3, 5, 7}.
- *(Optional)* INT4 quantization of weights via `bitsandbytes` or hand-rolled.

Three named regimes anchor the (target × draft) plane and are reported individually:

- **R1 — both uncompressed:** baseline MHA spec decode. Establishes the reference acceptance rate and tok/s.
- **R2 — target only compressed:** measures the verification-speedup vs acceptance-rate-shift trade-off when only the target is converted (the naive application).
- **R3 — both compressed at matched ratios (coupled distortion):** measures whether applying the same SVD pipeline + calibration data to both models preserves draft-target alignment relative to R2.

Metrics per cell: tok/s, acceptance rate, perplexity (target only — draft perplexity is irrelevant), max usable sequence length, peak VRAM.

Output:
- Pareto frontier chart (throughput vs perplexity, parameterised by target compression and faceted by regime R1/R2/R3).
- A "regime delta" plot: acceptance rate at R3 minus acceptance rate at R2 across compression ratios — directly tests the coupled-distortion hypothesis.
- 5–10 page writeup. Target: workshop / blog post.
- Reproducible code + cached calibration data + per-cell logs.

**Stage C success criteria:**
- Phase diagram across (target compression × draft compression × K) showing where compression is complementary to spec decode vs where it costs more in acceptance than it gains in verification speed.
- A clear answer on whether coupled distortion (R3) preserves acceptance materially better than target-only (R2). Either sign is publishable; null result is publishable too.
- One headline number for the abstract / writeup (e.g., "4× KV compression on both target and draft preserves 95% of baseline acceptance rate while extending usable context from X to Y at fixed 16 GB VRAM").

---

## Objective, initialization, and healing design

This section formalizes *what* the conversion + healing pipeline minimizes and *why* it is structured as it is. It supersedes the informal "healing = LoRA on the calibration set" of Stage B step 5.

### Prior art and where this sits

Every component of this pipeline has canonical prior art; the design is a re-derivation of the established compression playbook, not a new algorithm. The relevant lineage:

- **MLA (target architecture):** DeepSeek-V2 (Liu et al., arXiv:2405.04434, 2024) — low-rank joint KV compression, 93.3% KV reduction.
- **Post-hoc GQA→MLA conversion:** TransMLA (Meng et al., arXiv:2502.07864, NeurIPS 2025 Spotlight) — first framework, ~6B-token heal. CARE (arXiv:2603.17946, Mar 2026) — the closest single method to this one (see below). Attention Editing (arXiv:2604.05688, 2026) — random-init + progressive distillation alternative.
- **Activation-aware SVD:** ASVD (Yuan et al., arXiv:2312.05821, 2023) — introduced weighting SVD by input-activation statistics; SVD-LLM (Wang et al., arXiv:2403.07378, ICLR 2024) — truncation-aware whitening (the `√C·W` formulation used here).
- **Second-order / Fisher-weighted compression:** OBD (LeCun et al., 1990) → OBS (Hassibi & Stork, 1992) → OBC (Frantar et al., 2022) → GPTQ/SparseGPT. FWSVD (Hsu et al., 2022) weights SVD by the diagonal Fisher; **GFWSVD (Chekalina et al., arXiv:2505.17974, 2025)** generalizes it to a K-FAC Fisher `I_F = A ⊗ B` (input-covariance ⊗ output-gradient-covariance) — the exact factorization invoked in Phase 1.
- **Distillation recovery:** Hinton, Vinyals & Dean (2015) — the foundational KL-to-teacher-logits mechanism.
- **LoRA healing of an attention swap:** LoLCATs (Zhang et al., arXiv:2410.10254, 2024) — LoRA recovers quality after a subquadratic-attention swap.
- **Non-uniform rank allocation:** ASVD's sensitivity-rank search (STRS); AdaSVD/adaCR (arXiv:2502.01403, 2025); CARE's water-filling.

**CARE is the nearest published method** — it combines activation-aware (covariance-whitened) SVD init + water-filling rank allocation + CE+KL healing distillation for MHA/GQA→MLA, and it evaluates Qwen3-4B. This pipeline differs in four concrete ways, none of which CARE covers together: (1) **joint** `[K_nope; V]` latent (shared down-projection, closer to native MLA) vs CARE's **separate** per-matrix K and V factoring; (2) an explicit **Fisher / output-gradient** weighting (GFWSVD-style) on top of the input-covariance whitening CARE stops at; (3) **distill-to-raw-original** as the primary clean estimand (CARE mixes CE+KL and full-finetunes) with **LoRA-scoped** healing; (4) a characterization of **why naive rank allocation fails** that corroborates and explains CARE's own PPL-vs-uniform result. The specific assembly — Fisher-weighted SVD init + distill-to-original refinement + sensitivity-weighted allocation on a *joint* MLA latent — does not appear as a single published method.

### What we actually minimize

The pipeline targets one thing: the compressed model should reproduce the **original model's output distribution**. Per-layer reconstruction error is only a *proxy* for this — adopted because it is closed-form, not because it is the objective. Formally, with original (teacher) `M₀` producing `p₀(·|x)` and compressed student `M_θ = MLA(M₀) + LoRA` producing `p_θ(·|x)`, the ideal objective is

    min_θ  E_x  KL( p₀(·|x) ‖ p_θ(·|x) )

approximated over a finite corpus `D` — we can only be "behavior-aware" on the data we have. **Corollary:** fidelity is only as broad as `D`. Distilling on WikiText preserves the original on WikiText-like inputs, with no guarantee off-distribution. Use as diverse a `D` as feasible, evaluate across domains, and scope claims to `D`'s support.

### Two phases: loss-aware closed-form init, then exact-loss refinement

The objective above is non-convex, so we split it into a cheap init and an iterative refinement — and the key principle is *use the true loss where you can iterate, and its best closed-form surrogate where you cannot.*

**Phase 1 — initialization (closed-form).** A compression perturbation `ΔW` changes the loss, to second order at a converged model, by `ΔL ≈ ½ vec(ΔW)ᵀ F vec(ΔW)` where `F` is the Fisher (Gauss-Newton stand-in for the Hessian). So **Fisher-weighting *is* the loss, to second order** — the best closed-form surrogate. The current activation-aware SVD minimizes `tr(ΔW · C · ΔWᵀ)` with `C = E_D[xxᵀ]`: this is the *change in each layer's output*, weighted by the **input** covariance, implicitly assuming all output directions matter equally to the loss. That is only the input half of the Fisher; the missing **output-sensitivity** half (a backward-pass quantity) is exactly why per-layer reconstruction misranks layers — see the adaptive-rank result below. A Fisher-weighted SVD (empirical Fisher diagonal = `E_D[g²]`, one backward pass) restores the missing half and yields a loss-aware init in closed form.

**Phase 2 — refinement (iterative, exact loss).** Gradient-train the MLA factors (SVD-initialized) and the LoRA adapters (q/o/mlp) **jointly** against the true objective. Since we have the real gradient here, we do **not** use the Fisher surrogate — a quadratic approximation of the loss is strictly dominated by the loss itself. The refinement objective is **forward-KL distillation to a teacher**, `E_{x∈D} Σ_t KL(p_teacher ‖ p_θ)` (the student's cross-entropy against the teacher's soft labels), measured on a **held-out** slice for honest reporting.

### Teacher choice (both correct; primary vs. specialized)

Distilling to a teacher — rather than to one-hot data labels — is what removes the domain-adaptation confound: cross-entropy on data makes the student simultaneously *recover compression* and *learn adaptation*, and those are inseparable under one-hot targets (this is why the CE-heal frontier required an explicit control run to interpret). Choosing the teacher is choosing *what uncompressed behavior to preserve*.

**Primary — teacher = raw original `M₀` (behavior preservation).** The student learns to be "the original model, compressed." This is the fundamental, task-agnostic question and it is self-cleaning:
- No adaptation is injected, so there is *nothing to control for* — the number is honest by construction.
- The reference apparatus collapses to a single model: compression cost = held-out `PPL(student) − PPL(M₀)`, or held-out `KL`/top-1 agreement to `M₀`. **No separate control run.**
- The LoRA is zero-initialized and its role is purely **compression-correction** (not adaptation).
- The ceiling is `M₀`: `KL → 0` means indistinguishable from the original.

**Secondary — teacher = control `M_ctrl` = `M₀ + LoRA` adapted to a task `T` (specialized).** The student learns to be "the task-adapted model, compressed." Worth running and describing, but not the main focus. Isolates the compression cost *on top of* an adaptation: warm-start the student's LoRA from `M_ctrl`'s LoRA so that at init the *only* teacher-student difference is the MLA compression, and training descends exactly that perturbation.

### Relationship to the results already obtained

- The committed **CE-heal frontier** (V-only 1.33× at +0.5% … 4× at +16% vs a same-budget FT control) is the *task-adaptation* variant, measured with an explicit control. It stays as-is. The distill-to-`M₀` numbers are **not comparable** to it — they target ≈ `PPL(M₀)` + cost (above 21.64), not the adaptation-driven 12.91 — because they measure a different, cleaner quantity (behavior preservation vs. task performance).
- The **adaptive-rank negative result** (raw-σ² water-filling ≈ 6× worse than uniform on PPL) is the same missing ingredient in a different place: allocating rank by input-weighted reconstruction error drops the output-sensitivity half of the Fisher, so it starves early layers whose errors propagate most. This **corroborates CARE**, whose own table shows adjusted-rank has *higher* PPL than uniform (Llama-8B rank 256: CARE-U 48.35 vs CARE-E 49.43) — non-uniform allocation trades PPL for accuracy, it does not clearly win on PPL. Our raw-σ² score is catastrophic where CARE's **normalized residual-reduction score** `σ²_{r+1} / Σ_{m>r} σ²_m` is merely a near-tie; the normalization makes layers of different spectral scale comparable and prevents early-layer starvation. Two principled fixes to test: (a) CARE's normalized score, (b) Fisher/output-gradient weighting — the *same* signal that makes the Phase-1 init loss-aware.

### Concrete next experiments

1. **Normalized-score rank allocation (reconciliation with CARE).** Swap the raw-σ² allocator score for CARE's normalized residual-reduction `σ²_{r+1} / Σ_{m>r} σ²_m`; re-heal one 4× point. Tests whether normalization lifts the adaptive result from +98% toward uniform and quantifies the naive-vs-normalized gap. Cheapest, highest-information — one-line scorer change + one heal.
2. **Distill-to-original at 4×.** Teacher `M₀`, forward-KL, zero-init LoRA, SVD warm start. Report held-out PPL, KL, and top-1 agreement vs `M₀`. Primary result.
3. **Fisher-weighted SVD init.** Empirical Fisher / output-gradient Kronecker factor from one backward pass on `D`; compare pre-FT and post-distill against the covariance-only (`√C`) init CARE stops at.
4. **Fisher-weighted rank allocation.** Revisit adaptive rank with the output-sensitivity signal instead of reconstruction spectra.
5. **Secondary: teacher = control variant**, LoRA warm-started from `control_ft/best.pt`.

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

Apache License 2.0 — see [`LICENSE`](./LICENSE) for the full text.
