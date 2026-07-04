# Post-hoc MHA→MLA Conversion on a Single GPU — Design

**Activation-aware SVD with partial-RoPE for KV-cache compression of an already-trained transformer, validated on a from-scratch inference engine.**

---

## Problem

LLM inference on a single consumer GPU is bottlenecked by KV-cache memory and HBM bandwidth. **Multi-head Latent Attention (MLA)**, introduced in DeepSeek-V2 and used by every 2026 frontier model, compresses the KV cache 4–8× by absorbing the K and V projections into a low-rank latent. But the large existing population of MHA/GQA models — Llama-3, Qwen3, Mistral — predates MLA and ships with a full-size cache.

**Post-hoc conversion of MHA/GQA → MLA without retraining** is an emerging research line (TransMLA, MHA2MLA, the SVD-LLM family, CARE). The core tool is a calibration-data-weighted SVD that finds the best low-rank approximation of the K/V projections *on the activations the model actually produces*.

The catch is RoPE. Its position-dependent rotation sits between the Q and K projections and blocks the "absorb" trick that fuses the Q projection into attention over the compressed latent. The standard workaround — **partial RoPE**, from DeepSeek-V3 — splits each head into a no-rotation subspace that is compressed and absorbed, and a small rotated subspace that stays full-rank and is concatenated at attention time.

**The question:** how cleanly can a public MHA/GQA model be converted to MLA without retraining, and how does the resulting KV compression interact with speculative decoding at a fixed 16 GB VRAM budget?

## Hypothesis and its outcome

The original falsifiable hypothesis was: post-hoc conversion of Qwen3-4B via activation-aware SVD with partial-RoPE achieves **≥4× KV compression at ≤2% perplexity degradation, without any finetune.**

**Outcome: falsified without healing, recovered with a short one.** No-FT 4× compression lands at +826% PPL — the rank-deficiency of the K/V subspaces dominates, and no partial-RoPE fix reaches it. A ~3 h healing finetune closes the gap to **+16% at 4×** (and +0.5% at 1.33×), measured against a same-budget finetuned control. So the precise form of the claim is: *4× KV compression is achievable at modest, bounded cost, but requires a short heal, not zero training.*

## The interaction question (planned sweep)

The headline study is how MLA compression interacts with speculative decoding, at a fixed 16 GB VRAM ceiling, across four regimes:

1. **Both models uncompressed** — baseline speculative decode; reference acceptance rate and throughput.
2. **Target compressed, draft not** — the naive application. The draft was calibrated against the *original* target's distribution, so post-conversion drift in the target's logits may lower draft–target agreement, partly offsetting the cheaper KV reads.
3. **Both compressed at matched ratios (coupled distortion)** — the non-obvious case. Applying the same SVD pipeline and calibration data to both models makes their distortions *correlated* — both drift in similar directions — which may preserve, or even improve, draft–target alignment relative to (2).
4. **Draft specialized to the compressed target** — instead of shrinking the draft by SVD, align it. The single-token acceptance rate is exactly `Σ_i min(p_i, q_i) = 1 − TVD(p, q)`, where `TVD(p,q) = ½ Σ_i |p_i − q_i|` is the total variation distance (Leviathan et al.). So attach a LoRA to the draft and train it to minimize `E_workload[TVD(p_target, p_draft)]` — maximizing acceptance directly — reusing the same top-K teacher machinery as the heal (here the target is the teacher). This tests "is it better to *shrink* the draft (3) or *align* a small draft to the compressed target (4)?" and unifies the compression and speculative-decode halves under one objective. Caveats to report: exposure bias (train on draft rollouts, not just static text) and single-token TVD vs accept-*run*-length.

At 4× compression with ≤2% perplexity loss, ~98% of token-level argmaxes are unchanged, so acceptance may barely move — the sign of every interaction is a priori unclear, and a "no significant interaction" finding is as informative as a positive one. Deliverable: a throughput-vs-perplexity Pareto frontier faceted by regime, plus a regime-delta plot (acceptance under coupled distortion minus target-only, and align-vs-shrink for the specialized draft), and a short writeup.

---

## Components

### Inference engine (built)

A from-scratch sequential speculative-decoding engine for Qwen3, and the validation harness for everything downstream — every conversion variant is checked by running it end-to-end here.

- Manual Qwen3 forward pass (RMSNorm, RoPE, GQA + QK-norm, SwiGLU, decoder block), no `transformers.AutoModelForCausalLM` in the hot path. **Bit-exact with HuggingFace on Qwen3-0.6B.**
- Greedy decode matches HuggingFace tok/s to one decimal (38.9 tok/s) — confirms the manual path isn't pathologically slow despite no `torch.compile`.
- Speculative decode swept over K ∈ {1,2,3,4,5,7}, 100 Dolly-15k prompts × 200 tokens, 16 GB Blackwell GPU, BF16. **Headline: 1.29× over greedy at K=2–3** with a Qwen3-0.6B draft (50.3 vs 38.9 tok/s). Acceptance peaks at 78.9% at K=1 but per-round overhead wins at K=2–3; a Qwen3-1.7B draft raises acceptance but its per-step cost outruns the gain, so the smaller draft wins at every K.

### Fused Triton kernels (built — negative end-to-end result)

Profiling Qwen3-4B BF16 decode showed the hot path is *not* attention (1.5% of wall time) but the per-step linear projections (matmul 77%). At batch=1 these are M=1 GEMV operations that are HBM-bandwidth-bound, not compute-bound. Two fused kernels were shipped and both cleared their isolation microbench gates (fused gate-up-silu 1.05×; fused QKV 1.5–3× vs three separate Linears):

**Neither improved end-to-end speculative decode**, and the QKV kernel slightly hurt it. The reasons, recorded as warnings:
- **Per-call dispatch overhead × ~14k spec calls per generation.** cuBLAS via PyTorch's caching dispatcher is faster on the per-call axis than Triton's launch + autotune-cache lookup; the microbench wins (~6–12 µs/call) were tiny against ~5 s of decode wall time (max theoretical e2e win ~3%).
- **BF16 reduction-order non-associativity drops acceptance.** Triton's tiling reduces in a different order than cuBLAS, occasionally flipping an argmax under BF16; greedy verify rejects the rest of the round on a single flip (acceptance fell ~3–4 points).
- **cuBLAS M=1 GEMV is already near peak HBM bandwidth** (~70% of a 16 GB Blackwell card's ~448 GB/s at 38.9 tok/s). A custom kernel can shave a few percent; it cannot shave 50%.

Takeaway for the rest of the project: don't spend optimization budget on more batch=1 GEMV kernels. If more speculative speedup is wanted, the real levers are CUDA Graphs (amortize dispatch), probabilistic verify (recover the BF16 acceptance loss), or a fused MLA-attention kernel (a genuinely different shape and bottleneck).

### MLA conversion pipeline (built)

An `MLAttention` module drops into the existing decoder block in place of `Attention`, parameterized by (compression rank, no-rope/rope split).

1. **Calibration** — forward Qwen3-4B over ~1k calibration chunks; accumulate per-layer input covariance `C = E[xxᵀ]` (and, for the Fisher init, the output-gradient factor via one backward pass).
2. **Activation-aware joint SVD** — one SVD per layer over the *stacked* `[W_K_nope; W_V]` rows, giving a **shared** down-projection over K-nope and V (the MLA point), truncated to minimize `‖(W − AB)X‖` over the calibration distribution.
3. **Partial-RoPE split** — decompose each head into `d_nope + d_rope`; the no-rope subspace goes through the latent, the rope subspace keeps the original Qwen3 RoPE frequency pairs (a permutation places them contiguously at the head's end) and stays full-rank.
4. **Healing finetune** — full-train the new MLA projections (`dkv`, `uk_nope`, `uv`, `kr`) plus LoRA r=16 on Q/O and the MLP; everything else frozen, gradient-checkpointed to fit Qwen3-4B in 16 GB. Trainable ≈ 92.6 M params.
5. **Evaluation** — streaming perplexity and KL-to-original on held-out WikiText-103 (disjoint from calibration).

CLI: `python -m mla.convert Qwen/Qwen3-4B --rank 256 --d-rope 32 --calib wt103-1k`, then `python -m mla.heal --mla-artifact <...>.pt`. Results are in the README; raw artifacts under `experiments/`.

---

## Objective, initialization, and healing

This section states *what* the pipeline minimizes and *why* it is structured as it is. Every component has canonical prior art; the design is a considered assembly of the established compression playbook onto a joint MLA latent, with one genuinely new piece.

### Prior art and where this sits

- **MLA (target architecture):** DeepSeek-V2 (Liu et al., 2024).
- **Post-hoc GQA→MLA:** TransMLA (Meng et al., NeurIPS 2025); **CARE** (arXiv:2603.17946, 2026) — the nearest single method; Attention Editing (2026).
- **Activation-aware SVD:** ASVD (Yuan et al., 2023); SVD-LLM (Wang et al., ICLR 2024) — the truncation-aware `√C·W` whitening used here.
- **Fisher-weighted compression:** OBD/OBS/OBC → GPTQ; FWSVD (Hsu et al., 2022); **GFWSVD** (Chekalina et al., 2025) — K-FAC Fisher `F ≈ C ⊗ G` (input-covariance ⊗ output-gradient-covariance), the factorization invoked below.
- **Distillation recovery:** Hinton et al. (2015). **LoRA healing of an attention swap:** LoLCATs (Zhang et al., 2024).

**CARE is the nearest published method** — activation-aware SVD init + water-filling rank allocation + CE+KL healing, evaluated on Qwen3-4B. This pipeline differs in four ways CARE does not combine: (1) a **joint** `[K_nope; V]` latent with a shared down-projection (closer to native MLA) vs CARE's separate per-matrix factoring; (2) an explicit **output-gradient / Fisher** weighting on top of the input-covariance whitening CARE stops at; (3) **distill-to-raw-original** as the primary clean estimand with **LoRA-scoped** healing; (4) a characterization of **why naive rank allocation fails** that explains CARE's own PPL-vs-uniform result.

### What we actually minimize

The pipeline targets one thing: the compressed model should reproduce the **original model's output distribution**. With original teacher `M₀` producing `p₀` and compressed student `M_θ = MLA(M₀) + LoRA` producing `p_θ`, the ideal objective is

    min_θ  E_x  KL( p₀(·|x) ‖ p_θ(·|x) )

approximated over a finite corpus `D`. Per-layer reconstruction error is only a closed-form *proxy* for this. **Corollary:** fidelity is only as broad as `D` — distilling on WikiText preserves the original on WikiText-like inputs, with no guarantee off-distribution. Scope claims to `D`'s support.

### Two phases: loss-aware init, then exact-loss refinement

The objective is non-convex, so split it — *use the true loss where you can iterate, and its best closed-form surrogate where you cannot.*

**Phase 1 — closed-form init.** A compression perturbation `ΔW` changes the loss, to second order at a converged model, by `ΔL ≈ ½ vec(ΔW)ᵀ F vec(ΔW)` with `F` the Fisher. So **Fisher-weighting *is* the loss to second order.** The plain activation-aware SVD minimizes `tr(ΔW C ΔWᵀ)` — the change in each layer's *output*, weighted by the **input** covariance `C`, implicitly treating all output directions as equally important. That is only the input half of the Fisher. Adding the missing **output-sensitivity** half (`G`, one backward pass — the two-sided `√G·W·√C`) restores a loss-aware init.

**Phase 2 — iterative refinement.** Gradient-train the SVD-initialized MLA factors and the LoRA adapters *jointly* against the true objective — forward-KL distillation to a teacher, `E_{x∈D} Σ_t KL(p_teacher ‖ p_θ)`, on a held-out slice. Since the real gradient is available here, no Fisher surrogate is used — the quadratic approximation is dominated by the loss itself.

### Teacher choice

Distilling to a teacher rather than one-hot data labels is what removes the domain-adaptation confound: cross-entropy on data makes the student *recover compression* and *learn adaptation* inseparably (which is why the CE-heal frontier needs an explicit control to interpret).

- **Primary — teacher = raw original `M₀`.** The student learns "the original model, compressed." Self-cleaning: nothing to control for, the reference collapses to a single model (cost = held-out `KL`/agreement to `M₀`), the LoRA is pure compression-correction, and `KL → 0` means indistinguishable from the original.
- **Secondary — teacher = a task-adapted `M₀ + LoRA`.** The student learns "the task-adapted model, compressed," isolating compression cost *on top of* an adaptation; warm-start the student's LoRA from the control's so the only teacher–student difference at init is the MLA compression.

### How this connects to the results

- The committed **CE-heal frontier** (1.33× at +0.5% … 4× at +16% vs a same-budget control) is the *task-adaptation* variant, measured with an explicit control. The **distill-to-`M₀`** numbers (KL 0.64 → 0.40, top-1 70% → 76%; PPL rises to ~19 near `M₀`'s 21.6, by design) are **not comparable** to it — they measure behavior preservation, a different and cleaner quantity.
- The **adaptive-rank negative result** is the same missing ingredient in a different place: allocating rank by input-weighted reconstruction error drops the output-sensitivity half of the Fisher, so it starves early layers whose errors propagate most (raw-σ² ≈ +98% vs control; CARE's normalized score recovers most of it to +19.7% but still trails uniform's +16%). This corroborates CARE's own PPL-vs-uniform finding and points at the same fix — output-gradient weighting.
- The **two-sided Fisher init** wins the A/B on every metric: pre-heal PPL 201 → 126, and post-distill KL 0.397 → 0.379, top-1 76.42 → 76.52%, PPL 19.16 → 19.06. It is the one component here with no published precedent (CARE stops at `√C`).
- The **acceptance-objective A/B** confirms KL is a surrogate, not the deployment target. Healing the 4× model by minimizing top-K TVD (= maximizing single-token acceptance `Σ min`) *directly* beats KL-distillation on acceptance (**80.3% vs 73.8%**), and — surprisingly — on top-1 (79.2 vs 76.4) and even eval-KL (0.297 vs 0.397): the sampling-aligned objective finds a strictly better point. The exact `min` (hard 0/1 gradient) beat soft-min smoothing (τ=0.05 → 78.2%), which over-flattens the many tiny-probability tokens; a τ=0.01–0.02 sweep is the obvious follow-up but direct already wins.

- The **TVD-tuned draft** shows the same objective transfers to speculative decoding, end to end. A rank-16 LoRA on the Qwen3-0.6B draft, trained to minimize TVD to the Qwen3-4B target on the workload, raises the draft's single-token acceptance against the target from **58.9% → 67.8%** (top-1 60.8% → 68.6%) — a larger gain than the compression-healing case (+8.9 vs +6.5 points). Merging the LoRA back into the draft (zero inference overhead, identical speed to the base draft), the acceptance gain cashes out as **measured throughput**: at the optimal draft depth K=3, spec decode goes **52.1 → 56.2 tok/s** (1.37× → **1.48×** over greedy; end-to-end greedy-verify acceptance 70.3% → 77.8%), a 1.08× draft-over-draft speedup, with the tuned draft degrading more slowly as K grows. So *aligning* a small draft to the workload is a working alternative to *shrinking* it — the two levers of the planned joint sweep now measured under one objective. The speedup is capped near 1.5× by the draft's relative per-forward cost `c ≈ 0.3` (fixed per-round overhead inflates it above the 0.15 param-ratio), which pins optimal K at 3; a larger (e.g. quantized) target or lower per-round overhead (CUDA graphs) would drop `c`, raise optimal K, and lift the ceiling. Follow-up: a narrow-workload run (higher acceptance → higher optimal K), and align-vs-shrink at matched draft budget.

**Remaining:** Fisher-weighted *rank allocation* (revisit adaptive rank with the output-sensitivity signal); the end-to-end speculative-decode sweep (throughput, not just acceptance); and the secondary control-teacher distill variant.

---

## Hardware

- **Target:** a 16 GB Blackwell-class consumer GPU (RTX 5060 Ti). Qwen3-4B in BF16 ≈ 8 GB; the KV cache at long context dominates the rest. The 16 GB ceiling is the forcing function for the whole study — "compress to fit" is a real constraint, not an academic one.
- Engine and primitive development on a 16 GB MacBook M2 (CPU/MPS, no Triton/CUDA); kernel, calibration, and sweep runs on the CUDA desktop.

## Repo layout

```
src/
  engine/    model.py, attention.py, mla.py (MLAttention runtime + compressed cache),
             kv_cache.py, sampler.py, spec_decode.py, weights.py
  kernels/   fused_ffn.py, fused_qkv.py, swap.py, _autotune.py
  mla/       calibrate.py (covariance + output-Fisher), svd.py (activation-aware SVD),
             convert.py (joint SVD + partial-RoPE CLI), swap.py, heal.py (healing finetune)
  bench/     demo.py, e2e.py (sweep driver), profile.py, eval_ppl.py, microbench_*.py
experiments/ calibration artifacts, conversion runs, eval summaries, frontier/KL plots
tests/       120 tests; CPU subset runs in CI
```

## Open questions

- **Calibration sensitivity** — how much does calibration-set composition (base vs instruct, 100 vs 10k samples) move the no-FT perplexity?
- **Optimal partial-RoPE split** — what `(d_nope, d_rope)` minimizes perplexity at fixed head-dim budget?
- **Compression × speculative-decode interaction** — does a smaller KV help acceptance via cache locality, hurt it via K/V mismatch with the draft, or both by regime? (The planned sweep.)
- **Adaptive rank done right** — can an end-to-end sensitivity signal (gradient/Hessian/ablation) finally beat uniform where local spectra cannot?

## Future extensions

Deferred, each a natural follow-on: probabilistic speculative verify (rejection sampling, for sampled generation); INT4/FP8 weight quantization stacked on MLA; paged KV cache for multi-batch serving; extension of the conversion pipeline to hybrid linear/full-attention models. An SM-partitioning study (via CUDA Green Contexts) on the same engine substrate is archived in [`DESIGN-sm-partitioning.md`](./DESIGN-sm-partitioning.md).

## References

- DeepSeek-AI. "DeepSeek-V2" (2024); "DeepSeek-V3 Technical Report" (2024).
- Meng et al. "TransMLA: Multi-Head Latent Attention Is All You Need" (2025).
- Liu et al. "MHA2MLA: Post-Hoc Conversion of MHA to MLA" (2025).
- Yuan et al. "ASVD: Activation-aware Singular Value Decomposition" (2023).
- Wang et al. "SVD-LLM: Truncation-aware SVD for LLM Compression" (ICLR 2024).
- Hsu et al. "Language Model Compression with Weighted Low-Rank Factorization" (FWSVD, 2022); Chekalina et al. "GFWSVD" (2025).
- Leviathan, Kalman, Matias. "Fast Inference from Transformers via Speculative Decoding" (ICML 2023).

## License

Apache License 2.0 — see [`LICENSE`](./LICENSE).
