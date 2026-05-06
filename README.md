# Post-hoc MHA→MLA Conversion on a Single GPU

Running a large language model token-by-token is bottlenecked by KV-cache memory and HBM bandwidth: every new token has to read the full set of cached keys and values for every previous position. **Multi-head Latent Attention (MLA)**, introduced in DeepSeek-V2 and used by every 2026 frontier model (DeepSeek V3/V4, Kimi K2.6, Nemotron 3), compresses that cache by 4–8× by absorbing the K and V projections into a low-rank latent. But the existing population of MHA/GQA models — Llama-3, Qwen3, Mistral — was trained without MLA and is stuck with full-size caches.

This project converts an existing MHA-architecture model (Qwen3-4B) to MLA *after the fact*, without retraining, using activation-aware SVD with a partial-RoPE split. The conversion is validated on a from-scratch single-GPU inference engine, and the headline study is how the resulting KV compression interacts with speculative decoding at a fixed 16 GB VRAM budget.

The catch the partial-RoPE split solves: position-dependent rotation (RoPE) breaks MLA's "absorb" trick that fuses Q-projection with attention. The standard workaround is to split each attention head into a NoPE latent-absorbed subspace and a small RoPE-only concatenated subspace — the design DeepSeek-V3 uses natively, applied here as a conversion target.

## Status

- **Stage A — engine** — done. From-scratch sequential speculative-decoding inference engine for Qwen3-4B + Qwen3-0.6B. Manual forward pass, bit-exact with HuggingFace on Qwen3-0.6B; **1.29× spec/greedy speedup** at K=2–3 on the formal e2e bench (see below).
- **Stage A.5 — Triton kernel** — done. Two fused Triton kernels (gate-up-silu, QKV projection) shipped with microbench wins of 1.05× and 1.50–2.98×; both clear isolation gates but do not improve the spec-decode hot path end-to-end (per-call dispatch overhead × ~14k spec calls erodes the per-call gain, and BF16 reduction-order shifts drop draft/target acceptance from 64.5% to 54.5%). The 1.5× e2e ship gate was not reached.
- **Stage B — MLA conversion** — pending. Calibration pipeline + activation-aware SVD + partial-RoPE split.
- **Stage C — interaction characterization** — pending. Joint sweep over compression × spec-decode K at fixed VRAM.

See [`DESIGN.md`](./DESIGN.md) for the full plan: hypothesis, staged approach, hardware, references, and the future-extensions list (which still includes the original SM-partitioning research question).

## Engine speedups (Stage A end-to-end)

100 Dolly-15k prompts × 200 generated tokens, Qwen3-4B target in BF16 on RTX 5060 Ti 16 GB. Greedy baseline: **38.9 tok/s** (matches HuggingFace's `AutoModelForCausalLM` greedy at `use_cache=True` to one decimal place).

| draft | K=1 | K=2 | K=3 | K=4 | K=5 | K=7 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 46.6 — 1.21× (acc 78.9%) | 49.7 — 1.29× (69.8%) | **50.3 — 1.29×** (64.5%) | 48.7 — 1.25× (58.3%) | 47.5 — 1.22× (53.2%) | 41.4 — 1.06× (42.7%) |
| Qwen3-1.7B | 43.7 — 1.13× (acc 82.6%) | 46.2 — 1.20× (75.8%) | **46.3 — 1.19×** (70.3%) | 44.7 — 1.15× (63.7%) | 42.1 — 1.08× (57.5%) | 39.0 — 1.00× (50.5%) |

**Optimal K = 2–3 for both drafts** (K=2 and K=3 are tied within run-to-run noise). The curve is sharply peaked: K=1 leaves performance on the table because per-round overhead dominates one drafted token, and K≥4 wastes draft work that the target rejects. Acceptance climbs monotonically as K drops (peaking at 78.9% / 82.6% at K=1), so the bottleneck above K=3 is acceptance, not draft cost; below K=2 it flips and the bottleneck is round overhead. The 0.6B draft beats the 1.7B at every K despite a lower acceptance rate — the 1.7B's larger draft-step compute cost outruns its acceptance-rate gain. K=7 is the breakeven point for the 1.7B draft; beyond that, drafting more tokens than the target will accept loses to a plain greedy run.

## Approach at a glance

- **Stage A** — manual forward pass for Qwen3 (GQA + RoPE + RMSNorm + SwiGLU + Qwen3 QK-norm); contiguous KV cache; greedy speculative decoding with Qwen3-0.6B as the draft. PyTorch + Triton; no `transformers.AutoModelForCausalLM` in the hot path.
- **Stage A.5** — profile on a CUDA GPU; one Triton kernel for whichever op profiling identifies as the decode bottleneck.
- **Stage B** — calibrate on ~1k instruction samples; activation-aware SVD on K/V projections; partial-RoPE split following DeepSeek-V3; eval on perplexity and downstream tasks.
- **Stage C** — sweep target-compression × draft-compression × draft-K at fixed 16 GB VRAM. Three named regimes — both uncompressed, target only, both compressed at matched ratios — directly test whether coupling the SVD distortion across target and draft preserves spec-decode acceptance rate relative to compressing the target alone. Pareto frontier of throughput vs perplexity.

Target hardware: RTX 5060 Ti 16 GB (Blackwell). Development on MacBook M2 (CPU/MPS for engine work; CUDA desktop for Triton, profiling, and the sweeps).

## License

TBD before public release.
