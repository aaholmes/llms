# Post-hoc MHA→MLA Conversion on a Single GPU

Running a large language model token-by-token is bottlenecked by KV-cache memory and HBM bandwidth: every new token has to read the full set of cached keys and values for every previous position. **Multi-head Latent Attention (MLA)**, introduced in DeepSeek-V2 and used by every 2026 frontier model (DeepSeek V3/V4, Kimi K2.6, Nemotron 3), compresses that cache by 4–8× by absorbing the K and V projections into a low-rank latent. But the existing population of MHA/GQA models — Llama-3, Qwen3, Mistral — was trained without MLA and is stuck with full-size caches.

This project converts an existing MHA-architecture model (Qwen3-4B) to MLA *after the fact*, without retraining, using activation-aware SVD with a partial-RoPE split. The conversion is validated on a from-scratch single-GPU inference engine, and the headline study is how the resulting KV compression interacts with speculative decoding at a fixed 16 GB VRAM budget.

The catch the partial-RoPE split solves: position-dependent rotation (RoPE) breaks MLA's "absorb" trick that fuses Q-projection with attention. The standard workaround is to split each attention head into a NoPE latent-absorbed subspace and a small RoPE-only concatenated subspace — the design DeepSeek-V3 uses natively, applied here as a conversion target.

## Status

- **Stage A — engine** — done. From-scratch sequential speculative-decoding inference engine for Qwen3-4B + Qwen3-0.6B. Manual forward pass, bit-exact with HuggingFace on Qwen3-0.6B; 34 passing tests.
- **Stage A.5 — Triton kernel** — pending GPU-machine work.
- **Stage B — MLA conversion** — pending. Calibration pipeline + activation-aware SVD + partial-RoPE split.
- **Stage C — interaction characterization** — pending. Joint sweep over compression × spec-decode K at fixed VRAM.

See [`DESIGN.md`](./DESIGN.md) for the full plan: hypothesis, staged approach, hardware, references, and the future-extensions list (which still includes the original SM-partitioning research question).

## Approach at a glance

- **Stage A** — manual forward pass for Qwen3 (GQA + RoPE + RMSNorm + SwiGLU + Qwen3 QK-norm); contiguous KV cache; greedy speculative decoding with Qwen3-0.6B as the draft. PyTorch + Triton; no `transformers.AutoModelForCausalLM` in the hot path.
- **Stage A.5** — profile on a CUDA GPU; one Triton kernel for whichever op profiling identifies as the decode bottleneck.
- **Stage B** — calibrate on ~1k instruction samples; activation-aware SVD on K/V projections; partial-RoPE split following DeepSeek-V3; eval on perplexity and downstream tasks.
- **Stage C** — sweep KV compression × draft-K (× INT4 quantization, optional) at fixed 16 GB VRAM. Pareto frontier of throughput vs perplexity.

Target hardware: RTX 5060 Ti 16 GB (Blackwell). Development on MacBook M2 (CPU/MPS for engine work; CUDA desktop for Triton, profiling, and the sweeps).

## License

TBD before public release.
