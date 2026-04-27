# Async Speculative Decoding on a Single GPU

Running a large language model token-by-token is bottlenecked by memory bandwidth, not compute — every output token requires streaming the full set of model weights out of HBM. Speculative decoding amortizes that cost: a small "draft" model proposes several tokens cheaply, and the large "target" model verifies them in a single batched pass. If the draft is right often enough, you get more tokens per HBM read.

The catch on a single GPU is that running draft and target concurrently makes them fight for the same memory pipe, so the concurrency speedup you'd naively expect often disappears. NVIDIA's CUDA Green Contexts let you carve up the GPU's streaming multiprocessors between two contexts, which should — in principle — help by giving each model its own slice of L2 cache and reducing the contention.

This project asks when that actually pays off, and by how much. The output is a phase diagram across model-size pairs, SM split ratios, draft length, and batch size, plus the small inference engine needed to produce it.

## Status

Pre-implementation — only the design exists. See [`DESIGN.md`](./DESIGN.md) for the full plan: hypothesis, staged build, hardware requirements, and what counts as success.

## Plan

Three stages, each gated on the previous one shipping a working artifact:

1. **Sequential baseline.** A from-scratch PyTorch implementation of speculative decode using Llama-3.2-1B as the target and Qwen-2.5-0.5B as the draft, with one custom Triton kernel on the hot path. No `transformers.generate()` — the forward pass is hand-written so kernels stay swappable.
2. **Validation spike.** Run draft and target on two CUDA streams with no partitioning, profile with Nsight Compute. The numbers decide whether stage 3 is worth the time.
3. **Partitioned engine and sweep.** Bring in Green Contexts via `cuda-python`, run the factorial sweep, write it up.

Hardware: RTX 5060 Ti 16GB (consumer Blackwell).

## License

TBD before public release.
