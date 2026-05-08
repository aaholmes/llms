# Stage B perplexity sweep — partial-rope

Variant: `partial-rope` (partial-RoPE compresses both K-nope and V via shared latent)
Model: `Qwen/Qwen3-4B` (36 layers, hidden=2560, num_kv_heads=8, head_dim=128)
Calibration: `experiments/stage_b/calib/wt103_1k.pt` (256000 tokens, train split, seed=42)
Eval slice: 250 chunks × 1024 tokens from WikiText-103 `validation` split (disjoint from calibration)
Dtype: bfloat16; device: cuda

| Config | rank | d_rope | KV B/tok | Compression | PPL | Δ PPL % | Avg discarded energy |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | — | — | 4096 | 1.00× | 21.6425 | +0.00% | 0.00e+00 |
| r256_drope32 | 256 | 32 | 1024 | 4.00× | 200.3313 | +825.64% | 7.64e+02 |
| r192_drope32 | 192 | 32 | 896 | 4.57× | 521.4987 | +2309.60% | 9.63e+02 |
| r128_drope32 | 128 | 32 | 768 | 5.33× | 2253.3008 | +10311.44% | 1.33e+03 |
| r128_drope64 | 128 | 64 | 1280 | 3.20× | 617.4141 | +2752.78% | 1.19e+03 |
| r96_drope32 | 96 | 32 | 704 | 5.82× | 8205.8281 | +37815.27% | 1.64e+03 |
| r64_drope32 | 64 | 32 | 640 | 6.40× | 155706.4219 | +719346.16% | 2.07e+03 |

**Hard gate missed at this calibration.** Lowest-Δ config is `r256_drope32`: +825.64 % PPL at 4.00×. Contingency: scale calibration to 4k×256, switch to per-layer adaptive rank, or promote LoRA healing-FT.

**KV bytes/token** is per-layer; full-context cache size scales linearly with context length and number of layers.
**Compression ratio** is baseline KV-bytes-per-token / MLA KV-bytes-per-token.
**Δ PPL %** is `(ppl_mla - ppl_baseline) / ppl_baseline * 100`.
**Discarded energy** is the analytic C-weighted reconstruction error averaged across layers (rises with depth, as deeper layers carry more diverse activations).
