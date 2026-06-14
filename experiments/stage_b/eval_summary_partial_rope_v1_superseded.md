# Stage B perplexity sweep

Model: `Qwen/Qwen3-4B` (36 layers, hidden=2560, num_kv_heads=8, head_dim=128)
Calibration: `experiments/stage_b/calib/wt103_1k.pt` (256000 tokens, train split, seed=42)
Eval slice: 250 chunks × 1024 tokens from WikiText-103 `validation` split (disjoint from calibration)
Dtype: bfloat16; device: cuda

| Config | rank | d_rope | KV B/tok | Compression | PPL | Δ PPL % | Avg discarded energy |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | — | — | 4096 | 1.00× | 21.6425 | +0.00% | 0.00e+00 |
| r256_drope32 | 256 | 32 | 1024 | 4.00× | 1986.7064 | +9079.64% | 7.57e+02 |
| r192_drope32 | 192 | 32 | 896 | 4.57× | 3586.7280 | +16472.58% | 9.53e+02 |
| r128_drope32 | 128 | 32 | 768 | 5.33× | 2970.2017 | +13623.91% | 1.31e+03 |
| r128_drope64 | 128 | 64 | 1280 | 3.20× | 1191.4692 | +5405.22% | 1.17e+03 |
| r96_drope32 | 96 | 32 | 704 | 5.82× | 3699.5703 | +16993.97% | 1.61e+03 |
| r64_drope32 | 64 | 32 | 640 | 6.40× | 5572.5098 | +25647.95% | 2.01e+03 |

**Hard gate missed at this calibration.** Lowest-Δ config is `r128_drope64`: +5405.22 % PPL at 3.20×. Contingency: scale calibration to 4k×256, switch to per-layer adaptive rank, or promote LoRA healing-FT.

**KV bytes/token** is per-layer; full-context cache size scales linearly with context length and number of layers.
**Compression ratio** is baseline KV-bytes-per-token / MLA KV-bytes-per-token.
**Δ PPL %** is `(ppl_mla - ppl_baseline) / ppl_baseline * 100`.
**Discarded energy** is the analytic C-weighted reconstruction error averaged across layers (rises with depth, as deeper layers carry more diverse activations).
