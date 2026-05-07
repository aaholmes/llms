# Stage B perplexity sweep — v-only

Variant: `v-only` (V-only: d_rope=head_dim so K stays uncompressed and full-RoPE)
Model: `Qwen/Qwen3-4B` (36 layers, hidden=2560, num_kv_heads=8, head_dim=128)
Calibration: `experiments/stage_b/calib/wt103_1k.pt` (256000 tokens, train split, seed=42)
Eval slice: 250 chunks × 1024 tokens from WikiText-103 `validation` split (disjoint from calibration)
Dtype: bfloat16; device: cuda

| Config | rank | d_rope | KV B/tok | Compression | PPL | Δ PPL % | Avg discarded energy |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | — | — | 4096 | 1.00× | 21.6425 | +0.00% | 0.00e+00 |
| r1024_drope128 | 1024 | 128 | 4096 | 1.00× | 21.6406 | -0.01% | 0.00e+00 |
| r768_drope128 | 768 | 128 | 3584 | 1.14× | 25.6824 | +18.67% | 4.09e+01 |
| r512_drope128 | 512 | 128 | 3072 | 1.33× | 27.1185 | +25.30% | 1.61e+02 |
| r384_drope128 | 384 | 128 | 2816 | 1.45× | 29.8705 | +38.02% | 2.72e+02 |
| r256_drope128 | 256 | 128 | 2560 | 1.60× | 35.4481 | +63.79% | 4.45e+02 |
| r128_drope128 | 128 | 128 | 2304 | 1.78× | 89.8598 | +315.20% | 7.93e+02 |

**Hard gate met.** Highest-compression config under PPL Δ ≤ 2 % is `r1024_drope128`: **1.00× compression** at **+-0.01 % PPL** (21.6406 vs baseline 21.6425).

**KV bytes/token** is per-layer; full-context cache size scales linearly with context length and number of layers.
**Compression ratio** is baseline KV-bytes-per-token / MLA KV-bytes-per-token.
**Δ PPL %** is `(ppl_mla - ppl_baseline) / ppl_baseline * 100`.
**Discarded energy** is the analytic C-weighted reconstruction error averaged across layers (rises with depth, as deeper layers carry more diverse activations).
