# Stage A.5 ship-gate evaluation

- GPU: **NVIDIA GeForce RTX 5060 Ti** (cap 12.0, 15.5 GB)
- target: `Qwen/Qwen3-4B`, draft: `Qwen/Qwen3-1.7B`, dtype=bfloat16
- prompts: 100 (sampled from Dolly-15k, seed=42); max_new=200

## Ship gate: not computed (need both phase B and phase C)

## Per-config tok/s

| config | median | p10 | p90 | extra |
|---|---:|---:|---:|---|
| `ours_greedy_eager` | 38.6 | 38.5 | 38.6 | — |
| `ours_spec_eager_K1` | 43.7 | 41.9 | 46.4 | acc=82.6%, rounds_med=109.0 |
| `ours_spec_eager_K2` | 46.2 | 42.3 | 51.0 | acc=75.8%, rounds_med=79.5 |

## Spec speedup over greedy, by branch

| spec config | K | spec tok/s | greedy tok/s | speedup |
|---|---:|---:|---:|---:|
| `ours_spec_eager_K1` | 1 | 43.7 | 38.6 | **1.13×** |
| `ours_spec_eager_K2` | 2 | 46.2 | 38.6 | **1.20×** |
