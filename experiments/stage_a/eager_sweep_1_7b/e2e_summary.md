# Stage A.5 ship-gate evaluation

- GPU: **NVIDIA GeForce RTX 5060 Ti** (cap 12.0, 15.5 GB)
- target: `Qwen/Qwen3-4B`, draft: `Qwen/Qwen3-1.7B`, dtype=bfloat16
- prompts: 100 (sampled from Dolly-15k, seed=42); max_new=200

## Ship gate: not computed (need both phase B and phase C)

## Per-config tok/s

| config | median | p10 | p90 | extra |
|---|---:|---:|---:|---|
| `ours_greedy_eager` | 38.9 | 38.8 | 39.0 | — |
| `ours_spec_eager_K3` | 46.3 | 41.3 | 54.3 | acc=70.3%, rounds_med=65.0 |
| `ours_spec_eager_K4` | 44.7 | 38.0 | 54.7 | acc=63.7%, rounds_med=56.5 |
| `ours_spec_eager_K5` | 42.1 | 35.2 | 54.3 | acc=57.5%, rounds_med=52.0 |
| `ours_spec_eager_K7` | 39.0 | 28.7 | 51.7 | acc=50.5%, rounds_med=44.0 |

## Spec speedup over greedy, by branch

| spec config | K | spec tok/s | greedy tok/s | speedup |
|---|---:|---:|---:|---:|
| `ours_spec_eager_K3` | 3 | 46.3 | 38.9 | **1.19×** |
| `ours_spec_eager_K4` | 4 | 44.7 | 38.9 | **1.15×** |
| `ours_spec_eager_K5` | 5 | 42.1 | 38.9 | **1.08×** |
| `ours_spec_eager_K7` | 7 | 39.0 | 38.9 | **1.00×** |
