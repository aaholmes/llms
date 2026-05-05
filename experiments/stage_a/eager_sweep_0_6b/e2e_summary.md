# Stage A.5 ship-gate evaluation

- GPU: **NVIDIA GeForce RTX 5060 Ti** (cap 12.0, 15.5 GB)
- target: `Qwen/Qwen3-4B`, draft: `Qwen/Qwen3-0.6B`, dtype=bfloat16
- prompts: 100 (sampled from Dolly-15k, seed=42); max_new=200

## Ship gate: not computed (need both phase B and phase C)

## Per-config tok/s

| config | median | p10 | p90 | extra |
|---|---:|---:|---:|---|
| `ours_greedy_eager` | 38.9 | 38.8 | 38.9 | — |
| `ours_spec_eager_K3` | 50.3 | 40.6 | 59.5 | acc=64.5%, rounds_med=68.0 |
| `ours_spec_eager_K4` | 48.7 | 37.2 | 60.4 | acc=58.3%, rounds_med=60.0 |
| `ours_spec_eager_K5` | 47.5 | 34.9 | 60.4 | acc=53.2%, rounds_med=54.5 |
| `ours_spec_eager_K7` | 41.4 | 29.4 | 57.6 | acc=42.7%, rounds_med=50.5 |

## Spec speedup over greedy, by branch

| spec config | K | spec tok/s | greedy tok/s | speedup |
|---|---:|---:|---:|---:|
| `ours_spec_eager_K3` | 3 | 50.3 | 38.9 | **1.29×** |
| `ours_spec_eager_K4` | 4 | 48.7 | 38.9 | **1.25×** |
| `ours_spec_eager_K5` | 5 | 47.5 | 38.9 | **1.22×** |
| `ours_spec_eager_K7` | 7 | 41.4 | 38.9 | **1.06×** |
