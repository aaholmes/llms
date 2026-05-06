# Stage A.5 ship-gate evaluation

- GPU: **NVIDIA GeForce RTX 5060 Ti** (cap 12.0, 15.5 GB)
- target: `Qwen/Qwen3-4B`, draft: `Qwen/Qwen3-0.6B`, dtype=bfloat16
- prompts: 100 (sampled from Dolly-15k, seed=42); max_new=200

## Ship gate: not computed (need both phase B and phase C)

## Per-config tok/s

| config | median | p10 | p90 | extra |
|---|---:|---:|---:|---|
| `ours_greedy_eager` | 38.6 | 38.5 | 38.7 | — |
| `ours_spec_eager_K1` | 46.6 | 43.0 | 49.5 | acc=78.9%, rounds_med=111.5 |
| `ours_spec_eager_K2` | 49.7 | 43.4 | 56.3 | acc=69.8%, rounds_med=83.5 |

## Spec speedup over greedy, by branch

| spec config | K | spec tok/s | greedy tok/s | speedup |
|---|---:|---:|---:|---:|
| `ours_spec_eager_K1` | 1 | 46.6 | 38.6 | **1.21×** |
| `ours_spec_eager_K2` | 2 | 49.7 | 38.6 | **1.29×** |
