# Stage A.5 ship-gate evaluation

- GPU: **NVIDIA GeForce RTX 5060 Ti** (cap 12.0, 15.5 GB)
- target: `Qwen/Qwen3-4B`, draft: `Qwen/Qwen3-1.7B`, dtype=bfloat16
- prompts: 100 (sampled from Dolly-15k, seed=42); max_new=200

## Ship gate: **FAIL** (ratio 1.148, target 1.50)

- numerator: `ours_spec_triton_K4.median_tps`
- denominator: `ours_greedy_triton.median_tps`

## Per-config tok/s

| config | median | p10 | p90 | extra |
|---|---:|---:|---:|---|
| `ours_greedy_eager` | 38.9 | 38.8 | 39.0 | — |
| `ours_spec_eager_K4` | 44.7 | 38.0 | 54.6 | acc=63.7%, rounds_med=56.5 |
| `ours_greedy_triton` | 39.1 | 33.3 | 39.2 | — |
| `ours_spec_triton_K3` | 44.0 | 37.6 | 55.1 | acc=70.6%, rounds_med=64.0 |
| `ours_spec_triton_K4` | 44.9 | 36.3 | 56.6 | acc=63.4%, rounds_med=57.0 |
| `ours_spec_triton_K5` | 44.0 | 34.0 | 54.8 | acc=59.6%, rounds_med=50.0 |
| `ours_spec_triton_K7` | 38.4 | 29.2 | 53.7 | acc=49.5%, rounds_med=45.0 |

## Spec speedup over greedy, by branch

| spec config | K | spec tok/s | greedy tok/s | speedup |
|---|---:|---:|---:|---:|
| `ours_spec_eager_K4` | 4 | 44.7 | 38.9 | **1.15×** |
| `ours_spec_triton_K3` | 3 | 44.0 | 39.1 | **1.12×** |
| `ours_spec_triton_K4` | 4 | 44.9 | 39.1 | **1.15×** |
| `ours_spec_triton_K5` | 5 | 44.0 | 39.1 | **1.12×** |
| `ours_spec_triton_K7` | 7 | 38.4 | 39.1 | **0.98×** |

## Triton kernel net effect on spec decode

At K=4: eager spec = **44.7 tok/s** (acc 63.7%), triton spec = **44.9 tok/s** (acc 63.4%). Triton helped spec by **1.00×**.

## Best K on Triton spec branch

K=4: **44.9 tok/s** (acc 63.4%, 1.15× over greedy_triton).
