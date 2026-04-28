# Stage A.5 ship-gate evaluation

- GPU: **NVIDIA GeForce RTX 5060 Ti** (cap 12.0, 15.5 GB)
- target: `Qwen/Qwen3-4B`, draft: `Qwen/Qwen3-0.6B`, dtype=bfloat16
- prompts: 100 (sampled from Dolly-15k, seed=42); max_new=200

## Ship gate: **FAIL** (ratio 1.114, target 1.50)

- numerator: `ours_spec_triton_K4.median_tps`
- denominator: `ours_greedy_triton.median_tps`

## Per-config tok/s

| config | median | p10 | p90 | extra |
|---|---:|---:|---:|---|
| `hf_greedy` | 39.0 | 39.0 | 39.1 | — |
| `ours_greedy_eager` | 39.0 | 39.0 | 39.1 | — |
| `ours_spec_eager_K4` | 49.8 | 38.1 | 61.9 | acc=58.3%, rounds_med=60.0 |
| `ours_greedy_triton` | 39.3 | 33.4 | 39.4 | — |
| `ours_spec_triton_K3` | 44.2 | 36.2 | 56.2 | acc=60.8%, rounds_med=71.0 |
| `ours_spec_triton_K4` | 43.7 | 35.3 | 57.4 | acc=54.5%, rounds_med=63.5 |
| `ours_spec_triton_K5` | 42.2 | 31.4 | 54.5 | acc=50.7%, rounds_med=57.0 |
| `ours_spec_triton_K7` | 36.6 | 26.2 | 55.5 | acc=41.4%, rounds_med=52.0 |

## Sanity

Our eager greedy runs at **39.0 tok/s** vs HuggingFace's **39.0 tok/s** on the same Qwen3-4B BF16 weights (1.00× HF). The kernel-free path is on par with the reference implementation.

## Spec speedup over greedy, by branch

| spec config | K | spec tok/s | greedy tok/s | speedup |
|---|---:|---:|---:|---:|
| `ours_spec_eager_K4` | 4 | 49.8 | 39.0 | **1.28×** |
| `ours_spec_triton_K3` | 3 | 44.2 | 39.3 | **1.13×** |
| `ours_spec_triton_K4` | 4 | 43.7 | 39.3 | **1.11×** |
| `ours_spec_triton_K5` | 5 | 42.2 | 39.3 | **1.08×** |
| `ours_spec_triton_K7` | 7 | 36.6 | 39.3 | **0.93×** |

## Triton kernel net effect on spec decode

At K=4: eager spec = **49.8 tok/s** (acc 58.3%), triton spec = **43.7 tok/s** (acc 54.5%). Triton hurt spec by **0.88×**.

## Best K on Triton spec branch

K=3: **44.2 tok/s** (acc 60.8%, 1.13× over greedy_triton).
