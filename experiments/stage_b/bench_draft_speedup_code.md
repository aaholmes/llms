# End-to-end spec-decode speedup: base vs TVD-tuned draft

Qwen3-0.6B draft → Qwen3-4B target, greedy-verify, 24 code prompts × 128→256 tok. Greedy target baseline: **38.0 tok/s**. Tuned draft has its LoRA merged (same speed as base; only weights differ).

| Draft | K | tok/s | speedup vs greedy | acceptance |
|---|---:|---:|---:|---:|
| base | 1 | 48.1 | 1.27× | 88.9% |
| base | 2 | 52.8 | 1.39× | 81.1% |
| base | 3 | 55.9 | 1.47× | 77.1% |
| base | 4 | 56.6 | 1.49× | 73.3% |
| base | 5 | 55.5 | 1.46× | 68.4% |
| base | 6 | 54.2 | 1.43× | 64.8% |
| base | 8 | 49.9 | 1.32× | 56.7% |
| code-tuned | 1 | 48.7 | 1.28× | 91.7% |
| code-tuned | 2 | 55.5 | 1.46× | 87.6% |
| code-tuned | 3 | 58.1 | 1.53× | 82.7% |
| code-tuned | 4 | 60.6 | 1.60× | 81.2% |
| code-tuned | 5 | 59.5 | 1.57× | 75.7% |
| code-tuned | 6 | 59.4 | 1.57× | 73.4% |
| code-tuned | 8 | 57.6 | 1.52× | 67.6% |

**Best-K: base 56.6 tok/s (1.49×) → tuned 60.6 tok/s (1.60×)** — a 1.07× throughput gain from TVD-tuning the draft (acceptance 73.3% → 81.2%).
