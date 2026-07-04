# End-to-end spec-decode speedup: base vs TVD-tuned draft

Qwen3-0.6B draft → Qwen3-4B target, greedy-verify, 24 WikiText-val prompts × 128→256 tok. Greedy target baseline: **38.0 tok/s**. Tuned draft has its LoRA merged (same speed as base; only weights differ).

| Draft | K | tok/s | speedup vs greedy | acceptance |
|---|---:|---:|---:|---:|
| base | 1 | 46.5 | 1.23× | 82.5% |
| base | 2 | 50.7 | 1.34× | 75.1% |
| base | 3 | 52.1 | 1.37× | 70.3% |
| base | 4 | 50.6 | 1.33× | 62.6% |
| base | 5 | 50.0 | 1.32× | 58.9% |
| base | 6 | 48.5 | 1.28× | 55.5% |
| base | 8 | 42.4 | 1.12× | 46.2% |
| tvd-tuned | 1 | 47.3 | 1.24× | 85.8% |
| tvd-tuned | 2 | 53.1 | 1.40× | 81.3% |
| tvd-tuned | 3 | 56.2 | 1.48× | 77.8% |
| tvd-tuned | 4 | 55.2 | 1.46× | 71.8% |
| tvd-tuned | 5 | 53.6 | 1.41× | 65.5% |
| tvd-tuned | 6 | 52.1 | 1.37× | 61.7% |
| tvd-tuned | 8 | 50.2 | 1.32× | 56.8% |

**Best-K: base 52.1 tok/s (1.37×) → tuned 56.2 tok/s (1.48×)** — a 1.08× throughput gain from TVD-tuning the draft (acceptance 70.3% → 77.8%).
