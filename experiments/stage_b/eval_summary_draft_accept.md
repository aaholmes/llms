# Draft acceptance vs target (Qwen3-0.6B → Qwen3-4B): base vs TVD-tuned

Expected single-token speculative-decode acceptance = `Σ min(p_target, p_draft)` = `1 − TVD`, per token over 64×1024-tok WT103-val. Higher = more drafted tokens accepted = faster spec decode.

| Draft | Accept (1−TVD) ↑ | Top-1 agree ↑ |
|---|---:|---:|
| Qwen3-0.6B (base draft) | 58.9% | 60.8% |
| Qwen3-0.6B (TVD-tuned) | 67.8% | 68.6% |
