# Behavioral fidelity to the original

Per-token agreement of each healed model with the frozen original, over a 64-chunk × 1024-tok WT103-val subset. **Accept** = `Σ min(p_orig, p_healed)` = 1−TVD = the sampling-aligned metric (expected single-token speculative-decode acceptance rate). **Top-1** = greedy-decode agreement. **KL** is the training surrogate, reported for completeness. Accept=100% / top1=100% / KL=0 ⇒ identical to the original.

| Config | Compression | Accept (1−TVD) ↑ | Top-1 agree ↑ | KL (nats) ↓ |
|---|---:|---:|---:|---:|
| FT control (uncompressed) | 1.00× | 70.4% | 74.5% | 0.522 |
| r512_drope128 (V-only) | 1.33× | 69.5% | 73.2% | 0.557 |
| r256_drope32 (uniform) | 4.00× | 66.0% | 70.0% | 0.639 |
| r256_drope32 (adaptive raw) | 4.00× | 53.5% | 58.5% | 1.116 |
| r256_drope32 (adaptive mr128) | 4.00× | 63.3% | 67.6% | 0.723 |
| r256_drope32 (adaptive norm) | 4.00× | 65.1% | 69.1% | 0.665 |
| r192_drope32 | 4.57× | 64.3% | 68.6% | 0.692 |
| r128_drope32 | 5.33× | 61.2% | 65.7% | 0.799 |
| r256_drope32 (distill-to-orig) | 4.00× | 73.8% | 76.4% | 0.397 |
| r256_drope32 (fisher-distill) | 4.00× | 74.6% | 76.5% | 0.379 |
