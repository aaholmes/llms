# Behavioral fidelity to the original: KL + top-1 agreement

Forward KL(p_original ‖ p_healed) and top-1 argmax agreement, per token, over a 64-chunk × 1024-tok WT103-val subset. KL=0 / top1=100% would be behaviorally identical to the original.

| Config | Compression | KL to orig (nats) ↓ | Top-1 agree ↑ |
|---|---:|---:|---:|
| FT control (uncompressed) | 1.00× | 0.522 | 74.5% |
| r512_drope128 (V-only) | 1.33× | 0.557 | 73.2% |
| r256_drope32 (uniform) | 4.00× | 0.639 | 70.0% |
| r256_drope32 (adaptive raw) | 4.00× | 1.116 | 58.5% |
| r256_drope32 (adaptive mr128) | 4.00× | 0.723 | 67.6% |
| r256_drope32 (adaptive norm) | 4.00× | 0.665 | 69.1% |
| r192_drope32 | 4.57× | 0.692 | 68.6% |
| r128_drope32 | 5.33× | 0.799 | 65.7% |
