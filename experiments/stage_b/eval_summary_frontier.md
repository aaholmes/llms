# Post-heal compression–PPL frontier (250-chunk WT103-val slice)

baseline=21.64  |  same-budget FT control=11.13 (uncompressed)

| Config | Compression | PPL | Δ% vs baseline | Δ% vs FT control |
|---|---:|---:|---:|---:|
| r256_drope32 (uniform) | 4.00× | 12.91 | -40.3% | +16.0% |
| r256_drope32 (adaptive) | 4.00× | 22.01 | +1.7% | +97.8% |
| r256_drope32 (adaptive mr128) | 4.00× | 14.29 | -34.0% | +28.3% |
| r192_drope32 | 4.57× | 13.81 | -36.2% | +24.0% |
| r128_drope32 | 5.33× | 15.65 | -27.7% | +40.6% |
| r512_drope128 (V-only) | 1.33× | 11.19 | -48.3% | +0.5% |
