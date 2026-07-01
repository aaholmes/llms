"""Plot post-heal perplexity vs KV compression ratio for the Stage B frontier."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SB = Path("experiments/stage_b")
data = json.loads((SB / "eval_summary_frontier.json").read_text())
base, control = data["baseline_ppl"], data["control_ppl"]
pts = {p["label"]: (p["compression"], p["ppl"]) for p in data["points"]}

# Uniform frontier (the good curve), in compression order.
uniform = [
    ("r512 V-only", "r512_drope128 (V-only)"),
    ("r256 uniform", "r256_drope32 (uniform)"),
    ("r192", "r192_drope32"),
    ("r128", "r128_drope32"),
]
uni_x = [pts[k][0] for _, k in uniform]
uni_y = [pts[k][1] for _, k in uniform]

fig, ax = plt.subplots(figsize=(9, 6))

# Reference lines (labels left-aligned inside the axes to avoid edge clipping).
ax.axhline(base, ls="--", color="0.5", lw=1)
ax.text(1.05, base + 0.12, f"baseline {base:.2f}", va="bottom", color="0.4", fontsize=9)
ax.axhline(control, ls="--", color="seagreen", lw=1)
ax.text(3.6, control + 0.12, f"FT control {control:.2f} (uncompressed)", va="bottom",
        color="seagreen", fontsize=9)

# Uniform frontier.
ax.plot(uni_x, uni_y, "-o", color="tab:blue", lw=2, ms=8, label="uniform rank (frontier)", zorder=3)
# Per-point label offsets (dx, dy in points) to avoid the frontier line / ref lines.
uni_off = [(10, 10), (10, -20), (10, -20), (-6, 10)]
for (short, _), x, y, off in zip(uniform, uni_x, uni_y, uni_off):
    ax.annotate(f"{short}\n{y:.2f} ({x:.2f}×)", (x, y), textcoords="offset points",
                xytext=off, fontsize=9, color="tab:blue")

# Adaptive points at 4x (off the frontier).
adaptive = [
    ("adaptive\nmin_rank=8", "r256_drope32 (adaptive)", "tab:red"),
    ("adaptive\nmin_rank=128", "r256_drope32 (adaptive mr128)", "tab:orange"),
]
for short, key, c in adaptive:
    x, y = pts[key]
    ax.plot(x, y, "X", color=c, ms=13, zorder=4)
    ax.annotate(f"{short}\n{y:.2f}", (x, y), textcoords="offset points",
                xytext=(10, 4), fontsize=9, color=c, fontweight="bold")

ax.set_xlabel("KV-cache compression ratio  (× vs baseline)", fontsize=11)
ax.set_ylabel("Perplexity (250-chunk WikiText-103-val)", fontsize=11)
ax.set_title("Post-heal perplexity vs KV compression — Qwen3-4B MHA→MLA, 16 GB", fontsize=12)
ax.set_xlim(1.0, 6.1)
ax.margins(y=0.08)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", fontsize=10)

out = SB / "frontier_plot.png"
fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"wrote {out}")
