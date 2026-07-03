"""Plot behavioral fidelity to the original (KL + top-1) for the 4x point:
CE-heal vs distill-to-original vs fisher-distill, with the uncompressed CE
control as a reference line."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SB = Path("experiments/stage_b")
data = json.loads((SB / "eval_summary_frontier_kl.json").read_text())
pts = {p["label"]: p for p in data["points"]}

# Bars, worst→best (top→bottom after invert).
rows = [
    ("CE-heal (uniform)", "r256_drope32 (uniform)", "tab:blue"),
    ("distill-to-original", "r256_drope32 (distill-to-orig)", "tab:green"),
    ("fisher-distill", "r256_drope32 (fisher-distill)", "tab:olive"),
]
labels = [r[0] for r in rows]
kl = [pts[r[1]]["kl_nats"] for r in rows]
top1 = [pts[r[1]]["top1_agree_pct"] for r in rows]
colors = [r[2] for r in rows]
ctrl_kl = pts["FT control (uncompressed)"]["kl_nats"]

fig, ax = plt.subplots(figsize=(9, 4.2))
y = range(len(rows))
ax.barh(list(y), kl, color=colors, zorder=3)
ax.axvline(ctrl_kl, ls="--", color="0.5", lw=1)
ax.text(ctrl_kl, len(rows) - 0.4, f"  uncompressed CE control {ctrl_kl:.2f}",
        color="0.4", fontsize=9, va="top")
for i, (k, t) in enumerate(zip(kl, top1)):
    ax.text(k + 0.01, i, f"  {k:.3f} nats   ({t:.1f}% top-1)", va="center", fontsize=10)

ax.set_yticks(list(y))
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel("KL(original ‖ healed) — nats per token   (lower = closer to original)", fontsize=11)
ax.set_title("Behavioral fidelity to the original @ 4× KV compression — Qwen3-4B", fontsize=12)
ax.set_xlim(0, max(kl) * 1.5)
ax.grid(True, axis="x", alpha=0.3)
fig.tight_layout()
out = SB / "kl_plot.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
