"""Plot behavioral fidelity to the original at the 4x point, in ACCEPTANCE
(1 - TVD) — the speculative-decode deployment metric — across training objectives:
CE-heal, distill-to-original (KL), fisher-distill, and direct-acceptance. KL is
kept as a secondary annotation. The uncompressed CE control is the reference line.

Acceptance is the fraction of the healed model's tokens the original would keep;
it is what a verifier acts on, and training on it directly beats training on the
KL surrogate on acceptance, KL, and top-1 alike.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SB = Path("experiments/stage_b")
data = json.loads((SB / "eval_summary_frontier_kl.json").read_text())
pts = {p["label"]: p for p in data["points"]}

# Bars in training-objective progression, worst→best acceptance (top→bottom after invert).
rows = [
    ("CE-heal (data cross-entropy)", "r256_drope32 (uniform)", "tab:blue"),
    ("distill to original (KL)", "r256_drope32 (distill-to-orig)", "tab:green"),
    ("fisher-distill (KL + Fisher init)", "r256_drope32 (fisher-distill)", "tab:olive"),
    ("direct TVD (1 − TVD objective)", "r256_drope32 (accept-direct)", "tab:red"),
]
labels = [r[0] for r in rows]
accept = [pts[r[1]]["accept_pct"] for r in rows]
kl = [pts[r[1]]["kl_nats"] for r in rows]
top1 = [pts[r[1]]["top1_agree_pct"] for r in rows]
colors = [r[2] for r in rows]
ctrl = pts["FT control (uncompressed)"]["accept_pct"]

fig, ax = plt.subplots(figsize=(9.5, 4.4))
y = range(len(rows))
ax.barh(list(y), accept, color=colors, zorder=3)
ax.axvline(ctrl, ls="--", color="0.5", lw=1)
ax.text(ctrl, -0.45, f"uncompressed FT control {ctrl:.1f}%", color="0.4",
        fontsize=9, va="center", ha="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", lw=0.5))
for i, (a, k, t) in enumerate(zip(accept, kl, top1)):
    ax.text(a + 0.4, i, f"  {a:.1f}%   (KL {k:.2f}, {t:.1f}% top-1)", va="center", fontsize=10)

ax.set_yticks(list(y))
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel("Fidelity to original — 1 − TVD, % mass agreed   "
              "(= accept-rate if drafted; higher = closer)", fontsize=11)
ax.set_title("Behavioral fidelity by training objective @ 4× KV compression — Qwen3-4B",
             fontsize=12)
ax.set_xlim(0, max(accept) * 1.42)
ax.grid(True, axis="x", alpha=0.3)
fig.tight_layout()
out = SB / "kl_plot.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
