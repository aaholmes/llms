"""Plot end-to-end spec-decode speedup vs draft depth K, base vs TVD-tuned draft.

Two workloads (broad WikiText, narrow Python code) show the same lever: a TVD-tuned
draft LoRA raises single-token acceptance, which lifts the whole speedup-vs-K curve
and pushes the optimal K rightward. The narrow workload starts from higher acceptance,
so the gain is larger and the peak moves further right.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SB = Path("experiments/stage_b")


def series(path):
    d = json.loads((SB / path).read_text())
    drafts = d["drafts"]
    base = {int(k): v["speedup"] for k, v in drafts["base"].items()}
    tuned_key = next(k for k in drafts if k != "base")
    tuned = {int(k): v["speedup"] for k, v in drafts[tuned_key].items()}
    return base, tuned


wiki_b, wiki_t = series("bench_draft_speedup.json")
code_b, code_t = series("bench_draft_speedup_code.json")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

panels = [
    (axes[0], "Broad workload (WikiText-103)", wiki_b, wiki_t, "tab:blue"),
    (axes[1], "Narrow workload (Python code)", code_b, code_t, "tab:green"),
]

for ax, title, b, t, color in panels:
    ks = sorted(b)
    ax.plot(ks, [b[k] for k in ks], "--o", color="0.5", lw=2, ms=7, label="base 0.6B draft")
    ax.plot(ks, [t[k] for k in ks], "-o", color=color, lw=2.5, ms=8, label="TVD-tuned draft")
    # mark each curve's peak
    for d, c in [(b, "0.5"), (t, color)]:
        kbest = max(d, key=d.get)
        ax.plot(kbest, d[kbest], "*", color=c, ms=20, zorder=5)
        ax.annotate(f"{d[kbest]:.2f}×", (kbest, d[kbest]), textcoords="offset points",
                    xytext=(6, 8), fontsize=11, fontweight="bold", color=c)
    ax.axhline(1.0, ls=":", color="0.7", lw=1)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("draft depth K", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", fontsize=10)

axes[0].set_ylabel("end-to-end speedup vs greedy target", fontsize=11)
fig.suptitle("Draft-tuning as a TVD-LoRA corrector: acceptance ↑ lifts the speedup curve — "
             "Qwen3-0.6B draft → Qwen3-4B target, 16 GB", fontsize=12.5)

out = SB / "draft_speedup_plot.png"
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out, dpi=150)
print(f"wrote {out}")
