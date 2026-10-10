"""Summarize the retrieval eval: accuracy per cell and paired difference from uncompressed.

Reads ``experiments/stage_b/niah/<arm>.jsonl`` (written by ``niah_eval.py``) and
writes ``niah_summary.md`` / ``niah_summary.json``. No GPU needed.

Accuracy cells carry a 95% Wilson interval over prompts (for multi-query, where a
prompt scores 0–1 in quarters, this treats each prompt as one item). Differences
carry a 95% bootstrap interval, resampling prompts jointly for both arms.

Run:
    uv run python -m experiments.stage_b.niah_analyze
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from bench.niah import TASKS, paired_diff_interval, score, wilson_interval

OUT = Path("experiments/stage_b/niah")
BASE = "uncompressed"


def load() -> dict:
    """{arm: {(task, length): {idx: score}}}"""
    data: dict = {}
    for path in sorted(OUT.glob("*.jsonl")):
        cells = defaultdict(dict)
        for line in path.read_text().splitlines():
            r = json.loads(line)
            # Rescore from the stored output so a scorer change needs no rerun.
            cells[(r["task"], r["length"])][r["idx"]] = score(r["output"], SimpleNamespace(**r))
        data[path.stem] = dict(cells)
    return data


def summarize(data: dict) -> list[dict]:
    rows = []
    for arm, cells in data.items():
        for (task, length), scores in sorted(cells.items()):
            n = len(scores)
            mean = sum(scores.values()) / n
            lo, hi = wilson_interval(mean, n)
            row = {"arm": arm, "task": task, "length": length, "n": n,
                   "acc": mean, "acc_lo": lo, "acc_hi": hi}
            base = data.get(BASE, {}).get((task, length))
            if arm != BASE and base:
                shared = sorted(set(scores) & set(base))
                d, dlo, dhi = paired_diff_interval(
                    [scores[i] for i in shared], [base[i] for i in shared], seed=0)
                row.update(n_paired=len(shared), diff=d, diff_lo=dlo, diff_hi=dhi)
            rows.append(row)
    return rows


def to_markdown(rows: list[dict]) -> str:
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    arms.sort(key=lambda a: (a != BASE, a))
    lengths = sorted({r["length"] for r in rows})
    by = {(r["arm"], r["task"], r["length"]): r for r in rows}
    head = "| Arm | " + " | ".join(f"{L // 1024}K" for L in lengths) + " |"
    sep = "|---|" + "---:|" * len(lengths)
    out = []
    for task in TASKS:
        out += [f"## {task}: accuracy % [95% interval], n per cell in parentheses", "", head, sep]
        for arm in arms:
            cells = []
            for L in lengths:
                r = by.get((arm, task, L))
                cells.append("—" if r is None else
                             f"{100 * r['acc']:.1f} [{100 * r['acc_lo']:.1f}, {100 * r['acc_hi']:.1f}] ({r['n']})")
            out.append(f"| {arm} | " + " | ".join(cells) + " |")
        out += ["", f"## {task}: difference from {BASE}, percentage points [95% interval]", "", head, sep]
        for arm in arms:
            if arm == BASE:
                continue
            cells = []
            for L in lengths:
                r = by.get((arm, task, L))
                cells.append("—" if r is None or "diff" not in r else
                             f"{100 * r['diff']:+.1f} [{100 * r['diff_lo']:+.1f}, {100 * r['diff_hi']:+.1f}]")
            out.append(f"| {arm} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    rows = summarize(load())
    (OUT / "niah_summary.json").write_text(json.dumps(rows, indent=2))
    md = to_markdown(rows)
    (OUT / "niah_summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
