"""Batch-eval every healed operating point on the 250-chunk WT103-val slice.

Reconstructs each healed model (base -> apply_mla -> wrap_lora -> load_trainable)
and evaluates on the SAME slice the sweep/eval_healed used, so the whole
post-heal compression-PPL frontier is directly comparable. Baseline (21.64)
and the FT control (11.13, amortized across all points) are the references.

Emits experiments/stage_b/eval_summary_frontier.{md,json}.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from bench.eval_ppl import evaluate_ppl
from engine.model import Qwen3Model
from engine.weights import load_weights
from mla.calibrate import load_wikitext103_chunks
from mla.heal import load_trainable, wrap_lora
from mla.swap import apply_mla

MODEL_ID = "Qwen/Qwen3-4B"
DEV, DT = "cuda", torch.bfloat16
LORA_RANK, LORA_ALPHA = 16, 32.0
CONTROL_PPL = 11.1326  # uncompressed, same-budget FT control (amortized)

# (label, converted-artifact, heal-out-dir, nominal compression)
POINTS = [
    ("r256_drope32 (uniform)", "r256_drope32.pt", "heal_r256_drope32", 4.00),
    ("r256_drope32 (adaptive)", "r256_drope32_adaptive.pt", "heal_r256_drope32_adaptive", 4.00),
    ("r256_drope32 (adaptive mr128)", "r256_drope32_adaptive_mr128.pt", "heal_r256_drope32_adaptive_mr128", 4.00),
    ("r256_drope32 (adaptive normalized)", "r256_drope32_adaptive_norm.pt", "heal_r256_drope32_adaptive_norm", 4.00),
    ("r256_drope32 (distill-to-orig)", "r256_drope32.pt", "heal_r256_drope32_distill", 4.00),
    ("r256_drope32 (fisher-distill)", "r256_drope32_fisher.pt", "heal_r256_drope32_fisher_distill", 4.00),
    ("r192_drope32", "r192_drope32.pt", "heal_r192_drope32", 4.57),
    ("r128_drope32", "r128_drope32.pt", "heal_r128_drope32", 5.33),
    ("r512_drope128 (V-only)", "r512_drope128.pt", "heal_r512_drope128", 1.33),
]
SB = Path("experiments/stage_b")


def eval_one(chunks, artifact_path: Path, best_path: Path) -> float:
    loaded = load_weights(MODEL_ID, dtype=DT, device="cpu")
    model = Qwen3Model.from_loaded(loaded).to(dtype=DT, device=DEV)
    del loaded
    torch.cuda.empty_cache()
    apply_mla(model, torch.load(artifact_path, map_location="cpu", weights_only=False))
    wrap_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA)
    missing = load_trainable(model, best_path)
    if missing:
        print(f"  WARN unmatched trainable: {missing[:4]}", flush=True)
    model.eval()
    m = evaluate_ppl(model, chunks, progress_every=None)
    ppl = m["ppl"]
    del model
    torch.cuda.empty_cache()
    return ppl


def main() -> int:
    chunks = load_wikitext103_chunks(
        n_samples=256, chunk_tokens=1024, tokenizer_id=MODEL_ID, split="validation"
    )
    print(f"[frontier-eval] {len(chunks)} chunks", flush=True)

    # Baseline on the same slice (sanity: should reproduce ~21.64).
    loaded = load_weights(MODEL_ID, dtype=DT, device="cpu")
    model = Qwen3Model.from_loaded(loaded).to(dtype=DT, device=DEV).eval()
    del loaded
    torch.cuda.empty_cache()
    base_ppl = evaluate_ppl(model, chunks, progress_every=None)["ppl"]
    del model
    torch.cuda.empty_cache()
    print(f"[frontier-eval] baseline={base_ppl:.4f}", flush=True)

    rows = []
    for label, art, out, comp in POINTS:
        art_p, best_p = SB / art, SB / out / "best.pt"
        if not best_p.exists():
            print(f"[frontier-eval] SKIP {label} (no {best_p})", flush=True)
            continue
        ppl = eval_one(chunks, art_p, best_p)
        rows.append({
            "label": label, "compression": comp, "ppl": ppl,
            "delta_vs_baseline_pct": (ppl - base_ppl) / base_ppl * 100.0,
            "delta_vs_control_pct": (ppl - CONTROL_PPL) / CONTROL_PPL * 100.0,
        })
        print(f"[frontier-eval] {label:26s} comp={comp:.2f}x ppl={ppl:.4f} "
              f"dbase={rows[-1]['delta_vs_baseline_pct']:+.1f}% "
              f"dctrl={rows[-1]['delta_vs_control_pct']:+.1f}%", flush=True)

    result = {"baseline_ppl": base_ppl, "control_ppl": CONTROL_PPL, "points": rows}
    (SB / "eval_summary_frontier.json").write_text(json.dumps(result, indent=2))

    md = ["# Post-heal compression–PPL frontier (250-chunk WT103-val slice)\n",
          f"baseline={base_ppl:.2f}  |  same-budget FT control={CONTROL_PPL:.2f} (uncompressed)\n",
          "| Config | Compression | PPL | Δ% vs baseline | Δ% vs FT control |",
          "|---|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['label']} | {r['compression']:.2f}× | {r['ppl']:.2f} | "
                  f"{r['delta_vs_baseline_pct']:+.1f}% | {r['delta_vs_control_pct']:+.1f}% |")
    (SB / "eval_summary_frontier.md").write_text("\n".join(md) + "\n")
    print("[frontier-eval] wrote eval_summary_frontier.{md,json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
