"""Behavioral-fidelity eval: KL-to-original + top-1 agreement across the frontier.

PPL asks "how well does the healed model predict the data." The project's actual
objective is "does the healed model behave like the *original*." That is measured
directly by the per-token divergence of the healed model's next-token distribution
from the original's — forward KL(p_orig ‖ p_healed) — and by top-1 agreement
(fraction of positions where the two models' argmax match).

Two 4B models don't co-fit in 16 GB, so this is a two-pass design: pass 1 runs the
original and stages its per-position log-probs to CPU RAM (fp16); pass 2 loads each
healed model in turn and accumulates KL + agreement against the staged reference.
Uses a 64-chunk subset of the same seed-42 WT103-val slice as the PPL frontier
(~65k scored positions — a stable KL mean).

Emits experiments/stage_b/eval_summary_frontier_kl.{md,json}.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from engine.model import Qwen3Model
from engine.weights import load_weights
from mla.calibrate import load_wikitext103_chunks
from mla.heal import load_trainable, wrap_lora
from mla.swap import apply_mla, alloc_mla_cache

MODEL_ID = "Qwen/Qwen3-4B"
DEV, DT = "cuda", torch.bfloat16
LORA_RANK, LORA_ALPHA = 16, 32.0
N_CHUNKS = 64
CHUNK_TOKENS = 1024
SB = Path("experiments/stage_b")

# (label, converted-artifact | None, heal-out-dir, compression). None artifact
# = no MLA swap (the uncompressed FT control — a reference for adaptation drift).
POINTS = [
    ("FT control (uncompressed)", None, "control_ft", 1.00),
    ("r512_drope128 (V-only)", "r512_drope128.pt", "heal_r512_drope128", 1.33),
    ("r256_drope32 (uniform)", "r256_drope32.pt", "heal_r256_drope32", 4.00),
    ("r256_drope32 (adaptive raw)", "r256_drope32_adaptive.pt", "heal_r256_drope32_adaptive", 4.00),
    ("r256_drope32 (adaptive mr128)", "r256_drope32_adaptive_mr128.pt", "heal_r256_drope32_adaptive_mr128", 4.00),
    ("r256_drope32 (adaptive norm)", "r256_drope32_adaptive_norm.pt", "heal_r256_drope32_adaptive_norm", 4.00),
    ("r192_drope32", "r192_drope32.pt", "heal_r192_drope32", 4.57),
    ("r128_drope32", "r128_drope32.pt", "heal_r128_drope32", 5.33),
]


def _build_model(artifact_name, best_dir):
    """Reconstruct a healed (or control) model: base → [apply_mla] → LoRA → load_trainable."""
    loaded = load_weights(MODEL_ID, dtype=DT, device="cpu")
    model = Qwen3Model.from_loaded(loaded).to(dtype=DT, device=DEV)
    del loaded
    torch.cuda.empty_cache()
    is_mla = artifact_name is not None
    if is_mla:
        apply_mla(model, torch.load(SB / artifact_name, map_location="cpu", weights_only=False))
    wrap_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA)
    missing = load_trainable(model, SB / best_dir / "best.pt")
    if missing:
        print(f"  WARN unmatched trainable: {missing[:4]}", flush=True)
    model.eval()
    return model, is_mla


def _alloc(model, is_mla, T):
    return alloc_mla_cache(model, max_seq_len=T) if is_mla else model.alloc_cache(T)


@torch.inference_mode()
def _logprobs(model, is_mla, ids):
    """log_softmax over scored positions [0, T-1) — the next-token distributions."""
    cache = _alloc(model, is_mla, ids.shape[1])
    logits = model(ids.to(DEV), cache, start_pos=0)
    return F.log_softmax(logits[0, :-1].float(), dim=-1)  # (T-1, V) on GPU


def main() -> int:
    chunks = load_wikitext103_chunks(
        n_samples=N_CHUNKS, chunk_tokens=CHUNK_TOKENS, tokenizer_id=MODEL_ID, split="validation"
    )
    print(f"[kl] {len(chunks)} chunks (subset of the seed-42 slice)", flush=True)

    # --- Pass 1: original model → stage per-chunk log-probs (fp16) + argmax on CPU ---
    print("[kl] pass 1: original model (staging log-probs to CPU)", flush=True)
    loaded = load_weights(MODEL_ID, dtype=DT, device="cpu")
    orig = Qwen3Model.from_loaded(loaded).to(dtype=DT, device=DEV).eval()
    del loaded
    torch.cuda.empty_cache()

    lp_ref, am_ref = [], []
    for i, ids in enumerate(chunks):
        lp = _logprobs(orig, False, ids)
        lp_ref.append(lp.half().cpu())
        am_ref.append(lp.argmax(-1).cpu())
        if (i + 1) % 16 == 0:
            print(f"[kl]   staged {i+1}/{len(chunks)}", flush=True)
    del orig
    torch.cuda.empty_cache()

    # --- Pass 2: each point → KL(orig ‖ point) + top-1 agreement vs staged reference ---
    rows = []
    for label, art, best_dir, comp in POINTS:
        if not (SB / best_dir / "best.pt").exists():
            print(f"[kl] SKIP {label} (no {best_dir}/best.pt)", flush=True)
            continue
        model, is_mla = _build_model(art, best_dir)
        kl_sum, agree, n = 0.0, 0, 0
        for i, ids in enumerate(chunks):
            lp_h = _logprobs(model, is_mla, ids)                 # (T-1,V) GPU
            lp_o = lp_ref[i].float().to(DEV)                     # (T-1,V) GPU
            p_o = lp_o.exp()
            kl_sum += float((p_o * (lp_o - lp_h)).sum(-1).sum())  # Σ_pos Σ_v p_o (lp_o-lp_h)
            agree += int((lp_h.argmax(-1) == am_ref[i].to(DEV)).sum())
            n += lp_h.shape[0]
        kl = kl_sum / n
        top1 = 100.0 * agree / n
        rows.append({"label": label, "compression": comp, "kl_nats": kl, "top1_agree_pct": top1})
        print(f"[kl] {label:32s} comp={comp:.2f}x  KL={kl:.4f} nats  top1={top1:.2f}%", flush=True)
        del model
        torch.cuda.empty_cache()

    (SB / "eval_summary_frontier_kl.json").write_text(
        json.dumps({"n_chunks": len(chunks), "points": rows}, indent=2))
    md = ["# Behavioral fidelity to the original: KL + top-1 agreement",
          f"\nForward KL(p_original ‖ p_healed) and top-1 argmax agreement, per token, "
          f"over a {len(chunks)}-chunk × {CHUNK_TOKENS}-tok WT103-val subset. "
          f"KL=0 / top1=100% would be behaviorally identical to the original.\n",
          "| Config | Compression | KL to orig (nats) ↓ | Top-1 agree ↑ |",
          "|---|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['label']} | {r['compression']:.2f}× | {r['kl_nats']:.3f} | {r['top1_agree_pct']:.1f}% |")
    (SB / "eval_summary_frontier_kl.md").write_text("\n".join(md) + "\n")
    print("[kl] wrote eval_summary_frontier_kl.{md,json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
