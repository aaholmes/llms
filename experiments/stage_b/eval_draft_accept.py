"""Speculative-decode acceptance of a draft vs a target: base vs TVD-tuned draft.

The expected single-token spec-decode acceptance rate is Σ_v min(p_target, p_draft)
= 1 − TVD (Leviathan et al.). This measures it for the Qwen3-0.6B draft against the
Qwen3-4B target, BEFORE and AFTER TVD-tuning the draft's LoRA — the first evidence
for "a workload-tuned LoRA raises draft acceptance at negligible cost."

Two-pass (draft and target don't need to co-reside if we stage): pass 1 runs the
target once and stages its per-position full-vocab log-probs to CPU; pass 2 loads
each draft variant and accumulates acceptance/TVD/top-1 against the staged target.
Same 64-chunk WT103-val subset as eval_frontier_kl.py. Draft and target share the
Qwen3 tokenizer/vocab, so token ids align.

Emits experiments/stage_b/eval_summary_draft_accept.{md,json}.
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

TARGET_ID = "Qwen/Qwen3-4B"
DRAFT_ID = "Qwen/Qwen3-0.6B"
DEV, DT = "cuda", torch.bfloat16
LORA_RANK, LORA_ALPHA = 16, 32.0
N_CHUNKS = 64
CHUNK_TOKENS = 1024
SB = Path("experiments/stage_b")

# (label, heal-out-dir | None) — None = untuned base draft.
DRAFTS = [
    ("Qwen3-0.6B (base draft)", None),
    ("Qwen3-0.6B (TVD-tuned)", "heal_draft_0_6b_tvd"),
]


@torch.inference_mode()
def _logprobs(model, ids):
    cache = model.alloc_cache(ids.shape[1])
    logits = model(ids.to(DEV), cache, start_pos=0)
    return F.log_softmax(logits[0, :-1].float(), dim=-1)  # (T-1, V)


def main() -> int:
    chunks = load_wikitext103_chunks(
        n_samples=N_CHUNKS, chunk_tokens=CHUNK_TOKENS, tokenizer_id=TARGET_ID, split="validation"
    )
    print(f"[draft] {len(chunks)} chunks", flush=True)

    # Pass 1: target (4B) → stage per-chunk log-probs + argmax on CPU.
    print("[draft] pass 1: target (Qwen3-4B) staging log-probs", flush=True)
    loaded = load_weights(TARGET_ID, dtype=DT, device="cpu")
    tgt = Qwen3Model.from_loaded(loaded).to(dtype=DT, device=DEV).eval()
    del loaded
    torch.cuda.empty_cache()
    lp_ref, am_ref = [], []
    for i, ids in enumerate(chunks):
        lp = _logprobs(tgt, ids)
        lp_ref.append(lp.half().cpu())
        am_ref.append(lp.argmax(-1).cpu())
    del tgt
    torch.cuda.empty_cache()

    # Pass 2: each draft variant → acceptance / TVD / top-1 vs the target.
    rows = []
    for label, heal_dir in DRAFTS:
        if heal_dir is not None and not (SB / heal_dir / "best.pt").exists():
            print(f"[draft] SKIP {label} (no {heal_dir}/best.pt)", flush=True)
            continue
        loaded = load_weights(DRAFT_ID, dtype=DT, device="cpu")
        model = Qwen3Model.from_loaded(loaded).to(dtype=DT, device=DEV)
        del loaded
        torch.cuda.empty_cache()
        if heal_dir is not None:
            wrap_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA)
            miss = load_trainable(model, SB / heal_dir / "best.pt")
            if miss:
                print(f"  WARN unmatched: {miss[:4]}", flush=True)
        model.eval()
        accept_sum, agree, n = 0.0, 0, 0
        for i, ids in enumerate(chunks):
            lp_d = _logprobs(model, ids)              # (T-1,V) draft
            p_t = lp_ref[i].float().to(DEV).exp()     # target probs
            accept_sum += float(torch.minimum(p_t, lp_d.exp()).sum(-1).sum())
            agree += int((lp_d.argmax(-1) == am_ref[i].to(DEV)).sum())
            n += lp_d.shape[0]
        accept = 100.0 * accept_sum / n
        top1 = 100.0 * agree / n
        rows.append({"label": label, "accept_pct": accept, "top1_agree_pct": top1,
                     "tvd": 1.0 - accept / 100.0})
        print(f"[draft] {label:28s} accept={accept:.2f}%  top1={top1:.2f}%", flush=True)
        del model
        torch.cuda.empty_cache()

    (SB / "eval_summary_draft_accept.json").write_text(json.dumps({"points": rows}, indent=2))
    md = ["# Draft acceptance vs target (Qwen3-0.6B → Qwen3-4B): base vs TVD-tuned\n",
          f"Expected single-token speculative-decode acceptance = `Σ min(p_target, p_draft)` "
          f"= `1 − TVD`, per token over {len(chunks)}×{CHUNK_TOKENS}-tok WT103-val. Higher = "
          f"more drafted tokens accepted = faster spec decode.\n",
          "| Draft | Accept (1−TVD) ↑ | Top-1 agree ↑ |", "|---|---:|---:|"]
    for r in rows:
        md.append(f"| {r['label']} | {r['accept_pct']:.1f}% | {r['top1_agree_pct']:.1f}% |")
    (SB / "eval_summary_draft_accept.md").write_text("\n".join(md) + "\n")
    print("[draft] wrote eval_summary_draft_accept.{md,json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
