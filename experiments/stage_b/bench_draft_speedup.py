"""End-to-end speculative-decode speedup: base vs TVD-tuned draft.

Converts the draft's acceptance gain (base 58.9% → TVD-tuned 67.8% vs the 4B
target) into a measured throughput number. Both drafts are plain Qwen3-0.6B
models (the tuned one has its LoRA *merged* into the weights, so it runs at
exactly the base draft's speed — any tok/s difference is acceptance, not adapter
overhead). Greedy-verify speculative decode, so both drafts produce the *same*
tokens as plain greedy; only speed differs.

Measured on WikiText-val prompts (the workload the draft was tuned on), swept
over draft depth K, against a plain-greedy target baseline. Emits
experiments/stage_b/bench_draft_speedup.{md,json}.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

# Fight fragmentation across the many per-prompt cache alloc/free cycles — must
# be set before torch initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from engine.model import Qwen3Model
from engine.spec_decode import speculative_generate
from engine.weights import load_weights
from mla.calibrate import load_wikitext103_chunks
from mla.heal import load_trainable, merge_lora, wrap_lora

TARGET_ID = "Qwen/Qwen3-4B"
DRAFT_ID = "Qwen/Qwen3-0.6B"
DEV, DT = "cuda", torch.bfloat16
N_PROMPTS, PROMPT_TOK, MAX_NEW = 24, 128, 256
KS = [1, 2, 3, 4, 5, 6, 8]
SB = Path("experiments/stage_b")


def _plain(model_id):
    loaded = load_weights(model_id, dtype=DT, device="cpu")
    m = Qwen3Model.from_loaded(loaded).to(dtype=DT, device=DEV).eval()
    del loaded
    torch.cuda.empty_cache()
    return m


def _tuned_draft(heal_dir):
    loaded = load_weights(DRAFT_ID, dtype=DT, device="cpu")
    m = Qwen3Model.from_loaded(loaded).to(dtype=DT, device=DEV)
    del loaded
    torch.cuda.empty_cache()
    wrap_lora(m, rank=16, alpha=32.0)
    load_trainable(m, SB / heal_dir / "best.pt")
    merge_lora(m)  # fold LoRA in → plain model, base-draft speed
    return m.eval()


@torch.inference_mode()
def _greedy(model, ids, max_new):
    cache = model.alloc_cache(ids.shape[1] + max_new + 4)
    logits = model(ids.to(DEV), cache, start_pos=0)
    tok = int(logits[0, -1].argmax())
    pos = ids.shape[1]
    for _ in range(max_new - 1):
        logits = model(torch.tensor([[tok]], device=DEV), cache, start_pos=pos)
        pos += 1
        tok = int(logits[0, -1].argmax())


def _time(fn):
    torch.cuda.synchronize()
    t0 = time.time()
    fn()
    torch.cuda.synchronize()
    return time.time() - t0


def _bench_greedy(target, prompts):
    _greedy(target, prompts[0][:, :8], 4)  # warmup
    walls = [_time(lambda: _greedy(target, p, MAX_NEW)) for p in prompts]
    return MAX_NEW / statistics.median(walls)


@torch.inference_mode()  # else the forward graph is retained → activation blowup → OOM
def _bench_spec(target, draft, prompts, K):
    speculative_generate(target=target, draft=draft, prompt_ids=prompts[0][:, :8].to(DEV),
                         max_new_tokens=4, K=K)  # warmup
    walls, accs = [], []
    for p in prompts:
        torch.cuda.empty_cache()  # release the previous prompt's transient caches
        p = p.to(DEV)
        torch.cuda.synchronize()
        t0 = time.time()
        _, stats = speculative_generate(target=target, draft=draft, prompt_ids=p,
                                        max_new_tokens=MAX_NEW, K=K)
        torch.cuda.synchronize()
        walls.append(time.time() - t0)
        accs.append(stats.acceptance_rate)
    return MAX_NEW / statistics.median(walls), statistics.median(accs)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="wikitext", choices=["wikitext", "code"],
                    help="held-out prompt workload (must match the draft's tuning corpus)")
    ap.add_argument("--tuned-dir", default="heal_draft_0_6b_tvd", help="tuned-draft heal dir")
    ap.add_argument("--tuned-label", default="tvd-tuned")
    ap.add_argument("--out-suffix", default="", help="suffix for output md/json (e.g. _code)")
    args = ap.parse_args()

    if args.corpus == "code":
        from mla.calibrate import load_code_chunks
        # held-out: skip past the ~10-12k examples the teacher/tuning consumed
        chunks = load_code_chunks(n_samples=N_PROMPTS + 8, chunk_tokens=PROMPT_TOK,
                                  tokenizer_id=TARGET_ID, skip=20000)
    else:
        chunks = load_wikitext103_chunks(n_samples=N_PROMPTS, chunk_tokens=PROMPT_TOK,
                                         tokenizer_id=TARGET_ID, split="validation")
    prompts = [c[:, :PROMPT_TOK] for c in chunks][:N_PROMPTS]
    print(f"[bench] {len(prompts)} {args.corpus} prompts × {PROMPT_TOK} tok → {MAX_NEW} new", flush=True)

    target = _plain(TARGET_ID)
    greedy_tps = _bench_greedy(target, prompts)
    torch.cuda.empty_cache()
    print(f"[bench] greedy target: {greedy_tps:.1f} tok/s", flush=True)

    results = {"greedy_tok_s": greedy_tps, "corpus": args.corpus, "drafts": {}}
    for label, heal_dir in [("base", None), (args.tuned_label, args.tuned_dir)]:
        draft = _plain(DRAFT_ID) if heal_dir is None else _tuned_draft(heal_dir)
        per_k = {}
        for K in KS:
            tps, acc = _bench_spec(target, draft, prompts, K)
            per_k[K] = {"tok_s": tps, "acceptance": acc, "speedup": tps / greedy_tps}
            print(f"[bench] {label:10s} K={K}  {tps:.1f} tok/s  {tps/greedy_tps:.2f}×  "
                  f"acc={acc*100:.1f}%", flush=True)
        results["drafts"][label] = per_k
        del draft
        torch.cuda.empty_cache()

    sfx = args.out_suffix
    (SB / f"bench_draft_speedup{sfx}.json").write_text(json.dumps(results, indent=2))
    best = {lab: max(d.values(), key=lambda r: r["tok_s"]) for lab, d in results["drafts"].items()}
    md = ["# End-to-end spec-decode speedup: base vs TVD-tuned draft\n",
          f"Qwen3-0.6B draft → Qwen3-4B target, greedy-verify, {len(prompts)} {args.corpus} "
          f"prompts × {PROMPT_TOK}→{MAX_NEW} tok. Greedy target baseline: **{greedy_tps:.1f} tok/s**. "
          f"Tuned draft has its LoRA merged (same speed as base; only weights differ).\n",
          "| Draft | K | tok/s | speedup vs greedy | acceptance |", "|---|---:|---:|---:|---:|"]
    for label, d in results["drafts"].items():
        for K, r in d.items():
            md.append(f"| {label} | {K} | {r['tok_s']:.1f} | {r['speedup']:.2f}× | {r['acceptance']*100:.1f}% |")
    b, t = best["base"], best[args.tuned_label]
    md.append(f"\n**Best-K: base {b['tok_s']:.1f} tok/s ({b['speedup']:.2f}×) → "
              f"tuned {t['tok_s']:.1f} tok/s ({t['speedup']:.2f}×)** "
              f"— a {t['tok_s']/b['tok_s']:.2f}× throughput gain from TVD-tuning the draft "
              f"(acceptance {b['acceptance']*100:.1f}% → {t['acceptance']*100:.1f}%).")
    (SB / f"bench_draft_speedup{sfx}.md").write_text("\n".join(md) + "\n")
    print(f"[bench] wrote bench_draft_speedup{sfx}.{{md,json}}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
