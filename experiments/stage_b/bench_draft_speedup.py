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
import statistics
import time
from pathlib import Path

import torch

from engine.model import Qwen3Model
from engine.spec_decode import speculative_generate
from engine.weights import load_weights
from mla.calibrate import load_wikitext103_chunks
from mla.heal import load_trainable, merge_lora, wrap_lora

TARGET_ID = "Qwen/Qwen3-4B"
DRAFT_ID = "Qwen/Qwen3-0.6B"
DEV, DT = "cuda", torch.bfloat16
N_PROMPTS, PROMPT_TOK, MAX_NEW = 24, 128, 128
KS = [2, 3, 4]
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


def _bench_spec(target, draft, prompts, K):
    speculative_generate(target=target, draft=draft, prompt_ids=prompts[0][:, :8].to(DEV),
                         max_new_tokens=4, K=K)  # warmup
    walls, accs = [], []
    for p in prompts:
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
    chunks = load_wikitext103_chunks(n_samples=N_PROMPTS, chunk_tokens=PROMPT_TOK,
                                     tokenizer_id=TARGET_ID, split="validation")
    prompts = [c[:, :PROMPT_TOK] for c in chunks]
    print(f"[bench] {len(prompts)} prompts × {PROMPT_TOK} tok → {MAX_NEW} new", flush=True)

    target = _plain(TARGET_ID)
    greedy_tps = _bench_greedy(target, prompts)
    print(f"[bench] greedy target: {greedy_tps:.1f} tok/s", flush=True)

    results = {"greedy_tok_s": greedy_tps, "drafts": {}}
    for label, heal_dir in [("base", None), ("tvd-tuned", "heal_draft_0_6b_tvd")]:
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

    (SB / "bench_draft_speedup.json").write_text(json.dumps(results, indent=2))
    best = {lab: max(d.values(), key=lambda r: r["tok_s"]) for lab, d in results["drafts"].items()}
    md = ["# End-to-end spec-decode speedup: base vs TVD-tuned draft\n",
          f"Qwen3-0.6B draft → Qwen3-4B target, greedy-verify, {len(prompts)} WikiText-val "
          f"prompts × {PROMPT_TOK}→{MAX_NEW} tok. Greedy target baseline: **{greedy_tps:.1f} tok/s**. "
          f"Tuned draft has its LoRA merged (same speed as base; only weights differ).\n",
          "| Draft | K | tok/s | speedup vs greedy | acceptance |", "|---|---:|---:|---:|---:|"]
    for label, d in results["drafts"].items():
        for K, r in d.items():
            md.append(f"| {label} | {K} | {r['tok_s']:.1f} | {r['speedup']:.2f}× | {r['acceptance']*100:.1f}% |")
    b, t = best["base"], best["tvd-tuned"]
    md.append(f"\n**Best-K: base {b['tok_s']:.1f} tok/s ({b['speedup']:.2f}×) → "
              f"tuned {t['tok_s']:.1f} tok/s ({t['speedup']:.2f}×)** "
              f"— a {t['tok_s']/b['tok_s']:.2f}× throughput gain from TVD-tuning the draft "
              f"(acceptance {b['acceptance']*100:.1f}% → {t['acceptance']*100:.1f}%).")
    (SB / "bench_draft_speedup.md").write_text("\n".join(md) + "\n")
    print("[bench] wrote bench_draft_speedup.{md,json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
