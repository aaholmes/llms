"""Long-context retrieval (needle-in-a-haystack) on the uncompressed and MLA targets.

Collection step only: writes one JSONL row per (arm, task, length, prompt) to
``experiments/stage_b/niah/<arm>.jsonl``. ``niah_analyze.py`` reads those files.
Every arm sees identical token ids, built once and cached in ``niah/prompts.pt``.
Resumable: rows already on disk are skipped.

Grid: 3 tasks × context lengths 1K/4K/8K/16K/32K × 100 prompts, greedy decoding,
Qwen3 chat template with thinking off.

Run (GPU):
    uv run python -m experiments.stage_b.niah_eval                 # full grid, all arms
    uv run python -m experiments.stage_b.niah_eval --arms uncompressed --lengths 32768 --n 2
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoTokenizer

from bench.niah import TASKS, fit_sample, generate_greedy, score
from engine.model import Qwen3Model
from engine.weights import load_weights
from mla.heal import load_trainable, merge_lora, wrap_lora
from mla.swap import apply_mla

TARGET_ID = "Qwen/Qwen3-4B"
DEV, DT = "cuda", torch.bfloat16
SB = Path("experiments/stage_b")
OUT = SB / "niah"
# 1024 matches the heal's training length; 4K–32K are the paper's columns.
LENGTHS = [1024, 4096, 8192, 16384, 32768]
N = 100
MAX_NEW = {"single": 48, "multi_key": 48, "multi_query": 128}
RESERVE = 128  # prompt is fitted to length − RESERVE so prompt + answer ≤ length
CHUNK = 512

# label → (mla-artifact | None, dir holding best.pt | None)
ARMS = {
    "uncompressed": (None, None),
    "control-ft": (None, "control_ft"),
    "mla-1.33x": ("r512_drope128.pt", "heal_r512_drope128"),
    "mla-4x": ("r256_drope32.pt", "heal_r256_drope32"),
    "mla-5.33x": ("r128_drope32.pt", "heal_r128_drope32"),
    "mla-4x-unhealed": ("r256_drope32.pt", None),
    "mla-1.33x-unhealed": ("r512_drope128.pt", None),
    # 4× healed by matching the uncompressed model's top-k distribution on WikiText.
    "mla-4x-distill": ("r256_drope32.pt", "heal_r256_drope32_distill"),
    # Un-healed conversions whose SVD was calibrated on chat-formatted prompts.
    "mla-4x-unhealed-chatcal": ("r256_drope32_chatcal.pt", None),
    "mla-1.33x-unhealed-chatcal": ("r512_drope128_chatcal.pt", None),
}


def _seed(task: str, length: int, i: int) -> int:
    return TASKS.index(task) * 10_000_019 + length * 1_009 + i


def build_prompts(tok, lengths, n) -> dict:
    """{(task, length, i): {"ids": LongTensor, **sample fields}} for the whole grid."""
    def chat_ids(text: str) -> list[int]:
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, enable_thinking=False, tokenize=False,
        )
        return tok(rendered, add_special_tokens=False)["input_ids"]

    prompts = {}
    for task in TASKS:
        for length in lengths:
            for i in range(n):
                s = fit_sample(task, budget=length - RESERVE,
                               count_tokens=lambda t: len(chat_ids(t)),
                               seed=_seed(task, length, i))
                prompts[(task, length, i)] = {
                    "ids": torch.tensor(chat_ids(s.text), dtype=torch.int32),
                    "sample": s,
                }
            print(f"[niah] built {task} @ {length}", flush=True)
    return prompts


def load_or_build_prompts(tok, lengths, n) -> dict:
    path = OUT / "prompts.pt"
    cached = torch.load(path, weights_only=False) if path.exists() else {}
    want = [(t, L, i) for t in TASKS for L in lengths for i in range(n)]
    if all(k in cached for k in want):
        return cached
    # Prompts are a pure function of (task, length, i), so rebuilding the missing
    # cells reproduces the cached ones exactly.
    cached.update(build_prompts(tok, lengths, n))
    torch.save(cached, path)
    return cached


def build_arm(artifact, heal_dir) -> Qwen3Model:
    # CPU-first load then move: loading straight to CUDA double-allocates 8 GB.
    loaded = load_weights(TARGET_ID, dtype=DT, device="cpu")
    m = Qwen3Model.from_loaded(loaded).to(dtype=DT, device=DEV).eval()
    del loaded
    gc.collect(); torch.cuda.empty_cache()
    if artifact is not None:
        apply_mla(m, torch.load(SB / artifact, map_location="cpu", weights_only=False))
    if heal_dir is not None:
        wrap_lora(m, rank=16, alpha=32.0)
        unmatched = load_trainable(m, SB / heal_dir / "best.pt")
        if unmatched:
            raise RuntimeError(f"{heal_dir}: unmatched keys {unmatched[:5]}")
        merge_lora(m)
    return m.eval()


def run_arm(label, prompts, lengths, n, tok) -> None:
    path = OUT / f"{label}.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["task"], r["length"], r["idx"]))
    todo = [(t, L, i) for L in lengths for t in TASKS for i in range(n) if (t, L, i) not in done]
    if not todo:
        print(f"[niah] {label}: complete", flush=True)
        return

    model = build_arm(*ARMS[label])
    eos = (tok.eos_token_id, tok.convert_tokens_to_ids("<|endoftext|>"))
    t_arm = time.time()
    with path.open("a") as f:
        for j, (task, length, i) in enumerate(todo):
            p = prompts[(task, length, i)]
            s = p["sample"]
            ids = p["ids"].long().unsqueeze(0).to(DEV)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            out_ids = generate_greedy(model, ids, max_new=MAX_NEW[task], eos_ids=eos, chunk=CHUNK)
            wall = time.time() - t0
            text = tok.decode(out_ids, skip_special_tokens=True)
            row = {
                "arm": label, "task": task, "length": length, "idx": i,
                "prompt_tokens": ids.shape[1], "score": score(text, s),
                "output": text, "keys": s.keys, "numbers": s.numbers,
                "query_idx": s.query_idx, "depths": s.depths,
                "wall_s": round(wall, 2),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            }
            f.write(json.dumps(row) + "\n"); f.flush()
            if j % 25 == 0 or j == len(todo) - 1:
                print(f"[niah] {label} {j + 1}/{len(todo)} {task}@{length} "
                      f"score={row['score']:.2f} wall={wall:.1f}s peak={row['peak_gb']}GB "
                      f"elapsed={(time.time() - t_arm) / 60:.1f}min", flush=True)
    del model
    gc.collect(); torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--lengths", nargs="+", type=int, default=LENGTHS)
    ap.add_argument("--n", type=int, default=N)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TARGET_ID)
    prompts = load_or_build_prompts(tok, args.lengths, args.n)
    for label in args.arms:
        run_arm(label, prompts, args.lengths, args.n, tok)


if __name__ == "__main__":
    main()
