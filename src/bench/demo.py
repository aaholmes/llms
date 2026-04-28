"""End-to-end demo: load Qwen3 models, run greedy and spec-decode on a prompt.

This is for sanity checking, not measurement. Real benchmarks live in e2e.py
once the GPU desktop work begins.

Examples:
  # self-spec on the small model (works on M2; ~2 GB BF16):
  uv run python -m bench.demo --prompt "Hello, my name is"
  # asymmetric target/draft (intended for the 16 GB GPU):
  uv run python -m bench.demo --prompt "..." --target Qwen/Qwen3-4B --draft Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoTokenizer

from engine.model import Qwen3Model
from engine.sampler import greedy
from engine.spec_decode import speculative_generate
from engine.weights import load_weights

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _autodetect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _run_greedy(model: Qwen3Model, prompt_ids: torch.Tensor, n_new: int) -> list[int]:
    cache = model.alloc_cache(prompt_ids.shape[1] + n_new + 8)
    out: list[int] = []
    with torch.inference_mode():
        logits = model(prompt_ids, cache, start_pos=0)
        nt = greedy(logits[:, -1, :]).unsqueeze(0)
        out.append(int(nt.item()))
        for _ in range(n_new - 1):
            logits = model(nt, cache)
            nt = greedy(logits[:, -1, :]).unsqueeze(0)
            out.append(int(nt.item()))
    return out


def _load(model_id: str, dtype: torch.dtype, device: str) -> Qwen3Model:
    loaded = load_weights(model_id, dtype=dtype, device=device)
    return Qwen3Model.from_loaded(loaded).to(dtype=dtype, device=device).eval()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--target", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--draft", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=list(DTYPES))
    p.add_argument(
        "--use-triton", action="store_true",
        help="Enable Triton kernels (Stage A.5a: fused gate-up-silu).",
    )
    args = p.parse_args()

    device = _autodetect_device() if args.device == "auto" else args.device
    dtype = DTYPES[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.target)
    prompt_ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)
    print(
        f"prompt: {args.prompt!r} ({prompt_ids.shape[1]} tokens) "
        f"device={device} dtype={args.dtype} triton={args.use_triton}"
    )

    print(f"loading target: {args.target}")
    target = _load(args.target, dtype, device)
    if args.draft == args.target:
        draft = target
        print("(self-spec — draft is the target)")
    else:
        print(f"loading draft:  {args.draft}")
        draft = _load(args.draft, dtype, device)

    # Apply Triton kernels and prewarm autotune AFTER all loads — both
    # operations allocate transient GPU memory (W_qkv concat in
    # apply_triton_kernels; cache-flush buffer in autotune do_bench) that
    # won't fit alongside the staged state-dict duplicates load_weights
    # leaves on GPU until GC.
    if args.use_triton:
        from kernels import apply_triton_kernels, prewarm_triton_kernels

        apply_triton_kernels(target)
        prewarm_triton_kernels(target)
        if draft is not target:
            apply_triton_kernels(draft)
            prewarm_triton_kernels(draft)

    # Greedy baseline
    t0 = time.perf_counter()
    g = _run_greedy(target, prompt_ids, args.max_new)
    t_g = time.perf_counter() - t0
    print()
    print("=== greedy ===")
    print(tok.decode(g))
    print(f"  {args.max_new} tokens in {t_g:.2f}s = {args.max_new / t_g:.1f} tok/s")

    # Spec decode
    t0 = time.perf_counter()
    with torch.inference_mode():
        s, stats = speculative_generate(
            target=target,
            draft=draft,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new,
            K=args.K,
        )
    t_s = time.perf_counter() - t0
    print()
    print(f"=== spec decode (K={args.K}) ===")
    print(tok.decode(s))
    print(f"  {args.max_new} tokens in {t_s:.2f}s = {args.max_new / t_s:.1f} tok/s")
    print(f"  speedup vs greedy: {t_g / t_s:.2f}x")
    print(
        f"  rounds={stats.rounds} drafted={stats.drafted_tokens} "
        f"accepted={stats.accepted_drafted} bonus={stats.bonus_rounds} "
        f"acc_rate={stats.acceptance_rate:.2%}"
    )
    if g[: min(8, args.max_new)] == s[: min(8, args.max_new)]:
        print("  greedy and spec agree on first 8 tokens (sanity ✓)")
    else:
        print(f"  WARNING: greedy and spec diverge: g[:8]={g[:8]} s[:8]={s[:8]}")
        print("  (BF16 batched-vs-sequential drift is expected for short outputs; FP32 should agree.)")


if __name__ == "__main__":
    main()
