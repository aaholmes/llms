"""Stage A.5 ship-gate evaluation: end-to-end greedy + spec-decode bench.

Runs Qwen3-4B over a 100-prompt × 200-token Dolly-15k sample, comparing five
configurations:

  hf_greedy            HuggingFace AutoModelForCausalLM, greedy
  ours_greedy_eager    our engine, no Triton kernels, greedy
  ours_spec_eager      our engine, no Triton kernels, spec K=4
  ours_greedy_triton   our engine + fused gate-up-silu + fused QKV, greedy
  ours_spec_triton     our engine + Triton kernels, spec K-sweep ({3,4,5,7})

Three Qwen3-4B-class models cannot fit on 16 GB simultaneously, so phases
load sequentially:

  Phase A — HF greedy
  Phase B — ours eager (greedy + spec K=4)
  Phase C — ours + Triton (greedy + spec K-sweep)

Outputs (under --out, default ``experiments/stage_a/``):

  e2e_results.json    full per-prompt timings + percentiles + acceptance stats
  e2e_summary.md      headline pass/fail vs the 1.5× ship gate

Usage:
    uv run python -m bench.e2e
    uv run python -m bench.e2e --n-prompts 20 --K 4 --phase B C   # quick

The formal Stage A.5 ship gate (DESIGN.md):
    median(ours_spec_triton_K4) / median(ours_greedy_triton) >= 1.5
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

# Make PyTorch's CUDA allocator more willing to return memory between phases —
# we load and unload three Qwen3-4B-class models sequentially on a 16 GB GPU
# and stock fragmentation patterns can pin a few hundred MB at the wrong time.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoTokenizer

from bench._e2e_prompts import PROMPTS
from engine.model import Qwen3Model
from engine.sampler import greedy
from engine.spec_decode import speculative_generate
from engine.weights import load_weights


# ---------- timing + summary helpers ---------------------------------------


@dataclass
class ResultRow:
    label: str
    median_tps: float
    p10_tps: float
    p90_tps: float
    n_prompts: int
    n_tokens_per_prompt: int
    per_prompt_wall_s: list[float] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def _percentiles(walls: list[float], n_tokens: int) -> tuple[float, float, float]:
    """Return (median_tps, p10_tps, p90_tps) from per-prompt wall times."""
    tps = sorted(n_tokens / w for w in walls)
    n = len(tps)
    return (
        statistics.median(tps),
        tps[n // 10],
        tps[(n * 9) // 10] if n >= 10 else tps[-1],
    )


def _make_row(
    label: str,
    walls: list[float],
    n_tokens: int,
    *,
    extra: dict | None = None,
) -> ResultRow:
    median_tps, p10, p90 = _percentiles(walls, n_tokens)
    return ResultRow(
        label=label,
        median_tps=median_tps,
        p10_tps=p10,
        p90_tps=p90,
        n_prompts=len(walls),
        n_tokens_per_prompt=n_tokens,
        per_prompt_wall_s=walls,
        extra=extra or {},
    )


def _time_generation(
    fn: Callable[[], object],
) -> tuple[object, float]:
    """Time ``fn()`` end-to-end on the GPU using CUDA events."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        return out, start.elapsed_time(end) / 1000.0
    # CPU fallback for the smoke test on dev boxes.
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


# ---------- generators ------------------------------------------------------


def _our_greedy(model: Qwen3Model, prompt_ids: torch.Tensor, max_new: int) -> list[int]:
    cache = model.alloc_cache(prompt_ids.shape[1] + max_new + 4)
    out: list[int] = []
    with torch.inference_mode():
        logits = model(prompt_ids, cache, start_pos=0)
        nt = greedy(logits[:, -1, :]).unsqueeze(0)
        out.append(int(nt.item()))
        for _ in range(max_new - 1):
            logits = model(nt, cache)
            nt = greedy(logits[:, -1, :]).unsqueeze(0)
            out.append(int(nt.item()))
    return out


def _our_spec(
    target: Qwen3Model,
    draft: Qwen3Model,
    prompt_ids: torch.Tensor,
    max_new: int,
    K: int,
):
    with torch.inference_mode():
        return speculative_generate(
            target=target,
            draft=draft,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new,
            K=K,
        )


def _hf_greedy(hf_model, prompt_ids: torch.Tensor, max_new: int):
    """HF ``generate`` with greedy + use_cache=True. EOS suppressed via
    ``min_new_tokens=max_new_tokens`` so every prompt produces exactly
    ``max_new`` tokens (apples-to-apples wall comparison)."""
    with torch.inference_mode():
        return hf_model.generate(
            prompt_ids,
            do_sample=False,
            use_cache=True,
            max_new_tokens=max_new,
            min_new_tokens=max_new,
        )


# ---------- model loading ---------------------------------------------------


def _load_ours(model_id: str, dtype: torch.dtype, device: str) -> Qwen3Model:
    """Load weights to CPU first, then move the model to GPU.

    Avoids the 2× GPU transient that ``load_weights(device='cuda')`` +
    ``model.to('cuda')`` creates — relevant on a 16 GB GPU where any residue
    from a prior phase can push the transient past the budget.
    """
    loaded = load_weights(model_id, dtype=dtype, device="cpu")
    model = Qwen3Model.from_loaded(loaded)
    del loaded
    gc.collect()
    return model.to(dtype=dtype, device=device).eval()


def _free():
    """Aggressive GPU memory release between phases. Two ``gc.collect()``
    passes catch cyclic refs (HF generation caches keep a few)."""
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------- phase runners ---------------------------------------------------


def phase_a_hf(
    *, target_id: str, prompts_ids: list[torch.Tensor], max_new: int,
    dtype: torch.dtype, device: str,
) -> list[ResultRow]:
    """HuggingFace greedy on Qwen3-4B."""
    from transformers import AutoModelForCausalLM

    print(f"[e2e/A] loading HF {target_id} ...")
    hf = (
        AutoModelForCausalLM.from_pretrained(target_id, dtype=dtype)
        .to(device)
        .eval()
    )

    # Warmup once before timing — first call into HF generate hits cuBLAS init,
    # cuDNN init, and any internal state setup.
    print("[e2e/A] HF warmup ...")
    _hf_greedy(hf, prompts_ids[0][:, :8], max_new=4)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"[e2e/A] HF greedy: {len(prompts_ids)} prompts × {max_new} tokens ...")
    walls: list[float] = []
    for i, ids in enumerate(prompts_ids):
        _, wall = _time_generation(lambda: _hf_greedy(hf, ids, max_new=max_new))
        walls.append(wall)
        if (i + 1) % 20 == 0:
            print(
                f"[e2e/A]   {i + 1}/{len(prompts_ids)}: "
                f"median {statistics.median(walls):.2f}s"
            )

    rows = [_make_row("hf_greedy", walls, max_new)]

    del hf
    _free()
    return rows


def phase_b_ours_eager(
    *, target_id: str, draft_id: str, prompts_ids: list[torch.Tensor],
    max_new: int, K: int, dtype: torch.dtype, device: str,
) -> list[ResultRow]:
    """Our engine, no Triton kernels: greedy + spec K=4."""
    print(f"[e2e/B] loading ours target {target_id} ...")
    target = _load_ours(target_id, dtype, device)
    _free()
    if target_id == draft_id:
        draft = target
        print("[e2e/B] (self-spec — draft is the target)")
    else:
        print(f"[e2e/B] loading ours draft {draft_id} ...")
        draft = _load_ours(draft_id, dtype, device)
        _free()

    print("[e2e/B] warmup ...")
    _our_greedy(target, prompts_ids[0][:, :8], max_new=4)
    _our_spec(target, draft, prompts_ids[0][:, :8], max_new=4, K=K)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"[e2e/B] greedy: {len(prompts_ids)} prompts × {max_new} tokens ...")
    g_walls: list[float] = []
    for i, ids in enumerate(prompts_ids):
        _, wall = _time_generation(lambda: _our_greedy(target, ids, max_new=max_new))
        g_walls.append(wall)
        if (i + 1) % 20 == 0:
            print(
                f"[e2e/B]   greedy {i + 1}/{len(prompts_ids)}: "
                f"median {statistics.median(g_walls):.2f}s"
            )

    print(f"[e2e/B] spec K={K}: {len(prompts_ids)} prompts × {max_new} tokens ...")
    s_walls: list[float] = []
    s_accs: list[float] = []
    s_rounds: list[int] = []
    for i, ids in enumerate(prompts_ids):
        result, wall = _time_generation(
            lambda: _our_spec(target, draft, ids, max_new=max_new, K=K)
        )
        _, stats = result
        s_walls.append(wall)
        s_accs.append(stats.acceptance_rate)
        s_rounds.append(stats.rounds)
        if (i + 1) % 20 == 0:
            print(
                f"[e2e/B]   spec   {i + 1}/{len(prompts_ids)}: "
                f"median {statistics.median(s_walls):.2f}s, "
                f"acc {statistics.median(s_accs):.1%}"
            )

    rows = [
        _make_row("ours_greedy_eager", g_walls, max_new),
        _make_row(
            f"ours_spec_eager_K{K}",
            s_walls,
            max_new,
            extra={
                "acceptance_rate_median": statistics.median(s_accs),
                "rounds_median": statistics.median(s_rounds),
                "K": K,
            },
        ),
    ]

    same_model = draft is target
    del target
    if not same_model:
        del draft
    _free()
    return rows


def phase_c_ours_triton(
    *, target_id: str, draft_id: str, prompts_ids: list[torch.Tensor],
    max_new: int, K_values: list[int], dtype: torch.dtype, device: str,
) -> list[ResultRow]:
    """Our engine + Triton kernels: greedy + spec K-sweep."""
    from kernels import apply_triton_kernels, prewarm_triton_kernels

    print(f"[e2e/C] loading ours target {target_id} ...")
    target = _load_ours(target_id, dtype, device)
    _free()
    if target_id == draft_id:
        draft = target
        print("[e2e/C] (self-spec — draft is the target)")
    else:
        print(f"[e2e/C] loading ours draft {draft_id} ...")
        draft = _load_ours(draft_id, dtype, device)
        _free()

    print("[e2e/C] applying Triton kernels + prewarming autotune ...")
    apply_triton_kernels(target)
    prewarm_triton_kernels(target)
    if draft is not target:
        apply_triton_kernels(draft)
        prewarm_triton_kernels(draft)

    print("[e2e/C] warmup ...")
    _our_greedy(target, prompts_ids[0][:, :8], max_new=4)
    for K in K_values:
        _our_spec(target, draft, prompts_ids[0][:, :8], max_new=4, K=K)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"[e2e/C] greedy: {len(prompts_ids)} prompts × {max_new} tokens ...")
    g_walls: list[float] = []
    for i, ids in enumerate(prompts_ids):
        _, wall = _time_generation(lambda: _our_greedy(target, ids, max_new=max_new))
        g_walls.append(wall)
        if (i + 1) % 20 == 0:
            print(
                f"[e2e/C]   greedy {i + 1}/{len(prompts_ids)}: "
                f"median {statistics.median(g_walls):.2f}s"
            )

    rows: list[ResultRow] = [_make_row("ours_greedy_triton", g_walls, max_new)]

    for K in K_values:
        print(
            f"[e2e/C] spec K={K}: {len(prompts_ids)} prompts × {max_new} tokens ..."
        )
        s_walls: list[float] = []
        s_accs: list[float] = []
        s_rounds: list[int] = []
        for i, ids in enumerate(prompts_ids):
            result, wall = _time_generation(
                lambda: _our_spec(target, draft, ids, max_new=max_new, K=K)
            )
            _, stats = result
            s_walls.append(wall)
            s_accs.append(stats.acceptance_rate)
            s_rounds.append(stats.rounds)
            if (i + 1) % 20 == 0:
                print(
                    f"[e2e/C]   spec K={K} {i + 1}/{len(prompts_ids)}: "
                    f"median {statistics.median(s_walls):.2f}s, "
                    f"acc {statistics.median(s_accs):.1%}"
                )
        rows.append(
            _make_row(
                f"ours_spec_triton_K{K}",
                s_walls,
                max_new,
                extra={
                    "acceptance_rate_median": statistics.median(s_accs),
                    "rounds_median": statistics.median(s_rounds),
                    "K": K,
                },
            )
        )

    same_model = draft is target
    del target
    if not same_model:
        del draft
    _free()
    return rows


# ---------- output writers --------------------------------------------------


def _gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"name": "cpu", "cap": "n/a", "mem_gb": 0.0}
    p = torch.cuda.get_device_properties(0)
    return {
        "name": p.name,
        "cap": f"{p.major}.{p.minor}",
        "mem_gb": p.total_memory / (1024**3),
    }


def _ship_gate(rows: list[ResultRow], target: float = 1.5) -> dict | None:
    """Compute the formal Stage A.5 ship gate ratio if both rows are present."""
    by_label = {r.label: r for r in rows}
    num = by_label.get("ours_spec_triton_K4")
    den = by_label.get("ours_greedy_triton")
    if num is None or den is None:
        return None
    ratio = num.median_tps / den.median_tps
    return {
        "numerator": "ours_spec_triton_K4.median_tps",
        "denominator": "ours_greedy_triton.median_tps",
        "ratio": ratio,
        "target": target,
        "passed": ratio >= target,
    }


def _write_json(out: Path, rows: list[ResultRow], config: dict, gpu: dict) -> None:
    payload = {
        "gpu": gpu,
        "config": config,
        "results": [asdict(r) for r in rows],
    }
    gate = _ship_gate(rows)
    if gate is not None:
        payload["ship_gate"] = gate
    out.write_text(json.dumps(payload, indent=2))


def _write_summary(out: Path, rows: list[ResultRow], config: dict, gpu: dict) -> None:
    lines: list[str] = []
    lines.append("# Stage A.5 ship-gate evaluation\n")
    lines.append(
        f"- GPU: **{gpu['name']}** (cap {gpu['cap']}, {gpu['mem_gb']:.1f} GB)"
    )
    lines.append(
        f"- target: `{config['target']}`, draft: `{config['draft']}`, dtype={config['dtype']}"
    )
    lines.append(
        f"- prompts: {config['n_prompts']} (sampled from Dolly-15k, seed=42); "
        f"max_new={config['max_new_tokens']}"
    )
    lines.append("")

    gate = _ship_gate(rows)
    if gate is not None:
        verdict = "**PASS**" if gate["passed"] else "**FAIL**"
        lines.append(
            f"## Ship gate: {verdict} (ratio {gate['ratio']:.3f}, target {gate['target']:.2f})\n"
        )
        lines.append(f"- numerator: `{gate['numerator']}`")
        lines.append(f"- denominator: `{gate['denominator']}`")
        lines.append("")
    else:
        lines.append("## Ship gate: not computed (need both phase B and phase C)\n")

    lines.append("## Per-config tok/s\n")
    lines.append("| config | median | p10 | p90 | extra |")
    lines.append("|---|---:|---:|---:|---|")
    for r in rows:
        extra_bits = []
        if "acceptance_rate_median" in r.extra:
            extra_bits.append(f"acc={r.extra['acceptance_rate_median']:.1%}")
        if "rounds_median" in r.extra:
            extra_bits.append(f"rounds_med={r.extra['rounds_median']}")
        extras = ", ".join(extra_bits) or "—"
        lines.append(
            f"| `{r.label}` | {r.median_tps:.1f} | {r.p10_tps:.1f} | "
            f"{r.p90_tps:.1f} | {extras} |"
        )
    lines.append("")

    by_label = {r.label: r for r in rows}

    # Sanity narrative
    if "hf_greedy" in by_label and "ours_greedy_eager" in by_label:
        hf = by_label["hf_greedy"].median_tps
        ours = by_label["ours_greedy_eager"].median_tps
        lines.append(
            f"## Sanity\n\nOur eager greedy runs at **{ours:.1f} tok/s** vs "
            f"HuggingFace's **{hf:.1f} tok/s** on the same Qwen3-4B BF16 weights "
            f"({ours / hf:.2f}× HF). The kernel-free path is on par with the "
            f"reference implementation.\n"
        )

    # Spec speedup over greedy, per branch
    def _row(label: str) -> ResultRow | None:
        return by_label.get(label)

    eager_g = _row("ours_greedy_eager")
    triton_g = _row("ours_greedy_triton")
    spec_rows = sorted(
        [r for r in rows if r.label.startswith("ours_spec_")],
        key=lambda r: (
            "triton" in r.label,
            r.extra.get("K", 0),
        ),
    )
    if (eager_g or triton_g) and spec_rows:
        lines.append("## Spec speedup over greedy, by branch\n")
        lines.append("| spec config | K | spec tok/s | greedy tok/s | speedup |")
        lines.append("|---|---:|---:|---:|---:|")
        for sr in spec_rows:
            branch_g = triton_g if "triton" in sr.label else eager_g
            if branch_g is None:
                continue
            speedup = sr.median_tps / branch_g.median_tps
            lines.append(
                f"| `{sr.label}` | {sr.extra.get('K', '?')} | "
                f"{sr.median_tps:.1f} | {branch_g.median_tps:.1f} | "
                f"**{speedup:.2f}×** |"
            )
        lines.append("")

    # Did the Triton kernels help or hurt spec? Compare same-K eager vs triton.
    if "ours_spec_eager_K4" in by_label and "ours_spec_triton_K4" in by_label:
        ek = by_label["ours_spec_eager_K4"]
        tk = by_label["ours_spec_triton_K4"]
        ratio = tk.median_tps / ek.median_tps
        verb = "helped" if ratio > 1.0 else "hurt"
        lines.append("## Triton kernel net effect on spec decode\n")
        lines.append(
            f"At K=4: eager spec = **{ek.median_tps:.1f} tok/s** "
            f"(acc {ek.extra.get('acceptance_rate_median', 0):.1%}), "
            f"triton spec = **{tk.median_tps:.1f} tok/s** "
            f"(acc {tk.extra.get('acceptance_rate_median', 0):.1%}). "
            f"Triton {verb} spec by **{ratio:.2f}×**."
        )
        lines.append("")

    # Best K on the kerneled spec branch.
    triton_specs = [
        r for r in rows
        if r.label.startswith("ours_spec_triton_K") and triton_g is not None
    ]
    if triton_specs:
        best = max(triton_specs, key=lambda r: r.median_tps)
        lines.append("## Best K on Triton spec branch\n")
        lines.append(
            f"K={best.extra.get('K')}: **{best.median_tps:.1f} tok/s** "
            f"(acc {best.extra.get('acceptance_rate_median', 0):.1%}, "
            f"{best.median_tps / triton_g.median_tps:.2f}× over greedy_triton).\n"
        )

    out.write_text("\n".join(lines))


# ---------- entry point -----------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=str, default="Qwen/Qwen3-4B")
    p.add_argument("--draft", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--n-prompts", type=int, default=len(PROMPTS))
    p.add_argument("--max-new", type=int, default=200)
    p.add_argument(
        "--K", type=int, nargs="+", default=[3, 4, 5, 7],
        help="Spec K values for the Triton branch sweep. Phase B always uses K=4.",
    )
    p.add_argument(
        "--phase", type=str, nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"],
    )
    p.add_argument("--out", type=Path, default=Path("experiments/stage_a"))
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n = min(args.n_prompts, len(PROMPTS))
    prompts = PROMPTS[:n]
    print(f"[e2e] using {n} prompts (out of {len(PROMPTS)} available)")

    # Tokenize once with the engine's tokenizer; same tokens reused across phases.
    tok = AutoTokenizer.from_pretrained(args.target)
    prompts_ids = [
        tok(p, return_tensors="pt").input_ids.to(device) for p in prompts
    ]
    prompt_lens = [int(ids.shape[1]) for ids in prompts_ids]
    print(
        f"[e2e] prompt token lengths: min={min(prompt_lens)}, "
        f"median={statistics.median(prompt_lens)}, max={max(prompt_lens)}"
    )

    config = {
        "target": args.target,
        "draft": args.draft,
        "dtype": args.dtype,
        "n_prompts": n,
        "max_new_tokens": args.max_new,
        "K_sweep": list(args.K),
        "phases": args.phase,
    }
    gpu = _gpu_info()
    json_path = args.out / "e2e_results.json"
    md_path = args.out / "e2e_summary.md"

    rows: list[ResultRow] = []

    def _checkpoint() -> None:
        """Persist after each phase so a later crash doesn't drop earlier work."""
        _write_json(json_path, rows, config, gpu)
        _write_summary(md_path, rows, config, gpu)

    if "A" in args.phase:
        rows.extend(
            phase_a_hf(
                target_id=args.target,
                prompts_ids=prompts_ids,
                max_new=args.max_new,
                dtype=dtype,
                device=device,
            )
        )
        _checkpoint()
    if "B" in args.phase:
        rows.extend(
            phase_b_ours_eager(
                target_id=args.target,
                draft_id=args.draft,
                prompts_ids=prompts_ids,
                max_new=args.max_new,
                K=4,
                dtype=dtype,
                device=device,
            )
        )
        _checkpoint()
    if "C" in args.phase:
        rows.extend(
            phase_c_ours_triton(
                target_id=args.target,
                draft_id=args.draft,
                prompts_ids=prompts_ids,
                max_new=args.max_new,
                K_values=list(args.K),
                dtype=dtype,
                device=device,
            )
        )
        _checkpoint()

    print()
    print(f"[e2e] wrote {json_path}")
    print(f"[e2e] wrote {md_path}")
    gate = _ship_gate(rows)
    if gate is not None:
        verdict = "PASS" if gate["passed"] else "FAIL"
        print(
            f"[e2e] ship gate: {verdict} "
            f"(spec_triton_K4 / greedy_triton = {gate['ratio']:.3f}, target {gate['target']:.2f})"
        )


if __name__ == "__main__":
    main()
