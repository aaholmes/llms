"""Profile Qwen3 decode hot path.

Stage A.5 step 1: identify the dominant op in autoregressive decode so the
single Triton kernel choice is data-driven, not assumed. The headline question
this script answers is: which op (or op category) consumes >30% of CUDA time
during steady-state decode of Qwen3-4B on the target GPU?

Outputs (written under --out, default experiments/stage_a/):
  - profile_summary.md       human-readable narrative + recommendation
  - profile_data.json        raw aggregated per-op + per-category stats
  - torch_trace_greedy.json  Chrome trace for the greedy run
  - torch_trace_spec.json    Chrome trace for the spec-decode run

Usage (GPU desktop):
  uv run python -m bench.profile \\
      --target Qwen/Qwen3-4B --draft Qwen/Qwen3-0.6B \\
      --prompt "The capital of France is" \\
      --decode-steps 200 --K 4

For kernel-level metrics (HBM throughput, L2 hit rate, SM occupancy), wrap
this script with Nsight Compute instead — example at the bottom of the summary.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoTokenizer

from engine.model import Qwen3Model
from engine.sampler import greedy
from engine.spec_decode import speculative_generate
from engine.weights import load_weights

DEFAULT_PROMPT = (
    "The history of computing is a story of doubling. Every decade or so, the "
    "machinery shrinks while what it can do roughly doubles. The interesting "
    "question is whether that pattern survives when"
)


# ---------- categorization ---------------------------------------------------

# torch.profiler reports events at multiple layers — framework ops (aten::*,
# user-named record_function regions) and the underlying device kernels — and
# their self-CUDA times overlap. To avoid double-counting we classify each
# event as either FRAMEWORK_OP or DEVICE_KERNEL and aggregate one layer at a
# time. The simplest reliable signal: framework ops have CPU time > 0 (Python
# called them); raw CUDA kernel events have CPU time == 0.

_USER_REGIONS = {"greedy_decode_steady_state", "spec_decode_steady_state"}

# Category rules over framework op names. First match wins; specific first.
_FRAMEWORK_CATEGORY_RULES: list[tuple[str, str]] = [
    ("scaled_dot_product_attention", "attention_core"),
    ("flash_attention", "attention_core"),
    ("efficient_attention", "attention_core"),
    ("repeat_interleave", "gqa_expand"),
    ("index_put", "kv_io"),
    ("copy_", "kv_io"),
    ("aten::slice", "memory_layout"),
    ("aten::as_strided", "memory_layout"),
    ("aten::cat", "memory_layout"),
    ("aten::transpose", "memory_layout"),
    ("aten::view", "memory_layout"),
    ("aten::reshape", "memory_layout"),
    ("aten::contiguous", "memory_layout"),
    ("aten::_to_copy", "memory_layout"),
    ("aten::expand", "memory_layout"),
    ("aten::permute", "memory_layout"),
    ("aten::embedding", "embedding"),
    ("rms_norm", "norm_act"),
    ("rsqrt", "norm_act"),
    ("layer_norm", "norm_act"),
    ("silu", "norm_act"),
    ("aten::mean", "norm_act"),
    ("aten::pow", "norm_act"),
    ("addmm", "matmul"),
    ("matmul", "matmul"),
    ("bmm", "matmul"),
    ("aten::mm", "matmul"),
    ("linear", "matmul"),
    ("aten::mul", "elementwise"),
    ("aten::add", "elementwise"),
    ("aten::div", "elementwise"),
    ("aten::sub", "elementwise"),
    ("aten::neg", "elementwise"),
    ("argmax", "sampling"),
    ("topk", "sampling"),
]


def categorize(name: str) -> str:
    for needle, cat in _FRAMEWORK_CATEGORY_RULES:
        if needle in name:
            return cat
    return "other"


def is_framework_op(row: "OpRow") -> bool:
    """Heuristic: framework ops (aten::*, custom autograd nodes) have CPU time;
    pure CUDA kernel events do not. Filter our user-named regions out too."""
    if row.name in _USER_REGIONS:
        return False
    return row.self_cpu_us > 0


def is_device_kernel(row: "OpRow") -> bool:
    if row.name in _USER_REGIONS:
        return False
    return row.self_cpu_us == 0 and row.self_cuda_us > 0


# ---------- profiler runners -------------------------------------------------


@dataclass
class OpRow:
    name: str
    self_cuda_us: float = 0.0
    self_cpu_us: float = 0.0
    cuda_us: float = 0.0  # includes children (a "device time")
    cpu_us: float = 0.0
    count: int = 0


@dataclass
class ProfileResult:
    label: str
    wall_seconds: float
    decode_steps: int
    tokens_emitted: int
    rows: list[OpRow] = field(default_factory=list)
    total_self_cuda_us: float = 0.0
    total_self_cpu_us: float = 0.0
    total_kernel_launches: int = 0
    extra: dict = field(default_factory=dict)


def _collect_rows(prof: torch.profiler.profile) -> list[OpRow]:
    """Roll up profiler events by op name (averages over invocations)."""
    bucket: dict[str, OpRow] = {}
    for ev in prof.key_averages():
        row = bucket.setdefault(ev.key, OpRow(name=ev.key))
        # In recent torch the units are microseconds.
        row.self_cuda_us += float(ev.self_device_time_total)
        row.self_cpu_us += float(ev.self_cpu_time_total)
        row.cuda_us += float(ev.device_time_total)
        row.cpu_us += float(ev.cpu_time_total)
        row.count += int(ev.count)
    return list(bucket.values())


def _summarize(label: str, prof: torch.profiler.profile, wall: float, steps: int, tokens: int) -> ProfileResult:
    rows = _collect_rows(prof)
    rows.sort(key=lambda r: r.self_cuda_us, reverse=True)
    total_self_cuda = sum(r.self_cuda_us for r in rows)
    total_self_cpu = sum(r.self_cpu_us for r in rows)
    # Heuristic kernel-launch count: events whose name starts "void " or contains
    # "kernel" are device kernels; aten ops that include cuda time launch >=1.
    total_launches = 0
    for r in rows:
        if r.self_cuda_us > 0:
            total_launches += r.count
    return ProfileResult(
        label=label,
        wall_seconds=wall,
        decode_steps=steps,
        tokens_emitted=tokens,
        rows=rows,
        total_self_cuda_us=total_self_cuda,
        total_self_cpu_us=total_self_cpu,
        total_kernel_launches=total_launches,
    )


def run_greedy_profile(
    model: Qwen3Model,
    prompt_ids: torch.Tensor,
    *,
    decode_steps: int,
    warmup: int,
    trace_path: Path | None,
) -> ProfileResult:
    """Profile ``decode_steps`` steady-state greedy decode steps after warmup."""
    cache = model.alloc_cache(prompt_ids.shape[1] + warmup + decode_steps + 8)
    with torch.inference_mode():
        # Prefill (not profiled).
        logits = model(prompt_ids, cache, start_pos=0)
        nt = greedy(logits[:, -1, :]).unsqueeze(0)
        # Warmup decode (not profiled).
        for _ in range(warmup):
            logits = model(nt, cache)
            nt = greedy(logits[:, -1, :]).unsqueeze(0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        ) as prof:
            with record_function("greedy_decode_steady_state"):
                t0 = time.perf_counter()
                for _ in range(decode_steps):
                    logits = model(nt, cache)
                    nt = greedy(logits[:, -1, :]).unsqueeze(0)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                wall = time.perf_counter() - t0

    if trace_path is not None:
        prof.export_chrome_trace(str(trace_path))

    return _summarize("greedy", prof, wall, decode_steps, decode_steps)


def run_spec_profile(
    target: Qwen3Model,
    draft: Qwen3Model,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    K: int,
    warmup_rounds: int,
    trace_path: Path | None,
) -> ProfileResult:
    """Profile a steady-state speculative-decode run after a warmup phase."""
    # Warmup is just a separate full speculative_generate call to populate
    # caches in CUDA's allocator and hit any first-call autotune paths.
    with torch.inference_mode():
        _, _ = speculative_generate(
            target=target,
            draft=draft,
            prompt_ids=prompt_ids,
            max_new_tokens=max(1, warmup_rounds * K),
            K=K,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        ) as prof:
            with record_function("spec_decode_steady_state"):
                t0 = time.perf_counter()
                out, stats = speculative_generate(
                    target=target,
                    draft=draft,
                    prompt_ids=prompt_ids,
                    max_new_tokens=max_new_tokens,
                    K=K,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                wall = time.perf_counter() - t0

    if trace_path is not None:
        prof.export_chrome_trace(str(trace_path))

    result = _summarize("spec", prof, wall, stats.rounds, len(out))
    result.extra = {
        "K": K,
        "rounds": stats.rounds,
        "drafted_tokens": stats.drafted_tokens,
        "accepted_drafted": stats.accepted_drafted,
        "bonus_rounds": stats.bonus_rounds,
        "acceptance_rate": stats.acceptance_rate,
    }
    return result


# ---------- summary writers --------------------------------------------------


def category_breakdown(rows: list[OpRow]) -> list[tuple[str, float, int]]:
    """Return [(category, self_cuda_us, count), ...] over framework ops only."""
    agg: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    for r in rows:
        if not is_framework_op(r):
            continue
        slot = agg[categorize(r.name)]
        slot[0] += r.self_cuda_us
        slot[1] += r.count
    return sorted(((c, t, n) for c, (t, n) in agg.items()), key=lambda x: x[1], reverse=True)


def framework_total_cuda(rows: list[OpRow]) -> float:
    return sum(r.self_cuda_us for r in rows if is_framework_op(r))


def write_json(path: Path, results: list[ProfileResult], gpu_info: dict) -> None:
    payload = {
        "gpu": gpu_info,
        "runs": [],
    }
    for r in results:
        framework_ops = [op for op in r.rows if is_framework_op(op)]
        device_kernels = [op for op in r.rows if is_device_kernel(op)]
        framework_ops.sort(key=lambda x: x.self_cuda_us, reverse=True)
        device_kernels.sort(key=lambda x: x.self_cuda_us, reverse=True)
        payload["runs"].append(
            {
                "label": r.label,
                "wall_seconds": r.wall_seconds,
                "decode_steps": r.decode_steps,
                "tokens_emitted": r.tokens_emitted,
                "tokens_per_second": (r.tokens_emitted / r.wall_seconds) if r.wall_seconds > 0 else 0.0,
                "framework_total_self_cuda_us": framework_total_cuda(r.rows),
                "framework_total_self_cpu_us": sum(op.self_cpu_us for op in framework_ops),
                "kernel_launches_total": sum(op.count for op in device_kernels),
                "extra": r.extra,
                "top_framework_ops": [
                    {
                        "name": op.name,
                        "category": categorize(op.name),
                        "self_cuda_us": op.self_cuda_us,
                        "self_cpu_us": op.self_cpu_us,
                        "count": op.count,
                    }
                    for op in framework_ops[:40]
                ],
                "top_device_kernels": [
                    {
                        "name": op.name,
                        "self_cuda_us": op.self_cuda_us,
                        "count": op.count,
                    }
                    for op in device_kernels[:20]
                ],
                "categories": [
                    {"category": c, "self_cuda_us": t, "count": n}
                    for c, t, n in category_breakdown(r.rows)
                ],
            }
        )
    path.write_text(json.dumps(payload, indent=2))


def _fmt_pct(num: float, denom: float) -> str:
    if denom <= 0:
        return "  -  "
    return f"{100.0 * num / denom:5.1f}%"


def write_markdown(
    path: Path,
    results: list[ProfileResult],
    gpu_info: dict,
    args: argparse.Namespace,
) -> None:
    lines: list[str] = []
    lines.append("# Stage A.5 — Profile summary\n")
    lines.append(f"- GPU: **{gpu_info['name']}** (cap {gpu_info['cap']}, {gpu_info['mem_gb']:.1f} GB)")
    lines.append(f"- torch: {torch.__version__}")
    lines.append(f"- target: `{args.target}`")
    lines.append(f"- draft:  `{args.draft}`")
    lines.append(f"- prompt tokens: {args.prompt_tokens}")
    lines.append(f"- decode steps (greedy): {args.decode_steps}; spec-decode K={args.K}, max_new={args.max_new}")
    lines.append("")

    headline = []
    for r in results:
        tps = (r.tokens_emitted / r.wall_seconds) if r.wall_seconds > 0 else 0.0
        headline.append(f"  - **{r.label}**: {tps:.1f} tok/s ({r.tokens_emitted} tokens in {r.wall_seconds:.2f}s)")
    lines.append("## Headline\n")
    lines.extend(headline)
    lines.append("")

    for r in results:
        lines.append(f"## Run: {r.label}\n")
        if r.label == "spec" and r.extra:
            lines.append(
                f"- rounds={r.extra['rounds']}, accepted={r.extra['accepted_drafted']}/"
                f"{r.extra['drafted_tokens']} ({r.extra['acceptance_rate']:.1%}), "
                f"bonus_rounds={r.extra['bonus_rounds']}"
            )
        framework_ops = [op for op in r.rows if is_framework_op(op)]
        device_kernels = [op for op in r.rows if is_device_kernel(op)]
        framework_ops.sort(key=lambda x: x.self_cuda_us, reverse=True)
        device_kernels.sort(key=lambda x: x.self_cuda_us, reverse=True)
        cuda_total = sum(op.self_cuda_us for op in framework_ops)
        cpu_total = sum(op.self_cpu_us for op in framework_ops)
        kernel_total = sum(op.self_cuda_us for op in device_kernels)
        kernel_count = sum(op.count for op in device_kernels)
        wall_us = r.wall_seconds * 1e6
        lines.append(f"- wall time: {r.wall_seconds:.3f} s")
        lines.append(
            f"- framework-op self-CUDA total: {cuda_total/1e6:.3f} s "
            f"({_fmt_pct(cuda_total, wall_us)} of wall)"
        )
        lines.append(
            f"- device-kernel self-CUDA total: {kernel_total/1e6:.3f} s "
            f"({_fmt_pct(kernel_total, wall_us)} of wall — cross-check)"
        )
        lines.append(f"- framework-op self-CPU total: {cpu_total/1e6:.3f} s")
        if r.decode_steps:
            lines.append(f"- mean wall per step: {wall_us / r.decode_steps:.1f} µs")
            lines.append(
                f"- mean kernel launches per step: "
                f"{kernel_count / max(1, r.decode_steps):.0f}"
            )
        # If the host can't issue kernels fast enough, CUDA is idle while CPU plans.
        # The hint: total self-CPU time of framework ops > total kernel time on device.
        if cpu_total > kernel_total * 1.2 and kernel_total > 0:
            lines.append(
                f"- **launch-bound signal**: framework CPU {cpu_total/1e6:.3f}s > "
                f"device kernels {kernel_total/1e6:.3f}s. Host-side Python/dispatcher "
                f"overhead is on the critical path; CUDA Graphs or a single fused decode-step "
                f"kernel would address this in addition to any per-op speedup."
            )
        lines.append("")

        # Category breakdown — over framework ops only (no double counting)
        lines.append("### Category breakdown over framework ops (self-CUDA, % of wall)\n")
        lines.append("| category | self-CUDA µs | % of wall | events |")
        lines.append("|---|---:|---:|---:|")
        for cat, t, n in category_breakdown(r.rows):
            lines.append(f"| {cat} | {t:.0f} | {_fmt_pct(t, wall_us)} | {n} |")
        lines.append("")

        # Top framework ops
        lines.append("### Top 15 framework ops by self-CUDA time\n")
        lines.append("| op | category | self-CUDA µs | % wall | self-CPU µs | count |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for op in framework_ops[:15]:
            lines.append(
                f"| `{op.name}` | {categorize(op.name)} | {op.self_cuda_us:.0f} | "
                f"{_fmt_pct(op.self_cuda_us, wall_us)} | {op.self_cpu_us:.0f} | {op.count} |"
            )
        lines.append("")

        # Cross-check: top device kernels (CUDA-side names; useful for kernel selection)
        lines.append("### Top 10 device kernels (cross-check)\n")
        lines.append("| kernel | self-CUDA µs | % wall | launches |")
        lines.append("|---|---:|---:|---:|")
        for op in device_kernels[:10]:
            short = op.name if len(op.name) <= 110 else op.name[:107] + "..."
            lines.append(
                f"| `{short}` | {op.self_cuda_us:.0f} | "
                f"{_fmt_pct(op.self_cuda_us, wall_us)} | {op.count} |"
            )
        lines.append("")

    # Recommendation block
    lines.append("## Bottleneck and recommendation\n")
    g = next((r for r in results if r.label == "greedy"), None)
    if g is not None:
        framework_ops = [op for op in g.rows if is_framework_op(op)]
        device_kernels = [op for op in g.rows if is_device_kernel(op)]
        framework_ops.sort(key=lambda x: x.self_cuda_us, reverse=True)
        device_kernels.sort(key=lambda x: x.self_cuda_us, reverse=True)
        wall_us = g.wall_seconds * 1e6
        cpu_total = sum(op.self_cpu_us for op in framework_ops)
        kernel_total = sum(op.self_cuda_us for op in device_kernels)
        cats = category_breakdown(g.rows)
        if cats:
            top_cat, top_t, _ = cats[0]
            pct = 100.0 * top_t / max(1.0, wall_us)
            lines.append(
                f"On the **greedy** decode, the dominant framework category is **{top_cat}** "
                f"at **{pct:.1f}%** of wall time."
            )
            if pct >= 30.0:
                lines.append(
                    f"\nThat clears the >30% bar in DESIGN.md, so the Stage A.5 Triton kernel "
                    f"target is in the **{top_cat}** family."
                )
            else:
                lines.append(
                    f"\nNo single category clears the 30% bar — the workload is fragmented. "
                    f"Re-examine the per-op table; the kernel target may need to fuse multiple "
                    f"ops rather than replace one."
                )
        # Specific GEMV vs attention call-out, since this is the headline story
        # for batch=1 / short-context decode and the most common reader confusion.
        matmul_t = next((t for c, t, _ in cats if c == "matmul"), 0.0)
        attn_t = next((t for c, t, _ in cats if c == "attention_core"), 0.0)
        if matmul_t > 0 and attn_t >= 0:
            lines.append("")
            lines.append(
                f"### GEMV vs attention\n"
                f"- linear projections (`matmul`): **{_fmt_pct(matmul_t, wall_us)}** of wall\n"
                f"- attention core (`flash/sdpa`): **{_fmt_pct(attn_t, wall_us)}** of wall\n"
            )
            if matmul_t > attn_t * 5:
                lines.append(
                    "Linear projections dominate by >5×. At batch=1 / T=1 every linear is a "
                    "GEMV (matrix-vector), and is HBM-bandwidth bound: the weight matrix is "
                    "streamed once per token with ~no reuse. Attention core is cheap because the "
                    "context is short. The right Triton target is a **fused linear/GEMV kernel** "
                    "(e.g., fused QKV or fused gate+up+silu+down for the FFN), not a fused "
                    "attention kernel. Revisit attention only at long context (Stage C sweeps)."
                )
        # Launch-bound check
        if cpu_total > kernel_total * 1.2 and kernel_total > 0:
            lines.append(
                "\n**Launch-bound signal**: framework-op CPU time exceeds total device-kernel "
                "time by >20%. Host-side Python/dispatcher overhead is on the critical path; "
                "CUDA Graphs (replay the decode step) or a single fused decode-step kernel "
                "would attack this. Consider before chasing per-op kernel speedups."
            )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Reproduce / extend\n")
    lines.append("Re-run this profile (regenerates traces and summary):\n")
    lines.append("```bash")
    lines.append(
        f"uv run python -m bench.profile --target {args.target} --draft {args.draft} "
        f"--decode-steps {args.decode_steps} --K {args.K} --max-new {args.max_new}"
    )
    lines.append("```\n")
    lines.append("Kernel-level metrics with Nsight Compute (HBM throughput, L2 hit, occupancy):\n")
    lines.append("```bash")
    lines.append(
        "ncu --set full --target-processes all --launch-skip 200 --launch-count 50 \\\n"
        "    -o experiments/stage_a/ncu_decode -f \\\n"
        f"    uv run python -m bench.profile --target {args.target} --draft {args.draft} "
        f"--decode-steps 80 --skip-spec"
    )
    lines.append("```\n")
    lines.append("View Chrome traces: open `chrome://tracing` and load `torch_trace_*.json`.")

    path.write_text("\n".join(lines))


# ---------- entry point ------------------------------------------------------


def _autodevice() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"name": "cpu", "cap": "n/a", "mem_gb": 0.0}
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "cap": f"{props.major}.{props.minor}",
        "mem_gb": props.total_memory / (1024**3),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=str, default="Qwen/Qwen3-4B")
    p.add_argument("--draft", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    p.add_argument("--decode-steps", type=int, default=200, help="Greedy decode steps to profile.")
    p.add_argument("--max-new", type=int, default=200, help="Max new tokens for spec-decode profile.")
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--warmup", type=int, default=10, help="Greedy warmup decode steps before profiling.")
    p.add_argument("--warmup-rounds", type=int, default=3, help="Spec-decode warmup rounds before profiling.")
    p.add_argument("--out", type=Path, default=Path("experiments/stage_a"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--skip-greedy", action="store_true")
    p.add_argument("--skip-spec", action="store_true")
    p.add_argument(
        "--use-triton", action="store_true",
        help="Enable Triton kernels (Stage A.5a: fused gate-up-silu).",
    )
    args = p.parse_args()

    device = _autodevice() if args.device == "auto" else torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    args.out.mkdir(parents=True, exist_ok=True)

    def _load(model_id: str) -> Qwen3Model:
        # Scope the staged state-dict tensors locally so the duplicate GPU copy
        # is freed before the next model loads (same pattern as bench/demo.py).
        loaded = load_weights(model_id, dtype=dtype, device=device)
        model = (
            Qwen3Model.from_loaded(loaded).to(dtype=dtype, device=device).eval()
        )
        if args.use_triton:
            from kernels import apply_triton_kernels

            apply_triton_kernels(model)
        return model

    print(f"[profile] device={device} dtype={args.dtype}")
    print(f"[profile] loading target {args.target} ...")
    target = _load(args.target)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.target == args.draft or args.skip_spec:
        draft = target
    else:
        print(f"[profile] loading draft  {args.draft} ...")
        draft = _load(args.draft)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tok = AutoTokenizer.from_pretrained(args.target)
    prompt_ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)
    args.prompt_tokens = int(prompt_ids.shape[1])
    print(f"[profile] prompt tokens: {args.prompt_tokens}")

    results: list[ProfileResult] = []

    if not args.skip_greedy:
        print(f"[profile] greedy: warmup={args.warmup}, decode_steps={args.decode_steps} ...")
        gres = run_greedy_profile(
            target,
            prompt_ids,
            decode_steps=args.decode_steps,
            warmup=args.warmup,
            trace_path=args.out / "torch_trace_greedy.json",
        )
        print(
            f"[profile] greedy: {gres.tokens_emitted} tokens in {gres.wall_seconds:.2f}s "
            f"= {gres.tokens_emitted / gres.wall_seconds:.1f} tok/s"
        )
        results.append(gres)

    if not args.skip_spec:
        print(f"[profile] spec:   K={args.K}, max_new={args.max_new}, warmup_rounds={args.warmup_rounds} ...")
        sres = run_spec_profile(
            target,
            draft,
            prompt_ids,
            max_new_tokens=args.max_new,
            K=args.K,
            warmup_rounds=args.warmup_rounds,
            trace_path=args.out / "torch_trace_spec.json",
        )
        print(
            f"[profile] spec:   {sres.tokens_emitted} tokens in {sres.wall_seconds:.2f}s "
            f"= {sres.tokens_emitted / sres.wall_seconds:.1f} tok/s "
            f"(acc={sres.extra['acceptance_rate']:.1%})"
        )
        results.append(sres)

    gpu_info = _gpu_info()
    write_json(args.out / "profile_data.json", results, gpu_info)
    write_markdown(args.out / "profile_summary.md", results, gpu_info, args)
    print(f"[profile] wrote {args.out / 'profile_summary.md'}")
    print(f"[profile] wrote {args.out / 'profile_data.json'}")


if __name__ == "__main__":
    main()
