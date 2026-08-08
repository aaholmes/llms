"""Stage C — the joint sweep at a fixed VRAM ceiling.

The study the whole pipeline was built to run: how KV compression and
speculative decoding interact, across the four regimes in DESIGN.md.

  1. ``baseline``      — both models uncompressed. Reference acceptance/throughput.
  2. ``target-only``   — target compressed, draft not. The naive application: the
                         draft was calibrated against the *original* target, so
                         post-conversion logit drift may lower agreement and
                         partly offset the cheaper KV reads.
  3. ``coupled``       — both compressed, same pipeline and calibration data, so
                         the two distortions are *correlated*. May preserve or
                         even improve draft–target alignment relative to (2).
  4. ``specialized``   — small dense draft with a LoRA TVD-tuned against the
                         *compressed* target. Align the draft instead of
                         shrinking it.

(3) vs (4) is the question: is it better to **shrink** the draft or **align** it?

The sign of every interaction is a priori unclear and "no significant
interaction" is a real result, so this driver reports every cell it ran and
every cell it skipped — it never silently narrows the grid.

Each cell emits throughput, speedup vs plain greedy, acceptance, perplexity,
and peak VRAM, appended to ``results.jsonl`` as it completes. Re-running
resumes: cells already present are skipped unless ``--force``.

Usage (GPU desktop):
    python -m experiments.stage_c.sweep --dry-run      # validate artifacts, print plan
    python -m experiments.stage_c.sweep --out experiments/stage_c/run1
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Must precede torch's CUDA init — many per-prompt cache alloc/free cycles
# fragment the allocator otherwise.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO = Path(__file__).resolve().parents[2]
SB = REPO / "experiments" / "stage_b"

# `python -m experiments.stage_c.sweep` from the repo root doesn't put src/ on
# the path the way pytest's pythonpath setting does; do it here so the driver
# runs from a bare checkout without an env var.
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import torch

TARGET_ID = "Qwen/Qwen3-4B"
DRAFT_ID = "Qwen/Qwen3-0.6B"
LORA_RANK, LORA_ALPHA = 16, 32.0


# --------------------------------------------------------------------------
# Grid definition
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """How to build one model. ``dense`` needs nothing but the id.

    ``artifact``/``heal_dir`` are repo-relative to ``experiments/stage_b``.
    ``kind``:
      - ``dense``      — plain HF weights.
      - ``mla``        — dense → apply_mla(artifact) → wrap_lora → load heal_dir/best.pt
      - ``lora-merged``— dense → wrap_lora → load heal_dir/best.pt → merge (draft-tuning)
    """

    kind: str
    model_id: str
    artifact: str | None = None
    heal_dir: str | None = None
    label: str = ""

    def required_paths(self) -> list[Path]:
        paths = []
        if self.artifact:
            paths.append(SB / self.artifact)
        if self.heal_dir:
            paths.append(SB / self.heal_dir / "best.pt")
        return paths

    def describe(self) -> str:
        return self.label or self.kind


@dataclass(frozen=True)
class Regime:
    name: str
    target: ModelSpec
    draft: ModelSpec
    compression: float  # nominal KV compression of the *target*
    note: str = ""


DENSE_TARGET = ModelSpec("dense", TARGET_ID, label="Qwen3-4B dense")
DENSE_DRAFT = ModelSpec("dense", DRAFT_ID, label="Qwen3-0.6B dense")


def _mla_target(rank: int, drope: int, compression: float) -> ModelSpec:
    stem = f"r{rank}_drope{drope}"
    return ModelSpec("mla", TARGET_ID, f"{stem}.pt", f"heal_{stem}", f"MLA {stem} ({compression:.2f}×)")


def default_regimes(compressions: list[tuple[int, int, float]]) -> list[Regime]:
    """Build the four-regime grid over the requested target compression points.

    ``compressions`` is a list of ``(rank, d_rope, nominal_compression)`` — the
    healed operating points from Stage B's frontier.
    """
    regimes = [
        Regime("baseline", DENSE_TARGET, DENSE_DRAFT, 1.0,
               "reference: neither model compressed"),
    ]
    for rank, drope, comp in compressions:
        tgt = _mla_target(rank, drope, comp)
        stem = f"r{rank}_drope{drope}"
        regimes.append(Regime("target-only", tgt, DENSE_DRAFT, comp,
                              "draft calibrated against the *original* target"))
        regimes.append(Regime(
            "coupled", tgt,
            # Same conversion pipeline and calibration data applied to the draft,
            # so its distortion is correlated with the target's.
            ModelSpec("mla", DRAFT_ID, f"draft_{stem}.pt", f"heal_draft_{stem}",
                      f"MLA draft {stem}"),
            comp, "both compressed — correlated distortion"))
        regimes.append(Regime(
            "specialized", tgt,
            ModelSpec("lora-merged", DRAFT_ID, None, f"heal_draft_0_6b_tvd_{stem}",
                      f"TVD-tuned draft → {stem}"),
            comp, "draft LoRA TVD-tuned against the *compressed* target"))
    return regimes


@dataclass(frozen=True)
class Cell:
    regime: str
    target_label: str
    draft_label: str
    compression: float
    K: int

    def key(self) -> str:
        return f"{self.regime}|{self.target_label}|{self.draft_label}|K{self.K}"


def build_plan(regimes: list[Regime], ks: list[int]) -> list[tuple[Regime, Cell]]:
    return [
        (r, Cell(r.name, r.target.describe(), r.draft.describe(), r.compression, K))
        for r in regimes
        for K in ks
    ]


def missing_artifacts(regime: Regime) -> list[Path]:
    return [p for p in regime.target.required_paths() + regime.draft.required_paths() if not p.exists()]


def load_done_keys(results_path: Path) -> set[str]:
    """Keys already recorded, so a re-run resumes instead of redoing GPU hours."""
    if not results_path.exists():
        return set()
    done = set()
    for line in results_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(json.loads(line)["key"])
        except (json.JSONDecodeError, KeyError):
            # A partially-written final line from a killed run: ignore it and
            # let the cell re-run rather than trusting a truncated record.
            continue
    return done


# --------------------------------------------------------------------------
# Environment capture (reproducibility bar: an external reviewer reproduces
# the central plots from published code in under a day)
# --------------------------------------------------------------------------


def _git_rev() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, stderr=subprocess.DEVNULL
        )
        return bool(out.decode().strip())
    except Exception:
        return False


def capture_environment() -> dict:
    env = {
        "git_rev": _git_rev(),
        "git_dirty": _git_dirty(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "argv": sys.argv[1:],
    }
    if torch.cuda.is_available():
        env["cuda"] = torch.version.cuda
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
    return env


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------


def build_model(spec: ModelSpec, *, device: str, dtype: torch.dtype):
    from engine.model import Qwen3Model
    from engine.weights import load_weights
    from mla.heal import load_trainable, merge_lora, wrap_lora
    from mla.swap import apply_mla

    loaded = load_weights(spec.model_id, dtype=dtype, device="cpu")
    model = Qwen3Model.from_loaded(loaded).to(dtype=dtype, device=device)
    del loaded
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if spec.kind == "dense":
        return model.eval()

    if spec.kind == "mla":
        artifact = torch.load(SB / spec.artifact, map_location="cpu", weights_only=False)
        apply_mla(model, artifact)
        wrap_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA)
        missing = load_trainable(model, SB / spec.heal_dir / "best.pt")
        if missing:
            print(f"  WARN unmatched trainable in {spec.heal_dir}: {missing[:4]}", flush=True)
        # Merged so the healed model runs at plain-MLA speed — any throughput
        # difference between regimes is the architecture, not adapter overhead.
        merge_lora(model)
        return model.eval()

    if spec.kind == "lora-merged":
        wrap_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA)
        missing = load_trainable(model, SB / spec.heal_dir / "best.pt")
        if missing:
            print(f"  WARN unmatched trainable in {spec.heal_dir}: {missing[:4]}", flush=True)
        merge_lora(model)
        return model.eval()

    raise ValueError(f"unknown ModelSpec.kind={spec.kind!r}")


def _alloc_cache(model, max_seq_len: int):
    from engine.mla import MLAttention
    from mla.swap import alloc_mla_cache

    if any(isinstance(m, MLAttention) for m in model.modules()):
        return alloc_mla_cache(model, max_seq_len)
    return model.alloc_cache(max_seq_len)


def kv_bytes_at(model, context_len: int) -> int:
    """KV-cache footprint at ``context_len`` — the quantity the 16 GB ceiling binds on."""
    cache = _alloc_cache(model, context_len)
    total = 0
    for attr in ("k", "v", "kv", "c_kv", "k_rope"):
        tensors = getattr(cache, attr, None)
        if tensors is None:
            continue
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]
        total += sum(t.numel() * t.element_size() for t in tensors)
    del cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return total


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.inference_mode()
def _greedy(model, ids, max_new: int, device: str) -> None:
    cache = _alloc_cache(model, ids.shape[1] + max_new + 4)
    logits = model(ids.to(device), cache, start_pos=0)
    tok = int(logits[0, -1].argmax())
    pos = ids.shape[1]
    for _ in range(max_new - 1):
        logits = model(torch.tensor([[tok]], device=device), cache, start_pos=pos)
        pos += 1
        tok = int(logits[0, -1].argmax())


def bench_greedy(model, prompts, *, max_new: int, device: str) -> float:
    _greedy(model, prompts[0][:, :8], 4, device)  # warmup
    walls = []
    for p in prompts:
        _sync()
        t0 = time.time()
        _greedy(model, p, max_new, device)
        _sync()
        walls.append(time.time() - t0)
    return max_new / statistics.median(walls)


@torch.inference_mode()  # else the forward graph is retained → activation blowup → OOM
def bench_spec(target, draft, prompts, K: int, *, max_new: int, device: str) -> tuple[float, float]:
    from engine.spec_decode import speculative_generate

    speculative_generate(target=target, draft=draft,
                         prompt_ids=prompts[0][:, :8].to(device),
                         max_new_tokens=4, K=K)  # warmup
    walls, accs = [], []
    for p in prompts:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        p = p.to(device)
        _sync()
        t0 = time.time()
        _, stats = speculative_generate(target=target, draft=draft, prompt_ids=p,
                                        max_new_tokens=max_new, K=K)
        _sync()
        walls.append(time.time() - t0)
        accs.append(stats.acceptance_rate)
    return max_new / statistics.median(walls), statistics.median(accs)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def parse_compressions(raw: list[str]) -> list[tuple[int, int, float]]:
    """``"256:32:4.0"`` → ``(256, 32, 4.0)``."""
    out = []
    for item in raw:
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"--compression expects rank:d_rope:ratio, got {item!r}")
        out.append((int(parts[0]), int(parts[1]), float(parts[2])))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="experiments/stage_c/run1", help="output directory")
    ap.add_argument("--corpus", default="wikitext", choices=["wikitext", "code"])
    ap.add_argument("--compression", nargs="+", default=["512:128:1.33", "256:32:4.0", "128:32:5.33"],
                    help="target operating points as rank:d_rope:nominal_ratio")
    ap.add_argument("--K", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 8],
                    help="draft depths to sweep")
    ap.add_argument("--regimes", nargs="+", default=None,
                    help="subset of {baseline,target-only,coupled,specialized}")
    ap.add_argument("--n-prompts", type=int, default=24)
    ap.add_argument("--prompt-tok", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--ppl-chunks", type=int, default=256)
    ap.add_argument("--ppl-chunk-tokens", type=int, default=1024)
    ap.add_argument("--context-len", type=int, default=32768,
                    help="context length at which the KV budget is accounted")
    ap.add_argument("--vram-budget-gb", type=float, default=16.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-ppl", action="store_true", help="throughput only (faster smoke run)")
    ap.add_argument("--force", action="store_true", help="ignore existing results and redo every cell")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate artifacts and print the cell plan without loading models")
    args = ap.parse_args(argv)

    regimes = default_regimes(parse_compressions(args.compression))
    if args.regimes:
        wanted = set(args.regimes)
        unknown = wanted - {r.name for r in regimes}
        if unknown:
            ap.error(f"unknown regime(s): {sorted(unknown)}")
        regimes = [r for r in regimes if r.name in wanted]

    out_dir = Path(args.out)
    results_path = out_dir / "results.jsonl"

    # --- artifact validation, before anything expensive -------------------
    runnable, skipped = [], []
    for r in regimes:
        missing = missing_artifacts(r)
        (skipped if missing else runnable).append((r, missing))

    print(f"[stage-c] {len(runnable)} regime(s) runnable, {len(skipped)} skipped", flush=True)
    for r, missing in skipped:
        # Explicit, never silent — a skipped regime is a hole in the grid and
        # must read as one in the log and the summary.
        print(f"[stage-c] SKIP regime={r.name} comp={r.compression:.2f}× "
              f"({r.draft.describe()}) — missing: {[str(p.relative_to(REPO)) for p in missing]}",
              flush=True)

    plan = build_plan([r for r, _ in runnable], args.K)
    done = set() if args.force else load_done_keys(results_path)
    todo = [(r, c) for r, c in plan if c.key() not in done]
    print(f"[stage-c] {len(plan)} cells planned, {len(plan) - len(todo)} already done, "
          f"{len(todo)} to run", flush=True)

    if args.dry_run:
        for r, c in plan:
            mark = "done" if c.key() in done else "todo"
            print(f"  [{mark}] {c.key()}")
        if skipped:
            print(f"\n{len(skipped)} regime(s) cannot run — convert/heal the missing "
                  f"artifacts on the GPU box first.")
        return 0

    if not torch.cuda.is_available():
        ap.error("no CUDA device: the sweep is a GPU-desktop run. Use --dry-run to validate the plan.")

    out_dir.mkdir(parents=True, exist_ok=True)
    set_seeds(args.seed)
    env = capture_environment()
    env["args"] = vars(args)
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2, default=str))
    if env["git_dirty"]:
        print("[stage-c] WARNING: working tree is dirty — results will not be "
              "reproducible from git_rev alone", flush=True)

    device, dtype = "cuda", torch.bfloat16

    # --- workload ---------------------------------------------------------
    from mla.calibrate import load_wikitext103_chunks

    if args.corpus == "code":
        from mla.calibrate import load_code_chunks
        chunks = load_code_chunks(n_samples=args.n_prompts + 8, chunk_tokens=args.prompt_tok,
                                  tokenizer_id=TARGET_ID, skip=20000)
    else:
        chunks = load_wikitext103_chunks(n_samples=args.n_prompts, chunk_tokens=args.prompt_tok,
                                         tokenizer_id=TARGET_ID, split="validation")
    prompts = [c[:, : args.prompt_tok] for c in chunks][: args.n_prompts]

    ppl_chunks = []
    if not args.skip_ppl:
        ppl_chunks = load_wikitext103_chunks(
            n_samples=args.ppl_chunks, chunk_tokens=args.ppl_chunk_tokens,
            tokenizer_id=TARGET_ID, split="validation")
    print(f"[stage-c] {len(prompts)} {args.corpus} prompts × {args.prompt_tok}→{args.max_new} tok; "
          f"{len(ppl_chunks)} ppl chunks", flush=True)

    # --- group cells by target so the 4B model is loaded once per config ---
    from bench.eval_ppl import evaluate_ppl

    by_target: dict[str, list[tuple[Regime, Cell]]] = {}
    for r, c in todo:
        by_target.setdefault(r.target.describe(), []).append((r, c))

    results_fh = results_path.open("a")
    n_ok = n_fail = 0
    for target_label, group in by_target.items():
        target_spec = group[0][0].target
        print(f"\n[stage-c] === target: {target_label} ===", flush=True)
        try:
            target = build_model(target_spec, device=device, dtype=dtype)
        except Exception as exc:  # a bad artifact shouldn't abandon the rest of the grid
            print(f"[stage-c] FAILED to build target {target_label}: {exc}", flush=True)
            n_fail += len(group)
            continue

        torch.cuda.reset_peak_memory_stats()
        target_kv_bytes = kv_bytes_at(target, args.context_len)
        greedy_tps = bench_greedy(target, prompts, max_new=args.max_new, device=device)
        target_ppl = None
        if ppl_chunks:
            target_ppl = evaluate_ppl(target, ppl_chunks, progress_every=None)["ppl"]
        print(f"[stage-c] greedy={greedy_tps:.1f} tok/s  ppl={target_ppl}  "
              f"KV@{args.context_len}={target_kv_bytes/2**30:.2f} GiB", flush=True)

        by_draft: dict[str, list[tuple[Regime, Cell]]] = {}
        for r, c in group:
            by_draft.setdefault(r.draft.describe(), []).append((r, c))

        for draft_label, dgroup in by_draft.items():
            draft_spec = dgroup[0][0].draft
            try:
                draft = build_model(draft_spec, device=device, dtype=dtype)
            except Exception as exc:
                print(f"[stage-c] FAILED to build draft {draft_label}: {exc}", flush=True)
                n_fail += len(dgroup)
                continue

            for regime, cell in dgroup:
                try:
                    tps, acc = bench_spec(target, draft, prompts, cell.K,
                                          max_new=args.max_new, device=device)
                except Exception as exc:
                    print(f"[stage-c] FAILED {cell.key()}: {exc}", flush=True)
                    n_fail += 1
                    continue
                peak = torch.cuda.max_memory_allocated()
                rec = {
                    "key": cell.key(),
                    **asdict(cell),
                    "note": regime.note,
                    "tok_s": tps,
                    "greedy_tok_s": greedy_tps,
                    "speedup": tps / greedy_tps,
                    "acceptance": acc,
                    "ppl": target_ppl,
                    "kv_bytes_at_context": target_kv_bytes,
                    "context_len": args.context_len,
                    "peak_vram_bytes": peak,
                    "within_budget": peak <= args.vram_budget_gb * 2**30,
                    "seed": args.seed,
                    "corpus": args.corpus,
                }
                results_fh.write(json.dumps(rec) + "\n")
                results_fh.flush()  # a killed run keeps every completed cell
                n_ok += 1
                print(f"[stage-c] {cell.regime:12s} {draft_label:24s} K={cell.K}  "
                      f"{tps:7.1f} tok/s  {tps/greedy_tps:.2f}×  acc={acc*100:5.1f}%  "
                      f"peak={peak/2**30:.1f} GiB", flush=True)

            del draft
            torch.cuda.empty_cache()

        del target
        torch.cuda.empty_cache()

    results_fh.close()
    print(f"\n[stage-c] {n_ok} cells completed, {n_fail} failed, "
          f"{len(skipped)} regime(s) skipped for missing artifacts", flush=True)
    write_summary(out_dir, skipped=skipped, n_ok=n_ok, n_fail=n_fail)
    return 0 if n_fail == 0 else 1


def write_summary(out_dir: Path, *, skipped: list, n_ok: int, n_fail: int) -> None:
    """Markdown summary from the accumulated JSONL (includes prior resumed runs)."""
    results_path = out_dir / "results.jsonl"
    rows = [json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]
    if not rows:
        return
    rows.sort(key=lambda r: (r["regime"], r["compression"], r["K"]))

    md = ["# Stage C — joint sweep at fixed VRAM\n",
          f"{n_ok} cells this run, {n_fail} failed, {len(rows)} total on file.\n",
          "| Regime | Target | Draft | Comp | K | tok/s | speedup | acceptance | PPL | peak VRAM |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        ppl = f"{r['ppl']:.2f}" if r.get("ppl") is not None else "—"
        md.append(
            f"| {r['regime']} | {r['target_label']} | {r['draft_label']} | "
            f"{r['compression']:.2f}× | {r['K']} | {r['tok_s']:.1f} | {r['speedup']:.2f}× | "
            f"{r['acceptance']*100:.1f}% | {ppl} | {r['peak_vram_bytes']/2**30:.1f} GiB |")

    # Best-K per regime × compression — the row that goes in the writeup.
    md += ["\n## Best-K per regime\n",
           "| Regime | Comp | best K | speedup | acceptance |", "|---|---:|---:|---:|---:|"]
    groups: dict[tuple[str, float], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["regime"], r["compression"]), []).append(r)
    for (regime, comp), rs in sorted(groups.items()):
        b = max(rs, key=lambda r: r["tok_s"])
        md.append(f"| {regime} | {comp:.2f}× | {b['K']} | {b['speedup']:.2f}× | "
                  f"{b['acceptance']*100:.1f}% |")

    if skipped:
        md.append("\n## Not run\n")
        md.append("These regimes were skipped for missing artifacts — the grid has holes here:\n")
        for r, missing in skipped:
            md.append(f"- `{r.name}` @ {r.compression:.2f}× — missing "
                      f"{', '.join(str(p.name) for p in missing)}")

    over = [r for r in rows if not r.get("within_budget", True)]
    if over:
        md.append(f"\n**{len(over)} cell(s) exceeded the VRAM budget** and are not "
                  f"deployable at the stated ceiling.")

    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    print(f"[stage-c] wrote {out_dir/'summary.md'}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
