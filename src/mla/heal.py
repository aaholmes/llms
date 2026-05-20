"""Healing finetune for a post-hoc MHA→MLA-converted Qwen3 model.

Stage B's joint-SVD conversion produces a model with the right architecture
but degraded perplexity (e.g. +826% on Qwen3-4B at 4× KV compression, no FT).
Published recipes (MHA2MLA, TransMLA) close this gap with full-parameter LM-
loss finetuning over ~0.3–0.6% of pretraining tokens. We can't fit full FT of
a 4B model in 16 GB, so the recipe here is **hybrid**: train the new MLA
projections (``dkv``, ``uk_nope``, ``uv``, ``kr``) fully — they're SVD-init'd
and need to *move*, not just receive a low-rank delta — and wrap the rest of
the linears (Q, O, gate, up, down) with LoRA.

Reuses the existing pipeline:
  - ``mla.swap.apply_mla`` to load a converted artifact onto Qwen3Model
  - ``mla.calibrate.load_wikitext103_chunks`` for train and validation data
  - ``bench.eval_ppl.evaluate_ppl`` as the validation hook

CLI:
    python -m mla.heal \\
        --mla-artifact experiments/stage_b/qwen_qwen3_4b_r256_drope32.pt \\
        --data wt103 --split-seed 17 \\
        --steps 25000 --seq-len 1024 --micro-batch 1 --grad-accum 32 \\
        --lora-rank 16 --lora-alpha 32 \\
        --lr-mla 5e-5 --lr-lora 1e-4 --warmup-frac 0.03 \\
        --val-every 200 --val-chunks 250 \\
        --out experiments/stage_b/heal_r256_drope32
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from bench.eval_ppl import evaluate_ppl
from engine.mla import MLAttention
from engine.model import Qwen3Model
from engine.weights import load_weights
from mla.calibrate import load_wikitext103_chunks
from mla.swap import alloc_mla_cache, apply_mla


# --- LoRA wrapper ---------------------------------------------------------


class LoraLinear(nn.Module):
    """nn.Linear + low-rank additive update y = base(x) + (B @ A @ x) * scaling.

    ``base`` is held as a child module and frozen by ``setup_trainable``.
    ``lora_A`` is kaiming-uniform init; ``lora_B`` is zero-init so at step 0
    the wrapped layer is bit-identical to the original.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if base.bias is not None:
            raise ValueError("LoraLinear assumes bias=False (Qwen3 convention)")
        in_features = base.in_features
        out_features = base.out_features
        self.base = base
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        # Standard LoRA init: A kaiming, B zeros.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        # Match base dtype/device.
        self.lora_A.to(dtype=base.weight.dtype, device=base.weight.device)
        self.lora_B.to(dtype=base.weight.dtype, device=base.weight.device)
        self.scaling = alpha / rank
        self.rank = rank
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scaling


# Leaf names to wrap. Matches Qwen3 decoder block + (post-MLA) attention:
# ``layers.i.attn.q/o`` and ``layers.i.ffn.gate/up/down``. The MLA-specific
# projections (``dkv``, ``uk_nope``, ``uv``, ``kr``) are intentionally absent —
# they're trained fully, not via LoRA delta.
DEFAULT_LORA_TARGETS: tuple[str, ...] = ("q", "o", "gate", "up", "down")


def wrap_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    target_leaf_names: tuple[str, ...] = DEFAULT_LORA_TARGETS,
) -> int:
    """Replace nn.Linear children whose attribute name is in ``target_leaf_names`` with LoraLinear.

    Walks all submodules; for each one, iterates over named children and
    swaps Linears whose attribute name matches. Matching is by exact leaf
    name (the last component of the qualified path), so 'up' matches
    ``ffn.up`` but not ``attn.uk_nope`` and not ``attn.uv``.

    Returns the number of layers wrapped.
    """
    wrapped = 0
    targets = set(target_leaf_names)
    for parent in model.modules():
        for child_name, child in list(parent.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(parent, child_name, LoraLinear(child, rank=rank, alpha=alpha))
                wrapped += 1
    return wrapped


# --- Trainable-parameter setup --------------------------------------------


@dataclass
class TrainableInventory:
    mla_proj_params: list[nn.Parameter]
    lora_params: list[nn.Parameter]

    @property
    def n_mla_proj(self) -> int:
        return sum(p.numel() for p in self.mla_proj_params)

    @property
    def n_lora(self) -> int:
        return sum(p.numel() for p in self.lora_params)

    @property
    def n_total(self) -> int:
        return self.n_mla_proj + self.n_lora


def setup_trainable(model: nn.Module) -> TrainableInventory:
    """Freeze the whole model, then unfreeze MLA projections and LoRA params.

    Trainable scope:
      - ``MLAttention.{dkv, uk_nope, uv, kr}.weight`` (every MLA layer)
      - ``LoraLinear.{lora_A, lora_B}.weight`` (every wrapped Linear)

    Everything else stays frozen: embeddings, RMSNorms, base Q/O/MLP weights
    underneath LoRA adapters, the lm_head (if untied), and the per-head
    QK-norm weights.

    Returns the inventory so the caller can build optimizer param groups.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    mla_proj_params: list[nn.Parameter] = []
    lora_params: list[nn.Parameter] = []

    for module in model.modules():
        if isinstance(module, MLAttention):
            for proj in (module.dkv, module.uk_nope, module.uv, module.kr):
                if proj is not None:
                    proj.weight.requires_grad_(True)
                    mla_proj_params.append(proj.weight)
        elif isinstance(module, LoraLinear):
            module.lora_A.weight.requires_grad_(True)
            module.lora_B.weight.requires_grad_(True)
            lora_params.append(module.lora_A.weight)
            lora_params.append(module.lora_B.weight)

    return TrainableInventory(mla_proj_params=mla_proj_params, lora_params=lora_params)


def build_optimizer(
    inventory: TrainableInventory,
    *,
    lr_mla: float,
    lr_lora: float,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.95),
) -> torch.optim.Optimizer:
    """AdamW with two param groups; stores ``_initial_lr`` for the scheduler."""
    groups = []
    if inventory.mla_proj_params:
        groups.append({
            "params": inventory.mla_proj_params,
            "lr": lr_mla,
            "_initial_lr": lr_mla,
            "name": "mla_proj",
        })
    if inventory.lora_params:
        groups.append({
            "params": inventory.lora_params,
            "lr": lr_lora,
            "_initial_lr": lr_lora,
            "name": "lora",
        })
    if not groups:
        raise ValueError("no trainable parameters — did setup_trainable run?")
    return torch.optim.AdamW(groups, betas=betas, weight_decay=weight_decay)


# --- Scheduler ------------------------------------------------------------


def cosine_warmup_multiplier(step: int, total_steps: int, warmup_frac: float) -> float:
    """LR multiplier in [0, 1]. Linear warmup → cosine decay to 0."""
    warmup_steps = max(1, int(warmup_frac * total_steps))
    if step < warmup_steps:
        return (step + 1) / warmup_steps  # +1 so step 0 already has nonzero lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def apply_lr_multiplier(optim: torch.optim.Optimizer, mult: float) -> None:
    for pg in optim.param_groups:
        pg["lr"] = pg["_initial_lr"] * mult


# --- Checkpoint I/O -------------------------------------------------------


def save_trainable(model: nn.Module, path: Path | str) -> None:
    """Save only requires_grad=True params (trainable scope)."""
    state = {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters() if p.requires_grad
    }
    torch.save({"trainable_state": state}, Path(path))


def load_trainable(model: nn.Module, path: Path | str) -> list[str]:
    """Restore trainable params from save_trainable. Returns list of unmatched names."""
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    state = ckpt["trainable_state"]
    name_to_param = dict(model.named_parameters())
    missing: list[str] = []
    for name, t in state.items():
        if name not in name_to_param:
            missing.append(name)
            continue
        p = name_to_param[name]
        with torch.no_grad():
            p.copy_(t.to(dtype=p.dtype, device=p.device))
    return missing


# --- Training loop --------------------------------------------------------


def chunk_ce_loss(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Shift-by-one cross-entropy in fp32, mean over predicted positions.

    logits: (B, T, V); ids: (B, T) long. Returns scalar loss with grad.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = ids[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.float().view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


def train_loop(
    model: Qwen3Model,
    optim: torch.optim.Optimizer,
    inventory: TrainableInventory,
    train_chunks: list[torch.Tensor],
    val_chunks: list[torch.Tensor],
    *,
    steps: int,
    grad_accum: int,
    seq_len: int,
    warmup_frac: float,
    val_every: int,
    log_every: int,
    out_dir: Path,
    grad_clip: float = 1.0,
    seed: int = 0,
    progress_writer=None,
) -> dict:
    """Train ``model`` for ``steps`` optimizer steps. Returns final summary dict."""
    device = next(model.parameters()).device
    log_path = out_dir / "training_log.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator().manual_seed(seed)
    all_trainable = inventory.mla_proj_params + inventory.lora_params

    best_val_ppl = float("inf")
    best_step = -1
    t0 = time.time()

    with log_path.open("w") as log_f:
        for step in range(steps):
            mult = cosine_warmup_multiplier(step, steps, warmup_frac)
            apply_lr_multiplier(optim, mult)

            optim.zero_grad(set_to_none=True)
            step_loss = 0.0

            for _ in range(grad_accum):
                idx = int(torch.randint(0, len(train_chunks), (1,), generator=rng).item())
                ids = train_chunks[idx].to(device)
                # Tight cache: one chunk's worth, freed at end of iter.
                cache = alloc_mla_cache(model, max_seq_len=ids.shape[1])
                logits = model(ids, cache, start_pos=0)
                loss = chunk_ce_loss(logits, ids)
                (loss / grad_accum).backward()
                step_loss += float(loss.item()) / grad_accum
                del cache, logits, loss

            grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=grad_clip)
            optim.step()

            entry = {
                "step": step,
                "train_loss": step_loss,
                "grad_norm": float(grad_norm.item()),
                "lr_mla": optim.param_groups[0]["lr"],
                "lr_lora": (
                    optim.param_groups[1]["lr"] if len(optim.param_groups) > 1 else None
                ),
                "wall_seconds": time.time() - t0,
            }

            do_val = (step + 1) % val_every == 0 or step == steps - 1
            if do_val:
                model.eval()
                val = evaluate_ppl(model, val_chunks)
                model.train()
                entry["val_ppl"] = val["ppl"]
                entry["val_tokens"] = val["token_count"]
                if val["ppl"] < best_val_ppl:
                    best_val_ppl = val["ppl"]
                    best_step = step
                    save_trainable(model, out_dir / "best.pt")
                    entry["best"] = True

            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()
            if progress_writer is not None and (step + 1) % log_every == 0:
                msg = (
                    f"[heal] step {step+1}/{steps} "
                    f"loss={step_loss:.4f} grad_norm={float(grad_norm):.2f}"
                )
                if "val_ppl" in entry:
                    msg += f" val_ppl={entry['val_ppl']:.4f}"
                progress_writer(msg)

    return {
        "best_val_ppl": best_val_ppl,
        "best_step": best_step,
        "wall_seconds": time.time() - t0,
        "n_trainable_total": inventory.n_total,
        "n_trainable_mla_proj": inventory.n_mla_proj,
        "n_trainable_lora": inventory.n_lora,
    }


# --- CLI ------------------------------------------------------------------


def _git_rev() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _write_summary(out_dir: Path, summary: dict) -> None:
    summary_path = out_dir / "summary.md"
    lines = ["# Healing finetune summary\n"]
    for k, v in summary.items():
        lines.append(f"- **{k}**: {v}")
    summary_path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--mla-artifact", required=True,
        help="path to the converted MLA artifact (.pt) produced by mla.convert",
    )
    ap.add_argument("--out", required=True, help="output dir for logs and checkpoints")
    ap.add_argument(
        "--data", default="wt103", choices=["wt103"],
        help="training data source (only wt103 for now)",
    )
    ap.add_argument("--split-seed", type=int, default=17, help="train-split sampling seed")
    ap.add_argument("--steps", type=int, default=3000, help="optimizer steps")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--micro-batch", type=int, default=1, help="(unused for now — micro_batch=1)")
    ap.add_argument("--grad-accum", type=int, default=32)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--lr-mla", type=float, default=5e-5)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--val-chunks", type=int, default=250)
    ap.add_argument("--train-chunks", type=int, default=20000,
                    help="how many unique training chunks to load (cycled by random sampling)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
        help="model + activation dtype",
    )
    ap.add_argument(
        "--no-grad-ckpt", action="store_true",
        help="disable gradient checkpointing (faster, more memory)",
    )
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps({**vars(args), "git_rev": _git_rev()}, indent=2))

    if args.micro_batch != 1:
        # Documented limitation: training chunks are (1, T) — extending to
        # micro_batch>1 requires stacking and padding, deferred to a follow-up.
        raise NotImplementedError("micro_batch > 1 not implemented; use --grad-accum instead")

    dtype = getattr(torch, args.dtype)

    print(f"[heal] loading artifact {args.mla_artifact}", flush=True)
    artifact = torch.load(args.mla_artifact, map_location="cpu", weights_only=False)
    model_id = artifact["meta"]["model_id"]
    print(f"[heal]   model_id={model_id} rank={artifact['meta']['rank']} "
          f"d_rope={artifact['meta']['d_rope']}", flush=True)

    print(f"[heal] loading base weights {model_id}", flush=True)
    loaded = load_weights(model_id, dtype=dtype, device=args.device)
    model = Qwen3Model.from_loaded(loaded).to(dtype=dtype, device=args.device)

    print("[heal] applying MLA swap", flush=True)
    apply_mla(model, artifact)

    print(f"[heal] wrapping LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})", flush=True)
    n_wrapped = wrap_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
    print(f"[heal]   wrapped {n_wrapped} Linear layers", flush=True)

    if not args.no_grad_ckpt:
        model.gradient_checkpointing = True
        print("[heal] gradient checkpointing enabled", flush=True)

    inventory = setup_trainable(model)
    print(
        f"[heal] trainable: {inventory.n_total/1e6:.2f}M params "
        f"(mla_proj={inventory.n_mla_proj/1e6:.2f}M, lora={inventory.n_lora/1e6:.2f}M)",
        flush=True,
    )

    optim = build_optimizer(inventory, lr_mla=args.lr_mla, lr_lora=args.lr_lora)

    print(f"[heal] loading {args.train_chunks} train chunks (seq_len={args.seq_len})", flush=True)
    train_chunks = load_wikitext103_chunks(
        n_samples=args.train_chunks,
        chunk_tokens=args.seq_len,
        tokenizer_id=model_id,
        seed=args.split_seed,
        split="train",
    )
    print(f"[heal]   got {len(train_chunks)} chunks", flush=True)

    print(f"[heal] loading {args.val_chunks} validation chunks", flush=True)
    val_chunks = load_wikitext103_chunks(
        n_samples=args.val_chunks,
        chunk_tokens=args.seq_len,
        tokenizer_id=model_id,
        seed=args.split_seed,
        split="validation",
    )
    print(f"[heal]   got {len(val_chunks)} chunks", flush=True)

    # Baseline pre-training PPL on the validation slice.
    print("[heal] measuring pre-FT validation PPL", flush=True)
    model.eval()
    pre_val = evaluate_ppl(model, val_chunks)
    model.train()
    print(f"[heal]   pre-FT val_ppl={pre_val['ppl']:.4f}", flush=True)

    summary_in = vars(args).copy()
    summary_in["pre_ft_val_ppl"] = pre_val["ppl"]

    train_summary = train_loop(
        model, optim, inventory, train_chunks, val_chunks,
        steps=args.steps,
        grad_accum=args.grad_accum,
        seq_len=args.seq_len,
        warmup_frac=args.warmup_frac,
        val_every=args.val_every,
        log_every=args.log_every,
        out_dir=out_dir,
        grad_clip=args.grad_clip,
        seed=args.split_seed,
        progress_writer=lambda m: print(m, flush=True),
    )

    summary = {**summary_in, **train_summary}
    _write_summary(out_dir, summary)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[heal] done. best_val_ppl={train_summary['best_val_ppl']:.4f} "
          f"at step {train_summary['best_step']}; wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
