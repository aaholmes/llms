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


def _alloc_cache(model: Qwen3Model, max_seq_len: int):
    """Compressed MLA cache if the model has MLAttention, else the plain GQA cache.

    Lets the harness heal either an MLA-converted model or a plain model (e.g. a
    speculative-decode *draft* being TVD-tuned to a target — no MLA swap)."""
    if any(isinstance(m, MLAttention) for m in model.modules()):
        return alloc_mla_cache(model, max_seq_len=max_seq_len)
    return model.alloc_cache(max_seq_len)


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


def merge_lora(model: nn.Module) -> nn.Module:
    """Fold every ``LoraLinear`` back into a plain ``nn.Linear`` (W ← W + s·B·A).

    After merging, the model is architecturally identical to the un-adapted
    original — no LoRA branch, zero inference overhead — so a merged tuned draft
    runs at exactly the base draft's speed and any throughput difference is
    attributable to the weights (acceptance), not adapter compute. Mutates in
    place; returns the model."""
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, LoraLinear):
                with torch.no_grad():
                    delta = (child.lora_B.weight @ child.lora_A.weight) * child.scaling
                    child.base.weight.add_(delta.to(child.base.weight.dtype))
                setattr(parent, name, child.base)
    return model


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


def chunk_topk_kl_loss(
    logits: torch.Tensor,
    teacher_ids: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Forward KL(teacher ‖ student) over the teacher's top-K support (distillation).

    Both distributions are compared on the teacher's top-K support, each
    renormalized over those K tokens (a peaked LM puts ~all mass there, so the
    out-of-K remainder is negligible). Restricting to the shared support makes
    the loss a proper distribution match: student ≡ teacher ⇒ KL = 0.

    Parameters
    ----------
    logits : (1, T, V) student logits (grad).
    teacher_ids : (T-1, K) long — teacher top-K token indices at scored positions.
    teacher_logits : (T-1, K) float — teacher top-K raw logits.
    temperature : softmax temperature τ; loss scaled by τ² (standard KD scaling).

    Returns scalar KL, mean over the T-1 scored positions.
    """
    tau = temperature
    student_logits = logits[0, :-1].float() / tau                     # (T-1, V)
    student_logits_k = student_logits.gather(-1, teacher_ids)         # (T-1, K)
    student_lp_k = F.log_softmax(student_logits_k, dim=-1)            # renorm over K
    teacher_p_k = F.softmax(teacher_logits.float() / tau, dim=-1)     # renorm over K
    kl = (teacher_p_k * (teacher_p_k.clamp_min(1e-12).log() - student_lp_k)).sum(-1).mean()
    return kl * (tau * tau)


def chunk_topk_accept_loss(
    logits: torch.Tensor,
    teacher_ids: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    softmin_tau: float = 0.0,
) -> torch.Tensor:
    """Total-variation-distance loss ``TVD = 1 − Σ min(p_teacher, p_student)`` on the top-K support.

    The total variation distance ``TVD(p, q) = 1 − Σ_i min(p_i, q_i) = ½ Σ_i |p_i − q_i|``;
    its complement ``Σ min(p, q) = 1 − TVD`` is the expected single-token
    speculative-decode acceptance rate (Leviathan et al.), so minimizing TVD
    *directly* maximizes acceptance — the sampling-aligned target, unlike KL which
    is a mode-covering surrogate with a different optimum under a constrained model.

    ``softmin_tau``:
      - ``0`` → exact ``min`` (autograd-valid a.e.; hard 0/1 gradient).
      - ``>0`` → numerically-stable soft-min ``min(a,b) − τ·softplus(−|a−b|/τ)``,
        whose gradient w.r.t. the student is ``σ((p_t − p_s)/τ)`` — a graded
        sigmoid of the probability gap. → exact min as ``τ → 0``.

    Both distributions are renormalized over the teacher's top-K support (matching
    :func:`chunk_topk_kl_loss`). Returns ``1 − overlap`` (≥ 0, = 0 at a match).
    """
    tau_t = temperature
    student_logits_k = (logits[0, :-1].float() / tau_t).gather(-1, teacher_ids)  # (T-1,K)
    p_s = F.softmax(student_logits_k, dim=-1)
    p_t = F.softmax(teacher_logits.float() / tau_t, dim=-1)
    if softmin_tau and softmin_tau > 0:
        s = softmin_tau
        smin = torch.minimum(p_t, p_s) - s * F.softplus(-(p_t - p_s).abs() / s)
        overlap = smin.sum(-1)
    else:
        overlap = torch.minimum(p_t, p_s).sum(-1)
    return (1.0 - overlap).mean()


def train_loop(
    model: Qwen3Model,
    optim: torch.optim.Optimizer,
    inventory: TrainableInventory,
    train_chunks: list[torch.Tensor],
    val_chunks: list[torch.Tensor],
    *,
    teacher: dict | None = None,
    steps: int,
    grad_accum: int,
    seq_len: int,
    warmup_frac: float,
    val_every: int,
    log_every: int,
    out_dir: Path,
    grad_clip: float = 1.0,
    seed: int = 0,
    max_seconds: float | None = None,
    progress_writer=None,
) -> dict:
    """Train ``model`` for ``steps`` optimizer steps. Returns final summary dict.

    If ``max_seconds`` is set, training stops cleanly at the first step
    boundary past the budget (a final validation runs before stopping), so
    the cosine schedule should be sized via ``steps`` to roughly match.
    """
    device = next(model.parameters()).device
    log_path = out_dir / "training_log.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator().manual_seed(seed)
    all_trainable = inventory.mla_proj_params + inventory.lora_params

    best_val_ppl = float("inf")
    best_step = -1
    stopped_early = False
    t0 = time.time()

    with log_path.open("w") as log_f:
        for step in range(steps):
            out_of_time = max_seconds is not None and time.time() - t0 > max_seconds
            mult = cosine_warmup_multiplier(step, steps, warmup_frac)
            apply_lr_multiplier(optim, mult)

            optim.zero_grad(set_to_none=True)
            step_loss = 0.0

            for _ in range(grad_accum):
                idx = int(torch.randint(0, len(train_chunks), (1,), generator=rng).item())
                ids = train_chunks[idx].to(device)
                # Tight cache: one chunk's worth, freed at end of iter.
                cache = _alloc_cache(model, ids.shape[1])
                logits = model(ids, cache, start_pos=0)
                if teacher is not None:
                    # Distill-to-original against the frozen teacher's precomputed
                    # top-K logits (aligned to train_chunks by index). Objective:
                    # 'kl' (forward KL surrogate) or 'accept' (direct/soft-min TVD,
                    # targeting speculative-decode acceptance).
                    t_ids = teacher["topk_ids"][idx].to(device)
                    t_logits = teacher["topk_logits"][idx].to(device)
                    if teacher["objective"] == "accept":
                        loss = chunk_topk_accept_loss(
                            logits, t_ids, t_logits, temperature=teacher["temp"],
                            softmin_tau=teacher["softmin_tau"],
                        )
                    else:
                        loss = chunk_topk_kl_loss(
                            logits, t_ids, t_logits, temperature=teacher["temp"]
                        )
                    if teacher["ce_weight"] > 0:
                        loss = loss + teacher["ce_weight"] * chunk_ce_loss(logits, ids)
                else:
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

            do_val = (step + 1) % val_every == 0 or step == steps - 1 or out_of_time
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

            if out_of_time:
                stopped_early = True
                if progress_writer is not None:
                    progress_writer(
                        f"[heal] time budget reached after step {step+1}/{steps}; stopping"
                    )
                break

    return {
        "best_val_ppl": best_val_ppl,
        "best_step": best_step,
        "stopped_early": stopped_early,
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
        "--mla-artifact", default=None,
        help="path to the converted MLA artifact (.pt) from mla.convert. Omit to heal "
             "a *plain* model (no MLA swap) named by --model-id — e.g. TVD-tuning a "
             "speculative-decode draft to a target.",
    )
    ap.add_argument(
        "--model-id", default=None,
        help="HF model id for a plain (no-MLA) heal, when --mla-artifact is omitted.",
    )
    ap.add_argument("--out", required=True, help="output dir for logs and checkpoints")
    ap.add_argument(
        "--data", default="wt103", choices=["wt103"],
        help="training data source (only wt103 for now)",
    )
    ap.add_argument("--split-seed", type=int, default=17, help="train-split sampling seed")
    ap.add_argument("--steps", type=int, default=3000, help="optimizer steps")
    ap.add_argument(
        "--max-hours", type=float, default=None,
        help="wall-clock budget; stop cleanly (with a final val) at the first "
             "step boundary past it. Size --steps to roughly match so the "
             "cosine schedule isn't cut off mid-decay.",
    )
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
    ap.add_argument("--val-corpus", default="wikitext", choices=["wikitext", "code"],
                    help="validation workload; use 'code' when TVD-tuning a draft on code")
    ap.add_argument("--train-chunks", type=int, default=20000,
                    help="how many unique training chunks to load (cycled by random sampling)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument(
        "--teacher-topk", default=None,
        help="path to a precomputed teacher top-K distillation file (from "
             "experiments/stage_b/precompute_teacher_topk.py). If set, healing "
             "trains on forward KL to the frozen original instead of cross-entropy "
             "on data; its chunk pool replaces the sampled train chunks.",
    )
    ap.add_argument("--distill-temp", type=float, default=1.0,
                    help="distillation softmax temperature τ. Default 1.0.")
    ap.add_argument("--distill-ce-weight", type=float, default=0.0,
                    help="optional CE-on-data mix added to the distill loss. Default 0.")
    ap.add_argument("--distill-objective", default="kl", choices=["kl", "accept"],
                    help="distill target: 'kl' (forward-KL surrogate) or 'accept' "
                         "(1−TVD = speculative-decode acceptance, the sampling-aligned target)")
    ap.add_argument("--softmin-tau", type=float, default=0.0,
                    help="soft-min smoothing for --distill-objective accept: 0 = exact "
                         "min (hard gradient); >0 = smoothed (graded sigmoid gradient)")
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

    if args.mla_artifact:
        print(f"[heal] loading artifact {args.mla_artifact}", flush=True)
        artifact = torch.load(args.mla_artifact, map_location="cpu", weights_only=False)
        model_id = artifact["meta"]["model_id"]
        print(f"[heal]   model_id={model_id} rank={artifact['meta']['rank']} "
              f"d_rope={artifact['meta']['d_rope']}", flush=True)
    elif args.model_id:
        artifact = None
        model_id = args.model_id
        print(f"[heal] plain heal (no MLA swap), model_id={model_id}", flush=True)
    else:
        raise SystemExit("provide either --mla-artifact or --model-id")

    print(f"[heal] loading base weights {model_id}", flush=True)
    # Stage the HF weights on CPU and move only the constructed model to the
    # device. Loading straight to cuda keeps the 8 GB staging state dict
    # resident alongside the model's own 8 GB copy — two full models on a
    # 16 GB card OOMs before training starts. Free the staging copy once the
    # model owns its weights.
    loaded = load_weights(model_id, dtype=dtype, device="cpu")
    model = Qwen3Model.from_loaded(loaded).to(dtype=dtype, device=args.device)
    del loaded
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if artifact is not None:
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

    teacher = None
    if args.teacher_topk:
        # Distill-to-original: the training pool and its per-token labels come
        # from the precomputed teacher file, so chunks and top-K stay aligned.
        print(f"[heal] distillation mode: loading teacher top-K {args.teacher_topk}", flush=True)
        tk = torch.load(args.teacher_topk, map_location="cpu", weights_only=False)
        train_chunks = [c for c in tk["chunk_ids"]]  # list of (1, T) long
        teacher = {
            "topk_ids": tk["topk_ids"],        # (N, T-1, K) long, CPU
            "topk_logits": tk["topk_logits"],  # (N, T-1, K) float16, CPU
            "temp": args.distill_temp,
            "ce_weight": args.distill_ce_weight,
            "objective": args.distill_objective,
            "softmin_tau": args.softmin_tau,
        }
        print(f"[heal]   teacher pool: {len(train_chunks)} chunks, K={tk['topk_ids'].shape[-1]}, "
              f"teacher_model={tk.get('model_id')}, objective={args.distill_objective}"
              f"{' (softmin_tau=%.3g)' % args.softmin_tau if args.distill_objective=='accept' else ''}, "
              f"τ={args.distill_temp}, ce_weight={args.distill_ce_weight}", flush=True)
    else:
        print(f"[heal] loading {args.train_chunks} train chunks (seq_len={args.seq_len})", flush=True)
        train_chunks = load_wikitext103_chunks(
            n_samples=args.train_chunks,
            chunk_tokens=args.seq_len,
            tokenizer_id=model_id,
            seed=args.split_seed,
            split="train",
        )
        print(f"[heal]   got {len(train_chunks)} chunks", flush=True)

    print(f"[heal] loading {args.val_chunks} {args.val_corpus} validation chunks", flush=True)
    if args.val_corpus == "code":
        from mla.calibrate import load_code_chunks
        # held-out slice: past the teacher/tuning region (skip=0..~12k), before the bench's skip=20000
        val_chunks = load_code_chunks(
            n_samples=args.val_chunks, chunk_tokens=args.seq_len,
            tokenizer_id=model_id, skip=15000,
        )
    else:
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
        teacher=teacher,
        steps=args.steps,
        grad_accum=args.grad_accum,
        seq_len=args.seq_len,
        warmup_frac=args.warmup_frac,
        val_every=args.val_every,
        log_every=args.log_every,
        out_dir=out_dir,
        grad_clip=args.grad_clip,
        seed=args.split_seed,
        max_seconds=args.max_hours * 3600.0 if args.max_hours is not None else None,
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
