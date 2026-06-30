"""Same-budget WT103 finetune CONTROL on the UNCONVERTED Qwen3-4B.

Isolates how much of the healed model's PPL gain is plain WikiText-103
domain adaptation (which the never-finetuned baseline never received) vs.
genuine MLA-compression recovery. The honest 4x-compression cost is then
   healed_ppl  vs  control_ppl
not  healed_ppl vs raw-baseline_ppl.

Recipe mirrors the heal run exactly (same train/val slices via split_seed=17,
same LoRA targets q/o/gate/up/down at rank=16 alpha=32, same 400 steps /
grad_accum=32 / seq_len=1024 / lr_lora=1e-4 / cosine+3% warmup), with two
unavoidable differences that are inherent to "no conversion":
  - no MLA projections exist, so only the 28.9M LoRA params train
    (healed additionally full-trains 63.7M SVD-init MLA projections);
  - the plain GQA KV cache is used instead of the compressed MLA cache.
This is the standard post-hoc-MLA control: best same-recipe FT of the baseline.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from bench.eval_ppl import evaluate_ppl
from engine.model import Qwen3Model
from engine.weights import load_weights
from mla.calibrate import load_wikitext103_chunks
from mla.heal import (
    apply_lr_multiplier,
    build_optimizer,
    chunk_ce_loss,
    cosine_warmup_multiplier,
    save_trainable,
    setup_trainable,
    wrap_lora,
)

MODEL_ID = "Qwen/Qwen3-4B"
OUT = Path("experiments/stage_b/control_ft")
DEVICE = "cuda"
DTYPE = torch.bfloat16
STEPS = 400
GRAD_ACCUM = 32
SEQ_LEN = 1024
LORA_RANK = 16
LORA_ALPHA = 32.0
LR_LORA = 1e-4
WARMUP_FRAC = 0.03
GRAD_CLIP = 1.0
VAL_EVERY = 30
VAL_CHUNKS = 50
TRAIN_CHUNKS = 20000
SPLIT_SEED = 17
MAX_HOURS = 9.0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    loaded = load_weights(MODEL_ID, dtype=DTYPE, device="cpu")
    model = Qwen3Model.from_loaded(loaded).to(dtype=DTYPE, device=DEVICE)
    del loaded
    torch.cuda.empty_cache()

    # No apply_mla — this is the unconverted baseline. LoRA on the same targets.
    n_wrapped = wrap_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA)
    print(f"[control] wrapped {n_wrapped} Linear layers (LoRA only, no MLA)", flush=True)
    model.gradient_checkpointing = True
    inventory = setup_trainable(model)  # no MLAttention -> only LoRA trains
    print(f"[control] trainable: {inventory.n_total/1e6:.2f}M "
          f"(mla_proj={inventory.n_mla_proj/1e6:.2f}M, lora={inventory.n_lora/1e6:.2f}M)",
          flush=True)
    optim = build_optimizer(inventory, lr_mla=LR_LORA, lr_lora=LR_LORA)

    train_chunks = load_wikitext103_chunks(n_samples=TRAIN_CHUNKS, chunk_tokens=SEQ_LEN,
                                           tokenizer_id=MODEL_ID, seed=SPLIT_SEED, split="train")
    val_chunks = load_wikitext103_chunks(n_samples=VAL_CHUNKS, chunk_tokens=SEQ_LEN,
                                         tokenizer_id=MODEL_ID, seed=SPLIT_SEED, split="validation")
    print(f"[control] {len(train_chunks)} train / {len(val_chunks)} val chunks", flush=True)

    model.eval()
    pre = evaluate_ppl(model, val_chunks)
    model.train()
    print(f"[control] pre-FT val_ppl={pre['ppl']:.4f}", flush=True)

    all_trainable = inventory.mla_proj_params + inventory.lora_params
    rng = torch.Generator().manual_seed(SPLIT_SEED)
    best_val = float("inf")
    best_step = -1
    t0 = time.time()
    max_seconds = MAX_HOURS * 3600.0
    log_path = OUT / "training_log.jsonl"

    with log_path.open("w") as log_f:
        for step in range(STEPS):
            out_of_time = time.time() - t0 > max_seconds
            apply_lr_multiplier(optim, cosine_warmup_multiplier(step, STEPS, WARMUP_FRAC))
            optim.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(GRAD_ACCUM):
                idx = int(torch.randint(0, len(train_chunks), (1,), generator=rng).item())
                ids = train_chunks[idx].to(DEVICE)
                cache = model.alloc_cache(ids.shape[1])  # plain GQA cache
                logits = model(ids, cache, start_pos=0)
                loss = chunk_ce_loss(logits, ids)
                (loss / GRAD_ACCUM).backward()
                step_loss += float(loss.item()) / GRAD_ACCUM
                del cache, logits, loss
            grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=GRAD_CLIP)
            optim.step()

            entry = {"step": step, "train_loss": step_loss, "grad_norm": float(grad_norm.item()),
                     "lr_lora": optim.param_groups[-1]["lr"], "wall_seconds": time.time() - t0}
            do_val = (step + 1) % VAL_EVERY == 0 or step == STEPS - 1 or out_of_time
            if do_val:
                model.eval()
                val = evaluate_ppl(model, val_chunks)
                model.train()
                entry["val_ppl"] = val["ppl"]
                if val["ppl"] < best_val:
                    best_val = val["ppl"]
                    best_step = step
                    entry["best"] = True
                    save_trainable(model, OUT / "best.pt")
                print(f"[control] step {step+1}/{STEPS} loss={step_loss:.4f} "
                      f"val_ppl={val['ppl']:.4f}", flush=True)
            elif (step + 1) % 10 == 0:
                print(f"[control] step {step+1}/{STEPS} loss={step_loss:.4f} "
                      f"grad_norm={entry['grad_norm']:.2f}", flush=True)
            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()
            if out_of_time:
                break

    summary = {"pre_ft_val_ppl": pre["ppl"], "best_val_ppl": best_val, "best_step": best_step,
               "steps": STEPS, "wall_seconds": time.time() - t0,
               "n_trainable_lora": inventory.n_lora}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[control] done. best_val_ppl={best_val:.4f} at step {best_step}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
