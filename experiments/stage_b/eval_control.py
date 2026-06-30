import json
from pathlib import Path
import torch
from bench.eval_ppl import evaluate_ppl
from engine.model import Qwen3Model
from engine.weights import load_weights
from mla.calibrate import load_wikitext103_chunks
from mla.heal import load_trainable, wrap_lora

DEV, DT = "cuda", torch.bfloat16
chunks = load_wikitext103_chunks(n_samples=256, chunk_tokens=1024,
                                 tokenizer_id="Qwen/Qwen3-4B", split="validation")
print(f"[ctrl-eval] {len(chunks)} chunks", flush=True)
loaded = load_weights("Qwen/Qwen3-4B", dtype=DT, device="cpu")
model = Qwen3Model.from_loaded(loaded).to(dtype=DT, device=DEV)
del loaded; torch.cuda.empty_cache()
wrap_lora(model, rank=16, alpha=32.0)
miss = load_trainable(model, "experiments/stage_b/control_ft/best.pt")
if miss: print("WARN unmatched:", miss[:5], flush=True)
model.eval()
m = evaluate_ppl(model, chunks, progress_every=64)
print(f"[ctrl-eval] control PPL (250-chunk slice) = {m['ppl']:.4f}", flush=True)
Path("experiments/stage_b/eval_summary_control.json").write_text(
    json.dumps({"control_ft_ppl_250chunk": m["ppl"], "tokens": m["token_count"]}, indent=2))
