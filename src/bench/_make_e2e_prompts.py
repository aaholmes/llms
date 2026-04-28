"""One-shot generator for the e2e bench prompt set.

Downloads ``databricks/databricks-dolly-15k``, filters to instruction-only
entries with reasonable token lengths, samples 100 deterministically, and
writes them as a Python list literal at ``src/bench/_e2e_prompts.py``.

Run:
    uv run python -m bench._make_e2e_prompts

Idempotent: same seed produces the same output. The generated file is
committed so the e2e bench works without re-downloading.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

DOLLY_REPO = "databricks/databricks-dolly-15k"
DOLLY_FILE = "databricks-dolly-15k.jsonl"
TOKENIZER_FOR_LENGTH = "Qwen/Qwen3-4B"  # the engine's target

# Output bounds (in Qwen3-tokenized prompt length). Keep prompts short enough
# that we stay decode-dominated for max_new=200, not prefill-dominated.
MIN_TOKENS = 5
MAX_TOKENS = 50

OUT_FILE = Path(__file__).parent / "_e2e_prompts.py"


def _is_clean_ascii(s: str) -> bool:
    """Reject non-ASCII or control-character-heavy strings."""
    if not s:
        return False
    return all(0x20 <= ord(c) <= 0x7E or c in "\n\t" for c in s)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--out", type=Path, default=OUT_FILE)
    args = p.parse_args()

    print(f"[make-prompts] downloading {DOLLY_REPO}/{DOLLY_FILE} ...")
    path = hf_hub_download(
        repo_id=DOLLY_REPO, filename=DOLLY_FILE, repo_type="dataset"
    )
    print(f"[make-prompts] -> {path}")

    print(f"[make-prompts] loading tokenizer {TOKENIZER_FOR_LENGTH} ...")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_FOR_LENGTH)

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"[make-prompts] loaded {len(rows)} dolly rows")

    # Filter: instruction-only (empty context), clean ASCII, in length range.
    seen: set[str] = set()
    candidates: list[str] = []
    for row in rows:
        instr = (row.get("instruction") or "").strip()
        ctx = (row.get("context") or "").strip()
        if ctx:
            continue
        if not _is_clean_ascii(instr):
            continue
        if instr in seen:
            continue
        n_tok = len(tok(instr, add_special_tokens=True).input_ids)
        if n_tok < MIN_TOKENS or n_tok > MAX_TOKENS:
            continue
        seen.add(instr)
        candidates.append(instr)
    print(f"[make-prompts] {len(candidates)} candidates after filter")

    if len(candidates) < args.n:
        raise SystemExit(
            f"Only {len(candidates)} prompts pass filters; need {args.n}. "
            f"Loosen MIN_TOKENS/MAX_TOKENS or relax the ASCII filter."
        )

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    sample = candidates[: args.n]

    # Sort by tokenized length so the inlined list reads in length order
    # (purely cosmetic; deterministic sampling order is preserved by the seed).
    sample.sort(key=lambda s: len(tok(s, add_special_tokens=True).input_ids))

    print(f"[make-prompts] writing {args.out} ...")
    with args.out.open("w", encoding="utf-8") as f:
        f.write(
            f'"""Sampled from {DOLLY_REPO} (seed={args.seed}, '
            f"n={args.n}, len in [{MIN_TOKENS}, {MAX_TOKENS}] Qwen3 tokens, "
            f'instruction-only).\n\n'
            f"Regenerate with: ``uv run python -m bench._make_e2e_prompts``.\n"
            f'"""\n\n'
        )
        f.write("PROMPTS: list[str] = [\n")
        for prompt in sample:
            f.write(f"    {prompt!r},\n")
        f.write("]\n")
    print(f"[make-prompts] wrote {len(sample)} prompts")


if __name__ == "__main__":
    main()
