"""Needle-in-a-haystack retrieval eval (RULER-style) for long-context checks.

A prompt hides one or four "special magic number" sentences in repeated filler
text and then asks for them back. Three tasks:

- ``single``      : one needle, ask for it.
- ``multi_key``   : four needles, ask for one (the others are distractors).
- ``multi_query`` : four needles, ask for all four; scored by the fraction found.

The prompt wording follows the sample prompts printed in the appendix of
"On-Policy Attention Linearization" (arXiv:2609.31947); the filler sentences are
RULER's. Prompt construction and scoring are pure functions; ``prefill_chunked``
and ``generate_greedy`` run a prompt of tens of thousands of tokens through the
engine without materializing a full T×T attention matrix.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

TASKS = ("single", "multi_key", "multi_query")
N_NEEDLES = {"single": 1, "multi_key": 4, "multi_query": 4}

HEADER = (
    "Some special magic numbers are hidden within the following text. "
    "Make sure to memorize them. I will quiz you about the numbers afterwards."
)
FILLER = (
    "The grass is green.",
    "The sky is blue.",
    "The sun is yellow.",
    "Here we go.",
    "There and back again.",
)
ADJECTIVES = (
    "spirited", "gentle", "noble", "luminous", "vivid", "placid", "mellow", "rugged",
    "quiet", "amber", "brisk", "crimson", "dusty", "eager", "frosty", "golden",
    "hollow", "ivory", "jolly", "keen", "lively", "misty", "nimble", "ochre",
    "proud", "rapid", "silent", "tidy", "urban", "velvet", "wary", "young",
    "zesty", "bold", "calm", "daring", "earnest", "faint", "grand", "humble",
)
NOUNS = (
    "fiddle", "pillar", "meadow", "ember", "walnut", "lantern", "bramble", "harbor",
    "anchor", "basket", "candle", "dagger", "engine", "falcon", "garden", "hammer",
    "island", "jacket", "kettle", "ladder", "marble", "needle", "orchard", "pebble",
    "quiver", "ribbon", "saddle", "tunnel", "urchin", "valley", "window", "yarrow",
    "zipper", "beacon", "cradle", "drawer", "feather", "glacier", "hinge", "kernel",
)


@dataclass(frozen=True)
class Sample:
    task: str
    text: str
    keys: tuple[str, ...]
    numbers: tuple[str, ...]
    query_idx: tuple[int, ...]  # which needles the question asks for
    depths: tuple[float, ...]  # needle position as a fraction of the haystack


def _question(keys: Sequence[str]) -> str:
    if len(keys) == 1:
        return f"What is the special magic number for {keys[0]} mentioned in the provided text?"
    listed = ", ".join(keys[:-1]) + f", and {keys[-1]}"
    return (
        f"What are the special magic numbers for {listed} mentioned in the provided text? "
        "State each key with its number."
    )


def build_sample(task: str, n_filler: int, rng: random.Random) -> Sample:
    """Build one prompt with ``n_filler`` filler sentences.

    Everything random (keys, numbers, depths, queried needle) is drawn before
    ``n_filler`` is used, so two calls with equal-state ``rng`` and different
    ``n_filler`` hide identical needles at matching relative depths.
    """
    n = N_NEEDLES[task]
    keys = tuple(
        f"{a}-{b}" for a, b in zip(rng.sample(ADJECTIVES, n), rng.sample(NOUNS, n))
    )
    numbers = tuple(str(x) for x in rng.sample(range(1_000_000, 10_000_000), n))
    depths = tuple(sorted(rng.random() for _ in range(n)))
    if task == "multi_query":
        query_idx = tuple(range(n))
    else:
        query_idx = (rng.randrange(n),)

    # Needle i goes in before filler sentence ``slots[i]``; depths are sorted, so
    # needles appear in key order.
    slots = [int(d * (n_filler + 1)) for d in depths]
    parts: list[str] = []
    needle = 0
    for i in range(n_filler + 1):
        while needle < n and slots[needle] == i:
            parts.append(
                f"One of the special magic numbers for {keys[needle]} is: {numbers[needle]}."
            )
            needle += 1
        if i < n_filler:
            parts.append(FILLER[i % len(FILLER)])

    question = _question([keys[i] for i in query_idx])
    text = f"{HEADER}\n\n{' '.join(parts)}\n\n{question}"
    return Sample(task, text, keys, numbers, query_idx, depths)


def fit_sample(
    task: str, *, budget: int, count_tokens: Callable[[str], int], seed: int
) -> Sample:
    """Largest prompt for ``seed`` whose ``count_tokens(text)`` is ≤ ``budget``."""
    def make(n_filler: int) -> Sample:
        return build_sample(task, n_filler, random.Random(seed))

    if count_tokens(make(0).text) > budget:
        raise ValueError(f"budget {budget} is too small for a {task} prompt")
    lo, hi = 0, 1
    while count_tokens(make(hi).text) <= budget:
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if count_tokens(make(mid).text) <= budget:
            lo = mid
        else:
            hi = mid
    return make(lo)


def score(output: str, sample: Sample) -> float:
    """Fraction of the queried numbers that appear in ``output``.

    Commas between digits are dropped first, so "7,191,887" counts as 7191887.
    """
    output = re.sub(r"(?<=\d),(?=\d)", "", output)
    wanted = [sample.numbers[i] for i in sample.query_idx]
    return sum(n in output for n in wanted) / len(wanted)


def prefill_chunked(model, cache, ids: torch.Tensor, chunk: int) -> torch.Tensor:
    """Feed ``ids`` through ``model`` ``chunk`` tokens at a time; return the last logit.

    Attention memory is then ``chunk × T`` per head, not ``T × T``.
    """
    logits = None
    for start in range(0, ids.shape[1], chunk):
        logits = model(ids[:, start : start + chunk], cache, start_pos=start)
    return logits[:, -1]


@torch.inference_mode()
def generate_greedy(
    model, ids: torch.Tensor, *, max_new: int, eos_ids: Sequence[int], chunk: int
) -> list[int]:
    """Greedy continuation of ``ids`` (shape ``(1, T)``), stopping before any EOS id."""
    cache = model.alloc_cache(ids.shape[1] + max_new)
    last = prefill_chunked(model, cache, ids, chunk)
    out: list[int] = []
    pos = ids.shape[1]
    for _ in range(max_new):
        tok = int(last[0].argmax())
        if tok in eos_ids:
            break
        out.append(tok)
        if len(out) == max_new:
            break
        last = model(torch.tensor([[tok]], device=ids.device), cache, start_pos=pos)[:, -1]
        pos += 1
    return out


def wilson_interval(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion ``p`` observed on ``n`` items."""
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def paired_diff_interval(
    a: Sequence[float], b: Sequence[float], *, seed: int, n_boot: int = 10_000
) -> tuple[float, float, float]:
    """Mean of ``a − b`` over paired items, with a 95% bootstrap interval.

    Items are resampled jointly for both arms, so the interval is on the
    difference itself.
    """
    d = torch.tensor(a, dtype=torch.float64) - torch.tensor(b, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(d), (n_boot, len(d)), generator=g)
    means = d[idx].mean(dim=1)
    lo, hi = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return float(d.mean()), float(lo), float(hi)
