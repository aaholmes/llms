"""Tests for the needle-in-a-haystack retrieval eval.

Prompt construction and scoring are pure functions tested without a model;
chunked prefill and greedy generation are checked on the tiny synthetic model.
"""

from __future__ import annotations

import random

import pytest
import torch

from bench.niah import (
    FILLER,
    TASKS,
    build_sample,
    fit_sample,
    generate_greedy,
    prefill_chunked,
    score,
)
from engine.model import Qwen3Model

from ._tiny import Cfg, tiny_loaded


def _words(text: str) -> int:
    return len(text.split())


# --- prompt construction ------------------------------------------------------

@pytest.mark.parametrize("task", TASKS)
def test_build_sample_contains_every_needle_and_the_query(task: str) -> None:
    s = build_sample(task, n_filler=200, rng=random.Random(0))

    assert s.text.startswith("Some special magic numbers are hidden within the following text.")
    for key, number in zip(s.keys, s.numbers):
        assert f"One of the special magic numbers for {key} is: {number}." in s.text
    for i in s.query_idx:
        assert s.keys[i] in s.text.rsplit("\n\n", 1)[1]
    assert sum(s.text.count(sentence) for sentence in FILLER) == 200


def test_needle_and_query_counts_per_task() -> None:
    rng = random.Random(1)
    assert len(build_sample("single", 50, rng).keys) == 1
    multi_key = build_sample("multi_key", 50, rng)
    assert len(multi_key.keys) == 4 and len(multi_key.query_idx) == 1
    multi_query = build_sample("multi_query", 50, rng)
    assert len(multi_query.keys) == 4 and multi_query.query_idx == (0, 1, 2, 3)
    assert "State each key with its number." in multi_query.text


def test_keys_and_numbers_are_distinct_seven_digit() -> None:
    s = build_sample("multi_query", 50, random.Random(2))
    assert len(set(s.keys)) == 4 and len(set(s.numbers)) == 4
    assert all(len(n) == 7 and n.isdigit() and n[0] != "0" for n in s.numbers)


def test_build_sample_is_deterministic_in_the_seed() -> None:
    a = build_sample("multi_key", 80, random.Random(7))
    b = build_sample("multi_key", 80, random.Random(7))
    c = build_sample("multi_key", 80, random.Random(8))
    assert a == b
    assert a.text != c.text


@pytest.mark.parametrize("task", TASKS)
def test_fit_sample_fills_the_budget_without_exceeding_it(task: str) -> None:
    budget = 3000
    s = fit_sample(task, budget=budget, count_tokens=_words, seed=3)

    assert _words(s.text) <= budget
    # One more filler sentence is at most 5 words; the fit should be that tight.
    assert _words(s.text) > budget - 6


def test_fit_sample_keeps_needles_when_the_budget_changes() -> None:
    small = fit_sample("multi_key", budget=1000, count_tokens=_words, seed=5)
    large = fit_sample("multi_key", budget=4000, count_tokens=_words, seed=5)
    assert small.keys == large.keys and small.numbers == large.numbers


def test_fit_sample_rejects_a_budget_too_small_for_the_needles() -> None:
    with pytest.raises(ValueError):
        fit_sample("multi_query", budget=10, count_tokens=_words, seed=0)


# --- scoring --------------------------------------------------------------------

def test_score_single_is_one_only_when_the_number_appears() -> None:
    s = build_sample("single", 20, random.Random(0))
    assert score(f"The number is **{s.numbers[0]}**.", s) == 1.0
    assert score("The number is 1234567.", s) == 0.0


def test_score_multi_key_ignores_the_distractor_numbers() -> None:
    s = build_sample("multi_key", 20, random.Random(0))
    wrong = next(n for i, n in enumerate(s.numbers) if i != s.query_idx[0])
    assert score(wrong, s) == 0.0
    assert score(s.numbers[s.query_idx[0]], s) == 1.0


def test_score_multi_query_gives_partial_credit() -> None:
    s = build_sample("multi_query", 20, random.Random(0))
    assert score(f"{s.numbers[0]} and {s.numbers[2]}", s) == 0.5
    assert score(" ".join(s.numbers), s) == 1.0


# --- chunked prefill and generation ---------------------------------------------

def _tiny_model() -> Qwen3Model:
    return Qwen3Model.from_loaded(tiny_loaded(Cfg())).eval()


@torch.inference_mode()
def test_prefill_chunked_matches_single_pass() -> None:
    model = _tiny_model()
    ids = torch.randint(0, 32, (1, 23), generator=torch.Generator().manual_seed(0))

    ref = model(ids, model.alloc_cache(32), start_pos=0)[:, -1]
    cache = model.alloc_cache(32)
    got = prefill_chunked(model, cache, ids, chunk=5)

    assert cache.cur_len == 23
    assert torch.allclose(got, ref, atol=1e-5)


@torch.inference_mode()
def test_generate_greedy_matches_full_recompute_and_stops_at_eos() -> None:
    model = _tiny_model()
    ids = torch.randint(0, 32, (1, 11), generator=torch.Generator().manual_seed(1))

    expected, cur = [], ids
    for _ in range(6):
        tok = int(model(cur, model.alloc_cache(32), start_pos=0)[0, -1].argmax())
        expected.append(tok)
        cur = torch.cat([cur, torch.tensor([[tok]])], dim=1)

    assert generate_greedy(model, ids, max_new=6, eos_ids=(), chunk=4) == expected
    stopped = generate_greedy(model, ids, max_new=6, eos_ids=(expected[2],), chunk=4)
    assert stopped == expected[: expected.index(expected[2])]


# --- intervals --------------------------------------------------------------------

def test_wilson_interval_known_values() -> None:
    from bench.niah import wilson_interval

    lo, hi = wilson_interval(1.0, 100)
    assert abs(hi - 1.0) < 1e-9 and abs(lo - 0.9630) < 1e-3
    lo, hi = wilson_interval(0.5, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3


def test_paired_diff_interval_excludes_zero_for_a_clear_gap_only() -> None:
    from bench.niah import paired_diff_interval

    base = [1.0] * 100
    worse = [0.0] * 60 + [1.0] * 40
    diff, lo, hi = paired_diff_interval(worse, base, seed=0)
    assert abs(diff + 0.6) < 1e-9 and hi < 0

    same = [1.0, 0.0] * 50
    diff, lo, hi = paired_diff_interval(same, same, seed=0)
    assert diff == 0.0 and lo == 0.0 and hi == 0.0


def test_score_accepts_a_number_written_with_thousands_separators() -> None:
    s = build_sample("single", 20, random.Random(0))
    n = s.numbers[0]
    grouped = f"{n[0]},{n[1:4]},{n[4:]}"
    assert score(f"The number is **{grouped}**.", s) == 1.0
    # A comma followed by a space separates two numbers; it must not join them.
    assert score(f"{n[:4]}, {n[4:]}", s) == 0.0
