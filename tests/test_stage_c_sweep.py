"""Stage C sweep driver: plan construction, resume, artifact gating, summary.

The measurement itself needs a GPU, but everything that decides *what gets
measured* is pure and testable here — which is the part that would silently
corrupt a multi-hour run if it were wrong.
"""

from __future__ import annotations

import json

import pytest

from experiments.stage_c.sweep import (
    Cell,
    ModelSpec,
    Regime,
    build_plan,
    capture_environment,
    default_regimes,
    load_done_keys,
    missing_artifacts,
    parse_compressions,
    write_summary,
)


# --- grid -----------------------------------------------------------------


def test_default_regimes_cover_all_four_per_compression_point() -> None:
    regimes = default_regimes([(256, 32, 4.0), (128, 32, 5.33)])
    names = [r.name for r in regimes]
    assert names.count("baseline") == 1  # baseline is compression-independent
    for n in ("target-only", "coupled", "specialized"):
        assert names.count(n) == 2
    assert len(regimes) == 1 + 3 * 2


def test_coupled_and_specialized_differ_in_draft_kind() -> None:
    """The align-vs-shrink question is exactly this distinction; don't collapse it."""
    regimes = {r.name: r for r in default_regimes([(256, 32, 4.0)])}
    assert regimes["coupled"].draft.kind == "mla"          # shrink the draft
    assert regimes["specialized"].draft.kind == "lora-merged"  # align the draft
    assert regimes["target-only"].draft.kind == "dense"


def test_baseline_requires_no_artifacts() -> None:
    baseline = default_regimes([])[0]
    assert baseline.name == "baseline"
    assert missing_artifacts(baseline) == []


def test_compressed_regimes_require_artifacts(tmp_path) -> None:
    regime = [r for r in default_regimes([(256, 32, 4.0)]) if r.name == "target-only"][0]
    missing = missing_artifacts(regime)
    assert missing, "an unconverted operating point must report missing artifacts"
    assert any("r256_drope32" in p.name or "best.pt" == p.name for p in missing)


def test_plan_is_regimes_times_ks() -> None:
    regimes = default_regimes([(256, 32, 4.0)])
    plan = build_plan(regimes, [1, 2, 3])
    assert len(plan) == len(regimes) * 3
    assert len({c.key() for _, c in plan}) == len(plan), "cell keys must be unique"


def test_cell_key_distinguishes_regime_and_k() -> None:
    a = Cell("coupled", "t", "d", 4.0, 3)
    b = Cell("specialized", "t", "d", 4.0, 3)
    c = Cell("coupled", "t", "d", 4.0, 4)
    assert len({a.key(), b.key(), c.key()}) == 3


# --- compression parsing --------------------------------------------------


def test_parse_compressions() -> None:
    assert parse_compressions(["256:32:4.0", "128:32:5.33"]) == [
        (256, 32, 4.0),
        (128, 32, 5.33),
    ]


def test_parse_compressions_rejects_malformed() -> None:
    with pytest.raises(ValueError, match="rank:d_rope:ratio"):
        parse_compressions(["256:32"])


# --- resume ---------------------------------------------------------------


def test_load_done_keys_empty_when_absent(tmp_path) -> None:
    assert load_done_keys(tmp_path / "nope.jsonl") == set()


def test_load_done_keys_reads_completed_cells(tmp_path) -> None:
    p = tmp_path / "results.jsonl"
    p.write_text('{"key": "a"}\n{"key": "b"}\n')
    assert load_done_keys(p) == {"a", "b"}


def test_truncated_final_line_is_not_counted_done(tmp_path) -> None:
    """A killed run leaves a partial line; that cell must re-run, not be trusted."""
    p = tmp_path / "results.jsonl"
    p.write_text('{"key": "a"}\n{"key": "b", "tok_s": 12')
    assert load_done_keys(p) == {"a"}


def test_resume_skips_only_done_cells(tmp_path) -> None:
    regimes = default_regimes([])
    plan = build_plan(regimes, [1, 2])
    p = tmp_path / "results.jsonl"
    p.write_text(json.dumps({"key": plan[0][1].key()}) + "\n")
    done = load_done_keys(p)
    todo = [c for _, c in plan if c.key() not in done]
    assert len(todo) == len(plan) - 1


# --- environment capture --------------------------------------------------


def test_environment_capture_records_provenance() -> None:
    env = capture_environment()
    for field in ("git_rev", "git_dirty", "python", "platform", "torch"):
        assert field in env
    assert isinstance(env["git_dirty"], bool)


# --- summary --------------------------------------------------------------


def _row(**kw) -> dict:
    base = {
        "key": "k", "regime": "baseline", "target_label": "T", "draft_label": "D",
        "compression": 1.0, "K": 3, "tok_s": 100.0, "greedy_tok_s": 70.0,
        "speedup": 1.43, "acceptance": 0.7, "ppl": 21.6,
        "peak_vram_bytes": 10 * 2**30, "within_budget": True,
    }
    base.update(kw)
    return base


def test_summary_picks_best_k_by_throughput(tmp_path) -> None:
    (tmp_path / "results.jsonl").write_text(
        "\n".join(json.dumps(_row(K=k, tok_s=t)) for k, t in [(2, 90.0), (3, 110.0), (4, 95.0)])
    )
    write_summary(tmp_path, skipped=[], n_ok=3, n_fail=0)
    md = (tmp_path / "summary.md").read_text()
    assert "## Best-K per regime" in md
    best_line = [l for l in md.splitlines() if l.startswith("| baseline | 1.00×")][-1]
    assert "| 3 |" in best_line


def test_summary_reports_skipped_regimes_as_holes(tmp_path) -> None:
    """Silent truncation reads as 'covered everything' — it must not."""
    (tmp_path / "results.jsonl").write_text(json.dumps(_row()) + "\n")
    regime = Regime("coupled", ModelSpec("dense", "t"), ModelSpec("mla", "d", "x.pt"), 4.0)
    write_summary(tmp_path, skipped=[(regime, [tmp_path / "x.pt"])], n_ok=1, n_fail=0)
    md = (tmp_path / "summary.md").read_text()
    assert "Not run" in md and "coupled" in md and "x.pt" in md


def test_summary_flags_over_budget_cells(tmp_path) -> None:
    (tmp_path / "results.jsonl").write_text(
        json.dumps(_row(peak_vram_bytes=18 * 2**30, within_budget=False)) + "\n"
    )
    write_summary(tmp_path, skipped=[], n_ok=1, n_fail=0)
    assert "exceeded the VRAM budget" in (tmp_path / "summary.md").read_text()


def test_summary_handles_missing_ppl(tmp_path) -> None:
    (tmp_path / "results.jsonl").write_text(json.dumps(_row(ppl=None)) + "\n")
    write_summary(tmp_path, skipped=[], n_ok=1, n_fail=0)
    assert "—" in (tmp_path / "summary.md").read_text()


def test_summary_noop_on_empty_results(tmp_path) -> None:
    (tmp_path / "results.jsonl").write_text("")
    write_summary(tmp_path, skipped=[], n_ok=0, n_fail=0)
    assert not (tmp_path / "summary.md").exists()
