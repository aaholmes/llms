"""Smoke test for ``bench.e2e`` — verifies the harness runs end-to-end and
produces a well-formed JSON output, in a few seconds rather than the ~65 min
the formal sweep takes.

Uses Phase B only (our engine, no Triton, no HF), 2 prompts, max_new=8 — the
smallest useful invocation that exercises the full code path.
"""

from __future__ import annotations

import json
import sys

import pytest


@pytest.mark.requires_draft
@pytest.mark.requires_cuda
def test_e2e_smoke(tmp_path, monkeypatch, draft_model_id):
    """Bench harness runs at minimum-viable size and produces valid JSON."""
    from bench import e2e

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "e2e",
            "--target", draft_model_id,   # use the small model as both target and draft
            "--draft", draft_model_id,
            "--n-prompts", "2",
            "--max-new", "8",
            "--K", "4",
            "--phase", "B",
            "--out", str(tmp_path),
        ],
    )

    e2e.main()

    results_file = tmp_path / "e2e_results.json"
    assert results_file.exists(), "bench.e2e didn't write e2e_results.json"
    data = json.loads(results_file.read_text())

    assert "results" in data
    assert "config" in data
    labels = {row["label"] for row in data["results"]}
    assert "ours_greedy_eager" in labels
    assert any("ours_spec_eager" in lab for lab in labels), labels
