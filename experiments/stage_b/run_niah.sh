#!/usr/bin/env bash
# Retrieval eval, cheapest and most decisive cells first. Each step resumes from
# the rows already on disk, so later steps only fill what earlier ones left.
set -euo pipefail
cd "$(dirname "$0")/../.."
run() { uv run python -m experiments.stage_b.niah_eval "$@"; }
run --lengths 1024 4096          # every arm at short context
run --arms uncompressed          # reference arm, all lengths
run --arms control-ft mla-1.33x mla-1.33x-unhealed   # arms that answer at short context
run                              # everything else
uv run python -m experiments.stage_b.niah_analyze
