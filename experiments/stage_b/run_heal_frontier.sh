#!/usr/bin/env bash
# Sequential healing queue to map the post-heal compression-PPL frontier.
#
# Heals each operating point with the EXACT recipe used for r256_drope32
# (via run_heal_overnight.sh) so compression ratio is the only variable.
# The uncompressed FT control (experiments/stage_b/control_ft) is amortized
# across all of these — no per-point control needed.
#
# Per-run budget capped at 5h (each ~3.3h at 400 steps) so a slow run can't
# starve the rest of the queue. Total ~10-15h unattended.
#
# Usage:
#   nohup bash experiments/stage_b/run_heal_frontier.sh > experiments/stage_b/heal_frontier.log 2>&1 &
#   tail -f experiments/stage_b/heal_frontier.log
set -uo pipefail
cd "$(dirname "$0")/../.."

# (artifact basename, out dir) pairs.
POINTS=(
  "r192_drope32"
  "r128_drope32"
  "r512_drope128"
)

for name in "${POINTS[@]}"; do
  art="experiments/stage_b/${name}.pt"
  out="experiments/stage_b/heal_${name}"
  echo "===================================================================="
  echo "[frontier] $(date -Is) starting heal: ${name}"
  echo "===================================================================="
  if [ ! -f "$art" ]; then
    echo "[frontier] MISSING artifact ${art}; skipping" >&2
    continue
  fi
  if [ -f "${out}/summary.json" ]; then
    echo "[frontier] ${out} already has summary.json; skipping (delete to re-run)"
    continue
  fi
  ARTIFACT="$art" OUT="$out" MAX_HOURS=5 \
    bash experiments/stage_b/run_heal_overnight.sh \
    || echo "[frontier] heal ${name} exited non-zero; continuing queue" >&2
  echo "[frontier] $(date -Is) finished heal: ${name}"
done

echo "[frontier] $(date -Is) queue complete"
