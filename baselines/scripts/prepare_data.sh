#!/usr/bin/env bash
# Build the MAESTRO-E style datasets for the Polytune / LadderSym baselines
# from ALIGN bundles.
#
#   scripts/prepare_data.sh smoke   # tiny sets for loader / training smoke tests
#   scripts/prepare_data.sh full    # full align_v1 (10k multi + 2k raw) + real_test
#   scripts/prepare_data.sh all     # both
#
# Extra args after the mode are forwarded to prepare_dataset.py (e.g. --overwrite,
# --workers 4).  Do NOT run "full" while synth-pipeline/output_* are being
# regenerated.
set -euo pipefail

BASELINES=${BASELINE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
BASE=$(cd "$BASELINES/.." && pwd)
PY=${POLYTUNE_PYTHON:-$BASELINES/envs/polytune/bin/python}
PREP=$BASELINES/common/prepare_dataset.py
DATA=$BASELINES/data

# 28 small bundles generated with the final synth-pipeline code (12 procedural + 16 real-score snippets);
# audio/MIDI/labels only (no alignment.npz / mels), kept in-repo so the smoke sets are reproducible.
SMOKE_BUNDLES=$DATA/smoke_bundles

MODE=${1:-smoke}
shift || true
EXTRA=("$@")

smoke() {
  echo "== smoke_align (12 gen + 16 real-score snippet bundles; 50/25/25 split) =="
  "$PY" "$PREP" --out "$DATA/smoke_align" \
    --set multi="$SMOKE_BUNDLES/multi" \
    --set raw="$SMOKE_BUNDLES/raw" \
    --val-frac 0.25 --test-frac 0.25 --seed 365 --workers 2 ${EXTRA[@]+"${EXTRA[@]}"}

  echo "== smoke_real (5 real clarinet recordings, empty labels, all 'test') =="
  "$PY" "$PREP" --out "$DATA/smoke_real" \
    --set real="$BASE/data/test" --real-test --limit 5 --workers 2 ${EXTRA[@]+"${EXTRA[@]}"}
}

full() {
  echo "== align_v1 (10k procedurally generated + 2k real-score snippets) =="
  # Both sets use a deterministic per-clip 90/5/5 split (seed 365).  The raw set
  # has only 3 source scores (001, Mozart K.622, Weber), so "--split-by-source raw"
  # would hand one whole score to test, one to validation and leave one for
  # training; keep per-clip splitting and remember that raw snippets of the same
  # score overlap in measures across splits (same-piece practice setting).
  "$PY" "$PREP" --out "$DATA/align_v1" \
    --set multi="$BASE/synth-pipeline/output_10k_multi" \
    --set raw="$BASE/synth-pipeline/output_2k_rawdata" \
    --val-frac 0.05 --test-frac 0.05 --seed 365 \
    --workers "${WORKERS:-4}" ${EXTRA[@]+"${EXTRA[@]}"}

  echo "== real_test (73 real clarinet recordings, empty labels, all 'test') =="
  "$PY" "$PREP" --out "$DATA/real_test" \
    --set real="$BASE/data/test" --real-test --workers "${WORKERS:-4}" ${EXTRA[@]+"${EXTRA[@]}"}
}

case "$MODE" in
  smoke) smoke ;;
  full) full ;;
  all) smoke; full ;;
  *) echo "usage: $0 {smoke|full|all} [extra prepare_dataset.py args]" >&2; exit 2 ;;
esac
