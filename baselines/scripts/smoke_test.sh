#!/usr/bin/env bash
# Usage: smoke_test.sh [cpu|mps|cuda] [DATA_ROOT]
# Functional test only: shortened decoding cannot be used as research results.
set -euo pipefail
B=${BASELINE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PROFILE=${1:-cpu}
DATA=${2:-$B/data/smoke_align}
DATA=$(cd "$DATA" && pwd)
STAMP=$(date +%Y%m%d_%H%M%S)
export MPLBACKEND=Agg
"$B/envs/polytune/bin/python" "$B/common/audit_dataset.py" --data "$DATA"
for variant in polytune prompted unprompted; do
  if [[ "$variant" == polytune ]]; then
    flavor=polytune; repo=Polytune; extra=()
  else
    flavor=laddersym; repo=LadderSym; extra=("--$variant")
  fi
  py="$B/envs/$flavor/bin/python"
  BASELINE_FLAVOR="$repo" "$py" "$B/tests/test_regressions.py"
  (cd "$B/$repo" && "$py" "$B/common/verify_loaders.py" --flavor "$flavor" --root "$DATA" --items 2)
  run="smoke_${variant}_${STAMP}"
  budget=(event_length=128)
  [[ "$flavor" != laddersym ]] || budget+=(prompt_length=256)
  "$B/scripts/${flavor}_train.sh" --data "$DATA" --profile "$PROFILE" --smoke \
    --run-name "$run" ${extra[@]+"${extra[@]}"} -- "${budget[@]}"
  ckpt=$(find "$B/runs/$flavor/$run" -name last.ckpt -print | head -n 1)
  [[ -n "$ckpt" ]] || { echo "No Lightning checkpoint was saved" >&2; exit 1; }
  "$B/scripts/${flavor}_eval.sh" --data "$DATA" --profile "$PROFILE" --ckpt "$ckpt" \
    ${extra[@]+"${extra[@]}"} --first-n 1 --max-length 16 --tag "$run" -- "${budget[@]}"
done
echo "PASS: three training variants, strict checkpoint reloads, inference and note_metrics.json"
