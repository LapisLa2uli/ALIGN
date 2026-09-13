#!/usr/bin/env bash
# Run a trained Polytune checkpoint on ONE performance/reference wav pair (no dataset layout needed).
#
# Usage:
#   polytune_infer.sh --ckpt <path.ckpt|.pt|.pth> (--mistake <perf.wav> --score <ref.wav> | --bundle <ALIGN bundle dir>)
#                     [--out <pred.mid>] [--profile auto|cuda|mps|cpu] [--batch-size N] [--max-length 1024]
#                     [--config-name config_align] [--verbose] [--dry-run]
#
#   --mistake/--score  performance ("mistake") and reference ("score") audio, any sample rate (resampled to 16 kHz)
#   --bundle           shorthand: <dir>/performance_audio.wav + <dir>/reference_audio.wav
#   --out              output MIDI (default $POLYTUNE_RUNS/infer/<name>/mix.mid); tracks are named
#                      extra / missing / correct (error classes 1/2/3)
#   --profile          device (auto = cuda > mps > cpu)
#   --verbose          keep InferenceHandler's very chatty stdout (default: written to <out>.log)
#
# Driver: baselines/common/polytune_infer.py (loads the model via Hydra compose of config_align).
set -euo pipefail

BASELINES=${BASELINE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export PYTHONPATH="$BASELINES/common${PYTHONPATH:+:$PYTHONPATH}"
REPO=${POLYTUNE_REPO:-$BASELINES/Polytune}
PY=${POLYTUNE_PYTHON:-$BASELINES/envs/polytune/bin/python}
RUNS=${POLYTUNE_RUNS:-$BASELINES/runs/polytune}
DRIVER=$BASELINES/common/polytune_infer.py

CKPT=""; MISTAKE=""; SCORE=""; BUNDLE=""; OUT=""
PROFILE=${POLYTUNE_PROFILE:-auto}
BATCH_SIZE=1; MAX_LENGTH=1024; CONFIG_NAME=config_align
VERBOSE=0; DRY_RUN=0

usage() { sed -n '2,/^set -euo/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'; }
die() { echo "[polytune_infer] error: $*" >&2; exit 2; }
abspath() { echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)        CKPT=$2; shift 2 ;;
    --mistake)     MISTAKE=$2; shift 2 ;;
    --score)       SCORE=$2; shift 2 ;;
    --bundle)      BUNDLE=$2; shift 2 ;;
    --out)         OUT=$2; shift 2 ;;
    --profile)     PROFILE=$2; shift 2 ;;
    --batch-size)  BATCH_SIZE=$2; shift 2 ;;
    --max-length)  MAX_LENGTH=$2; shift 2 ;;
    --config-name) CONFIG_NAME=$2; shift 2 ;;
    --verbose)     VERBOSE=1; shift ;;
    --dry-run)     DRY_RUN=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *)             die "unknown argument: $1" ;;
  esac
done

case "$PROFILE" in auto|cuda|mps|cpu) ;; *) die "--profile must be auto, cuda, mps or cpu" ;; esac
[[ -n "$CKPT" ]] || die "--ckpt is required"
[[ -f "$CKPT" ]] || die "checkpoint not found: $CKPT"
CKPT=$(abspath "$CKPT")
if [[ -n "$BUNDLE" ]]; then
  [[ -d "$BUNDLE" ]] || die "bundle dir not found: $BUNDLE"
  MISTAKE=$BUNDLE/performance_audio.wav
  SCORE=$BUNDLE/reference_audio.wav
fi
[[ -n "$MISTAKE" && -n "$SCORE" ]] || die "--mistake and --score (or --bundle) are required"
[[ -f "$MISTAKE" ]] || die "mistake wav not found: $MISTAKE"
[[ -f "$SCORE" ]] || die "score wav not found: $SCORE"
MISTAKE=$(abspath "$MISTAKE"); SCORE=$(abspath "$SCORE")
[[ -x "$PY" ]] || die "python not found: $PY"
[[ -f "$DRIVER" ]] || die "driver not found: $DRIVER"

if [[ -z "$OUT" ]]; then
  base=$(basename "$MISTAKE"); base=${base%.*}
  case "$base" in mix|performance_audio) name=$(basename "$(dirname "$MISTAKE")") ;; *) name=$base ;; esac
  OUT=$RUNS/infer/$name/mix.mid
fi
mkdir -p "$(dirname "$OUT")"
OUT=$(abspath "$OUT")

export MPLBACKEND=Agg
export WANDB_MODE=offline
if [[ $PROFILE == mps ]]; then export PYTORCH_ENABLE_MPS_FALLBACK=1; fi
if [[ $PROFILE == cpu ]]; then export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}; fi

CMD=("$PY" "$DRIVER" --repo "$REPO" --config-name "$CONFIG_NAME" --ckpt "$CKPT"
     --mistake "$MISTAKE" --score "$SCORE" --out "$OUT" --device "$PROFILE"
     --batch-size "$BATCH_SIZE" --max-length "$MAX_LENGTH")
if (( VERBOSE )); then CMD+=(--verbose); fi

echo "[polytune_infer] checkpoint : $CKPT"
echo "[polytune_infer] mistake    : $MISTAKE"
echo "[polytune_infer] score      : $SCORE"
echo "[polytune_infer] output     : $OUT   (device profile: $PROFILE)"
printf '[polytune_infer] command    :'; printf ' %q' "${CMD[@]}"; printf '\n'
if (( DRY_RUN )); then exit 0; fi
exec "${CMD[@]}"
