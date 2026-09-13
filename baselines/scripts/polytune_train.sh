#!/usr/bin/env bash
# Train Polytune (AAAI 2025) on an ALIGN dataset converted to the MAESTRO-E layout.
#
# Usage:
#   polytune_train.sh [--data <DATA_ROOT>] [--split-json <split.json>] [--run-name NAME]
#                     [--profile cuda|mps|cpu] [--epochs N] [--batch-size N] [--resume <ckpt>]
#                     [--smoke] [--dry-run] [-- <extra hydra overrides...>]
#
#   --data        <DATA_ROOT> with label/, mistake/, score/, split.json (default: $ALIGN_BASELINE_ROOT)
#   --split-json  split file (default: $ALIGN_BASELINE_SPLIT_JSON, else <DATA_ROOT>/split.json)
#   --run-name    outputs go to $POLYTUNE_RUNS/<run-name> (default: smoke_<profile> or align_<timestamp>)
#   --profile     cuda (yaml defaults: gpu + bf16-mixed) | mps | cpu (32-true, strategy auto, num_workers 0)
#   --epochs      overrides num_epochs (feeds trainer.max_epochs and optim.num_epochs)
#   --batch-size  dataloader.{train,val}.batch_size (default 4; 1 with --smoke)
#   --resume      path=<x.ckpt>: validate, then continue training from that Lightning checkpoint
#   --smoke       1 epoch, 2 train batches, 1 val batch, batch_size 1, num_rows_per_batch 1, warmup 1
#   --dry-run     only print the resolved command
#   Anything after "--" (or any unrecognised token) is passed through as a Hydra override.
#
# Environment (optional):
#   POLYTUNE_PYTHON   interpreter (default baselines/envs/polytune/bin/python)
#   POLYTUNE_REPO     repo dir    (default baselines/Polytune)
#   POLYTUNE_RUNS     run root    (default baselines/runs/polytune)
#   POLYTUNE_PROFILE  default for --profile (default cuda)
#   OMP_NUM_THREADS   CPU threads (default 2 for --profile cpu)
#
# optim.num_steps_per_epoch is retained for config compatibility and diagnostics.
# The patched task uses Lightning estimated_stepping_batches for its LR schedule.
# Outputs in the run dir: .hydra/, train_polytune.log (hydra), train_stdout.log (this script),
# wandb/ (offline), polytune_ALIGN/<wandb-run-id>/checkpoints/*.ckpt (Lightning) and
# polytune_ALIGN/version_0/checkpoints/last.pt (plain state_dict saved by train_polytune.py).
set -euo pipefail

BASELINES=${BASELINE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export PYTHONPATH="$BASELINES/common${PYTHONPATH:+:$PYTHONPATH}"
REPO=${POLYTUNE_REPO:-$BASELINES/Polytune}
PY=${POLYTUNE_PYTHON:-$BASELINES/envs/polytune/bin/python}
RUNS=${POLYTUNE_RUNS:-$BASELINES/runs/polytune}

DATA=${ALIGN_BASELINE_ROOT:-}
SPLIT_JSON=${ALIGN_BASELINE_SPLIT_JSON:-}
RUN_NAME=""
PROFILE=${POLYTUNE_PROFILE:-cuda}
EPOCHS=""
BATCH_SIZE=""
RESUME=""
SMOKE=0
DRY_RUN=0
EXTRA=()

usage() { sed -n '2,/^set -euo/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'; }
die() { echo "[polytune_train] error: $*" >&2; exit 2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)       DATA=$2; shift 2 ;;
    --split-json) SPLIT_JSON=$2; shift 2 ;;
    --run-name)   RUN_NAME=$2; shift 2 ;;
    --profile)    PROFILE=$2; shift 2 ;;
    --epochs)     EPOCHS=$2; shift 2 ;;
    --batch-size) BATCH_SIZE=$2; shift 2 ;;
    --resume)     RESUME=$2; shift 2 ;;
    --smoke)      SMOKE=1; shift ;;
    --dry-run)    DRY_RUN=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    --)           shift; EXTRA+=("$@"); break ;;
    *)            EXTRA+=("$1"); shift ;;
  esac
done

case "$PROFILE" in cuda|mps|cpu) ;; *) die "--profile must be cuda, mps or cpu (got '$PROFILE')" ;; esac
[[ -n "$DATA" ]] || die "--data <DATA_ROOT> (or ALIGN_BASELINE_ROOT) is required"
[[ -d "$DATA" ]] || die "data root not found: $DATA"
DATA=$(cd "$DATA" && pwd)
if [[ -z "$SPLIT_JSON" ]]; then
  SPLIT_JSON="$DATA/split.json"
  [[ ! -f "$DATA/split.audited.json" ]] || SPLIT_JSON="$DATA/split.audited.json"
fi
[[ -f "$SPLIT_JSON" ]] || die "split json not found: $SPLIT_JSON"
SPLIT_JSON=$(cd "$(dirname "$SPLIT_JSON")" && pwd)/$(basename "$SPLIT_JSON")
[[ -x "$PY" ]] || die "python not found: $PY"
[[ -f "$REPO/train_polytune.py" ]] || die "Polytune repo not found: $REPO"
if [[ -n "$RESUME" ]]; then
  [[ -f "$RESUME" ]] || die "--resume checkpoint not found: $RESUME"
  [[ "$RESUME" == *.ckpt ]] || die "--resume expects a Lightning .ckpt (train_polytune.py treats .pth as MAE weights)"
  RESUME=$(cd "$(dirname "$RESUME")" && pwd)/$(basename "$RESUME")
fi

if (( SMOKE )); then
  EPOCHS=1
  : "${BATCH_SIZE:=1}"
  : "${RUN_NAME:=smoke_$PROFILE}"
else
  : "${BATCH_SIZE:=4}"
  : "${RUN_NAME:=align_$(date +%Y%m%d_%H%M%S)}"
fi
BS=$BATCH_SIZE
RUN_DIR=$RUNS/$RUN_NAME

# Informational ceil(n_train / batch_size); optimizer scheduling uses Lightning actual steps.
N_TRAIN=$("$PY" - "$SPLIT_JSON" <<'EOF'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
print(sum(1 for v in d["split"].values() if v == "train"))
EOF
)
STEPS=$(( (N_TRAIN + BS - 1) / BS ))
(( STEPS > 0 )) || STEPS=1

OVERRIDES=(
  --config-name config_align
  "hydra.run.dir=$RUN_DIR"
  "optim.num_steps_per_epoch=$STEPS"
  "dataloader.train.batch_size=$BS"
  "dataloader.val.batch_size=$BS"
)
if [[ -n "$EPOCHS" ]]; then OVERRIDES+=("num_epochs=$EPOCHS"); fi
if [[ -n "$RESUME" ]]; then OVERRIDES+=("path='$RESUME'"); fi
case "$PROFILE" in
  cuda) ;;
  mps|cpu)
    OVERRIDES+=(
      "trainer.accelerator=$PROFILE"
      trainer.precision=32-true
      trainer.strategy=auto
      dataloader.train.num_workers=0
      dataloader.val.num_workers=0
    ) ;;
esac
if (( SMOKE )); then
  OVERRIDES+=(
    +trainer.limit_train_batches=2
    +trainer.limit_val_batches=1
    trainer.num_sanity_val_steps=0
    num_rows_per_batch=1
    optim.warmup_steps=1
    modelcheckpoint.every_n_epochs=1
    trainer.check_val_every_n_epoch=1
  )
fi
OVERRIDES+=(${EXTRA[@]+"${EXTRA[@]}"})

export ALIGN_BASELINE_ROOT=$DATA
export ALIGN_BASELINE_SPLIT_JSON=$SPLIT_JSON
export WANDB_MODE=offline
export WANDB_SILENT=true
export WANDB_DIR=$RUN_DIR
export MPLBACKEND=Agg
export HYDRA_FULL_ERROR=1
if [[ $PROFILE == mps ]]; then export PYTORCH_ENABLE_MPS_FALLBACK=1; fi
if [[ $PROFILE == cpu ]]; then export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}; fi

CMD=("$PY" train_polytune.py "${OVERRIDES[@]}")
echo "[polytune_train] data       : $DATA"
echo "[polytune_train] split json : $SPLIT_JSON (n_train=$N_TRAIN, batch_size=$BS -> optim.num_steps_per_epoch=$STEPS)"
echo "[polytune_train] profile    : $PROFILE   run dir: $RUN_DIR"
echo "[polytune_train] env        : ALIGN_BASELINE_ROOT=$ALIGN_BASELINE_ROOT ALIGN_BASELINE_SPLIT_JSON=$ALIGN_BASELINE_SPLIT_JSON WANDB_MODE=offline MPLBACKEND=Agg"
printf '[polytune_train] command    : cd %q &&' "$REPO"; printf ' %q' "${CMD[@]}"; printf '\n'
if (( DRY_RUN )); then exit 0; fi

mkdir -p "$RUN_DIR"
cd "$REPO"
set +e
"${CMD[@]}" 2>&1 | tee "$RUN_DIR/train_stdout.log"
STATUS=${PIPESTATUS[0]}
set -e
echo "[polytune_train] exit status: $STATUS   (stdout log: $RUN_DIR/train_stdout.log)"
echo "[polytune_train] checkpoints under $RUN_DIR:"
find "$RUN_DIR" \( -name '*.ckpt' -o -name '*.pt' \) -print | sed 's/^/    /'
exit "$STATUS"
