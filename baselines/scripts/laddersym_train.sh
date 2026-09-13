#!/usr/bin/env bash
# Train LadderSym (prompted or unprompted) on ALIGN bundles exported to the MAESTRO-E layout.
#
# Usage:
#   laddersym_train.sh [--data <DATA_ROOT>] [--split-json <split.json>] [--run-name NAME]
#                      [--profile cuda|mps|cpu] [--prompted|--unprompted] [--epochs N]
#                      [--warm-start <ckpt>] [--smoke] [-- <extra hydra overrides...>]
#
# Defaults: --data ${ALIGN_BASELINE_ROOT} (must be set or passed), split json <DATA_ROOT>/split.json,
#           --prompted, --profile auto (cuda if available, else mps, else cpu),
#           run name laddersym_<prompted|unprompted>_<YYYYmmdd-HHMMSS>.
# Output:   $B/runs/laddersym/<run-name>/ (Hydra run dir;
#           Lightning checkpoints under lightning_logs/ or the ModelCheckpoint dir, the exported
#           state dict *.pt next to the last checkpoint, wandb/ offline files).
# Env:      WANDB_MODE=offline and MPLBACKEND=Agg are exported; ALIGN_BASELINE_ROOT / ALIGN_BASELINE_SPLIT_JSON
#           are exported for config/dataset/ALIGN.yaml.  LADDERSYM_DEBUG_PRINTS=1 restores per-step prints.
set -euo pipefail

BASELINES=${BASELINE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export PYTHONPATH="$BASELINES/common${PYTHONPATH:+:$PYTHONPATH}"
REPO=${LADDERSYM_REPO:-$BASELINES/LadderSym}
PY=${LADDERSYM_PYTHON:-$BASELINES/envs/laddersym/bin/python}
RUNS=${LADDERSYM_RUNS:-$BASELINES/runs/laddersym}

DATA_ROOT="${ALIGN_BASELINE_ROOT:-}"
SPLIT_JSON="${ALIGN_BASELINE_SPLIT_JSON:-}"
RUN_NAME=""
PROFILE="auto"
VARIANT="prompted"
EPOCHS=""
WARM_START=""
RESUME=""
DRY_RUN=0
BATCH_SIZE=""
SMOKE=0
EXTRA=()

usage() { sed -n '2,20p' "$0"; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)        DATA_ROOT="$2"; shift 2 ;;
    --split-json)  SPLIT_JSON="$2"; shift 2 ;;
    --run-name)    RUN_NAME="$2"; shift 2 ;;
    --profile)     PROFILE="$2"; shift 2 ;;
    --prompted)    VARIANT="prompted"; shift ;;
    --unprompted)  VARIANT="unprompted"; shift ;;
    --epochs)      EPOCHS="$2"; shift 2 ;;
    --resume)      RESUME="$2"; shift 2 ;;
    --dry-run)     DRY_RUN=1; shift ;;
    --batch-size)  BATCH_SIZE="$2"; shift 2 ;;
    --warm-start)  WARM_START="$2"; shift 2 ;;
    --smoke)       SMOKE=1; shift ;;
    -h|--help)     usage 0 ;;
    --)            shift; EXTRA=("$@"); break ;;
    *)             EXTRA+=("$1"); shift ;;   # bare hydra overrides are passed through
  esac
done

[[ -n "$DATA_ROOT" ]] || { echo "error: --data <DATA_ROOT> (or ALIGN_BASELINE_ROOT) is required" >&2; usage 2; }
DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"
if [[ -z "$SPLIT_JSON" ]]; then
  SPLIT_JSON="$DATA_ROOT/split.json"
  [[ ! -f "$DATA_ROOT/split.audited.json" ]] || SPLIT_JSON="$DATA_ROOT/split.audited.json"
fi
[[ -f "$SPLIT_JSON" ]] || { echo "error: split json not found: $SPLIT_JSON" >&2; exit 2; }
[[ -x "$PY" ]] || { echo "error: venv python not found: $PY" >&2; exit 2; }

if [[ "$PROFILE" == "auto" ]]; then
  PROFILE="$("$PY" -c 'import torch; print("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))' 2>/dev/null || echo cpu)"
fi
case "$PROFILE" in cuda|mps|cpu) ;; *) echo "error: --profile must be cuda|mps|cpu (got $PROFILE)" >&2; exit 2 ;; esac

if [[ "$VARIANT" == "prompted" ]]; then CONFIG=config_align_prompted; else CONFIG=config_align; fi
[[ -n "$RUN_NAME" ]] || RUN_NAME="laddersym_${VARIANT}_$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$RUNS/$RUN_NAME"
(( DRY_RUN )) || mkdir -p "$RUN_DIR"

export ALIGN_BASELINE_ROOT="$DATA_ROOT"
export ALIGN_BASELINE_SPLIT_JSON="$SPLIT_JSON"
export WANDB_MODE=offline
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1   # only matters for --profile mps

OVERRIDES=( "hydra.run.dir=$RUN_DIR" )

case "$PROFILE" in
  cuda)
    # keep the config defaults: accelerator gpu, bf16-mixed, strategy auto
    ;;
  mps|cpu)
    OVERRIDES+=( "trainer.accelerator=$PROFILE" "trainer.precision=32-true" "trainer.strategy=auto" "trainer.devices=1"
                 "dataloader.train.num_workers=0" "dataloader.val.num_workers=0" )
    ;;
esac

SPLIT_JSON=$(cd "$(dirname "$SPLIT_JSON")" && pwd)/$(basename "$SPLIT_JSON")
export ALIGN_BASELINE_SPLIT_JSON="$SPLIT_JSON"
[[ "$PROFILE" != cpu ]] || export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
[[ -z "$RESUME" || -z "$WARM_START" ]] || { echo "error: --resume and --warm-start are exclusive" >&2; exit 2; }
if [[ -n "$RESUME" ]]; then
  [[ -f "$RESUME" && "$RESUME" == *.ckpt ]] || { echo "error: --resume needs a Lightning .ckpt" >&2; exit 2; }
  OVERRIDES+=( "path='$(cd "$(dirname "$RESUME")" && pwd)/$(basename "$RESUME")'" "use_lightweight_checkpoint=False" )
fi
[[ -z "$BATCH_SIZE" ]] || OVERRIDES+=( "dataloader.train.batch_size=$BATCH_SIZE" "dataloader.val.batch_size=$BATCH_SIZE" )
[[ -n "$EPOCHS" ]] && OVERRIDES+=( "num_epochs=$EPOCHS" )
if [[ -n "$WARM_START" ]]; then
  [[ -f "$WARM_START" ]] || { echo "error: --warm-start checkpoint not found: $WARM_START" >&2; exit 2; }
  OVERRIDES+=( "path='$(cd "$(dirname "$WARM_START")" && pwd)/$(basename "$WARM_START")'" "use_lightweight_checkpoint=True" )
fi

if [[ "$SMOKE" == "1" ]]; then
  OVERRIDES+=( "num_epochs=1" "+trainer.limit_train_batches=2" "+trainer.limit_val_batches=1"
               "trainer.num_sanity_val_steps=0" "dataloader.train.batch_size=1" "dataloader.val.batch_size=1"
               "num_rows_per_batch=1" "optim.warmup_steps=1" "modelcheckpoint.every_n_epochs=1"
               "trainer.check_val_every_n_epoch=1" "trainer.log_every_n_steps=1" )
fi

OVERRIDES+=( "${EXTRA[@]+"${EXTRA[@]}"}" )

echo "== LadderSym train"
echo "   variant   : $VARIANT ($CONFIG)"
echo "   profile   : $PROFILE"
echo "   data root : $DATA_ROOT"
echo "   split json: $SPLIT_JSON"
echo "   run dir   : $RUN_DIR"
echo "   overrides : ${OVERRIDES[*]}"
(( ! DRY_RUN )) || exit 0
cd "$REPO"
set -x
exec "$PY" train_laddersym.py --config-name "$CONFIG" "${OVERRIDES[@]}" 2>&1 | tee "$RUN_DIR/train.log"
