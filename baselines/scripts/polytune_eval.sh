#!/usr/bin/env bash
# Evaluate a Polytune checkpoint on an ALIGN dataset (MAESTRO-E layout) with the official
# test_polytune.py (InferenceHandler + evaluate_errors.evaluate_main, per-class onset F1).
#
# Usage:
#   polytune_eval.sh --ckpt <path.ckpt|.pt|.pth> --data <DATA_ROOT> [--split-json <split.json>]
#                    [--split test|validation|train] [--profile auto|cuda|mps|cpu] [--tag NAME]
#                    [--first-n K] [--batch-size N] [--max-length N] [--bundles <ALIGN bundle root>]...
#                    [--dry-run] [-- <extra hydra overrides...>]
#
#   --ckpt        Lightning .ckpt or plain state_dict .pt/.pth (e.g. .../version_0/checkpoints/last.pt)
#   --data        <DATA_ROOT> (default $ALIGN_BASELINE_ROOT); --split-json defaults to <DATA_ROOT>/split.json
#   --split       which split.json split to evaluate (default test). test_polytune.py hard-codes
#                 split="test", so for other splits a derived split file <run dir>/split_as_test.json
#                 relabels the chosen split to "test" (everything else to "train").
#   --profile     device: auto (cuda > mps > cpu, default) or force one via POLYTUNE_DEVICE
#   --tag         eval.exp_tag_name; predictions land in <run dir>/<tag>/<track_id>/mix.mid
#                 (default <ckpt stem>_<split>); run dir = $POLYTUNE_RUNS/eval_<tag>
#   --first-n K   eval.eval_first_n_examples (NB: upstream takes a random window of K pieces)
#   --batch-size  eval.batch_size (segments per generate() call, default 1)
#   --max-length  eval.max_length: decoder tokens per 2.048 s segment (default 1024 = upstream).
#                 Greedy decoding has no KV cache, so SMOKE TESTS on CPU should pass e.g. 32.
#   --bundles     ALIGN bundle root(s) for common/eval_bridge.py (repeatable). Default: the distinct
#                 parent dirs of the "bundle" paths recorded in <DATA_ROOT>/manifest.json.
#   --dry-run     only print the resolved command
#
# Environment (optional): POLYTUNE_PYTHON, POLYTUNE_REPO, POLYTUNE_RUNS (see polytune_train.sh),
#   EVAL_BRIDGE_PYTHON / EVAL_BRIDGE_ARGS for baselines/common/eval_bridge.py (called if present;
#   default args: --pred-dir <run>/<tag> --bundles <root>... [--ids-from <split json> when no --first-n]
#   --split test --out <run>/eval_bridge.json).
set -euo pipefail

BASELINES=${BASELINE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export PYTHONPATH="$BASELINES/common${PYTHONPATH:+:$PYTHONPATH}"
REPO=${POLYTUNE_REPO:-$BASELINES/Polytune}
PY=${POLYTUNE_PYTHON:-$BASELINES/envs/polytune/bin/python}
RUNS=${POLYTUNE_RUNS:-$BASELINES/runs/polytune}
BRIDGE=$BASELINES/common/eval_bridge.py

CKPT=""
DATA=${ALIGN_BASELINE_ROOT:-}
SPLIT_JSON=${ALIGN_BASELINE_SPLIT_JSON:-}
SPLIT=test
PROFILE=${POLYTUNE_PROFILE:-auto}
TAG=""
FIRST_N=""
BATCH_SIZE=1
MAX_LENGTH=""
BUNDLES=()
DRY_RUN=0
EXTRA=()

usage() { sed -n '2,/^set -euo/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'; }
die() { echo "[polytune_eval] error: $*" >&2; exit 2; }
abspath() { echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)       CKPT=$2; shift 2 ;;
    --data)       DATA=$2; shift 2 ;;
    --split-json) SPLIT_JSON=$2; shift 2 ;;
    --split)      SPLIT=$2; shift 2 ;;
    --profile)    PROFILE=$2; shift 2 ;;
    --tag)        TAG=$2; shift 2 ;;
    --first-n)    FIRST_N=$2; shift 2 ;;
    --batch-size) BATCH_SIZE=$2; shift 2 ;;
    --max-length) MAX_LENGTH=$2; shift 2 ;;
    --bundles)    BUNDLES+=("$2"); shift 2 ;;
    --dry-run)    DRY_RUN=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    --)           shift; EXTRA+=("$@"); break ;;
    *)            EXTRA+=("$1"); shift ;;
  esac
done

case "$PROFILE" in auto|cuda|mps|cpu) ;; *) die "--profile must be auto, cuda, mps or cpu (got '$PROFILE')" ;; esac
case "$SPLIT" in train|validation|test) ;; *) die "--split must be train, validation or test (got '$SPLIT')" ;; esac
[[ -n "$CKPT" ]] || die "--ckpt is required"
[[ -f "$CKPT" ]] || die "checkpoint not found: $CKPT"
case "$CKPT" in *.ckpt|*.pt|*.pth) ;; *) die "--ckpt must end in .ckpt, .pt or .pth" ;; esac
CKPT=$(abspath "$CKPT")
[[ -n "$DATA" ]] || die "--data <DATA_ROOT> (or ALIGN_BASELINE_ROOT) is required"
[[ -d "$DATA" ]] || die "data root not found: $DATA"
DATA=$(cd "$DATA" && pwd)
if [[ -z "$SPLIT_JSON" ]]; then
  SPLIT_JSON="$DATA/split.json"
  [[ ! -f "$DATA/split.audited.json" ]] || SPLIT_JSON="$DATA/split.audited.json"
fi
[[ -f "$SPLIT_JSON" ]] || die "split json not found: $SPLIT_JSON"
SPLIT_JSON=$(abspath "$SPLIT_JSON")
[[ -x "$PY" ]] || die "python not found: $PY"
[[ -f "$REPO/test_polytune.py" ]] || die "Polytune repo not found: $REPO"

for i in "${!BUNDLES[@]}"; do BUNDLES[$i]=$(cd "${BUNDLES[$i]}" && pwd); done
if [[ -z "$TAG" ]]; then
  stem=$(basename "$CKPT"); stem=${stem%.*}
  TAG=$(printf '%s_%s' "$stem" "$SPLIT" | tr -c 'A-Za-z0-9_.-\n' '_')
fi
RUN_DIR=$RUNS/eval_$TAG
PRED_DIR=$RUN_DIR/$TAG

# test_polytune.py evaluates split_to_numbers["test"] only -> derive a split file if needed.
EFFECTIVE_SPLIT_JSON=$SPLIT_JSON
if [[ "$SPLIT" != test ]]; then
  EFFECTIVE_SPLIT_JSON=$RUN_DIR/split_as_test.json
  if (( ! DRY_RUN )); then
    mkdir -p "$RUN_DIR"
    "$PY" - "$SPLIT_JSON" "$SPLIT" "$EFFECTIVE_SPLIT_JSON" <<'EOF'
import json, sys
src, want, dst = sys.argv[1:4]
with open(src) as f:
    d = json.load(f)
d["split"] = {k: ("test" if v == want else "train") for k, v in d["split"].items()}
with open(dst, "w") as f:
    json.dump(d, f, indent=1)
print(f"[polytune_eval] wrote {dst}: {sum(v == 'test' for v in d['split'].values())} '{want}' entries relabelled as test")
EOF
  fi
fi

OVERRIDES=(
  --config-name config_align
  "path='$CKPT'"
  eval.eval_dataset=MAESTRO
  "eval.exp_tag_name=$TAG"
  "dataset.test.root_dir=$DATA"
  "dataset.test.split_json_path=$EFFECTIVE_SPLIT_JSON"
  dataset.test.split=test
  "eval.batch_size=$BATCH_SIZE"
  "hydra.run.dir=$RUN_DIR"
)
if [[ -n "$FIRST_N" ]]; then OVERRIDES+=("eval.eval_first_n_examples=$FIRST_N"); fi
if [[ -n "$MAX_LENGTH" ]]; then OVERRIDES+=("eval.max_length=$MAX_LENGTH"); fi
OVERRIDES+=(${EXTRA[@]+"${EXTRA[@]}"})

export ALIGN_BASELINE_ROOT=$DATA
export ALIGN_BASELINE_SPLIT_JSON=$EFFECTIVE_SPLIT_JSON
export WANDB_MODE=offline
export MPLBACKEND=Agg
export HYDRA_FULL_ERROR=1
if [[ $PROFILE != auto ]]; then export POLYTUNE_DEVICE=$PROFILE; fi
if [[ $PROFILE == mps ]]; then export PYTORCH_ENABLE_MPS_FALLBACK=1; fi
if [[ $PROFILE == cpu ]]; then export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}; fi

CMD=("$PY" test_polytune.py "${OVERRIDES[@]}")
echo "[polytune_eval] checkpoint : $CKPT"
echo "[polytune_eval] data       : $DATA   split: $SPLIT   split json: $EFFECTIVE_SPLIT_JSON"
echo "[polytune_eval] profile    : $PROFILE   run dir: $RUN_DIR"
echo "[polytune_eval] predictions: $PRED_DIR/<track_id>/mix.mid  (tracks named extra / missing / correct)"
printf '[polytune_eval] command    : cd %q &&' "$REPO"; printf ' %q' "${CMD[@]}"; printf '\n'
if (( DRY_RUN )); then exit 0; fi

mkdir -p "$RUN_DIR"
[[ ! -e "$RUN_DIR/$TAG" ]] || { echo "error: predictions already exist; choose a new --tag" >&2; exit 2; }
cd "$REPO"
set +e
"${CMD[@]}" 2>&1 | tee "$RUN_DIR/eval_stdout.log"
STATUS=${PIPESTATUS[0]}
set -e
echo "[polytune_eval] exit status: $STATUS   (stdout log: $RUN_DIR/eval_stdout.log)"
N_PRED=0
if [[ -d "$PRED_DIR" ]]; then N_PRED=$(find "$PRED_DIR" -name mix.mid | wc -l | tr -d ' '); fi
echo "[polytune_eval] $N_PRED predicted MIDIs in $PRED_DIR/<track_id>/mix.mid"
if [[ "$N_PRED" -eq 0 && "$STATUS" -eq 0 ]]; then
  # InferenceHandler.inference() swallows exceptions (traceback on stdout) and test_polytune.py still exits 0.
  echo "[polytune_eval] ERROR: no prediction was written -- look for 'Traceback' in $RUN_DIR/eval_stdout.log"
  STATUS=1
fi
echo "[polytune_eval] official per-class onset F1 (evaluate_errors.py) is printed above / in eval_stdout.log"

[[ "$STATUS" -eq 0 ]] || exit "$STATUS"
"$PY" "$BASELINES/common/evaluate_notes.py" --data "$ALIGN_BASELINE_ROOT" \
  --pred-dir "$PRED_DIR" --out "$RUN_DIR/note_metrics.json" | tee "$RUN_DIR/note_metrics.log"

if [[ -f "$BRIDGE" && "$N_PRED" -gt 0 ]]; then
  BRIDGE_PY=${EVAL_BRIDGE_PYTHON:-$PY}
  if [[ -n "${EVAL_BRIDGE_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    BRIDGE_ARGS=($EVAL_BRIDGE_ARGS)
  else
    # Bundle roots: --bundles, else the distinct parents of manifest.json "bundle" paths.
    if [[ ${#BUNDLES[@]} -eq 0 && -f "$DATA/manifest.json" ]]; then
      while IFS= read -r line; do BUNDLES+=("$line"); done < <("$PY" - "$DATA/manifest.json" <<'EOF2'
import json, os, sys
m = json.load(open(sys.argv[1]))
tracks = m.get("tracks", {})
tracks = tracks.values() if isinstance(tracks, dict) else tracks
roots = sorted({os.path.dirname(t["bundle"]) for t in tracks if t.get("bundle")})
print("\n".join(r for r in roots if os.path.isdir(r)))
EOF2
)
    fi
    if [[ ${#BUNDLES[@]} -eq 0 ]]; then
      echo "[polytune_eval] eval bridge not run: no ALIGN bundle roots (pass --bundles <root> or keep <DATA_ROOT>/manifest.json)"
      exit "$STATUS"
    fi
    BRIDGE_ARGS=(--pred-dir "$PRED_DIR")
    for b in "${BUNDLES[@]}"; do BRIDGE_ARGS+=(--bundles "$b"); done
    # With --first-n only a subset was predicted: evaluate what exists instead of flagging the rest as empty.
    BRIDGE_ARGS+=(--ids-from "$PRED_DIR/evaluated_split.json" --split test)
    BRIDGE_ARGS+=(--out "$RUN_DIR/eval_bridge.json")
  fi
  printf '[polytune_eval] eval bridge: %q' "$BRIDGE_PY"; printf ' %q' "$BRIDGE" "${BRIDGE_ARGS[@]}"; printf '\n'
  set +e
  "$BRIDGE_PY" "$BRIDGE" "${BRIDGE_ARGS[@]}" 2>&1 | tee "$RUN_DIR/eval_bridge_stdout.log"
  BSTATUS=${PIPESTATUS[0]}
  set -e
  if [[ "$BSTATUS" -ne 0 ]]; then
    echo "[polytune_eval] eval bridge failed (exit $BSTATUS, see $RUN_DIR/eval_bridge_stdout.log)"
    [[ "$STATUS" -eq 0 ]] && STATUS=$BSTATUS
  else
    echo "[polytune_eval] eval bridge results: $RUN_DIR/eval_bridge.json"
  fi
else
  echo "[polytune_eval] eval bridge not run ($BRIDGE missing or no predictions)"
fi
exit "$STATUS"
