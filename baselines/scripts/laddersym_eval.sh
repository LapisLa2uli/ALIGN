#!/usr/bin/env bash
# Evaluate a LadderSym checkpoint on the test split of an ALIGN export (MAESTRO-E layout) with the
# official test_laddersym.py (+ evaluate_errors.py), then run the shared eval bridge if present.
#
# Usage:
#   laddersym_eval.sh --ckpt <model.ckpt|.pt|.pth> --data <DATA_ROOT> [--split-json <split.json>]
#                     [--prompted|--unprompted] [--split test] [--profile cuda|mps|cpu] [--tag TAG]
#                     [--first-n K] [--max-length N] [--bundles <ALIGN bundle root>]... [--randomize-prompt]
#                     [-- <extra hydra overrides...>]
#
# Defaults: --prompted, --split test, --tag align_<split>, device auto (cuda -> mps -> cpu),
#           deterministic prompts (LADDERSYM_DETERMINISTIC_PROMPT=1; pass --randomize-prompt for the
#           original shuffled-prompt behaviour).
# Output:   $B/runs/laddersym/eval_<tag>/<tag>/<track_id>/mix.mid
#           (tracks named extra/missing/correct, note.instrument 1/2/3) and per-class onset F1 on stdout
#           (also tee'd to eval.log).
# Note:     test_laddersym.py hard-codes split="test".  For --split train|validation this script writes a
#           derived split json in the run dir that relabels the requested split as "test".
#           --max-length N sets eval.max_length (new decoder tokens per 2.048 s segment; default 1024 =
#           upstream).  Decoding has no KV cache, so SMOKE TESTS on CPU should pass e.g. 16.
#           --bundles gives the ALIGN bundle root(s) for common/eval_bridge.py (repeatable); default: the
#           distinct parents of the "bundle" paths in <DATA_ROOT>/manifest.json.  Bridge call:
#           eval_bridge.py --pred-dir <run>/<tag> --bundles <root>... [--ids-from <split json> --split test
#           when no --first-n] --out <run>/eval_bridge.json
set -euo pipefail

BASELINES=${BASELINE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export PYTHONPATH="$BASELINES/common${PYTHONPATH:+:$PYTHONPATH}"
REPO=${LADDERSYM_REPO:-$BASELINES/LadderSym}
PY=${LADDERSYM_PYTHON:-$BASELINES/envs/laddersym/bin/python}
RUNS=${LADDERSYM_RUNS:-$BASELINES/runs/laddersym}
BRIDGE=$BASELINES/common/eval_bridge.py

CKPT=""
DATA_ROOT="${ALIGN_BASELINE_ROOT:-}"
SPLIT_JSON="${ALIGN_BASELINE_SPLIT_JSON:-}"
VARIANT="prompted"
SPLIT="test"
PROFILE="auto"
TAG=""
FIRST_N=""
MAX_LENGTH=""
BUNDLES=()
DETERMINISTIC=1
EXTRA=()

usage() { sed -n '2,26p' "$0"; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)        CKPT="$2"; shift 2 ;;
    --data)        DATA_ROOT="$2"; shift 2 ;;
    --split-json)  SPLIT_JSON="$2"; shift 2 ;;
    --prompted)    VARIANT="prompted"; shift ;;
    --unprompted)  VARIANT="unprompted"; shift ;;
    --split)       SPLIT="$2"; shift 2 ;;
    --profile)     PROFILE="$2"; shift 2 ;;
    --tag)         TAG="$2"; shift 2 ;;
    --first-n)     FIRST_N="$2"; shift 2 ;;
    --max-length)  MAX_LENGTH="$2"; shift 2 ;;
    --bundles)     BUNDLES+=("$2"); shift 2 ;;
    --randomize-prompt) DETERMINISTIC=0; shift ;;
    -h|--help)     usage 0 ;;
    --)            shift; EXTRA=("$@"); break ;;
    *)             EXTRA+=("$1"); shift ;;
  esac
done

[[ -n "$CKPT" ]] || { echo "error: --ckpt is required" >&2; usage 2; }
[[ -f "$CKPT" ]] || { echo "error: checkpoint not found: $CKPT" >&2; exit 2; }
CKPT="$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")"
[[ -n "$DATA_ROOT" ]] || { echo "error: --data <DATA_ROOT> (or ALIGN_BASELINE_ROOT) is required" >&2; usage 2; }
DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"
if [[ -z "$SPLIT_JSON" ]]; then
  SPLIT_JSON="$DATA_ROOT/split.json"
  [[ ! -f "$DATA_ROOT/split.audited.json" ]] || SPLIT_JSON="$DATA_ROOT/split.audited.json"
fi
[[ -f "$SPLIT_JSON" ]] || { echo "error: split json not found: $SPLIT_JSON" >&2; exit 2; }
SPLIT_JSON=$(cd "$(dirname "$SPLIT_JSON")" && pwd)/$(basename "$SPLIT_JSON")
case "$SPLIT" in train|validation|test) ;; *) echo "error: --split must be train|validation|test" >&2; exit 2 ;; esac
[[ -x "$PY" ]] || { echo "error: venv python not found: $PY" >&2; exit 2; }

for i in "${!BUNDLES[@]}"; do BUNDLES[$i]=$(cd "${BUNDLES[$i]}" && pwd); done
if [[ "$VARIANT" == "prompted" ]]; then CONFIG=config_align_prompted; else CONFIG=config_align; fi
[[ -n "$TAG" ]] || TAG="align_${SPLIT}"
RUN_DIR="$RUNS/eval_${TAG}"
mkdir -p "$RUN_DIR"

if [[ "$SPLIT" != "test" ]]; then
  DERIVED="$RUN_DIR/split_${SPLIT}_as_test.json"
  "$PY" - "$SPLIT_JSON" "$SPLIT" "$DERIVED" <<'PYEOF'
import json, sys
src, want, dst = sys.argv[1:4]
d = json.load(open(src))
d["split"] = {k: ("test" if v == want else "train") for k, v in d["split"].items()}
json.dump(d, open(dst, "w"))
print(f"derived split json: {dst} ({sum(v == 'test' for v in d['split'].values())} tracks relabelled test)")
PYEOF
  SPLIT_JSON="$DERIVED"
fi

export ALIGN_BASELINE_ROOT="$DATA_ROOT"
export ALIGN_BASELINE_SPLIT_JSON="$SPLIT_JSON"
export WANDB_MODE=offline
export MPLBACKEND=Agg
export PYTORCH_ENABLE_MPS_FALLBACK=1
export LADDERSYM_DETERMINISTIC_PROMPT="$DETERMINISTIC"
case "$PROFILE" in
  auto) ;;                                  # patched test_laddersym.py picks cuda -> mps -> cpu
  cuda|mps|cpu) export LADDERSYM_DEVICE="$PROFILE" ;;
  *) echo "error: --profile must be cuda|mps|cpu" >&2; exit 2 ;;
esac

OVERRIDES=( "hydra.run.dir=$RUN_DIR" "path='$CKPT'" "eval.eval_dataset=MAESTRO" "eval.exp_tag_name=$TAG" )
[[ -n "$FIRST_N" ]] && OVERRIDES+=( "eval.eval_first_n_examples=$FIRST_N" )
[[ -n "$MAX_LENGTH" ]] && OVERRIDES+=( "eval.max_length=$MAX_LENGTH" )
OVERRIDES+=( "${EXTRA[@]+"${EXTRA[@]}"}" )

echo "== LadderSym eval"
echo "   variant   : $VARIANT ($CONFIG)"
echo "   ckpt      : $CKPT"
echo "   data root : $DATA_ROOT"
echo "   split     : $SPLIT  (json: $SPLIT_JSON)"
echo "   device    : ${LADDERSYM_DEVICE:-auto}"
echo "   run dir   : $RUN_DIR   (pred MIDIs in $RUN_DIR/$TAG/<track_id>/mix.mid)"
echo "   overrides : ${OVERRIDES[*]}"
[[ ! -e "$RUN_DIR/$TAG" ]] || { echo "error: predictions already exist; choose a new --tag" >&2; exit 2; }
cd "$REPO"
set -x
set +e
"$PY" test_laddersym.py --config-name "$CONFIG" "${OVERRIDES[@]}" 2>&1 | tee "$RUN_DIR/eval.log"
STATUS=${PIPESTATUS[0]}
set -e
set +x
PRED_DIR="$RUN_DIR/$TAG"
N_PRED=0
if [[ -d "$PRED_DIR" ]]; then N_PRED=$(find "$PRED_DIR" -name mix.mid | wc -l | tr -d ' '); fi
echo "== test_laddersym.py exit status: $STATUS; $N_PRED predicted MIDI(s) in $PRED_DIR/<track_id>/mix.mid (log: $RUN_DIR/eval.log)"
if [[ "$N_PRED" -eq 0 && "$STATUS" -eq 0 ]]; then
  # InferenceHandler.inference() swallows exceptions (traceback on stdout) and test_laddersym.py still exits 0.
  echo "== ERROR: no prediction was written -- look for 'Traceback' in $RUN_DIR/eval.log" >&2
  STATUS=1
fi

[[ "$STATUS" -eq 0 ]] || exit "$STATUS"
"$PY" "$BASELINES/common/evaluate_notes.py" --data "$ALIGN_BASELINE_ROOT" \
  --pred-dir "$PRED_DIR" --out "$RUN_DIR/note_metrics.json" | tee "$RUN_DIR/note_metrics.log"

if [[ -f "$BRIDGE" && "$N_PRED" -gt 0 ]]; then
  # Bundle roots: --bundles, else the distinct parents of manifest.json "bundle" paths.
  if [[ ${#BUNDLES[@]} -eq 0 && -f "$DATA_ROOT/manifest.json" ]]; then
    while IFS= read -r line; do BUNDLES+=("$line"); done < <("$PY" - "$DATA_ROOT/manifest.json" <<'PYEOF'
import json, os, sys
m = json.load(open(sys.argv[1]))
tracks = m.get("tracks", {})
tracks = tracks.values() if isinstance(tracks, dict) else tracks
roots = sorted({os.path.dirname(t["bundle"]) for t in tracks if t.get("bundle")})
print("\n".join(r for r in roots if os.path.isdir(r)))
PYEOF
)
  fi
  if [[ ${#BUNDLES[@]} -eq 0 ]]; then
    echo "== eval bridge not run: no ALIGN bundle roots (pass --bundles <root> or keep <DATA_ROOT>/manifest.json)"
    exit "$STATUS"
  fi
  BRIDGE_ARGS=( --pred-dir "$PRED_DIR" )
  for b in "${BUNDLES[@]}"; do BRIDGE_ARGS+=( --bundles "$b" ); done
  # With --first-n only a subset was predicted: evaluate what exists instead of flagging the rest as empty.
  BRIDGE_ARGS+=( --ids-from "$PRED_DIR/evaluated_split.json" --split test )
  BRIDGE_ARGS+=( --out "$RUN_DIR/eval_bridge.json" )
  echo "== eval bridge: $BRIDGE ${BRIDGE_ARGS[*]}"
  set +e
  "$PY" "$BRIDGE" "${BRIDGE_ARGS[@]}" 2>&1 | tee "$RUN_DIR/bridge.log"
  BSTATUS=${PIPESTATUS[0]}
  set -e
  if [[ "$BSTATUS" -ne 0 ]]; then
    echo "warning: eval_bridge.py failed (exit $BSTATUS; see $RUN_DIR/bridge.log; predictions are in $PRED_DIR)" >&2
    [[ "$STATUS" -eq 0 ]] && STATUS=$BSTATUS
  else
    echo "== eval bridge results: $RUN_DIR/eval_bridge.json"
  fi
else
  echo "== eval bridge not run ($BRIDGE missing or no predictions)"
fi
exit "$STATUS"
