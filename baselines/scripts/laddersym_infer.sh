#!/usr/bin/env bash
# Run LadderSym inference (laddersym_test_inference.py) on one piece folder, a folder of piece
# folders, or explicit files.  Writes a MIDI whose tracks are named extra/missing/correct.
#
# Usage:
#   laddersym_infer.sh --ckpt <model.ckpt> --piece-dir <dir with score.wav mistake.wav score.mid>
#   laddersym_infer.sh --ckpt <model.ckpt> --pieces-root <dir of piece dirs> [--pieces a,b,c]
#   laddersym_infer.sh --ckpt <model.ckpt> --mistake <wav> --score <wav> [--prompt <score.mid>] --out-dir <dir>
#   common options: [--prompted|--unprompted] [--out-name NAME.mid] [--overwrite] [--profile cuda|mps|cpu]
#                   [--randomize-prompt] [--batch-size N] [--max-length N] [-- <extra hydra overrides...>]
#
# Notes: laddersym_test_inference.py only loads Lightning .ckpt files (not .pt/.pth).  The prompt
#        (score.mid) is required for --prompted and ignored for --unprompted.  Output goes to the piece
#        dir (or --out-dir) as <out-name> (default laddersym_MT3Net_output.mid).  Prompts are decoded
#        deterministically unless --randomize-prompt is given.  --max-length N caps the new decoder
#        tokens per 2.048 s segment (default 1024; decoding has no KV cache -> use e.g. 16 for smoke runs).
set -euo pipefail

BASELINES=${BASELINE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export PYTHONPATH="$BASELINES/common${PYTHONPATH:+:$PYTHONPATH}"
REPO=${LADDERSYM_REPO:-$BASELINES/LadderSym}
PY=${LADDERSYM_PYTHON:-$BASELINES/envs/laddersym/bin/python}
RUNS=${LADDERSYM_RUNS:-$BASELINES/runs/laddersym}

CKPT=""; PIECE_DIR=""; PIECES_ROOT=""; PIECES=""; MISTAKE=""; SCORE=""; PROMPT=""; OUT_DIR=""
VARIANT="prompted"; OUT_NAME=""; OVERWRITE=0; PROFILE="auto"; DETERMINISTIC=1; BATCH=""; MAX_LENGTH=""
EXTRA=()

usage() { sed -n '2,17p' "$0"; exit "${1:-0}"; }
abspath() { local d; d="$(cd "$(dirname "$1")" && pwd)"; echo "$d/$(basename "$1")"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)         CKPT="$2"; shift 2 ;;
    --piece-dir)    PIECE_DIR="$2"; shift 2 ;;
    --pieces-root)  PIECES_ROOT="$2"; shift 2 ;;
    --pieces)       PIECES="$2"; shift 2 ;;
    --mistake)      MISTAKE="$2"; shift 2 ;;
    --score)        SCORE="$2"; shift 2 ;;
    --prompt)       PROMPT="$2"; shift 2 ;;
    --out-dir)      OUT_DIR="$2"; shift 2 ;;
    --prompted)     VARIANT="prompted"; shift ;;
    --unprompted)   VARIANT="unprompted"; shift ;;
    --out-name)     OUT_NAME="$2"; shift 2 ;;
    --overwrite)    OVERWRITE=1; shift ;;
    --profile)      PROFILE="$2"; shift 2 ;;
    --randomize-prompt) DETERMINISTIC=0; shift ;;
    --batch-size)   BATCH="$2"; shift 2 ;;
    --max-length)   MAX_LENGTH="$2"; shift 2 ;;
    -h|--help)      usage 0 ;;
    --)             shift; EXTRA=("$@"); break ;;
    *)              EXTRA+=("$1"); shift ;;
  esac
done

[[ -n "$CKPT" ]] || { echo "error: --ckpt is required" >&2; usage 2; }
[[ -f "$CKPT" ]] || { echo "error: checkpoint not found: $CKPT" >&2; exit 2; }
[[ "$CKPT" == *.ckpt ]] || { echo "error: laddersym_test_inference.py only loads .ckpt files (got $CKPT)" >&2; exit 2; }
CKPT="$(abspath "$CKPT")"
[[ -x "$PY" ]] || { echo "error: venv python not found: $PY" >&2; exit 2; }

if [[ "$VARIANT" == "prompted" ]]; then CONFIG=config_align_prompted; else CONFIG=config_align; fi
RUN_DIR="$RUNS/infer_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"

export WANDB_MODE=offline
export MPLBACKEND=Agg
export PYTORCH_ENABLE_MPS_FALLBACK=1
export LADDERSYM_DETERMINISTIC_PROMPT="$DETERMINISTIC"
case "$PROFILE" in
  auto) ;;
  cuda|mps|cpu) export LADDERSYM_DEVICE="$PROFILE" ;;
  *) echo "error: --profile must be cuda|mps|cpu" >&2; exit 2 ;;
esac

OVERRIDES=( "hydra.run.dir=$RUN_DIR" "path='$CKPT'" )
[[ -n "$OUT_NAME" ]] && OVERRIDES+=( "output_mid_name=$OUT_NAME" )
[[ "$OVERWRITE" == "1" ]] && OVERRIDES+=( "overwrite=true" )
[[ -n "$BATCH" ]] && OVERRIDES+=( "+batch_size=$BATCH" )
[[ -n "$MAX_LENGTH" ]] && OVERRIDES+=( "+max_length=$MAX_LENGTH" )

if [[ -n "$PIECE_DIR" ]]; then
  [[ -d "$PIECE_DIR" ]] || { echo "error: --piece-dir not found: $PIECE_DIR" >&2; exit 2; }
  PIECE_DIR="$(cd "$PIECE_DIR" && pwd)"
  for f in score.wav mistake.wav; do [[ -f "$PIECE_DIR/$f" ]] || { echo "error: missing $PIECE_DIR/$f" >&2; exit 2; }; done
  if [[ "$VARIANT" == "prompted" && ! -f "$PIECE_DIR/score.mid" ]]; then echo "error: prompted inference needs $PIECE_DIR/score.mid" >&2; exit 2; fi
  OVERRIDES+=( "dataset_dir=$(dirname "$PIECE_DIR")" "pieces=[$(basename "$PIECE_DIR")]" )
  [[ "$VARIANT" == "unprompted" ]] && OVERRIDES+=( "prompt_filename=null" )
elif [[ -n "$PIECES_ROOT" ]]; then
  [[ -d "$PIECES_ROOT" ]] || { echo "error: --pieces-root not found: $PIECES_ROOT" >&2; exit 2; }
  OVERRIDES+=( "dataset_dir=$(cd "$PIECES_ROOT" && pwd)" )
  [[ -n "$PIECES" ]] && OVERRIDES+=( "pieces=[$PIECES]" )
  [[ "$VARIANT" == "unprompted" ]] && OVERRIDES+=( "prompt_filename=null" )
else
  [[ -n "$MISTAKE" && -n "$SCORE" && -n "$OUT_DIR" ]] || { echo "error: need --piece-dir, --pieces-root, or --mistake/--score/--out-dir" >&2; usage 2; }
  [[ -f "$MISTAKE" ]] || { echo "error: --mistake not found: $MISTAKE" >&2; exit 2; }
  [[ -f "$SCORE" ]] || { echo "error: --score not found: $SCORE" >&2; exit 2; }
  mkdir -p "$OUT_DIR"
  OVERRIDES+=( "+mistake_file=$(abspath "$MISTAKE")" "+score_file=$(abspath "$SCORE")" "+output_dir=$(cd "$OUT_DIR" && pwd)" )
  if [[ -n "$PROMPT" ]]; then
    [[ -f "$PROMPT" ]] || { echo "error: --prompt not found: $PROMPT" >&2; exit 2; }
    OVERRIDES+=( "+prompt_file=$(abspath "$PROMPT")" )
  elif [[ "$VARIANT" == "prompted" ]]; then
    echo "error: prompted inference needs --prompt <score.mid>" >&2; exit 2
  else
    OVERRIDES+=( "+prompt_file=null" )
  fi
fi

OVERRIDES+=( "${EXTRA[@]+"${EXTRA[@]}"}" )

echo "== LadderSym infer"
echo "   variant   : $VARIANT ($CONFIG)"
echo "   ckpt      : $CKPT"
echo "   device    : ${LADDERSYM_DEVICE:-auto}"
echo "   run dir   : $RUN_DIR"
echo "   overrides : ${OVERRIDES[*]}"
cd "$REPO"
set -x
exec "$PY" laddersym_test_inference.py --config-path "$REPO/config" --config-name "$CONFIG" "${OVERRIDES[@]}" 2>&1 | tee "$RUN_DIR/infer.log"
