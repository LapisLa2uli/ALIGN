#!/usr/bin/env bash
# Train both baselines on this Mac (Apple Silicon, MPS) on data/align_v1.
#
#   scripts/mac_train_all.sh [DATA_ROOT] [EPOCHS]        (defaults: data/align_v1, 3)
#
# Reduced "Mac profile" (see README, section 3):
#   * event_length=256 (targets) and prompt_length=384 (LadderSym prompt) instead of the upstream 1024/1024.
#     ALIGN segments are monophonic: measured max 93 target / 193 prompt tokens per 2.048 s segment, so nothing
#     is truncated, while decoder memory/time drop ~4x (the 1024+1024 LadderSym prompted decoder OOMs on MPS).
#   * fp32 (MPS has no bf16 autocast in torch 2.3), num_workers=0 (spawned workers were 20x slower here),
#     batch 2 pieces (Polytune 2 x 4 rows, LadderSym 2 x 2 rows).
#   * The three runs are SEQUENTIAL: LadderSym-prompted -> Polytune -> LadderSym-unprompted.  Running two
#     models at once with batch 4 exhausted the 32 GB unified memory (23 GB swap, ~20x slower); one model at
#     batch 2 measured ~3 s/step (LadderSym prompted) and ~7 s/step (Polytune) on smoke data.
#   * warmup 1000 steps (upstream 4000 assumes ~113k steps); checkpoints every epoch (top-3 by val_loss + last).
# Logs: runs/polytune/<run>/train_stdout.log, runs/laddersym/<run>/train.log, plus runs/mac_train_all.log.
set -uo pipefail
B=${BASELINE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATA=${1:-$B/data/align_v1}
EPOCHS=${2:-3}
TS=$(date +%Y%m%d-%H%M%S)
LOG=$B/runs/mac_train_all.log
mkdir -p "$B/runs"
echo "[$(date)] mac_train_all start data=$DATA epochs=$EPOCHS tag=$TS (sequential, batch 2)" | tee -a "$LOG"

COMMON=(dataloader.train.num_workers=0 dataloader.val.num_workers=0 dataloader.train.batch_size=2 dataloader.val.batch_size=2 optim.warmup_steps=1000 event_length=256)

"$B/scripts/laddersym_train.sh" --profile mps --prompted --data "$DATA" --epochs "$EPOCHS" --run-name "laddersym_prompted_align_v1_mac_$TS" \
    -- "${COMMON[@]}" num_rows_per_batch=2 prompt_length=384 > /dev/null 2>&1
echo "[$(date)] laddersym prompted finished exit=$?" | tee -a "$LOG"

"$B/scripts/polytune_train.sh" --profile mps --data "$DATA" --epochs "$EPOCHS" --run-name "polytune_align_v1_mac_$TS" \
    -- "${COMMON[@]}" num_rows_per_batch=4 > /dev/null 2>&1
echo "[$(date)] polytune finished exit=$?" | tee -a "$LOG"

"$B/scripts/laddersym_train.sh" --profile mps --unprompted --data "$DATA" --epochs "$EPOCHS" --run-name "laddersym_unprompted_align_v1_mac_$TS" \
    -- "${COMMON[@]}" num_rows_per_batch=2 > /dev/null 2>&1
echo "[$(date)] laddersym unprompted finished exit=$?" | tee -a "$LOG"

echo "[$(date)] mac_train_all done" | tee -a "$LOG"
