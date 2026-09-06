# ALIGN error detector

Score-informed clarinet error detector for ALIGN bundles. Official gold is schema **1.2**: a contiguous clean-score melody (`score_part`, `pitches`, `note_ids`) on `verified_score.musicxml`, first pass only. Official synth score is **set-F1** (`eval-melodies`): exclusive 1-1 matching of predicted vs gold pitch lists as the same event, not timestamp IoU and not slice/containment.

Two models share that gold and that metric:

| Track | CLI | Object | Output |
|-------|-----|--------|--------|
| **A** four-stage pipeline | `run` / `train-stages` | Time crops, then attach a score-part | `pipeline_pred.json` schema 1.2 |
| **B** melody-first | `run-melody` / `train-melody` | Per-note type on the clean score, then contiguous runs | `melody_pred.json` schema 1.2 |

Legacy **RUMAA-lite** (`train` / `infer`) stays available. It did not learn error classes; prefer A or B.

Never read `performance_score.musicxml`. The gold score is always the clean `verified_score.musicxml`. Written MIDI is the target; Bb clarinet audio sounds two semitones lower.

## Schema 1.2 gold

The on-disk bundle is unchanged (`verified_score.musicxml`, `performance_audio.wav`, `performance_mel.npy`, `labels.json`). What changed is label meaning ([methodology.md](../methodology.md) §10–11):

- Gold is the **contiguous clean-score melody**, not the wall-clock fault.
- **First-pass only.** Comments with `repeated pass` or `(pass N)` are not gold. One `repetition` label is kept.
- **`extra_note` gold is the neighbors** on the clean score, not the inserted note.
- **`repetition`** has `extra_copies` (`1` = two plays, `2` = three) and a **0.2–1.0 s silent gap** before the replay. `repeats_label_range` is the first-pass source. The gap is not part of the gold melody.
- Official synth metric: Hungarian 1-1 set matching. A pair matches only if the pitch lists are the same event (equal, or LCS-Dice ≥ 0.80 with length ratio ≥ 0.60). A slice of gold, or a pred that contains gold, does **not** match. Type is ignored. Empty vs empty scores 1.

Trainers drop repeated-pass copies (`first_pass_labels` in `alignmodel.stages.gold`). Timbre types (`click`, `squeak`, `bad_start`, `bad_timbre`) are ignored by both trainers.

## Four-stage pipeline (Model A)

| Stage | What it does | ALIGN types |
|------|--------|-------------|
| 1 Restarts | Score graph, silence/hold + chroma copy cuts, partial-match beam. `extra_copies==2` is two sequential replays; the silent gap is a hard negative | `repetition` with `repeats_label_range` and `extra_copies` |
| 2 Edits | On each unfolded segment: chroma DTW, Match / Insert / Delete / Substitute. Extra crops use the neighbor window stored on first-pass gold | `missed_note`, `extra_note`, `wrong_note`, `intonation_error` |
| 3 Rhythm | EWMA + far-window, or `RhythmNet` if `stage3.pt` is present | `rhythm_error` |
| 4 Timbre | Untrained heuristics on extra/wrong crops (off by default) | `squeak`, `bad_start`, `bad_timbre` |

After stages 1–3 the pipeline maps each scored label onto the clean score (extra → neighbors, then pad 1–2 notes) and writes `score_part` / `pitches` / `note_ids`. Consecutive restarts that share a source become one `repetition` with `extra_copies` 1 or 2.

Stages share one `PipelineState`. `EditCropNet` module names stay the same so an old `stage2.pt` still loads.

## Melody-first (Model B)

A separate checkpoint family. It does **not** emit one label per DTW pair.

- Encoders: reuse `AudioEncoder` / `ScoreEncoder` / `HierarchicalFusion` on `performance_mel.npy` and sounding notes from `verified_score.musicxml`.
- Per-note type: `match`, `miss`, `wrong`, `extra`, `rhythm`, `intonation`, `repetition`.
- Clip-level `extra_copies` ∈ {0, 1, 2}.
- Decode contiguous same-type non-match runs to schema 1.2 labels. Times are the mapped note onsets.

Empty clips (all `match`) are valid and score 1.0 against empty gold.

## Setup

Same conda env as DataCreate (`MusicEval`):

```powershell
cd "D:\stuff\Audio Evaluation\ALIGN"
conda activate MusicEval
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ./DataCreate
pip install -e ./align-model
```

## Run

```powershell
align-model run --sample ".\synth-pipeline\output\synth_gen_0010"
align-model run --sample ".\synth-pipeline\output\synth_gen_0010" --timbre
align-model run-melody --sample ".\synth-pipeline\output\synth_gen_0010" --ckpt ".\align-model\runs\melody\best.pt"
align-model smoke --data ".\synth-pipeline\output"
align-model eval-melodies --data ".\synth-pipeline\output"
align-model eval-melodies --data ".\synth-pipeline\output" --pred melody_pred.json
```

`eval-melodies` is the official synth metric. Gold is the schema **1.2** `pitches` list on each first-pass label. Predictions that already store `pitches` are used as-is; time-only preds are mapped onto the clean score with the same core-plus-pad rules (extras use the notes before and after the insert).

- Precision = matched predictions / all predictions (each pred ≤ 1 gold)
- Recall = matched golds / all golds (each gold ≤ 1 pred)
- Headline = F1 (`melody_f1`)

`smoke` prints that score first; timestamp `repetition_iou` is still included (first-pass gold only).

`run` writes `pipeline_pred.json`. `run-melody` writes `melody_pred.json`. Both are schema **1.2**. Stage 4 is opt-in via `--timbre`.

## Train

Learned heads on `performance_mel.npy`. Default train root is `synth-pipeline/output` (schema 1.2 synth). DataCreate `001–020` is a real-take check; most of those folders are still 1.1 / sparsely converted.

### Model A — stages 1–3

| Stage | Model | Supervision |
|------|--------|-------------|
| 1 | Siamese `RestartScorer` on paired mel spans | first-pass `repetition` vs source; split `extra_copies==2`; hard-neg the silent gap |
| 2 | `EditCropNet` on mel crops | first-pass match / miss / extra / wrong / intonation (extra times are neighbor windows) |
| 3 | `RhythmNet` crop + aux | first-pass `rhythm_error` spans |

```powershell
align-model train-stages --data ".\synth-pipeline\output" --out ".\align-model\runs\stages" --epochs 8 --device cuda
align-model run --sample ".\synth-pipeline\output\synth_gen_0010" --weights ".\align-model\runs\stages" --device cuda
```

### Model B — melody-first

```powershell
align-model train-melody --data ".\synth-pipeline\output" --out ".\align-model\runs\melody" --epochs 8 --device cuda
align-model run-melody --sample ".\synth-pipeline\output\synth_gen_0010" --ckpt ".\align-model\runs\melody\best.pt"
align-model eval-melodies --data ".\synth-pipeline\output" --pred melody_pred.json
```

### Legacy RUMAA-lite

Joint transformer in `align-model/runs/rumaa-lite/`. Val error acc stayed 0; prefer A or B. Training now also uses first-pass gold only.

```powershell
align-model train --data ".\synth-pipeline\output" --out ".\align-model\runs\rumaa-lite" --epochs 20 --batch-size 4 --device cuda
align-model infer --ckpt ".\align-model\runs\rumaa-lite\best.pt" --sample ".\synth-pipeline\output\synth_gen_0010" --device cuda
```

Each sample needs `verified_score.musicxml` and `performance_audio.wav` (plus `performance_mel.npy` for training).
