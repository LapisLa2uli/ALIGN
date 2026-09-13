# ALIGN error detector

Score-informed clarinet error detector for ALIGN bundles. Official gold is schema **1.2**: a contiguous clean-score melody (`score_part`, `pitches`, `note_ids`) on `verified_score.musicxml`, first pass only. Official synth score is **set-F1** (`eval-melodies`): exclusive 1-1 matching of predicted vs gold pitch lists as the same event, then type (wrong type on a range hit is 0.5, not 1). Not timestamp IoU and not slice/containment.

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
- Official synth metric: Hungarian 1-1 set matching on pitch lists (the range). A pair is a range hit if the lists are the same event (equal, or LCS-Dice ≥ 0.80 with length ratio ≥ 0.60). A slice of gold, or a pred that contains gold, does **not** match. A range hit with the same error type scores 1; a range hit with the wrong type scores 0.5. Empty vs empty scores 1.

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

### Learned note alignment

Stage 3 can use a separate audio-to-note and note-to-score alignment checkpoint
instead of DataCreate DTW. The aligner supports inserted, deleted, substituted,
and replayed notes; replayed performance notes may point to the same written
score range as the first pass.

```powershell
python scripts/run_full_note_alignment.py

align-model run --sample "E:\output\synth_gen_0042" `
  --weights "align-model\runs\stages-random12k-typed" `
  --alignment-weights "align-model\runs\note-align-full-e-s365\weights"
```

The full training command uses both `E:\output` and
`E:\output_2k_rawdata`, creates a frozen stratified split, reconstructs only
strictly validated synth lineage maps, trains the note recognizer and aligner,
and evaluates against symbolic and DTW baselines. `performance_score.musicxml`
and synth MIDI are training/evaluation supervision only; inference reads
`performance_mel.npy` and `verified_score.musicxml`.

### Basic Pitch clarinet refiner

The v3 note path uses the official Basic Pitch 0.4.0 onset/note/contour maps,
PESTO fine pitch, and a compact PyTorch monophonic interval refiner. Its
checkpoint loader remains backward-compatible with v1/v2 `NoteFrameNet`
weights. The current training run is deliberately procedural-only while the
raw-derived corpus is being edited.

```powershell
# Isolated TensorFlow/Basic Pitch environment
python -m venv align-model\.venv-amt-bench --system-site-packages
align-model\.venv-amt-bench\Scripts\pip install `
  -r align-model\requirements-amt-benchmark.txt `
  --extra-index-url https://download.pytorch.org/whl/cu124

python align-model\scripts\run_basic_pitch_refiner.py `
  --max-train-samples 256 --max-val-samples 80 --max-test-samples 80
```

The strict manifest contains only `procedural12k` rows, requires exact
`note_map.json/rendered_notes`, and records the effective acoustic-to-written
transpose explicitly. Basic Pitch and PESTO caches are SHA-256 keyed, so edited
WAV files cannot reuse stale activations. A candidate refiner is promoted only
if it beats frozen Basic Pitch and passes note-count gates; otherwise
`weights/note_decoder.json` keeps frozen Basic Pitch canonical.

The current bounded promotion run retained frozen Basic Pitch: the refiner
reached 0.304 test-ID F1 versus 0.827 for Basic Pitch. PESTO reduced ordinary
cents error, but intonation-only error stayed near 60 cents because 98% of
labeled intonation regions in the audited procedural WAVs measure within 20
cents of zero. `regenerate_audio.py` now preserves pitch-bend labels in ordinary
rerenders; existing affected WAVs must be regenerated before cents training can
pass its promotion gate.

The basic production pipeline is transcription-first:

1. Basic Pitch transcribes the WAV once and stores the written notes.
2. Layer 1 finds repeated note phrases entirely inside that transcription.
3. Repeated notes reuse the source phrase's score mapping; Layer 2 emits only
   `wrong_note`, `extra_note`, and `missed_note`.
4. Layer 3 uses those same repetition-aware pairs for conservative duration
   errors.

Requesting Layer 2 or 3 automatically runs its prerequisites. Intonation
detection is disabled by default and filtered from final output.

Layer 1 also includes a small note-sequence scorer trained on 1,000 procedural
bundles. Long phrases use deterministic tempo-tolerant matching; the learned
scorer is only a one-note rescue when no long repetition was found, using the
fact that the repeated note looks like an insertion/extra relative to the
first pass.

```powershell
python align-model\scripts\train_note_repetition.py `
  --manifest align-model\runs\basic-pitch-refiner-procedural-s365\split.json `
  --out align-model\runs\basic-pitch-refiner-procedural-s365\weights\note_repetition.pt `
  --train-samples 1000 --val-samples 200
```

On 100 procedural test-ID clips, this hybrid Layer 1 scored 0.755 repetition
F1 (71/89 repetitions found, 99 predictions), versus 0.753 without the learned
one-note rescue (70/89 found, 97 predictions).

A contextual GRU note aligner was also trained on 10,000 procedural exact maps
after removing gold replay copies from the first-pass sequence. It reached
0.711 validation note-position accuracy, but failed the pipeline promotion
test: 0.0244 held-out mapping F1 versus 0.0256 for deterministic edit
alignment on the same 100 Basic Pitch clips. The checkpoint is retained under
`candidates/contextual_note_aligner.pt`; it is intentionally absent from
`weights/`, so production keeps the stronger deterministic aligner.

The corrected `E:\outputRaw_sf_10k` experiment is isolated under
`runs/contextual-aligner-outputRaw_sf-1k`. Basic Pitch was frozen; Layer 1 and
the contextual aligner were retrained on exactly 1,000 bundles, calibrated on
200 validation bundles, and evaluated on 200 separate test bundles. Contextual
mapping reached 0.470 F1 versus 0.450 for deterministic alignment (8,902 versus
8,728 exactly placed notes), so `weights/contextual_note_aligner.pt` remains
active for that dataset-specific pipeline.

The promoted aligner was then fine-tuned on the actual cached Basic Pitch +
Layer 1 sequences with a monotonic sequence-level alignment loss. This raised
200-clip mapping F1 from 0.470 to 0.503. Clarinet-range/harmonic filtering,
weak-onset same-pitch merge, tied-score-note collapse, and a lower score-note
deletion cost address spurious high notes, artificial splits, tied extensions,
and missing-note cascade failures. On transcription alone, cleanup improves
default F1 from 0.648 to 0.650 while reducing the note-count ratio from 1.077
to 1.018; validation-calibrated thresholds raise F1 to 0.657 at a 1.024 count
ratio. The calibrated decoder raises final mapping F1 to 0.509.

Two recovery alternatives were evaluated on the same 200 clips after
calibration: overlapping multi-start window consensus scored 0.463 F1 and
conservative dynamic revision scored 0.506. Both remain available through
`PipelineConfig.note_alignment_strategy`, but `contextual` remains the default
because it scored best.

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
