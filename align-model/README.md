# ALIGN error detector

Score-informed clarinet error detector for ALIGN bundles. Official gold is schema **1.2** with canonical score-event identity on `verified_score.musicxml`, first pass only. Official scoring is exclusive one-to-one note-wise F1: same canonical location and type receives 1.0, same location with a different type receives 0.5, and a different location receives 0. Pitch lists and timestamps are validation/diagnostic data, not location identity.

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
- Official metric: Hungarian one-to-one matching on canonical score-event identity. Same identity/type scores 1, same identity/wrong type scores 0.5, and wrong identity scores 0. Equal pitches in another score location do not match. Empty vs empty scores 1.

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

Contextual-aligner training now preserves the manifest's split membership:
requests exceeding either split's available sample count fail instead of
moving validation examples into training. Missing or duplicate note maps and
overlap between training and validation also fail before training. Set
`--train-samples` and `--val-samples` to counts within the frozen splits.

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

`eval-melodies` is the official synth metric. Gold is schema **1.2** labels with canonical score-event identity (`score_part` / `note_ids` / audited extras). Predictions without a validated identity remain unmatched. Schema **1.1** gold without a projection is officially unavailable.

- Matching is exclusive one-to-one on canonical location, never pitch lists or timestamps
- Same location and type receives 1.0; same location with a different type receives 0.5; a wrong location receives 0
- Headline = official note-wise F1 (`melody_f1` / `official_note_wise`)
- Pitch-list similarity remains under `legacy_pitch_similarity_*`

`smoke` prints that score first; timestamp `repetition_iou` is still included as a diagnostic (first-pass gold only).

`run` writes `pipeline_pred.json`. `run-melody` writes `melody_pred.json`. Both are schema **1.2**. Stage 4 is opt-in via `--timbre`.

## Train

Full command-line flag reference, working-directory conventions, and parameter recipes: [TRAINING.md](TRAINING.md). The selected hyperparameter card for each model: [HYPERPARAMETERS.md](HYPERPARAMETERS.md). Discover live flags with `align-model train-stages --help` or `python align-model/scripts/<script>.py --help`.

Learned heads on `performance_mel.npy`. Default train root is `synth-pipeline/output` (schema 1.2 synth). DataCreate currently contains 94 real-take/fixture folders (`001`–`093` plus `demo_001`); all remain schema 1.1 and only 3 currently have `note_alignment_v2.json`.

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

## Model and dataset registry

This section records the repository's model history. Counts are bundle counts, not note counts, unless explicitly labeled otherwise. Smoke runs validate code paths and are not treated as trained model versions.

### Dataset versions used by ALIGN

| Dataset / manifest name | Origin and methodology | Frozen count or requested size | Use in this repository |
|---|---|---:|---|
| `1000dataexport` | Early procedural MusicXML with one planted error per clip | 1,000 bundles | Early RUMAA-lite and stage-model development |
| `procedural12k` / `E:\output` | Procedurally generated 8–16-measure melodies with multiple planted errors, repeats, schema 1.2 labels, and exact synthetic note lineage | 12,000 rows in the full manifest; strict refiner subset 11,488: 9,197 train, 1,138 val, 1,153 test-ID | Model A/B, NoteFrameNet, exact-map aligners, Basic Pitch/refiner experiments |
| `raw2k` / `output_2k_rawdata` | Random 8–16-measure snippets from uploaded `RawData/Score` MusicXML, then synthetic corruption/rendering | 2,100 rows in the frozen full manifest: 1,268 train, 140 val, 692 test-OOD | Raw-score generalization and OOD evaluation |
| Full note-alignment manifest | `procedural12k` plus `raw2k`, stratified by corpus/source/repetition/duration/intonation/recording/render | 14,100 rows: 10,871 train, 1,338 val, 1,199 test-ID, 692 test-OOD | NoteFrameNet v1/v2 and learned pairwise aligner |
| `outputRaw_sf_10k` | Raw-score snippets rendered with FreePats (`soundfont_v1`), score `001` excluded, exact `note_map.json` supervision | 10,000 rows: 8,004 train, 999 val, 997 test-ID | Current dataset-specific Basic Pitch, Layer 1, and contextual aligner |
| Model B bakeoff split | Fixed `random12k` train/validation partition with a separate 100-bundle holdout | 10,800 train, 1,200 val, 100 evaluated holdout bundles | Controlled v1–v7 architecture comparison |
| DataCreate real takes `001–093` plus `demo_001` | Human recordings aligned to clean scores; all currently remain schema 1.1 and only 3 have note-first alignment artifacts | 94 inspection bundles, not a training corpus | Real-audio sanity/OOD checks only |

The full split files are the authoritative membership lists:

- `runs/note-align-full-e-s365/split.json`
- `runs/basic-pitch-refiner-procedural-s365/split.json`
- `runs/contextual-aligner-outputRaw_sf-1k/split.json`

### Transcription models

| Version | Methodology and architecture | Training data | Training hyperparameters | Result / status |
|---|---|---|---|---|
| NoteFrameNet v1 | Log-mel → four frequency-striding convolution blocks → five 48-channel temporal residual blocks; independent voiced, written-pitch, onset, and offset heads. No multiscale skip, F0, or cents head. | 10,871 full-manifest train + 1,338 val bundles | AdamW, LR `3e-4`, batch 8, 1,024-frame crops, 2 crops/clip, dropout 0.12, up to 30 epochs, patience 5, seed 365; calibrated voiced/onset/offset thresholds 0.55/0.30/0.30 | Best val F1 0.0417 at epoch 12; calibrated F1 0.0547. Rejected. |
| NoteFrameNet v2 | Three spectral blocks, multiscale projection, 128 temporal channels, ten dilated depthwise residual blocks, PESTO F0 input, and voiced/pitch/onset/offset/cents heads. Written-pitch and signed-cents targets come from exact note maps/labels. | Same 10,871 train + 1,338 val bundles | AdamW, LR `3e-4`, batch 8, 1,024-frame crops, 2 crops/clip, dropout 0.12, patience 5, seed 365; loss weights voiced 1, pitch 1, onset 2, offset 1.5, cents 0.5; decode thresholds 0.50/0.45/0.40 | Best val F1 0.1115 at epoch 10; cents MAE 2.96 on matched notes. Better than v1 but rejected for low absolute F1. |
| Basic Pitch 0.4.0 frozen | Spotify Basic Pitch onset/note/contour network, decoded in written-pitch space. Repository cleanup enforces monophony, written MIDI 50–96, removes harmonic/high-note artifacts, and merges weak same-pitch splits. | No repository training; upstream pretrained weights. Thresholds calibrated on repository validation audio. | Production: onset 0.55, frame 0.35, min note 55 ms, 45–2,600 Hz frontend range, same-pitch merge gap 0.10 s, merge onset 0.60, harmonic overlap 0.55 | Current decoder. On `outputRaw_sf_10k` 200-test transcription: P/R/F1 0.649/0.665/0.657, predicted/target count ratio 1.024. |
| Basic Pitch clarinet refiner v3 | Frozen Basic Pitch maps + PESTO fine pitch → 96-channel, six-block temporal residual refiner → sparse monophonic interval Semi-CRF. Predicts voice, pitch residual, boundaries, confidence, and cents. | Procedural-only strict manifest; bounded run used 256 train bundles and 80 validation clips, tested on 80 | AdamW, LR `3e-4`, weight decay `1e-3`, batch 8, 384 frames, 2 crops/clip, grad clip 2, max 8 requested/6 completed epochs, patience 4, interval weight 0.20, seed 365 | Test-ID F1 0.304 versus frozen Basic Pitch 0.827. Not promoted. |
| Basic Pitch TensorFlow fine-tune | Optional staged fine-tuning of official Basic Pitch: heads, then branches, then trunk | Requires an explicit frozen manifest; procedural-only mode available | Batch 4; phases 2/4/2 epochs; LR `3e-4`/`1e-4`/`3e-5`; seed 365 | Only smoke artifact is retained; not production. |

Basic Pitch cache/frontend version is `align-basic-pitch-0.4.0-v2`; PESTO feature version is 2.0.1. Cache keys include the WAV SHA-256 and effective audio-to-written transpose.

### External transcription benchmark

These are frozen upstream models, not models trained by this repository. Each was calibrated on 40 clips and evaluated on up to 80 clips per split from the full note-alignment manifest (actual valid counts: 75 val, 78 test-ID, 80 test-OOD). The F1 below is an acoustic onset/pitch diagnostic and is not official note-wise model F1. Promotion still requires end-to-end canonical score-event scoring.

| Model | Methodology | Calibrated decoder settings | Val F1 | Test-ID F1 | Test-OOD F1 |
|---|---|---|---:|---:|---:|
| Basic Pitch 0.4.0 | Joint onset/frame/contour AMT network with note-event decoding | onset 0.50, frame 0.40, min 55 ms | 0.826 | 0.827 | 0.558 |
| Tsumugi checkpoint | Neural frame features plus sparse interval Semi-CRF decoding | checkpoint-native interval decode | 0.823 | 0.822 | 0.557 |
| PESTO 2.0.1 | Monophonic F0 estimator segmented into notes | confidence 0.85, onset 0.50, min 90 ms, pitch-change 0.65 semitone, max gap 2 frames | 0.434 | 0.407 | 0.434 |
| PENN | Neural F0 estimator segmented into notes | confidence 0.40, onset 0.20, min 50 ms, pitch-change 0.65, max gap 2 | 0.454 | 0.461 | 0.319 |
| TorchCREPE tiny | CREPE F0 estimator segmented into notes | confidence 0.10, onset 0.20, min 90 ms, pitch-change 0.65, max gap 2 | 0.310 | 0.283 | 0.235 |

Basic Pitch was selected because it matched Tsumugi while being easier to cache, calibrate, and integrate. PESTO remains useful for cents/F0 features, not note segmentation.

### Layer 1 repetition model

The production Layer 1 is hybrid:

- Deterministic sequence search proposes tempo-tolerant repeated phrases from transcribed pitch and timing.
- `NoteRepetitionScorer` rescues ambiguous one-note repeats. It is a two-hidden-layer MLP over candidate sequence features (`hidden_dim=48`, SiLU, dropout 0.10), with up to 32 source candidates per note, 64 notes per phrase, and two extra repetitions.

Both retained checkpoints used 1,000 train and 200 validation bundles, AdamW (`lr=8e-4`, weight decay `1e-3`), weighted BCE, batch 512, 25 epochs, and seed 365:

| Version | Dataset | Best candidate-row val F1 | End-to-end repetition result |
|---|---|---:|---|
| Procedural Layer 1 | `procedural12k` | 0.624 | 0.755 F1 on 100 test-ID clips (hybrid), versus 0.753 deterministic |
| `outputRaw_sf_10k` Layer 1 | Raw-score SoundFont corpus only | 0.666 | Used by the current dataset-specific pipeline |

The learned scorer is deliberately a fallback; long phrase matches remain deterministic.

### Note-to-score aligners

| Version | Methodology | Training data and hyperparameters | Result / status |
|---|---|---|---|
| Deterministic | Monotonic edit alignment with pitch-dominant match costs and explicit insertion/deletion/substitution operations | No training. Current score-note deletion cost 0.90; repeat links are applied before first-pass mapping. | `outputRaw_sf_10k` 200-test F1 0.450 before later decoder calibration; baseline/fallback. |
| Pairwise learned v1 | Tiny residual MLP (`hidden=48`, dropout 0.08) scores match/substitute/extra/deletion/reject features; dynamic program combines learned and hand costs | Full 14,100-row manifest; AdamW LR `8e-4`, weight decay `1e-3`, batch 2,048, 8 epochs default, seed 365; synthetic augmentation drop 0.08, pitch error 0.10, spurious 0.08, timing jitter 35 ms | Validation mapping F1 about 0.245 after calibration. Rejected. |
| Pairwise learned v2 | Same model with revised operation generation/calibration | Same full manifest; 10-epoch retained experiment | Validation mapping F1 about 0.252. Rejected. |
| Pairwise optimized v3 | Corrected pitch targets, rendered-note supervision, faster/vectorized data generation, and revised bootstrap | Same full manifest; 9-epoch optimized run | Validation mapping F1 about 0.297. Rejected for weak end-to-end mapping. |
| Contextual exact-map v1 | Bidirectional 2-layer GRU sequence encoder (`hidden=64`, dropout 0.10) plus pairwise score matrix and monotonic dynamic program | `procedural12k`: 10,000 train, 300 val; AdamW LR `5e-4`, weight decay `1e-3`, batch 32, 8 epochs, grad clip 2, augmented pitch/drop/timing noise | Val note-position accuracy 0.711; pipeline F1 0.0244 versus deterministic 0.0256. Retained as a candidate only. |
| Contextual exact-map v2 | Same architecture trained only on `outputRaw_sf_10k` | Exactly 1,000 train + 200 val, 8 epochs, batch 32, LR `5e-4`, weight decay `1e-3`, seed 365 | Val note accuracy 0.782; held-out mapping F1 0.470 versus deterministic 0.450. Promoted, then used to initialize cached-sequence training. |
| Contextual cached-sequence v3 | Fine-tunes v2 on actual cached Basic Pitch + Layer 1 sequences. Structured monotonic sequence NLL plus 0.25 local loss aligns training inputs with inference inputs. | `outputRaw_sf_10k`: 1,000 train + 200 val; AdamW LR `2e-4`, weight decay `1e-3`, batch 8, 5 epochs, grad clip 2 | Val note accuracy 0.665; 200-test mapping F1 0.503 before final threshold calibration. Current checkpoint. |

Two non-learned recovery decoders were evaluated with the same current transcription and 200-test split:

| Strategy | Method | Mapping F1 |
|---|---|---:|
| `contextual` | Current contextual v3 dynamic program | **0.509** |
| `revision` | Conservative local remapping after a sustained mismatch burst | 0.506 |
| `contextual_continuation` | Contextual v3 plus a strict replay/resume rule: after removing a proposed replay, at least two of the next three mapped notes must continue immediately after the source score span | 0.468 |
| `multi_start` | Overlapping-window alignments combined by consensus | 0.463 |

`PipelineConfig.note_alignment_strategy="contextual"` is the default.

The continuation branch is retained as a precision-oriented experiment, not
promoted. On the same locked 200-clip `outputRaw_sf_10k` test subset it raised
precision from 0.511 to 0.582 and reduced predicted repetition spans from 475
to 52, but recall fell from 0.507 to 0.392 and overall F1 fell from 0.509 to
0.468. It detected repetitions in only 49 clips while exact lineage marks
replays in 166 clips. Results are stored in
`runs/contextual-aligner-outputRaw_sf-1k/evaluation-contextual-continuation-200.json`.

### Error-detector model families

#### Model A: staged crop models

| Stage/version | Architecture and target | Main datasets / run names | Hyperparameters and status |
|---|---|---|---|
| A1 `RestartScorer` | Siamese three-layer 1-D convolution encoder, mean+max pooling, 64-d embedding, comparison features `[a,b,|a-b|,a*b]`, dropout 0.15, binary head | `stages` (early 1k), `stages-random12k`, `stages-raw2k`, typed/set-soft reporting runs | AdamW, LR `1e-3`, weight decay `1e-2`, cosine LR schedule, batch 32, 8 epochs default, 10% sample-level val, seed 365. Superseded by transcription-based Layer 1. |
| A2 `EditCropNet` | Two 1-D convolutions + global pool + five-way crop classifier: match/missed/extra/wrong/intonation | Same stage runs; later `stage2-binary-gate`, `stage2-binary-hardneg`, `stage2-full`, and capped `stage2-20k` | AdamW, LR `1e-3`, weight decay `1e-2`, batch 32, 8 epochs. The optimized cap is 20,000 examples: 8k match + 3k each error class. Hard-negative binary gate won the 20-clip calibration comparison (F1 0.129). Current note-first Layer 2 mostly uses aligned transcription instead. |
| A3 `RhythmNet` | 64-d mel crop encoder plus four timing/energy auxiliaries → 16-d aux MLP → dropout-0.15 binary head | `stages`, `stages-random12k`, `stages-raw2k`, `model-a-improve/stage3-procedural` | AdamW, LR `1e-3`, weight decay `1e-2`, batch 32, 8 epochs, 10% val. Final Model-A calibration set logit threshold 20, suppressing noisy learned rhythm predictions; note-first conservative duration rules are preferred. |
| A4 timbre | Duration/energy heuristics for squeak, bad start, and bad timbre | No training set/checkpoint | Off by default. |

The final pre-note-first Model A experiment trained only on `E:\output` (no raw-derived data) and scored 0.206 type-aware melody F1 on a locked 100-bundle holdout. It is historical, not the current production path.

`stages-random12k-setsoft` and `stages-random12k-typed` use the same stage architectures and training defaults. Their names distinguish soft similarity reporting from the later hard type-aware metric; they are not additional neural architectures.

#### Model B: melody-first versions v1–v7

All versions share 128-bin mel input, `d_model=256`, four-head attention, four audio encoder layers, four score encoder layers, two fusion layers, dropout 0.10, audio stride 4, at most 1,920 audio frames and 128 score notes. The controlled bakeoff used 10,800 `random12k` train bundles, 1,200 validation bundles, batch 2, AdamW (`lr=2e-4`, weight decay `1e-2`), grad clip 1, seed 365, and shared early stopping. All stopped after 3 epochs/16,200 steps.

| Version | Methodology | Decode-specific hyperparameters | 100-bundle holdout F1 | Status |
|---|---|---|---:|---|
| v1 control | Seven-way per-note type CE + clip-level repeat-copy head; contiguous same-type runs | Loss weights: error 6, copies 1, coverage 2, error auxiliary 3 | **0.249** | Best bakeoff version, but below Model A/current note-first path |
| v2 sparse | Match-biased seven-way CE, separately normalized match/error losses | error CE 1.25, match CE 1, match decode bias 0.8, confidence gate 0.55 | 0.000 | Collapsed to all match |
| v3 BIO | BIO span-boundary head + type-on-span head + copy head | O bias 0.4, B/I bias -0.2; valid `B I*` runs only | 0.000 | Collapsed to no spans |
| v4 binary | Binary fault mask (balanced BCE + Dice), then type only on fault notes | fault threshold 0.50 | 0.156 | Predicted one broad fault per clip |
| v5 Dice | Type-agnostic core mask with BCE + soft Dice and auxiliary type loss | threshold 0.50, run length 2–16, type aux 0.25 | 0.000 | Collapsed to no spans |
| v6 CRF | Seven-way emissions with trainable linear-chain CRF and constrained Viterbi | maximum error run 16 notes | 0.000 | Collapsed to all match |
| v7 combo | Class-weighted seven-way CE + Dice fault objective + confidence-gated run decode | gate 0.55; class weights 1.0–1.5 | 0.000 | Collapsed to all match |

Other retained Model B directories (`melody-random12k`, `melody-random12k-set`, `melody-raw2k`, and `melody-raw2k-set`) are earlier metric/dataset variants of the v1 architecture. They are historical and not production checkpoints.

### Legacy RUMAA-lite

RUMAA-lite is a joint score/audio Transformer using the same `AudioEncoder`, `ScoreEncoder`, and `HierarchicalFusion` backbone as Model B. Defaults: 128 mel bins, width 256, four heads, 4/4/2 audio/score/fusion layers, dropout 0.10, stride 4, max 1,920 frames and 128 notes. Training defaults are 20 epochs, batch 4, AdamW LR `2e-4`, weight decay `1e-2`, 10% val, grad clip 1, seed 365. Its validation error accuracy remained zero, so it must not be selected for new work.

### Full-joint outputRaw data release

`data-audit/joint-outputraw-full-v1` re-audits exactly 10,000
`E:\outputRaw_sf_10k` bundles. It admits 9,736: 4,544 Mozart train, 358
fingerprint-isolated Mozart validation, and 4,022 metadata-only Weber locked
test rows. The lockbox excludes 812 rows in 102 duplicate components touching
788 historically evaluated ids. Intonation is masked throughout. Canonical
targets include true MusicXML tie-chain projection (never slurs), exact
transcription intervals, replay source/resume/copy targets, note operations,
and duration/rhythm targets.

The immutable train/validation pack is
`data-packed/joint-outputraw-full-v1-shard64` (4,902 records, 32.14 GB).
On the measured i7-13700H/32 GB/NVMe system, packed loading reached 108.19
rows/s for 100 rows and 67.82 rows/s for 1,000 rows, versus 1.00 and 1.13
rows/s for NPZ + SQLite: 107.84x and 59.88x faster. Repeated full-layout and
worker tests selected 64-row shards, four workers, prefetch 16, at most four
open shards, record checksum validation, no pinned memory, and seed 20260915.
Use `PackedJointDataset` and persist `PackedCursor.to_dict()` for exact resume.
`runs/joint-outputraw-full-v1/DATA_READY.json` is the authoritative verified
paths/counts/hashes marker; test features and targets are intentionally absent.

### Frozen downstream error heads v1

`runs/joint-outputraw-full-v1/error-heads-v1` trains Layer 2/3 heads on all
4,544 verified train rows and calibrates only on all 358 validation rows. The
4,022-row lockbox remains metadata-only. Basic Pitch candidate configuration
`align-joint-candidates-v2-short-rescue` and the completed replay-aware joint
aligner/decoder are frozen; the shared aligner/decoder checkpoint SHA-256 is
`91baaa26dc408cd5ffdadea66bcb9fe9b61fac3bda4754f0a4f20392750fa2f4`.
Actual decoded paths supply training inputs, while a separately trained oracle
upstream ablation measures the downstream ceiling. Intonation stays masked.

The selected eager FP32 CPU setting processed 626,048 head rows/s at batch
32,768. Both variants trained for eight epochs with AdamW (`lr=2e-4`, weight
decay `1e-4`), focal/balanced classification, deviation regression, atomic
mid-epoch state, and validation-only per-class thresholds.

| Validation path | Layer 2 typed error P/R/F1 | Layer 2 macro F1 | Rhythm P/R/F1 | Deviation MAE | Schema 1.2 typed-range F1 |
|---|---:|---:|---:|---:|---:|
| Frozen predicted upstream | 0.185 / 0.475 / **0.266** | 0.369 | 0.150 / 0.023 / 0.040 | 70.0 ms | 0.052 |
| Oracle upstream ceiling | 0.816 / 0.949 / **0.877** | 0.749 | 0.238 / 0.005 / 0.010 | 54.6 ms | 0.877 |
| Current path/rhythm rules | 0.093 / 0.277 / 0.139 | 0.320 | 0.048 / 0.390 / 0.085 | 176.1 ms | not available |

Predicted-upstream per-type F1 is 0.850 match, 0.263 wrong note, 0.281
extra note, and 0.081 missed note, with 30.41 false error labels per clip.
The learned Layer 2 head materially beats current operation rules, but rhythm
F1 and absolute schema-range quality do not pass promotion quality. This run
is therefore retained as an isolated experiment and no production weights
were modified. Historical EditCropNet and RhythmNet require mel crops absent
from this verified packed release, so their old results are recorded as
non-comparable rather than relabeled as same-validation measurements.

### Frozen downstream error heads v2

`runs/joint-outputraw-full-v1/error-heads-v2` corrects the failed v1 target,
rhythm, decoding, and schema contracts while reusing the compatible frozen
upstream extraction. Rhythm supervision now requires an exact predicted
score mapping and excludes replay/restart rows. Layer 2 uses validation-tuned
operation/context thresholds and suppresses duplicate operations on the same
decoded score event. Layer 3 combines learned probabilities with robust local
tempo-normalized duration/IOI evidence, uncertainty gating, subtype output,
and deviation regression.

On all 358 validation clips, predicted-upstream Layer 2 typed-error
precision/recall/F1 is **0.203 / 0.499 / 0.289**, versus v1 F1 0.266 and rules
F1 0.139. Macro F1 is 0.357 and false labels fall from 30.41 to 28.39 per
clip. Per-class F1 is 0.861 match, 0.276 wrong note, 0.293 extra note, and
0.000 missed note; the missed-note limitation remains explicit. Predicted
rhythm precision/recall/F1 is **0.100 / 0.324 / 0.153**, versus v1 0.040 and
rules 0.088, with 70.3 ms deviation MAE. Short/long subtype F1 is
0.088/0.068. The oracle-upstream Layer 2 ceiling is 0.972 and oracle rhythm
F1 is 0.209.

Historical schema output used the now-legacy exclusive schema 1.2 pitch-list metric,
one-note audited padding, contiguous operation merging, clean neighbours for
extras, and one source-range/copy-count label per repetition. Conservative
per-type emission thresholds are stored in the candidate checkpoint.
The original report's **0.211** headline included repetition. The later
gold-isolated integrated rerun scored the requested four error types at 0.092
legacy pitch-list F1 (range-only 0.129), so v2 is retained as an experiment rather than
promoted. The lockbox and production weights remain untouched.

### Localization-first error heads v3

`runs/joint-outputraw-full-v1/error-heads-v3` keeps the completed v2 head and
frozen upstream hashes but replaces per-row schema emission with a calibrated
sequence cluster decoder. Wrong/rhythm ranges stay on aligned score events;
extras require immediate mapped neighbours on both sides; score-order DELETE
runs require local resynchronization and merge across consecutive misses.
Type-specific thresholds, hysteresis, support, overlap suppression, one-to-one
competition, uncertainty controls, and one-note official padding are encoded
in `decode_config.json`. No classifier retraining was performed.

On all 358 validation clips, the maximum-F1 policy historically reached four-type
legacy pitch-list P/R/F1 **0.137 / 0.327 / 0.193** (95% F1 CI 0.176-0.210), up from v2
0.092. Per-type F1 is 0.154 wrong, 0.105 missed, 0.166 extra, and 0.080 rhythm;
range-only F1 rises from 0.129 to 0.256. Including repetition gives 0.264 F1.
A separately frozen false-control policy reaches **0.144 F1** while reducing
unmatched schema labels from 1,480 to 1,439.5 (4.02/clip), and retains nonzero
missed-note output.

These v2/v3 values are not canonical note-wise results and require rerun under
`align-note-wise-score-event-metric-v1`; they must not be used for promotion.

The oracle decomposition finds no score-index corruption: all 1,338 evaluated
gold ranges reconstruct exactly from the frozen clean score. Gold score-index,
gold clustering/padding, and metric-sanity documents each score 0.998; replacing
only predicted operation cores raises F1 to 0.472, while replacing only type on
existing ranges reaches 0.129. Localization and operation exposure therefore
remain the next bottlenecks. The two-process integrity check passed; the
lockbox, production weights, and promotion state remain unchanged.

#### Legacy half-credit timestamp diagnostic

The strict timestamp metric remains unchanged and requires both a location hit
and an exact label type. A separate `align-typed-location-metric-v1`
diagnostic uses maximum-weight one-to-one assignment: an IoU/onset location hit
receives 1.0 for the same type and 0.5 for a different type. This must not be
applied to schema 1.2 pitch-list/range scores a second time: that metric already
uses `TYPE_MISMATCH_SCALE = 0.5`.

On the 18 non-empty schema-1.1 DataCreate documents explicitly attributed to
`ai_f0_align`, v3 strict IoU>=0.3 F1 is 0.0000 and the separate half-credit F1
is 0.0218 (95% clip-bootstrap CI 0.0057-0.0422). At 50/100/250/500 ms, strict
F1 is 0.0000/0.0000/0.0187/0.0249 and half-credit F1 is
0.0093/0.0187/0.0498/0.0935. The same frozen-prediction rerun gives half-credit
IoU F1 0.0000 for v2 and 0.0213 for rules (strict rules F1 0.0064). These are
agent-label agreement diagnostics, not independent human accuracy. Candidate
counts and score-conditioned coverage remain proxies, not note F1, because no
independent performed-note transcription is available. Full artifacts are in
`runs/eval-datacreate-error-heads-v3-agent-labels-20260915/half-credit-metric-v1/`.

### Direct-operation hybrid error heads v4

`runs/joint-outputraw-full-v1/error-heads-v4` derives wrong, extra, and missed
candidates from frozen joint substitutions, unlinked acoustic events, and
locally resynchronized DELETE runs. Reliability uses the decoder's exact local
n-best operation/span softmax, path margin and alternative entropy, acoustic
confidence, stable neighbors, repeat state, and delete+insert ambiguity. These
direct scores are blended with the completed v2 learned head before the v3
one-to-one sequence cluster decoder.

The first deterministic hybrid reached 0.206 four-type F1, short of the
material-gain gate. A validation-only, CPU-trained linear operation-core
selector was therefore fitted without changing any upstream model weights.
On the separately frozen 358-clip validation inference it reaches official
P/R/F1 **0.210 / 0.245 / 0.226** (95% F1 CI 0.204–0.250). Per-type F1 is
0.191 wrong, 0.133 missed, 0.245 extra, and 0.069 rhythm; range-only F1 is
0.279 and the historical score including repetition is 0.324. The selected
policy also satisfies the false-control gate at 1,225 effective false labels
(3.42/clip), 214.5 fewer than v3's balanced policy.

The v4 candidate clears the recorded validation gates but remains unpromoted:
the selector was calibrated on validation, the lockbox remains sealed, and
production weights/configuration are unchanged. `feature_freeze_manifest.json`,
`freeze_manifest.json`, `integrity.json`, and `verification.json` record the
two-process gold isolation and artifact hashes.

### Train-only calibrated error heads v5

`runs/joint-outputraw-full-v1/error-heads-v5` removes v4's validation leakage.
A deterministic leakage-group split assigned 3,634 training rows across 1,846
groups to selector fitting and 910 rows across 475 disjoint groups to blend,
threshold, clustering, and padding calibration. The selector adds
tempo-normalized duration/IOI evidence for rhythm localization. Five candidate
policies were declared before validation; selector weights and policy hashes
were frozen before any of the 358 validation targets were opened.

The separate one-shot validation process scores learned-only, direct-only, and
train-selected hybrid four-type F1 at **0.191 / 0.200 / 0.209**, respectively.
Hybrid P/R/F1 is **0.165 / 0.285 / 0.209** (95% F1 CI 0.191–0.229), with
per-type F1 0.192 wrong, 0.127 missed, 0.188 extra, and 0.079 rhythm.
Range-only F1 is 0.267, type-only F1 is 0.439, and including repetition is
0.292. The max-F1 policy emits 5.34 effective false labels per clip. Its
predeclared high-precision counterpart lowers this to 2.18/clip but scores only
0.166 F1.

The honest estimate is 0.017 below exploratory same-set v4 and improves over
the frozen learned-only baseline by 0.0178, narrowly missing the required 0.02
material-gain gate. No post-validation tuning or second validation opening was
performed. V5 is not promoted; the lockbox and production remain unchanged.

### Reading run metadata

- `history.json`, `*_history.json`, and `*.history.json` record actual completed epochs and validation metrics.
- Checkpoints store architecture configs and calibrated thresholds; those values override source defaults when loaded.
- `evaluation-*.json` files record the exact manifest, clip count, decoder/alignment strategy, and aggregate counts.
- Directory names containing `smoke`, `dev`, `bench`, `candidate`, or `pre-*` are diagnostic snapshots, not separately promoted production versions.
