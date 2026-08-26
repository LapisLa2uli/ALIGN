# ALIGN error detector

Score-informed clarinet error detector for ALIGN bundles. The default path is a **four-stage pipeline** (practice restarts, pitch edits, rhythm, optional timbre). Legacy **RUMAA-lite** train/infer commands remain available.

## Four-stage pipeline

| Stage | What it does | ALIGN types |
|------|--------|-------------|
| 1 Restarts | Score graph (written repeats vs practice jumps), silence/hold + chroma copy cuts, partial-match beam | `repetition` with `repeats_label_range` |
| 2 Edits | On each unfolded segment: chroma DTW, Match / Insert / Delete / Substitute | `missed_note`, `extra_note`, `wrong_note`, `intonation_error` |
| 3 Rhythm | EWMA + far-window duration ratios on **paired** notes only | `rhythm_error` |
| 4 Timbre | Untrained heuristics on extra/wrong crops (off by default) | `squeak`, `bad_start`, `bad_timbre` |

Stages share one `PipelineState` (unfolded performance↔score segments), not four independent detectors.

## Setup

Same conda env as DataCreate (`MusicEval`):

```powershell
cd "D:\stuff\Audio Evaluation\ALIGN"
conda activate MusicEval
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ./align-model
```

## Run

```powershell
align-model run --sample ".\synth-pipeline\output\synth_gen_0010"
align-model run --sample ".\synth-pipeline\output\synth_gen_0010" --timbre
align-model smoke --data ".\synth-pipeline\output"
```

Writes `pipeline_pred.json` in the sample folder (labels schema `1.1`, `source: pipeline`). Stage 4 is opt-in via `--timbre`.

## Train stages 1-3

Learned heads on top of the classical pipeline (GPU, uses `performance_mel.npy`):

| Stage | Model | Supervision |
|------|--------|-------------|
| 1 | MLP on paired mel-span pools | gold `repetition` vs random spans |
| 2 | Conv1d on mel crops | match / miss / extra / wrong / intonation |
| 3 | MLP on duration-ratio features | gold `rhythm_error` on mapped notes |

```powershell
align-model train-stages --data ".\synth-pipeline\output" --out ".\align-model\runs\stages" --epochs 8 --device cuda
align-model run --sample ".\synth-pipeline\output\synth_gen_0010" --weights ".\align-model\runs\stages" --device cuda
```

```powershell
align-model run --sample ".\synth-pipeline\output\synth_gen_0010" --device cuda
align-model train --data ".\synth-pipeline\output" --device cuda --batch-size 4
```

Each sample needs `verified_score.musicxml` and `performance_audio.wav`.

## Legacy RUMAA-lite

Joint transformer trained in `align-model/runs/rumaa-lite/`. It did not learn error classes (val error acc stayed 0); prefer the staged pipeline.

```powershell
align-model train --data ".\synth-pipeline\output" --out ".\align-model\runs\rumaa-lite" --epochs 20 --batch-size 4 --device cuda
align-model infer --ckpt ".\align-model\runs\rumaa-lite\best.pt" --sample ".\synth-pipeline\output\synth_gen_0010" --device cuda
```
