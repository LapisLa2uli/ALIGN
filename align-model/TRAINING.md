# Command-line training guide

This document is the practical CLI reference for training ALIGN models. The hyperparameter card that produced each model's best comparable result lives in [HYPERPARAMETERS.md](HYPERPARAMETERS.md). Model history, promotion status, and published metrics live in [README.md](README.md). Schema and official scoring live in [`methodology.md`](../methodology.md).

Every trainer below is a Python `argparse` program. Flags always override the documented defaults. There is no separate config-file trainer for the main `align-model` commands; you change parameters by passing flags.

## 1. Which trainer to use

| Goal | Command | Status |
|---|---|---|
| Production Layer 1 repetition scorer | `python align-model/scripts/train_note_repetition.py` | Current production checkpoint family |
| Production contextual note aligner | `python align-model/scripts/train_contextual_note_aligner.py` then `train_cached_contextual_aligner.py` | Current production checkpoint family |
| Historical four-stage Model A | `align-model train-stages` | Historical. Beaten by the note-first path |
| Historical melody-first Model B | `align-model train-melody` | Historical bakeoff family. Best bakeoff version (v1) scored below Model A / note-first |
| Legacy RUMAA-lite | `align-model train` | Do not use for new work. Validation error accuracy stayed 0 |
| NoteFrameNet transcriber | `python align-model/scripts/train_note_transcriber.py` | Rejected. Best calibrated F1 stayed far below frozen Basic Pitch |
| Basic Pitch clarinet refiner | `python align-model/scripts/train_note_refiner.py` | Not promoted. Test-ID F1 0.304 vs frozen Basic Pitch 0.827 |
| Joint outputRaw packed-release stack | `python scripts/train_outputraw_full_pipeline.py` from `align-model/` | Experimental. Lockbox must stay sealed |
| Frozen downstream error heads | `python scripts/train_error_heads*.py` from `align-model/` | Isolated experiments. None promoted |

Production inference still uses frozen Basic Pitch plus the Layer 1 / contextual-aligner weights under `runs/contextual-aligner-outputRaw_sf-1k/weights/`. Training a new checkpoint does not install it into that bundle until you copy it there yourself.

## 2. Environment and working directory

```powershell
cd "D:\stuff\Audio Evaluation\ALIGN"
conda activate MusicEval
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ./DataCreate
pip install -e ./align-model
```

Two invocation styles exist:

| Style | Typical working directory | Example |
|---|---|---|
| Installed console script | repository root | `align-model train-stages --data ...` |
| Script path | repository root for note-first scripts; `align-model/` for joint `runs/...` defaults | `python align-model\scripts\train_note_repetition.py ...` |

Joint packed-release scripts default to paths such as `runs/joint-outputraw-full-v1/DATA_READY.json`. Those relatives resolve against the **current working directory**. Either `cd align-model` first, or pass prefixed paths such as `align-model\runs\...`.

PowerShell line continuation uses a backtick. Quote paths that contain spaces. Flag names are `--kebab-case`.

```powershell
# Discover every flag for a command
align-model train-stages --help
python align-model\scripts\train_note_repetition.py --help
python align-model\scripts\train_outputraw_full_pipeline.py --help
```

`--help` is the live contract. If this file and `--help` disagree, trust `--help`.

## 3. Data each trainer expects

Training never reads `performance_score.musicxml` as the gold score. The clean score is always `verified_score.musicxml`. Written MIDI is the target; Bb clarinet audio sounds two semitones lower.

| Trainer family | Required files | Split source |
|---|---|---|
| `train`, `train-stages`, `train-melody` | Per-bundle `verified_score.musicxml`, `performance_audio.wav`, `performance_mel.npy`, `labels.json` | Random 90/10 sample split, seed 365, unless `--max-samples` / `--overfit` shrinks it |
| Layer 1 / contextual aligner | Frozen `split.json` plus exact `note_map.json` / `rendered_notes` | Manifest membership is authoritative. Requests larger than a split fail instead of leaking validation into train |
| NoteFrameNet | One or more bundle `--root`s plus a JSON/JSONL `--manifest` | Manifest `train` / `val` rows |
| Joint packed release | `DATA_READY.json` and the 4,902-record packed shard pack | Immutable 4,544 train / 358 val. Locked test is metadata-only and must not be opened |

Official evaluation is exclusive one-to-one **note-wise F1** on canonical score-event identity. Same location and type scores 1.0, same location with a different type scores 0.5, and a wrong location scores 0. Timestamp IoU is a diagnostic only.

## 4. How to change parameters

Pass a flag. Almost every trainer maps CLI flags onto a dataclass (`TrainConfig`, `StageTrainConfig`, `MelodyTrainConfig`, `NoteTrainConfig`, and so on). Unspecified flags keep the defaults shown below.

```powershell
# Change epochs, learning rate, batch size, device, and output directory
align-model train-stages `
  --data ".\synth-pipeline\output" `
  --out ".\align-model\runs\stages-debug" `
  --epochs 3 `
  --batch-size 16 `
  --lr 5e-4 `
  --device cuda
```

Common patterns:

| Intent | Typical flags |
|---|---|
| Shorter debug run | `--epochs 1`, `--max-samples 32`, `--overfit 8`, `--max-train-rows 16` |
| Resume a crashed run | `--resume path\to\last.pt` or `--resume-checkpoint ...` |
| CPU instead of GPU | `--device cpu` (some joint scripts also have `--path-device`) |
| Reproducible comparison | keep `--seed` fixed (365 for note-first; 20260915/20260916/20260919 for joint families) |
| Smaller memory footprint | lower `--batch-size`, raise `--accumulation-steps` / `--gradient-accumulation` |
| Architecture width | `--hidden-dim`, `--channels`, `--temporal-blocks`, `--temporal-kind` |
| Data subset | `--max-samples`, `--train-samples`, `--val-samples`, `--max-train-rows` |

Flags that are **not** exposed on a given command cannot be changed from the CLI. Those values live in the trainer dataclass and stay at source defaults unless you edit code. Check `--help` before assuming a knob exists.

Boolean flags are `store_true` unless noted. Presence enables them:

```powershell
align-model train-stages --skip-holdout --mine-heuristic-edits
python align-model\scripts\train_mel_transcriber_v1.py --compile --amp bf16
```

Some joint scripts use `BooleanOptionalAction`:

```powershell
python align-model\scripts\train_error_heads.py --resume
python align-model\scripts\train_error_heads.py --no-resume
```

## 5. Installed CLI: `align-model`

After `pip install -e ./align-model`, the `align-model` console script is `alignmodel.cli:main`.

```powershell
align-model --help
align-model train --help
align-model train-stages --help
align-model train-melody --help
```

### 5.1 `train-stages` — Model A (historical)

Trains Stage 1 `RestartScorer`, Stage 2 `EditCropNet`, and Stage 3 `RhythmNet` on first-pass gold. Writes `stage1.pt` / `stage2.pt` / `stage3.pt` under `--out`.

Earlier Model A used chroma-DTW crops and a 20,000-example Stage 2 cap because unfiltered crops were dominated by matches and drowned the four error classes. `--stage2-max-train-examples` is that cap. The later note-first pipeline replaced these crop models because held-out type-aware F1 stayed low (0.206 on a locked 100-bundle holdout). This command remains available for reproduction, not promotion.

```powershell
align-model train-stages `
  --data ".\synth-pipeline\output" `
  --out ".\align-model\runs\stages" `
  --epochs 8 `
  --batch-size 32 `
  --lr 1e-3 `
  --device cuda `
  --stages 1,2,3
```

| Flag | Default | Meaning |
|---|---|---|
| `--data` | `E:/output` | Root of synth bundle folders |
| `--out` | `align-model/runs/stages` | Checkpoint directory |
| `--epochs` | `8` | Epochs per requested stage |
| `--batch-size` | `32` | Crops per optimizer step |
| `--lr` | `1e-3` | AdamW learning rate. Weight decay is fixed at `1e-2` in code |
| `--device` | `cuda` | `cuda`, `cpu`, or `auto` |
| `--stages` | `1,2,3` | Comma-separated subset, for example `2` or `1,3` |
| `--max-samples` | `0` | If > 0, cap the number of bundles. `0` uses every bundle |
| `--skip-holdout` | off | Drop the seed-365 official holdout clips from training |
| `--mine-heuristic-edits` | off | Add Stage 2 crops from chroma-DTW proposals labeled against gold |
| `--heuristic-mine-max` | `0` | Clip cap for heuristic mining. `0` uses every training clip |
| `--stage2-max-train-examples` | `20000` | Stratified Stage 2 crop cap. `0` keeps every crop |

Values **not** on the CLI, fixed in `StageTrainConfig`: `seed=365`, `val_fraction=0.1`.

Use the resulting directory with inference:

```powershell
align-model run `
  --sample ".\synth-pipeline\output\synth_gen_0010" `
  --weights ".\align-model\runs\stages" `
  --device cuda
```

### 5.2 `train-melody` — Model B (historical)

Trains the melody-first transformer. The controlled bakeoff compared v1–v7 on a frozen 10,800/1,200 split. Only v1 produced a usable holdout F1 (0.249). v2–v7 collapsed to all-match or empty spans, so the CLI still defaults to `--variant v1`.

```powershell
align-model train-melody `
  --data ".\synth-pipeline\output" `
  --out ".\align-model\runs\melody" `
  --epochs 8 `
  --batch-size 4 `
  --lr 2e-4 `
  --device cuda `
  --variant v1
```

| Flag | Default | Meaning |
|---|---|---|
| `--data` | `E:/output` | Synth bundle root |
| `--out` | `align-model/runs/melody` | Writes `best.pt`, `history.json` |
| `--epochs` | `8` | Maximum epochs. Shared early stopping can stop sooner |
| `--batch-size` | `4` | Bundles per step. Bakeoff used `2` |
| `--lr` | `2e-4` | AdamW learning rate |
| `--device` | `cuda` | Compute device |
| `--max-samples` | `0` | Bundle cap. `0` = all |
| `--overfit` | `0` | If > 0, train on that many bundles only (debug) |
| `--variant` | `v1` | `v1` control, `v3` BIO, `v5` dice, `v7` combo |

Not on the CLI: `weight_decay=1e-2`, `seed=365`, `grad_clip=1.0`, `d_model=256`, four encoder layers, dropout 0.10, and the v1 loss weights `error=6`, `copies=1`, `coverage=2`, `error_aux=3`.

```powershell
align-model run-melody `
  --sample ".\synth-pipeline\output\synth_gen_0010" `
  --ckpt ".\align-model\runs\melody\best.pt"
align-model eval-melodies `
  --data ".\synth-pipeline\output" `
  --pred melody_pred.json
```

### 5.3 `train` — RUMAA-lite (legacy)

Joint score/audio transformer. Kept for reproduction. Validation error accuracy remained zero, so do not select it for new work.

```powershell
align-model train `
  --data ".\synth-pipeline\output" `
  --out ".\align-model\runs\rumaa-lite" `
  --epochs 20 `
  --batch-size 4 `
  --lr 2e-4 `
  --device cuda
```

| Flag | Default | Meaning |
|---|---|---|
| `--data` | `synth-pipeline/1000dataexport` | Bundle root |
| `--out` | `align-model/runs/rumaa-lite` | Checkpoint directory |
| `--epochs` | `20` | Training epochs |
| `--batch-size` | `4` | Bundles per step |
| `--lr` | `2e-4` | AdamW learning rate |
| `--overfit` | `0` | If > 0, restrict to that many bundles |
| `--device` | `cuda` | Compute device |

Not on the CLI: `weight_decay=1e-2`, `seed=365`, `val_fraction=0.1`, `grad_clip=1.0`, `ModelConfig` width 256.

```powershell
align-model infer `
  --ckpt ".\align-model\runs\rumaa-lite\best.pt" `
  --sample ".\synth-pipeline\output\synth_gen_0010" `
  --device cuda
```

## 6. Production note-first training

Run these from the repository root. Use a frozen `split.json`; do not resplit.

The earlier pairwise learned aligner (tiny residual MLP over match/substitute/extra/delete features) peaked around 0.297 validation mapping F1 and was rejected. The later contextual GRU sequence aligner is the production family because, on `outputRaw_sf_10k`, it beat deterministic edit alignment (0.470 vs 0.450 mapping F1) and then improved further when fine-tuned on cached Basic Pitch + Layer 1 sequences (0.503, then 0.509 after calibration).

### 6.1 Layer 1 repetition scorer

```powershell
python align-model\scripts\train_note_repetition.py `
  --manifest align-model\runs\contextual-aligner-outputRaw_sf-1k\split.json `
  --out align-model\runs\contextual-aligner-outputRaw_sf-1k\weights\note_repetition.pt `
  --train-samples 1000 `
  --val-samples 200 `
  --epochs 25 `
  --batch-size 512 `
  --device cuda `
  --seed 365
```

| Flag | Default | Meaning |
|---|---|---|
| `--manifest` | required | Frozen split JSON |
| `--out` | required | Output `.pt` path |
| `--train-samples` | `1000` | Must not exceed the manifest train split |
| `--val-samples` | `200` | Must not exceed the manifest val split |
| `--epochs` | `25` | Training epochs |
| `--batch-size` | `512` | Candidate rows per step |
| `--device` | `cuda` | Compute device |
| `--seed` | `365` | Data order and initialization |

Optimizer values not on the CLI: AdamW `lr=8e-4`, weight decay `1e-3`, hidden dim 48, dropout 0.10.

`--train-samples` / `--val-samples` larger than the frozen split raise an error. That is intentional: earlier code silently moved validation examples into training.

### 6.2 Contextual aligner (exact-map pretrain)

```powershell
python align-model\scripts\train_contextual_note_aligner.py `
  --manifest align-model\runs\contextual-aligner-outputRaw_sf-1k\split.json `
  --out align-model\runs\contextual-aligner-outputRaw_sf-1k\weights\contextual_note_aligner.pt `
  --train-samples 1000 `
  --val-samples 200 `
  --epochs 8 `
  --batch-size 32 `
  --device cuda `
  --seed 365
```

| Flag | Default | Meaning |
|---|---|---|
| `--manifest` | required | Frozen split |
| `--out` | required | Output checkpoint |
| `--train-samples` | `10000` | Production `outputRaw_sf_10k` run used `1000` |
| `--val-samples` | `300` | Production run used `200` |
| `--epochs` | `8` | Training epochs |
| `--batch-size` | `32` | Sequences per step |
| `--device` | `cuda` | Compute device |
| `--seed` | `365` | Reproducibility |

Not on the CLI: AdamW `lr=5e-4`, weight decay `1e-3`, `hidden_dim=64`, 2 GRU layers, dropout 0.10, grad clip 2.

### 6.3 Cached-sequence fine-tune

The exact-map model was then fine-tuned on actual cached Basic Pitch + Layer 1 sequences so train-time inputs match inference. That change is why v3 exists: v2 still trained on synthetic exact maps, then lost accuracy when fed real transcriptions.

```powershell
python align-model\scripts\train_cached_contextual_aligner.py `
  --manifest align-model\runs\contextual-aligner-outputRaw_sf-1k\split.json `
  --basic-cache-root path\to\basic-pitch-cache `
  --repetition-checkpoint align-model\runs\contextual-aligner-outputRaw_sf-1k\weights\note_repetition.pt `
  --initial-checkpoint align-model\runs\contextual-aligner-outputRaw_sf-1k\weights\contextual_note_aligner.pt `
  --out align-model\runs\contextual-aligner-outputRaw_sf-1k\weights\contextual_note_aligner.pt `
  --train-samples 1000 `
  --val-samples 200 `
  --epochs 5 `
  --batch-size 8 `
  --device cuda
```

| Flag | Default | Meaning |
|---|---|---|
| `--manifest` | required | Same frozen split |
| `--basic-cache-root` | required | SHA-256 keyed Basic Pitch cache |
| `--repetition-checkpoint` | required | Layer 1 `.pt` from §6.1 |
| `--initial-checkpoint` | required | Exact-map checkpoint from §6.2 |
| `--out` | required | Fine-tuned checkpoint |
| `--train-samples` | `1000` | Train clip count |
| `--val-samples` | `200` | Val clip count |
| `--epochs` | `5` | Fine-tune epochs |
| `--batch-size` | `8` | Smaller than pretrain because sequence NLL is heavier |
| `--device` | `cuda` | Compute device |

Not on the CLI: AdamW `lr=2e-4`, weight decay `1e-3`, grad clip 2, local-loss weight 0.25.

### 6.4 Basic Pitch clarinet refiner (not promoted)

The v1/v2 NoteFrameNet transcribers learned notes from log-mel and reached only 0.05–0.11 val F1. The later refiner froze Basic Pitch maps, added PESTO F0, and trained a compact residual Semi-CRF. That targeted the low absolute F1, but the bounded promotion run still lost to frozen Basic Pitch (0.304 vs 0.827 test-ID F1), so production keeps frozen Basic Pitch.

```powershell
python align-model\scripts\train_note_refiner.py `
  --manifest align-model\runs\basic-pitch-refiner-procedural-s365\split.json `
  --basic-cache-root path\to\basic-pitch-cache `
  --pesto-cache-root path\to\pesto-cache `
  --out align-model\runs\basic-pitch-refiner-procedural-s365\refiner `
  --epochs 12 `
  --batch-size 8 `
  --crop-frames 384 `
  --crops-per-clip 2 `
  --device cuda `
  --seed 365
```

| Flag | Default | Meaning |
|---|---|---|
| `--manifest` | required | Strict procedural-only split |
| `--basic-cache-root` | required | Basic Pitch feature cache |
| `--pesto-cache-root` | required | PESTO feature cache |
| `--out` | required | Run directory |
| `--epochs` | `12` | Max epochs (patience can stop earlier) |
| `--batch-size` | `8` | Crops per step |
| `--crop-frames` | `384` | Temporal crop length |
| `--crops-per-clip` | `2` | Random crops per bundle |
| `--workers` | `8` | DataLoader workers |
| `--max-train-samples` | `0` | `0` = all train rows |
| `--max-val-samples` | `80` | Val clip cap |
| `--interval-weight` | `0.20` | Semi-CRF interval loss weight |
| `--augment-probability` | `0.75` | Refiner augmentation rate |
| `--short-note-weight` | `2.0` | Extra weight on short positives |
| `--hard-negative-ratio` | `1.0` | Hard-negative sampling |
| `--same-pitch-split-probability` | `0.45` | Synthetic same-pitch split noise |
| `--channels` | `96` | Temporal residual width |
| `--temporal-blocks` | `6` | Residual depth |
| `--midi-min` / `--midi-max` | `36` / `108` | Written MIDI range |
| `--device` | `cuda` | Compute device |
| `--seed` | `365` | Reproducibility |
| `--resume` | none | Resume checkpoint |

### 6.5 NoteFrameNet transcriber (rejected)

Kept for reproduction of v1/v2. Do not promote from this trainer.

```powershell
python align-model\scripts\train_note_transcriber.py `
  --root E:\output `
  --root E:\output_2k_rawdata `
  --manifest align-model\runs\note-align-full-e-s365\split.json `
  --out align-model\runs\note-transcriber `
  --epochs 30 `
  --batch-size 8 `
  --lr 3e-4 `
  --device auto `
  --seed 365
```

| Flag | Default | Meaning |
|---|---|---|
| `--root` | required, repeatable | Bundle roots |
| `--manifest` | required | Train/val split |
| `--out` | required | Run directory (`best.pt`, `last.pt`, `history.json`) |
| `--epochs` | `30` | Total target epochs. `--resume` does not add extra epochs |
| `--batch-size` | `8` | Training crops per step |
| `--crop-frames` | `1024` | Crop length |
| `--crops-per-clip` | `2` | Crops drawn per bundle |
| `--lr` | `3e-4` | AdamW learning rate |
| `--workers` | `0` | DataLoader workers |
| `--prefetch-factor` | `4` | Only used when `--workers > 0` |
| `--infer-batch-size` | `4` | Validation decode batch |
| `--infer-window-frames` | `2048` | Sliding-window inference length |
| `--infer-overlap-frames` | `512` | Window overlap |
| `--resume` | none | `last.pt` to continue |
| `--device` | `auto` | `cuda` if available, else `cpu` |
| `--seed` | `365` | Reproducibility |
| `--patience` | `5` | Early-stop patience |
| `--max-val-clips` | `80` | Val clip cap during training |
| `--calibrate-val-clips` | `0` | Extra calibration clips; `0` skips |
| `--midi-min` / `--midi-max` | `36` / `108` | Pitch range |
| `--channels` | `64` | Spectral width (v2 default) |
| `--temporal-channels` | `128` | Temporal width |
| `--spectral-blocks` | `3` | v2 spectral depth. v1 used 4 |
| `--temporal-blocks` | `10` | v2 temporal depth. v1 used 5 |

### 6.6 Full note-alignment orchestrator

`run_full_note_alignment.py` prepares the 14,100-row split, backfills exact maps, trains NoteFrameNet v2, trains the pairwise aligner, and evaluates. It is the historical E: drive pipeline, not the production Basic Pitch path.

```powershell
python align-model\scripts\run_full_note_alignment.py `
  --procedural-root E:\output `
  --raw-root E:\output_2k_rawdata `
  --out align-model\runs\note-align-full-e-s365 `
  --device cuda `
  --transcriber-epochs 24 `
  --aligner-epochs 12
```

| Flag | Default | Meaning |
|---|---|---|
| `--procedural-root` | `E:/output` | `procedural12k` bundles |
| `--raw-root` | `E:/output_2k_rawdata` | `raw2k` bundles |
| `--out` | `align-model/runs/note-align-full-e-s365` | Split, caches, weights |
| `--device` | `cuda` | Passed through to child trainers |
| `--workers` | `8` | Note-map backfill workers |
| `--seed` | `365` | Split seed |
| `--transcriber-epochs` | `24` | NoteFrameNet epochs |
| `--transcriber-workers` | `16` | Transcriber DataLoader workers |
| `--transcriber-batch-size` | `64` | Transcriber batch |
| `--transcriber-prefetch-factor` | `3` | Prefetch |
| `--transcriber-max-val-clips` | `80` | Val clip cap |
| `--aligner-epochs` | `12` | Pairwise aligner epochs |
| `--force-targets` | off | Rebuild exact-map caches |

### 6.7 Pairwise learned aligner

```powershell
python align-model\scripts\train_note_aligner.py E:\output `
  --manifest align-model\runs\note-align-full-e-s365\split.json `
  --output-dir align-model\runs\note-aligner `
  --epochs 8 `
  --batch-size 2048 `
  --lr 8e-4 `
  --device auto
```

The first positional argument is the bundle or exact-cache root.

| Flag | Default | Meaning |
|---|---|---|
| `--output-dir` | `runs/note-aligner` | Checkpoints |
| `--cache-dir` | none | Separate exact-map cache |
| `--manifest` | none | Frozen split; otherwise a random val split |
| `--epochs` | `8` | Training epochs |
| `--batch-size` | `2048` | Operation rows per step (minimum 32) |
| `--lr` | `8e-4` | AdamW learning rate |
| `--weight-decay` | `1e-3` | AdamW decay |
| `--hidden-dim` | `48` | MLP width |
| `--learned-weight` | `0.65` | Blend with hand-written costs, clipped to `[0, 1]` |
| `--device` | `auto` | Compute device |
| `--seed` | `365` | Reproducibility |
| `--max-samples` | `0` | Map cap |
| `--augmentations-per-map` | `1` | Synthetic corruptions per map |
| `--drop-probability` | `0.08` | Random deletion noise |
| `--pitch-error-probability` | `0.10` | Random pitch substitution |
| `--spurious-probability` | `0.08` | Inserted extra notes |
| `--timing-jitter-sec` | `0.035` | Onset jitter |
| `--calibration-maps` | `128` | Maps used to calibrate costs |
| `--early-stop-patience` | `3` | Early stop |
| `--transcriber-ckpt` | none | Also train on decoded audio notes |
| `--build-missing-cache` | off | Build exact caches from synth scores |

## 7. Joint outputRaw packed-release training

These scripts consume `data-packed/joint-outputraw-full-v1-shard64` through `DATA_READY.json`. Run them from `align-model/` unless you override every path.

The earlier joint path trained a sparse CRF decoder on Basic Pitch candidates, then attached Layer 2/3 error heads. Error-heads v1 used a failed target/rhythm/schema contract and scored 0.266 Layer 2 F1 with near-zero rhythm F1. v2 corrected those contracts (0.289 Layer 2 F1, 0.153 rhythm F1) but still failed promotion on isolated schema-range quality. Later v3–v5 changed decoding and calibration rather than the upstream encoder; none were promoted, and the 4,022-row lockbox stays sealed.

### 7.1 Full staged pipeline

```powershell
cd "D:\stuff\Audio Evaluation\ALIGN\align-model"
python scripts\train_outputraw_full_pipeline.py `
  --ready-marker runs\joint-outputraw-full-v1\DATA_READY.json `
  --output-dir runs\joint-outputraw-full-v1\training-v1 `
  --hardware-profile path\to\hardware-profile.json `
  --initialize-checkpoint path\to\joint_decoder.pt `
  --device cuda `
  --path-device cpu `
  --seed 20260915
```

| Flag | Default | Meaning |
|---|---|---|
| `--ready-marker` | `runs/joint-outputraw-full-v1/DATA_READY.json` | Verified pack marker |
| `--output-dir` | `runs/joint-outputraw-full-v1/training-v1` | Run directory |
| `--hardware-profile` | required | Measured loader/device profile |
| `--actual-training-profile` | none | Optional measured training profile |
| `--initialize-checkpoint` | required | Pretrained joint decoder |
| `--initialize-full-checkpoint` | none | Full-head initialization |
| `--candidate-rescorer-checkpoint` | none | Optional candidate scorer |
| `--minimum-candidate-f1` | `0.85` | Gate on the rescorer's stored val F1 |
| `--resume-checkpoint` | none | Continue a previous pipeline state |
| `--prepared-local-cache` | none | Packed local-example cache |
| `--prepared-max-open-shards` | `64` | Open shard cap |
| `--device` | required `cpu` or `cuda` | Acoustic/structure device |
| `--path-device` | `cpu` | Structured path-decoding device |
| `--resource-status` | `runs/TRAINING_RESOURCE_STATUS.json` | GPU/CPU lease file |
| `--seed` | `20260915` | Reproducibility |
| `--acoustic-epochs` | `1` | Stage epoch counts |
| `--structure-epochs` | `1` | |
| `--errors-epochs` | `1` | |
| `--joint-epochs` | `3` | |
| `--path-epochs` | `1` | |
| `--path-samples-per-epoch` | `1000` | Path-stage subset |
| `--learning-rate` | `2e-4` | Main AdamW LR |
| `--path-learning-rate` | `5e-5` | Path-stage LR |
| `--weight-decay` | `1e-4` | AdamW decay |
| `--gradient-accumulation` | `1` | Accumulated steps |
| `--checkpoint-every` | `250` | Rows between checkpoints |
| `--local-checkpoint-every` | none | Override for local stage |
| `--path-checkpoint-every` | none | Override for path stage |
| `--max-train-examples` | none | Train subset. Required with `--overfit-mode` |
| `--max-val-examples` | none | Val subset |
| `--overfit-mode` | off | Reuse one tiny subset across stages |
| `--bootstrap-replicates` | `1000` | Metric bootstrap |
| `--path-component-dim` | `32` | Path network width |
| `--residual-scale` | `0.20` | Residual mixing |
| `--structure-path-weight` | `0.0` | Structure-to-path loss mix |
| `--path-transfer-mode` | `exact` | `exact` or `inflate` |
| `--minimum-transfer-coverage` | `1.0` | Transfer coverage gate |
| `--freeze-pretrained-path` | off | Keep the validated legacy path frozen |
| `--legacy-path-lr-scale` | `0.05` | LR scale if the legacy path is not frozen |
| `--max-options` | `12` | Lattice option cap |
| `--max-states` | `48` | Lattice state cap |
| `--max-delete-events` | `24` | Delete-run cap |

If `PAUSE_REQUESTED.json` exists in `--output-dir`, the script refuses to start until that file is removed.

Overfit debug:

```powershell
python scripts\train_outputraw_full_pipeline.py `
  --hardware-profile path\to\hardware-profile.json `
  --initialize-checkpoint path\to\joint_decoder.pt `
  --device cuda `
  --overfit-mode `
  --max-train-examples 8 `
  --max-val-examples 4 `
  --acoustic-epochs 1 `
  --joint-epochs 1
```

### 7.2 Track B mel transcriber

```powershell
python scripts\train_mel_transcriber_v1.py `
  --cache path\to\mel-cache `
  --output-dir runs\joint-outputraw-full-v1\mel-transcriber-v1\full-training `
  --resource-status runs\TRAINING_RESOURCE_STATUS.json `
  --device cuda `
  --epochs 12 `
  --batch-size 8 `
  --learning-rate 3e-4 `
  --amp bf16
```

| Flag | Default | Meaning |
|---|---|---|
| `--cache` | required | Packed high-resolution mel cache |
| `--output-dir` | required | `best.pt`, `last.pt`, `history.json`, `config.json` |
| `--resource-status` | required | Lease file |
| `--resume` | none | Resume checkpoint |
| `--device` | `cuda` | Compute device |
| `--epochs` | `12` | Max epochs |
| `--batch-size` | `8` | Crops per step |
| `--crop-frames` | `1024` | Crop length |
| `--crops-per-clip` | `2` | Crops per bundle |
| `--learning-rate` | `3e-4` | AdamW LR |
| `--accumulation-steps` | `1` | Effective batch = batch × this |
| `--workers` | `4` | DataLoader workers |
| `--prefetch-factor` | `4` | Prefetch |
| `--checkpoint-every-steps` | `100` | Mid-epoch save |
| `--calibration-max-clips` | `64` | Decode-threshold calibration cap |
| `--calibration-fraction` | `0.05` | Fraction of val used to calibrate |
| `--calibration-every-epochs` | `2` | Calibration cadence |
| `--patience` | `3` | Early stop |
| `--amp` | `bf16` | `bf16`, `fp16`, or `none` |
| `--compile` | off | `torch.compile` |
| `--max-train-rows` | none | Train subset |
| `--augmentation-probability` | `0.55` | Mel augmentation rate |
| `--temporal-kind` | `tcn` | `tcn` or `bigru` |
| `--temporal-dim` | `128` | Temporal width |
| `--temporal-blocks` | `8` | Temporal depth |
| `--conv-channels` | `32` | Frontend channels |
| `--seed` | `20260916` | Reproducibility |

Effective batch size without changing memory:

```powershell
python scripts\train_mel_transcriber_v1.py `
  --cache path\to\mel-cache `
  --output-dir runs\...\mel-debug `
  --resource-status runs\TRAINING_RESOURCE_STATUS.json `
  --batch-size 4 `
  --accumulation-steps 2 `
  --max-train-rows 32 `
  --epochs 1
```

### 7.3 Joint decoder and end-to-end fine-tune

```powershell
python scripts\train_joint_decoder.py `
  --manifest path\to\split.json `
  --cache-root path\to\basic-pitch-cache `
  --output-dir runs\joint-decoder `
  --epochs 4 `
  --learning-rate 8e-4 `
  --device cuda
```

| Flag | Default | Meaning |
|---|---|---|
| `--manifest` / `--cache-root` / `--output-dir` | required | Data and output |
| `--initialize-checkpoint` | none | Warm start |
| `--seed` | `365` | Reproducibility |
| `--warmup-epochs` | `1` | Warmup before main epochs |
| `--epochs` | `4` | Main path-CRF epochs |
| `--learning-rate` | `8e-4` | AdamW LR |
| `--minimum-candidate-confidence` | `0.65` | Global candidate gate |
| `--hidden-dim` | `64` | Edge network width |
| `--gradient-accumulation` | `8` | Path-batch accumulation |
| `--max-train-samples` / `--max-val-samples` | none | Subsets |
| `--device` | `cuda` | Compute device |
| `--max-options` | `32` | Lattice caps |
| `--max-states` | `256` | |
| `--max-delete-events` | `16` | |
| `--noise-inference-bias` | `-3.0` | Bias against noise emissions |
| `--enable-continuation-feature` | off | Replay/resume feature |
| `--continuation-score-weight` | `0.0` | Continuation loss weight |
| `--continuation-hard-negative-copies` | `0` | Hard negatives for copies |
| `--repeat-fragment-penalty` | `0.0` | Penalty on short replay fragments |

`train_end_to_end_joint.py` then fine-tunes local edges and the path CRF together. Important extra knobs: `--local-epochs`, `--path-epochs`, `--local-learning-rate` (`2e-4`), `--path-learning-rate` (`8e-5`), `--local-distillation-weight` (`0.5`), `--residual-scale` (`0.10`), `--resume-checkpoint`, `--train-legacy-during-local`, and `--disable-continuation-feature`.

### 7.4 Candidate rescorer

```powershell
python scripts\train_candidate_rescorer.py `
  --manifest path\to\split.json `
  --basic-cache-root path\to\basic-pitch-cache `
  --example-cache-path path\to\examples `
  --output-dir runs\candidate-rescorer `
  --epochs 3 `
  --learning-rate 3e-4 `
  --device cuda
```

| Flag | Default | Meaning |
|---|---|---|
| `--epochs` | `3` | Training epochs |
| `--batch-candidates` | `65536` | Candidates per step |
| `--learning-rate` | `3e-4` | AdamW LR |
| `--hidden-dim` | `64` | MLP width |
| `--dropout` | `0.05` | Dropout |
| `--candidate-floor` | `0.50` | Lowest retained confidence |
| `--global-candidate-gate` | `0.65` | Production reference gate, left unchanged |
| `--hard-negative-ratio` | `1.0` | Negative sampling |
| `--short-weight-lt-80ms` / `120ms` / `180ms` | `4.0` / `3.0` / `2.0` | Extra loss on short notes |
| `--workers` / `--prefetch` | `3` / `6` | Loader |
| `--resume-checkpoint` | none | Resume |
| `--checkpoint-every-clips` | `0` | `0` disables mid-epoch saves |
| `--packed-training-root` | none | Optional packed examples |

### 7.5 Frozen error heads

v1 trains new Layer 2/3 heads on frozen upstream paths:

```powershell
python scripts\train_error_heads.py `
  --ready-marker runs\joint-outputraw-full-v1\DATA_READY.json `
  --upstream-checkpoint path\to\joint_decoder.pt `
  --output-dir runs\joint-outputraw-full-v1\error-heads-v1 `
  --phase all `
  --epochs 8 `
  --device cpu `
  --seed 20260915
```

| Flag | Default | Meaning |
|---|---|---|
| `--phase` | `all` | `all`, `extract`, `train`, `evaluate` |
| `--workers` | `4` | Feature-extract workers |
| `--epochs` | `8` | Head-training epochs |
| `--max-train` / `--max-val` | none | Subsets |
| `--device` | `cpu` | Heads were trained on CPU |
| `--resume` / `--no-resume` | resume on | Mid-epoch atomic resume |

v2 reuses the v1 example cache and canonical targets; add `--v1-cache`, `--canonical-targets`, and `--phase prepare|train|evaluate`.

v3–v5 are phase machines, not from-scratch trainers:

```powershell
python scripts\train_error_heads_v3.py analyze
python scripts\train_error_heads_v3.py freeze
python scripts\train_error_heads_v3.py score --freeze-manifest path\to\freeze_manifest.json

python scripts\train_error_heads_v5.py fit
python scripts\train_error_heads_v5.py freeze
python scripts\train_error_heads_v5.py score
python scripts\train_error_heads_v5.py verify
```

v5's `fit` uses only a leakage-grouped training subset. Do not point `--canonical-targets` at lockbox rows.

## 8. Other experimental trainers

These are specialized research scripts. Always start with `--help`. Defaults below are the argparse defaults, not promoted hyperparameters.

| Script | What it trains | Flags you will usually change |
|---|---|---|
| `train_identity_crf_v1.py` | Ornament identity CRF | `--epochs 4`, `--hidden 48`, `--learning-rate 1e-3`, `--runtime fast_v2\|reference_v1`, `--max-train-rows`, `--resume` |
| `train_drop_emit_lattice_v1.py` | Drop/emit lattice | `--epochs 3`, `--hidden 64`, `--learning-rate 1e-3`, `--max-train-rows`, `--checkpoint-every-steps` |
| `train_activation_candidate_scorer_v1.py` | Activation candidate scorer | `--epochs 4`, `--batch-size 4096`, `--learning-rate 5e-4`, `--resume` |
| `train_sequence_mapper_v1.py` | Fixed-score sequence mapper | `--epochs 12`, `--batch-size 24`, `--learning-rate 3e-4`, `--hidden-dim 32`, `--resume` |
| `train_mel_mapper_v1.py` | Mel operation mapper | `--epochs 6`, `--learning-rate 3e-4`, `--clips-per-step 8` |
| `train_orn_multipitch_v1.py` | Ornament multipitch | `--epochs 8`, `--batch-size 24`, `--learning-rate 3e-4`, `--amp bf16`, `--resume` |
| `train_basic_pitch_polyphonic_v1.py` | Polyphonic Basic Pitch residual | `--epochs 8`, `--batch-size 16`, `--learning-rate 2e-4`, `--resume` |
| `train_fast_note_multipitch_v1.py` | Fast-note multipitch | same family as ORN multipitch; inspect `--help` |
| `train_canonical_candidate_rescorer.py` | Packed-release candidate rescorer | `--epochs 6`, `--batch-size 8192`, `--learning-rate 2e-4`, `--resume-checkpoint` |
| `train_grammar_emission_crf_v2.py` | Grammar emission CRF | `--rows 1536`, `--seed 20260917` |
| `train_grammar_operation_core_v2.py` | Grammar operation core | `--rows 512` |
| `train_joint_plan_path_ranker_v3.py` | Plan-path ranker | `--rows 256`, `--max-candidates 12`, `--epochs 200` |
| `train_replay_plan_ranker_v3.py` | Replay plan ranker | inspect `--help` |
| `train_ornament_mapper_prior_v1.py` | Ornament mapper prior | inspect `--help` |

Most of these require `--ready-marker` or `--release-manifest` plus `--expected-*-sha256` hashes. Wrong hashes fail before training on purpose.

## 9. Recipes

### Overfit one batch

Use this to prove the command, device, and I/O path before a full run.

```powershell
align-model train-melody --data ".\synth-pipeline\output" --out ".\align-model\runs\melody-overfit" --overfit 4 --epochs 3 --device cuda
python align-model\scripts\train_mel_transcriber_v1.py --cache CACHE --output-dir OUT --resource-status STATUS --max-train-rows 8 --epochs 1 --device cuda
```

### Resume after a crash

```powershell
python align-model\scripts\train_note_transcriber.py --root E:\output --manifest SPLIT --out RUN --resume RUN\last.pt --epochs 30
python align-model\scripts\train_mel_transcriber_v1.py --cache CACHE --output-dir RUN --resource-status STATUS --resume RUN\last.pt
```

`--epochs` on NoteFrameNet is the **total** target, not additional epochs.

### Fit a 8 GB GPU

Lower batch size first. If the trainer has accumulation, restore the effective batch that way.

```powershell
align-model train-stages --batch-size 8 --epochs 8 --device cuda
python align-model\scripts\train_mel_transcriber_v1.py --batch-size 4 --accumulation-steps 2 --amp bf16 --workers 2
```

### Change architecture width

Only trainers that expose width flags can do this from the CLI:

```powershell
python align-model\scripts\train_note_transcriber.py --channels 48 --temporal-channels 96 --temporal-blocks 6 ...
python align-model\scripts\train_mel_transcriber_v1.py --temporal-kind bigru --temporal-dim 96 --temporal-blocks 6 ...
python align-model\scripts\train_note_aligner.py ROOT --hidden-dim 64
```

`align-model train-melody` cannot change `d_model` or layer counts from the CLI.

### Change the train/val split size without editing files

```powershell
align-model train-stages --max-samples 200
python align-model\scripts\train_note_repetition.py --manifest SPLIT --out OUT --train-samples 256 --val-samples 64
```

Do not pass `--train-samples` larger than the frozen split.

### Force CPU

```powershell
align-model train-stages --device cpu
python align-model\scripts\train_identity_crf_v1.py --device cpu ...
```

Joint path decoding often stays on CPU even when acoustics use CUDA (`--path-device cpu`).

## 10. What a successful run writes

Look in `--out` / `--output-dir`:

| File | Role |
|---|---|
| `best.pt` | Selected checkpoint (usually best validation metric) |
| `last.pt` | Most recent epoch, used for `--resume` |
| `mid_epoch_checkpoint.pt` | Crash recovery inside an epoch |
| `history.json` | Completed epochs, losses, and the metric used for selection |
| `config.json` | Frozen CLI/dataclass snapshot |
| `candidate-epoch-NNN.pt` | Periodic candidates |
| `PAUSE_REQUESTED.json` | Manual stop for the full joint pipeline |

`history.json` is the source of truth for how many epochs actually ran. Directory names containing `smoke`, `overfit`, `dev`, or `candidate` are diagnostics, not production.

Copy a checkpoint into a weights directory only after you decide to use it:

```powershell
Copy-Item align-model\runs\...\best.pt `
  align-model\runs\contextual-aligner-outputRaw_sf-1k\weights\contextual_note_aligner.pt
```

Training does not do that copy automatically.

## 11. After training

```powershell
# Note-first / Model A pipeline
align-model run --sample path\to\bundle --weights path\to\weights --device cuda

# Melody-first
align-model run-melody --sample path\to\bundle --ckpt path\to\best.pt
align-model eval-melodies --data path\to\bundles --pred melody_pred.json
```

Headline numbers must be official note-wise F1 (`melody_f1` / `official_note_wise`). Do not use timestamp IoU, pitch-list similarity, or checkpoint train loss for promotion.

## 12. Quick flag index

Shared names across many trainers:

| Flag | Typical effect |
|---|---|
| `--data` / `--root` / `--cache` / `--manifest` / `--ready-marker` | Where training examples come from |
| `--out` / `--output-dir` | Where checkpoints go |
| `--epochs` | Maximum epochs |
| `--batch-size` | Examples per step |
| `--lr` / `--learning-rate` | Optimizer step size |
| `--weight-decay` | AdamW decay |
| `--device` | `cuda`, `cpu`, or `auto` |
| `--seed` | Reproducibility |
| `--workers` | DataLoader processes |
| `--resume` / `--resume-checkpoint` | Continue from a checkpoint |
| `--max-samples` / `--max-train-rows` / `--overfit` | Subset the data |
| `--patience` | Early stopping |
| `--amp` | Mixed precision |
| `--hidden-dim` / `--channels` / `--temporal-*` | Model width/depth |
| `--resource-status` | Single-machine GPU/CPU lease |

If a flag from this table is missing on a command, that knob is hardcoded for that trainer.
