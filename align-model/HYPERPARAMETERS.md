# Best-run hyperparameters

This document records the **selected hyperparameter card** for each ALIGN model: the settings that produced that model's best comparable result. It is not a catalog of CLI defaults. To change flags when you retrain, use [TRAINING.md](TRAINING.md). Model history and promotion status also live in [README.md](README.md).

"Best" here means the configuration that won **inside that model family**, on the comparison that family actually used. A later version is not automatically better. Official headline scoring is exclusive one-to-one **note-wise F1** on canonical score-event identity (same location and type = 1.0, same location and different type = 0.5, wrong location = 0). Timestamp IoU and acoustic onset F1 are diagnostics only.

Run artifacts (`history.json`, checkpoint `train_config`, `evaluation-*.json`) override this file when they exist. The `runs/` directories are the source of truth for completed epochs and calibrated thresholds.

## How to read a card

| Field | Meaning |
|---|---|
| Winning run | The variant and dataset that produced the best result |
| Why it won | What earlier versions did, the failure, and the change that targeted it |
| Status | Promoted, retained as a candidate, rejected, or experimental |
| Headline | The number used to select it, with the exact metric name |
| Reproduce | The CLI that reconstructs this card. Unlisted knobs stay at source defaults |

## Summary

| Family | Winning card | Headline | Status |
|---|---|---|---|
| Transcription (production) | Frozen Basic Pitch 0.4.0, cleanup + calibrated decode | 0.657 P/R/F1 on `outputRaw_sf_10k` 200-test transcription | Current decoder |
| Transcription (standalone AMT bench) | Same Basic Pitch, onset 0.50 / frame 0.40 | 0.827 test-ID acoustic F1 | Diagnostic only; not official note-wise F1 |
| Layer 1 | `outputRaw_sf_10k` hybrid scorer | 0.666 candidate-row val F1; used by the current pipeline | Promoted for that dataset |
| Note-to-score aligner | Contextual cached-sequence v3 + `contextual` decode | 0.509 mapping F1 on the locked 200-test subset | Current aligner |
| Model A | Procedural-only stages 1–3 | 0.206 type-aware melody F1 on 100-bundle holdout | Historical |
| Model B | Bakeoff v1 control | 0.249 holdout F1 | Best bakeoff version; below Model A / note-first |
| RUMAA-lite | Default transformer | Val error acc 0 | Do not use |
| NoteFrameNet | v2 | Calibrated val F1 0.1115 | Rejected |
| Clarinet refiner | v3 bounded run | Test-ID F1 0.304 vs Basic Pitch 0.827 | Not promoted |
| Pairwise aligner | Optimized v3 | Val mapping F1 ~0.297 | Rejected |
| Joint error heads (trained classifier) | v2 predicted-upstream | Layer 2 typed-error F1 0.289 | Experimental |
| Joint error heads (honest four-type) | v5 hybrid | Four-type F1 0.209 | Experimental; not promoted |

---

## 1. Production note-first stack

Production inference loads `runs/contextual-aligner-outputRaw_sf-1k/weights/`: frozen Basic Pitch decoder, Layer 1 `note_repetition.pt`, and contextual `contextual_note_aligner.pt`. Those weights were trained only on `outputRaw_sf_10k` (1,000 train / 200 val from a frozen 8,004 / 999 / 997 split). Reported final comparisons use 200 held-out test bundles.

### 1.1 Frozen Basic Pitch 0.4.0 (current transcriber)

Earlier NoteFrameNet v1/v2 learned notes from log-mel and stalled at 0.05–0.11 val F1. Frozen Basic Pitch was adopted because the standalone AMT bench matched Tsumugi (0.827 vs 0.822 test-ID) while being easier to cache. A later clarinet refiner tried to close remaining monophonic errors and lost (0.304 vs 0.827), so production keeps frozen Basic Pitch.

Two different calibrations exist. They are **not interchangeable**.

**Production pipeline decode** (cleanup + monophony + validation-calibrated thresholds on `outputRaw_sf_10k`):

| Setting | Value |
|---|---|
| Upstream weights | Spotify Basic Pitch 0.4.0 (not trained here) |
| Cache / frontend | `align-basic-pitch-0.4.0-v2` |
| Onset threshold | `0.55` |
| Frame threshold | `0.35` |
| Minimum note | 55 ms |
| Frontend frequency range | 45–2,600 Hz |
| Written MIDI keep range | 50–96 |
| Same-pitch merge gap | 0.10 s |
| Merge onset threshold | 0.60 |
| Harmonic overlap drop | 0.55 |
| Cleanup | Monophony, harmonic/high-note filter, weak-onset same-pitch merge |
| Headline | P/R/F1 **0.649 / 0.665 / 0.657**, pred/target count ratio 1.024 on 200 test clips |

Cleanup alone moved default F1 from 0.648 to 0.650 and the count ratio from 1.077 to 1.018. The 0.55 / 0.35 pair is the validation-calibrated step that reached 0.657.

**Standalone AMT bench** (no ALIGN cleanup; 40-clip calibration, up to 80 clips/split on the full note-alignment manifest):

| Setting | Value |
|---|---|
| Onset / frame / min note | `0.50` / `0.40` / 55 ms |
| Val / test-ID / test-OOD F1 | 0.826 / **0.827** / 0.558 |

That acoustic onset/pitch F1 is not official note-wise model F1 and is not used for checkpoint promotion.

Source `BasicPitchDecodeConfig` defaults are the bench pair (`0.50` / `0.40`). Loaded `note_decoder.json` overrides them for production.

### 1.2 Layer 1 `NoteRepetitionScorer`

Earlier Stage 1 `RestartScorer` scored paired mel crops. It could not see transcribed note identity, so long phrase repeats and one-note replays were mixed into one Siamese decision. Layer 1 replaced that with deterministic tempo-tolerant phrase search plus a small learned rescue for one-note cases.

The procedural checkpoint reached 0.755 hybrid repetition F1 on 100 test-ID clips versus 0.753 deterministic (71/89 vs 70/89 found). The `outputRaw_sf_10k` card has the better candidate-row val F1 (0.666 vs 0.624) and is the production weight.

| Setting | Winning value |
|---|---|
| Dataset | `outputRaw_sf_10k`, frozen `runs/contextual-aligner-outputRaw_sf-1k/split.json` |
| Train / val bundles | 1,000 / 200 |
| Architecture | 2-hidden-layer MLP, `hidden_dim=48`, SiLU, dropout 0.10 |
| Features | Sequence-candidate vector, feature version 2 (includes continuation / restart-gap) |
| Caps | 32 sources/note, 64 notes/phrase, 2 extra repetitions |
| Loss | Weighted BCE (`pos_weight = negatives / positives`) |
| Optimizer | AdamW, **lr `8e-4`**, weight decay `1e-3` |
| Batch / epochs / seed | 512 / 25 / 365 |
| Decode threshold | `0.65` (model config default; checkpoint may override) |
| Pipeline gate | `note_repetition_min_confidence=0.80`, `use_note_repetition_model=True` |
| Headline | Candidate-row val F1 **0.666** |
| Status | Promoted for the dataset-specific pipeline |

```powershell
python align-model\scripts\train_note_repetition.py `
  --manifest align-model\runs\contextual-aligner-outputRaw_sf-1k\split.json `
  --out align-model\runs\contextual-aligner-outputRaw_sf-1k\weights\note_repetition.pt `
  --train-samples 1000 --val-samples 200 --epochs 25 --batch-size 512 --seed 365
```

Long phrases stay deterministic. Do not raise `--train-samples` above the frozen train split.

### 1.3 Contextual note aligner

Pairwise learned aligners (tiny residual MLP over match/sub/extra/delete features) peaked at about 0.297 val mapping F1 and were rejected. Contextual exact-map v1 then used a bidirectional GRU on `procedural12k` (10,000 / 300) and reached 0.711 val note-position accuracy, but **failed pipeline promotion**: 0.0244 mapping F1 versus 0.0256 deterministic on 100 Basic Pitch clips. v2 kept the same architecture and trained only on `outputRaw_sf_10k` (1,000 / 200). That closed the domain gap and beat deterministic alignment 0.470 vs 0.450.

v2 still trained on synthetic exact maps, then lost accuracy on real Basic Pitch + Layer 1 sequences. Cached-sequence v3 fine-tunes v2 on those actual inference inputs (monotonic sequence NLL + 0.25 local loss). Mapping F1 rose from 0.470 to 0.503, then 0.509 after decoder calibration. That is the current checkpoint.

**v2 exact-map pretrain (required initializer):**

| Setting | Winning value |
|---|---|
| Data | `outputRaw_sf_10k`, 1,000 train / 200 val |
| Architecture | 2-layer bidirectional GRU, `hidden=64`, dropout 0.10, deletion cost 0.90 |
| Optimizer | AdamW, **lr `5e-4`**, weight decay `1e-3`, grad clip 2 |
| Batch / epochs / seed | 32 / 8 / 365 |
| Headline | Val note accuracy 0.782; 200-test mapping F1 0.470 |
| Status | Promoted, then used only as v3 initialization |

```powershell
python align-model\scripts\train_contextual_note_aligner.py `
  --manifest ...\split.json --out ...\contextual_note_aligner.pt `
  --train-samples 1000 --val-samples 200 --epochs 8 --batch-size 32 --seed 365
```

**v3 cached-sequence fine-tune (current weights):**

| Setting | Winning value |
|---|---|
| Inputs | Cached Basic Pitch notes + Layer 1 repeats, same 1,000 / 200 split |
| Initialize from | v2 exact-map checkpoint |
| Optimizer | AdamW, **lr `2e-4`**, weight decay `1e-3`, grad clip 2 |
| Loss | Structured monotonic sequence NLL + **0.25** local loss |
| Batch / epochs | 8 / 5 |
| Deletion cost | 0.90 |
| Headline | Val note accuracy 0.665; 200-test mapping F1 **0.503** before threshold calibration, **0.509** after |
| Status | Current production aligner |

```powershell
python align-model\scripts\train_cached_contextual_aligner.py `
  --manifest ...\split.json `
  --basic-cache-root CACHE `
  --repetition-checkpoint ...\note_repetition.pt `
  --initial-checkpoint ...\contextual_note_aligner.pt `
  --out ...\contextual_note_aligner.pt `
  --train-samples 1000 --val-samples 200 --epochs 5 --batch-size 8
```

### 1.4 Alignment decode strategy

The same v3 checkpoint was decoded four ways on the locked 200-test subset:

| Strategy | Mapping F1 | Status |
|---|---:|---|
| `contextual` | **0.509** | Default (`PipelineConfig.note_alignment_strategy`) |
| `revision` | 0.506 | Available, not default |
| `contextual_continuation` | 0.468 | Precision experiment. Precision 0.582 vs 0.511, recall 0.392 vs 0.507. Not promoted |
| `multi_start` | 0.463 | Available, not default |

Use `contextual`. The continuation rule is the later decode variant; it did not beat `contextual` on F1.

---

## 2. Historical error detectors

### 2.1 Model A (four-stage crops)

Model A is the last pre-note-first error detector. It trains three crop networks on first-pass gold. The final comparable run trained **only on `E:\output`** (no raw-derived data) and scored **0.206** type-aware melody F1 on a locked 100-bundle holdout. That is the family best. It is historical; the note-first stack replaced it.

Shared training card for stages 1–3:

| Setting | Winning value |
|---|---|
| Data | Procedural `E:\output` / `random12k` |
| Optimizer | AdamW, **lr `1e-3`**, weight decay `1e-2`, cosine schedule |
| Batch / epochs | 32 / 8 |
| Val / seed | 10% sample-level split / 365 |
| Audio | Cached `performance_mel.npy`, 128 bins |

**Stage 1 `RestartScorer`**

| Setting | Value |
|---|---|
| Architecture | Siamese 3-layer 1-D conv, mean+max pool, 64-d embedding, features `[a, b, \|a-b\|, a*b]`, dropout 0.15, binary head |
| Supervision | First-pass `repetition` vs source; split `extra_copies==2`; silent gap is a hard negative |

**Stage 2 `EditCropNet`**

Unfiltered crops were match-dominated and drowned the four error classes. The winning data card caps training at **20,000** examples: 8k match + 3k each of miss / extra / wrong / intonation. A hard-negative binary gate won a 20-clip calibration comparison (F1 0.129) over the five-way head, but the full 100-bundle Model A number above still uses the staged pipeline, not that isolated gate.

| Setting | Value |
|---|---|
| Architecture | Two 1-D convs + global pool, 5-way (match / missed / extra / wrong / intonation) |
| Stage 2 cap | `--stage2-max-train-examples 20000` |

**Stage 3 `RhythmNet`**

Learned rhythm logits were noisy. The selected Model-A decode **sets `logit_threshold=20`**, which suppresses almost all learned rhythm predictions. Conservative duration rules later replaced this head.

```powershell
align-model train-stages --data "E:\output" --out ".\align-model\runs\stages" `
  --epochs 8 --batch-size 32 --lr 1e-3 --stages 1,2,3 --stage2-max-train-examples 20000
```

`stages-random12k-setsoft` and `stages-random12k-typed` reuse this card. The names are metric variants, not new architectures.

### 2.2 Model B (melody-first bakeoff)

All seven versions shared one backbone and one training card so only the objective/decode changed:

| Shared setting | Value |
|---|---|
| Data | 10,800 `random12k` train / 1,200 val; 100-bundle holdout |
| Backbone | 128-bin mel, `d_model=256`, 4 heads, 4 audio / 4 score / 2 fusion layers, dropout 0.10, stride 4 |
| Sequence caps | 1,920 audio frames, 128 score notes |
| Optimizer | AdamW, **lr `2e-4`**, weight decay `1e-2`, grad clip 1 |
| Batch / seed | **2** / 365 |
| Early stop | Shared; every version stopped at 3 epochs / 16,200 steps |

v1 is the only usable bakeoff version (holdout F1 **0.249**). v2/v3/v5/v6/v7 collapsed to all-match or empty spans. v4 scored 0.156 by emitting one broad fault per clip. v1 is therefore the family best, but it is still below Model A's 0.206 and below the note-first path.

**v1 winning extras:**

| Setting | Value |
|---|---|
| Heads | 7-way per-note type CE + clip-level `extra_copies` ∈ {0,1,2} |
| Loss weights | error **6**, copies **1**, coverage **2**, error auxiliary **3** |
| Decode | Contiguous same-type non-match runs |
| `--variant` | `v1` |

```powershell
align-model train-melody --data ".\synth-pipeline\output" --out ".\align-model\runs\melody" `
  --epochs 8 --batch-size 2 --lr 2e-4 --variant v1
```

CLI default `--batch-size 4` is **not** the bakeoff winner. Use `2` to reconstruct the 0.249 card. Other retained Model B directories (`melody-random12k`, `melody-raw2k`, and the `-set` variants) are earlier metric/dataset copies of v1, not better architectures.

### 2.3 RUMAA-lite

Same backbone as Model B. Defaults: 20 epochs, batch 4, AdamW lr `2e-4`, weight decay `1e-2`, 10% val, grad clip 1, seed 365. Validation error accuracy stayed 0. There is no winning hyperparameter card. Do not select this model.

---

## 3. Rejected transcription and pairwise alignment

### 3.1 NoteFrameNet v2 (best rejected transcriber)

v1 used four spectral blocks, 48-channel temporal residuals, and no F0/cents head. Best val F1 was 0.0417 at epoch 12 (calibrated 0.0547). v2 added multiscale skips, 128 temporal channels, ten dilated blocks, PESTO F0, and a cents head. That raised val F1 to **0.1115** at epoch 10. Absolute F1 stayed far below frozen Basic Pitch, so both were rejected. v2 is the family best.

| Setting | Winning v2 value |
|---|---|
| Data | Full 14,100-row manifest: 10,871 train / 1,338 val |
| Architecture | 3 spectral blocks, 64 spectral channels, 128 temporal channels, 10 temporal blocks, dropout 0.12, MIDI 36–108, multiscale + F0 + cents on |
| Optimizer | AdamW, **lr `3e-4`**, weight decay `1e-3`, grad clip 2 |
| Crops | 1,024 frames, 2 crops/clip, batch 8 |
| Epochs / patience / seed | up to 30 / 5 / 365; best at epoch **10** |
| Loss weights | voiced 1, pitch 1, onset 2, offset 1.5, cents 0.5; onset pos-weight 12, offset pos-weight 10 |
| Selected decode | voiced / onset / offset **0.50 / 0.45 / 0.40** |
| Headline | Best val F1 **0.1115**; cents MAE 2.96 on matched notes |
| Status | Rejected |

v1's calibrated decode (0.55 / 0.30 / 0.30) belongs only to v1.

### 3.2 Basic Pitch clarinet refiner v3

This later refiner froze Basic Pitch maps and trained a residual Semi-CRF on top, targeting NoteFrameNet's low F1. The bounded promotion run still lost to frozen Basic Pitch (0.304 vs 0.827 test-ID F1) and was not promoted. The card below is that run, not a production recipe.

| Setting | Value |
|---|---|
| Data | Procedural-only strict manifest; 256 train / 80 val / 80 test |
| Architecture | 96-channel, 6-block temporal residual + sparse interval Semi-CRF |
| Optimizer | AdamW, **lr `3e-4`**, weight decay `1e-3`, grad clip 2 |
| Crops | 384 frames, 2 crops/clip, batch 8 |
| Epochs | 8 requested, 6 completed, patience 4 |
| Interval weight | 0.20 |
| Seed | 365 |
| Headline | Test-ID F1 **0.304** vs frozen Basic Pitch **0.827** |
| Status | Not promoted |

### 3.3 Pairwise learned aligner v3

v1 (~0.245) and v2 (~0.252) used the same tiny MLP. v3 corrected pitch targets, used rendered-note supervision, and vectorized generation. Val mapping F1 rose to about **0.297**. That is the pairwise family best. End-to-end mapping stayed too weak, so all three were rejected in favor of the contextual GRU.

| Setting | Winning v3 value |
|---|---|
| Data | Full 14,100-row manifest |
| Architecture | Residual MLP, `hidden=48`, dropout 0.08; blend weight 0.65 with hand costs |
| Optimizer | AdamW, **lr `8e-4`**, weight decay `1e-3` |
| Batch / epochs / seed | 2,048 / 9 retained / 365 |
| Augmentation | drop 0.08, pitch error 0.10, spurious 0.08, timing jitter 35 ms |
| Status | Rejected |

---

## 4. Joint outputRaw experiments

These cards use the sealed `joint-outputraw-full-v1` release: 4,544 Mozart train, 358 fingerprint-isolated Mozart val, 4,022 metadata-only locked Weber test rows. Intonation is masked. None of the error-head versions are promoted.

### 4.1 Packed loader (selected I/O card)

Repeated layout/worker tests on the measured i7-13700H / 32 GB / NVMe machine selected:

| Setting | Value |
|---|---|
| Pack | `data-packed/joint-outputraw-full-v1-shard64` (4,902 records, 32.14 GB) |
| Shard size | 64 rows |
| Workers / prefetch | 4 / 16 |
| Max open shards | 4 |
| Pinned memory | off |
| Checksums | on |
| Seed | 20260915 |
| Throughput | 108.19 rows/s @ 100 rows, 67.82 rows/s @ 1,000 rows (vs ~1 row/s NPZ+SQLite) |

### 4.2 Frozen joint decoder used by error heads

Error-heads v1–v5 reuse one completed replay-aware joint aligner/decoder. They do not retrain it.

| Setting | Value |
|---|---|
| Candidate generation | `align-joint-candidates-v2-short-rescue` |
| Short-rescue decode | min note 30 ms, `adaptive_short_note_rescue=True`, plus the two legacy high-recall grids and the frozen 0.50/0.40 decode |
| Global candidate gate | 0.65 |
| Pairing tolerance | 50 ms |
| Checkpoint SHA-256 | `91baaa26dc408cd5ffdadea66bcb9fe9b61fac3bda4754f0a4f20392750fa2f4` |

The later end-to-end fine-tune card (local then path) that produced the replay-aware decoder uses:

| Setting | Value |
|---|---|
| Local stage | 1 epoch, lr `2e-4`, distillation 0.5, batch 65,536 edges, freeze legacy path |
| Path stage | 1 epoch, lr `8e-5`, 1,000 train samples, accumulation 8, path device CPU |
| Weight decay / clip | `1e-4` / 5.0 |
| Hidden / component / residual | 64 / 32 / 0.10 |
| Lattice | max options 12, max states 48, max deletes 24, noise bias `-6.0` |
| Continuation | enabled, score weight **0.35**, 1 hard-negative copy, fragment penalty 0.25 |
| Seed | 365 |

### 4.3 Error heads

v1 trained Layer 2/3 heads on frozen paths but used a failed target/rhythm/schema contract (Layer 2 F1 0.266, rhythm F1 0.040, schema-range F1 0.052). v2 kept the same optimizer and corrected those contracts. That is the best **trained classifier** card. v3–v5 then changed decoding/calibration rather than the encoder; their four-type numbers are not official note-wise F1 and must not be used for promotion.

**Shared v1/v2 training card:**

| Setting | Value |
|---|---|
| Data | All 4,544 train rows; calibrate on all 358 val rows; lockbox unopened |
| Architecture | MLP, `hidden_dim=128`, dropout 0.10; Layer 2 classes match/wrong/extra/missed; rhythm short/long |
| Optimizer | AdamW, **lr `2e-4`**, weight decay `1e-4` |
| Loss | Focal CE, gamma **1.5**, measured class weights; balanced rhythm BCE; deviation regression |
| Device / batch | Eager FP32 CPU, batch **32,768** (626,048 rows/s) |
| Epochs / seed | 8 / 20260915 |
| Thresholds | Validation-only per class |

**v2 predicted-upstream headline (best trained heads):**

| Metric | Value |
|---|---|
| Layer 2 typed-error P/R/F1 | 0.203 / 0.499 / **0.289** (v1 was 0.266; rules 0.139) |
| Rhythm P/R/F1 | 0.100 / 0.324 / **0.153** (v1 0.040) |
| Deviation MAE | 70.3 ms |
| Isolated four-type legacy pitch-list F1 | 0.092 (range-only 0.129) after gold isolation |

Oracle-upstream Layer 2 F1 0.972 shows the remaining ceiling is upstream localization, not this head's optimizer.

**Later decode cards (no classifier retraining):**

| Version | What changed | Four-type F1 | Caveat |
|---|---|---:|---|
| v3 max-F1 cluster decode | Sequence clustering, hysteresis, one-to-one NMS | 0.193 | Legacy pitch-list; not note-wise |
| v4 hybrid + val-fitted linear selector | Direct operations blended with v2 | **0.226** | Selector fitted on validation (leakage) |
| v5 train-only selector | Leakage-group 3,634 fit / 910 calibrate; val opened once | 0.209 | Honest; missed the 0.02 material-gain gate |

If the question is "best trained Layer 2/3 hyperparameters," use **v2**. If the question is "best honest four-type policy," use **v5**. v4 looks better than v5 only because it tuned on validation. None are production.

---

## 5. External AMT bench (not trained here)

Calibrated on 40 clips; evaluated on up to 80 clips/split from the full note-alignment manifest. Acoustic onset/pitch F1 only.

| Model | Winning decode | Test-ID F1 |
|---|---|---:|
| Basic Pitch 0.4.0 | onset 0.50, frame 0.40, min 55 ms | **0.827** |
| Tsumugi | checkpoint-native interval decode | 0.822 |
| PENN | conf 0.40, onset 0.20, min 50 ms, pitch-change 0.65, max gap 2 | 0.461 |
| PESTO 2.0.1 | conf 0.85, onset 0.50, min 90 ms, pitch-change 0.65, max gap 2 | 0.407 |
| TorchCREPE tiny | conf 0.10, onset 0.20, min 90 ms, pitch-change 0.65, max gap 2 | 0.283 |

Basic Pitch was selected for integration, not because a repository training run beat Tsumugi.

---

## 6. Reconstructing a card

1. Copy the PowerShell block on that card, or the nearest command in [TRAINING.md](TRAINING.md).
2. Keep `--seed` at the listed value (365 for note-first / A / B; 20260915 for error heads).
3. Do not raise `--train-samples` / `--val-samples` above the frozen split.
4. After training, compare `history.json` to the headline in this file. If they disagree, the run artifact wins.
5. Promote only on official note-wise F1. Do not promote from train loss, timestamp IoU, or standalone AMT F1.
