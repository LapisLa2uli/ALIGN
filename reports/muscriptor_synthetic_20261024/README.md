# MuScriptor synthetic transcription benchmark — 2026-10-24

**MuScriptor medium scored 88.22% pitch-sequence F1 on 100 randomly selected
clips from the frozen dataset 9.2 test split.** Precision was 88.55%, recall
87.89%, and the clip-bootstrap 95% interval was 84.48–91.33%.

The authoritative report is [results_audited.json](results_audited.json).
All 106 inference requests succeeded; no selected clips were dropped.

## Main result: realistic synthetic clarinet audio

| Model / population | Clips | Reference notes | Precision | Recall | Sequence F1 |
|---|---:|---:|---:|---:|---:|
| MuScriptor medium, all selected test clips | 100 | 11,710 | 88.55% | 87.89% | **88.22%** |
| MuScriptor, procedural clips | 28 | 3,518 | 87.54% | 79.31% | 83.22% |
| MuScriptor, score-derived clips | 72 | 8,192 | 88.93% | 91.58% | 90.23% |
| Saved ALIGN CTC R2A, exactly the same 100 clips | 100 | 11,710 | 97.45% | 96.34% | **96.89%** |

MuScriptor is 8.67 percentage points below the saved CTC result on this sample.
CTC was trained on the dataset's training split; MuScriptor used its upstream
pretrained medium weights without ALIGN fine-tuning. The CTC comparison reuses
saved predictions; it is not a fresh CTC inference run.

The metric is the longest common subsequence of predicted and reference
**written MIDI pitches**, computed separately per clip and micro-averaged:
`precision = total matches / total predicted notes`,
`recall = total matches / total reference notes`,
`F1 = 2 * total matches / (total predicted + total reference notes)`.
MuScriptor emitted 11,623 notes, of which 10,292 matched.
Timing, note offsets, and instrument identity do not enter this metric.
All predicted instruments and duplicate notes are retained in the denominator.

Dataset 9.2 contains Muse Sounds renders with phone-style degradation. Its
music21 MIDI timestamps drift from the rendered audio: the existing 40-clip
[timing audit](../../align-model/runs/realistic92-transcriber-v1/timing-diagnostic.json)
found a median onset offset of approximately 0.48 seconds. Consequently, its
39.41% exact-pitch/50-ms-onset diagnostic is not a reliable measure of model
timing accuracy. Sequence F1 follows the established
[dataset evaluation protocol](../../align-model/runs/realistic92-transcriber-v1/RESULT_SUMMARY.md).
This is **not** ALIGN's official score-event/error-detection F1.

## Separate onset-accuracy check: six retained FreePats clips

| Metric | Reference notes | Predicted notes | Matches | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Exact pitch and onset within 50 ms | 509 | 594 | 468 | 78.79% | 91.94% | **84.86%** |
| Pitch-sequence LCS | 509 | 594 | 487 | 81.99% | 95.68% | 88.30% |

These are all six retained clips in `E:/ALIGN_Data/outputRaw_sf_10k`, not a
random sample of its former 10,000 clips. They are a small timing diagnostic,
not a broad estimate of SoundFont performance. Note offsets are not scored.

**Reference audit:** five of the six legacy `note_map.json` pitch sequences
disagreed with the performance MIDI. In the most severe case,
`synth_WeberITAV_0045`, the note map retained only 3 of 79 MIDI notes; another
clip retained 25 of 125. The `soundfont_v1` renderer consumes
`performance_audio.mid` directly, so all six clips were rescored against that
MIDI's note-on events and tempo map, using metadata to convert MIDI pitches to
written pitch. The same cached predictions were reused without modification.
No dataset files were changed. [GOLD_AUDIT.json](GOLD_AUDIT.json) records MIDI
hashes, count discrepancies, and the superseded scores.

The preliminary 61.62% onset score in `results.json` used stale note maps and
is invalid. Use **results_audited.json**. The main 100-clip result is unchanged:
all 100 frozen reference pitch sequences also matched their performance MIDI.

## Selection, inference, and verification

- Fixed sample: 100 of the frozen 2,028 test clips, selected using
  `random.Random(20261024).sample(sorted(test_names), 100)` before inference.
  The sample contains 28 procedural clips and 72 clips from the two held-out
  source scores. The confidence interval uses 2,000 clip-bootstrap resamples;
  it does not establish generalization across a broad population of scores.
- Verified the frozen split and test-hash-manifest SHA-256 values, then verified
  audio and reference-note hashes for every selected realistic clip.
- MuScriptor 0.4.0, upstream commit
  `8a1ef88672cc23c26619adaff37ec48d82c0b906`, medium model revision
  `f32236969308476e01fd3aae67357de5feb05a2d`.
  Model SHA-256:
  `ac80adbdf85d87231735fd948af7013441c0afced316c4e9067fd5d8a7fb97ec`.
- NVIDIA RTX 2000 Ada laptop GPU; CUDA 12.8 PyTorch 2.11; float16 model weights.
  Default greedy decoding, batch size 1, prelude forcing enabled; no instrument
  hint, score input, pitch-range filtering, note merging, or test-set tuning.
- Used raw `/transcribe` note events with tempo detection disabled, avoiding
  MIDI bar-padding, quantization, and beat-based onset adjustments. Metadata
  supplied the sounding-to-written pitch conversion (+2 semitones here).
- The LCS and onset-match implementations agreed with existing workspace
  scorers on 1,000 randomized test cases each.
- 62.88 minutes of realistic audio took 14.32 minutes of inference; the six
  FreePats clips added 3.14 minutes of audio and 48.68 seconds of inference.
  Reported inference time includes local HTTP transfer and audio decoding.

## Reproduce or inspect

With the local server running with the configuration above, from the ALIGN root:

```powershell
muscriptor\.venv\Scripts\python.exe align-model\scripts\benchmark_muscriptor_local.py --out reports\muscriptor_synthetic_20261024 --count 100
muscriptor\.venv\Scripts\python.exe align-model\scripts\audit_muscriptor_synthetic_gold.py --run reports\muscriptor_synthetic_20261024
```

The first command reuses saved predictions when configuration and input hashes
match. Use a new output directory for an independent inference run, then pass
that directory to the audit command.

- [manifest.json](manifest.json): frozen selection, model identity, and inputs.
- [results_audited.json](results_audited.json): authoritative aggregates and per-clip scores.
- [GOLD_AUDIT.json](GOLD_AUDIT.json): legacy reference discrepancies and corrected sources.
- `predictions/<corpus>/<sample>.json`: original model events and inference times.
- [Benchmark script](../../align-model/scripts/benchmark_muscriptor_local.py).
- [Reference audit script](../../align-model/scripts/audit_muscriptor_synthetic_gold.py).
