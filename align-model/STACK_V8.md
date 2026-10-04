# Acoustic same-pitch repair (v8)

This separately versioned pipeline adds a score-free postprocessor after v7
transcription, before alignment. It examines every pair of adjacent equal MIDI
pitches. It combines short-window mel continuity, centered 12 ms waveform RMS,
and the transcriber's voiced/onset/rearticulation outputs. Only continuous,
voiced audio with no clear attack is merged. Missing, ambiguous, or insufficient
context retains both notes. It does not use decoder end times to infer silence:
those ends were synthesized from the following candidate's start.

The existing v7 files, checkpoint, frozen candidate and GUI default are unchanged.
No retraining is needed. v8 remains experimental and is not promoted: evaluation
found no accuracy improvement and low sensitivity to artificial splits.

## Run

From the repository root in PowerShell:

```powershell
& align-model/.venv-amt-bench/Scripts/python.exe align-model/scripts/run_stack_v8.py `
  --score DataCreate/samples/020/verified_score.musicxml `
  --audio DataCreate/samples/020/performance_audio.wav `
  --output align-model/runs/stack-v8/example-020.json
```

The runner accepts WAV/MP3 via the existing audio loader, resamples to 22050 Hz,
and defaults to `runs/stack-v8/CANDIDATE_STACK_V8.json`. The same written
B-flat-clarinet pitch convention as v7 applies. Candidate source, checkpoint and
verifier hashes are checked. The output schema is `align-score-feedback-v8`.

`diagnostics.same_pitch_repair` contains the threshold configuration, original
candidate index groups for each output note, and one decision per original
same-pitch boundary. Each decision contains time, pitch, reason and measured
evidence when context is sufficient. Decisions are made against the original
boundaries, so chains are order-independent. A merged note spans the source
notes, keeps the maximum confidence and is optional only if all sources were
optional; conflicting alternative-pitch hypotheses are cleared.

Alignment and error gating run on the repaired transcription. Feedback still
uses canonical reference-score event indices as its primary identity; timestamps
are only for playback. This is not a cosmetic deletion of final JSON labels.

## API

`alignmodel.transcription.same_pitch_v1.repair_same_pitch(rows, outputs, mel,
audio=audio)` returns `(repaired_rows, audit)`. Rows use the existing v7 format.
Mel must use the v3 frontend: normalized `[128 long; 64 short]` bands and centered
256/22050-second frames. Arrays must share the same recording and time origin.
Without waveform input, cached `rms_db` is supported and its use is recorded;
longer RMS windows can hide brief interruptions. Deployment and the paired
evaluation both use the waveform.

The higher-level `alignmodel.joint.stack_v8.align_outputs` follows the v7 API
with additional `mel` and optional `audio` keyword arguments. Threshold overrides
use `config['same_pitch']`; the returned audit records the actual values.

## Validation

See `runs/stack-v8/REPORT.md` and the JSON artifacts beside it. Reproduce with:

```powershell
& align-model/.venv-amt-bench/Scripts/python.exe align-model/scripts/evaluate_same_pitch_v1.py
& align-model/.venv-amt-bench/Scripts/python.exe align-model/scripts/check_same_pitch_controls.py
```

The comparison reuses v7 acoustic caches and loads original recordings. Both
variants rerun alignment with identical weights and gates. The 90 synthetic
clips are previously inspected validation data, not an unseen test. DataCreate
001-094 are processed; only the 25 reviewed takes contribute accuracy metrics.
Unlabeled recordings are not assumed clean. Labels 095 onward are excluded.

## Known limits

- Fixed heuristic thresholds, not a learned or calibrated split classifier.
- Conservative 90 ms context on each side plus 25 ms separation from neighboring
  attacks makes fast same-pitch runs frequently unassessable.
- Vibrato, dynamics and timbre changes can block merging even inside one note.
  Model onset false positives can also block a valid repair.
- Very soft repeated attacks can resemble a sustained note; zero false merges
  in this evaluation is not a general guarantee.
- Input adjacency includes optional candidates; equal primary notes separated
  by another pitch candidate are not treated as adjacent.
- No score-derived rule forces repeated notated pitches to remain separate:
  an actual missed rearticulation should be resolved by downstream alignment.
