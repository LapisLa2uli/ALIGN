# v9: relaxed same-pitch repair and 10 ms gap detection

The requested experiment is implemented as v9, preserving the v7/v8 candidates.
It passes all 316 previous injected-split controls and detects all 316 inserted
10 ms silences on the same recordings. It **does not solve the overall problem**:
457/543 genuine repeat controls are incorrectly merged and synthetic reference
note error F1 falls from 0.898968 to 0.839192. It remains experimental. At the
user's request, DataCreate's GUI now runs this candidate for Re-align and
Re-label so review and regeneration use the same version. This does not promote
its evaluation status or change the frozen candidate.

## Changes

- Separate waveform envelope: centered 4 ms RMS at 1 ms intervals. The existing
  mel features retain their 11.61 ms hop; gap resolution does not depend on them.
- A gap requires a low-energy core with higher-energy flanks on both sides.
  Default depth is 18 dB. This detects short interruptions, including boundaries
  between coarse mel frames, without interpreting a one-way fade as a gap.
- Adaptive context: up to 90 ms on each side, clipped to neighboring starts with
  a 5 ms margin. Insufficient or missing data still retains the notes.
- Candidate energy variation limit increases from 2.5 to **54 dB**. Spectral step
  and drift limits increase from 0.06/0.06 to **0.35/0.65** in normalized mel units.
- The model's onset/rearticulation and voiced outputs remain in diagnostics;
  their vetoes are disabled in this permissive candidate. Increasing energy and
  spectral bounds alone would not make all prior split controls pass.
- Every original adjacent equal-pitch pair has a decision record, source indices,
  and available acoustic features. Merged rows are realigned to the score before
  error labels are generated.

The 316 injected controls were used to select upper variation bounds, so their
100% success is a calibration result, not held-out accuracy. Both the `GUARDED`
and `ALL_PASS` candidate files end up identical: none of these positives crossed
the fixed 18 dB gap threshold. No DataCreate labels selected these parameters.

## Run

From the repository root in PowerShell:

```powershell
& align-model/.venv-amt-bench/Scripts/python.exe align-model/scripts/run_stack_v9.py `
  --score DataCreate/samples/020/verified_score.musicxml `
  --audio DataCreate/samples/020/performance_audio.wav `
  --output align-model/runs/stack-v9/example-020.json
```

The runner defaults to `runs/stack-v9/CANDIDATE_STACK_V9.json`, accepts WAV/MP3
through the existing audio loader and retains the written B-flat-clarinet pitch
convention. It checks candidate source, checkpoint and verifier hashes. Feedback
uses the `align-score-feedback-v9` schema with canonical reference-note indices.

API: `alignmodel.transcription.same_pitch_v2.repair_same_pitch(rows, outputs, mel,
audio=waveform, config=SamePitchConfig(...))`. Use the candidate's `same_pitch`
configuration for the evaluated operating point. Missing waveform abstains;
cached coarse RMS alone is not sufficient to claim 10 ms gap support.
`acoustic_evidence` can precompute an envelope for repeated calls on one recording.
The wrapper `alignmodel.joint.stack_v9.align_outputs` performs repair and alignment.

## Reproduce

```powershell
& align-model/.venv-amt-bench/Scripts/python.exe align-model/scripts/calibrate_same_pitch_v2.py
& align-model/.venv-amt-bench/Scripts/python.exe align-model/scripts/check_10ms_gaps_v2.py
& align-model/.venv-amt-bench/Scripts/python.exe align-model/scripts/evaluate_same_pitch_v2.py
```

Read `runs/stack-v9/REPORT.md` for results and limitations. The 90 synthetic clips
are development data used previously; DataCreate 001-094 are processed but only
25 reviewed takes are scored. Unlabeled takes are not assumed clean. DataCreate
095 onward is excluded.

## Limits of the gap claim

The method detects 316/316 inserted 10 ms silences and 311/316 inserted 10 ms
26 dB attenuations. Real noise/reverberation can fill a gap, and genuine tongued
or legato repetitions need not contain a deep silent interval. None of the 842
naturally decoded pairs crossed the configured gap threshold. The method must
not interpret absence of a detected gap as proof of one note: this permissive
experiment demonstrates the resulting false merges. Very short or marginal
contexts remain unassessed; the low-core timing is not an exact gap-duration
estimator.
