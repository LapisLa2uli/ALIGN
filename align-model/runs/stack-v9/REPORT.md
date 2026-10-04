# v9: 10 ms gaps and relaxed split repair

**Outcome: all injected split controls now pass, but the method does not solve the error-labeling problem and hurts synthetic accuracy.** The version is available through the v9 runner; the GUI default and previous candidates are preserved.

## What changed

- Waveform envelope uses a centered 4 ms RMS window at 1 ms steps, independent of the 11.61 ms mel hop.
- The gap veto requires at least an 18 dB dip with energy recovery on both sides; nominal gap scale is 10 ms.
- Energy variation cutoff: 2.5 -> 54 dB. Normalized mel step: 0.06 -> 0.35. Mel drift: 0.06 -> 0.65.
- Model attack/voicing vetoes are disabled for this permissive experiment; those signals remain in the audit. Relaxing variation alone cannot make every old positive pass.
- Context adapts to nearby notes, with a 5 ms margin instead of requiring the entire old fixed window.
- Repaired transcription is realigned before reference-score error labels are generated.

## Controlled split and gap checks

| Check | Result |
|---|---:|
| Existing injected false splits repaired | 316 / 316 (100%) |
| Genuine repeated-note controls incorrectly merged | 457 / 543 (84.16%) |
| Inserted 10 ms complete silences detected | 316 / 316 (100%) |
| Inserted 10 ms 26 dB attenuations detected | 311 / 316 (98.42%) |

The 316 positive controls set the variation bounds; their 100% rate is a **calibration result**, not independent test accuracy. Genuine repeats were not used to tune the bounds. Of 543 genuine repeats, 84 lacked sufficient context and 2 exceeded another bound; the other 457 were merged incorrectly. The original v8 repaired 5/316 false splits and incorrectly merged 0/543 genuine repeats.

The gap checks alter copies of the original waveform at the known control midpoint; cached mel/model evidence is held fixed to isolate the gap detector. Source recordings are untouched. Controlled-tone tests also cover different pitches, sub-millisecond offsets, 10/20/50 ms silences, attenuation and short context. These are not guarantees for arbitrary recording noise or reverberation.

## End-to-end reference-note evaluation

Paired evaluation uses the same checkpoint, gates and acoustic caches. Accuracy uses canonical reference-score note identity and type, not timestamp overlap. The 90 synthetic validation clips have been inspected before and also supply calibration controls; this is a development comparison, not a sealed test. DataCreate did not select parameters.

| Dataset | Pairs checked | Merged | Clips changed | Content-error F1 before | After |
|---|---:|---:|---:|---:|---:|
| realistic92 (30 clips) | 276 | 270 | 26 | 0.878238 | 0.800000 |
| fast102 (30 clips) | 76 | 72 | 23 | 0.954468 | 0.937087 |
| dclike11 (30 clips) | 209 | 203 | 23 | 0.864198 | 0.780488 |
| DataCreate 001-094 | 281 | 272 | 75 | See below | See below |

Macro synthetic content-error F1: **0.898968 -> 0.839192**, a decline of 5.98 percentage points.

All three synthetic families regress. Correct extra-note identities fall 285 -> 247 in realistic92 and 1294 -> 1255 in fast102. Correct wrong-note identities fall 31 -> 26, 84 -> 77, and 22 -> 20 in the three families. Merging genuine repeats changes alignment and removes useful performance evidence.

## DataCreate

All 001-094 recordings are processed. Only 25 reviewed takes contribute accuracy; unlabeled takes are not treated as clean. Labels 095 onward are excluded.

| Type | Correct before -> after | Predictions before -> after | Gold | F1 before -> after |
|---|---:|---:|---:|---:|
| extra_note | 2 -> 2 | 32 -> 31 | 12 | 0.090909 -> 0.093023 |
| missed_note | 1 -> 1 | 8 -> 8 | 8 | 0.125000 -> 0.125000 |
| wrong_note | 1 -> 1 | 22 -> 21 | 22 | 0.045455 -> 0.046512 |
| repetition | 0 -> 0 | 9 -> 8 | 4 | 0.000000 -> 0.000000 |

There are three fewer false-positive label predictions overall, with no additional correctly identified errors. This is a small precision change, not a solution or recall improvement. The same 12 takes remain alignment_uncertain: 050, 053, 058-067.

Changed reviewed label identities:

- 008: removes extra-note label on score events 24-25.
- 012: replaces an incorrect repetition on events 25-30 with an incorrect extra label on 28-29.
- 013: removes wrong-note label on event 130.
- 030: removes extra-note label on events 36-37.
- 039: changes repetition span 79-97 to 86-97; it remains incorrect.

Indices above are zero-based canonical score events.

## Why this still fails

No naturally decoded pair crossed the 18 dB gap threshold. The processor merged 817/842 pairs (97.03%). Among the 25 retained boundaries, 12 lacked context, 11 exceeded a spectral bound and 2 exceeded the energy bound. A clear 10 ms silent gap is detectable, but genuine repeated notes can have connected sound, a shallow tongue interruption, or a reverberant tail. Absence of a deep gap therefore does not establish that two notes are one.

The real-repeat controls had a maximum measured bounded dip of about 6.16 dB, while positive sustained-note controls reached 6.97 dB. Even lowering this single gap threshold cannot perfectly separate these populations. Requiring every sustained-note control to pass leaves substantial overlap with genuine repetitions. A useful next classifier needs localized attack evidence and harmonic continuity jointly, trained on both kinds of boundary and selected by downstream reference-note accuracy.

## Artifacts and verification

- `CANDIDATE_STACK_V9.json`: evaluated permissive configuration, checkpoint and source hashes.
- `calibration.json`: selected bounds and positive/negative control summary. GUARDED and ALL_PASS are identical in this run.
- `control_features.json`: measured control features before calibration.
- `control_decisions.json`: decisions recomputed using each final configuration.
- `gap_10ms_validation.json`: every waveform-interruption check.
- `evaluation_protocol.json`: paired-evaluation manifest and limitations.
- `synthetic.json`, `datacreate-before.json`, `datacreate-after.json`: detailed reference-note metrics.
- `same_pitch_audit.json`, `predictions/before/`, `predictions/after/`: all boundary decisions and feedback.
- `example-020.json`: successful fresh waveform-to-feedback CLI run.
- **44 tests passed**, covering v7/v8 regression and the new processor.

Recommendation: keep this permissive version experimental; do not promote it as an accuracy improvement.
