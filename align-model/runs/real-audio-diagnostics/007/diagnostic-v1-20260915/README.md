# Sample 007 real-audio transcription diagnostic

Status: diagnostic only; no production parameter is promoted. The sealed
synthetic test set was not read. Shared transcriber and joint-training modules
were not modified.

## Finding

The report that roughly half the fast notes are absent is real in the current
joint/UI artifact, but it is primarily a pre-lattice confidence-gating failure,
not a partial take, score-excerpt error, stale cache, transposition error, or a
raw canonical-transcriber count failure.

- Audio: 34.911973 s, mono 22,050 Hz; trim metadata is internally consistent
  with 5.2763--40.1883 s of the 43.54 s source.
- Score: measures 114 beat 2 through 130, 243 events (238 notes and 5 rests).
  An independently recomputed transcription-blind DTW spans 0.2438--34.9073 s,
  covering the complete recording and all requested measures.
- Cache: the Basic Pitch cache WAV SHA-256 exactly matches the current audio
  (`606335f4...d4c73c`), uses Basic Pitch 0.4.0 frontend v2, written pitch, and
  the configured +2 semitone sounding-to-written shift.
- Pitch convention: the timing-paired written-minus-score pitch mode is 0
  semitones; the implied score-minus-sounding interval is +2.
- The canonical frozen transcriber emits 244 notes for 238 score notes.
- The audio-only high-recall union emits 284 candidates, but the production
  minimum confidence of 0.65 retains only 139 (58.4% of the score count).
  The existing `note_alignment_v2.json` reports exactly those 139 candidates.
- Timing-free pitch-sequence coverage is an optimistic upper bound, but it
  isolates the gate clearly: 224/238 (94.12%) before the gate versus 134/238
  (56.30%) after the 0.65 gate.

The current sample `alignment.npz` is not an independent timing reference: its
engine is `align-joint`, and it was regenerated as a compatibility path from
the 139-event joint output. This diagnostic therefore recomputed DTW from the
performance and rendered reference audio in a temporary directory and saved
the resulting path only under this run directory.

## Exact proxy metrics

Independent DTW is adequate for coverage/segment diagnosis but not accurate
enough for strict fast-note timing: broad exact-pitch pairs have 108.99 ms
median absolute onset error (75th percentile 158.01 ms; 90th 231.14 ms).
Consequently, the following 20/50/100 ms values are reproducible strict proxies,
not ground-truth event accuracy:

- Canonical frozen, 244 predictions: 18/32/61 matches; F1
  0.074689/0.132780/0.253112.
- Joint v1 all, 284 predictions: 20/38/70 matches; F1
  0.076628/0.145594/0.268199.
- Joint v2 all, 284 predictions: exactly identical to v1 on this sample.
- Joint v2 at confidence >=0.65, 139 predictions: 7/15/29 matches; F1
  0.037135/0.079576/0.153846.

At 50 ms, cumulative duration-bucket recall is:

- Canonical: <80 ms 1/3; <120 ms 20/159; <180 ms 30/232;
  <250 ms 30/233.
- V2 all: <80 ms 1/3; <120 ms 26/159; <180 ms 36/232;
  <250 ms 36/233.
- V2 confidence >=0.65: <80 ms 1/3; <120 ms 9/159; <180 ms
  13/232; <250 ms 13/233.

Same-pitch fragmentation is not the dominant failure here. The canonical stream
has 5 adjacent same-pitch boundaries and none has a weak second onset. The
high-recall union has 33 boundaries, only one weak (0.35% per prediction).

For the strict 100 ms proxy, matched versus missed median activation evidence
is sharply separated: onset 0.586 vs 0.139, frame peak 0.762 vs 0.109, and
contour peak 0.547 vs 0.120. In contrast, RMS (-21.10 vs -20.33 dBFS), SNR proxy
(6.16 vs 6.94 dB), centroid (1,141 vs 1,134 Hz), and flatness have near-zero
point-biserial correlation. This supports activation/confidence calibration
over a global energy or timbre threshold, although note-level conclusions are
limited by DTW timing error.

## Single-sample sensitivity (not generalizable)

V1 and v2 candidate sets are byte-for-byte identical by rounded
pitch/start/end identity on this sample. The 72-setting rescue/merge sweep did
not produce a defensible gain: its best strict 100 ms proxy had 256 candidates,
63 matches, and F1 0.255061, essentially the canonical 0.253112 with 12 more
predictions. No rescue/merge setting should be promoted from this result.

The confidence-gate sequence-coverage sweep is more decisive:

- 0.65: 139 candidates, 134 sequence matches, 56.30% upper-bound recall.
- 0.60: 185 candidates, 170 matches, 71.43% recall.
- 0.55: 216 candidates, 192 matches, 80.67% recall.
- 0.50: 240 candidates, 208 matches, 87.39% recall and 32 unmatched candidates.
- No gate: 284 candidates, 224 matches, 94.12% recall and 60 unmatched candidates.

These are score-conditioned, timing-free sensitivity figures, not validation
metrics. They identify 0.50 and 0.55 as ablations to run, not new defaults.

## Recommended retraining/validation follow-up

1. Keep candidate generation audio-only, but validate lattice admission at
   minimum confidence 0.50 and 0.55. Do not hard-drop candidates at 0.65 before
   the learned path can use score and structural evidence.
2. Calibrate candidate confidence and use duration-weighted positive loss for
   <120 ms and <180 ms notes, with hard negatives in the 0.45--0.65 confidence
   band. The current median canonical confidence is 0.635, below production's
   hard gate.
3. Retain onset/contour/timbre augmentation in v2 training, but do not further
   loosen rescue thresholds based on sample 007: rescue did not add coverage
   here. Higher temporal resolution is also not the smallest next step; the
   frontend grid is about 11.6 ms for mostly 80--180 ms events.
4. Validate on the unsealed validation split using note F1, count ratio,
   duration-bucket recall, and extra-note rate before changing production.
   Score-conditioned transcription candidate generation is not indicated.

## Artifacts and reproducibility

- `report.json`: complete machine-readable metrics and provenance.
- `audio_only_candidates.json`: canonical, v1, v2, and gated streams.
- `basic_pitch_cache.hash-validated.npz`: diagnostic copy tied to the current
  WAV hash.
- `independent_dtw.npz` and `independent_score_events.json`: transcription-blind
  timing proxy.
- `missed_short_events.json`: strict 100 ms missed-event list with acoustic
  evidence; intentionally includes the timing-reference caveat.
- `sequence_unmatched_short_events.upper_bound.json`: timing-free optimistic
  unmatched list.
- `rescue_merge_sweep.non_generalizable.json` and
  `confidence_gate_sweep.non_generalizable.json`: single-sample sensitivity.

Re-run from `align-model`:

```powershell
$env:PYTHONPATH = "src;..\DataCreate\src;..\synth-pipeline\src"
$env:CUDA_VISIBLE_DEVICES = "-1"
.\.venv-amt-bench\Scripts\python.exe scripts\diagnose_real_audio_transcription.py `
  --sample ..\DataCreate\samples\007 `
  --out runs\real-audio-diagnostics\007\diagnostic-v1-20260915
```
