# Transcriber Track A: postprocess v3

Status: experimental; production is unchanged. The 4,022-row lockbox was not
opened. All train/validation work used DATA_READY pack
`9359b82e...d62a82bd`.

## Predeclared comparison

`PREDECLARED.json` was written before this Track A validation opening. It
freezes baseline, preprocessing-only, postprocessing-only, combined, and
combined+rescorer definitions. The optional preprocessing implementation is
now hash/versioned and non-destructive, but preprocessing-only and multi-view
combined were not advanced to validation because they do not yet have a
train-fold benchmark. They therefore have no validation result and must not be
silently equated with the evaluated packed candidate stream.

## Evaluated results (all 358 validation rows)

The authoritative current full-pipeline baseline is transcription P/R/F1
0.601923/0.690928/0.643362 (95% bootstrap F1
0.626478--0.660959), alignment F1 0.632252, combined F1 0.610432, and
transcription count ratio 1.147867.

The earlier postprocessing-only packed v2 stream used activation-map
continuity merging and short-note rescue but no learned admission model. With
the frozen legacy aligner it produced canonical transcriber-only P/R/F1
0.575193/0.674223/0.620783 and count ratio 1.172169. Its downstream
note-wise F1 including aligner deletions was 0.617774. This is below the
authoritative baseline and is rejected.

The later compact rescorer changes admission rather than candidate generation.
It was selected at threshold 0.62 on a deterministic 4,092/452 split of the
4,544 training rows only. It weights positives below 80/120/180 ms and hard
negatives in the 0.45--0.65 confidence band. On all 358 validation rows:

- Canonical note-wise transcription P/R/F1:
  **0.785997 / 0.737057 / 0.760741**.
- Bootstrap F1 (1,000 row-level replicates):
  **0.747894--0.772813**, median 0.760658.
- Predicted/target count: 37,892/40,408, ratio **0.937735**.
- Downstream fixed-aligner note-wise P/R/F1:
  **0.711297 / 0.678485 / 0.694504**.
- Same-pitch split diagnostic: 518, rate **0.013670** (legacy diagnostic
  baseline approximately 0.166).
- Timestamp-only 50 ms note P/R/F1:
  0.898316/0.842383/0.869451.
- Timestamp-only short recall: <80 ms 30/49 = **0.612245**;
  <120 ms 1,980/2,402 = **0.824313**; <180 ms 20,755/24,141 =
  **0.859741**.
- Per-source equals aggregate because outputRaw development data has one
  source, `MozartClConcertoA`.

The official transcription gain over the authoritative baseline is +0.117379
absolute F1; its lower bootstrap bound is above the baseline upper bound.
Count ratio and split rate improve. However, <80 ms and <120 ms timestamp
diagnostic recall are lower than the stated legacy diagnostics (0.653 and
0.844), and optional preprocessing/multi-view candidates have not passed a
train-fold benchmark. The rescorer is therefore retained as a strong
experimental candidate, not promoted.

## Code and safeguards

- `src/alignmodel/transcription/track_a.py` adds gentle pre-emphasis and RMS
  normalization on a temporary audio copy. Cache identity includes source WAV
  SHA-256, preprocessing config SHA-256, frontend version, Basic Pitch version,
  and pitch policy.
- `scripts/eval_outputraw_note_wise_fallback.py` now separates canonical
  transcriber identity from downstream fixed-aligner scoring and adds a
  1,000-replicate bootstrap.
- Existing activation cleanup preserves strong rearticulation, merges only
  weak/continuous same-pitch boundaries, rescues short events only with
  onset+frame+contour+margin evidence, and rejects onset-only noise.

Verification: 23 distinct focused joint/rescorer/transcription tests passed
(including preprocessing immutability, hash invalidation, and config
roundtrip). IDE lints reported no errors; `py_compile` and `git diff --check`
passed.

## Artifacts

- `PREDECLARED.json`: frozen candidate set and promotion gate.
- `postprocessing-only.json`: full-358 packed v2 fixed-aligner result.
- `combined-rescorer-packed-v2.json`: initial typed downstream result.
- `combined-rescorer-official.json`: canonical transcriber, bootstrap,
  per-source, downstream aligner, and legacy timestamp diagnostics.
- `candidate-rescorer-note-wise-v2/candidate_rescorer.pt`: train-only compact
  rescorer (SHA-256
  `50880dadaf8f2a6166e149b85b11ab10193a04f09afec80dd1633fc4ee790bf7`).

Next promotion prerequisite: benchmark the versioned preprocessing view on
group-disjoint training folds, then open validation once for
preprocessing-only and true multi-view combined candidates. Preserve the
current production default until those candidates maintain the official gain
without the observed short-note recall regression.

## Train-fold follow-up (2026-09-16)

No additional validation rows were opened in this follow-up. Track B held the
GPU, so Track A used bounded CPU/cache work only.

The duration-conditioned admission sweep used 522 rows from 248 held-out
training leakage groups. Its objective was declared in
`train-fold-calibration-v1.json`: maximize mean <80/<120/<180 ms recall while
allowing at most 0.005 candidate-identity F1 loss, 5% false-split inflation,
and no count-ratio increase versus the 0.62 gate. Only the existing global
0.62 schedule passed all constraints:

- Global 0.62: P/R/F1 0.901016/0.960122/0.929631, count ratio 1.065599,
  false-split rate 0.012208, short objective 0.986631.
- Most permissive short schedule: recall rose 0.001406 and the short objective
  rose 0.005610, but F1 fell to 0.928163, count ratio rose to 1.071895, and
  false-split rate rose to 0.013540. It was rejected.

The duration feature is not directly suppressing short positives. Removing
duration features lowers predicted probability for <80 ms positives by only
0.0132 on average, but lowers <80 ms negatives by 0.0520. Thus the feature
helps short candidates indiscriminately and separates negatives poorly. The
larger defect is target mismatch in the first trainer: weights were based on
candidate duration, so artificial short fragments paired to long rendered
events were upweighted as short positives. The trainer now derives positive
duration weights from the exact paired rendered target; this change remains
untrained and unvalidated.

The predeclared preprocessing view was also benchmarked on eight distinct
held-out training leakage groups (861 targets) using hash-validated caches.
Preprocessing-only underperformed original canonical decoding (F1 0.801366
versus 0.831728). Adding the view before the existing rescorer improved the
small-fold postprocess+rescorer control from 0.831528 to 0.863473, raised
<120 ms diagnostic recall from 5/11 to 6/11 and <180 ms from 286/382 to
290/382, and reduced false split rate from 0.004994 to 0.002472. However,
this bounded fold contained no <80 ms targets and retained no measured strong
same-pitch rearticulations after either rescorer path. It is insufficient to
freeze a winner.

`PREDECLARED.json` names the candidate families and broad promotion gates, but
does not authorize winner selection from a fold with no <80 ms support or an
unverified rearticulation guard. Per the requested protocol, Track A therefore
stops before a second full-358 validation opening. The required next comparison
is a larger group-disjoint train benchmark stratified to contain <80 ms events
and strong rearticulations, using the corrected target-duration weighting.
Production and the 0.760741 validation candidate remain unchanged.
