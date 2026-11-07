# Alignment-based rhythm classifier pilot

Implemented 2026-11-06. This is a controlled oracle-timing experiment, not an
acoustic or real-performance accuracy claim.

## Experiment

The registered E: synthetic corpora were unavailable on this host. The runner
therefore generated 80 new symbolic synthetic bundles using the repository's
existing six rhythm injectors: late start, early start, early end, late end,
temporary tempo change, and uneven rhythm. Each split also includes clean and
smooth-tempo controls. There are 40 selection clips and 40 separately seeded
check clips, 48 notes per clip. Each bundle stores both MusicXML scores and
oracle_alignment.json with exact canonical identity and tempo-integrated timing.
No waveform synthesis or acoustic inference was performed: this isolates the
requested correct-alignment experiment. Existing sealed tests were not opened.

Global tempo varies by 0.8–1.25, articulation by 0.9–1.0 and onset jitter by
±4 ms. Smooth-tempo negatives gradually change by 15%. Gold identifies notes
whose duration was changed by the injector before those nuisance variations.
This is per-note binary rhythm detection, without padded label ranges or
half-credit matching. It does not measure exact phrase-level label grouping.

Baseline: EWMA alpha 0.3, log-ratio threshold 0.25, minimum absolute deviation
60 ms. Expected duration is normalized by estimated tempo, not compared directly
with a reference recording. The EWMA initialization uses the first six eligible
ratios' median.

The selection-set adequacy target was precision >=85% and recall >=80%. EWMA
failed, so the runner compared EWMA thresholds and leave-one-out neighboring-note
median baselines. Only selection scores chose the policy. The selected policy
uses a 65-note window, log-ratio threshold 0.20 and minimum deviation 60 ms.
The current note is excluded from its median. Both thresholds must pass.

## Independent check results

| Method / timing evidence | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Initial EWMA, exact offsets | 38 | 9 | 47 | 80.85% | 44.71% | 57.58% |
| Selected median, exact offsets | 81 | 0 | 4 | 100.00% | 95.29% | 97.59% |
| Selected median, onsets only | 75 | 10 | 10 | 88.24% | 88.24% | 88.24% |

The check population has 85 affected notes. The exact-offset median missed four
tempo-change notes and detected all injected early/late starts, early/late ends
and unequal-duration notes. Neither method emitted errors on the check split's
ten clean/smooth-tempo clips.

EWMA adapts to a short tempo-error region and can flag the return to normal.
A wider median is more resistant to sparse/short corruption. This advantage
depends on the error affecting less than half of its context.

Onset-only evaluation missed every release-only error (ten notes) and added ten
neighboring false positives around early/late attacks. Its timing evidence
describes the preceding inter-onset interval, not a precisely localized attack.
Do not present its 88.24% aggregate score as reliable release detection.

## Runtime integration

- joint/rhythm_v1.py: score/event-only API; supports ewma and median policies.
- joint/stack_v9_passage.py: appends rhythm labels after accepted content alignment.
- scripts/publish_datacreate_v9.py: explicitly chooses predicted-offset mode for
  MuScriptor; CTC uses IOIs because its ends are next-onset estimates.
- DataCreate freshness checks require rhythm_revision, preventing old cached
  feedback from silently omitting the new stage.
- The GUI adapter retains rhythm explanations and validates canonical identities.
- The standalone v9 runner records the rhythm revision too.

Input events come from the raw acoustic alignment, never matches inferred by the
missed-note gate. Extras, replays, duplicate mappings and merged score spans are
excluded. IOIs require consecutive score identities and adjacent performance
events. Uncertain clip alignment withholds all rhythm feedback.

Output adds rhythm_error, signed deviation_ms (positive = longer), short/long
subtype, timing source, canonical affected note identities and separate context.
diagnostics.rhythm reports assessed indices and timing limitations. Content
alignment coverage is independent of rhythm coverage.

MuScriptor offsets are predictions, not ground truth. Both recognition and
mapping errors can produce false rhythm labels. The default policy is a pilot;
no real-audio improvement has been established. Written tempo instructions are
not explicitly modeled by this quarter-length-ratio implementation. Long tempo
departures, majority-corrupted windows, short excerpts, expressive articulation,
ornaments and arbitrary practice restarts require further validation. The
classifier does not distinguish early attack from early release by subtype.

## Reproduction

From the repository root, use the existing AMT/MusicEval Python:

    python align-model/scripts/evaluate_rhythm_oracle.py --output reports/rhythm-oracle-new

The output directory must be new. report.json contains check predictions and
per-kind metrics; selection.json contains all compared configurations.
The bundles are the complete generated inputs. No human labels or model weights
are changed. Generation/calibration use the same injector family, so the check
split tests new seeds rather than unseen repertoire or realistic audio.

Programmatic use:

    classify_rhythm(index.events, alignment.events,
        measured_offsets=True,
        config=RhythmConfig(method='median', window=65, log_threshold=.20))

Use measured_offsets=False for next-onset-derived ends. EWMA remains available
with RhythmConfig(method='ewma'). Insufficient context returns no labels and an
explicit status, not a claim of clean playing.


Verification: 48 focused classifier, passage, GUI adapter, bridge and transcriber-selection tests passed. Publisher and runner compile checks passed. The bridge regression also verifies that feedback without the rhythm revision is rejected as stale.
