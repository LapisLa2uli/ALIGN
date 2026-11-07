# DataCreate 001–050: MuScriptor vs. CTC

Published MuScriptor medium + ALIGN v9 agent labels for all 50 takes. Scored 25 against the pre-seed user archive (66 human labels). 004 was explicitly reviewed clean. The remaining takes lack verified manual review and were not scored.

The same ALIGN v9 candidate and feedback thresholds were used for both. The CTC baseline uses the original, unedited publisher outputs. Only the labels published to the GUI are scored; an abstention contributes no predictions.

| Label metric | Pipeline | Precision | Recall | F1 | Match credit | Predictions | Human labels |
|---|---|---:|---:|---:|---:|---:|---:|
| All error types, official | CTC + ALIGN | 5.88% | 6.06% | 5.97% | 4 | 68 | 66 |
| All error types, official | MuScriptor + ALIGN | 3.64% | 3.03% | 3.31% | 2 | 55 | 66 |
| All error types, strict | CTC + ALIGN | 5.88% | 6.06% | 5.97% | 4 | 68 | 66 |
| All error types, strict | MuScriptor + ALIGN | 1.82% | 1.52% | 1.65% | 1 | 55 | 66 |
| Four content types, strict | CTC + ALIGN | 5.88% | 8.70% | 7.02% | 4 | 68 | 46 |
| Four content types, strict | MuScriptor + ALIGN | 1.82% | 2.17% | 1.98% | 1 | 55 | 46 |

Official micro-F1 matches errors one-to-one on the exact affected score-note location, excluding display padding. Matching the type earns one point; the wrong type at that location earns half. Strict F1 requires the same type. Content types: wrong note, missed note, extra note, repetition. This is not a metric of transcription or timestamp accuracy.

## Strict results by type

| Error type | Human labels | CTC calls | CTC matches | MuScriptor calls | MuScriptor matches |
|---|---:|---:|---:|---:|---:|
| bad_start | 5 | 0 | 0 | 0 | 0 |
| bad_timbre | 4 | 0 | 0 | 0 | 0 |
| extra_note | 12 | 31 | 2 | 20 | 1 |
| missed_note | 8 | 8 | 1 | 2 | 0 |
| repetition | 4 | 8 | 0 | 13 | 0 |
| rhythm_error | 5 | 0 | 0 | 0 | 0 |
| sliding | 4 | 0 | 0 | 0 | 0 |
| squeak | 2 | 0 | 0 | 0 | 0 |
| wrong_note | 22 | 21 | 1 | 20 | 0 |

Official micro-F1 difference (MuScriptor minus CTC): -2.66 percentage points; paired 95% bootstrap interval -10.32 to +5.36 points (10,000 resamples of reviewed takes, seed 20261105). The interval describes this small reviewed sample.

Agent labels installed: 148 across 50 takes. Alignment status other than ok: 047, 050. All before/after user-file hashes matched. The live user-label files were agent-seeded and were not used as ground truth.

## Per-take results

A dash means there was no verified human reference. A reviewed take with zero labels still counts against precision if it receives a prediction.

| Take | Status | Installed labels | Human labels | CTC match credit | MuScriptor match credit |
|---|---|---:|---:|---:|---:|
| 001 | ok | 1 | 1 | 0.0 | 1.0 |
| 002 | ok | 5 | 4 | 0.0 | 0.0 |
| 003 | ok | 2 | 1 | 0.0 | 0.0 |
| 004 | ok | 1 | 0 | 0.0 | 0.0 |
| 005 | ok | 5 | 6 | 0.0 | 0.0 |
| 006 | ok | 1 | 4 | 0.0 | 0.5 |
| 007 | ok | 8 | — | — | — |
| 008 | ok | 1 | 2 | 0.0 | 0.0 |
| 009 | ok | 1 | — | — | — |
| 010 | ok | 0 | 2 | 0.0 | 0.0 |
| 011 | ok | 1 | 1 | 0.0 | 0.0 |
| 012 | ok | 3 | 3 | 0.0 | 0.0 |
| 013 | ok | 6 | 6 | 0.0 | 0.0 |
| 014 | ok | 0 | 1 | 0.0 | 0.0 |
| 015 | ok | 8 | — | — | — |
| 016 | ok | 0 | 1 | 0.0 | 0.0 |
| 017 | ok | 0 | 1 | 0.0 | 0.0 |
| 018 | ok | 2 | — | — | — |
| 019 | ok | 1 | 2 | 0.0 | 0.0 |
| 020 | ok | 10 | 3 | 2.0 | 0.0 |
| 021 | ok | 2 | — | — | — |
| 022 | ok | 1 | — | — | — |
| 023 | ok | 2 | — | — | — |
| 024 | ok | 2 | — | — | — |
| 025 | ok | 1 | — | — | — |
| 026 | ok | 5 | — | — | — |
| 027 | ok | 0 | — | — | — |
| 028 | ok | 0 | — | — | — |
| 029 | ok | 0 | 1 | 0.0 | 0.0 |
| 030 | ok | 0 | 4 | 0.0 | 0.0 |
| 031 | ok | 3 | — | — | — |
| 032 | ok | 2 | — | — | — |
| 033 | ok | 1 | 3 | 0.0 | 0.5 |
| 034 | ok | 1 | 2 | 0.0 | 0.0 |
| 035 | ok | 3 | 5 | 1.0 | 0.0 |
| 036 | ok | 4 | 5 | 1.0 | 0.0 |
| 037 | ok | 2 | — | — | — |
| 038 | ok | 7 | — | — | — |
| 039 | ok | 4 | 3 | 0.0 | 0.0 |
| 040 | ok | 1 | — | — | — |
| 041 | ok | 2 | 4 | 0.0 | 0.0 |
| 042 | ok | 40 | — | — | — |
| 043 | ok | 3 | 1 | 0.0 | 0.0 |
| 044 | ok | 0 | — | — | — |
| 045 | ok | 1 | — | — | — |
| 046 | ok | 2 | — | — | — |
| 047 | alignment_uncertain | 0 | — | — | — |
| 048 | ok | 2 | — | — | — |
| 049 | ok | 1 | — | — | — |
| 050 | alignment_uncertain | 0 | — | — | — |

[Full per-take metrics and provenance](evaluation.json) · [Installation manifest](publish/manifest.json) · [Previous GUI artifacts](publish/backup/). The pre-seed user archive was read only.
