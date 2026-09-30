# ORN v6 mel lockbox result

## Decision

The frozen mel/identity candidate **failed** the exact 0.80 objective.

- Precision: **0.793074**
- Recall: **0.761706**
- F1: **0.777074**
- Bootstrap 95% interval: **0.751596–0.804634**
- Bootstrap median: 0.777687
- Predictions / gold: 6,338 / 6,599
- Credit: 5,026.5

The observed F1, not its rounded value or bootstrap median, controls the
prespecified decision. This is a failed lockbox and cannot authorize tuning on
v6 or selection of another test.

## Version rationale

The earlier ORN v5 mel candidate used min_confidence=0.72 with softer
boundary/rearticulation thresholds after a recall-oriented grid on opened
v3+v4. It reached development F1 0.807944, then lockbox F1 0.786213 after
over-predicting (6,985 vs 6,622) and collapsing precision to 0.765784.

The later v6 candidate was selected only after sealing an independent 130-row
holdout. On pooled opened v3+v4+v5 (364 rows) a precision-stable grid chose
min_confidence=0.75 with default decode thresholds (no soft-boundary
overrides), reaching development F1 0.802877 with precision 0.819453 and
recall 0.786959. On the sealed v6 lockbox both precision and recall fell
(0.793074 / 0.761706), yielding F1 0.777074 with slight under-prediction
(6,338 vs 6,599). The later version is therefore **rejected**, not promoted.

## Per-type lockbox F1

- Match: 0.919669
- Copy: 0.731513
- Extra: 0.420050
- Substitute: 0.459559
- Missed note: 0.380313

## Isolation

- Candidate SHA-256:
  2bece8c2aba8242ec7b13ac6327521344fcee60d85cbbc4124e5df865086a2dc
- Protocol SHA-256:
  61a832099060f28d5e9643792533fb7757f795fb71e9ef3d7f6fdcb404743399
- Frozen predictions SHA-256:
  de561f84f6ef660d3188ffed4054f52a3aa2598d7f29144372015dd11b04c39d
- Opening record SHA-256:
  8e3313f0763b57aa85bdae359ecc2960c708482f8724b39411c6d7cce4482ad5
- Report SHA-256:
  8b95f9c0a47afc3a54bbe15016de0f8e0643e76a8c6ad77a1036bdae93e2a02d

All 130 predictions were frozen and hashed before the opening sentinels and
before key/target access. Timestamp metrics were not used.
