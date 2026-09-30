# ORN v7 hybrid lockbox result

## Decision

The frozen mel-plus-Basic-Pitch candidate **passed** the exact 0.80 objective.

- Precision: **0.808184**
- Recall: **0.825341**
- F1: **0.816673**
- Bootstrap 95% interval: **0.798071–0.834349**
- Bootstrap median: 0.816263
- Predictions / gold: 6,879 / 6,736
- Credit: 5,559.5

The observed F1, not its rounded value or bootstrap median, controls the
prespecified decision. The bootstrap interval extends slightly below 0.80.
This is a passed lockbox and does not authorize further tuning on v7.

## Version rationale

The earlier ORN v6 candidate used mel transcription at min_confidence 0.75
with default decode thresholds and no Basic Pitch gap fill. On its sealed
130-row lockbox it under-predicted (6,338 vs 6,599) and scored F1 0.777074.
On the same opened v3–v6 pool later used for v7 development, that mel-only
setting scored F1 0.795928 (precision 0.812349, recall 0.780158). A separate
Basic Pitch plus identity grid on that pool stayed near F1 0.751.

The later candidate was selected only after sealing an independent 137-row
holdout. It keeps the mel notes and adds sanitized Basic Pitch 0.4.0 notes
whose overlap with any mel note is at most 0.02 seconds. On the 494 opened
rows that change reached F1 0.818999 (precision 0.815930, recall 0.822090),
compared with mel-only F1 0.795928 on those same rows. The sealed v7 point
estimate is F1 0.816673. The later version therefore **passed** its one-shot
lockbox.

## Per-type lockbox F1

- Match: 0.939285
- Copy: 0.779970
- Extra: 0.434827
- Substitute: 0.461812
- Missed note: 0.450495

## Isolation

- Candidate SHA-256:
  30c5b9b60fbf55ecc9fd52a1efe553dd0a5c20fd61ff47546fb7147d85386c8c
- Protocol SHA-256:
  30087e6fdb0cef3536a242648fcd396e02e93c23f124cc2f8a701d6058cc8e52
- Frozen predictions SHA-256:
  85753053d0635382cd142f98fcaeed92126955ba019666c3ab93495d82d187f0
- Opening record SHA-256:
  1c8ce377a8da5615e36cc32097567aa456d4901487e68f8ac36546d590968c8b
- Report SHA-256:
  e16c88818aceb740c9fdc504cbd40d8b1d98ce7091e73c4ab2ffcd5feda4219d

All 137 predictions were frozen and hashed before the opening sentinels and
before key/target access. Timestamp metrics were not used.
