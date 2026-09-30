# ORN v5 mel lockbox result

## Decision

The frozen mel/identity candidate **failed** the exact 0.80 objective.

- Precision: **0.765784**
- Recall: **0.807762**
- F1: **0.786213**
- Bootstrap 95% interval: **0.759873–0.812184**
- Bootstrap median: 0.786199
- Predictions / gold: 6,985 / 6,622
- Credit: 5,349.0

The observed F1, not its rounded value or bootstrap median, controls the
prespecified decision. This is a failed lockbox and cannot authorize tuning on
v5 or selection of another test.

## Version rationale

The earlier ORN v4 mel candidate used `min_confidence=0.75` with default decode
thresholds. It reached development F1 0.814699 on opened v3, then lockbox F1
0.798517 (recall-limited shortfall of 0.001483).

The later v5 candidate was selected only after sealing an independent 140-row
holdout. On pooled opened v3+v4 (224 rows) a decode grid chose
`min_confidence=0.72` with softer boundary/rearticulation thresholds, reaching
development F1 0.807944 with higher recall (0.827528) than precision
(0.789265). On the sealed v5 lockbox that recall-oriented choice over-predicted
(6,985 vs 6,622) and precision fell to 0.765784, so overall F1 dropped to
0.786213. The later version is therefore **rejected**, not promoted.

## Per-type lockbox F1

- Match: 0.921532
- Copy: 0.725176
- Extra: 0.443259
- Substitute: 0.380789
- Missed note: 0.346535

## Isolation

- Candidate SHA-256:
  `d9b9221e654e72f7429a146cf7aca8e4f5e953a7d79a42fea276ad1459731a00`
- Protocol SHA-256:
  `e91c09185e174481e9ec6b0c2f8c0e0a28f96f93fb6ef05be7f559a4eaabcb09`
- Frozen predictions SHA-256:
  `0d52b6ea1efe8abe072a2a4b47c37e4e290f9e184d2df9329f4a657023970f8a`
- Opening record SHA-256:
  `782641fd944e9e269f581f2255770beca95f3ff175d9571eff8a1483632f7c1d`
- Report SHA-256:
  `780a3a934e4db37ffa8f516dfbe3b38ab8d98bcbb47c70b7ecd70d1ab38581e8`

All 140 predictions were frozen and hashed before the opening sentinels and
before key/target access. Timestamp metrics were not used.
