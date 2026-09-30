# ORN v4 mel lockbox result

## Decision

The frozen mel/identity candidate **failed** the exact 0.80 objective.

- Precision: **0.814335**
- Recall: **0.783302**
- F1: **0.798517**
- Bootstrap 95% interval: **0.767821–0.828288**
- Bootstrap median: 0.800088
- Predictions / gold: 5,553 / 5,773
- Credit: 4,522.0

The observed F1, not its rounded value or bootstrap median, controls the
prespecified decision. The shortfall is 0.001483 F1 (8.5 credits at the
observed supports). This is a failed lockbox and cannot authorize tuning on
v4 or selection of another test.

## Version rationale

The earlier Basic Pitch local-rescue model reached v3 lockbox F1 0.732413
because copy/EXTRA precision collapsed. A stricter Basic Pitch decode improved
opened-v3 development to only 0.778431.

The later model replaced Basic Pitch with the repository-trained,
boundary-aware mel transcriber while retaining the frozen identity CRF.
Before v4 opening, confidence 0.75 reached development F1 0.814699
(P/R 0.829167/0.800726). On the independently generated and encrypted
110-row v4 lockbox it reached F1 0.798517. The later version materially
improves the prior distribution failure but still misses the declared
threshold, so it remains rejected rather than promoted.

## Isolation

- Candidate SHA-256:
  `ea149d8d8011abdb3e35ac5627384da99e66a4c2a9032f2ce0f59c6fffaa3eaf`
- Protocol SHA-256:
  `398424f9b3ca432fcff15d3ed5afd741494600e9122af3530a14e1bcb0a07bd7`
- Frozen predictions SHA-256:
  `4be6ff800e035354cea62fccedf54cbc955a0059303980b9dd40c827b9cb89ac`
- Opening record SHA-256:
  `18d4e2563ae2e69e4590abfe3f4c4f8230fe292c195fc4641783773c2b4e6361`
- Report SHA-256:
  `e11aed6ce00e84cba8d717b75e4f1f10dbb31cd2a1e46de07a4c84f8b3eb601c`

All 110 predictions were frozen and hashed before the opening sentinels and
before key/target access. Timestamp metrics were not used.
