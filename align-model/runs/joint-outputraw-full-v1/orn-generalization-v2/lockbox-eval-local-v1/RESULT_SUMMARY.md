# ORN v2 local-rescue lockbox result

## Decision

The frozen 0.80 objective **did not pass**.

- Official exclusive note-wise precision: **0.799843**
- Official exclusive note-wise recall: **0.796846**
- Official exclusive note-wise F1: **0.798342**
- Bootstrap 95% interval: **0.775540–0.821041** (1,000 replicates)
- Support: 3,190 predictions, 3,202 gold events, 2,551.5 credit
- Prespecified threshold: **0.800000**
- Shortfall: **0.001658 F1**, equivalent to 5.5 additional credits at the observed supports

The result is a failed lockbox, not an achieved or rounded-up 0.80 result.
No post-lockbox tuning or alternate held-out selection is permitted.

## Version rationale

The earlier `template-rescue-v3` candidate used a single global linear
score-to-audio timing map. It reached calibration F1 0.806803, but only
0.792178 on the 111-row split after that split was reclassified as development.
This exposed sensitivity to local tempo variation.

The later local-rescue candidate interpolated ornament times from the frozen
identity CRF's own target-free canonical score/copy anchors. On comparable
development populations it reached F1 0.803941 (64 rows) and 0.801359
(111 rows), with combined F1 0.802271. It was frozen before lockbox inference.
The later candidate remains **rejected for the 0.80 objective** because its
one-shot lockbox F1 was 0.798342. It also does not satisfy or alter the
original v2 0.85 promotion gate.

## Isolation and audit

- All 57 predictions were frozen and hashed before the key or targets were read.
- Frozen predictions SHA-256:
  `16e81d5fa08c80532401d061d04ddeb7e614fef8e2fd7839f12fc9419a826c5e`
- Opening record SHA-256:
  `a943193e739c3b695e2f0439f6bfdc2a1a7b03ea224711cd82e5ad52c82f935c`
- Report SHA-256:
  `6a79d8ae62f6e90fa2e7443f5dda85b90620a69093b6c41c3afecff71ce9abcd`
- Protocol SHA-256:
  `0b72096989a76a3ac0b83f939ce0c3e51d95fb58f05aad1909486af22cb189da`
- Candidate SHA-256:
  `f44b7f44e5d2f4b7ba99c97de5a478f94f16c8842d87da1c5235c53138605f9f`

An earlier diagnostic loader parsed the full development-target archive,
invalidating `open_validation` as held-out before any metric was computed from
that split. This was recorded in `OPEN_VALIDATION_CONTAMINATION.json`; the
split was subsequently used only as development. The encrypted lockbox was
the only held-out result used for the decision above.

## Per-type lockbox F1

- Match: 0.950181 (support 1,427)
- Copy: 0.777972 (support 1,116)
- Extra: 0.348000 (support 534)
- Substitute: 0.489796 (support 64)
- Missed note: 0.352941 (support 61)

Timestamp overlap was not used for selection or the headline result.
