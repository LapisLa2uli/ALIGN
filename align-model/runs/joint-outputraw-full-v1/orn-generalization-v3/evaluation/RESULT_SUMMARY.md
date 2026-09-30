# ORN v3 one-shot lockbox result

## Decision

The frozen candidate **failed** the 0.80 objective.

- Official precision: **0.669270**
- Official recall: **0.808711**
- Official F1: **0.732413**
- Bootstrap 95% interval: **0.709439–0.752901** (1,000 replicates)
- Predictions / gold: **6,658 / 5,510**
- Credit: **4,456.0**
- Prespecified success threshold: **0.800000**

The failure is primarily excess predictions, not a rounding-edge miss.
No post-lockbox tuning or alternate test selection is allowed.

## Version rationale

The earlier v2 local-rescue candidate used 175 development rows and reached
development F1 0.802271, but narrowly failed its 57-row lockbox at 0.798342.
That version remained experimental and was rejected for the 0.80 objective.

Before any post-v2 tuning, v3 generated and strictly audited a new independent
population. Of 360 fixed-seed rows, 114 passed exact global-lineage and
identity round-trip checks; all 114 were encrypted, and the plaintext source
was deleted. The opened v2 lockbox then became development. The unchanged
evidence-0.40/window-0.15 candidate had aggregate pre-v3 development F1
0.801221 over 232 rows and was frozen.

On the independent v3 lockbox, the later candidate fell to F1 0.732413 because
copy and EXTRA predictions inflated. Comparable per-type F1 was:

- Match: 0.928336
- Copy: 0.682023
- Extra: 0.283822
- Substitute: 0.430839
- Missed note: 0.328267

The later version is therefore also **rejected**, not promoted.

## Isolation evidence

- Candidate SHA-256:
  `a70679959cda097d00b458380a69fadde3987215ef37d2b11271fabd815540a0`
- Protocol SHA-256:
  `b352ad9d9bd72df17a8b0fe53bf1cf6418824ddb7715618431e1ac45f7298176`
- Frozen predictions SHA-256:
  `aea8a5a48d8d7fcc0a269a141f7b14af51ec4b13c9d66cccc639148d477f99c2`
- Opening record SHA-256:
  `8d44c95da9aea9835d758c6130e6021a81f0b6e131786879796c37257ef4c18c`
- Lockbox report SHA-256:
  `9776feef0595edf7017ec45460c74da6454c479cc62462ca5a2590a6a8e85ecd`

All predictions were generated and hashed from public audio/score inputs before
the opening sentinels were written and before the AES key or target envelope
was read. Timestamp metrics were not used.
