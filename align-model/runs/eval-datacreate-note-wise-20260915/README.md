# DataCreate canonical note-wise evaluation — 2026-09-15

All 94 label documents were inventoried. Provenance was determined only from
document-level fields. There are 91 `annotator_id: ai_f0_align` documents
(18 nonempty, 49 labels), one explicit local user document
(`004`, `annotator_id: annotator01`, empty/unreviewed), and two ambiguous
documents (`005`, six wrong-note labels but no document annotator; `demo_001`,
empty). Label-level `source: manual` did not move `005` into the user subset.

The stricter all-label canonical audit accepted 10/18 agent clips and 28/49
individual rows. Eight clips were excluded as a unit because at least one gold
row failed: 19 rows had pitch lists inconsistent with their claimed canonical
score range and two lacked a valid canonical identity. The official score
therefore covers 23 labels on clips
`001,006,010,012,017,019,020,029,033,034`. This is stricter than the earlier
11/18 supported-type-only audit because it validates every gold type.

Official agent-label results:

- v3: P/R/F1 `0/0/0`, 167 predictions, 23 gold
- balanced v3: P/R/F1 `0/0/0`, 67 predictions, 23 gold
- v2: P/R/F1 `0/0/0`, 20 predictions, 23 gold
- rules: P `0.000923`, R `0.021739`, F1 `0.001770`, one half-credit match,
  2,000-clip-bootstrap 95% CI `[0, 0.006623]`

All per-type F1 values are zero. These numbers measure agreement with
agent-generated labels, not independent accuracy.

The user subset has no verifiable nonempty gold: `004` is empty/unreviewed.
Its official result is unavailable. `005` remains explicitly ambiguous rather
than being guessed human from label-level sources.

As a separate `manual_provenance_ambiguous` follow-up, sample `005` was audited
without changing that classification. Two of its six wrong-note rows have
unique validated canonical ranges: indices `23–27` and `29–34`. Four rows were
rejected because their pitch lists do not validate the claimed score ranges
(`37–41`, `43–47`, `37–47`, and `131–136`). Against the two accepted rows,
V3, balanced V3, V2, and rules all have P/R/F1 `0/0/0`, no full- or half-credit
matches, and bootstrap interval `[0,0]`. Prediction counts are respectively
29, 8, 6, and 60. This result remains provenance-ambiguous and is never pooled
with agent or explicit-user results.

Predictions were not regenerated. V3, balanced V3, V2, and rules documents
come from the previously hash-verified 94-sample freeze. V4/V5 were omitted
because that freeze does not contain their deterministic documents or the
upstream state required to recreate them without reopening inference.
Timestamp metrics are not included. Lockbox and production state are
unchanged.
