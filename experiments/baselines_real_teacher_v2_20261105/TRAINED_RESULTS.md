# Actual trained-model assessment: 10, 20, and 40 recordings

Every row for N=10/20/40 below uses a genuinely fine-tuned checkpoint. The checkpoint is selected by validation reference-note F1 among updates 100/200/300, with earlier-update ties. This supplemental rule was frozen before any test generation. Rejected candidates are included for scientific assessment, not promoted for use. The separate guarded-selection report is RESULTS.md.

Primary metric: exact canonical reference-score note set and error type; timestamps are not the matching standard. All 21 DataCreate recordings 100–120 are inferred, and 17 with 76 usable AudioEval v9 + MuScriptor medium pseudo-gold labels are scored. Abstentions 101/105/114/117 are excluded from scoring. This is the same previously examined benchmark, not a new holdout.

| Model | Training recordings | Trained update | Precision | Recall | Strict F1 | Change vs corrected control | Paired 95% interval | Validation stability guard |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| polytune | 0 | 0 | 2.47% | 11.84% | 4.08% | +0.00% | +0.00% to +0.00% | control |
| polytune | 10 | 200 | 2.67% | 11.84% | 4.36% | +0.28% | -2.84% to +3.49% | failed |
| polytune | 20 | 200 | 3.32% | 14.47% | 5.41% | +1.32% | -1.44% to +4.31% | failed |
| polytune | 40 | 200 | 3.15% | 14.47% | 5.18% | +1.09% | -1.57% to +4.78% | failed |
| laddersym | 0 | 0 | 1.74% | 3.95% | 2.42% | +0.00% | +0.00% to +0.00% | control |
| laddersym | 10 | 200 | 2.05% | 7.89% | 3.26% | +0.84% | -2.83% to +5.07% | failed |
| laddersym | 20 | 100 | 2.05% | 3.95% | 2.70% | +0.28% | -4.65% to +6.03% | failed |
| laddersym | 40 | 100 | 2.13% | 3.95% | 2.76% | +0.35% | -4.58% to +6.08% | failed |

## Comparison with previous fine-tunes

| Model | Recordings | Previous published F1 | Previous outputs + v2 adapter | Revised actual trained F1 |
|---|---:|---:|---:|---:|
| polytune | 0 | 6.78% | 6.78% | 4.08% |
| polytune | 10 | 6.29% | 6.29% | 4.36% |
| polytune | 20 | 1.17% | 1.17% | 5.41% |
| polytune | 40 | 0.00% | 1.21% | 5.18% |
| laddersym | 0 | 6.86% | 6.86% | 2.42% |
| laddersym | 10 | 0.00% | 0.00% | 3.26% |
| laddersym | 20 | 0.00% | 0.00% | 2.70% |
| laddersym | 40 | 0.00% | 0.00% | 2.76% |

## Error types and output stability

| Model | Recordings | Wrong F1 | Missed F1 | Extra F1 | Repetition F1 | Exact / predicted / teacher labels | Test capped windows |
|---|---:|---:|---:|---:|---:|---:|---:|
| polytune | 0 | 10.59% | 0.00% | 0.00% | 0.00% | 9 / 365 / 76 | 0 / 320 |
| polytune | 10 | 9.47% | 0.00% | 0.00% | 0.00% | 9 / 337 / 76 | 0 / 320 |
| polytune | 20 | 11.17% | 0.00% | 0.00% | 0.00% | 11 / 331 / 76 | 0 / 320 |
| polytune | 40 | 9.52% | 0.00% | 1.07% | 0.00% | 11 / 349 / 76 | 0 / 320 |
| laddersym | 0 | 4.69% | 0.00% | 0.00% | 0.00% | 3 / 172 / 76 | 0 / 320 |
| laddersym | 10 | 7.94% | 0.00% | 3.12% | 0.00% | 6 / 292 / 76 | 2 / 320 |
| laddersym | 20 | 5.08% | 0.00% | 0.00% | 0.00% | 3 / 146 / 76 | 0 / 320 |
| laddersym | 40 | 5.41% | 0.00% | 0.00% | 0.00% | 3 / 141 / 76 | 0 / 320 |

A failed validation stability guard means this candidate produced more length-capped outputs than allowed, even if its F1 happened to rise. The guard and unchanged-model fallback remain in force in the original selection report; these diagnostic results do not override them.

The teacher labels are not human ground truth. Exact range matching is sensitive to grouping, and native Extra/Missing/Correct conversion still limits wrong-note and repetition reconstruction. One seed and one nested subset order were used. Confidence intervals are paired recording bootstraps, not estimates of seed variance. Original checkpoints, human labels, previous results, and application defaults are preserved.

See RESULTS.md for preprocessing, training settings, context audit, and runtime validation; ANALYSIS.md for all validation candidates; trained_results.json for all per-recording trained-test results.
