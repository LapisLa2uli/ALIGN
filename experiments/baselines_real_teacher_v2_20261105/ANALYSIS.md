# Assessment interpretation

Primary scores require exact reference-score note sets and error types, including repetition copy counts. The teacher is AudioEval v9 + MuScriptor medium; these are not independently human-verified labels.

## Selected test outcomes

| Model | Training recordings | Selected update | Test exact matches | Test predictions | Test strict F1 | Change from corrected control |
|---|---:|---:|---:|---:|---:|---:|
| polytune | 0 | 0 | 9 / 76 | 365 | 4.08% | +0.00% |
| polytune | 10 | 0 | 9 / 76 | 365 | 4.08% | +0.00% |
| polytune | 20 | 0 | 9 / 76 | 365 | 4.08% | +0.00% |
| polytune | 40 | 0 | 9 / 76 | 365 | 4.08% | +0.00% |
| laddersym | 0 | 0 | 3 / 76 | 172 | 2.42% | +0.00% |
| laddersym | 10 | 0 | 3 / 76 | 172 | 2.42% | +0.00% |
| laddersym | 20 | 0 | 3 / 76 | 172 | 2.42% | +0.00% |
| laddersym | 40 | 0 | 3 / 76 | 172 | 2.42% | +0.00% |

polytune: training with 10, 20, 40 recordings selected the unchanged-model fallback. Those test rows reuse the control predictions and do not demonstrate learning from those recordings.

laddersym: training with 10, 20, 40 recordings selected the unchanged-model fallback. Those test rows reuse the control predictions and do not demonstrate learning from those recordings.

## Every validation candidate

All six fine-tunes ran 300 updates. Free generation, not teacher-forced loss, evaluated every checkpoint. The unchanged model is eligible; trained candidates must not exceed its capped-window rate by more than 0.005. Earlier updates win F1 ties. No test score selects checkpoints.

| Model | Recordings | Update | Exact matches | Predictions | Strict F1 | Capped windows | Eligible | Selected |
|---|---:|---:|---:|---:|---:|---:|---|---|
| polytune | 0 | 0 | 1 / 37 | 1096 | 0.18% | 17 / 173 | yes | control |
| polytune | 10 | 100 | 1 / 37 | 752 | 0.25% | 26 / 173 | no | no |
| polytune | 10 | 200 | 2 / 37 | 1363 | 0.29% | 25 / 173 | no | no |
| polytune | 10 | 300 | 0 / 37 | 2083 | 0.00% | 31 / 173 | no | no |
| polytune | 20 | 100 | 2 / 37 | 1346 | 0.29% | 23 / 173 | no | no |
| polytune | 20 | 200 | 3 / 37 | 1870 | 0.31% | 33 / 173 | no | no |
| polytune | 20 | 300 | 2 / 37 | 2146 | 0.18% | 31 / 173 | no | no |
| polytune | 40 | 100 | 1 / 37 | 965 | 0.20% | 24 / 173 | no | no |
| polytune | 40 | 200 | 3 / 37 | 2170 | 0.27% | 34 / 173 | no | no |
| polytune | 40 | 300 | 0 / 37 | 2205 | 0.00% | 29 / 173 | no | no |
| laddersym | 0 | 0 | 3 / 37 | 2296 | 0.26% | 21 / 173 | yes | control |
| laddersym | 10 | 100 | 1 / 37 | 2246 | 0.09% | 20 / 173 | yes | no |
| laddersym | 10 | 200 | 4 / 37 | 4457 | 0.18% | 33 / 173 | no | no |
| laddersym | 10 | 300 | 4 / 37 | 5084 | 0.16% | 36 / 173 | no | no |
| laddersym | 20 | 100 | 2 / 37 | 2647 | 0.15% | 22 / 173 | no | no |
| laddersym | 20 | 200 | 1 / 37 | 4425 | 0.04% | 33 / 173 | no | no |
| laddersym | 20 | 300 | 1 / 37 | 4889 | 0.04% | 33 / 173 | no | no |
| laddersym | 40 | 100 | 2 / 37 | 2793 | 0.14% | 22 / 173 | no | no |
| laddersym | 40 | 200 | 1 / 37 | 4298 | 0.05% | 29 / 173 | no | no |
| laddersym | 40 | 300 | 2 / 37 | 4859 | 0.08% | 33 / 173 | no | no |

## Remaining output problems

- polytune, N=10: 356 unmatched predicted labels and 67 unmatched teacher labels. Largest false-positive contributors: 113 (0 exact / 36 predictions), 106 (4 exact / 35 predictions), 112 (1 exact / 32 predictions). No exact recovery for: missed_note, extra_note, repetition. Capped generation: 0 / 320 windows.
- polytune, N=20: 356 unmatched predicted labels and 67 unmatched teacher labels. Largest false-positive contributors: 113 (0 exact / 36 predictions), 106 (4 exact / 35 predictions), 112 (1 exact / 32 predictions). No exact recovery for: missed_note, extra_note, repetition. Capped generation: 0 / 320 windows.
- polytune, N=40: 356 unmatched predicted labels and 67 unmatched teacher labels. Largest false-positive contributors: 113 (0 exact / 36 predictions), 106 (4 exact / 35 predictions), 112 (1 exact / 32 predictions). No exact recovery for: missed_note, extra_note, repetition. Capped generation: 0 / 320 windows.
- laddersym, N=10: 169 unmatched predicted labels and 73 unmatched teacher labels. Largest false-positive contributors: 106 (2 exact / 22 predictions), 102 (0 exact / 17 predictions), 107 (0 exact / 16 predictions). No exact recovery for: missed_note, extra_note, repetition. Capped generation: 0 / 320 windows.
- laddersym, N=20: 169 unmatched predicted labels and 73 unmatched teacher labels. Largest false-positive contributors: 106 (2 exact / 22 predictions), 102 (0 exact / 17 predictions), 107 (0 exact / 16 predictions). No exact recovery for: missed_note, extra_note, repetition. Capped generation: 0 / 320 windows.
- laddersym, N=40: 169 unmatched predicted labels and 73 unmatched teacher labels. Largest false-positive contributors: 106 (2 exact / 22 predictions), 102 (0 exact / 17 predictions), 107 (0 exact / 16 predictions). No exact recovery for: missed_note, extra_note, repetition. Capped generation: 0 / 320 windows.

Capped generation is evidence of decoder instability; it is not a physical repeated-note diagnosis. Legitimate adjacent equal-pitch notes are not suppressed. Mislocated ranges, incorrect error types, extra/missing pairing, and repetition reconstruction can all produce strict-score failures. An error range that overlaps the right notes but has different boundaries receives no exact match under this metric.

The context audit established a preprocessing improvement, not an end-to-end accuracy guarantee. On auditable N40 training note occurrences, the corresponding reference note was outside the supplied crop 50.08% before retrieval and 8.92% afterward. The remaining mismatch and excluded ambiguous reference renders limit that audit.

Conservative decoder adapters prevent changes to the original weights, but do not guarantee stable free generation. The KL term uses teacher-forced prefixes; it does not directly penalize loops reached only during autoregressive decoding. This is a plausible limitation of the training objective, not a proven sole cause. The experiment does not isolate the effects of LoRA, balancing, retrieval, and regularization from one another.

## Interpretation limits

The same DataCreate 100–120 benchmark was examined previously. Four teacher abstentions are excluded rather than treated as clean recordings. One training seed and one nested subset order were used. Paired confidence intervals in RESULTS.md describe variation across the 17 scored recordings, not training-seed uncertainty.

Test scores compare agreement with the teacher and cannot establish human-verified error accuracy. The selected checkpoints remain experimental; application defaults were preserved. Stronger claims need manually checked score-note labels, additional recordings, and a new evaluation split.
