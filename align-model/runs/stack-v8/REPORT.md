# v8 same-pitch postprocessor evaluation

The processor is implemented and runnable, but this evaluation shows **no improvement** over selected v7. It merged zero naturally decoded boundaries. It remains experimental; v7 and the GUI default are unchanged.

## Paired comparison

Both variants use the selected v7 checkpoint, missed gate 0.80, identical acoustic caches and original waveform evidence. Thresholds were fixed heuristics, not fitted to these labels. Evaluation uses canonical reference-score note identity, not timestamp overlap. Synthetic clips are previously inspected development validation data.

| Dataset | Clips | Equal-pitch pairs checked | Merged | Content-error F1 before | After |
|---|---:|---:|---:|---:|---:|
| realistic92 | 30 | 276 | 0 | 0.878238 | 0.878238 |
| fast102 | 30 | 76 | 0 | 0.954468 | 0.954468 |
| dclike11 | 30 | 209 | 0 | 0.864198 | 0.864198 |
| DataCreate 001-094 | 94 | 281 | 0 | See below | Unchanged |

Macro synthetic content-error F1: 0.898968 before and after.

DataCreate accuracy includes only 25 reviewed takes (including explicitly clean 004). Unlabeled takes were not treated as clean. Labels 095 onward were not read.

| Error type | Correct identities | Predictions | Gold labels | F1 before and after |
|---|---:|---:|---:|---:|
| extra_note | 2 | 32 | 12 | 0.090909 |
| missed_note | 1 | 8 | 8 | 0.125000 |
| wrong_note | 1 | 22 | 22 | 0.045455 |
| repetition | 0 | 9 | 4 | 0.000000 |

The existing DataCreate weaknesses remain: many incorrect extra/wrong-note identities, low missed-note recall, no exact repetition hits, and 12 takes marked alignment_uncertain. This processor addresses only adjacent equal-pitch emissions, not general transcription or alignment errors.

## Why boundaries were retained

| Reason | Boundaries |
|---|---:|
| energy_boundary | 29 |
| insufficient_context | 327 |
| separate_attack | 481 |
| silence_or_unvoiced | 4 |
| spectral_boundary | 1 |

All 842 pairs have an audit entry; 327 lack enough isolated context for a confident acoustic decision. Retention reasons are conservative classifier decisions, not ground-truth proof of separate notes.

## Positive and negative controls

Using the same 90 synthetic recordings and gold rendered-note lineage, an artificial boundary was inserted halfway through each isolated rendered note lasting at least 0.5 seconds. These controls do not alter audio or train the model. Genuine adjacent equal rendered notes provide negative controls.

- Artificial splits removed: **5 / 316 (1.58%)**.
- Genuine repeat boundaries incorrectly merged: **0 / 543**.

This demonstrates working merge behavior but poor sensitivity. Among artificial splits, 221 were blocked by energy variation, 58 by model attack evidence, 16 by silence/voicing, and 16 by spectral variation. Gold timing is used only to construct controls; it is not used in deployment or the paired label comparison. The apparent safety and sensitivity estimates are limited to these controls.

## Verification and artifacts

- 19 focused tests passed, including repair-to-alignment reference identity, tones, attacks, silence, spectral changes, chains, malformed input and missing context.
- A fresh waveform-to-feedback CLI run completed on DataCreate 020; `example-020.json`.
- `CANDIDATE_STACK_V8.json`: fixed configuration and source/checkpoint hashes.
- `evaluation_protocol.json`: input job list and comparison protocol.
- `same_pitch_audit.json`: all decisions and source candidate groups.
- `predictions/before/` and `predictions/after/`: per-clip feedback and alignments.
- `synthetic.json`, `datacreate-before.json`, `datacreate-after.json`: canonical note metrics.
- `controls.json`: individual positive/negative control decisions.
- `COMPLETED.json`: completion counts and aggregate summary.

No default promotion or accuracy-improvement claim is justified by these results.
