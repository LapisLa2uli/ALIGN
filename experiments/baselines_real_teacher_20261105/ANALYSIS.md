# Interpretation and diagnostics

These scores measure agreement with AudioEval v9 + MuScriptor medium, using exact canonical reference-note sets and error types. They are not an independent estimate of human-verified accuracy.

## Observed outcome

**Every fine-tuned condition scored below its unchanged pretrained baseline.** Under this fixed 200-update, single-seed protocol, the real-data fine-tunes did not improve the requested reference-score labeling metric. This is a negative result for this training setup, not proof that these architectures cannot benefit from real data.

- PolyTune: baseline F1 **6.78%**; highest fine-tuned F1 **6.29%** at 10 recordings. Its exact matches fell from 8 to 5, while predicted error labels fell from 160 to 83. The precision gain was outweighed by lower recall. The paired change interval at 10 recordings is -5.63 to +3.43 percentage points, so that small observed decline is inconclusive.
- LadderSym: baseline F1 **6.86%**; highest fine-tuned F1 **2.35%** at 30 recordings. The 20/30/40-recording models predict only 18/9/23 error labels across all 17 scored recordings. At 50 recordings it instead produces 487 labels, 432 of which cannot be located in the reference score. Samples 102 and 104 account for 245 and 226 predictions respectively, with no exact matches in either sample.
- None of the 12 conditions exactly recovered any of the 5 extra-note or 13 repetition labels. Missed-note recall was also very low. Most exact matches came from wrong-note labels.
- Allowing half credit for a wrong error type at the correct score location does not reverse the result: every fine-tuned condition also scores below its corresponding baseline under that secondary metric.
- Average training loss decreased in every run, but held-out error-label F1 did not. The observations are consistent with poor transfer and unstable output classification; they do not by themselves identify a single causal mechanism. Sparse teacher errors, fixed training budget, input/target representation differences, and the unchanged score-localization adapter are possible contributors.

The checkpoints remain experimental artifacts. No result here supports replacing the existing baseline with one of these fine-tunes. See `RESULTS.md` for the complete precision/recall table and paired intervals.

## Native output and localization

Counts below use only the 17 scored recordings. Native note counts precede conversion into error-label ranges. Unlocated label ranges and unclassified notes remain false positives in the primary score.

| Model | N | Native Correct | Native Extra | Native Missing | Predicted error labels | Unlocated labels | Exact matches / 76 |
|---|---:|---:|---:|---:|---:|---:|---:|
| polytune | 0 | 519 | 146 | 102 | 160 | 78 | 8 / 76 |
| polytune | 10 | 962 | 77 | 57 | 83 | 48 | 5 / 76 |
| polytune | 20 | 779 | 86 | 65 | 95 | 65 | 1 / 76 |
| polytune | 30 | 669 | 65 | 61 | 73 | 35 | 3 / 76 |
| polytune | 40 | 616 | 81 | 63 | 89 | 58 | 0 / 76 |
| polytune | 50 | 694 | 82 | 66 | 95 | 58 | 3 / 76 |
| laddersym | 0 | 603 | 94 | 104 | 128 | 40 | 7 / 76 |
| laddersym | 10 | 692 | 148 | 157 | 165 | 156 | 0 / 76 |
| laddersym | 20 | 716 | 16 | 12 | 18 | 15 | 0 / 76 |
| laddersym | 30 | 580 | 9 | 3 | 9 | 6 | 1 / 76 |
| laddersym | 40 | 729 | 21 | 11 | 23 | 19 | 0 / 76 |
| laddersym | 50 | 647 | 371 | 396 | 487 | 432 | 1 / 76 |

## Error-type recall

Recall is exact label-location/type recall. A partial overlap with a longer error range receives no strict credit.

| Model | N | Wrong (49) | Missed (9) | Extra (5) | Repetition (13) |
|---|---:|---:|---:|---:|---:|
| polytune | 0 | 16.3% | 0.0% | 0.0% | 0.0% |
| polytune | 10 | 8.2% | 11.1% | 0.0% | 0.0% |
| polytune | 20 | 2.0% | 0.0% | 0.0% | 0.0% |
| polytune | 30 | 6.1% | 0.0% | 0.0% | 0.0% |
| polytune | 40 | 0.0% | 0.0% | 0.0% | 0.0% |
| polytune | 50 | 6.1% | 0.0% | 0.0% | 0.0% |
| laddersym | 0 | 12.2% | 11.1% | 0.0% | 0.0% |
| laddersym | 10 | 0.0% | 0.0% | 0.0% | 0.0% |
| laddersym | 20 | 0.0% | 0.0% | 0.0% | 0.0% |
| laddersym | 30 | 2.0% | 0.0% | 0.0% | 0.0% |
| laddersym | 40 | 0.0% | 0.0% | 0.0% | 0.0% |
| laddersym | 50 | 2.0% | 0.0% | 0.0% | 0.0% |

## Fixed training budget

| Recordings | Available windows | Unique windows sampled | Sampled windows with native errors / 800 |
|---:|---:|---:|---:|
| 10 | 285 | 254 | 176 / 800 |
| 20 | 551 | 371 | 162 / 800 |
| 30 | 801 | 452 | 127 / 800 |
| 40 | 1077 | 500 | 141 / 800 |
| 50 | 1326 | 532 | 169 / 800 |

Both models use the same window draws. More recordings provide greater diversity but fewer revisits per recording under this budget. A non-monotonic learning curve does not establish that more data is intrinsically harmful, or that 200 updates is sufficient for every dataset size.

## Coverage and limitations

- All 21 samples were decoded for all 12 conditions. Four teacher abstentions (101, 105, 114, 117) are excluded from F1, not relabeled as clean.
- The 17 scored samples contain 677 canonical score notes. Seven are marked unassessed by the teacher (100: three; 119: four). Unpublished teacher decisions are not independent negative labels; disagreement can reflect a teacher omission.
- Exact range matching is sensitive to grouping: finding some correct affected notes inside a different range earns no strict match. Native transcription, error classification, and score localization all contribute to the final score.
- Original reference renders can contain expanded repeats, ornaments, or chord notes outside the existing canonical parser's note list. This is particularly relevant to 100, 104, and 109; see `data_audit.json` and `README.md`.
- There is no detected identical training/test audio or full canonical pitch sequence, and available source-score names are disjoint. Missing metadata prevents a complete proof of piece-level independence.
- The bootstrap intervals in `RESULTS.md` use paired resampling of recordings; there is only one training seed and one nested subset ordering. The highest score on this test set should be treated as exploratory, not a deployment selection validated on another holdout.

See `results.json` for every per-recording match and the secondary type-mismatch-credit metric, `training_diagnostics.json` for training loss and exposure, and `final_audit.json` for preservation and integrity checks.
