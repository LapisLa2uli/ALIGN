# Secondary reference-note/type assessment

This diagnostic was declared during validation, before test generation, and is not used to select checkpoints. It measures individual reference-note/type agreement while reducing the penalty for different label grouping. The primary exact-range score remains in TRAINED_RESULTS.md. Timestamps are not used in either metric.

Wrong and missed labels contribute one unit per affected canonical reference note. Repetitions contribute one unit per note per extra copy. An extra-note insertion contributes one unit identified by its exact canonical-neighbor tuple. Duplicate predictions cannot earn duplicate credit beyond gold multiplicity; unlocated and unclassified outputs count as false positives.

| Model | Training recordings | Exact-range F1 | Note/type precision | Note/type recall | Note/type F1 | Paired change vs control 95% interval |
|---|---:|---:|---:|---:|---:|---:|
| polytune | 0 | 4.08% | 2.93% | 5.58% | 3.85% | +0.00% to +0.00% |
| polytune | 10 | 4.36% | 3.68% | 6.60% | 4.73% | -1.79% to +3.25% |
| polytune | 20 | 5.41% | 4.06% | 7.11% | 5.17% | -0.56% to +3.05% |
| polytune | 40 | 5.18% | 4.13% | 7.61% | 5.36% | -0.34% to +3.57% |
| laddersym | 0 | 2.42% | 2.87% | 2.54% | 2.70% | +0.00% to +0.00% |
| laddersym | 10 | 3.26% | 2.03% | 3.05% | 2.43% | -3.80% to +3.59% |
| laddersym | 20 | 2.70% | 2.03% | 1.52% | 1.74% | -5.31% to +4.10% |
| laddersym | 40 | 2.76% | 2.13% | 1.52% | 1.78% | -5.31% to +4.13% |

These scores still measure agreement with model-generated teacher labels, not human-verified accuracy. The unit weighting differs from the primary metric: long wrong/missed/repeated ranges contribute more units, and the metric does not require the same segmentation into labels. All six nonzero conditions use actual trained checkpoints chosen only by validation F1.
