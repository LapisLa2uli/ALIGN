# Short plans (deferred)

## data creation and initial inference

Future improvements for Stage 5 alignment / candidate detection and the annotate → train path. Not implemented yet; current rhythm detection uses note/rest-event EWMA + far-window (see `DataCreate` Stage 5).

- **Residual ≠ rhythm:** Stop treating high chroma L2 residual as `rhythm_error`. Add a separate residual / match-quality detector (or map to a clearer type) while still storing `frame_residuals` in `alignment.npz`.
- **Early release:** Dedicated detector when a score note’s mapped span still expects sound but performance energy drops to rest for the remainder of the event.
- **Ornaments / trills:** Handling beyond generic `extra_note` (onset peaks during a notated sustain; written-out grace notes as first-class events).
- **Dedupe EWMA + far-window:** Merge overlapping rhythm candidates from both evaluators into one region more aggressively when comments differ.
- **Breath / phrase-gap FPs:** Reduce false rhythm flags on intentional breaths and phrase endings (clarinet).
- **Score-relative interval ratios:** Detect dotted-vs-even and similar pattern errors via `(perf_IOI_i / perf_IOI_{i-1})` vs score IOI ratios, not only absolute duration vs EWMA.
- **Unreviewed auto labels:** Do not train on raw `source: auto` candidates without human confirm/reject; document risk if that path is ever enabled.
