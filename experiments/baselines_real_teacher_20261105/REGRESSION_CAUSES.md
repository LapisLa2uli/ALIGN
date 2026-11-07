# Follow-up diagnosis of the deterioration

The learning curve is a valid measurement of this implementation, but it is not evidence that real-data fine-tuning is inherently harmful. Follow-up inspection found concrete weaknesses in the training inputs and score-label adapter, as well as a reproducible decoder regression. No original predictions, training targets, checkpoint weights, metrics, or protocol settings were changed during this investigation.

## 1. Reference conditioning is often the wrong musical passage

`prepare_data.py` resamples and zero-pads the performance and reference audio to equal lengths. It does not align their musical positions. Both native dataset loaders then take a 2.048-second performance window at time t and a 4.096-second reference context from approximately t - 1.024 to t + 3.072. The reference context is compressed into 256 mel frames; this is intentional context compression, not accidental truncation.

Real performances contain different tempos, pauses, and repeats, so the matching score note can lie well outside that context. According to the teacher's alignment, the target can say Correct while the corresponding expected note is absent from the supplied reference passage. Such examples can encourage a model to ignore the reference or learn inconsistent error classifications.

The diagnostic uses only the 32 selected training recordings whose canonical XML pitch sequence exactly matches the sounding reference MIDI sequence. It reproduces the actual random window draws and checks teacher-matched Correct target notes against the corresponding MIDI note's interval. It adds 100ms of boundary tolerance. Recordings with ambiguous MIDI/XML sequences are excluded, not guessed.

| Training recordings | Audited sampled windows / 800 | Checked correct-note occurrences | Corresponding reference note outside supplied context | Fraction |
|---:|---:|---:|---:|---:|
| 10 | 405 | 4,245 | 2,358 | 55.5% |
| 20 | 472 | 4,666 | 2,284 | 48.9% |
| 30 | 526 | 5,045 | 2,575 | 51.0% |
| 40 | 472 | 4,469 | 1,976 | 44.2% |
| 50 | 520 | 4,950 | 1,955 | 39.5% |

These are sampled note occurrences, including repeat visits and overlapping windows, not percentages of distinct notes in the entire corpus. They depend on the teacher's score assignment and are not an independent human alignment audit.

Example: in sample 032, a teacher-matched performance note around 25.52s corresponds to a reference MIDI note at 20.17s. A sampled performance window at 25.68–27.73s receives reference audio at 24.66–28.75s, missing that expected note entirely.

This is a concrete preparation limitation that should have been checked before interpreting the negative learning curve. Its causal contribution requires a corrected-window ablation; the current run cannot quantify how much of the deterioration it explains.

## 2. LadderSym develops repetitive generation bursts

An instrumented rerun of sample 102 reproduced both the original and fine-tuned saved MIDI byte-for-byte.

For window 8 (roughly 16.384–18.432s):

- Original checkpoint: **30 output tokens**, including end-of-sequence.
- 50-recording fine-tune: **1,013 output tokens**, eventually reaching end-of-sequence, with repeated Missing-class/pitch token pairs.
- The fine-tuned MIDI contains **100 Extra and 163 Missing notes** whose onsets fall in this 2.048-second bin. They are within the actual audio duration, not simply notes in a padded tail.

Related bursts appear in sample 106 for the 10-recording model and sample 104 for the 50-recording model. Other fine-tuned LadderSym conditions emit very few errors overall. This confirms unstable free generation, not just a stricter evaluation penalty. It does not alone explain why the weights became unstable.

## 3. Sparse errors and extensive parameter updates are plausible contributors

The chosen 50 training recordings contain 5,914 Correct, 543 Extra, and 110 Missing native note targets before unknown-window exclusion. Only about 16–22% of sampled windows overlap native error targets. The remaining windows mainly teach ordinary note sequences.

All approximately 192 million trainable PolyTune parameters and 172 million LadderSym parameters were updated, with no synthetic-data replay, frozen-backbone constraint, or separate validation-based stopping. Training loss decreased, but the objective is teacher-forced token prediction; it does not directly reward free-running generation stability or correct reference-note identities. Poor transfer, forgetting, and class imbalance are plausible mechanisms. They have not been separately established by controlled ablations.

Each run sampled 800 windows regardless of dataset size. Larger recording counts therefore did not receive more optimizer updates, and the curve is not a comparison at equal epochs or individually optimized convergence.

## 4. The unchanged adapter limits what the score can measure

`baselines/common/canonical_adapter.py` assigns score indices to Correct/Wrong/Missing anchors, then recognizes some Extra runs as copies. **Ordinary Extra events never receive neighboring score-note identities.** Consequently, their converted labels cannot pass the strict reference-note metric. This affects the zero-data models too and cannot explain the entire relative decline, but it invalidates interpreting zero extra-note F1 as proof that the neural model cannot detect extras.

Training also represents wrong notes as Extra + Missing and repetitions as Extra; source score ranges and repetition copy counts are not explicit native targets. The fixed downstream adapter must reconstruct them. Hundreds of erroneous Missing/Extra events can overwhelm that localization stage: LadderSym at 50 recordings has 432 unlocated labels out of 487 predicted labels.

## Implication for the next experiment

First correct musical correspondence of training and inference reference windows using a label-free method available at inference, and assign canonical neighbor identities to ordinary extra notes. Then test a more conservative adaptation setup with error-balanced sampling, preservation of pretrained behavior, and a separate validation set. Keep the existing experiment as a negative baseline; do not tune on the 100–120 test labels and present that as a fresh held-out result.

Evidence: `regression_diagnostics.json`, `decoder_regression_trace.json`, `diagnose_regression.py`, and `trace_decoder_regression.py`. The original learning curve remains in `RESULTS.md`.
