# Shared outputRaw canonical evaluation

The comparison uses all 358 validation IDs from the uploaded frozen
`joint-outputraw-full-v1` split. Both baselines retain their previously selected
best checkpoints: Polytune 31 completed epochs, prompted LadderSym 26.

Only the canonical combined-pipeline score will be reported in the paper.
The shared scorer is `align-note-wise-score-event-metric-v1`: exact location
and complete type receive 1 credit; exact location and a different type receive
0.5; wrong or missing identity receives no credit. All predictions, including
unlocated events and duplicate predicted deletions, remain in the denominator.

## Data identity

- Frozen split SHA-256: `40f4900c142862fd06a69b1adcb2049701655db5ccb0266b925ff8cdc04bce6d`.
- Canonical target SQLite SHA-256: `137d6069c2909e111ef663f87bb193801e79f056916c695d343b1a641bf61584`.
- All 358 performance WAVs and MIDIs match the source manifest byte for byte;
  symbolic clean/performed/deleted lineage also matches exactly.
- 26 bundles reuse existing local inputs; 332 were reconstructed from the source
  score and original sample seed. No test targets or test audio were opened.
- Full canonical target support is 40,712 events, including score deletions.
- The baseline training and selection used a different 10,049/1,116 split.
  Exactly 21 evaluation audio files occur in baseline training, and three in
  its original validation set. This is a comparison of existing checkpoints,
  not a controlled comparison with matched training data or wholly unseen audio.
  The paper must disclose this overlap.

## Fixed conversion

`baselines/common/canonical_adapter.py` uses only native predictions and the
clean reference score. It pairs mutually nearest Extra/Missing onsets within
100 ms with different pitches into substitutions. Correct, substitution, and
remaining Missing events form anchors for global monotonic edit alignment:
unit insertion/deletion cost, 1.5 pitch mismatch cost, deterministic diagonal
priority on ties. Reference gaps do not create missing-note predictions.
Remaining Extra runs of at least three notes may map to previously visited
contiguous score sequences with at least 80% pitch agreement. The longest
accepted sequence is selected; ties prefer more pitch matches and the latest
source position. Copy passes follow prior predicted visits.

The converter emits no rhythm detections because neither native baseline has
that class. Rhythm stays in the combined gold type; any mismatch is scored by
the same fractional-credit rule. Unlinked extras receive no invented gold
rendered-event ID. All baseline events are retained, except that an explicit
Extra/Missing substitution pair becomes one substituted event.

Parameters are fixed before inspecting any validation score. The freeze
process has an audit hook denying access to uploaded targets, SQLite files,
label documents, and note maps. A separate process verifies frozen prediction
hashes before opening gold. Five tests cover substitution/deletion identities,
repeat source/pass, retained unmatched predictions, absence of invented
missing predictions, and exact parity with Ours' combined-pipeline metric.

## Reproduction

Use `align-model/runs/env/bin/python` for these scripts:

1. `baselines/scripts/restore_outputraw_validation.py` restores and verifies inputs.
2. `baselines/scripts/prepare_outputraw_evaluation.py` builds the baseline loader layout.
3. The exact checkpoint choices and launch commands are recorded in
   `checkpoint_selection.json` and `inference_launch.json`.
4. `baselines/scripts/evaluate_outputraw_canonical.py freeze` fixes score locations.
5. The same script's `score` subcommand runs separately against uploaded targets.

Run status and complete predictions are under
`baselines/runs/evaluate_20260915/outputraw358_canonical/`.
The frozen protocol and implementation hashes are in `frozen_protocol.json`.
Auxiliary native scores created by upstream inference wrappers are internal
artifacts and are not the reported paper metric.

## Completed results

All 358 predictions per baseline are frozen and scored against identical gold.
Canonical combined-pipeline micro-F1: Polytune **0.4139**, prompted LadderSym
**0.3471**. Ours retains the paper's existing **0.6104** result.
See `results.json` and `status.json` for precision, recall, confidence intervals,
checkpoint epochs, source hashes, and final publication status.

Overleaf push succeeded at `f160c26ee46118817818298bc88f648e486e46da`. The experiment section now has three
tables and two new vector figures (dataset partitions and unified F1 with 95%
confidence intervals), replacing two previous tables. The final PDF compiles
without warnings or overfull boxes.
