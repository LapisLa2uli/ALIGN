# Baselines on the user-provided 034.zip test set

This experiment evaluates the existing Polytune and prompted LadderSym best
checkpoints on the exact two selections requested by the user:

- Full: all 40 clips, 001–040.
- Filtered: 30 clips after excluding 005, 007, 010, 012, 013, 020, 026, 030,
  034, and 036.

Each model runs inference once on all 40 clips. Both summaries use the same
frozen predictions. No checkpoint, threshold, conversion parameter, or sample
exclusion is chosen using these test scores. The selected checkpoints completed
31 epochs for Polytune and 26 epochs for LadderSym. Their exact paths and hashes
are recorded in `checkpoint_selection.json`.

## Ground truth and interpretation

The user explicitly confirmed that `labels.json` is human-reviewed ground truth
and that empty label documents indicate clean performances. That confirmation
takes precedence over stale `annotator_id=ai_f0_align` metadata. Empty documents
are retained: any predicted errors on those recordings count as false positives.
`labels_agent.json`, existing transcription outputs, and alignment caches are
not used to construct model predictions.

The main taxonomy has five error types: wrong_note, missed_note, extra_note,
rhythm_error, repetition. There are 38 gold error events in the full set and 18
in the filtered set. Normal notes are not included in the error-event metric.
The separate all-annotated-types view also includes click, bad_start, squeak,
sliding, and bad_timbre (55 full / 29 filtered gold events). Both models lack
native predictions for these extra acoustic classes and rhythm_error; their
gold events remain in the respective denominators. A shared-four-type view
omits rhythm_error and is provided explicitly as a different taxonomy.

There is no complete performed-note gold transcription in this archive.
Consequently, the native note-level F1 and the synthetic combined-pipeline F1
(which includes normal notes) are unavailable. The inference loaders use empty
MIDI placeholders; any zero scores printed against them are not test results.

The pre-existing canonical label audit finds inconsistent score-range/pitch
metadata on some gold labels and missing repetition identities. Both requested
sets contain such labels, so their official canonical note-wise scores are
marked unavailable. No invalid clips or labels are silently dropped and no
gold locations are reconstructed from model predictions. `label_audit.json`
retains every finding; `results.json` narrows the findings to the five task types.

## Available timestamp diagnostics

Native named Extra/Missing/Correct MIDI tracks are converted using the existing
`baselines/common/eval_bridge.py::notes_to_spans` policy with its unchanged
defaults. This pairs nearby Extra/Missing predictions into wrong notes, retains
unpaired missing notes, recognizes qualifying extra-note repetition runs, and
retains other extras. Parameters are saved in `protocol.json`. Unnamed decoder
notes are retained as unmatched unclassified predictions, never assigned a
guessed class.

Prediction conversion runs separately from scoring, with a file-access guard
that denies gold/bundle reads. MIDI and converted-output hashes are frozen;
scoring checks converted hashes and code identity before reading gold.

Timestamp matching is typed, maximum-cardinality, exclusive one-to-one:

- Onset differences at most 50, 100, and 200 ms.
- Interval IoU at least 0.3 and 0.5, reported separately.

These are full-credit timestamp diagnostics, **not canonical location scores**.
Micro precision/recall/F1 pool matched, predicted, and gold event counts over
all clips. Macro F1 averages categories with gold support. Confidence intervals
use 2,000 clip bootstrap replicates, seed 365. The primary timestamp diagnostic
is 50 ms onset F1, fixed before scoring; all criteria remain in the report.

## Artifacts and reproduction

- `dataset_manifest.json`: archive hash, exact selections, user confirmation.
- `inference_launch.json`: exact commands, GPUs, logs, prediction directories.
- `protocol.json`: fixed conversion/scoring choices and code hashes.
- `polytune_frozen/`, `laddersym_frozen/`: converted predictions and manifests.
- `results.json`: both models, both selections, all taxonomies, per-clip counts,
  per-type metrics, and bootstrap intervals.
- `summary.csv`: flat table of aggregate metrics.
- `verification.json`: matching, aggregation, split, and integrity checks.

Preparation used:

```sh
align-model/runs/env/bin/python baselines/common/prepare_dataset.py \
  --set zip034=experiments/baselines_034_20260918/bundles \
  --out baselines/data/real_034_20260918 --real-test --workers 4
```

After inference completes, freeze each model in a separate process and score:

```sh
align-model/runs/env/bin/python experiments/baselines_034_20260918/evaluate.py freeze --model polytune
align-model/runs/env/bin/python experiments/baselines_034_20260918/evaluate.py freeze --model laddersym
align-model/runs/env/bin/python experiments/baselines_034_20260918/evaluate.py score
```

Freeze output directories are deliberately exclusive to avoid overwriting
predictions. Existing frozen outputs can be rescored using the final command.
The initial LadderSym launch rejected a wrapper argument before inference;
the corrected launch uses `eval.batch_size=4`. Both launch attempts are retained.
Decoder invalid-event warnings follow the existing native decoder's skip policy;
they do not cause any test clips to be dropped. The final integrity audit records
prediction coverage and log warning counts. Paper files are not modified.
