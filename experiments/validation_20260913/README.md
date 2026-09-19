# Shared validation experiment: execution status

**Polytune and prompted LadderSym training has been stopped on request.** Both use
the same audited 10,049 training / 1,116 validation split of the September 14
corpus. Full native validation F1 has been computed for their retained best
checkpoints, which completed 31 and 26 epochs. These cannot be described as a
strict 30-epoch experiment; see the [evaluation record](../baselines_f1_20260915/README.md).
ALIGN training remains pending. The measured baseline scores are in
[results.json](../baselines_f1_20260915/results.json), and the local draft contains
the native baseline table. No changes have been pushed to Overleaf. See the [baseline run record](../baselines_synth_20260914/README.md)
and `baselines/runs/start_20260914/training_status.json` for live progress.

The requested paper clone failed because this server has no usable Git
authentication for `git.overleaf.com`. A new corpus of 12,000 source bundles
has now been generated; see [DATA_GENERATION.md](DATA_GENERATION.md). An exact
baseline supervision export and renderer-timing audit retained 11,165 clips,
excluding 835 with ambiguous supervision or inconsistent rendered timing.
Full converted-label checks and both official loader checks passed before
training. A common comparison metric, ALIGN training and Basic Pitch caches
remain to be completed. An authenticated paper checkout remains needed.
Do not replace measured results with the
historical numbers in `align-model/README.md` or the original baseline papers.

## Prepared artifacts

- `experiments.draft.tex`: a provisional experiments section grounded in the
  current transcription-first implementation. It describes evaluation on a
  common validation set, without giving a reason for that choice. Comments
  identify run-dependent details that must be completed.
- `references.bib`: citations to the original baseline papers.
- `preview.tex` and the locally compiled `preview.pdf`: a two-page working
  preview, visibly marked as a working draft with baseline training in progress.
- A fix in `align-model/src/alignmodel/contextual_align_train.py`: requesting
  more training data than the frozen split contains now fails rather than
  borrowing validation examples. Missing maps, duplicate maps, overlapping
  splits, and insufficient validation samples also fail before training.

The draft is not an edit of the inaccessible Overleaf manuscript. Its dataset
description, table structure, citation keys, and training settings must be
reconciled with that manuscript and the actual run artifacts before publication.

## Environment and verification

Initial ALIGN checkout: `3cad0ddc36b08f85282f9f89ad4ae58a6b8f4100`.

| Environment | Location | Verification |
| --- | --- | --- |
| Polytune | `baselines/envs/polytune` | Dependency check; CUDA forward/backward; 13 regression tests passed, 1 LadderSym-only test skipped |
| LadderSym | `baselines/envs/laddersym` | Dependency check; CUDA forward/backward; all 14 regression tests passed with `BASELINE_FLAVOR=LadderSym` |
| ALIGN | `align-model/runs/env` | Dependency check; repetition scorer and contextual aligner CUDA forward/backward; 6 split integrity and 2 sequence-loss tests passed |

Python is 3.11.14. Both baselines use their pinned upstream revisions and
the existing repository patches. Their `doctor.py` checks confirmed an NVIDIA
RTX A5500, CUDA wheel 12.1, and bf16 support. ALIGN has PyTorch 2.3.0+cu121,
NumPy 1.26.4, Basic Pitch 0.4.0, and TensorFlow 2.15.0.post1 installed.
These environment checks were followed by full-budget baseline CUDA preflights,
optimizer updates, validation and strict checkpoint reloads. Both formal runs
have now passed startup and are updating parameters; full validation inference
has not yet run. Detailed evidence is under `baselines/runs/start_20260914/`.

Logs are under `baselines/runs/setup_20260913/`; baseline dependency freezes
are in each environment's `installed.freeze.txt`, and ALIGN's freeze is
`baselines/runs/setup_20260913/align.freeze.txt`. Generated environments, logs,
and PDF build products are ignored by Git.

## Dataset status and remaining comparison requirements

The historical audited `align_v1` manifest has 10,725 training and 594 validation
entries. Its SHA-256 is
`598faa8f2402e34b7b0395363d0596b3e1f24d7448fa76aa8e4bb41fc7281c88`.
These historical counts do not apply to the current run. The new shared split
has 10,049 training and 1,116 validation entries, with SHA-256
`b32edf500e7dece094801cccaf3495d072be4333232b0ea1afc9f7eaf6157b6c`.

1. The converted dataset is `baselines/data/align_synth_20260914`; original
   sources are `synth-pipeline/output_10k_multi` and `output_2k_rawdata`.
   Preserve the same training and validation membership for ALIGN. Baseline
   class labels, source composition and split overlap have been checked.
   The source-derived portion of the legacy dataset uses a clip split, so
   results on it cannot support claims about unseen pieces.
2. Baseline supervision compatibility is resolved by the audited
   `export_synth_supervision.py` exporter and renderer-timing audit. ALIGN
   training still requires exact `note_map.json` lineage. Its repetition/contextual loaders explicitly
   restrict rows to `procedural12k`. Do not rename raw-derived rows to bypass
   that check or silently substitute a new generated dataset. Resolve the
   original maps and raw-derived support before training ALIGN on this split.
   Only the converted baseline dataset is insufficient to train
   the current ALIGN method.
3. Confirm a common metric with the actual annotation schema. Baseline
   `evaluate_notes.py` measures native extra/missing/correct note F1, while
   ALIGN's `eval_melodies.py` measures score-melody region F1. Their numeric
   values must not share an unlabeled F1 comparison column. The draft's
   proposed common error-region comparison uses the existing
   `eval_bridge.py` note-to-region adapter, but that adapter's wall-clock
   onset metric still needs validation against the supplied labels. Schema
   1.2 melody windows are not necessarily wall-clock fault onsets. If using
   melody-set F1 for the shared comparison, implement and verify a
   score-mapping adapter for baseline predictions first, without using gold
   maps at inference. Do not treat time-only predictions as aligned score
   positions across repeats or tempo changes.
4. The second baseline is prompted LadderSym. Native baseline outputs
   contain correct, extra, and missing notes; derived wrong-note/repetition
   regions must be explicitly described as adapter outputs. Report rhythm
   separately and keep intonation disabled for the current ALIGN method.
5. Keep evaluated IDs and full prediction coverage, all final configurations,
   calibration settings, initialization source, completed epochs, selected
   checkpoint hashes, and per-example metrics. Select and calibrate using
   validation only. Any inference failures must be resolved or disclosed,
   rather than silently removing examples.

The existing [GPU runbook](../../baselines/docs/GPU_RUNBOOK.md) contains
baseline training commands. Use 40 epochs as its initial experiment budget,
with the full 1,024-token event/prompt settings, and pass `--split validation`
explicitly to both evaluation wrappers. The running models use one clip and
one segment per batch, with 16 gradient-accumulation steps and bf16 precision.
Do not start a truncated or synthetic smoke run and report it as this experiment.

After data compatibility is resolved, ALIGN's current training sequence is
`train_note_repetition.py`, `train_contextual_note_aligner.py`, Basic Pitch
feature caching, and `train_cached_contextual_aligner.py`, followed by
validation calibration and full pipeline evaluation. Request only sample
counts actually present in the immutable training split. The existing
`eval_note_first_pipeline.py` defaults to 20 samples and `test_id`; explicitly
use `--split val --max-samples 0` when running full validation, with the
correctly reconciled manifest. Inspect loaded model components so that a
missing decoder or aligner cannot silently select a different pipeline.

## Paper sources and implementation references

- [Polytune paper](https://arxiv.org/html/2501.02030v1), sections 3 and 4.
- [LadderSym paper](https://arxiv.org/html/2510.08580v2), sections 3, A.4, A.5.
- `align-model/src/alignmodel/pipeline.py`: stage prerequisites and output.
- `align-model/src/alignmodel/stages/contextual_note_aligner.py`: GRU architecture.
- `align-model/src/alignmodel/stages/note_repetition_model.py`: repetition scorer.
- `align-model/src/alignmodel/stages/rhythm.py`: active note-duration detector.
- `align-model/src/alignmodel/cached_alignment_train.py`: structured fine-tuning.
- `align-model/src/alignmodel/eval_melodies.py` and `DataCreate/src/datacreate/melody.py`: official melody metric.
- `baselines/common/evaluate_notes.py` and `eval_bridge.py`: native note metrics and legacy region adapter.

To rebuild the draft, run `pdflatex preview.tex`, `bibtex preview`, then
`pdflatex preview.tex` twice in this directory.
