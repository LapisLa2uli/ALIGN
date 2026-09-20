# ALIGN: Polytune and LadderSym Baselines

This integration uses the authors' implementations to train, run inference, and evaluate three note classes on ALIGN's frozen synthetic data: Extra, Missing, and Correct. LadderSym supports both prompted (with reference MIDI as an additional input) and unprompted settings.

**Start with the [GPU server runbook](docs/GPU_RUNBOOK.md)** for code and data transfer, installation, smoke tests, full training, checkpoint resumption, evaluation, GPU memory issues, and troubleshooting.

ALIGN comparison F1 is official note-wise score-event identity (`official_note_wise` in `note_metrics.json`), produced by mapping Extra/Missing/Correct notes onto canonical score locations. The authors' mir_eval onset 50 ms / 50 cents protocol remains available as `legacy_mir_eval_onset_50ms` and is not the ALIGN headline.

For the corrected 40-clip real recording test set, see the [real-data F1 calculation and workflow](docs/REAL_DATA_EVALUATION.md): error-event conversion, typed time matching, micro-F1, current retrained results, rescoring commands, and interval-merging analysis.

The [reproduction audit](docs/REPRODUCTION_AUDIT.md) documents the fixes, test evidence, differences from the authors' experiments, and remaining validation gaps. This work integrates the baselines with ALIGN data. Passing smoke tests or retraining on ALIGN does not establish reproduction of the papers' reported scores.

## Pinned Versions

| Project | Pinned code version | Usage |
|---|---|---|
| ALIGN | Upstream base `2424130` | The baseline integration is committed to the main repository; record the actual checkout commit for each run. |
| [Polytune](https://github.com/ben2002chou/Polytune) | `d2055bb21759d457c8f21c1cf2e47c79af6248f5` + patches | `scripts/polytune_train.sh` / `scripts/polytune_eval.sh` |
| [LadderSym](https://github.com/ben2002chou/LadderSym) | `381179754cf6bcb435f9decf0d5e24eada6c68ec` + patches | `scripts/laddersym_train.sh` / `scripts/laddersym_eval.sh` |

Keep the authors' repositories at the pinned commits. **Do not run `git pull` inside the nested repositories**, as the patches may no longer apply. `scripts/bootstrap.sh` applies `patches/*.patch` to clean clones and copies `configs/`. Repeated runs detect already applied patches; conflicts stop the setup.

## Repository Layout

- `scripts/`: Setup, training, evaluation, and transfer entry points that work from any working directory.
- `common/`: Data conversion, validation, strict checkpoint loading, and evaluation that matches notes by class name.
- `configs/`: Versioned ALIGN configurations copied into the authors' repositories by bootstrap.
- `patches/`: Complete modifications to the authors' repositories and their documentation, including inference fixes.
- `envs/requirements.*.server.txt` and `*.server.lock.txt`: Server dependencies and version constraints for Python 3.11.
- `tests/`: Regression tests for decoding, checkpoints, labels, and evaluation.
- `splits/align_v1/`: The preserved original split, the audited split, and reasons for the 86 excluded samples.
- `Polytune/`, `LadderSym/`, and `envs/{polytune,laddersym}/`: Rebuildable local directories ignored by Git.
- `data/` and `runs/`: Local datasets and results, ignored by Git and transferred separately when migrating.

## Keep the Dataset Version Fixed

The existing local `data/align_v1` dataset originally contains 12,000 tracks labeled with the legacy `note_labels.json` schema. A full audit of supervision semantics excluded 86 samples with issues such as unmapped reference notes or inconsistent note counts, retaining **11,914 tracks: train/validation/test = 10,725/594/595**. The original data and `split.json` remain unchanged; training and evaluation prefer the added `split.audited.json` by default. This frozen dataset can be used for training independently of the current generator source. Transfer and use it first; running the baselines does not require restoring the old generator changes.

The upstream generator at the base revision above produces `note_map.json`, whose semantics differ from `note_labels.json`. In particular, it does not directly provide complete missing-note timings on the performance timeline. The converter rejects this input unless it has first passed a validated supervision export. The September 14 corpus uses the exporter described below; the frozen legacy data already contains its labels.

For the newly generated September 14 corpus, `scripts/export_synth_supervision.py`
now provides a separate, audited export. It replays the original seeded generator,
tracks deleted notes as tagged rests through subsequent edits and repetitions,
and verifies the recreated clean/performed lineage and both original MIDI files.
It rejects ambiguous ties or incomplete correspondence. Accepted targets are
written in a separate directory; the original bundles remain unchanged.
See [the new training run](../experiments/baselines_synth_20260914/README.md) for
commands, the shared split, exclusions, settings, and live status. This new
dataset does not replace the frozen legacy `align_v1` corpus or inherit its audit.

The raw portion of `data/align_v1` contains clips drawn from a small set of shared scores and uses a per-clip split. It does not support claims of generalization to unseen pieces. For experiments across pieces, create a new dataset version with `prepare_dataset.py --split-by-source raw` and use the same split for every model being compared. This option raises an error when fewer than three known sources are available.

Historical Mac logs are described in the [legacy notes](docs/LEGACY_20260909.md); they are not evidence for the current validation results.
