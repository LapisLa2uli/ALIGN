# Baseline training on the September 14 synthetic corpus

This run trains Polytune and prompted LadderSym from scratch on the same
reviewed subset of the newly generated 12,000-clip clarinet corpus. The original
source bundles are `synth-pipeline/output_10k_multi` (10,000 procedural clips)
and `synth-pipeline/output_2k_rawdata` (2,000 existing-score snippets).

**September 15 update:** Both training processes were stopped at the user's
request. Polytune's last checkpoint completed 34 epochs; LadderSym's last
checkpoint completed 28 epochs, and its interrupted next epoch is not retained.
The surviving best checkpoints completed **31 and 26 epochs**, respectively.
The user confirmed evaluating the existing best checkpoints and recording their
actual epochs. This run is not a strict 30-epoch comparison. Full-validation
native F1 is complete for these existing best checkpoints.
See [the evaluation record](../baselines_f1_20260915/README.md).

The shared converted dataset is `baselines/data/align_synth_20260914`.
`split.json` assigns 90% of accepted clips to training and 10% to validation,
with seed 365 and stratification between the procedural and raw-derived sets.
Membership is fixed before either full training run starts. This is a clip
split within the sampled score collection, not evidence of generalization
to unseen pieces. The legacy `align_v1` split and its exclusion counts do not
apply to this corpus.

The final split contains **10,049 training clips and 1,116 validation clips**.
Its SHA-256 is
`b32edf500e7dece094801cccaf3495d072be4333232b0ea1afc9f7eaf6157b6c`.
All 11,165 converted bundles passed the header/layout audit and an independent
cross-check of exported label-MIDI counts, pitches and onsets against the
source MIDI. Eight examples from each full validation loader also passed
token-to-label checks.

## Supervision export

`baselines/common/synth_supervision.py` replays the exact per-sample seed and
worker-local index from the original generator configuration. During replay,
an injected rest retains the deleted note's clean identity through subsequent
edits and repeated copies. The exporter verifies that all clean/performed note
lineage matches the original `note_map.json` and that regenerated reference and
performance MIDI events and tempo maps match the originals. It does not render
new audio.

Every reference and performance MIDI event must correspond to a complete,
contiguous group of notated notes at matching quarter-note offsets and pitch.
Unrendered notes, partially changed ties, and inconsistent missing-rest lineage
are excluded with explicit reasons. The three baseline classes are:

- Correct: first-pass performed events matching the reference pitch.
- Extra: inserted events, substituted pitches, and all replay copies.
- Missing: expected pitches for substitutions, at the actual first-pass
  performed event time; or deleted notes, at the tagged rest's location under
  the final performance MIDI tempo map. Repeated missing notes are not counted
  again.

Timing never comes from dense DTW costs, interpolation across played notes,
or the padded error-region windows. Original audio, scores, maps and error
labels remain unchanged. `baselines/data/synth_20260914_targets` contains the
exported `note_labels.json` files, links to the source artifacts, source/config/
implementation hashes, and `supervision_export.json` with every exclusion.
The historical mapping-warning count alone is not used as an exclusion rule;
the exporter independently reconstructs and verifies correspondence.

A second audit replays the actual `tinysoundfont` MIDI event parser used for
rendering and requires every reference/performance note's pitch and timing to
agree with MIDI-derived supervision within 3 ms. It also rejects targets beyond
the available padded audio length. This caught tempo changes interpreted
differently by the renderer; note-file integrity alone cannot detect this issue.
The first audit accepted 11,907 and excluded 93 bundles. The rendering audit
excluded another 742 (657 procedural, 85 raw-derived). The final data contains
**11,165 clips: 9,325 procedural and 1,840 raw-derived**, with all original
12,000 bundles retained unchanged. Both models use the identical final subset.
See `baselines/data/synth_20260914_reviewed/render_timing_audit.json`.

Commands from the ALIGN root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
align-model/runs/env/bin/python baselines/scripts/export_synth_supervision.py \
  --set procedural12k=synth-pipeline/output_10k_multi \
  --config procedural12k=synth-pipeline/config/multi_error_10k.yaml \
  --set rawdata2k=synth-pipeline/output_2k_rawdata \
  --config rawdata2k=synth-pipeline/config/rawdata_snippets_2k.yaml \
  --out baselines/data/synth_20260914_targets --workers 8

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
align-model/runs/env/bin/python baselines/scripts/audit_synth_render_timing.py \
  --targets baselines/data/synth_20260914_targets \
  --out baselines/data/synth_20260914_reviewed --workers 8

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
baselines/envs/polytune/bin/python baselines/common/prepare_dataset.py \
  --set procedural12k=baselines/data/synth_20260914_reviewed/procedural12k \
  --set rawdata2k=baselines/data/synth_20260914_reviewed/rawdata2k \
  --out baselines/data/align_synth_20260914 \
  --workers 8 --val-frac 0.1 --test-frac 0 --seed 365
```

Conversion resamples both WAVs to 16 kHz mono PCM16, pads them to equal length
for the official loaders, retains reference MIDI on its own timeline, and
exports the three note-class MIDIs. This is the existing baseline input
convention. It does not time-warp reference audio to the performance.

Because other jobs were rapidly consuming the shared disk, the converted
`mistake/` and `score/` directories point to
`/dev/shm/weixi_align_baselines_20260914`. This uses approximately 31 GB of the
host's available memory instead of duplicating audio on disk. Original bundles,
all three-class targets, split/manifest files, logs and training checkpoints are
stored durably under the project. The audio cache is temporary and is lost on
host/container restart. Recreate its `mistake` and `score` directories and rerun
the conversion command to rebuild it from the original sources; verify the
recorded split SHA-256 before resuming any checkpoint. The waveform format and
training inputs are identical to ordinary disk-backed conversion.

## Training settings and verification

| Setting | Polytune | Prompted LadderSym |
| --- | --- | --- |
| Initialization | Random, seed 365 | Random, seed 365 |
| Epoch budget | 40 | 40 |
| Precision | bf16 mixed | bf16 mixed |
| Clips per batch | 1 | 1 |
| Segments per clip per batch | 1 | 1 |
| Gradient accumulation | 16 | 16 |
| Event token budget | 1,024 | 1,024 |
| Symbolic prompt budget | None | 1,024 |
| Peak learning rate | 0.00002 | 0.00002 |
| Warmup | 4,000 optimizer steps | 4,000 optimizer steps |
| Checkpoint selection | Validation loss | Validation loss |
| Retained checkpoints | Best and last | Best and last |

Both models use the pinned upstream revisions and existing integration patches
documented in `baselines/README.md`. Scheduler step counts come from Lightning's
actual optimizer updates. Except for a possible final partial accumulation,
both models accumulate 16 clips/segments per optimizer update. These are new
full training runs; no two-step preflight weights are used for initialization.

Before full training, both official loaders decoded targets back to the
expected class/pitch events for eight actual source items. CUDA preflights used
the full event and prompt budgets, performed optimizer updates and validation,
saved Lightning checkpoints, and reloaded all weight tensors strictly. Model
weights and losses were finite. Full-budget memory probes measured 4,410 MiB
reserved for Polytune and 6,032 MiB for prompted LadderSym with batch size one.
The final runs use one clip per batch and 16 accumulation steps to accommodate
other GPU users. The supervisor requires another 2 GiB above the measured
reservation before launch. Three new supervision tests cover exact missing-note times across
a tempo change, repeats/substitutions, and rejection of ambiguous/missing
lineage. The existing baseline regression suites passed in both environments.
Preflight model files were removed after strict reload checks to reclaim disk;
their logs and hashes remain under `baselines/runs/start_20260914`.

## Live artifacts

Both formal training runs are now stopped. Their original GPUs were 1 and 5.
`training_status.json` records the stop; `checkpoint_inventory_at_stop.json`
records the retained checkpoint epochs, steps, validation losses and hashes.

The run directory is `baselines/runs/start_20260914`:

- `run_manifest.json`: final dataset counts, split hash, configurations and provenance.
- `training_status.json`: supervisor and training process IDs, GPU assignments,
  current epoch/batch progress, losses, and free disk space.
- `polytune_training.log` and `laddersym_training.log`: full training stdout.
- `supervisor.log`: process state and disk monitoring.
- `supervision_export.log`, `data_conversion.log`, loader and checkpoint
  verification logs: preparation evidence.

Training outputs are under `baselines/runs/{polytune,laddersym}/synth_20260914_s365`.
During training, the supervisor ran independently of the terminal, started each model only when
its preferred GPU has sufficient free memory (or selects another available GPU), and pauses its own process
groups below 30 GiB free disk space and resumed at 40 GiB. A lock prevents a
second supervisor for the same run, and the recorded split hash is checked
before launch. Existing run directories require explicit checkpoint resumption.

Epoch-level validation losses select checkpoints. Full validation prediction
and note-F1 evaluation follow training; startup losses and preflight results
are not final validation scores. No Overleaf results have been published.
