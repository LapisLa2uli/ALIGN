# Native baseline F1 on the shared validation set

**Completed:** both models have predictions and metrics for all 1,116 validation
clips. Values below are micro-F1 percentages; overall retains class identity.

| Model | Extra | Missing | Correct | Overall |
| --- | ---: | ---: | ---: | ---: |
| Polytune | 17.15 | 0.94 | 18.41 | 16.99 |
| Prompted LadderSym | 7.79 | 2.26 | 16.15 | 11.66 |

Machine-readable results are in [results.json](results.json). Both models use
identical validation IDs and reference counts. Overall counts unclassified notes
as false positives: Polytune has none; LadderSym has five in one clip. These
notes remain in the saved MIDI, and no class is inferred from track position.

Both training jobs were stopped on September 15. Evaluation uses the existing
best checkpoints selected by minimum validation loss, with exact epoch metadata:

| Model | Completed epochs in selected checkpoint | Global step | Validation loss |
| --- | ---: | ---: | ---: |
| Polytune | 31 (`epoch=30`) | 19,499 | 1.0822654963 |
| Prompted LadderSym | 26 (`epoch=25`) | 16,354 | 1.0683450699 |

The original learning-rate schedule had a 40-epoch budget. Polytune was stopped
after 34 complete epochs and LadderSym during its 29th epoch, with 28 complete
epochs in its last saved checkpoint. The user confirmed evaluation of the
existing best checkpoints, with their actual 31/26 epochs reported. No checkpoint
epoch fields have been modified, and this is not a strict 30-epoch comparison.

## Data and inference

All **1,116 validation clips** from `baselines/data/align_synth_20260914/split.json`
are evaluated. Original split SHA-256:
`b32edf500e7dece094801cccaf3495d072be4333232b0ea1afc9f7eaf6157b6c`.
The original split remains unchanged. Duration-balanced shards are disjoint,
and their predictions must cover every original validation ID exactly once.
Shard split files only support the upstream entry points' internal split naming.

The inference handlers load all checkpoint weights strictly and use float32,
greedy decoding and a 1,024-token generation budget. Polytune uses the original
generation path. LadderSym uses the corrected start-token handling and verified
KV cache described below.
Prompted LadderSym uses the 1,024-token symbolic prompt budget from training,
with deterministic prompt order. Four segments are batched per generation call.
Two actual validation clips per model passed inference and native scoring before
the complete run. These preflight scores are not the full-validation results.

`baselines/scripts/evaluate_validation.py` launches the official evaluation
wrappers in independent shards, records process IDs and progress, audits merged
prediction coverage, and runs `baselines/common/evaluate_notes.py` over the full
validation set. The wrappers' `--native-only` option avoids the separate,
unvalidated score-region bridge. Model weights and note-matching rules are
preserved; the LadderSym generation corrections are described below.

## LadderSym generation correction

The prompted training decoder concatenates the padded prompt with right-shifted
targets, whose first position is an unmasked start token. The original generate
method instead predicted from the final masked prompt-padding position. An
actual-checkpoint diagnostic confirmed a 1,024-position generation prefix versus
the 1,025-position prefix used in training. Generation now explicitly appends
the same start token and mask as training. No weights or training labels change.
The before/after diagnostic and regression test verify the input/mask equality;
this is a correctness fix, not checkpoint selection by F1.

An optional KV cache avoids recomputing the full padded prompt at each token.
The decoder preserves positional offsets and only applies bidirectional prompt
attention in the initial pass. A small decoder test compares cached states with
full-prefix states through three successive tokens. Actual-checkpoint greedy
tokens also match, and two full validation-clip outputs are byte-identical MIDI
files with identical metrics, comparing cached versus uncached corrected
generation. All 17 final LadderSym regressions passed, including invalid-class
scoring; Polytune passed 14, with three LadderSym-only tests skipped. This run enables the cache with
`LADDERSYM_USE_CACHE=1`; the default remains off for other callers.

Initial LadderSym predictions made before the start-token fix are retained as
diagnostics and excluded from the final result. Polytune predictions are reused
unchanged after checking checkpoint/split identity and full coverage.

## Metrics and artifacts

Report extra, missing and correct note precision, recall and F1 separately.
Matching uses `mir_eval`, with 50 ms onset tolerance, 50 cents pitch tolerance,
and no offset constraint. Classes are read from MIDI track names. Micro-F1 pools
matched/predicted/reference counts across all clips; per-clip mean F1 is also
saved. A class-agnostic transcription score is recorded separately and must not
be called error-detection F1. These metrics are not ALIGN's score-melody F1.

The decoder's initial error class is 0, outside the valid 1--3 vocabulary. If
notes are generated before a class token, they produce an unnamed track.
`evaluate_notes.py --allow-unclassified` keeps such notes in an explicit invalid
bucket: they receive no class-match credit and count as false positives in
overall class-aware F1. The class-agnostic diagnostic includes their pitches and
times. Unknown nonempty track names still fail validation. The scorer never
assigns classes by track order or silently discards unnamed notes.

Final run directory: `baselines/runs/evaluate_20260915/validation_f1_20260915`.

After Polytune completed, its GPUs were released to LadderSym. The remaining
871 LadderSym clips were repartitioned across six GPUs; 245 completed corrected
predictions were verified and reused. Cached predictions and newly inferred
shards must remain disjoint and together cover all 1,116 validation IDs.

All MIDI inference completed. Editing wrapper comments while the running shells
waited for Python caused a post-inference shell error; original exit codes and
logs remain preserved. Native scoring was rerun directly on the complete merged
predictions and passed all coverage/count checks. The supervisor now snapshots
its shell wrappers before launch to prevent this failure in future runs.

- `evaluation_manifest.json`: exact selection, split, inference settings and commands.
- `evaluation_status.json`: live process state and per-shard prediction counts.
- `results.json`: full-set summary, created only after all predictions and metrics pass.
- `{polytune,laddersym}_note_metrics.json`: pooled and per-clip metrics.
- `{polytune,laddersym}_predictions`: complete prediction collections.
- Individual shard logs and split files remain for reproducibility.

Checkpoint hashes and the epoch-budget qualification are recorded in
`baselines/runs/evaluate_20260915/confirmed_checkpoint_selection.json`. Original training
outputs and source data are preserved. No Overleaf manuscript has been updated.
