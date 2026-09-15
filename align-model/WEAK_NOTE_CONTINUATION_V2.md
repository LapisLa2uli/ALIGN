# Weak-note and replay-continuation candidate v2

Status: implemented and unit-validated, not promoted. The sealed 6,644-row
test split was not read. Existing `align-sparse-joint-v1` and
`align-end-to-end-joint-v1` checkpoints retain v1 candidate generation at
inference.

## Code paths and versions

- `transcription/basic_pitch.py` adds activation-map short-note rescue and
  continuity-aware same-pitch split merging. Frozen Basic Pitch keeps rescue
  disabled unless a serialized `BasicPitchDecodeConfig` enables it.
- `joint/candidates.py` names the new candidate stream
  `align-joint-candidates-v2-short-rescue`. New sparse and end-to-end
  checkpoints are `align-sparse-joint-v2` and
  `align-end-to-end-joint-v2`; the inference loader routes older checkpoints
  to the legacy candidate stream.
- `transcription/refiner_data.py` creates cached-map approximations of band
  attenuation, filtered/breathy timbre, low-energy short notes, false short
  onset noise, and artificial same-pitch splits. Positive short notes and an
  equal configurable number of hard negatives receive explicit loss weights.
- `joint/lattice.py` computes a soft, bounded continuation compatibility
  feature for repeat-enter/replay edges. It tolerates one skipped observed and
  one skipped score note. Divergent matching-prefix alternatives are duplicated
  as local-training hard negatives, while repeated nested entries receive a
  configurable fragmentation penalty.
- `stages/note_repetition_model.py` adds continuation and restart-gap features
  to Layer 1 v2. Its loader preserves 12-feature v1 checkpoints.
- `stages/repetition.py` consolidates overlapping or adjacent replay fragments.
  The rejected `contextual_continuation` hard filter remains experimental and
  unchanged; v2 uses soft evidence instead.

## Serialized parameters

`BasicPitchDecodeConfig` stores all rescue thresholds, duration limits,
confidence floor, merge gap, onset, frame/contour continuity, and boundary
window parameters.

`RefinerAugmentConfig` stores augmentation probabilities, attenuation ranges,
short-note duration/weight, hard-negative ratio/weight, breath-noise level, and
artificial-split probability. It is embedded in the refiner checkpoint's
`train_config`.

`LatticeConfig` stores continuation enablement, lookahead, candidate/score skip
budgets, pitch tolerance, soft-score weight, hard-negative copies, and repeat
fragment penalty. `NoteRepetitionModelConfig` stores its feature version and
continuation lookahead/skip budgets.

## Verification

Focused command:

```powershell
C:\Users\Hank\.conda\envs\MusicEval\python.exe -m pytest `
  tests/test_transcription_cleanup.py tests/test_basic_pitch_frontend.py `
  tests/test_joint_lattice.py tests/test_joint_end_to_end.py `
  tests/test_note_repetition_model.py tests/test_note_refiner.py -q
```

Result: 39 passed. Modified modules also passed `py_compile`,
`git diff --check`, and IDE lint diagnostics.

The tests cover weak short-note rescue, onset-only noise rejection,
weak-boundary fragment merging, strong rearticulation preservation,
one-error replay continuation, divergent continuation rejection, replay-span
consolidation, component gradients, augmentation balance, and config/checkpoint
round trips.

## Validation and promotion

No dataset ablation was launched because the protected full-path CPU run and
parallel end-to-end CPU/GPU run were active. This avoids resource contention.
After they finish, retrain into a new directory (for example
`runs/joint-audit-v2/end-to-end-v2/weak-note-continuation-v2`) and evaluate
only the 1,871-row `val` split from `data-audit/2026-09-14-v2/split.json`.

Compare note F1, predicted/target count ratio, short-note recall, same-pitch
split rate, and repeat precision/recall/F1 against the matching v1 validation
checkpoint. Promotion remains blocked until the validation metrics improve;
the locked test remains untouched until a candidate is selected.
