# V10: protect adjacent identical notes without changing other mappings

Runtime revision: `v10-boundary-v1`. Candidate:
`runs/stack-v10/CANDIDATE_STACK_V10.json`. This is a separate experimental version;
v9's frozen candidate and acoustic checkpoint are unchanged. V10 uses the current
passage runtime (v9-passage-v2, including its earliest-equivalent-passage policy).

## Behavior

1. Run the existing v9 transcriber, decoder, same-pitch repair and bounded passage
   aligner. Retain that result as the fallback.
2. Look only at v9-merged groups that map across an equal number of consecutive,
   distinct, same-pitch canonical score notes. Require primary CTC candidates
   with confidence at least 0.5. Tied score notes are already one canonical note.
3. A small random forest votes on repeat evidence using the existing waveform
   gap depth/duration, energy variation, spectral step/drift, attack and voiced
   measurements. Restore a decoded boundary only with both structural support
   and a repeat vote >=0.5. No new onset is invented; no different-pitch candidate
   is removed. V9's 10 ms gap veto remains in force.
4. Realign the proposed transcription. Accept it only if the restored candidates
   map to separate consecutive score notes and all mappings outside the affected
   groups, outside deletions, and outside error-label identities/copy counts are
   unchanged. Otherwise return the original v9 result, recording the reason.

This is intentionally conservative: it cannot recover a repeat that the CTC
decoder never emitted, a group with unequal candidate/score counts, or a genuine
extra rearticulation over a single notated note. It may still mistake an artifact
for an attack when a sustained sound spans multiple repeated score notes. The
outside-region guard preserves other outputs, not correctness within the region.

## Data and training limits

E: was disconnected during this task, confirmed outside the sandbox. The old
raw acoustic evaluation caches were also unavailable. Fresh training on E: and
the full paired synthetic end-to-end evaluation could not run.

The boundary forest was fitted on **442 usable archived acoustic control rows**
from historical synthetic development recordings, split by sample (within each
family: fit/fit/selection/check). The original controls contain 543 genuine
same-pitch boundaries and 316 injected midpoint splits. No DataCreate labels or
sealed test data selected parameters. This is reused development material,
including controls previously used to calibrate v9; it is not independent proof
of generalization. The check subset was also inspected during exploratory
threshold comparisons, so it must not be described as a blind test. The fixed
0.5 class vote is not a calibrated probability.
The forest uses 96 trees, depth <=5, minimum leaf size 3, seed 20261005. Its JSON
inference matches sklearn probabilities; runtime does not require sklearn.

An archived-feature replay with *assumed correct structural support* reduced
false merging of repeat controls from 457/543 to 65/543 while retaining all
316/316 single-note split repairs. This is a conditional repair-rule diagnostic,
not fresh waveform or end-to-end accuracy. In the adverse case where a false
split is mapped over two score notes, 57/316 controls receive a repeat vote:
the structural assumption matters. Acoustic-only check-subset votes recover
91/117 repeats but also vote repeat on 18/76 artificial splits. These figures
explain why the acoustic classifier is guarded rather than deployed alone.

## Fresh DataCreate comparison — 2026-10-05

Both versions used the same freshly computed acoustic outputs and same passage
aligner on 001–095. Labels >=095 were never used as accuracy truth. Evaluation
artifacts are in `runs/stack-v10/predictions`, `paired` and `evaluation.json`.

- **57 boundaries restored in 33 takes.** Four proposed refinements fell back
  because they failed the protected-output or restored-mapping checks.
- **007:** 260 -> 261 candidates; retains the single-note repair at 20.550 s,
  restores the boundary at 27.133 s between canonical score notes 179 and 180
  (zero-based). All other mappings and all eight error labels are retained.
- **095:** unchanged; both repetition labels and both wrong-note labels remain.
- Error types, canonical note sets and repetition counts are identical across
  all 95 takes. Playback ranges differ on 071 and 073; their note identities do
  not change. Scores, audio and candidate-source hashes were verified unchanged.
- Against the 25 reviewed takes in 001–094, reference-note content-error and
  repetition metrics are unchanged. These sparse labels do not measure every
  repeated-note boundary; restored boundaries require human/audio verification.
- Average refinement overhead was about 0.415 s in the paired warm-process run;
  007 added 1.109 s. Clips needing refinement perform a second bounded alignment.
- The 3,934-note full-score CLI smoke test completes with status ok, three labels,
  and two accepted restorations. This is a functional check, not accuracy truth.
- **76 focused tests pass** in MusicEval, including gap preservation, fast-note
  invariance, single-note merging, repeated-note identity recovery, rollback,
  version dispatch, passage behavior and UI label contracts.

These results support improved handling of already-decoded same-pitch boundaries
with preserved outside output. They do not establish a general no-regression
accuracy guarantee. Full synthetic reference-note F1 and fast/repeat/single-note
stratified evaluation remain outstanding until E: is available.

## Run and review

```powershell
& 'C:\Users\Hank\.conda\envs\MusicEval\python.exe' `
  align-model/scripts/run_stack_v10.py `
  --score DataCreate/samples/007/verified_score.musicxml `
  --audio DataCreate/samples/007/performance_audio.wav `
  --output align-model/runs/stack-v10/example-007.json
```

V10 has its own DataCreate configuration. From `DataCreate`:

```powershell
datacreate --config config/v10.yaml serve --port 8767
```

The separate v10 server was started at http://127.0.0.1:8767/. Sample 007 was
published with backups under `DataCreate/work/v10-ui-007-20261005`, and its live
Re-label action confirmed `align_stack_v10`. Re-label on an older sample under
this configuration regenerates it as v10; the live 095 action verified that
regeneration and retained all four labels. The default config remains v9.
Both GUIs share the sample directory, so the most recently generated version
is the one displayed there; using the v9 Re-label action regenerates v9 output.
Human labels remain untouched.

Training and inspection commands:
`scripts/train_boundary_v10.py`, `scripts/check_boundary_v10_controls.py`, and
`scripts/evaluate_stack_v10.py`. The first explicitly uses archived features,
not E: recordings. Candidate hashes protect the classifier and versioned sources.
