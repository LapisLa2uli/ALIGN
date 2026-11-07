# Active campaign continuation notes

This is an operational snapshot, not the final assessment. Read `progress.json`
and the supervisor logs for current progress. Do not repeat completed work.

## Request and evaluation

Implement improvements following the previous real-data fine-tune regression,
and run PolyTune and LadderSym at **10, 20, 40 recordings only**. Finish actual
training, validation, DataCreate 100–120 inference, scoring, reporting, and
integrity checks. Two unchanged-checkpoint controls use the corrected frontend
and adapter. The user most recently asked for a time estimate; we answered
**2–3 hours remaining**, then continued the original work. This was not a stop.

Primary metric is canonical reference-score note identity and error type, not
timestamps. Teacher labels are AudioEval v9 + MuScriptor medium pseudo-gold.
Infer all 21 test recordings; score 17 (76 labels). Teacher abstentions are
101, 105, 114, 117. This is the previously examined benchmark, not a new holdout.
Preserve old experiments, checkpoints, human labels, app defaults, unrelated
working changes. No subagents, no automation needed. No AGENTS.md found.

## Files and environment

- Run: `experiments/baselines_real_teacher_v2_20261105/`.
- Previous reusable experiment: `experiments/baselines_real_teacher_20261105/`.
- Use `PYTHONNOUSERSITE=1` and `baselines/envs/polytune/Scripts/python.exe` for
  all baseline jobs. This venv inherits MusicEval's torch but has pinned baseline
  dependencies. Do not install conflicting packages into MusicEval.
- One GPU job at a time. RTX 2000 Ada laptop 8GB.
- Relative shell paths and patches work. The UI exposes C:/Users/chloe/ALIGN,
  but the actual Windows filesystem path for Start-Process is
  `D:\stuff\Audio Evaluation\ALIGN`. Literal C: paths may not exist even when
  tool outputs display them. Python `Path.resolve()` uses D:.

## Implemented and verified

- `reference_alignment.py`: label-free CQT/chroma Viterbi with costly backward
  and forward jumps; retrieves 4.096s reference context per 2.048s performance
  window. Same reference crop for audio and LadderSym MIDI prompts. All 73 maps
  and training/inference feature caches are complete.
- `features.py`, `prepare_features.py`: native mel compression factor 2 retained.
  Shared train/inference frontend. Training performance tensor parity <1e-5.
- `baselines/common/canonical_adapter_v2.py` and
  `baselines/scripts/label_datacreate_baselines_v2.py`: add canonical neighbors
  for ordinary extra notes; do not bridge backwards/copy boundaries.
- `lora.py`, `engine.py`: decoder q/v rank4 alpha8 dropout.05, 114,688 trainable
  parameters per model. All original weights frozen. AdamW LR1e-4, 300 updates,
  accumulation4, weighted native CE + .2 KL to base on teacher-forced prefixes.
  Half error windows, half clean; balance recordings within strata.
- Eight meaningful tests passed (`tests.log`). Both models passed real
  forward/backward/save/load/inference smoke tests, initial-logit equality,
  nonzero adapter updates, frozen base equality. LadderSym generation must
  replace prompt padding -100 with pad ID; this is fixed in `engine.py`.
- Original `protocol.json` frozen, SHA256
  `e39ca7a795cf8a67ca2f9a34e5db4df6c5b889266bdf430fd0e55a67c3346437`.
  Do not edit frozen source files unnecessarily. Protocol selection: full
  generation on validation12, candidates0/100/200/300; highest strict F1 among
  candidates with capped-window rate <= baseline+.005; ties earlier step.
  Step0 is an explicit unchanged-model fallback, not a trained improvement.
- Context audit: N40 outside-reference note occurrences 50.08% -> 8.92%; val
  41.16% -> 7.25%, only XML/MIDI-unambiguous scores. Test labels not used.
- Splits train40 / validation12 / test21 have no exact audio or XML hashes or
  available source filenames in common. Metadata incomplete; composition-level
  independence is not proved. `split_audit.json`, `context_audit.json` saved.
- Training exposure JSON: unique windows247/386/548 of available285/551/1077
  for N10/20/40; every trial draws600 error +600 clean windows.

## Verified inference optimization

PolyTune's native generate ignores use_cache and recomputes full prefix. Added
`cached_decode.py`, `engine_cached.py`, and `run_campaign_cached.py` as a separate
runtime-only amendment. `verify_cached_decode.py` compared **every token on 287
saved validation windows** from unchanged PolyTune and N10-step100. **All exact**,
including length-capped outputs. 958.688s ->493.858s (~1.94x). Results saved in
`cache_parity.json` and source hashes in `runtime_amendment.json`, before any
test generation. Training/settings/selection/scoring unchanged. Original frozen
files remain untouched. Final wrapper appends runtime note to RESULTS.md and
verifies amendment hashes.

## Running supervisor (do not launch a duplicate)

The initial tool-bound process was interrupted by a tool-host reset. Resumed
via hidden Windows Start-Process, which survives tool-session resets:

```
baselines/envs/polytune/Scripts/python.exe experiments/baselines_real_teacher_v2_20261105/run_campaign_cached.py
```

Background supervisor launcher PID14188, Python child PID8308 at launch; check
CIM command lines to find current processes. **Do not stop DataCreate serve
PID13796**, which is unrelated. Current stdout:
`campaign_resume_20261106.log`; stderr: `campaign_resume_20261106.err.log`.
Old interrupted logs `campaign.log` and `campaign_cached.log` are retained.
Per-job logs are `logs/<stage>.log`; `progress.json` names current stage.

At this snapshot: PolyTune N10 and N20 have completed training. N10 all three
validation passes complete; N20-step100 complete; N20-step200 currently finishing
recording039 (9/12). N20-step300, PolyTune N40, all LadderSym jobs, all test jobs,
scoring, final audit still remain. The supervisor handles all automatically.

Completed validation results so far (not test results):

| PolyTune | Strict F1 | Capped windows /173 |
|---|---:|---:|
| Unchanged corrected control |0.1765%|17|
| N10 step100 |0.2535%|26|
| N10 step200 |0.2857%|25|
| N10 step300 |0%|31|
| N20 step100 |0.2892%|23|

All N10 candidates fail the stability guard; selected N10 will be step0.
N20-step100 also fails. User has been informed candidly that input coverage
improved but the decoder's output instability remains. Do not claim success
from lower training loss. N10 train took147s; N20 roughly190s. Validation is
slow because long repetitive outputs consume token budgets. Keep monitoring,
use waits <=60s, and send concise meaningful progress updates.

## Finish

`run_campaign_cached.py` sequentially completes remaining training/validation,
writes `selection.json` before test inference, decodes controls and selected
models, then runs `report_results.py` and `audit_final.py`. For step0 fallbacks,
the control predictions are reused and clearly reported as fallback.

`report_results.py` produces RESULTS.md, results.json, learning_curve.png with
strict P/R/F1, per-type/sample counts, paired2000-recording bootstrap intervals,
and old outputs rescored under v2 adapter to separate adapter-only effects.
`audit_final.py` checks all original/frozen source/input/output/checkpoint hashes,
human labels, six trainings, output counts, and no-gold generation. Wrapper then
adds runtime amendment verification to final_audit.json.

After supervisor completes: inspect actual metrics, selection, final audit and
the plot; investigate/fix any execution/reporting failure without changing the
frozen experiment's decisions. Add concise interpretation of actual remaining
failure modes if needed. Open RESULTS.md in Codex and provide a self-contained
final response with the 10/20/40 F1 table, controls, changes, test limitations,
and report link. Do not stop after merely launching jobs. No app-default promotion
has been made, and a negative or mixed result must be reported honestly.

Additional post-hoc reporting helpers added after protocol freeze (no inference
or scoring changes): run `analyze_outputs.py --phase both` and
`report_interpretation.py` after final_audit passes. They generate output-count
diagnostics and ANALYSIS.md including every validation candidate, especially
important when a trial falls back to step0. They are not automatically called
by the frozen supervisor, so finish these manually.

Important scope completion: before any test generation, a supplemental actual-
trained assessment was declared in `trained_assessment_protocol.json` using
`assess_trained.py freeze`. It ensures every N10/20/40 fine-tune is actually
tested, even if the original stability-gated selection falls back to step0.
Rule: highest validation strict F1 among trained steps100/200/300, earlier ties;
report guard failure explicitly. NO test score selects weights. After the main
supervisor completes, run `assess_trained.py run` sequentially on the same GPU.
It reuses already decoded trained candidates when paths match and produces
TRAINED_RESULTS.md, trained_results.json, trained_selection.json, and a separate
trained_assessment_audit.json. Primary user-facing comparison should include
these actual-trained trial scores and the corrected controls; also explain
guarded fallback choices. Do not finish with only repeated control test scores.
This additional assessment is necessary to fully answer the requested six
trained-data trials and was explained to the user before test scoring.

Also declared before test generation: `note_unit_assessment.py freeze` stores
`note_unit_protocol.json`. After actual-trained assessment, run
`note_unit_assessment.py run`. It produces NOTE_UNIT_RESULTS.md and JSON, a
secondary note/type multiset F1 diagnostic to distinguish correct notes with
different grouping from wrong note identities. It never changes selection or
the primary exact-range metric. Self-tests cover grouping invariance, duplicate
prediction penalties, extra gap identity, repetitions, and unlocated false
positives. Then rerun `audit_final.py` to verify all additional test files too.

## Latest continuation additions (2026-11-06, ~03:00 local)

- ETA updated to 1.5–2.5 hours remaining from current state.
- Main campaign is still live: LadderSym N20 step300 validation; N40 remains.
- Added `finish_assessment.py` and launched it as a hidden background process
  (launcher PID45948). It polls main progress + final runtime-verified audit,
  then runs assess_trained, note_unit_assessment, output diagnostics,
  interpretation, final audit, and runtime hash verification sequentially.
  **Do not duplicate these assessments while this runner is alive.**
- Completion runner logs: `completion.log`, `completion.err.log`;
  progress: `completion_progress.json`; final combined audit:
  `completion_audit.json`. It waits up to four hours for the main campaign.
- `plot_trained_results.py` is prepared. Run it after completion, view
  `trained_learning_curve.png`, and include it in a concise final report.
- Monitoring functions cell `g2:38` runs ten read-only checks 45 seconds apart;
  it does not own or control the GPU jobs.
- Actual native Windows execution path in progress commands is
  `D:\stuff\Audio Evaluation\ALIGN`; relative Resolve-Path + Start-Process
  works. Literal C: apply_patch failed; relative apply_patch worked.
- Review the revised unchanged-model controls against the previous controls,
  not only trained vs revised control. Validation already shows excessive
  predictions before fine-tuning. Better reference crop coverage is not
  evidence of improved end-to-end accuracy.

## Update: all training and validation complete (~03:29 local)

Main campaign now runs test controls; PolyTune control finished all 21 in about
2 minutes with 0 capped windows. Test recordings are much faster than validation.
ETA updated to roughly 20–40 minutes remaining, allowing trained-output slowdown.
Guarded selection is step0 for all six trials. Supplemental actual-trained
choices from validation: PolyTune N10/N20/N40 step200; LadderSym N10 step200,
N20/N40 step100. All six of those actual-trained choices fail the validation
stability guard. No test score selected a checkpoint.
Completion runner PID45948 (child34288) is alive and will automatically start the
six actual-trained tests after main campaign audit. Do not launch duplicates.
Monitor functions cell now `g2:41`, reads every45 seconds and stops at completion.

## COMPLETE (2026-11-06, ~03:50 local)

All training, validation, both revised controls, all six actual trained test
passes, both metrics, output diagnostics, and final audits are complete.
No campaign, completion-runner, or engine processes remain running.
`completion_audit.json` passed; final integrity reports1199 hash checks.
Runtime amendment, original checkpoints/predictions, human labels, and teacher
labels verified. Eight unit/integration tests and both model smoke checks passed.

Primary strict F1 (0/10/20/40): PolyTune4.08/4.36/5.41/5.18%;
LadderSym2.42/3.26/2.70/2.76%.
Individual-note/type F1: PolyTune3.85/4.73/5.17/5.36%;
LadderSym2.70/2.43/1.74/1.78%.
All paired change confidence intervals include zero; all six actual trained
choices failed validation stability. No missed/repetition matches under either
metric. All revised trained strict scores remain below original controls
(PolyTune6.78%, LadderSym6.86%). No application default promotion.

Read SUMMARY.md for the final balanced interpretation and all metrics.
TRAINED_RESULTS.md received formatting-only whitespace fixes; frozen source
assess_trained.py unchanged. `trained_learning_curve.png` was generated and
visually checked; source is plot_trained_results.py (post-hoc reporting only).
SUMMARY.md was queued for opening in Codex. The monitoring functions cell g2:41
completed by itself. No required experiment work remains.
