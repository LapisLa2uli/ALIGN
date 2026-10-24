# Historical experiment handoff — completed

**The work below is finished.** Read `RESULTS.md`, `completion.json`, and
`CANDIDATE_SOLO_V1.json` for the final selected model, held-out evaluation,
DataCreate regression, and verification. Do not restart the historical jobs
or tune the frozen candidate against its opened test split.

Task: finish the shared solo-instrument transcriber, including actual evaluation
and a frozen usable candidate. Do not stop while training is merely running.

Physical Python cwd is `D:/stuff/Audio Evaluation/ALIGN`, although the shell/app
often displays `C:/Users/will61/ALIGN`. Use relative apply_patch paths; the C:
absolute path does not exist in the underlying filesystem. Python is always
`C:/Users/Hank/.conda/envs/MusicEval/python.exe`, with PYTHONNOUSERSITE=1.
No subagents. Original checkpoint/default configs must remain unchanged.

At host time Oct 17 02:26 (+08), jobs:

- Session **68639**: `run_solo_refinement.py`. Chromatic refinement epoch 2 of 2
  is underway; afterward it automatically evaluates baseline, transfer,
  augmented, chromatic on the full musical validation set (1272 clips).
  Watch `refinement_status.json`, `chromatic_refinement.log`,
  `validation_*.log`. It ends at VALIDATION_READY_TEST_UNOPENED.
- Session **41196**: range-validation generation completed (288 clips);
  it next audits all audio and prepares `E:/ALIGN_Data/solo_range_validation_cache_v1`
  on CPU. Watch `range_quality.log` and `range_prepare.log`.
- Main campaign (session 17454) finished. Transfer 2 epochs, augmented 6 epochs.

Datasets:

- Musical corpus `E:/ALIGN_Data/solo_multinstrument_v1`: 12,000 clips, 147,474
  notes, 17.282 h; 24 instruments, each 400 train/50 val/50 test. Cached under
  `solo_multi_cache_v1`, plus 600 legacy clarinet train and 72 validation clips.
- Chromatic supplement `E:/ALIGN_Data/solo_chromatic_support_v1`: 4,608 training
  clips, 36,864 notes, 3.0415 h; cached `solo_chromatic_cache_v1`. Every instrument
  pitch is covered. All new generated training/core audio: 16,608 clips, 20.32 h.
- Independent range diagnostic `solo_range_validation_v1`: 288 validation clips,
  0.190 h, new seed and rhythm, six low/mid/high pitches per instrument/bank.
  It targets missing/sparse pitches (notably MIDI 32 and 104 absent in musical
  training). No model has trained on it; final musical test remains unopened.

QA: all core/supplement audio passes finite/clipping/per-note audibility checks.
FluidR3 contrabass misses keys 59–67, and several repeated-note renders became
silent. Generator now probes bank ranges and checks rendered note windows,
falling back to the other bank. Core 91 fallback clips, supplement 23. Audio
montages saved; NO perceptual listening performed (no reliable listening tool).

Progress on 500-clip development subset (macro includes legacy clarinet):
baseline .6734, transfer best .8441, augmented epoch 5 best .95214 (epoch 6
.94844), chromatic epoch 1 .94597. Augmented legacy F1 ~.984 vs baseline .9895.
Chromatic second epoch and full validation will determine selection. It may
trade a little musical accuracy for actual coverage of previously unseen pitches.

Next steps:

1. Wait for jobs and inspect full validation. Evaluate baseline/augmented/
   chromatic on range cache with `solo_experiment.py eval --cache ...
   --checkpoint ... --limit 72 --output runs/solo-v1/validation_range_*.json`.
2. If useful, consider global decoder blank-scale tuning on validation only
   (currently .5); don't open test until selection is frozen. No tuning has yet
   occurred. If changing decoder, update freeze helper to use chosen report's
   blank_scale instead of its current fixed .5.
3. Freeze reviewed choice with `freeze_solo_candidate.py --variant ... --reason
   ...`. It validates checkpoint hashes, refuses if test results already exist,
   writes CANDIDATE_SOLO_V1.json and copies validation_candidate_full.json.
   Consider incorporating range validation evidence in that manifest/reason.
4. Run paired final test, limit 50, on `solo_multi_cache_v1`, saving
   baseline_test.json and candidate_test.json (same 1200 clips).
5. Run `evaluate_solo_datacreate.py --checkpoint <selected> --output
   align-model/runs/solo-v1/datacreate --device cpu`. CPU permits reuse of the
   completed baseline cache. All 25 human takes plus 007/095 are inspected.
6. Run selected-candidate standalone/publisher smoke on copied scratch sample,
   not live DataCreate. Check protected human-label/audio/score hashes.
7. `report_solo_results.py` generates RESULTS.md, comparison_statistics.json,
   instrument_f1.png. Review chart (legend may need placement adjustment), add
   range-diagnostic and actual selection findings, update main documentation.
8. Final tests/diff check as warranted and verify original checkpoint hash
   35cf4d3e540c04c86270a966f7c2b6abdd7827d9643964cc6763bf7bd0ccec3b unchanged.

Real gold: live labels.json files are all copied v9 AGENT labels. Original human
labels are in `DataCreate/archives/user-labels-before-agent-seed-20261006T074109Z-763e1b7d`.
The evaluation helper now verifies archive checksums and corresponding archived
agent provenance score/audio hashes against current inputs. All 25 eligible
001–094 takes match; 007 and 095 are NOT gold. Gold >=095 excluded. Baseline
strict canonical identity feedback F1: wrong .04651, missed .125, extra .09302,
repetition 0. These weak results must be reported honestly; do not substitute
agent labels or timing accuracy. Cached report-v10.json is valid, 25 gold takes.

Implementation is opt-in `stack-solo-v1`, model outputs sounding MIDI 28–108,
no instrument conditioning. Original dual-resolution architecture retained.
Score transpose metadata or explicit override maps to written pitches. Generic
config DataCreate/config/solo.yaml defaults concert pitch for absent metadata;
old DataCreate clarinet XML needs written_to_sounding -2. Aligner is bounded v9
path; old clarinet-specific boundary repair/verifier are not applied to new
instruments. Inherited error gates are experimental outside clarinet.

Tests: 23 focused tests passed, then added chromatic coverage test passed (4
synth tests total); latest 10 bridge tests passed. Original full focused suite
would now total 24. Human-archive checksum and exclusion tests included.
Pip check passes; psutil was added to align-model dependencies and installed in
MusicEval. All other deps including httpx/matplotlib present.

Smoke: `smoke-trial/` standalone, score alignment and publisher succeeded on a
copied flute training sample. UI TestClient GET / and note-alignment endpoint
both 200; torch not imported into UI server. Run again with frozen final model.
Standalone CLI now defaults to the candidate manifest (not augmented/best.pt)
and adds RMS evidence for alignment. Bridge corpus transcription also uses the
candidate, converting output to written pitches for its legacy locator.

Final report must distinguish 24-program synthetic results from real-world
coverage (only clarinet recordings available). No literal universality claim,
no listening claim, and no claim of acoustic note-offset estimation (ends are
next-onset estimates). Defaults stay v9/v10; provide opt-in config and commands.
