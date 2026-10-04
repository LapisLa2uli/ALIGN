# Stack v7 — score-note-first experimental version

Implemented and trained on 2026-10-04. This is a separate candidate; the annotation GUI and stack v6 are unchanged.

## What changed

- `transcription/transition_v7.py`: label-preserving transition/attack and quiet-band augmentation, a frozen clean-input teacher consistency objective, and attack-aware optional transition candidates. A short note remains available to the score aligner instead of being erased by duration alone. No replacement neural architecture: v7 fine-tunes the existing dual-resolution CTC network and retains its checkpoint format.
- `joint/restarts_v4.py` and `joint/robust_dp_aligner_v4.py`: bounded proposals at note boundaries, including two disjoint local restart regions, alongside the old measure/ornament hypotheses. Numerical DP primitives remain shared; frozen v3 code is not edited. Proposal limits: 2–24-note phrases, at most 12 distinct starts, up to two copies per region. Proposals require adjacent phrase evidence; arbitrary/nested restarts are not solved.
- `joint/stack_v7.py`: returns `alignment_uncertain` below match fraction 0.45, with explicitly unassessed score indices. Otherwise produces affected `score_event_indices` separately from `context_score_event_indices`. Canonical notes collapse ties; these indices are not raw MusicXML element ordinals. Missing/repetition playback times are estimated or null, never invented as 0–0.05 seconds.
- Missed-note gate threshold calibrated from 0.96 to 0.80 on synthetic selection data. Extra threshold remains 0.85. Existing gate/verifier weights were reused; the DP aligner is algorithmic, not a newly trained neural aligner.

## Training performed

Two epochs of real GPU fine tuning, initialized from stack v6, on 1,536 distinct training clips: 512 each from E:/outputRaw_realistic_10k (9.2), E:/outputRaw_fast16_focus_5k (10.2), and E:/outputRaw_dclike_11 (11.2).

Training reads the existing checksum-verified dual-mel caches derived from those E: datasets. Eligible repaired-lineage train membership is checked against the frozen splits, and the corresponding source WAVs must exist. No validation/test sample is in the selected training subset. This was a bounded pilot, not a full-dataset training run. Batch 8, 1,024-frame crops, AdamW LR 5e-5; CTC + 0.5 frame loss + 0.15 clean-teacher consistency. Training manifest and cache pack IDs are under `runs/stack-v7/training_manifest.json`. Source E: bundles were not modified.

Both `epoch-01.pt` and `epoch-02.pt` were saved. Mean training loss fell from 0.6681 to 0.6608. These losses alone do not establish better labeling.

## Selection and results

Metric: exact canonical score-note identity and error type for substitutions, extras and missed notes, micro-aggregated per family and then macro-averaged across families. Timestamps and ordinary correct-note accuracy do not select the candidate. Repetition passage accuracy is reported separately on real data, not included in this content-error selection metric. Synthetic extras use the existing pitch-sequence LCS identity adapter; this is an inherited evaluation limitation.

The prior 90-clip synthetic validation audit was split deterministically into 45 selection and 45 check clips, 15 per family in each half. These are previously used development/validation data, not a new unseen or sealed test. Neither E: sealed test data nor DataCreate labels selected weights or thresholds. Gate calibration was a second experiment after the initial validation comparison. Ties use the larger, more conservative missed threshold; 0.50 and 0.80 tied on selection.

| Variant | Selection error F1 | Check error F1 |
|---|---:|---:|
| v6 baseline | 86.92% | 90.31% |
| v7 decoder/aligner, v6 weights, original gate | 86.92% | 90.31% |
| v7 epoch 1, original gate | 86.19% | 90.22% |
| v7 epoch 2, original gate | 86.64% | 90.00% |
| Selected v7: v6 weights, missed threshold 0.80 | 87.16% | 92.50% |

**The newly trained transcriber did not beat the baseline and was not selected.** Default v7 retains the stronger v6 transcriber weights. The trained v7 checkpoints and an explicitly experimental epoch-2 candidate are available for further work. The selected gain comes from the gate operating point, not demonstrated transcription improvement or successful local-restart recovery on this corpus.

Check-half missed-note recall changes:

| Family | v6 | Selected v7 | Gold missed notes |
|---|---:|---:|---:|
| 9.2 | 46.7% | 73.3% | 15 |
| 10.2 | 40.0% | 50.0% | 10 |
| 11.2 | 60.0% | 100% | 5 |

Missed precision remains 100% on these small subsets (11, 5 and 5 correct calls). This does not establish perfect generalization.

## DataCreate comparison

Fresh inference on 001–094; labels from 095 onward were excluded entirely. Scored 24 manually labeled takes plus the explicitly reviewed clean take 004. Other recordings were not treated as error-free. Evaluation uses affected score cores, excludes padding, and does not match by time. Missing repetition counts retain the existing evaluator's default of one.

| Type | v6 P / R | Selected v7 P / R |
|---|---:|---:|
| Extra | 6.25% / 16.67% | 6.25% / 16.67% |
| Missed | 12.50% / 12.50% | 12.50% / 12.50% |
| Wrong | 4.76% / 4.55% | 4.55% / 4.55% |
| Repetition | 0% / 0% | 0% / 0% |

One additional wrong-note prediction is unmatched. No real-data accuracy improvement is established. Existing gold projection inconsistencies and overlapping labels remain unchanged.

The clip-quality check returns uncertain on 12 of the 94 recordings: 050, 053, 058–067. These have no gold in the reviewed subset, so abstention does not account for gains in the table. On take 063, this prevents the old 107-substitution flood from appearing as reliable feedback.

The transition decoder marked **zero** candidates optional in these 94 recordings, and no output selected two local restart regions. The new logic passes constructed regression tests, but its real-data benefit has not been demonstrated. Do not describe transition artifacts or practice restarts as solved. The new transcriber needs broader training/data work; simply increasing confidence thresholds remains inadequate.

## Run

From the repository root in PowerShell:

```powershell
& 'align-model/.venv-amt-bench/Scripts/python.exe' `
  align-model/scripts/run_stack_v7.py `
  --score DataCreate/samples/020/verified_score.musicxml `
  --audio DataCreate/samples/020/performance_audio.wav `
  --output align-model/runs/stack-v7/my-prediction.json
```

Default candidate: `runs/stack-v7/CANDIDATE_STACK_V7.json` (selected weights and calibrated threshold). To inspect the newly trained epoch 2, pass `--candidate align-model/runs/stack-v7/CANDIDATE_STACK_V7_TRAINED_EPOCH_02.json`. Both remain experimental (`promoted: false`). Checkpoint and v7 source hashes are verified by the runner. The runner assumes a written B-flat clarinet score, like training; take 001's concert-pitch exception is not automatically adapted.

Output schema `align-score-feedback-v7` intentionally distinguishes `status`, core note identities, contextual notes, and unassessed notes. It is not automatically copied into legacy `labels_agent.json`; a GUI integration must preserve those distinctions.

For another training pilot, use a **new** output directory:

```powershell
& 'align-model/.venv-amt-bench/Scripts/python.exe' `
  align-model/scripts/train_stack_v7.py `
  --output align-model/runs/stack-v7-next --epochs 2 --train-per-family 512
```

`train_stack_v7.py` saves both model candidates and the final optimizer state. Automatic checkpoint resume is not implemented. `evaluate_stack_v7.py` and `calibrate_stack_v7.py` reproduce this experiment in `runs/stack-v7`; their caches belong to this experiment and should not be reused after changing data or weights. Use a new run directory/version for follow-up experiments. Candidate selection should include passage-level repetition accuracy and larger unseen validation before promotion.

## Verification and artifacts

- 12 regression tests passed: new transition/restart/uncertainty behavior plus existing score-location metric tests.
- CLI smoke test completed on take 020 and wrote five labels to `runs/stack-v7/example-020.json`.
- `training_manifest.json`, `history.json`: actual data selection and completed training.
- `validation_protocol.json`, `validation.json`, `calibration.json`: fixed populations and all compared metrics.
- `datacreate-baseline.json`, `datacreate-baseline-m0.80.json`: gold/prediction score-note details.
- `predictions/`: versioned synthetic and DataCreate outputs.

Stack v6 and the GUI configuration remain available unchanged. The remaining central problem is real-recording note-label accuracy, not playback timestamps.
