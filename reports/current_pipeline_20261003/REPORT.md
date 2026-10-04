# Current pipeline diagnostic audit — 2026-10-03

This is a fresh inference audit of the latest experimental stack v6, not the older joint-CRF pipeline currently wired into the annotation GUI. Model weights, gates, and source data were not changed. All generated artifacts are in this folder.

## Scope and reproducibility

- 184 fresh audio transcriptions and alignments; zero failures.
- Synthetic: deterministic random sample of 30 eligible validation clips each from 9.2, 10.2, and 11.2, seed 20261003. These validation populations were used during model development; this is a diagnostic rerun, not a new held-out generalization result. Sealed test data were not rerun.
- DataCreate: inference on takes 001–094 only. Takes 095 onward were excluded entirely. Scoring uses 24 takes with manual labels plus take 004, explicitly saved with no errors: 25 reviewed takes. Remaining takes are inspected for output behavior but not scored as clean negatives.
- Candidate: `align-model/runs/precision-v4/CANDIDATE_STACK_V6.json`; transcriber `v5-dclike/best.pt`, robust DP aligner v3, presence verifier, extra gate 0.85 and missed gate 0.96.
- Checkpoint and verifier hashes match. All listed source hashes match after normalizing Windows CRLF to LF; four raw-byte hashes differ only in line endings.
- `manifest.json` contains sample selection and source audio/score SHA-256 hashes. `run_audit.py` is the executable audit. Synthetic lineage is loaded only after inference for evaluation; inference score indexing does not receive lineage.
- `synthetic.json` contains before/after-gate metrics and deletion diagnostics. `datacreate.json` contains gold, predictions and matching details. `alignments/` preserves all 184 inference outputs. `diagnostics.json` summarizes cases.

## Synthetic results

Numbers below are percentages. P/R means precision/recall, evaluated on canonical score-event identities using the repository's event-level evaluator.

| Family | Clips | Extra P/R | Missed P/R | Wrong-note P/R | Combined alignment F1 |
|---|---:|---:|---:|---:|---:|
| 9.2 realistic | 30 | 96.9 / 81.7 | 100 / 47.2 | 86.1 / 91.2 | 96.5 |
| 10.2 fast | 30 | 99.4 / 92.4 | 100 / 45.2 | 97.7 / 97.7 | 99.0 |
| 11.2 DataCreate-like | 30 | 85.7 / 46.2 | 100 / 71.4 | 91.7 / 95.7 | 99.4 |

Small-support warning for interpretation: 11.2 has only 13 gold extra events and 7 gold missed notes in this subset; its 100% missed precision means 5 correct calls. The 9.2 and 10.2 missed-note results mean 17/17 and 14/14 correct calls, respectively. These are not broad 100%-reliability claims.

Combined alignment F1 includes ordinary matches and copied notes. A diagnostic micro-F1 over just substitutions, extras and missed events is 86.9% / 95.4% / 83.5% for these three subsets. It excludes repetition as a passage-level error and is not a replacement headline metric. Synthetic event metrics also differ in unit of counting from the DataCreate feedback-label metrics: several extras in one score gap become one feedback label. Synthetic unlinked extra identities are established with the repository's pitch-sequence LCS evaluation adapter, not direct inference knowledge of gold identities.

### Precision gates lose real mistakes

| Family | Missed recall before gates | After gates | True misses removed by gates | True misses never proposed |
|---|---:|---:|---:|---:|
| 9.2 | 77.8% | 47.2% | 11 of 36 | 8 of 36 |
| 10.2 | 100% | 45.2% | 17 of 31 | 0 |
| 11.2 | 100% | 71.4% | 2 of 7 | 0 |

Example: `synth_Weber1_25513` has planted misses at canonical notes 15 and 33. Both are proposed before gating and both are removed. Its final feedback therefore misses both known errors.

In 9.2, extra-note recall also falls from 87.4% to 81.7%; 20 correct extra detections are suppressed. Content-error micro-F1 falls from 89.2% to 86.9% despite higher precision. This is an explicit conservative tradeoff, not simply better detection.

## DataCreate 001–094, reviewed subset

Scoring preserves the existing repository convention: canonicalized score ranges with stored padding, exclusive one-to-one matching, and only manual gold. The normalizer defaults unspecified repetition copy counts to one, as in the existing evaluator. No metric comparisons were unavailable.

| Type | Gold labels | Predictions | Exact hits | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Extra | 12 | 32 | 2 | 6.3% | 16.7% | 9.1% |
| Missed | 8 | 8 | 1 | 12.5% | 12.5% | 12.5% |
| Wrong | 22 | 21 | 1 | 4.8% | 4.5% | 4.7% |
| Repetition | 4 | 9 | 0 | 0% | 0% | 0% |

The lenient same-type/core-overlap-within-one-note diagnostic finds 2 extras, 1 missed note, 3 wrong notes, and 2 repetitions. Thus exact boundaries explain some repetition/wrong-note penalties but do not explain the poor extra/missed agreement.

The earlier saved 001–094 report counted 24 positive takes. This audit includes explicitly reviewed clean take 004, which adds one false extra prediction (32 instead of 31).

Twenty additional manual labels are outside stack v6's output classes: 5 rhythm errors, 5 bad starts, 4 sliding labels, 4 bad-timbre labels, and 2 squeaks. They are excluded from the four-type table, but represent feedback the current stack cannot provide in those categories.

These are agreement scores against existing manual files, not independently adjudicated acoustic truth. Some labels contain overlapping ranges or conflicting projection fields even below take 095. For example, take 005 includes individual wrong-note regions and an overlapping aggregate region; take 041 has overlapping repetition labels. The audit leaves all of them unchanged.

## Concrete output problems

1. **Brief transitions survive as confident extra notes.** Take 004 is explicitly reviewed as clean, but the model reports written MIDI 79 at 8.3824–8.3940 s: just 11.6 ms, with transcription confidence 0.964. Take 034 has a similar MIDI 64 extra at 13.1425–13.1541 s, confidence 0.990. The spectrogram views show these candidates at pitch transitions; they are consistent with transition artifacts rather than separate stable attacks. Transcription confidence is not calibrated error probability. Across all 94 takes, 51 of 331 retained extra events are under 50 ms and 77 under 80 ms. Those counts include unreviewed takes and do not mean every short event is false.

2. **Gating hides true omissions.** The synthetic ablation above isolates this directly. The high missed-note precision comes with losing more than half the true misses in the 9.2 and 10.2 subsets. On the 94 real takes, gates suppress 673 extra and 933 missed calls, but their truth cannot be determined without additional gold.

3. **Low-quality whole-clip alignment still produces confident-looking feedback.** Take 063 has match fraction 0.107, yet outputs 107 substitutions and only 13 matches. No clip-level abstention occurs. The gate suppresses 22 deletion calls, while the substitution flood survives. This take has no usable manual gold, so the cause could include score-excerpt mismatch or transcription/alignment failure; the demonstrated problem is that unreliable global alignment does not trigger a useful rejection state. The frozen gate's default minimum match fraction is zero.

4. **Repeat representation is too restrictive for practice.** The grammar permits one contiguous measure-aligned source passage with one or two additional copies. It cannot express several independent local restarts in the same take. Take 035 has two manual repetition regions; the model emits one different range with two extra copies. Take 041 has partially overlapping repeat labels; the model's larger predicted passage gets a lenient hit but no exact match. Both model restrictions and annotation boundary conventions contribute.

5. **The feedback exporter loses playback timing for misses and repeats.** `labels_from_stack_alignment` assigns both types the placeholder 0.0–0.05 s. On take 020 it correctly identifies missed score note 206, whose gold audio interval is 28.635–28.8351 s, but exports it at the beginning of the recording. Repetition labels also have no `repeats_label_range`. Score-only evaluation does not penalize these unusable playback timestamps. This is an exporter defect separate from acoustic model accuracy.

6. **Score location and audio disagreement require diagnosis, not just looser scoring.** Take 020's gold wrong note is index 79 at 12.558–12.6651 s. Around that time the alignment emits matches at indices 82–83 and produces no wrong-note label there; its wrong-note call is instead at index 144 around 20.38 s. This can involve score projection or alignment, and the inspected data do not establish which side is right. Pitch recognition, score mapping, and error classification need separate audits.

7. **Feedback coverage is incomplete.** Rhythm, intonation and technique/timbre problems are not emitted by this stack. These classes exist elsewhere in the repository and GUI, but their presence in the taxonomy does not imply current model support.

8. **Aggregate accuracy conceals the user-facing failures.** In the 11.2 subset, combined alignment F1 is 99.4%, while extra recall is only 46.2%. Reporting ordinary-note alignment accuracy without error-specific recall can give an inaccurate impression of feedback reliability. A missing JSON error label currently cannot distinguish a confident correct performance from an abstained error decision.

## Suggested priorities

1. Repair missing/repetition timestamps and export uncertainty/abstention explicitly.
2. Add a whole-clip alignment quality check before emitting substitutions; inspect low-match takes for wrong score excerpts and pitch conventions.
3. Build a reviewed real-audio set of transition artifacts, soft/fast notes, and same-pitch reattacks; improve candidate segmentation and train/calibrate the error gates against it with a frozen split.
4. Expand repeat handling to local note ranges and multiple restart regions, then audit score projections and overlapping gold labels.
5. Track per-type error precision/recall and error-only performance alongside alignment metrics. Avoid solving the problem solely by increasing thresholds.

## Evidence files

- `figures/004.png`: extra-note false alarm on a reviewed clean recording.
- `figures/034.png`: high-confidence 11.6 ms transition candidate.
- `figures/020.png`: output near the human-marked wrong-note interval.
- `datacreate.json`: full feedback labels and matching evidence for all 25 scored takes.
- `synthetic.json`: fresh synthetic results and before/after-gate comparison.
- `manifest.json`: frozen configuration checks and exact input hashes.

To reproduce in the project's AMT environment: run `reports/current_pipeline_20261003/run_audit.py`, then `analyze.py`. The script can resume its own cache, so use a fresh output folder for a new uncached run or after changing inputs. The inference cache is local to this audit; original sample folders are never overwritten.
