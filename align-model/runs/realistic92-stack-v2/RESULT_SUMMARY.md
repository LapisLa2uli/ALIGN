# Transcriber + aligner stack v3 (9.2 and fast 10.2)

## Decision

The frozen stack **passed** the > 0.95 combined target on both held-out test
populations. Each was evaluated once (`TEST_EVALUATIONS.jsonl`).

| Test population | Clips | Combined note-wise F1 | 95% CI | Precision / recall |
|---|---:|---:|---:|---:|
| 9.2 test (score-grouped: WeberITAV, cruel-angels, procedural) | 1,836 | **0.9671** | 0.9653–0.9690 | 0.9748 / 0.9596 |
| Fast 10.2 test (procedural) | 597 | **0.9904** | 0.9897–0.9911 | 0.9964 / 0.9844 |

Combined F1 is the official exclusive note-wise metric on canonical
`verified_score` events: frozen transcriber → aligner, scored against
repaired gold. Span-less predicted notes take the rendered index of the gold
note they pair with along the pitch sequence; unpaired notes never match.

Per-type combined F1 on 9.2 test: match 0.978, copy 0.965, extra 0.905,
substitute 0.872, missed note 0.671.

## Short notes and false positives (transcriber, 9.2)

| Measure | Round-2 CTC (previous), val | v3 stack, val | v3 stack, **test** |
|---|---:|---:|---:|
| Pitch-sequence F1 | 0.963 | 0.977 | 0.982 |
| Recall, gold notes < 50 ms | 0.751 | 0.910 | 0.886 |
| Recall, gold notes 50–80 ms | 0.805 | 0.863 | 0.903 |
| Recall, notes next to a same-pitch neighbour | 0.904 | 0.935 | 0.934 |
| Precision | 0.974 | 0.984 | 0.989 |
| False-positive rate (unpaired predictions) | 2.6% | 1.6% | 1.2% |

Short-note recall rose by roughly 9–14 points while false positives fell.
Durations are the gold render durations from `note_map`. They categorize
notes only and are not compared with predicted timing.

## Version rationale

**Earlier stack.** The round-2 CTC transcriber used one 93 ms analysis
window and about ±0.9 s of context. It fed the perfect-transcription DP
aligner v1 and reached combined val F1 **0.940**.
- The breakdown showed the largest losses were misses: 75% recall under
  50 ms, and 90% on notes next to a same-pitch neighbour.
- Same-pitch false re-emissions were rare, about 1% of emissions.
- Onset and boundary heads barely separated true from false emissions.
- About 40% of the combined loss came from the aligner mishandling notes the
  transcriber got right, once transcription errors were present.

**Decoder changes.** A structured decoder merged weak same-pitch
re-emissions, recovered repeats from attack peaks, and removed A-B-A flicker.
It did not help: the false splits it targets are rare, and head evidence
doesn't separate them (`decode-val-grid1.json`). It was not promoted. The
kept decoder is greedy CTC plus "rich" outputs for the aligner: per-note
confidence, runner-up pitch, and optional weak-peak candidates.

**Transcriber v3.** Changes:
- a 23 ms analysis-window branch alongside the 93 ms branch;
- a 12-block dilated TCN with about ±2.9 s context;
- realistic mel augmentation: vibrato and drift, swells, reverb tails, and
  deeper articulation dips;
- targets re-aligned with the round-2 model.

It was trained on 9.2 train, then fine-tuned on 9.2 plus fast 10.2 train.
9.2 val pitch-sequence F1 rose from 0.963 to 0.977, and fast val from 0.947
to 0.989.

**Aligner v2.** It keeps v1's repeat and ornament grammar, and adds:
- confidence-scaled insertion;
- dropping low-confidence notes that fit nowhere;
- optional weak-peak candidates that can fill score notes;
- runner-up-pitch matches;
- a fallback when no hypothesis passes the length filter.

With the same round-2 transcriptions it lifted combined val F1 from 0.940 to
0.945. After retuning costs on the even half of 9.2 val (`tune-aligner-v2-v3c.json`)
and pairing with v3, combined F1 is 0.966 on 9.2 val. It is 0.961 on the
val half the tuning never saw, and 0.990 on fast val.

**Status.** Promoted for 9.2 / fast evaluation. It is not validated on real
recordings; DataCreate remains out of domain.

## Frozen candidate

`CANDIDATE_STACK_V3.json` (SHA-256 `e8d0f327efe50fec4dfd78023f75e8aceb71acf60230ce8f976f365699995eb4`):

- Transcriber: `v3-c-fast/best.pt` (SHA-256 `60663218…50cad9`), 2.09M parameters.
  - 192-band input: 128 long-window mel bands + 64 short-window bands.
  - Rich decoding at blank scale 0.5, candidate threshold 0.08.
- Aligner: robust DP v2 with costs substitute 1.1, insert 1.2, delete 1.0,
  copy 1.0, drop below confidence 0.85 at 0.8, optional match 0.15,
  runner-up match 0.6. Other settings are as in v1.
- Code hashes for every model, decoder, aligner, and scoring module are
  recorded in the candidate file and verified by the test driver.

## Data and splits

- **9.2.** The score-grouped split frozen for the transcriber work.
  Aligner evaluation population: repaired-lineage eligible clips
  (`runs/realistic92-aligner-v1/ALIGNER_FREEZE.json`).
- **Fast 10.2** (`E:\outputRaw_fast16_focus_5k`, degradation `realistic_v2`).
  - Split frozen before training: 4,000 / 400 / 600 (`runs/fast102-v1/split.json`).
  - Lineage repaired with the same fail-closed reconstruction: 4,973 of 5,000 eligible.
  - Evaluation population: `runs/fast102-v1/ALIGNER_FREEZE.json`.

Training used train splits only. Checkpoints, blank scale, and aligner costs
were chosen on val only.

## Artifacts

| Artifact | SHA-256 |
|---|---|
| `TEST_STACK_V3_realistic92.json` | `fc7acc20a0a0fe425d33e5f55918208b4fd99ed242c4a0e54e30512330514b8d` |
| `TEST_STACK_V3_fast102.json` | `7f84446243ae34a863891ef8db47129169da2b1db952830fd55217bb38acb4a5` |

Code:
- `src/alignmodel/transcription/mel_ctc_v3.py`
- `src/alignmodel/transcription/ctc_decode_v2.py`
- `src/alignmodel/joint/robust_dp_aligner_v2.py`
- scripts:
  - `train_mel_ctc_v3_realistic92.py`, `build_realistic92_aligned_cache.py` (`--dual-mel`, `--aligner-kind ctc_base`)
  - `cache_v3_outputs.py`, `decode_eval_realistic92.py`, `e2e_eval_realistic92.py`
  - `tune_aligner_v2_realistic92.py`, `eval_stack_v3_test.py`

Note: the first v3 run crashed with an out-of-memory error during epoch 7,
while other jobs were running. Training resumed from the epoch-6 checkpoint
(`v3-b`).
