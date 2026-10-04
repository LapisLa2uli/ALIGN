# Extra / missed-note precision (stack v6) and DataCreate-like data (family 11)

Goal: per-type precision > 0.95 for extra notes and missed notes (official
exclusive note-wise metric), measured on sealed synthetic test sets, with
DataCreate precision against the human labels reported alongside.

## Result

Stack v6 (`CANDIDATE_STACK_V6.json`) passes on every sealed test population:

| Test population | Clips | Extra P (95% CI) | Extra R | Missed P (95% CI) | Missed R | Combined F1 |
|---|---:|---:|---:|---:|---:|---:|
| 9.2 (WeberITAV, cruel-angels, procedural) | 1,836 | **0.983** (0.980-0.986) | 0.82 | **0.984** (0.976-0.992) | 0.43 | 0.967 |
| fast 10.2 | 597 | **0.994** (0.993-0.995) | 0.92 | **1.000** (1.000-1.000) | 0.50 | 0.989 |
| 11.2 test (new pilot) | 260 | **1.000** (56 calls) | 0.41 | **1.000** (27 calls) | 0.33 | 0.985 |

Combined F1 is unchanged from stack v3 (9.2 test 0.967, fast 0.990): the
precision comes from withholding low-evidence calls, which lowers recall of
extra and missed notes. `TEST_EVALUATIONS.jsonl` lists each candidate's
single test read; v6 appears twice because the freeze file was written twice
(identical content apart from the freeze timestamp, identical results).

| Candidate | 9.2 extra P | 9.2 missed P | fast extra P | fast missed P | why it changed |
|---|---:|---:|---:|---:|---|
| v4 | 0.980 | 0.932 | 0.994 | 0.974 | timing pass + logistic gates (val even half) |
| v5 | 0.988 | 0.902 | 0.995 | 0.997 | gates refit on all val (clip CV) - did not transfer to unseen scores |
| **v6** | **0.983** | **0.984** | **0.994** | **1.000** | new transcriber, note-presence verifier, leave-one-score-out gates |

## What changed

**Aligner v3** (`src/alignmodel/joint/robust_dp_aligner_v3.py`), on top of v2's
repeat grammar, ornament template and DP:
- timing pass: matched notes give a tempo map; a second DP adds a capped
  onset-deviation cost (weight 0.2), breaking ties between identical pitches;
- an unexplained note directly after a replayed note is labelled as part of
  the replay (the replayed copy of an extra note);
- every extra and missed call gets evidence features and passes a logistic
  gate; no missed call on ornamented notes or at the unplayed ends of a take.

**Note-presence verifier** (`src/alignmodel/joint/presence_verifier_v1.py`):
a small CNN that looks at the dual-resolution mel around the expected onset
of a score note, with harmonic templates of its pitch and its neighbours, and
says whether the note was played. Trained on 62k examples from the 9.2 / 10.2
/ 11.x training splits (planted missed notes vs played notes, both centred
where an aligner would expect them). Val AUC 0.992. It is the feature that
made missed-note gating generalize: with it, held-out precision on scores the
gate never saw rose from about 0.93 (Weber1) / 0.81 (introduction-theme) to
0.977 / 0.979.

**Transcriber** (`runs/precision-v4/v5-dclike/best.pt`): v3 fine-tuned with
label-preserving realism augmentation (transition blips, attack scoops,
mid-note dips; `src/alignmodel/transcription/realism_augment_v4.py`), then on
9.2 + 10.2 + 11.x train. 9.2 val pitch F1 0.978, 11.x val 0.992.

**Gate selection** (`scripts/fit_gates_v5.py --fold-by group`): leave one
source score out at a time (Weber1, introduction-theme, each 11.x training
score, each procedural set). Missed threshold 0.96: the lowest with every
held-out group at >= 0.97. Extra threshold 0.85: highest worst-dataset
held-out precision with recall > 0.3 (9.2 0.962, fast 0.997, 11.x 0.855).
Thresholds were chosen on val only; DataCreate was not used.

## Why missed-note precision was hard

On the 9.2 test, v4/v5 stayed at 0.90-0.93 although val said 0.97-0.98:
9.2 val has one real score (Weber1), the test has two others. Wrong missed
calls were played notes the transcriber did not hear (ornament realizations,
merged repeated notes, short notes in fast runs). Features derived from the
transcriber cannot separate those from real skips on an unfamiliar score; the
presence verifier looks at the audio directly.

## DataCreate against the human labels

Population: 33 takes with GUI ("manual") labels plus one take saved by a
human with none (95 labels). Lenient match = same type, core score range
within one note.

| Stack | extra predicted / matching | missed predicted / matching | wrong-note predicted / matching |
|---|---:|---:|---:|
| stack v3 (as delivered) | 105 / 9 | 56 / 4 | 30 / 2 |
| v5 gates | 7 / 0 | 13 / 2 | 32 / 2 |
| v6 | 33 / 3 | 9 / 1 | 27 / 4 |
| v6, extra threshold 0.97 (tuned on DataCreate, not used for the tests) | 8 / 0 | 9 / 1 | 27 / 4 |

False alarms fall by 70-95%, but precision against the human labels stays
far below 0.95. Most v3 false extras on real audio were 10-40 ms pitch blips
at slurred note changes and attack transients, which Muse Sounds never
renders; the realism augmentation reduced them only slightly. Some remaining
missed calls look audibly right (e.g. take 012 note 35 and take 036 note 77:
the score note is absent in the spectrogram) but are not labelled.

## Data

- E: pruned (about 1 TB freed): 3.1, 5.x, 6.0, 7.x, 8.x keep a few bundles,
  `_config/` and `PRUNED.json`.
- Family 11 (`synth-pipeline/config/dclike_11_*.yaml`, registry 11.x): written-
  out ornaments, DataCreate register and error mix, held-over missed notes,
  degrade realistic_v3, two note_map lineage fixes (repeated notes folded as
  ties; repeat gaps shifting the timed matching). 3,300 bundles, 2,875 pass
  lineage repair.
- 11.2 is still acoustically separable from DataCreate (mel-statistics AUC
  0.998): quiet passages lack 7-9 dB of mid-band noise. That is the next
  degrade change.
