# Dataset 9.2 perfect-transcription aligner result

## Decision

The frozen structured DP aligner **passed** the > 0.95 target on the held-out
9.2 test population.

- Official exclusive note-wise F1: **0.982620**
- Precision / recall: 0.983610 / 0.981631
- Clip bootstrap 95% interval (1,000 replicates): **0.981169–0.983863**
- Procedural / RawData F1: 0.983329 / 0.982177
- Per-type F1: match 0.9926, copy 0.9754, extra 0.9632, substitute 0.8491,
  missed note 0.8235
- Clips: 1,836 (no aligner failures)

Task: the input is the gold rendered note sequence (pitch, start, end in
rendered order), i.e. a perfect transcription. The aligner maps each note to a
canonical `verified_score.musicxml` event span, copy pass, and relationship
(match / substitute / copy / extra), and predicts missed score events.
Scoring uses the official metric: exclusive one-to-one matching on
canonical score-event identity. Same location and type earns 1.0, same
location with another type 0.5, and a wrong location 0. Span-less extras are
identified by rendered index. Timestamps are not used for scoring.

## Target repair

The shipped `note_map.json` rendered lineage has mapping errors. Examples:
rendered notes assigned to score events out of order, and copy notes whose
span merges two performed notes from different passes (span 19–40). The
frozen identity CRF scored 0.777 against that raw gold on 20 train clips; no
aligner can learn those labels.

Targets were rebuilt with the ORN fail-closed reconstruction
(`global_ornament_lineage_v2`): the performance MIDI is aligned to the
performance score's ornament template, and any clip that does not round-trip
exactly is rejected. One 9.2-specific fix was needed. music21 emits grace
notes at the principal's onset, so MIDI order within a shared onset is
arbitrary. Each shared-onset group is now permuted to best match the template
before the unchanged exact check (`scripts/repair_realistic92_lineage.py`).
That raised procedural pass rates from 2/30 to 58/60 in development.

Eligibility depends only on reconstruction validity; no model output was
read. It was frozen in `ALIGNER_FREEZE.json` before any val or test scoring.

| Split | Eligible / clips | Procedural | RawData scores |
|---|---:|---:|---|
| Train | 6,628 / 6,864 | 3,935 / 4,000 | 001 720/720, Mozart 686/716, howls 635/720, spirited-away 652/708 |
| Val | 935 / 1,108 | 395 / 400 | Weber1 540/708 |
| Test | 1,836 / 2,028 | 585 / 600 | WeberITAV 641/708, cruel-angels 610/720 |

The split is the score-grouped 9.2 split frozen for the transcriber work
(`runs/realistic92-transcriber-v1/split.json`).

## Aligner

`src/alignmodel/joint/perfect_dp_aligner_v1.py` enumerates repeat hypotheses
(no repeat, or one measure window repeated 1–2 extra times) and expands each
into the reference score's ornament template. The principal note is linked to
the score event; trill, mordent, turn, and grace auxiliaries become template
extras. A numba two-layer DP aligns the note pitches to each template with
these costs:

| Operation | Cost |
|---|---:|
| exact linked match / exact ornament | 0 |
| substitute | 1.4 |
| extra | 1.0 |
| missed | 1.0 |
| skip ornament unit | 0.35 |
| ornament pitch within 2 semitones | 0.30 |
| same-pitch merge (one rendered note covering consecutive same-pitch score events) | 0.15 |
| per repeat copy | 0.5 |

The cheapest hypothesis wins. Costs were hand-set and checked on train clips
only (0.982 on 400 train clips); no val tuning was done before freezing.

## Comparison and version rationale

| Aligner | Gold | Val F1 (935 clips) | Test F1 | Val runtime (6 workers) |
|---|---|---:|---:|---:|
| Frozen identity CRF fast v2 | raw note_map | 0.777 (20 train clips only) | — | — |
| Frozen identity CRF fast v2 | repaired | **0.990** | not evaluated | about 20 min |
| Structured DP v1 (frozen candidate) | repaired | 0.985 | **0.983** | about 75 s |

The earlier identity CRF was trained on ORN data and appeared weak on 9.2
(0.777), but that measurement used unrepaired gold. On repaired gold it is the
more accurate aligner on val (0.990 vs 0.985), mainly on substitutes (0.906 vs
0.862) and copies. The DP aligner needs no training and is about 16× faster.
It is the candidate that was frozen and tested, and it passed on test. It is
not claimed to beat the identity CRF: the comparable val evaluation favors
the CRF. Both remain weakest on missed-note detection (0.80–0.83 F1).

## Artifacts

| Artifact | SHA-256 |
|---|---|
| `ALIGNER_FREEZE.json` | `c660525a5710068af10bc7fa57d63a31d4521ee2498545b8ec1546af6628a00d` |
| `CANDIDATE_DP_V1.json` | `f4d8006052cfcf953ccf7031e115af166c31783c986194b6756f76c8111c5b96` |
| `TEST_DP_V1.json` | `dc8a627e9e4223a47a7a1b6bdb698071c2f4531111b6cebf44b617da12b3ac48` |

Other files: `val-dp-v1-default.json`, `val-identity-crf-baseline.json`,
`repair-v1/REPAIR_AUDIT.json`, `TEST_EVALUATIONS.jsonl`.
