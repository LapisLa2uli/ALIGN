# Dataset 9.2 transcriber result

## Decision

The frozen mel CTC transcriber **passed** the 0.95 target on the held-out 9.2
test split.

- Written-pitch sequence LCS F1: **0.965128**
- Precision / recall: 0.973813 / 0.956597
- Clip bootstrap 95% interval (2,000 replicates): **0.963715–0.966395**
- Procedural / RawData F1: 0.961750 / 0.966999
- Predicted / gold notes: 257,648 / 262,285 over 2,028 clips

The metric is score-agnostic: the longest common subsequence of predicted and
gold written pitches per clip, micro-averaged. It ignores timing. It is the
metric used for the earlier 9.2 transcriber comparison. It is not the official
exclusive note-wise score-event F1, which needs a score-aware mapper.

## Split

`split.json` was frozen before any 9.2 training (`FREEZE.json`, split SHA-256
`1cb5acc9…6a`). RawData snippets overlap within a source score, so whole
scores are assigned to one split.

| Split | Clips | Scores |
|---|---:|---|
| Train | 6,864 | 001, MozartClConcertoA, howls-moving-castle, spirited-away + 4,000 procedural |
| Val | 1,108 | Weber1 + 400 procedural |
| Test | 2,028 | WeberITAV, a-cruel-angels-thesis + 600 procedural |

Test audio and `note_map.json` hashes were recorded at the freeze and verified
at evaluation. The test split was evaluated once, for this candidate
(`TEST_EVALUATIONS.jsonl`).

## Version rationale

**Frozen Basic Pitch 0.4.0** (no 9.2 training) scored LCS F1 0.811 on a
250-clip 9.2 sample. **Mel transcriber v1 epoch 018**, trained on earlier
FreePats data, scored 0.994 on its own synthetic validation but only 0.756 on
9.2 val (best decode, confidence 0.70). The gap is domain shift to Muse Sounds
plus phone-style degradation.

Training on 9.2 needed frame targets, but `note_map` timestamps come from the
music21 MIDI render, not the Muse Sounds audio. On 40 train clips, onsets were
off by a median 0.48 s, with about 0.6 s left after a linear time fit
(`timing-diagnostic.json`). The gold pitch order does match the audio, so each
clip's gold sequence was placed on frames by forced Viterbi alignment over the
current model's posteriors (`forced_align_v1.py`). With the epoch-018 aligner,
0.35% of about 1.0 million gold notes were skipped. With the round-1 aligner,
0.13% were skipped. So the audio realizes essentially the whole gold sequence,
including renderer ornament notes.

**Round 1** fine-tuned epoch 018 on the first alignment (degraded + clean
Muse Sounds audio). Val F1 rose to 0.894. It remained recall-limited: 31% of
notes next to a same-pitch neighbor and 56% of notes under 80 ms were missed,
because onset peaks were broad and the decoder merged repeated notes
(`val-r1-errors.json`).

**Round 2 (promoted)** re-aligned targets with the round-1 model and added a
CTC head over 49 written pitches plus blank on the shared encoder. It trains
on the pitch tokens whose aligned midpoint falls inside each crop, jointly
with the frame losses at weight 0.5. CTC separates repeated notes with blanks
and does not need exact onset frames. Full val F1 is 0.962 (0.9615 with plain
greedy decoding). The 0.3 blank scale was chosen on val. Test F1 0.965
confirms it on unseen scores.

## Artifacts

| Artifact | SHA-256 |
|---|---|
| `CANDIDATE_CTC_R2A.json` | `cd318cabc7640dbec46383e2cb5f73907c476044db55897893d88d7be3d717f7` |
| `TEST_CTC_R2A.json` | `3f7821754c4f775fabc4f5c1205dbf76c948a9c51421b0f45128475524a2c65f` |
| `ctc-r2-a/best.pt` | listed in the candidate file |

Code: `scripts/freeze_realistic92_transcriber_split.py`,
`scripts/build_realistic92_aligned_cache.py`,
`scripts/train_mel_ctc_realistic92.py`, `scripts/eval_realistic92_ctc.py`,
`src/alignmodel/transcription/forced_align_v1.py`,
`src/alignmodel/transcription/mel_ctc_v1.py`.
