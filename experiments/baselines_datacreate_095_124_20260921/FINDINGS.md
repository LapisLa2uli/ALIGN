# DataCreate 095-124 three-system test

Official exclusive note-wise scoring on clips 095-124. Timestamp IoU is diagnostic only and was not used for ranking.

## What ran

- AudioEval freeze: `align-model/runs/eval-datacreate-095-124-20260921/` (30/30 succeeded, labels not read during freeze).
- Stack: frozen Basic Pitch candidates, joint aligner/decoder, error-heads v2. This is not the synthetic exact-32 combined pipeline.
- Polytune / LadderSym: 16 kHz evaluation audio prepared at `baselines/data/datacreate_095_124_20260921`. ALIGN-trained checkpoints were not on this machine; GPU restore host `i-2.gpushare.com` refused connections; Hugging Face author checkpoints timed out. Those two systems are unscored.

## Gold

- 30 clips requested.
- 25 auditable under canonical score-event identity.
- Unevaluable: 095, 096, 097, 098, 100 (missing/invalid identity).
- Manual-source auditable: 099 only.
- Agent-source auditable: 101-124. These labels were copied from AudioEval-family agent proposals after empty human files. They are working gold, not independent human annotation.

## AudioEval official note-wise (25 auditable clips)

- Precision 0.1108, recall 0.0871, F1 **0.0975**, 95% clip-bootstrap [0.0568, 0.1391]
- Credit 19.5 / predicted 176 / gold 224 (19 full, 1 half)
- Per-type F1: wrong 0.163, missed 0.000, extra 0.054, rhythm 0.000, repetition 0.123
- Gold supports: wrong 21, missed 33, extra 47, rhythm 5, repetition 118

This is experimental. It does not replace the synthetic 358-clip table and must not be promoted as a human-held-out result.
