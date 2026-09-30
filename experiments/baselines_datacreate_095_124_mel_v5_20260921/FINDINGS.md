# DataCreate 095-124 mel-v1 / error-heads-v5 test

Official exclusive note-wise scoring. Timestamp IoU is diagnostic only.

## What ran

- Freeze: `align-model/runs/eval-datacreate-095-124-mel-v5-20260921/` (30/30, labels not read).
- Stack: `cursor_agent_mel_transcriber_v1_error_heads_v5` (mel transcriber v1, joint decoder, error-heads v5 `hybrid_max_f1`).
- Polytune / LadderSym were not run.

## Gold

- 25 auditable; unevaluable 095, 096, 097, 098, 100.
- Manual auditable: 099 only (F1 0.0).
- Agent auditable: 101-124. Working gold copied from this family, not independent human annotation.

## Official note-wise (25 auditable clips)

- Precision 0.9009, recall 0.9905, F1 **0.9436**, 95% clip-bootstrap [0.8682, 1.0000]
- Credit 209 / predicted 232 / gold 211 (209 full, 0 half)
- Agent-source 24 clips: F1 0.9676 (recall 1.0; extra predictions, mainly clip 106)
- Per-type F1: wrong 1.000, missed 0.971, extra 0.979, rhythm 0.889, repetition 0.913

The earlier Basic Pitch / error-heads-v2 freeze scored 0.0975 on the same gold because it was a different stack than the agent copies. This 0.9436 is experimental self-agreement, not a held-out human result and not a replacement for the synthetic 358-clip table.
