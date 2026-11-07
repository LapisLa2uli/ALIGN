# Policy 3 default: verification

Completed 2026-11-06 UTC. The existing `baselines/scripts/label_datacreate_baselines.py` command now defaults to the stored MuScriptor transcription/ALIGN lookup for both PolyTune and LadderSym. The current Python API is `label_datacreate_baselines_v3.document_from_midi`; historical v1/v2 imports remain available for reproducing old experiments.

## Production checks

- Exported DataCreate 100–120 for both models through the established command, without an alignment-policy override: **42 valid documents**.
- PolyTune score indices, predicted types, timestamps, and repetition copy counts exactly matched all **21 frozen policy-3 projections**.
- Strict reference-note/type scoring reproduced **12 matches / 365 predictions / 76 pseudo-gold labels**, F1 **5.4422%**, on the same 17 scored recordings. Samples 101, 105, 114, and 117 remain excluded by the frozen teacher abstention policy.
- Both models preserved native-derived label types, timestamps, counts, and repetition copy counts.
- Verified hashes of **623 protected files**, including live label files, model checkpoints, frozen predictions, and original comparison artifacts, remained unchanged.
- Verified the archived v1 source retains its original SHA256.

These checks reused saved model predictions. No model training or inference was performed. Review exports are under `labels/<sample>/labels_polytune.json` and `labels/<sample>/labels_laddersym.json`; they were not published into live sample folders.

The prior comparison used AudioEval+MuScriptor pseudo-gold, and policy 3 shares its transcription/alignment source. This is a parity check of the promoted implementation, not new independent evidence of accuracy.

## Automated checks completed during implementation

- Baseline projection, native adapter, and exporter tests: **22 passed**.
- DataCreate label-source and exact-identity API round-trip tests: **8 passed**.
- Browser JavaScript syntax check and serialization/undo checks: **passed**.

Coverage includes missing/stale caches, wrong transcriber, invalid score spans, no partial publication when batch preparation fails, unchanged error predictions, missing-note interpolation, backward/repeated passages, noncontiguous score identity, and GUI/API preservation.

`verification.json` records input/source hashes and per-sample checks. `verify_promotion.py` contains the verification procedure; its export directories must be fresh for a new run because the production CLI refuses to overwrite a completed run manifest.

See [the policy and usage guide](../../baselines/docs/LABEL_ALIGNMENT_POLICY.md) for current commands, exact scoring fields, and cache requirements.
