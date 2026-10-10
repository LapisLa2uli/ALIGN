# Methodology revision — 2026-10-06

Pulled the Overleaf project from a5ec89a to 35bcf88 before editing. This revision updates the method and connected descriptions. It does not rerun experiments or replace their numerical results.

## Implementation evidence

Paths are relative to the ALIGN repository.

- Deployment: `DataCreate/config/default.yaml` (v9) and `DataCreate/config/v10.yaml` (separate v10).
- Acoustic architecture: `align-model/src/alignmodel/transcription/mel_ctc_v3.py` and `mel_v1.py`.
- Selected settings: `align-model/runs/precision-v4/v5-dclike/config.json` and `history.json`; training in `align-model/scripts/train_mel_ctc_v3_realistic92.py`.
- Decoder: `align-model/src/alignmodel/transcription/ctc_decode_v2.py` and `transition_v7.py`.
- Boundary repair: `align-model/src/alignmodel/transcription/same_pitch_v2.py`, `same_pitch_v3.py`, and `align-model/src/alignmodel/joint/stack_v10.py`.
- Passage policy: `align-model/src/alignmodel/joint/passage_v1.py` and `stack_v9_passage.py`. The current earliest-plausible policy supersedes older ambiguity-abstention prose in PASSAGE_V1.md.
- Alignment: `align-model/src/alignmodel/joint/robust_dp_aligner_v2.py`, `robust_dp_aligner_v3.py`, `robust_dp_aligner_v5.py`, `restarts_v4.py`, and `ornament_mapper_v1.py`.
- Verification and export: `align-model/src/alignmodel/joint/presence_verifier_v1.py`, `robust_dp_aligner_v3.py`, and `stack_v7.py`.
- Candidate manifests: `align-model/runs/precision-v4/CANDIDATE_STACK_V6.json`, `align-model/runs/stack-v9/CANDIDATE_STACK_V9.json`, and `align-model/runs/stack-v10/CANDIDATE_STACK_V10.json`.

## Scope and version distinctions

- The selected acoustic checkpoint uses temporal width 256; the class default of 192 is not the trained configuration.
- The aligner uses configured edit costs and a second timing pass, not a learned neural path scorer.
- v9 is the application default. v10 is an experimental boundary restoration extension; neither is described as promoted or independently validated.
- Current feedback does not emit rhythm, intonation, timbre, or separate content errors within replayed copies.
- Existing result tables and case studies retain their historical configurations. No new accuracy claims are inferred from them.
- Historical Basic Pitch architecture and training details remain in the appendix, separately from current implementation settings.
- The replacement pipeline diagram is editable LaTeX in `figures/pipeline-current.tex`.
- Windows checkout converted line endings in the two newly synced alignment PDFs, invalidating their byte offsets. Their original Git blobs were restored exactly; `.gitattributes` now preserves PDF bytes. Figure content and committed PDF blobs are unchanged.
