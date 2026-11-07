# Rhythm feedback: release-independent onset timing

Updated 2026-11-06. Revision: `aligned-rhythm-v2-onsets`.

The previous measured-offset path compared each note's audible sustain with its written duration. This caused normal articulation gaps, short phrase endings, and varying legato overlap to become rhythm suggestions, particularly with MuScriptor offsets. The correction uses the spacing between consecutive, unambiguously aligned attacks for both transcribers. Release times never determine the tempo estimate or the rhythm decision.

Short releases, legato overlap and final-note releases are not errors by themselves. The last note without a subsequent attack remains unassessed. Real early/late attacks can still be detected. The feedback label describes the interval to the next attack and its playback region ends at that attack. Existing guards exclude extra notes, copies, merged or duplicated mappings, skipped score intervals and uncertain alignments.

The deployed estimator remains the neighboring median with a 65-interval window, 0.20 log-ratio threshold and 60 ms minimum deviation. This is a change in what is assessed, not a blanket suppression of short timing intervals. The compatibility argument `measured_offsets` now records input capability only. The revision bump makes earlier feedback stale when the server checks freshness.

The oracle evaluator now defines gold using injected changes in consecutive onset intervals, rather than changes in audible duration. This follows the corrected product requirement; release-only injectors are negative controls. Its old duration-based reports remain historical and are not directly comparable to this run. Three explicit controls cover short releases, overlapping legato and shortened final notes.

The new experiment contains 110 symbolic clips with exact known alignment: 55 selection and 55 check clips. The unchanged deployed median policy achieves 63 true positives, 0 false positives and 11 false negatives on the check split (precision 100%, recall 85.1%). All 35 clean/articulation/gradual-tempo check controls have zero predicted errors. EWMA scores 32 true positives, 3 false positives and 42 false negatives; median remains selected. These are small synthetic results with oracle alignment, not measured real-audio accuracy.

Take 002 was republished from its existing full-context MuScriptor transcript. Its rhythm suggestions changed from 14 to 7. The transcript and note mapping are exactly unchanged. The remaining suggestions concern onset spacing; they have not been independently validated as performer mistakes. The human label file, metadata, score, full audio and trimmed audio are hash-protected and unchanged. Prior generated files are backed up under `publish002/backup/002/`.

The updated GUI on port 8766 was restarted and verified through a live Re-label request: HTTP 200, seven rhythm labels and no alignment regeneration. A subsequent GUI GET also reports seven suggestions. The separate original server on port 8765 was not restarted in this change; use <http://127.0.0.1:8766/> for the current implementation.

Validation covers release invariance for both estimators and both offset-availability modes, detection of genuinely early/late attacks even with short or overlapping releases, local tempo changes, ambiguous alignment boundaries, the feedback wrapper, GUI label validation and cache revision handling. The focused suite passes 78 tests, with one existing Starlette/httpx deprecation warning. Python compilation and whitespace checks pass.

Evidence files: `report.json`, `selection.json`, `bundles/`, `002_before.json`, `002_after.json`, `002_verification.json` and `live_verification.json`. `refresh_002.py` preserves the existing transcription while republishing the feedback; its output directory must be new before rerunning.
