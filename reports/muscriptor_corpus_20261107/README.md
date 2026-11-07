# DataCreate MuScriptor corpus refresh

This run refreshed the transcription, score alignment, and agent labels for all 124 numbered bundles plus `demo_001`.

## Results

- 125 bundles have installed MuScriptor-medium transcriptions and `note_alignment_v2.json` outputs.
- 124 numbered transcription outputs are provenance-valid; 108 were generated in the final batch and 16 were reused because their existing output already matched the current model, written-pitch shift, score hash, and audio-context signature.
- Audio context: 45 bundles were transcribed from a verified full recording and projected onto the trimmed timeline; 80 used the available trimmed/current recording because no verifiable full recording was present.
- Alignment status: 105 `ok`; 20 `alignment_uncertain` (listed below). The uncertain status is preserved as a review signal; those samples were not silently treated as confident matches.
- All 125 bundles were relabeled from the refreshed alignment. The numbered relabel report contains 485 rhythm, 425 wrong-note, 106 extra-note, 24 missed-note, and 66 repetition labels; `demo_001` adds three labels.
- Rhythm labels use inter-onset timing and explicitly exclude note-release duration, so note/legato ends are not classified as rhythm errors solely because the release does not fill the nominal score duration.
- Protected `labels.json`, audio, and verified scores were byte-for-byte unchanged (`protected_files_unchanged: true`).

The aligner also received a boundary fix in `robust_dp_aligner_v3.py`: onset-window reductions are clamped to available frames, preventing a short recording whose final event lies beyond the onset array from aborting the corpus publish.

## Alignment and piece audit

Each row records the MusicXML work title, the exact score part, expected metadata measure range, aligned measure range, passage selection, match fraction, and audio context:

- [alignment_audit.json](./alignment_audit.json)
- [alignment_audit.csv](./alignment_audit.csv)

The corpus contains these score works/parts: `Title` / `1stCl`, `Concerto in A Major, K.622 for Clarinet and Orchestra` / `Clarinette en Sib`, `Howls moving castle` / `Kèn Clarinet Bb`, `Spirited Away Medley for Clarinet` / `Bb Clarinet`, `Erstes Konzert` / `Solo Clarinet`, and `Demo Scale` / `单簧管`. Ten files have no MusicXML work-title and are reported as `Untitled score`; their part is `Clarinette en Sib`.

The 20 review samples are: `047`, `050`, `053`, `055`, `056`, `058`, `059`, `060`, `061`, `062`, `063`, `064`, `065`, `066`, `067`, `101`, `105`, `114`, `117`, and `123`. Most have the correct expected measure range but low match fraction; `055`, `056`, `060`, `061`, `063`, `064`, `065`, `066`, and `067` have no confident bounded passage and therefore no mapped score events.

## Reproducibility artifacts

- [transcription_manifest.json](./transcription_manifest.json)
- [publish/manifest.json](./publish/manifest.json)
- [publish_demo/manifest.json](./publish_demo/manifest.json)
- [relabel/REPORT.json](./relabel/REPORT.json)

Focused aligner validation: `17 passed` (`test_stack_v7.py`, `test_passage_v1.py`).
