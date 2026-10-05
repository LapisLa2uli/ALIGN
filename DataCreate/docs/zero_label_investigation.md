# Studio zero-label investigation (2026-10-04)

The Studio bridge is running the configured stack-v9 waveform inference and
passes its predictions to the shared feedback pipeline. It is not substituting
the empty human-annotation template. The frozen model and thresholds were not
changed in this fix.

## Saved takes

- `2f6745699b1347da88144d3a23c1ec48`, `ad399add8789471f867ec15e41067751`,
  and `d19a4cd745bd4b20b9b4247af44c240a` contain identical performance WAVs
  and identical Demo Scale scores. These are not three independent recordings.
- Fresh inference on that scale gives status `ok`, 49 reference notes,
  match fraction 0.960784, and zero labels. The aligner proposes two extra notes
  at 9.75 and 11.27 seconds. The trained extra-note gate scores them 0.1632
  and 0.4619, below its frozen 0.85 threshold, so both are withheld. One optional
  low-confidence decoded candidate is dropped by the aligner. This explains the
  empty result; it does not establish whether those passages were played correctly.
- `b6ab8d7b9362423d96c5ad2697b5467f` selects
  `a-cruel-angels-thesis-clarinet.musicxml` (583 canonical notes), against 87
  repaired decoded notes. The match fraction is 0.413793, below the candidate's
  0.45 minimum. The detector explicitly returns `alignment_uncertain`, withholds
  all feedback, and marks every score note unassessed. The previous Studio flow
  discarded that distinction and narrated the empty labels as a successful run.
- A separate fresh inference of that same waveform against the available
  `howls-moving-castle.musicxml` gives status `ok`, match fraction **0.908046**,
  and **11 labels: 8 wrong notes and 3 missed notes**. This strongly indicates
  an incorrect selected score in the newer take. The comparison is saved as
  `DataCreate/work/label-diagnosis/new-take-howls-comparison.json`; the original
  take and its selected score were not replaced. Predictions still require
  musical review and are not asserted to be ground truth.

## Fix and verification

`analysis_status.py` reads model assessment separately from the label array.
Studio blocks narration and speech retry for inconclusive alignment, returns an
actionable message, and suppresses misleading older narration in its API. The
shared narration entry point also rejects uncertain model documents and checks
the companion alignment for pipeline candidate input. Valid zero-label results
remain allowed, with confidence-filtering context shown in Studio.

66 Studio/feedback tests passed, including inconclusive-run blocking, old-result
recovery, confidence-filtering explanations, direct narration safeguards, and
existing audio-excerpt integration tests.

A controlled copy of the real scale changes only score note index 5 by one
semitone. Fresh waveform inference through `DataCreatePipeline.run_stage5`
(the same bridge used by Studio) emits one `wrong_note`, attached to
`note_0005`, at 5.2593–5.4683 seconds. Original samples are untouched. Diagnostic
outputs and the control are in `DataCreate/work/label-diagnosis/`.

This fixes loss of assessment status, not the experimental model's recognition
accuracy. v9 has documented false merges and weak real-recording recall; see
`align-model/STACK_V9.md`. Changing confidence thresholds simply to force nonzero
labels would not establish correctness.
