# V9 passage-location runtime revision

`v9-passage-v1` leaves the frozen v9 model, candidate hashes and calibration
unchanged. The DataCreate publisher and `scripts/run_stack_v9.py` now use
`joint/stack_v9_passage.py`.

1. Infer mel outputs once; decode and repair same-pitch boundaries as in v9.
2. Match reliable transcribed pitches against the full canonical score with
   confidence-weighted subsequence edit distance. Two rolling DP rows give
   O(score length) memory; unmatched score prefixes and suffixes are free.
3. Retain up to four distinct locations; add eight context notes on each side.
4. Run the versioned v5 detailed aligner on each window, then compare its
   normalized alignment cost and retrieval similarity. Similar independent
   locations produce `alignment_uncertain`, with no labels.
5. Offset all local event spans, missed-note identities, deletion sets and
   repetition ranges back to the original canonical index before feedback.
   Unplayed outer notes remain unassessed. Context padding is not a missing-note
   claim. The input MusicXML is never changed.

The v5 aligner reuses v4 operations and evidence gates but replaces the eager
full-score repeat grammar with a bounded iterator. It retains local acoustic
restart proposals first, then measure spans ranked by expected performed length.
Limits: 512 score notes/window, 4 windows, 1,024 hypotheses/window, 2,048 expanded
template notes and 2 million DP cells/hypothesis. Locator work is capped at
20 million score/performance cells. Oversized or unsupported searches abstain;
there is no unbounded fallback. Supplied excerpts <=128 notes skip retrieval.

Limitations: this assumes the existing monophonic written B-flat-clarinet pitch
convention. Identical passages can remain ambiguous; upload a specific excerpt
to resolve this. Long performances covering >512 score notes need segmentation
or a future streaming aligner. Widely separated jumps are not supported by a
single passage window. Locator confidence is heuristic, not calibrated accuracy.
V9's known permissive same-pitch merging behavior remains unchanged.

Verification (2026-10-05): the previously problematic Studio take
`7bc61b3e6fcd48fb91e3a9f6a893dc00` has 3,934 canonical notes across 813 measures.
An isolated fresh waveform run completed in 16.59 seconds, peak working set
1.353 GiB and peak sampled private memory 2.601 GiB. It retrieved core notes
782–937 (measures 172–192), with a padded window of 172 notes; detailed matching
extended into the padding. This is a resource/functional check, not human-label
accuracy validation. The previous live worker was observed around 14 GiB resident
and 26–36 GiB private; that was not a controlled peak benchmark.

DataCreate 095 completed in 8.83 seconds, peak working set 1.353 GiB. Its two
repetition and two wrong-note labels were retained. Labels >=095 were not used
as evaluation truth.

All 61 focused passage, alignment and DataCreate integration tests passed in
MusicEval. Both frontend scripts passed Node syntax checks. A live POST to
`/api/samples/095/re-label` on port 8765 regenerated the alignment with provenance
`v9-passage-v1`, retained both repetition labels and both wrong-note labels, and
preserved the audio, score and human-label hashes. Its verification record is
`DataCreate/work/passage-live-20261005/095-relabel-verification.json`.

Reproduce resource checks with MusicEval:

```powershell
python align-model/scripts/profile_passage_ui.py --sample <sample-folder> --output <new-output-folder>
```

The profiler copies only inputs into a new workspace directory, runs the actual
single-sample publisher, records Windows working-set/private memory, and verifies
the inputs and human labels remain unchanged. Reports are under
`DataCreate/work/passage-profile-20261005/`.
