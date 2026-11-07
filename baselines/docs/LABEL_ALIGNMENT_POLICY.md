# Current baseline label alignment: policy 3

Policy 3 is the default for new **PolyTune and LadderSym** label exports and evaluations. It uses the stored MuScriptor performance transcription and its ALIGN reference-score mapping to place baseline error intervals on score notes. Reference-note identity and error type remain the primary evaluation target.

The existing command now selects policy 3 without an extra flag:

```powershell
$env:PYTHONNOUSERSITE='1'
& baselines/envs/polytune/Scripts/python.exe baselines/scripts/label_datacreate_baselines.py `
  --model polytune `
  --pred-dir <prediction-directory> `
  --samples DataCreate/samples `
  --output <new-export-run-directory>
```

Use `--model laddersym` for LadderSym. Each prediction directory contains `<sample>/mix.mid`; `evaluated_ids.json` is used when present, otherwise IDs are discovered from those MIDI files. The command stages and validates the entire batch before publishing any labels, backs up existing baseline label files in the run archive, and writes `labels_polytune.json` or `labels_laddersym.json`. Human and AudioEval agent labels are not overwritten. Choose a new output directory for each run.

Each sample must contain `performance_audio.wav`, `verified_score.musicxml`, and a MuScriptor-backed `note_alignment_v2.json`. The converter requires matching audio and score hashes in the alignment provenance. A missing, stale, or different-transcriber cache raises an actionable error; there is no silent fallback to pitch-sequence alignment. Create the MuScriptor alignment using DataCreate's existing MuScriptor re-align flow first. `--alignment-root` can point at a separate set of sample folders containing those caches. `--labels-root` writes sample label files elsewhere for review or evaluation without publishing them into the live samples.

For **new Python callers**, use the current API:

```python
from label_datacreate_baselines_v3 import document_from_midi

document = document_from_midi(
    midi_path, score_path, model="polytune", sample_id="106",
    # Optional; by default these are beside the MusicXML:
    alignment_path=alignment_path,
    audio_path=audio_path,
)
```

The shared projection implementation lives in `baselines/common/transcription_label_alignment.py`, with no dependency on experiment folders. It preserves the tested policy-3 rules: temporal lookup for wrong notes, additional interpolated score gaps for missing notes, surrounding anchors for extras, and covered-note unions for repetitions. The predicted types, times, label counts, and repetition copy counts remain unchanged. Unsupported locations remain unlocated.

## Output and scoring contract

- `baseline_labeling.alignment_policy = 3` and `adapter_version = stored_transcription_lookup_v1` identify the adopted policy.
- `score_event_indices` and `core_note_ids` identify **exact affected notes**, including any gaps in a noncontiguous set.
- `context_score_event_indices`, the outer `score_part`, `note_ids`, and `pitches` provide the padded GUI context. Do not score those padded ranges or fill holes between affected notes.
- `score_part.core_start_note_index/core_end_note_index` bound the core for display; the explicit index set is authoritative for scoring.
- `location_evidence` records the supporting transcript events. The GUI serializer, undo data, and API preserve explicit identities and repetition copy counts. Manual score selections replace the evidence with a manual-selection marker.

Pass the label dictionaries directly to `datacreate.melody.match_note_wise_labels_detail` with `type_mismatch_credit=0` for strict scoring. It prefers explicit score indices. Do **not** pass v3 output through old experiment helpers that replace explicit index sets with contiguous `score_part` ranges. Unlocated and unclassified predictions must still count against precision; `baseline_labeling.unclassified_count` records the latter.

Policy 3 consumes allowlisted transcript timing, pitch, score spans, ignored-event flags, and score rhythm. It does not use stored error labels, relationship classes, or repetition decisions to assign locations. When evaluation labels come from the same MuScriptor/ALIGN source, report that shared-source dependence. This change does not add performance-pass identity or remove very short predictions; those remain known evaluation limitations.

## Historical reproduction

The v1/v2 import APIs remain unchanged for old frozen experiments. New evaluations must import **v3**, not those historical APIs. The command supports explicit `--alignment-policy legacy-v1` and `legacy-v2` overrides only for reproduction. The old `label_datacreate_baselines.py` module's `document_from_midi` and `main` functions retain their v1 behavior; executing the file as a command dispatches to the current v3 CLI.

The original entry-point file is preserved byte-for-byte in `baselines/legacy/policy12_20261106/label_datacreate_baselines.py`. Historical hashes refer to that archived content; the current entry point has intentionally changed to dispatch to policy 3. The original experiment results and policy comparison remain unchanged. No old result should be relabeled as a policy-3 result.

The policy comparison is in `experiments/polytune_alignment_policies_20261106/REPORT.md`. Its rules were promoted, not tuned further on the test results. Existing live baseline labels are not automatically regenerated merely by changing this default; the next export applies policy 3.

Production verification is recorded in [`reports/baseline_policy3_default_20261107/README.md`](../../reports/baseline_policy3_default_20261107/README.md): 42 exports across both models passed, and all 21 PolyTune mappings exactly reproduced the frozen policy-3 projections.
