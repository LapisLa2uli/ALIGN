# Seed-42 synthetic data generation

The requested generation job is supervised by
`synth-pipeline/scripts/generate_12k.py`. It runs the two original commands
in order, using 8 workers, seed 42, the FreePats SoundFont, and
`--skip-existing`:

- `synth-pipeline/output_10k_multi`: 10,000 procedural samples.
- `synth-pipeline/output_2k_rawdata`: 2,000 snippets from `RawData/Score`.

The requested 12,000 samples have been generated. The data recipe, 22,050 Hz
mono audio, scores, injected errors, and mel features are preserved. With the
user's authorization, all production `alignment.npz` archives omit only the
dense diagnostic `dtw_cost` array and carry `dtw_cost_omitted=True`. All other
NPZ members were verified byte-for-byte using SHA-256 before atomic replacement.
This saved **161.763 GB**. DTW computation and the full saved warping paths are
unchanged. Future synthetic generation uses this compact format by default;
`render.save_dtw_cost: true` restores the diagnostic matrix for debugging.

The measured corpus after compacting and repairing MIDI-facing maps is:

| Dataset | Samples | GB (decimal) | GiB | Performance hours |
| --- | ---: | ---: | ---: | ---: |
| Procedural scores | 10,000 | 72.024 | 67.077 | 115.982 |
| Existing-score snippets | 2,000 | 11.625 | 10.826 | 18.100 |
| Total | 12,000 | 83.648 | 77.904 | 134.082 |

These sizes include all files in the two output roots. Separate environments,
pilot samples, repair backups, logs and provenance are not included. The shared
volume has concurrent writers, so free space can change independently of this
job. The earlier 244.4 GB figure described a different full-matrix generation.
This new corpus is not the frozen legacy `align_v1` dataset, and its labels
must not be assigned the legacy audit's validity claims or split checksum.

## Runtime

Python environment: `align-model/runs/env` (Python 3.11).
`DataCreate`, `synth-pipeline`, and `tinysoundfont==0.3.7` are installed.
PortAudio headers and libraries are installed locally under
`align-model/runs/render-runtime`; PyAudio's runtime search path points to
that directory. No administrator installation was required. The original
`RawData.zip` was extracted without replacing differing existing files.
The generator reads the three MusicXML scores; it does not use the MSCZ
file as a MusicXML source. The bundled FreePats SF2 was already present.

Run artifacts are under `align-model/runs/data_generation_20260914/`:

- `status.json`: current phase, completed counts, free space, and process IDs.
- `supervisor.log`: progress and disk-pause messages.
- `procedural.log` / `raw_snippets.log`: full generation logs.
- `generation_manifest.json`, copied YAMLs, `input_files.json`,
  `environment.freeze.txt`, and `implementation.patch`: provenance.
- `postprocessing_implementation.patch`, `postprocessing_provenance.json`,
  and the manifest's `postprocessing` section: compact storage and MIDI-map fixes.
- `dtw_compaction.jsonl` and `dtw_compaction.summary.json`: per-file retained
  member hashes and measured storage recovery for all 12,000 archives.
- `midi_map_repair/`: original note-map backups, before/after hashes and counts.
- `dataset_summary.json`: generated only after both requested datasets are
  complete and inventoried. Its absence means the full job is not complete.

The supervisor pauses its generation process group below 15 GiB of free
space and resumes when at least 20 GiB is free. This protects partially
written bundles when other users or jobs consume the shared volume. It
does not reserve enough space for the entire remaining dataset.

To resume after a stopped supervisor, run from the repository root:

```bash
align-model/runs/env/bin/python -u synth-pipeline/scripts/generate_12k.py \
  --run-dir align-model/runs/data_generation_20260914
```

A file lock prevents two supervisors from using the same run directory.
Do not start another generation command against these output roots while
the existing supervisor is running or paused. Current complete-bundle
checks require nonempty scores, WAVs, MIDIs, labels, metadata, note maps,
candidate labels, alignment data, and both mel feature files.

## Validation and training suitability

Preflight generated 8 procedural and 3 raw-source samples under the run
directory's `pilot_multi` and `pilot_raw` directories, separate from the
production data. All 11 passed checks for audio format, mel features,
annotation schema, note index bounds, pitch convention, alignment arrays,
and NPZ CRC integrity. See `pilot_audit.json`. The existing 35 generator
tests passed; the expanded 15-test resume suite also passed.

The raw pilot reported mapping warnings for 2 of its 3 examples.
`render_validation.unmapped_performed_notes` must be audited before using
rendered-note correspondence supervision. A successful file-generation
run is not proof that every correspondence or acoustic error label is
correct. `pilot_intonation_audit.json` contains a limited YIN check of
intonation regions; it is not a full acoustic calibration test, and label
windows need not coincide with the exact detuned-note interval.

The initial final inventory found one empty rendered-note map
(`synth_WeberITAV_0953`). Its MIDI and audio contained all 24 performed notes;
music21's MIDI-to-score quantization had collapsed fast triplets into Chords
that the old Note-only scan ignored. The same parser also quantized timing
and split sustained events in other bundles. MIDI-facing maps were therefore
checked for all 12,000 clips and refreshed for 11,777 directly from the original
MIDI events, including tempo changes. The generation-time clean/performed
identities, deleted-note indices, scores, audio, error labels, features and
DTW paths were preserved. Original maps were backed up first. MIDI event
timing is now read without score quantization for newly generated samples too.
Error-label time windows in existing bundles were not rewritten by this repair.

After the storage and MIDI timing fixes, 56 synthetic/storage tests and 34
DataCreate alignment tests passed. An end-to-end compact-format smoke sample
also passed the bundle audit and NPZ CRC check. Its WAVs, MIDIs and mels were
byte-identical to the corresponding original seed-42 production sample.

The final `data_audit.json` reports **12,000 complete bundles, zero partial
bundles and zero artifact failures**, including NPZ CRC checks and comparison
of every rendered event's pitch/onset/offset with the original MIDI. All 12,000
archives use compact storage. The independent expected-ID inventory found no
missing, duplicate or unexpected sample IDs.

Mapping warnings remain for **354 procedural and 1,159 raw-source samples**:
some notated performance notes have no rendered-event correspondence. These
1,513 warnings require a supervision review before training selection; they
are not file-corruption failures. Neither this integrity audit nor the earlier
pilot establishes that all semantic labels or acoustic deviations are correct.

The generator calls DataCreate's chroma-based `run_alignment` directly.
The inherited `alignment_params.engine: note_first` in metadata comes from
the shared DataCreate configuration; it does not mean a learned ALIGN
model was used to produce these synthetic alignment files.

For a read-only audit of the completed data:

```bash
align-model/runs/env/bin/python synth-pipeline/scripts/audit_generated_data.py \
  --root synth-pipeline/output_10k_multi \
  --root synth-pipeline/output_2k_rawdata \
  --out align-model/runs/data_generation_20260914/data_audit.json \
  --verify-crc
```

The auditor reports partial bundles and mapping warnings independently
from malformed artifacts. Check the supervisor's expected sample counts
as well as this report before describing the dataset as complete.

These bundles export `note_map.json`, not legacy `note_labels.json`.
The baseline converter still requires a validated schema adapter before
training Polytune/LadderSym on this newly generated corpus. Do not silently
drop missing-note supervision or reuse the older corpus's audit exclusions.
