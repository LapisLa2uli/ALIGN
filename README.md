# ALIGN / MusicEval

Clarinet performance-analysis workspace containing three cooperating Python projects:

| Directory | Responsibility | Detailed documentation |
|---|---|---|
| [`DataCreate/`](DataCreate/) | Ingest scores and recordings, run automatic alignment, review candidates, and build training bundles | [`DataCreate/README.md`](DataCreate/README.md) |
| [`synth-pipeline/`](synth-pipeline/) | Generate reproducible synthetic clarinet performances with exact error labels and note lineage | [`synth-pipeline/README.md`](synth-pipeline/README.md) |
| [`align-model/`](align-model/) | Transcribe notes, detect repetitions, align notes to the score, and classify note/rhythm errors | [`align-model/README.md`](align-model/README.md) |

The shared data and evaluation rules are specified in [`methodology.md`](methodology.md). That document is normative for label schema and metrics; component READMEs describe implementation, datasets, model versions, and run-specific hyperparameters.

## Current pipeline

1. Convert the performance to 22,050 Hz mono audio.
2. Transcribe written clarinet notes with Basic Pitch 0.4.0 and the repository's calibrated monophonic cleanup.
3. Detect repeated phrases from the transcription (Layer 1).
4. Align first-pass and repeated notes to the clean score with the contextual note aligner.
5. Classify wrong, extra, and missed notes (Layer 2).
6. Optionally detect conservative duration/rhythm errors from the same mapping (ALIGN Layer 3).
7. Present automatic candidates in DataCreate for human review. The current DataCreate bridge runs Layers 1–2 only; Layer 3 remains available through ALIGN directly.

Intonation-error output is currently disabled. The synthetic corpus contains intonation labels, but an acoustic audit found that many historical WAVs did not preserve the corresponding pitch bends.

## Production model bundle

DataCreate currently points to:

```text
align-model/runs/contextual-aligner-outputRaw_sf-1k/weights/
├── note_decoder.json                 # calibrated frozen Basic Pitch decoder
├── note_repetition.pt                # Layer 1 repetition candidate scorer
└── contextual_note_aligner.pt        # repetition-aware note-to-score aligner
```

These learned ALIGN-specific weights were trained only on `outputRaw_sf_10k`: 1,000 training bundles and 200 validation bundles. The frozen split contains 8,004 train, 999 validation, and 997 test-ID bundles; reported final comparisons use 200 held-out test bundles. See the complete registry and all historical model versions in [`align-model/README.md`](align-model/README.md).

## Pitch convention

- MusicXML, labels, transcriptions, and model targets use **written MIDI pitch**.
- Bb-clarinet audio uses **sounding pitch = written pitch - 2 semitones**.
- Loaders recover written pitch through `effective_audio_transpose` recorded in metadata. They must not infer pitch space from a dataset name.

## Setup

```powershell
cd "D:\stuff\Audio Evaluation\ALIGN"
conda env create -f DataCreate/environment.yml
conda activate MusicEval
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ./DataCreate
pip install -e ./synth-pipeline
pip install -e ./align-model
```

Basic Pitch and PESTO use an isolated environment because their TensorFlow/ONNX dependency pins differ:

```powershell
python -m venv align-model\.venv-amt-bench --system-site-packages
align-model\.venv-amt-bench\Scripts\pip install `
  -r align-model\requirements-amt-benchmark.txt `
  --extra-index-url https://download.pytorch.org/whl/cu124
```

## Common commands

```powershell
# Generate synthetic bundles
synth-pipeline --config synth-pipeline/config/rawdata_sf_10k.yaml `
  generate --count 10000 --workers 8 --seed 42

# Process and annotate a real take
datacreate run --score path/to/score.musicxml `
  --performance path/to/take.wav --sample-id take_001
datacreate serve

# Run the current note-first detector
align-model run --sample path/to/bundle `
  --weights align-model/runs/contextual-aligner-outputRaw_sf-1k/weights
```

## Reproducibility

- Frozen manifests under `align-model/runs/*/split.json` are the source of truth for train/validation/test membership.
- Run histories and evaluation JSON files under the same run directory are the source of truth for trained epochs, calibrated thresholds, and reported metrics.
- Basic Pitch and PESTO caches include source-audio SHA-256 hashes; edited WAVs cannot silently reuse stale activations.
- `performance_score.musicxml` and `note_map.json` may be used as synthetic supervision, but inference reads only the clean score and performance audio/features.
