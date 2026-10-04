# ALIGN / MusicEval

Clarinet performance-analysis workspace containing three cooperating Python projects:

| Directory | Responsibility | Detailed documentation |
|---|---|---|
| [`DataCreate/`](DataCreate/) | Ingest scores and recordings, run automatic alignment, review candidates, and build training bundles | [`DataCreate/README.md`](DataCreate/README.md) |
| [`synth-pipeline/`](synth-pipeline/) | Generate reproducible synthetic clarinet performances with exact error labels and note lineage | [`synth-pipeline/README.md`](synth-pipeline/README.md) |
| [`align-model/`](align-model/) | Transcribe notes, detect repetitions, align notes to the score, and classify note/rhythm errors | [`align-model/README.md`](align-model/README.md), [`align-model/TRAINING.md`](align-model/TRAINING.md), [`align-model/HYPERPARAMETERS.md`](align-model/HYPERPARAMETERS.md) |

Start with [`DOCUMENTATION.md`](DOCUMENTATION.md). It covers the sample layout, how to read `labels.json` and the other output files, the official note-wise score, dataset versions, and which model is which.

[`methodology.md`](methodology.md) is the specification for the label schema and the official score. Component READMEs describe implementation and historical runs. Where they still call the contextual aligner or pitch-list matching current, `DOCUMENTATION.md` and the code it cites take precedence.

Two systems are in use, and they are not the same:

- The annotation GUI aligns with the joint path CRF (`paths.note_alignment_checkpoint` in `DataCreate/config/default.yaml`) and shows `note_alignment_v2.json`.
- Sealed experiments use stack v6, frozen in `align-model/runs/precision-v4/CANDIDATE_STACK_V6.json`. Those alignments are under `align-model/runs/precision-v4/dc-v6/` and are not what the GUI displays.

The contextual-aligner weights in `align-model/runs/contextual-aligner-outputRaw_sf-1k/weights/` remain the fallback if the joint checkpoint is missing. They were trained on dataset 6.0.

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

The [practice studio](DataCreate/docs/practice_studio.md) is available at
`http://127.0.0.1:8765/studio` after `datacreate serve`: select a MusicXML score,
record with a live mel visual, then stop to run analysis and receive spoken MP3
feedback. Configure narration and speech providers before recording.

Spoken feedback from existing label results is available through
`datacreate-feedback` (302.AI → Fish Audio MP3). See the
[setup and usage guide](DataCreate/docs/spoken_feedback.md) and
[feedback configuration](DataCreate/config/feedback.yaml).

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

# Train (see align-model/TRAINING.md for every flag)
align-model train-stages --help
python align-model/scripts/train_note_repetition.py --help
```

## Reproducibility

- Frozen manifests under `align-model/runs/*/split.json` are the source of truth for train/validation/test membership.
- Run histories and evaluation JSON files under the same run directory are the source of truth for trained epochs, calibrated thresholds, and reported metrics.
- Basic Pitch and PESTO caches include source-audio SHA-256 hashes; edited WAVs cannot silently reuse stale activations.
- `performance_score.musicxml` and `note_map.json` may be used as synthetic supervision, but inference reads only the clean score and performance audio/features.
