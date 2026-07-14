# MusicEval Data Creation Pipeline

Local pipeline for turning MusicXML/PDF scores plus performance recordings into labeled training-data bundles for music performance error detection.

## Setup

### 1. Conda environment

```powershell
cd "D:\stuff\Audio Evaluation\ALIGN\DataCreate"
conda env create -f environment.yml
conda activate MusicEval
pip install -e .
```

### 2. External tools (configure in `config/default.yaml`)

| Tool | Purpose | Windows example path |
|------|---------|---------------------|
| **MuseScore 4.2+** | Reference audio synthesis (Stage 3) | `C:/Program Files/MuseScore 4/bin/MuseScore4.exe` |
| **Audiveris** | PDF → MusicXML OMR (Stage 2) | `C:/Program Files/Audiveris/Audiveris.exe` |

Windows installs use the jpackage layout (`Audiveris.exe`, `app/audiveris.jar`, `runtime/`). Set either `paths.audiveris` to the `.exe` or `paths.audiveris_home` to the install folder.

Set `paths.musescore`, `paths.audiveris`, and `paths.soundfont` in `config/default.yaml`. Defaults point at the standard MuseScore 4 install and its bundled `MS Basic.sf3` SoundFont.

> **Stage 3 reference audio:** MuseScore’s direct WAV export is unreliable from Python on Windows (exit code 1331). The pipeline exports MIDI via MuseScore, then renders WAV with `tinysoundfont` and the configured SoundFont. MuseScore **4.2+** is still required for MIDI export.

## Pipeline stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `stage1_ingest` | Validate MusicXML/MXL |
| 2 | `stage2_omr` | Audiveris PDF → draft MusicXML + manual correction |
| 3 | `stage3_reference` | MuseScore MIDI export + SoundFont render → `reference_audio.wav` |
| 4 | `stage4_performance` | Ingest/record performance audio |
| 5 | `stage5_alignment` | Chroma/CQT DTW + candidate error detection |
| 6 | Web UI | NLE-style annotation (`datacreate serve`) |
| 7 | `stage7_features` | Log-mel spectrograms + alignment.npz |
| 8 | `stage8_bundle` | Final sample directory layout |
| 9 | `stage9_synthetic` | Programmatic score corruption + auto-labels |

## Usage

### Batch mode (Stages 1–5, 7 — no UI)

Single sample:

```powershell
conda activate MusicEval
datacreate run --score fixtures/demo_scale.musicxml --performance path/to/recording.wav --sample-id demo_001
```

**Batch range** — process many performance takes against one shared MusicXML (e.g. partial recordings `001`–`014` in `RawData/Audio/`):

```powershell
cd "D:\stuff\Audio Evaluation\ALIGN\DataCreate"
datacreate batch-range --from 1 --to 14
# or explicitly:
datacreate batch-range --score "../RawData/Score/001.musicxml" --audio-dir "../RawData/Audio" --from 001 --to 014
```

Each ID becomes a sample folder (`samples/001/`, `samples/002/`, …). Already-processed IDs are skipped unless you pass `--no-skip-existing`. Supported audio extensions: `.mp3`, `.m4a`, `.wav`, `.flac`, etc.

Configure default raw-data paths in `config/default.yaml`:

```yaml
paths:
  raw_data_score: ../RawData/Score
  raw_data_audio: ../RawData/Audio
```

Output lands in `samples/<sample_id>/`:

```
sample_<id>/
├── verified_score.musicxml
├── performance_audio.wav
├── reference_audio.wav
├── performance_mel.npy
├── reference_mel.npy
├── performance_mel_preview.png
├── reference_mel_preview.png
├── alignment.npz
├── candidates.json
├── labels.json
└── metadata.json
```

### Annotation UI (Stage 6)

```powershell
datacreate serve
```

Open http://127.0.0.1:8765 — zoomable waveform with draggable regions (wavesurfer.js), score view (OpenSheetMusicDisplay), candidate confirm/reject workflow.

Use the **Batch process** bar to run an ID range (e.g. 1–14) against the shared score, then pick any **Sample ID** from the dropdown to segment, trim, and label that take.

**Partial-piece workflow:** If your recording covers only part of a long score, use the score toolbar to pick a measure range (e.g. 12–48). Click **Apply segment & regenerate reference** to extract that slice from the full MusicXML and synthesize matching reference audio. Use the green **trim** handles on the waveform to discard leading/trailing silence, then **Apply trim & re-align**.

### Synthetic batch mode (Stage 9)

```powershell
datacreate-synthetic --score fixtures/demo_scale.musicxml --count 3
```

### Schema validation

```powershell
datacreate-validate samples/
datacreate-validate samples/demo_001/labels.json
```

### Resume mid-annotation

```powershell
datacreate resume --sample-id demo_001
datacreate serve
```

## Configuration

All tunables live in `config/default.yaml`:

- **taxonomy** — closed label enum (add types here, no code change needed)
- **alignment** — DTW band, ms/cents thresholds, feature type
- **mel** — spectrogram parameters stored in `metadata.json`
- **paths** — binary locations, sample roots
- **review.sampling_rate** — inter-annotator review fraction

## Labels schema

See `DataCollectionPipelinePrompt.md` for full `labels.json` / `candidates.json` schema (version `1.1`).

Key fields per label: `id`, `source`, `start_time`, `end_time`, `type`, `severity`, `deviation_cents`, `deviation_ms`, `repeats_label_range`.

`source` values: `auto`, `auto_confirmed`, `auto_edited`, `auto_rejected`, `manual`, `synthetic`.

Stage 5 DTW auto-candidates can be any of: `wrong_note`, `intonation_error`, `rhythm_error`, `missed_note`, `extra_note` (not rhythm-only). Timing, warping-slope, and high residual all map to `rhythm_error`; missed/extra are unmatched DTW frames. Samples that look rhythm-heavy after re-align are often threshold-driven rather than a missing detector.

## Adding a new error type

1. Add the type string to `taxonomy` in `config/default.yaml`
2. Bump `schema_version` if field requirements change
3. Run `datacreate-validate` on existing corpus

## Fixtures

```powershell
python scripts/create_fixture_score.py
```

Generates `fixtures/demo_scale.musicxml`. Process it end-to-end once MuseScore is configured.
