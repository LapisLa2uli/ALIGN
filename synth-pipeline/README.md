# Synth pipeline

Generate **MusicXML clarinet scores** and ALIGN sample bundles whose performance audio contains a known error (wrong, missing, or extra note, a rhythm error, or an intonation / tuning error). After any of those errors, there is a configurable chance the audio **repeats the measure(s)** that contain it, then continues.

This is separate from DataCreate Stage 9, which corrupts an existing score in place. Here the **clean score stays correct**; only the synthesized performance is wrong.

## Setup

Use the same conda environment as DataCreate (`MusicEval`):

```powershell
cd "D:\stuff\Audio Evaluation\ALIGN"
conda activate MusicEval
pip install -e ./DataCreate
pip install -e ./synth-pipeline
```

MIDI is written with **music21**. Audio is rendered with tinysoundfont and a **clarinet SoundFont** you choose (`freepats` by default). See [`soundfonts/README.md`](soundfonts/README.md).

## Usage

Procedural original scores (default):

```powershell
cd "D:\stuff\Audio Evaluation\ALIGN\synth-pipeline"
synth-pipeline fetch-soundfonts
synth-pipeline list-soundfonts
synth-pipeline generate --count 10 --soundfont freepats
synth-pipeline generate --count 50 --workers 5 --soundfont freepats
synth-pipeline generate --count 5 --soundfont u220
synth-pipeline generate --count 5 --soundfont mcb
synth-pipeline generate --count 20 --seed 42 --output ../DataCreate/samples/synthetic
```

Corrupt existing MusicXML (still rendered as clarinet):

```powershell
synth-pipeline generate --score ../RawData/Score/001.musicxml --count 5
synth-pipeline generate --score ../RawData/Score --count 8
```

`--config` selects a YAML file. Generation knobs (`measures_min`/`max`, keys, meters, tempo, pitch range) and error weights / `repetition_prob` live in `config/default.yaml`.

## Output

Each sample is an ALIGN bundle:

```
output/synth_gen_0010/
├── verified_score.musicxml      # clean notation (ground truth)
├── performance_score.musicxml   # errored (and maybe repeated) render source
├── reference_audio.wav          # clarinet, clean score
├── performance_audio.wav        # clarinet, with the injected error
├── labels.json                  # source: synthetic, including repetition links
├── candidates.json
├── alignment.npz
├── performance_mel.npy
├── reference_mel.npy
└── metadata.json
```

`labels.json` uses the DataCreate taxonomy: `wrong_note`, `missed_note`, `extra_note`, `rhythm_error`, `intonation_error` (same written pitch, audio detuned by cents via MIDI pitch bend; `deviation_cents` is stored), and `repetition` with `repeats_label_range` pointing at the first pass of the bad measure(s). Any error type can trigger that restart (`repetition_prob`).

Open the output root in the DataCreate annotator (`datacreate serve`) like any other sample directory if you copy or generate into `DataCreate/samples/`.
