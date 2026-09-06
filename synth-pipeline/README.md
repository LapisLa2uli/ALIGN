# Synth pipeline

Generate **MusicXML clarinet scores** and ALIGN sample bundles whose performance audio contains known faults. The **clean score stays correct**; only the synthesized performance is wrong. This is the volume path for synthetic gold. It is separate from DataCreate Stage 9, which corrupts an existing score in place.

Gold is a **score part** on `verified_score.musicxml` (schema **1.2**), not the wall-clock interval of the fault. Times stay on the file for the annotator. Full methodology: [`../methodology.md`](../methodology.md).

## What is planted

A clip draws **1–8 content errors** (`per_clip_min` / `per_clip_max`) from five equally weighted types. They do not overlap in time.

| Type | Performance change | Gold core (then 1–2 notes of pad on each side) |
|------|--------------------|------------------------------------------------|
| `wrong_note` | ±1 or ±2 semitones, **or** a squeak MIDI **C6–A7** (`squeak.prob`) | The written note that was changed |
| `missed_note` | Written note replaced by a rest | That written note |
| `extra_note` | Split a note; insert a neighbor or a C6–A7 squeak | Clean notes **before and after** the insert (not the extra itself). Last-note insert: last written note only |
| `intonation_error` | Same pitch class; MIDI pitch bend 40–80 cents, sometimes a run of up to 4 notes | The detuned written note(s) |
| `rhythm_error` | One of the kinds below | The written note(s) whose timing changed |

**Rhythm kinds** (`rhythm_kinds`), tried in random order until one succeeds:

- `late_start` — rest inserted before the note; offset unchanged
- `early_start` — onset steals a preceding rest
- `late_end` — offset steals a following rest
- `early_end` — note shortened; rest fills the tail
- `tempo_change` — MetronomeMark at 0.68–0.82× or 1.2–1.4× for 1–2 measures, then restore
- `uneven` — 3–4 durations in a bar reweighted so the bar still sums

**Repetition** is not one of the 1–8 draws.

1. After the content errors, with `repetition_prob` (0.80 in the 10k config) replay the measure(s) that contain them. This is the usual case.
2. If that did not fire, with `standalone_repetition_prob` (0.20) replay a random sounding measure on its own.

`repeat_extra_copies_weights`: `1` = two plays (0.7), `2` = three plays (0.3). Immediately before the copy, insert a silent **0.2–1.0 s** gap (`repeat_gap_seconds`) so the restart sounds like the player stopping to adjust. The gap is not part of the gold melody. `repeats_label_range` is the first-pass span. Repeated-pass copies of content labels are **not** gold.

Padding is `melody_pad_notes: [1, 2]`, chosen per clip, clamped at the phrase ends (no wrap).

## Setup

Use the same conda environment as DataCreate (`MusicEval`):

```powershell
cd "D:\stuff\Audio Evaluation\ALIGN"
conda activate MusicEval
pip install -e ./DataCreate
pip install -e ./synth-pipeline
```

MIDI is written with **music21**. Audio is rendered with tinysoundfont and a **clarinet SoundFont** (`freepats` by default). See [`soundfonts/README.md`](soundfonts/README.md).

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

**10,000 multi-error clips** (1–8 errors, equal type weights, squeaks, rhythm kinds, after-error and standalone repetition, 0.2–1 s restart gap):

```powershell
cd "D:\stuff\Audio Evaluation\ALIGN\synth-pipeline"
synth-pipeline generate --config config/multi_error_10k.yaml --count 10000 --workers 8 --soundfont freepats --seed 42 --output ./output_10k_multi
```

Corrupt existing MusicXML (still rendered as clarinet):

```powershell
synth-pipeline generate --score ../RawData/Score/001.musicxml --count 5
synth-pipeline generate --score ../RawData/Score --count 8
```

`--config` selects a YAML file.

| File | Role |
|------|------|
| `config/default.yaml` | One content error per clip; repetition 35%; no squeaks; no standalone repeat; no restart gap |
| `config/multi_error_10k.yaml` | 1–8 errors; equal weights; `squeak.prob` 0.22 (C6–A7); `repetition_prob` 0.80; `standalone_repetition_prob` 0.20; `repeat_gap_seconds` `[0.2, 1.0]` |

Existing 1.1 bundles (or 1.2 extras that still have a single-note core) can be rewritten onto the current gold rules:

```powershell
python scripts/convert_labels_to_melody.py --root ./output --force --pad-random
synth-pipeline convert-labels --root ./output --force --pad-random --workers 8
```

`--force` overwrites existing `score_part` / `pitches`. `--pad-random` picks pad ∈ {1, 2} unless the file already stored `pad_notes`. Extra-note conversion maps the extra’s **performance time** onto the clean score, then expands to the notes before and after. Repeat to convert more roots: `--root ./output --root ./1000dataexport`.

## Output

Each sample is an ALIGN bundle:

```
output/synth_gen_0010/
├── verified_score.musicxml      # clean notation (ground truth)
├── performance_score.musicxml   # errored (and maybe repeated) render source
├── reference_audio.wav          # clarinet, clean score
├── performance_audio.wav        # clarinet, with the injected error
├── labels.json                  # source: synthetic, schema 1.2
├── candidates.json
├── alignment.npz
├── performance_mel.npy
├── reference_mel.npy
└── metadata.json
```

`labels.json` fields beyond schema 1.1: `score_part` (`start_note_index`, `end_note_index`, `start_measure`, `end_measure`, `pad_notes`), `pitches`, `note_ids`, and on `repetition` `extra_copies`.

Eval (`align-model eval-melodies`) treats a predicted melody as correct if either pitch list is a **contiguous part** of the other (or they are equal). One gold can validate several predictions and the reverse. Headline is F1. Repeated-pass labels are ignored.

Open the output root in the DataCreate annotator (`datacreate serve`) like any other sample directory if you copy or generate into `DataCreate/samples/`.
