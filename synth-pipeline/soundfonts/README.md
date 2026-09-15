# Clarinet SoundFonts

Place clarinet-only SF2 files here. MIDI is still written with **music21**; tinysoundfont renders the chosen bank.

| ID | File | Program | Source |
|---|---|---|---|
| `freepats` | `freepats/Clarinet-20190818.sf2` | 0 | [FreePats Clarinet](http://freepats.zenvoid.org/Reed/clarinet.html) (CC0) |
| `u220` | `u220/u220_clarinet.sf2` | 0 | [Roland U220 Winds clarinet](https://www.polyphone.io/en/soundfonts/reeds/219-roland-u220-winds-clarinet) |
| `mcb` | `mcb/mcb.sf2` | 0 | [Maestro Clarinet Base](https://musical-artifacts.com/artifacts/2135) (CC BY 3.0, Mats Helgesson) |
| `msbasic` | MuseScore `MS Basic.sf3` | 71 | Installed with MuseScore 4 |

```powershell
synth-pipeline fetch-soundfonts
synth-pipeline list-soundfonts
```

`fetch-soundfonts` can install FreePats automatically. Polyphone and Musical Artifacts block scripted downloads: save `u220_clarinet.sf2` and `mcb.sf2` into the folders above, then:

```powershell
synth-pipeline generate --count 5 --soundfont freepats
synth-pipeline generate --count 5 --soundfont u220
synth-pipeline generate --count 5 --soundfont mcb
```

## Renderer settings and dataset use

All banks are driven by the same renderer so timbre, not synthesis logic, is the intended variable:

| Parameter | Value |
|---|---|
| MIDI writer | `music21` |
| Audio engine | `tinysoundfont` |
| Sample rate | 22,050 Hz mono |
| Synth gain | -6 dB |
| Render tail | 2.0 s |
| Chunk size | 4,096 samples |
| Bb-clarinet convention | sounding MIDI = written MIDI - 2 semitones |
| Current render marker | `soundfont_v1` |

The synth is cached once per worker process and keyed by resolved SoundFont path, sample rate, and gain. The pipeline sends explicit note transpose and pitch-bend events; changing the SoundFont must not change `labels.json` written pitches or `note_map.json` clean indices.

Known corpus versions:

| Corpus | Bank | Size |
|---|---|---:|
| `output_10k_multi` | FreePats by documented command | 10,000 requested; configured root is not currently present |
| `output_2k_rawdata` | FreePats by config | 2,000 requested |
| `outputRaw_sf_10k` | FreePats, `soundfont_v1` | 10,000 accepted |
| Small default/soundfont comparison runs | `freepats`, `u220`, `mcb`, or `msbasic` selected by CLI | User-selected |

No model is trained in this directory. Model datasets, training counts, and hyperparameters are listed in [`../../align-model/README.md`](../../align-model/README.md); generation settings are listed in [`../README.md`](../README.md).

When adding a bank, record its license and source URL here, add its path/program to `src/synthpipeline/soundfonts.py`, and regenerate a small deterministic seed before producing a corpus. Do not mix banks inside a model split unless `audio_render` and soundfont identity are included in the split stratification.
