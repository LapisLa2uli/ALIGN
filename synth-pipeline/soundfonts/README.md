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
