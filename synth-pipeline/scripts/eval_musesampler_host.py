"""Render Muse Sounds clarinet clips and time them vs FreePats SoundFont.

Direct ctypes hosting of MuseSamplerCoreLib fails ``ms_init`` from Python: the
sampler talks to Muse Hub only from an SDK-enabled binary (MuseScore Studio).
This eval still *loads* the DLL (version, exports), then renders listen-files
through MuseScore Studio CLI ``--sound-profile MuseSounds`` — the same DLL,
inside the licensed host — and converts MP3 to WAV.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthpipeline.musesampler_host import (  # noqa: E402
    DEFAULT_DLL,
    MuseSamplerError,
    MuseSamplerHost,
    SamplerNote,
)
from synthpipeline.soundfonts import SOUNDFONT_ROOT  # noqa: E402

OUT_DIR = Path(r"D:\stuff\Audio Evaluation\ALIGN\DataCreate\work\musesampler_eval")
SF2 = SOUNDFONT_ROOT / "freepats" / "Clarinet-20190818.sf2"
MUSESCORE = Path(r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe")
FFMPEG = Path(r"C:\Users\Hank\.conda\envs\MusicEval\Library\bin\ffmpeg.exe")
SAMPLE_A_XML = Path(r"E:\outputRaw_sf_10k\synth_MozartClConcertoA_0042\performance_score.musicxml")
SAMPLE_A_MID = Path(r"E:\outputRaw_sf_10k\synth_MozartClConcertoA_0042\performance_audio.mid")
SAMPLE_B_XML = Path(r"E:\outputRaw_sf_10k\synth_WeberITAV_0043\performance_score.musicxml")
SAMPLE_B_MID = Path(r"E:\outputRaw_sf_10k\synth_WeberITAV_0043\performance_audio.mid")
SAMPLE_RATE = 44100
TAIL_SEC = 1.2

SCALE_MUSICXML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN"
 "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1">
      <part-name>Clarinet in Bb</part-name>
      <part-abbreviation>Bb Cl.</part-abbreviation>
      <score-instrument id="P1-I1">
        <instrument-name>Clarinet in Bb</instrument-name>
      </score-instrument>
      <midi-device id="P1-I1" port="1"></midi-device>
      <midi-instrument id="P1-I1">
        <midi-channel>1</midi-channel>
        <midi-program>72</midi-program>
        <volume>78.7402</volume>
        <pan>0</pan>
      </midi-instrument>
    </score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>-2</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
        <transpose>
          <diatonic>-1</diatonic>
          <chromatic>-2</chromatic>
        </transpose>
      </attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>F</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
    </measure>
    <measure number="2">
      <note><pitch><step>G</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>A</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>B</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
    </measure>
    <measure number="3">
      <note><rest/><duration>4</duration><type>whole</type></note>
      <barline location="right"><bar-style>light-heavy</bar-style></barline>
    </measure>
  </part>
</score-partwise>
"""


def _write_wav(path: Path, stereo: np.ndarray, sr: int) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(stereo)))) if stereo.size else 0.0
    sf.write(str(path), stereo, sr, subtype="PCM_16")
    return {
        "path": str(path),
        "seconds": float(len(stereo) / sr) if sr else 0.0,
        "peak": peak,
        "rms": rms,
        "silent": peak < 1e-5,
        "bytes": path.stat().st_size if path.exists() else 0,
    }


def _scale_notes() -> list[SamplerNote]:
    keys = [60, 62, 64, 65, 67, 69, 71, 72]
    notes: list[SamplerNote] = []
    t = 200_000
    for key in keys:
        notes.append(SamplerNote(pitch=key, start_us=t, duration_us=400_000, tempo=120.0))
        t += 450_000
    return notes


def _render_soundfont_midi(midi_path: Path, out_wav: Path, sr: int) -> tuple[dict, float]:
    import tinysoundfont
    from tinysoundfont.midi import load
    from tinysoundfont.sequencer import Sequencer

    started = time.perf_counter()
    synth = tinysoundfont.Synth(samplerate=sr, gain=-6)
    synth.sfload(str(SF2))
    for ch in range(16):
        synth.program_change(ch, 0, False)
    sequencer = Sequencer(synth)
    events = load(str(midi_path), persistent=False)
    sequencer.add(events)
    duration = max(event.t for event in events) + TAIL_SEC
    remaining = int(duration * sr)
    chunks = []
    while remaining > 0:
        n = min(4096, remaining)
        buf = synth.generate(n)
        chunks.append(np.frombuffer(buf, dtype=np.float32).reshape(-1, 2).copy())
        remaining -= n
    stereo = np.concatenate(chunks, axis=0)
    return _write_wav(out_wav, stereo, sr), time.perf_counter() - started


def _render_soundfont_notes(notes: list[SamplerNote], out_wav: Path, sr: int) -> tuple[dict, float]:
    import tinysoundfont

    started = time.perf_counter()
    synth = tinysoundfont.Synth(samplerate=sr, gain=-6)
    synth.sfload(str(SF2))
    synth.program_change(0, 0, False)
    pieces = []
    cursor = 0.0
    for note in notes:
        start = note.start_us / 1_000_000.0
        dur = note.duration_us / 1_000_000.0
        gap = int(max(0.0, start - cursor) * sr)
        if gap:
            pieces.append(np.frombuffer(synth.generate(gap), dtype=np.float32).reshape(-1, 2).copy())
        synth.noteon(0, note.pitch, 96)
        n = max(1, int(dur * sr))
        pieces.append(np.frombuffer(synth.generate(n), dtype=np.float32).reshape(-1, 2).copy())
        synth.noteoff(0, note.pitch)
        cursor = start + dur
    pieces.append(
        np.frombuffer(synth.generate(int(TAIL_SEC * sr)), dtype=np.float32).reshape(-1, 2).copy()
    )
    stereo = np.concatenate(pieces, axis=0)
    return _write_wav(out_wav, stereo, sr), time.perf_counter() - started


def _probe_dll(report: dict) -> None:
    report["dll"] = str(DEFAULT_DLL)
    report["dll_exists"] = DEFAULT_DLL.is_file()
    try:
        host = MuseSamplerHost(sample_rate=SAMPLE_RATE)
        host.open()
        try:
            report["sampler_version"] = host.version()
            report["dll_init_ok"] = True
        finally:
            host.close()
    except MuseSamplerError as exc:
        report["dll_init_ok"] = False
        report["dll_init_error"] = str(exc)
        # Version getters work without init.
        import ctypes
        import os

        os.add_dll_directory(str(DEFAULT_DLL.parent))
        lib = ctypes.CDLL(str(DEFAULT_DLL))
        lib.ms_get_version_major.restype = ctypes.c_int
        lib.ms_get_version_minor.restype = ctypes.c_int
        lib.ms_get_version_revision.restype = ctypes.c_int
        lib.ms_get_version_string.restype = ctypes.c_char_p
        report["sampler_version"] = (
            f"{lib.ms_get_version_major()}.{lib.ms_get_version_minor()}."
            f"{lib.ms_get_version_revision()}"
        )
        raw = lib.ms_get_version_string()
        if raw:
            report["sampler_version_string"] = raw.decode("utf-8", "replace")
    print(
        f"DLL init ok={report.get('dll_init_ok')} version={report.get('sampler_version')}",
        flush=True,
    )


def _musescore_export(xml_path: Path, mp3_path: Path) -> dict:
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    if mp3_path.exists():
        mp3_path.unlink()
    cmd = [
        str(MUSESCORE),
        "--sound-profile",
        "MuseSounds",
        "-f",
        "-o",
        str(mp3_path),
        str(xml_path),
    ]
    started = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - started
    return {
        "cmd": cmd,
        "exit": proc.returncode,
        "wall_sec": elapsed,
        "bytes": mp3_path.stat().st_size if mp3_path.exists() else 0,
        "stderr_tail": (proc.stderr or "")[-500:],
    }


def _mp3_to_wav(mp3_path: Path, wav_path: Path) -> dict:
    cmd = [str(FFMPEG), "-y", "-i", str(mp3_path), "-ar", str(SAMPLE_RATE), str(wav_path)]
    started = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - started
    info: dict = {"ffmpeg_sec": elapsed, "exit": proc.returncode}
    if wav_path.exists():
        audio, sr = sf.read(str(wav_path), always_2d=True, dtype="float32")
        info.update(
            {
                "path": str(wav_path),
                "seconds": float(len(audio) / sr),
                "peak": float(np.max(np.abs(audio))),
                "rms": float(np.sqrt(np.mean(np.square(audio)))),
                "silent": float(np.max(np.abs(audio))) < 1e-5,
                "bytes": wav_path.stat().st_size,
                "channels": int(audio.shape[1]),
                "sample_rate": int(sr),
            }
        )
    return info


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "out_dir": str(OUT_DIR),
        "sample_rate": SAMPLE_RATE,
        "musescore": str(MUSESCORE),
        "renders": [],
    }
    _probe_dll(report)

    scale_xml = OUT_DIR / "01_scale_written.musicxml"
    scale_xml.write_text(SCALE_MUSICXML, encoding="utf-8")

    jobs = [
        ("01_scale_written", scale_xml, None, _scale_notes()),
        ("05_mozart_0042", SAMPLE_A_XML, SAMPLE_A_MID, None),
        ("06_weber_0043", SAMPLE_B_XML, SAMPLE_B_MID, None),
    ]

    for name, xml_path, midi_path, notes in jobs:
        if not xml_path.is_file():
            print(f"skip {name}: missing {xml_path}", flush=True)
            continue
        mp3_path = OUT_DIR / f"{name}_musesounds.mp3"
        wav_path = OUT_DIR / f"{name}_musesounds.wav"
        sf_path = OUT_DIR / f"{name}_soundfont.wav"
        export = _musescore_export(xml_path, mp3_path)
        print(
            f"{name} MuseSounds CLI exit={export['exit']} {export['wall_sec']:.2f}s "
            f"mp3={export['bytes']} bytes",
            flush=True,
        )
        wav_info = {}
        if export["exit"] == 0 and mp3_path.exists():
            wav_info = _mp3_to_wav(mp3_path, wav_path)
        if notes is not None:
            sf_info, sf_sec = _render_soundfont_notes(notes, sf_path, SAMPLE_RATE)
        elif midi_path and midi_path.is_file():
            sf_info, sf_sec = _render_soundfont_midi(midi_path, sf_path, SAMPLE_RATE)
        else:
            sf_info, sf_sec = {}, None
        ms_dur = wav_info.get("seconds") or 0.0
        item = {
            "name": name,
            "xml": str(xml_path),
            "musesounds_cli": export,
            "musesounds_wav": wav_info,
            "soundfont": {**sf_info, "wall_sec": sf_sec} if sf_info else None,
        }
        if ms_dur and export["wall_sec"]:
            item["musesounds_rtf"] = export["wall_sec"] / ms_dur
        if sf_info and sf_sec and sf_info.get("seconds"):
            item["soundfont_rtf"] = sf_sec / sf_info["seconds"]
        report["renders"].append(item)

    (OUT_DIR / "timings.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "Muse Sounds clarinet eval",
        f"Output: {OUT_DIR}",
        f"MuseSampler DLL: {report.get('dll')} version {report.get('sampler_version')}",
        f"Direct ctypes ms_init: {report.get('dll_init_ok')} ({report.get('dll_init_error', 'ok')})",
        "",
        "MuseSounds WAVs were rendered by MuseScore Studio CLI with --sound-profile MuseSounds",
        "(same MuseSamplerCoreLib.dll, inside the licensed MuseScore host), then ffmpeg MP3->WAV.",
        "WAV/FLAC export is still broken in this MuseScore build (exit 1331); MP3/OGG work.",
        "Companion *_soundfont.wav files are FreePats Clarinet via tinysoundfont.",
        "",
    ]
    for item in report["renders"]:
        wav = item.get("musesounds_wav") or {}
        sf_ = item.get("soundfont") or {}
        lines.append(
            f"{item['name']}: MuseSounds CLI {item['musesounds_cli']['wall_sec']:.2f}s "
            f"for {wav.get('seconds', 0):.2f}s audio "
            f"(rtf {item.get('musesounds_rtf', float('nan')):.2f}x, silent={wav.get('silent')}) "
            f"vs SoundFont {sf_.get('wall_sec', float('nan')):.3f}s "
            f"(rtf {item.get('soundfont_rtf', float('nan')):.2f}x)"
        )
    text = "\n".join(lines) + "\n"
    (OUT_DIR / "LISTEN.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
