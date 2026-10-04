"""Local-only extraction and assembly of spoken feedback with performance clips."""
from __future__ import annotations

from math import gcd
from pathlib import Path
import hashlib

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def resolve_performance(source: Path, document, explicit: Path | None) -> Path | None:
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise ValueError("The supplied performance recording does not exist.")
        return path
    name = document.get("audio_reference") if isinstance(document, dict) else None
    if name is not None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("audio_reference must name the recording used for label timestamps.")
        path = (source.parent / name).resolve()
        if not path.is_relative_to(source.parent.resolve()):
            raise ValueError("audio_reference must stay within the sample directory; use --performance for an external file.")
    else:
        path = source.parent / "performance_audio.wav"
    return path if path.is_file() else None


def locate_excerpts(report: dict, recording: Path, padding: float) -> dict:
    info = sf.info(recording)
    clips = {}
    for index, label in enumerate(report["labels"]):
        if "start_time" not in label:
            continue
        start, end = label["start_time"], label["end_time"]
        if start >= info.duration or end > info.duration + 1 / info.samplerate:
            raise ValueError("A label time is outside the performance recording. Use the matching trimmed audio.")
        clips[index] = {"label_start_time": start, "label_end_time": end,
                        "source_start_time": max(0.0, start - padding),
                        "source_end_time": min(info.duration, end + padding)}
    return clips


def locate_reference_excerpts(report: dict, recording: Path, score: Path, midi: Path) -> dict:
    from datacreate.feedback_score import label_note_indices, reference_note_times

    events = reference_note_times(score, midi)
    duration = sf.info(recording).duration
    clips = {}
    for index, label in enumerate(report["labels"]):
        indices = sorted(set(label_note_indices(label, context=True)))
        if not indices:
            continue
        if indices[-1] >= len(events):
            raise ValueError("A label refers beyond the reference MIDI notes.")
        start = min(events[i]["start"] for i in indices)
        end = max(events[i]["end"] for i in indices)
        following = [event["start"] for event in events if event["start"] >= end]
        # Keep note release where there is room, without including the next note.
        end = min(end + .06, min(following, default=duration), duration)
        if not 0 <= start < end or any(events[i]["end"] > duration for i in indices):
            raise ValueError("Reference MIDI times exceed the reference recording. Re-render the reference.")
        clips[index] = {"source_start_time": start, "source_end_time": end,
                        "score_note_indices": indices, "timing_source": "reference_audio.mid",
                        "scope": "context" if (label.get("score_part") or {}).get("pad_notes", 0) else "core"}
    return clips


def extract_clip(recording: Path, clip: dict, destination: Path) -> dict:
    with sf.SoundFile(recording) as source:
        start = int(clip["source_start_time"] * source.samplerate)
        end = min(len(source), int(np.ceil(clip["source_end_time"] * source.samplerate)))
        source.seek(start)
        audio = source.read(end - start, dtype="float32", always_2d=True)
        rate = source.samplerate
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Performance excerpt contains no valid audio.")
    # Keep pitch, tempo, channels and relative dynamics intact in the saved excerpt.
    sf.write(destination, audio, rate, subtype="FLOAT")
    return {**clip, "file": destination.name, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "source_start_frame": start, "source_end_frame": end, "sample_rate": rate}


def checked_clip(plan_directory: Path, clip: dict) -> Path:
    name = clip.get("file")
    if not isinstance(name, str) or Path(name).name != name or Path(name).is_absolute():
        raise ValueError("Saved excerpt filenames must be local to the playback plan.")
    path = (plan_directory / name).resolve()
    if not path.is_relative_to(plan_directory.resolve()) or not path.is_file():
        raise ValueError("A saved performance excerpt is missing or outside the plan directory.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != clip.get("sha256"):
        raise ValueError("A saved performance excerpt has changed; regenerate the plan from labels.")
    return path


def compose_audio(entries: list[dict], output: Path, rate: int = 44100) -> list[dict]:
    """Resample to stereo, keep natural pauses, and encode a single final MP3."""
    pieces, timeline = [], []
    cursor = 0
    for entry in entries:
        audio, source_rate = sf.read(entry["path"], dtype="float32", always_2d=True)
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError("Cannot assemble an empty or invalid audio segment.")
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        elif audio.shape[1] != 2:
            raise ValueError("Audio assembly supports mono or stereo recordings.")
        if source_rate != rate:
            factor = gcd(source_rate, rate)
            audio = resample_poly(audio, rate // factor, source_rate // factor, axis=0)
        gain = 1.0
        if entry["kind"] in {"performance", "reference"}:
            # A single bounded gain preserves expressive dynamics inside the excerpt.
            rms = float(np.sqrt(np.mean(audio ** 2)))
            peak = float(np.max(np.abs(audio)))
            if rms > 0.00001 and peak > 0:
                gain = min(4.0, 0.075 / rms, 0.95 / peak)
                audio *= gain
            fade = min(int(rate * 0.008), len(audio) // 2)
            if fade:
                audio[:fade] *= np.linspace(0, 1, fade)[:, None]
                audio[-fade:] *= np.linspace(1, 0, fade)[:, None]
        if pieces:
            gap = np.zeros((int(rate * 0.3), 2), dtype="float32")
            pieces.append(gap)
            cursor += len(gap)
        timeline.append({key: value for key, value in entry.items() if key != "path"} |
                        {"file": Path(entry["path"]).name, "start_time": cursor / rate,
                         "end_time": (cursor + len(audio)) / rate, "gain": gain})
        pieces.append(audio)
        cursor += len(audio)
    audio = np.concatenate(pieces)
    # Attenuate the whole mix only if needed to avoid clipping; never time-stretch.
    peak = float(np.max(np.abs(audio)))
    master_gain = min(1.0, 0.95 / peak) if peak else 1.0
    audio *= master_gain
    temporary = output.with_suffix(".mp3.part")
    try:
        sf.write(temporary, audio, rate, format="MP3")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    for item in timeline:
        item["master_gain"] = master_gain
    return timeline
