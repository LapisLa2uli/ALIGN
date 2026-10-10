"""Local-only extraction and assembly of spoken feedback with performance clips."""
from __future__ import annotations

from math import gcd
from pathlib import Path
import hashlib
import re
import subprocess

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt


MIX_TARGET_LUFS = -20.0
SNIPPET_GAP_SECONDS = 0.5
SNIPPET_REVERB_MIX = 0.08
SNIPPET_REVERB_TAIL_SECONDS = 0.25


def prepare_speech_pace(source: Path, text: str, language: str, min_wpm: float) -> tuple[Path, dict]:
    """Gently correct slow English narration without shifting voice pitch.

    Never stretches music, slows faster speech, or estimates Chinese from words.
    Short cues are excluded because pauses dominate their measured word rate.
    Keep raw synthesis for replay; adjusted audio is a separate lossless file.
    """
    if not min_wpm or not language.lower().startswith(("english", "en")):
        return source, {}
    words = len(re.findall(r"\b[A-Za-z]+(?:['-][A-Za-z]+)*\b", text))
    if words < 12:
        return source, {}
    audio, rate = sf.read(source, dtype="float32", always_2d=True)
    seconds = len(audio) / rate
    if not seconds:
        raise ValueError("Cannot adjust empty speech audio.")
    measured = words * 60 / seconds
    factor = min(1.25, max(1.0, min_wpm / measured))
    metadata = {"speech_source_file": source.name, "speech_source_seconds": seconds,
                "speech_source_wpm": measured, "speech_speed_factor": 1.0}
    if factor < 1.03:
        return source, metadata
    from datacreate.audio_utils import _find_ffmpeg
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise ValueError("Speech pacing requires ffmpeg or imageio-ffmpeg.")
    destination = source.with_name(source.stem + "-paced.wav")
    temporary = destination.with_suffix(".wav.part")
    try:
        result = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-af", f"atempo={factor:.8f}", "-c:a", "pcm_f32le",
            "-f", "wav", str(temporary)], capture_output=True, timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            raise ValueError("Speech pacing failed; the original narration is preserved.")
        info = sf.info(temporary)
        if not info.frames or info.samplerate != rate:
            raise ValueError("Speech pacing returned invalid audio.")
        temporary.replace(destination)
    except subprocess.TimeoutExpired:
        raise ValueError("Speech pacing timed out; the original narration is preserved.") from None
    finally:
        temporary.unlink(missing_ok=True)
    metadata.update(speech_speed_factor=factor, speech_output_wpm=words * 60 / info.duration)
    return destination, metadata


def measure_loudness(audio: np.ndarray, rate: int) -> float | None:
    """Gated K-weighted loudness; shorten the window for notes under 400 ms."""
    if not len(audio) or not np.any(audio):
        return None
    meter = pyln.Meter(rate, block_size=min(0.4, len(audio) / rate))
    with np.errstate(divide="ignore", invalid="ignore"):
        level = float(meter.integrated_loudness(audio))
    return level if np.isfinite(level) else None


def _small_room(audio: np.ndarray, rate: int) -> np.ndarray:
    """Quiet, short reflections with a fully retained decay, on music only."""
    frames = int(round(rate * SNIPPET_REVERB_TAIL_SECONDS))
    time = np.arange(frames) / rate
    output = np.pad(audio, ((0, frames), (0, 0))).astype(np.float64)
    lowpass = butter(2, min(4500, rate * 0.4), fs=rate, output="sos")
    for channel in range(audio.shape[1]):
        rng = np.random.default_rng(1701 + channel)
        impulse = sosfilt(lowpass, rng.standard_normal(frames)) * np.exp(-time / 0.036)
        impulse[:int(rate * 0.014)] = 0  # Small room pre-delay; keep the direct attack.
        impulse *= np.linspace(1, 0, frames)
        impulse /= max(float(np.sqrt(np.sum(impulse ** 2))), 1e-12)
        wet = fftconvolve(audio[:, channel], impulse)
        output[:len(wet), channel] += SNIPPET_REVERB_MIX * wet
    return output


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
    """Match speech/music loudness, add subtle room tone and unhurried pauses."""
    if not entries:
        raise ValueError("Cannot assemble an empty playback plan.")
    prepared = []
    target = MIX_TARGET_LUFS
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
        dry_frames = len(audio)
        snippet = entry["kind"] in {"performance", "reference"}
        if snippet:
            fade = min(int(rate * 0.008), len(audio) // 2)
            if fade:
                audio[:fade] *= np.linspace(0, 1, fade)[:, None]
                audio[-fade:] *= np.linspace(1, 0, fade)[:, None]
            audio = _small_room(audio, rate)
        level = measure_loudness(audio, rate)
        # Fourfold oversampling estimates inter-sample peaks. Use one shared
        # feasible target instead of peak-capping one clip and leaving it quieter.
        peak = float(np.max(np.abs(resample_poly(audio, 4, 1, axis=0))))
        if level is not None and peak > 0:
            target = min(target, level - 2.0 - 20 * np.log10(peak), level + 30.0)
        prepared.append((entry, audio, level, dry_frames, snippet))

    pieces, timeline = [], []
    cursor = 0
    if prepared[0][4]:
        pieces.append(np.zeros((round(rate * SNIPPET_GAP_SECONDS), 2)))
        cursor += len(pieces[-1])
    for index, (entry, audio, level, dry_frames, snippet) in enumerate(prepared):
        gain = 10 ** ((target - level) / 20) if level is not None else 1.0
        audio = audio * gain  # One constant gain preserves dynamics inside each segment.
        has_next = index + 1 < len(prepared)
        next_is_snippet = has_next and prepared[index + 1][4]
        gap_after = SNIPPET_GAP_SECONDS if snippet or next_is_snippet else 0.3 if has_next else 0.0
        timeline.append({key: value for key, value in entry.items() if key != "path"} |
                        {"file": Path(entry["path"]).name, "start_time": cursor / rate,
                         "end_time": (cursor + len(audio)) / rate, "gain": gain,
                         "input_lufs": level, "output_lufs": measure_loudness(audio, rate),
                         "target_lufs": target, "dry_end_time": (cursor + dry_frames) / rate,
                         "reverb_mix": SNIPPET_REVERB_MIX if snippet else 0.0,
                         "reverb_tail_seconds": (len(audio) - dry_frames) / rate,
                         "silence_after_seconds": gap_after})
        pieces.append(audio)
        cursor += len(audio)
        if gap_after:
            pieces.append(np.zeros((round(rate * gap_after), 2)))
            cursor += len(pieces[-1])
    audio = np.concatenate(pieces)
    # Final numerical guard; loudness matching already reserves two dB of headroom.
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
        if item["output_lufs"] is not None:
            item["output_lufs"] += 20 * np.log10(master_gain)
    return timeline
