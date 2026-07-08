from __future__ import annotations

import logging
import platform
import subprocess
from pathlib import Path

import numpy as np

from datacreate.audio_utils import save_wav
from datacreate.config import PipelineConfig
from datacreate.utils import resolve_binary


def find_musescore(config: PipelineConfig) -> Path | None:
    fallbacks = []
    if platform.system() == "Windows":
        fallbacks = [
            r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
            r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe",
        ]
    elif platform.system() == "Darwin":
        fallbacks = ["/Applications/MuseScore 4.app/Contents/MacOS/mscore"]
    else:
        fallbacks = ["/usr/bin/mscore", "/usr/local/bin/mscore"]
    return resolve_binary(config, "musescore", fallbacks)


def find_soundfont(config: PipelineConfig) -> Path | None:
    configured = config.paths.get("soundfont")
    if configured:
        path = Path(configured)
        if path.exists():
            return path
    if platform.system() == "Windows":
        fallback = Path(r"C:\Program Files\MuseScore 4\sound\MS Basic.sf3")
        if fallback.exists():
            return fallback
    return None


def parse_version(text: str) -> tuple[int, ...]:
    digits: list[int] = []
    for part in text.replace("_", ".").split("."):
        chunk = "".join(ch for ch in part if ch.isdigit())
        if chunk:
            digits.append(int(chunk))
    return tuple(digits or [0])


def _musescore_cli_path(path: Path) -> str:
    return path.resolve().as_posix()


def _powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _run_musescore(
    binary: Path,
    args: list[str],
    logger: logging.Logger,
) -> subprocess.CompletedProcess[str]:
    if platform.system() == "Windows":
        parts = ["&", _powershell_quote(str(binary.resolve()))] + [
            _powershell_quote(arg) for arg in args
        ]
        command = " ".join(parts)
        logger.info("Running MuseScore via PowerShell: %s", command)
        return subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            check=False,
        )

    cmd = [str(binary), *args]
    logger.info("Running MuseScore: %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def check_musescore_version(config: PipelineConfig, logger: logging.Logger) -> str | None:
    binary = find_musescore(config)
    if binary is None:
        logger.warning("MuseScore binary not found; configure paths.musescore in config")
        return None
    result = _run_musescore(binary, ["--version"], logger)
    version_text = (result.stdout or result.stderr or "").strip()
    logger.info("MuseScore version output: %s", version_text)
    min_version = parse_version(str(config.musescore.get("min_version", "4.2.0")))
    current = parse_version(version_text)
    if current and current < min_version:
        logger.warning(
            "MuseScore %s is below recommended %s; CLI export may produce silent audio",
            ".".join(map(str, current)),
            ".".join(map(str, min_version)),
        )
    return version_text


def export_score_to_midi(
    config: PipelineConfig,
    score_path: Path,
    output_midi: Path,
    logger: logging.Logger,
) -> None:
    binary = find_musescore(config)
    if binary is None:
        raise RuntimeError(
            "MuseScore not configured. Set paths.musescore in config/default.yaml"
        )

    output_midi.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "-o",
        _musescore_cli_path(output_midi),
        _musescore_cli_path(score_path),
        "-f",
    ]
    soundfont = config.paths.get("soundfont")
    if soundfont:
        args.extend(["-S", _musescore_cli_path(Path(soundfont))])

    result = _run_musescore(binary, args, logger)
    if result.stdout:
        logger.info("MuseScore stdout:\n%s", result.stdout)
    if result.stderr:
        logger.warning("MuseScore stderr:\n%s", result.stderr)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"MuseScore MIDI export failed with code {result.returncode}"
            + (f": {detail}" if detail else "")
        )
    if not output_midi.exists() or output_midi.stat().st_size == 0:
        raise RuntimeError(f"MuseScore did not produce {output_midi}")


def render_midi_to_wav(
    midi_path: Path,
    output_wav: Path,
    config: PipelineConfig,
    logger: logging.Logger,
) -> None:
    try:
        import tinysoundfont
        from tinysoundfont.midi import load
        from tinysoundfont.sequencer import Sequencer
    except ImportError as exc:
        raise RuntimeError(
            "tinysoundfont is required for reference synthesis. "
            "Install with: pip install tinysoundfont"
        ) from exc

    soundfont = find_soundfont(config)
    if soundfont is None:
        raise RuntimeError(
            "No SoundFont found. Set paths.soundfont in config/default.yaml "
            "(e.g. C:/Program Files/MuseScore 4/sound/MS Basic.sf3)"
        )

    sample_rate = config.sample_rate()
    gain_db = float(config.musescore.get("synthesizer_gain_db", -6))
    tail_seconds = float(config.musescore.get("tail_seconds", 2.0))
    chunk_size = int(config.musescore.get("render_chunk_size", 4096))

    synth = tinysoundfont.Synth(samplerate=sample_rate, gain=gain_db)
    synth.sfload(str(soundfont))
    for channel in range(16):
        synth.program_change(channel, 0, channel == 9)

    sequencer = Sequencer(synth)
    events = load(str(midi_path), persistent=False)
    if not events:
        raise RuntimeError(f"No MIDI events found in {midi_path}")
    sequencer.add(events)
    duration = max(event.t for event in events) + tail_seconds
    logger.info(
        "Rendering MIDI via SoundFont (%s): %.2fs, %d events",
        soundfont.name,
        duration,
        len(events),
    )

    chunks: list[np.ndarray] = []
    remaining = int(duration * sample_rate)
    while remaining > 0:
        count = min(chunk_size, remaining)
        buffer = synth.generate(count)
        stereo = np.frombuffer(buffer, dtype=np.float32).reshape(-1, 2)
        chunks.append(stereo)
        remaining -= count

    stereo = np.concatenate(chunks, axis=0)
    mono = stereo.mean(axis=1)
    save_wav(output_wav, mono, sample_rate)
    logger.info("Wrote reference audio %s (%d samples)", output_wav, mono.size)


def render_score_to_wav(
    config: PipelineConfig,
    score_path: Path,
    output_wav: Path,
    logger: logging.Logger,
) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    midi_path = output_wav.with_suffix(".mid")
    export_score_to_midi(config, score_path, midi_path, logger)
    render_midi_to_wav(midi_path, output_wav, config, logger)
