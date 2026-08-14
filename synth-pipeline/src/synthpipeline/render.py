from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from datacreate.audio_utils import audio_stats, is_silent, save_wav
from datacreate.config import PipelineConfig
from datacreate.tools.musescore import export_score_to_midi, find_soundfont


GM_CLARINET = 71


def render_score_as_clarinet(
    dc_config: PipelineConfig,
    score_path: Path,
    output_wav: Path,
    logger: logging.Logger,
    clarinet_program: int = GM_CLARINET,
) -> Path:
    """Export MusicXML via MuseScore MIDI, then render with GM clarinet."""
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    midi_path = output_wav.with_suffix(".mid")
    export_score_to_midi(dc_config, score_path, midi_path, logger)
    render_midi_clarinet(midi_path, output_wav, dc_config, logger, clarinet_program)
    audio, _ = _load_wav_mono(output_wav, dc_config.sample_rate())
    stats = audio_stats(audio)
    logger.info("Rendered %s stats: %s", output_wav.name, stats)
    if is_silent(audio):
        raise RuntimeError(
            f"Rendered audio is silent ({output_wav}). "
            "Check MuseScore version (>=4.2) and SoundFont."
        )
    return output_wav


def render_midi_clarinet(
    midi_path: Path,
    output_wav: Path,
    config: PipelineConfig,
    logger: logging.Logger,
    clarinet_program: int = GM_CLARINET,
) -> None:
    try:
        import tinysoundfont
        from tinysoundfont.midi import load
        from tinysoundfont.sequencer import Sequencer
    except ImportError as exc:
        raise RuntimeError(
            "tinysoundfont is required. Install with: pip install tinysoundfont"
        ) from exc

    soundfont = find_soundfont(config)
    if soundfont is None:
        raise RuntimeError(
            "No SoundFont found. Set paths.soundfont in config/default.yaml"
        )

    sample_rate = config.sample_rate()
    gain_db = float(config.musescore.get("synthesizer_gain_db", -6))
    tail_seconds = float(config.musescore.get("tail_seconds", 2.0))
    chunk_size = int(config.musescore.get("render_chunk_size", 4096))

    synth = tinysoundfont.Synth(samplerate=sample_rate, gain=gain_db)
    synth.sfload(str(soundfont))
    for channel in range(16):
        if channel == 9:
            synth.program_change(channel, 0, True)
        else:
            synth.program_change(channel, clarinet_program, 0)

    sequencer = Sequencer(synth)
    events = load(str(midi_path), persistent=False)
    if not events:
        raise RuntimeError(f"No MIDI events found in {midi_path}")
    _force_clarinet_program(events, clarinet_program)
    sequencer.add(events)
    duration = max(event.t for event in events) + tail_seconds
    logger.info(
        "Rendering clarinet MIDI via SoundFont (%s): %.2fs, %d events, program %d",
        soundfont.name,
        duration,
        len(events),
        clarinet_program,
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
    logger.info("Wrote clarinet audio %s (%d samples)", output_wav, mono.size)


def _force_clarinet_program(events, clarinet_program: int) -> None:
    for ev in events:
        channel = getattr(ev, "channel", None)
        if channel == 9:
            continue
        for attr in ("program", "preset", "instrument"):
            if hasattr(ev, attr) and getattr(ev, attr) is not None:
                try:
                    setattr(ev, attr, clarinet_program)
                except (AttributeError, TypeError):
                    pass


def _load_wav_mono(path: Path, sample_rate: int):
    from datacreate.audio_utils import load_audio

    return load_audio(path, sample_rate, mono=True)
