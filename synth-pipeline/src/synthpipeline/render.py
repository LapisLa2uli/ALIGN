from __future__ import annotations

import copy
import logging
import time
from pathlib import Path

import numpy as np

from music21 import converter, note, stream

from datacreate.audio_utils import audio_stats, is_silent, save_wav
from datacreate.config import PipelineConfig
from datacreate.tools.musescore import export_score_to_midi, find_soundfont
from synthpipeline.scoregen import write_midi


GM_CLARINET = 71
MIDI_BACKENDS = ("music21", "musescore")
PITCH_BEND_CENTER = 8192
PITCH_BEND_RANGE_SEMITONES = 2.0

# tinysoundfont.sfload of MS Basic.sf3 is ~20-30s; reuse one Synth per process.
_synth_cache: dict[tuple, object] = {}


def render_score_as_clarinet(
    dc_config: PipelineConfig,
    score_path: Path,
    output_wav: Path,
    logger: logging.Logger,
    clarinet_program: int = GM_CLARINET,
    midi_backend: str = "music21",
    score: stream.Score | None = None,
    pitch_bends: list[dict] | None = None,
    bpm: float | None = None,
    sounding_transpose: int = -2,
) -> Path:
    """Export MIDI (music21 or MuseScore), then render with GM clarinet.

    ``sounding_transpose`` is applied only to the MIDI sent to the SoundFont
    (Bb clarinet: written C sounds Bb). MusicXML on disk stays written pitch.
    """
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    midi_path = output_wav.with_suffix(".mid")
    backend = (midi_backend or "music21").lower()
    if backend == "music21":
        export_score_to_midi_music21(
            score, score_path, midi_path, logger, sounding_transpose=sounding_transpose
        )
    elif backend == "musescore":
        export_score_to_midi(dc_config, score_path, midi_path, logger)
    else:
        raise ValueError(f"Unknown midi_backend {midi_backend!r}; use music21 or musescore")
    render_midi_clarinet(
        midi_path,
        output_wav,
        dc_config,
        logger,
        clarinet_program,
        pitch_bends=pitch_bends,
        bpm=bpm,
    )
    audio, _ = _load_wav_mono(output_wav, dc_config.sample_rate())
    stats = audio_stats(audio)
    logger.info("Rendered %s stats: %s", output_wav.name, stats)
    if is_silent(audio):
        raise RuntimeError(
            f"Rendered audio is silent ({output_wav}). "
            "Check MuseScore version (>=4.2) and SoundFont."
        )
    return output_wav


def transpose_notes_chromatic(score: stream.Score, semitones: int) -> stream.Score:
    """Shift each note by semitones without music21 Stream/key-signature transpose.

    ``Score.transpose(-2)`` and ``pitch.midi -= 2`` corrupt generated clarinet
    scores in keys such as Bb major. Assigning ``pitch.transpose(...)`` is safe.
    """
    if not semitones:
        return score
    shift = int(semitones)
    for item in score.flatten().getElementsByClass(note.Note):
        item.pitch = item.pitch.transpose(shift)
    return score


def export_score_to_midi_music21(
    score: stream.Score | None,
    score_path: Path,
    output_midi: Path,
    logger: logging.Logger,
    sounding_transpose: int = -2,
) -> Path:
    if score is None:
        score = converter.parse(str(score_path))
    else:
        score = copy.deepcopy(score)
    from synthpipeline.midi_player import strip_ornaments

    strip_ornaments(score)
    if sounding_transpose:
        transpose_notes_chromatic(score, int(sounding_transpose))
        logger.info(
            "Writing MIDI via music21 at sounding pitch (%+d semitones): %s",
            int(sounding_transpose),
            output_midi,
        )
    else:
        logger.info("Writing MIDI via music21: %s", output_midi)
    write_midi(score, output_midi)
    if not output_midi.exists() or output_midi.stat().st_size == 0:
        raise RuntimeError(f"music21 did not produce {output_midi}")
    return output_midi


def render_midi_clarinet(
    midi_path: Path,
    output_wav: Path,
    config: PipelineConfig,
    logger: logging.Logger,
    clarinet_program: int = GM_CLARINET,
    pitch_bends: list[dict] | None = None,
    bpm: float | None = None,
    note_transpose: int = 0,
) -> None:
    try:
        from tinysoundfont.midi import load
    except ImportError as exc:
        raise RuntimeError(
            "tinysoundfont is required. Install with: pip install tinysoundfont"
        ) from exc

    from synthpipeline.midi_player import render_midi_events

    sample_rate = config.sample_rate()
    tail_seconds = min(0.4, float(config.musescore.get("tail_seconds", 0.35)))

    events = load(str(midi_path), persistent=False)
    if not events:
        raise RuntimeError(f"No MIDI events found in {midi_path}")
    _force_clarinet_program(events, clarinet_program)
    if note_transpose:
        _transpose_note_events(events, int(note_transpose))
    if pitch_bends:
        n_added = _inject_pitch_bends(events, pitch_bends, bpm or 120.0)
        logger.info("Injected %d pitch-bend events for intonation", n_added)
    logger.info(
        "Rendering clarinet MIDI via oscillator (%s): %d events, program %d",
        midi_path.name,
        len(events),
        clarinet_program,
    )
    mono = render_midi_events(events, sample_rate, tail_seconds=tail_seconds)
    save_wav(output_wav, mono, sample_rate)
    logger.info("Wrote clarinet audio %s (%d samples)", output_wav, mono.size)


def _cached_synth(tinysoundfont, soundfont: Path, sample_rate: int, gain_db: float, logger: logging.Logger):
    key = (str(soundfont.resolve()), int(sample_rate), float(gain_db))
    synth = _synth_cache.get(key)
    if synth is not None:
        return synth
    logger.info("Loading SoundFont %s (once per process)...", soundfont)
    started = time.perf_counter()
    synth = tinysoundfont.Synth(samplerate=sample_rate, gain=gain_db)
    synth.sfload(str(soundfont))
    logger.info("SoundFont loaded in %.2fs", time.perf_counter() - started)
    _synth_cache[key] = synth
    return synth


def _transpose_note_events(events, semitones: int) -> None:
    from tinysoundfont.midi import NoteOff, NoteOn

    shift = int(semitones)
    if not shift:
        return
    for ev in events:
        action = ev.action
        if isinstance(action, (NoteOn, NoteOff)) and getattr(action, "key", None) is not None:
            action.key = max(0, min(127, int(action.key) + shift))


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


def cents_to_pitch_bend(
    cents: float, range_semitones: float = PITCH_BEND_RANGE_SEMITONES
) -> int:
    span = float(range_semitones) * 100.0
    if span <= 0:
        return PITCH_BEND_CENTER
    value = int(round(PITCH_BEND_CENTER + (float(cents) / span) * PITCH_BEND_CENTER))
    return max(0, min(16383, value))


def _inject_pitch_bends(events: list, regions: list[dict], bpm: float) -> int:
    """Insert channel pitch-bend events so selected notes sound off-tune."""
    from tinysoundfont.midi import Event, NoteOff, NoteOn, PitchBend

    from synthpipeline.timing import ql_to_seconds

    extra = []
    for region in regions:
        cents = float(region["cents"])
        t0 = ql_to_seconds(float(region["ql_start"]), bpm)
        t1 = ql_to_seconds(float(region["ql_end"]), bpm)
        t0, t1 = _snap_region_to_notes(events, t0, t1)
        bend = cents_to_pitch_bend(cents)
        channels = _note_channels_in_span(events, t0, t1)
        for channel in channels:
            extra.append(
                Event(
                    action=PitchBend(pitch_bend=bend),
                    t=max(0.0, t0 - 0.003),
                    channel=channel,
                    persistent=False,
                )
            )
            extra.append(
                Event(
                    action=PitchBend(pitch_bend=PITCH_BEND_CENTER),
                    t=t1 + 0.003,
                    channel=channel,
                    persistent=False,
                )
            )
    events.extend(extra)
    events.sort(key=lambda ev: (float(ev.t), _pitch_bend_sort_key(ev)))
    return len(extra)


def _snap_region_to_notes(events: list, t0: float, t1: float) -> tuple[float, float]:
    from tinysoundfont.midi import NoteOff, NoteOn

    ons = [float(ev.t) for ev in events if isinstance(ev.action, NoteOn)]
    offs = [float(ev.t) for ev in events if isinstance(ev.action, NoteOff)]
    if ons:
        t0 = min(ons, key=lambda t: abs(t - t0))
    after = [t for t in offs if t >= t0 - 1e-4]
    if after:
        t1 = min(after, key=lambda t: abs(t - t1))
    return t0, max(t1, t0 + 0.05)


def _note_channels_in_span(events: list, t0: float, t1: float) -> list[int]:
    from tinysoundfont.midi import NoteOn

    channels = {
        int(ev.channel)
        for ev in events
        if isinstance(ev.action, NoteOn) and t0 - 0.01 <= ev.t <= t1 + 0.01
    }
    channels.discard(9)
    return sorted(channels) or [0]


def _pitch_bend_sort_key(ev) -> int:
    from tinysoundfont.midi import NoteOff, NoteOn, PitchBend

    action = ev.action
    if isinstance(action, PitchBend):
        return 0 if action.pitch_bend != PITCH_BEND_CENTER else 3
    if isinstance(action, NoteOn):
        return 1
    if isinstance(action, NoteOff):
        return 2
    return 1


def _load_wav_mono(path: Path, sample_rate: int):
    from datacreate.audio_utils import load_audio

    return load_audio(path, sample_rate, mono=True)
