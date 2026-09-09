"""Oscillator MIDI renderer.

SoundFonts drop clarinet squeaks above their sample range and tinysoundfont
can leave tremolo/ornament note-ons hanging for the whole file. This player
pairs on/off, clips drones, and synthesizes every MIDI key 0-127.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PITCH_BEND_CENTER = 8192
PITCH_BEND_RANGE_SEMITONES = 2.0
MAX_NOTE_SEC = 2.8
MIN_NOTE_SEC = 0.03


@dataclass
class PlayedNote:
    start: float
    end: float
    key: int
    velocity: int
    channel: int = 0
    cents: float = 0.0


def midi_key_to_hz(key: int, cents: float = 0.0) -> float:
    key = max(0, min(127, int(key)))
    return 440.0 * (2.0 ** ((key - 69) / 12.0 + float(cents) / 1200.0))


def events_to_notes(
    events,
    *,
    max_note_sec: float = MAX_NOTE_SEC,
    tail_seconds: float = 0.0,
) -> list[PlayedNote]:
    """Convert tinysoundfont MIDI events into finite notes."""
    from tinysoundfont.midi import NoteOff, NoteOn, PitchBend

    active: dict[tuple[int, int], list[tuple[float, int]]] = {}
    bends: dict[int, list[tuple[float, float]]] = {}
    notes: list[PlayedNote] = []
    last_t = 0.0

    def cents_at(channel: int, t: float) -> float:
        series = bends.get(channel) or [(0.0, 0.0)]
        current = series[0][1]
        for tb, cents in series:
            if tb <= t:
                current = cents
            else:
                break
        return current

    def close(channel: int, key: int, t: float) -> None:
        stack = active.get((channel, key))
        if not stack:
            return
        start, vel = stack.pop()
        end = max(start + MIN_NOTE_SEC, min(float(t), start + max_note_sec))
        notes.append(
            PlayedNote(
                start=start,
                end=end,
                key=int(key),
                velocity=int(vel),
                channel=int(channel),
                cents=cents_at(channel, start),
            )
        )

    for ev in events:
        t = float(getattr(ev, "t", 0.0))
        last_t = max(last_t, t)
        channel = int(getattr(ev, "channel", 0) or 0)
        action = ev.action
        if isinstance(action, PitchBend):
            raw = int(getattr(action, "pitch_bend", PITCH_BEND_CENTER))
            cents = (
                (raw - PITCH_BEND_CENTER)
                / float(PITCH_BEND_CENTER)
                * PITCH_BEND_RANGE_SEMITONES
                * 100.0
            )
            bends.setdefault(channel, []).append((t, cents))
            continue
        if isinstance(action, NoteOn):
            key = int(action.key)
            vel = int(getattr(action, "velocity", 90) or 0)
            if vel <= 0:
                close(channel, key, t)
                continue
            stack = active.setdefault((channel, key), [])
            if stack:
                close(channel, key, t)
            stack.append((t, vel))
            continue
        if isinstance(action, NoteOff):
            close(channel, int(action.key), t)

    hang_end = last_t + max(0.0, float(tail_seconds))
    for (channel, key), stack in list(active.items()):
        while stack:
            close(channel, key, hang_end)

    return _clip_overlapping_drones(notes)


def _clip_overlapping_drones(notes: list[PlayedNote]) -> list[PlayedNote]:
    """Clip a note that sits under many others (stuck tremolo / missing off)."""
    if len(notes) < 4:
        return notes
    durs = np.array([max(MIN_NOTE_SEC, n.end - n.start) for n in notes], dtype=np.float64)
    median = float(np.median(durs))
    cap = max(1.15, 3.0 * median)
    ordered = sorted(notes, key=lambda n: (n.start, n.end))
    for note in ordered:
        dur = note.end - note.start
        if dur <= cap:
            continue
        overlaps = 0
        next_start = None
        for other in ordered:
            if other is note:
                continue
            if other.start >= note.end:
                if next_start is None:
                    next_start = other.start
                break
            if other.start >= note.start - 1e-6 and other.start < note.end:
                overlaps += 1
                if next_start is None and other.start > note.start + MIN_NOTE_SEC:
                    next_start = other.start
        if overlaps < 2:
            continue
        clipped = note.start + max(MIN_NOTE_SEC, median)
        if next_start is not None:
            clipped = min(clipped, max(note.start + MIN_NOTE_SEC, next_start))
        note.end = min(note.end, clipped)
    return notes


def render_notes(
    notes: list[PlayedNote],
    sample_rate: int,
    *,
    tail_seconds: float = 0.15,
) -> np.ndarray:
    if not notes:
        n = max(1, int(round(float(tail_seconds) * sample_rate)))
        return np.zeros(n, dtype=np.float32)
    sr = int(sample_rate)
    end_t = max(n.end for n in notes) + max(0.05, float(tail_seconds))
    n_samples = max(1, int(np.ceil(end_t * sr)))
    mix = np.zeros(n_samples, dtype=np.float64)
    for note in notes:
        i0 = max(0, int(round(note.start * sr)))
        i1 = min(n_samples, int(round(note.end * sr)))
        if i1 - i0 < 2:
            i1 = min(n_samples, i0 + 2)
        mix[i0:i1] += _clarinet_tone(
            i1 - i0,
            sr,
            midi_key_to_hz(note.key, note.cents),
            max(1, int(note.velocity)),
        )
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 1e-8:
        mix *= 0.89 / peak
    return mix.astype(np.float32)


def _clarinet_tone(n: int, sr: int, freq: float, velocity: int) -> np.ndarray:
    """Odd-harmonic clarinet-ish oscillator. Full MIDI range, including squeaks."""
    t = np.arange(n, dtype=np.float64) / float(sr)
    nyquist = 0.45 * float(sr)
    freq = float(max(20.0, min(freq, nyquist - 80.0)))
    wave = np.sin(2.0 * np.pi * freq * t)
    for harm, amp in ((3, 0.42), (5, 0.16), (7, 0.07)):
        f_h = freq * harm
        if f_h >= nyquist:
            break
        wave += amp * np.sin(2.0 * np.pi * f_h * t)
    if freq >= 1046.5:  # C6 and above: a little noise so squeaks read as attacks
        rng = np.random.default_rng(int(freq * 10) % (2**32))
        burst = min(n, int(0.012 * sr))
        noise = rng.normal(0.0, 0.22, size=burst)
        wave[:burst] += noise * np.linspace(1.0, 0.0, burst)
    attack = min(n, max(2, int(0.006 * sr)))
    release = min(n, max(4, int(0.035 * sr)))
    env = np.ones(n, dtype=np.float64)
    env[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float64)
    env[-release:] *= np.linspace(1.0, 0.0, release, dtype=np.float64)
    amp = 0.18 * (max(1, velocity) / 127.0)
    return (wave * env * amp).astype(np.float64)


def render_midi_events(
    events,
    sample_rate: int,
    *,
    tail_seconds: float = 0.25,
) -> np.ndarray:
    notes = events_to_notes(events, tail_seconds=tail_seconds)
    return render_notes(notes, sample_rate, tail_seconds=tail_seconds)


def strip_ornaments(score) -> None:
    """Drop tremolo/trill marks so MIDI export is one note with a real off."""
    try:
        from music21 import note
    except Exception:
        return
    drop = {
        "Tremolo",
        "Trill",
        "Mordent",
        "InvertedMordent",
        "Turn",
        "InvertedTurn",
        "Shake",
        "Schleifer",
    }
    for el in score.recurse().getElementsByClass(note.Note):
        kept = [expr for expr in (el.expressions or []) if type(expr).__name__ not in drop]
        if len(kept) != len(el.expressions or []):
            el.expressions = kept
