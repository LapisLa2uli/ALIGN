import numpy as np
from tinysoundfont.midi import Event, NoteOff, NoteOn

from synthpipeline.midi_player import events_to_notes, midi_key_to_hz, render_notes


def _ev(action, t: float, channel: int = 0) -> Event:
    return Event(action=action, t=t, channel=channel, persistent=False)


def test_hanging_note_on_is_capped() -> None:
    events = [
        _ev(NoteOn(key=60, velocity=90), 0.0),
        _ev(NoteOn(key=64, velocity=80), 8.0),
        _ev(NoteOff(key=64), 8.4),
    ]
    notes = events_to_notes(events)
    stuck = next(n for n in notes if n.key == 60)
    assert stuck.end - stuck.start <= 2.85
    assert stuck.end - stuck.start >= 0.03


def test_drone_under_melody_is_clipped() -> None:
    events = [_ev(NoteOn(key=48, velocity=70), 0.0)]
    t = 0.15
    for i in range(8):
        key = 64 + (i % 3)
        events.append(_ev(NoteOn(key=key, velocity=90), t))
        events.append(_ev(NoteOff(key=key), t + 0.12))
        t += 0.14
    events.append(_ev(NoteOff(key=48), 12.0))
    notes = events_to_notes(events)
    drone = next(n for n in notes if n.key == 48)
    assert drone.end - drone.start < 1.5


def test_squeak_c7_is_audible() -> None:
    notes = events_to_notes(
        [
            _ev(NoteOn(key=96, velocity=100), 0.0),
            _ev(NoteOff(key=96), 0.25),
        ]
    )
    audio = render_notes(notes, 22050, tail_seconds=0.05)
    assert float(np.max(np.abs(audio))) > 0.05
    spec = np.abs(np.fft.rfft(audio * np.hanning(audio.size)))
    freqs = np.fft.rfftfreq(audio.size, d=1.0 / 22050)
    peak = float(freqs[int(np.argmax(spec))])
    expected = midi_key_to_hz(96)
    assert abs(peak - expected) < 40.0
