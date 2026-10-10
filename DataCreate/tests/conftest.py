import json

import numpy as np
import pytest


@pytest.fixture
def synthesis_inputs():
    """A four-note bar with unequal performed timing and one added note."""
    def write(directory):
        import mido
        from music21 import meter, note, stream
        score, part = stream.Score(), stream.Part()
        measure = stream.Measure(number=2)
        measure.append(meter.TimeSignature("4/4"))
        for pitch in [60, 62, 64, 65]:
            measure.append(note.Note(pitch, quarterLength=1))
        part.append(measure)
        score.append(part)
        score.write("musicxml", fp=directory / "verified_score.musicxml")
        midi = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        midi.tracks.append(track)
        for pitch in [60, 62, 64, 65]:
            track.append(mido.Message("note_on", note=pitch, velocity=80))
            track.append(mido.Message("note_off", note=pitch, time=480))
        midi.save(directory / "reference_audio.mid")
        notes = [{"pitch": p, "start": s, "end": e} for p, s, e in
                 [(60, .2, .65), (62, .7, 1.1), (69, 1.15, 1.4), (64, 1.4, 1.8), (65, 2., 2.7)]]
        events = [{"sounding_index": i, "perf_start": notes[j]["start"], "perf_end": notes[j]["end"]}
                  for i, j in enumerate([0, 1, 3, 4])]
        (directory / "note_alignment_v2.json").write_text(json.dumps({"events": events, "transcribed_notes": notes}))
        return notes
    return write


@pytest.fixture
def fake_soundfont(monkeypatch, tmp_path):
    """Exercise the real event scheduler without requiring a system SoundFont."""
    from datacreate import feedback_synthesis
    from datacreate.tools import musescore
    class Synth:
        def __init__(self):
            self.active = set()
            self.position = 0
        def sounds_off(self):
            self.active.clear()
        def program_change(self, channel, program):
            pass
        def noteon(self, channel, pitch, velocity):
            self.active.add(pitch)
        def noteoff(self, channel, pitch):
            self.active.discard(pitch)
        def generate(self, count):
            time = (np.arange(count) + self.position) / 44100
            wave = sum((.05 * np.sin(2*np.pi*440*2**((p-69)/12)*time) for p in self.active), np.zeros(count))
            self.position += count
            return np.repeat(wave[:, None], 2, axis=1).astype(np.float32).tobytes()
    monkeypatch.setattr(feedback_synthesis, "synthesis_soundfont", lambda: tmp_path / "test.sf2")
    monkeypatch.setattr(musescore, "_cached_synth", lambda *args: Synth())
