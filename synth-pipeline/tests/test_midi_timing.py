import mido
import numpy as np

from synthpipeline.timing import midi_note_times


def test_fast_triplets_are_separate_midi_events(tmp_path):
    path = tmp_path / "triplets.mid"
    midi = mido.MidiFile(ticks_per_beat=10080)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=857143))
    pitches = [70, 74, 77, 82, 77, 74] * 2
    for pitch in pitches:
        track.append(mido.Message("note_on", note=pitch, velocity=90))
        track.append(mido.Message("note_off", note=pitch, time=1680))
    midi.save(path)
    times = midi_note_times(path)
    assert [row[0] for row in times] == pitches
    np.testing.assert_allclose([row[1] for row in times], np.arange(12) * 857143 / 6e6)
    np.testing.assert_allclose([row[2] for row in times], np.arange(1, 13) * 857143 / 6e6)


def test_multiple_tracks_tempo_changes_and_velocity_zero_off(tmp_path):
    path = tmp_path / "tempo.mid"
    midi = mido.MidiFile(ticks_per_beat=480)
    tempo = mido.MidiTrack([
        mido.MetaMessage("set_tempo", tempo=500000),
        mido.MetaMessage("set_tempo", tempo=1000000, time=480),
    ])
    notes = mido.MidiTrack([
        mido.Message("note_on", note=60, velocity=80),
        mido.Message("note_on", note=60, velocity=0, time=960),
    ])
    midi.tracks.extend([tempo, notes])
    midi.save(path)
    assert midi_note_times(path) == [(60, 0.0, 1.5)]
