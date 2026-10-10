import json

import mido
import pytest
from music21 import converter, meter, note, stream

from datacreate.feedback import add_score_locations, build_messages, FeedbackConfig, prepare_report
from datacreate.feedback_score import reference_note_times


def write_score(path, numbers, pitches=("C4", "D4", "E4", "F4")):
    score, part = stream.Score(), stream.Part()
    for number in numbers:
        measure = stream.Measure(number=number)
        measure.append(meter.TimeSignature("4/4"))
        for pitch in pitches:
            measure.append(note.Note(pitch, quarterLength=1))
        part.append(measure)
    score.append(part)
    score.write("musicxml", fp=path)
    return path


@pytest.mark.parametrize("selected_numbers", [[12, 13], [1, 2]])
def test_bar_numbers_are_global_without_double_offset(tmp_path, selected_numbers):
    write_score(tmp_path / "full_score.musicxml", range(1, 15))
    selected = write_score(tmp_path / "verified_score.musicxml", selected_numbers)
    (tmp_path / "metadata.json").write_text(json.dumps({"score_segment": {"start_measure": 12, "end_measure": 13}}))
    report = prepare_report([{"type": "wrong_note", "score_event_indices": [5, 6]}])
    add_score_locations(report, selected)
    assert report["labels"][0]["score_location"]["phrase"] == "the 2nd to 3rd notes of bar 13"
    messages = build_messages(report, FeedbackConfig(), {0}, {0})
    row = json.loads(messages[1]["content"])["report"]["labels"][0]
    assert row["score_location"]["phrase"] == "bar 13"
    assert "spans" not in row["score_location"]
    assert row["reference_available"] is True


def test_partial_bar_note_count_includes_notes_before_selection(tmp_path):
    write_score(tmp_path / "full_score.musicxml", range(1, 14))
    selected = write_score(tmp_path / "verified_score.musicxml", [1], pitches=("E4", "F4"))
    (tmp_path / "metadata.json").write_text(json.dumps({"score_segment": {"start_measure": 12, "end_measure": 12, "start_beat": 3}}))
    report = prepare_report([{"type": "wrong_note", "note_id": "note_0000"}])
    add_score_locations(report, selected)
    assert report["labels"][0]["score_location"]["phrase"] == "the 3rd note of bar 12"


def test_partial_bar_without_full_score_uses_bar_only(tmp_path):
    selected = write_score(tmp_path / "verified_score.musicxml", [1], pitches=("E4", "F4"))
    (tmp_path / "metadata.json").write_text(json.dumps({"score_segment": {"start_measure": 12, "end_measure": 12, "start_beat": 3}}))
    report = prepare_report([{"type": "wrong_note", "note_id": "note_0000"}])
    add_score_locations(report, selected)
    assert report["labels"][0]["score_location"]["phrase"] == "bar 12"


def test_full_score_measure_enumeration_is_used(tmp_path):
    write_score(tmp_path / "full_score.musicxml", [0, 1, 2])
    selected = write_score(tmp_path / "verified_score.musicxml", [2])
    report = prepare_report([{"type": "wrong_note", "note_id": "note_0000"}])
    add_score_locations(report, selected)
    assert report["labels"][0]["score_location"]["phrase"] == "the 1st note of bar 3"


def test_reference_midi_uses_tempo_changes_and_written_pitch_transposition(tmp_path):
    score = write_score(tmp_path / "verified_score.musicxml", [1])
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    for index, pitch in enumerate([58, 60, 62, 63]):  # written C D E F, sounding down two
        if index in (0, 2):
            track.append(mido.MetaMessage("set_tempo", tempo=500000 if index == 0 else 1000000))
        track.append(mido.Message("note_on", note=pitch, velocity=80))
        track.append(mido.Message("note_off", note=pitch, time=480))
    path = tmp_path / "reference_audio.mid"
    midi.save(path)
    events = reference_note_times(score, path)
    assert [e["start"] for e in events] == [0, .5, 1, 2]
    assert [e["end"] for e in events] == [.5, 1, 2, 3]
    track[4].note = 61
    midi.save(path)
    with pytest.raises(ValueError, match="disagree|match"):
        reference_note_times(score, path)


def test_rendered_trill_and_grace_keep_canonical_identity(tmp_path):
    from music21 import expressions
    score, part = stream.Score(), stream.Part()
    part.append(note.Note('C4', quarterLength=1))
    trill = note.Note('D4', quarterLength=1)
    trill.expressions.append(expressions.Trill())
    part.append(trill)
    part.append(note.Note('D4').getGrace())
    part.append(note.Note('E4', quarterLength=1))
    part.append(note.Note('F4', quarterLength=1))
    score.append(part)
    path = tmp_path / 'score.musicxml'
    score.write('musicxml', fp=path)
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    # Rendered written C, D-E-D-E trill, grace D, E, F; sounding down two.
    for pitch, duration in [(58,480),(60,120),(62,120),(60,120),(62,120),
                            (60,60),(62,420),(63,480)]:
        track.append(mido.Message('note_on', note=pitch, velocity=80))
        track.append(mido.Message('note_off', note=pitch, time=duration))
        if len(track) == 2:
            track.append(mido.MetaMessage('set_tempo', tempo=1000000))
    midi_path = tmp_path / 'reference.mid'
    midi.save(midi_path)
    events = reference_note_times(path, midi_path)
    assert len(events) == 4
    assert [e['start'] for e in events] == [0, .5, 1.5, 2.5]
    assert [e['end'] for e in events] == [.5, 1.5, 2.5, 3.5]
    # A wrong pitch inside an ornament must not be accepted as arbitrary decoration.
    track[5].note = track[6].note = 70
    midi.save(midi_path)
    with pytest.raises(ValueError, match='ornament timeline'):
        reference_note_times(path, midi_path)


def test_extra_midi_notes_without_notated_ornaments_are_rejected(tmp_path):
    path = write_score(tmp_path / 'score.musicxml', [1])
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    for pitch, duration in [(60,240),(60,240),(62,480),(64,480),(65,480)]:
        track.append(mido.Message('note_on', note=pitch, velocity=80))
        track.append(mido.Message('note_off', note=pitch, time=duration))
    midi_path = tmp_path / 'reference.mid'
    midi.save(midi_path)
    with pytest.raises(ValueError, match='ornament timeline'):
        reference_note_times(path, midi_path)


@pytest.mark.parametrize('variant', ['valid', 'wrong_pitch', 'wrong_timing', 'unslashed', 'absent'])
def test_musescore_early_grace_in_sixteenth_run(tmp_path, variant):
    """Bar 144: the written grace before A is played before the preceding G#."""
    score, part = stream.Score(), stream.Part()
    for pitch in ['D6', 'A5', 'G#5']:
        part.append(note.Note(pitch, quarterLength=.25))
    if variant != 'absent':
        grace = note.Note('B5').getGrace()
        grace.duration.slash = variant != 'unslashed'
        part.append(grace)
    part.append(note.Note('A5', quarterLength=.25))
    part.append(note.Note('G5', quarterLength=1))
    score.append(part)
    path = tmp_path / 'score.musicxml'
    score.write('musicxml', fp=path)
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack(); midi.tracks.append(track)
    # Sounding down two semitones; MuseScore releases notes one tick early.
    split = 40 if variant == 'wrong_timing' else 60
    for pitch, ticks in [(84,120),(79,split),(82 if variant == 'wrong_pitch' else 81,120-split),
                         (78,120),(79,120),(77,480)]:
        track.append(mido.Message('note_on', note=pitch, velocity=80))
        track.append(mido.Message('note_off', note=pitch, time=ticks-1))
        track.append(mido.MetaMessage('text', text='', time=1))
    midi_path = tmp_path / 'reference.mid'
    midi.save(midi_path)
    if variant == 'valid':
        result = reference_note_times(path, midi_path)
        assert [e['pitch'] for e in result] == [84,79,78,79,77]
        assert result[1]['start'] == pytest.approx(.125)
        assert result[1]['end'] == pytest.approx(239/960)
    else:
        with pytest.raises(ValueError, match='ornament timeline'):
            reference_note_times(path, midi_path)


def test_octave_chords_keep_canonical_indices_and_validate_all_tones(tmp_path):
    from music21 import chord, tie
    from datacreate.feedback_score import ReferenceMismatchError
    score, part = stream.Score(), stream.Part()
    part.append(note.Note('C4', quarterLength=1))
    for kind in ('start', 'stop'):
        tones = chord.Chord(['D4', 'D5'], quarterLength=1)
        tones.tie = tie.Tie(kind)
        part.append(tones)
    part.append(note.Note('E4', quarterLength=1))
    score.append(part)
    path = tmp_path / 'score.musicxml'
    score.write('musicxml', fp=path)
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    for kind, pitch, ticks in [('note_on',58,0),('note_off',58,480),
                              ('note_on',60,0),('note_on',72,0),
                              ('note_off',60,960),('note_off',72,0),
                              ('note_on',62,0),('note_off',62,480)]:
        track.append(mido.Message(kind, note=pitch, velocity=80, time=ticks))
    midi_path = tmp_path / 'reference.mid'
    midi.save(midi_path)
    result = reference_note_times(path, midi_path)
    assert [e['pitch'] for e in result] == [58, 62]
    assert [e['start'] for e in result] == [0, 1.5]
    track[3].note = track[5].note = 73
    midi.save(midi_path)
    with pytest.raises(ReferenceMismatchError, match='chord notes'):
        reference_note_times(path, midi_path)


@pytest.mark.parametrize('valid', [True, False])
def test_reference_rebuild_validates_before_replacing_originals(tmp_path, monkeypatch, valid):
    from datacreate.config import PipelineConfig
    from datacreate.feedback_score import regenerate_reference, ReferenceMismatchError
    from datacreate.tools import musescore
    score = write_score(tmp_path / 'score.musicxml', [1])
    midi = tmp_path / 'reference.mid'
    audio = tmp_path / 'reference.wav'
    midi.write_bytes(b'old midi'); audio.write_bytes(b'old audio')
    fresh = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack(); fresh.tracks.append(track)
    for pitch in ([60,62,64,65] if valid else [60,61,64,65]):
        track.append(mido.Message('note_on', note=pitch, velocity=80))
        track.append(mido.Message('note_off', note=pitch, time=480))
    monkeypatch.setattr(musescore, 'export_score_to_midi', lambda config, score, dest, log: fresh.save(dest))
    monkeypatch.setattr(musescore, 'render_midi_to_wav', lambda midi, dest, config, log: dest.write_bytes(b'fresh audio'))
    if valid:
        regenerate_reference(score, midi, config=PipelineConfig())
        assert len(reference_note_times(score, midi)) == 4
        assert audio.read_bytes() == b'fresh audio'
        backup, = tmp_path.glob('reference-before-rebuild-*')
        assert (backup / midi.name).read_bytes() == b'old midi'
        assert (backup / audio.name).read_bytes() == b'old audio'
    else:
        with pytest.raises(ReferenceMismatchError):
            regenerate_reference(score, midi, config=PipelineConfig())
        assert midi.read_bytes() == b'old midi' and audio.read_bytes() == b'old audio'
