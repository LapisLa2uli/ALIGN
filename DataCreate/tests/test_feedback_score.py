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
