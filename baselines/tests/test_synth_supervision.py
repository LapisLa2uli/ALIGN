from pathlib import Path
import sys

import mido
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from synth_supervision import MidiTimeline, SupervisionError, make_labels


def _midi(path, notes, tempos=((0, 500000),)):
    midi = mido.MidiFile(ticks_per_beat=480)
    events = [(round(q * 480), mido.MetaMessage("set_tempo", tempo=t)) for q, t in tempos]
    for pitch, start, end in notes:
        events.extend([(round(start * 480), mido.Message("note_on", note=pitch, velocity=90)),
                       (round(end * 480), mido.Message("note_off", note=pitch))])
    events.sort(key=lambda row: row[0])
    track, previous = mido.MidiTrack(), 0
    for tick, message in events:
        track.append(message.copy(time=tick-previous))
        previous = tick
    midi.tracks.append(track)
    midi.save(path)
    return MidiTimeline(path)


def _note(pitch, start, duration=1, clean=None, copy=0):
    return dict(pitch_midi=pitch+2, onset_ql=start, duration_ql=duration,
                clean_index=clean, copy_pass=copy)


def test_missing_timing_follows_final_tempo_and_repeat_is_extra(tmp_path):
    ref = _midi(tmp_path / "ref.mid", [(60,0,1), (62,1,2), (64,2,3)])
    perf = _midi(tmp_path / "perf.mid", [(61,0,1), (64,2,3), (61,4,5)], ((0,500000),(1,1000000)))
    lineage = dict(clean_notes=[_note(60,0),_note(62,1),_note(64,2)],
                   performed_notes=[_note(61,0,clean=0),_note(64,2,clean=2),_note(61,4,clean=0,copy=1)])
    result = make_labels(lineage, ref, perf, {1:(1,2)})
    assert [(n["cls"],n["sub"]) for n in result["performance_notes"]] == [
        ("extra","wrong"),("correct","correct"),("extra","repeat")]
    assert [(n["sounding_pitch"],n["onset"],n["offset"]) for n in result["missed_notes"]] == [
        (60,0,0.5),(62,0.5,1.5)]


def test_partial_wrong_tie_is_rejected(tmp_path):
    ref = _midi(tmp_path / "ref.mid", [(60,0,2)])
    perf = _midi(tmp_path / "perf.mid", [(61,0,1),(60,1,2)])
    lineage = dict(clean_notes=[_note(60,0),_note(60,1)],
                   performed_notes=[_note(61,0,clean=0),_note(60,1,clean=1)])
    with pytest.raises(SupervisionError, match="partially_changed_reference_tie"):
        make_labels(lineage, ref, perf, {})


def test_missing_note_requires_injected_rest_lineage(tmp_path):
    ref = _midi(tmp_path / "ref.mid", [(60,0,1),(62,1,2)])
    perf = _midi(tmp_path / "perf.mid", [(60,0,1)])
    lineage = dict(clean_notes=[_note(60,0),_note(62,1)], performed_notes=[_note(60,0,clean=0)])
    with pytest.raises(SupervisionError, match="without_injected_rest"):
        make_labels(lineage, ref, perf, {})
