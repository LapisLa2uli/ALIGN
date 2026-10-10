import json

import numpy as np
import pytest
import soundfile as sf

from datacreate.feedback import prepare_report
from datacreate.feedback_synthesis import prepare_examples, render_example, slow_pair


@pytest.mark.parametrize('failure', ['repaired', 'still_mismatched', 'alignment'])
def test_feedback_recovers_reference_once_before_narration(tmp_path, synthesis_inputs, monkeypatch, failure):
    from datacreate import feedback_score, feedback_synthesis
    from datacreate.feedback import run_feedback, FeedbackError
    synthesis_inputs(tmp_path)
    source = tmp_path / 'labels.json'
    source.write_text(json.dumps({'labels': [{'type':'wrong_note', 'score_event_indices':[1]}]}))
    original = feedback_synthesis.prepare_examples
    calls, rebuilds, progress = [], [], []
    def prepare(*args):
        calls.append(True)
        if failure == 'alignment':
            raise ValueError('Invalid alignment')
        if len(calls) == 1 or failure == 'still_mismatched':
            raise feedback_score.ReferenceMismatchError('Mismatched reference')
        return original(*args)
    monkeypatch.setattr(feedback_synthesis, 'prepare_examples', prepare)
    monkeypatch.setattr(feedback_score, 'regenerate_reference', lambda *args, **kwargs: rebuilds.append(kwargs))
    kwargs = dict(dry_run=True, output_dir=tmp_path / 'output', detail_progress=progress.append)
    if failure == 'repaired':
        output = run_feedback(source, **kwargs)
        assert (output / 'request.json').is_file()
        assert any('Regenerating' in p['message'] for p in progress)
    else:
        with pytest.raises(FeedbackError):
            run_feedback(source, **kwargs)
    assert len(rebuilds) == (0 if failure == 'alignment' else 1)
    assert len(calls) == (1 if failure == 'alignment' else 2)


def examples(directory, labels):
    report = prepare_report(labels)
    result = prepare_examples(report, directory, directory / "verified_score.musicxml", directory / "reference_audio.mid")
    return report, result


def test_shared_slowdown_preserves_every_relative_time():
    reference = [{"pitch": 60, "start": 0., "end": .1}, {"pitch": 62, "start": .1, "end": .2}]
    performance = [{"pitch": 61, "start": .03, "end": .17}, {"pitch": 62, "start": .2, "end": .45}]
    ref, perf, factor = slow_pair(reference, performance, 2, 2.7)
    assert 1 < factor <= 4
    for original, scaled in [(reference, ref), (performance, perf)]:
        for a, b in zip(original, scaled):
            assert a["pitch"] == b["pitch"]
            assert b["start"] == pytest.approx(a["start"] * factor)
            assert b["end"] == pytest.approx(a["end"] * factor)
    assert slow_pair(reference, performance, .05, .1)[2] == 4
    assert slow_pair([{"pitch": 60, "start": 0, "end": 4}], [], 4, 4)[2] == 1


def test_whole_bar_preserves_extras_wrong_pitches_and_gaps(tmp_path, synthesis_inputs):
    original = synthesis_inputs(tmp_path)
    path = tmp_path / "note_alignment_v2.json"
    data = json.loads(path.read_text())
    data["transcribed_notes"][1]["pitch"] = 63
    data["transcribed_notes"][1]["alignment_pitch"] = 62
    data["transcribed_notes"].append({"pitch": 99, "start": .3, "end": .5, "ignored": True})
    path.write_text(json.dumps(data))
    report, pairs = examples(tmp_path, [{"type": "wrong_note", "score_event_indices": [1]}])
    ref, perf = pairs[0]["reference"], pairs[0]["performance"]
    assert ref["bars"] == perf["bars"] == [2]
    assert len(ref["notes"]) == 4
    assert [n["pitch"] for n in perf["notes"]] == [60, 63, 69, 64, 65]
    factor = ref["slowdown_factor"]
    assert factor == perf["slowdown_factor"]
    for old, new in zip(original, perf["notes"]):
        assert new["start"] == pytest.approx((old["start"] - .2)*factor)
        assert new["end"] == pytest.approx((old["end"] - .2)*factor)
    assert report["labels"][0]["playback_example"]["whole_bars"] is True


def test_missed_note_is_not_inserted_into_transcription(tmp_path, synthesis_inputs):
    synthesis_inputs(tmp_path)
    path = tmp_path / "note_alignment_v2.json"
    data = json.loads(path.read_text())
    data["transcribed_notes"] = [n for n in data["transcribed_notes"] if n["pitch"] != 64]
    data["events"] = [e for e in data["events"] if e["sounding_index"] != 2]
    path.write_text(json.dumps(data))
    _, pairs = examples(tmp_path, [{"type": "missed_note", "score_event_indices": [2]}])
    assert 64 in [n["pitch"] for n in pairs[0]["reference"]["notes"]]
    assert 64 not in [n["pitch"] for n in pairs[0]["performance"]["notes"]]


@pytest.mark.parametrize("pitch_space,expected_shift", [("written", -2), ("sounding", 0), (None, -2)])
def test_transposing_instrument_uses_sounding_pitches(tmp_path, synthesis_inputs, pitch_space, expected_shift):
    import mido
    original = synthesis_inputs(tmp_path)
    midi_path = tmp_path / "reference_audio.mid"
    midi = mido.MidiFile(midi_path)
    for track in midi.tracks:
        for message in track:
            if message.type in {"note_on", "note_off"}:
                message.note -= 2
    midi.save(midi_path)
    alignment_path = tmp_path / "note_alignment_v2.json"
    data = json.loads(alignment_path.read_text())
    data["provenance"] = {"pitch_convention": "written Bb-clarinet; sounding = written - 2"}
    if pitch_space is not None:
        data["pitch_space"] = pitch_space
    alignment_path.write_text(json.dumps(data))
    _, pairs = examples(tmp_path, [{"type": "extra_note", "score_event_indices": [1]}])
    reference, performance = pairs[0]["reference"], pairs[0]["performance"]
    assert [n["pitch"] for n in reference["notes"]] == [58, 60, 62, 63]
    assert [n["pitch"] for n in performance["notes"]] == [n["pitch"]+expected_shift for n in original]
    assert performance["pitch_shift_semitones"] == expected_shift
    for source, rendered in zip(original, performance["notes"]):
        factor = performance["slowdown_factor"]
        assert rendered["start"] == pytest.approx((source["start"]-.2)*factor)
        assert rendered["end"] == pytest.approx((source["end"]-.2)*factor)
    # A supplied concert-pitch transcription takes priority over ALIGN's axes.
    override = tmp_path / "concert.json"
    override.write_text(json.dumps({"pitch_space": "sounding", "transcribed_notes": original}))
    report = prepare_report([{"type": "extra_note", "score_event_indices": [1]}])
    overridden = prepare_examples(report, tmp_path, tmp_path / "verified_score.musicxml", midi_path,
                                  transcription_path=override)[0]["performance"]
    assert [n["pitch"] for n in overridden["notes"]] == [n["pitch"] for n in original]


def test_cross_bar_range_and_full_score_enumeration(tmp_path, synthesis_inputs):
    from test_feedback_score import write_score
    import mido
    synthesis_inputs(tmp_path)
    write_score(tmp_path / "full_score.musicxml", range(1, 15))
    write_score(tmp_path / "verified_score.musicxml", [1, 2])
    (tmp_path / "metadata.json").write_text(json.dumps({"score_segment": {"start_measure": 12, "end_measure": 13}}))
    midi_path = tmp_path / "reference_audio.mid"
    midi = mido.MidiFile(midi_path)
    midi.tracks[0].pop()  # end_of_track
    midi.tracks[0].extend([m.copy() for m in list(midi.tracks[0])])
    midi.save(midi_path)
    path = tmp_path / "note_alignment_v2.json"
    data = json.loads(path.read_text())
    data["events"] += [{**e, "sounding_index": e["sounding_index"]+4, "perf_start": e["perf_start"]+3,
                        "perf_end": e["perf_end"]+3} for e in list(data["events"])]
    data["transcribed_notes"] += [{**n, "start": n["start"]+3, "end": n["end"]+3} for n in list(data["transcribed_notes"])]
    path.write_text(json.dumps(data))
    report, pairs = examples(tmp_path, [{"type": "wrong_note", "score_event_indices": [3, 4]}])
    assert report["labels"][0]["playback_example"]["phrase"] == "bars 12 to 13"
    assert pairs[0]["reference"]["bars"] == [12, 13]
    assert len(pairs[0]["reference"]["notes"]) == 8
    assert len(pairs[0]["performance"]["notes"]) == 10


def test_partial_selection_fills_reference_but_not_unrecorded_performance(tmp_path, synthesis_inputs):
    from test_feedback_score import write_score
    import mido
    synthesis_inputs(tmp_path)
    write_score(tmp_path / "full_score.musicxml", range(1, 13))
    write_score(tmp_path / "verified_score.musicxml", [1], pitches=("E4", "F4"))
    (tmp_path / "metadata.json").write_text(json.dumps({"score_segment": {"start_measure": 12, "end_measure": 12, "start_beat": 3}}))
    midi_path = tmp_path / "reference_audio.mid"
    midi = mido.MidiFile(midi_path)
    del midi.tracks[0][:4]
    midi.save(midi_path)
    path = tmp_path / "note_alignment_v2.json"
    path.write_text(json.dumps({"events": [{"sounding_index": 0, "perf_start": .2, "perf_end": .8},
                                          {"sounding_index": 1, "perf_start": .9, "perf_end": 1.5}],
                               "transcribed_notes": [{"pitch": 64, "start": .2, "end": .8}, {"pitch": 65, "start": .9, "end": 1.5}]}))
    _, pairs = examples(tmp_path, [{"type": "wrong_note", "score_event_indices": [0]}])
    ref, perf = pairs[0]["reference"], pairs[0]["performance"]
    assert ref["bars"] == [12]
    assert [n["pitch"] for n in ref["notes"]] == [60, 62, 64, 65]
    assert [n["pitch"] for n in perf["notes"]] == [64, 65]
    assert perf["partial_recording"] is True


def test_no_alignment_fails_without_raw_audio_fallback(tmp_path, synthesis_inputs):
    synthesis_inputs(tmp_path)
    (tmp_path / "note_alignment_v2.json").unlink()
    with pytest.raises(ValueError, match="note_alignment"):
        examples(tmp_path, [{"type": "wrong_note", "score_event_indices": [0]}])


def test_reference_preserves_rendered_ornaments_and_tempo_changes(tmp_path, synthesis_inputs):
    import mido
    from music21 import converter, expressions, note
    synthesis_inputs(tmp_path)
    score_path = tmp_path / "verified_score.musicxml"
    score = converter.parse(score_path)
    list(score.recurse().getElementsByClass(note.Note))[1].expressions.append(expressions.Trill())
    score.write("musicxml", fp=score_path)
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    for i, (pitch, duration) in enumerate([(60, 480), (62, 120), (64, 120), (62, 120), (64, 120), (64, 480), (65, 480)]):
        if i == 1:
            track.append(mido.MetaMessage("set_tempo", tempo=1000000))
        track.append(mido.Message("note_on", note=pitch, velocity=80))
        track.append(mido.Message("note_off", note=pitch, time=duration))
    midi.save(tmp_path / "reference_audio.mid")
    _, pairs = examples(tmp_path, [{"type": "wrong_note", "score_event_indices": [1]}])
    ref = pairs[0]["reference"]
    assert [n["pitch"] for n in ref["notes"]] == [60, 62, 64, 62, 64, 64, 65]
    factor = ref["slowdown_factor"]
    assert [n["start"]/factor for n in ref["notes"]] == pytest.approx([0, .5, .75, 1., 1.25, 1.5, 2.5])


def test_renderer_schedules_notes_and_retains_silence(tmp_path, fake_soundfont):
    example = {"notes": [{"pitch": 60, "start": .1, "end": .2}, {"pitch": 64, "start": .4, "end": .6}],
               "duration_seconds": .8, "program": 71, "synthesized": True}
    path = tmp_path / "example.wav"
    meta = render_example(example, path)
    audio, rate = sf.read(path)
    assert rate == 44100 and len(audio)/rate == pytest.approx(.95)
    assert np.max(np.abs(audio[int(.25*rate):int(.35*rate)])) == 0
    assert np.max(np.abs(audio[int(.45*rate):int(.55*rate)])) > .01
    assert np.isfinite(audio).all()
    assert json.loads(path.with_suffix(".notes.json").read_text()) == example
    assert meta["note_count"] == 2 and meta["sha256"]
