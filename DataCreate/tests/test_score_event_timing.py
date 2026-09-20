from __future__ import annotations

import json

from music21 import meter, note, stream

from datacreate.note_alignment import build_score_events


def test_score_events_prefer_note_first_performance_times(tmp_path):
    sample = tmp_path / "001"
    sample.mkdir()

    part = stream.Part()
    part.insert(0, meter.TimeSignature("4/4"))
    measure = stream.Measure(number=1)
    measure.append(note.Note("C4", quarterLength=1.0))
    measure.append(note.Rest(quarterLength=1.0))
    measure.append(note.Note("D4", quarterLength=1.0))
    part.append(measure)
    score = stream.Score()
    score.insert(0, part)
    score.write("musicxml", fp=str(sample / "verified_score.musicxml"))

    (sample / "note_alignment_v2.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "engine": "align-joint",
                "events": [
                    {
                        "score_index": 0,
                        "perf_start": 2.0,
                        "perf_end": 2.4,
                        "is_repetition": False,
                    },
                    {
                        "score_index": 0,
                        "perf_start": 8.0,
                        "perf_end": 8.4,
                        "is_repetition": True,
                    },
                    {
                        "score_index": 1,
                        "perf_start": 4.5,
                        "perf_end": 5.0,
                        "is_repetition": False,
                    },
                ],
                "transcribed_notes": [],
                "note_mapping": [],
            }
        ),
        encoding="utf-8",
    )

    payload = build_score_events(sample)
    sounding = [event for event in payload["events"] if not event["is_rest"]]
    assert payload["performance_timing_source"] == "note_alignment_v2"
    assert sounding[0]["perf_start"] == 2.0
    assert sounding[0]["perf_end"] == 2.4
    assert sounding[1]["perf_start"] == 4.5
    assert sounding[1]["perf_end"] == 5.0
    assert all(
        event["performance_timing_source"] == "note_alignment_v2"
        for event in sounding
    )
