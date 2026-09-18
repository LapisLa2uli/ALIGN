from __future__ import annotations

from pathlib import Path

from music21 import expressions, note, stream, tie

from alignmodel.joint.global_ornament_lineage_v2 import (
    reconstruct_global_ornament_lineage,
)
from alignmodel.joint.index import ScoreEventIndex


def _write(path: Path, notes: list[note.Note]) -> None:
    score = stream.Score()
    part = stream.Part(id="clarinet")
    measure = stream.Measure(number=1)
    for value in notes:
        measure.append(value)
    part.append(measure)
    score.append(part)
    score.write("musicxml", fp=path)


def _clean_row(index: int, pitch: int, onset: float) -> dict:
    return {
        "clean_index": index,
        "deleted": False,
        "pitch_midi": pitch,
        "pitch": note.Note(midi=pitch).pitch.nameWithOctave,
        "onset_ql": onset,
        "duration_ql": 1.0,
        "measure": 1,
    }


def _performed_row(
    index: int,
    pitch: int,
    onset: float,
    *,
    clean: int | None,
    relationship: str,
    copy_pass: int = 0,
) -> dict:
    return {
        "performed_index": index,
        "clean_index": clean,
        "relationship": relationship,
        "origin_relationship": (
            "extra" if clean is None else relationship
        ),
        "copy_pass": copy_pass,
        "pitch_midi": pitch,
        "pitch": note.Note(midi=pitch).pitch.nameWithOctave,
        "onset_ql": onset,
        "duration_ql": 1.0,
        "measure": 1,
    }


def _lineage(clean: list[dict], performed: list[dict], deleted=()) -> dict:
    for value in deleted:
        clean[value]["deleted"] = True
    return {
        "schema_version": "1.0",
        "kind": "synth_note_lineage",
        "clean_note_count": len(clean),
        "performed_note_count": len(performed),
        "clean_notes": clean,
        "performed_notes": performed,
        "deleted_clean_notes": list(deleted),
        "rendered_notes": [],
    }


def _midi(pitches: list[int]) -> dict:
    return {
        "notes": [
            {
                "pitch": pitch,
                "start": index * 0.1,
                "end": index * 0.1 + 0.09,
            }
            for index, pitch in enumerate(pitches)
        ]
    }


def test_mordent_and_planted_extra_are_distinguished(tmp_path: Path) -> None:
    verified = tmp_path / "verified.musicxml"
    performance = tmp_path / "performance.musicxml"
    principal = note.Note("C4", quarterLength=1.0)
    principal.expressions.append(expressions.Mordent())
    planted = note.Note("D4", quarterLength=1.0)
    _write(verified, [note.Note("C4", quarterLength=1.0)])
    _write(performance, [principal, planted])
    result = reconstruct_global_ornament_lineage(
        _lineage(
            [_clean_row(0, 60, 0.0)],
            [
                _performed_row(0, 60, 0.0, clean=0, relationship="match"),
                _performed_row(1, 62, 1.0, clean=None, relationship="extra"),
            ],
        ),
        performance,
        verified,
        _midi([60, 59, 60, 62]),
        written_shift=0,
    )
    origins = [row["extra_origin"] for row in result.lineage["rendered_notes"]]
    assert origins == [
        "generator_performed_lineage",
        "renderer_ornament",
        "renderer_ornament",
        "planted_extra",
    ]
    index = ScoreEventIndex.from_musicxml(verified, result.lineage)
    assert [event.relationship for event in index.rendered_events] == [
        "match",
        "extra",
        "extra",
        "extra",
    ]


def test_substitution_and_delete_identities_survive(tmp_path: Path) -> None:
    verified = tmp_path / "verified.musicxml"
    performance = tmp_path / "performance.musicxml"
    _write(
        verified,
        [
            note.Note("C4", quarterLength=1.0),
            note.Note("D4", quarterLength=1.0),
        ],
    )
    _write(performance, [note.Note("C#4", quarterLength=1.0)])
    result = reconstruct_global_ornament_lineage(
        _lineage(
            [_clean_row(0, 60, 0.0), _clean_row(1, 62, 1.0)],
            [
                _performed_row(
                    0, 61, 0.0, clean=0, relationship="substitute"
                )
            ],
            deleted=(1,),
        ),
        performance,
        verified,
        _midi([61]),
        written_shift=0,
    )
    index = ScoreEventIndex.from_musicxml(verified, result.lineage)
    assert index.rendered_events[0].relationship == "substitute"
    assert index.rendered_events[0].score_span == (0, 1)
    assert index.deleted_event_indices == frozenset({1})


def test_repeat_copy_and_tied_chain_are_legal(tmp_path: Path) -> None:
    verified = tmp_path / "verified.musicxml"
    performance = tmp_path / "performance.musicxml"
    _write(verified, [note.Note("C4", quarterLength=1.0)])
    _write(
        performance,
        [
            note.Note("C4", quarterLength=1.0),
            note.Note("C4", quarterLength=1.0),
        ],
    )
    repeated = reconstruct_global_ornament_lineage(
        _lineage(
            [_clean_row(0, 60, 0.0)],
            [
                _performed_row(0, 60, 0.0, clean=0, relationship="match"),
                _performed_row(
                    1,
                    60,
                    1.0,
                    clean=0,
                    relationship="copy",
                    copy_pass=1,
                ),
            ],
        ),
        performance,
        verified,
        _midi([60, 60]),
        written_shift=0,
    )
    repeat_index = ScoreEventIndex.from_musicxml(verified, repeated.lineage)
    assert [event.copy_pass for event in repeat_index.rendered_events] == [0, 1]

    tied_verified = tmp_path / "tied-verified.musicxml"
    tied_performance = tmp_path / "tied-performance.musicxml"
    first = note.Note("C4", quarterLength=1.0)
    first.tie = tie.Tie("start")
    second = note.Note("C4", quarterLength=1.0)
    second.tie = tie.Tie("stop")
    _write(tied_verified, [first, second])
    first_perf = note.Note("C4", quarterLength=1.0)
    first_perf.tie = tie.Tie("start")
    second_perf = note.Note("C4", quarterLength=1.0)
    second_perf.tie = tie.Tie("stop")
    _write(tied_performance, [first_perf, second_perf])
    tied = reconstruct_global_ornament_lineage(
        _lineage(
            [_clean_row(0, 60, 0.0), _clean_row(1, 60, 1.0)],
            [
                _performed_row(0, 60, 0.0, clean=0, relationship="match"),
                _performed_row(1, 60, 1.0, clean=1, relationship="match"),
            ],
        ),
        tied_performance,
        tied_verified,
        _midi([60]),
        written_shift=0,
    )
    tied_index = ScoreEventIndex.from_musicxml(tied_verified, tied.lineage)
    assert len(tied_index.events) == 1
    assert tied.lineage["rendered_notes"][0]["performed_indices"] == [0, 1]
