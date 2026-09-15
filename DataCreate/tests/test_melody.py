from pathlib import Path

from music21 import duration, meter, note, stream, tempo, tie

from datacreate.melody import (
    ScoreSoundingNote,
    WeakMelody,
    extra_neighbor_core,
    is_contiguous_part,
    match_melodies,
    match_melodies_detail,
    match_note_wise_labels_detail,
    melody_similarity,
    padded_melody,
    parse_sounding_notes,
)


def _located(kind: str, first: int, last: int, **extra):
    return {
        "type": kind,
        "score_part": {
            "start_note_index": first,
            "end_note_index": last,
        },
        **extra,
    }


def _notes(n: int, start_measure: int = 1) -> list[ScoreSoundingNote]:
    out = []
    for i in range(n):
        out.append(
            ScoreSoundingNote(
                index=i,
                pitch=60 + i,
                start=float(i),
                end=float(i) + 0.5,
                ql_start=float(i),
                ql_end=float(i) + 1.0,
                measure=start_measure + i // 4,
                note_id=f"note_{i:04d}",
            )
        )
    return out


def test_pad_clamps_at_phrase_start():
    notes = _notes(5)
    span = padded_melody(notes, 0, 1, pad_notes=2)
    assert span.start_note_index == 0
    assert span.end_note_index == 2
    assert span.pitches == [60, 61, 62]
    assert span.pad_notes == 2


def test_pad_clamps_at_phrase_end():
    notes = _notes(5)
    span = padded_melody(notes, 4, 5, pad_notes=2)
    assert span.start_note_index == 2
    assert span.end_note_index == 4
    assert span.pitches == [62, 63, 64]


def test_pad_both_sides():
    notes = _notes(8)
    span = padded_melody(notes, 3, 4, pad_notes=2)
    assert span.start_note_index == 1
    assert span.end_note_index == 5
    assert span.pitches == [61, 62, 63, 64, 65]


def test_extra_neighbor_core_middle_then_pad():
    notes = _notes(8)
    i0, i1 = extra_neighbor_core(notes, 3)
    assert (i0, i1) == (3, 5)
    span = padded_melody(notes, i0, i1, pad_notes=2)
    assert span.pitches == [61, 62, 63, 64, 65, 66]


def test_extra_neighbor_core_last_note_has_no_after():
    notes = _notes(5)
    assert extra_neighbor_core(notes, 4) == (4, 5)


def test_melody_similarity_lcs():
    assert melody_similarity([1, 2, 3], [1, 2, 3]) == 1.0
    assert melody_similarity([1, 2, 3], [1, 2, 3, 4]) == 2 * 3 / 7
    assert melody_similarity([], [1]) == 0.0
    assert melody_similarity([], []) == 1.0


def test_contiguous_part_is_a_slice_not_a_subsequence():
    assert is_contiguous_part([62, 64], [60, 62, 64, 65])
    assert is_contiguous_part([60, 62, 64], [60, 62, 64])
    assert not is_contiguous_part([60, 64], [60, 62, 64])
    assert not is_contiguous_part([60, 62, 64, 65], [60, 62, 64])


def test_match_melodies_set_equal_and_near():
    gold = [
        WeakMelody(pitches=[60, 62, 64], type="wrong_note"),
        WeakMelody(pitches=[67, 69, 71], type="repetition"),
    ]
    pred = [
        WeakMelody(pitches=[67, 69, 71], type="repetition"),
        WeakMelody(pitches=[60, 62, 64], type="wrong_note"),
    ]
    f1, precision = match_melodies(gold, pred)
    assert f1 == 1.0
    assert precision == 1.0


def test_match_melodies_wrong_type_is_half():
    gold = [WeakMelody(pitches=[60, 62, 64], type="wrong_note")]
    pred = [WeakMelody(pitches=[60, 62, 64], type="extra_note")]
    f1, precision = match_melodies(gold, pred)
    assert f1 == 0.5
    assert precision == 0.5


def test_ignore_type_gives_full_credit_on_type_mismatch():
    gold = [WeakMelody(pitches=[60, 62, 64], type="wrong_note")]
    pred = [WeakMelody(pitches=[60, 62, 64], type="extra_note")]
    sensitive = match_melodies_detail(gold, pred, ignore_type=False)
    insensitive = match_melodies_detail(gold, pred, ignore_type=True)
    assert sensitive["f1"] == 0.5
    assert insensitive["f1"] == 1.0
    assert insensitive["precision"] == 1.0
    assert insensitive["recall"] == 1.0


def test_pred_inside_gold_is_not_a_match():
    gold = [WeakMelody(pitches=[60, 62, 64, 65, 67])]
    pred = [WeakMelody(pitches=[62, 64, 65])]
    f1, precision = match_melodies(gold, pred)
    assert f1 == 0.0
    assert precision == 0.0


def test_gold_inside_pred_is_not_a_match():
    gold = [WeakMelody(pitches=[62, 64])]
    pred = [WeakMelody(pitches=[60, 62, 64, 65])]
    f1, precision = match_melodies(gold, pred)
    assert f1 == 0.0
    assert precision == 0.0


def test_unmatched_pred_lowers_precision():
    gold = [WeakMelody(pitches=[60, 62], type="wrong_note")]
    pred = [
        WeakMelody(pitches=[60, 62], type="wrong_note"),
        WeakMelody(pitches=[72, 74], type="extra_note"),
    ]
    f1, precision = match_melodies(gold, pred)
    assert precision == 0.5
    assert abs(f1 - (2 * 0.5 * 1.0) / 1.5) < 1e-9


def test_exclusive_one_gold_one_pred():
    gold = [WeakMelody(pitches=[60, 62, 64, 65])]
    pred = [
        WeakMelody(pitches=[60, 62, 64, 65]),
        WeakMelody(pitches=[60, 62, 64, 65]),
    ]
    f1, precision = match_melodies(gold, pred)
    assert precision == 0.5
    assert abs(f1 - (2 * 0.5 * 1.0) / 1.5) < 1e-9


def test_parse_sounding_notes_skips_grace_and_folds_ties(tmp_path: Path):
    part = stream.Part()
    part.insert(0, tempo.MetronomeMark(number=60))
    part.insert(0, meter.TimeSignature("4/4"))
    measure = stream.Measure(number=1)
    grace = note.Note("D5")
    grace.duration = duration.GraceDuration(0.25)
    start = note.Note("C4", quarterLength=2.0)
    start.tie = tie.Tie("start")
    stop = note.Note("C4", quarterLength=2.0)
    stop.tie = tie.Tie("stop")
    measure.append(grace)
    measure.append(start)
    measure.append(stop)
    part.append(measure)
    score = stream.Score()
    score.insert(0, part)
    path = tmp_path / "sounding.musicxml"
    score.write("musicxml", fp=str(path))
    notes = parse_sounding_notes(path)
    assert [n.pitch for n in notes] == [60]
    assert abs(notes[0].ql_end - notes[0].ql_start - 4.0) < 1e-6


def test_note_wise_exact_location_and_fractional_type_credit():
    exact = match_note_wise_labels_detail(
        [_located("wrong_note", 2, 2)],
        [_located("wrong_note", 2, 2)],
    )
    wrong_type = match_note_wise_labels_detail(
        [_located("wrong_note", 2, 2)],
        [_located("rhythm_error", 2, 2)],
    )
    assert exact["f1"] == 1.0
    assert wrong_type["f1"] == 0.5


def test_note_wise_same_pitch_wrong_location_is_zero():
    gold = _located("wrong_note", 1, 1, pitches=[60])
    pred = _located("wrong_note", 8, 8, pitches=[60])
    assert match_note_wise_labels_detail([gold], [pred])["f1"] == 0.0


def test_note_wise_duplicate_is_exclusive_and_global():
    duplicate = match_note_wise_labels_detail(
        [_located("wrong_note", 1, 1)],
        [
            _located("wrong_note", 1, 1),
            _located("wrong_note", 1, 1),
        ],
    )
    assert duplicate["credit"] == 1.0
    assert duplicate["precision"] == 0.5
    global_result = match_note_wise_labels_detail(
        [
            _located("a", 1, 1),
            _located("b", 1, 1),
        ],
        [
            _located("a", 1, 1),
            _located("a", 1, 1),
        ],
    )
    assert global_result["credit"] == 1.5


def test_note_wise_extras_ties_repetition_and_empty():
    extra = match_note_wise_labels_detail(
        [{"type": "extra_note", "rendered_index": 0}],
        [{"type": "extra_note", "rendered_index": 0}],
    )
    tied = match_note_wise_labels_detail(
        [_located("wrong_note", 2, 4)],
        [_located("wrong_note", 2, 4)],
    )
    repetition = match_note_wise_labels_detail(
        [_located("repetition", 3, 6, extra_copies=2)],
        [_located("repetition", 3, 6, extra_copies=2)],
    )
    wrong_copy_count = match_note_wise_labels_detail(
        [_located("repetition", 3, 6, extra_copies=2)],
        [_located("repetition", 3, 6, extra_copies=1)],
    )
    empty = match_note_wise_labels_detail([], [])
    assert extra["f1"] == tied["f1"] == repetition["f1"] == empty["f1"] == 1.0
    assert wrong_copy_count["f1"] == 0.0


def test_schema_1_1_without_projection_is_unavailable():
    result = match_note_wise_labels_detail(
        [{"type": "wrong_note", "start_time": 1.0, "end_time": 2.0}],
        [],
    )
    assert result["status"] == "unavailable"
