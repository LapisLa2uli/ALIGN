from datacreate.melody import (
    ScoreSoundingNote,
    WeakMelody,
    extra_neighbor_core,
    is_contiguous_part,
    match_melodies,
    melody_similarity,
    padded_melody,
)


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
        WeakMelody(pitches=[67, 69, 71], type="extra_note"),
        WeakMelody(pitches=[60, 62, 64], type="wrong_note"),
    ]
    f1, precision = match_melodies(gold, pred)
    assert f1 == 1.0
    assert precision == 1.0


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
