from alignmodel.melody import (
    WeakMelody,
    extra_neighbor_core,
    gold_melodies_from_labels,
    match_melodies,
    match_melodies_detail,
    pred_melodies_from_labels,
)
from datacreate.melody import ScoreSoundingNote


def test_gold_skips_repeated_pass_and_dedupes():
    labels = [
        {
            "type": "wrong_note",
            "comment": "shifted -1 (60 -> 59) (first pass)",
            "pitches": [59, 60, 62],
        },
        {
            "type": "wrong_note",
            "comment": "shifted -1 (60 -> 59) (repeated pass)",
            "pitches": [59, 60, 62],
        },
        {
            "type": "repetition",
            "comment": "repeated measure containing wrong_note",
            "pitches": [55, 57, 59, 60, 62],
            "extra_copies": 1,
        },
    ]
    gold = gold_melodies_from_labels(labels)
    assert [g.type for g in gold] == ["wrong_note", "repetition"]
    assert gold[1].extra_copies == 1


def test_empty_match_is_perfect():
    f1, precision = match_melodies([], [])
    assert f1 == 1.0
    assert precision == 1.0


def test_pred_miss_is_zero():
    gold = [WeakMelody(pitches=[60], type="wrong_note")]
    f1, precision = match_melodies(gold, [])
    assert f1 == 0.0
    assert precision == 0.0


def test_containment_detail_counts():
    gold = [WeakMelody(pitches=[60, 62, 64, 65])]
    pred = [
        WeakMelody(pitches=[62, 64]),
        WeakMelody(pitches=[70, 72]),
    ]
    detail = match_melodies_detail(gold, pred)
    assert detail["n_pred_correct"] == 1
    assert detail["n_gold_covered"] == 1
    assert detail["precision"] == 0.5
    assert detail["recall"] == 1.0


def test_pred_extra_maps_to_neighbors_plus_pad():
    notes = [
        ScoreSoundingNote(
            index=i,
            pitch=60 + i,
            start=float(i),
            end=float(i) + 0.8,
            ql_start=float(i),
            ql_end=float(i) + 1.0,
            measure=1,
            note_id=f"note_{i:04d}",
        )
        for i in range(6)
    ]
    assert extra_neighbor_core(notes, 2) == (2, 4)
    pred = pred_melodies_from_labels(
        [{"type": "extra_note", "start_time": 2.1, "end_time": 2.4}],
        notes,
        pad_notes=1,
    )
    assert len(pred) == 1
    assert pred[0].pitches == [61, 62, 63, 64]
