from alignmodel.melody import (
    WeakMelody,
    extra_neighbor_core,
    gold_melodies_from_labels,
    match_melodies,
    match_melodies_detail,
    pred_melodies_from_labels,
)
from alignmodel.melody_model import decode_note_runs
from alignmodel.stages.gold import extra_copies_of, first_pass_labels, gap_span, replay_spans
from alignmodel.types import (
    PipelineConfig,
    PipelineLabel,
    PipelineState,
    RepeatRange,
    ScoreGraph,
    labels_document,
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


def test_set_match_equal_lists():
    gold = [WeakMelody(pitches=[60, 62, 64, 65], type="wrong_note")]
    pred = [WeakMelody(pitches=[60, 62, 64, 65], type="repetition")]
    f1, precision = match_melodies(gold, pred)
    assert f1 == 1.0
    assert precision == 1.0


def test_slice_and_containment_do_not_match():
    gold = [WeakMelody(pitches=[60, 62, 64, 65])]
    pred = [
        WeakMelody(pitches=[62, 64]),
        WeakMelody(pitches=[70, 72]),
    ]
    detail = match_melodies_detail(gold, pred)
    assert detail["n_matched"] == 0
    assert detail["precision"] == 0.0
    assert detail["recall"] == 0.0


def test_whole_score_pred_scores_poorly():
    gold = [
        WeakMelody(pitches=[60, 62, 64, 65]),
        WeakMelody(pitches=[67, 69, 71]),
    ]
    pred = [WeakMelody(pitches=list(range(50, 90)))]
    f1, precision = match_melodies(gold, pred)
    assert f1 == 0.0
    assert precision == 0.0


def test_exclusive_one_to_one():
    gold = [WeakMelody(pitches=[60, 62, 64])]
    pred = [
        WeakMelody(pitches=[60, 62, 64]),
        WeakMelody(pitches=[60, 62, 64]),
    ]
    detail = match_melodies_detail(gold, pred)
    assert detail["n_matched"] == 1
    assert detail["precision"] == 0.5
    assert detail["recall"] == 1.0


def test_near_equal_sequence_matches():
    gold = [WeakMelody(pitches=[60, 62, 64, 65])]
    pred = [WeakMelody(pitches=[60, 62, 64, 65, 67])]
    f1, precision = match_melodies(gold, pred)
    assert precision == 1.0
    assert f1 == 1.0


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


def test_first_pass_drops_repeated_pass():
    labels = [
        {"type": "wrong_note", "comment": "x (first pass)"},
        {"type": "wrong_note", "comment": "x (repeated pass)"},
        {"type": "wrong_note", "comment": "x (pass 2)"},
        {"type": "repetition", "comment": "restart"},
    ]
    kept = first_pass_labels(labels)
    assert [lab["type"] for lab in kept] == ["wrong_note", "repetition"]


def test_replay_and_gap_geometry():
    lab = {
        "type": "repetition",
        "start_time": 4.0,
        "end_time": 6.0,
        "extra_copies": 2,
        "repeats_label_range": {"start_time": 1.0, "end_time": 3.2},
    }
    assert extra_copies_of(lab) == 2
    assert replay_spans(lab) == [(4.0, 5.0), (5.0, 6.0)]
    assert gap_span(lab) == (3.2, 4.0)


def test_decode_contiguous_runs_emits_schema12():
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
        for i in range(5)
    ]
    labels = decode_note_runs(
        ["match", "wrong", "wrong", "match", "repetition"],
        notes,
        extra_copies=2,
        pad_notes=0,
        max_run_frac=1.0,
        max_run_notes=16,
    )
    assert [lab["type"] for lab in labels] == ["wrong_note", "repetition"]
    assert labels[0]["pitches"] == [61, 62]
    assert labels[1]["extra_copies"] == 2
    assert labels[1]["score_part"]["start_note_index"] == 4


def test_decode_splits_clip_wide_repetition():
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
        for i in range(20)
    ]
    labels = decode_note_runs(
        ["repetition"] * 20,
        notes,
        extra_copies=1,
        pad_notes=0,
        max_run_frac=0.45,
        max_run_notes=16,
    )
    assert len(labels) >= 2
    assert all(len(lab["pitches"]) <= 5 for lab in labels)
    assert max(lab["score_part"]["end_note_index"] - lab["score_part"]["start_note_index"] for lab in labels) < 16


def test_labels_document_is_schema_12():
    state = PipelineState(
        sample_id="demo",
        sample_dir=".",
        sr=22050,
        duration_sec=2.0,
        hop_sec=0.023,
        config=PipelineConfig(),
        score=ScoreGraph(),
    )
    state.labels.append(
        PipelineLabel(
            id="pipe_000",
            type="repetition",
            start_time=1.0,
            end_time=1.8,
            extra_copies=1,
            pitches=[60, 62],
            note_ids=["note_0000", "note_0001"],
            repeats_label_range=RepeatRange(0.2, 0.9),
        )
    )
    doc = labels_document(state)
    assert doc["schema_version"] == "1.2"
    assert doc["labels"][0]["pitches"] == [60, 62]
    assert doc["labels"][0]["extra_copies"] == 1
