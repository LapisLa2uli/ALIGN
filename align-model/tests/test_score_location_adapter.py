import inspect

from alignmodel.bakeoff import runner
from alignmodel.bakeoff.even_common import official_set_metrics
from alignmodel.joint.score_location_adapter import evaluate_class_notes_note_wise
from alignmodel.melody import official_label_metrics
from alignmodel.melody_train import train_melody


def test_adapter_exact_location_and_type_is_one():
    gold = {
        "correct": [(0.0, 0.4, 60)],
        "missing": [(0.5, 0.9, 62)],
        "extra": [(1.2, 1.5, 64)],
    }
    result = evaluate_class_notes_note_wise(gold, gold)
    assert result["status"] == "available"
    assert result["f1"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


def test_adapter_wrong_type_at_same_location_is_half():
    gold = {
        "correct": [(0.0, 0.4, 60)],
        "missing": [],
        "extra": [],
    }
    pred = {
        "correct": [],
        "missing": [],
        "extra": [(0.0, 0.4, 60)],
    }
    result = evaluate_class_notes_note_wise(gold, pred)
    assert result["f1"] == 0.5
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5


def test_adapter_wrong_location_is_zero():
    gold = {
        "correct": [(0.0, 0.4, 60)],
        "missing": [],
        "extra": [],
    }
    pred = {
        "correct": [(1.0, 1.4, 72)],
        "missing": [],
        "extra": [],
    }
    result = evaluate_class_notes_note_wise(gold, pred)
    assert result["f1"] == 0.0


def test_adapter_matching_is_exclusive():
    gold = {
        "correct": [(0.0, 0.4, 60)],
        "missing": [],
        "extra": [],
    }
    pred = {
        "correct": [(0.0, 0.4, 60), (0.02, 0.42, 60)],
        "missing": [],
        "extra": [],
    }
    result = evaluate_class_notes_note_wise(gold, pred)
    assert result["predicted"] == 2
    assert result["gold"] == 1
    assert result["credit"] == 1.0
    assert result["precision"] == 0.5
    assert result["recall"] == 1.0


def test_adapter_empty_empty_is_one():
    empty = {"correct": [], "missing": [], "extra": []}
    result = evaluate_class_notes_note_wise(empty, empty)
    assert result["f1"] == 1.0


def test_official_set_metrics_uses_score_identity_not_pitch_lists():
    gold = [
        {
            "type": "wrong_note",
            "score_part": {"start_note_index": 0, "end_note_index": 2},
            "note_ids": ["note_0", "note_1", "note_2"],
            "pitches": [60, 62, 64],
        }
    ]
    same_pitches_elsewhere = [
        {
            "type": "wrong_note",
            "score_part": {"start_note_index": 8, "end_note_index": 10},
            "note_ids": ["note_8", "note_9", "note_10"],
            "pitches": [60, 62, 64],
        }
    ]
    wrong_type = [
        {
            "type": "missed_note",
            "score_part": {"start_note_index": 0, "end_note_index": 2},
            "note_ids": ["note_0", "note_1", "note_2"],
            "pitches": [60, 62, 64],
        }
    ]
    moved = official_set_metrics(gold, same_pitches_elsewhere, score_event_count=16)
    typed = official_set_metrics(gold, wrong_type, score_event_count=16)
    exact = official_set_metrics(gold, gold, score_event_count=16)
    assert moved["note_wise_f1"] == 0.0
    assert typed["note_wise_f1"] == 0.5
    assert exact["note_wise_f1"] == 1.0
    assert moved["legacy_pitch_similarity_f1"] == 1.0


def test_official_label_metrics_empty_is_one():
    metrics = official_label_metrics([], [])
    assert metrics["f1"] == 1.0
    assert metrics["status"] == "available"


def test_bakeoff_and_melody_train_select_on_note_wise_f1():
    bakeoff = inspect.getsource(runner.train_variant)
    melody = inspect.getsource(train_melody)
    for source in (bakeoff, melody):
        assert 'val_metrics["note_wise_f1"] >= best_f1' in source
        assert 'val_metrics["note_wise_f1"] >= patience_best' in source
        assert 'val_metrics["set_f1"] >= best_f1' not in source
