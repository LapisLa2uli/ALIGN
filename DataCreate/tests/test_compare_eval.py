from datacreate.web.compare_eval import pair_labels


def test_pair_labels_type_mismatch_is_half():
    gold = [
        {
            "id": "g1",
            "type": "wrong_note",
            "start_time": 1.0,
            "end_time": 2.0,
            "pitches": [60, 62, 64],
            "score_part": {"start_note_index": 4, "end_note_index": 6},
        }
    ]
    pred = [
        {
            "id": "p1",
            "type": "rhythm_error",
            "start_time": 1.0,
            "end_time": 2.0,
            "pitches": [60, 62, 64],
            "score_part": {"start_note_index": 4, "end_note_index": 6},
        }
    ]
    g_out, p_out = pair_labels(gold, pred)
    assert g_out[0]["compare"]["match"] == "type_mismatch"
    assert p_out[0]["compare"]["match"] == "type_mismatch"
    assert p_out[0]["compare"]["credit"] == 0.5
    assert p_out[0]["compare"]["pair_id"] == "g1"


def test_pair_labels_same_type_is_full():
    gold = [{
        "id": "g1",
        "type": "rhythm_error",
        "pitches": [60, 62],
        "score_part": {"start_note_index": 2, "end_note_index": 3},
    }]
    pred = [{
        "id": "p1",
        "type": "rhythm_error",
        "pitches": [60, 62],
        "score_part": {"start_note_index": 2, "end_note_index": 3},
    }]
    g_out, p_out = pair_labels(gold, pred)
    assert g_out[0]["compare"]["match"] == "full"
    assert p_out[0]["compare"]["credit"] == 1.0
