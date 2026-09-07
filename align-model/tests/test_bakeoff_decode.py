from alignmodel.bakeoff.common import BIO_B, BIO_I, BIO_O, span_targets_from_score_y
from alignmodel.bakeoff.v3 import decode_bio_runs
from alignmodel.bakeoff.v5 import decode_mask_runs
from alignmodel.config import MELODY_NOTE_CLASSES
from alignmodel.melody_model import class_index
from datacreate.melody import ScoreSoundingNote


def _notes(n: int) -> list[ScoreSoundingNote]:
    return [
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
        for i in range(n)
    ]


def test_span_targets_and_bio_decode_no_tiles():
    match = class_index("match")
    wrong = class_index("wrong")
    y = [match] * 20
    y[2:10] = [wrong] * 8
    import numpy as np

    bio, typ, mask = span_targets_from_score_y(np.array(y), 20, match)
    assert int(bio[2]) == BIO_B
    assert all(int(x) == BIO_I for x in bio[3:10])
    assert int(bio[1]) == BIO_O
    assert mask[2:10].sum() == 8
    labels = decode_bio_runs(
        [int(x) for x in bio[:20]],
        [int(x) for x in typ[:20]],
        _notes(20),
        extra_copies=1,
        pad_notes=2,
    )
    assert len(labels) == 1
    assert labels[0]["type"] == "wrong_note"
    assert labels[0]["score_part"]["pad_notes"] == 2


def test_mask_decode_min2_max16_no_tiles():
    notes = _notes(20)
    mask = [False] * 20
    mask[1] = True
    mask[3:19] = [True] * 16
    labels = decode_mask_runs(
        mask,
        [class_index("wrong")] * 20,
        notes,
        extra_copies=1,
        scores=[0.9] * 20,
        pad_notes=0,
    )
    assert all(2 <= (lab["score_part"]["end_note_index"] - lab["score_part"]["start_note_index"] + 1) <= 16 for lab in labels)
    assert sum(1 for lab in labels if lab["type"] == "wrong_note") >= 1
    singles = decode_mask_runs(
        [True] + [False] * 4,
        [class_index("wrong")] * 5,
        _notes(5),
        extra_copies=1,
        pad_notes=0,
    )
    assert singles == []


def test_v7_gate_skips_low_confidence():
    import torch

    from alignmodel.bakeoff.v7 import decode_gated_runs

    n = 6
    logits = torch.zeros(n, len(MELODY_NOTE_CLASSES))
    logits[:, class_index("match")] = 2.0
    logits[1:4, class_index("wrong")] = 0.1
    labels = decode_gated_runs(logits, _notes(n), extra_copies=1, gate=0.55, pad_notes=0)
    assert labels == []
    logits[1:4, class_index("wrong")] = 4.0
    labels = decode_gated_runs(logits, _notes(n), extra_copies=1, gate=0.55, pad_notes=0)
    assert len(labels) == 1
    assert labels[0]["type"] == "wrong_note"
    assert labels[0]["score_part"]["end_note_index"] - labels[0]["score_part"]["start_note_index"] == 2
