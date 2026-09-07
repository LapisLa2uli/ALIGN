import torch

from alignmodel.bakeoff.even_common import decode_fault_runs, decode_runs_no_tile
from alignmodel.bakeoff.v6 import LinearChainCRF, MAX_ERROR_RUN
from alignmodel.config import MELODY_NOTE_CLASSES
from alignmodel.melody_model import class_index
from datacreate.melody import ScoreSoundingNote


def _notes(n: int):
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


def test_v2_gate_and_no_tile():
    notes = _notes(20)
    types = ["repetition"] * 20
    probs = torch.zeros(20, len(MELODY_NOTE_CLASSES))
    probs[:, class_index("repetition")] = 0.4
    labels = decode_runs_no_tile(types, notes, 1, type_probs=probs, conf_min=0.55)
    assert labels == []
    probs[:, class_index("repetition")] = 0.8
    labels = decode_runs_no_tile(types, notes, 1, type_probs=probs, conf_min=0.55)
    assert len(labels) == 1
    assert labels[0]["type"] == "repetition"
    assert labels[0]["score_part"]["end_note_index"] - labels[0]["score_part"]["start_note_index"] >= 19


def test_v4_fault_runs():
    notes = _notes(8)
    fault = torch.tensor([0.1, 0.8, 0.9, 0.7, 0.2, 0.8, 0.8, 0.1])
    logits = torch.zeros(8, len(MELODY_NOTE_CLASSES))
    logits[:, class_index("wrong")] = 2.0
    types, labels = decode_fault_runs(fault, logits, notes, extra_copies=1, threshold=0.5)
    assert types[0] == "match"
    assert types[1] == "wrong"
    assert types[4] == "match"
    assert [lab["type"] for lab in labels] == ["wrong_note", "wrong_note"]


def test_crf_nll_and_max_run():
    crf = LinearChainCRF(len(MELODY_NOTE_CLASSES))
    emissions = torch.randn(2, 12, len(MELODY_NOTE_CLASSES), requires_grad=True)
    tags = torch.zeros(2, 12, dtype=torch.long)
    tags[0, 2:6] = class_index("wrong")
    mask = torch.ones(2, 12, dtype=torch.bool)
    loss = crf.nll(emissions, tags, mask)
    loss.backward()
    assert emissions.grad is not None
    long = torch.zeros(1, 24, len(MELODY_NOTE_CLASSES))
    long[0, :, class_index("repetition")] = 5.0
    ids = crf.decode(long, torch.ones(1, 24, dtype=torch.bool), MAX_ERROR_RUN)[0]
    run = 0
    best = 0
    for i in ids:
        if i != class_index("match"):
            run += 1
            best = max(best, run)
        else:
            run = 0
    assert best <= MAX_ERROR_RUN
