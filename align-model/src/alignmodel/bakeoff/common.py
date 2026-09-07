from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from alignmodel.config import MELODY_ALIGN_TYPES, MELODY_NOTE_CLASSES
from alignmodel.melody import ScoreSoundingNote
from datacreate.melody import padded_melody

BIO_O = 0
BIO_B = 1
BIO_I = 2
FAULT_CLASSES = ("miss", "wrong", "extra", "rhythm", "intonation", "repetition")


def fault_index(name: str) -> int:
    return FAULT_CLASSES.index(name)


def span_targets_from_score_y(score_y, n: int, match_i: int) -> tuple:
    """Derive BIO / fault-type / binary mask from priority-painted score_y."""
    import numpy as np

    bio_y = np.zeros(score_y.shape, dtype=np.int64)
    type_span_y = np.zeros(score_y.shape, dtype=np.int64)
    mask_y = np.zeros(score_y.shape, dtype=np.float32)
    i = 0
    while i < n:
        cls = int(score_y[i])
        if cls == match_i:
            i += 1
            continue
        j = i + 1
        while j < n and int(score_y[j]) == cls:
            j += 1
        bio_y[i] = BIO_B
        if j - i > 1:
            bio_y[i + 1 : j] = BIO_I
        name = MELODY_NOTE_CLASSES[cls]
        if name in FAULT_CLASSES:
            type_span_y[i:j] = fault_index(name)
        mask_y[i:j] = 1.0
        i = j
    return bio_y, type_span_y, mask_y


def emit_schema12(
    notes: list[ScoreSoundingNote],
    lo: int,
    hi: int,
    kind: str,
    extra_copies: int,
    pad_notes: int = 2,
) -> dict[str, Any]:
    """Core [lo, hi) → schema 1.2 label. ``kind`` is a melody class name."""
    span = padded_melody(notes, lo, hi, pad_notes)
    span_notes = notes[span.start_note_index : span.end_note_index + 1]
    item: dict[str, Any] = {
        "id": "mel_000",
        "source": "melody",
        "type": MELODY_ALIGN_TYPES[kind],
        "start_time": round(span_notes[0].start, 4),
        "end_time": round(span_notes[-1].end, 4),
        "score_part": {
            "start_note_index": span.start_note_index,
            "end_note_index": span.end_note_index,
            "pad_notes": span.pad_notes,
            "start_measure": span.start_measure,
            "end_measure": span.end_measure,
        },
        "pitches": list(span.pitches),
        "note_ids": list(span.note_ids),
    }
    if kind == "repetition":
        item["extra_copies"] = 1 if extra_copies <= 0 else min(2, extra_copies)
        item["repeats_label_range"] = {
            "start_time": item["start_time"],
            "end_time": item["end_time"],
        }
    return item


def soft_dice(pred: Tensor, target: Tensor, mask: Tensor | None = None, eps: float = 1e-6) -> Tensor:
    if mask is not None:
        pred = pred * mask
        target = target * mask
    num = 2.0 * (pred * target).sum()
    den = pred.sum() + target.sum()
    return 1.0 - (num + eps) / (den + eps)


def copies_from_out(out: dict[str, Tensor], index: int) -> int:
    return int(out["copies_logits"][index].argmax(-1).detach().cpu())
