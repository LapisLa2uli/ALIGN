"""V7: match-default CE weights + Dice fault mask + gated no-tile decode."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from alignmodel.bakeoff.common import copies_from_out, emit_schema12, soft_dice
from alignmodel.config import MELODY_ALIGN_TYPES, MELODY_NOTE_CLASSES
from alignmodel.melody import ScoreSoundingNote
from alignmodel.melody_model import class_index

GATE = 0.55

# match CE 1; errors in [1, 1.5]
CLASS_WEIGHTS = {
    "match": 1.0,
    "miss": 1.35,
    "wrong": 1.5,
    "extra": 1.4,
    "rhythm": 1.25,
    "intonation": 1.2,
    "repetition": 1.0,
}


def class_weight_tensor(device: torch.device) -> Tensor:
    w = torch.ones(len(MELODY_NOTE_CLASSES), device=device)
    for name, val in CLASS_WEIGHTS.items():
        w[class_index(name)] = val
    return w


def compute_loss(outputs: dict[str, Tensor], batch: dict, cfg) -> tuple[Tensor, dict[str, float]]:
    type_logits = outputs["type_logits"]
    note_mask = batch["note_mask"]
    y = batch["score_y"]
    match_i = class_index("match")
    weights = class_weight_tensor(type_logits.device)
    ce = nn.functional.cross_entropy(
        type_logits.transpose(1, 2), y, weight=weights, reduction="none"
    )
    ce_loss = (ce * note_mask).sum() / note_mask.sum().clamp_min(1)
    probs = nn.functional.softmax(type_logits, dim=-1)
    p_fault = (1.0 - probs[..., match_i]) * note_mask.float()
    gold_fault = ((y != match_i) & note_mask).float()
    dice_loss = soft_dice(p_fault, gold_fault)
    copies_loss = nn.functional.cross_entropy(outputs["copies_logits"], batch["copies_y"])
    total = ce_loss + dice_loss + cfg.copies_loss_weight * copies_loss
    return total, {
        "loss": float(total.detach()),
        "type": float(ce_loss.detach()),
        "err": float(dice_loss.detach()),
        "copies": float(copies_loss.detach()),
        "coverage": 0.0,
    }


def decode_gated_runs(
    type_logits: Tensor,
    notes: list[ScoreSoundingNote],
    extra_copies: int,
    gate: float = GATE,
    pad_notes: int = 2,
) -> list[dict[str, Any]]:
    """Contiguous same-type non-match runs; emit only if mean softmax >= gate. No tiles."""
    n = min(int(type_logits.size(0)), len(notes))
    if n == 0:
        return []
    probs = torch.softmax(type_logits[:n], dim=-1)
    ids = probs.argmax(-1).tolist()
    if isinstance(ids, int):
        ids = [ids]
    types = [MELODY_NOTE_CLASSES[int(i)] for i in ids]
    labels: list[dict[str, Any]] = []
    i = 0
    while i < n:
        kind = types[i]
        if kind == "match" or kind not in MELODY_ALIGN_TYPES:
            i += 1
            continue
        j = i + 1
        while j < n and types[j] == kind:
            j += 1
        cls = class_index(kind)
        mean_p = float(probs[i:j, cls].mean())
        if mean_p >= gate:
            item = emit_schema12(notes, i, j, kind, extra_copies, pad_notes=pad_notes)
            item["id"] = f"mel_{len(labels):03d}"
            labels.append(item)
        i = j
    return labels


def decode_sample(out: dict[str, Tensor], index: int, n: int, notes: list[ScoreSoundingNote]):
    return decode_gated_runs(
        out["type_logits"][index],
        notes,
        copies_from_out(out, index),
    )
