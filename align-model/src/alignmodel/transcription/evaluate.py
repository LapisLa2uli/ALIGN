from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from .decode import TransNote


def _as_note(value) -> TransNote:
    if isinstance(value, TransNote):
        return value
    if isinstance(value, dict):
        return TransNote(
            pitch=int(value["pitch"]),
            start=float(value["start"]),
            end=float(value["end"]),
            confidence=float(value.get("confidence", 1.0)),
            cents=float(value.get("cents", 0.0)),
            pitch_candidates=tuple(value.get("pitch_candidates") or ()),
        )
    pitch, start, end = value[:3]
    cents = float(value[3]) if len(value) > 3 else 0.0
    return TransNote(int(pitch), float(start), float(end), 1.0, cents=cents)


def match_notes(
    predicted: Iterable[TransNote | dict | tuple],
    target: Iterable[TransNote | dict | tuple],
    *,
    onset_tolerance_sec: float = 0.050,
) -> list[tuple[int, int]]:
    """Maximum monotonic matching with exact written semitone and <=50 ms onset."""

    pred = [_as_note(n) for n in predicted]
    gold = [_as_note(n) for n in target]
    pred_by_pitch: dict[int, list[int]] = defaultdict(list)
    gold_by_pitch: dict[int, list[int]] = defaultdict(list)
    for i, note in enumerate(pred):
        pred_by_pitch[note.pitch].append(i)
    for i, note in enumerate(gold):
        gold_by_pitch[note.pitch].append(i)

    pairs: list[tuple[int, int]] = []
    for pitch in sorted(set(pred_by_pitch) & set(gold_by_pitch)):
        pp = sorted(pred_by_pitch[pitch], key=lambda i: pred[i].start)
        gg = sorted(gold_by_pitch[pitch], key=lambda i: gold[i].start)
        i = j = 0
        tolerance = onset_tolerance_sec + 1e-9
        while i < len(pp) and j < len(gg):
            p_start = pred[pp[i]].start
            g_start = gold[gg[j]].start
            if p_start < g_start - tolerance:
                i += 1
            elif g_start < p_start - tolerance:
                j += 1
            else:
                pairs.append((pp[i], gg[j]))
                i += 1
                j += 1
    return sorted(pairs)


def match_notes_by_onset(
    predicted: Iterable[TransNote | dict | tuple],
    target: Iterable[TransNote | dict | tuple],
    *,
    onset_tolerance_sec: float = 0.050,
) -> list[tuple[int, int]]:
    """Greedy one-to-one pairing by onset only, used for pitch-error analysis."""

    pred = [_as_note(n) for n in predicted]
    gold = [_as_note(n) for n in target]
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    tolerance = onset_tolerance_sec + 1e-9
    for i, note in sorted(enumerate(pred), key=lambda item: item[1].start):
        best_j = None
        best_dt = None
        for j, other in enumerate(gold):
            if j in used:
                continue
            delta = abs(note.start - other.start)
            if delta <= tolerance and (best_dt is None or delta < best_dt):
                best_j = j
                best_dt = delta
        if best_j is not None:
            used.add(best_j)
            pairs.append((i, best_j))
    return pairs


def evaluate_note_lists(
    predicted: Iterable[TransNote | dict | tuple],
    target: Iterable[TransNote | dict | tuple],
    *,
    onset_tolerance_sec: float = 0.050,
) -> dict[str, float | int | None]:
    pred = [_as_note(n) for n in predicted]
    gold = [_as_note(n) for n in target]
    pairs = match_notes(pred, gold, onset_tolerance_sec=onset_tolerance_sec)
    matched = len(pairs)
    precision = matched / max(len(pred), 1)
    recall = matched / max(len(gold), 1)
    if not pred and not gold:
        precision = recall = 1.0
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    onset_errors = [abs(pred[i].start - gold[j].start) for i, j in pairs]
    offset_errors = [abs(pred[i].end - gold[j].end) for i, j in pairs]
    cents_errors = [abs(pred[i].cents - gold[j].cents) for i, j in pairs]
    onset_pairs = match_notes_by_onset(
        pred, gold, onset_tolerance_sec=onset_tolerance_sec
    )
    pitch_deltas = [pred[i].pitch - gold[j].pitch for i, j in onset_pairs]
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "onset_mae_sec": float(np.mean(onset_errors)) if onset_errors else None,
        "offset_mae_sec": float(np.mean(offset_errors)) if offset_errors else None,
        "cents_mae": float(np.mean(cents_errors)) if cents_errors else None,
        "n_pred": len(pred),
        "n_target": len(gold),
        "n_matched": matched,
        "n_onset_aligned": len(onset_pairs),
        "n_semitone_errors": sum(abs(delta) == 1 for delta in pitch_deltas),
        "n_octave_errors": sum(abs(delta) == 12 for delta in pitch_deltas),
        "n_plus_minus_2": sum(abs(delta) == 2 for delta in pitch_deltas),
        "pred_target_ratio": len(pred) / max(len(gold), 1),
    }
