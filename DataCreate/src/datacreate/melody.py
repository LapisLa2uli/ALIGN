from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from music21 import converter, note, stream, tempo


@dataclass(frozen=True)
class ScoreSoundingNote:
    index: int
    pitch: int
    start: float
    end: float
    ql_start: float
    ql_end: float
    measure: int | None
    note_id: str


@dataclass
class MelodySpan:
    start_note_index: int
    end_note_index: int
    pad_notes: int
    pitches: list[int]
    note_ids: list[str]
    start_measure: int | None = None
    end_measure: int | None = None

    def as_fields(self) -> dict[str, Any]:
        return {
            "score_part": {
                "start_measure": self.start_measure,
                "start_note_index": self.start_note_index,
                "end_measure": self.end_measure,
                "end_note_index": self.end_note_index,
                "pad_notes": self.pad_notes,
            },
            "pitches": list(self.pitches),
            "note_ids": list(self.note_ids),
        }


def score_bpm(score) -> float:
    for mark in score.flatten().getElementsByClass(tempo.MetronomeMark):
        if mark.number:
            return float(mark.number)
    return 120.0


def parse_sounding_notes(score_or_path) -> list[ScoreSoundingNote]:
    if isinstance(score_or_path, (str, Path)):
        parsed = converter.parse(str(score_or_path))
    else:
        parsed = score_or_path
    bpm = score_bpm(parsed)
    sec_per_ql = 60.0 / bpm
    raw: list[tuple[float, int, float, float, int | None]] = []
    for n in parsed.recurse().getElementsByClass(note.Note):
        if n.duration.isGrace:
            continue
        try:
            start_ql = float(n.getOffsetInHierarchy(parsed))
        except Exception:
            start_ql = float(n.offset)
        dur_ql = float(n.duration.quarterLength or 0.0)
        start = start_ql * sec_per_ql
        end = (start_ql + dur_ql) * sec_per_ql
        if end <= start:
            end = start + 0.05
        measure = n.getContextByClass(stream.Measure)
        measure_num = int(measure.number) if measure and measure.number is not None else None
        raw.append((start_ql, int(n.pitch.midi), start, end, measure_num))
    raw.sort(key=lambda item: (item[0], item[1]))
    notes: list[ScoreSoundingNote] = []
    for i, (ql0, pitch, start, end, measure) in enumerate(raw):
        notes.append(
            ScoreSoundingNote(
                index=i,
                pitch=pitch,
                start=start,
                end=end,
                ql_start=ql0,
                ql_end=ql0 + max(end - start, 0.05) / sec_per_ql,
                measure=measure,
                note_id=f"note_{i:04d}",
            )
        )
    return notes


def extra_neighbor_core(
    notes: list[ScoreSoundingNote], anchor_index: int
) -> tuple[int, int]:
    """Clean-score span around an extra: the note before it and the note after it."""
    if not notes:
        return 0, 0
    n = len(notes)
    i0 = max(0, min(int(anchor_index), n - 1))
    return i0, min(n, i0 + 2)


def padded_melody(
    notes: list[ScoreSoundingNote],
    core_i0: int,
    core_i1: int,
    pad_notes: int,
) -> MelodySpan:
    """Expand [core_i0, core_i1) by pad_notes on each side. Clamp, no wrap."""
    if not notes:
        raise ValueError("No sounding notes")
    n = len(notes)
    i0 = max(0, min(int(core_i0), n - 1))
    i1 = max(i0 + 1, min(int(core_i1), n))
    pad = max(0, int(pad_notes))
    lo = max(0, i0 - pad)
    hi = min(n, i1 + pad)
    span_notes = notes[lo:hi]
    return MelodySpan(
        start_note_index=lo,
        end_note_index=hi - 1,
        pad_notes=pad,
        pitches=[item.pitch for item in span_notes],
        note_ids=[item.note_id for item in span_notes],
        start_measure=span_notes[0].measure,
        end_measure=span_notes[-1].measure,
    )


def notes_in_measures(
    notes: list[ScoreSoundingNote], measures: Iterable[int]
) -> tuple[int, int] | None:
    wanted = {int(m) for m in measures}
    idxs = [n.index for n in notes if n.measure is not None and n.measure in wanted]
    if not idxs:
        return None
    return min(idxs), max(idxs) + 1


def notes_overlapping_time(
    notes: list[ScoreSoundingNote], t0: float, t1: float
) -> tuple[int, int] | None:
    idxs = [n.index for n in notes if n.start < t1 and t0 < n.end]
    if not idxs:
        nearest = min(notes, key=lambda n: min(abs(n.start - t0), abs(n.end - t0)))
        return nearest.index, nearest.index + 1
    return min(idxs), max(idxs) + 1


def notes_for_measure_pitch(
    notes: list[ScoreSoundingNote], measure: int | None, pitch: int | None
) -> tuple[int, int] | None:
    cands = notes
    if measure is not None:
        cands = [n for n in cands if n.measure == measure]
    if pitch is not None:
        pitched = [n for n in cands if n.pitch == pitch]
        if pitched:
            cands = pitched
    if not cands:
        return None
    return cands[0].index, cands[-1].index + 1


_MIDI_IN_COMMENT = re.compile(r"MIDI\s+(\d+)")
_SHIFT_ORIG = re.compile(r"\((\d+)\s*->\s*(\d+)\)")
_MIDI_LIST = re.compile(r"MIDI\s+([\d,\s]+)")


def midi_from_comment(comment: str | None) -> list[int]:
    if not comment:
        return []
    listed = _MIDI_LIST.search(comment)
    if listed and "," in listed.group(1):
        return [int(x) for x in re.findall(r"\d+", listed.group(1))]
    shift = _SHIFT_ORIG.search(comment)
    if shift:
        return [int(shift.group(1))]
    single = _MIDI_IN_COMMENT.search(comment)
    if single:
        return [int(single.group(1))]
    return []


def is_repeated_pass(label: dict[str, Any]) -> bool:
    comment = str(label.get("comment") or "")
    if "repeated pass" in comment:
        return True
    if "first pass" in comment:
        return False
    return bool(re.search(r"\(pass \d+\)", comment))


def label_already_converted(label: dict[str, Any]) -> bool:
    part = label.get("score_part")
    pitches = label.get("pitches")
    return isinstance(part, dict) and isinstance(pitches, list) and len(pitches) > 0


def lcs_length(a: list[int], b: list[int]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            if x == y:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def melody_similarity(a: list[int], b: list[int]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return (2.0 * lcs_length(a, b)) / float(len(a) + len(b))


def note_set_iou(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


@dataclass
class WeakMelody:
    pitches: list[int]
    type: str | None = None
    extra_copies: int | None = None
    note_ids: list[str] = field(default_factory=list)


MATCH_SIMILARITY_THRESHOLD = 0.80
MATCH_LENGTH_RATIO = 0.60


def is_contiguous_part(inner: list[int], outer: list[int]) -> bool:
    """True if `inner` is empty-false and appears as a consecutive slice of `outer`.

    Kept as a helper. Official melody eval no longer uses slice/containment matching.
    """
    if not inner or not outer:
        return False
    n, m = len(inner), len(outer)
    if n > m:
        return False
    for i in range(m - n + 1):
        if outer[i : i + n] == inner:
            return True
    return False


def melodies_containment_match(a: list[int], b: list[int]) -> bool:
    """Retired default. Slice/containment match; not used by ``match_melodies``."""
    if not a and not b:
        return True
    return is_contiguous_part(a, b) or is_contiguous_part(b, a)


def melody_pair_score(a: list[int], b: list[int]) -> float:
    """Same-event similarity in ``[0, 1]``.

    LCS Dice (``melody_similarity``) if the lists have similar length, else 0.
    Equal lists score 1. A short slice of a long list, or a whole-score dump
    that contains a gold melody, scores 0 because ``min/max`` length <
    ``MATCH_LENGTH_RATIO``. Type is applied later by ``melody_label_score``.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if min(la, lb) / max(la, lb) < MATCH_LENGTH_RATIO:
        return 0.0
    return melody_similarity(a, b)


def melodies_set_match(a: list[int], b: list[int]) -> bool:
    return melody_pair_score(a, b) >= MATCH_SIMILARITY_THRESHOLD


def _exclusive_pairs(scores: list[list[float]]) -> list[tuple[int, int, float]]:
    """1-1 assignment maximizing score. Hungarian, greedy fallback."""
    n_pred = len(scores)
    n_gold = len(scores[0]) if scores else 0
    if n_pred == 0 or n_gold == 0:
        return []
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        mat = np.asarray(scores, dtype=np.float64)
        rows, cols = linear_sum_assignment(-mat)
        return [(int(i), int(j), float(mat[i, j])) for i, j in zip(rows, cols)]
    except Exception:
        cells = [
            (scores[i][j], i, j)
            for i in range(n_pred)
            for j in range(n_gold)
        ]
        cells.sort(reverse=True)
        used_p: set[int] = set()
        used_g: set[int] = set()
        out: list[tuple[int, int, float]] = []
        for score, i, j in cells:
            if i in used_p or j in used_g:
                continue
            used_p.add(i)
            used_g.add(j)
            out.append((i, j, score))
        return out


TYPE_MISMATCH_SCALE = 0.5


def _types_agree(pred: WeakMelody, gold: WeakMelody) -> bool:
    pt, gt = pred.type, gold.type
    if pt is None and gt is None:
        return True
    if pt is None or gt is None:
        return False
    return str(pt) == str(gt)


def melody_label_score(
    pred: WeakMelody,
    gold: WeakMelody,
    *,
    soft: bool = False,
    ignore_type: bool = False,
) -> float:
    """Pitch-list similarity, then type: mismatch halves a positive score.

    Assignment still uses pitch lists only. After a pair is a range hit,
    matching types keep full credit; a wrong type scales it by
    ``TYPE_MISMATCH_SCALE`` (0.5). Hard mode: exact/near range is 1 or 0.5.
    Soft mode: raw similarity, or half of that on a type mismatch.
    ``ignore_type=True`` skips the type check (old type-insensitive scoring).
    """
    pitch = melody_pair_score(pred.pitches, gold.pitches)
    if pitch <= 0.0:
        return 0.0
    if not soft and pitch < MATCH_SIMILARITY_THRESHOLD:
        return 0.0
    credit = pitch if soft else 1.0
    if ignore_type or _types_agree(pred, gold):
        return credit
    return TYPE_MISMATCH_SCALE * credit


def match_melodies(
    gold: list[WeakMelody], pred: list[WeakMelody]
) -> tuple[float, float]:
    """Exclusive set-F1 and precision.

    Each prediction matches at most one gold and vice versa (Hungarian 1-1
    on pitch-list similarity). A pair is a range hit only when the pitch
    lists are the same event: equal, or LCS-Dice ≥ ``MATCH_SIMILARITY_THRESHOLD``
    with length ratio ≥ ``MATCH_LENGTH_RATIO``. Slice/containment does not
    match. A range hit with the same type scores 1; a range hit with a
    different type scores ``TYPE_MISMATCH_SCALE`` (0.5).

    Returns ``(F1, precision)``. Use ``match_melodies_detail`` for recall.
    """
    detail = match_melodies_detail(gold, pred)
    return detail["f1"], detail["precision"]


def match_melodies_detail(
    gold: list[WeakMelody],
    pred: list[WeakMelody],
    *,
    soft: bool = False,
    ignore_type: bool = False,
) -> dict[str, float]:
    """Exclusive set scores with type-aware credit.

    Hungarian is 1-1 on pitch-list similarity. Each assigned pair then
    contributes ``melody_label_score`` (hard: 1 / 0.5 / 0; soft: similarity
    or half of it when types differ). ``ignore_type=True`` gives full range
    credit even when types differ.
    """
    empty = {
        "f1": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "n_pred_correct": 0,
        "n_gold_covered": 0,
        "n_matched": 0,
        "similarity_sum": 0.0,
    }
    if not gold and not pred:
        return empty
    zero = {
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "n_pred_correct": 0,
        "n_gold_covered": 0,
        "n_matched": 0,
        "similarity_sum": 0.0,
    }
    if not gold or not pred:
        return zero
    scores = [
        [melody_pair_score(p.pitches, g.pitches) for g in gold] for p in pred
    ]
    pairs = _exclusive_pairs(scores)
    sim_sum = float(
        sum(
            melody_label_score(pred[i], gold[j], soft=soft, ignore_type=ignore_type)
            for i, j, _ in pairs
        )
    )
    precision = sim_sum / len(pred)
    recall = sim_sum / len(gold)
    n_matched = sim_sum
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "n_pred_correct": n_matched,
        "n_gold_covered": n_matched,
        "n_matched": n_matched,
        "similarity_sum": sim_sum,
    }
