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


def is_contiguous_part(inner: list[int], outer: list[int]) -> bool:
    """True if `inner` is empty-false and appears as a consecutive slice of `outer`."""
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
    """Pred is correct if it is part/whole of gold, or gold is part of pred."""
    if not a and not b:
        return True
    return is_contiguous_part(a, b) or is_contiguous_part(b, a)


def match_melodies(
    gold: list[WeakMelody], pred: list[WeakMelody]
) -> tuple[float, float]:
    """Containment F1 and precision.

    A predicted melody is correct if its pitch list is a contiguous part of
    a gold melody, equals one, or contains a gold melody as a contiguous part.
    One gold can validate several preds and the reverse. Returns
    (F1, precision) so existing callers keep a two-tuple; recall is
    recoverable as ``2*F1*prec / max(F1+prec, eps)`` but callers that need
    it should use ``match_melodies_detail``.
    """
    detail = match_melodies_detail(gold, pred)
    return detail["f1"], detail["precision"]


def match_melodies_detail(
    gold: list[WeakMelody], pred: list[WeakMelody]
) -> dict[str, float]:
    if not gold and not pred:
        return {"f1": 1.0, "precision": 1.0, "recall": 1.0, "n_pred_correct": 0, "n_gold_covered": 0}
    if not gold:
        return {
            "f1": 0.0,
            "precision": 0.0,
            "recall": 1.0 if not pred else 0.0,
            "n_pred_correct": 0,
            "n_gold_covered": 0,
        }
    if not pred:
        return {
            "f1": 0.0,
            "precision": 1.0 if not gold else 0.0,
            "recall": 0.0,
            "n_pred_correct": 0,
            "n_gold_covered": 0,
        }
    pred_ok = [
        any(melodies_containment_match(p.pitches, g.pitches) for g in gold) for p in pred
    ]
    gold_ok = [
        any(melodies_containment_match(p.pitches, g.pitches) for p in pred) for g in gold
    ]
    precision = sum(pred_ok) / len(pred_ok)
    recall = sum(gold_ok) / len(gold_ok)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "n_pred_correct": int(sum(pred_ok)),
        "n_gold_covered": int(sum(gold_ok)),
    }
