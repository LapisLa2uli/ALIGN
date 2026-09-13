"""Sounding-note cleanup: drop decorations and fold tied/extended notes."""

from __future__ import annotations

from typing import Any, Iterable

from music21 import chord, note, spanner, stream

_TIE_START = frozenset({"start", "continue"})
_TIE_STOP = frozenset({"continue", "stop"})
_DECORATIVE_NOTE_SIZES = frozenset({"cue", "grace"})
# MusicXML offsets after barlines are often a few 1e-3 QL off.
_TIE_GAP_QL = 0.05


def is_decorative_element(el) -> bool:
    """True for grace, cue, hidden, or zero-length ornamental notes."""
    if isinstance(el, note.Rest):
        return False
    duration = getattr(el, "duration", None)
    if duration is not None:
        if bool(getattr(duration, "isGrace", False)):
            return True
        try:
            if float(duration.quarterLength or 0.0) <= 0.0:
                return True
        except (TypeError, ValueError):
            return True
    style = getattr(el, "style", None)
    if style is not None:
        if bool(getattr(style, "hideObjectOnPrint", False)):
            return True
        note_size = getattr(style, "noteSize", None)
        if note_size in _DECORATIVE_NOTE_SIZES:
            return True
    return False


def element_tie_type(el) -> str | None:
    tie_obj = getattr(el, "tie", None)
    if tie_obj is None:
        return None
    kind = getattr(tie_obj, "type", None)
    return str(kind) if kind else None


def slur_adjacent_ids(score) -> set[tuple[int, int]]:
    """Ordered (first, second) id pairs of notes that a slur joins consecutively."""
    pairs: set[tuple[int, int]] = set()
    for slur in score.recurse().getElementsByClass(spanner.Slur):
        try:
            spanned = [
                item
                for item in slur.getSpannedElements()
                if isinstance(item, (note.Note, chord.Chord, note.Unpitched))
            ]
        except Exception:  # noqa: BLE001
            continue
        for left, right in zip(spanned, spanned[1:]):
            pairs.add((id(left), id(right)))
    return pairs


def voice_key(el, part_idx: int) -> tuple[int, Any]:
    voice = el.getContextByClass(stream.Voice)
    if voice is None:
        return (part_idx, None)
    return (part_idx, getattr(voice, "id", None) or id(voice))


def _abuts(prev_end_ql: float, next_start_ql: float) -> bool:
    gap = float(next_start_ql) - float(prev_end_ql)
    return -_TIE_GAP_QL <= gap <= _TIE_GAP_QL


def _can_fold_tied(prev: dict[str, Any], nxt: dict[str, Any], slur_pairs: set[tuple[int, int]]) -> bool:
    if prev.get("is_rest") or nxt.get("is_rest"):
        return False
    if prev.get("voice") != nxt.get("voice"):
        return False
    if prev.get("midi") is None or prev.get("midi") != nxt.get("midi"):
        return False
    prev_end = float(prev["offset_ql"]) + float(prev["duration_ql"])
    if not _abuts(prev_end, float(nxt["offset_ql"])):
        return False
    prev_tie = prev.get("tie_type")
    next_tie = nxt.get("tie_type")
    if prev_tie in _TIE_START and next_tie in _TIE_STOP:
        return True
    if prev_tie in _TIE_START and next_tie is None:
        return True
    if prev_tie is None and next_tie in _TIE_STOP:
        return True
    left_id = prev.get("el_id")
    right_id = nxt.get("el_id")
    if left_id is not None and right_id is not None and (left_id, right_id) in slur_pairs:
        return True
    return False


def collapse_tied_records(
    records: Iterable[dict[str, Any]],
    slur_pairs: set[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    """Fold abutting same-pitch ties (and slur-as-tie OMR) into one sounding note."""
    pairs = slur_pairs or set()
    ordered = sorted(
        records,
        key=lambda row: (
            row.get("voice") or (0, None),
            float(row["offset_ql"]),
            float(row["duration_ql"]),
        ),
    )
    merged: list[dict[str, Any]] = []
    for row in ordered:
        if merged and _can_fold_tied(merged[-1], row, pairs):
            prev = merged[-1]
            end_ql = float(row["offset_ql"]) + float(row["duration_ql"])
            prev["duration_ql"] = end_ql - float(prev["offset_ql"])
            if row.get("ref_end") is not None:
                prev["ref_end"] = row["ref_end"]
            if row.get("end") is not None:
                prev["end"] = row["end"]
            prev["tie_type"] = row.get("tie_type") or prev.get("tie_type")
            sources = list(prev.get("source_el_ids") or [prev.get("el_id")])
            sources.append(row.get("el_id"))
            prev["source_el_ids"] = [item for item in sources if item is not None]
            continue
        merged.append(dict(row))
    return merged
