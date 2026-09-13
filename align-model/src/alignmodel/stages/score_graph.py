from __future__ import annotations

from pathlib import Path

from music21 import bar, converter, note, spanner, stream, tempo

from alignmodel.types import GraphNote, LegalEdge, ScoreGraph

_TIE_START = frozenset({"start", "continue"})
_TIE_STOP = frozenset({"continue", "stop"})
_DECORATIVE_NOTE_SIZES = frozenset({"cue", "grace"})
_TIE_GAP_QL = 0.05


def _bpm(score) -> float:
    for mark in score.flatten().getElementsByClass(tempo.MetronomeMark):
        if mark.number:
            return float(mark.number)
    return 120.0


def _is_decorative(el) -> bool:
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
        if getattr(style, "noteSize", None) in _DECORATIVE_NOTE_SIZES:
            return True
    return False


def _tie_type(el) -> str | None:
    tie_obj = getattr(el, "tie", None)
    if tie_obj is None:
        return None
    kind = getattr(tie_obj, "type", None)
    return str(kind) if kind else None


def _slur_pairs(score) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for slur in score.recurse().getElementsByClass(spanner.Slur):
        try:
            spanned = [
                item
                for item in slur.getSpannedElements()
                if isinstance(item, note.Note)
            ]
        except Exception:  # noqa: BLE001
            continue
        for left, right in zip(spanned, spanned[1:]):
            pairs.add((id(left), id(right)))
    return pairs


def _abuts(prev_end_ql: float, next_start_ql: float) -> bool:
    return -_TIE_GAP_QL <= (next_start_ql - prev_end_ql) <= _TIE_GAP_QL


def build_score_graph(score_path: Path) -> ScoreGraph:
    parsed = converter.parse(str(score_path))
    bpm = _bpm(parsed)
    sec_per_ql = 60.0 / bpm
    slur_pairs = _slur_pairs(parsed)
    notes: list[GraphNote] = []
    source_ids: list[int] = []
    idx = 0
    for n in parsed.recurse().getElementsByClass(note.Note):
        if _is_decorative(n):
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
        notes.append(
            GraphNote(
                index=idx,
                pitch=int(n.pitch.midi),
                start=start,
                end=end,
                duration=end - start,
                ql_start=start_ql,
                ql_end=start_ql + dur_ql,
                measure=measure_num,
                tie_type=_tie_type(n),
                source_note_indices=[idx],
            )
        )
        source_ids.append(id(n))
        idx += 1
    order = sorted(range(len(notes)), key=lambda i: (notes[i].start, notes[i].pitch))
    notes = [notes[i] for i in order]
    source_ids = [source_ids[i] for i in order]
    collapsed: list[GraphNote] = []
    collapsed_ids: list[int] = []
    for current, current_id in zip(notes, source_ids):
        previous = collapsed[-1] if collapsed else None
        prev_id = collapsed_ids[-1] if collapsed_ids else None
        same_pitch = previous is not None and previous.pitch == current.pitch
        abutting = previous is not None and _abuts(previous.ql_end, current.ql_start)
        tied = (
            same_pitch
            and abutting
            and (
                (
                    previous.tie_type in _TIE_START
                    and current.tie_type in _TIE_STOP
                )
                or (
                    previous.tie_type in _TIE_START
                    and current.tie_type is None
                )
                or (
                    previous.tie_type is None
                    and current.tie_type in _TIE_STOP
                )
                or (prev_id, current_id) in slur_pairs
            )
        )
        if previous is not None and tied:
            previous.end = max(previous.end, current.end)
            previous.ql_end = max(previous.ql_end, current.ql_end)
            previous.duration = previous.end - previous.start
            previous.tie_type = current.tie_type or previous.tie_type
            previous.source_note_indices = [
                *(previous.source_note_indices or []),
                *(current.source_note_indices or []),
            ]
            collapsed_ids[-1] = current_id
            continue
        collapsed.append(current)
        collapsed_ids.append(current_id)
    notes = collapsed
    for i, n in enumerate(notes):
        n.index = i

    edges: list[LegalEdge] = []
    for i in range(len(notes) - 1):
        edges.append(LegalEdge(from_index=i, to_index=i + 1, kind="next"))
    edges.extend(_written_repeat_edges(parsed, notes))
    duration = notes[-1].end if notes else 0.0
    return ScoreGraph(notes=notes, legal_edges=edges, duration_sec=duration, bpm=bpm)


def _first_last_in_measure(
    notes: list[GraphNote], measure: int
) -> tuple[int | None, int | None]:
    idxs = [n.index for n in notes if n.measure == measure]
    if not idxs:
        return None, None
    return idxs[0], idxs[-1]


def _written_repeat_edges(score, notes: list[GraphNote]) -> list[LegalEdge]:
    if not notes:
        return []
    starts: list[int] = []
    ends: list[int] = []
    for el in score.recurse().getElementsByClass(bar.Repeat):
        measure = el.getContextByClass(stream.Measure)
        if measure is None or measure.number is None:
            continue
        m = int(measure.number)
        direction = str(getattr(el, "direction", "")).lower()
        if direction == "start":
            starts.append(m)
        else:
            ends.append(m)
    edges: list[LegalEdge] = []
    for start_m, end_m in zip(starts, ends):
        _first_s, last_e = _first_last_in_measure(notes, end_m)
        first_s, _last_s = _first_last_in_measure(notes, start_m)
        if last_e is not None and first_s is not None:
            edges.append(LegalEdge(from_index=last_e, to_index=first_s, kind="written_repeat"))

    for br in score.recurse().getElementsByClass(spanner.RepeatBracket):
        try:
            measures = [int(m.number) for m in br.getSpannedElements() if getattr(m, "number", None)]
        except Exception:
            measures = []
        if len(measures) >= 2:
            _f, last = _first_last_in_measure(notes, measures[-1])
            first, _l = _first_last_in_measure(notes, measures[0])
            if last is not None and first is not None:
                edges.append(LegalEdge(from_index=last, to_index=first, kind="volta"))
    return edges


def legal_targets(graph: ScoreGraph, from_index: int) -> set[int]:
    out = {from_index + 1} if from_index + 1 < len(graph.notes) else set()
    for edge in graph.legal_edges:
        if edge.from_index == from_index:
            out.add(edge.to_index)
    return out


def span_is_legal_continuation(graph: ScoreGraph, cursor: int, i0: int, i1: int) -> bool:
    if i0 == cursor:
        return True
    if i0 in legal_targets(graph, max(0, cursor - 1)):
        return True
    return False
