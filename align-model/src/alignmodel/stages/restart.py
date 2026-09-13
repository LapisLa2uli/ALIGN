from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alignmodel.audio import chroma_slice, dtw_normalized_cost, score_chroma_template
from alignmodel.stages.repetition import (
    RepetitionCandidate,
    find_past_repetitions,
    silence_intervals,
)
from alignmodel.stages.score_graph import span_is_legal_continuation
from alignmodel.types import (
    GraphNote,
    PipelineLabel,
    PipelineState,
    RepeatRange,
    RestartHypothesis,
    ScoreGraph,
    UnfoldedSegment,
    next_label_id,
)


@dataclass
class _SpanMatch:
    i0: int
    i1: int
    cost: float
    is_restart: bool


def run_stage1(
    state: PipelineState,
    chroma: np.ndarray,
    ref_chroma: np.ndarray | None = None,
    *,
    audio: np.ndarray | None = None,
    mel: np.ndarray | None = None,
    learned=None,
) -> None:
    if state.transcribed_notes:
        from alignmodel.stages.repetition import apply_note_repetitions

        apply_note_repetitions(state, learned=learned)
        state.beam = [
            RestartHypothesis(
                segments=list(state.segments),
                total_cost=0.0,
                unexplained_sec=0.0,
                score=0.0,
            )
        ]
        state.stages_run.append(1)
        return
    notes = state.score.notes
    cfg = state.config
    if not notes:
        state.segments = [
            UnfoldedSegment(0.0, state.duration_sec, 0, 0, 0.0, False, None)
        ]
        state.stages_run.append(1)
        return

    repetitions: list[RepetitionCandidate] = []
    if audio is not None:
        silences = silence_intervals(
            audio,
            state.sr,
            silence_db=float(cfg.silence_db),
            min_silence_sec=float(cfg.min_silence_sec),
            hop_length=int(cfg.hop_length),
        )
        repetitions = find_past_repetitions(
            chroma,
            state.hop_sec,
            state.duration_sec,
            silences,
            max_lookback_sec=float(cfg.repetition_max_lookback_sec),
            search_step_sec=float(cfg.repetition_search_step_sec),
            probe_sec=float(cfg.repetition_probe_sec),
            min_confidence=float(cfg.repetition_min_confidence),
        )
    state.segments = _segments_from_retrieval(state, repetitions)
    state.beam = [
        RestartHypothesis(
            segments=list(state.segments),
            total_cost=0.0,
            unexplained_sec=sum(
                max(0.0, item.start_time - item.source_end) for item in repetitions
            ),
            score=0.0,
        )
    ]
    _apply_retrieved_repetitions(state, repetitions)
    state.stages_run.append(1)


def _note_span_for_times(
    notes: list[GraphNote], start_time: float, end_time: float
) -> tuple[int, int]:
    hits = [
        note.index
        for note in notes
        if note.end > start_time and note.start < end_time and not note.is_rest
    ]
    if hits:
        return min(hits), max(hits) + 1
    nearest = min(
        range(len(notes)),
        key=lambda i: abs(float(notes[i].start) - float(start_time)),
    )
    return nearest, min(len(notes), nearest + 1)


def _segments_from_retrieval(
    state: PipelineState, repetitions: list[RepetitionCandidate]
) -> list[UnfoldedSegment]:
    """Build monotonic first-pass segments plus explicit retrieved replay spans."""
    notes = state.score.notes
    if not repetitions:
        return [
            UnfoldedSegment(
                0.0,
                state.duration_sec,
                0,
                len(notes),
                0.0,
            )
        ]
    segments: list[UnfoldedSegment] = []
    perf_cursor = 0.0
    score_cursor = 0.0
    score_end = max(float(state.score.duration_sec), float(notes[-1].end))
    for item in sorted(repetitions, key=lambda value: value.start_time):
        normal_end = max(perf_cursor, item.source_end)
        if normal_end - perf_cursor >= 0.05:
            i0, i1 = _note_span_for_times(notes, score_cursor, item.source_end)
            segments.append(
                UnfoldedSegment(perf_cursor, normal_end, i0, i1, 0.0)
            )
        source_i0, source_i1 = _note_span_for_times(
            notes, item.source_start, item.source_end
        )
        source = RepeatRange(item.source_start, item.source_end)
        segments.append(
            UnfoldedSegment(
                item.start_time,
                item.end_time,
                source_i0,
                source_i1,
                0.0,
                True,
                source,
            )
        )
        perf_cursor = item.end_time
        score_cursor = item.source_end
    if state.duration_sec - perf_cursor >= 0.05:
        i0, i1 = _note_span_for_times(notes, score_cursor, score_end + 1e-3)
        segments.append(
            UnfoldedSegment(perf_cursor, state.duration_sec, i0, i1, 0.0)
        )
    return segments


def _apply_retrieved_repetitions(
    state: PipelineState, repetitions: list[RepetitionCandidate]
) -> None:
    """Mark alignment segments and emit labels from retrieval detections."""
    for item in repetitions:
        source = RepeatRange(item.source_start, item.source_end)
        repeated_segments = [
            seg
            for seg in state.segments
            if seg.perf_start < item.end_time and item.start_time < seg.perf_end
        ]
        for seg in repeated_segments:
            seg.is_repetition = True
            seg.repeats_label_range = source
        state.labels.append(
            PipelineLabel(
                id=next_label_id(state),
                type="repetition",
                start_time=item.start_time,
                end_time=item.end_time,
                source="pipeline",
                comment=f"past-span retrieval confidence={item.confidence:.3f}",
                repeats_label_range=source,
                extra_copies=item.extra_copies,
            )
        )


def _mark_score_replays(state: PipelineState) -> None:
    """If two adjacent windows mapped onto the same score notes, the later is a restart."""
    for i in range(1, len(state.segments)):
        prev = state.segments[i - 1]
        cur = state.segments[i]
        prev_n = prev.score_i1 - prev.score_i0
        cur_n = cur.score_i1 - cur.score_i0
        short = min(prev_n, cur_n)
        if short < 2:
            continue
        overlap = min(prev.score_i1, cur.score_i1) - max(prev.score_i0, cur.score_i0)
        if overlap >= 0.6 * short:
            cur.is_repetition = True
            if cur.repeats_label_range is None:
                cur.repeats_label_range = RepeatRange(prev.perf_start, prev.perf_end)
            cur.score_i0 = prev.score_i0
            cur.score_i1 = prev.score_i1


def _search_beam(
    graph: ScoreGraph,
    chroma: np.ndarray,
    windows: list[tuple[float, float]],
    hop_sec: float,
    k: int,
    cfg,
    ref_chroma: np.ndarray | None,
    copy_cuts: list[float] | None = None,
) -> list[RestartHypothesis]:
    notes = graph.notes
    n = len(notes)
    copy_cuts = copy_cuts or []
    # Each beam item: cursor, visited list of (i0,i1,perf_start,perf_end), cost, unexplained, segments
    BeamItem = tuple[int, list[tuple[int, int, float, float]], float, float, list[UnfoldedSegment]]
    beam: list[BeamItem] = [(0, [], 0.0, 0.0, [])]

    for p0, p1 in windows:
        win = chroma_slice(chroma, p0, p1, hop_sec)
        at_copy = any(abs(p0 - c) < 0.35 for c in copy_cuts)
        nxt: list[BeamItem] = []
        for cursor, visited, cost, unexplained, segs in beam:
            matches = _candidate_spans(
                graph, notes, win, hop_sec, cursor, visited, cfg, ref_chroma
            )
            if at_copy and segs:
                prev = segs[-1]
                replay = _replay_span(notes, win, hop_sec, prev, ref_chroma)
                if replay is not None:
                    matches = [replay] + [m for m in matches if not (m.i0 == replay.i0 and m.i1 == replay.i1)]
            if not matches:
                matches = [
                    _SpanMatch(
                        i0=min(cursor, n - 1),
                        i1=n,
                        cost=1.0,
                        is_restart=False,
                    )
                ]
            for match in matches[:k]:
                restart = match.is_restart
                repeats = None
                if restart:
                    src = _source_span(visited, match.i0, match.i1)
                    if src is not None:
                        repeats = RepeatRange(start_time=src[0], end_time=src[1])
                seg = UnfoldedSegment(
                    perf_start=p0,
                    perf_end=p1,
                    score_i0=match.i0,
                    score_i1=match.i1,
                    dtw_cost=match.cost,
                    is_repetition=restart,
                    repeats_label_range=repeats,
                )
                new_vis = list(visited)
                new_vis.append((match.i0, match.i1, p0, p1))
                new_cursor = match.i1 if not restart else cursor
                extra = 0.0 if match.cost < 0.35 else (p1 - p0) * 0.25
                nxt.append(
                    (
                        new_cursor,
                        new_vis,
                        cost + match.cost,
                        unexplained + extra,
                        segs + [seg],
                    )
                )
        nxt.sort(key=lambda item: item[2] + 0.15 * item[3])
        beam = nxt[:k]

    out: list[RestartHypothesis] = []
    for _cursor, _vis, cost, unexplained, segs in beam:
        if not segs:
            continue
        out.append(
            RestartHypothesis(
                segments=segs,
                total_cost=cost,
                unexplained_sec=unexplained,
                score=cost + 0.15 * unexplained,
            )
        )
    return out


def _replay_span(
    notes: list[GraphNote],
    win: np.ndarray,
    hop_sec: float,
    prev: UnfoldedSegment,
    ref_chroma: np.ndarray | None,
) -> _SpanMatch | None:
    if prev.score_i1 <= prev.score_i0:
        return None
    span = notes[prev.score_i0 : prev.score_i1]
    if not span:
        return None
    tmpl = _span_template(span, hop_sec, ref_chroma)
    cost, _wp = dtw_normalized_cost(tmpl, win)
    return _SpanMatch(i0=prev.score_i0, i1=prev.score_i1, cost=cost * 0.82, is_restart=True)


def _source_span(
    visited: list[tuple[int, int, float, float]], i0: int, i1: int
) -> tuple[float, float] | None:
    best = None
    best_ov = 0
    for v0, v1, p0, p1 in visited:
        ov = min(v1, i1) - max(v0, i0)
        if ov > best_ov:
            best_ov = ov
            best = (p0, p1)
    return best


def _overlaps_visited(visited: list[tuple[int, int, float, float]], i0: int, i1: int) -> bool:
    for v0, v1, _p0, _p1 in visited:
        if min(v1, i1) - max(v0, i0) > 0:
            return True
    return False


def _span_template(
    notes: list[GraphNote], hop_sec: float, ref_chroma: np.ndarray | None
) -> np.ndarray:
    if ref_chroma is not None:
        return chroma_slice(ref_chroma, notes[0].start, notes[-1].end, hop_sec)
    return score_chroma_template(notes, hop_sec)


def _candidate_spans(
    graph: ScoreGraph,
    notes: list[GraphNote],
    win: np.ndarray,
    hop_sec: float,
    cursor: int,
    visited: list[tuple[int, int, float, float]],
    cfg,
    ref_chroma: np.ndarray | None,
) -> list[_SpanMatch]:
    n = len(notes)
    if n == 0:
        return []
    win_dur = max(win.shape[1] * hop_sec, 0.25)
    matches: list[_SpanMatch] = []
    starts = list(range(n))
    # Prefer searching near the cursor first.
    starts.sort(key=lambda i: abs(i - cursor))
    budget = min(n, 24)
    for i0 in starts[:budget]:
        span_start = notes[i0].start
        i1 = i0 + 1
        while i1 < n and notes[i1 - 1].end - span_start < win_dur * cfg.span_dur_lo:
            i1 += 1
        for end in range(max(i0 + 1, i1 - 1), min(n, i1 + 3) + 1):
            if end <= i0:
                continue
            span_dur = notes[end - 1].end - span_start
            if span_dur > win_dur * cfg.span_dur_hi:
                break
            if span_dur < win_dur * cfg.span_dur_lo and end < n:
                continue
            tmpl = _span_template(notes[i0:end], hop_sec, ref_chroma)
            cost, _wp = dtw_normalized_cost(tmpl, win)
            legal = span_is_legal_continuation(graph, cursor, i0, end)
            restart = (not legal) and _overlaps_visited(visited, i0, end)
            if cursor > 0 and not legal and not restart:
                cost += 0.08 * abs(i0 - cursor) / max(n, 1)
            if legal:
                cost *= 0.92
            matches.append(_SpanMatch(i0=i0, i1=end, cost=cost, is_restart=restart))
    matches.sort(key=lambda m: (m.cost, m.i0))
    # Keep diverse starts
    uniq: list[_SpanMatch] = []
    seen: set[int] = set()
    for m in matches:
        if m.i0 in seen:
            continue
        seen.add(m.i0)
        uniq.append(m)
        if len(uniq) >= 6:
            break
    return uniq


def _emit_repetition_labels(state: PipelineState) -> None:
    groups: dict[tuple[float | None, float | None], list[UnfoldedSegment]] = {}
    for seg in state.segments:
        if not seg.is_repetition:
            continue
        src = seg.repeats_label_range
        key = (
            round(src.start_time, 3) if src is not None else None,
            round(src.end_time, 3) if src is not None else None,
        )
        groups.setdefault(key, []).append(seg)
    for segs in groups.values():
        segs = sorted(segs, key=lambda item: item.perf_start)
        copies = min(2, max(1, len(segs)))
        lab = PipelineLabel(
            id=next_label_id(state),
            type="repetition",
            start_time=segs[0].perf_start,
            end_time=segs[-1].perf_end,
            comment="practice restart (stage 1)",
            repeats_label_range=segs[0].repeats_label_range,
            extra_copies=copies,
        )
        state.labels.append(lab)
