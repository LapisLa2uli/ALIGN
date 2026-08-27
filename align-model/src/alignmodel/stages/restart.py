from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alignmodel.audio import chroma_slice, dtw_normalized_cost, score_chroma_template
from alignmodel.stages.boundaries import window_times, _merge_cuts
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
    mel: np.ndarray | None = None,
    learned=None,
) -> None:
    notes = state.score.notes
    cfg = state.config
    if not notes:
        state.segments = [
            UnfoldedSegment(0.0, state.duration_sec, 0, 0, 0.0, False, None)
        ]
        state.stages_run.append(1)
        return

    windows = window_times(state.boundaries, state.duration_sec, cfg.min_window_sec)
    copy_cuts: list[float] = []
    if learned is not None and getattr(learned, "restart", None) is not None and mel is not None:
        from alignmodel.stages.learned import propose_copy_cuts

        copy_cuts = propose_copy_cuts(mel, learned, state.duration_sec)
        if copy_cuts:
            state.boundaries = _merge_cuts(
                list(state.boundaries) + copy_cuts,
                state.duration_sec,
                min_gap=cfg.min_window_sec * 0.5,
            )
            windows = window_times(state.boundaries, state.duration_sec, cfg.min_window_sec)
    beam = _search_beam(
        state.score,
        chroma,
        windows,
        state.hop_sec,
        cfg.beam_k,
        cfg,
        ref_chroma,
        copy_cuts=copy_cuts,
    )
    if not beam:
        beam = [
            RestartHypothesis(
                segments=[
                    UnfoldedSegment(
                        0.0,
                        state.duration_sec,
                        0,
                        len(notes),
                        0.0,
                        False,
                        None,
                    )
                ],
                total_cost=0.0,
                unexplained_sec=0.0,
                score=0.0,
            )
        ]
    beam.sort(key=lambda h: h.score)
    for rank, hyp in enumerate(beam):
        for seg in hyp.segments:
            seg.hypothesis_rank = rank
    state.beam = beam[: cfg.beam_k]
    chosen = state.beam[0]
    state.segments = chosen.segments
    for seg in state.segments:
        seg.is_repetition = False
        seg.repeats_label_range = None
    _mark_score_replays(state)
    if learned is not None and getattr(learned, "restart", None) is not None and mel is not None:
        from alignmodel.stages.learned import apply_learned_restarts

        apply_learned_restarts(state, mel, learned)
    _emit_repetition_labels(state)
    state.stages_run.append(1)


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
    for seg in state.segments:
        if not seg.is_repetition:
            continue
        lab = PipelineLabel(
            id=next_label_id(state),
            type="repetition",
            start_time=seg.perf_start,
            end_time=seg.perf_end,
            comment="practice restart (stage 1)",
            repeats_label_range=seg.repeats_label_range,
        )
        state.labels.append(lab)
