from __future__ import annotations

import numpy as np

from alignmodel.audio import (
    cents_off,
    chroma_slice,
    dtw_normalized_cost,
    mean_chroma,
    pitch_class_mismatch,
    score_chroma_template,
)
from alignmodel.types import (
    GraphNote,
    PairedEvent,
    PipelineLabel,
    PipelineState,
    UnfoldedSegment,
    next_label_id,
)


def run_stage2(
    state: PipelineState,
    audio: np.ndarray,
    chroma: np.ndarray,
    ref_chroma: np.ndarray | None = None,
    *,
    mel: np.ndarray | None = None,
    learned=None,
) -> None:
    if not state.segments:
        state.segments = [
            UnfoldedSegment(
                0.0,
                state.duration_sec,
                0,
                len(state.score.notes),
                0.0,
                False,
                None,
            )
        ]
    pairs: list[PairedEvent] = []
    for seg in state.segments:
        pairs.extend(_edits_for_segment(state, seg, audio, chroma, ref_chroma))
    state.pairs = pairs
    if learned is not None and mel is not None:
        from alignmodel.stages.learned import apply_learned_edits

        apply_learned_edits(state, mel, learned)
    state.stages_run.append(2)


def _edits_for_segment(
    state: PipelineState,
    seg: UnfoldedSegment,
    audio: np.ndarray,
    chroma: np.ndarray,
    ref_chroma: np.ndarray | None,
) -> list[PairedEvent]:
    notes = state.score.notes[seg.score_i0 : seg.score_i1]
    cfg = state.config
    hop = state.hop_sec
    if not notes:
        _maybe_extra_span(state, seg.perf_start, seg.perf_end, "empty score span")
        return []

    if ref_chroma is not None:
        tmpl = chroma_slice(ref_chroma, notes[0].start, notes[-1].end, hop)
    else:
        tmpl = score_chroma_template(notes, hop)
    win = chroma_slice(chroma, seg.perf_start, seg.perf_end, hop)
    _cost, wp = dtw_normalized_cost(tmpl, win)
    t0 = notes[0].start
    events = _map_notes_to_perf(notes, wp, t0, hop, seg.perf_start)
    events = _refine_onsets(events, audio, state.sr, cfg)

    tmpl_used = set(int(r) for r, _p in wp) if len(wp) else set()
    perf_used = set(int(p) for _r, p in wp) if len(wp) else set()

    pairs: list[PairedEvent] = []
    for ev in events:
        note: GraphNote = ev["note"]
        p0, p1 = float(ev["perf_start"]), float(ev["perf_end"])
        if p1 <= p0:
            p1 = p0 + 0.05
        vec_score = np.zeros(12, dtype=np.float64)
        vec_score[int(note.pitch) % 12] = 1.0
        vec_perf = mean_chroma(chroma, p0, p1, hop)
        vec_ref = (
            mean_chroma(ref_chroma, note.start, note.end, hop)
            if ref_chroma is not None
            else vec_score
        )
        cents = cents_off(vec_ref, vec_perf) if ref_chroma is not None else None
        mismatch = pitch_class_mismatch(vec_score, vec_perf, cfg.chroma_peak_min)
        tmpl_i0 = int(round((note.start - t0) / hop))
        tmpl_i1 = max(tmpl_i0 + 1, int(round((note.end - t0) / hop)))
        covered = any(tmpl_i0 <= f < tmpl_i1 for f in tmpl_used)
        short = note.duration < cfg.min_miss_sec

        if not covered and not short:
            state.labels.append(
                PipelineLabel(
                    id=next_label_id(state),
                    type="missed_note",
                    start_time=p0,
                    end_time=p1,
                    comment=f"uncovered score MIDI {note.pitch}",
                    measure_number=note.measure,
                    note_id=f"note_{note.index:04d}",
                )
            )
            continue
        if mismatch:
            state.labels.append(
                PipelineLabel(
                    id=next_label_id(state),
                    type="wrong_note",
                    start_time=p0,
                    end_time=p1,
                    comment=f"pitch-class mismatch MIDI {note.pitch}",
                    deviation_cents=cents,
                    measure_number=note.measure,
                    note_id=f"note_{note.index:04d}",
                )
            )
            pairs.append(
                PairedEvent(
                    score_index=note.index,
                    pitch=note.pitch,
                    ref_start=note.start,
                    ref_end=note.end,
                    perf_start=p0,
                    perf_end=p1,
                    kind="substitute",
                    cents=cents,
                    measure=note.measure,
                )
            )
            continue
        if cents is not None and abs(cents) > cfg.cents_tolerance and not mismatch:
            state.labels.append(
                PipelineLabel(
                    id=next_label_id(state),
                    type="intonation_error",
                    start_time=p0,
                    end_time=p1,
                    comment="chroma cents over tolerance",
                    deviation_cents=cents,
                    measure_number=note.measure,
                    note_id=f"note_{note.index:04d}",
                )
            )
        pairs.append(
            PairedEvent(
                score_index=note.index,
                pitch=note.pitch,
                ref_start=note.start,
                ref_end=note.end,
                perf_start=p0,
                perf_end=p1,
                kind="match",
                cents=cents,
                measure=note.measure,
            )
        )

    n_perf = win.shape[1]
    unused: list[tuple[float, float]] = []
    on = None
    for i in range(n_perf):
        t = seg.perf_start + i * hop
        energy = float(np.max(win[:, i])) if n_perf else 0.0
        if i not in perf_used and energy > cfg.chroma_peak_min:
            if on is None:
                on = t
        elif on is not None:
            unused.append((on, t))
            on = None
    if on is not None:
        unused.append((on, seg.perf_end))
    for a, b in _merge_spans(unused, gap=0.05):
        if b - a >= cfg.min_extra_sec:
            # skip extras that sit entirely inside a matched note
            if _inside_pair(a, b, pairs):
                continue
            _maybe_extra_span(state, a, b, "unmatched performance frames")
    return pairs


def _map_notes_to_perf(
    notes: list[GraphNote],
    wp: np.ndarray,
    t0: float,
    hop: float,
    perf_origin: float,
) -> list[dict]:
    events = []
    if len(wp) == 0:
        cursor = perf_origin
        for note in notes:
            dur = max(note.duration, 0.05)
            events.append(
                {
                    "note": note,
                    "perf_start": cursor,
                    "perf_end": cursor + dur,
                }
            )
            cursor += dur
        return events
    buckets: dict[int, list[int]] = {n.index: [] for n in notes}
    for ref_i, perf_i in wp:
        t = t0 + int(ref_i) * hop
        for note in notes:
            if note.start <= t < note.end or (
                note is notes[-1] and abs(t - note.end) < hop
            ):
                buckets[note.index].append(int(perf_i))
                break
    for note in notes:
        frames = buckets.get(note.index) or []
        if frames:
            p0 = perf_origin + min(frames) * hop
            p1 = perf_origin + (max(frames) + 1) * hop
        else:
            p0 = perf_origin + max(0.0, note.start - t0)
            p1 = p0 + max(note.duration, 0.05)
        events.append({"note": note, "perf_start": p0, "perf_end": max(p1, p0 + 0.04)})
    return events


def _refine_onsets(events: list[dict], audio: np.ndarray, sr: int, cfg) -> list[dict]:
    if audio is None or len(audio) == 0 or sr <= 0:
        return events
    hop = 256
    frame_length = 1024
    n = len(audio)
    if n < frame_length:
        return events
    frames = 1 + (n - frame_length) // hop
    rms = np.empty(frames, dtype=np.float64)
    for i in range(frames):
        start = i * hop
        chunk = audio[start : start + frame_length]
        rms[i] = float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))
    hop_sec = hop / float(sr)
    eps = 1e-12
    rise = 10.0 ** (cfg.onset_rise_db / 20.0)
    prev_end = 0.0
    out = []
    for ev in events:
        t0 = float(ev["perf_start"])
        t1 = float(ev["perf_end"])
        item = dict(ev)
        if t1 - t0 < 0.04:
            out.append(item)
            prev_end = t1
            continue
        search_lo = max(prev_end, max(0.0, t0 - cfg.onset_lookback_sec))
        search_hi = min(t1 - 0.04, t0 + cfg.onset_max_shift_sec)
        if search_hi <= search_lo + hop_sec:
            out.append(item)
            prev_end = t1
            continue
        i0 = max(0, int(search_lo / hop_sec))
        i1 = min(frames, int(np.ceil(search_hi / hop_sec)) + 1)
        if i1 - i0 < 3:
            out.append(item)
            prev_end = t1
            continue
        window = rms[i0:i1]
        floor = max(float(np.percentile(window, 20)), eps)
        thresh = floor * rise
        onset_idx = None
        for k, val in enumerate(window):
            if val >= thresh and (k == 0 or window[k - 1] < thresh * 0.85):
                onset_idx = i0 + k
                break
        if onset_idx is not None:
            new_start = min(max(onset_idx * hop_sec, search_lo), search_hi)
            if t1 - new_start >= 0.04:
                item["perf_start"] = new_start
        out.append(item)
        prev_end = float(item["perf_end"])
    return out


def _merge_spans(spans: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    if not spans:
        return []
    ordered = sorted(spans)
    merged = [ordered[0]]
    for a, b in ordered[1:]:
        la, lb = merged[-1]
        if a <= lb + gap:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged


def _inside_pair(a: float, b: float, pairs: list[PairedEvent]) -> bool:
    mid = 0.5 * (a + b)
    for p in pairs:
        if p.perf_start <= mid <= p.perf_end:
            return True
    return False


def _maybe_extra_span(state: PipelineState, start: float, end: float, comment: str) -> None:
    if end - start < state.config.min_extra_sec:
        return
    state.labels.append(
        PipelineLabel(
            id=next_label_id(state),
            type="extra_note",
            start_time=start,
            end_time=end,
            comment=comment,
        )
    )
