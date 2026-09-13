from __future__ import annotations

import numpy as np

from alignmodel.audio import (
    cents_off,
    chroma_slice,
    dtw_normalized_cost,
    mean_chroma,
    pitch_class_mismatch,
    score_chroma_template,
    spectral_midi_frames,
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
    ref_audio: np.ndarray | None = None,
    mel: np.ndarray | None = None,
    learned=None,
) -> None:
    if state.transcribed_notes and learned is not None:
        _run_transcribed_note_errors(state, learned=learned, mel=mel)
        state.labels = [
            label for label in state.labels if label.type != "intonation_error"
        ]
        state.stages_run.append(2)
        return
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
    perf_f0 = ref_f0 = None
    if ref_audio is not None:
        perf_f0 = spectral_midi_frames(audio, state.sr, hop_length=state.config.hop_length)
        ref_f0 = spectral_midi_frames(
            ref_audio, state.sr, hop_length=state.config.hop_length
        )
    pairs: list[PairedEvent] = []
    for seg in state.segments:
        label_count = len(state.labels)
        segment_pairs = _edits_for_segment(
            state,
            seg,
            audio,
            chroma,
            ref_chroma,
            perf_f0=perf_f0,
            ref_f0=ref_f0,
        )
        if seg.is_repetition:
            # Replayed errors are repeated-pass annotations, not first-pass
            # mistakes in the official melody metric.
            del state.labels[label_count:]
        pairs.extend(segment_pairs)
    state.pairs = pairs
    if learned is not None and mel is not None:
        from alignmodel.stages.learned import apply_learned_edits

        apply_learned_edits(state, mel, learned)
    if not bool(getattr(state.config, "detect_intonation", False)):
        state.labels = [
            label for label in state.labels if label.type != "intonation_error"
        ]
    state.stages_run.append(2)


def _run_transcribed_note_errors(
    state: PipelineState, *, learned, mel=None
) -> None:
    """Layer 2: classify discrete note errors from the shared note mapping."""

    from alignmodel.stages.dc_alignment import ensure_rhythm_pairs

    pairs = ensure_rhythm_pairs(state, learned=learned, mel=mel)
    state.pairs = list(pairs)
    mapping = state.note_mapping
    if len(mapping) != len(state.transcribed_notes):
        mapping = [None] * len(state.transcribed_notes)
    mapped_score = {index for index in mapping if index is not None}
    minimum_confidence = float(state.config.note_error_min_confidence)
    for index, played in enumerate(state.transcribed_notes):
        target = mapping[index]
        if target is None:
            if played.confidence < minimum_confidence:
                continue
            state.labels.append(
                PipelineLabel(
                    id=next_label_id(state),
                    type="extra_note",
                    start_time=played.start,
                    end_time=played.end,
                    comment="transcribed note not mapped to written score",
                    pitches=[played.pitch],
                )
            )
            continue
        if not (0 <= target < len(state.score.notes)):
            continue
        written = state.score.notes[target]
        if (
            played.pitch != written.pitch
            and played.confidence >= minimum_confidence
        ):
            state.labels.append(
                PipelineLabel(
                    id=next_label_id(state),
                    type="wrong_note",
                    start_time=played.start,
                    end_time=played.end,
                    comment=(
                        f"played MIDI {played.pitch}; "
                        f"expected written MIDI {written.pitch}"
                    ),
                    measure_number=written.measure,
                    note_id=f"note_{written.index:04d}",
                    pitches=[played.pitch],
                )
            )
    for written in state.score.notes:
        if written.is_rest or written.index in mapped_score:
            continue
        state.labels.append(
            PipelineLabel(
                id=next_label_id(state),
                type="missed_note",
                start_time=written.start,
                end_time=written.end,
                comment=f"written MIDI {written.pitch} was not transcribed",
                measure_number=written.measure,
                note_id=f"note_{written.index:04d}",
                pitches=[written.pitch],
            )
        )


def _median_f0(
    track: tuple[np.ndarray, np.ndarray] | None,
    start: float,
    end: float,
    hop_sec: float,
) -> float | None:
    if track is None:
        return None
    midi, strength = track
    if midi.size == 0:
        return None
    i0 = max(0, min(len(midi), int(round(start / hop_sec)) + 1))
    i1 = max(i0 + 1, min(len(midi), int(round(end / hop_sec)) - 1))
    values = midi[i0:i1]
    weights = strength[i0:i1]
    valid = np.isfinite(values) & (weights > 0.0)
    if not np.any(valid):
        return None
    values = values[valid]
    weights = weights[valid]
    floor = 0.25 * float(np.max(weights))
    strong = values[weights >= floor]
    if strong.size < 2:
        strong = values
    return float(np.median(strong))


def _fine_cents_for_event(
    perf_track: tuple[np.ndarray, np.ndarray] | None,
    ref_track: tuple[np.ndarray, np.ndarray] | None,
    perf_start: float,
    perf_end: float,
    ref_start: float,
    ref_end: float,
    hop_sec: float,
) -> float | None:
    perf_midi = _median_f0(perf_track, perf_start, perf_end, hop_sec)
    ref_midi = _median_f0(ref_track, ref_start, ref_end, hop_sec)
    if perf_midi is None or ref_midi is None:
        return None
    return float(np.clip(100.0 * (perf_midi - ref_midi), -300.0, 300.0))


def _f0_intonation_spans(
    warping_path: np.ndarray,
    perf_track: tuple[np.ndarray, np.ndarray] | None,
    ref_track: tuple[np.ndarray, np.ndarray] | None,
    *,
    perf_origin: float,
    ref_origin: float,
    hop_sec: float,
    tolerance: float,
) -> list[tuple[float, float, float]]:
    """Group sustained DTW-aligned 25–95 cent deviations."""
    if perf_track is None or ref_track is None or warping_path.size == 0:
        return []
    perf_midi, perf_strength = perf_track
    ref_midi, ref_strength = ref_track
    perf_base = int(round(perf_origin / hop_sec))
    ref_base = int(round(ref_origin / hop_sec))
    rows: list[tuple[int, float]] = []
    for ref_local, perf_local in np.asarray(warping_path, dtype=np.int64):
        pi = perf_base + int(perf_local)
        ri = ref_base + int(ref_local)
        if not (0 <= pi < len(perf_midi) and 0 <= ri < len(ref_midi)):
            continue
        if not (np.isfinite(perf_midi[pi]) and np.isfinite(ref_midi[ri])):
            continue
        if perf_strength[pi] <= 0.0 or ref_strength[ri] <= 0.0:
            continue
        cents = 100.0 * float(perf_midi[pi] - ref_midi[ri])
        if max(25.0, tolerance) <= abs(cents) < 95.0:
            rows.append((pi, cents))
    if not rows:
        return []
    by_frame: dict[int, list[float]] = {}
    for frame, cents in rows:
        by_frame.setdefault(frame, []).append(cents)
    points = sorted((frame, float(np.median(values))) for frame, values in by_frame.items())
    spans: list[tuple[float, float, float]] = []
    group = [points[0]]
    for point in points[1:]:
        same_sign = np.sign(point[1]) == np.sign(group[-1][1])
        if point[0] - group[-1][0] <= 2 and same_sign:
            group.append(point)
            continue
        if len(group) >= 3:
            spans.append(
                (
                    group[0][0] * hop_sec,
                    (group[-1][0] + 1) * hop_sec,
                    float(np.median([value for _, value in group])),
                )
            )
        group = [point]
    if len(group) >= 3:
        spans.append(
            (
                group[0][0] * hop_sec,
                (group[-1][0] + 1) * hop_sec,
                float(np.median([value for _, value in group])),
            )
        )
    return spans


def _edits_for_segment(
    state: PipelineState,
    seg: UnfoldedSegment,
    audio: np.ndarray,
    chroma: np.ndarray,
    ref_chroma: np.ndarray | None,
    *,
    perf_f0: tuple[np.ndarray, np.ndarray] | None = None,
    ref_f0: tuple[np.ndarray, np.ndarray] | None = None,
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
    for cents_start, cents_end, cents_value in _f0_intonation_spans(
        wp,
        perf_f0,
        ref_f0,
        perf_origin=seg.perf_start,
        ref_origin=t0,
        hop_sec=cfg.hop_length / float(state.sr),
        tolerance=float(cfg.cents_tolerance),
    ):
        state.labels.append(
            PipelineLabel(
                id=next_label_id(state),
                type="intonation_error",
                start_time=cents_start,
                end_time=cents_end,
                comment="DTW-aligned fine-pitch deviation",
                deviation_cents=cents_value,
            )
        )

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
        cents = _fine_cents_for_event(
            perf_f0,
            ref_f0,
            p0,
            p1,
            note.start,
            note.end,
            state.config.hop_length / float(state.sr),
        )
        if cents is None and ref_chroma is not None:
            cents = cents_off(vec_ref, vec_perf)
        # Compare against the rendered clean reference when available. Using
        # the written score pitch here creates a systematic wrong-note flood
        # for transposing instruments and masks sub-semitone intonation.
        mismatch = pitch_class_mismatch(vec_ref, vec_perf, cfg.chroma_peak_min)
        ref_peak = int(np.argmax(vec_ref))
        perf_peak = int(np.argmax(vec_perf))
        peak_steps = min((perf_peak - ref_peak) % 12, (ref_peak - perf_peak) % 12)
        intonation_like = (
            cents is not None
            and cfg.cents_tolerance < abs(cents) < 95.0
            and peak_steps <= 1
        )
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
        if mismatch and not intonation_like:
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
        if intonation_like:
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
