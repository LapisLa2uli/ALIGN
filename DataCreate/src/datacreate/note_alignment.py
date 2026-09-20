from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from music21 import chord, converter, note, stream, tempo

from datacreate.score_notes import (
    collapse_tied_records,
    element_tie_type,
    is_decorative_element,
    slur_adjacent_ids,
    voice_key,
)
from datacreate.utils import read_json

_PITCH_TYPES = (note.Note, note.Rest, note.Unpitched, chord.Chord)
_PC_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _seconds_per_quarter(score) -> float:
    for el in score.flatten().getElementsByClass(tempo.MetronomeMark):
        if el.number:
            return 60.0 / float(el.number)
    return 0.5  # 120 BPM default


def _tempo_map(score) -> list[tuple[float, float]]:
    """(offset_ql, seconds_per_quarter) marks, sorted, covering offset 0."""
    marks: list[tuple[float, float]] = []
    for el in score.flatten().getElementsByClass(tempo.MetronomeMark):
        if not el.number:
            continue
        try:
            off = float(el.getOffsetInHierarchy(score))
        except Exception:  # noqa: BLE001
            off = float(getattr(el, "offset", 0.0))
        marks.append((off, 60.0 / float(el.number)))
    marks.sort(key=lambda item: item[0])
    if not marks:
        return [(0.0, 0.5)]
    if marks[0][0] > 1e-9:
        marks.insert(0, (0.0, marks[0][1]))
    # Last mark at a given offset wins.
    collapsed: list[tuple[float, float]] = []
    for off, spq in marks:
        if collapsed and abs(off - collapsed[-1][0]) < 1e-6:
            collapsed[-1] = (off, spq)
        else:
            collapsed.append((off, spq))
    return collapsed


def _ql_to_sec(offset_ql: float, tempo_map: list[tuple[float, float]]) -> float:
    sec = 0.0
    offset_ql = max(0.0, float(offset_ql))
    for i, (off, spq) in enumerate(tempo_map):
        next_off = tempo_map[i + 1][0] if i + 1 < len(tempo_map) else offset_ql
        end = min(offset_ql, next_off) if i + 1 < len(tempo_map) else offset_ql
        if end > off:
            sec += (end - off) * spq
        if i + 1 < len(tempo_map) and offset_ql <= next_off + 1e-12:
            break
    return sec


def _element_offset_ql(el, score, part) -> float:
    try:
        return float(el.getOffsetInHierarchy(score))
    except Exception:  # noqa: BLE001
        try:
            return float(el.getOffsetInHierarchy(part))
        except Exception:  # noqa: BLE001
            return float(getattr(el, "offset", 0.0))


def _pitch_label(el) -> str | None:
    if isinstance(el, note.Rest):
        return "rest"
    if isinstance(el, chord.Chord):
        return el.pitchedCommonName or "chord"
    if isinstance(el, note.Note):
        return el.pitch.nameWithOctave
    return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _midi_pitch_name(midi: int | None) -> str | None:
    if midi is None:
        return None
    pitch = int(midi)
    if pitch < 0 or pitch > 127:
        return None
    return f"{_PC_NAMES[pitch % 12]}{pitch // 12 - 1}"


def _normalize_transcribed_notes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """GUI-ready transcription notes plus the score index each one maps to."""
    mapping = payload.get("note_mapping") or []
    notes: list[dict[str, Any]] = []
    for index, raw in enumerate(payload.get("transcribed_notes") or []):
        midi = raw.get("midi")
        if midi is None:
            midi = raw.get("pitch") if isinstance(raw.get("pitch"), int) else None
        midi_i = _optional_int(midi)
        start = float(raw.get("start") or 0.0)
        end = float(raw.get("end") or start + 0.05)
        if end < start + 0.001:
            end = start + 0.001
        score_index = None
        if index < len(mapping):
            score_index = _optional_int(mapping[index])
        elif raw.get("score_index") is not None:
            score_index = _optional_int(raw.get("score_index"))
        notes.append(
            {
                "id": f"trans_{index:04d}",
                "index": index,
                "midi": midi_i,
                "pitch": _midi_pitch_name(midi_i),
                "start": round(start, 4),
                "end": round(end, 4),
                "perf_start": round(start, 4),
                "perf_end": round(end, 4),
                "confidence": float(raw.get("confidence") or 1.0),
                "cents": float(raw.get("cents") or 0.0),
                "score_index": score_index,
                "is_rest": False,
                "duration_ql": round(max(0.0625, (end - start) * 2.0), 4),
            }
        )
    return notes


def _midi_value(el) -> int | None:
    if isinstance(el, note.Rest):
        return None
    if isinstance(el, note.Note):
        return int(el.pitch.midi)
    if isinstance(el, chord.Chord) and el.pitches:
        return int(el.pitches[0].midi)
    return None


def _extract_score_events(score_path: Path) -> list[dict[str, Any]]:
    score = converter.parse(str(score_path))
    if not score.parts:
        return []

    tempo_map = _tempo_map(score)
    slur_pairs = slur_adjacent_ids(score)
    records: list[dict[str, Any]] = []

    for part_idx, part in enumerate(score.parts):
        for el in part.recurse().notesAndRests:
            if not isinstance(el, _PITCH_TYPES):
                continue
            if is_decorative_element(el):
                continue
            # Hierarchy offset before getContextByClass: that call can change
            # activeSite so el.offset becomes measure-relative.
            offset_ql = _element_offset_ql(el, score, part)
            duration_ql = float(el.duration.quarterLength)
            measure = el.getContextByClass(stream.Measure)
            measure_num = int(measure.number) if measure and measure.number is not None else None
            ref_start = _ql_to_sec(offset_ql, tempo_map)
            ref_end = _ql_to_sec(offset_ql + duration_ql, tempo_map)
            records.append(
                {
                    "part": part_idx,
                    "voice": voice_key(el, part_idx),
                    "el_id": id(el),
                    "measure": measure_num,
                    "offset_ql": offset_ql,
                    "duration_ql": duration_ql,
                    "is_rest": isinstance(el, note.Rest),
                    "pitch": _pitch_label(el),
                    "midi": _midi_value(el),
                    "tie_type": element_tie_type(el),
                    "ref_start": ref_start,
                    "ref_end": max(ref_end, ref_start + 0.001),
                }
            )

    events = collapse_tied_records(records, slur_pairs)
    events.sort(key=lambda ev: (int(ev["part"]), float(ev["offset_ql"]), float(ev["duration_ql"])))
    cleaned: list[dict[str, Any]] = []
    for i, ev in enumerate(events):
        cleaned.append(
            {
                "id": f"note_{i:04d}",
                "part": ev["part"],
                "measure": ev["measure"],
                "offset_ql": round(float(ev["offset_ql"]), 4),
                "duration_ql": round(float(ev["duration_ql"]), 4),
                "is_rest": bool(ev["is_rest"]),
                "pitch": ev["pitch"],
                "midi": ev["midi"],
                "ref_start": round(float(ev["ref_start"]), 4),
                "ref_end": round(max(float(ev["ref_end"]), float(ev["ref_start"]) + 0.001), 4),
            }
        )
    return cleaned


def _fill_ref_to_perf(mapping: np.ndarray) -> np.ndarray:
    """Hold edges and interpolate interior gaps; keep the map non-decreasing.

    Unmapped leading/trailing frames (trimmed silence, band edges) must not
    fall back to an identity frame index — that jumps the path and can invert
    later score events.
    """
    out = np.asarray(mapping, dtype=np.float64).copy()
    n = len(out)
    if n == 0:
        return out
    valid = np.flatnonzero(~np.isnan(out))
    if valid.size == 0:
        return np.arange(n, dtype=np.float64)
    first_perf = float(out[valid[0]])
    if valid[0] > 0:
        # Opening rests / trimmed ref silence occupy [0, first sounding].
        out[: valid[0]] = np.linspace(0.0, first_perf, valid[0], endpoint=False)
    out[valid[-1] + 1 :] = out[valid[-1]]
    still_nan = np.isnan(out)
    if np.any(still_nan):
        known = np.flatnonzero(~still_nan)
        out[still_nan] = np.interp(np.flatnonzero(still_nan), known, out[known])
    for i in range(1, n):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def _build_ref_to_perf(wp: np.ndarray, n_ref: int) -> np.ndarray:
    buckets: dict[int, list[int]] = {}
    for ref_i, perf_i in wp:
        ri, pi = int(ref_i), int(perf_i)
        buckets.setdefault(ri, []).append(pi)

    mapping = np.full(n_ref, np.nan, dtype=np.float64)
    for ref_i, perf_list in buckets.items():
        if 0 <= ref_i < n_ref:
            mapping[ref_i] = float(np.median(perf_list))
    return _fill_ref_to_perf(mapping)


def _interp_ref_to_perf(ref_frame: float, ref_to_perf: np.ndarray) -> float:
    n = len(ref_to_perf)
    if n == 0:
        return 0.0
    ref_frame = max(0.0, min(ref_frame, n - 1))
    lo = int(np.floor(ref_frame))
    hi = min(lo + 1, n - 1)
    frac = ref_frame - lo
    v_lo = ref_to_perf[lo]
    v_hi = ref_to_perf[hi]
    if np.isnan(v_lo) and np.isnan(v_hi):
        return float(ref_to_perf[0]) if n and not np.isnan(ref_to_perf[0]) else 0.0
    if np.isnan(v_lo):
        return float(v_hi)
    if np.isnan(v_hi):
        return float(v_lo)
    return float(v_lo * (1 - frac) + v_hi * frac)


def _audio_time_for_ql(offset_ql: float, ql_end: float, audio_dur: float) -> float:
    """Place a score offset on the rendered reference-audio timeline.

    MusicXML tempo and MuseScore/MIDI tempo often disagree on excerpts (a late
    Andante mark back-filled, MIDI defaulting to 120). Quarter-length is the
    shared axis of the score and the render, so we map ql → audio seconds.
    """
    if ql_end <= 1e-9 or audio_dur <= 0:
        return 0.0
    return float(max(0.0, offset_ql) / ql_end) * audio_dur


def _ref_sec_to_perf_sec(ref_sec: float, frame_to_sec: float, ref_to_perf: np.ndarray) -> float:
    ref_frame = ref_sec / frame_to_sec
    perf_frame = _interp_ref_to_perf(ref_frame, ref_to_perf)
    return perf_frame * frame_to_sec


def _residual_for_ref_range(
    ref_start: float,
    ref_end: float,
    frame_to_sec: float,
    wp: np.ndarray,
    residuals: np.ndarray,
) -> float | None:
    ref_lo = int(np.floor(ref_start / frame_to_sec))
    ref_hi = int(np.ceil(ref_end / frame_to_sec))
    vals: list[float] = []
    for k in range(wp.shape[0]):
        ref_i = int(wp[k, 0])
        if ref_lo <= ref_i <= ref_hi:
            vals.append(float(residuals[k]))
    if not vals:
        return None
    return round(float(np.mean(vals)), 4)


def refine_event_onsets(
    events: list[dict[str, Any]],
    audio: np.ndarray,
    sr: int,
    *,
    lookback_sec: float = 0.15,
    max_shift_sec: float = 0.6,
    frame_length: int = 1024,
    hop_length: int = 256,
    rise_db: float = 8.0,
    min_note_sec: float = 0.04,
) -> list[dict[str, Any]]:
    """Snap non-rest perf_start to the first energy rise inside the DTW window.

    DTW often maps score onsets into leading silence; this moves the boundary to
    the acoustic attack so EWMA/staff spans match heard notes more closely.
    Rests are left unchanged.
    """
    if audio is None or len(audio) == 0 or sr <= 0:
        return events

    # RMS envelope for the whole take (cheap, shared across events).
    n = len(audio)
    if n < frame_length:
        return events
    frames = 1 + (n - frame_length) // hop_length
    rms = np.empty(frames, dtype=np.float64)
    for i in range(frames):
        start = i * hop_length
        chunk = audio[start : start + frame_length]
        rms[i] = float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))
    hop_sec = hop_length / float(sr)
    eps = 1e-12

    refined: list[dict[str, Any]] = []
    prev_end = 0.0
    for ev in events:
        out = dict(ev)
        if out.get("is_rest"):
            refined.append(out)
            prev_end = float(out["perf_end"])
            continue

        t0 = float(out["perf_start"])
        t1 = float(out["perf_end"])
        if t1 - t0 < min_note_sec:
            refined.append(out)
            prev_end = t1
            continue

        search_lo = max(0.0, t0 - lookback_sec)
        # Prefer not to steal the previous event's body.
        search_lo = max(search_lo, prev_end)
        search_hi = min(t1 - min_note_sec, t0 + max_shift_sec)
        if search_hi <= search_lo + hop_sec:
            refined.append(out)
            prev_end = t1
            continue

        i0 = max(0, int(search_lo / hop_sec))
        i1 = min(frames, int(np.ceil(search_hi / hop_sec)) + 1)
        if i1 - i0 < 3:
            refined.append(out)
            prev_end = t1
            continue

        window = rms[i0:i1]
        # Silence floor from quieter end of the window (leading silence).
        floor = float(np.percentile(window, 20))
        floor = max(floor, eps)
        thresh = floor * (10.0 ** (rise_db / 20.0))

        onset_idx = None
        for k, val in enumerate(window):
            if val >= thresh:
                # Require a short rising edge vs the previous frame when possible.
                if k == 0 or window[k - 1] < thresh * 0.85:
                    onset_idx = i0 + k
                    break
        if onset_idx is None:
            # Fallback: strongest relative rise in the first half of the window.
            half = max(2, len(window) // 2)
            diffs = np.diff(window[:half], prepend=window[0])
            k = int(np.argmax(diffs))
            if diffs[k] > 0:
                onset_idx = i0 + k

        if onset_idx is not None:
            new_start = onset_idx * hop_sec
            # Only move start later into the note (or slightly earlier via lookback),
            # and keep a usable duration.
            new_start = min(max(new_start, search_lo), search_hi)
            if t1 - new_start >= min_note_sec:
                out["perf_start_dtw"] = round(t0, 4)
                out["perf_start"] = round(new_start, 4)

        refined.append(out)
        prev_end = float(out["perf_end"])
    return refined


def align_score_events(
    score_path: Path,
    wp: np.ndarray,
    n_ref: int,
    frame_to_sec: float,
    residuals: np.ndarray | None = None,
    perf_audio: np.ndarray | None = None,
    sample_rate: int | None = None,
    onset_refine: bool = True,
    onset_lookback_sec: float = 0.15,
    onset_max_shift_sec: float = 0.6,
    onset_rise_db: float = 8.0,
    phrase_min_rest_ql: float = 0.25,
) -> list[dict[str, Any]]:
    """Map MusicXML note/rest events onto performance time via the DTW path.

    Shared by the annotate UI (`build_note_alignment`) and Stage 5 rhythm detection.
    When ``perf_audio`` is provided and ``onset_refine`` is True, non-rest
    ``perf_start`` values are snapped to the first energy rise in the window.
    """
    score_events = _extract_score_events(score_path)
    ref_to_perf = _build_ref_to_perf(wp, n_ref)
    ql_end = max(
        (float(ev["offset_ql"]) + float(ev["duration_ql"])) for ev in score_events
    ) if score_events else 1.0
    audio_dur = max(frame_to_sec, (max(n_ref, 1) - 1) * frame_to_sec)

    aligned_events: list[dict[str, Any]] = []
    for ev in score_events:
        audio_start = _audio_time_for_ql(float(ev["offset_ql"]), ql_end, audio_dur)
        audio_end = _audio_time_for_ql(
            float(ev["offset_ql"]) + float(ev["duration_ql"]), ql_end, audio_dur
        )
        perf_start = _ref_sec_to_perf_sec(audio_start, frame_to_sec, ref_to_perf)
        perf_end = _ref_sec_to_perf_sec(audio_end, frame_to_sec, ref_to_perf)
        if perf_end < perf_start:
            perf_start, perf_end = perf_end, perf_start
        residual = None
        if residuals is not None:
            residual = _residual_for_ref_range(
                audio_start, audio_end, frame_to_sec, wp, residuals
            )
        aligned_events.append(
            {
                **ev,
                "perf_start": round(perf_start, 4),
                "perf_end": round(max(perf_end, perf_start + 0.001), 4),
                "residual_mean": residual,
            }
        )

    _enforce_monotonic_perf_times(aligned_events)
    _redistribute_crushed_phrases(aligned_events, min_rest_ql=phrase_min_rest_ql)
    if perf_audio is not None and sample_rate is not None and sample_rate > 0:
        _snap_phrases_to_voiced(
            aligned_events,
            perf_audio,
            int(sample_rate),
            frame_to_sec,
            min_rest_ql=phrase_min_rest_ql,
        )

    if (
        onset_refine
        and perf_audio is not None
        and sample_rate is not None
        and sample_rate > 0
    ):
        aligned_events = refine_event_onsets(
            aligned_events,
            perf_audio,
            int(sample_rate),
            lookback_sec=onset_lookback_sec,
            max_shift_sec=onset_max_shift_sec,
            rise_db=onset_rise_db,
        )
    return aligned_events


def _phrase_groups(
    events: list[dict[str, Any]], min_rest_ql: float = 0.25
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    for ev in events:
        if ev.get("is_rest") and float(ev.get("duration_ql", 0.0)) >= min_rest_ql:
            if cur:
                groups.append(cur)
                cur = []
            continue
        cur.append(ev)
    if cur:
        groups.append(cur)
    return groups


def _redistribute_crushed_phrases(
    events: list[dict[str, Any]],
    min_rest_ql: float = 0.25,
    crush_ratio: float = 0.3,
) -> list[dict[str, Any]]:
    """Spread notes inside a phrase when DTW piled them onto one frame.

    Phrase start/end stay on the DTW envelope; interior times follow score ql
    so a late arpeggio after a rest is not swallowed by the previous figure.
    """
    for group in _phrase_groups(events, min_rest_ql=min_rest_ql):
        sounding = [ev for ev in group if not ev.get("is_rest")]
        if len(sounding) < 3:
            continue
        n_crushed = 0
        for ev in sounding:
            ref_d = max(1e-4, float(ev["ref_end"]) - float(ev["ref_start"]))
            perf_d = float(ev["perf_end"]) - float(ev["perf_start"])
            if perf_d / ref_d < crush_ratio:
                n_crushed += 1
        if n_crushed < 3 and n_crushed / len(sounding) < 0.25:
            continue
        p0 = min(float(ev["perf_start"]) for ev in group)
        p1 = max(float(ev["perf_end"]) for ev in group)
        total_ql = sum(max(1e-4, float(ev["duration_ql"])) for ev in group)
        if p1 <= p0 + 0.05 or total_ql <= 1e-6:
            continue
        cursor = p0
        for ev in group:
            span = (float(ev["duration_ql"]) / total_ql) * (p1 - p0)
            ev["perf_start"] = round(cursor, 4)
            ev["perf_end"] = round(max(cursor + span, cursor + 0.001), 4)
            cursor = float(ev["perf_end"])
    return events


def _frame_rms(audio: np.ndarray, sr: int, hop_sec: float, frame_sec: float = 0.05) -> np.ndarray:
    hop = max(1, int(round(hop_sec * sr)))
    frame = max(hop * 2, int(round(frame_sec * sr)))
    n = 1 + max(0, (len(audio) - frame) // hop)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    rms = np.empty(n, dtype=np.float64)
    for i in range(n):
        chunk = audio[i * hop : i * hop + frame]
        rms[i] = float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))
    return rms


def _voiced_islands(
    rms: np.ndarray,
    hop_sec: float,
    thresh: float = 0.08,
    min_sec: float = 0.18,
    merge_gap: float = 0.32,
) -> list[tuple[float, float]]:
    if rms.size == 0:
        return []
    peak = float(np.percentile(rms, 95))
    if peak < 1e-10:
        return []
    voiced = rms / peak >= thresh
    raw: list[tuple[float, float]] = []
    i = 0
    while i < len(voiced):
        if not voiced[i]:
            i += 1
            continue
        j = i
        while j < len(voiced) and voiced[j]:
            j += 1
        t0, t1 = i * hop_sec, j * hop_sec
        if t1 - t0 >= min_sec:
            raw.append((t0, t1))
        i = j
    merged: list[tuple[float, float]] = []
    for t0, t1 in raw:
        if merged and t0 - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], t1)
        else:
            merged.append((t0, t1))
    return merged


def _voiced_span_inside(
    p0: float, p1: float, rms: np.ndarray, hop_sec: float, thresh: float
) -> tuple[float, float] | None:
    if rms.size == 0 or p1 <= p0:
        return None
    peak = float(np.percentile(rms, 95))
    if peak < 1e-10:
        return None
    i0 = max(0, int(np.floor(p0 / hop_sec)))
    i1 = min(len(rms), max(i0 + 1, int(np.ceil(p1 / hop_sec))))
    voiced = np.flatnonzero(rms[i0:i1] / peak >= thresh)
    if voiced.size == 0:
        return None
    return (i0 + int(voiced[0])) * hop_sec, (i0 + int(voiced[-1]) + 1) * hop_sec


def _fit_phrase_envelope(
    p0: float,
    p1: float,
    rms: np.ndarray,
    hop_sec: float,
    islands: list[tuple[float, float]],
    thresh: float = 0.08,
) -> tuple[float, float]:
    """Trim a DTW phrase envelope to sounding audio; snap if it sat on a rest."""
    orig = max(p1 - p0, 1e-3)
    overlapping = []
    for a, b in islands:
        ov = min(p1, b) - max(p0, a)
        if ov < 0.12:
            continue
        # Drop an island that mostly belongs to the next phrase.
        if b > p1 and (b - p1) > max(0.4, p1 - a):
            continue
        overlapping.append((a, b))
    if overlapping:
        t0 = max(overlapping[0][0], p0 - 0.12)
        t1 = min(overlapping[-1][1], p1)
        if t1 - t0 >= max(0.28, 0.35 * orig):
            return t0, max(t1, t0 + hop_sec)
        return overlapping[0][0], min(overlapping[0][1], p1 + 0.2)

    inner = _voiced_span_inside(p0, p1, rms, hop_sec, thresh)
    if inner is not None and (inner[1] - inner[0]) >= 0.2:
        return inner

    prev = None
    nxt = None
    for a, b in islands:
        if b <= p1 + 0.25:
            prev = (a, b)
        if nxt is None and a >= p0 - 0.25:
            nxt = (a, b)
    if prev and (p0 - prev[1]) <= (nxt[0] - p1 if nxt else 1e9):
        return prev
    if nxt:
        return nxt
    return p0, p1


def _longest_voiced_run(times: list[float], hop_sec: float, gap: float = 0.35) -> tuple[float, float]:
    if not times:
        return 0.0, hop_sec
    best_a = best_b = times[0]
    run_a = prev = times[0]
    for t in times[1:]:
        if t - prev > gap:
            if prev - run_a >= best_b - best_a:
                best_a, best_b = run_a, prev
            run_a = t
        prev = t
    if prev - run_a >= best_b - best_a:
        best_a, best_b = run_a, prev
    return float(best_a), float(best_b + hop_sec)


def _pack_group_on_voiced(
    group: list[dict[str, Any]],
    t0: float,
    t1: float,
    rms: np.ndarray,
    hop_sec: float,
    thresh: float = 0.08,
) -> None:
    peak = float(np.percentile(rms, 95)) if rms.size else 0.0
    i0 = max(0, int(np.floor(t0 / hop_sec)))
    i1 = min(len(rms), max(i0 + 2, int(np.ceil(t1 / hop_sec))))
    times = [
        k * hop_sec
        for k in range(i0, i1)
        if peak >= 1e-10 and rms[k] / peak >= thresh
    ]
    if len(times) < 4:
        n = max(8, int(round((t1 - t0) / max(hop_sec, 1e-3))))
        times = list(np.linspace(t0, max(t1, t0 + hop_sec), n))
    sounding = [ev for ev in group if not ev.get("is_rest")]
    total_ql = sum(max(1e-4, float(ev["duration_ql"])) for ev in sounding) or 1.0
    cursor_ql = 0.0
    last = t0
    assigned: dict[int, tuple[float, float]] = {}
    n_t = len(times)
    for ev in sounding:
        frac0 = cursor_ql / total_ql
        cursor_ql += max(1e-4, float(ev["duration_ql"]))
        frac1 = min(1.0, cursor_ql / total_ql)
        i_a = int(frac0 * (n_t - 1))
        i_b = int(frac1 * (n_t - 1))
        a, b = _longest_voiced_run(times[i_a : i_b + 1], hop_sec)
        if b <= a:
            b = a + hop_sec
        assigned[id(ev)] = (a, b)
        last = b
    for ev in group:
        if ev.get("is_rest"):
            ev["perf_start"] = round(last, 4)
            ev["perf_end"] = round(last + 0.001, 4)
            continue
        a, b = assigned[id(ev)]
        ev["perf_start"] = round(a, 4)
        ev["perf_end"] = round(max(b, a + 0.001), 4)
        last = float(ev["perf_end"])


def _group_is_crushed(
    group: list[dict[str, Any]], crush_ratio: float = 0.3
) -> bool:
    sounding = [ev for ev in group if not ev.get("is_rest")]
    if len(sounding) < 3:
        return False
    n_crushed = 0
    for ev in sounding:
        ref_d = max(1e-4, float(ev["ref_end"]) - float(ev["ref_start"]))
        perf_d = float(ev["perf_end"]) - float(ev["perf_start"])
        if perf_d / ref_d < crush_ratio:
            n_crushed += 1
    return n_crushed >= 3 or n_crushed / len(sounding) >= 0.25


def _rescale_group_times(
    group: list[dict[str, Any]], p0: float, p1: float, t0: float, t1: float
) -> None:
    """Keep DTW-relative spacing while clipping a phrase onto a shorter island."""
    span = max(p1 - p0, 1e-6)
    scale = (t1 - t0) / span
    for ev in group:
        a = t0 + (float(ev["perf_start"]) - p0) * scale
        b = t0 + (float(ev["perf_end"]) - p0) * scale
        ev["perf_start"] = round(a, 4)
        ev["perf_end"] = round(max(b, a + 0.001), 4)


def _snap_phrases_to_voiced(
    events: list[dict[str, Any]],
    audio: np.ndarray,
    sr: int,
    hop_sec: float,
    thresh: float = 0.08,
    min_rest_ql: float = 0.25,
) -> list[dict[str, Any]]:
    """Keep phrase notes on sounding audio so tails cannot spill into rests.

    Healthy DTW timing is left alone. Uniform ql-packing is only used when a
    phrase sat on silence or was crushed onto a few frames.
    """
    if audio is None or len(audio) == 0 or sr <= 0 or not events:
        return events
    hop_sec = max(float(hop_sec), 1e-4)
    rms = _frame_rms(audio, sr, hop_sec)
    islands = _voiced_islands(rms, hop_sec, thresh=thresh)
    if not islands:
        return events
    for group in _phrase_groups(events, min_rest_ql=min_rest_ql):
        if not group:
            continue
        p0 = min(float(ev["perf_start"]) for ev in group)
        p1 = max(float(ev["perf_end"]) for ev in group)
        t0, t1 = _fit_phrase_envelope(p0, p1, rms, hop_sec, islands, thresh=thresh)
        if t1 <= t0 + 0.05:
            continue
        jumped = t0 > p1 + 0.12 or t1 < p0 - 0.12
        if _group_is_crushed(group) or jumped:
            _pack_group_on_voiced(group, t0, t1, rms, hop_sec, thresh=thresh)
        elif t0 > p0 + 0.10 or t1 < p1 - 0.10:
            _rescale_group_times(group, p0, p1, t0, t1)
    return events


def note_edge_clustering(
    events: list[dict[str, Any]],
    perf_dur: float,
    *,
    edge: float = 0.25,
    min_notes: int = 12,
    cluster_frac: float = 0.5,
    ref_dur: float | None = None,
) -> dict[str, Any]:
    """Detect a DTW pile-up: half the notes in the first or last ``edge`` of the take."""
    times = [float(ev["perf_start"]) for ev in events if not ev.get("is_rest")]
    n = len(times)
    out: dict[str, Any] = {
        "n": n,
        "frac_first": 0.0,
        "frac_last": 0.0,
        "clustered": False,
        "occupied": 0.0,
        "hole": 0.0,
    }
    if n < min_notes or perf_dur <= 0.4:
        return out
    first_cut = float(edge) * float(perf_dur)
    last_cut = (1.0 - float(edge)) * float(perf_dur)
    times_sorted = sorted(times)
    frac_first = sum(t <= first_cut for t in times) / n
    frac_last = sum(t >= last_cut for t in times) / n
    occupied = times_sorted[-1] - times_sorted[0]
    hole = 0.0
    for a, b in zip(times_sorted, times_sorted[1:]):
        hole = max(hole, b - a)
    out["frac_first"] = float(frac_first)
    out["frac_last"] = float(frac_last)
    out["occupied"] = float(occupied)
    out["hole"] = float(hole)
    piled = bool(frac_first >= cluster_frac or frac_last >= cluster_frac)
    # Looping extras make a first-pass mapping look like a start pile on the
    # raw take. That is OK when notes fill an excerpt-length region with no
    # restart hole. A 040-style crush still fails: occupied << take.
    looping = ref_dur is not None and float(perf_dur) > 1.45 * max(float(ref_dur), 1.0)
    incomplete = ref_dur is not None and float(ref_dur) > 1.60 * float(perf_dur)
    if looping and piled:
        expected = min(float(perf_dur), max(float(ref_dur), 1.0) * 1.15)
        filled = occupied >= 0.55 * expected
        bimodal = hole >= 3.0 and hole >= 0.20 * float(perf_dur)
        crushed = occupied < 0.45 * min(float(perf_dur), expected)
        piled = bool(bimodal or crushed or not filled)
    elif incomplete and piled and frac_last >= cluster_frac and frac_first < cluster_frac:
        # Unplayed coda pinned at the end of a short take.
        piled = False
    out["clustered"] = piled
    return out


def _enforce_monotonic_perf_times(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep mapped spans in score order so later bars cannot precede earlier ones."""
    last_end = 0.0
    for ev in events:
        start = float(ev["perf_start"])
        end = float(ev["perf_end"])
        if start < last_end:
            start = last_end
        if end < start + 0.001:
            end = start + 0.001
        ev["perf_start"] = round(start, 4)
        ev["perf_end"] = round(end, 4)
        last_end = end
    return events


def build_note_alignment(sample_dir: Path, logger: logging.Logger | None = None) -> dict[str, Any]:
    from datacreate.audio_utils import load_audio
    from datacreate.config import PipelineConfig
    from datacreate.sample_prep import ensure_full_score

    logger = logger or logging.getLogger(__name__)
    note_first_path = sample_dir / "note_alignment_v2.json"
    if note_first_path.exists():
        payload = read_json(note_first_path)
        engine = str(payload.get("engine") or "")
        if engine not in {"align-note-first", "align-joint"}:
            raise ValueError(f"Unknown note alignment engine in {note_first_path}")
        summary = dict(payload.get("summary") or {})
        summary.setdefault("engine", engine)
        summary.setdefault("event_count", len(payload.get("events") or []))
        summary["alignment_path"] = str(note_first_path)
        transcribed = _normalize_transcribed_notes(payload)
        mapping = [_optional_int(value) for value in (payload.get("note_mapping") or [])]
        summary.setdefault("transcribed_note_count", len(transcribed))
        summary.setdefault(
            "mapped_note_count",
            sum(value is not None for value in mapping),
        )
        logger.info(
            "Loaded %s alignment for %s: %d events, %d transcribed",
            engine,
            sample_dir.name,
            summary["event_count"],
            len(transcribed),
        )
        return {
            "events": list(payload.get("events") or []),
            "transcribed_notes": transcribed,
            "note_mapping": mapping,
            "summary": summary,
        }
    align_path = sample_dir / "alignment.npz"
    if not align_path.exists():
        raise FileNotFoundError(f"Alignment not found: {align_path}")

    score_path = sample_dir / "verified_score.musicxml"
    if not score_path.exists():
        score_path = ensure_full_score(sample_dir)

    data = np.load(align_path)
    wp = data["warping_path"]
    residuals = data["frame_residuals"]
    hop = int(data["hop_length"])
    sr = int(data["sample_rate"])
    frame_to_sec = hop / sr
    n_ref = int(data["ref_features"].shape[1])

    align_cfg = PipelineConfig.load().alignment or {}

    perf_audio = None
    perf_path = sample_dir / "performance_audio.wav"
    if perf_path.exists():
        try:
            perf_audio, _ = load_audio(perf_path, sr, mono=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load performance audio for onset refine: %s", exc)

    aligned_events = align_score_events(
        score_path,
        wp,
        n_ref,
        frame_to_sec,
        residuals=residuals,
        perf_audio=perf_audio,
        sample_rate=sr,
        onset_refine=bool(align_cfg.get("onset_refine", True)),
        onset_lookback_sec=float(align_cfg.get("onset_lookback_sec", 0.15)),
        onset_max_shift_sec=float(align_cfg.get("onset_max_shift_sec", 0.6)),
        onset_rise_db=float(align_cfg.get("onset_rise_db", 8.0)),
        phrase_min_rest_ql=float(align_cfg.get("phrase_min_rest_ql", 0.25)),
    )

    candidate_count = 0
    cand_path = sample_dir / "candidates.json"
    if cand_path.exists():
        candidate_count = len(read_json(cand_path).get("labels", []))

    from datacreate.score_segment import measures_in_written_order

    score_order_ok = True
    try:
        score_order_ok = measures_in_written_order(converter.parse(str(score_path)))
    except Exception:  # noqa: BLE001
        pass
    if not score_order_ok:
        logger.warning(
            "Score measures are out of written order in %s; re-apply the "
            "score segment so reference audio matches the excerpt.",
            score_path.name,
        )

    summary = {
        "warping_path_length": int(wp.shape[0]),
        "ref_frames": n_ref,
        "perf_frames": int(data["perf_features"].shape[1]),
        "hop_length": hop,
        "sample_rate": sr,
        "frame_to_sec": round(frame_to_sec, 6),
        "mean_residual": round(float(np.mean(residuals)), 4),
        "max_residual": round(float(np.max(residuals)), 4),
        "candidate_count": candidate_count,
        "event_count": len(aligned_events),
        "onset_refine": bool(align_cfg.get("onset_refine", True)),
        "score_order_ok": score_order_ok,
    }
    logger.info(
        "Built note alignment for %s: %d events (onset_refine=%s)",
        sample_dir.name,
        len(aligned_events),
        summary["onset_refine"],
    )
    return {
        "events": aligned_events,
        "transcribed_notes": [],
        "note_mapping": [],
        "summary": summary,
    }


def _wav_duration_sec(path: Path) -> float:
    import wave

    with wave.open(str(path), "rb") as fh:
        frames = fh.getnframes()
        rate = fh.getframerate()
    if rate <= 0:
        return 0.0
    return frames / float(rate)


def _annotate_sounding_indices(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sounding = 0
    for ev in events:
        if ev.get("is_rest"):
            ev["sounding_index"] = None
            ev["note_id"] = None
        else:
            ev["sounding_index"] = sounding
            ev["note_id"] = f"note_{sounding:04d}"
            sounding += 1
    return events


def _map_events_from_alignment(
    score_path: Path, align_path: Path, sample_dir: Path | None = None
) -> list[dict[str, Any]]:
    from datacreate.audio_utils import load_audio

    data = np.load(align_path)
    wp = data["warping_path"]
    residuals = data["frame_residuals"]
    hop = int(data["hop_length"])
    sr = int(data["sample_rate"])
    frame_to_sec = hop / sr
    n_ref = int(data["ref_features"].shape[1])
    perf_audio = None
    if sample_dir is not None:
        perf_path = sample_dir / "performance_audio.wav"
        if perf_path.exists():
            try:
                perf_audio, _ = load_audio(perf_path, sr, mono=True)
            except Exception:  # noqa: BLE001
                perf_audio = None
    from datacreate.config import PipelineConfig

    min_rest = float((PipelineConfig.load().alignment or {}).get("phrase_min_rest_ql", 0.25))
    return align_score_events(
        score_path,
        wp,
        n_ref,
        frame_to_sec,
        residuals=residuals,
        perf_audio=perf_audio,
        sample_rate=sr if perf_audio is not None else None,
        onset_refine=False,
        phrase_min_rest_ql=min_rest,
    )


def _attach_performance_times(
    events: list[dict[str, Any]],
    sample_dir: Path,
    score_path: Path,
    logger: logging.Logger,
) -> bool:
    """Keep reference layout; add performance times for waveform labels when possible."""
    align_path = sample_dir / "alignment.npz"
    if align_path.exists():
        try:
            mapped = {
                ev["id"]: ev for ev in _map_events_from_alignment(
                    score_path, align_path, sample_dir
                )
            }
            for ev in events:
                src = mapped.get(ev["id"])
                if not src:
                    continue
                ev["perf_start"] = src.get("perf_start")
                ev["perf_end"] = src.get("perf_end")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not map reference notes onto performance time: %s", exc)

    perf_dur = 0.0
    perf_path = sample_dir / "performance_audio.wav"
    if perf_path.exists():
        try:
            perf_dur = _wav_duration_sec(perf_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read performance duration: %s", exc)
    ref_end = max((float(ev["ref_end"]) for ev in events), default=0.0)
    scale = (perf_dur / ref_end) if ref_end > 1e-6 and perf_dur > 0 else 1.0
    for ev in events:
        start = float(ev["ref_start"]) * scale
        end = float(ev["ref_end"]) * scale
        ev["perf_start"] = round(start, 4)
        ev["perf_end"] = round(max(end, start + 0.001), 4)
    return False


def _attach_note_first_performance_times(
    events: list[dict[str, Any]],
    sample_dir: Path,
    logger: logging.Logger,
) -> bool:
    """Prefer joint-decoder note times over the reference-audio DTW map.

    The annotator displays reference-score events, but label regions belong on
    the performance timeline.  ``note_alignment_v2.json`` is the authoritative
    score-note to transcribed-note alignment, so its event times must win over
    the older reference-audio warping path.
    """

    path = sample_dir / "note_alignment_v2.json"
    if not path.exists():
        return False
    try:
        payload = read_json(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read note-first alignment times: %s", exc)
        return False

    by_note_id: dict[str, list[dict[str, Any]]] = {}
    by_score_index: dict[int, list[dict[str, Any]]] = {}
    for aligned in payload.get("events") or []:
        note_id = aligned.get("note_id")
        if note_id:
            by_note_id.setdefault(str(note_id), []).append(aligned)
        score_index = _optional_int(
            aligned.get("score_index", aligned.get("sounding_index"))
        )
        if score_index is None:
            continue
        start = aligned.get("perf_start")
        end = aligned.get("perf_end")
        if start is None or end is None:
            continue
        by_score_index.setdefault(score_index, []).append(aligned)

    attached = 0
    for ev in events:
        score_index = _optional_int(ev.get("sounding_index"))
        if score_index is None:
            continue
        candidates = (
            by_note_id.get(str(ev.get("note_id")))
            or by_score_index.get(score_index)
            or []
        )
        if not candidates:
            continue
        # A repeated performance can map more than once to one written note.
        # Keep the canonical non-repetition event on the reference staff; the
        # transcription row still shows every performed repetition separately.
        aligned = min(
            candidates,
            key=lambda item: (
                bool(item.get("is_repetition")),
                float(item.get("perf_start") or 0.0),
            ),
        )
        start = float(aligned["perf_start"])
        end = max(start + 0.001, float(aligned["perf_end"]))
        ev["perf_start"] = round(start, 4)
        ev["perf_end"] = round(end, 4)
        ev["performance_timing_source"] = "note_alignment_v2"
        attached += 1
    return attached > 0


def build_score_events(
    sample_dir: Path, logger: logging.Logger | None = None
) -> dict[str, Any]:
    """Reference-score notes/rests for the annotator highlight staff.

    Layout times stay on the clean score (``ref_start`` / ``ref_end``).
    Performance times are attached only so a staff selection can place a
    waveform label.
    """
    from datacreate.sample_prep import ensure_full_score

    logger = logger or logging.getLogger(__name__)
    score_path = sample_dir / "verified_score.musicxml"
    if not score_path.exists():
        score_path = ensure_full_score(sample_dir)

    events = _annotate_sounding_indices(_extract_score_events(score_path))
    legacy_aligned = _attach_performance_times(events, sample_dir, score_path, logger)
    note_first_aligned = _attach_note_first_performance_times(
        events, sample_dir, logger
    )
    aligned = note_first_aligned or legacy_aligned
    timing_source = (
        "note_alignment_v2"
        if note_first_aligned
        else ("alignment.npz" if legacy_aligned else "duration_scale")
    )
    logger.info(
        "Built reference score events for %s: %d events (aligned=%s, source=%s)",
        sample_dir.name,
        len(events),
        aligned,
        timing_source,
    )
    return {
        "events": events,
        "aligned": aligned,
        "layout": "reference",
        "performance_timing_source": timing_source,
        "summary": {
            "event_count": len(events),
            "aligned": aligned,
            "performance_timing_source": timing_source,
        },
    }
