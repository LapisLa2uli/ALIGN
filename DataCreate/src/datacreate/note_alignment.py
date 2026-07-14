from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from music21 import chord, converter, note, stream, tempo

from datacreate.sample_prep import ensure_full_score
from datacreate.utils import read_json

_PITCH_TYPES = (note.Note, note.Rest, note.Unpitched, chord.Chord)


def _seconds_per_quarter(score) -> float:
    for el in score.flatten().getElementsByClass(tempo.MetronomeMark):
        if el.number:
            return 60.0 / float(el.number)
    return 0.5  # 120 BPM default


def _pitch_label(el) -> str | None:
    if isinstance(el, note.Rest):
        return "rest"
    if isinstance(el, chord.Chord):
        return el.pitchedCommonName or "chord"
    if isinstance(el, note.Note):
        return el.pitch.nameWithOctave
    return None


def _extract_score_events(score_path: Path) -> list[dict[str, Any]]:
    score = converter.parse(str(score_path))
    if not score.parts:
        return []

    sec_per_ql = _seconds_per_quarter(score)
    events: list[dict[str, Any]] = []
    event_idx = 0

    for part_idx, part in enumerate(score.parts):
        flat = part.flatten()
        for el in flat.notesAndRests:
            if not isinstance(el, _PITCH_TYPES):
                continue
            measure = el.getContextByClass(stream.Measure)
            measure_num = int(measure.number) if measure and measure.number is not None else None
            offset_ql = float(el.offset)
            duration_ql = float(el.duration.quarterLength)
            ref_start = offset_ql * sec_per_ql
            ref_end = (offset_ql + duration_ql) * sec_per_ql
            events.append(
                {
                    "id": f"note_{event_idx:04d}",
                    "part": part_idx,
                    "measure": measure_num,
                    "offset_ql": round(offset_ql, 4),
                    "duration_ql": round(duration_ql, 4),
                    "is_rest": isinstance(el, note.Rest),
                    "pitch": _pitch_label(el),
                    "ref_start": round(ref_start, 4),
                    "ref_end": round(ref_end, 4),
                }
            )
            event_idx += 1

    return events


def _build_ref_to_perf(wp: np.ndarray, n_ref: int) -> np.ndarray:
    buckets: dict[int, list[int]] = {}
    for ref_i, perf_i in wp:
        ri, pi = int(ref_i), int(perf_i)
        buckets.setdefault(ri, []).append(pi)

    mapping = np.full(n_ref, np.nan, dtype=np.float64)
    for ref_i, perf_list in buckets.items():
        if 0 <= ref_i < n_ref:
            mapping[ref_i] = float(np.median(perf_list))
    return mapping


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
        return ref_frame
    if np.isnan(v_lo):
        return float(v_hi)
    if np.isnan(v_hi):
        return float(v_lo)
    return float(v_lo * (1 - frac) + v_hi * frac)


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


def build_note_alignment(sample_dir: Path, logger: logging.Logger | None = None) -> dict[str, Any]:
    logger = logger or logging.getLogger(__name__)
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

    score_events = _extract_score_events(score_path)
    ref_to_perf = _build_ref_to_perf(wp, n_ref)

    aligned_events: list[dict[str, Any]] = []
    for ev in score_events:
        perf_start = _ref_sec_to_perf_sec(ev["ref_start"], frame_to_sec, ref_to_perf)
        perf_end = _ref_sec_to_perf_sec(ev["ref_end"], frame_to_sec, ref_to_perf)
        if perf_end < perf_start:
            perf_start, perf_end = perf_end, perf_start
        residual = _residual_for_ref_range(
            ev["ref_start"], ev["ref_end"], frame_to_sec, wp, residuals
        )
        aligned_events.append(
            {
                **ev,
                "perf_start": round(perf_start, 4),
                "perf_end": round(max(perf_end, perf_start + 0.001), 4),
                "residual_mean": residual,
            }
        )

    candidate_count = 0
    cand_path = sample_dir / "candidates.json"
    if cand_path.exists():
        candidate_count = len(read_json(cand_path).get("labels", []))

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
    }
    logger.info(
        "Built note alignment for %s: %d events",
        sample_dir.name,
        len(aligned_events),
    )
    return {"events": aligned_events, "summary": summary}
