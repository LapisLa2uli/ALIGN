from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from music21 import chord, converter, note, stream, tempo

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
) -> list[dict[str, Any]]:
    """Map MusicXML note/rest events onto performance time via the DTW path.

    Shared by the annotate UI (`build_note_alignment`) and Stage 5 rhythm detection.
    When ``perf_audio`` is provided and ``onset_refine`` is True, non-rest
    ``perf_start`` values are snapped to the first energy rise in the window.
    """
    score_events = _extract_score_events(score_path)
    ref_to_perf = _build_ref_to_perf(wp, n_ref)

    aligned_events: list[dict[str, Any]] = []
    for ev in score_events:
        perf_start = _ref_sec_to_perf_sec(ev["ref_start"], frame_to_sec, ref_to_perf)
        perf_end = _ref_sec_to_perf_sec(ev["ref_end"], frame_to_sec, ref_to_perf)
        if perf_end < perf_start:
            perf_start, perf_end = perf_end, perf_start
        residual = None
        if residuals is not None:
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


def build_note_alignment(sample_dir: Path, logger: logging.Logger | None = None) -> dict[str, Any]:
    from datacreate.audio_utils import load_audio
    from datacreate.config import PipelineConfig
    from datacreate.sample_prep import ensure_full_score

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
        "onset_refine": bool(align_cfg.get("onset_refine", True)),
    }
    logger.info(
        "Built note alignment for %s: %d events (onset_refine=%s)",
        sample_dir.name,
        len(aligned_events),
        summary["onset_refine"],
    )
    return {"events": aligned_events, "summary": summary}
