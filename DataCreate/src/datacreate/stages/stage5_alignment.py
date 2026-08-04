from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datacreate.audio_utils import load_audio
import librosa
import numpy as np
from librosa.sequence import dtw

from datacreate.config import PipelineConfig
from datacreate.models import Label
from datacreate.note_alignment import align_score_events

_DEBUG_LOG = Path(__file__).resolve().parents[4] / "debug-e01e0d.log"


def _debug_log(
    location: str,
    message: str,
    data: dict[str, Any],
    hypothesis_id: str,
    run_id: str = "pre-fix",
) -> None:
    # region agent log
    try:
        payload = {
            "sessionId": "e01e0d",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with _DEBUG_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
    except OSError:
        pass
    # endregion


def _zero_norm_columns(feat: np.ndarray, eps: float = 1e-8) -> int:
    return int(np.sum(np.linalg.norm(feat, axis=0) < eps))


def _sanitize_features(feat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Replace silent chroma frames so cosine DTW stays finite."""
    out = feat.copy()
    norms = np.linalg.norm(out, axis=0)
    silent = norms < eps
    if np.any(silent):
        out[:, silent] = 1.0 / np.sqrt(out.shape[0])
    return out


@dataclass
class AlignmentResult:
    candidates: list[Label]
    alignment_path: Path
    warping_path: np.ndarray
    wp: np.ndarray


def extract_features(audio: np.ndarray, sr: int, config: PipelineConfig) -> np.ndarray:
    feature = str(config.alignment.get("feature", "chroma")).lower()
    if feature == "cqt":
        bins = int(config.alignment.get("cqt_bins", 84))
        return librosa.feature.chroma_cqt(y=audio, sr=sr, n_bins=bins)
    return librosa.feature.chroma_cqt(y=audio, sr=sr)


def run_alignment(
    performance_wav: Path,
    reference_wav: Path,
    sample_dir: Path,
    config: PipelineConfig,
    logger: logging.Logger,
) -> AlignmentResult:
    sr = config.sample_rate()
    perf, _ = load_audio(performance_wav, sr, mono=True)
    ref, _ = load_audio(reference_wav, sr, mono=True)

    # region agent log
    _debug_log(
        "stage5_alignment.py:run_alignment",
        "loaded audio",
        {
            "perf_samples": int(len(perf)),
            "ref_samples": int(len(ref)),
            "perf_nan": int(np.isnan(perf).sum()),
            "ref_nan": int(np.isnan(ref).sum()),
            "perf_rms": float(np.sqrt(np.mean(np.square(perf)))),
            "ref_rms": float(np.sqrt(np.mean(np.square(ref)))),
        },
        "A,B",
    )
    # endregion

    perf_feat = extract_features(perf, sr, config)
    ref_feat = extract_features(ref, sr, config)

    # region agent log
    _debug_log(
        "stage5_alignment.py:run_alignment",
        "raw features before sanitize",
        {
            "perf_shape": list(perf_feat.shape),
            "ref_shape": list(ref_feat.shape),
            "perf_zero_norm_cols": _zero_norm_columns(perf_feat),
            "ref_zero_norm_cols": _zero_norm_columns(ref_feat),
            "perf_feat_nan": int(np.isnan(perf_feat).sum()),
            "ref_feat_nan": int(np.isnan(ref_feat).sum()),
        },
        "C,D",
    )
    # endregion

    perf_feat = _sanitize_features(perf_feat)
    ref_feat = _sanitize_features(ref_feat)

    # region agent log
    _debug_log(
        "stage5_alignment.py:run_alignment",
        "features after sanitize",
        {
            "perf_zero_norm_cols": _zero_norm_columns(perf_feat),
            "ref_zero_norm_cols": _zero_norm_columns(ref_feat),
        },
        "D",
        run_id="post-fix",
    )
    # endregion

    band_ratio = float(config.alignment.get("dtw_band_ratio", 0.1))
    cost_matrix, wp = _run_dtw(ref_feat, perf_feat, band_ratio, config, logger)

    hop = int(config.mel.get("hop_length", 512))
    frame_to_sec = hop / sr
    residuals = _frame_residuals(ref_feat, perf_feat, wp)

    score_path = sample_dir / "verified_score.musicxml"
    if not score_path.exists():
        from datacreate.sample_prep import ensure_full_score

        score_path = ensure_full_score(sample_dir)
    aligned_events = align_score_events(
        score_path,
        wp,
        int(ref_feat.shape[1]),
        frame_to_sec,
        residuals=residuals,
        perf_audio=perf,
        sample_rate=sr,
        onset_refine=bool(config.alignment.get("onset_refine", True)),
        onset_lookback_sec=float(config.alignment.get("onset_lookback_sec", 0.15)),
        onset_max_shift_sec=float(config.alignment.get("onset_max_shift_sec", 0.6)),
        onset_rise_db=float(config.alignment.get("onset_rise_db", 8.0)),
    )

    candidates = _detect_candidates(
        ref_feat,
        perf_feat,
        wp,
        frame_to_sec,
        config,
        logger,
        residuals,
        aligned_events,
    )

    alignment_path = sample_dir / "alignment.npz"
    np.savez(
        alignment_path,
        ref_features=ref_feat,
        perf_features=perf_feat,
        warping_path=wp,
        dtw_cost=cost_matrix,
        frame_residuals=residuals,
        hop_length=hop,
        sample_rate=sr,
    )
    logger.info("Alignment saved to %s; %d candidates", alignment_path, len(candidates))
    return AlignmentResult(candidates, alignment_path, wp, wp)


def _run_dtw(
    ref_feat: np.ndarray,
    perf_feat: np.ndarray,
    band_ratio: float,
    config: PipelineConfig,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray]:
    if config.alignment.get("jump_dtw", True):
        logger.info("Running bounded DTW (band_rad=%.3f)", band_ratio)
    try:
        cost_matrix, wp = dtw(
            X=ref_feat,
            Y=perf_feat,
            metric="cosine",
            subseq=False,
            band_rad=band_ratio,
        )
    except Exception as exc:
        # region agent log
        _debug_log(
            "stage5_alignment.py:_run_dtw",
            "dtw failed",
            {
                "error": str(exc),
                "ref_zero_norm_cols": _zero_norm_columns(ref_feat),
                "perf_zero_norm_cols": _zero_norm_columns(perf_feat),
            },
            "D",
        )
        # endregion
        raise
    # region agent log
    _debug_log(
        "stage5_alignment.py:_run_dtw",
        "dtw succeeded",
        {
            "cost_nan": int(np.isnan(cost_matrix).sum()),
            "wp_shape": list(wp.shape),
        },
        "D",
        run_id="post-fix",
    )
    # endregion
    return cost_matrix, wp


def _frame_residuals(ref_feat: np.ndarray, perf_feat: np.ndarray, wp: np.ndarray) -> np.ndarray:
    residuals = []
    for ref_i, perf_i in wp:
        diff = ref_feat[:, int(ref_i)] - perf_feat[:, int(perf_i)]
        residuals.append(float(np.linalg.norm(diff)))
    return np.asarray(residuals, dtype=np.float32)


def _detect_candidates(
    ref_feat: np.ndarray,
    perf_feat: np.ndarray,
    wp: np.ndarray,
    frame_to_sec: float,
    config: PipelineConfig,
    logger: logging.Logger,
    residuals: np.ndarray,
    aligned_events: list[dict[str, Any]] | None = None,
) -> list[Label]:
    cents_tol = float(config.alignment.get("cents_tolerance", 20))
    min_dur = float(config.alignment.get("min_candidate_duration_sec", 0.15))

    candidates: list[Label] = []
    idx = 0

    matched_ref = set()
    matched_perf = set()
    for k in range(wp.shape[0]):
        ref_i, perf_i = int(wp[k, 0]), int(wp[k, 1])
        matched_ref.add(ref_i)
        matched_perf.add(perf_i)

        if k == 0:
            continue
        prev_ref, prev_perf = int(wp[k - 1, 0]), int(wp[k - 1, 1])
        delta_ref = ref_i - prev_ref
        delta_perf = perf_i - prev_perf
        if delta_ref <= 0:
            continue
        t_start = prev_perf * frame_to_sec
        t_end = perf_i * frame_to_sec

        pitch_diff = _pitch_class_mismatch(ref_feat, perf_feat, ref_i, perf_i)
        timing_ms = abs(delta_perf - delta_ref) * frame_to_sec * 1000
        cents = _cents_off(ref_feat, perf_feat, ref_i, perf_i)

        if pitch_diff:
            candidates.append(
                _make_candidate(idx, t_start, t_end, "wrong_note", cents, timing_ms, min_dur)
            )
            idx += 1
        elif cents and abs(cents) > cents_tol:
            candidates.append(
                _make_candidate(idx, t_start, t_end, "intonation_error", cents, timing_ms, min_dur)
            )
            idx += 1

    rhythm_cands, idx = _detect_rhythm_errors(
        aligned_events or [], config, min_dur, idx
    )
    candidates.extend(rhythm_cands)

    for ref_i in range(ref_feat.shape[1]):
        if ref_i not in matched_ref:
            t = ref_i * frame_to_sec
            candidates.append(
                _make_candidate(
                    idx, max(0, t - min_dur / 2), t + min_dur / 2, "missed_note", None, None, min_dur
                )
            )
            idx += 1

    for perf_i in range(perf_feat.shape[1]):
        if perf_i not in matched_perf:
            t = perf_i * frame_to_sec
            candidates.append(
                _make_candidate(
                    idx, max(0, t - min_dur / 2), t + min_dur / 2, "extra_note", None, None, min_dur
                )
            )
            idx += 1

    candidates = _merge_candidates(candidates, min_dur)

    if not candidates:
        logger.info("Zero candidates after detection (%d aligned score events)", len(aligned_events or []))
    else:
        logger.info("Detected %d alignment candidates", len(candidates))
    return candidates


def _event_duration_ratio(ev: dict[str, Any], eps: float = 1e-4) -> float | None:
    ref_dur = float(ev["ref_end"]) - float(ev["ref_start"])
    perf_dur = float(ev["perf_end"]) - float(ev["perf_start"])
    if ref_dur < eps or perf_dur < eps:
        return None
    return perf_dur / ref_dur


def _merge_consecutive_rests(
    events: list[dict[str, Any]], eps: float = 1e-3
) -> list[dict[str, Any]]:
    """Coalesce adjacent rests on the same part into one span for EWMA.

    DTW often mis-splits silence between consecutive rests; treating them as one
    event avoids fake tempo jumps.
    """
    if not events:
        return events

    by_part: dict[int, list[dict[str, Any]]] = {}
    for ev in events:
        by_part.setdefault(int(ev.get("part", 0)), []).append(ev)

    merged: list[dict[str, Any]] = []
    for part in sorted(by_part.keys()):
        part_events = by_part[part]
        i = 0
        while i < len(part_events):
            cur = dict(part_events[i])
            if cur.get("is_rest"):
                j = i + 1
                while j < len(part_events):
                    nxt = part_events[j]
                    if not nxt.get("is_rest"):
                        break
                    if abs(float(nxt["ref_start"]) - float(cur["ref_end"])) > eps:
                        break
                    cur["ref_end"] = float(nxt["ref_end"])
                    cur["perf_end"] = float(nxt["perf_end"])
                    if "duration_ql" in cur and "duration_ql" in nxt:
                        cur["duration_ql"] = float(cur["duration_ql"]) + float(nxt["duration_ql"])
                    j += 1
                merged.append(cur)
                i = j
            else:
                merged.append(cur)
                i += 1
    return merged


def _detect_rhythm_errors(
    aligned_events: list[dict[str, Any]],
    config: PipelineConfig,
    min_dur: float,
    idx: int,
) -> tuple[list[Label], int]:
    """Flag rhythm_error from note/rest duration ratios via EWMA + far-window."""
    alpha = float(config.alignment.get("rhythm_ewma_alpha", 0.3))
    ewma_thresh = float(config.alignment.get("rhythm_ewma_log_threshold", 0.25))
    far_window = int(config.alignment.get("rhythm_far_window", 12))
    far_gap = int(config.alignment.get("rhythm_far_gap", 6))
    far_thresh = float(config.alignment.get("rhythm_far_log_threshold", 0.35))

    candidates: list[Label] = []
    by_part: dict[int, list[dict[str, Any]]] = {}
    for ev in _merge_consecutive_rests(aligned_events or []):
        part = int(ev.get("part", 0))
        by_part.setdefault(part, []).append(ev)

    for part_events in by_part.values():
        ratios: list[float | None] = [_event_duration_ratio(ev) for ev in part_events]
        ewma: float | None = None
        ewma_flagged: set[int] = set()

        for i, (ev, ratio) in enumerate(zip(part_events, ratios)):
            if ratio is None:
                continue
            if ewma is None:
                ewma = ratio
                continue
            log_jump = abs(float(np.log(ratio / ewma)))
            if log_jump > ewma_thresh:
                ewma_flagged.add(i)
                deviation_ms = (float(ev["perf_end"]) - float(ev["perf_start"])) * 1000
                candidates.append(
                    _make_candidate(
                        idx,
                        float(ev["perf_start"]),
                        float(ev["perf_end"]),
                        "rhythm_error",
                        None,
                        deviation_ms,
                        min_dur,
                        comment=f"ewma tempo jump (log={log_jump:.3f})",
                    )
                )
                idx += 1
            ewma = alpha * ratio + (1.0 - alpha) * ewma

        for i, (ev, ratio) in enumerate(zip(part_events, ratios)):
            if ratio is None or i in ewma_flagged:
                continue
            # Lag window: events ending `far_gap` before i, length `far_window`.
            end = i - far_gap
            if end <= 0:
                continue
            start = max(0, end - far_window)
            window_ratios = [r for r in ratios[start:end] if r is not None]
            if len(window_ratios) < max(3, far_window // 3):
                continue
            far_median = float(np.median(window_ratios))
            if far_median <= 0:
                continue
            log_drift = abs(float(np.log(ratio / far_median)))
            if log_drift > far_thresh:
                deviation_ms = (float(ev["perf_end"]) - float(ev["perf_start"])) * 1000
                candidates.append(
                    _make_candidate(
                        idx,
                        float(ev["perf_start"]),
                        float(ev["perf_end"]),
                        "rhythm_error",
                        None,
                        deviation_ms,
                        min_dur,
                        comment=f"far-window tempo drift (log={log_drift:.3f})",
                    )
                )
                idx += 1

    return candidates, idx


def _merge_candidates(candidates: list[Label], min_dur: float) -> list[Label]:
    if not candidates:
        return candidates
    ordered = sorted(candidates, key=lambda c: (c.start_time, c.end_time))
    merged: list[Label] = [ordered[0]]
    for cand in ordered[1:]:
        last = merged[-1]
        if cand.type == last.type and cand.start_time <= last.end_time + 0.05:
            merged[-1] = Label(
                id=last.id,
                source=last.source,
                start_time=last.start_time,
                end_time=max(last.end_time, cand.end_time),
                type=last.type,
                deviation_cents=last.deviation_cents or cand.deviation_cents,
                deviation_ms=last.deviation_ms or cand.deviation_ms,
                comment=last.comment or cand.comment,
            )
        else:
            merged.append(cand)

    out: list[Label] = []
    for i, cand in enumerate(merged):
        start, end = _expand_to_min_duration(cand.start_time, cand.end_time, min_dur)
        out.append(
            Label(
                id=f"cand_{i:03d}",
                source=cand.source,
                start_time=round(start, 4),
                end_time=round(end, 4),
                type=cand.type,
                deviation_cents=cand.deviation_cents,
                deviation_ms=cand.deviation_ms,
                comment=cand.comment,
            )
        )
    return out


def _expand_to_min_duration(start: float, end: float, min_dur: float) -> tuple[float, float]:
    duration = end - start
    if duration >= min_dur:
        return start, end
    center = (start + end) / 2
    half = min_dur / 2
    return max(0.0, center - half), center + half


def _pitch_class_mismatch(
    ref_feat: np.ndarray, perf_feat: np.ndarray, ref_i: int, perf_i: int
) -> bool:
    ref_peak = int(np.argmax(ref_feat[:, ref_i]))
    perf_peak = int(np.argmax(perf_feat[:, perf_i]))
    ref_strength = float(ref_feat[ref_peak, ref_i])
    perf_strength = float(perf_feat[perf_peak, perf_i])
    if ref_strength < 0.2 or perf_strength < 0.2:
        return False
    return ref_peak != perf_peak


def _cents_off(
    ref_feat: np.ndarray, perf_feat: np.ndarray, ref_i: int, perf_i: int
) -> float | None:
    ref_vec = ref_feat[:, ref_i]
    perf_vec = perf_feat[:, perf_i]
    if float(np.max(ref_vec)) < 0.15:
        return None
    dot = float(np.dot(ref_vec, perf_vec))
    norm = float(np.linalg.norm(ref_vec) * np.linalg.norm(perf_vec))
    if norm < 1e-6:
        return None
    similarity = max(-1.0, min(1.0, dot / norm))
    angle = float(np.arccos(similarity))
    return angle * 1200.0 / np.pi


def _make_candidate(
    idx: int,
    start: float,
    end: float,
    label_type: str,
    cents: float | None,
    ms: float | None,
    min_dur: float = 0.15,
    comment: str | None = None,
) -> Label:
    start, end = _expand_to_min_duration(start, end, min_dur)
    return Label(
        id=f"cand_{idx:03d}",
        source="auto",
        start_time=round(start, 4),
        end_time=round(max(end, start + 0.01), 4),
        type=label_type,
        deviation_cents=round(cents, 2) if cents is not None else None,
        deviation_ms=round(ms, 2) if ms is not None else None,
        comment=comment,
    )


def write_candidates(
    candidates: list[Label], sample_dir: Path, schema_version: str
) -> Path:
    from datacreate.utils import write_json

    path = sample_dir / "candidates.json"
    payload = {
        "schema_version": schema_version,
        "labels": [c.model_dump(exclude_none=True) for c in candidates],
    }
    write_json(path, payload)
    return path
