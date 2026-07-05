from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from librosa.sequence import dtw

from datacreate.config import PipelineConfig
from datacreate.models import Label


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
    perf, _ = librosa.load(performance_wav, sr=sr, mono=True)
    ref, _ = librosa.load(reference_wav, sr=sr, mono=True)

    perf_feat = extract_features(perf, sr, config)
    ref_feat = extract_features(ref, sr, config)

    band_ratio = float(config.alignment.get("dtw_band_ratio", 0.1))
    max_w = max(1, int(band_ratio * max(perf_feat.shape[1], ref_feat.shape[1])))
    wp, cost = _run_dtw(ref_feat, perf_feat, max_w, config, logger)

    hop = int(config.mel.get("hop_length", 512))
    frame_to_sec = hop / sr
    candidates = _detect_candidates(ref_feat, perf_feat, wp, frame_to_sec, config, logger)

    alignment_path = sample_dir / "alignment.npz"
    residuals = _frame_residuals(ref_feat, perf_feat, wp)
    np.savez(
        alignment_path,
        ref_features=ref_feat,
        perf_features=perf_feat,
        warping_path=wp,
        dtw_cost=cost,
        frame_residuals=residuals,
        hop_length=hop,
        sample_rate=sr,
    )
    logger.info("Alignment saved to %s; %d candidates", alignment_path, len(candidates))
    return AlignmentResult(candidates, alignment_path, wp, wp)


def _run_dtw(
    ref_feat: np.ndarray,
    perf_feat: np.ndarray,
    max_w: int,
    config: PipelineConfig,
    logger: logging.Logger,
) -> tuple[np.ndarray, float]:
    if config.alignment.get("jump_dtw", True):
        logger.info("Running bounded DTW (jump-tolerant band=%d frames)", max_w)
    wp, cost = dtw(
        X=ref_feat,
        Y=perf_feat,
        metric="cosine",
        subseq=False,
        band=max_w,
    )
    return wp, float(cost)


def _frame_residuals(ref_feat: np.ndarray, perf_feat: np.ndarray, wp: np.ndarray) -> np.ndarray:
    residuals = []
    for ref_i, perf_i in wp.T:
        diff = ref_feat[:, ref_i] - perf_feat[:, perf_i]
        residuals.append(float(np.linalg.norm(diff)))
    return np.asarray(residuals, dtype=np.float32)


def _detect_candidates(
    ref_feat: np.ndarray,
    perf_feat: np.ndarray,
    wp: np.ndarray,
    frame_to_sec: float,
    config: PipelineConfig,
    logger: logging.Logger,
) -> list[Label]:
    ms_tol = float(config.alignment.get("ms_tolerance", 100))
    cents_tol = float(config.alignment.get("cents_tolerance", 20))
    slope_min = float(config.alignment.get("warping_slope_min", 0.5))
    slope_max = float(config.alignment.get("warping_slope_max", 2.0))

    candidates: list[Label] = []
    idx = 0

    matched_ref = set()
    matched_perf = set()
    for k in range(wp.shape[1]):
        ref_i, perf_i = int(wp[0, k]), int(wp[1, k])
        matched_ref.add(ref_i)
        matched_perf.add(perf_i)

        if k == 0:
            continue
        prev_ref, prev_perf = int(wp[0, k - 1]), int(wp[1, k - 1])
        delta_ref = ref_i - prev_ref
        delta_perf = perf_i - prev_perf
        if delta_ref <= 0:
            continue
        slope = delta_perf / delta_ref
        t_start = prev_perf * frame_to_sec
        t_end = perf_i * frame_to_sec

        pitch_diff = _pitch_class_mismatch(ref_feat, perf_feat, ref_i, perf_i)
        timing_ms = abs(delta_perf - delta_ref) * frame_to_sec * 1000
        cents = _cents_off(ref_feat, perf_feat, ref_i, perf_i)

        if pitch_diff:
            candidates.append(
                _make_candidate(idx, t_start, t_end, "wrong_pitch", cents, timing_ms)
            )
            idx += 1
        elif cents and abs(cents) > cents_tol:
            candidates.append(
                _make_candidate(idx, t_start, t_end, "intonation_error", cents, timing_ms)
            )
            idx += 1
        elif timing_ms > ms_tol:
            candidates.append(
                _make_candidate(idx, t_start, t_end, "rhythm_error", cents, timing_ms)
            )
            idx += 1

        if slope < slope_min or slope > slope_max:
            candidates.append(
                _make_candidate(
                    idx,
                    t_start,
                    t_end,
                    "rhythm_error",
                    cents,
                    timing_ms,
                    comment="structural discontinuity in warping path",
                )
            )
            idx += 1

    for ref_i in range(ref_feat.shape[1]):
        if ref_i not in matched_ref:
            t = ref_i * frame_to_sec
            candidates.append(
                _make_candidate(idx, max(0, t - 0.05), t + 0.05, "missed_note", None, None)
            )
            idx += 1

    for perf_i in range(perf_feat.shape[1]):
        if perf_i not in matched_perf:
            t = perf_i * frame_to_sec
            candidates.append(
                _make_candidate(idx, max(0, t - 0.05), t + 0.05, "extra_note", None, None)
            )
            idx += 1

    logger.info("Detected %d alignment candidates", len(candidates))
    return candidates


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
    comment: str | None = None,
) -> Label:
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
