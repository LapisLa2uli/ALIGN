from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datacreate.audio_utils import load_audio, sounding_span
import librosa
import numpy as np
from librosa.sequence import dtw

from datacreate.config import PipelineConfig
from datacreate.models import Label
from datacreate.note_alignment import (
    _audio_time_for_ql,
    _build_ref_to_perf,
    _extract_score_events,
    _interp_ref_to_perf,
    align_score_events,
    note_edge_clustering,
)


def _zero_norm_columns(feat: np.ndarray, eps: float = 1e-8) -> int:
    return int(np.sum(np.linalg.norm(feat, axis=0) < eps))


_CHROMA_BINS = 12


def _chroma(feat: np.ndarray) -> np.ndarray:
    if feat.shape[0] > _CHROMA_BINS:
        return feat[:_CHROMA_BINS]
    return feat


def _sanitize_features(feat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Replace silent chroma frames so cosine DTW stays finite.

    Energy (extra rows) is left alone so silence stays distinct from notes.
    """
    out = feat.copy()
    chroma = _chroma(out)
    norms = np.linalg.norm(chroma, axis=0)
    silent = norms < eps
    if np.any(silent):
        chroma[:, silent] = 1.0 / np.sqrt(chroma.shape[0])
    return out


@dataclass
class AlignmentResult:
    candidates: list[Label]
    alignment_path: Path
    warping_path: np.ndarray
    wp: np.ndarray


def extract_features(audio: np.ndarray, sr: int, config: PipelineConfig) -> np.ndarray:
    hop = int(config.mel.get("hop_length", 512))
    feature = str(config.alignment.get("feature", "chroma")).lower()
    if feature == "cqt":
        bins = int(config.alignment.get("cqt_bins", 84))
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, n_bins=bins, hop_length=hop)
    else:
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop)
    midi = _midi_contour_row(audio, sr, hop, int(chroma.shape[1]))
    stacked = np.vstack([chroma, midi]) if midi is not None else chroma
    feat = _with_energy(stacked, audio, sr, hop, config)
    return _gate_midi_by_energy(feat)


def _with_energy(
    chroma: np.ndarray,
    audio: np.ndarray,
    sr: int,
    hop: int,
    config: PipelineConfig,
) -> np.ndarray:
    """Append a loudness row so interior rests do not match sounding notes."""
    weight = float(config.alignment.get("energy_weight", 1.5))
    if weight <= 0 or chroma.size == 0:
        return chroma
    frame_length = min(len(audio), max(hop * 2, 1024))
    rms = librosa.feature.rms(y=audio, hop_length=hop, frame_length=frame_length)[0]
    n = chroma.shape[1]
    if rms.size != n:
        if rms.size == 0:
            energy = np.zeros(n, dtype=np.float64)
        else:
            energy = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, rms.size), rms)
    else:
        energy = rms.astype(np.float64, copy=True)
    peak = float(np.percentile(energy, 95)) if energy.size else 0.0
    if peak < 1e-8:
        normed = np.zeros_like(energy)
    else:
        normed = np.clip(energy / peak, 0.0, 1.0)
    return np.vstack([chroma, weight * normed])


def _midi_contour_row(
    audio: np.ndarray,
    sr: int,
    hop: int,
    n_frames: int,
) -> np.ndarray | None:
    """One row of MIDI/24 so DTW follows melody height, not just chroma class."""
    if n_frames < 4 or audio is None or len(audio) < hop * 4:
        return np.zeros((1, max(n_frames, 1)), dtype=np.float64)
    fmin = float(librosa.note_to_hz("D3"))
    fmax = float(librosa.note_to_hz("C7"))
    try:
        f0 = librosa.yin(
            audio,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            hop_length=hop,
            frame_length=min(2048, max(512, (len(audio) // hop) * hop or 512)),
        )
    except Exception:
        return np.zeros((1, n_frames), dtype=np.float64)
    midi = np.zeros(int(f0.size), dtype=np.float64)
    ok = np.isfinite(f0) & (f0 >= fmin)
    if np.any(ok):
        midi[ok] = librosa.hz_to_midi(f0[ok])
    if midi.size != n_frames:
        if midi.size == 0:
            midi = np.zeros(n_frames, dtype=np.float64)
        else:
            midi = np.interp(
                np.linspace(0.0, 1.0, n_frames),
                np.linspace(0.0, 1.0, midi.size),
                midi,
            )
    return (midi / 24.0).reshape(1, -1)


def _gate_midi_by_energy(feat: np.ndarray, thresh: float = 0.08) -> np.ndarray:
    """Drop F0 on silent frames so rests do not impersonate a pitch."""
    if feat.shape[0] < 14:
        return feat
    energy = np.abs(feat[-1])
    peak = float(np.max(energy)) if energy.size else 0.0
    if peak < 1e-12:
        feat = feat.copy()
        feat[-2] = 0.0
        return feat
    silent = energy / peak < thresh
    if np.any(silent):
        feat = feat.copy()
        feat[-2, silent] = 0.0
    return feat


def _voiced_from_energy(
    energy: np.ndarray,
    *,
    silence_thresh: float = 0.08,
    rise_db: float = 12.0,
) -> np.ndarray:
    """Voiced frames from max-normalized energy, with a noise-floor fallback.

    A fixed 0.08-of-peak cut drops *pp* clarinet notes and can keep noisy rests.
    Anything clearly above the quiet-frame floor is treated as sounding.
    """
    n = int(energy.size)
    if n == 0:
        return np.zeros(0, dtype=bool)
    peak = float(np.max(energy))
    if peak < 1e-12:
        return np.zeros(n, dtype=bool)
    normed = energy / peak
    db = 20.0 * np.log10(np.clip(normed, 1e-8, 1.0))
    floor = float(np.percentile(db, 20))
    return (normed >= silence_thresh) | (db >= floor + rise_db)


def silence_keep_mask(
    energy: np.ndarray,
    hop_sec: float,
    *,
    silence_thresh: float = 0.08,
    keep_silence_sec: float = 0.16,
) -> np.ndarray:
    """Keep voiced frames plus a short collar on each rest; drop rest interiors.

    Long score rests vs a player who barely waits would otherwise consume the
    Sakoe–Chiba band and shift every later note.
    """
    n = int(energy.size)
    if n == 0:
        return np.zeros(0, dtype=bool)
    voiced = _voiced_from_energy(energy, silence_thresh=silence_thresh)
    keep = voiced.copy()
    keep_n = max(2, int(round(keep_silence_sec / max(hop_sec, 1e-6))))
    i = 0
    while i < n:
        if voiced[i]:
            i += 1
            continue
        j = i
        while j < n and not voiced[j]:
            j += 1
        if (j - i) <= 2 * keep_n:
            keep[i:j] = True
        else:
            keep[i : i + keep_n] = True
            keep[j - keep_n : j] = True
        i = j
    keep[0] = True
    keep[-1] = True
    return keep


def _compress_for_dtw(
    feat: np.ndarray,
    hop: int,
    sr: int,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (compressed features, original frame index per compressed frame)."""
    n = feat.shape[1]
    idx = np.arange(n, dtype=np.int32)
    if n < 16 or not config.alignment.get("compress_silence", True):
        return feat, idx
    energy = np.abs(feat[-1]) if feat.shape[0] > _CHROMA_BINS else np.zeros(n)
    if float(np.max(energy)) < 1e-8:
        return feat, idx
    hop_sec = hop / float(sr)
    keep = silence_keep_mask(
        energy,
        hop_sec,
        silence_thresh=float(config.alignment.get("silence_energy_thresh", 0.08)),
        keep_silence_sec=float(config.alignment.get("keep_silence_sec", 0.16)),
    )
    kept = np.flatnonzero(keep)
    # Always compress when a real rest run was removed; do not bail at 85%.
    if kept.size < 8 or kept.size >= n:
        return feat, idx
    return feat[:, kept], kept.astype(np.int32)


def _dtw_cost(
    ref_feat: np.ndarray,
    perf_feat: np.ndarray,
    energy_weight: float,
    midi_weight: float = 1.5,
) -> np.ndarray:
    """Chroma distance, plus energy and melody-height so DTW follows the line."""
    ref_c = _chroma(ref_feat).astype(np.float64, copy=False)
    perf_c = _chroma(perf_feat).astype(np.float64, copy=False)
    ref_n = np.linalg.norm(ref_c, axis=0, keepdims=True)
    perf_n = np.linalg.norm(perf_c, axis=0, keepdims=True)
    ref_u = ref_c / np.clip(ref_n, 1e-8, None)
    perf_u = perf_c / np.clip(perf_n, 1e-8, None)
    cost = np.clip(1.0 - ref_u.T @ perf_u, 0.0, 2.0)
    if energy_weight > 0 and ref_feat.shape[0] > _CHROMA_BINS and perf_feat.shape[0] > _CHROMA_BINS:
        ref_e = np.abs(ref_feat[-1].astype(np.float64))
        perf_e = np.abs(perf_feat[-1].astype(np.float64))
        ref_e = ref_e / max(float(np.max(ref_e)), 1e-8)
        perf_e = perf_e / max(float(np.max(perf_e)), 1e-8)
        cost = cost + float(energy_weight) * np.abs(ref_e[:, None] - perf_e[None, :])
    if midi_weight > 0 and ref_feat.shape[0] >= 14 and perf_feat.shape[0] >= 14:
        ref_m = ref_feat[-2].astype(np.float64)
        perf_m = perf_feat[-2].astype(np.float64)
        both = (ref_m[:, None] > 0.08) & (perf_m[None, :] > 0.08)
        delta = np.abs(ref_m[:, None] - perf_m[None, :])
        cost = cost + float(midi_weight) * np.where(both, delta, 0.0)
    return cost


def _sample_to_frame(sample: int, hop: int, n_frames: int) -> int:
    if n_frames <= 0:
        return 0
    return int(max(0, min(n_frames - 1, sample // max(1, hop))))


def _sounding_feature_span(
    n_samples: int,
    n_frames: int,
    hop: int,
    start_sample: int,
    end_sample: int,
) -> tuple[int, int]:
    """Convert a sounding sample span to a half-open feature-frame slice."""
    if n_frames <= 1:
        return 0, n_frames
    i0 = _sample_to_frame(start_sample, hop, n_frames)
    i1 = _sample_to_frame(max(start_sample, end_sample - 1), hop, n_frames) + 1
    i1 = max(i0 + 1, min(n_frames, i1))
    # Keep the full take when the slice would be a handful of frames.
    if i1 - i0 < min(8, n_frames):
        return 0, n_frames
    if start_sample <= hop and end_sample >= n_samples - hop:
        return 0, n_frames
    return i0, i1


def _dtw_band_ratio(n_ref: int, n_perf: int, configured: float) -> float:
    """Sakoe–Chiba radius. Fast takes stay near the compression diagonal."""
    if n_perf < n_ref * 0.92:
        # A wide band lets the path pile early ref frames onto the opening
        # (half the notes in the first seconds). Keep local wiggle only.
        return float(np.clip(configured, 0.08, 0.18))
    longer = max(n_ref, n_perf, 1)
    mismatch = abs(n_ref - n_perf) / longer
    return float(min(0.5, max(configured, mismatch + 0.05)))


def _score_phrase_spans(
    score_path: Path,
    audio_dur: float,
    min_rest_ql: float = 0.25,
) -> list[tuple[float, float]]:
    """Sounding-phrase spans on the reference-audio timeline, split at long rests."""
    if not score_path.exists() or audio_dur <= 0:
        return []
    events = _extract_score_events(score_path)
    if not events:
        return []
    ql_end = max(float(ev["offset_ql"]) + float(ev["duration_ql"]) for ev in events)
    phrases: list[tuple[float, float]] = []
    start = end = None
    for ev in events:
        long_rest = bool(ev.get("is_rest")) and float(ev["duration_ql"]) >= min_rest_ql
        if long_rest:
            if start is not None and end is not None:
                phrases.append((start, end))
            start = end = None
            continue
        if ev.get("is_rest"):
            continue
        a0 = _audio_time_for_ql(float(ev["offset_ql"]), ql_end, audio_dur)
        a1 = _audio_time_for_ql(
            float(ev["offset_ql"]) + float(ev["duration_ql"]), ql_end, audio_dur
        )
        if start is None:
            start = a0
        end = a1
    if start is not None and end is not None:
        phrases.append((start, end))
    return phrases


def _densify_warping_path(wp: np.ndarray) -> np.ndarray:
    """Fill jumps so compressed/phrase gaps stay matched (no fake missed notes)."""
    if wp is None or len(wp) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    order = np.lexsort((wp[:, 1], wp[:, 0]))
    src = wp[order].astype(np.int32, copy=False)
    out = [src[0].tolist()]
    for i in range(1, len(src)):
        r0, p0 = out[-1]
        r1, p1 = int(src[i, 0]), int(src[i, 1])
        if r1 == r0 and p1 == p0:
            continue
        dr = r1 - r0
        dp = p1 - p0
        steps = max(abs(dr), abs(dp), 1)
        for t in range(1, steps + 1):
            out.append(
                [r0 + int(round(dr * t / steps)), p0 + int(round(dp * t / steps))]
            )
    return np.asarray(out, dtype=np.int32)


def _voiced_bounds(feat: np.ndarray, thresh: float = 0.08) -> tuple[int, int]:
    n = int(feat.shape[1])
    if n == 0 or feat.shape[0] <= _CHROMA_BINS:
        return 0, n
    energy = np.abs(feat[-1])
    peak = float(np.max(energy))
    if peak < 1e-12:
        return 0, n
    voiced = np.flatnonzero(energy / peak >= thresh)
    if voiced.size == 0:
        return 0, n
    return int(voiced[0]), int(voiced[-1]) + 1


def _linear_sounding_path(
    n_ref: int,
    n_perf: int,
    perf_feat: np.ndarray,
    prefix_n: int | None = None,
) -> np.ndarray:
    """Uniform ref→perf map onto the sounding performance (no end-pin pile).

    ``prefix_n`` maps only the played score onto the take; leftover frames pin
    to the last sounding sample so an unplayed coda is not smeared backwards.
    """
    i0, i1 = _voiced_bounds(perf_feat)
    if i1 - i0 < 8:
        i0, i1 = 0, n_perf
    n_ref = max(1, int(n_ref))
    used = n_ref if prefix_n is None else int(max(8, min(n_ref, prefix_n)))
    rs = np.arange(used, dtype=np.int32)
    ps = np.linspace(i0, max(i0, i1 - 1), len(rs))
    path = np.column_stack(
        [rs, np.clip(ps, 0, max(0, n_perf - 1)).astype(np.int32)]
    )
    if used < n_ref:
        last_p = int(np.clip(i1 - 1, 0, max(0, n_perf - 1)))
        tail = np.column_stack(
            [
                np.arange(used, n_ref, dtype=np.int32),
                np.full(n_ref - used, last_p, dtype=np.int32),
            ]
        )
        path = np.vstack([path, tail])
    return path


def _midi_path_error(ref_feat: np.ndarray, perf_feat: np.ndarray, wp: np.ndarray) -> float:
    """Mean |ΔMIDI/24| on voiced frames. High = melody does not follow the path."""
    if wp is None or len(wp) == 0 or ref_feat.shape[0] < 14 or perf_feat.shape[0] < 14:
        return 9.0
    ref_m = ref_feat[-2]
    perf_m = perf_feat[-2]
    n_ref = int(ref_m.size)
    n_perf = int(perf_m.size)
    diffs: list[float] = []
    step = max(1, len(wp) // 400)
    for r, p in wp[::step]:
        ri = int(max(0, min(n_ref - 1, r)))
        pi = int(max(0, min(n_perf - 1, p)))
        rm = float(ref_m[ri])
        pm = float(perf_m[pi])
        if rm > 0.08 and pm > 0.08:
            diffs.append(abs(rm - pm))
    if len(diffs) < 8:
        return 9.0
    return float(np.mean(diffs))


def _best_contour_linear_path(
    ref_feat: np.ndarray,
    perf_feat: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """Linear map whose prefix best matches performance melody height."""
    n_ref = int(ref_feat.shape[1])
    n_perf = int(perf_feat.shape[1])
    best_wp = _linear_sounding_path(n_ref, n_perf, perf_feat)
    best_err = _midi_path_error(ref_feat, perf_feat, best_wp)
    best_prefix = n_ref
    # Incomplete takes (short audio, long excerpt) score better on a prefix.
    for frac in (1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.42, 0.35):
        prefix = int(round(frac * n_ref))
        if prefix < 16:
            continue
        cand = _linear_sounding_path(n_ref, n_perf, perf_feat, prefix_n=prefix)
        err = _midi_path_error(ref_feat, perf_feat, cand)
        if err < best_err - 0.005:
            best_err = err
            best_wp = cand
            best_prefix = prefix
    return best_wp, best_prefix, best_err


def _repair_clustered_path(
    wp: np.ndarray,
    score_path: Path,
    ref_feat: np.ndarray,
    perf_feat: np.ndarray,
    hop: int,
    sr: int,
    logger: logging.Logger,
) -> np.ndarray:
    """If notes pile at either end, replace with a melody-matched linear path.

    Chroma DTW on repeating figures can pin the path at either end. A linear
    map onto sounding audio spreads score order across the take so melody
    height can still match. Skip when the performance is longer than the
    score (practice restarts) — extras must not be smeared into the notes.
    Incomplete excerpts keep a score prefix; leftover frames pin at the end.
    """
    if not score_path.exists():
        return wp
    fts = hop / float(sr)
    n_ref = int(ref_feat.shape[1])
    n_perf = int(perf_feat.shape[1])
    perf_dur = n_perf * fts
    events = align_score_events(
        score_path,
        wp,
        n_ref,
        fts,
        onset_refine=False,
        perf_audio=None,
    )
    stats = note_edge_clustering(events, perf_dur, ref_dur=n_ref * fts)
    pace = _global_pace(ref_feat, perf_feat)
    incomplete = _voiced_length(ref_feat) > 1.65 * max(8, _voiced_length(perf_feat))
    # End-pile of leftover score on a short take is expected; start-pile is not.
    start_pile = bool(stats["clustered"] and stats["frac_first"] >= 0.5)
    end_pile = bool(stats["clustered"] and stats["frac_last"] >= 0.5 and not incomplete)
    if not (start_pile or end_pile):
        return wp
    if pace > 1.20:
        logger.info(
            "Clustered notes (first=%.2f last=%.2f) but pace=%.2f; keep restart path",
            stats["frac_first"],
            stats["frac_last"],
            pace,
        )
        return wp
    dtw_err = _midi_path_error(ref_feat, perf_feat, wp)
    linear, prefix, lin_err = _best_contour_linear_path(ref_feat, perf_feat)
    lin_events = align_score_events(
        score_path,
        linear,
        n_ref,
        fts,
        onset_refine=False,
        perf_audio=None,
    )
    lin_stats = note_edge_clustering(lin_events, perf_dur, ref_dur=n_ref * fts)
    lin_start = bool(lin_stats["clustered"] and lin_stats["frac_first"] >= 0.5)
    if lin_start:
        logger.info(
            "Clustered notes (first=%.2f last=%.2f); linear still piled, keep DTW",
            stats["frac_first"],
            stats["frac_last"],
        )
        return wp
    logger.info(
        "Clustered notes (first=%.2f last=%.2f); linear prefix %d/%d (err %.3f→%.3f)",
        stats["frac_first"],
        stats["frac_last"],
        prefix,
        n_ref,
        dtw_err,
        lin_err,
    )
    return linear


def _run_subseq_dtw(
    query: np.ndarray,
    target: np.ndarray,
    energy_weight: float,
) -> np.ndarray:
    cost = _dtw_cost(query, target, energy_weight)
    _, wp = dtw(C=cost, metric="euclidean", subseq=True)
    return np.asarray(wp, dtype=np.int32)


def _map_times_through_path(
    times: list[float], wp: np.ndarray, n_ref: int, fts: float
) -> list[float]:
    ref_to_perf = _build_ref_to_perf(wp, n_ref)
    out: list[float] = []
    for t in times:
        frame = max(0.0, float(t) / max(fts, 1e-9))
        out.append(_interp_ref_to_perf(frame, ref_to_perf) * fts)
    return out


def _splice_path(
    wp: np.ndarray, segment: np.ndarray, clear_from_ref: int | None = None
) -> np.ndarray:
    if segment.size == 0:
        return wp
    r0 = int(segment[:, 0].min())
    r1 = int(segment[:, 0].max())
    lo = r0 if clear_from_ref is None else min(int(clear_from_ref), r0)
    keep = (wp[:, 0] < lo) | (wp[:, 0] > r1)
    return np.vstack([wp[keep], segment])


def _voiced_length(feat: np.ndarray, thresh: float = 0.08) -> int:
    if feat.size == 0 or feat.shape[0] <= _CHROMA_BINS:
        return int(feat.shape[1]) if feat.size else 0
    energy = np.abs(feat[-1])
    peak = float(np.max(energy))
    if peak < 1e-12:
        return 0
    return int(np.count_nonzero(energy / peak >= thresh))


def _global_pace(ref_feat: np.ndarray, perf_feat: np.ndarray) -> float:
    ref_n = max(8, _voiced_length(ref_feat))
    perf_n = max(8, _voiced_length(perf_feat))
    # Fast takes (half the written duration) must go below 0.85; otherwise
    # phrase 0's window eats the next figure and leftover notes pile up.
    return float(np.clip(perf_n / ref_n, 0.40, 1.55))


def _first_voiced_island(
    feat: np.ndarray,
    start: int,
    end: int,
    hop: int,
    sr: int,
    merge_gap_sec: float = 0.16,
    expected_sec: float | None = None,
    pace: float = 1.0,
) -> tuple[int, int]:
    """First voiced run in [start, end), merging brief articulation gaps.

    When ``expected_sec`` is set, short dips cannot end the island before
    ~88% of the written phrase, and a long slur is capped near the phrase.
    Gaps longer than ``long_merge`` (~0.42s) always end the island so a
    written rest or practice stop cannot swallow the next figure.
    """
    if feat.size == 0 or feat.shape[0] <= _CHROMA_BINS:
        return start, end
    energy = np.abs(feat[-1])
    peak = float(np.max(energy)) if energy.size else 0.0
    if peak < 1e-12:
        return start, end
    voiced = energy / peak >= 0.08
    n = int(voiced.size)
    i0 = max(0, min(start, n))
    i1 = max(i0, min(end, n))
    i = i0
    while i < i1 and not voiced[i]:
        i += 1
    if i >= i1:
        return start, end
    fts = hop / float(sr)
    merge_n = max(2, int(round(merge_gap_sec / max(fts, 1e-6))))
    long_merge = max(merge_n, int(round(0.42 / max(fts, 1e-6))))
    min_keep = 0
    max_keep = i1 - i
    if expected_sec is not None and expected_sec > 0:
        # Shrink min_keep when the take is faster than written. Do not grow it
        # with pace>1: extras inflate pace and would swallow the next figure.
        keep_pace = float(np.clip(pace, 0.40, 1.0))
        min_keep = int(round(0.88 * expected_sec * keep_pace / max(fts, 1e-6)))
        max_keep = int(round(1.15 * expected_sec * max(pace, 0.40) / max(fts, 1e-6))) + 4
    j = i
    while j < i1 and (j - i) < max_keep:
        if voiced[j]:
            j += 1
            continue
        gap = 0
        k = j
        while k < i1 and not voiced[k] and gap < long_merge:
            k += 1
            gap += 1
        if k < i1 and voiced[k] and (
            gap < merge_n or ((j - i) < min_keep and gap < long_merge)
        ):
            j = k
            continue
        break
    return i, max(j, i + 1)


def _match_phrase_in_window(
    ref_feat: np.ndarray,
    perf_feat: np.ndarray,
    t0: float,
    t1: float,
    window_lo: float,
    window_hi: float,
    hop: int,
    sr: int,
    config: PipelineConfig,
    expected_sec: float | None = None,
    pace: float = 1.0,
) -> np.ndarray | None:
    """Bounded DTW of one score phrase into a performance window (global frames)."""
    fts = hop / float(sr)
    n_ref = int(ref_feat.shape[1])
    n_perf = int(perf_feat.shape[1])
    energy_weight = float(config.alignment.get("energy_weight", 1.5))
    midi_weight = float(config.alignment.get("midi_weight", 1.5))
    r0 = int(max(0, min(n_ref - 1, np.floor(t0 / fts))))
    r1 = int(max(r0 + 8, min(n_ref, np.ceil(t1 / fts) + 1)))
    q_lo, q_hi = _voiced_bounds(ref_feat[:, r0:r1])
    rq0, rq1 = r0 + q_lo, r0 + q_hi
    if rq1 - rq0 < 8:
        rq0, rq1 = r0, r1
    y0 = int(max(0, min(n_perf - 1, np.floor(window_lo / fts))))
    y1 = int(max(y0 + 8, min(n_perf, np.ceil(window_hi / fts))))
    qlen = rq1 - rq0
    # Stay on the first voiced island after the previous phrase. Using
    # first-to-last voiced would include the next figure and end-pin into it.
    i0, i1 = _first_voiced_island(
        perf_feat, y0, y1, hop, sr, expected_sec=expected_sec, pace=pace
    )
    exp_sec = expected_sec if expected_sec and expected_sec > 0 else qlen * hop / float(sr)
    keep_pace = float(np.clip(pace, 0.40, 1.0))
    min_island = max(8, int(0.55 * exp_sec * keep_pace / max(fts, 1e-6)))
    island_sec = (i1 - i0) * fts
    gap_after = 0.0
    if i1 < y1 and perf_feat.shape[0] > _CHROMA_BINS:
        energy = np.abs(perf_feat[-1])
        peak = float(np.max(energy)) if energy.size else 0.0
        if peak >= 1e-12:
            voiced = energy / peak >= 0.08
            k = i1
            n_voiced = int(voiced.size)
            while k < y1 and k < n_voiced and not voiced[k]:
                k += 1
            gap_after = (k - i1) * fts
    # Accept a slightly short first island when a clear stop follows it.
    # Compare against paced duration: a 2x take's 6s island is enough for a
    # 12s written phrase, and rejecting it lets DTW pin into the next figure.
    clear_break = gap_after >= 0.45 and island_sec >= 0.45 * exp_sec * keep_pace
    if i1 - i0 >= min_island or clear_break:
        y0, y1 = i0, i1
    elif i0 > y0:
        y0 = i0
    if y1 - y0 < 8 or qlen < 8:
        return None
    query = ref_feat[:, rq0:rq1]
    target = perf_feat[:, y0:y1]
    q_comp, q_idx = _compress_for_dtw(query, hop, sr, config)
    t_comp, t_idx = _compress_for_dtw(target, hop, sr, config)
    # Bounded DTW pins the phrase to the *start* of the remaining audio.
    # Subseq matching is avoided here: a later repeat of the same figure
    # would otherwise steal the path and lag every later phrase.
    try:
        if min(q_comp.shape[1], t_comp.shape[1]) >= 8:
            band = _dtw_band_ratio(int(q_comp.shape[1]), int(t_comp.shape[1]), 0.25)
            _, seg = dtw(
                C=_dtw_cost(q_comp, t_comp, energy_weight, midi_weight=midi_weight),
                metric="euclidean",
                subseq=False,
                band_rad=band,
            )
            seg = np.asarray(seg, dtype=np.int32)
            seg[:, 0] = q_idx[np.clip(seg[:, 0], 0, len(q_idx) - 1)]
            seg[:, 1] = t_idx[np.clip(seg[:, 1], 0, len(t_idx) - 1)]
        else:
            return None
    except Exception:
        return None
    if seg is None or seg.size == 0:
        return None
    seg = seg.copy()
    seg[:, 0] += rq0
    seg[:, 1] += y0
    m0 = int(seg[:, 1].min())
    m1 = int(seg[:, 1].max()) + 1
    ylen = m1 - m0
    if ylen < max(8, int(0.55 * qlen * keep_pace)) or ylen > int(2.4 * qlen) + 8:
        return None
    if rq0 > r0:
        lead = np.column_stack(
            [np.arange(r0, rq0, dtype=np.int32), np.full(rq0 - r0, m0, dtype=np.int32)]
        )
        seg = np.vstack([lead, seg])
    if r1 > int(seg[:, 0].max()) + 1:
        last_p = int(np.clip(m1 - 1, 0, n_perf - 1))
        tail_r0 = int(seg[:, 0].max()) + 1
        tail = np.column_stack(
            [
                np.arange(tail_r0, r1, dtype=np.int32),
                np.full(max(0, r1 - tail_r0), last_p, dtype=np.int32),
            ]
        )
        if len(tail):
            seg = np.vstack([seg, tail])
    return seg


def _silence_gaps(
    feat: np.ndarray,
    start_sec: float,
    end_sec: float,
    hop: int,
    sr: int,
    min_gap_sec: float = 0.35,
) -> list[tuple[float, float]]:
    """Voiced-energy gaps of at least ``min_gap_sec`` in [start_sec, end_sec]."""
    if feat.size == 0 or feat.shape[0] <= _CHROMA_BINS or end_sec <= start_sec:
        return []
    fts = hop / float(sr)
    energy = np.abs(feat[-1])
    peak = float(np.max(energy)) if energy.size else 0.0
    if peak < 1e-12:
        return []
    voiced = energy / peak >= 0.08
    i0 = int(max(0, min(len(voiced), np.floor(start_sec / fts))))
    i1 = int(max(i0, min(len(voiced), np.ceil(end_sec / fts))))
    min_n = max(3, int(round(min_gap_sec / max(fts, 1e-6))))
    gaps: list[tuple[float, float]] = []
    i = i0
    while i < i1:
        if voiced[i]:
            i += 1
            continue
        j = i
        while j < i1 and not voiced[j]:
            j += 1
        if j - i >= min_n:
            gaps.append((i * fts, j * fts))
        i = j
    return gaps


def _skip_restart_extra(
    cursor: float,
    remain_ref: float,
    pace: float,
    perf_feat: np.ndarray,
    hop: int,
    sr: int,
    n_perf: int,
    logger: logging.Logger | None = None,
    current_ref: float = 0.0,
) -> float:
    """Skip a practice restart so leftover score sits on the continuation.

    After a stop, players often replay earlier figures and then continue.
    If leftover audio is much longer than leftover score, jump past the extra
    prefix (after the last gap that still leaves enough audio for the score).
    A trailing breath near the end must not cancel a mid-take restart.
    """
    fts = hop / float(sr)
    perf_end = n_perf * fts
    leftover = perf_end - cursor
    need = max(0.4, remain_ref * pace * 1.15)
    if leftover <= need * 1.40:
        return cursor
    # Continuous in-order figure: the next island starts immediately and is
    # the right length. A looping take's last phrase looks like this — do
    # not jump it onto a later playthrough.
    if current_ref > 0.2:
        i0, i1 = _first_voiced_island(
            perf_feat,
            min(n_perf - 1, max(0, int(cursor / fts))),
            n_perf,
            hop,
            sr,
            expected_sec=current_ref,
            pace=pace,
        )
        delay = i0 * fts - cursor
        island = (i1 - i0) * fts
        if delay < 0.32 and 0.55 * current_ref <= island <= 1.40 * current_ref * max(pace, 0.85):
            return cursor
    # Only jump when the player actually stopped. A long continuous tail is
    # extras after the written ending — do not steal the last figure.
    gaps = _silence_gaps(perf_feat, cursor, perf_end, hop, sr, min_gap_sec=0.35)
    if not gaps:
        return cursor
    min_after = remain_ref * pace * 0.70
    suitable = [g1 for _g0, g1 in gaps if (perf_end - g1) >= min_after]
    if not suitable:
        return cursor
    new_cursor = suitable[-1]
    leftover = perf_end - new_cursor
    later_gaps = [g1 for _g0, g1 in gaps if g1 > new_cursor + 0.05]
    # One long island after the stop is restart+continuation; keep the tail.
    if not later_gaps and leftover > need * 1.40:
        new_cursor = max(new_cursor, perf_end - need)
    skip = new_cursor - cursor
    # A 90s jump is a looping take, not a restart of the last phrase.
    if skip > max(12.0, need * 1.35):
        if logger is not None:
            logger.info(
                "Skip %.2fs exceeds restart cap; keep cursor at %.2fs",
                skip,
                cursor,
            )
        return cursor
    if logger is not None and skip > 0.2:
        logger.info(
            "Skipped %.2fs of restart extra (cursor %.2f -> %.2f, leftover score %.2fs)",
            skip,
            cursor,
            new_cursor,
            remain_ref,
        )
    return new_cursor


def _align_phrases_sequential(
    wp: np.ndarray,
    phrases: list[tuple[float, float]],
    ref_feat: np.ndarray,
    perf_feat: np.ndarray,
    hop: int,
    sr: int,
    config: PipelineConfig,
    logger: logging.Logger,
) -> np.ndarray:
    """Match each rest-separated phrase in score order.

    Global chroma DTW on repeating arpeggios often maps an early figure onto a
    later one. Pinning each phrase to the audio after the previous phrase stops
    that lag from cascading.
    """
    if len(phrases) < 2:
        return wp
    fts = hop / float(sr)
    n_perf = int(perf_feat.shape[1])
    slack = float(config.alignment.get("phrase_slack", 1.12))
    pace = _global_pace(ref_feat, perf_feat)
    cursor = 0.0
    logger.info(
        "Sequential phrase DTW: %d rest-separated figures (pace=%.2f)",
        len(phrases),
        pace,
    )
    for idx, (t0, t1) in enumerate(phrases):
        ref_dur = max(t1 - t0, 1e-3)
        remain = sum(max(p[1] - p[0], 0.0) for p in phrases[idx + 1 :])
        leftover_score = ref_dur + remain
        perf_end = n_perf * fts
        # Only the last figure may jump a practice restart. Jumping earlier
        # phrases onto the final loop piles every later note at the end.
        if idx == len(phrases) - 1:
            cursor = _skip_restart_extra(
                cursor,
                leftover_score,
                pace,
                perf_feat,
                hop,
                sr,
                n_perf,
                logger,
                current_ref=ref_dur,
            )
        window_lo = cursor
        expected = ref_dur * pace * slack
        if idx + 1 < len(phrases):
            reserve = remain * pace * 0.35
            window_hi = min(perf_end - reserve, window_lo + expected + 0.20)
            # Floor is paced: 80% of the slow written duration would eat the
            # next figure on a 6/8 take rendered at quarter=120.
            window_hi = max(window_hi, window_lo + ref_dur * pace * 0.80)
        else:
            # Do not end-pin the last figure across a restart / extra tail.
            window_hi = min(perf_end, window_lo + expected + 0.35)
        window_hi = min(perf_end, max(window_hi, window_lo + 0.3))
        expected_perf = ref_dur * pace
        seg = _match_phrase_in_window(
            ref_feat,
            perf_feat,
            t0,
            t1,
            window_lo,
            window_hi,
            hop,
            sr,
            config,
            expected_sec=ref_dur,
            pace=pace,
        )
        if seg is None and idx + 1 < len(phrases):
            wider = min(perf_end - remain * pace * 0.20, window_lo + expected * 1.25 + 0.45)
            if wider > window_hi + 0.12:
                seg = _match_phrase_in_window(
                    ref_feat,
                    perf_feat,
                    t0,
                    t1,
                    window_lo,
                    wider,
                    hop,
                    sr,
                    config,
                    expected_sec=ref_dur,
                    pace=pace,
                )
        if seg is None:
            nxt0, nxt1 = _first_voiced_island(
                perf_feat,
                min(n_perf - 1, int(cursor / fts) + 2),
                n_perf,
                hop,
                sr,
                expected_sec=ref_dur,
                pace=pace,
            )
            if nxt1 - nxt0 >= 8:
                seg = _match_phrase_in_window(
                    ref_feat,
                    perf_feat,
                    t0,
                    t1,
                    nxt0 * fts,
                    min(perf_end, nxt1 * fts + 0.25),
                    hop,
                    sr,
                    config,
                    expected_sec=ref_dur,
                    pace=pace,
                )
        if seg is None and idx == len(phrases) - 1:
            n_ref = int(ref_feat.shape[1])
            r0 = int(max(0, min(n_ref - 1, np.floor(t0 / fts))))
            r1 = int(max(r0 + 8, min(n_ref, np.ceil(t1 / fts) + 1)))
            y0 = int(max(0, min(n_perf - 1, np.floor(cursor / fts))))
            span = min(n_perf - y0, max(8, int(round(expected / fts)) + 4))
            y1 = y0 + span
            span_sec = (y1 - y0) * fts
            need = ref_dur * pace
            # A 12s leftover phrase must not be linearly packed into 3s of audio.
            if (
                r1 - r0 >= 8
                and y1 - y0 >= 8
                and 0.70 * need <= span_sec <= expected * 1.55 + 0.4
            ):
                rs = np.arange(r0, r1, dtype=np.int32)
                ps = np.linspace(y0, y1 - 1, r1 - r0)
                seg = np.column_stack([rs, np.clip(ps, 0, n_perf - 1).astype(np.int32)])
                logger.info(
                    "Phrase %d linear leftover %.2f–%.2fs -> perf %.2f–%.2fs",
                    idx, t0, t1, y0 * fts, y1 * fts,
                )
        if seg is None:
            n_ref = int(ref_feat.shape[1])
            r0 = int(max(0, min(n_ref - 1, np.floor(t0 / fts))))
            r1 = int(max(r0 + 1, min(n_ref, np.ceil(t1 / fts) + 1)))
            # Unmatched leftover after the take is consumed: pin to the end
            # so global DTW cannot smear later bars onto already-used audio.
            if cursor >= perf_end - 0.25 and r1 > r0:
                last_p = max(0, n_perf - 1)
                pin = np.column_stack(
                    [
                        np.arange(r0, r1, dtype=np.int32),
                        np.full(r1 - r0, last_p, dtype=np.int32),
                    ]
                )
                clear_from = 0 if idx == 0 else int(np.floor(phrases[idx - 1][1] / fts))
                wp = _splice_path(wp, pin, clear_from_ref=clear_from)
                logger.info(
                    "Phrase %d pinned leftover %.2f–%.2fs at end (%.2fs)",
                    idx,
                    t0,
                    t1,
                    perf_end,
                )
            else:
                cursor = min(perf_end, cursor + ref_dur * pace)
                logger.info(
                    "Phrase %d kept walking cursor at %.2fs (no local match)",
                    idx,
                    cursor,
                )
            continue
        m0 = int(seg[:, 1].min())
        m1 = int(seg[:, 1].max()) + 1
        clear_from = 0 if idx == 0 else int(np.floor(phrases[idx - 1][1] / fts))
        wp = _splice_path(wp, seg, clear_from_ref=clear_from)
        _, island_end = _first_voiced_island(
            perf_feat, m0, n_perf, hop, sr, expected_sec=ref_dur, pace=pace
        )
        cursor = max(m1, island_end) * fts
        logger.info(
            "Phrase %d ref %.2f–%.2fs -> perf %.2f–%.2fs",
            idx,
            t0,
            t1,
            m0 * fts,
            m1 * fts,
        )
    return wp


def _repair_crushed_phrases(
    wp: np.ndarray,
    phrases: list[tuple[float, float]],
    ref_feat: np.ndarray,
    perf_feat: np.ndarray,
    hop: int,
    sr: int,
    config: PipelineConfig,
    logger: logging.Logger,
) -> np.ndarray:
    return _align_phrases_sequential(
        wp, phrases, ref_feat, perf_feat, hop, sr, config, logger
    )


def run_alignment(
    performance_wav: Path,
    reference_wav: Path,
    sample_dir: Path,
    config: PipelineConfig,
    logger: logging.Logger,
    *,
    detect_candidates: bool = True,
) -> AlignmentResult:
    sr = config.sample_rate()
    hop = int(config.mel.get("hop_length", 512))
    perf, _ = load_audio(performance_wav, sr, mono=True)
    ref, _ = load_audio(reference_wav, sr, mono=True)

    perf_feat = _sanitize_features(extract_features(perf, sr, config))
    ref_feat = _sanitize_features(extract_features(ref, sr, config))

    ref_i0, ref_i1 = 0, int(ref_feat.shape[1])
    perf_i0, perf_i1 = 0, int(perf_feat.shape[1])
    if config.alignment.get("trim_silence", True):
        top_db = float(config.alignment.get("silence_top_db", 35.0))
        pad_sec = float(config.alignment.get("silence_pad_sec", 0.08))
        ref_s0, ref_s1 = sounding_span(ref, sr, top_db=top_db, pad_sec=pad_sec, hop_length=hop)
        perf_s0, perf_s1 = sounding_span(perf, sr, top_db=top_db, pad_sec=pad_sec, hop_length=hop)
        ref_i0, ref_i1 = _sounding_feature_span(len(ref), ref_feat.shape[1], hop, ref_s0, ref_s1)
        perf_i0, perf_i1 = _sounding_feature_span(
            len(perf), perf_feat.shape[1], hop, perf_s0, perf_s1
        )
        logger.info(
            "DTW sounding windows ref[%d:%d]/%d  perf[%d:%d]/%d  "
            "(ref silence %.2f–%.2fs, perf silence %.2f–%.2fs)",
            ref_i0,
            ref_i1,
            ref_feat.shape[1],
            perf_i0,
            perf_i1,
            perf_feat.shape[1],
            ref_s0 / sr,
            (len(ref) - ref_s1) / sr,
            perf_s0 / sr,
            (len(perf) - perf_s1) / sr,
        )

    score_path = sample_dir / "verified_score.musicxml"
    if not score_path.exists():
        from datacreate.sample_prep import ensure_full_score

        score_path = ensure_full_score(sample_dir)

    ref_slice = ref_feat[:, ref_i0:ref_i1]
    perf_slice = perf_feat[:, perf_i0:perf_i1]
    ref_comp, ref_idx = _compress_for_dtw(ref_slice, hop, sr, config)
    perf_comp, perf_idx = _compress_for_dtw(perf_slice, hop, sr, config)
    if ref_comp.shape[1] < ref_slice.shape[1] or perf_comp.shape[1] < perf_slice.shape[1]:
        logger.info(
            "Compressed rest frames for DTW ref %d→%d  perf %d→%d",
            ref_slice.shape[1],
            ref_comp.shape[1],
            perf_slice.shape[1],
            perf_comp.shape[1],
        )
    band_ratio = _dtw_band_ratio(
        int(ref_comp.shape[1]),
        int(perf_comp.shape[1]),
        float(config.alignment.get("dtw_band_ratio", 0.1)),
    )
    cost_matrix, wp_comp = _run_dtw(ref_comp, perf_comp, band_ratio, config, logger)
    wp = np.empty_like(wp_comp)
    wp[:, 0] = ref_idx[np.clip(wp_comp[:, 0].astype(int), 0, len(ref_idx) - 1)] + ref_i0
    wp[:, 1] = perf_idx[np.clip(wp_comp[:, 1].astype(int), 0, len(perf_idx) - 1)] + perf_i0

    if config.alignment.get("phrase_dtw", True):
        phrases = _score_phrase_spans(
            score_path,
            len(ref) / float(sr),
            float(config.alignment.get("phrase_min_rest_ql", 0.25)),
        )
        if len(phrases) >= 2:
            wp = _align_phrases_sequential(
                wp, phrases, ref_feat, perf_feat, hop, sr, config, logger
            )

    if config.alignment.get("cluster_repair", True):
        wp = _repair_clustered_path(
            wp, score_path, ref_feat, perf_feat, hop, sr, logger
        )

    wp = _densify_warping_path(wp)
    wp[:, 0] = np.clip(wp[:, 0], 0, int(ref_feat.shape[1]) - 1)
    wp[:, 1] = np.clip(wp[:, 1], 0, int(perf_feat.shape[1]) - 1)
    frame_to_sec = hop / sr
    residuals = _frame_residuals(ref_feat, perf_feat, wp)
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
        phrase_min_rest_ql=float(config.alignment.get("phrase_min_rest_ql", 0.25)),
    )
    if config.alignment.get("cluster_repair", True):
        snap_stats = note_edge_clustering(
            aligned_events,
            int(perf_feat.shape[1]) * frame_to_sec,
            ref_dur=int(ref_feat.shape[1]) * frame_to_sec,
        )
        pace = _global_pace(ref_feat, perf_feat)
        incomplete = _voiced_length(ref_feat) > 1.65 * max(
            8, _voiced_length(perf_feat)
        )
        start_pile = snap_stats["clustered"] and snap_stats["frac_first"] >= 0.5
        end_pile = (
            snap_stats["clustered"]
            and snap_stats["frac_last"] >= 0.5
            and not incomplete
        )
        if (start_pile or end_pile) and pace <= 1.20:
            logger.info(
                "Post-snap clustering (first=%.2f last=%.2f); melody-linear path",
                snap_stats["frac_first"],
                snap_stats["frac_last"],
            )
            wp, _, _ = _best_contour_linear_path(ref_feat, perf_feat)
            wp = _densify_warping_path(wp)
            wp[:, 0] = np.clip(wp[:, 0], 0, int(ref_feat.shape[1]) - 1)
            wp[:, 1] = np.clip(wp[:, 1], 0, int(perf_feat.shape[1]) - 1)
            residuals = _frame_residuals(ref_feat, perf_feat, wp)
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
                phrase_min_rest_ql=float(config.alignment.get("phrase_min_rest_ql", 0.25)),
            )

    candidates: list[Label] = []
    if detect_candidates:
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
        silence_frames=np.asarray([ref_i0, ref_i1, perf_i0, perf_i1], dtype=np.int32),
    )
    if detect_candidates:
        logger.info("Alignment saved to %s; %d candidates", alignment_path, len(candidates))
    else:
        logger.info("Alignment saved to %s (DTW only)", alignment_path)
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
        energy_weight = float(config.alignment.get("energy_weight", 1.5))
        midi_weight = float(config.alignment.get("midi_weight", 1.5))
        cost = _dtw_cost(ref_feat, perf_feat, energy_weight, midi_weight=midi_weight)
        cost_matrix, wp = dtw(
            C=cost,
            metric="euclidean",
            subseq=False,
            band_rad=band_ratio,
        )
    except Exception:
        logger.exception(
            "dtw failed (ref_zero_norm=%d perf_zero_norm=%d band=%.3f)",
            _zero_norm_columns(ref_feat),
            _zero_norm_columns(perf_feat),
            band_ratio,
        )
        raise
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

    ref_silent = _voiced_from_energy(
        np.abs(ref_feat[-1]) if ref_feat.shape[0] > _CHROMA_BINS else np.ones(ref_feat.shape[1])
    )
    perf_silent = _voiced_from_energy(
        np.abs(perf_feat[-1]) if perf_feat.shape[0] > _CHROMA_BINS else np.ones(perf_feat.shape[1])
    )
    # `_voiced_from_energy` is True for notes; invert for rest frames.
    ref_silent = ~ref_silent
    perf_silent = ~perf_silent

    for ref_i in range(ref_feat.shape[1]):
        if ref_i not in matched_ref and not bool(ref_silent[ref_i]):
            t = ref_i * frame_to_sec
            candidates.append(
                _make_candidate(
                    idx, max(0, t - min_dur / 2), t + min_dur / 2, "missed_note", None, None, min_dur
                )
            )
            idx += 1

    for perf_i in range(perf_feat.shape[1]):
        if perf_i not in matched_perf and not bool(perf_silent[perf_i]):
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
    ref_ch = _chroma(ref_feat)[:, ref_i]
    perf_ch = _chroma(perf_feat)[:, perf_i]
    ref_peak = int(np.argmax(ref_ch))
    perf_peak = int(np.argmax(perf_ch))
    ref_strength = float(ref_ch[ref_peak])
    perf_strength = float(perf_ch[perf_peak])
    if ref_strength < 0.2 or perf_strength < 0.2:
        return False
    return ref_peak != perf_peak


def _cents_off(
    ref_feat: np.ndarray, perf_feat: np.ndarray, ref_i: int, perf_i: int
) -> float | None:
    ref_vec = _chroma(ref_feat)[:, ref_i]
    perf_vec = _chroma(perf_feat)[:, perf_i]
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
