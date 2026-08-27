from __future__ import annotations

import numpy as np

from alignmodel.audio import rms_db
from alignmodel.types import PipelineConfig, PipelineState


def detect_boundaries(
    audio: np.ndarray,
    chroma: np.ndarray,
    hop_sec: float,
    duration: float,
    cfg: PipelineConfig,
) -> list[float]:
    cuts: list[float] = []
    cuts.extend(_silence_cuts(audio, hop_sec, cfg))
    cuts.extend(_hold_reattack_cuts(audio, hop_sec, cfg))
    cuts.extend(_copy_cuts(chroma, hop_sec, duration, cfg))
    merged = _merge_cuts(cuts, duration, min_gap=cfg.min_window_sec * 0.5)
    return merged


def _silence_cuts(audio: np.ndarray, hop_sec: float, cfg: PipelineConfig) -> list[float]:
    db = rms_db(audio, hop_length=cfg.hop_length)
    silent = db < cfg.silence_db
    min_frames = max(2, int(round(cfg.min_silence_sec / hop_sec)))
    cuts: list[float] = []
    i = 0
    n = len(silent)
    while i < n:
        if not silent[i]:
            i += 1
            continue
        j = i
        while j < n and silent[j]:
            j += 1
        if j - i >= min_frames:
            mid = (i + j) / 2.0 * hop_sec
            cuts.append(float(mid))
        i = j
    return cuts


def _hold_reattack_cuts(audio: np.ndarray, hop_sec: float, cfg: PipelineConfig) -> list[float]:
    try:
        import librosa

        onset_env = librosa.onset.onset_strength(y=audio, sr=cfg.sample_rate, hop_length=cfg.hop_length)
        rms = librosa.feature.rms(y=audio, hop_length=cfg.hop_length)[0]
    except Exception:
        return []
    n = min(len(onset_env), len(rms))
    if n < 8:
        return []
    onset_env = onset_env[:n]
    rms = rms[:n]
    hold_frames = max(4, int(round(cfg.min_hold_sec / hop_sec)))
    rms_hi = float(np.percentile(rms, 50))
    onset_lo = float(np.percentile(onset_env, 40))
    cuts: list[float] = []
    i = 0
    while i < n - 2:
        if rms[i] < rms_hi:
            i += 1
            continue
        j = i
        while j < n and rms[j] >= rms_hi * 0.7 and onset_env[j] <= onset_lo * 1.4:
            j += 1
        if j - i >= hold_frames:
            k = j
            while k < min(n, j + int(0.4 / hop_sec)):
                if onset_env[k] > onset_lo * 2.0:
                    cuts.append(float(k * hop_sec))
                    break
                k += 1
        i = max(j, i + 1)
    return cuts


def _copy_cuts(
    chroma: np.ndarray, hop_sec: float, duration: float, cfg: PipelineConfig
) -> list[float]:
    """Boundaries where the window after t copies the window before t."""
    peaks: list[tuple[float, float, float]] = []
    for win_sec in (cfg.copy_window_sec, 1.0, 2.2):
        win = max(4, int(round(win_sec / hop_sec)))
        t_frames = chroma.shape[1]
        if t_frames < win * 2 + 2:
            continue
        col = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8)
        step = max(1, win // 4)
        sims: list[tuple[float, float]] = []
        t = win
        while t + win < t_frames:
            a = col[:, t - win : t]
            b = col[:, t : t + win]
            sim = float(np.mean(np.sum(a * b, axis=0)))
            sims.append((t * hop_sec, sim))
            t += step
        if not sims:
            continue
        values = np.array([s for _, s in sims])
        for i, (time, sim) in enumerate(sims):
            if sim < cfg.copy_sim_threshold:
                continue
            left = values[i - 1] if i > 0 else sim - 1
            right = values[i + 1] if i + 1 < len(values) else sim - 1
            if sim >= left and sim >= right:
                if cfg.min_window_sec < time < duration - cfg.min_window_sec:
                    peaks.append((sim, float(time), float(win_sec)))
    peaks.sort(reverse=True)
    seen: list[float] = []
    for _sim, t, _win_sec in peaks:
        if any(abs(t - u) < cfg.min_window_sec * 0.5 for u in seen):
            continue
        seen.append(t)
        if len(seen) >= 4:
            break
    return seen


def _merge_cuts(cuts: list[float], duration: float, min_gap: float) -> list[float]:
    ordered = sorted(t for t in cuts if min_gap < t < duration - min_gap)
    if not ordered:
        return []
    merged = [ordered[0]]
    for t in ordered[1:]:
        if t - merged[-1] >= min_gap:
            merged.append(t)
        else:
            merged[-1] = 0.5 * (merged[-1] + t)
    return merged


def window_times(boundaries: list[float], duration: float, min_window: float) -> list[tuple[float, float]]:
    edges = [0.0] + boundaries + [duration]
    windows: list[tuple[float, float]] = []
    for a, b in zip(edges, edges[1:]):
        if b - a >= min_window:
            windows.append((a, b))
        elif windows:
            prev_a, _ = windows[-1]
            windows[-1] = (prev_a, b)
        else:
            windows.append((a, b))
    if not windows:
        windows = [(0.0, duration)]
    return windows


def apply_boundaries(state: PipelineState, audio: np.ndarray, chroma: np.ndarray) -> None:
    cfg = state.config
    state.boundaries = detect_boundaries(audio, chroma, state.hop_sec, state.duration_sec, cfg)
