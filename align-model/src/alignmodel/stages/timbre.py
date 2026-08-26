from __future__ import annotations

import numpy as np

from alignmodel.types import PipelineLabel, PipelineState, next_label_id


def run_stage4(state: PipelineState, audio: np.ndarray) -> None:
    cfg = state.config
    sr = state.sr
    _maybe_bad_start(state, audio, sr)
    crop_types = {"wrong_note", "extra_note"}
    for lab in list(state.labels):
        if lab.type not in crop_types:
            continue
        if _is_squeak(audio, sr, lab.start_time, lab.end_time, cfg.squeak_max_sec):
            state.labels.append(
                PipelineLabel(
                    id=next_label_id(state),
                    type="squeak",
                    start_time=lab.start_time,
                    end_time=lab.end_time,
                    comment=f"noisy burst on {lab.type}",
                )
            )
    for pair in state.pairs:
        if pair.kind != "match":
            continue
        dur = pair.perf_end - pair.perf_start
        if dur < 0.25:
            continue
        if _is_bad_timbre(audio, sr, pair.perf_start, pair.perf_end):
            state.labels.append(
                PipelineLabel(
                    id=next_label_id(state),
                    type="bad_timbre",
                    start_time=pair.perf_start,
                    end_time=pair.perf_end,
                    comment="low harmonic ratio on matched sustain",
                    measure_number=pair.measure,
                    note_id=f"note_{pair.score_index:04d}",
                )
            )
    state.stages_run.append(4)


def _crop(audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    i0 = max(0, int(start * sr))
    i1 = min(len(audio), int(end * sr))
    if i1 <= i0 + 16:
        return np.zeros(0, dtype=np.float32)
    return audio[i0:i1]


def _n_fft(n_samples: int) -> int:
    return int(2 ** max(6, min(11, int(np.floor(np.log2(max(n_samples, 64)))))))


def _maybe_bad_start(state: PipelineState, audio: np.ndarray, sr: int) -> None:
    prefix = _crop(audio, sr, 0.0, state.config.bad_start_sec)
    if prefix.size < sr * 0.05:
        return
    rms = float(np.sqrt(np.mean(np.square(prefix)) + 1e-12))
    # Failed attack: quiet then a burst, or very noisy onset with little pitch.
    try:
        import librosa

        n_fft = _n_fft(len(prefix))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=prefix, sr=sr, n_fft=n_fft)))
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=prefix, n_fft=n_fft)))
    except Exception:
        centroid, flatness = 0.0, 0.0
    if rms < 0.004 and flatness > 0.15:
        state.labels.append(
            PipelineLabel(
                id=next_label_id(state),
                type="bad_start",
                start_time=0.0,
                end_time=state.config.bad_start_sec,
                comment="weak noisy attack",
            )
        )
        return
    if centroid > 3500 and flatness > 0.25:
        state.labels.append(
            PipelineLabel(
                id=next_label_id(state),
                type="bad_start",
                start_time=0.0,
                end_time=state.config.bad_start_sec,
                comment="harsh unpitched onset",
            )
        )


def _is_squeak(
    audio: np.ndarray, sr: int, start: float, end: float, max_sec: float
) -> bool:
    dur = end - start
    if dur <= 0.03 or dur > max_sec:
        return False
    crop = _crop(audio, sr, start, end)
    if crop.size < 64:
        return False
    try:
        import librosa

        n_fft = _n_fft(len(crop))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=crop, sr=sr, n_fft=n_fft)))
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=crop, n_fft=n_fft)))
        harm = librosa.effects.harmonic(crop)
        h_ratio = float(np.sqrt(np.mean(np.square(harm)) + 1e-12)) / (
            float(np.sqrt(np.mean(np.square(crop)) + 1e-12)) + 1e-8
        )
    except Exception:
        return False
    return centroid > 2800 and flatness > 0.18 and h_ratio < 0.55


def _is_bad_timbre(audio: np.ndarray, sr: int, start: float, end: float) -> bool:
    crop = _crop(audio, sr, start, end)
    if crop.size < 256:
        return False
    try:
        import librosa

        harm, perc = librosa.effects.hpss(crop)
        h = float(np.sqrt(np.mean(np.square(harm)) + 1e-12))
        p = float(np.sqrt(np.mean(np.square(perc)) + 1e-12))
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=crop, n_fft=_n_fft(len(crop)))))
    except Exception:
        return False
    if h + p < 1e-6:
        return False
    return (h / (h + p)) < 0.45 or flatness > 0.22
