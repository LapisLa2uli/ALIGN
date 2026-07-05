from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import scipy.io.wavfile as wavfile


def _normalize_int_audio(audio: np.ndarray) -> np.ndarray:
    if np.issubdtype(audio.dtype, np.integer):
        max_val = np.iinfo(audio.dtype).max
        return audio.astype(np.float32) / max_val
    return audio.astype(np.float32)


def load_audio(path: Path, sample_rate: int, mono: bool = True) -> tuple[np.ndarray, int]:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        sr, audio = wavfile.read(path)
        audio = _normalize_int_audio(audio)
        if audio.ndim > 1 and mono:
            audio = np.mean(audio, axis=1)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            sr = sample_rate
        return audio, sr
    audio, sr = librosa.load(path, sr=sample_rate, mono=mono)
    return audio, sr


def save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    wavfile.write(path, sample_rate, (clipped * 32767).astype(np.int16))


def is_silent(audio: np.ndarray, threshold: float = 1e-4) -> bool:
    if audio.size == 0:
        return True
    rms = float(np.sqrt(np.mean(np.square(audio))))
    peak = float(np.max(np.abs(audio)))
    return rms < threshold and peak < threshold


def audio_stats(audio: np.ndarray) -> dict[str, float]:
    return {
        "rms": float(np.sqrt(np.mean(np.square(audio)))),
        "peak": float(np.max(np.abs(audio))),
    }
