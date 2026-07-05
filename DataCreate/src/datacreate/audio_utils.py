from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import resample_poly


def _normalize_int_audio(audio: np.ndarray) -> np.ndarray:
    if np.issubdtype(audio.dtype, np.integer):
        max_val = np.iinfo(audio.dtype).max
        return audio.astype(np.float32) / max_val
    return audio.astype(np.float32)


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    gcd = np.gcd(orig_sr, target_sr)
    return resample_poly(audio, target_sr // gcd, orig_sr // gcd).astype(np.float32)


def _load_wav(path: Path, sample_rate: int, mono: bool) -> tuple[np.ndarray, int]:
    sr, audio = wavfile.read(path)
    audio = _normalize_int_audio(audio)
    if audio.ndim > 1 and mono:
        audio = np.mean(audio, axis=1)
    if sr != sample_rate:
        audio = _resample(audio, sr, sample_rate)
        sr = sample_rate
    return audio, sr


def _find_ffmpeg() -> str | None:
    import os
    import sys

    found = shutil.which("ffmpeg")
    if found:
        return found

    for prefix in (os.environ.get("CONDA_PREFIX"), sys.prefix):
        if not prefix:
            continue
        for rel in ("Library/bin/ffmpeg.exe", "Scripts/ffmpeg.exe", "bin/ffmpeg"):
            candidate = Path(prefix) / rel
            if candidate.exists():
                return str(candidate)
    return None


def _ffmpeg_to_wav(path: Path) -> Path | None:
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        return None
    fd, name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    tmp = Path(name)
    result = subprocess.run(
        [ffmpeg, "-y", "-i", str(path), "-acodec", "pcm_s16le", str(tmp)],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        return None
    return tmp


def load_audio(path: Path, sample_rate: int, mono: bool = True) -> tuple[np.ndarray, int]:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return _load_wav(path, sample_rate, mono)

    converted = _ffmpeg_to_wav(path)
    if converted is not None:
        try:
            return _load_wav(converted, sample_rate, mono)
        finally:
            converted.unlink(missing_ok=True)

    raise RuntimeError(
        f"Cannot load {path.suffix} audio without ffmpeg on PATH. "
        "Install ffmpeg (conda install -c conda-forge ffmpeg) or provide WAV/FLAC input."
    )


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
