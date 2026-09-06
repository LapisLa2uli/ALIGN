from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from datacreate.audio_utils import load_audio, save_wav

WAV_NAMES = ("performance_audio.wav", "reference_audio.wav")


def pitch_shift_duration_preserving(
    audio: np.ndarray,
    sample_rate: int,
    semitones: float,
    *,
    frame: int = 2048,
    hop: int = 512,
) -> np.ndarray:
    """Shift pitch by a ratio, then overlap-add back to the original length.

    Two semitones is a ~12% stretch, which OLA handles without changing label times.
    """
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not semitones:
        return y
    import soxr

    factor = 2.0 ** (-float(semitones) / 12.0)
    pitched = np.asarray(
        soxr.resample(y, sample_rate, sample_rate * factor),
        dtype=np.float32,
    )
    return _ola_to_length(pitched, y.size, frame=frame, hop=hop)


def _ola_to_length(
    audio: np.ndarray,
    out_len: int,
    *,
    frame: int = 2048,
    hop: int = 512,
) -> np.ndarray:
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    n = int(y.size)
    if out_len <= 0:
        return np.zeros(0, dtype=np.float32)
    if n == out_len:
        return y
    if n == 0:
        return np.zeros(out_len, dtype=np.float32)
    frame = min(int(frame), n, out_len)
    hop = max(1, min(int(hop), frame // 2 or 1))
    window = np.hanning(frame).astype(np.float32)
    out = np.zeros(out_len + frame, dtype=np.float32)
    weights = np.zeros(out_len + frame, dtype=np.float32)
    ratio = n / float(out_len)
    pos_out = 0
    while pos_out < out_len:
        pos_in = int(round(pos_out * ratio))
        chunk = np.zeros(frame, dtype=np.float32)
        if pos_in < n:
            take = min(frame, n - pos_in)
            chunk[:take] = y[pos_in : pos_in + take]
        sl = slice(pos_out, pos_out + frame)
        out[sl] += chunk * window
        weights[sl] += window
        pos_out += hop
    return (out / np.maximum(weights, 1e-6))[:out_len]


def _rewrite_mel(audio: np.ndarray, sample_rate: int, dest: Path) -> None:
    import librosa

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        fmin=30.0,
    )
    np.save(dest, librosa.power_to_db(mel, ref=np.max).astype(np.float32))


def transpose_bundle(
    sample_dir: Path,
    semitones: int = -2,
    *,
    force: bool = False,
    sample_rate: int = 22050,
) -> str:
    sample_dir = Path(sample_dir)
    perf = sample_dir / "performance_audio.wav"
    ref = sample_dir / "reference_audio.wav"
    if not perf.exists() or not ref.exists():
        return "skip_missing"
    meta_path = sample_dir / "metadata.json"
    meta: dict = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not force and int(meta.get("sounding_transpose") or 0) == int(semitones):
            return "skip_done"

    for name in WAV_NAMES:
        path = sample_dir / name
        audio, sr = load_audio(path, sample_rate, mono=True)
        shifted = pitch_shift_duration_preserving(audio, sr, semitones)
        save_wav(path, shifted, sr)
        try:
            _rewrite_mel(shifted, sr, sample_dir / f"{name.split('_')[0]}_mel.npy")
        except Exception:
            return "failed_mel"
    meta["sounding_transpose"] = int(semitones)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return "converted"


def _one(payload: tuple[str, int, bool, int]) -> str:
    sample, semis, force, sr = payload
    try:
        return transpose_bundle(Path(sample), semis, force=force, sample_rate=sr)
    except Exception as exc:
        print(f"failed {sample}: {type(exc).__name__}: {exc}", flush=True)
        return "failed"


def discover_bundles(root: Path) -> list[Path]:
    out: list[Path] = []
    if not root.is_dir():
        return out
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "performance_audio.wav").exists() and (path / "reference_audio.wav").exists():
            out.append(path)
    return out


def transpose_root(
    root: Path,
    semitones: int = -2,
    *,
    force: bool = False,
    workers: int = 8,
    sample_rate: int = 22050,
) -> dict[str, int]:
    dirs = discover_bundles(root)
    counts = {
        "converted": 0,
        "skip_done": 0,
        "skip_missing": 0,
        "failed": 0,
        "failed_mel": 0,
        "n_bundles": len(dirs),
    }
    jobs = [(str(p), int(semitones), bool(force), int(sample_rate)) for p in dirs]
    workers = max(1, int(workers))
    if workers == 1:
        statuses = [_one(job) for job in jobs]
    else:
        statuses = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_one, job) for job in jobs]
            for i, fut in enumerate(as_completed(futs), start=1):
                statuses.append(fut.result())
                if i % 100 == 0 or i == len(futs):
                    print(f"  {root}: {i}/{len(futs)}", flush=True)
    for status in statuses:
        counts[status] = counts.get(status, 0) + 1
    return counts
