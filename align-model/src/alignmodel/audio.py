from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
import torch
from librosa.sequence import dtw

from alignmodel.device import resolve_device
from alignmodel.types import GraphNote

_AUDIO_DEVICE = torch.device("cpu")


def set_audio_device(name: str) -> torch.device:
    global _AUDIO_DEVICE
    _AUDIO_DEVICE = resolve_device(name)
    return _AUDIO_DEVICE


def audio_device() -> torch.device:
    return _AUDIO_DEVICE


def load_mono(path: Path, sr: int) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=sr, mono=True)
    return audio.astype(np.float32)


@lru_cache(maxsize=8)
def _chroma_filterbank(sr: int, n_fft: int) -> np.ndarray:
    return librosa.filters.chroma(sr=sr, n_fft=n_fft, n_chroma=12).astype(np.float32)


def extract_chroma(
    audio: np.ndarray, sr: int, hop_length: int, n_fft: int = 2048
) -> tuple[np.ndarray, float]:
    device = _AUDIO_DEVICE
    hop_sec = hop_length / float(sr)
    if device.type == "cpu":
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length)
        return _sanitize_chroma(chroma), hop_sec

    wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).to(device)
    window = torch.hann_window(n_fft, device=device)
    spec = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
        center=True,
    )
    mag = spec.abs()
    fb = torch.from_numpy(_chroma_filterbank(sr, n_fft)).to(device=device, dtype=mag.dtype)
    chroma = fb @ mag
    chroma = _sanitize_chroma_torch(chroma)
    return chroma.detach().cpu().numpy().astype(np.float64), hop_sec


def _sanitize_chroma_torch(feat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norms = torch.linalg.norm(feat, dim=0)
    out = feat / norms.clamp_min(eps)
    silent = norms < eps
    if torch.any(silent):
        out = out.clone()
        out[:, silent] = 1.0 / (feat.size(0) ** 0.5)
    return out


def _sanitize_chroma(feat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    out = feat.astype(np.float64, copy=True)
    norms = np.linalg.norm(out, axis=0)
    silent = norms < eps
    if np.any(silent):
        out[:, silent] = 1.0 / np.sqrt(out.shape[0])
    return out


def rms_db(audio: np.ndarray, hop_length: int, frame_length: int = 2048) -> np.ndarray:
    rms = librosa.feature.rms(y=audio, hop_length=hop_length, frame_length=frame_length)[0]
    return librosa.amplitude_to_db(rms, ref=np.max)


def score_chroma_template(notes: list[GraphNote], hop_sec: float) -> np.ndarray:
    if not notes:
        return np.ones((12, 4), dtype=np.float64) / np.sqrt(12.0)
    t0 = notes[0].start
    t1 = max(n.end for n in notes)
    n_frames = max(4, int(round((t1 - t0) / hop_sec)))
    chroma = np.zeros((12, n_frames), dtype=np.float64)
    for note in notes:
        if note.is_rest:
            continue
        a = int(round((note.start - t0) / hop_sec))
        b = max(a + 1, int(round((note.end - t0) / hop_sec)))
        a = max(0, min(n_frames - 1, a))
        b = max(a + 1, min(n_frames, b))
        chroma[int(note.pitch) % 12, a:b] = 1.0
    return _sanitize_chroma(chroma)


def _cosine_cost_matrix(ref: np.ndarray, perf: np.ndarray) -> np.ndarray:
    device = _AUDIO_DEVICE
    r = torch.as_tensor(np.ascontiguousarray(ref, dtype=np.float32), device=device)
    p = torch.as_tensor(np.ascontiguousarray(perf, dtype=np.float32), device=device)
    r = r / r.norm(dim=0, keepdim=True).clamp_min(1e-8)
    p = p / p.norm(dim=0, keepdim=True).clamp_min(1e-8)
    sim = r.T @ p
    cost = (1.0 - sim).clamp_min(0.0)
    return cost.detach().cpu().numpy().astype(np.float64)


def dtw_normalized_cost(ref: np.ndarray, perf: np.ndarray) -> tuple[float, np.ndarray]:
    """Full cosine DTW. Returns (normalized cost, warping path [N,2] as ref,perf)."""
    if ref.size == 0 or perf.size == 0:
        return 1e6, np.zeros((0, 2), dtype=np.int32)
    ref = np.atleast_2d(ref)
    perf = np.atleast_2d(perf)
    if ref.shape[1] < 2:
        ref = np.repeat(ref, 2, axis=1)
    if perf.shape[1] < 2:
        perf = np.repeat(perf, 2, axis=1)
    try:
        if _AUDIO_DEVICE.type != "cpu":
            cost_mat = _cosine_cost_matrix(ref, perf)
            cost, wp = dtw(C=cost_mat, subseq=False)
        else:
            cost, wp = dtw(X=ref, Y=perf, metric="cosine", subseq=False)
    except Exception:
        return 1e6, np.zeros((0, 2), dtype=np.int32)
    wp = np.asarray(wp, dtype=np.int32)
    total = float(cost[-1, -1])
    norm = total / max(ref.shape[1] + perf.shape[1], 1)
    return norm, wp


def chroma_slice(chroma: np.ndarray, start_sec: float, end_sec: float, hop_sec: float) -> np.ndarray:
    i0 = max(0, int(round(start_sec / hop_sec)))
    i1 = max(i0 + 2, int(round(end_sec / hop_sec)))
    i1 = min(chroma.shape[1], i1)
    if i1 - i0 < 2:
        i0 = max(0, chroma.shape[1] - 2)
        i1 = chroma.shape[1]
    return chroma[:, i0:i1]


def cents_off(ref_vec: np.ndarray, perf_vec: np.ndarray) -> float | None:
    """Estimate signed sub-semitone offset around the reference pitch class.

    Chroma has one bin per semitone, so this uses the left/right energy
    imbalance around the expected bin. It is intentionally bounded to one
    semitone; pitch-class substitutions are handled separately.
    """
    ref_vec = np.asarray(ref_vec, dtype=np.float64)
    perf_vec = np.asarray(perf_vec, dtype=np.float64)
    if ref_vec.size != 12 or perf_vec.size != 12 or float(np.max(ref_vec)) < 0.15:
        return None
    center = int(np.argmax(ref_vec))

    def local_offset(vec: np.ndarray) -> float:
        left = max(0.0, float(vec[(center - 1) % 12]))
        middle = max(0.0, float(vec[center]))
        right = max(0.0, float(vec[(center + 1) % 12]))
        total = left + middle + right
        if total < 1e-8:
            return 0.0
        return 200.0 * (right - left) / total

    return float(np.clip(local_offset(perf_vec) - local_offset(ref_vec), -100.0, 100.0))


def pitch_class_mismatch(
    ref_vec: np.ndarray, perf_vec: np.ndarray, min_peak: float
) -> bool:
    ref_peak = int(np.argmax(ref_vec))
    perf_peak = int(np.argmax(perf_vec))
    if float(ref_vec[ref_peak]) < min_peak or float(perf_vec[perf_peak]) < min_peak:
        return False
    return ref_peak != perf_peak


def mean_chroma(chroma: np.ndarray, start_sec: float, end_sec: float, hop_sec: float) -> np.ndarray:
    sl = chroma_slice(chroma, start_sec, end_sec, hop_sec)
    return np.mean(sl, axis=1)


def spectral_midi_frames(
    audio: np.ndarray,
    sr: int,
    *,
    hop_length: int = 512,
    n_fft: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unrounded monophonic MIDI estimates and spectral strengths."""
    if audio.size < n_fft or sr <= 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    pitches, magnitudes = librosa.piptrack(
        y=np.asarray(audio, dtype=np.float32),
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=50.0,
        fmax=4200.0,
    )
    best = magnitudes.argmax(axis=0)
    columns = np.arange(pitches.shape[1])
    hz = pitches[best, columns]
    midi = librosa.hz_to_midi(np.maximum(hz, 1.0)).astype(np.float32)
    strength = magnitudes[best, columns].astype(np.float32)
    midi[hz <= 0.0] = np.nan
    return midi, strength
