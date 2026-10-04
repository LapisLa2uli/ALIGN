"""Make clean synthetic performance audio sound like a phone recording in a room.

The chain models what DataCreate recordings went through: player breath noise,
room reverb, a phone microphone's frequency response, background noise (room
tone, mains hum, occasional thumps), slow level drift, phone AGC, and a 48 kHz
AAC voice-memo encode. Every stage is time-aligned with its input (no net
delay), so score-note timestamps in labels.json stay valid.

Parameters come from the ``degrade`` block of a synth-pipeline config; ranges
``[lo, hi]`` are sampled uniformly per clip.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

DEGRADE_MARK = "realistic_v1"


def _u(rng: np.random.Generator, spec, default: float | None = None) -> float:
    if spec is None:
        if default is None:
            raise ValueError("missing range")
        return float(default)
    if isinstance(spec, (int, float)):
        return float(spec)
    lo, hi = float(spec[0]), float(spec[1])
    return float(rng.uniform(lo, hi))


def _hit(rng: np.random.Generator, stage: dict | None) -> bool:
    if not stage:
        return False
    return bool(rng.random() < float(stage.get("prob", 1.0)))


def _curve(freqs: np.ndarray, anchors_hz, gains_db) -> np.ndarray:
    """Piecewise-linear gain (dB) in log-frequency through the anchor points."""
    a = np.log(np.maximum(np.asarray(anchors_hz, dtype=np.float64), 1.0))
    g = np.asarray(gains_db, dtype=np.float64)
    return np.interp(np.log(np.maximum(freqs, 1.0)), a, g, left=g[0], right=g[-1])


def _jittered(rng: np.random.Generator, gains_db, jitter_db: float, tilt_db: float, anchors_hz):
    g = np.asarray(gains_db, dtype=np.float64).copy()
    if jitter_db > 0:
        raw = rng.normal(0.0, jitter_db, size=g.size)
        g += np.convolve(raw, [0.25, 0.5, 0.25], mode="same")
    if tilt_db:
        octaves = np.log2(np.asarray(anchors_hz, dtype=np.float64) / 1000.0)
        g += rng.uniform(-tilt_db, tilt_db) * octaves
    return g


def _apply_spectral_gain(x: np.ndarray, sr: int, anchors_hz, gains_db) -> np.ndarray:
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    spec *= 10.0 ** (_curve(freqs, anchors_hz, gains_db) / 20.0)
    return np.fft.irfft(spec, n=x.size)


def _colored_noise(rng: np.random.Generator, n: int, sr: int, anchors_hz, gains_db) -> np.ndarray:
    white = rng.standard_normal(n)
    out = _apply_spectral_gain(white, sr, anchors_hz, gains_db)
    return out / (np.sqrt(np.mean(out**2)) + 1e-12)


def _smooth_random(rng: np.random.Generator, n: int, sr: int, rate_hz: float) -> np.ndarray:
    """Zero-mean, unit-peak random curve that changes about ``rate_hz`` times per second."""
    k = max(2, int(np.ceil(n / sr * rate_hz)) + 2)
    pts = rng.uniform(-1.0, 1.0, size=k)
    t = np.linspace(0.0, k - 1.0, n)
    return np.interp(t, np.arange(k), pts)


def active_rms(x: np.ndarray, sr: int) -> float:
    hop = max(1, sr // 50)
    n = x.size // hop
    if n < 1:
        return float(np.sqrt(np.mean(x**2)) + 1e-12)
    frames = x[: n * hop].reshape(n, hop)
    e = np.mean(frames**2, axis=1) + 1e-12
    db = 10.0 * np.log10(e)
    keep = db > np.percentile(db, 95) - 20.0
    return float(np.sqrt(np.mean(e[keep])))


def envelope(x: np.ndarray, sr: int, smooth_ms: float = 20.0) -> np.ndarray:
    from scipy.signal import sosfiltfilt, butter

    sos = butter(2, 1000.0 / (smooth_ms * sr / 2.0), output="sos")
    env = sosfiltfilt(sos, np.abs(x))
    return np.maximum(env, 0.0)


def _band_limited_walk(
    rng: np.random.Generator, n: int, sr: int, rate_hz: float, low_hz: float = 0.0
) -> np.ndarray:
    """Unit-RMS random signal with spectrum between ``low_hz`` and ``rate_hz``."""
    from scipy.signal import butter, sosfiltfilt

    ctrl_sr = 200
    m = max(8, int(np.ceil(n / sr * ctrl_sr)) + 1)
    hi = min(0.95, rate_hz / (ctrl_sr / 2.0))
    if low_hz > 0 and low_hz < rate_hz:
        sos = butter(2, [low_hz / (ctrl_sr / 2.0), hi], btype="band", output="sos")
    else:
        sos = butter(2, hi, output="sos")
    walk = sosfiltfilt(sos, rng.standard_normal(m)) if m > 30 else rng.standard_normal(m)
    walk /= np.sqrt(np.mean(walk**2)) + 1e-12
    return np.interp(np.arange(n) / sr * ctrl_sr, np.arange(m), walk)


def micro_variation(
    rng: np.random.Generator, x: np.ndarray, sr: int, stage: dict
) -> tuple[np.ndarray, dict]:
    """Human-player pitch wander (a few cents) and amplitude shimmer.

    Pitch wander is applied by time-varying resampling whose read position
    never drifts more than a few milliseconds from the original clock.
    """
    cents = _u(rng, stage.get("pitch_cents"), 4.0)
    p_rate = _u(rng, stage.get("pitch_rate_hz"), 4.0)
    shimmer = _u(rng, stage.get("shimmer_db"), 0.5)
    s_rate = _u(rng, stage.get("shimmer_rate_hz"), 8.0)
    y = x
    max_drift_ms = 0.0
    if cents > 0:
        # Build the read-position offset directly as a bounded random curve;
        # its slope is the pitch deviation, scaled to ``cents`` RMS.
        walk = _band_limited_walk(rng, x.size, sr, p_rate, low_hz=0.5)
        slope = np.gradient(walk) * sr
        scale = (cents * np.log(2.0) / 1200.0) / (np.sqrt(np.mean(slope**2)) + 1e-12)
        drift = scale * walk * sr
        max_drift_ms = float(np.max(np.abs(drift)) / sr * 1000.0)
        pos = np.arange(x.size) + drift
        y = np.interp(np.clip(pos, 0, x.size - 1), np.arange(x.size), x)
    if shimmer > 0:
        y = y * 10.0 ** (shimmer * _band_limited_walk(rng, x.size, sr, s_rate) / 20.0)
    return y, {
        "pitch_cents": cents,
        "pitch_rate_hz": p_rate,
        "shimmer_db": shimmer,
        "shimmer_rate_hz": s_rate,
        "max_drift_ms": max_drift_ms,
    }


def dereverb(rng: np.random.Generator, x: np.ndarray, sr: int, stage: dict) -> tuple[np.ndarray, dict]:
    """Suppress the hall ambience baked into Muse Sounds samples.

    Late-reverb spectral subtraction (Lebart et al.): the reverberant power at
    frame t is predicted from the power ``delay_ms`` earlier, decayed by
    ``t60``; the STFT magnitude gain removes ``strength`` of that estimate.
    """
    import librosa

    t60 = _u(rng, stage.get("t60"), 1.5)
    strength = _u(rng, stage.get("strength"), 0.7)
    floor_db = _u(rng, stage.get("floor_db"), -12.0)
    delay_ms = float(stage.get("delay_ms", 50.0))
    n_fft, hop = 1024, 256
    spec = librosa.stft(x.astype(np.float32), n_fft=n_fft, hop_length=hop)
    power = np.abs(spec) ** 2
    d = max(1, int(round(delay_ms / 1000.0 * sr / hop)))
    decay = np.exp(-2.0 * 6.908 * (d * hop / sr) / max(t60, 0.1))
    late = np.zeros_like(power)
    late[:, d:] = decay * power[:, :-d]
    g2 = np.maximum(1.0 - strength * late / (power + 1e-12), 10.0 ** (floor_db / 10.0))
    gain = np.sqrt(g2)
    gain[:, 1:-1] = 0.25 * gain[:, :-2] + 0.5 * gain[:, 1:-1] + 0.25 * gain[:, 2:]
    y = librosa.istft(spec * gain, hop_length=hop, n_fft=n_fft, length=x.size)
    return y.astype(np.float64), {"t60": t60, "strength": strength, "floor_db": floor_db}


def sharpen_partials(
    rng: np.random.Generator, x: np.ndarray, sr: int, stage: dict
) -> tuple[np.ndarray, dict]:
    """Deepen valleys between partials (Muse Sounds partials are less separated
    than a real clarinet's). Within ``[low_hz, high_hz]`` each STFT frame's
    magnitudes are raised to ``gamma`` and rescaled to keep the band energy.
    """
    import librosa

    gamma = _u(rng, stage.get("gamma"), 1.15)
    lo = float(stage.get("low_hz", 600.0))
    hi = float(stage.get("high_hz", 7000.0))
    n_fft, hop = 2048, 256
    spec = librosa.stft(x.astype(np.float32), n_fft=n_fft, hop_length=hop)
    mag = np.abs(spec) + 1e-12
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    ramp = np.clip((freqs - lo * 0.7) / (lo * 0.3), 0, 1) * np.clip((hi * 1.3 - freqs) / (hi * 0.3), 0, 1)
    g_exp = 1.0 + (gamma - 1.0) * ramp[:, None]
    peak = mag.max(axis=0, keepdims=True)
    sharp = peak * (mag / peak) ** g_exp
    band = ramp[:, None] > 0
    e_in = np.sum((mag * band) ** 2, axis=0, keepdims=True)
    e_out = np.sum((sharp * band) ** 2, axis=0, keepdims=True) + 1e-24
    scale = np.where(band, np.sqrt(e_in / e_out), 1.0)
    gain = sharp * scale / mag
    y = librosa.istft(spec * gain, hop_length=hop, n_fft=n_fft, length=x.size)
    return y.astype(np.float64), {"gamma": gamma}


def tonguing(rng: np.random.Generator, x: np.ndarray, sr: int, stage: dict) -> tuple[np.ndarray, dict]:
    """Dip the level just before note onsets, like a player tonguing each note.

    Each dip ends exactly at the detected onset, so attacks and onset times
    are unchanged; only the tail of the previous note is shortened.
    """
    import librosa

    depth_db = _u(rng, stage.get("depth_db"), 10.0)
    width = _u(rng, stage.get("width_ms"), 25.0) / 1000.0
    share = _u(rng, stage.get("share"), 0.8)
    onsets = librosa.onset.onset_detect(
        y=x.astype(np.float32), sr=sr, hop_length=256, units="samples", backtrack=True,
        delta=float(stage.get("onset_delta", 0.07)),
    )
    gain_db = np.zeros(x.size)
    w = max(8, int(width * sr))
    n_down = int(0.7 * w)
    shape = np.concatenate(
        [
            0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n_down)),
            0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, w - n_down)),
        ]
    )
    used = 0
    for onset in onsets:
        if onset < w or rng.random() >= share:
            continue
        seg = slice(int(onset) - w, int(onset))
        dip = -depth_db * rng.uniform(0.6, 1.0) * shape
        gain_db[seg] = np.minimum(gain_db[seg], dip)
        used += 1
    return x * 10.0 ** (gain_db / 20.0), {
        "depth_db": depth_db,
        "width_ms": width * 1000.0,
        "share": share,
        "dips": used,
    }


def breath_noise(rng: np.random.Generator, x: np.ndarray, sr: int, stage: dict) -> tuple[np.ndarray, dict]:
    from scipy.signal import butter, sosfilt

    lo = _u(rng, stage.get("low_hz"), 1200.0)
    hi = min(_u(rng, stage.get("high_hz"), 7000.0), sr / 2.0 * 0.95)
    level_db = _u(rng, stage.get("level_db"), -32.0)
    sos = butter(2, [lo / (sr / 2.0), hi / (sr / 2.0)], btype="band", output="sos")
    noise = sosfilt(sos, rng.standard_normal(x.size))
    env = envelope(x, sr)
    shaped = noise * env
    scale = active_rms(x, sr) / (active_rms(shaped, sr) + 1e-12) * 10.0 ** (level_db / 20.0)
    return x + shaped * scale, {"low_hz": lo, "high_hz": hi, "level_db": level_db}


def room_impulse(rng: np.random.Generator, sr: int, stage: dict) -> tuple[np.ndarray, dict]:
    from scipy.signal import butter, sosfilt

    rt60 = _u(rng, stage.get("rt60"), 0.5)
    wet_db = _u(rng, stage.get("wet_db"), -8.0)
    predelay = _u(rng, stage.get("predelay_ms"), 8.0) / 1000.0
    hf_ratio = _u(rng, stage.get("hf_rt60_ratio"), 0.5)
    lf_ratio = _u(rng, stage.get("lf_rt60_ratio"), 1.1)
    n = int(sr * min(3.0, rt60 * 1.3 + predelay + 0.05))
    t = np.arange(n) / sr
    tail = np.zeros(n)
    splits = [(None, 500.0, lf_ratio), (500.0, 3000.0, 1.0), (3000.0, None, hf_ratio)]
    for lo, hi, ratio in splits:
        if lo is None:
            sos = butter(2, hi / (sr / 2.0), btype="low", output="sos")
        elif hi is None:
            sos = butter(2, lo / (sr / 2.0), btype="high", output="sos")
        else:
            sos = butter(2, [lo / (sr / 2.0), hi / (sr / 2.0)], btype="band", output="sos")
        band = sosfilt(sos, rng.standard_normal(n))
        tail += band * np.exp(-6.908 * t / max(0.05, rt60 * ratio))
    onset = int(predelay * sr)
    tail[:onset] = 0.0
    ramp = min(n - onset, int(0.01 * sr))
    if ramp > 0:
        tail[onset : onset + ramp] *= np.linspace(0.0, 1.0, ramp)
    n_taps = int(stage.get("early_taps", 6))
    early = np.zeros(n)
    for _ in range(n_taps):
        d = int(rng.uniform(0.002, max(0.004, predelay + 0.03)) * sr)
        if 0 < d < n:
            early[d] += rng.uniform(-1.0, 1.0) * np.exp(-d / sr / 0.03)
    wet = tail + early * 2.0
    wet *= 10.0 ** (wet_db / 20.0) / (np.sqrt(np.sum(wet**2)) + 1e-12)
    ir = wet
    ir[0] += 1.0
    info = {
        "rt60": rt60,
        "wet_db": wet_db,
        "predelay_ms": predelay * 1000.0,
        "hf_rt60_ratio": hf_ratio,
        "lf_rt60_ratio": lf_ratio,
    }
    return ir, info


def apply_reverb(rng: np.random.Generator, x: np.ndarray, sr: int, stage: dict) -> tuple[np.ndarray, dict]:
    from scipy.signal import fftconvolve

    ir, info = room_impulse(rng, sr, stage)
    return fftconvolve(x, ir)[: x.size], info


def background_noise(
    rng: np.random.Generator, x: np.ndarray, sr: int, stage: dict, ref_rms: float
) -> tuple[np.ndarray, dict]:
    anchors = stage["anchors_hz"]
    gains = _jittered(
        rng,
        stage["color_db"],
        float(stage.get("color_jitter_db", 0.0)),
        float(stage.get("tilt_jitter_db", 0.0)),
        anchors,
    )
    snr_db = _u(rng, stage.get("snr_db"), 35.0)
    noise = _colored_noise(rng, x.size, sr, anchors, gains)
    mod_db = _u(rng, stage.get("modulation_db"), 0.0)
    if mod_db > 0:
        wobble = _smooth_random(rng, x.size, sr, _u(rng, stage.get("modulation_rate_hz"), 0.3))
        noise *= 10.0 ** (mod_db * wobble / 20.0)
    noise *= ref_rms * 10.0 ** (-snr_db / 20.0)
    info: dict = {"snr_db": snr_db, "modulation_db": mod_db}

    hum = stage.get("hum") or {}
    if hum and rng.random() < float(hum.get("prob", 0.0)):
        f0 = float(hum.get("freq_hz", 50.0))
        t = np.arange(x.size) / sr
        tone = np.zeros(x.size)
        for k in range(1, int(hum.get("harmonics", 4)) + 1):
            if f0 * k < sr / 2.0:
                tone += (0.6 ** (k - 1)) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 2 * np.pi))
        tone /= np.sqrt(np.mean(tone**2)) + 1e-12
        hum_db = _u(rng, hum.get("level_db"), -50.0)
        noise += tone * ref_rms * 10.0 ** (hum_db / 20.0)
        info["hum_db"] = hum_db

    events = stage.get("events") or {}
    if events:
        rate = _u(rng, events.get("rate_per_min"), 0.0)
        count = int(rng.poisson(rate * x.size / sr / 60.0))
        for _ in range(count):
            dur = int(rng.uniform(0.01, 0.12) * sr)
            start = int(rng.integers(0, max(1, x.size - dur)))
            burst = rng.standard_normal(dur) * np.exp(-np.linspace(0, 6, dur))
            burst = _apply_spectral_gain(burst, sr, [100, 400, 2000, 8000], [0, -3, -12, -24])
            burst /= np.max(np.abs(burst)) + 1e-12
            level = ref_rms * 10.0 ** (_u(rng, events.get("level_db"), -25.0) / 20.0) * 3.0
            noise[start : start + dur] += burst * level
        info["events"] = count
    return x + noise, info


def compress(rng: np.random.Generator, x: np.ndarray, sr: int, stage: dict) -> tuple[np.ndarray, dict]:
    """Feed-forward RMS compressor/AGC; gain computed on 2.9 ms blocks, no lookahead."""
    thr = _u(rng, stage.get("threshold_db"), -24.0)
    ratio = _u(rng, stage.get("ratio"), 2.0)
    att = _u(rng, stage.get("attack_ms"), 10.0) / 1000.0
    rel = _u(rng, stage.get("release_ms"), 200.0) / 1000.0
    hop = 64
    n = int(np.ceil(x.size / hop))
    pad = np.zeros(n * hop)
    pad[: x.size] = x
    blocks = np.sqrt(np.mean(pad.reshape(n, hop) ** 2, axis=1) + 1e-12)
    peak = np.max(blocks)
    level_db = 20.0 * np.log10(blocks / (peak + 1e-12) + 1e-12)
    a_att = np.exp(-hop / (sr * max(att, 1e-4)))
    a_rel = np.exp(-hop / (sr * max(rel, 1e-4)))
    env = np.empty(n)
    cur = level_db[0]
    for i in range(n):
        target = level_db[i]
        coef = a_att if target > cur else a_rel
        cur = coef * cur + (1.0 - coef) * target
        env[i] = cur
    over = np.maximum(env - thr, 0.0)
    gain_db = -over * (1.0 - 1.0 / ratio)
    gain_db -= gain_db.max()
    centers = (np.arange(n) + 0.5) * hop
    gain = 10.0 ** (np.interp(np.arange(x.size), centers, gain_db) / 20.0)
    return x * gain, {"threshold_db": thr, "ratio": ratio, "attack_ms": att * 1e3, "release_ms": rel * 1e3}


def _find_ffmpeg() -> str:
    from synthpipeline.musesounds import find_ffmpeg

    return str(find_ffmpeg())


def codec_roundtrip(
    rng: np.random.Generator, x: np.ndarray, sr: int, stage: dict
) -> tuple[np.ndarray, dict]:
    """Encode to AAC in an .m4a (like a phone voice memo) and decode back to ``sr``."""
    import soundfile as sf

    bitrate = int(round(_u(rng, stage.get("bitrate_k"), 64.0)))
    codec_sr = int(stage.get("sample_rate", 48000))
    ffmpeg = _find_ffmpeg()
    fd, tmp_in = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    tmp_m4a = tmp_in[:-4] + ".m4a"
    tmp_out = tmp_in[:-4] + "_dec.wav"
    peak = float(np.max(np.abs(x))) + 1e-12
    try:
        sf.write(tmp_in, (x / peak * 0.9).astype(np.float32), sr, subtype="FLOAT")
        subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-i", tmp_in, "-ar", str(codec_sr), "-ac", "1",
             "-c:a", "aac", "-b:a", f"{bitrate}k", tmp_m4a],
            check=True, capture_output=True,
        )
        subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-i", tmp_m4a, "-ar", str(sr), "-ac", "1",
             "-c:a", "pcm_f32le", tmp_out],
            check=True, capture_output=True,
        )
        y, _ = sf.read(tmp_out, dtype="float64", always_2d=False)
    finally:
        for p in (tmp_in, tmp_m4a, tmp_out):
            Path(p).unlink(missing_ok=True)
    y = y * (peak / 0.9)
    lag = best_lag(x, y, max_lag=int(0.2 * sr))
    y = shift(y, -lag, x.size)
    return y, {"bitrate_k": bitrate, "sample_rate": codec_sr, "lag_samples": int(lag)}


def _xcorr_peak(a: np.ndarray, b: np.ndarray, max_lag: int) -> int:
    n = min(a.size, b.size)
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    size = 1 << int(np.ceil(np.log2(2 * n)))
    xc = np.fft.irfft(np.fft.rfft(b, size) * np.conj(np.fft.rfft(a, size)), size)
    cand = np.concatenate([xc[: max_lag + 1], xc[-max_lag:]])
    lags = np.concatenate([np.arange(max_lag + 1), np.arange(-max_lag, 0)])
    return int(lags[int(np.argmax(cand))])


def _log_envelope(x: np.ndarray, hop: int) -> np.ndarray:
    n = x.size // hop
    e = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-10)
    return np.diff(np.log(e), prepend=np.log(e[0]))


def best_lag(ref: np.ndarray, y: np.ndarray, max_lag: int) -> int:
    """Samples by which ``y`` trails ``ref`` (positive: y is late).

    Coarse lag from onset (log-envelope) correlation, which has no pitch-period
    ambiguity; refined by waveform correlation within one envelope hop.
    """
    hop = 16
    n = min(ref.size, y.size, 22050 * 30)
    coarse = hop * _xcorr_peak(
        _log_envelope(ref[:n], hop), _log_envelope(y[:n], hop), max(1, max_lag // hop)
    )
    shifted = shift(y[:n], -coarse, n)
    return coarse + _xcorr_peak(ref[:n], shifted, hop)


def shift(y: np.ndarray, lag: int, length: int) -> np.ndarray:
    out = np.zeros(length)
    if lag >= 0:
        seg = y[: max(0, length - lag)]
        out[lag : lag + seg.size] = seg
    else:
        seg = y[-lag : -lag + length]
        out[: seg.size] = seg
    return out


def soft_clip(rng: np.random.Generator, x: np.ndarray, stage: dict) -> tuple[np.ndarray, dict]:
    drive_db = _u(rng, stage.get("drive_db"), 3.0)
    peak = float(np.max(np.abs(x))) + 1e-12
    g = 10.0 ** (drive_db / 20.0) / peak
    return np.tanh(x * g) / np.tanh(g * peak) * peak, {"drive_db": drive_db}


def degrade_audio(
    audio: np.ndarray, sr: int, params: dict, rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    """Return degraded audio (same length as ``audio``) and the sampled parameters."""
    x = np.asarray(audio, dtype=np.float64).copy()
    info: dict = {"version": str(params.get("version", DEGRADE_MARK))}
    if x.size == 0 or not np.any(x):
        return x.astype(np.float32), info

    stage = params.get("dereverb")
    if _hit(rng, stage):
        x, info["dereverb"] = dereverb(rng, x, sr, stage)

    stage = params.get("sharpen")
    if _hit(rng, stage):
        x, info["sharpen"] = sharpen_partials(rng, x, sr, stage)

    stage = params.get("tonguing")
    if _hit(rng, stage):
        x, info["tonguing"] = tonguing(rng, x, sr, stage)

    stage = params.get("micro")
    if _hit(rng, stage):
        x, info["micro"] = micro_variation(rng, x, sr, stage)

    stage = params.get("breath")
    if _hit(rng, stage):
        x, info["breath"] = breath_noise(rng, x, sr, stage)

    stage = params.get("reverb")
    if _hit(rng, stage):
        x, info["reverb"] = apply_reverb(rng, x, sr, stage)

    stage = params.get("eq")
    if stage:
        gains = _jittered(
            rng,
            stage["gains_db"],
            float(stage.get("jitter_db", 0.0)),
            float(stage.get("tilt_jitter_db", 0.0)),
            stage["anchors_hz"],
        )
        x = _apply_spectral_gain(x, sr, stage["anchors_hz"], gains)
        info["eq_gains_db"] = [round(float(g), 2) for g in gains]

    ref_rms = active_rms(x, sr)
    stage = params.get("noise")
    if _hit(rng, stage):
        x, info["noise"] = background_noise(rng, x, sr, stage, ref_rms)

    stage = params.get("gain_wobble")
    if _hit(rng, stage):
        depth = _u(rng, stage.get("depth_db"), 1.0)
        rate = _u(rng, stage.get("rate_hz"), 0.2)
        x *= 10.0 ** (depth * _smooth_random(rng, x.size, sr, rate) / 20.0)
        info["gain_wobble"] = {"depth_db": depth, "rate_hz": rate}

    stage = params.get("compressor")
    if _hit(rng, stage):
        x, info["compressor"] = compress(rng, x, sr, stage)

    stage = params.get("codec")
    if _hit(rng, stage):
        x, info["codec"] = codec_roundtrip(rng, x, sr, stage)

    stage = params.get("clip")
    if _hit(rng, stage):
        x, info["clip"] = soft_clip(rng, x, stage)

    peak = float(np.max(np.abs(x))) + 1e-12
    target = _u(rng, params.get("peak"), 0.89)
    x = x * (target / peak)
    info["peak"] = target
    return x.astype(np.float32), _round(info)


def _round(obj):
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 4)
    return obj
