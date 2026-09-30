"""Log-mel statistics for comparing synthetic performance audio with real recordings.

Features use the DataCreate mel settings (n_fft 2048, hop 512, 128 mels,
fmin 30 Hz) so they describe what the model actually sees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

N_FFT = 2048
HOP = 512
N_MELS = 128
FMIN = 30.0
N_BANDS = 16
EPS = 1e-12

SCALAR_KEYS = (
    "floor_db",
    "median_db",
    "active_frac",
    "flatness_db",
    "centroid_hz",
    "rolloff_hz",
    "hf_ratio_db",
    "lf_ratio_db",
    "delta_db",
    "fall_db",
    "harmonic_contrast_db",
    "mel_p10",
    "mel_p25",
    "mel_p50",
    "mel_p75",
    "mel_floor_frac",
    "quiet_tilt_db",
)


@dataclass
class ClipStats:
    name: str
    scalars: dict[str, float]
    ltas_active: np.ndarray
    ltas_quiet: np.ndarray
    contrast: np.ndarray
    delta_bands: np.ndarray
    extra: dict = field(default_factory=dict)


def load_mono(path: Path, sample_rate: int = 22050, max_seconds: float | None = 60.0) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        g = int(np.gcd(int(sr), int(sample_rate)))
        audio = resample_poly(audio, sample_rate // g, int(sr) // g).astype(np.float32)
    if max_seconds:
        audio = audio[: int(max_seconds * sample_rate)]
    return audio


def _band_edges(n: int = N_MELS, bands: int = N_BANDS) -> list[tuple[int, int]]:
    step = n // bands
    return [(i * step, (i + 1) * step) for i in range(bands)]


def _pool_bands(curve: np.ndarray, bands: int = N_BANDS) -> np.ndarray:
    return np.array([curve[a:b].mean() for a, b in _band_edges(len(curve), bands)], dtype=np.float64)


def clip_stats(audio: np.ndarray, sample_rate: int = 22050, name: str = "") -> ClipStats:
    import librosa

    audio = np.asarray(audio, dtype=np.float32)
    stft = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP)) ** 2
    mel_fb = librosa.filters.mel(sr=sample_rate, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN)
    mel = mel_fb @ stft
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=N_FFT)

    energy = 10.0 * np.log10(mel.sum(axis=0) + EPS)
    ref = float(np.percentile(energy, 95))
    active = energy > ref - 20.0
    if active.sum() < 4:
        active = energy >= np.percentile(energy, 50)
    quiet = energy <= np.percentile(energy, 10)

    act_curve = 10.0 * np.log10(mel[:, active].mean(axis=1) + EPS)
    # Mean over 260 Hz–5 kHz (mel bins 8–95): robust to one dominant partial.
    peak = float(act_curve[8:96].mean())
    ltas_active = act_curve - peak
    ltas_quiet = 10.0 * np.log10(mel[:, quiet].mean(axis=1) + EPS) - peak

    log_mel = librosa.power_to_db(mel, ref=np.max)
    geo = np.exp(np.mean(np.log(mel[:, active] + EPS), axis=0))
    arith = np.mean(mel[:, active], axis=0) + EPS
    flatness_db = float(np.median(10.0 * np.log10(geo / arith + EPS)))

    spec_act = stft[:, active]
    tot = spec_act.sum(axis=0) + EPS
    centroid = float(np.median((freqs[:, None] * spec_act).sum(axis=0) / tot))
    cum = np.cumsum(spec_act, axis=0) / tot
    rolloff = float(np.median(freqs[np.argmax(cum >= 0.85, axis=0)]))
    hf = 10.0 * np.log10(spec_act[freqs >= 4000].sum() / tot.sum() + EPS)
    lf_all = stft.sum(axis=1)
    lf = 10.0 * np.log10(lf_all[freqs < 120].sum() / (lf_all.sum() + EPS) + EPS)

    clipped = np.maximum(log_mel, log_mel.max() - 80.0)
    delta = np.abs(np.diff(clipped, axis=1))
    both = active[1:] & active[:-1]
    delta_db = float(delta[:, both].mean()) if both.any() else float(delta.mean())
    delta_curve = delta[:, both].mean(axis=1) if both.any() else delta.mean(axis=1)
    de = np.diff(energy)
    falls = de[de < -1.0]
    fall_db = float(np.median(falls)) if falls.size else 0.0

    # Peak-to-valley depth between harmonics, measured on the linear-frequency
    # spectrum in 200–3000 Hz where clarinet partials are resolved.
    band = (freqs >= 200) & (freqs <= 3000)
    sub = 10.0 * np.log10(spec_act[band] + EPS)
    hc = float(np.median(np.percentile(sub, 95, axis=0) - np.percentile(sub, 30, axis=0)))

    contrast = librosa.feature.spectral_contrast(
        S=np.sqrt(stft), sr=sample_rate, n_fft=N_FFT, hop_length=HOP
    )[:, active].mean(axis=1)

    vals = clipped.ravel()
    q = ltas_quiet
    scalars = {
        "floor_db": float(np.percentile(energy, 10) - ref),
        "median_db": float(np.percentile(energy, 50) - ref),
        "active_frac": float(active.mean()),
        "flatness_db": flatness_db,
        "centroid_hz": centroid,
        "rolloff_hz": rolloff,
        "hf_ratio_db": float(hf),
        "lf_ratio_db": float(lf),
        "delta_db": delta_db,
        "fall_db": fall_db,
        "harmonic_contrast_db": hc,
        "mel_p10": float(np.percentile(vals, 10)),
        "mel_p25": float(np.percentile(vals, 25)),
        "mel_p50": float(np.percentile(vals, 50)),
        "mel_p75": float(np.percentile(vals, 75)),
        "mel_floor_frac": float(np.mean(vals <= vals.max() - 79.5)),
        "quiet_tilt_db": float(q[96:].mean() - q[:32].mean()),
    }
    return ClipStats(
        name=name,
        scalars=scalars,
        ltas_active=ltas_active,
        ltas_quiet=ltas_quiet,
        contrast=contrast,
        delta_bands=_pool_bands(delta_curve),
    )


def file_stats(path: Path, sample_rate: int = 22050, max_seconds: float | None = 60.0) -> ClipStats:
    return clip_stats(load_mono(Path(path), sample_rate, max_seconds), sample_rate, name=str(path))


def feature_matrix(stats: list[ClipStats]) -> np.ndarray:
    rows = []
    for s in stats:
        rows.append(
            np.concatenate(
                [
                    [s.scalars[k] for k in SCALAR_KEYS],
                    _pool_bands(s.ltas_active),
                    _pool_bands(s.ltas_quiet),
                    s.contrast,
                    s.delta_bands,
                ]
            )
        )
    return np.asarray(rows, dtype=np.float64)


def compare(real: list[ClipStats], synth: list[ClipStats]) -> dict:
    """Per-feature standardized median gaps plus a real-vs-synth classifier AUC."""
    out: dict = {"n_real": len(real), "n_synth": len(synth), "scalars": {}}
    for key in SCALAR_KEYS:
        r = np.array([s.scalars[key] for s in real])
        y = np.array([s.scalars[key] for s in synth])
        iqr = float(np.subtract(*np.percentile(r, [75, 25])))
        scale = max(iqr / 1.349, 0.02 * abs(float(np.median(r))) + 1e-3)
        out["scalars"][key] = {
            "real_median": float(np.median(r)),
            "synth_median": float(np.median(y)),
            "real_iqr": iqr,
            "synth_iqr": float(np.subtract(*np.percentile(y, [75, 25]))),
            "gap_sd": float((np.median(y) - np.median(r)) / scale),
        }
    ra = np.median([s.ltas_active for s in real], axis=0)
    sa = np.median([s.ltas_active for s in synth], axis=0)
    rq = np.median([s.ltas_quiet for s in real], axis=0)
    sq = np.median([s.ltas_quiet for s in synth], axis=0)
    out["_synth_ltas_active"] = sa.tolist()
    out["_synth_ltas_quiet"] = sq.tolist()
    out["ltas_active_mae_db"] = float(np.mean(np.abs(ra - sa)))
    out["ltas_quiet_mae_db"] = float(np.mean(np.abs(rq - sq)))
    out["ltas_active_bands"] = {
        "real": _pool_bands(ra).round(1).tolist(),
        "synth": _pool_bands(sa).round(1).tolist(),
    }
    out["ltas_quiet_bands"] = {
        "real": _pool_bands(rq).round(1).tolist(),
        "synth": _pool_bands(sq).round(1).tolist(),
    }
    out["delta_bands"] = {
        "real": np.median([s.delta_bands for s in real], axis=0).round(2).tolist(),
        "synth": np.median([s.delta_bands for s in synth], axis=0).round(2).tolist(),
    }
    out["contrast_bands"] = {
        "real": np.median([s.contrast for s in real], axis=0).round(1).tolist(),
        "synth": np.median([s.contrast for s in synth], axis=0).round(1).tolist(),
    }
    out["mean_abs_gap_sd"] = float(np.mean([abs(v["gap_sd"]) for v in out["scalars"].values()]))
    out["classifier_auc"] = classifier_auc(real, synth)
    out["top_features"] = feature_aucs(real, synth)[:8]
    return out


def feature_names() -> list[str]:
    names = list(SCALAR_KEYS)
    names += [f"ltas_active_b{i}" for i in range(N_BANDS)]
    names += [f"ltas_quiet_b{i}" for i in range(N_BANDS)]
    names += [f"contrast_b{i}" for i in range(7)]
    names += [f"delta_b{i}" for i in range(N_BANDS)]
    return names


def feature_aucs(real: list[ClipStats], synth: list[ClipStats]) -> list[tuple[str, float]]:
    """Single-feature separability, as max(AUC, 1 - AUC), most separable first."""
    from sklearn.metrics import roc_auc_score

    x = np.vstack([feature_matrix(real), feature_matrix(synth)])
    y = np.concatenate([np.ones(len(real)), np.zeros(len(synth))])
    rows = []
    for name, col in zip(feature_names(), x.T):
        auc = float(roc_auc_score(y, col))
        rows.append((name, round(max(auc, 1.0 - auc), 3)))
    return sorted(rows, key=lambda r: -r[1])


def classifier_auc(real: list[ClipStats], synth: list[ClipStats], seed: int = 0) -> float:
    """5-fold CV ROC AUC of a regularized logistic regression separating real from synth.

    0.5 means the feature set cannot tell the two apart.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.vstack([feature_matrix(real), feature_matrix(synth)])
    y = np.concatenate([np.ones(len(real)), np.zeros(len(synth))])
    # liblinear: the MusicEval scipy build crashes inside L-BFGS-B (0xc06d007f).
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, solver="liblinear")
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    prob = cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, prob))


def format_report(result: dict) -> str:
    lines = [
        f"real={result['n_real']} synth={result['n_synth']}  "
        f"classifier_auc={result['classifier_auc']:.3f}  "
        f"mean|gap|={result['mean_abs_gap_sd']:.2f} sd  "
        f"ltas_active_mae={result['ltas_active_mae_db']:.2f} dB  "
        f"ltas_quiet_mae={result['ltas_quiet_mae_db']:.2f} dB",
        f"{'feature':24s} {'real':>10s} {'synth':>10s} {'gap_sd':>8s} {'real_iqr':>9s} {'syn_iqr':>9s}",
    ]
    for key, row in result["scalars"].items():
        lines.append(
            f"{key:24s} {row['real_median']:10.2f} {row['synth_median']:10.2f} {row['gap_sd']:8.2f}"
            f" {row['real_iqr']:9.2f} {row['synth_iqr']:9.2f}"
        )
    lines.append("ltas_active real  " + " ".join(f"{v:6.1f}" for v in result["ltas_active_bands"]["real"]))
    lines.append("ltas_active synth " + " ".join(f"{v:6.1f}" for v in result["ltas_active_bands"]["synth"]))
    lines.append("ltas_quiet  real  " + " ".join(f"{v:6.1f}" for v in result["ltas_quiet_bands"]["real"]))
    lines.append("ltas_quiet  synth " + " ".join(f"{v:6.1f}" for v in result["ltas_quiet_bands"]["synth"]))
    lines.append("delta_bands real  " + " ".join(f"{v:6.2f}" for v in result["delta_bands"]["real"]))
    lines.append("delta_bands synth " + " ".join(f"{v:6.2f}" for v in result["delta_bands"]["synth"]))
    lines.append("contrast    real  " + " ".join(f"{v:6.1f}" for v in result["contrast_bands"]["real"]))
    lines.append("contrast    synth " + " ".join(f"{v:6.1f}" for v in result["contrast_bands"]["synth"]))
    lines.append("most separable: " + ", ".join(f"{n}={a:.2f}" for n, a in result["top_features"]))
    return "\n".join(lines)
