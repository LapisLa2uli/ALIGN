from __future__ import annotations

import logging
from pathlib import Path

from datacreate.audio_utils import load_audio
import librosa
import matplotlib.pyplot as plt
import numpy as np

from datacreate.config import PipelineConfig


def extract_mels(
    performance_wav: Path,
    reference_wav: Path,
    sample_dir: Path,
    config: PipelineConfig,
    logger: logging.Logger,
) -> dict[str, Path]:
    sr = config.sample_rate()
    mel_cfg = config.mel
    n_fft = int(mel_cfg.get("n_fft", 2048))
    hop = int(mel_cfg.get("hop_length", 512))
    n_mels = int(mel_cfg.get("n_mels", 128))
    fmin = float(mel_cfg.get("fmin", 30.0))
    fmax = mel_cfg.get("fmax")
    fmax_val = float(fmax) if fmax is not None else None

    outputs: dict[str, Path] = {}
    for name, wav in [("performance", performance_wav), ("reference", reference_wav)]:
        audio, _ = load_audio(wav, sr, mono=True)
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax_val,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        npy_path = sample_dir / f"{name}_mel.npy"
        np.save(npy_path, log_mel.astype(np.float32))
        png_path = sample_dir / f"{name}_mel_preview.png"
        _save_preview(log_mel, png_path, title=f"{name} log-mel")
        outputs[f"{name}_mel"] = npy_path
        outputs[f"{name}_mel_preview"] = png_path
        logger.info("Saved %s mel spectrogram %s", name, npy_path)
    return outputs


def _save_preview(log_mel: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        log_mel,
        x_axis="time",
        y_axis="mel",
        ax=ax,
        hop_length=512,
    )
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
