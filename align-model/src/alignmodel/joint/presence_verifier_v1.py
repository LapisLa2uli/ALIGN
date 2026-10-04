"""Score-informed note-presence verifier (v1).

Question answered: "was score note k (written pitch p, between neighbours with
pitches a and b) played at about frame t?" The input is a window of the
dual-resolution log-mel around the expected onset plus three harmonic templates
(target, previous and next pitch) over the same mel bands. The transcriber's
CTC output is peaky and can drop a played note entirely; the verifier looks
directly for the expected pitch's partials, so a missed-note call can be
withheld when the note is audibly present.

Trained on synthetic training splits only: positives are played score notes at
their forced-aligned onset, negatives are planted missed notes at the onset
interpolated from their played neighbours (rest or held previous note).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

WINDOW = 48
MARGIN = 6
BEFORE = 12
HARMONICS = 10


@dataclass(frozen=True)
class PresenceConfig:
    channels: int = 32
    hidden: int = 64
    audio_transpose: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=4)
def _band_freqs(n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    import librosa

    return librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)


def pitch_template(written_pitch: int, audio_transpose: int = 2) -> np.ndarray:
    """192-band harmonic template (128 long-window + 64 short-window bands)."""

    template = np.zeros(192, np.float32)
    if written_pitch is None or written_pitch < 0:
        return template
    f0 = 440.0 * 2 ** ((int(written_pitch) - int(audio_transpose) - 69) / 12)
    for offset, (count, fmin) in ((0, (128, 45.0)), (128, (64, 100.0))):
        freqs = _band_freqs(count, fmin, 8000.0)
        for harmonic in range(1, HARMONICS + 1):
            frequency = f0 * harmonic
            if frequency > 7800.0 or frequency < fmin:
                continue
            band = int(np.argmin(np.abs(freqs - frequency)))
            template[offset + band] = max(template[offset + band], 1.0 / np.sqrt(harmonic))
    return template


def extract_window(mel: np.ndarray, center: float) -> np.ndarray:
    """[192, WINDOW + 2*MARGIN] window starting BEFORE+MARGIN frames before ``center``."""

    start = int(round(center)) - BEFORE - MARGIN
    length = WINDOW + 2 * MARGIN
    output = np.full((mel.shape[0], length), float(mel.min()) if mel.size else -4.0, np.float32)
    lo, hi = max(0, start), min(mel.shape[1], start + length)
    if hi > lo:
        output[:, lo - start:hi - start] = mel[:, lo:hi]
    return output


class PresenceNet(nn.Module):
    def __init__(self, config: PresenceConfig = PresenceConfig()) -> None:
        super().__init__()
        self.config = config
        c = config.channels
        self.features = nn.Sequential(
            nn.Conv2d(5, c, (5, 3), padding=(2, 1)), nn.GELU(), nn.BatchNorm2d(c),
            nn.Conv2d(c, c, (5, 3), padding=(2, 1)), nn.GELU(), nn.BatchNorm2d(c),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(c, 2 * c, (5, 3), padding=(2, 1)), nn.GELU(), nn.BatchNorm2d(2 * c),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(2 * c, 2 * c, (3, 3), padding=1), nn.GELU(), nn.BatchNorm2d(2 * c),
        )
        self.head = nn.Sequential(nn.Linear(4 * c + 2, config.hidden), nn.GELU(), nn.Linear(config.hidden, 1))

    def forward(self, windows: torch.Tensor, templates: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        # windows [B, 192, W]; templates [B, 3, 192]; scalars [B, 2]
        frames = windows.shape[-1]
        target = templates[:, 0, :, None].expand(-1, -1, frames)
        previous = templates[:, 1, :, None].expand(-1, -1, frames)
        following = templates[:, 2, :, None].expand(-1, -1, frames)
        x = torch.stack([windows, target, previous, following, windows * target], dim=1)
        h = self.features(x)
        pooled = torch.cat([h.amax(dim=(2, 3)), h.mean(dim=(2, 3))], dim=1)
        return self.head(torch.cat([pooled, scalars], dim=1)).squeeze(1)


def scalars_for(duration_frames: float, uncertainty_frames: float) -> np.ndarray:
    return np.array([np.log1p(max(duration_frames, 0.0)) / 4.0, np.log1p(max(uncertainty_frames, 0.0)) / 4.0],
                    np.float32)


def save_verifier(path: Path, model: PresenceNet, extra: dict) -> None:
    torch.save({"schema_version": "align-presence-verifier-v1", "config": model.config.to_dict(),
                "state_dict": model.state_dict(), **extra}, path)


def load_verifier(path: Path, device) -> PresenceNet:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = PresenceNet(PresenceConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


@torch.inference_mode()
def score_presence(model: PresenceNet, mel: np.ndarray, queries: Sequence[tuple[float, int, int, int, float, float]],
                   device, offsets: Sequence[int] = (-3, 0, 3)) -> np.ndarray:
    """Probability that each queried score note is present.

    ``queries``: (center_frame, written_pitch, previous_pitch, next_pitch,
    duration_frames, uncertainty_frames). The maximum over small centre
    offsets is returned, so a withheld missed-note call needs the note to be
    absent at every nearby position.
    """

    if not queries:
        return np.zeros(0, np.float32)
    transpose = model.config.audio_transpose
    windows, templates, scalars = [], [], []
    for center, pitch, previous, following, duration, uncertainty in queries:
        for offset in offsets:
            window = extract_window(mel, center + offset)[:, MARGIN:MARGIN + WINDOW]
            windows.append(window)
            templates.append(np.stack([pitch_template(pitch, transpose), pitch_template(previous, transpose),
                                       pitch_template(following, transpose)]))
            scalars.append(scalars_for(duration, uncertainty))
    logits = model(torch.as_tensor(np.stack(windows), device=device),
                   torch.as_tensor(np.stack(templates), device=device),
                   torch.as_tensor(np.stack(scalars), device=device))
    probabilities = torch.sigmoid(logits).float().cpu().numpy().reshape(len(queries), len(offsets))
    return probabilities.max(axis=1)
