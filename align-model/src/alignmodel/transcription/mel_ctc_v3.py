"""Dual-resolution, long-context mel CTC transcriber (v3).

v2 (round-2 CTC) used one 2048-sample (93 ms) analysis window and a TCN that
sees about +/-0.9 s. Notes under 80 ms were blurred into neighbours and long
notes could be re-emitted mid-note. v3 adds a 512-sample (23 ms) window branch
for attacks and short notes, keeps the long window for pitch, and widens the
temporal context to about +/-2.9 s. Both branches share the 256-sample hop,
so frames align exactly; the model input is the two log-mels stacked as
[long (128 bands); short (64 bands)].
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .mel_v1 import MelFrontendConfig, extract_log_mel, _SpectralBlock


SCHEMA_VERSION = "align-mel-ctc-transcriber-v3"
LONG_FRONTEND = MelFrontendConfig()
SHORT_FRONTEND = MelFrontendConfig(n_fft=1024, win_length=512, n_mels=64, fmin=100.0)


@dataclass(frozen=True)
class DualMelConfig:
    long_mels: int = 128
    short_mels: int = 64
    midi_min: int = 52
    midi_max: int = 100
    conv_channels: int = 32
    temporal_dim: int = 192
    temporal_blocks: int = 12
    dilation_cycle: int = 6
    kernel_size: int = 5
    dropout: float = 0.10

    @property
    def n_mels(self) -> int:
        return self.long_mels + self.short_mels

    @property
    def n_pitches(self) -> int:
        return self.midi_max - self.midi_min + 1

    @property
    def context_frames_each_side(self) -> int:
        return sum(
            (self.kernel_size - 1) // 2 * 2 ** (index % self.dilation_cycle)
            for index in range(self.temporal_blocks)
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping | None) -> "DualMelConfig":
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: item for key, item in dict(value or {}).items() if key in allowed})


class _DilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, kernel, dilation=dilation,
                      padding=(kernel - 1) // 2 * dilation, groups=channels, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels * 2, 1),
            nn.GLU(dim=1),
            nn.Dropout(dropout),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.network(value)


def _branch(mels: int, channels: int, blocks: int) -> tuple[nn.Sequential, int]:
    layers = []
    incoming = 1
    for index in range(blocks):
        outgoing = channels if index else channels // 2
        layers.append(_SpectralBlock(incoming, outgoing))
        incoming = outgoing
    reduced = mels
    for _ in range(blocks):
        reduced = (reduced + 1) // 2
    return nn.Sequential(*layers), incoming * reduced


class DualMelCTCTranscriber(nn.Module):
    def __init__(self, config: DualMelConfig = DualMelConfig()) -> None:
        super().__init__()
        self.config = config
        width = config.temporal_dim
        self.long_spectral, long_features = _branch(config.long_mels, config.conv_channels, 4)
        self.short_spectral, short_features = _branch(config.short_mels, config.conv_channels, 3)
        self.projection = nn.Sequential(
            nn.Conv1d(long_features + short_features, width, 1, bias=False),
            nn.GroupNorm(8, width), nn.SiLU(),
        )
        self.mel_skip = nn.Sequential(
            nn.Conv1d(config.n_mels, width, 1, bias=False), nn.GroupNorm(8, width), nn.SiLU(),
        )
        self.temporal = nn.Sequential(*[
            _DilatedBlock(width, 2 ** (index % config.dilation_cycle), config.kernel_size, config.dropout)
            for index in range(config.temporal_blocks)
        ])
        self.voiced_head = nn.Conv1d(width, 1, 1)
        self.pitch_head = nn.Conv1d(width, config.n_pitches, 1)
        self.onset_head = nn.Conv1d(width, 1, 1)
        self.boundary_head = nn.Conv1d(width, 1, 1)
        self.rearticulation_head = nn.Conv1d(width, 1, 1)
        self.confidence_head = nn.Conv1d(width, 1, 1)
        self.ctc_head = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=1), nn.SiLU(),
            nn.Conv1d(width, config.n_pitches + 1, 1),
        )

    @property
    def vocabulary(self) -> int:
        return self.config.n_pitches + 1

    def _spectral(self, branch: nn.Sequential, mel: Tensor) -> Tensor:
        value = branch(mel.unsqueeze(1))
        return value.permute(0, 1, 3, 2).reshape(value.shape[0], -1, value.shape[3])

    def forward(self, mel: Tensor) -> dict[str, Tensor]:
        mel = mel.float()
        long_mel = mel[:, :self.config.long_mels]
        short_mel = mel[:, self.config.long_mels:]
        features = torch.cat(
            [self._spectral(self.long_spectral, long_mel), self._spectral(self.short_spectral, short_mel)],
            dim=1,
        )
        value = self.projection(features) + self.mel_skip(mel)
        value = self.temporal(value)
        return {
            "voiced_logits": self.voiced_head(value).squeeze(1),
            "pitch_logits": self.pitch_head(value).transpose(1, 2),
            "onset_logits": self.onset_head(value).squeeze(1),
            "boundary_logits": self.boundary_head(value).squeeze(1),
            "rearticulation_logits": self.rearticulation_head(value).squeeze(1),
            "confidence_logits": self.confidence_head(value).squeeze(1),
            "ctc_logits": self.ctc_head(value).transpose(1, 2),
        }


def extract_dual_mel(audio: np.ndarray, device: torch.device | str) -> tuple[np.ndarray, dict[str, Any]]:
    long_mel, long_norm = extract_log_mel(audio, LONG_FRONTEND, device=device)
    short_mel, short_norm = extract_log_mel(audio, SHORT_FRONTEND, device=device)
    frames = min(long_mel.shape[1], short_mel.shape[1])
    stacked = np.concatenate([long_mel[:, :frames], short_mel[:, :frames]], axis=0)
    return stacked, {"long": long_norm, "short": short_norm}


def dual_frontend_metadata() -> dict[str, Any]:
    return {
        **LONG_FRONTEND.to_dict(),
        "n_mels": LONG_FRONTEND.n_mels + SHORT_FRONTEND.n_mels,
        "dual": {"long": LONG_FRONTEND.to_dict(), "short": SHORT_FRONTEND.to_dict()},
    }


@torch.inference_mode()
def infer_dual_outputs(
    model: DualMelCTCTranscriber,
    mel: np.ndarray,
    device: torch.device | str,
    *,
    window_frames: int = 4096,
    overlap_frames: int = 1024,
    batch_size: int = 2,
) -> dict[str, np.ndarray]:
    """Overlap-add CTC softmax and frame heads over a full clip."""

    array = np.asarray(mel, dtype=np.float32)
    total = array.shape[1]
    heads = ("voiced", "onset", "boundary", "rearticulation")
    vocabulary = model.vocabulary
    if total == 0:
        return {"ctc": np.zeros((0, vocabulary), np.float32), **{h: np.zeros(0, np.float32) for h in heads}}
    stride = window_frames - overlap_frames
    starts = list(range(0, max(total - window_frames, 0) + 1, stride))
    final = max(0, total - window_frames)
    if not starts or starts[-1] != final:
        starts.append(final)
    taper = np.maximum(np.hanning(window_frames + 2)[1:-1].astype(np.float32), 0.05)
    sums = {"ctc": np.zeros((total, vocabulary), np.float64), **{h: np.zeros(total, np.float64) for h in heads}}
    weights = np.zeros(total, np.float64)
    target = torch.device(device)
    model.eval()
    for group_start in range(0, len(starts), batch_size):
        group = starts[group_start:group_start + batch_size]
        length_max = min(window_frames, total)
        windows = np.zeros((len(group), array.shape[0], length_max), np.float32)
        lengths = []
        for index, start in enumerate(group):
            length = min(length_max, total - start)
            windows[index, :, :length] = array[:, start:start + length]
            lengths.append(length)
        output = model(torch.from_numpy(windows).to(target))
        values = {"ctc": output["ctc_logits"].float().softmax(dim=-1).cpu().numpy()}
        for head in heads:
            values[head] = torch.sigmoid(output[f"{head}_logits"]).float().cpu().numpy()
        for index, (start, length) in enumerate(zip(group, lengths)):
            weight = taper[:length] if total > window_frames else np.ones(length, np.float32)
            sums["ctc"][start:start + length] += values["ctc"][index, :length] * weight[:, None]
            for head in heads:
                sums[head][start:start + length] += values[head][index, :length] * weight
            weights[start:start + length] += weight
    denominator = np.maximum(weights, 1e-9)
    return {"ctc": (sums["ctc"] / denominator[:, None]).astype(np.float32),
            **{h: (sums[h] / denominator).astype(np.float32) for h in heads}}


def augment_dual_batch(
    mel: Tensor, onset: Tensor | None, *, long_mels: int, probability: float = 0.6,
) -> Tensor:
    """Realistic perturbations applied per branch with shared timing.

    Adds (i) vibrato / slow pitch drift as a time-varying shift along the mel
    axis, (ii) dynamic swells, (iii) reverb tails via a decaying running max,
    (iv) occasional deeper articulation dips just before annotated onsets, plus
    the v1 gain/EQ/noise/band-mask perturbations.
    """

    from .mel_v1_data import augment_mel_batch

    batch, bands, frames = mel.shape
    device = mel.device
    result = mel.clone()
    time = torch.arange(frames, device=device, dtype=torch.float32)[None, :] * 0.011610
    # (i) vibrato + drift: shift in long-mel bands, scaled for the short branch.
    active = (torch.rand(batch, 1, device=device) < probability * 0.6).float()
    rate = torch.empty(batch, 1, device=device).uniform_(4.0, 7.0)
    depth = torch.empty(batch, 1, device=device).uniform_(0.1, 0.6)
    drift = torch.cumsum(torch.randn(batch, frames, device=device) * 0.01, dim=1)
    drift = drift - drift.mean(dim=1, keepdim=True)
    shift = active * (depth * torch.sin(2 * torch.pi * rate * time + torch.rand(batch, 1, device=device) * 6.28)
                      + drift.clamp(-0.6, 0.6))
    for lo, hi, scale in ((0, long_mels, 1.0), (long_mels, bands, 0.5)):
        segment = result[:, lo:hi]
        count = hi - lo
        positions = torch.arange(count, device=device, dtype=torch.float32)[None, :, None]
        source = (positions - scale * shift[:, None, :]).clamp(0, count - 1)
        low = source.floor().long()
        high = (low + 1).clamp(max=count - 1)
        frac = source - low.float()
        gathered_low = torch.gather(segment, 1, low)
        gathered_high = torch.gather(segment, 1, high)
        result[:, lo:hi] = gathered_low * (1 - frac) + gathered_high * frac
    # (ii) swells: slow gain curve shared by both branches.
    active = (torch.rand(batch, 1, 1, device=device) < probability * 0.5).float()
    curve = torch.zeros(batch, frames, device=device)
    for _ in range(3):
        frequency = torch.empty(batch, 1, device=device).uniform_(0.15, 1.5)
        phase = torch.rand(batch, 1, device=device) * 6.28
        curve = curve + torch.sin(2 * torch.pi * frequency * time + phase)
    amplitude = torch.empty(batch, 1, 1, device=device).uniform_(0.05, 0.25)
    result = result + active * amplitude * (curve / 3.0)[:, None, :]
    # (iii) reverb tail: decaying running max (applied frame-recursively on a coarse grid).
    active = torch.rand(batch, device=device) < probability * 0.4
    if bool(active.any()):
        decay = torch.empty(int(active.sum()), 1, device=device).uniform_(0.015, 0.06)
        mix = torch.empty(int(active.sum()), 1, 1, device=device).uniform_(0.3, 0.8)
        chosen = result[active]
        tail = chosen.clone()
        for frame in range(1, frames):
            tail[:, :, frame] = torch.maximum(chosen[:, :, frame], tail[:, :, frame - 1] - decay)
        result[active] = chosen + mix * (tail - chosen)
    # (iv) deeper articulation dips before annotated onsets.
    if onset is not None:
        active = (torch.rand(batch, 1, device=device) < probability * 0.3).float()
        marks = (onset > 0.5).float() * active
        dip = torch.zeros_like(marks)
        for back in range(1, 4):
            dip[:, :-back] = torch.maximum(dip[:, :-back], marks[:, back:] * (1.0 - 0.25 * (back - 1)))
        depth = torch.empty(batch, 1, device=device).uniform_(0.15, 0.5)
        result = result - (dip * depth)[:, None, :]
    # v1 perturbations per branch.
    result[:, :long_mels] = augment_mel_batch(result[:, :long_mels], probability=probability)
    result[:, long_mels:] = augment_mel_batch(result[:, long_mels:], probability=probability)
    return result.clamp(-4.5, 3.5)


def save_dual_checkpoint(path: Path, model: DualMelCTCTranscriber, extra: Mapping[str, Any]) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "model_config": model.config.to_dict(),
        "frontend": dual_frontend_metadata(),
        **dict(extra),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_dual_checkpoint(path: Path | str, device: torch.device | str) -> tuple[DualMelCTCTranscriber, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported v3 checkpoint")
    model = DualMelCTCTranscriber(DualMelConfig.from_dict(payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload
