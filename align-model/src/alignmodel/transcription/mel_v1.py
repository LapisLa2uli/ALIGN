"""High-resolution, score-free clarinet note transcriber.

This module is deliberately independent of Basic Pitch.  Audio inference uses
only a waveform and immutable instrument/pitch-space metadata; score-aware
mapping is a separate downstream evaluation step.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import librosa
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


SCHEMA_VERSION = "align-mel-transcriber-v1"
CACHE_SCHEMA_VERSION = "align-mel-packed-cache-v1"


@dataclass(frozen=True)
class MelFrontendConfig:
    sample_rate: int = 22050
    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 256
    n_mels: int = 128
    fmin: float = 45.0
    fmax: float = 8000.0
    power_floor: float = 1e-10
    normalization: str = "clip-q05-q50-q95-v1"

    @property
    def hop_sec(self) -> float:
        return self.hop_length / self.sample_rate

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping | None) -> "MelFrontendConfig":
        allowed = {item.name for item in fields(cls)}
        return cls(**{
            key: item for key, item in dict(value or {}).items() if key in allowed
        })


def load_audio_mono(path: Path | str, sample_rate: int) -> np.ndarray:
    """Load audio without modifying or writing the source bundle."""

    audio, _ = librosa.load(
        str(Path(path)), sr=sample_rate, mono=True, dtype=np.float32
    )
    if not np.isfinite(audio).all():
        raise ValueError(f"Non-finite audio samples: {path}")
    return np.asarray(audio, dtype=np.float32)


def extract_log_mel(
    audio: np.ndarray | Tensor,
    config: MelFrontendConfig = MelFrontendConfig(),
    *,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, dict[str, float]]:
    """Extract a robustly normalized log-mel while retaining weak attacks.

    There is no top-dB clipping.  One clip-wide affine normalization uses
    robust quantiles, so quiet attacks retain their local contrast instead of
    being erased by frame-wise normalization or a hard dynamic-range floor.
    """

    target = torch.device(device)
    waveform = torch.as_tensor(audio, dtype=torch.float32, device=target).flatten()
    if waveform.numel() == 0:
        raise ValueError("Cannot extract a mel spectrogram from empty audio")
    window = torch.hann_window(
        config.win_length, periodic=True, dtype=torch.float32, device=target
    )
    spectrum = torch.stft(
        waveform,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    power = spectrum.abs().square()
    mel_filter = librosa.filters.mel(
        sr=config.sample_rate,
        n_fft=config.n_fft,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
        htk=False,
        norm="slaney",
        dtype=np.float32,
    )
    mel = torch.as_tensor(mel_filter, device=target) @ power
    log_mel = 10.0 * torch.log10(mel.clamp_min(config.power_floor))
    flat = log_mel.flatten()
    quantiles = torch.quantile(
        flat, torch.tensor([0.05, 0.50, 0.95], device=target)
    )
    floor, center, high = (float(value) for value in quantiles)
    scale = max(high - floor, 12.0)
    normalized = ((log_mel - center) / scale).clamp(-4.5, 3.5)
    result = normalized.to(torch.float16).cpu().numpy()
    return result, {
        "q05_db": floor,
        "q50_db": center,
        "q95_db": high,
        "scale_db": scale,
    }


@dataclass(frozen=True)
class MelTranscriberConfig:
    n_mels: int = 128
    midi_min: int = 52
    midi_max: int = 100
    conv_channels: int = 32
    temporal_dim: int = 128
    spectral_blocks: int = 4
    temporal_blocks: int = 8
    temporal_kind: str = "tcn"
    gru_layers: int = 2
    dropout: float = 0.10
    use_mel_skip: bool = True

    @property
    def n_pitches(self) -> int:
        return self.midi_max - self.midi_min + 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping | None) -> "MelTranscriberConfig":
        supplied = dict(value or {})
        # Checkpoints produced before the direct high-resolution skip existed
        # reconstruct the exact earlier architecture.
        if supplied and "use_mel_skip" not in supplied:
            supplied["use_mel_skip"] = False
        allowed = {item.name for item in fields(cls)}
        return cls(**{
            key: item for key, item in supplied.items() if key in allowed
        })


class _SpectralBlock(nn.Module):
    """Downsample frequency only; the high-resolution time axis is immutable."""

    def __init__(self, incoming: int, outgoing: int) -> None:
        super().__init__()
        groups = min(8, outgoing)
        while outgoing % groups:
            groups -= 1
        self.network = nn.Sequential(
            nn.Conv2d(
                incoming, outgoing, kernel_size=3, stride=(2, 1),
                padding=1, bias=False,
            ),
            nn.GroupNorm(groups, outgoing),
            nn.SiLU(),
            nn.Conv2d(
                outgoing, outgoing, kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(groups, outgoing),
            nn.SiLU(),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.network(value)


class _TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.network = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=5,
                dilation=dilation,
                padding=2 * dilation,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels * 2, 1),
            nn.GLU(dim=1),
            nn.Dropout(dropout),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.network(value)


class MelNoteTranscriber(nn.Module):
    """Time-preserving spectral encoder plus an offline temporal model."""

    def __init__(
        self, config: MelTranscriberConfig = MelTranscriberConfig()
    ) -> None:
        super().__init__()
        self.config = config
        if config.temporal_kind not in {"tcn", "bigru"}:
            raise ValueError("temporal_kind must be 'tcn' or 'bigru'")
        blocks = []
        incoming = 1
        for index in range(config.spectral_blocks):
            outgoing = config.conv_channels if index else config.conv_channels // 2
            blocks.append(_SpectralBlock(incoming, outgoing))
            incoming = outgoing
        self.spectral = nn.Sequential(*blocks)
        reduced = config.n_mels
        for _ in range(config.spectral_blocks):
            reduced = (reduced + 1) // 2
        self.reduced_mels = reduced
        self.projection = nn.Sequential(
            nn.Conv1d(incoming * reduced, config.temporal_dim, 1, bias=False),
            nn.GroupNorm(8 if config.temporal_dim % 8 == 0 else 1, config.temporal_dim),
            nn.SiLU(),
        )
        self.mel_skip = (
            nn.Sequential(
                nn.Conv1d(config.n_mels, config.temporal_dim, 1, bias=False),
                nn.GroupNorm(
                    8 if config.temporal_dim % 8 == 0 else 1,
                    config.temporal_dim,
                ),
                nn.SiLU(),
            )
            if config.use_mel_skip else None
        )
        if config.temporal_kind == "tcn":
            self.temporal = nn.Sequential(*[
                _TCNBlock(
                    config.temporal_dim,
                    2 ** (index % 5),
                    config.dropout,
                )
                for index in range(config.temporal_blocks)
            ])
            temporal_output = config.temporal_dim
        else:
            hidden = max(32, config.temporal_dim // 2)
            self.temporal = nn.GRU(
                config.temporal_dim,
                hidden,
                num_layers=config.gru_layers,
                dropout=config.dropout if config.gru_layers > 1 else 0.0,
                bidirectional=True,
                batch_first=True,
            )
            temporal_output = hidden * 2
        self.voiced_head = nn.Conv1d(temporal_output, 1, 1)
        self.pitch_head = nn.Conv1d(
            temporal_output, config.n_pitches, 1
        )
        self.onset_head = nn.Conv1d(temporal_output, 1, 1)
        self.boundary_head = nn.Conv1d(temporal_output, 1, 1)
        self.rearticulation_head = nn.Conv1d(temporal_output, 1, 1)
        self.confidence_head = nn.Conv1d(temporal_output, 1, 1)

    def forward(self, mel: Tensor) -> dict[str, Tensor]:
        if mel.ndim != 3 or mel.shape[1] != self.config.n_mels:
            raise ValueError(
                f"Expected [B,{self.config.n_mels},T], got {tuple(mel.shape)}"
            )
        mel_value = mel.float()
        value = self.spectral(mel_value.unsqueeze(1))
        value = value.permute(0, 1, 3, 2).reshape(
            value.shape[0], -1, value.shape[3]
        )
        value = self.projection(value)
        if self.mel_skip is not None:
            value = value + self.mel_skip(mel_value)
        if self.config.temporal_kind == "bigru":
            value, _ = self.temporal(value.transpose(1, 2))
            value = value.transpose(1, 2)
        else:
            value = self.temporal(value)
        return {
            "voiced_logits": self.voiced_head(value).squeeze(1),
            "pitch_logits": self.pitch_head(value).transpose(1, 2),
            "onset_logits": self.onset_head(value).squeeze(1),
            "boundary_logits": self.boundary_head(value).squeeze(1),
            "rearticulation_logits": self.rearticulation_head(value).squeeze(1),
            "confidence_logits": self.confidence_head(value).squeeze(1),
        }


def make_mel_targets(
    notes: Iterable[Mapping | Sequence],
    *,
    frames: int,
    hop_sec: float,
    midi_min: int,
    midi_max: int,
    crop_start: int = 0,
) -> dict[str, np.ndarray]:
    """Rasterize exact rendered lineage and retain segment identity."""

    size = max(0, int(frames))
    voiced = np.zeros(size, np.float32)
    pitch = np.full(size, -1, np.int64)
    onset = np.zeros(size, np.float32)
    boundary = np.zeros(size, np.float32)
    rearticulation = np.zeros(size, np.float32)
    confidence = np.zeros(size, np.float32)
    event_id = np.full(size, -1, np.int64)
    duration_weight = np.ones(size, np.float32)
    previous_pitch: int | None = None
    absolute_end = crop_start + size
    for index, raw in enumerate(notes):
        if isinstance(raw, Mapping):
            midi = int(raw.get("pitch", raw.get("pitch_midi_written")))
            start = float(raw.get("start_sec", raw.get("start", 0.0)))
            end = float(raw.get("end_sec", raw.get("end", start)))
        else:
            midi, start, end = int(raw[0]), float(raw[1]), float(raw[2])
        first = max(0, int(math.floor(start / hop_sec)))
        last = max(first + 1, int(math.ceil(end / hop_sec)))
        local_first = max(first, crop_start) - crop_start
        local_last = min(last, absolute_end) - crop_start
        duration = max(0.0, end - start)
        weight = (
            4.0 if duration < 0.080 else
            2.5 if duration < 0.120 else
            1.6 if duration < 0.180 else 1.0
        )
        if local_first < local_last:
            voiced[local_first:local_last] = 1.0
            confidence[local_first:local_last] = 1.0
            duration_weight[local_first:local_last] = weight
            event_id[local_first:local_last] = index
            if midi_min <= midi <= midi_max:
                pitch[local_first:local_last] = midi - midi_min
        onset_frame = int(round(start / hop_sec))
        local_onset = onset_frame - crop_start
        if 0 <= local_onset < size:
            onset[local_onset] = 1.0
            boundary[local_onset] = 1.0
            duration_weight[local_onset] = max(
                duration_weight[local_onset], weight
            )
            if previous_pitch == midi:
                rearticulation[local_onset] = 1.0
        previous_pitch = midi
    return {
        "voiced": voiced,
        "pitch": pitch,
        "onset": onset,
        "boundary": boundary,
        "rearticulation": rearticulation,
        "confidence": confidence,
        "event_id": event_id,
        "duration_weight": duration_weight,
    }


def _focal_bce(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    positive_weight: float,
    gamma: float = 2.0,
    sample_weight: Tensor | None = None,
) -> Tensor:
    values = target.float()
    base = F.binary_cross_entropy_with_logits(
        logits, values, reduction="none",
        pos_weight=logits.new_tensor(positive_weight),
    )
    probability = torch.sigmoid(logits)
    focal = torch.where(values > 0.5, 1.0 - probability, probability).pow(gamma)
    weight = mask.float() * (sample_weight if sample_weight is not None else 1.0)
    return (base * focal * weight).sum() / weight.sum().clamp_min(1.0)


def mel_transcriber_loss(
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    *,
    pitch_class_weight: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Duration-aware frame and segment objective for one-event-per-note output."""

    valid = batch["frame_mask"].bool()
    duration_weight = batch["duration_weight"].float()
    voiced = _focal_bce(
        outputs["voiced_logits"], batch["voiced"], valid,
        positive_weight=1.0, gamma=0.0, sample_weight=duration_weight,
    )
    onset = _focal_bce(
        outputs["onset_logits"], batch["onset"], valid,
        positive_weight=18.0, gamma=2.0, sample_weight=duration_weight,
    )
    boundary = _focal_bce(
        outputs["boundary_logits"], batch["boundary"], valid,
        positive_weight=14.0, gamma=2.0, sample_weight=duration_weight,
    )
    rearticulation = _focal_bce(
        outputs["rearticulation_logits"], batch["rearticulation"], valid,
        positive_weight=24.0, gamma=2.0, sample_weight=duration_weight,
    )
    confidence = _focal_bce(
        outputs["confidence_logits"], batch["confidence"], valid,
        positive_weight=1.0, gamma=0.0, sample_weight=duration_weight,
    )
    pitch_mask = valid & (batch["pitch"] >= 0) & (batch["voiced"] > 0)
    pitch_values = F.cross_entropy(
        outputs["pitch_logits"].transpose(1, 2),
        batch["pitch"].long().masked_fill(~pitch_mask, -1),
        ignore_index=-1,
        reduction="none",
        label_smoothing=0.03,
        weight=pitch_class_weight,
    )
    pitch_weight = duration_weight * pitch_mask.float()
    pitch = (pitch_values * pitch_weight).sum() / pitch_weight.sum().clamp_min(1.0)

    event_ids = batch["event_id"].long()
    if bool(pitch_mask.any()):
        # Give every batch row a disjoint event-id range, then pool every gold
        # segment with one set of GPU scatter operations (no per-event sync).
        event_stride = event_ids.shape[1] + 1
        batch_offset = torch.arange(
            event_ids.shape[0], device=event_ids.device
        )[:, None] * event_stride
        global_ids = (event_ids + batch_offset)[pitch_mask]
        _, inverse = torch.unique(global_ids, sorted=True, return_inverse=True)
        event_count = int(inverse.max().item()) + 1
        logits_flat = outputs["pitch_logits"][pitch_mask].float()
        pooled_sum = logits_flat.new_zeros(
            event_count, logits_flat.shape[-1]
        )
        pooled_sum.index_add_(0, inverse, logits_flat)
        counts = torch.bincount(inverse, minlength=event_count).clamp_min(1)
        pooled = pooled_sum / counts[:, None]
        target_sum = torch.zeros(
            event_count, dtype=torch.float32, device=logits_flat.device
        )
        target_sum.index_add_(0, inverse, batch["pitch"][pitch_mask].float())
        target_pitch = (target_sum / counts).round().long()
        event_weight = torch.zeros(
            event_count, dtype=torch.float32, device=logits_flat.device
        )
        event_weight.scatter_reduce_(
            0, inverse, duration_weight[pitch_mask],
            reduce="amax", include_self=False,
        )
        if pitch_class_weight is not None:
            event_weight = event_weight * pitch_class_weight[target_pitch]
        segment = (
            F.cross_entropy(pooled, target_pitch, reduction="none")
            * event_weight
        ).mean()
    else:
        segment = outputs["voiced_logits"].sum() * 0.0

    voice_probability = torch.sigmoid(outputs["voiced_logits"])
    local_support = F.avg_pool1d(
        voice_probability.unsqueeze(1), kernel_size=7, stride=1, padding=3
    ).squeeze(1)
    isolated = (
        F.relu(voice_probability - local_support - 0.20).square()
        * valid.float()
        * (1.0 - batch["voiced"].float())
    ).sum() / valid.sum().clamp_min(1)
    # Suppress internal boundaries within a gold segment.  True same-pitch
    # rearticulations remain positive via the explicit boundary targets.
    same_event = (
        (event_ids[:, 1:] == event_ids[:, :-1])
        & (event_ids[:, 1:] >= 0)
        & valid[:, 1:]
        & valid[:, :-1]
    )
    internal_boundary = (
        torch.sigmoid(outputs["boundary_logits"][:, 1:]) * same_event.float()
    ).sum() / same_event.sum().clamp_min(1)

    total = (
        voiced
        + pitch
        + 2.2 * onset
        + 1.6 * boundary
        + 1.2 * rearticulation
        + 0.35 * confidence
        + 0.35 * segment
        + 0.15 * isolated
        + 0.10 * internal_boundary
    )
    parts = {
        "loss": total.detach(),
        "voiced": voiced.detach(),
        "pitch": pitch.detach(),
        "onset": onset.detach(),
        "boundary": boundary.detach(),
        "rearticulation": rearticulation.detach(),
        "confidence": confidence.detach(),
        "segment": segment.detach(),
        "min_duration": isolated.detach(),
        "merge_consistency": internal_boundary.detach(),
    }
    return total, parts


@dataclass(frozen=True)
class MelDecodeConfig:
    voice_on: float = 0.52
    voice_off: float = 0.35
    onset_threshold: float = 0.48
    boundary_threshold: float = 0.46
    strong_rearticulation: float = 0.64
    min_note_sec: float = 0.040
    merge_gap_sec: float = 0.045
    pitch_change_frames: int = 4
    alternatives: int = 3
    min_confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping | None) -> "MelDecodeConfig":
        allowed = {item.name for item in fields(cls)}
        return cls(**{
            key: item for key, item in dict(value or {}).items() if key in allowed
        })


@dataclass(frozen=True)
class MelNote:
    pitch: int
    start: float
    end: float
    confidence: float
    pitch_candidates: tuple[int, ...]
    candidate_confidences: tuple[float, ...]
    onset_strength: float
    boundary_strength: float

    def to_dict(self) -> dict:
        return asdict(self)


def _local_peaks(values: np.ndarray, threshold: float) -> list[int]:
    if not len(values):
        return []
    left = np.r_[-np.inf, values[:-1]]
    right = np.r_[values[1:], -np.inf]
    return np.flatnonzero(
        (values >= threshold) & (values >= left) & (values >= right)
    ).tolist()


def decode_mel_notes(
    probabilities: Mapping[str, np.ndarray],
    *,
    midi_min: int,
    hop_sec: float,
    config: MelDecodeConfig = MelDecodeConfig(),
) -> list[MelNote]:
    """Monophonic hysteresis decoder with merge-aware same-pitch handling."""

    voice = np.asarray(probabilities["voiced"], np.float32)
    pitch = np.asarray(probabilities["pitch"], np.float32)
    onset = np.asarray(probabilities["onset"], np.float32)
    boundary = np.asarray(probabilities["boundary"], np.float32)
    reart = np.asarray(probabilities["rearticulation"], np.float32)
    confidence = np.asarray(probabilities["confidence"], np.float32)
    frames = len(voice)
    if pitch.ndim != 2 or pitch.shape[0] != frames:
        raise ValueError("pitch must have shape [frames, pitches]")
    if any(len(value) != frames for value in (onset, boundary, reart, confidence)):
        raise ValueError("probability heads have different frame counts")

    runs: list[tuple[int, int]] = []
    active = False
    start = 0
    for index, value in enumerate(voice):
        if not active and value >= config.voice_on:
            active, start = True, index
        elif active and value < config.voice_off:
            runs.append((start, index))
            active = False
    if active:
        runs.append((start, frames))

    minimum = max(1, int(round(config.min_note_sec / hop_sec)))
    classes = pitch.argmax(axis=1) if frames else np.zeros(0, np.int64)
    proposed: list[MelNote] = []
    combined_boundary = np.maximum.reduce((onset, boundary, reart))
    peaks = _local_peaks(
        combined_boundary,
        min(config.onset_threshold, config.boundary_threshold),
    )
    for run_start, run_end in runs:
        split_values = {run_start, run_end}
        for peak in peaks:
            if not run_start + minimum <= peak <= run_end - minimum:
                continue
            if (
                onset[peak] >= config.onset_threshold
                or boundary[peak] >= config.boundary_threshold
                or reart[peak] >= config.strong_rearticulation
            ):
                split_values.add(peak)
        cursor = run_start + 1
        while cursor + config.pitch_change_frames <= run_end:
            new_class = int(classes[cursor])
            if (
                new_class != int(classes[cursor - 1])
                and np.all(
                    classes[cursor:cursor + config.pitch_change_frames]
                    == new_class
                )
                and cursor - run_start >= minimum
            ):
                split_values.add(cursor)
                cursor += config.pitch_change_frames
            else:
                cursor += 1
        splits = sorted(split_values)
        for first, last in zip(splits, splits[1:]):
            if last - first < minimum:
                continue
            weights = np.maximum(voice[first:last], 0.05)
            pitch_scores = (
                pitch[first:last] * weights[:, None]
            ).sum(axis=0) / weights.sum()
            ranked = np.argsort(pitch_scores)[::-1][
                :max(1, config.alternatives)
            ]
            chosen = int(ranked[0])
            onset_strength = float(
                onset[max(run_start, first - 1):min(run_end, first + 2)].max(
                    initial=0.0
                )
            )
            boundary_strength = float(
                max(boundary[first], reart[first], onset_strength)
            )
            score = float(np.clip(
                0.32 * voice[first:last].mean()
                + 0.38 * pitch_scores[chosen]
                + 0.20 * confidence[first:last].mean()
                + 0.10 * onset_strength,
                0.0,
                1.0,
            ))
            proposed.append(MelNote(
                pitch=midi_min + chosen,
                start=round(first * hop_sec, 6),
                end=round(last * hop_sec, 6),
                confidence=round(score, 6),
                pitch_candidates=tuple(midi_min + int(item) for item in ranked),
                candidate_confidences=tuple(
                    round(float(pitch_scores[item]), 6) for item in ranked
                ),
                onset_strength=round(onset_strength, 6),
                boundary_strength=round(boundary_strength, 6),
            ))

    merged: list[MelNote] = []
    for note in proposed:
        previous = merged[-1] if merged else None
        weak_boundary = (
            note.boundary_strength < config.strong_rearticulation
            and note.onset_strength < config.strong_rearticulation
        )
        if (
            previous is not None
            and previous.pitch == note.pitch
            and note.start - previous.end <= config.merge_gap_sec
            and weak_boundary
        ):
            duration_a = previous.end - previous.start
            duration_b = note.end - note.start
            merged[-1] = MelNote(
                pitch=previous.pitch,
                start=previous.start,
                end=note.end,
                confidence=round(
                    (previous.confidence * duration_a + note.confidence * duration_b)
                    / max(duration_a + duration_b, 1e-6),
                    6,
                ),
                pitch_candidates=previous.pitch_candidates,
                candidate_confidences=previous.candidate_confidences,
                onset_strength=previous.onset_strength,
                boundary_strength=previous.boundary_strength,
            )
        else:
            merged.append(note)
    return [
        note for note in merged
        if note.confidence >= config.min_confidence
    ]


@torch.inference_mode()
def infer_mel_probabilities(
    model: MelNoteTranscriber,
    mel: np.ndarray,
    device: torch.device | str,
    *,
    window_frames: int = 2048,
    overlap_frames: int = 512,
    batch_size: int = 4,
) -> dict[str, np.ndarray]:
    """Overlap-add full-clip inference without changing temporal resolution."""

    array = np.asarray(mel, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != model.config.n_mels:
        raise ValueError(
            f"Expected [{model.config.n_mels},T], got {array.shape}"
        )
    total = array.shape[1]
    names = ("voiced", "onset", "boundary", "rearticulation", "confidence")
    if total == 0:
        return {
            **{name: np.zeros(0, np.float32) for name in names},
            "pitch": np.zeros((0, model.config.n_pitches), np.float32),
        }
    if not 0 <= overlap_frames < window_frames:
        raise ValueError("Invalid inference overlap")
    stride = window_frames - overlap_frames
    starts = list(range(0, max(total - window_frames, 0) + 1, stride))
    final = max(0, total - window_frames)
    if not starts or starts[-1] != final:
        starts.append(final)
    taper = np.maximum(
        np.hanning(window_frames + 2)[1:-1].astype(np.float32), 0.05
    )
    sums = {name: np.zeros(total, np.float64) for name in names}
    sums["pitch"] = np.zeros(
        (total, model.config.n_pitches), np.float64
    )
    weights = np.zeros(total, np.float64)
    target = torch.device(device)
    model.eval()
    for group_start in range(0, len(starts), batch_size):
        group = starts[group_start:group_start + batch_size]
        windows = np.zeros(
            (len(group), model.config.n_mels, window_frames), np.float32
        )
        lengths = []
        for batch_index, start_frame in enumerate(group):
            length = min(window_frames, total - start_frame)
            windows[batch_index, :, :length] = array[
                :, start_frame:start_frame + length
            ]
            lengths.append(length)
        output = model(torch.from_numpy(windows).to(target))
        probability = {
            name: torch.sigmoid(output[f"{name}_logits"]).float().cpu().numpy()
            for name in names
        }
        probability["pitch"] = (
            output["pitch_logits"].softmax(dim=-1).float().cpu().numpy()
        )
        for batch_index, (start_frame, length) in enumerate(zip(group, lengths)):
            section = slice(start_frame, start_frame + length)
            weight = taper[:length]
            weights[section] += weight
            for name in names:
                sums[name][section] += probability[name][batch_index, :length] * weight
            sums["pitch"][section] += (
                probability["pitch"][batch_index, :length] * weight[:, None]
            )
    denominator = np.maximum(weights, 1e-8)
    return {
        **{
            name: (sums[name] / denominator).astype(np.float32)
            for name in names
        },
        "pitch": (sums["pitch"] / denominator[:, None]).astype(np.float32),
    }


def load_mel_checkpoint(
    path: Path | str,
    device: torch.device | str = "cpu",
) -> tuple[
    MelNoteTranscriber,
    MelFrontendConfig,
    MelDecodeConfig,
    dict,
]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported mel transcriber checkpoint")
    model = MelNoteTranscriber(
        MelTranscriberConfig.from_dict(payload["model_config"])
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return (
        model,
        MelFrontendConfig.from_dict(payload["frontend_config"]),
        MelDecodeConfig.from_dict(payload.get("decode_config")),
        payload,
    )

