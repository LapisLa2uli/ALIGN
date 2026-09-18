"""Polyphonic-safe high-resolution transcriber for rendered ornaments."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .mel_v1 import MelNoteTranscriber, MelTranscriberConfig
from .mel_v1_data import MelCacheRecord


SCHEMA_VERSION = "align-ornament-multipitch-transcriber-v1"


@dataclass(frozen=True)
class MultiPitchConfig:
    backbone: MelTranscriberConfig = MelTranscriberConfig()
    max_polyphony: int = 8
    dropout: float = 0.10

    def to_dict(self) -> dict[str, Any]:
        return {
            "backbone": self.backbone.to_dict(),
            "max_polyphony": self.max_polyphony,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MultiPitchConfig":
        return cls(
            backbone=MelTranscriberConfig.from_dict(value.get("backbone")),
            max_polyphony=int(value.get("max_polyphony", 8)),
            dropout=float(value.get("dropout", 0.10)),
        )


class OrnamentMultiPitchTranscriber(nn.Module):
    """Track-B encoder with independent pitch activity/onset/offset heads."""

    def __init__(self, config: MultiPitchConfig = MultiPitchConfig()) -> None:
        super().__init__()
        self.config = config
        self.backbone = MelNoteTranscriber(config.backbone)
        channels = (
            config.backbone.temporal_dim
            if config.backbone.temporal_kind == "tcn"
            else max(32, config.backbone.temporal_dim // 2) * 2
        )
        pitches = config.backbone.n_pitches
        self.refinement = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8 if channels % 8 == 0 else 1, channels),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.activity_head = nn.Conv1d(channels, pitches, 1)
        self.onset_head = nn.Conv1d(channels, pitches, 1)
        self.offset_head = nn.Conv1d(channels, pitches, 1)
        self.polyphony_head = nn.Conv1d(
            channels, config.max_polyphony + 1, 1
        )

    @property
    def midi_min(self) -> int:
        return self.config.backbone.midi_min

    @property
    def midi_max(self) -> int:
        return self.config.backbone.midi_max

    def initialize_track_b(self, track_b: MelNoteTranscriber) -> None:
        if track_b.config != self.config.backbone:
            raise ValueError("Track B and multi-pitch backbone configs differ")
        self.backbone.load_state_dict(track_b.state_dict())

    def forward(self, mel: Tensor) -> dict[str, Tensor]:
        encoded = self.backbone.encode(mel)
        encoded = encoded + self.refinement(encoded)
        return {
            "activity_logits": self.activity_head(encoded).transpose(1, 2),
            "onset_logits": self.onset_head(encoded).transpose(1, 2),
            "offset_logits": self.offset_head(encoded).transpose(1, 2),
            "polyphony_logits": self.polyphony_head(encoded).transpose(1, 2),
        }


def make_multipitch_targets(
    notes: Iterable[Mapping[str, Any] | Sequence[Any]],
    *,
    frames: int,
    hop_sec: float,
    midi_min: int,
    midi_max: int,
    crop_start: int = 0,
) -> dict[str, np.ndarray]:
    size = max(0, int(frames))
    pitches = midi_max - midi_min + 1
    activity = np.zeros((size, pitches), np.float32)
    onset = np.zeros((size, pitches), np.float32)
    offset = np.zeros((size, pitches), np.float32)
    duration_weight = np.ones((size, pitches), np.float32)
    absolute_end = crop_start + size
    for raw in notes:
        if isinstance(raw, Mapping):
            midi = int(
                raw.get(
                    "pitch",
                    raw.get("pitch_midi_written"),
                )
            )
            start = float(raw.get("start_sec", raw.get("start", 0.0)))
            end = float(raw.get("end_sec", raw.get("end", start)))
        else:
            midi, start, end = int(raw[0]), float(raw[1]), float(raw[2])
        pitch_index = midi - midi_min
        if not 0 <= pitch_index < pitches or end <= start:
            continue
        first = max(0, int(math.floor(start / hop_sec)))
        last = max(first + 1, int(math.ceil(end / hop_sec)))
        local_first = max(first, crop_start) - crop_start
        local_last = min(last, absolute_end) - crop_start
        duration = end - start
        weight = (
            5.0
            if duration < 0.080
            else 3.0
            if duration < 0.120
            else 1.8
            if duration < 0.180
            else 1.0
        )
        if local_first < local_last:
            activity[local_first:local_last, pitch_index] = 1.0
            duration_weight[local_first:local_last, pitch_index] = np.maximum(
                duration_weight[local_first:local_last, pitch_index], weight
            )
        onset_frame = int(round(start / hop_sec)) - crop_start
        if 0 <= onset_frame < size:
            onset[onset_frame, pitch_index] = 1.0
            duration_weight[onset_frame, pitch_index] = max(
                duration_weight[onset_frame, pitch_index], weight
            )
        offset_frame = int(round(end / hop_sec)) - crop_start
        if 0 <= offset_frame < size:
            offset[offset_frame, pitch_index] = 1.0
            duration_weight[offset_frame, pitch_index] = max(
                duration_weight[offset_frame, pitch_index], weight
            )
    polyphony = np.minimum(
        activity.sum(axis=1), max(0, pitches)
    ).astype(np.int64)
    return {
        "activity": activity,
        "onset": onset,
        "offset": offset,
        "polyphony": polyphony,
        "duration_weight": duration_weight,
    }


def _focal_bce(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    positive_weight: float,
    gamma: float,
    duration_weight: Tensor,
) -> Tensor:
    values = target.float()
    base = F.binary_cross_entropy_with_logits(
        logits,
        values,
        reduction="none",
        pos_weight=logits.new_tensor(positive_weight),
    )
    probability = torch.sigmoid(logits)
    focal = torch.where(
        values > 0.5, 1.0 - probability, probability
    ).pow(gamma)
    weight = mask.float() * torch.where(
        values > 0.5, duration_weight.float(), 1.0
    )
    return (base * focal * weight).sum() / weight.sum().clamp_min(1.0)


def multipitch_loss(
    output: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
) -> tuple[Tensor, dict[str, Tensor]]:
    frame_mask = batch["frame_mask"].bool().unsqueeze(-1)
    duration_weight = batch["duration_weight"].float()
    activity = _focal_bce(
        output["activity_logits"],
        batch["activity"],
        frame_mask,
        positive_weight=3.0,
        gamma=1.0,
        duration_weight=duration_weight,
    )
    onset = _focal_bce(
        output["onset_logits"],
        batch["onset"],
        frame_mask,
        positive_weight=32.0,
        gamma=2.0,
        duration_weight=duration_weight,
    )
    offset = _focal_bce(
        output["offset_logits"],
        batch["offset"],
        frame_mask,
        positive_weight=20.0,
        gamma=2.0,
        duration_weight=duration_weight,
    )
    polyphony_target = batch["polyphony"].long().clamp(
        max=output["polyphony_logits"].shape[-1] - 1
    )
    polyphony_values = F.cross_entropy(
        output["polyphony_logits"].transpose(1, 2),
        polyphony_target.masked_fill(~batch["frame_mask"].bool(), -1),
        ignore_index=-1,
        reduction="none",
    )
    polyphony = (
        polyphony_values * batch["frame_mask"].float()
    ).sum() / batch["frame_mask"].float().sum().clamp_min(1.0)
    total = activity + 0.75 * onset + 0.45 * offset + 0.20 * polyphony
    return total, {
        "total": total.detach(),
        "activity": activity.detach(),
        "onset": onset.detach(),
        "offset": offset.detach(),
        "polyphony": polyphony.detach(),
    }


class MultiPitchCropDataset(Dataset):
    """Deterministic crop loader for an immutable packed mel cache."""

    def __init__(
        self,
        cache_root: Path | str,
        records: Sequence[MelCacheRecord],
        *,
        crop_frames: int,
        epoch: int,
        seed: int,
        crops_per_clip: int = 1,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.records = tuple(records)
        self.crop_frames = int(crop_frames)
        self.epoch = int(epoch)
        self.seed = int(seed)
        self.crops_per_clip = max(1, int(crops_per_clip))
        metadata = json.loads(
            (self.cache_root / "metadata.json").read_text(encoding="utf-8")
        )
        from .mel_v1 import MelFrontendConfig

        self.frontend = MelFrontendConfig.from_dict(
            metadata["frontend_config"]
        )
        self._maps: dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return len(self.records) * self.crops_per_clip

    def _mel(self, record: MelCacheRecord) -> np.ndarray:
        mapping = self._maps.get(record.shard)
        if mapping is None:
            path = self.cache_root / (
                f"shard-{record.shard:05d}.mel.float16.bin"
            )
            values = path.stat().st_size // np.dtype("<f2").itemsize
            mapping = np.memmap(
                path,
                dtype="<f2",
                mode="r",
                shape=(
                    values // self.frontend.n_mels,
                    self.frontend.n_mels,
                ),
            )
            self._maps[record.shard] = mapping
        start = record.frame_offset
        return np.asarray(
            mapping[start : start + record.frame_count], dtype=np.float32
        ).T

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index % len(self.records)]
        mel = self._mel(record)
        maximum = max(0, record.frame_count - self.crop_frames)
        digest = hashlib.sha256(
            f"{self.seed}:{self.epoch}:{record.sample}:{index}".encode("utf-8")
        ).digest()
        crop_start = (
            int.from_bytes(digest[:8], "little") % (maximum + 1)
            if maximum
            else 0
        )
        valid_frames = min(
            self.crop_frames, record.frame_count - crop_start
        )
        crop = np.zeros(
            (self.frontend.n_mels, self.crop_frames), dtype=np.float32
        )
        crop[:, :valid_frames] = mel[
            :, crop_start : crop_start + valid_frames
        ]
        targets = make_multipitch_targets(
            record.target,
            frames=self.crop_frames,
            hop_sec=self.frontend.hop_sec,
            midi_min=52,
            midi_max=100,
            crop_start=crop_start,
        )
        frame_mask = np.zeros(self.crop_frames, dtype=np.bool_)
        frame_mask[:valid_frames] = True
        return {
            "mel": torch.from_numpy(crop),
            **{
                key: torch.from_numpy(value)
                for key, value in targets.items()
            },
            "frame_mask": torch.from_numpy(frame_mask),
            "sample": record.sample,
            "source": record.source,
            "audio_render": record.audio_render,
            "crop_start": crop_start,
        }


@dataclass(frozen=True)
class MultiPitchDecodeConfig:
    activity_on: float = 0.45
    activity_off: float = 0.25
    onset_threshold: float = 0.40
    offset_threshold: float = 0.45
    min_note_sec: float = 0.030
    release_frames: int = 2
    min_confidence: float = 0.20
    use_polyphony_gate: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any] | None
    ) -> "MultiPitchDecodeConfig":
        allowed = {item.name for item in fields(cls)}
        return cls(
            **{
                key: item
                for key, item in dict(value or {}).items()
                if key in allowed
            }
        )


@dataclass(frozen=True)
class MultiPitchNote:
    pitch: int
    start: float
    end: float
    confidence: float
    activity_strength: float
    onset_strength: float
    offset_strength: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _local_peaks(values: np.ndarray, threshold: float) -> set[int]:
    if not len(values):
        return set()
    left = np.r_[-np.inf, values[:-1]]
    right = np.r_[values[1:], -np.inf]
    return set(
        np.flatnonzero(
            (values >= threshold)
            & (values >= left)
            & (values >= right)
        ).tolist()
    )


def decode_multipitch_notes(
    probabilities: Mapping[str, np.ndarray],
    *,
    midi_min: int,
    hop_sec: float,
    config: MultiPitchDecodeConfig = MultiPitchDecodeConfig(),
) -> list[MultiPitchNote]:
    activity = np.asarray(probabilities["activity"], np.float32)
    onset = np.asarray(probabilities["onset"], np.float32)
    offset = np.asarray(probabilities["offset"], np.float32)
    if (
        activity.ndim != 2
        or onset.shape != activity.shape
        or offset.shape != activity.shape
    ):
        raise ValueError("Multi-pitch probabilities must all be [T,P]")
    frames, pitches = activity.shape
    polyphony_probability = probabilities.get("polyphony")
    if config.use_polyphony_gate and polyphony_probability is not None:
        polyphony_probability = np.asarray(polyphony_probability, np.float32)
        if (
            polyphony_probability.ndim != 2
            or polyphony_probability.shape[0] != frames
        ):
            raise ValueError("polyphony probabilities must be [T,K]")
        counts = np.argmax(polyphony_probability, axis=1)
        gated = np.zeros_like(activity)
        gated_onset = np.zeros_like(onset)
        gated_offset = np.zeros_like(offset)
        for frame, count in enumerate(counts):
            count = min(int(count), pitches)
            if count <= 0:
                continue
            selected = np.argpartition(activity[frame], -count)[-count:]
            gated[frame, selected] = activity[frame, selected]
            gated_onset[frame, selected] = onset[frame, selected]
            gated_offset[frame, selected] = offset[frame, selected]
        activity, onset, offset = gated, gated_onset, gated_offset
    notes = []
    minimum_frames = max(1, int(math.ceil(config.min_note_sec / hop_sec)))
    for pitch_index in range(pitches):
        onset_peaks = _local_peaks(
            onset[:, pitch_index], config.onset_threshold
        )
        active_start: int | None = None
        below = 0

        def emit(stop: int) -> None:
            nonlocal active_start
            assert active_start is not None
            stop = max(stop, active_start + 1)
            if stop - active_start < minimum_frames:
                active_start = None
                return
            section = slice(active_start, min(stop, frames))
            activity_strength = float(
                np.mean(activity[section, pitch_index])
            )
            onset_strength = float(onset[active_start, pitch_index])
            offset_strength = float(
                np.max(offset[max(active_start, stop - 2) : min(frames, stop + 1), pitch_index])
            )
            confidence = 0.65 * activity_strength + 0.35 * onset_strength
            if confidence >= config.min_confidence:
                notes.append(
                    MultiPitchNote(
                        pitch=midi_min + pitch_index,
                        start=active_start * hop_sec,
                        end=stop * hop_sec,
                        confidence=confidence,
                        activity_strength=activity_strength,
                        onset_strength=onset_strength,
                        offset_strength=offset_strength,
                    )
                )
            active_start = None

        for frame in range(frames):
            starts = (
                activity[frame, pitch_index] >= config.activity_on
                or (
                    frame in onset_peaks
                    and activity[frame, pitch_index] >= config.activity_off
                )
            )
            if active_start is None:
                if starts:
                    active_start = frame
                    below = 0
                continue
            if (
                frame in onset_peaks
                and activity[frame, pitch_index] >= config.activity_off
                and frame - active_start >= minimum_frames
            ):
                emit(frame)
                active_start = frame
                below = 0
                continue
            below = (
                below + 1
                if activity[frame, pitch_index] < config.activity_off
                else 0
            )
            explicit_offset = (
                offset[frame, pitch_index] >= config.offset_threshold
                and frame - active_start >= minimum_frames
            )
            if explicit_offset or below >= max(1, config.release_frames):
                stop = frame if explicit_offset else frame - below + 1
                emit(stop)
                below = 0
        if active_start is not None:
            emit(frames)
    return sorted(notes, key=lambda value: (value.start, value.pitch, value.end))


@torch.inference_mode()
def infer_multipitch_probabilities(
    model: OrnamentMultiPitchTranscriber,
    mel: np.ndarray,
    device: torch.device | str,
    *,
    window_frames: int = 2048,
    overlap_frames: int = 512,
    batch_size: int = 4,
) -> dict[str, np.ndarray]:
    array = np.asarray(mel, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != model.config.backbone.n_mels:
        raise ValueError(
            f"Expected [{model.config.backbone.n_mels},T], got {array.shape}"
        )
    total = array.shape[1]
    pitches = model.config.backbone.n_pitches
    if total == 0:
        return {
            name: np.zeros((0, pitches), np.float32)
            for name in ("activity", "onset", "offset")
        } | {
            "polyphony": np.zeros(
                (0, model.config.max_polyphony + 1), np.float32
            )
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
    shapes = {
        "activity": pitches,
        "onset": pitches,
        "offset": pitches,
        "polyphony": model.config.max_polyphony + 1,
    }
    sums = {
        name: np.zeros((total, width), np.float64)
        for name, width in shapes.items()
    }
    weights = np.zeros(total, np.float64)
    target = torch.device(device)
    model.eval()
    for group_start in range(0, len(starts), batch_size):
        group = starts[group_start : group_start + batch_size]
        windows = np.zeros(
            (
                len(group),
                model.config.backbone.n_mels,
                window_frames,
            ),
            np.float32,
        )
        lengths = []
        for batch_index, start in enumerate(group):
            length = min(window_frames, total - start)
            windows[batch_index, :, :length] = array[:, start : start + length]
            lengths.append(length)
        output = model(torch.from_numpy(windows).to(target))
        probability = {
            "activity": torch.sigmoid(output["activity_logits"])
            .float()
            .cpu()
            .numpy(),
            "onset": torch.sigmoid(output["onset_logits"])
            .float()
            .cpu()
            .numpy(),
            "offset": torch.sigmoid(output["offset_logits"])
            .float()
            .cpu()
            .numpy(),
            "polyphony": output["polyphony_logits"]
            .softmax(dim=-1)
            .float()
            .cpu()
            .numpy(),
        }
        for batch_index, (start, length) in enumerate(zip(group, lengths)):
            section = slice(start, start + length)
            weight = taper[:length]
            weights[section] += weight
            for name in shapes:
                sums[name][section] += (
                    probability[name][batch_index, :length] * weight[:, None]
                )
    denominator = np.maximum(weights[:, None], 1e-8)
    return {
        name: (value / denominator).astype(np.float32)
        for name, value in sums.items()
    }
