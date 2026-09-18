"""Residual polyphonic refiner over immutable Basic Pitch maps."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .basic_pitch import (
    BasicPitchDecodeConfig,
    BasicPitchFeatures,
    decode_basic_pitch_features,
)


SCHEMA_VERSION = "align-basic-pitch-polyphonic-refiner-v1"


@dataclass(frozen=True)
class BasicPitchPolyphonicConfig:
    midi_min: int = 52
    midi_max: int = 100
    hidden: int = 128
    blocks: int = 8
    dropout: float = 0.10

    @property
    def n_pitches(self) -> int:
        return self.midi_max - self.midi_min + 1

    @property
    def axis_start(self) -> int:
        return self.midi_min - 21

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _Block(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                5,
                padding=2 * dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(8 if channels % 8 == 0 else 1, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.network(value)


def _logit(value: Tensor) -> Tensor:
    value = value.clamp(1e-4, 1.0 - 1e-4)
    return torch.log(value) - torch.log1p(-value)


class BasicPitchPolyphonicRefiner(nn.Module):
    def __init__(
        self,
        config: BasicPitchPolyphonicConfig = BasicPitchPolyphonicConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        channels = config.n_pitches * 5
        self.input = nn.Sequential(
            nn.Conv1d(channels, config.hidden, 1, bias=False),
            nn.GroupNorm(8 if config.hidden % 8 == 0 else 1, config.hidden),
            nn.SiLU(),
        )
        self.temporal = nn.Sequential(
            *[
                _Block(
                    config.hidden,
                    2 ** (index % 5),
                    config.dropout,
                )
                for index in range(config.blocks)
            ]
        )
        self.activity_residual = nn.Conv1d(
            config.hidden, config.n_pitches, 1
        )
        self.onset_residual = nn.Conv1d(config.hidden, config.n_pitches, 1)
        nn.init.zeros_(self.activity_residual.weight)
        nn.init.zeros_(self.activity_residual.bias)
        nn.init.zeros_(self.onset_residual.weight)
        nn.init.zeros_(self.onset_residual.bias)

    def forward(
        self,
        note_map: Tensor,
        onset_map: Tensor,
        contour_map: Tensor,
    ) -> dict[str, Tensor]:
        if note_map.ndim != 3 or note_map.shape != onset_map.shape:
            raise ValueError("note/onset maps must be [B,T,88]")
        if contour_map.shape[:2] != note_map.shape[:2]:
            raise ValueError("contour time grid differs from note map")
        start = self.config.axis_start
        stop = start + self.config.n_pitches
        note = note_map[:, :, start:stop].float().clamp(0.0, 1.0)
        onset = onset_map[:, :, start:stop].float().clamp(0.0, 1.0)
        contour = contour_map.float().reshape(
            *contour_map.shape[:2], 88, 3
        )[:, :, start:stop]
        features = torch.cat(
            (note.unsqueeze(-1), onset.unsqueeze(-1), contour), dim=-1
        ).flatten(start_dim=2)
        hidden = self.temporal(self.input(features.transpose(1, 2)))
        return {
            "activity_logits": _logit(note)
            + self.activity_residual(hidden).transpose(1, 2),
            "onset_logits": _logit(onset)
            + self.onset_residual(hidden).transpose(1, 2),
            "activity_residual": self.activity_residual(hidden).transpose(1, 2),
            "onset_residual": self.onset_residual(hidden).transpose(1, 2),
        }


def make_basic_pitch_targets(
    notes: Sequence[Mapping[str, Any]],
    frame_times: np.ndarray,
    *,
    midi_min: int,
    midi_max: int,
) -> dict[str, np.ndarray]:
    times = np.asarray(frame_times, np.float64)
    pitches = midi_max - midi_min + 1
    activity = np.zeros((len(times), pitches), np.float32)
    onset = np.zeros((len(times), pitches), np.float32)
    weight = np.ones((len(times), pitches), np.float32)
    for row in notes:
        pitch = int(row["pitch"]) - midi_min
        start = float(row.get("start", row.get("start_sec")))
        end = float(row.get("end", row.get("end_sec")))
        if not 0 <= pitch < pitches or end <= start:
            continue
        selected = np.flatnonzero((times >= start) & (times < end))
        if not len(selected):
            selected = np.asarray(
                [int(np.argmin(np.abs(times - start)))], dtype=np.int64
            )
        duration = end - start
        duration_weight = (
            5.0
            if duration < 0.080
            else 3.0
            if duration < 0.120
            else 1.8
            if duration < 0.180
            else 1.0
        )
        activity[selected, pitch] = 1.0
        weight[selected, pitch] = np.maximum(
            weight[selected, pitch], duration_weight
        )
        onset_index = int(np.argmin(np.abs(times - start)))
        onset[onset_index, pitch] = 1.0
        weight[onset_index, pitch] = max(
            weight[onset_index, pitch], duration_weight
        )
    return {"activity": activity, "onset": onset, "duration_weight": weight}


def basic_pitch_refiner_loss(
    output: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
) -> tuple[Tensor, dict[str, Tensor]]:
    mask = batch["frame_mask"].bool().unsqueeze(-1)
    weight = batch["duration_weight"].float()

    def loss_for(
        logits: Tensor,
        target: Tensor,
        *,
        positive_weight: float,
        gamma: float,
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
        effective = mask.float() * torch.where(
            values > 0.5, weight, 1.0
        )
        return (base * focal * effective).sum() / effective.sum().clamp_min(1.0)

    activity = loss_for(
        output["activity_logits"],
        batch["activity"],
        positive_weight=3.0,
        gamma=1.0,
    )
    onset = loss_for(
        output["onset_logits"],
        batch["onset"],
        positive_weight=20.0,
        gamma=2.0,
    )
    regularization = 0.001 * (
        output["activity_residual"].square().mean()
        + output["onset_residual"].square().mean()
    )
    total = activity + 0.75 * onset + regularization
    return total, {
        "total": total.detach(),
        "activity": activity.detach(),
        "onset": onset.detach(),
        "residual_regularization": regularization.detach(),
    }


class BasicPitchRefinerDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[Mapping[str, Any]],
        *,
        crop_frames: int,
        crops_per_clip: int,
        epoch: int,
        seed: int,
    ) -> None:
        self.examples = tuple(examples)
        self.crop_frames = int(crop_frames)
        self.crops_per_clip = int(crops_per_clip)
        self.epoch = int(epoch)
        self.seed = int(seed)
        self._cache: dict[str, tuple[np.ndarray, ...]] = {}

    def __len__(self) -> int:
        return len(self.examples) * self.crops_per_clip

    def _arrays(self, example: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
        sample = str(example["sample"])
        cached = self._cache.get(sample)
        if cached is not None:
            return cached
        with np.load(example["cache"], allow_pickle=False) as saved:
            arrays = (
                np.asarray(saved["note"], np.float32),
                np.asarray(saved["onset"], np.float32),
                np.asarray(saved["contour"], np.float32),
                np.asarray(saved["frame_times"], np.float64),
            )
        self._cache[sample] = arrays
        return arrays

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index % len(self.examples)]
        note_map, onset_map, contour_map, frame_times = self._arrays(example)
        maximum = max(0, len(frame_times) - self.crop_frames)
        digest = hashlib.sha256(
            f"{self.seed}:{self.epoch}:{example['sample']}:{index}".encode(
                "utf-8"
            )
        ).digest()
        start = (
            int.from_bytes(digest[:8], "little") % (maximum + 1)
            if maximum
            else 0
        )
        stop = min(start + self.crop_frames, len(frame_times))
        valid = stop - start
        note = np.zeros((self.crop_frames, 88), np.float32)
        onset = np.zeros_like(note)
        contour = np.zeros((self.crop_frames, 264), np.float32)
        note[:valid] = note_map[start:stop]
        onset[:valid] = onset_map[start:stop]
        contour[:valid] = contour_map[start:stop]
        targets = make_basic_pitch_targets(
            example["target"],
            frame_times,
            midi_min=52,
            midi_max=100,
        )
        activity = np.zeros((self.crop_frames, 49), np.float32)
        onset_target = np.zeros_like(activity)
        duration_weight = np.ones_like(activity)
        activity[:valid] = targets["activity"][start:stop]
        onset_target[:valid] = targets["onset"][start:stop]
        duration_weight[:valid] = targets["duration_weight"][start:stop]
        frame_mask = np.zeros(self.crop_frames, np.bool_)
        frame_mask[:valid] = True
        return {
            "note_map": torch.from_numpy(note),
            "onset_map": torch.from_numpy(onset),
            "contour_map": torch.from_numpy(contour),
            "activity": torch.from_numpy(activity),
            "onset": torch.from_numpy(onset_target),
            "duration_weight": torch.from_numpy(duration_weight),
            "frame_mask": torch.from_numpy(frame_mask),
        }


@torch.inference_mode()
def refine_basic_pitch_features(
    model: BasicPitchPolyphonicRefiner,
    features: BasicPitchFeatures,
    device: torch.device | str,
    *,
    window_frames: int = 2048,
    overlap_frames: int = 512,
) -> BasicPitchFeatures:
    total = len(features.frame_times)
    stride = window_frames - overlap_frames
    starts = list(range(0, max(total - window_frames, 0) + 1, stride))
    final = max(0, total - window_frames)
    if not starts or starts[-1] != final:
        starts.append(final)
    sums_activity = np.zeros((total, model.config.n_pitches), np.float64)
    sums_onset = np.zeros_like(sums_activity)
    weights = np.zeros(total, np.float64)
    taper = np.maximum(
        np.hanning(window_frames + 2)[1:-1].astype(np.float32), 0.05
    )
    model.eval()
    target = torch.device(device)
    for start in starts:
        length = min(window_frames, total - start)
        note_map = np.zeros((1, window_frames, 88), np.float32)
        onset_map = np.zeros_like(note_map)
        contour_map = np.zeros((1, window_frames, 264), np.float32)
        note_map[0, :length] = features.note[start : start + length]
        onset_map[0, :length] = features.onset[start : start + length]
        contour_map[0, :length] = features.contour[start : start + length]
        output = model(
            torch.from_numpy(note_map).to(target),
            torch.from_numpy(onset_map).to(target),
            torch.from_numpy(contour_map).to(target),
        )
        activity = torch.sigmoid(output["activity_logits"])[0, :length]
        onset = torch.sigmoid(output["onset_logits"])[0, :length]
        section = slice(start, start + length)
        weight = taper[:length]
        sums_activity[section] += activity.float().cpu().numpy() * weight[:, None]
        sums_onset[section] += onset.float().cpu().numpy() * weight[:, None]
        weights[section] += weight
    refined_note = features.note.copy()
    refined_onset = features.onset.copy()
    start = model.config.axis_start
    stop = start + model.config.n_pitches
    denominator = np.maximum(weights[:, None], 1e-8)
    refined_note[:, start:stop] = (
        sums_activity / denominator
    ).astype(np.float32)
    refined_onset[:, start:stop] = (
        sums_onset / denominator
    ).astype(np.float32)
    return BasicPitchFeatures(
        refined_note,
        refined_onset,
        features.contour,
        features.frame_times,
        features.metadata,
    )


def decode_refined_basic_pitch(
    model: BasicPitchPolyphonicRefiner,
    features: BasicPitchFeatures,
    device: torch.device | str,
    config: BasicPitchDecodeConfig,
) -> list[Any]:
    return decode_basic_pitch_features(
        refine_basic_pitch_features(model, features, device),
        config,
    )
