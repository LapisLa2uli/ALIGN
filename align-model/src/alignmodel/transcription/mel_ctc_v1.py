"""CTC pitch-sequence head on the mel transcriber encoder.

The frame heads of ``MelNoteTranscriber`` need exact note timing, and the
decoder must separate repeated same-pitch notes from weak onset peaks. A CTC
head instead learns the written-pitch token sequence directly: repeated notes
are separated by blanks, and training needs only which notes fall inside a
crop, not their frame boundaries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .mel_v1 import (
    MelFrontendConfig,
    MelNoteTranscriber,
    MelTranscriberConfig,
)


SCHEMA_VERSION = "align-mel-ctc-transcriber-v1"
BLANK = 0


class MelCTCTranscriber(nn.Module):
    def __init__(self, config: MelTranscriberConfig = MelTranscriberConfig()) -> None:
        super().__init__()
        self.config = config
        self.base = MelNoteTranscriber(config)
        width = (
            config.temporal_dim if config.temporal_kind == "tcn"
            else max(32, config.temporal_dim // 2) * 2
        )
        self.ctc_head = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(width, config.n_pitches + 1, 1),
        )

    @property
    def vocabulary(self) -> int:
        return self.config.n_pitches + 1

    def forward(self, mel: Tensor) -> dict[str, Tensor]:
        value = self.base.encode(mel)
        base = self.base
        return {
            "voiced_logits": base.voiced_head(value).squeeze(1),
            "pitch_logits": base.pitch_head(value).transpose(1, 2),
            "onset_logits": base.onset_head(value).squeeze(1),
            "boundary_logits": base.boundary_head(value).squeeze(1),
            "rearticulation_logits": base.rearticulation_head(value).squeeze(1),
            "confidence_logits": base.confidence_head(value).squeeze(1),
            "ctc_logits": self.ctc_head(value).transpose(1, 2),
        }


def pitch_tokens(pitches: Sequence[int], midi_min: int, n_pitches: int) -> list[int]:
    return [
        int(np.clip(int(pitch) - midi_min, 0, n_pitches - 1)) + 1
        for pitch in pitches
    ]


def ctc_loss(
    logits: Tensor, frame_mask: Tensor, targets: Tensor, target_lengths: Tensor
) -> Tensor:
    log_probs = logits.float().log_softmax(dim=-1).transpose(0, 1)
    input_lengths = frame_mask.long().sum(dim=1)
    flat = torch.cat([
        targets[index, :int(length)] for index, length in enumerate(target_lengths)
    ]) if int(target_lengths.sum()) else targets.new_zeros(0)
    return F.ctc_loss(
        log_probs, flat, input_lengths, target_lengths,
        blank=BLANK, reduction="mean", zero_infinity=True,
    )


def greedy_decode(probabilities: np.ndarray, midi_min: int) -> list[tuple[int, int]]:
    """Return (written pitch, first frame) for each collapsed non-blank run."""

    best = probabilities.argmax(axis=1)
    output: list[tuple[int, int]] = []
    previous = BLANK
    for frame, token in enumerate(best.tolist()):
        if token != BLANK and token != previous:
            output.append((midi_min + token - 1, frame))
        previous = token
    return output


@torch.inference_mode()
def infer_ctc_probabilities(
    model: MelCTCTranscriber,
    mel: np.ndarray,
    device: torch.device | str,
    *,
    window_frames: int = 2048,
    overlap_frames: int = 512,
    batch_size: int = 4,
) -> np.ndarray:
    array = np.asarray(mel, dtype=np.float32)
    total = array.shape[1]
    vocabulary = model.vocabulary
    if total == 0:
        return np.zeros((0, vocabulary), np.float32)
    stride = window_frames - overlap_frames
    starts = list(range(0, max(total - window_frames, 0) + 1, stride))
    final = max(0, total - window_frames)
    if not starts or starts[-1] != final:
        starts.append(final)
    taper = np.maximum(np.hanning(window_frames + 2)[1:-1].astype(np.float32), 0.05)
    sums = np.zeros((total, vocabulary), np.float64)
    weights = np.zeros(total, np.float64)
    target = torch.device(device)
    model.eval()
    for group_start in range(0, len(starts), batch_size):
        group = starts[group_start:group_start + batch_size]
        windows = np.zeros((len(group), array.shape[0], window_frames), np.float32)
        lengths = []
        for index, start in enumerate(group):
            length = min(window_frames, total - start)
            windows[index, :, :length] = array[:, start:start + length]
            lengths.append(length)
        logits = model(torch.from_numpy(windows).to(target))["ctc_logits"]
        probability = logits.float().softmax(dim=-1).cpu().numpy()
        for index, (start, length) in enumerate(zip(group, lengths)):
            weight = taper[:length] if total > window_frames else np.ones(length, np.float32)
            sums[start:start + length] += probability[index, :length] * weight[:, None]
            weights[start:start + length] += weight
    return (sums / np.maximum(weights, 1e-9)[:, None]).astype(np.float32)


@torch.inference_mode()
def infer_ctc_outputs(
    model: MelCTCTranscriber,
    mel: np.ndarray,
    device: torch.device | str,
    *,
    window_frames: int = 2048,
    overlap_frames: int = 512,
    batch_size: int = 4,
) -> dict[str, np.ndarray]:
    """Overlap-add CTC softmax plus the frame heads (voiced/onset/boundary/rearticulation)."""

    array = np.asarray(mel, dtype=np.float32)
    total = array.shape[1]
    heads = ("voiced", "onset", "boundary", "rearticulation")
    vocabulary = model.vocabulary
    if total == 0:
        return {"ctc": np.zeros((0, vocabulary), np.float32),
                **{name: np.zeros(0, np.float32) for name in heads}}
    stride = window_frames - overlap_frames
    starts = list(range(0, max(total - window_frames, 0) + 1, stride))
    final = max(0, total - window_frames)
    if not starts or starts[-1] != final:
        starts.append(final)
    taper = np.maximum(np.hanning(window_frames + 2)[1:-1].astype(np.float32), 0.05)
    sums = {"ctc": np.zeros((total, vocabulary), np.float64),
            **{name: np.zeros(total, np.float64) for name in heads}}
    weights = np.zeros(total, np.float64)
    target = torch.device(device)
    model.eval()
    for group_start in range(0, len(starts), batch_size):
        group = starts[group_start:group_start + batch_size]
        windows = np.zeros((len(group), array.shape[0], window_frames), np.float32)
        lengths = []
        for index, start in enumerate(group):
            length = min(window_frames, total - start)
            windows[index, :, :length] = array[:, start:start + length]
            lengths.append(length)
        output = model(torch.from_numpy(windows).to(target))
        values = {"ctc": output["ctc_logits"].float().softmax(dim=-1).cpu().numpy()}
        for name in heads:
            values[name] = torch.sigmoid(output[f"{name}_logits"]).float().cpu().numpy()
        for index, (start, length) in enumerate(zip(group, lengths)):
            weight = taper[:length] if total > window_frames else np.ones(length, np.float32)
            sums["ctc"][start:start + length] += values["ctc"][index, :length] * weight[:, None]
            for name in heads:
                sums[name][start:start + length] += values[name][index, :length] * weight
            weights[start:start + length] += weight
    denominator = np.maximum(weights, 1e-9)
    return {
        "ctc": (sums["ctc"] / denominator[:, None]).astype(np.float32),
        **{name: (sums[name] / denominator).astype(np.float32) for name in heads},
    }


@dataclass(frozen=True)
class CTCCheckpointInfo:
    model_config: dict
    frontend_config: dict


def save_ctc_checkpoint(
    path: Path, model: MelCTCTranscriber, frontend: MelFrontendConfig,
    extra: Mapping[str, Any],
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "model_config": model.config.to_dict(),
        "frontend_config": frontend.to_dict(),
        **dict(extra),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_ctc_checkpoint(
    path: Path | str, device: torch.device | str
) -> tuple[MelCTCTranscriber, MelFrontendConfig, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported CTC checkpoint")
    model = MelCTCTranscriber(MelTranscriberConfig.from_dict(payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, MelFrontendConfig.from_dict(payload["frontend_config"]), payload
