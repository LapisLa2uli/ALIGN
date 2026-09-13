"""Trainable repetition scoring over transcribed note-sequence candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

FEATURE_NAMES = (
    "pitch_exact_fraction",
    "pitch_class_fraction",
    "interval_exact_fraction",
    "timing_consistency",
    "duration_consistency",
    "source_confidence",
    "repeat_confidence",
    "log_note_count",
    "gap_notes",
    "repeat_extra_fraction",
    "source_extra_fraction",
    "duration_ratio_error",
)


@dataclass(frozen=True)
class NoteRepeatCandidate:
    source_i0: int
    source_i1: int
    repeat_i0: int
    repeat_i1: int
    features: np.ndarray
    heuristic_confidence: float


@dataclass
class NoteRepetitionModelConfig:
    hidden_dim: int = 48
    dropout: float = 0.10
    threshold: float = 0.65
    max_sources_per_note: int = 32
    max_notes: int = 64
    max_repetitions: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


class NoteRepetitionScorer(nn.Module):
    def __init__(self, config: NoteRepetitionModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or NoteRepetitionModelConfig()
        self.net = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), self.config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _value(note: Any, name: str, default: float = 0.0) -> float:
    if isinstance(note, dict):
        return float(note.get(name, default))
    return float(getattr(note, name, default))


def repetition_features(
    notes: list[Any],
    source_i0: int,
    repeat_i0: int,
    length: int,
    *,
    extra_mask: list[bool] | None = None,
) -> np.ndarray:
    source = notes[source_i0 : source_i0 + length]
    repeated = notes[repeat_i0 : repeat_i0 + length]
    source_pitch = np.asarray([_value(note, "pitch") for note in source])
    repeat_pitch = np.asarray([_value(note, "pitch") for note in repeated])
    exact = float(np.mean(source_pitch == repeat_pitch))
    pitch_class = float(np.mean(source_pitch % 12 == repeat_pitch % 12))
    if length > 1:
        interval_exact = float(
            np.mean(np.diff(source_pitch) == np.diff(repeat_pitch))
        )
        source_ioi = np.diff([_value(note, "start") for note in source])
        repeat_ioi = np.diff([_value(note, "start") for note in repeated])
        valid = (source_ioi > 1e-4) & (repeat_ioi > 1e-4)
        if np.any(valid):
            ratios = repeat_ioi[valid] / source_ioi[valid]
            center = float(np.median(ratios))
            timing = float(
                np.exp(
                    -4.0
                    * np.median(np.abs(np.log(ratios / max(center, 1e-6))))
                )
            )
        else:
            timing = 0.0
    else:
        interval_exact = 1.0
        timing = 1.0
    source_duration = np.asarray(
        [max(1e-3, _value(note, "end") - _value(note, "start")) for note in source]
    )
    repeat_duration = np.asarray(
        [max(1e-3, _value(note, "end") - _value(note, "start")) for note in repeated]
    )
    ratios = repeat_duration / source_duration
    duration_center = float(np.median(ratios))
    duration_consistency = float(
        np.exp(
            -3.0
            * np.median(
                np.abs(np.log(ratios / max(duration_center, 1e-6)))
            )
        )
    )
    extra = extra_mask or [False] * len(notes)
    target_extra = float(
        np.mean(extra[repeat_i0 : repeat_i0 + length])
    )
    source_extra = float(
        np.mean(extra[source_i0 : source_i0 + length])
    )
    return np.asarray(
        [
            exact,
            pitch_class,
            interval_exact,
            timing,
            duration_consistency,
            np.mean([_value(note, "confidence", 1.0) for note in source]),
            np.mean([_value(note, "confidence", 1.0) for note in repeated]),
            np.log1p(length) / np.log(65.0),
            min(1.0, max(0, repeat_i0 - source_i0 - length) / 64.0),
            target_extra,
            source_extra,
            min(1.0, abs(np.log(max(duration_center, 1e-6))) / 1.5),
        ],
        dtype=np.float32,
    )


def propose_note_repeat_candidates(
    notes: list[Any],
    *,
    extra_mask: list[bool] | None = None,
    max_notes: int = 64,
    max_sources_per_note: int = 32,
) -> list[NoteRepeatCandidate]:
    """Propose one-note and longer repeats, using same-pitch starts as seeds."""

    pitches = [int(_value(note, "pitch")) for note in notes]
    output: dict[tuple[int, int, int], NoteRepeatCandidate] = {}
    for repeat_i0 in range(1, len(notes)):
        sources = [
            index
            for index in range(repeat_i0)
            if pitches[index] == pitches[repeat_i0]
        ][-max_sources_per_note:]
        for source_i0 in sources:
            maximum = min(
                max_notes,
                repeat_i0 - source_i0,
                len(notes) - repeat_i0,
            )
            mismatches = 0
            best = 1
            for length in range(1, maximum + 1):
                if pitches[source_i0 + length - 1] != pitches[
                    repeat_i0 + length - 1
                ]:
                    mismatches += 1
                if mismatches > max(1, int(round(0.15 * length))):
                    break
                exact = 1.0 - mismatches / length
                if exact >= 0.75:
                    best = length
            for length in {1, best}:
                features = repetition_features(
                    notes,
                    source_i0,
                    repeat_i0,
                    length,
                    extra_mask=extra_mask,
                )
                confidence = float(
                    0.72 * features[0]
                    + 0.13 * features[2]
                    + 0.15 * features[3]
                )
                output[(source_i0, repeat_i0, length)] = NoteRepeatCandidate(
                    source_i0,
                    source_i0 + length,
                    repeat_i0,
                    repeat_i0 + length,
                    features,
                    confidence,
                )
    return list(output.values())


def save_note_repetition_model(
    path,
    model: NoteRepetitionScorer,
    *,
    extra: dict | None = None,
):
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model.config.to_dict(),
            "feature_names": FEATURE_NAMES,
            "extra": dict(extra or {}),
        },
        path,
    )


def load_note_repetition_model(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    config = NoteRepetitionModelConfig(**payload.get("model_config", {}))
    model = NoteRepetitionScorer(config)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, dict(payload.get("extra") or {})
