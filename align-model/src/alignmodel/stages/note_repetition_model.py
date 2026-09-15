"""Trainable repetition scoring over transcribed note-sequence candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

FEATURE_NAMES_V1 = (
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
FEATURE_NAMES = (
    *FEATURE_NAMES_V1,
    "continuation_pitch_fraction",
    "continuation_alignment_score",
    "restart_gap_strength",
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
    feature_version: int = 2
    continuation_lookahead_notes: int = 3
    continuation_candidate_skips: int = 1
    continuation_source_skips: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


class NoteRepetitionScorer(nn.Module):
    def __init__(self, config: NoteRepetitionModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or NoteRepetitionModelConfig()
        self.feature_names = (
            FEATURE_NAMES_V1
            if int(self.config.feature_version) <= 1
            else FEATURE_NAMES
        )
        self.net = nn.Sequential(
            nn.Linear(len(self.feature_names), self.config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        expected = len(self.feature_names)
        if features.shape[-1] < expected:
            features = torch.nn.functional.pad(
                features, (0, expected - features.shape[-1])
            )
        elif features.shape[-1] > expected:
            features = features[..., :expected]
        return self.net(features).squeeze(-1)


def _value(note: Any, name: str, default: float = 0.0) -> float:
    if isinstance(note, dict):
        return float(note.get(name, default))
    return float(getattr(note, name, default))


def continuation_similarity(
    notes: list[Any],
    source_i0: int,
    repeat_i0: int,
    length: int,
    *,
    lookahead_notes: int = 3,
    candidate_skips: int = 1,
    source_skips: int = 1,
) -> tuple[float, float]:
    """Compare post-replay notes to the source phrase's expected continuation."""

    source_start = source_i0 + length
    repeat_start = repeat_i0 + length
    expected = [
        int(_value(note, "pitch"))
        for note in notes[
            source_start : min(
                repeat_i0,
                source_start + lookahead_notes + source_skips,
            )
        ]
    ]
    observed = [
        int(_value(note, "pitch"))
        for note in notes[
            repeat_start : repeat_start + lookahead_notes + candidate_skips
        ]
    ]
    target = min(max(1, lookahead_notes), len(expected))
    if target < 2 or len(observed) < 2:
        return 0.5, 0.0
    active: dict[tuple[int, int, int, int], int] = {(0, 0, 0, 0): 0}
    best = 0
    while active:
        following: dict[tuple[int, int, int, int], int] = {}
        for (obs_i, exp_i, obs_skips, exp_skips), matches in active.items():
            best = max(best, matches)
            if obs_i >= len(observed) or exp_i >= len(expected):
                continue
            if observed[obs_i] == expected[exp_i]:
                key = (obs_i + 1, exp_i + 1, obs_skips, exp_skips)
                following[key] = max(following.get(key, -1), matches + 1)
            if obs_skips < candidate_skips:
                key = (obs_i + 1, exp_i, obs_skips + 1, exp_skips)
                following[key] = max(following.get(key, -1), matches)
            if exp_skips < source_skips:
                key = (obs_i, exp_i + 1, obs_skips, exp_skips + 1)
                following[key] = max(following.get(key, -1), matches)
        active = following
    fraction = best / max(target, 1)
    return float(fraction), float(2.0 * fraction - 1.0)


def repetition_features(
    notes: list[Any],
    source_i0: int,
    repeat_i0: int,
    length: int,
    *,
    extra_mask: list[bool] | None = None,
    continuation_lookahead_notes: int = 3,
    continuation_candidate_skips: int = 1,
    continuation_source_skips: int = 1,
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
    continuation_fraction, continuation_score = continuation_similarity(
        notes,
        source_i0,
        repeat_i0,
        length,
        lookahead_notes=continuation_lookahead_notes,
        candidate_skips=continuation_candidate_skips,
        source_skips=continuation_source_skips,
    )
    preceding_end = (
        _value(notes[repeat_i0 - 1], "end")
        if repeat_i0 > 0
        else _value(notes[repeat_i0], "start")
    )
    restart_gap = max(
        0.0,
        _value(notes[repeat_i0], "start") - preceding_end,
    )
    reference_duration = float(np.median(source_duration))
    restart_gap_strength = min(
        1.0, restart_gap / max(0.20, reference_duration)
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
            continuation_fraction,
            continuation_score,
            restart_gap_strength,
        ],
        dtype=np.float32,
    )


def propose_note_repeat_candidates(
    notes: list[Any],
    *,
    extra_mask: list[bool] | None = None,
    max_notes: int = 64,
    max_sources_per_note: int = 32,
    continuation_lookahead_notes: int = 3,
    continuation_candidate_skips: int = 1,
    continuation_source_skips: int = 1,
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
                    continuation_lookahead_notes=continuation_lookahead_notes,
                    continuation_candidate_skips=continuation_candidate_skips,
                    continuation_source_skips=continuation_source_skips,
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
            "feature_names": model.feature_names,
            "extra": dict(extra or {}),
        },
        path,
    )


def load_note_repetition_model(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    values = dict(payload.get("model_config", {}))
    if "feature_version" not in values:
        values["feature_version"] = (
            1
            if len(payload.get("feature_names") or FEATURE_NAMES_V1)
            <= len(FEATURE_NAMES_V1)
            else 2
        )
    config = NoteRepetitionModelConfig(**values)
    model = NoteRepetitionScorer(config)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, dict(payload.get("extra") or {})
