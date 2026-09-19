"""High-recall, score-free interval proposals from Basic Pitch activations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from .basic_pitch import BasicPitchFeatures


SCHEMA_VERSION = "align-basic-pitch-activation-lattice-v1"


@dataclass(frozen=True)
class ActivationCandidate:
    pitch: int
    start: float
    end: float
    confidence: float
    note_peak: float
    note_mean: float
    onset_peak: float
    contour_peak: float
    onset_contrast: float
    lower_harmonic: float
    upper_harmonic: float
    duration_frames: int
    source_kind: str
    alternatives: tuple[int, ...]
    alternative_confidences: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _local_peaks(values: np.ndarray, floor: float) -> np.ndarray:
    left = np.r_[-np.inf, values[:-1]]
    right = np.r_[values[1:], -np.inf]
    return np.flatnonzero(
        (values >= floor) & (values >= left) & (values >= right)
    )


def _features(
    features: BasicPitchFeatures,
    *,
    pitch: int,
    start_frame: int,
    end_frame: int,
    source_kind: str,
) -> ActivationCandidate:
    axis = pitch - 21
    note_map = np.asarray(features.note[:, axis], np.float32)
    onset_map = np.asarray(features.onset[:, axis], np.float32)
    contour = np.asarray(
        features.contour[:, axis * 3 : axis * 3 + 3], np.float32
    )
    start_frame = max(0, min(start_frame, len(note_map) - 1))
    end_frame = max(start_frame + 1, min(end_frame, len(note_map)))
    section = slice(start_frame, end_frame)
    neighborhood = slice(max(0, start_frame - 2), min(len(note_map), start_frame + 3))
    onset_peak = float(onset_map[start_frame])
    onset_background = float(np.median(onset_map[neighborhood]))
    pitch_probabilities = np.asarray(features.note[start_frame], np.float32)
    top = np.argsort(pitch_probabilities)[-4:][::-1]
    hop = (
        float(np.median(np.diff(features.frame_times)))
        if len(features.frame_times) > 1
        else 256.0 / 22050.0
    )
    start = float(features.frame_times[start_frame])
    end_index = min(end_frame - 1, len(features.frame_times) - 1)
    end = max(float(features.frame_times[end_index]) + hop, start + hop)
    lower_axis = axis - 12
    upper_axis = axis + 12
    return ActivationCandidate(
        pitch=pitch,
        start=start,
        end=end,
        confidence=float(
            0.45 * np.max(note_map[section])
            + 0.35 * onset_peak
            + 0.20 * np.max(contour[section])
        ),
        note_peak=float(np.max(note_map[section])),
        note_mean=float(np.mean(note_map[section])),
        onset_peak=onset_peak,
        contour_peak=float(np.max(contour[section])),
        onset_contrast=onset_peak - onset_background,
        lower_harmonic=(
            float(features.note[start_frame, lower_axis])
            if 0 <= lower_axis < features.note.shape[1]
            else 0.0
        ),
        upper_harmonic=(
            float(features.note[start_frame, upper_axis])
            if 0 <= upper_axis < features.note.shape[1]
            else 0.0
        ),
        duration_frames=end_frame - start_frame,
        source_kind=source_kind,
        alternatives=tuple(int(value) + 21 for value in top),
        alternative_confidences=tuple(
            float(pitch_probabilities[value]) for value in top
        ),
    )


def generate_activation_lattice(
    features: BasicPitchFeatures,
    standard_notes: Sequence[Any] = (),
    *,
    midi_min: int = 52,
    midi_max: int = 100,
    onset_floor: float = 0.04,
    note_floors: tuple[float, ...] = (0.04, 0.08, 0.14),
    fixed_duration_frames: tuple[int, ...] = (2, 4, 8),
    max_onsets_per_pitch: int = 32,
    max_candidates: int = 1024,
) -> list[ActivationCandidate]:
    """Generate a fixed high-recall pool without reading score or targets."""

    if len(features.frame_times) == 0:
        return []
    records: dict[tuple[int, int, int], ActivationCandidate] = {}
    for note in standard_notes:
        frame = int(
            np.argmin(np.abs(features.frame_times - float(note.start)))
        )
        end_frame = int(
            np.searchsorted(features.frame_times, float(note.end), side="left")
        )
        candidate = _features(
            features,
            pitch=int(note.pitch),
            start_frame=frame,
            end_frame=max(frame + 1, end_frame),
            source_kind="standard_decode",
        )
        records[(candidate.pitch, frame, end_frame)] = candidate
    for pitch in range(midi_min, midi_max + 1):
        axis = pitch - 21
        note_map = np.asarray(features.note[:, axis], np.float32)
        onset_map = np.asarray(features.onset[:, axis], np.float32)
        starts = set(int(value) for value in _local_peaks(onset_map, onset_floor))
        for floor in note_floors:
            active = note_map >= floor
            starts.update(
                int(value)
                for value in np.flatnonzero(
                    active
                    & np.r_[True, ~active[:-1]]
                )
            )
        starts = sorted(
            starts,
            key=lambda value: (
                max(float(onset_map[value]), float(note_map[value])),
                -value,
            ),
            reverse=True,
        )[:max_onsets_per_pitch]
        for start in starts:
            for floor in note_floors:
                end = start + 1
                while (
                    end < len(note_map)
                    and end - start < 128
                    and float(note_map[end]) >= floor
                ):
                    end += 1
                candidate = _features(
                    features,
                    pitch=pitch,
                    start_frame=start,
                    end_frame=end,
                    source_kind=f"activation_run_{floor:.2f}",
                )
                key = (pitch, start, end)
                previous = records.get(key)
                if previous is None or candidate.confidence > previous.confidence:
                    records[key] = candidate
            for duration in fixed_duration_frames:
                end = min(len(note_map), start + duration)
                candidate = _features(
                    features,
                    pitch=pitch,
                    start_frame=start,
                    end_frame=end,
                    source_kind=f"fixed_{duration}_frames",
                )
                key = (pitch, start, end)
                previous = records.get(key)
                if previous is None or candidate.confidence > previous.confidence:
                    records[key] = candidate
    standard_keys = {
        key
        for key, value in records.items()
        if value.source_kind == "standard_decode"
    }
    ordered = sorted(
        records.items(),
        key=lambda item: (
            item[1].confidence,
            item[1].onset_peak,
            item[1].note_peak,
        ),
        reverse=True,
    )
    selected_keys = set(standard_keys)
    selected_keys.update(
        key for key, _value in ordered[: max(0, max_candidates - len(standard_keys))]
    )
    return sorted(
        (records[key] for key in selected_keys),
        key=lambda value: (value.start, value.pitch, value.end, value.source_kind),
    )
