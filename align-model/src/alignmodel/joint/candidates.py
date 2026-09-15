from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from alignmodel.stages.repetition import find_note_sequence_repetitions
from alignmodel.transcription.basic_pitch import (
    BasicPitchDecodeConfig,
    BasicPitchFeatures,
    FROZEN_DECODE_CONFIG,
    MIDI_OFFSET,
    decode_basic_pitch_features,
    sanitize_basic_pitch_notes,
)

from .lattice import JointCandidate, candidates_from_events
from .index import ScoreEvent

CANDIDATE_GENERATION_VERSION = "align-joint-candidates-v2-short-rescue"

LEGACY_CANDIDATE_GENERATION_VERSION = "align-joint-candidates-v1"
LEGACY_HIGH_RECALL_DECODE_CONFIGS = (
    replace(
        FROZEN_DECODE_CONFIG,
        onset_threshold=0.25,
        frame_threshold=0.25,
        minimum_note_length_ms=35.0,
        merge_same_pitch_gap_sec=0.06,
    ),
    replace(
        FROZEN_DECODE_CONFIG,
        onset_threshold=0.35,
        frame_threshold=0.30,
        minimum_note_length_ms=45.0,
    ),
    FROZEN_DECODE_CONFIG,
)
HIGH_RECALL_DECODE_CONFIGS = (
    replace(
        LEGACY_HIGH_RECALL_DECODE_CONFIGS[0],
        minimum_note_length_ms=30.0,
        adaptive_short_note_rescue=True,
    ),
    *LEGACY_HIGH_RECALL_DECODE_CONFIGS[1:],
)


def _merge_candidate(
    existing: JointCandidate, incoming: JointCandidate
) -> JointCandidate:
    if incoming.confidence > existing.confidence:
        primary, secondary = incoming, existing
    else:
        primary, secondary = existing, incoming
    return JointCandidate(
        pitch=primary.pitch,
        start=primary.start,
        end=max(primary.end, secondary.end),
        confidence=max(primary.confidence, secondary.confidence),
        score_hints=tuple(
            sorted(set(primary.score_hints) | set(secondary.score_hints))
        ),
        acoustic_features=primary.acoustic_features,
    )


def _acoustic_summary(
    candidate: JointCandidate,
    features: BasicPitchFeatures,
) -> tuple[float, ...]:
    """Summarize frozen activation evidence for a trainable projection."""

    axis = int(candidate.pitch) - MIDI_OFFSET
    start = int(
        np.searchsorted(features.frame_times, candidate.start, side="left")
    )
    end = int(
        np.searchsorted(features.frame_times, candidate.end, side="right")
    )
    start = min(max(start, 0), max(len(features.frame_times) - 1, 0))
    end = min(max(end, start + 1), len(features.frame_times))
    if not (0 <= axis < features.note.shape[1]) or end <= start:
        return (0.0,) * 5
    onset_lo = max(0, start - 2)
    onset_hi = min(len(features.frame_times), start + 3)
    onset_peak = float(np.max(features.onset[onset_lo:onset_hi, axis]))
    note_values = features.note[start:end, axis]
    note_peak = float(np.max(note_values))
    note_mean = float(np.mean(note_values))
    competing = np.max(features.note[start:end], axis=1)
    pitch_margin = float(np.mean(note_values - competing))
    contour_start = axis * 3
    contour_end = min(contour_start + 3, features.contour.shape[1])
    contour_support = (
        float(np.mean(np.max(features.contour[start:end, contour_start:contour_end], axis=1)))
        if contour_end > contour_start
        else 0.0
    )
    return (
        max(0.0, min(1.0, onset_peak)),
        max(0.0, min(1.0, note_peak)),
        max(0.0, min(1.0, note_mean)),
        max(-1.0, min(0.0, pitch_margin)),
        max(0.0, min(1.0, contour_support)),
    )


def basic_pitch_candidate_union(
    features: BasicPitchFeatures,
    configs: Sequence[BasicPitchDecodeConfig] = HIGH_RECALL_DECODE_CONFIGS,
    *,
    dedup_onset_sec: float = 0.035,
    minimum_confidence: float = 0.0,
) -> list[JointCandidate]:
    """Decode a high-recall union from one frozen Basic Pitch activation map."""

    candidates: list[JointCandidate] = []
    for config in configs:
        decoded = sanitize_basic_pitch_notes(
            decode_basic_pitch_features(features, config),
            features,
            config,
        )
        for candidate in candidates_from_events(decoded):
            match = next(
                (
                    index
                    for index, previous in enumerate(candidates)
                    if previous.pitch == candidate.pitch
                    and abs(previous.start - candidate.start) <= dedup_onset_sec
                ),
                None,
            )
            if match is None:
                candidates.append(candidate)
            else:
                candidates[match] = _merge_candidate(
                    candidates[match], candidate
                )
    return sorted(
        [
            replace(
                value,
                acoustic_features=_acoustic_summary(value, features),
            )
            for value in candidates
            if value.confidence >= minimum_confidence
        ],
        key=lambda value: (value.start, value.pitch, value.end),
    )


def add_score_repeat_hints(
    candidates: Sequence[JointCandidate],
    score: Sequence[ScoreEvent],
    *,
    max_hints: int = 4,
) -> list[JointCandidate]:
    """Attach score starts supported by score-blind repeated candidate phrases."""

    output = list(candidates)
    repetitions = find_note_sequence_repetitions(
        output,
        min_notes=6,
        max_notes=64,
        min_confidence=0.80,
    )
    for repetition in repetitions:
        length = min(
            repetition.source_i1 - repetition.source_i0,
            repetition.repeat_i1 - repetition.repeat_i0,
        )
        if length < 1 or length > len(score):
            continue
        pitches = [
            output[repetition.source_i0 + offset].pitch
            for offset in range(length)
        ]
        starts = []
        for score_start in range(0, len(score) - length + 1):
            mismatches = sum(
                score[score_start + offset].pitch != pitch
                for offset, pitch in enumerate(pitches)
            )
            starts.append((mismatches, score_start))
        starts.sort()
        allowed = max(1, int(round(0.15 * length)))
        hints = [
            score_start
            for mismatches, score_start in starts
            if mismatches <= allowed
        ][:max_hints]
        for offset in range(length):
            candidate_index = repetition.repeat_i0 + offset
            candidate_hints = {
                *(output[candidate_index].score_hints),
                *(score_start + offset for score_start in hints),
            }
            output[candidate_index] = replace(
                output[candidate_index],
                score_hints=tuple(sorted(candidate_hints)),
            )
    return output
