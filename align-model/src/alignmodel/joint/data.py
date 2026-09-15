from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from alignmodel.transcription.basic_pitch import (
    basic_pitch_cache_path,
    load_audio_metadata,
    load_basic_pitch_cache,
)
from alignmodel.validated_targets import target_note_map

from .candidates import add_score_repeat_hints, basic_pitch_candidate_union
from .index import JointEvent, ScoreEvent, ScoreEventIndex
from .lattice import JointCandidate
from .metrics import pair_exact_pitch_onset


@dataclass(frozen=True)
class JointTrainingExample:
    sample: str
    source: str
    candidates: tuple[JointCandidate, ...]
    score: tuple[ScoreEvent, ...]
    gold_spans: tuple[tuple[int, int] | None, ...]
    gold_keep_unlinked: tuple[bool, ...]
    target_events: tuple[JointEvent, ...]
    target_deletions: frozenset[int]


@dataclass(frozen=True)
class JointInferenceExample:
    sample: str
    source: str
    candidates: tuple[JointCandidate, ...]
    score: tuple[ScoreEvent, ...]


def _validated_basic_pitch(
    row: Mapping[str, Any],
    cache_root: Path,
):
    sample_dir = Path(str(row["sample_dir"]))
    corpus = str(row.get("corpus") or row.get("root") or "unknown")
    path = basic_pitch_cache_path(cache_root, sample_dir, corpus)
    metadata = load_audio_metadata(sample_dir)
    expected_hash = (row.get("source_hashes") or {}).get(
        "performance_audio.wav"
    )
    features = load_basic_pitch_cache(
        path,
        sample_dir / "performance_audio.wav",
        metadata,
        wav_sha256=str(expected_hash) if expected_hash else None,
    )
    if features is None:
        raise ValueError(f"Missing, stale, or wrong-policy Basic Pitch cache: {path}")
    return features


def _candidate_as_event(candidate: JointCandidate) -> JointEvent:
    return JointEvent(
        pitch=candidate.pitch,
        start=candidate.start,
        end=candidate.end,
        score_span=None,
        relationship="extra",
        confidence=candidate.confidence,
    )


def build_training_example(
    row: Mapping[str, Any],
    cache_root: Path,
    *,
    pairing_tolerance_sec: float = 0.050,
    minimum_candidate_confidence: float = 0.0,
) -> JointTrainingExample:
    """Build exact-lineage supervision. This function is training-only."""

    if row.get("target_db") is None or row.get("target_record") is None:
        raise ValueError("Joint training requires an audited SQLite target row")
    sample_dir = Path(str(row["sample_dir"]))
    score_path = sample_dir / "verified_score.musicxml"
    lineage = target_note_map(row)
    index = (
        ScoreEventIndex.from_musicxml(score_path, lineage)
        if score_path.is_file()
        else ScoreEventIndex.from_lineage(lineage)
    )
    candidates = tuple(
        add_score_repeat_hints(
            basic_pitch_candidate_union(
                _validated_basic_pitch(row, cache_root),
                minimum_confidence=minimum_candidate_confidence,
            ),
            index.events,
        )
    )
    candidate_events = tuple(_candidate_as_event(value) for value in candidates)
    pairs = pair_exact_pitch_onset(
        candidate_events,
        index.rendered_events,
        tolerance_sec=pairing_tolerance_sec,
    )
    gold_spans: list[tuple[int, int] | None] = [None] * len(candidates)
    gold_keep_unlinked = [False] * len(candidates)
    for candidate_index, target_index in pairs:
        gold_spans[candidate_index] = index.rendered_events[
            target_index
        ].score_span
        gold_keep_unlinked[candidate_index] = True
    return JointTrainingExample(
        sample=str(row.get("sample") or sample_dir.name),
        source=str(
            row.get("source_group")
            or row.get("source")
            or row.get("corpus")
            or "unknown"
        ),
        candidates=candidates,
        score=index.events,
        gold_spans=tuple(gold_spans),
        gold_keep_unlinked=tuple(gold_keep_unlinked),
        target_events=index.rendered_events,
        target_deletions=index.deleted_event_indices,
    )


def build_inference_example(
    row: Mapping[str, Any],
    cache_root: Path,
    *,
    minimum_candidate_confidence: float = 0.0,
) -> JointInferenceExample:
    """Build decoder inputs using only WAV-derived cache and verified score."""

    sample_dir = Path(str(row["sample_dir"]))
    score_index = ScoreEventIndex.from_musicxml(
        sample_dir / "verified_score.musicxml"
    )
    return JointInferenceExample(
        sample=str(row.get("sample") or sample_dir.name),
        source=str(
            row.get("source_group")
            or row.get("source")
            or row.get("corpus")
            or "unknown"
        ),
        candidates=tuple(
            add_score_repeat_hints(
                basic_pitch_candidate_union(
                    _validated_basic_pitch(row, cache_root),
                    minimum_confidence=minimum_candidate_confidence,
                ),
                score_index.events,
            )
        ),
        score=score_index.events,
    )


def load_inference_inputs(
    *,
    score_path: Path,
    wav_path: Path,
    cache_path: Path,
    source_metadata: Mapping[str, Any] | None = None,
) -> tuple[tuple[JointCandidate, ...], tuple[ScoreEvent, ...]]:
    """Load deployable inputs without touching lineage, MIDI, or sidecars."""

    features = load_basic_pitch_cache(
        cache_path,
        wav_path,
        source_metadata,
    )
    if features is None:
        raise ValueError(f"Invalid inference Basic Pitch cache: {cache_path}")
    index = ScoreEventIndex.from_musicxml(score_path)
    return (
        tuple(
            add_score_repeat_hints(
                basic_pitch_candidate_union(features),
                index.events,
            )
        ),
        index.events,
    )
