"""Deployable joint transcription + alignment for a DataCreate sample."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Callable

from .candidates import (
    CANDIDATE_GENERATION_VERSION,
    HIGH_RECALL_DECODE_CONFIGS,
    LEGACY_CANDIDATE_GENERATION_VERSION,
    LEGACY_HIGH_RECALL_DECODE_CONFIGS,
    add_score_repeat_hints,
    basic_pitch_candidate_union,
)
from .index import JointEvent, ScoreEvent, ScoreEventIndex
from .lattice import JointCandidate, JointOperation, LatticeConfig, LatticePath, SparseJointLattice
from .train import load_joint_model

_PC_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _midi_pitch_name(midi: int) -> str:
    pitch = int(midi)
    return f"{_PC_NAMES[pitch % 12]}{pitch // 12 - 1}"


@dataclass(frozen=True)
class JointSampleResult:
    sample_id: str
    candidates: tuple[JointCandidate, ...]
    score: tuple[ScoreEvent, ...]
    path: LatticePath
    events: tuple[JointEvent, ...]
    checkpoint: Path
    minimum_candidate_confidence: float
    cache_path: Path
    candidate_generation: str = LEGACY_CANDIDATE_GENERATION_VERSION
    checkpoint_schema: str = "unknown"


def _training_confidence(payload: dict[str, Any], fallback: float) -> float:
    training = payload.get("training") or {}
    value = training.get("minimum_candidate_confidence")
    if value is None:
        return fallback
    return float(value)


def candidate_generation_for_checkpoint(payload: dict[str, Any]) -> str:
    training = payload.get("training") or {}
    frontend = training.get("frontend") or {}
    version = (
        frontend.get("candidate_generation")
        or training.get("candidate_generation")
    )
    return (
        CANDIDATE_GENERATION_VERSION
        if version == CANDIDATE_GENERATION_VERSION
        else LEGACY_CANDIDATE_GENERATION_VERSION
    )


def candidate_configs_for_checkpoint(payload: dict[str, Any]):
    if candidate_generation_for_checkpoint(payload) == CANDIDATE_GENERATION_VERSION:
        return HIGH_RECALL_DECODE_CONFIGS
    return LEGACY_HIGH_RECALL_DECODE_CONFIGS


def infer_joint_sample(
    sample_dir: Path | str,
    checkpoint: Path | str,
    *,
    device: str = "cpu",
    cache_path: Path | str | None = None,
    minimum_candidate_confidence: float | None = None,
    progress: Callable[[str], None] | None = None,
) -> JointSampleResult:
    """Transcribe with frozen Basic Pitch and decode a joint path CRF."""

    sample = Path(sample_dir)
    ckpt = Path(checkpoint)
    wav = sample / "performance_audio.wav"
    score_path = sample / "verified_score.musicxml"
    if not wav.is_file():
        raise FileNotFoundError(wav)
    if not score_path.is_file():
        raise FileNotFoundError(score_path)

    from alignmodel.transcription.basic_pitch import (
        extract_sample_basic_pitch_features,
    )

    cache = Path(cache_path) if cache_path is not None else (
        sample / "basic_pitch_cache.npz"
    )
    if progress:
        progress("transcriber")
    features = extract_sample_basic_pitch_features(sample, cache_path=cache)
    index = ScoreEventIndex.from_musicxml(score_path)
    model, lattice_config, payload = load_joint_model(ckpt, device=device)
    candidate_generation = candidate_generation_for_checkpoint(payload)
    confidence = (
        float(minimum_candidate_confidence)
        if minimum_candidate_confidence is not None
        else _training_confidence(payload, 0.65)
    )
    candidates = tuple(
        add_score_repeat_hints(
            basic_pitch_candidate_union(
                features,
                configs=candidate_configs_for_checkpoint(payload),
                minimum_confidence=confidence,
            ),
            index.events,
        )
    )
    if progress:
        progress("aligner")
    lattice = SparseJointLattice(model, lattice_config or LatticeConfig())
    path = lattice.decode(candidates, index.events)
    events = tuple(path.joint_events(candidates))
    return JointSampleResult(
        sample_id=sample.name,
        candidates=candidates,
        score=index.events,
        path=path,
        events=events,
        checkpoint=ckpt,
        minimum_candidate_confidence=confidence,
        cache_path=cache,
        candidate_generation=candidate_generation,
        checkpoint_schema=str(payload.get("schema_version") or "unknown"),
    )


def match_sounding_index(
    score_event: ScoreEvent,
    sounding: Sequence[Any],
    *,
    ql_slack: float = 0.05,
) -> int | None:
    """Map a joint score event onto a GUI sounding-note index."""

    best: int | None = None
    best_delta: float | None = None
    pitch = int(score_event.pitch)
    onset = float(score_event.ql_start)
    for note in sounding:
        if int(getattr(note, "pitch")) != pitch:
            continue
        start = float(getattr(note, "ql_start"))
        end = float(getattr(note, "ql_end", start))
        if onset < start - ql_slack or onset >= end + ql_slack:
            continue
        delta = abs(onset - start)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best = int(getattr(note, "index"))
    return best


def build_gui_alignment_payload(
    result: JointSampleResult,
    sounding: Sequence[Any],
) -> dict[str, Any]:
    """Emit the DataCreate `note_alignment_v2.json` document.

    Kept path events remain the alignment surface. Joint ``NOISE`` steps are
    still written into ``transcribed_notes`` with ``ignored=True`` so the GUI
    can show extras the post-processor discarded without treating them as
    real ``extra_note`` labels.
    """

    sounding_by_index = {
        int(getattr(note, "index")): note for note in sounding
    }
    transcribed: list[dict[str, Any]] = []
    mapping: list[int | None] = []
    events: list[dict[str, Any]] = []
    ignored_count = 0
    for step in result.path.steps:
        candidate = result.candidates[step.candidate_index]
        ignored = step.operation == JointOperation.NOISE
        score_index = None
        if not ignored and step.score_span is not None:
            score_index = match_sounding_index(
                result.score[step.score_span[0]], sounding
            )
        note_payload: dict[str, Any] = {
            "pitch": int(candidate.pitch),
            "start": float(candidate.start),
            "end": float(candidate.end),
            "confidence": float(candidate.confidence),
        }
        if ignored:
            note_payload["ignored"] = True
            note_payload["ignored_reason"] = "joint_noise"
            ignored_count += 1
        transcribed.append(note_payload)
        mapping.append(None if ignored else score_index)
        if ignored or score_index is None:
            continue
        written = sounding_by_index.get(score_index)
        if written is None:
            continue
        relationship = (
            "copy"
            if step.structural_operation
            in {JointOperation.REPEAT_ENTER, JointOperation.REPLAY}
            else (
                "match"
                if step.operation == JointOperation.MATCH
                else "substitute"
            )
        )
        event_index = len(events)
        midi = int(getattr(written, "pitch"))
        events.append(
            {
                "id": f"aligned_{event_index:05d}",
                "note_id": f"note_{score_index:04d}",
                "sounding_index": score_index,
                "score_index": score_index,
                "is_rest": False,
                "pitch": _midi_pitch_name(midi),
                "midi": midi,
                "measure": getattr(written, "measure", None),
                "duration_ql": float(getattr(written, "ql_end"))
                - float(getattr(written, "ql_start")),
                "ref_start": float(getattr(written, "start")),
                "ref_end": float(getattr(written, "end")),
                "perf_start": float(candidate.start),
                "perf_end": float(candidate.end),
                "alignment_kind": relationship,
                "is_repetition": relationship == "copy",
            }
        )
    kept_count = sum(
        1 for step in result.path.steps if step.operation != JointOperation.NOISE
    )
    return {
        "format_version": 2,
        "engine": "align-joint",
        "sample_id": result.sample_id,
        "events": events,
        "labels": [],
        "transcribed_notes": transcribed,
        "note_mapping": mapping,
        "repetitions": [],
        "summary": {
            "engine": "align-joint",
            "backend": "joint-path-crf",
            "checkpoint": str(result.checkpoint),
            "checkpoint_schema": result.checkpoint_schema,
            "candidate_generation": result.candidate_generation,
            "event_count": len(events),
            "transcribed_note_count": len(transcribed),
            "mapped_note_count": sum(value is not None for value in mapping),
            "ignored_note_count": ignored_count,
            "candidate_count": len(result.candidates),
            "kept_note_count": kept_count,
            "score_event_count": len(result.score),
            "minimum_candidate_confidence": result.minimum_candidate_confidence,
            "path_score": float(result.path.score),
        },
    }
