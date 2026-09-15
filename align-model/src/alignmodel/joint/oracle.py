from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from alignmodel.stages.repetition import find_note_sequence_repetitions
from alignmodel.validated_targets import target_note_map

from .index import JointEvent, ScoreEventIndex
from .metrics import (
    JointMetricSample,
    evaluate_joint_dataset,
    pair_exact_pitch_onset,
)


_FORBIDDEN_SPLIT_TERMS = ("test", "sealed", "challenge")


@dataclass(frozen=True)
class OracleHarnessConfig:
    manifest: Path
    split: str
    output: Path
    candidate_root: Path | None = None
    note_map_root: Path | None = None
    limit: int | None = None
    candidate_pairing_tolerance_sec: float = 0.050
    minimum_candidate_confidence: float = 0.0
    repetition_min_notes: int = 6
    repetition_max_notes: int = 64
    repetition_min_confidence: float = 0.80


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc


def _manifest_rows(document: Any, split: str) -> list[Mapping[str, Any]]:
    if not isinstance(document, Mapping):
        raise ValueError("Manifest must be a JSON object keyed by split")
    container = document.get("splits", document)
    if not isinstance(container, Mapping) or split not in container:
        raise ValueError(f"Manifest has no explicit {split!r} split")
    rows = container[split]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Manifest split {split!r} must be a non-empty list")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Manifest row {index} is not an object")
        declared = row.get("split")
        if declared is not None and str(declared) != split:
            raise ValueError(
                f"Manifest row {index} declares split {declared!r}, "
                f"not {split!r}"
            )
    return rows


def _assert_non_test_split(split: str) -> None:
    normalized = split.strip().lower()
    if not normalized:
        raise ValueError("An explicit non-test split is required")
    if any(term in normalized for term in _FORBIDDEN_SPLIT_TERMS):
        raise ValueError(
            f"Refusing oracle evaluation on protected split {split!r}"
        )


def _sample_dir(row: Mapping[str, Any]) -> Path:
    value = row.get("sample_dir")
    if not value:
        raise ValueError("Manifest row lacks sample_dir")
    path = Path(str(value))
    if not path.is_dir():
        raise FileNotFoundError(f"Sample directory does not exist: {path}")
    return path


def _note_map_path(
    row: Mapping[str, Any],
    sample_dir: Path,
    root: Path | None,
) -> Path:
    if row.get("note_map"):
        candidates = [Path(str(row["note_map"]))]
    elif root is not None:
        corpus = str(row.get("corpus") or row.get("root") or "")
        candidates = [
            root / corpus / sample_dir.name / "note_map.json",
            root / sample_dir.name / "note_map.json",
        ]
    else:
        candidates = [sample_dir / "note_map.json"]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Exact note_map.json not found; checked "
        + ", ".join(str(path) for path in candidates)
    )


def _lineage_for_row(
    row: Mapping[str, Any],
    sample_dir: Path,
    root: Path | None,
) -> Mapping[str, Any]:
    if row.get("target_db") is not None or row.get("target_record") is not None:
        if row.get("target_db") is None or row.get("target_record") is None:
            raise ValueError("Manifest row has an incomplete SQLite target reference")
        return target_note_map(row)
    return _read_json(_note_map_path(row, sample_dir, root))


def _candidate_paths(
    root: Path, row: Mapping[str, Any], sample_dir: Path
) -> list[Path]:
    corpus = str(row.get("corpus") or row.get("root") or "")
    stems = [
        root / corpus / sample_dir.name,
        root / "basic-pitch" / corpus / sample_dir.name,
        root / sample_dir.name,
    ]
    return [path.with_suffix(suffix) for path in stems for suffix in (".npz", ".json")]


def _plain_candidate_events(document: Any, path: Path) -> list[JointEvent]:
    rows = document.get("notes") if isinstance(document, Mapping) else document
    if not isinstance(rows, list):
        raise ValueError(f"Candidate JSON {path} must contain a notes list")
    events = [
        JointEvent(
            pitch=int(row.get("pitch", row.get("written_pitch"))),
            start=float(row["start"]),
            end=max(float(row["end"]), float(row["start"]) + 0.001),
            score_span=None,
            relationship="extra",
            confidence=float(row.get("confidence", 1.0)),
        )
        for row in rows
    ]
    return sorted(events, key=lambda value: (value.start, value.pitch))


def _load_npz_candidates(
    path: Path,
    *,
    row: Mapping[str, Any],
    sample_dir: Path,
) -> list[JointEvent]:
    with np.load(path, allow_pickle=False) as saved:
        names = set(saved.files)
        if {"pitch", "start", "end"} <= names:
            pitches = np.asarray(saved["pitch"])
            starts = np.asarray(saved["start"])
            ends = np.asarray(saved["end"])
            confidence = (
                np.asarray(saved["confidence"])
                if "confidence" in names
                else np.ones_like(starts, dtype=np.float32)
            )
            if not (
                pitches.ndim == starts.ndim == ends.ndim == confidence.ndim == 1
                and len(pitches) == len(starts) == len(ends) == len(confidence)
            ):
                raise ValueError(f"Malformed decoded candidate arrays in {path}")
            return sorted(
                [
                    JointEvent(
                        pitch=int(pitch),
                        start=float(start),
                        end=max(float(end), float(start) + 0.001),
                        score_span=None,
                        relationship="extra",
                        confidence=float(conf),
                    )
                    for pitch, start, end, conf in zip(
                        pitches, starts, ends, confidence
                    )
                ],
                key=lambda value: (value.start, value.pitch),
            )

    # Raw Basic Pitch activations are accepted only through the cache validator.
    from alignmodel.joint.candidates import basic_pitch_candidate_union
    from alignmodel.transcription.basic_pitch import (
        load_audio_metadata,
        load_basic_pitch_cache,
    )

    wav_path = Path(
        str(row.get("audio_path") or sample_dir / "performance_audio.wav")
    )
    if not wav_path.is_file():
        raise FileNotFoundError(f"Audio for cache validation is missing: {wav_path}")
    source_metadata = load_audio_metadata(sample_dir)
    expected_hash = (row.get("source_hashes") or {}).get(
        "performance_audio.wav"
    )
    features = load_basic_pitch_cache(
        path,
        wav_path,
        source_metadata,
        wav_sha256=str(expected_hash) if expected_hash else None,
    )
    if features is None:
        raise ValueError(f"Stale or malformed Basic Pitch cache: {path}")
    decoded = basic_pitch_candidate_union(features)
    return sorted(
        [
            JointEvent(
                pitch=int(value.pitch),
                start=float(value.start),
                end=max(float(value.end), float(value.start) + 0.001),
                score_span=None,
                relationship="extra",
                confidence=float(value.confidence),
            )
            for value in decoded
        ],
        key=lambda value: (value.start, value.pitch),
    )


def _load_candidates(
    root: Path,
    row: Mapping[str, Any],
    sample_dir: Path,
    *,
    minimum_confidence: float = 0.0,
) -> list[JointEvent]:
    candidates = _candidate_paths(root, row, sample_dir)
    path = next((value for value in candidates if value.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            "Candidate cache not found; checked "
            + ", ".join(str(value) for value in candidates)
        )
    if path.suffix == ".json":
        values = _plain_candidate_events(_read_json(path), path)
    else:
        values = _load_npz_candidates(path, row=row, sample_dir=sample_dir)
    return [
        value for value in values if value.confidence >= minimum_confidence
    ]


def _transfer_oracle_mapping(
    candidates: Sequence[JointEvent],
    target: Sequence[JointEvent],
    tolerance_sec: float,
) -> list[JointEvent]:
    pairs = pair_exact_pitch_onset(
        candidates, target, tolerance_sec=tolerance_sec
    )
    target_by_prediction = {pred: gold for pred, gold in pairs}
    projected: list[JointEvent] = []
    for index, candidate in enumerate(candidates):
        gold_index = target_by_prediction.get(index)
        if gold_index is None:
            projected.append(candidate)
            continue
        gold = target[gold_index]
        projected.append(
            replace(
                candidate,
                score_span=gold.score_span,
                relationship=gold.relationship,
                copy_pass=gold.copy_pass,
                origin_relationship=gold.origin_relationship,
            )
        )
    return projected


def _predicted_repeat_events(
    target: Sequence[JointEvent],
    config: OracleHarnessConfig,
) -> list[JointEvent]:
    repetitions = find_note_sequence_repetitions(
        list(target),
        min_notes=config.repetition_min_notes,
        max_notes=config.repetition_max_notes,
        min_confidence=config.repetition_min_confidence,
    )
    projected = [
        replace(
            event,
            relationship=(
                "extra"
                if event.is_extra
                else (
                    event.origin_relationship
                    if event.origin_relationship in {"match", "substitute"}
                    else "match"
                )
            ),
            copy_pass=0,
        )
        for event in target
    ]
    for repetition in repetitions:
        length = min(
            repetition.source_i1 - repetition.source_i0,
            repetition.repeat_i1 - repetition.repeat_i0,
        )
        for offset in range(length):
            source = projected[repetition.source_i0 + offset]
            repeat_index = repetition.repeat_i0 + offset
            projected[repeat_index] = replace(
                projected[repeat_index],
                score_span=source.score_span,
                relationship="copy",
                copy_pass=1,
                origin_relationship=source.relationship,
            )
    return projected


def _atomic_json_write(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def run_oracle_harness(config: OracleHarnessConfig) -> dict[str, Any]:
    """Run diagnostic oracle controls on one explicit, non-test manifest split."""

    _assert_non_test_split(config.split)
    manifest_path = config.manifest.resolve()
    rows = _manifest_rows(_read_json(manifest_path), config.split)
    if config.limit is not None:
        if config.limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[: config.limit]

    stage_samples: dict[str, list[JointMetricSample]] = {
        "oracle_notes_oracle_repeats": [],
        "oracle_notes_predicted_repeats": [],
    }
    if config.candidate_root is not None:
        stage_samples["acoustic_candidates_oracle_repeats"] = []

    sample_ids: list[str] = []
    for row in rows:
        sample_dir = _sample_dir(row)
        score_path = Path(
            str(row.get("score_path") or sample_dir / "verified_score.musicxml")
        )
        if not score_path.is_file():
            raise FileNotFoundError(f"Verified score is missing: {score_path}")
        lineage = _lineage_for_row(row, sample_dir, config.note_map_root)
        index = ScoreEventIndex.from_musicxml(score_path, lineage)
        target = index.rendered_events
        if not target:
            raise ValueError(f"No rendered/performed events for {sample_dir}")
        source = str(row.get("source") or row.get("corpus") or "unknown")
        sample_id = str(row.get("sample") or sample_dir.name)
        sample_ids.append(sample_id)

        common = {
            "target": target,
            "source": source,
            "target_deletions": index.deleted_event_indices,
            "score_event_count": index.score_event_count,
        }
        stage_samples["oracle_notes_oracle_repeats"].append(
            JointMetricSample(
                predicted=target,
                predicted_deletions=index.deleted_event_indices,
                **common,
            )
        )
        stage_samples["oracle_notes_predicted_repeats"].append(
            JointMetricSample(
                predicted=_predicted_repeat_events(target, config),
                predicted_deletions=index.deleted_event_indices,
                **common,
            )
        )
        if config.candidate_root is not None:
            candidates = _load_candidates(
                config.candidate_root,
                row,
                sample_dir,
                minimum_confidence=config.minimum_candidate_confidence,
            )
            stage_samples["acoustic_candidates_oracle_repeats"].append(
                JointMetricSample(
                    predicted=_transfer_oracle_mapping(
                        candidates,
                        target,
                        config.candidate_pairing_tolerance_sec,
                    ),
                    **common,
                )
            )

    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    result: dict[str, Any] = {
        "schema_version": "align-joint-oracle-v1",
        "diagnostic_only": True,
        "split": config.split,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "n_samples": len(rows),
        "sample_ids": sample_ids,
        "candidate_pairing_uses_oracle_mapping": (
            config.candidate_root is not None
        ),
        "stage_definitions": {
            "oracle_notes_oracle_repeats": (
                "Exact rendered lineage, mapping, copy state, and deletions."
            ),
            "oracle_notes_predicted_repeats": (
                "Exact rendered notes/mapping with score-blind heuristic "
                "repeat-state predictions."
            ),
            **(
                {
                    "acoustic_candidates_oracle_repeats": (
                        "Cached acoustic notes paired to rendered lineage; "
                        "paired events receive oracle mapping/copy state. "
                        "This is an acoustic ceiling, not deployable inference."
                    )
                }
                if config.candidate_root is not None
                else {}
            ),
        },
        "stages": {
            stage: evaluate_joint_dataset(samples)
            for stage, samples in stage_samples.items()
        },
    }
    _atomic_json_write(config.output, result)
    return result
