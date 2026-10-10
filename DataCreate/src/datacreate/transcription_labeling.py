"""Restore model feedback or derive conservative labels from transcriptions.

Versioned model exports carry final feedback that is preserved when re-labeling.
For legacy artifacts, this module compares the decoded pitch sequence with the
verified score using an independent edit-distance implementation. It does not
run ALIGN's learned or deterministic aligners. Agent annotations can be reviewed
separately from ``labels.json``.
"""

from __future__ import annotations

import json
from copy import deepcopy
import statistics
import wave
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from datacreate.melody import (
    ScoreSoundingNote,
    extra_neighbor_core,
    padded_melody,
    parse_sounding_notes,
)

AGENT_LABEL_FILENAME = "labels_agent.json"
AGENT_ANNOTATOR_ID = "cursor_agent_transcription_review_v1"
ALIGNMENT_AGENT_ANNOTATOR_ID = "cursor_agent_current_alignment_review_v1"
MAX_LABELS_PER_TYPE = 10
NOTE_ALIGNMENT_FILENAME = "note_alignment_v2.json"


def _has_model_feedback(payload: Mapping[str, Any]) -> bool:
    """Model feedback must not be reconstructed from a lossy scalar mapping.

    Before this contract was explicit, v9 exports already carried final gated
    labels, but `repetitions` contained labels rather than legacy note ranges.
    Keep those installed exports readable without requiring another ML run.
    Empty final labels are also authoritative (clean take or model abstention).
    """
    contract = payload.get("label_generation") or {}
    if contract.get("schema_version") == "datacreate-model-feedback-v1":
        if not isinstance(payload.get("labels"), list):
            raise ValueError("Model feedback artifact must contain a labels list")
        return True
    summary = payload.get("summary") or {}
    repair = (payload.get("diagnostics") or {}).get("same_pitch_repair") or {}
    return (
        payload.get("engine") == "align-joint"
        and summary.get("backend") == "ALIGN v9 (experimental)"
        and repair.get("schema_version") == "same-pitch-repair-v2"
        and isinstance(payload.get("labels"), list)
    )


def _model_feedback_document(payload: dict[str, Any]) -> dict[str, Any]:
    """Preserve canonical identities, copy counts, gates and playback ranges."""
    from datacreate.models import LabelsDocument

    contract = payload.get("label_generation") or {}
    summary = payload.get("summary") or {}
    diagnostics = payload.get("diagnostics") or {}
    status = diagnostics.get("status", summary.get("status"))
    if status is None:
        raise ValueError("Model feedback artifact is missing its assessment status")
    labels = deepcopy(payload["labels"]) if status == "ok" else []
    score_only = deepcopy(payload.get("score_only_labels") or []) if status == "ok" else []
    for label in labels:
        label["source"] = "agent"
    counts = dict(sorted(Counter(label["type"] for label in labels).items()))
    document = {
        "schema_version": "1.2",
        "audio_reference": "performance_audio.wav",
        "annotator_id": contract.get("annotator_id", "align_stack_v9_review"),
        "self_reported": [],
        "labels": labels,
        "agent_labeling": {
            **deepcopy(payload.get("provenance") or {}),
            "method": contract.get("method", "align_stack_v9"),
            "status": status,
            "alignment_artifact": NOTE_ALIGNMENT_FILENAME,
            "alignment_engine": payload.get("engine"),
            "label_source": "alignment_model_feedback",
            "uses_project_alignment_or_error_models": True,
            "training_performed": False,
            "maximum_labels_per_type": None,
            "raw_counts_by_type": counts,
            "kept_counts_by_type": counts,
            "dismissed_types": [],
            "replaced_previous_agent_labels": True,
            "score_only_labels": score_only,
            "labels_without_playback_time": [r['id'] for r in score_only],
        },
    }
    # Validate without serializing through the old Label model: it drops newer
    # fields such as score_event_indices and timing_status.
    LabelsDocument.model_validate(document)
    return document


@dataclass(frozen=True)
class TranscribedNote:
    pitch: int
    start: float
    end: float
    confidence: float


@dataclass(frozen=True)
class AlignmentOperation:
    kind: str
    score_index: int | None
    transcription_index: int | None


@dataclass(frozen=True)
class RepetitionMatch:
    inserted_indices: tuple[int, ...]
    score_start: int
    score_end: int
    repeat_start: float
    repeat_end: float
    source_start: float
    source_end: float
    similarity: float


def load_transcription(path: Path) -> list[TranscribedNote]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    notes = []
    for raw in payload.get("transcribed_notes") or []:
        if raw.get("ignored"):
            continue
        start = float(raw.get("start") or 0.0)
        end = max(start + 0.001, float(raw.get("end") or start + 0.05))
        notes.append(
            TranscribedNote(
                pitch=int(raw["pitch"]),
                start=start,
                end=end,
                confidence=float(raw.get("confidence") or 0.0),
            )
        )
    return sorted(notes, key=lambda item: (item.start, item.end, item.pitch))


def _substitution_cost(score_pitch: int, heard_pitch: int) -> float:
    difference = abs(int(score_pitch) - int(heard_pitch))
    if difference == 0:
        return 0.0
    if difference <= 2:
        return 1.05
    if difference <= 5:
        return 1.30
    return 1.60


def align_pitch_sequences(
    score: Sequence[ScoreSoundingNote],
    transcription: Sequence[TranscribedNote],
) -> list[AlignmentOperation]:
    """Needleman-Wunsch alignment written for this annotation pass.

    Exact pitches are strongly preferred. A gap costs less than a remote pitch
    substitution, which prevents one missed phrase from becoming a cascade of
    implausible wrong-note labels.
    """

    n_score = len(score)
    n_heard = len(transcription)
    gap_cost = 0.90
    costs = [[0.0] * (n_heard + 1) for _ in range(n_score + 1)]
    parents = [[""] * (n_heard + 1) for _ in range(n_score + 1)]
    for i in range(1, n_score + 1):
        costs[i][0] = i * gap_cost
        parents[i][0] = "delete"
    for j in range(1, n_heard + 1):
        costs[0][j] = j * gap_cost
        parents[0][j] = "insert"

    for i in range(1, n_score + 1):
        score_pitch = score[i - 1].pitch
        for j in range(1, n_heard + 1):
            heard_pitch = transcription[j - 1].pitch
            diagonal = costs[i - 1][j - 1] + _substitution_cost(
                score_pitch, heard_pitch
            )
            delete = costs[i - 1][j] + gap_cost
            insert = costs[i][j - 1] + gap_cost
            # The ordering makes exact diagonal matches win deterministic ties.
            value, _, action = min(
                (
                    (diagonal, 0, "match" if score_pitch == heard_pitch else "substitute"),
                    (delete, 1, "delete"),
                    (insert, 2, "insert"),
                )
            )
            costs[i][j] = value
            parents[i][j] = action

    operations: list[AlignmentOperation] = []
    i, j = n_score, n_heard
    while i or j:
        action = parents[i][j]
        if action in {"match", "substitute"}:
            operations.append(AlignmentOperation(action, i - 1, j - 1))
            i -= 1
            j -= 1
        elif action == "delete":
            operations.append(AlignmentOperation(action, i - 1, None))
            i -= 1
        elif action == "insert":
            operations.append(AlignmentOperation(action, None, j - 1))
            j -= 1
        else:
            raise RuntimeError(f"Missing alignment predecessor at ({i}, {j})")
    operations.reverse()
    return operations


def _lcs_length(a: Sequence[int], b: Sequence[int]) -> int:
    previous = [0] * (len(b) + 1)
    for left in a:
        current = [0] * (len(b) + 1)
        for j, right in enumerate(b, 1):
            current[j] = (
                previous[j - 1] + 1
                if left == right
                else max(previous[j], current[j - 1])
            )
        previous = current
    return previous[-1]


def _sequence_similarity(a: Sequence[int], b: Sequence[int]) -> float:
    if not a or not b:
        return 0.0
    return 2.0 * _lcs_length(a, b) / float(len(a) + len(b))


def _insertion_runs(
    operations: Sequence[AlignmentOperation],
) -> Iterable[tuple[list[int], int]]:
    run: list[int] = []
    score_cursor = 0
    run_cursor = 0
    for operation in operations:
        if operation.score_index is not None:
            score_cursor = operation.score_index + 1
        if operation.kind == "insert" and operation.transcription_index is not None:
            if not run:
                run_cursor = score_cursor
            run.append(operation.transcription_index)
            continue
        if run:
            yield run, run_cursor
            run = []
    if run:
        yield run, run_cursor


def detect_repetitions(
    score: Sequence[ScoreSoundingNote],
    transcription: Sequence[TranscribedNote],
    operations: Sequence[AlignmentOperation],
    *,
    minimum_notes: int = 4,
    minimum_similarity: float = 0.82,
) -> list[RepetitionMatch]:
    """Turn insertion runs that closely replay a score phrase into repetitions."""

    trans_for_score = {
        operation.score_index: operation.transcription_index
        for operation in operations
        if operation.kind in {"match", "substitute"}
        and operation.score_index is not None
        and operation.transcription_index is not None
    }
    matches: list[RepetitionMatch] = []
    for run, score_cursor in _insertion_runs(operations):
        if len(run) < minimum_notes:
            continue
        heard = [transcription[index].pitch for index in run]
        best: tuple[float, int, int] | None = None
        minimum_length = max(minimum_notes, len(heard) - 3)
        maximum_length = min(64, len(heard) + 3, len(score))
        search_start = max(0, score_cursor - 128)
        for length in range(minimum_length, maximum_length + 1):
            latest_start = min(score_cursor, len(score) - length)
            for start in range(search_start, latest_start + 1):
                end = start + length
                similarity = _sequence_similarity(
                    heard, [note.pitch for note in score[start:end]]
                )
                candidate = (similarity, start, end)
                if best is None or candidate > best:
                    best = candidate
        if best is None or best[0] < minimum_similarity:
            continue
        similarity, score_start, score_end = best
        mapped = [
            trans_for_score[index]
            for index in range(score_start, score_end)
            if trans_for_score.get(index) is not None
        ]
        if len(mapped) < minimum_notes:
            continue
        inserted_start = transcription[run[0]].start
        inserted_end = transcription[run[-1]].end
        mapped_start = transcription[min(mapped)].start
        mapped_end = transcription[max(mapped)].end
        if inserted_start < mapped_start:
            source_start, source_end = inserted_start, inserted_end
            repeat_start, repeat_end = mapped_start, mapped_end
        else:
            source_start, source_end = mapped_start, mapped_end
            repeat_start, repeat_end = inserted_start, inserted_end
        matches.append(
            RepetitionMatch(
                inserted_indices=tuple(run),
                score_start=score_start,
                score_end=score_end,
                repeat_start=repeat_start,
                repeat_end=repeat_end,
                source_start=source_start,
                source_end=source_end,
                similarity=similarity,
            )
        )
    return matches


def _audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def _missed_note_time(
    operation_index: int,
    operations: Sequence[AlignmentOperation],
    transcription: Sequence[TranscribedNote],
    audio_duration: float,
) -> tuple[float, float]:
    before = next(
        (
            transcription[operation.transcription_index].end
            for operation in reversed(operations[:operation_index])
            if operation.transcription_index is not None
        ),
        0.0,
    )
    after = next(
        (
            transcription[operation.transcription_index].start
            for operation in operations[operation_index + 1 :]
            if operation.transcription_index is not None
        ),
        audio_duration,
    )
    if after <= before:
        after = min(audio_duration, before + 0.12)
    center = (before + after) / 2.0
    width = min(0.30, max(0.08, (after - before) / 2.0))
    start = max(0.0, center - width / 2.0)
    end = min(audio_duration, max(start + 0.05, center + width / 2.0))
    return start, end


def _score_fields(
    score: list[ScoreSoundingNote],
    core_start: int,
    core_end: int,
    *,
    pad_notes: int = 2,
) -> dict[str, Any]:
    span = padded_melody(score, core_start, core_end, pad_notes)
    fields = span.as_fields()
    fields["score_part"]["core_start_note_index"] = core_start
    fields["score_part"]["core_end_note_index"] = core_end - 1
    fields["core_note_ids"] = [
        score[index].note_id for index in range(core_start, core_end)
    ]
    return fields


def _base_label(
    label_id: str,
    type_name: str,
    start: float,
    end: float,
    comment: str,
) -> dict[str, Any]:
    return {
        "id": label_id,
        "source": "agent",
        "start_time": round(float(start), 4),
        "end_time": round(max(float(end), float(start) + 0.001), 4),
        "type": type_name,
        "severity": 2,
        "deviation_cents": None,
        "deviation_ms": None,
        "measure_number": None,
        "note_id": None,
        "comment": comment,
        "repeats_label_range": None,
        "score_part": None,
        "pitches": None,
        "note_ids": None,
        "core_note_ids": None,
        "extra_copies": None,
    }


def _rhythm_labels(
    score: list[ScoreSoundingNote],
    transcription: list[TranscribedNote],
    operations: Sequence[AlignmentOperation],
    occupied_score: set[int],
) -> list[dict[str, Any]]:
    exact = [
        (operation.score_index, operation.transcription_index)
        for operation in operations
        if operation.kind == "match"
        and operation.score_index is not None
        and operation.transcription_index is not None
    ]
    intervals = []
    for (score_a, trans_a), (score_b, trans_b) in zip(exact, exact[1:]):
        if score_b != score_a + 1 or trans_b != trans_a + 1:
            continue
        score_delta = score[score_b].ql_start - score[score_a].ql_start
        perf_delta = transcription[trans_b].start - transcription[trans_a].start
        if score_delta > 0 and perf_delta > 0:
            intervals.append((score_a, score_b, trans_a, trans_b, perf_delta / score_delta))
    if len(intervals) < 6:
        return []
    tempo_scale = statistics.median(value[-1] for value in intervals)
    if tempo_scale <= 0:
        return []
    labels = []
    for score_a, score_b, trans_a, trans_b, raw_ratio in intervals:
        relative = raw_ratio / tempo_scale
        if 0.42 <= relative <= 2.20:
            continue
        if score_a in occupied_score or score_b in occupied_score:
            continue
        start = transcription[trans_a].start
        end = max(transcription[trans_b].end, start + 0.05)
        label = _base_label(
            f"agent_rhythm_{score_a:04d}",
            "rhythm_error",
            start,
            end,
            (
                "Agent transcription review: adjacent-note onset spacing was "
                f"{relative:.2f}x the take's median score-relative spacing."
            ),
        )
        label.update(_score_fields(score, score_a, score_b + 1))
        label["measure_number"] = score[score_a].measure
        label["note_id"] = score[score_a].note_id
        label["deviation_ms"] = round(
            1000.0 * (raw_ratio - tempo_scale) * max(
                0.0, score[score_b].ql_start - score[score_a].ql_start
            ),
            1,
        )
        labels.append(label)
    return labels


def build_agent_label_document(
    sample_dir: Path,
    *,
    maximum_per_type: int = MAX_LABELS_PER_TYPE,
) -> dict[str, Any]:
    score = parse_sounding_notes(sample_dir / "verified_score.musicxml")
    transcription = load_transcription(sample_dir / "transcription_notes.json")
    operations = align_pitch_sequences(score, transcription)
    repetitions = detect_repetitions(score, transcription, operations)
    repeated_insertions = {
        index for match in repetitions for index in match.inserted_indices
    }
    audio_duration = _audio_duration(sample_dir / "performance_audio.wav")

    labels: list[dict[str, Any]] = []
    occupied_score: set[int] = set()
    for operation_index, operation in enumerate(operations):
        if operation.kind == "match":
            continue
        if operation.kind == "insert":
            trans_index = operation.transcription_index
            if trans_index is None or trans_index in repeated_insertions:
                continue
            note = transcription[trans_index]
            next_score = next(
                (
                    candidate.score_index
                    for candidate in operations[operation_index + 1 :]
                    if candidate.score_index is not None
                ),
                len(score) - 1,
            )
            anchor = max(0, int(next_score or 0) - 1)
            core_start, core_end = extra_neighbor_core(score, anchor)
            label = _base_label(
                f"agent_extra_{trans_index:04d}",
                "extra_note",
                note.start,
                note.end,
                (
                    f"Agent transcription review: Basic Pitch heard MIDI "
                    f"{note.pitch} here, but it had no score counterpart "
                    f"(confidence {note.confidence:.2f})."
                ),
            )
            label.update(_score_fields(score, core_start, core_end))
            label["measure_number"] = score[core_start].measure
            labels.append(label)
            occupied_score.update(range(core_start, core_end))
        elif operation.kind == "delete" and operation.score_index is not None:
            score_index = operation.score_index
            start, end = _missed_note_time(
                operation_index, operations, transcription, audio_duration
            )
            note = score[score_index]
            label = _base_label(
                f"agent_missed_{score_index:04d}",
                "missed_note",
                start,
                end,
                (
                    f"Agent transcription review: no transcribed note matched "
                    f"score MIDI {note.pitch}."
                ),
            )
            label.update(_score_fields(score, score_index, score_index + 1))
            label["measure_number"] = note.measure
            label["note_id"] = note.note_id
            labels.append(label)
            occupied_score.add(score_index)
        elif (
            operation.kind == "substitute"
            and operation.score_index is not None
            and operation.transcription_index is not None
        ):
            score_index = operation.score_index
            trans_index = operation.transcription_index
            written = score[score_index]
            heard = transcription[trans_index]
            label = _base_label(
                f"agent_wrong_{score_index:04d}_{trans_index:04d}",
                "wrong_note",
                heard.start,
                heard.end,
                (
                    f"Agent transcription review: Basic Pitch heard MIDI "
                    f"{heard.pitch}; the score has MIDI {written.pitch} "
                    f"(confidence {heard.confidence:.2f})."
                ),
            )
            label.update(_score_fields(score, score_index, score_index + 1))
            label["measure_number"] = written.measure
            label["note_id"] = written.note_id
            labels.append(label)
            occupied_score.add(score_index)

    for repetition_index, match in enumerate(repetitions):
        label = _base_label(
            f"agent_repetition_{repetition_index:03d}",
            "repetition",
            match.repeat_start,
            match.repeat_end,
            (
                "Agent transcription review: an unmatched note run closely "
                f"replayed this score passage (pitch-sequence similarity "
                f"{match.similarity:.2f})."
            ),
        )
        label.update(_score_fields(score, match.score_start, match.score_end))
        label["measure_number"] = score[match.score_start].measure
        label["note_id"] = score[match.score_start].note_id
        label["repeats_label_range"] = {
            "start_time": round(match.source_start, 4),
            "end_time": round(match.source_end, 4),
        }
        label["extra_copies"] = 1
        labels.append(label)
        occupied_score.update(range(match.score_start, match.score_end))

    labels.extend(
        _rhythm_labels(score, transcription, operations, occupied_score)
    )
    raw_counts = Counter(label["type"] for label in labels)
    dismissed_types = sorted(
        type_name
        for type_name, count in raw_counts.items()
        if count > maximum_per_type
    )
    kept = [
        label for label in labels if label["type"] not in dismissed_types
    ]
    kept.sort(key=lambda item: (item["start_time"], item["end_time"], item["type"]))
    return {
        "schema_version": "1.2",
        "audio_reference": "performance_audio.wav",
        "annotator_id": AGENT_ANNOTATOR_ID,
        "self_reported": [],
        "labels": kept,
        "agent_labeling": {
            "method": "independent_pitch_sequence_review_v1",
            "transcriber": "basic-pitch-frozen",
            "uses_project_alignment_or_error_models": False,
            "maximum_labels_per_type": maximum_per_type,
            "raw_counts_by_type": dict(sorted(raw_counts.items())),
            "kept_counts_by_type": dict(
                sorted(Counter(label["type"] for label in kept).items())
            ),
            "dismissed_types": dismissed_types,
            "score_note_count": len(score),
            "transcribed_note_count": len(transcription),
        },
    }


def write_agent_labels(
    sample_dir: Path,
    *,
    maximum_per_type: int = MAX_LABELS_PER_TYPE,
) -> Path:
    document = build_agent_label_document(
        sample_dir, maximum_per_type=maximum_per_type
    )
    output = sample_dir / AGENT_LABEL_FILENAME
    output.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


def _operations_from_note_mapping(
    score: Sequence[ScoreSoundingNote],
    transcription: Sequence[TranscribedNote],
    note_mapping: Sequence[Any],
) -> list[AlignmentOperation]:
    """Translate a joint note_mapping into edit operations for labeling."""

    operations: list[AlignmentOperation] = []
    next_score = 0
    covered: set[int] = set()
    for transcription_index, raw_score_index in enumerate(note_mapping):
        if transcription_index >= len(transcription):
            break
        if raw_score_index is None:
            operations.append(AlignmentOperation("insert", None, transcription_index))
            continue
        score_index = int(raw_score_index)
        if not (0 <= score_index < len(score)):
            operations.append(AlignmentOperation("insert", None, transcription_index))
            continue
        while next_score < score_index:
            if next_score not in covered:
                operations.append(AlignmentOperation("delete", next_score, None))
            next_score += 1
        heard = transcription[transcription_index].pitch
        written = score[score_index].pitch
        kind = "match" if heard == written else "substitute"
        operations.append(AlignmentOperation(kind, score_index, transcription_index))
        covered.add(score_index)
        next_score = max(next_score, score_index + 1)
    while next_score < len(score):
        if next_score not in covered:
            operations.append(AlignmentOperation("delete", next_score, None))
        next_score += 1
    return operations


def _repetitions_from_alignment_payload(
    payload: dict[str, Any],
    transcription: Sequence[TranscribedNote],
    note_mapping: Sequence[Any],
    *,
    old_to_new: Mapping[int, int] | None = None,
) -> list[RepetitionMatch]:
    matches: list[RepetitionMatch] = []
    index_map = old_to_new or {}

    def _remap(index: int) -> int | None:
        if not index_map:
            return index
        return index_map.get(index)

    for raw in payload.get("repetitions") or []:
        if not isinstance(raw, dict):
            continue
        repeat_i0 = int(raw.get("repeat_i0", -1))
        repeat_i1 = int(raw.get("repeat_i1", -1))
        if repeat_i0 < 0 or repeat_i1 <= repeat_i0:
            continue
        inserted_old = range(repeat_i0, repeat_i1)
        inserted = tuple(
            remapped
            for old_index in inserted_old
            if (remapped := _remap(old_index)) is not None
            and remapped < len(transcription)
        )
        if not inserted:
            continue
        source_i0 = int(raw.get("source_i0", -1))
        source_i1 = int(raw.get("source_i1", -1))
        score_indices = []
        for transcription_index_old in range(max(0, source_i0), max(0, source_i1)):
            transcription_index = _remap(transcription_index_old)
            if transcription_index is None or transcription_index >= len(note_mapping):
                continue
            mapped = note_mapping[transcription_index]
            if mapped is not None:
                score_indices.append(int(mapped))
        if not score_indices:
            # Fallback: treat payload fields as score indices when present.
            if raw.get("score_start") is not None and raw.get("score_end") is not None:
                score_indices = list(
                    range(int(raw["score_start"]), int(raw["score_end"]))
                )
            else:
                continue
        score_start = min(score_indices)
        score_end = max(score_indices) + 1
        source_new = _remap(source_i0) if source_i0 >= 0 else None
        source_end_new = _remap(max(0, source_i1 - 1)) if source_i1 > 0 else None
        matches.append(
            RepetitionMatch(
                inserted_indices=inserted,
                score_start=score_start,
                score_end=score_end,
                repeat_start=float(
                    raw.get("repeat_start", transcription[inserted[0]].start)
                ),
                repeat_end=float(
                    raw.get("repeat_end", transcription[inserted[-1]].end)
                ),
                source_start=float(
                    raw.get(
                        "source_start",
                        transcription[source_new].start
                        if source_new is not None
                        else transcription[inserted[0]].start,
                    )
                ),
                source_end=float(
                    raw.get(
                        "source_end",
                        transcription[source_end_new].end
                        if source_end_new is not None
                        else transcription[inserted[-1]].end,
                    )
                ),
                similarity=float(raw.get("confidence", raw.get("similarity", 1.0))),
            )
        )
    return matches


def _sync_transcription_notes(
    sample_dir: Path, transcription: Sequence[TranscribedNote]
) -> None:
    payload = {
        "source": NOTE_ALIGNMENT_FILENAME,
        "transcribed_notes": [
            {
                "pitch": note.pitch,
                "start": note.start,
                "end": note.end,
                "confidence": note.confidence,
            }
            for note in transcription
        ],
    }
    (sample_dir / "transcription_notes.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_agent_label_document_from_note_alignment(
    sample_dir: Path,
    *,
    maximum_per_type: int = MAX_LABELS_PER_TYPE,
) -> dict[str, Any]:
    """Build agent labels from the sample's current joint alignment artifact."""

    alignment_path = sample_dir / NOTE_ALIGNMENT_FILENAME
    if not alignment_path.is_file():
        raise FileNotFoundError(alignment_path)
    payload = json.loads(alignment_path.read_text(encoding="utf-8"))
    return build_agent_label_document_from_alignment_payload(
        sample_dir, payload, maximum_per_type=maximum_per_type
    )


def build_agent_label_document_from_alignment_payload(
    sample_dir: Path,
    payload: dict[str, Any],
    *,
    maximum_per_type: int = MAX_LABELS_PER_TYPE,
) -> dict[str, Any]:
    """Build agent labels from a note-alignment payload (``note_alignment_v2`` schema)."""

    if _has_model_feedback(payload):
        return _model_feedback_document(payload)

    score = parse_sounding_notes(sample_dir / "verified_score.musicxml")
    transcription, mapping, old_to_new = _kept_notes_and_mapping_from_alignment(
        payload
    )
    if not transcription and not score:
        raise ValueError("Alignment has no transcribed notes or score notes")
    if len(mapping) < len(transcription):
        mapping.extend([None] * (len(transcription) - len(mapping)))
    operations = _operations_from_note_mapping(score, transcription, mapping)
    repetitions = _repetitions_from_alignment_payload(
        payload, transcription, mapping, old_to_new=old_to_new
    )
    if not repetitions:
        repetitions = detect_repetitions(score, transcription, operations)
    repeated_insertions = {
        index for match in repetitions for index in match.inserted_indices
    }
    audio_duration = _audio_duration(sample_dir / "performance_audio.wav")

    labels: list[dict[str, Any]] = []
    occupied_score: set[int] = set()
    for operation_index, operation in enumerate(operations):
        if operation.kind == "match":
            continue
        if operation.kind == "insert":
            trans_index = operation.transcription_index
            if trans_index is None or trans_index in repeated_insertions:
                continue
            note = transcription[trans_index]
            next_score = next(
                (
                    candidate.score_index
                    for candidate in operations[operation_index + 1 :]
                    if candidate.score_index is not None
                ),
                len(score) - 1,
            )
            anchor = max(0, int(next_score or 0) - 1)
            core_start, core_end = extra_neighbor_core(score, anchor)
            label = _base_label(
                f"agent_extra_{trans_index:04d}",
                "extra_note",
                note.start,
                note.end,
                (
                    "Current alignment review: transcribed MIDI "
                    f"{note.pitch} has no score counterpart "
                    f"(confidence {note.confidence:.2f})."
                ),
            )
            if score:
                label.update(_score_fields(score, core_start, core_end))
                label["measure_number"] = score[core_start].measure
                occupied_score.update(range(core_start, core_end))
            labels.append(label)
        elif operation.kind == "delete" and operation.score_index is not None:
            score_index = operation.score_index
            start, end = _missed_note_time(
                operation_index, operations, transcription, audio_duration
            )
            note = score[score_index]
            label = _base_label(
                f"agent_missed_{score_index:04d}",
                "missed_note",
                start,
                end,
                (
                    "Current alignment review: no transcribed note matched "
                    f"score MIDI {note.pitch}."
                ),
            )
            label.update(_score_fields(score, score_index, score_index + 1))
            label["measure_number"] = note.measure
            label["note_id"] = note.note_id
            labels.append(label)
            occupied_score.add(score_index)
        elif (
            operation.kind == "substitute"
            and operation.score_index is not None
            and operation.transcription_index is not None
        ):
            score_index = operation.score_index
            trans_index = operation.transcription_index
            written = score[score_index]
            heard = transcription[trans_index]
            label = _base_label(
                f"agent_wrong_{score_index:04d}_{trans_index:04d}",
                "wrong_note",
                heard.start,
                heard.end,
                (
                    "Current alignment review: transcribed MIDI "
                    f"{heard.pitch}; score has MIDI {written.pitch} "
                    f"(confidence {heard.confidence:.2f})."
                ),
            )
            label.update(_score_fields(score, score_index, score_index + 1))
            label["measure_number"] = written.measure
            label["note_id"] = written.note_id
            labels.append(label)
            occupied_score.add(score_index)

    for repetition_index, match in enumerate(repetitions):
        if not score or match.score_end > len(score):
            continue
        label = _base_label(
            f"agent_repetition_{repetition_index:03d}",
            "repetition",
            match.repeat_start,
            match.repeat_end,
            (
                "Current alignment review: unmatched notes replayed this "
                f"score passage (similarity {match.similarity:.2f})."
            ),
        )
        label.update(_score_fields(score, match.score_start, match.score_end))
        label["measure_number"] = score[match.score_start].measure
        label["note_id"] = score[match.score_start].note_id
        label["repeats_label_range"] = {
            "start_time": round(match.source_start, 4),
            "end_time": round(match.source_end, 4),
        }
        label["extra_copies"] = 1
        labels.append(label)
        occupied_score.update(range(match.score_start, match.score_end))

    labels.extend(_rhythm_labels(score, transcription, operations, occupied_score))
    raw_counts = Counter(label["type"] for label in labels)
    dismissed_types = sorted(
        type_name
        for type_name, count in raw_counts.items()
        if count > maximum_per_type
    )
    kept = [label for label in labels if label["type"] not in dismissed_types]
    kept.sort(key=lambda item: (item["start_time"], item["end_time"], item["type"]))
    return {
        "schema_version": "1.2",
        "audio_reference": "performance_audio.wav",
        "annotator_id": ALIGNMENT_AGENT_ANNOTATOR_ID,
        "self_reported": [],
        "labels": kept,
        "agent_labeling": {
            "method": "current_note_alignment_review_v1",
            "transcriber": "current_sample_transcription",
            "alignment_artifact": NOTE_ALIGNMENT_FILENAME,
            "alignment_engine": payload.get("engine"),
            "uses_project_alignment_or_error_models": True,
            "uses_error_heads": False,
            "maximum_labels_per_type": maximum_per_type,
            "raw_counts_by_type": dict(sorted(raw_counts.items())),
            "kept_counts_by_type": dict(
                sorted(Counter(label["type"] for label in kept).items())
            ),
            "dismissed_types": dismissed_types,
            "score_note_count": len(score),
            "transcribed_note_count": len(transcription),
            "mapped_note_count": sum(
                value is not None for value in mapping[: len(transcription)]
            ),
            "replaced_previous_agent_labels": True,
        },
    }


def notes_from_alignment_payload(payload: dict[str, Any]) -> list[TranscribedNote]:
    return load_transcription_notes(payload.get("transcribed_notes") or [])


def load_transcription_notes(raw_notes: Sequence[dict[str, Any]]) -> list[TranscribedNote]:
    notes = []
    for raw in raw_notes:
        if raw.get("ignored"):
            # Joint post-processor noise: visible in the GUI, not an error label.
            continue
        pitch = raw.get("pitch")
        if pitch is None:
            pitch = raw.get("midi")
        if pitch is None:
            continue
        start = float(raw.get("start") or raw.get("perf_start") or 0.0)
        end = max(start + 0.001, float(raw.get("end") or raw.get("perf_end") or start + 0.05))
        notes.append(
            TranscribedNote(
                pitch=int(pitch),
                start=start,
                end=end,
                confidence=float(raw.get("confidence") or 0.0),
            )
        )
    return sorted(notes, key=lambda item: (item.start, item.end, item.pitch))


def _kept_notes_and_mapping_from_alignment(
    payload: dict[str, Any],
) -> tuple[list[TranscribedNote], list[Any], dict[int, int]]:
    """Drop ignored extras while remapping note_mapping / repetition indices."""

    raw_notes = list(payload.get("transcribed_notes") or [])
    raw_mapping = list(payload.get("note_mapping") or [])
    notes: list[TranscribedNote] = []
    mapping: list[Any] = []
    old_to_new: dict[int, int] = {}
    for old_index, raw in enumerate(raw_notes):
        if raw.get("ignored"):
            continue
        pitch = raw.get("pitch")
        if pitch is None:
            pitch = raw.get("midi")
        if pitch is None:
            continue
        start = float(raw.get("start") or raw.get("perf_start") or 0.0)
        end = max(
            start + 0.001,
            float(raw.get("end") or raw.get("perf_end") or start + 0.05),
        )
        old_to_new[old_index] = len(notes)
        notes.append(
            TranscribedNote(
                pitch=int(pitch),
                start=start,
                end=end,
                confidence=float(raw.get("confidence") or 0.0),
            )
        )
        mapping.append(raw_mapping[old_index] if old_index < len(raw_mapping) else None)
    return notes, mapping, old_to_new


def relabel_sample_from_current_alignment(
    sample_dir: Path,
    *,
    maximum_per_type: int = MAX_LABELS_PER_TYPE,
) -> dict[str, Any]:
    """Rewrite ``labels_agent.json`` from current alignment/transcription artifacts."""

    alignment_path = sample_dir / NOTE_ALIGNMENT_FILENAME
    if alignment_path.is_file():
        payload = json.loads(alignment_path.read_text(encoding="utf-8"))
        if _has_model_feedback(payload):
            from datacreate.feedback_visibility import score_only_feedback
            payload['score_only_labels'] = score_only_feedback(payload, sample_dir)
        document = build_agent_label_document_from_alignment_payload(
            sample_dir, payload, maximum_per_type=maximum_per_type
        )
        if not _has_model_feedback(payload):
            # Legacy relabeling creates a normalized transcription. Versioned
            # model exports already have one, with merge lineage and provenance.
            _sync_transcription_notes(sample_dir, notes_from_alignment_payload(payload))
        source = NOTE_ALIGNMENT_FILENAME
    elif (sample_dir / "transcription_notes.json").is_file():
        document = build_agent_label_document(
            sample_dir, maximum_per_type=maximum_per_type
        )
        source = "transcription_notes.json"
    else:
        raise FileNotFoundError(
            f"{sample_dir.name}: need {NOTE_ALIGNMENT_FILENAME} or transcription_notes.json"
        )

    output = sample_dir / AGENT_LABEL_FILENAME
    output.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(output),
        "source": source,
        "label_count": len(document["labels"]),
        "score_only_label_count": len(document['agent_labeling'].get('score_only_labels') or []),
        "counts_by_type": dict(document["agent_labeling"]["kept_counts_by_type"]),
        "dismissed_types": list(document["agent_labeling"]["dismissed_types"]),
        "method": document["agent_labeling"]["method"],
    }
