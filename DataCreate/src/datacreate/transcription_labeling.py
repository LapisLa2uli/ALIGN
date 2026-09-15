"""Conservative labels derived from Basic Pitch note transcriptions.

This module deliberately does not call ALIGN's learned or deterministic
aligners.  It compares the decoded pitch sequence with the verified score using
an independent edit-distance implementation so the resulting annotations can
be reviewed separately from ``labels.json``.
"""

from __future__ import annotations

import json
import statistics
import wave
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from datacreate.melody import (
    ScoreSoundingNote,
    extra_neighbor_core,
    padded_melody,
    parse_sounding_notes,
)

AGENT_LABEL_FILENAME = "labels_agent.json"
AGENT_ANNOTATOR_ID = "cursor_agent_transcription_review_v1"
MAX_LABELS_PER_TYPE = 10


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
