"""Map timestamped Extra/Missing/Correct notes onto canonical score events."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from .index import JointEvent
from .metrics import (
    JointMetricSample,
    evaluate_joint_dataset,
    evaluate_joint_events,
    pair_exact_pitch_onset,
)

CLASS_KEYS = ("extra", "missing", "correct")
NOTE_WISE_ADAPTER_SCHEMA = "align-note-wise-score-event-metric-v1"


NoteTriple = tuple[float, float, int]


def _as_note(value: NoteTriple | Mapping[str, Any]) -> NoteTriple:
    if isinstance(value, Mapping):
        start = float(value.get("start", value.get("onset", value.get("start_time"))))
        end = float(value.get("end", value.get("offset", value.get("end_time"))))
        pitch = int(value.get("pitch", value.get("sounding_pitch")))
        return start, end, pitch
    start, end, pitch = value
    return float(start), float(end), int(pitch)


def _unlinked_event(note: NoteTriple, *, rendered_index: int | None = None) -> JointEvent:
    start, end, pitch = _as_note(note)
    return JointEvent(
        pitch=pitch,
        start=start,
        end=max(end, start + 0.001),
        score_span=None,
        relationship="extra",
        rendered_index=rendered_index,
    )


def score_notes_from_class_gold(
    gold_by_class: Mapping[str, Sequence[NoteTriple | Mapping[str, Any]]],
) -> list[NoteTriple]:
    """Build a reference partition from gold Correct and Missing notes."""

    notes = [
        _as_note(value)
        for key in ("correct", "missing")
        for value in gold_by_class.get(key) or ()
    ]
    notes.sort(key=lambda item: (item[0], item[2], item[1]))
    return notes


def score_events_from_notes(
    score_notes: Sequence[NoteTriple | Mapping[str, Any]],
) -> list[JointEvent]:
    events = []
    for index, note in enumerate(score_notes):
        start, end, pitch = _as_note(note)
        events.append(
            JointEvent(
                pitch=pitch,
                start=start,
                end=max(end, start + 0.001),
                score_span=(index, index + 1),
                relationship="match",
            )
        )
    return events


def adapt_class_notes(
    notes_by_class: Mapping[str, Sequence[NoteTriple | Mapping[str, Any]]],
    score_notes: Sequence[NoteTriple | Mapping[str, Any]],
    *,
    tolerance_sec: float = 0.050,
) -> tuple[list[JointEvent], frozenset[int]]:
    """Project Extra/Missing/Correct notes onto exclusive score-event identities.

    Performed Correct and Extra notes are paired onto score events by exact pitch
    and onset. A paired Correct note becomes ``match`` (or ``substitute`` if the
    written pitch differs). A paired Extra note keeps type ``extra`` at that
    score location so a type mismatch can receive half credit. Unpaired Extra
    notes receive an audited extra identity. Missing notes pair onto remaining
    score events and become deletions.
    """

    score = score_events_from_notes(score_notes)
    performed: list[JointEvent] = []
    performed_class: list[str] = []
    for kind in ("correct", "extra"):
        for note in notes_by_class.get(kind) or ():
            performed.append(_unlinked_event(note))
            performed_class.append(kind)
    performed_order = sorted(range(len(performed)), key=lambda index: performed[index].start)
    performed = [performed[index] for index in performed_order]
    performed_class = [performed_class[index] for index in performed_order]

    pairs = pair_exact_pitch_onset(performed, score, tolerance_sec=tolerance_sec)
    paired_performed = {pred_index for pred_index, _ in pairs}
    paired_score = {gold_index for _, gold_index in pairs}

    events: list[JointEvent] = []
    for pred_index, gold_index in pairs:
        score_event = score[gold_index]
        source = performed[pred_index]
        kind = performed_class[pred_index]
        if kind == "correct":
            relationship = (
                "match" if source.pitch == score_event.pitch else "substitute"
            )
            events.append(
                replace(
                    source,
                    score_span=score_event.score_span,
                    relationship=relationship,
                )
            )
            continue
        events.append(
            replace(
                source,
                score_span=score_event.score_span,
                relationship="extra",
            )
        )

    extra_identity = 0
    for pred_index, source in enumerate(performed):
        if pred_index in paired_performed:
            continue
        events.append(
            replace(
                source,
                relationship="extra",
                rendered_index=extra_identity,
            )
        )
        extra_identity += 1

    missing = [
        _unlinked_event(note)
        for note in notes_by_class.get("missing") or ()
    ]
    missing.sort(key=lambda event: event.start)
    remaining_indices = [
        index for index in range(len(score)) if index not in paired_score
    ]
    remaining = [score[index] for index in remaining_indices]
    miss_pairs = pair_exact_pitch_onset(
        missing, remaining, tolerance_sec=tolerance_sec
    )
    deletions = {remaining_indices[gold_index] for _, gold_index in miss_pairs}
    events.sort(key=lambda event: (event.start, event.pitch, event.end))
    return events, frozenset(int(value) for value in deletions)


def _transfer_extra_identities(
    predicted: Sequence[JointEvent],
    target: Sequence[JointEvent],
    *,
    tolerance_sec: float,
) -> list[JointEvent]:
    """Copy gold extra identities onto pitch+onset matched unmatched extras."""

    pred_extras = [
        (index, event)
        for index, event in enumerate(predicted)
        if event.score_span is None
    ]
    gold_extras = [
        (index, event)
        for index, event in enumerate(target)
        if event.score_span is None
    ]
    if not pred_extras:
        return list(predicted)
    pairs = pair_exact_pitch_onset(
        [event for _, event in pred_extras],
        [event for _, event in gold_extras],
        tolerance_sec=tolerance_sec,
    )
    paired = {int(pred_index) for pred_index, _ in pairs}
    remapped = list(predicted)
    for pred_index, gold_index in pairs:
        source_index = pred_extras[pred_index][0]
        gold_event = gold_extras[gold_index][1]
        remapped[source_index] = replace(
            remapped[source_index],
            rendered_index=gold_event.rendered_index,
        )
    for local_index, (source_index, event) in enumerate(pred_extras):
        if local_index in paired:
            continue
        remapped[source_index] = replace(event, rendered_index=None)
    return remapped


def evaluate_class_notes_note_wise(
    gold_by_class: Mapping[str, Sequence[NoteTriple | Mapping[str, Any]]],
    pred_by_class: Mapping[str, Sequence[NoteTriple | Mapping[str, Any]]],
    score_notes: Sequence[NoteTriple | Mapping[str, Any]] | None = None,
    *,
    tolerance_sec: float = 0.050,
) -> dict[str, Any]:
    """Score class-labeled notes with the official note-wise matcher."""

    reference = (
        list(score_notes)
        if score_notes is not None
        else score_notes_from_class_gold(gold_by_class)
    )
    predicted, predicted_deletions = adapt_class_notes(
        pred_by_class, reference, tolerance_sec=tolerance_sec
    )
    target, target_deletions = adapt_class_notes(
        gold_by_class, reference, tolerance_sec=tolerance_sec
    )
    predicted = _transfer_extra_identities(
        predicted, target, tolerance_sec=tolerance_sec
    )
    report = evaluate_joint_events(
        predicted,
        target,
        predicted_deletions=predicted_deletions,
        target_deletions=target_deletions,
        score_event_count=len(reference),
    )
    official = dict(report["official_note_wise"])
    official["adapter"] = {
        "schema_version": NOTE_WISE_ADAPTER_SCHEMA,
        "tolerance_sec": float(tolerance_sec),
        "score_event_count": len(reference),
        "synthesized_score_notes": score_notes is None,
    }
    return official


def class_notes_metric_sample(
    gold_by_class: Mapping[str, Sequence[NoteTriple | Mapping[str, Any]]],
    pred_by_class: Mapping[str, Sequence[NoteTriple | Mapping[str, Any]]],
    score_notes: Sequence[NoteTriple | Mapping[str, Any]] | None = None,
    *,
    source: str = "unknown",
    tolerance_sec: float = 0.050,
) -> JointMetricSample:
    reference = (
        list(score_notes)
        if score_notes is not None
        else score_notes_from_class_gold(gold_by_class)
    )
    predicted, predicted_deletions = adapt_class_notes(
        pred_by_class, reference, tolerance_sec=tolerance_sec
    )
    target, target_deletions = adapt_class_notes(
        gold_by_class, reference, tolerance_sec=tolerance_sec
    )
    predicted = _transfer_extra_identities(
        predicted, target, tolerance_sec=tolerance_sec
    )
    return JointMetricSample(
        predicted=predicted,
        target=target,
        source=source,
        predicted_deletions=predicted_deletions,
        target_deletions=target_deletions,
        score_event_count=len(reference),
    )


def evaluate_class_notes_dataset(
    samples: Sequence[tuple[Mapping[str, Sequence[Any]], Mapping[str, Sequence[Any]], Sequence[Any] | None]],
    *,
    tolerance_sec: float = 0.050,
) -> dict[str, Any]:
    metric_samples = [
        class_notes_metric_sample(
            gold, pred, score, source=f"clip_{index}", tolerance_sec=tolerance_sec
        )
        for index, (gold, pred, score) in enumerate(samples)
    ]
    return evaluate_joint_dataset(metric_samples)
