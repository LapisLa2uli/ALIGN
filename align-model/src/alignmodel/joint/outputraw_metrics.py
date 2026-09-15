"""Validation metrics for the outputRaw full joint pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from alignmodel.melody import match_note_wise_labels_detail

from .index import JointEvent
from .metrics import (
    JointMetricSample,
    evaluate_joint_dataset,
    pair_exact_pitch_onset,
)


@dataclass(frozen=True)
class FullPipelineMetricSample:
    predicted: Sequence[JointEvent]
    target: Sequence[JointEvent]
    predicted_layer2: Sequence[str]
    target_layer2: Sequence[str] = ()
    predicted_rhythm: Sequence[bool] = ()
    target_rhythm: Sequence[bool] = ()
    predicted_duration_sec: Sequence[float] = ()
    predicted_deletions: frozenset[int] = frozenset()
    target_deletions: frozenset[int] = frozenset()
    predicted_resume_events: Sequence[int] = ()
    target_resume_events: Sequence[int] = ()
    score_event_count: int | None = None
    source: str = "unknown"

    def validate(self) -> None:
        if len(self.predicted_layer2) != len(self.predicted):
            raise ValueError("predicted_layer2 must align with predicted events")
        if self.target_layer2 and len(self.target_layer2) != len(self.target):
            raise ValueError("target_layer2 must align with target events")
        if self.predicted_rhythm and len(self.predicted_rhythm) != len(
            self.predicted
        ):
            raise ValueError("predicted_rhythm must align with predicted events")
        if self.target_rhythm and len(self.target_rhythm) != len(self.target):
            raise ValueError("target_rhythm must align with target events")
        if self.predicted_duration_sec and len(
            self.predicted_duration_sec
        ) != len(self.predicted):
            raise ValueError(
                "predicted_duration_sec must align with predicted events"
            )


def _prf(correct: int, predicted: int, target: int) -> dict[str, float | int]:
    if not predicted and not target:
        precision = recall = f1 = 1.0
    else:
        precision = correct / predicted if predicted else 0.0
        recall = correct / target if target else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct": correct,
        "predicted": predicted,
        "target": target,
    }


def _target_layer2(sample: FullPipelineMetricSample) -> tuple[str, ...]:
    if sample.target_layer2:
        return tuple(sample.target_layer2)
    return tuple(
        "extra_note"
        if event.is_extra
        else "wrong_note"
        if (
            event.relationship == "substitute"
            or event.origin_relationship == "substitute"
        )
        else "match"
        for event in sample.target
    )


def _rhythm_values(
    values: Sequence[bool],
    count: int,
) -> tuple[bool, ...]:
    return tuple(bool(value) for value in values) if values else (False,) * count


def _same_pitch_split_count(events: Sequence[JointEvent]) -> int:
    return sum(
        current.pitch == previous.pitch
        and current.start - previous.end <= 0.100
        for previous, current in zip(events, events[1:])
    )


def _bootstrap(
    counts: Sequence[tuple[int, int, int]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, float | int]:
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        selected = generator.integers(0, len(counts), size=len(counts))
        correct = sum(counts[index][0] for index in selected)
        predicted = sum(counts[index][1] for index in selected)
        target = sum(counts[index][2] for index in selected)
        precision = correct / predicted if predicted else 0.0
        recall = correct / target if target else 0.0
        values.append(
            2.0 * precision * recall / max(precision + recall, 1e-12)
        )
    return {
        "replicates": replicates,
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _event_label(event: JointEvent, kind: str) -> dict[str, object]:
    label: dict[str, object] = {"type": kind}
    if event.score_span is not None:
        label["score_event_indices"] = list(
            range(event.score_span[0], event.score_span[1])
        )
        label["copy_pass"] = event.copy_pass
    elif event.rendered_index is not None:
        label["rendered_index"] = event.rendered_index
    return label


def _note_metric(
    gold: Sequence[dict[str, object]],
    predicted: Sequence[dict[str, object]],
    score_event_count: int | None,
) -> dict[str, object]:
    detail = match_note_wise_labels_detail(
        list(gold),
        list(predicted),
        score_event_count=score_event_count,
    )
    if detail["status"] != "available":
        raise ValueError(f"Official note-wise metric unavailable: {detail['reason']}")
    return detail


def _fractional_prf(
    credit: float, predicted: int, target: int
) -> dict[str, float | int]:
    if not predicted and not target:
        precision = recall = f1 = 1.0
    else:
        precision = credit / predicted if predicted else 0.0
        recall = credit / target if target else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "credit": credit,
        "predicted": predicted,
        "target": target,
    }


def evaluate_full_pipeline(
    samples: Sequence[FullPipelineMetricSample],
    *,
    rhythm_threshold: float = 0.5,
    seed: int = 365,
    bootstrap_replicates: int = 1000,
) -> dict[str, object]:
    """Report every promotion metric without consulting on-disk gold."""

    if not samples:
        raise ValueError("Full-pipeline validation requires at least one sample")
    del rhythm_threshold  # Inputs are intentionally thresholded by the caller.
    for sample in samples:
        sample.validate()

    mapping = evaluate_joint_dataset(
        [
            JointMetricSample(
                predicted=sample.predicted,
                target=sample.target,
                source=sample.source,
                predicted_deletions=sample.predicted_deletions,
                target_deletions=sample.target_deletions,
                score_event_count=sample.score_event_count,
            )
            for sample in samples
        ]
    )

    note_correct = note_predicted = note_target = 0
    short_counts = {
        "lt_80ms": [0, 0],
        "lt_120ms": [0, 0],
        "lt_180ms": [0, 0],
    }
    split_count = 0
    layer2_counts = {
        name: [0, 0, 0]
        for name in ("wrong_note", "extra_note", "missed_note")
    }
    rhythm_counts = [0, 0, 0]
    duration_absolute_errors = []
    duration_relative_errors = []
    repeat_counts = [0, 0, 0]
    resume_correct = resume_total = 0
    combined_totals = [0, 0, 0]
    combined_per_clip: list[tuple[int, int, int]] = []
    official_combined = [0.0, 0, 0]
    official_combined_per_clip: list[tuple[float, int, int]] = []
    official_layer2 = {
        name: [0.0, 0, 0]
        for name in ("wrong_note", "extra_note", "missed_note")
    }
    official_rhythm = [0.0, 0, 0]
    official_repeat = [0.0, 0, 0]

    for sample in samples:
        target_layer2 = _target_layer2(sample)
        predicted_rhythm = _rhythm_values(
            sample.predicted_rhythm, len(sample.predicted)
        )
        target_rhythm = _rhythm_values(
            sample.target_rhythm, len(sample.target)
        )
        for type_name in ("wrong_note", "extra_note"):
            pred_labels = [
                _event_label(sample.predicted[index], type_name)
                for index, value in enumerate(sample.predicted_layer2)
                if value == type_name
            ]
            gold_labels = [
                _event_label(sample.target[index], type_name)
                for index, value in enumerate(target_layer2)
                if value == type_name
            ]
            detail = _note_metric(
                gold_labels, pred_labels, sample.score_event_count
            )
            values = official_layer2[type_name]
            values[0] += float(detail["credit"])
            values[1] += len(pred_labels)
            values[2] += len(gold_labels)
        missed = official_layer2["missed_note"]
        missed[0] += len(sample.predicted_deletions & sample.target_deletions)
        missed[1] += len(sample.predicted_deletions)
        missed[2] += len(sample.target_deletions)
        pred_rhythm_labels = [
            _event_label(sample.predicted[index], "rhythm_error")
            for index, value in enumerate(predicted_rhythm)
            if value
        ]
        gold_rhythm_labels = [
            _event_label(sample.target[index], "rhythm_error")
            for index, value in enumerate(target_rhythm)
            if value
        ]
        rhythm_detail = _note_metric(
            gold_rhythm_labels,
            pred_rhythm_labels,
            sample.score_event_count,
        )
        official_rhythm[0] += float(rhythm_detail["credit"])
        official_rhythm[1] += len(pred_rhythm_labels)
        official_rhythm[2] += len(gold_rhythm_labels)
        pred_repeat_labels = [
            _event_label(value, "copy")
            for value in sample.predicted
            if value.is_copy
        ]
        gold_repeat_labels = [
            _event_label(value, "copy")
            for value in sample.target
            if value.is_copy
        ]
        repeat_detail = _note_metric(
            gold_repeat_labels,
            pred_repeat_labels,
            sample.score_event_count,
        )
        official_repeat[0] += float(repeat_detail["credit"])
        official_repeat[1] += len(pred_repeat_labels)
        official_repeat[2] += len(gold_repeat_labels)
        pred_combined = [
            _event_label(
                event,
                "|".join(
                    (
                        event.relationship,
                        str(sample.predicted_layer2[index]),
                        "rhythm" if predicted_rhythm[index] else "no_rhythm",
                        "copy" if event.is_copy else "ordinary",
                    )
                ),
            )
            for index, event in enumerate(sample.predicted)
        ]
        gold_combined = [
            _event_label(
                event,
                "|".join(
                    (
                        event.relationship,
                        str(target_layer2[index]),
                        "rhythm" if target_rhythm[index] else "no_rhythm",
                        "copy" if event.is_copy else "ordinary",
                    )
                ),
            )
            for index, event in enumerate(sample.target)
        ]
        pred_combined.extend(
            {"type": "missed_note", "score_event_indices": [value]}
            for value in sorted(sample.predicted_deletions)
        )
        gold_combined.extend(
            {"type": "missed_note", "score_event_indices": [value]}
            for value in sorted(sample.target_deletions)
        )
        combined_detail = _note_metric(
            gold_combined, pred_combined, sample.score_event_count
        )
        clip_official = (
            float(combined_detail["credit"]),
            len(pred_combined),
            len(gold_combined),
        )
        official_combined_per_clip.append(clip_official)
        for index, value in enumerate(clip_official):
            official_combined[index] += value
        pairs = pair_exact_pitch_onset(
            sample.predicted, sample.target, tolerance_sec=0.050
        )
        note_correct += len(pairs)
        note_predicted += len(sample.predicted)
        note_target += len(sample.target)
        paired_target = {target_index for _, target_index in pairs}
        for label, threshold in (
            ("lt_80ms", 0.080),
            ("lt_120ms", 0.120),
            ("lt_180ms", 0.180),
        ):
            selected = {
                index
                for index, event in enumerate(sample.target)
                if event.end - event.start < threshold
            }
            short_counts[label][0] += len(paired_target & selected)
            short_counts[label][1] += len(selected)
        split_count += _same_pitch_split_count(sample.predicted)

        pair_lookup = {pred: target for pred, target in pairs}
        for type_name in ("wrong_note", "extra_note"):
            predicted_indices = {
                index
                for index, value in enumerate(sample.predicted_layer2)
                if value == type_name
            }
            target_indices = {
                index
                for index, value in enumerate(target_layer2)
                if value == type_name
            }
            correct = sum(
                target_index in target_indices
                for pred_index, target_index in pairs
                if pred_index in predicted_indices
            )
            values = layer2_counts[type_name]
            values[0] += correct
            values[1] += len(predicted_indices)
            values[2] += len(target_indices)
        missed = layer2_counts["missed_note"]
        missed[0] += len(
            sample.predicted_deletions & sample.target_deletions
        )
        missed[1] += len(sample.predicted_deletions)
        missed[2] += len(sample.target_deletions)

        predicted_rhythm_indices = {
            index for index, value in enumerate(predicted_rhythm) if value
        }
        target_rhythm_indices = {
            index for index, value in enumerate(target_rhythm) if value
        }
        rhythm_counts[0] += sum(
            target_index in target_rhythm_indices
            for pred_index, target_index in pairs
            if pred_index in predicted_rhythm_indices
        )
        rhythm_counts[1] += len(predicted_rhythm_indices)
        rhythm_counts[2] += len(target_rhythm_indices)
        predicted_durations = (
            tuple(float(value) for value in sample.predicted_duration_sec)
            if sample.predicted_duration_sec
            else tuple(event.end - event.start for event in sample.predicted)
        )
        for pred_index, target_index in pairs:
            target_duration = max(
                sample.target[target_index].end
                - sample.target[target_index].start,
                1e-3,
            )
            error = abs(predicted_durations[pred_index] - target_duration)
            duration_absolute_errors.append(error)
            duration_relative_errors.append(error / target_duration)

        predicted_copy = {
            index
            for index, event in enumerate(sample.predicted)
            if event.is_copy
        }
        target_copy = {
            index for index, event in enumerate(sample.target) if event.is_copy
        }
        repeat_counts[0] += sum(
            target_index in target_copy
            and sample.predicted[pred_index].score_span
            == sample.target[target_index].score_span
            for pred_index, target_index in pairs
            if pred_index in predicted_copy
        )
        repeat_counts[1] += len(predicted_copy)
        repeat_counts[2] += len(target_copy)
        for predicted, target in zip(
            sample.predicted_resume_events,
            sample.target_resume_events,
        ):
            resume_correct += int(int(predicted) == int(target))
            resume_total += 1
        resume_total += abs(
            len(sample.predicted_resume_events)
            - len(sample.target_resume_events)
        )

        event_correct = 0
        for pred_index, target_index in pair_lookup.items():
            event_correct += int(
                sample.predicted[pred_index].score_span
                == sample.target[target_index].score_span
                and sample.predicted_layer2[pred_index]
                == target_layer2[target_index]
                and predicted_rhythm[pred_index] == target_rhythm[target_index]
                and sample.predicted[pred_index].is_copy
                == sample.target[target_index].is_copy
            )
        deletion_correct = len(
            sample.predicted_deletions & sample.target_deletions
        )
        clip_counts = (
            event_correct + deletion_correct,
            len(sample.predicted) + len(sample.predicted_deletions),
            len(sample.target) + len(sample.target_deletions),
        )
        combined_per_clip.append(clip_counts)
        for index, value in enumerate(clip_counts):
            combined_totals[index] += value

    transcription = _prf(note_correct, note_predicted, note_target)
    transcription.update(
        {
            "count_ratio": note_predicted / max(note_target, 1),
            "short_note_recall": {
                name: {
                    "matched": values[0],
                    "target": values[1],
                    "recall": values[0] / max(values[1], 1),
                }
                for name, values in short_counts.items()
            },
            "same_pitch_split_count": split_count,
            "same_pitch_split_rate": split_count / max(note_predicted, 1),
        }
    )
    diagnostic_timestamp_layer2 = {
        name: _prf(*values) for name, values in layer2_counts.items()
    }
    diagnostic_timestamp_layer2["macro_f1"] = sum(
        float(value["f1"]) for value in diagnostic_timestamp_layer2.values()
    ) / len(layer2_counts)
    layer2 = {
        name: _fractional_prf(*values)
        for name, values in official_layer2.items()
    }
    layer2["macro_f1"] = sum(
        float(value["f1"]) for value in layer2.values()
    ) / len(official_layer2)
    diagnostic_timestamp_repeat = _prf(*repeat_counts)
    repeat = _fractional_prf(*official_repeat)
    repeat["resume_accuracy"] = (
        resume_correct / resume_total if resume_total else 1.0
    )
    repeat["resume_correct"] = resume_correct
    repeat["resume_target"] = resume_total
    diagnostic_timestamp_combined = _prf(*combined_totals)
    combined = _fractional_prf(*official_combined)
    combined["bootstrap"] = _bootstrap(
        official_combined_per_clip,
        seed=seed,
        replicates=bootstrap_replicates,
    )
    return {
        "metric_schema": "align-note-wise-score-event-metric-v1",
        "transcription": mapping["aggregate"]["official_note_wise"],
        "repeat": repeat,
        "mapping": mapping,
        "layer2": layer2,
        "layer3": {
            "rhythm": _fractional_prf(*official_rhythm),
            "duration": {
                "paired": len(duration_absolute_errors),
                "mae_seconds": (
                    float(np.mean(duration_absolute_errors))
                    if duration_absolute_errors
                    else None
                ),
                "median_relative_error": (
                    float(np.median(duration_relative_errors))
                    if duration_relative_errors
                    else None
                ),
                "within_20_percent_accuracy": (
                    sum(value <= 0.20 for value in duration_relative_errors)
                    / len(duration_relative_errors)
                    if duration_relative_errors
                    else None
                ),
            },
        },
        "combined": combined,
        "diagnostic_timestamp_transcription_50ms": transcription,
        "diagnostic_timestamp_layer2_50ms": diagnostic_timestamp_layer2,
        "diagnostic_timestamp_repeat_50ms": diagnostic_timestamp_repeat,
        "diagnostic_timestamp_rhythm_50ms": _prf(*rhythm_counts),
        "diagnostic_timestamp_combined_50ms": diagnostic_timestamp_combined,
        "intonation_masked": True,
    }
