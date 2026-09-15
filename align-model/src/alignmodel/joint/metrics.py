from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from alignmodel.melody import match_note_wise_labels_detail

from .index import JointEvent


DEFAULT_TOLERANCES_SEC = (0.020, 0.050, 0.100)
TYPED_LOCATION_METRIC_SCHEMA = "align-typed-location-metric-v1"


@dataclass(frozen=True)
class JointMetricSample:
    predicted: Sequence[JointEvent]
    target: Sequence[JointEvent]
    source: str = "unknown"
    predicted_deletions: frozenset[int] = frozenset()
    target_deletions: frozenset[int] = frozenset()
    score_event_count: int | None = None


def _label_interval_iou(
    predicted: Mapping[str, Any],
    target: Mapping[str, Any],
) -> float:
    pred_start = float(predicted["start_time"])
    pred_end = float(predicted["end_time"])
    target_start = float(target["start_time"])
    target_end = float(target["end_time"])
    intersection = max(
        0.0,
        min(pred_end, target_end) - max(pred_start, target_start),
    )
    union = max(pred_end, target_end) - min(pred_start, target_start)
    return intersection / union if union > 0.0 else 0.0


def evaluate_typed_location_labels(
    predicted: Sequence[Mapping[str, Any]],
    target: Sequence[Mapping[str, Any]],
    *,
    criterion: str = "iou_0.3",
    type_mismatch_credit: float = 0.5,
) -> dict[str, Any]:
    """One-to-one label-location metric with partial type-mismatch credit.

    A location hit receives full credit when types agree and
    ``type_mismatch_credit`` otherwise. Hungarian assignment maximizes total
    credit globally; input ordering cannot change a greedy matching decision.
    Existing strict metrics are separate and are not redefined here.
    """

    mismatch_credit = float(type_mismatch_credit)
    if not 0.0 <= mismatch_credit <= 1.0:
        raise ValueError("type_mismatch_credit must be within [0, 1]")
    if criterion == "iou_0.3":

        def location_hit(
            left: Mapping[str, Any], right: Mapping[str, Any]
        ) -> bool:
            return _label_interval_iou(left, right) >= 0.3

    elif criterion.startswith("onset_") and criterion.endswith("ms"):
        tolerance_ms = int(criterion[6:-2])
        if tolerance_ms <= 0:
            raise ValueError("Onset tolerance must be positive")
        tolerance = tolerance_ms / 1000.0

        def location_hit(
            left: Mapping[str, Any], right: Mapping[str, Any]
        ) -> bool:
            return (
                abs(
                    float(left["start_time"])
                    - float(right["start_time"])
                )
                <= tolerance + 1e-12
            )

    else:
        raise ValueError(f"Unsupported location criterion: {criterion}")

    predicted_count = len(predicted)
    target_count = len(target)
    if not predicted_count or not target_count:
        perfect_empty = not predicted_count and not target_count
        return {
            "schema_version": TYPED_LOCATION_METRIC_SCHEMA,
            "criterion": criterion,
            "type_mismatch_credit": mismatch_credit,
            "precision": 1.0 if perfect_empty else 0.0,
            "recall": 1.0 if perfect_empty else 0.0,
            "f1": 1.0 if perfect_empty else 0.0,
            "credit": 0.0,
            "predicted": predicted_count,
            "gold": target_count,
            "pair_counts": {
                "full_credit": 0,
                "half_credit": 0,
                "zero_credit": 0,
            },
            "unmatched_predictions": predicted_count,
            "unmatched_gold": target_count,
        }

    weights = np.zeros((predicted_count, target_count), dtype=np.float64)
    for pred_index, prediction in enumerate(predicted):
        for target_index, gold in enumerate(target):
            if not location_hit(prediction, gold):
                continue
            weights[pred_index, target_index] = (
                1.0
                if str(prediction.get("type")) == str(gold.get("type"))
                else mismatch_credit
            )
    rows, columns = linear_sum_assignment(-weights)
    assigned = [float(weights[row, column]) for row, column in zip(rows, columns)]
    full_credit = sum(abs(value - 1.0) <= 1e-12 for value in assigned)
    partial_credit = sum(
        value > 0.0 and abs(value - 1.0) > 1e-12 for value in assigned
    )
    zero_credit = len(assigned) - full_credit - partial_credit
    credit = float(sum(assigned))
    precision = credit / predicted_count
    recall = credit / target_count
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "schema_version": TYPED_LOCATION_METRIC_SCHEMA,
        "criterion": criterion,
        "type_mismatch_credit": mismatch_credit,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "credit": credit,
        "predicted": predicted_count,
        "gold": target_count,
        "pair_counts": {
            "full_credit": full_credit,
            "half_credit": partial_credit,
            "zero_credit": zero_credit,
        },
        "unmatched_predictions": predicted_count - len(rows),
        "unmatched_gold": target_count - len(columns),
    }


def _prf(correct: int, predicted: int, target: int) -> dict[str, float]:
    if predicted == 0 and target == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    precision = correct / predicted if predicted else 0.0
    recall = correct / target if target else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def _validate_events(
    events: Sequence[JointEvent],
    *,
    score_event_count: int | None,
    role: str,
) -> None:
    previous_start = float("-inf")
    for index, event in enumerate(events):
        if event.start < previous_start:
            raise ValueError(f"{role} events are not ordered at index {index}")
        previous_start = event.start
        if event.score_span is not None and score_event_count is not None:
            start, end = event.score_span
            if end > score_event_count:
                raise ValueError(
                    f"{role} event {index} span {event.score_span} exceeds "
                    f"score event count {score_event_count}"
                )


def _validate_deletions(
    values: Iterable[int],
    *,
    score_event_count: int | None,
    role: str,
) -> frozenset[int]:
    result = frozenset(int(value) for value in values)
    if any(value < 0 for value in result):
        raise ValueError(f"{role} deletion indices must be non-negative")
    if (
        score_event_count is not None
        and any(value >= score_event_count for value in result)
    ):
        raise ValueError(
            f"{role} deletion index exceeds score event count "
            f"{score_event_count}"
        )
    return result


def _note_wise_event_label(event: JointEvent) -> dict[str, Any]:
    label: dict[str, Any] = {"type": event.relationship}
    if event.score_span is not None:
        label["score_event_indices"] = list(
            range(event.score_span[0], event.score_span[1])
        )
        label["copy_pass"] = event.copy_pass
    elif event.rendered_index is not None:
        label["rendered_index"] = event.rendered_index
    return label


def _official_note_wise_report(
    predicted: Sequence[JointEvent],
    target: Sequence[JointEvent],
    predicted_deletions: frozenset[int],
    target_deletions: frozenset[int],
    *,
    score_event_count: int | None,
) -> dict[str, Any]:
    pred_labels = [_note_wise_event_label(value) for value in predicted]
    gold_labels = [_note_wise_event_label(value) for value in target]
    pred_labels.extend(
        {"type": "missed_note", "score_event_indices": [value]}
        for value in sorted(predicted_deletions)
    )
    gold_labels.extend(
        {"type": "missed_note", "score_event_indices": [value]}
        for value in sorted(target_deletions)
    )
    detail = match_note_wise_labels_detail(
        gold_labels,
        pred_labels,
        score_event_count=score_event_count,
    )
    if detail["status"] != "available":
        raise ValueError(f"Canonical target identity unavailable: {detail['reason']}")
    return detail


def _match_exact_pitch_onset(
    predicted: Sequence[JointEvent],
    target: Sequence[JointEvent],
    tolerance_sec: float,
) -> list[tuple[int, int]]:
    """Maximum ordered pairing; tie-break only by onset error, never mapping."""

    n_pred, n_target = len(predicted), len(target)
    counts = [[0] * (n_target + 1) for _ in range(n_pred + 1)]
    neg_errors = [[0.0] * (n_target + 1) for _ in range(n_pred + 1)]
    # 0 = skip prediction, 1 = skip target, 2 = pair.
    parents = [[0] * (n_target + 1) for _ in range(n_pred + 1)]
    for j in range(1, n_target + 1):
        parents[0][j] = 1
    tolerance = float(tolerance_sec) + 1e-12
    for i in range(1, n_pred + 1):
        for j in range(1, n_target + 1):
            candidates = [
                (counts[i - 1][j], neg_errors[i - 1][j], 0),
                (counts[i][j - 1], neg_errors[i][j - 1], 1),
            ]
            pred = predicted[i - 1]
            gold = target[j - 1]
            delta = abs(pred.start - gold.start)
            if pred.pitch == gold.pitch and delta <= tolerance:
                candidates.append(
                    (
                        counts[i - 1][j - 1] + 1,
                        neg_errors[i - 1][j - 1] - delta,
                        2,
                    )
                )
            count, neg_error, action = max(
                candidates, key=lambda value: (value[0], value[1], value[2])
            )
            counts[i][j] = count
            neg_errors[i][j] = neg_error
            parents[i][j] = action

    pairs: list[tuple[int, int]] = []
    i, j = n_pred, n_target
    while i or j:
        action = parents[i][j]
        if action == 2:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif action == 1:
            j -= 1
        else:
            i -= 1
    pairs.reverse()
    return pairs


def pair_exact_pitch_onset(
    predicted: Sequence[JointEvent],
    target: Sequence[JointEvent],
    *,
    tolerance_sec: float = 0.050,
) -> list[tuple[int, int]]:
    """Public deterministic acoustic pairing with no mapping-aware tie-break."""

    _validate_events(predicted, score_event_count=None, role="Predicted")
    _validate_events(target, score_event_count=None, role="Target")
    if tolerance_sec <= 0:
        raise ValueError("Onset tolerance must be positive")
    return _match_exact_pitch_onset(predicted, target, tolerance_sec)


def _tolerance_counts(
    predicted: Sequence[JointEvent],
    target: Sequence[JointEvent],
    tolerance_sec: float,
) -> dict[str, int]:
    pairs = _match_exact_pitch_onset(predicted, target, tolerance_sec)
    joint_correct = sum(
        predicted[i].score_span == target[j].score_span for i, j in pairs
    )
    paired_linked = sum(not target[j].is_extra for _, j in pairs)
    mapping_correct = sum(
        not target[j].is_extra
        and predicted[i].score_span == target[j].score_span
        for i, j in pairs
    )
    copy_correct = sum(
        target[j].is_copy
        and predicted[i].is_copy
        and predicted[i].score_span == target[j].score_span
        for i, j in pairs
    )
    copy_mapping_correct = sum(
        target[j].is_copy
        and predicted[i].score_span == target[j].score_span
        for i, j in pairs
    )
    extra_correct = sum(
        predicted[i].is_extra and target[j].is_extra for i, j in pairs
    )
    substitution_correct = sum(
        target[j].relationship == "substitute"
        and predicted[i].score_span == target[j].score_span
        for i, j in pairs
    )
    return {
        "predicted": len(predicted),
        "target": len(target),
        "paired": len(pairs),
        "joint_correct": joint_correct,
        "paired_linked": paired_linked,
        "mapping_correct": mapping_correct,
        "predicted_copy": sum(event.is_copy for event in predicted),
        "target_copy": sum(event.is_copy for event in target),
        "copy_correct": copy_correct,
        "copy_mapping_correct": copy_mapping_correct,
        "predicted_extra": sum(event.is_extra for event in predicted),
        "target_extra": sum(event.is_extra for event in target),
        "extra_correct": extra_correct,
        "target_substitution": sum(
            event.relationship == "substitute" for event in target
        ),
        "substitution_correct": substitution_correct,
    }


def _report_from_counts(counts: Mapping[str, int]) -> dict[str, object]:
    note = _prf(counts["paired"], counts["predicted"], counts["target"])
    joint = _prf(
        counts["joint_correct"], counts["predicted"], counts["target"]
    )
    copy = _prf(
        counts["copy_correct"],
        counts["predicted_copy"],
        counts["target_copy"],
    )
    extra = _prf(
        counts["extra_correct"],
        counts["predicted_extra"],
        counts["target_extra"],
    )
    # This preserves the old mapped-event precision/recall interpretation:
    # extras are negatives and score-linked rendered events are positives.
    current_prf = _prf(
        counts["mapping_correct"],
        counts["predicted"] - counts["predicted_extra"],
        counts["target"] - counts["target_extra"],
    )
    current = {
        **current_prf,
        "exact_accuracy": (
            counts["joint_correct"] / counts["target"]
            if counts["target"]
            else 1.0
        ),
        "copy_recall": (
            counts["copy_mapping_correct"] / counts["target_copy"]
            if counts["target_copy"]
            else 0.0
        ),
        "n_correct": counts["mapping_correct"],
        "n_assigned": counts["predicted"] - counts["predicted_extra"],
        "n_positive": counts["target"] - counts["target_extra"],
        "n_copy_correct": counts["copy_mapping_correct"],
        "n_copy": counts["target_copy"],
        "n_transcription_matches": counts["paired"],
    }
    return {
        "note": note,
        "joint": joint,
        "conditional_mapping_accuracy": (
            counts["mapping_correct"] / counts["paired_linked"]
            if counts["paired_linked"]
            else None
        ),
        "copy": {**copy, "recall": copy["recall"]},
        "extras": extra,
        "substitution_mapping_accuracy": (
            counts["substitution_correct"] / counts["target_substitution"]
            if counts["target_substitution"]
            else None
        ),
        "current_mapping": current,
        "counts": dict(counts),
    }


def evaluate_joint_events(
    predicted: Sequence[JointEvent],
    target: Sequence[JointEvent],
    *,
    predicted_deletions: Iterable[int] = (),
    target_deletions: Iterable[int] = (),
    score_event_count: int | None = None,
    tolerances_sec: Sequence[float] = DEFAULT_TOLERANCES_SEC,
) -> dict[str, object]:
    """Evaluate official note identity plus explicit timestamp diagnostics."""

    predicted = tuple(predicted)
    target = tuple(target)
    _validate_events(
        predicted, score_event_count=score_event_count, role="Predicted"
    )
    _validate_events(target, score_event_count=score_event_count, role="Target")
    pred_deletions = _validate_deletions(
        predicted_deletions,
        score_event_count=score_event_count,
        role="Predicted",
    )
    gold_deletions = _validate_deletions(
        target_deletions,
        score_event_count=score_event_count,
        role="Target",
    )

    by_tolerance: dict[str, object] = {}
    for tolerance in tolerances_sec:
        if tolerance <= 0:
            raise ValueError("Onset tolerances must be positive")
        key = f"{round(float(tolerance) * 1000):d}ms"
        by_tolerance[key] = _report_from_counts(
            _tolerance_counts(predicted, target, float(tolerance))
        )

    deletion_correct = len(pred_deletions & gold_deletions)
    deletion_counts = {
        "predicted": len(pred_deletions),
        "target": len(gold_deletions),
        "correct": deletion_correct,
    }
    deletion_report = {
        **_prf(
            deletion_correct,
            len(pred_deletions),
            len(gold_deletions),
        ),
        "counts": deletion_counts,
    }
    current_50ms = by_tolerance.get("50ms")
    official = _official_note_wise_report(
        predicted,
        target,
        pred_deletions,
        gold_deletions,
        score_event_count=score_event_count,
    )
    return {
        "official_note_wise": official,
        "diagnostic_timestamp_tolerances": by_tolerance,
        "tolerances": by_tolerance,
        "deletions": deletion_report,
        "legacy_current_metric_50ms": (
            current_50ms["current_mapping"]
            if isinstance(current_50ms, dict)
            else None
        ),
        "current_metric_50ms": (
            current_50ms["current_mapping"]
            if isinstance(current_50ms, dict)
            else None
        ),
    }


def _sum_counts(reports: Sequence[Mapping[str, object]]) -> dict[str, int]:
    keys = {
        key
        for report in reports
        for key in (report.get("counts") or {}).keys()  # type: ignore[union-attr]
    }
    return {
        key: sum(
            int((report.get("counts") or {}).get(key, 0))  # type: ignore[union-attr]
            for report in reports
        )
        for key in keys
    }


def evaluate_joint_dataset(
    samples: Sequence[JointMetricSample],
    *,
    tolerances_sec: Sequence[float] = DEFAULT_TOLERANCES_SEC,
) -> dict[str, object]:
    """Micro-aggregate clips overall and by declared data source."""

    reports: list[tuple[JointMetricSample, dict[str, object]]] = []
    for sample in samples:
        reports.append(
            (
                sample,
                evaluate_joint_events(
                    sample.predicted,
                    sample.target,
                    predicted_deletions=sample.predicted_deletions,
                    target_deletions=sample.target_deletions,
                    score_event_count=sample.score_event_count,
                    tolerances_sec=tolerances_sec,
                ),
            )
        )

    def aggregate(
        selected: Sequence[tuple[JointMetricSample, dict[str, object]]],
    ) -> dict[str, object]:
        tolerance_reports: dict[str, object] = {}
        for tolerance in tolerances_sec:
            key = f"{round(float(tolerance) * 1000):d}ms"
            children = [
                report["tolerances"][key]  # type: ignore[index]
                for _, report in selected
            ]
            tolerance_reports[key] = _report_from_counts(_sum_counts(children))
        deletion_children = [report["deletions"] for _, report in selected]
        deletion_counts = _sum_counts(deletion_children)
        deletion = {
            **_prf(
                deletion_counts.get("correct", 0),
                deletion_counts.get("predicted", 0),
                deletion_counts.get("target", 0),
            ),
            "counts": deletion_counts,
        }
        current = tolerance_reports.get("50ms")
        official_children = [
            report["official_note_wise"] for _, report in selected
        ]
        official_credit = sum(float(value["credit"]) for value in official_children)
        official_predicted = sum(
            int(value["predicted"]) for value in official_children
        )
        official_gold = sum(int(value["gold"]) for value in official_children)
        official_precision = (
            official_credit / official_predicted if official_predicted else 0.0
        )
        official_recall = official_credit / official_gold if official_gold else 0.0
        if not official_predicted and not official_gold:
            official_precision = official_recall = 1.0
        official_f1 = (
            2.0
            * official_precision
            * official_recall
            / (official_precision + official_recall)
            if official_precision + official_recall
            else 0.0
        )
        return {
            "n_samples": len(selected),
            "official_note_wise": {
                "schema_version": "align-note-wise-score-event-metric-v1",
                "status": "available",
                "type_mismatch_credit": 0.5,
                "credit": official_credit,
                "predicted": official_predicted,
                "gold": official_gold,
                "precision": official_precision,
                "recall": official_recall,
                "f1": official_f1,
            },
            "diagnostic_timestamp_tolerances": tolerance_reports,
            "tolerances": tolerance_reports,
            "deletions": deletion,
            "legacy_current_metric_50ms": (
                current["current_mapping"] if isinstance(current, dict) else None
            ),
            "current_metric_50ms": (
                current["current_mapping"] if isinstance(current, dict) else None
            ),
        }

    sources = sorted({sample.source for sample, _ in reports})
    return {
        "aggregate": aggregate(reports),
        "per_source": {
            source: aggregate(
                [entry for entry in reports if entry[0].source == source]
            )
            for source in sources
        },
    }
