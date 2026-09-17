from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence

import numpy as np

from .index import JointEvent, ScoreEvent
from .lattice import JointCandidate


@dataclass(frozen=True)
class GrammarCosts:
    substitution: float = 1.5
    extra: float = 1.0
    deletion: float = 1.0
    repeat: float = 0.75
    no_restart_gap: float = 0.25
    timing: float = 0.0
    duration: float = 0.0


@dataclass(frozen=True)
class GrammarHypothesis:
    source_span: tuple[int, int] | None
    copies: int
    units: tuple[tuple[int, int], ...]
    times: tuple[float, ...]


def grammar_hypotheses(
    score: Sequence[ScoreEvent],
    performed_count: int,
    *,
    edit_slack: int = 32,
) -> tuple[GrammarHypothesis, ...]:
    base = tuple((index, 0) for index in range(len(score)))
    output = [
        GrammarHypothesis(
            None,
            0,
            base,
            tuple(float(event.ql_start) for event in score),
        )
    ]
    identified_parts = [
        event
        for event in score
        if event.part is not None and event.source_indices
    ]
    repeated_part = (
        min(identified_parts, key=lambda event: min(event.source_indices)).part
        if identified_parts
        else None
    )
    measures = []
    for event in score:
        if repeated_part is not None and event.part != repeated_part:
            continue
        if not measures or measures[-1] != event.measure:
            measures.append(event.measure)
    spans = []
    for start_position in range(len(measures)):
        for measure_count in range(1, len(measures) - start_position + 1):
            selected = set(measures[start_position : start_position + measure_count])
            if len(selected) != measure_count:
                continue
            indices = [
                index
                for index, event in enumerate(score)
                if event.measure in selected
                and (repeated_part is None or event.part == repeated_part)
            ]
            if indices:
                spans.append(tuple(indices))
    observed_extra = performed_count - len(score)
    for block in spans:
        source_start, source_end = min(block), max(block) + 1
        for copies in (1, 2):
            expected_extra = len(block) * copies
            if abs(expected_extra - observed_extra) > edit_slack:
                continue
            block_start_ql = min(score[index].ql_start for index in block)
            block_end_ql = max(score[index].ql_end for index in block)
            block_duration = block_end_ql - block_start_ql
            scheduled = []
            for index, event in enumerate(score):
                shifted = (
                    (repeated_part is None or event.part == repeated_part)
                    and event.ql_start >= block_end_ql - 1e-9
                )
                scheduled.append(
                    (
                        event.ql_start + (block_duration * copies if shifted else 0.0),
                        event.pitch,
                        min(event.source_indices or (index,)),
                        index,
                        0,
                    )
                )
            for copy_pass in range(1, copies + 1):
                for index in block:
                    event = score[index]
                    scheduled.append(
                        (
                            event.ql_start + block_duration * copy_pass,
                            event.pitch,
                            min(event.source_indices or (index,)),
                            index,
                            copy_pass,
                        )
                    )
            scheduled.sort(key=lambda row: (row[0], row[1], row[2], row[4]))
            units = tuple((row[3], row[4]) for row in scheduled)
            output.append(
                GrammarHypothesis(
                    (source_start, source_end),
                    copies,
                    units,
                    tuple(float(row[0]) for row in scheduled),
                )
            )
    return tuple(output)


def _align(
    events: Sequence[JointCandidate],
    score: Sequence[ScoreEvent],
    hypothesis: GrammarHypothesis,
    costs: GrammarCosts,
    operation_probabilities: np.ndarray | None = None,
    operation_weight: float = 0.4,
    location_log_probabilities: np.ndarray | None = None,
    location_weight: float = 0.2,
    emission_log_probabilities: np.ndarray | None = None,
    extra_log_probabilities: np.ndarray | None = None,
    emission_weight: float = 0.2,
) -> tuple[float, list[tuple[str, int | None, int | None]]]:
    rows = len(events)
    columns = len(hypothesis.units)
    values = np.full((rows + 1, columns + 1), np.inf, dtype=np.float32)
    back = np.zeros((rows + 1, columns + 1), dtype=np.uint8)
    values[0, 0] = costs.repeat * hypothesis.copies
    event_start = min((event.start for event in events), default=0.0)
    event_extent = max(
        max((event.end for event in events), default=event_start + 1.0)
        - event_start,
        1e-6,
    )
    score_start = min(hypothesis.times, default=0.0)
    score_extent = max(
        max(
            (
                unit_time
                + score[hypothesis.units[index][0]].ql_end
                - score[hypothesis.units[index][0]].ql_start
                for index, unit_time in enumerate(hypothesis.times)
            ),
            default=score_start + 1.0,
        )
        - score_start,
        1e-6,
    )
    for row in range(1, rows + 1):
        values[row, 0] = values[row - 1, 0] + costs.extra
        back[row, 0] = 2
    for column in range(1, columns + 1):
        values[0, column] = values[0, column - 1] + costs.deletion
        back[0, column] = 3
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            score_index, _copy_pass = hypothesis.units[column - 1]
            copy_pass = hypothesis.units[column - 1][1]
            diagonal_type = (
                1
                if copy_pass
                else 0
                if events[row - 1].pitch == score[score_index].pitch
                else 2
            )
            diagonal = values[row - 1, column - 1] + (
                0.0
                if events[row - 1].pitch == score[score_index].pitch
                else costs.substitution
            )
            event_position = (events[row - 1].start - event_start) / event_extent
            score_position = (
                hypothesis.times[column - 1] - score_start
            ) / score_extent
            diagonal += (
                costs.timing
                * abs(event_position - score_position)
                * min(rows, columns)
            )
            event_duration = (
                events[row - 1].end - events[row - 1].start
            ) / event_extent
            score_duration = (
                score[score_index].ql_end - score[score_index].ql_start
            ) / score_extent
            diagonal += (
                costs.duration
                * abs(event_duration - score_duration)
                * min(rows, columns)
            )
            extra = values[row - 1, column] + costs.extra
            if operation_probabilities is not None:
                diagonal += operation_weight * -np.log(
                    max(float(operation_probabilities[row - 1, diagonal_type]), 1e-6)
                )
                extra += operation_weight * -np.log(
                    max(float(operation_probabilities[row - 1, 3]), 1e-6)
                )
            if location_log_probabilities is not None:
                diagonal -= location_weight * float(
                    location_log_probabilities[row - 1, score_index]
                )
                extra -= location_weight * float(
                    location_log_probabilities[row - 1, -1]
                )
            if emission_log_probabilities is not None:
                diagonal -= emission_weight * float(
                    emission_log_probabilities[
                        row - 1, score_index, min(copy_pass, 2)
                    ]
                )
            if extra_log_probabilities is not None:
                extra -= emission_weight * float(
                    extra_log_probabilities[row - 1]
                )
            deletion = values[row, column - 1] + costs.deletion
            best = min(diagonal, extra, deletion)
            values[row, column] = best
            back[row, column] = (
                1 if diagonal <= extra and diagonal <= deletion
                else 2 if extra <= deletion
                else 3
            )
    row, column = rows, columns
    actions = []
    while row or column:
        operation = int(back[row, column])
        if operation == 1:
            actions.append(("diagonal", row - 1, column - 1))
            row -= 1
            column -= 1
        elif operation == 2:
            actions.append(("extra", row - 1, None))
            row -= 1
        elif operation == 3:
            actions.append(("delete", None, column - 1))
            column -= 1
        else:
            raise RuntimeError("Grammar Viterbi backtrace is incomplete")
    actions.reverse()
    return float(values[rows, columns]), actions


def _restart_adjustment(
    events: Sequence[JointCandidate],
    hypothesis: GrammarHypothesis,
    actions: Sequence[tuple[str, int | None, int | None]],
    costs: GrammarCosts,
) -> float:
    if hypothesis.copies == 0 or hypothesis.source_span is None:
        return 0.0
    source_start, source_end = hypothesis.source_span
    copy_unit = source_end
    first_copy_event = next(
        (
            event_index
            for operation, event_index, unit_index in actions
            if operation == "diagonal"
            and unit_index == copy_unit
            and event_index is not None
        ),
        None,
    )
    if first_copy_event is None or first_copy_event <= 0:
        return costs.no_restart_gap
    iois = [
        max(events[index].start - events[index - 1].start, 1e-3)
        for index in range(1, len(events))
    ]
    median = float(np.median(iois)) if iois else 0.0
    gap = events[first_copy_event].start - events[first_copy_event - 1].end
    return 0.0 if gap >= max(0.05, 0.5 * median) else costs.no_restart_gap


def decode_grammar_mapper(
    events: Sequence[JointCandidate],
    score: Sequence[ScoreEvent],
    *,
    costs: GrammarCosts = GrammarCosts(),
    operation_probabilities: np.ndarray | None = None,
    operation_weight: float = 0.4,
    location_log_probabilities: np.ndarray | None = None,
    location_weight: float = 0.2,
    emission_log_probabilities: np.ndarray | None = None,
    extra_log_probabilities: np.ndarray | None = None,
    emission_weight: float = 0.2,
    forced_hypothesis: GrammarHypothesis | None = None,
    plan_log_probabilities: np.ndarray | None = None,
    plan_weight: float = 0.2,
) -> tuple[tuple[JointEvent, ...], dict[str, object]]:
    hypotheses = (
        (forced_hypothesis,)
        if forced_hypothesis is not None
        else grammar_hypotheses(score, len(events))
    )
    ranked = []
    for hypothesis_index, hypothesis in enumerate(hypotheses):
        value, actions = _align(
            events,
            score,
            hypothesis,
            costs,
            operation_probabilities,
            operation_weight,
            location_log_probabilities,
            location_weight,
            emission_log_probabilities,
            extra_log_probabilities,
            emission_weight,
        )
        value += _restart_adjustment(events, hypothesis, actions, costs)
        if plan_log_probabilities is not None:
            value -= plan_weight * float(
                plan_log_probabilities[hypothesis_index]
            )
        ranked.append((value, hypothesis, actions))
    value, selected, actions = min(
        ranked,
        key=lambda row: (
            row[0],
            row[1].copies,
            row[1].source_span or (-1, -1),
        ),
    )
    pending: dict[int, list[int]] = {0: [], 1: [], 2: []}
    output = []
    for operation, event_index, unit_index in actions:
        if operation == "delete":
            assert unit_index is not None
            score_index, copy_pass = selected.units[unit_index]
            pending[copy_pass].append(score_index)
            continue
        if operation == "extra":
            assert event_index is not None
            event = events[event_index]
            output.append(
                JointEvent(
                    pitch=event.pitch,
                    start=event.start,
                    end=event.end,
                    score_span=None,
                    relationship="extra",
                    rendered_index=event_index,
                    confidence=event.confidence,
                )
            )
            continue
        assert event_index is not None and unit_index is not None
        event = events[event_index]
        score_index, copy_pass = selected.units[unit_index]
        deleted = pending[copy_pass]
        span_start = min([score_index, *deleted]) if deleted else score_index
        pending[copy_pass] = []
        origin = (
            "match"
            if event.pitch == score[score_index].pitch
            else "substitute"
        )
        output.append(
            JointEvent(
                pitch=event.pitch,
                start=event.start,
                end=event.end,
                score_span=(span_start, score_index + 1),
                relationship="copy" if copy_pass else origin,
                copy_pass=copy_pass,
                origin_relationship=origin,
                rendered_index=event_index,
                confidence=event.confidence,
            )
        )
    return tuple(output), {
        "cost": value,
        "source_span": selected.source_span,
        "copies": selected.copies,
        "hypotheses": len(hypotheses),
    }


def operation_features(
    candidates: Sequence[JointCandidate],
    mapped: Sequence[JointEvent],
    score: Sequence[ScoreEvent],
    event_index: int,
    grammar: dict[str, object],
) -> np.ndarray:
    candidate = candidates[event_index]
    event = mapped[event_index]
    previous = candidates[event_index - 1] if event_index else None
    following = (
        candidates[event_index + 1]
        if event_index + 1 < len(candidates)
        else None
    )
    score_index = (
        event.score_span[-1] - 1 if event.score_span is not None else -1
    )
    score_pitch = score[score_index].pitch if score_index >= 0 else candidate.pitch
    relationship = "copy" if event.is_copy else event.relationship
    previous_score = (
        mapped[event_index - 1].score_span[-1] - 1
        if event_index
        and mapped[event_index - 1].score_span is not None
        else -1
    )
    next_score = (
        mapped[event_index + 1].score_span[-1] - 1
        if event_index + 1 < len(mapped)
        and mapped[event_index + 1].score_span is not None
        else -1
    )
    source_span = grammar.get("source_span")
    return np.asarray(
        [
            *(float(relationship == name) for name in ("match", "copy", "substitute", "extra")),
            float(event.copy_pass),
            float(candidate.confidence),
            min(candidate.end - candidate.start, 2.0),
            min(candidate.start - previous.start, 2.0) if previous else 0.0,
            min(following.start - candidate.start, 2.0) if following else 0.0,
            (candidate.pitch - score_pitch) / 12.0,
            float(candidate.pitch == score_pitch),
            float(
                event.score_span[1] - event.score_span[0]
                if event.score_span is not None
                else 0
            ),
            event_index / max(len(candidates) - 1, 1),
            float(grammar["copies"]),
            float(previous is not None and previous.pitch == candidate.pitch),
            float(following is not None and following.pitch == candidate.pitch),
            (
                (candidate.pitch - previous.pitch) / 12.0
                if previous is not None
                else 0.0
            ),
            (
                (following.pitch - candidate.pitch) / 12.0
                if following is not None
                else 0.0
            ),
            score_index / max(len(score) - 1, 1),
            (
                (score_index - previous_score) / 16.0
                if score_index >= 0 and previous_score >= 0
                else 0.0
            ),
            (
                (next_score - score_index) / 16.0
                if score_index >= 0 and next_score >= 0
                else 0.0
            ),
            (event_index - max(score_index, 0)) / max(len(candidates), 1),
            (
                (score_index - source_span[0]) / max(source_span[1] - source_span[0], 1)
                if source_span is not None and score_index >= 0
                else 0.0
            ),
            (
                copy_positions := sum(
                    value.is_copy
                    for value in mapped[:event_index]
                    if value.copy_pass == event.copy_pass and event.copy_pass > 0
                )
            )
            / max(sum(value.copy_pass == event.copy_pass for value in mapped), 1),
        ],
        dtype=np.float32,
    )


def emission_features(
    candidates: Sequence[JointCandidate],
    score: Sequence[ScoreEvent],
    event_index: int,
    score_index: int,
    copy_pass: int,
) -> np.ndarray:
    event = candidates[event_index]
    previous = candidates[event_index - 1] if event_index else None
    following = (
        candidates[event_index + 1]
        if event_index + 1 < len(candidates)
        else None
    )


def plan_features(
    events: Sequence[JointCandidate],
    score: Sequence[ScoreEvent],
    hypothesis: GrammarHypothesis,
    *,
    costs: GrammarCosts = GrammarCosts(),
) -> np.ndarray:
    performed_pitch = tuple(event.pitch for event in events)
    unfolded_pitch = tuple(score[index].pitch for index, _copy in hypothesis.units)
    similarity = SequenceMatcher(
        None, performed_pitch, unfolded_pitch, autojunk=False
    ).ratio()
    source_start, source_end = hypothesis.source_span or (0, 0)
    block_pitch = tuple(
        score[index].pitch for index in range(source_start, source_end)
    )
    occurrences = 0
    if block_pitch:
        for start in range(0, len(performed_pitch) - len(block_pitch) + 1):
            occurrences += int(
                performed_pitch[start : start + len(block_pitch)] == block_pitch
            )
    diagonal = min(len(events), len(hypothesis.units)) * similarity
    extras = max(0, len(events) - len(hypothesis.units))
    deletions = max(0, len(hypothesis.units) - len(events))
    value = (1.0 - similarity) * max(len(events), len(hypothesis.units))
    return np.asarray(
        [
            float(hypothesis.copies == 0),
            float(hypothesis.copies == 1),
            float(hypothesis.copies == 2),
            source_start / max(len(score), 1),
            source_end / max(len(score), 1),
            (source_end - source_start) / max(len(score), 1),
            len(hypothesis.units) / max(len(score), 1),
            (len(events) - len(score)) / max(len(score), 1),
            abs(len(events) - len(hypothesis.units)) / max(len(events), 1),
            value / max(len(events), 1),
            similarity,
            diagonal / max(len(events), 1),
            extras / max(len(events), 1),
            deletions / max(len(score), 1),
            min(occurrences, 4) / 4.0,
            0.0,
        ],
        dtype=np.float32,
    )
    if score_index >= 0:
        target = score[score_index]
        score_end = max((value.ql_end for value in score), default=1.0)
        pitch_delta = (event.pitch - target.pitch) / 12.0
        score_position = target.ql_start / max(score_end, 1e-6)
        score_duration = (target.ql_end - target.ql_start) / max(score_end, 1e-6)
    else:
        pitch_delta = score_position = score_duration = 0.0
    event_end = max((value.end for value in candidates), default=1.0)
    context = []
    for offset in (-3, -2, -1, 1, 2, 3):
        candidate_at = event_index + offset
        score_at = score_index + offset
        context.append(
            float(
                score_index >= 0
                and 0 <= candidate_at < len(candidates)
                and 0 <= score_at < len(score)
                and candidates[candidate_at].pitch == score[score_at].pitch
            )
        )
    return np.asarray(
        [
            float(score_index < 0),
            pitch_delta,
            float(score_index >= 0 and abs(pitch_delta) < 1e-9),
            event_index / max(len(candidates) - 1, 1),
            score_position,
            event.start / max(event_end, 1e-6),
            min(event.end - event.start, 2.0),
            score_duration,
            float(event.confidence),
            float(copy_pass == 0),
            float(copy_pass == 1),
            float(copy_pass == 2),
            *context,
            (
                (event.pitch - previous.pitch) / 12.0
                if previous is not None
                else 0.0
            ),
            (
                (following.pitch - event.pitch) / 12.0
                if following is not None
                else 0.0
            ),
        ],
        dtype=np.float32,
    )
