"""Score-informed mapper with explicit rendered-ornament EXTRA states.

The acoustic candidates remain score-free.  This mapper reads only the
verified score and candidates.  It expands notated ornaments into a structural
template matching the repository's deterministic renderer, while retaining
the official canonical convention: ornament realizations without score-event
lineage are predicted as exclusive rendered EXTRA identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import inf, log
from pathlib import Path
from typing import Sequence

from music21 import converter, note

from .grammar_mapper_v2 import (
    GrammarCosts,
    GrammarHypothesis,
    decode_grammar_mapper,
    grammar_hypotheses,
)
from .index import JointEvent, ScoreEvent, _ordered_source_notes
from .lattice import JointCandidate


@dataclass(frozen=True)
class OrnamentMapperCosts:
    exact_linked: float = 0.0
    substitution: float = 1.35
    candidate_extra: float = 0.80
    score_deletion: float = 0.90
    exact_ornament: float = 0.0
    ornament_miss: float = 0.20
    ornament_wrong_pitch: float = 2.50
    repeat: float = 0.75
    timing: float = 0.12


@dataclass(frozen=True)
class OrnamentTemplateUnit:
    pitch: int
    time: float
    kind: str
    score_index: int | None
    copy_pass: int
    ornament_kind: str | None = None

    @property
    def is_ornament(self) -> bool:
        return self.kind == "ornament_extra"


@dataclass(frozen=True)
class _ScorePattern:
    prefix: tuple[tuple[int, str], ...]
    body: tuple[tuple[int, str], ...]


def _expression_pattern(element: note.Note) -> tuple[tuple[int, str], ...]:
    names = {type(value).__name__ for value in (element.expressions or [])}
    pitch = int(element.pitch.midi)
    duration = float(element.duration.quarterLength)
    if names & {"Trill", "Shake"}:
        steps = max(4, min(12, int(round(duration / 0.125))))
        if steps % 2:
            steps += 1
        return tuple(
            (
                pitch if index == 0 or index % 2 == 0 else min(127, pitch + 1),
                "principal" if index == 0 else "trill",
            )
            for index in range(steps)
        )
    if "Mordent" in names:
        return (
            (pitch, "principal"),
            (max(0, pitch - 1), "mordent"),
            (pitch, "mordent"),
        )
    if "InvertedMordent" in names:
        return (
            (pitch, "principal"),
            (min(127, pitch + 1), "inverted_mordent"),
            (pitch, "inverted_mordent"),
        )
    if names & {"Turn", "InvertedTurn"}:
        values = (
            (min(127, pitch + 1), pitch, max(0, pitch - 1), pitch)
            if "Turn" in names
            else (max(0, pitch - 1), pitch, min(127, pitch + 1), pitch)
        )
        return tuple(
            (value, "principal" if index == 0 else "turn")
            for index, value in enumerate(values)
        )
    return ((pitch, "principal"),)


@lru_cache(maxsize=512)
def score_ornament_patterns(
    score_path: Path | str,
    score_events: tuple[ScoreEvent, ...],
) -> tuple[_ScorePattern, ...]:
    """Extract one renderer-compatible ornament pattern per canonical event."""

    parsed = converter.parse(str(Path(score_path)))
    ordered = _ordered_source_notes(parsed)
    element_to_source = {
        id(element): source_index
        for source_index, element, *_rest in ordered
    }
    prefix_by_source: dict[int, list[tuple[int, str]]] = {}
    pending_graces: list[tuple[int, str]] = []
    for element in parsed.recurse().getElementsByClass(note.Note):
        if bool(element.duration.isGrace):
            pending_graces.append((int(element.pitch.midi), "grace"))
            continue
        source_index = element_to_source.get(id(element))
        if source_index is not None:
            prefix_by_source[source_index] = list(pending_graces)
        pending_graces.clear()
    source_elements = {
        source_index: element for source_index, element, *_rest in ordered
    }
    patterns = []
    for event in score_events:
        source_index = event.source_indices[0]
        element = source_elements[source_index]
        prefix = tuple(prefix_by_source.get(source_index, ()))
        body = _expression_pattern(element)
        patterns.append(_ScorePattern(prefix=prefix, body=body))
    if len(patterns) != len(score_events):
        raise ValueError("Ornament patterns do not match canonical score events")
    return tuple(patterns)


def expand_ornament_hypothesis(
    score: Sequence[ScoreEvent],
    patterns: Sequence[_ScorePattern],
    hypothesis: GrammarHypothesis,
) -> tuple[OrnamentTemplateUnit, ...]:
    if len(patterns) != len(score):
        raise ValueError("One ornament pattern is required per score event")
    output = []
    for unit_position, (score_index, copy_pass) in enumerate(hypothesis.units):
        event = score[score_index]
        pattern = patterns[score_index]
        base_time = float(hypothesis.times[unit_position])
        duration = max(float(event.ql_end - event.ql_start), 0.001)
        for prefix_position, (pitch, kind) in enumerate(pattern.prefix):
            output.append(
                OrnamentTemplateUnit(
                    pitch=pitch,
                    time=base_time - 1e-4 * (len(pattern.prefix) - prefix_position),
                    kind="ornament_extra",
                    score_index=None,
                    copy_pass=copy_pass,
                    ornament_kind=kind,
                )
            )
        body_size = max(len(pattern.body), 1)
        for body_position, (pitch, kind) in enumerate(pattern.body):
            is_principal = body_position == 0
            output.append(
                OrnamentTemplateUnit(
                    pitch=pitch,
                    time=base_time + duration * body_position / body_size,
                    kind="linked" if is_principal else "ornament_extra",
                    score_index=score_index if is_principal else None,
                    copy_pass=copy_pass,
                    ornament_kind=None if is_principal else kind,
                )
            )
    return tuple(output)


def _normalized_positions(
    events: Sequence[JointCandidate],
    template: Sequence[OrnamentTemplateUnit],
) -> tuple[list[float], list[float]]:
    event_start = min((value.start for value in events), default=0.0)
    event_end = max((value.start for value in events), default=event_start + 1.0)
    event_extent = max(event_end - event_start, 1e-6)
    template_start = min((value.time for value in template), default=0.0)
    template_end = max(
        (value.time for value in template), default=template_start + 1.0
    )
    template_extent = max(template_end - template_start, 1e-6)
    return (
        [(value.start - event_start) / event_extent for value in events],
        [(value.time - template_start) / template_extent for value in template],
    )


def _align_template(
    events: Sequence[JointCandidate],
    template: Sequence[OrnamentTemplateUnit],
    costs: OrnamentMapperCosts,
    extra_probabilities: Sequence[float] | None = None,
    extra_weight: float = 0.0,
) -> tuple[float, list[tuple[str, int | None, int | None]]]:
    rows = len(events)
    columns = len(template)
    values = [[inf] * (columns + 1) for _ in range(rows + 1)]
    back = [[0] * (columns + 1) for _ in range(rows + 1)]
    values[0][0] = 0.0
    for row in range(1, rows + 1):
        probability = (
            float(extra_probabilities[row - 1])
            if extra_probabilities is not None
            else 0.5
        )
        values[row][0] = (
            values[row - 1][0]
            + costs.candidate_extra
            + extra_weight * -log(max(probability, 1e-6))
        )
        back[row][0] = 2
    for column in range(1, columns + 1):
        unit = template[column - 1]
        values[0][column] = values[0][column - 1] + (
            costs.ornament_miss if unit.is_ornament else costs.score_deletion
        )
        back[0][column] = 3
    event_position, template_position = _normalized_positions(events, template)
    for row in range(1, rows + 1):
        event = events[row - 1]
        probability = (
            float(extra_probabilities[row - 1])
            if extra_probabilities is not None
            else 0.5
        )
        for column in range(1, columns + 1):
            unit = template[column - 1]
            exact = event.pitch == unit.pitch
            if unit.is_ornament:
                diagonal_cost = (
                    costs.exact_ornament
                    if exact
                    else costs.ornament_wrong_pitch
                )
            else:
                diagonal_cost = (
                    costs.exact_linked if exact else costs.substitution
                )
            diagonal = (
                values[row - 1][column - 1]
                + diagonal_cost
                + costs.timing
                * abs(event_position[row - 1] - template_position[column - 1])
                + extra_weight
                * -log(
                    max(
                        probability if unit.is_ornament else 1.0 - probability,
                        1e-6,
                    )
                )
            )
            extra = (
                values[row - 1][column]
                + costs.candidate_extra
                + extra_weight * -log(max(probability, 1e-6))
            )
            deletion = values[row][column - 1] + (
                costs.ornament_miss
                if unit.is_ornament
                else costs.score_deletion
            )
            best = min(diagonal, extra, deletion)
            values[row][column] = best
            back[row][column] = (
                1
                if diagonal <= extra and diagonal <= deletion
                else 2
                if extra <= deletion
                else 3
            )
    actions = []
    row, column = rows, columns
    while row or column:
        operation = back[row][column]
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
            raise RuntimeError("Ornament mapper backtrace is incomplete")
    actions.reverse()
    return values[rows][columns], actions


def decode_ornament_mapper(
    events: Sequence[JointCandidate],
    score: Sequence[ScoreEvent],
    score_path: Path | str,
    *,
    costs: OrnamentMapperCosts = OrnamentMapperCosts(),
    extra_probabilities: Sequence[float] | None = None,
    extra_weight: float = 0.0,
) -> tuple[tuple[JointEvent, ...], frozenset[int], dict[str, object]]:
    """Map events with explicit ornament extras and deletion identities."""

    if extra_probabilities is not None and len(extra_probabilities) != len(events):
        raise ValueError("One extra probability is required per candidate event")
    patterns = score_ornament_patterns(score_path, tuple(score))
    ranked = []
    for hypothesis in grammar_hypotheses(score, len(events)):
        template = expand_ornament_hypothesis(score, patterns, hypothesis)
        value, actions = _align_template(
            events,
            template,
            costs,
            extra_probabilities,
            extra_weight,
        )
        value += costs.repeat * hypothesis.copies
        ranked.append((value, hypothesis, template, actions))
    value, hypothesis, template, actions = min(
        ranked,
        key=lambda row: (
            row[0],
            row[1].copies,
            row[1].source_span or (-1, -1),
        ),
    )
    output = []
    deletions = set()
    matched_ornaments = missed_ornaments = 0
    for operation, event_index, template_index in actions:
        if operation == "delete":
            assert template_index is not None
            unit = template[template_index]
            if unit.is_ornament:
                missed_ornaments += 1
            elif unit.copy_pass == 0 and unit.score_index is not None:
                deletions.add(unit.score_index)
            continue
        assert event_index is not None
        event = events[event_index]
        if operation == "extra":
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
        assert template_index is not None
        unit = template[template_index]
        if unit.is_ornament:
            matched_ornaments += 1
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
        assert unit.score_index is not None
        origin = "match" if event.pitch == unit.pitch else "substitute"
        output.append(
            JointEvent(
                pitch=event.pitch,
                start=event.start,
                end=event.end,
                score_span=(unit.score_index, unit.score_index + 1),
                relationship="copy" if unit.copy_pass else origin,
                copy_pass=unit.copy_pass,
                origin_relationship=origin,
                rendered_index=event_index,
                confidence=event.confidence,
            )
        )
    return tuple(output), frozenset(deletions), {
        "cost": float(value),
        "source_span": hypothesis.source_span,
        "copies": hypothesis.copies,
        "hypotheses": len(ranked),
        "template_units": len(template),
        "template_ornaments": sum(value.is_ornament for value in template),
        "matched_template_ornaments": matched_ornaments,
        "missed_template_ornaments": missed_ornaments,
        "predicted_deletions": len(deletions),
        "extra_probability_weight": float(extra_weight),
    }


def ornament_mapper_features(
    events: Sequence[JointCandidate],
    mapped: Sequence[JointEvent],
    score: Sequence[ScoreEvent],
    score_path: Path | str,
) -> list[list[float]]:
    """Target-free event features for a supervised extra-state prior."""

    if len(events) != len(mapped):
        raise ValueError("Mapped events must preserve candidate cardinality")
    patterns = score_ornament_patterns(score_path, tuple(score))
    score_start = min((event.ql_start for event in score), default=0.0)
    score_end = max((event.ql_end for event in score), default=score_start + 1.0)
    score_extent = max(score_end - score_start, 1e-6)
    event_start = min((event.start for event in events), default=0.0)
    event_end = max((event.end for event in events), default=event_start + 1.0)
    event_extent = max(event_end - event_start, 1e-6)
    score_pitch_counts: dict[int, int] = {}
    for score_event in score:
        score_pitch_counts[score_event.pitch] = (
            score_pitch_counts.get(score_event.pitch, 0) + 1
        )
    output = []
    for index, (event, mapping) in enumerate(zip(events, mapped)):
        previous = events[index - 1] if index else None
        following = events[index + 1] if index + 1 < len(events) else None
        score_index = (
            mapping.score_span[-1] - 1
            if mapping.score_span is not None
            else -1
        )
        score_event = score[score_index] if score_index >= 0 else None
        overlap = [
            other
            for other_index, other in enumerate(events)
            if other_index != index
            and other.start < event.end
            and event.start < other.end
        ]
        local = [
            other
            for other_index, other in enumerate(events)
            if other_index != index
            and abs(other.start - event.start) <= 0.25
        ]
        relationship = "copy" if mapping.is_copy else mapping.relationship
        ornament_count = (
            len(patterns[score_index].prefix)
            + max(0, len(patterns[score_index].body) - 1)
            if score_index >= 0
            else 0
        )
        event_position = (event.start - event_start) / event_extent
        score_position = (
            (score_event.ql_start - score_start) / score_extent
            if score_event is not None
            else event_position
        )
        output.append(
            [
                *(float(relationship == name) for name in ("match", "copy", "substitute", "extra")),
                float(event.confidence),
                min(event.end - event.start, 2.0),
                event_position,
                (event.start - previous.start) if previous is not None else 0.0,
                (following.start - event.start) if following is not None else 0.0,
                (event.start - previous.end) if previous is not None else 0.0,
                float(event.pitch - previous.pitch) / 12.0
                if previous is not None
                else 0.0,
                float(following.pitch - event.pitch) / 12.0
                if following is not None
                else 0.0,
                float(previous is not None and previous.pitch == event.pitch),
                float(following is not None and following.pitch == event.pitch),
                float(len(overlap)),
                float(sum(other.pitch < event.pitch for other in overlap)),
                float(sum(other.pitch > event.pitch for other in overlap)),
                float(len(local)),
                float(score_index) / max(len(score) - 1, 1),
                float(
                    event.pitch - score_event.pitch
                    if score_event is not None
                    else 0
                )
                / 12.0,
                float(
                    score_event is not None and event.pitch == score_event.pitch
                ),
                event_position - score_position,
                float(ornament_count),
                float(score_pitch_counts.get(event.pitch, 0))
                / max(len(score), 1),
                float(mapping.copy_pass),
            ]
        )
    return output


def decode_hard_extra_mapper(
    events: Sequence[JointCandidate],
    score: Sequence[ScoreEvent],
    extra_mask: Sequence[bool],
    *,
    grammar_costs: GrammarCosts = GrammarCosts(),
) -> tuple[tuple[JointEvent, ...], frozenset[int], dict[str, object]]:
    """Force classified extras aside, then map the remaining sequence globally."""

    if len(extra_mask) != len(events):
        raise ValueError("One hard extra decision is required per candidate")
    linked_indices = [
        index for index, value in enumerate(extra_mask) if not bool(value)
    ]
    linked = tuple(events[index] for index in linked_indices)
    mapped, grammar = decode_grammar_mapper(
        linked, score, costs=grammar_costs
    )
    mapped_by_original = {}
    for subset_index, event in enumerate(mapped):
        original_index = linked_indices[subset_index]
        mapped_by_original[original_index] = JointEvent(
            pitch=event.pitch,
            start=event.start,
            end=event.end,
            score_span=event.score_span,
            relationship=event.relationship,
            copy_pass=event.copy_pass,
            origin_relationship=event.origin_relationship,
            rendered_index=original_index,
            source_indices=event.source_indices,
            confidence=event.confidence,
        )
    output = []
    for index, candidate in enumerate(events):
        if bool(extra_mask[index]):
            output.append(
                JointEvent(
                    pitch=candidate.pitch,
                    start=candidate.start,
                    end=candidate.end,
                    score_span=None,
                    relationship="extra",
                    rendered_index=index,
                    confidence=candidate.confidence,
                )
            )
        else:
            output.append(mapped_by_original[index])
    covered = {
        score_index
        for event in output
        if event.score_span is not None and event.copy_pass == 0
        for score_index in range(*event.score_span)
    }
    deletions = frozenset(set(range(len(score))) - covered)
    return tuple(output), deletions, {
        **grammar,
        "hard_extra_events": sum(bool(value) for value in extra_mask),
        "linked_events": len(linked),
        "predicted_deletions": len(deletions),
    }
