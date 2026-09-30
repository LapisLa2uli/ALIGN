"""Structured DP aligner for perfect transcriptions against a reference score.

The performance is modelled as one pass through the reference score, with an
optional repeated measure window (1-2 extra copies) and renderer ornament
expansions. For each repeat hypothesis the ornament-expanded template is
aligned to the transcribed pitch sequence with match, substitute, extra,
missed-note, ornament, and same-pitch merge operations; the cheapest
hypothesis wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numba
import numpy as np

from .grammar_mapper_v2 import grammar_hypotheses
from .index import JointEvent, ScoreEvent
from .ornament_mapper_v1 import (
    OrnamentTemplateUnit,
    expand_ornament_hypothesis,
    score_ornament_patterns,
)


INF = 1e18
OP_NONE, OP_MATCH, OP_ORNAMENT, OP_INSERT, OP_DELETE, OP_MERGE = range(6)


@dataclass(frozen=True)
class PerfectDPCosts:
    substitute: float = 1.4
    insert: float = 1.0
    delete: float = 1.0
    ornament_skip: float = 0.35
    ornament_near: float = 0.30
    merge: float = 0.15
    copy: float = 0.5
    length_slack: int = 24


@numba.njit(cache=True)
def _dp(
    note_pitch: np.ndarray,
    unit_pitch: np.ndarray,
    unit_linked: np.ndarray,
    unit_mergeable: np.ndarray,
    costs: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, int]:
    substitute, insert, delete, ornament_skip, ornament_near, merge = (
        costs[0], costs[1], costs[2], costs[3], costs[4], costs[5]
    )
    n = note_pitch.shape[0]
    m = unit_pitch.shape[0]
    # layer 0: any state; layer 1: note i-1 aligned to unit j-1 by match/merge.
    cost = np.full((2, n + 1, m + 1), INF)
    back = np.zeros((2, n + 1, m + 1), np.int8)
    from_layer = np.zeros((2, n + 1, m + 1), np.int8)
    cost[0, 0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best0 = INF
            op0 = OP_NONE
            layer0 = 0
            best1 = INF
            op1 = OP_NONE
            layer1 = 0
            if i > 0 and j > 0:
                previous = min(cost[0, i - 1, j - 1], cost[1, i - 1, j - 1])
                previous_layer = 0 if cost[0, i - 1, j - 1] <= cost[1, i - 1, j - 1] else 1
                if unit_linked[j - 1]:
                    value = previous + (0.0 if note_pitch[i - 1] == unit_pitch[j - 1] else substitute)
                    if value < best1:
                        best1, op1, layer1 = value, OP_MATCH, previous_layer
                else:
                    difference = abs(note_pitch[i - 1] - unit_pitch[j - 1])
                    if difference <= 2:
                        value = previous + (0.0 if difference == 0 else ornament_near)
                        if value < best0:
                            best0, op0, layer0 = value, OP_ORNAMENT, previous_layer
                # merge: note i-1 already matched to unit j-2 also covers unit j-1
                if (
                    unit_mergeable[j - 1]
                    and j >= 2
                    and note_pitch[i - 1] == unit_pitch[j - 1]
                    and cost[1, i, j - 1] < INF
                ):
                    value = cost[1, i, j - 1] + merge
                    if value < best1:
                        best1, op1, layer1 = value, OP_MERGE, 1
            if i > 0:
                previous = min(cost[0, i - 1, j], cost[1, i - 1, j])
                previous_layer = 0 if cost[0, i - 1, j] <= cost[1, i - 1, j] else 1
                value = previous + insert
                if value < best0:
                    best0, op0, layer0 = value, OP_INSERT, previous_layer
            if j > 0:
                previous = min(cost[0, i, j - 1], cost[1, i, j - 1])
                previous_layer = 0 if cost[0, i, j - 1] <= cost[1, i, j - 1] else 1
                value = previous + (delete if unit_linked[j - 1] else ornament_skip)
                if value < best0:
                    best0, op0, layer0 = value, OP_DELETE, previous_layer
            cost[0, i, j] = best0
            back[0, i, j] = op0
            from_layer[0, i, j] = layer0
            cost[1, i, j] = best1
            back[1, i, j] = op1
            from_layer[1, i, j] = layer1
    final_layer = 0 if cost[0, n, m] <= cost[1, n, m] else 1
    total = min(cost[0, n, m], cost[1, n, m])
    return total, back, from_layer, final_layer


def _mergeable(template: Sequence[OrnamentTemplateUnit]) -> np.ndarray:
    flags = np.zeros(len(template), np.bool_)
    for position in range(1, len(template)):
        previous, current = template[position - 1], template[position]
        flags[position] = (
            previous.kind == "linked"
            and current.kind == "linked"
            and previous.copy_pass == current.copy_pass
            and previous.pitch == current.pitch
            and previous.score_index is not None
            and current.score_index == previous.score_index + 1
        )
    return flags


def _backtrace(back, from_layer, cost_layers_final_layer, n, m):
    operations = []
    layer = cost_layers_final_layer
    i, j = n, m
    while i > 0 or j > 0:
        op = int(back[layer, i, j])
        next_layer = int(from_layer[layer, i, j])
        operations.append((op, i - 1, j - 1))
        if op in (OP_MATCH, OP_ORNAMENT):
            i, j = i - 1, j - 1
        elif op == OP_INSERT:
            i -= 1
        elif op in (OP_DELETE, OP_MERGE):
            j -= 1
        else:
            raise RuntimeError("DP backtrace reached an unset cell")
        layer = next_layer
    return operations[::-1]


@dataclass(frozen=True)
class PerfectAlignment:
    events: tuple[JointEvent, ...]
    deletions: frozenset[int]
    cost: float
    source_span: tuple[int, int] | None
    copies: int


def align_perfect(
    notes: Sequence[tuple[int, float, float]],
    score: Sequence[ScoreEvent],
    score_path: Path | str,
    costs: PerfectDPCosts = PerfectDPCosts(),
) -> PerfectAlignment:
    score_tuple = tuple(score)
    patterns = score_ornament_patterns(score_path, score_tuple)
    note_pitch = np.array([int(pitch) for pitch, _s, _e in notes], np.int64)
    cost_vector = np.array([
        costs.substitute, costs.insert, costs.delete,
        costs.ornament_skip, costs.ornament_near, costs.merge,
    ], np.float64)
    best = None
    for hypothesis in grammar_hypotheses(score_tuple, len(notes), edit_slack=100000):
        template = expand_ornament_hypothesis(score_tuple, patterns, hypothesis)
        if abs(len(template) - len(notes)) > costs.length_slack + 4 * len(score_tuple) // 10:
            continue
        unit_pitch = np.array([unit.pitch for unit in template], np.int64)
        unit_linked = np.array([unit.kind == "linked" for unit in template], np.bool_)
        total, back, from_layer, final_layer = _dp(
            note_pitch, unit_pitch, unit_linked, _mergeable(template), cost_vector
        )
        total += costs.copy * hypothesis.copies
        if best is None or total < best[0]:
            best = (total, hypothesis, template, back, from_layer, final_layer)
    if best is None:
        raise ValueError("No repeat hypothesis within the length slack")
    total, hypothesis, template, back, from_layer, final_layer = best
    n, m = len(notes), len(template)
    operations = _backtrace(back, from_layer, final_layer, n, m) if n or m else []

    assignment: dict[int, list[int]] = {}
    ornament_notes: set[int] = set()
    inserted: list[int] = []
    deleted_units: list[int] = []
    for op, i, j in operations:
        if op == OP_MATCH:
            assignment[i] = [j]
        elif op == OP_MERGE:
            assignment[i].append(j)
        elif op == OP_ORNAMENT:
            ornament_notes.add(i)
        elif op == OP_INSERT:
            inserted.append(i)
        elif op == OP_DELETE and template[j].kind == "linked":
            deleted_units.append(j)

    pass_of_note: dict[int, int] = {i: template[units[0]].copy_pass for i, units in assignment.items()}
    events = []
    for i, (pitch, start, end) in enumerate(notes):
        end = max(float(end), float(start) + 0.001)
        if i in assignment:
            units = [template[j] for j in assignment[i]]
            indices = [int(unit.score_index) for unit in units]
            copy_pass = int(units[0].copy_pass)
            relationship = (
                "copy" if copy_pass > 0
                else "match" if int(pitch) == units[0].pitch
                else "substitute"
            )
            events.append(JointEvent(
                pitch=int(pitch), start=float(start), end=end,
                score_span=(min(indices), max(indices) + 1),
                relationship=relationship, copy_pass=copy_pass,
                rendered_index=i,
            ))
            continue
        relationship = "extra"
        if i not in ornament_notes:
            before = [pass_of_note[k] for k in range(i - 1, -1, -1) if k in pass_of_note][:1]
            after = [pass_of_note[k] for k in range(i + 1, len(notes)) if k in pass_of_note][:1]
            if before and after and before[0] > 0 and after[0] > 0:
                relationship = "copy"
        events.append(JointEvent(
            pitch=int(pitch), start=float(start), end=end, score_span=None,
            relationship=relationship, rendered_index=i,
        ))
    deletions = frozenset(
        int(template[j].score_index) for j in deleted_units if template[j].copy_pass == 0
    )
    return PerfectAlignment(
        events=tuple(events), deletions=deletions, cost=float(total),
        source_span=hypothesis.source_span, copies=int(hypothesis.copies),
    )
