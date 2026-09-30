"""Score-informed DP aligner for imperfect (transcribed) note sequences.

The v1 aligner assumes a perfect transcription: every input note is either
aligned to the score template or reported as an extra, and a missing note can
only become a missed score event. With a real transcriber, false notes become
unmatchable extras and missed notes shift repeat-hypothesis and span choices.

v2 keeps v1's repeat hypotheses, ornament template, and operations, and adds
per-note evidence:

* confidence-scaled insertion, plus a ``drop`` operation that removes a
  low-confidence note that fits nowhere instead of reporting it as an extra;
* optional candidates (weak CTC peaks the greedy decoder skipped) that can
  fill a score note at a small cost or be dropped for free;
* a runner-up pitch per note that may match the score at a reduced cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numba
import numpy as np

from .grammar_mapper_v2 import grammar_hypotheses
from .index import JointEvent, ScoreEvent
from .ornament_mapper_v1 import expand_ornament_hypothesis, score_ornament_patterns
from .perfect_dp_aligner_v1 import _mergeable


INF = 1e18
OP_NONE, OP_MATCH, OP_ORNAMENT, OP_INSERT, OP_DELETE, OP_MERGE, OP_DROP, OP_ALT = range(8)


@dataclass(frozen=True)
class RobustDPCosts:
    substitute: float = 1.4
    insert: float = 1.0
    delete: float = 1.0
    ornament_skip: float = 0.35
    ornament_near: float = 0.30
    merge: float = 0.15
    copy: float = 0.5
    length_slack: int = 24
    drop_confidence: float = 0.5
    drop_cost: float = 0.8
    optional_match: float = 0.35
    alternative_match: float = 0.6
    alternative_min_confidence: float = 0.1


@dataclass(frozen=True)
class TranscribedNote:
    pitch: int
    start: float
    end: float
    confidence: float = 1.0
    optional: bool = False
    alternative_pitch: int = -1
    alternative_confidence: float = 0.0


@numba.njit(cache=True)
def _dp(note_pitch, note_alt, unit_pitch, unit_linked, unit_mergeable,
        insert_cost, drop_cost, match_extra, alt_cost, costs):
    substitute, delete, ornament_skip, ornament_near, merge = costs[0], costs[1], costs[2], costs[3], costs[4]
    n = note_pitch.shape[0]
    m = unit_pitch.shape[0]
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
                a = cost[0, i - 1, j - 1]
                b = cost[1, i - 1, j - 1]
                previous = a if a <= b else b
                previous_layer = 0 if a <= b else 1
                if unit_linked[j - 1]:
                    if note_pitch[i - 1] == unit_pitch[j - 1]:
                        value = previous + match_extra[i - 1]
                        if value < best1:
                            best1, op1, layer1 = value, OP_MATCH, previous_layer
                    else:
                        value = previous + match_extra[i - 1] + substitute
                        if value < best1:
                            best1, op1, layer1 = value, OP_MATCH, previous_layer
                        if note_alt[i - 1] == unit_pitch[j - 1]:
                            value = previous + match_extra[i - 1] + alt_cost[i - 1]
                            if value < best1:
                                best1, op1, layer1 = value, OP_ALT, previous_layer
                else:
                    difference = abs(note_pitch[i - 1] - unit_pitch[j - 1])
                    if difference <= 2:
                        value = previous + match_extra[i - 1] + (0.0 if difference == 0 else ornament_near)
                        if value < best0:
                            best0, op0, layer0 = value, OP_ORNAMENT, previous_layer
                if (unit_mergeable[j - 1] and j >= 2 and note_pitch[i - 1] == unit_pitch[j - 1]
                        and cost[1, i, j - 1] < INF):
                    value = cost[1, i, j - 1] + merge
                    if value < best1:
                        best1, op1, layer1 = value, OP_MERGE, 1
            if i > 0:
                a = cost[0, i - 1, j]
                b = cost[1, i - 1, j]
                previous = a if a <= b else b
                previous_layer = 0 if a <= b else 1
                value = previous + insert_cost[i - 1]
                if value < best0:
                    best0, op0, layer0 = value, OP_INSERT, previous_layer
                value = previous + drop_cost[i - 1]
                if value < best0:
                    best0, op0, layer0 = value, OP_DROP, previous_layer
            if j > 0:
                a = cost[0, i, j - 1]
                b = cost[1, i, j - 1]
                previous = a if a <= b else b
                previous_layer = 0 if a <= b else 1
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
    total = cost[0, n, m] if cost[0, n, m] <= cost[1, n, m] else cost[1, n, m]
    return total, back, from_layer, final_layer


def _backtrace(back, from_layer, layer, n, m):
    operations = []
    i, j = n, m
    while i > 0 or j > 0:
        op = int(back[layer, i, j])
        next_layer = int(from_layer[layer, i, j])
        operations.append((op, i - 1, j - 1))
        if op in (OP_MATCH, OP_ORNAMENT, OP_ALT):
            i, j = i - 1, j - 1
        elif op in (OP_INSERT, OP_DROP):
            i -= 1
        elif op in (OP_DELETE, OP_MERGE):
            j -= 1
        else:
            raise RuntimeError("DP backtrace reached an unset cell")
        layer = next_layer
    return operations[::-1]


@dataclass(frozen=True)
class RobustAlignment:
    events: tuple[JointEvent, ...]
    deletions: frozenset[int]
    cost: float
    source_span: tuple[int, int] | None
    copies: int
    kept_note_indices: tuple[int, ...]


def _as_notes(notes: Sequence) -> list[TranscribedNote]:
    output = []
    for value in notes:
        if isinstance(value, TranscribedNote):
            output.append(value)
        else:
            values = list(value)
            output.append(TranscribedNote(
                int(values[0]), float(values[1]), float(values[2]),
                float(values[3]) if len(values) > 3 else 1.0,
                bool(values[4]) if len(values) > 4 else False,
                int(values[5]) if len(values) > 5 else -1,
                float(values[6]) if len(values) > 6 else 0.0,
            ))
    return output


def align_robust(
    notes: Sequence,
    score: Sequence[ScoreEvent],
    score_path: Path | str,
    costs: RobustDPCosts = RobustDPCosts(),
) -> RobustAlignment:
    items = _as_notes(notes)
    score_tuple = tuple(score)
    patterns = score_ornament_patterns(score_path, score_tuple)
    note_pitch = np.array([note.pitch for note in items], np.int64)
    note_alt = np.array([
        note.alternative_pitch if note.alternative_confidence >= costs.alternative_min_confidence else -1
        for note in items
    ], np.int64)
    insert_cost = np.array([
        INF if note.optional else costs.insert for note in items
    ], np.float64)
    drop_cost = np.array([
        0.0 if note.optional
        else (costs.drop_cost if note.confidence < costs.drop_confidence else INF)
        for note in items
    ], np.float64)
    match_extra = np.array([costs.optional_match if note.optional else 0.0 for note in items], np.float64)
    alt_cost = np.array([costs.alternative_match for _ in items], np.float64)
    cost_vector = np.array([costs.substitute, costs.delete, costs.ornament_skip,
                            costs.ornament_near, costs.merge], np.float64)
    primary_count = sum(not note.optional for note in items)
    hypotheses = grammar_hypotheses(score_tuple, primary_count, edit_slack=100000)
    best = None
    for relaxed in (False, True):
        for hypothesis in hypotheses:
            template = expand_ornament_hypothesis(score_tuple, patterns, hypothesis)
            if not relaxed and abs(len(template) - primary_count) > costs.length_slack + 4 * len(score_tuple) // 10:
                continue
            unit_pitch = np.array([unit.pitch for unit in template], np.int64)
            unit_linked = np.array([unit.kind == "linked" for unit in template], np.bool_)
            total, back, from_layer, final_layer = _dp(
                note_pitch, note_alt, unit_pitch, unit_linked, _mergeable(template),
                insert_cost, drop_cost, match_extra, alt_cost, cost_vector,
            )
            total += costs.copy * hypothesis.copies
            if best is None or total < best[0]:
                best = (total, hypothesis, template, back, from_layer, final_layer)
        if best is not None:
            break
    if best is None:
        raise ValueError("No repeat hypothesis available")
    total, hypothesis, template, back, from_layer, final_layer = best
    n, m = len(items), len(template)
    operations = _backtrace(back, from_layer, final_layer, n, m) if n or m else []

    assignment: dict[int, list[int]] = {}
    use_alt: set[int] = set()
    ornament_notes: set[int] = set()
    inserted: set[int] = set()
    dropped: set[int] = set()
    deleted_units: list[int] = []
    for op, i, j in operations:
        if op in (OP_MATCH, OP_ALT):
            assignment[i] = [j]
            if op == OP_ALT:
                use_alt.add(i)
        elif op == OP_MERGE:
            assignment[i].append(j)
        elif op == OP_ORNAMENT:
            ornament_notes.add(i)
        elif op == OP_INSERT:
            inserted.add(i)
        elif op == OP_DROP:
            dropped.add(i)
        elif op == OP_DELETE and template[j].kind == "linked":
            deleted_units.append(j)

    kept = [i for i in range(n) if i not in dropped]
    output_index = {i: position for position, i in enumerate(kept)}
    pass_of_note = {i: template[units[0]].copy_pass for i, units in assignment.items()}
    events = []
    for i in kept:
        note = items[i]
        pitch = note.alternative_pitch if i in use_alt else note.pitch
        end = max(float(note.end), float(note.start) + 0.001)
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
                pitch=int(pitch), start=float(note.start), end=end,
                score_span=(min(indices), max(indices) + 1),
                relationship=relationship, copy_pass=copy_pass,
                rendered_index=output_index[i],
            ))
            continue
        relationship = "extra"
        if i not in ornament_notes:
            before = [pass_of_note[k] for k in range(i - 1, -1, -1) if k in pass_of_note][:1]
            after = [pass_of_note[k] for k in range(i + 1, n) if k in pass_of_note][:1]
            if before and after and before[0] > 0 and after[0] > 0:
                relationship = "copy"
        events.append(JointEvent(
            pitch=int(pitch), start=float(note.start), end=end, score_span=None,
            relationship=relationship, rendered_index=output_index[i],
        ))
    deletions = frozenset(
        int(template[j].score_index) for j in deleted_units if template[j].copy_pass == 0
    )
    return RobustAlignment(
        events=tuple(events), deletions=deletions, cost=float(total),
        source_span=hypothesis.source_span, copies=int(hypothesis.copies),
        kept_note_indices=tuple(kept),
    )
