"""Timing-aware, precision-gated score aligner for transcribed note sequences (v3).

v2 aligns on pitch alone. Among identical pitches (repeated notes, an extra
note next to a score note of the same pitch) every choice costs the same, and
v2 reports every unexplained transcribed note as an extra and every unmatched
score note as a missed note. On real recordings that produces many false
extras (10-40 ms pitch blips at slurred note changes and attacks) and false
missed notes (notes the transcriber did not hear, unplayed ends of a take).

v3 keeps v2's repeat grammar, ornament template and DP operations, and adds:

* artifact suppression: a note shorter than ``artifact_ioi`` (onset to next
  onset) becomes an optional candidate. It can still fill a score or ornament
  unit, or be dropped, but never becomes an extra on its own.
* a timing pass: matched notes of the first pass give a tempo map from
  template (performed-score) time to audio time; a second DP on the same repeat
  hypothesis adds a capped onset-deviation cost to matches, which breaks ties
  between identical pitches.
* gating (``gate_alignment``): each extra and missed note is checked against
  frame evidence from the transcriber. Unsupported calls are withheld, and a
  missed score note with strong evidence that it was played becomes an
  inferred match.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numba
import numpy as np

from .grammar_mapper_v2 import grammar_hypotheses
from .index import JointEvent, ScoreEvent
from .ornament_mapper_v1 import expand_ornament_hypothesis, score_ornament_patterns
from .perfect_dp_aligner_v1 import _mergeable
from .robust_dp_aligner_v2 import (
    INF, OP_ALT, OP_DELETE, OP_DROP, OP_INSERT, OP_MATCH, OP_MERGE, OP_NONE, OP_ORNAMENT,
    RobustDPCosts, TranscribedNote, _as_notes, _backtrace, _dp,
)


@dataclass(frozen=True)
class AlignerV3Config:
    costs: RobustDPCosts = field(default_factory=RobustDPCosts)
    artifact_ioi: float = 0.06
    artifact_keep_leaps: int = 3
    timing_weight: float = 0.0
    timing_tau: float = 0.06
    timing_tau_fraction: float = 0.35
    timing_cap: float = 1.0
    anchor_confidence: float = 0.9
    copy_after_replay: bool = False


@dataclass(frozen=True)
class GateConfig:
    enabled: bool = True
    extra_min_ioi: float = 0.0
    extra_same_pitch_min_ioi: float = 10.0
    extra_min_confidence: float = 0.0
    missed_max_pitch_probability: float = 1.0
    missed_max_voiced: float = 1.0
    missed_infer_pitch_probability: float = 2.0
    missed_edge_runs: bool = True
    missed_max_run: int = 10_000
    missed_min_level_drop_db: float | None = None
    missed_max_slot_onset: float = 2.0
    extra_model: dict | None = None
    extra_threshold: float = 0.5
    extra_model_origins: tuple[str, ...] = ("inserted", "ornament")
    missed_model: dict | None = None
    missed_threshold: float = 0.5
    missed_features: str = "v1"
    missed_never_ornamented: bool = False
    missed_max_presence: float = 2.0
    extra_features: str = "v1"
    clip_min_match_fraction: float = 0.0


@dataclass(frozen=True)
class AlignedExtra:
    event_position: int
    note_index: int
    origin: str


@dataclass(frozen=True)
class MissedUnit:
    score_index: int
    expected_time: float
    expected_duration: float
    previous_onset: float | None
    next_onset: float | None
    run_length: int
    edge: bool
    ornamented: bool = False
    nearby_similar: int = 0
    nearby_unexplained: int = 0
    ornamented_neighbor: bool = False
    in_repeat_span: bool = False
    neighbor_pitch_distance: int = 99
    interpolated_time: float = float("nan")
    previous_score_pitch: int = -1
    next_score_pitch: int = -1


@dataclass(frozen=True)
class AlignmentV3:
    events: tuple[JointEvent, ...]
    deletions: frozenset[int]
    cost: float
    source_span: tuple[int, int] | None
    copies: int
    kept_note_indices: tuple[int, ...]
    extras: tuple[AlignedExtra, ...]
    missed: tuple[MissedUnit, ...]
    notes: tuple[TranscribedNote, ...]
    artifact_notes: frozenset[int]
    seconds_per_ql: float
    match_fraction: float


@numba.njit(cache=True)
def _dp_timed(note_pitch, note_alt, unit_pitch, unit_linked, unit_mergeable,
              insert_cost, drop_cost, match_extra, alt_cost, costs, timing):
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
                    t = timing[i - 1, j - 1]
                    if note_pitch[i - 1] == unit_pitch[j - 1]:
                        value = previous + match_extra[i - 1] + t
                        if value < best1:
                            best1, op1, layer1 = value, OP_MATCH, previous_layer
                    else:
                        value = previous + match_extra[i - 1] + substitute + t
                        if value < best1:
                            best1, op1, layer1 = value, OP_MATCH, previous_layer
                        if note_alt[i - 1] == unit_pitch[j - 1]:
                            value = previous + match_extra[i - 1] + alt_cost[i - 1] + t
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


def mark_artifacts(items: Sequence[TranscribedNote], config: AlignerV3Config) -> tuple[list[TranscribedNote], frozenset[int]]:
    """Make very short committed notes optional unless they leap away from both neighbours."""

    if config.artifact_ioi <= 0:
        return list(items), frozenset()
    primary = [k for k, note in enumerate(items) if not note.optional]
    flagged = set()
    for position, k in enumerate(primary):
        if position + 1 >= len(primary):
            continue
        following = items[primary[position + 1]]
        if following.start - items[k].start >= config.artifact_ioi:
            continue
        pitch = items[k].pitch
        previous = items[primary[position - 1]].pitch if position > 0 else None
        neighbours = [value for value in (previous, following.pitch) if value is not None]
        if neighbours and all(abs(pitch - value) > config.artifact_keep_leaps for value in neighbours) and (
            previous is None or not (min(previous, following.pitch) < pitch < max(previous, following.pitch))
        ):
            continue
        flagged.add(k)
    output = [replace(note, optional=True) if k in flagged else note for k, note in enumerate(items)]
    return output, frozenset(flagged)


def _arrays(items: Sequence[TranscribedNote], costs: RobustDPCosts):
    note_pitch = np.array([note.pitch for note in items], np.int64)
    note_alt = np.array([
        note.alternative_pitch if note.alternative_confidence >= costs.alternative_min_confidence else -1
        for note in items
    ], np.int64)
    insert_cost = np.array([INF if note.optional else costs.insert for note in items], np.float64)
    drop_cost = np.array([
        0.0 if note.optional else (costs.drop_cost if note.confidence < costs.drop_confidence else INF)
        for note in items
    ], np.float64)
    match_extra = np.array([costs.optional_match if note.optional else 0.0 for note in items], np.float64)
    alt_cost = np.array([costs.alternative_match for _ in items], np.float64)
    cost_vector = np.array([costs.substitute, costs.delete, costs.ornament_skip,
                            costs.ornament_near, costs.merge], np.float64)
    return note_pitch, note_alt, insert_cost, drop_cost, match_extra, alt_cost, cost_vector


def _tempo_map(anchor_q: np.ndarray, anchor_t: np.ndarray, query_q: np.ndarray,
               exclude_self: bool = True) -> tuple[np.ndarray, float]:
    """Robust local-linear map from performed-score time (ql) to audio seconds."""

    if anchor_q.size == 0:
        return np.full(query_q.shape, np.nan), float("nan")
    order = np.argsort(anchor_q, kind="stable")
    q, t = anchor_q[order], anchor_t[order]
    slopes = np.diff(t) / np.maximum(np.diff(q), 1e-6)
    valid = np.diff(q) > 1e-6
    spq = float(np.median(slopes[valid])) if valid.any() else 0.5
    spq = spq if np.isfinite(spq) and spq > 0 else 0.5
    if q.size == 1:
        return t[0] + (query_q - q[0]) * spq, spq
    output = np.empty(query_q.shape)
    k = 4
    for position, value in enumerate(query_q):
        right = int(np.searchsorted(q, value, side="left"))
        lo, hi = max(0, right - k), min(q.size, right + k)
        qs, ts = q[lo:hi], t[lo:hi]
        if exclude_self:
            keep = np.abs(qs - value) > 1e-6
            if keep.sum() >= 2:
                qs, ts = qs[keep], ts[keep]
        if qs.size < 2 or np.ptp(qs) < 1e-6:
            output[position] = (ts.mean() if ts.size else t[0]) + (value - (qs.mean() if qs.size else q[0])) * spq
            continue
        weights = 1.0 / (1.0 + np.abs(qs - value))
        for _ in range(3):
            w = weights / weights.sum()
            qm, tm = float((w * qs).sum()), float((w * ts).sum())
            var = float((w * (qs - qm) ** 2).sum())
            slope = float((w * (qs - qm) * (ts - tm)).sum() / var) if var > 1e-9 else spq
            if not np.isfinite(slope) or slope <= 0:
                slope = spq
            residual = ts - (tm + slope * (qs - qm))
            scale = max(0.03, 1.5 * float(np.median(np.abs(residual))))
            weights = weights * np.minimum(1.0, scale / np.maximum(np.abs(residual), 1e-9))
        output[position] = tm + slope * (value - qm)
    return output, spq


def _assignment(operations):
    assignment: dict[int, list[int]] = {}
    use_alt: set[int] = set()
    ornament_notes: set[int] = set()
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
        elif op == OP_DROP:
            dropped.add(i)
        elif op == OP_DELETE:
            deleted_units.append(j)
    return assignment, use_alt, ornament_notes, dropped, deleted_units


def align_v3(
    notes: Sequence,
    score: Sequence[ScoreEvent],
    score_path: Path | str,
    config: AlignerV3Config = AlignerV3Config(),
) -> AlignmentV3:
    costs = config.costs
    raw_items = _as_notes(notes)
    items, artifacts = mark_artifacts(raw_items, config)
    score_tuple = tuple(score)
    patterns = score_ornament_patterns(score_path, score_tuple)
    arrays = _arrays(items, costs)
    note_pitch, note_alt, insert_cost, drop_cost, match_extra, alt_cost, cost_vector = arrays
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
                best = (total, hypothesis, template, back, from_layer, final_layer, unit_pitch, unit_linked)
        if best is not None:
            break
    if best is None:
        raise ValueError("No repeat hypothesis available")
    total, hypothesis, template, back, from_layer, final_layer, unit_pitch, unit_linked = best
    n, m = len(items), len(template)
    operations = _backtrace(back, from_layer, final_layer, n, m) if n or m else []
    assignment, use_alt, ornament_notes, dropped, deleted_units = _assignment(operations)

    unit_time = np.array([float(unit.time) for unit in template], np.float64)
    anchors = [
        (unit_time[units[0]], items[i].start) for i, units in assignment.items()
        if not items[i].optional and items[i].confidence >= config.anchor_confidence
        and template[units[0]].pitch == (items[i].alternative_pitch if i in use_alt else items[i].pitch)
    ]
    anchor_q = np.array([a for a, _b in anchors], np.float64)
    anchor_t = np.array([b for _a, b in anchors], np.float64)
    expected, spq = _tempo_map(anchor_q, anchor_t, unit_time)

    if config.timing_weight > 0 and anchor_q.size >= 4 and n and m:
        durations = np.array([
            max(float(score_tuple[unit.score_index].ql_end - score_tuple[unit.score_index].ql_start), 0.05)
            if unit.score_index is not None else 0.1 for unit in template
        ])
        tau = np.maximum(config.timing_tau, config.timing_tau_fraction * durations * spq)
        starts = np.array([note.start for note in items], np.float64)
        deviation = (starts[:, None] - expected[None, :]) / tau[None, :]
        timing = config.timing_weight * np.minimum(deviation * deviation, config.timing_cap)
        timing = np.where(np.isfinite(timing), timing, 0.0)
        total2, back, from_layer, final_layer = _dp_timed(
            note_pitch, note_alt, unit_pitch, unit_linked, _mergeable(template),
            insert_cost, drop_cost, match_extra, alt_cost, cost_vector, np.ascontiguousarray(timing),
        )
        total = total2 + costs.copy * hypothesis.copies
        operations = _backtrace(back, from_layer, final_layer, n, m)
        assignment, use_alt, ornament_notes, dropped, deleted_units = _assignment(operations)

    kept = [i for i in range(n) if i not in dropped]
    output_index = {i: position for position, i in enumerate(kept)}
    pass_of_note = {i: template[units[0]].copy_pass for i, units in assignment.items()}
    events: list[JointEvent] = []
    extras: list[AlignedExtra] = []
    for i in kept:
        note = items[i]
        pitch = note.alternative_pitch if i in use_alt else note.pitch
        end = max(float(note.end), float(note.start) + 0.001)
        if i in assignment:
            units = [template[j] for j in assignment[i]]
            indices = [int(unit.score_index) for unit in units]
            copy_pass = int(units[0].copy_pass)
            relationship = ("copy" if copy_pass > 0 else "match" if int(pitch) == units[0].pitch else "substitute")
            events.append(JointEvent(
                pitch=int(pitch), start=float(note.start), end=end,
                score_span=(min(indices), max(indices) + 1), relationship=relationship,
                copy_pass=copy_pass, rendered_index=output_index[i], confidence=float(note.confidence),
            ))
            continue
        relationship = "extra"
        origin = "ornament" if i in ornament_notes else "inserted"
        if i not in ornament_notes:
            before = [pass_of_note[k] for k in range(i - 1, -1, -1) if k in pass_of_note][:1]
            after = [pass_of_note[k] for k in range(i + 1, n) if k in pass_of_note][:1]
            # An unexplained note right after a replayed note belongs to the replay
            # (the replayed copy of an extra note), also at the end of the pass.
            if before and before[0] > 0 and (config.copy_after_replay or (after and after[0] > 0)):
                relationship = "copy"
                origin = "repeat_pass"
        extras.append(AlignedExtra(len(events), i, origin))
        events.append(JointEvent(
            pitch=int(pitch), start=float(note.start), end=end, score_span=None,
            relationship=relationship, rendered_index=output_index[i], confidence=float(note.confidence),
        ))

    linked_deleted = [j for j in deleted_units if template[j].kind == "linked" and template[j].copy_pass == 0]
    deletions = frozenset(int(template[j].score_index) for j in linked_deleted)
    unit_onset = {}
    unit_pitch_heard = {}
    for i, units in assignment.items():
        if i not in dropped:
            for j in units:
                unit_onset[j] = items[i].start
                unit_pitch_heard[j] = items[i].alternative_pitch if i in use_alt else items[i].pitch
    missed = []
    linked_positions = [j for j in range(m) if template[j].kind == "linked"]
    deleted_set = set(linked_deleted)
    first_matched = min(unit_onset) if unit_onset else None
    last_matched = max(unit_onset) if unit_onset else None
    for j in linked_deleted:
        position = linked_positions.index(j)
        lo = position
        while lo > 0 and linked_positions[lo - 1] in deleted_set:
            lo -= 1
        hi = position
        while hi + 1 < len(linked_positions) and linked_positions[hi + 1] in deleted_set:
            hi += 1
        previous_units = [k for k in range(j - 1, -1, -1) if k in unit_onset][:1]
        following_units = [k for k in range(j + 1, m) if k in unit_onset][:1]
        previous = [unit_onset[k] for k in previous_units]
        following = [unit_onset[k] for k in following_units]
        event = score_tuple[template[j].score_index]
        interpolated = float("nan")
        if previous_units and following_units:
            span = unit_time[following_units[0]] - unit_time[previous_units[0]]
            if span > 0:
                interpolated = previous[0] + (unit_time[j] - unit_time[previous_units[0]]) / span * (following[0] - previous[0])
        lo_t = previous[0] if previous else -1.0
        hi_t = following[0] if following else float("inf")
        unexplained = [i for i in range(n) if i not in assignment and lo_t < items[i].start < hi_t]
        similar = sum(1 for i in unexplained if abs(items[i].pitch - int(event.pitch)) <= 2)
        pattern = patterns[template[j].score_index]
        score_index = int(template[j].score_index)
        neighbour_patterns = [patterns[s] for s in (score_index - 1, score_index + 1) if 0 <= s < len(patterns)]
        heard_previous = [unit_pitch_heard[k] for k in range(j - 1, -1, -1) if k in unit_pitch_heard][:1]
        heard_next = [unit_pitch_heard[k] for k in range(j + 1, m) if k in unit_pitch_heard][:1]
        distances = [abs(int(value) - int(event.pitch)) for value in heard_previous + heard_next]
        span = hypothesis.source_span
        missed.append(MissedUnit(
            score_index=int(template[j].score_index),
            expected_time=float(expected[j]) if np.isfinite(expected[j]) else float("nan"),
            expected_duration=float(max(event.ql_end - event.ql_start, 0.05) * spq),
            previous_onset=previous[0] if previous else None,
            next_onset=following[0] if following else None,
            run_length=hi - lo + 1,
            edge=first_matched is None or j < first_matched or j > last_matched,
            ornamented=bool(pattern.prefix) or len(pattern.body) > 1,
            nearby_similar=similar,
            nearby_unexplained=len(unexplained),
            ornamented_neighbor=any(bool(p.prefix) or len(p.body) > 1 for p in neighbour_patterns),
            in_repeat_span=bool(hypothesis.copies and span and span[0] <= score_index < span[1]),
            neighbor_pitch_distance=min(distances) if distances else 99,
            interpolated_time=interpolated,
            previous_score_pitch=int(score_tuple[score_index - 1].pitch) if score_index > 0 else -1,
            next_score_pitch=int(score_tuple[score_index + 1].pitch) if score_index + 1 < len(score_tuple) else -1,
        ))
    linked_events = sum(1 for e in events if e.score_span is not None)
    return AlignmentV3(
        events=tuple(events), deletions=deletions, cost=float(total),
        source_span=hypothesis.source_span, copies=int(hypothesis.copies),
        kept_note_indices=tuple(kept), extras=tuple(extras), missed=tuple(missed),
        notes=tuple(items), artifact_notes=artifacts, seconds_per_ql=float(spq),
        match_fraction=sum(1 for e in events if e.relationship in ("match", "copy")) / max(len(events), 1),
    )


def frame_evidence(evidence: Mapping[str, Any], start: float, end: float, token: int | None) -> dict[str, float]:
    hop = float(evidence["hop"])
    ctc = evidence["ctc"]
    voiced = evidence["voiced"]
    a = max(0, int(np.floor(start / hop)))
    b = min(len(voiced), max(a + 1, int(np.ceil(end / hop))))
    if a >= len(voiced):
        return {"voiced": 0.0, "pitch_probability": 0.0, "frames": 0}
    pitch_probability = float(ctc[a:b, token].max()) if token is not None and 0 < token < ctc.shape[1] else 0.0
    return {"voiced": float(np.mean(voiced[a:b])), "pitch_probability": pitch_probability, "frames": b - a}


def missed_features(alignment: AlignmentV3, unit: MissedUnit, score: Sequence[ScoreEvent],
                    evidence: Mapping[str, Any]) -> dict[str, float]:
    pitch = int(score[unit.score_index].pitch)
    token = pitch - int(evidence["midi_min"]) + 1
    hop = float(evidence["hop"])
    lo = (unit.previous_onset + 2 * hop) if unit.previous_onset is not None else (
        unit.expected_time - unit.expected_duration if np.isfinite(unit.expected_time) else 0.0)
    hi = (unit.next_onset - hop) if unit.next_onset is not None else (
        unit.expected_time + 2 * unit.expected_duration if np.isfinite(unit.expected_time) else lo + 0.3)
    hi = max(hi, lo + hop)
    window = frame_evidence(evidence, lo, hi, token)
    slot_lo = unit.expected_time if np.isfinite(unit.expected_time) else lo
    slot_lo = min(max(slot_lo, lo), hi - hop)
    slot = frame_evidence(evidence, slot_lo, min(hi, slot_lo + max(unit.expected_duration, 2 * hop)), token)
    gap = (unit.next_onset - unit.previous_onset) if (unit.next_onset is not None and unit.previous_onset is not None) else float("nan")
    level_drop = float("nan")
    slot_onset = float("nan")
    rms = evidence.get("rms_db")
    if rms is not None and unit.previous_onset is not None and unit.next_onset is not None and gap > 4 * hop:
        a = int(unit.previous_onset / hop)
        split = int((unit.previous_onset + 0.35 * gap) / hop)
        b = int(unit.next_onset / hop) - 2
        if b > split + 1 and split > a:
            level_drop = float(np.median(rms[a:split]) - np.median(rms[split:b]))
            onset = evidence.get("onset")
            if onset is not None:
                slot_onset = float(np.max(onset[split:b]))
    presence = float("nan")
    scorer = evidence.get("presence")
    if scorer is not None:
        center_time = unit.interpolated_time if np.isfinite(unit.interpolated_time) else unit.expected_time
        if np.isfinite(center_time):
            uncertainty = (gap if np.isfinite(gap) else unit.expected_duration) * 0.05 / hop
            presence = float(scorer([(center_time / hop, pitch, unit.previous_score_pitch, unit.next_score_pitch,
                                      unit.expected_duration / hop, uncertainty)])[0])
    return {
        "presence": presence,
        "pitch_probability": window["pitch_probability"],
        "window_voiced": window["voiced"],
        "slot_voiced": slot["voiced"],
        "gap_seconds": gap,
        "expected_duration": unit.expected_duration,
        "run_length": unit.run_length,
        "edge": float(unit.edge),
        "level_drop_db": level_drop,
        "slot_onset": slot_onset,
        "ornamented": float(unit.ornamented),
        "nearby_similar": float(unit.nearby_similar),
        "nearby_unexplained": float(unit.nearby_unexplained),
        "gap_ratio": (gap / max(2 * unit.expected_duration, 1e-3)) if np.isfinite(gap) else float("nan"),
        "ornamented_neighbor": float(unit.ornamented_neighbor),
        "in_repeat_span": float(unit.in_repeat_span),
        "neighbor_pitch_distance": float(min(unit.neighbor_pitch_distance, 12)),
    }


def extra_features(alignment: AlignmentV3, extra: AlignedExtra, evidence: Mapping[str, Any] | None) -> dict[str, float]:
    events = alignment.events
    event = events[extra.event_position]
    position = extra.event_position
    previous = events[position - 1] if position > 0 else None
    following = events[position + 1] if position + 1 < len(events) else None
    ioi = (following.start - event.start) if following is not None else float("inf")
    same = any(other is not None and other.pitch == event.pitch for other in (previous, following))
    aba = previous is not None and following is not None and previous.pitch == following.pitch != event.pitch
    note = alignment.notes[extra.note_index]
    onset = 0.0
    if evidence is not None:
        hop = float(evidence["hop"])
        frame = int(round(event.start / hop))
        heads = evidence.get("onset")
        if heads is not None and len(heads):
            onset = float(np.max(heads[max(0, frame - 2):frame + 3]))
    paired = any(
        (unit.previous_onset is None or unit.previous_onset < event.start)
        and (unit.next_onset is None or event.start < unit.next_onset)
        for unit in alignment.missed
    )
    previous_step = abs(event.pitch - previous.pitch) if previous is not None else 99
    next_step = abs(event.pitch - following.pitch) if following is not None else 99
    previous_ioi = (event.start - previous.start) if previous is not None else float("inf")
    return {"ioi": ioi, "previous_ioi": previous_ioi, "same_pitch_neighbor": float(same), "aba": float(aba),
            "confidence": float(note.confidence), "onset": onset, "paired_with_missed": float(paired),
            "previous_step": float(previous_step), "next_step": float(next_step),
            "ornament": float(extra.origin == "ornament")}


def extra_vector(row: Mapping[str, float]) -> list[float]:
    ioi = min(float(row["ioi"]), 2.0)
    previous_ioi = min(float(row["previous_ioi"]), 2.0)
    return [float(np.log(max(ioi, 0.005))), float(np.log(max(previous_ioi, 0.005))), row["same_pitch_neighbor"],
            row["aba"], float(np.log(max(1.0 - min(row["confidence"], 0.9999), 1e-4))), row["onset"],
            row["paired_with_missed"], min(row["previous_step"], 12) / 12, min(row["next_step"], 12) / 12,
            row.get("ornament", 0.0)]


def extra_vector_v2(row: Mapping[str, float]) -> list[float]:
    previous_step, next_step = float(row["previous_step"]), float(row["next_step"])
    return extra_vector(row) + [
        float(previous_step == 0), float(previous_step in (1.0, 2.0)), float(previous_step >= 3),
        float(next_step == 0), float(next_step in (1.0, 2.0)), float(next_step >= 3),
        float(row["ioi"] < 0.1), float(row["previous_ioi"] < 0.06),
    ]


EXTRA_VECTORS = {"v1": extra_vector, "v2": extra_vector_v2}


def missed_vector(row: Mapping[str, float]) -> list[float]:
    def finite(value, default):
        return default if value is None or value != value else float(value)

    level = finite(row.get("level_drop_db"), 0.0)
    return [float(np.log(max(row["pitch_probability"], 1e-4))), row["window_voiced"], row["slot_voiced"],
            max(-10.0, min(level, 40.0)) / 10, finite(row.get("slot_onset"), 0.0),
            float(np.log(max(row["expected_duration"], 0.02))), min(row["run_length"], 5), row["edge"],
            row["ornamented"], min(row["nearby_similar"], 4), min(row["nearby_unexplained"], 6),
            min(finite(row.get("gap_ratio"), 1.0), 4.0)]


def missed_vector_v2(row: Mapping[str, float]) -> list[float]:
    distance = float(row.get("neighbor_pitch_distance", 12.0))
    return missed_vector(row) + [
        float(row.get("ornamented_neighbor", 0.0)), float(row.get("in_repeat_span", 0.0)),
        float(distance <= 2), min(distance, 12.0) / 12.0,
    ]


def missed_vector_v3(row: Mapping[str, float]) -> list[float]:
    presence = row.get("presence", float("nan"))
    presence = 0.5 if presence is None or presence != presence else float(presence)
    clipped = min(max(presence, 1e-4), 1 - 1e-4)
    return missed_vector_v2(row) + [presence, float(np.log(clipped / (1 - clipped)))]


MISSED_VECTORS = {"v1": missed_vector, "v2": missed_vector_v2, "v3": missed_vector_v3}


def logistic_score(model: Mapping[str, Any], vector: Sequence[float]) -> float:
    x = (np.asarray(vector, np.float64) - np.asarray(model["mean"])) / np.asarray(model["scale"])
    z = float(x @ np.asarray(model["coef"]) + float(model["intercept"]))
    return 1.0 / (1.0 + np.exp(-z))


def gate_alignment(
    alignment: AlignmentV3,
    score: Sequence[ScoreEvent],
    evidence: Mapping[str, Any] | None,
    config: GateConfig,
) -> tuple[tuple[JointEvent, ...], frozenset[int], dict[str, Any]]:
    """Withhold unsupported extra/missed calls; infer strongly supported missed notes as matches."""

    events = list(alignment.events)
    deletions = set(alignment.deletions)
    info = {"extras_withheld": 0, "missed_withheld": 0, "missed_inferred": 0, "clip_abstained": False}
    if not config.enabled:
        return tuple(events), frozenset(deletions), info
    remove: set[int] = set()
    if alignment.match_fraction < config.clip_min_match_fraction:
        info["clip_abstained"] = True
        for extra in alignment.extras:
            if extra.origin == "inserted":
                remove.add(extra.event_position)
        info["extras_withheld"] = len(remove)
        info["missed_withheld"] = len(deletions)
        deletions = set()
    else:
        for extra in alignment.extras:
            if extra.origin == "repeat_pass":
                continue
            features = extra_features(alignment, extra, evidence)
            if config.extra_model is not None:
                if extra.origin in config.extra_model_origins and logistic_score(
                        config.extra_model, EXTRA_VECTORS[config.extra_features](features)) < config.extra_threshold:
                    remove.add(extra.event_position)
                continue
            if extra.origin != "inserted":
                continue
            threshold = config.extra_same_pitch_min_ioi if features["same_pitch_neighbor"] else config.extra_min_ioi
            if features["ioi"] < threshold or features["confidence"] < config.extra_min_confidence:
                remove.add(extra.event_position)
        info["extras_withheld"] = len(remove)
        inferred: list[JointEvent] = []
        for unit in alignment.missed:
            if unit.score_index not in deletions:
                continue
            if unit.edge and not config.missed_edge_runs and unit.run_length >= 1:
                deletions.discard(unit.score_index)
                info["missed_withheld"] += 1
                continue
            if unit.run_length > config.missed_max_run:
                deletions.discard(unit.score_index)
                info["missed_withheld"] += 1
                continue
            if evidence is None:
                continue
            features = missed_features(alignment, unit, score, evidence)
            if features["pitch_probability"] >= config.missed_infer_pitch_probability and np.isfinite(unit.expected_time):
                deletions.discard(unit.score_index)
                start = unit.expected_time
                inferred.append(JointEvent(
                    pitch=int(score[unit.score_index].pitch), start=float(start),
                    end=float(start) + max(unit.expected_duration, 0.02),
                    score_span=(unit.score_index, unit.score_index + 1), relationship="match",
                    rendered_index=0, confidence=float(features["pitch_probability"]),
                ))
                info["missed_inferred"] += 1
                continue
            if config.missed_never_ornamented and (features["ornamented"] or features.get("ornamented_neighbor")):
                deletions.discard(unit.score_index)
                info["missed_withheld"] += 1
                continue
            presence = features.get("presence", float("nan"))
            if config.missed_max_presence < 1.0 and (not np.isfinite(presence) or presence > config.missed_max_presence):
                deletions.discard(unit.score_index)
                info["missed_withheld"] += 1
                continue
            if config.missed_model is not None:
                vector = MISSED_VECTORS[config.missed_features](features)
                if logistic_score(config.missed_model, vector) < config.missed_threshold:
                    deletions.discard(unit.score_index)
                    info["missed_withheld"] += 1
                continue
            level_ok = config.missed_min_level_drop_db is None or (
                np.isfinite(features["level_drop_db"]) and features["level_drop_db"] >= config.missed_min_level_drop_db)
            onset_ok = not np.isfinite(features["slot_onset"]) or features["slot_onset"] <= config.missed_max_slot_onset
            if (features["pitch_probability"] > config.missed_max_pitch_probability
                    or features["slot_voiced"] > config.missed_max_voiced or not level_ok or not onset_ok):
                deletions.discard(unit.score_index)
                info["missed_withheld"] += 1
        events.extend(inferred)
    kept = [event for position, event in enumerate(events) if position not in remove]
    kept.sort(key=lambda event: (event.start, event.rendered_index if event.rendered_index is not None else 0))
    kept = [replace(event, rendered_index=position) for position, event in enumerate(kept)]
    return tuple(kept), frozenset(deletions), info
