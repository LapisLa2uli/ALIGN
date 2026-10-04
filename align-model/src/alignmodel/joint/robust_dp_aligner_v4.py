"""Stack v7 aligner: v3 operations with bounded local/multiple restart search.

The frozen v3 source is untouched. The alignment body is intentionally
versioned here; shared numerical primitives and evidence gates are reused.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence
import numpy as np
from .index import JointEvent, ScoreEvent
from .robust_dp_aligner_v2 import _as_notes, _dp, _backtrace
from .perfect_dp_aligner_v1 import _mergeable
from .ornament_mapper_v1 import expand_ornament_hypothesis, score_ornament_patterns
from .robust_dp_aligner_v3 import (AlignerV3Config, AlignmentV3, AlignedExtra, MissedUnit,
    GateConfig, gate_alignment, mark_artifacts, _arrays, _assignment, _tempo_map, _dp_timed)
from .restarts_v4 import hypotheses_v4

@dataclass(frozen=True)
class AlignerV4Config(AlignerV3Config):
    max_local_proposals: int = 12

@dataclass(frozen=True)
class AlignmentV4(AlignmentV3):
    restart_regions: tuple[tuple[int,int,int], ...] = ()

def restart_regions(hypothesis):
    explicit=getattr(hypothesis,'restart_regions',())
    if explicit: return explicit
    if hypothesis.source_span and hypothesis.copies:
        return ((*hypothesis.source_span,hypothesis.copies),)
    return ()

def gate_v4(alignment, score, evidence, config=GateConfig(), *, minimum_match_fraction=0.45):
    events,deletions,info=gate_alignment(alignment,score,evidence,config)
    abstain=alignment.match_fraction < minimum_match_fraction
    info={**info,'status':'alignment_uncertain' if abstain else 'ok',
          'clip_abstained':abstain,'match_fraction':alignment.match_fraction,
          'minimum_match_fraction':minimum_match_fraction}
    if abstain:
        # Empty predictions remain false negatives in evaluation. Callers must
        # display status, not interpret this as an error-free performance.
        return (),frozenset(),info
    return events,deletions,info

def align_v4(
    notes: Sequence,
    score: Sequence[ScoreEvent],
    score_path: Path | str,
    config: AlignerV4Config = AlignerV4Config(),
) -> AlignmentV4:
    costs = config.costs
    raw_items = _as_notes(notes)
    items, artifacts = mark_artifacts(raw_items, config)
    score_tuple = tuple(score)
    patterns = score_ornament_patterns(score_path, score_tuple)
    arrays = _arrays(items, costs)
    note_pitch, note_alt, insert_cost, drop_cost, match_extra, alt_cost, cost_vector = arrays
    primary_count = sum(not note.optional for note in items)
    hypotheses = hypotheses_v4(score_tuple, items, primary_count, config.max_local_proposals)
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
    return AlignmentV4(
        restart_regions=restart_regions(hypothesis),
        events=tuple(events), deletions=deletions, cost=float(total),
        source_span=hypothesis.source_span, copies=int(hypothesis.copies),
        kept_note_indices=tuple(kept), extras=tuple(extras), missed=tuple(missed),
        notes=tuple(items), artifact_notes=artifacts, seconds_per_ql=float(spq),
        match_fraction=sum(1 for e in events if e.relationship in ("match", "copy")) / max(len(events), 1),
    )

