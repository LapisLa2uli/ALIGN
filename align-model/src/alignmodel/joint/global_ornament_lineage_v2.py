"""Authoritative global temporal reconstruction for rendered ornament lineage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .grammar_mapper_v2 import GrammarHypothesis
from .index import ScoreEventIndex
from .ornament_mapper_v1 import (
    OrnamentTemplateUnit,
    expand_ornament_hypothesis,
    score_ornament_patterns,
)


SCHEMA_VERSION = "align-global-ornament-lineage-v2"


@dataclass(frozen=True)
class ReconstructionResult:
    lineage: dict[str, Any]
    stats: dict[str, Any]


def _base_hypothesis(index: ScoreEventIndex) -> GrammarHypothesis:
    return GrammarHypothesis(
        source_span=None,
        copies=0,
        units=tuple((event.index, 0) for event in index.events),
        times=tuple(float(event.ql_start) for event in index.events),
    )


def _align_exact(
    raw_pitch: Sequence[int],
    template_pitch: Sequence[int],
) -> tuple[list[tuple[int, int]], list[int], list[int], int]:
    """Global minimum edit path with substitutions prohibited."""

    rows, columns = len(raw_pitch), len(template_pitch)
    cost = [[0] * (columns + 1) for _ in range(rows + 1)]
    back = [[0] * (columns + 1) for _ in range(rows + 1)]
    for row in range(1, rows + 1):
        cost[row][0] = row
        back[row][0] = 2
    for column in range(1, columns + 1):
        cost[0][column] = column
        back[0][column] = 3
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            diagonal = cost[row - 1][column - 1] + (
                0
                if raw_pitch[row - 1] == template_pitch[column - 1]
                else 3
            )
            raw_extra = cost[row - 1][column] + 1
            template_missing = cost[row][column - 1] + 1
            best = min(diagonal, raw_extra, template_missing)
            cost[row][column] = best
            back[row][column] = (
                1
                if diagonal <= raw_extra and diagonal <= template_missing
                else 2
                if raw_extra <= template_missing
                else 3
            )
    pairs = []
    raw_unmatched = []
    template_unmatched = []
    row, column = rows, columns
    while row or column:
        action = back[row][column]
        if action == 1:
            if raw_pitch[row - 1] == template_pitch[column - 1]:
                pairs.append((row - 1, column - 1))
            else:
                raw_unmatched.append(row - 1)
                template_unmatched.append(column - 1)
            row -= 1
            column -= 1
        elif action == 2:
            raw_unmatched.append(row - 1)
            row -= 1
        elif action == 3:
            template_unmatched.append(column - 1)
            column -= 1
        else:
            raise RuntimeError("Global ornament lineage backtrace failed")
    return (
        list(reversed(pairs)),
        list(reversed(raw_unmatched)),
        list(reversed(template_unmatched)),
        cost[rows][columns],
    )


def _performed_group_payload(
    performed_rows: Sequence[Mapping[str, Any]],
    source_indices: Sequence[int],
) -> tuple[list[int], list[int], str, int, str]:
    indices = [int(value) for value in source_indices]
    selected = [performed_rows[index] for index in indices]
    clean = list(
        dict.fromkeys(
            int(row["clean_index"])
            for row in selected
            if row.get("clean_index") is not None
        )
    )
    relationships = [str(row.get("relationship") or "extra") for row in selected]
    copy_pass = max((int(row.get("copy_pass") or 0) for row in selected), default=0)
    relationship = (
        "copy"
        if copy_pass or "copy" in relationships
        else relationships[0]
        if relationships
        else "extra"
    )
    planted = any(
        row.get("clean_index") is None
        and str(row.get("origin_relationship") or row.get("relationship"))
        == "extra"
        for row in selected
    )
    origin = (
        "planted_extra_copy"
        if planted and relationship == "copy"
        else "planted_extra"
        if planted
        else "generator_performed_lineage"
    )
    return indices, clean, relationship, copy_pass, origin


def reconstruct_global_ornament_lineage(
    original_lineage: Mapping[str, Any],
    performance_score_path: Path | str,
    verified_score_path: Path | str,
    raw_midi: Mapping[str, Any],
    *,
    written_shift: int,
) -> ReconstructionResult:
    """Reconstruct exact audible events without prediction-derived evidence."""

    performed_rows = list(original_lineage.get("performed_notes") or [])
    performance_index = ScoreEventIndex.from_musicxml(performance_score_path)
    source_count = len(performance_index.source_to_event)
    if source_count != len(performed_rows):
        raise ValueError(
            "Performance MusicXML/non-grace lineage count mismatch: "
            f"xml={source_count} lineage={len(performed_rows)}"
        )
    for source_index, row in enumerate(performed_rows):
        if (
            int(row.get("performed_index", -1)) != source_index
            or int(row.get("pitch_midi", -999))
            != performance_index.events[
                performance_index.event_for_source(source_index)
            ].pitch
        ):
            raise ValueError(
                f"Performance lineage identity mismatch at {source_index}"
            )
    patterns = score_ornament_patterns(
        performance_score_path, performance_index.events
    )
    template = expand_ornament_hypothesis(
        performance_index.events,
        patterns,
        _base_hypothesis(performance_index),
    )
    raw_notes = list(raw_midi.get("notes") or [])
    raw_pitch = [int(row["pitch"]) + int(written_shift) for row in raw_notes]
    template_pitch = [int(unit.pitch) for unit in template]
    pairs, raw_unmatched, template_unmatched, edit_cost = _align_exact(
        raw_pitch, template_pitch
    )
    pair_by_raw = {raw: expected for raw, expected in pairs}
    linked_template = {
        index for index, unit in enumerate(template) if unit.kind == "linked"
    }
    unmatched_linked = sorted(linked_template & set(template_unmatched))
    unmatched_ornament = sorted(
        set(template_unmatched) - linked_template
    )
    if unmatched_linked:
        raise ValueError(
            f"{len(unmatched_linked)} performed/tie units have no raw MIDI event"
        )
    if raw_unmatched or unmatched_ornament:
        raise ValueError(
            "Generator ornament template does not exactly round-trip raw MIDI: "
            f"raw_unmatched={len(raw_unmatched)} "
            f"ornament_unmatched={len(unmatched_ornament)}"
        )
    rendered = []
    mapped_performed = set()
    planted_extra_events = 0
    renderer_ornament_events = 0
    for rendered_index, actual in enumerate(raw_notes):
        template_index = pair_by_raw[rendered_index]
        unit: OrnamentTemplateUnit = template[template_index]
        if unit.kind == "ornament_extra":
            performed_indices: list[int] = []
            clean_indices: list[int] = []
            relationship = "extra"
            copy_pass = 0
            origin = "renderer_ornament"
            renderer_ornament_events += 1
        else:
            assert unit.score_index is not None
            performance_event = performance_index.events[unit.score_index]
            (
                performed_indices,
                clean_indices,
                relationship,
                copy_pass,
                origin,
            ) = _performed_group_payload(
                performed_rows, performance_event.source_indices
            )
            mapped_performed.update(performed_indices)
            planted_extra_events += int(origin.startswith("planted_extra"))
        rendered.append(
            {
                "rendered_index": rendered_index,
                "pitch_midi_sounding": int(actual["pitch"]),
                "pitch_midi_written": raw_pitch[rendered_index],
                "start_sec": round(float(actual["start"]), 9),
                "end_sec": round(
                    max(float(actual["end"]), float(actual["start"]) + 0.001),
                    9,
                ),
                "performed_indices": performed_indices,
                "clean_indices": clean_indices,
                "primary_clean_index": (
                    clean_indices[0] if clean_indices else None
                ),
                "relationship": relationship,
                "copy_pass": copy_pass,
                "extra_origin": origin,
                "ornament_kind": (
                    unit.ornament_kind
                    if unit.kind == "ornament_extra"
                    else None
                ),
                "template_index": template_index,
            }
        )
    missing_performed = sorted(
        set(range(len(performed_rows))) - mapped_performed
    )
    if missing_performed:
        raise ValueError(
            f"{len(missing_performed)} performed source identities are unmapped"
        )
    rebuilt = dict(original_lineage)
    rebuilt["schema_version"] = "2.0"
    rebuilt["representation_version"] = SCHEMA_VERSION
    rebuilt["rendered_notes"] = rendered
    rebuilt["rendered_note_count"] = len(rendered)
    rebuilt["render_repair"] = {
        "policy": SCHEMA_VERSION,
        "raw_midi_events": len(raw_notes),
        "template_units": len(template),
        "edit_cost": edit_cost,
        "raw_unmatched": len(raw_unmatched),
        "template_unmatched": len(template_unmatched),
        "mapped_performed_notes": len(mapped_performed),
        "unmapped_performed_notes": len(missing_performed),
        "planted_extra_events": planted_extra_events,
        "renderer_ornament_events": renderer_ornament_events,
    }
    verified_index = ScoreEventIndex.from_musicxml(
        verified_score_path, rebuilt
    )
    backward = 0
    previous_by_pass: dict[int, int] = {}
    for event in verified_index.rendered_events:
        if event.score_span is None:
            continue
        copy_pass = int(event.copy_pass)
        previous = previous_by_pass.get(copy_pass)
        if previous is not None and event.score_span[0] < previous:
            backward += 1
        previous_by_pass[copy_pass] = event.score_span[0]
    if backward:
        raise ValueError(
            f"Reconstructed canonical identity moves backward {backward} times"
        )
    starts = [float(row["start_sec"]) for row in rendered]
    if starts != sorted(starts):
        raise ValueError("Rendered events are not in raw MIDI temporal order")
    return ReconstructionResult(
        lineage=rebuilt,
        stats={
            **rebuilt["render_repair"],
            "canonical_score_events": len(verified_index.events),
            "canonical_backward_steps": backward,
            "unique_rendered_identity": [
                row["rendered_index"] for row in rendered
            ]
            == list(range(len(rendered))),
            "temporal_order": "raw_midi_start_pitch_end",
        },
    )
