from __future__ import annotations

import argparse
import atexit
import gzip
import hashlib
import json
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import audit_training_data as audit
from music21 import converter, note, stream
from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.outputraw_full import verify_data_ready
from alignmodel.training_resources import resource_lease


SCHEMA = "align-renderer-provenance-v7"
TIME_EPSILON = 1e-7
LEGAL_PASS_TRANSITIONS = {
    (0, 0),
    (0, 1),
    (1, 1),
    (1, 2),
    (2, 2),
    (1, 0),
    (2, 0),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _identity(row: dict[str, Any]) -> tuple[str, str]:
    clean = row.get("clean_index")
    if clean is not None:
        return ("clean", str(int(clean)))
    # Schema 1.0 did not serialize the generator's private extra-note ID.
    # Replay blocks also contain unique clean IDs, so this stable signature is
    # sufficient for source-block recovery; ambiguous blocks fail closed below.
    return (
        "extra_signature",
        f"{int(row['pitch_midi'])}:{float(row['duration_ql']):.9f}",
    )


def _temporal_groups(
    rows: list[dict[str, Any]], field: str
) -> tuple[list[int], list[list[int]]]:
    groups: list[list[int]] = []
    for index, row in enumerate(rows):
        value = float(row[field])
        if not groups or abs(value - float(rows[groups[-1][0]][field])) > TIME_EPSILON:
            groups.append([index])
        else:
            groups[-1].append(index)
    assignment = [0] * len(rows)
    for group_index, group in enumerate(groups):
        for index in group:
            assignment[index] = group_index
    return assignment, groups


def _xml_counts(path: Path) -> Counter[str]:
    root = ET.parse(path).getroot()
    counts: Counter[str] = Counter()
    for element in root.iter():
        name = element.tag.rsplit("}", 1)[-1]
        if name in {
            "note",
            "chord",
            "grace",
            "tie",
            "tied",
            "ornaments",
            "trill-mark",
            "mordent",
            "inverted-mordent",
            "turn",
            "voice",
        }:
            counts[name] += 1
    voices = {
        (element.text or "").strip()
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1] == "voice"
        and (element.text or "").strip()
    }
    counts["distinct_voice_ids"] = len(voices)
    return counts


def _performed_voice_owners(
    path: Path, performed: list[dict[str, Any]]
) -> list[str]:
    parsed = converter.parse(str(path))
    notes = [
        item
        for item in parsed.recurse().getElementsByClass(note.Note)
        if not item.duration.isGrace
    ]
    actual = []
    for source_index, item in enumerate(notes):
        onset = float(item.getOffsetInHierarchy(parsed))
        voice = item.getContextByClass(stream.Voice)
        part = item.getContextByClass(stream.Part)
        actual.append(
            (
                onset,
                int(item.pitch.midi),
                source_index,
                f"{getattr(part, 'id', None) or 'part'}:"
                f"{getattr(voice, 'id', None) or 'default'}",
            )
        )
    actual.sort(key=lambda value: (value[0], value[1], value[2]))
    expected = sorted(
        (
            float(row["onset_ql"]),
            int(row["pitch_midi"]),
            int(row["performed_index"]),
        )
        for row in performed
    )
    if len(actual) != len(expected) or any(
        abs(left[0] - right[0]) > TIME_EPSILON or left[1] != right[1]
        for left, right in zip(actual, expected)
    ):
        raise ValueError("Performance MusicXML voice ownership cannot be projected")
    return [row[3] for row in actual]


def _first_example(
    examples: dict[str, Any], name: str, payload: dict[str, Any]
) -> None:
    examples.setdefault(name, payload)


def _event_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    span = row.get("score_span")
    if span is None:
        location = row.get("canonical_location") or {}
        if location.get("kind") == "score_span":
            span = location.get("score_span")
    return (
        tuple(span) if span is not None else None,
        str(row["relationship"]),
        int(row["copy_pass"]),
        tuple(int(value) for value in row.get("source_indices") or []),
    )


def _validate_replay(
    sample: str, performed: list[dict[str, Any]]
) -> dict[str, Any]:
    seen: Counter[tuple[str, str]] = Counter()
    identities: list[tuple[str, str]] = []
    passes: list[int] = []
    for position, row in enumerate(performed):
        if int(row["performed_index"]) != position:
            raise ValueError(f"{sample}: performed identities are not contiguous")
        identity = _identity(row)
        copy_pass = int(row.get("copy_pass") or 0)
        expected_pass = seen[identity]
        exact_identity = row.get("clean_index") is not None
        if exact_identity and copy_pass != expected_pass:
            raise ValueError(
                f"{sample}: identity {identity} pass {copy_pass}, expected {expected_pass}"
            )
        relationship = str(row.get("relationship") or "extra")
        origin = str(row.get("origin_relationship") or relationship)
        if relationship != ("copy" if copy_pass else origin):
            raise ValueError(
                f"{sample}: illegal relationship/pass at performed {position}"
            )
        if exact_identity:
            seen[identity] += 1
        identities.append(identity)
        passes.append(copy_pass)

    transitions = Counter(zip(passes, passes[1:]))
    illegal = sorted(
        (left, right)
        for left, right in transitions
        if (left, right) not in LEGAL_PASS_TRANSITIONS
    )
    if illegal:
        raise ValueError(f"{sample}: illegal replay pass transitions {illegal}")

    first_pass = [
        identity for identity, copy_pass in zip(identities, passes) if copy_pass == 0
    ]
    segments: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(performed):
        copy_pass = passes[cursor]
        if copy_pass == 0:
            cursor += 1
            continue
        end = cursor + 1
        while end < len(performed) and passes[end] == copy_pass:
            end += 1
        copied = identities[cursor:end]
        matches = [
            start
            for start in range(len(first_pass) - len(copied) + 1)
            if first_pass[start : start + len(copied)] == copied
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{sample}: copy pass {copy_pass} has {len(matches)} source blocks"
            )
        source_start = matches[0]
        source_end = source_start + len(copied)
        resume = None
        if end < len(performed) and passes[end] == 0:
            resume = identities[end]
            expected = first_pass[source_end] if source_end < len(first_pass) else None
            if resume != expected:
                raise ValueError(
                    f"{sample}: replay resumes at {resume}, expected {expected}"
                )
        segments.append(
            {
                "copy_pass": copy_pass,
                "performed_span": [cursor, end],
                "source_first_pass_span": [source_start, source_end],
                "resume_identity": list(resume) if resume is not None else None,
            }
        )
        cursor = end
    return {
        "pass_transitions": transitions,
        "replay_segments": segments,
        "maximum_copy_pass": max(passes, default=0),
    }


def _reconstructed_lineage(
    note_map: dict[str, Any],
    midi_rows: list[dict[str, Any]],
    shift: int,
) -> dict[str, Any]:
    performed = list(note_map["performed_notes"])
    rendered = []
    for index, (actual, source) in enumerate(zip(midi_rows, performed)):
        clean = source.get("clean_index")
        rendered.append(
            {
                "rendered_index": index,
                "pitch_midi_sounding": int(actual["pitch"]),
                "pitch_midi_written": int(actual["pitch"]) + shift,
                "start_sec": round(float(actual["start"]), 9),
                "end_sec": round(
                    max(float(actual["end"]), float(actual["start"]) + 0.001), 9
                ),
                "performed_indices": [index],
                "clean_indices": [int(clean)] if clean is not None else [],
                "primary_clean_index": int(clean) if clean is not None else None,
                "relationship": str(source.get("relationship") or "extra"),
            }
        )
    rebuilt = dict(note_map)
    rebuilt["rendered_notes"] = rendered
    rebuilt["rendered_note_count"] = len(rendered)
    return rebuilt


def _score_part_valid(label: dict[str, Any], clean_count: int) -> bool:
    part = label.get("score_part")
    if part is None:
        return True
    try:
        start = int(part["start_note_index"])
        end = int(part["end_note_index"])
    except (KeyError, TypeError, ValueError):
        return False
    if start < 0 or end < start or end >= clean_count:
        return False
    note_ids = list(label.get("note_ids") or [])
    pitches = list(label.get("pitches") or [])
    expected = end - start + 1
    return (not note_ids or len(note_ids) == expected) and (
        not pitches or len(pitches) == expected
    )


def _audit_row(
    manifest_row: dict[str, Any],
    corrected: dict[str, Any],
    old: dict[str, Any] | None,
    examples: dict[str, Any],
) -> tuple[dict[str, Any], Counter[str], Counter[str]]:
    sample = str(manifest_row["sample"])
    sample_dir = Path(manifest_row["sample_dir"])
    for name, expected in manifest_row["source_hashes"].items():
        path = sample_dir / name
        if _sha256(path) != expected:
            raise ValueError(f"{sample}: source hash changed for {name}")
    note_map = json.loads(
        (sample_dir / "note_map.json").read_text(encoding="utf-8")
    )
    labels = json.loads(
        (sample_dir / "labels.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (sample_dir / "metadata.json").read_text(encoding="utf-8")
    )
    midi = audit._midi_events(sample_dir / "performance_audio.mid")
    if midi is None:
        raise ValueError(f"{sample}: raw performance MIDI is unreadable")
    midi_rows = list(midi["notes"])
    performed = list(note_map.get("performed_notes") or [])
    clean = list(note_map.get("clean_notes") or [])
    events = list(corrected.get("events") or [])
    shift = audit._inferred_midi_shift(note_map, midi, metadata)

    if not (len(midi_rows) == len(performed) == len(events)):
        raise ValueError(
            f"{sample}: raw/performed/corrected counts differ "
            f"{len(midi_rows)}/{len(performed)}/{len(events)}"
        )
    if any(
        int(actual["pitch"]) + shift != int(source["pitch_midi"])
        for actual, source in zip(midi_rows, performed)
    ):
        raise ValueError(f"{sample}: raw MIDI is not exact-pitch one-to-one")
    if any(
        float(right["start"]) + TIME_EPSILON < float(left["start"])
        for left, right in zip(midi_rows, midi_rows[1:])
    ):
        raise ValueError(f"{sample}: raw MIDI event list is not temporal")
    if any(
        float(right["onset_ql"]) + TIME_EPSILON < float(left["onset_ql"])
        for left, right in zip(performed, performed[1:])
    ):
        raise ValueError(f"{sample}: generator performed list is not temporal")

    replay = _validate_replay(sample, performed)
    rebuilt = _reconstructed_lineage(note_map, midi_rows, shift)
    score_index = ScoreEventIndex.from_musicxml(
        sample_dir / "verified_score.musicxml", rebuilt
    )
    projected = list(score_index.rendered_events)
    if len(projected) != len(events):
        raise ValueError(f"{sample}: round-trip rendered count changed")

    group_ids, groups = _temporal_groups(midi_rows, "start")
    simultaneous = [group for group in groups if len(group) > 1]
    maximum_active = 0
    overlap_events = 0
    overlapping_pairs: list[tuple[int, int]] = []
    active_indices: list[int] = []
    active_ends: list[float] = []
    for index, actual in enumerate(midi_rows):
        start = float(actual["start"])
        still_active = [
            (active_index, end)
            for active_index, end in zip(active_indices, active_ends)
            if end > start + TIME_EPSILON
        ]
        active_indices = [value[0] for value in still_active]
        active_ends = [value[1] for value in still_active]
        if active_ends:
            overlap_events += 1
            overlapping_pairs.extend(
                (active_index, index) for active_index in active_indices
            )
        active_indices.append(index)
        active_ends.append(float(actual["end"]))
        maximum_active = max(maximum_active, len(active_ends))

    xml = _xml_counts(sample_dir / "performance_score.musicxml")
    needs_voice_projection = bool(
        simultaneous or overlap_events or xml["distinct_voice_ids"] > 1
    )
    voice_owners = (
        _performed_voice_owners(
            sample_dir / "performance_score.musicxml", performed
        )
        if needs_voice_projection
        else ["clarinet:default"] * len(performed)
    )
    concurrent_owner_pairs = {
        tuple(sorted((voice_owners[left], voice_owners[right])))
        for left, right in overlapping_pairs
        if voice_owners[left] != voice_owners[right]
    }
    true_overlapping_strands = bool(concurrent_owner_pairs)

    output_events = []
    for index, (actual, source, expected, round_trip) in enumerate(
        zip(midi_rows, performed, events, projected)
    ):
        expected_signature = (
            tuple(expected["score_span"]) if expected["score_span"] is not None else None,
            str(expected["relationship"]),
            int(expected["copy_pass"]),
            tuple(int(value) for value in expected.get("source_indices") or []),
        )
        projected_signature = (
            round_trip.score_span,
            round_trip.relationship,
            round_trip.copy_pass,
            round_trip.source_indices,
        )
        if expected_signature != projected_signature:
            raise ValueError(
                f"{sample}: canonical round-trip differs at rendered {index}"
            )
        if int(expected["rendered_event_id"]) != index:
            raise ValueError(f"{sample}: rendered identities are not contiguous")
        if abs(float(expected["start_sec"]) - float(actual["start"])) > 1e-8:
            raise ValueError(f"{sample}: corrected onset differs from raw MIDI")
        if abs(float(expected["end_sec"]) - float(actual["end"])) > 1e-8:
            raise ValueError(f"{sample}: corrected offset differs from raw MIDI")
        output_events.append(
            {
                "rendered_event_id": index,
                "performed_index": index,
                "strand_id": voice_owners[index],
                "temporal_group": group_ids[index],
                "simultaneous_group_size": len(groups[group_ids[index]]),
                "pitch_sounding": int(actual["pitch"]),
                "pitch_written": int(actual["pitch"]) + shift,
                "start_sec": round(float(actual["start"]), 9),
                "end_sec": round(float(actual["end"]), 9),
                "generator_onset_ql": float(source["onset_ql"]),
                "generator_duration_ql": float(source["duration_ql"]),
                "generator_measure": source.get("measure"),
                "generator_identity": list(_identity(source)),
                "score_span": expected["score_span"],
                "source_indices": expected.get("source_indices") or [],
                "relationship": expected["relationship"],
                "origin_relationship": expected.get("origin_relationship"),
                "copy_pass": int(expected["copy_pass"]),
            }
        )

    counts: Counter[str] = Counter(
        {
            "raw_midi_events": len(midi_rows),
            "generator_performed_events": len(performed),
            "corrected_events": len(events),
            "simultaneous_groups": len(simultaneous),
            "simultaneous_events": sum(len(group) for group in simultaneous),
            "interval_overlap_events": overlap_events,
            "tie_aware_canonical_events": sum(
                len(event.source_indices) > 1 for event in score_index.events
            ),
            "canonical_source_notes": len(score_index.source_to_event),
            "canonical_events": len(score_index.events),
            "replay_segments": len(replay["replay_segments"]),
            "rows_with_true_overlapping_strands": true_overlapping_strands,
        }
    )
    for (left, right), value in replay["pass_transitions"].items():
        counts[f"corrected_pass_{left}_to_{right}"] += value

    category: Counter[str] = Counter(
        {
            "performance_xml_chord_followers": xml["chord"],
            "performance_xml_grace_notes": xml["grace"],
            "performance_xml_tie_marks": xml["tie"] + xml["tied"],
            "performance_xml_ornament_containers": xml["ornaments"],
            "performance_xml_distinct_voice_ids": xml["distinct_voice_ids"],
        }
    )
    if simultaneous:
        group = simultaneous[0]
        _first_example(
            examples,
            "simultaneous_voices_chords_or_ornaments",
            {
                "sample": sample,
                "rendered_indices": group,
                "pitches": [int(midi_rows[index]["pitch"]) for index in group],
                "start_sec": float(midi_rows[group[0]]["start"]),
            },
        )
    if overlap_events:
        _first_example(
            examples,
            "single_stream_interval_overlap",
            {"sample": sample, "maximum_active_events": maximum_active},
        )
    if true_overlapping_strands:
        _first_example(
            examples,
            "true_overlapping_strands",
            {
                "sample": sample,
                "owner_pairs": [
                    list(value) for value in sorted(concurrent_owner_pairs)
                ],
                "maximum_active_events": maximum_active,
            },
        )
    if any(xml[name] for name in ("chord", "grace", "tie", "tied", "ornaments")):
        _first_example(
            examples,
            "notated_chord_ornament_or_tie",
            {"sample": sample, "xml_counts": dict(xml)},
        )
    tied_events = [
        event.index for event in score_index.events if len(event.source_indices) > 1
    ]
    if tied_events:
        _first_example(
            examples,
            "canonical_ties",
            {
                "sample": sample,
                "canonical_event_indices": tied_events[:8],
                "count": len(tied_events),
            },
        )

    original = list(note_map.get("rendered_notes") or [])
    original_inversions = sum(
        float(right.get("start_sec", 0.0)) + TIME_EPSILON
        < float(left.get("start_sec", 0.0))
        for left, right in zip(original, original[1:])
    )
    counts["original_rendered_events"] += len(original)
    counts["original_rendered_temporal_inversions"] += original_inversions
    if original_inversions:
        _first_example(
            examples,
            "unsorted_original_rendered_list",
            {"sample": sample, "inversions": original_inversions},
        )

    label_rows = list(labels.get("labels") or [])
    label_inversions = sum(
        float(right["start_time"]) + TIME_EPSILON < float(left["start_time"])
        for left, right in zip(label_rows, label_rows[1:])
    )
    invalid_score_parts = sum(
        not _score_part_valid(label, len(clean)) for label in label_rows
    )
    counts["labels"] += len(label_rows)
    counts["label_list_temporal_inversions"] += label_inversions
    counts["invalid_label_score_parts"] += invalid_score_parts
    if label_inversions:
        _first_example(
            examples,
            "unsorted_annotation_label_list",
            {
                "sample": sample,
                "inversions": label_inversions,
                "label_order": [
                    {
                        "id": label.get("id"),
                        "start_time": label.get("start_time"),
                        "type": label.get("type"),
                    }
                    for label in label_rows[:8]
                ],
            },
        )
    if invalid_score_parts:
        raise ValueError(f"{sample}: {invalid_score_parts} invalid label score spans")

    packed_mismatches = relation_mismatches = 0
    old_illegal_pass = old_backward = old_wide_spans = 0
    if old is not None:
        old_events = list(old.get("events") or [])
        if len(old_events) != len(events):
            raise ValueError(f"{sample}: v2 packed event count changed")
        previous_by_pass: dict[int, int] = {}
        old_passes = []
        for old_event, event in zip(old_events, events):
            if _event_signature(old_event) != _event_signature(event):
                packed_mismatches += 1
            if (
                str(old_event["relationship"]) != str(event["relationship"])
                or int(old_event["copy_pass"]) != int(event["copy_pass"])
            ):
                relation_mismatches += 1
            span = _event_signature(old_event)[0]
            copy_pass = int(old_event["copy_pass"])
            old_passes.append(copy_pass)
            if span is not None:
                position = int(span[1]) - 1
                if (
                    copy_pass in previous_by_pass
                    and position < previous_by_pass[copy_pass]
                ):
                    old_backward += 1
                previous_by_pass[copy_pass] = position
                old_wide_spans += int(span[1]) - int(span[0]) > 1
        old_illegal_pass = sum(
            (left, right) not in LEGAL_PASS_TRANSITIONS
            for left, right in zip(old_passes, old_passes[1:])
        )
        counts["packed_v2_identity_mismatches"] += packed_mismatches
        counts["packed_v2_relationship_or_pass_mismatches"] += relation_mismatches
        counts["packed_v2_backward_within_pass"] += old_backward
        counts["packed_v2_illegal_pass_transitions"] += old_illegal_pass
        counts["packed_v2_wide_spans"] += old_wide_spans
        if packed_mismatches:
            _first_example(
                examples,
                "identity_transfer_error",
                {
                    "sample": sample,
                    "mismatched_rendered_events": packed_mismatches,
                    "relationship_or_pass_mismatches": relation_mismatches,
                    "backward_within_pass": old_backward,
                    "illegal_pass_transitions": old_illegal_pass,
                },
            )
            _first_example(
                examples,
                "source_index_ordering",
                {
                    "sample": sample,
                    "finding": (
                        "v2 packed raw-time rows referenced generator source "
                        "identities in a different order"
                    ),
                    "mismatched_rendered_events": packed_mismatches,
                },
            )

    if len({int(row["channel"]) for row in midi_rows}) > 1:
        _first_example(
            examples,
            "multiple_midi_channels",
            {
                "sample": sample,
                "channels": sorted({int(row["channel"]) for row in midi_rows}),
            },
        )
    counts["rows_with_multiple_midi_channels"] += (
        len({int(row["channel"]) for row in midi_rows}) > 1
    )
    counts["rows_with_simultaneity"] += bool(simultaneous)
    counts["rows_with_interval_overlap"] += overlap_events > 0
    counts["rows_with_replay"] += bool(replay["replay_segments"])
    counts["rows_with_identity_transfer_errors"] += packed_mismatches > 0
    counts["rows_with_unsorted_labels"] += label_inversions > 0
    counts["rows_with_unsorted_performed_source"] += 0

    row_document = {
        "schema_version": SCHEMA,
        "sample": sample,
        "source_hashes": manifest_row["source_hashes"],
        "ordering_semantics": {
            "performed_notes": "generator in-memory single-part sounding-note order",
            "raw_midi": "actual rendered note-on order, then pitch/end tie-break",
            "original_rendered_notes": "historical renderer attachment cache",
            "packed_v2_events": "raw-MIDI-authoritative v1 heuristic repair; rejected",
            "events": "actual rendered temporal order with generator identity",
            "labels": "annotation creation/pass order; not a temporal event sequence",
        },
        "ownership": {
            "instrument": "clarinet",
            "provenance": "performance MusicXML part/voice",
            "strand_count": len(set(voice_owners)),
            "true_overlapping_strands": true_overlapping_strands,
            "concurrent_owner_pairs": [
                list(value) for value in sorted(concurrent_owner_pairs)
            ],
            "multiple_midi_channels": len(
                {int(row["channel"]) for row in midi_rows}
            )
            > 1,
            "maximum_simultaneous_or_overlapping_events": maximum_active,
        },
        "replay_segments": replay["replay_segments"],
        "deleted_score_events": sorted(score_index.deleted_event_indices),
        "events": output_events,
    }
    return row_document, counts, category


def _iter_gzip(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _write_report(
    output: Path,
    *,
    split: str,
    admitted: int,
    excluded: list[dict[str, str]],
    totals: Counter[str],
    categories: Counter[str],
    examples: dict[str, Any],
    source_hashes: dict[str, str],
    train_supervision: Path | None,
) -> Path:
    expected_classes = (
        "unsorted_original_rendered_list",
        "unsorted_annotation_label_list",
        "source_index_ordering",
        "simultaneous_voices_chords_or_ornaments",
        "notated_chord_ornament_or_tie",
        "canonical_ties",
        "single_stream_interval_overlap",
        "identity_transfer_error",
        "multiple_midi_channels",
        "true_overlapping_strands",
    )
    complete_examples = {
        name: examples.get(
            name,
            {
                "representative": None,
                "finding": "No admitted train example observed"
                if split == "train"
                else "No admitted validation example observed",
            },
        )
        for name in expected_classes
    }
    genuine_strands = int(totals["rows_with_true_overlapping_strands"]) > 0
    document: dict[str, Any] = {
        "schema_version": f"{SCHEMA}-audit-report",
        "split": split,
        "rules_frozen_from_train": True,
        "admitted": admitted,
        "excluded": excluded,
        "totals": dict(totals),
        "source_notation_counts": dict(categories),
        "representative_examples": complete_examples,
        "conclusion": {
            "ownership": (
                "multiple overlapping MusicXML part/voice owners"
                if genuine_strands
                else "single non-overlapping clarinet owner"
            ),
            "apparent_concurrent_strands": (
                "confirmed from renderer overlap and MusicXML voice ownership"
                if genuine_strands
                else "rejected"
            ),
            "cause_of_v2_pass_flips": (
                "identity transfer accepted stale performed indices behind the "
                "performed cursor"
            ),
            "simultaneity_policy": (
                "equal raw MIDI onsets share a temporal group; ordering inside "
                "that group is explicit and does not create another strand"
            ),
            "labels_ordering": (
                "annotation creation/pass order, not renderer temporal order"
            ),
        },
        "validators": {
            "source_hashes": "exact",
            "raw_to_generator_identity": "exact pitch, one-to-one, zero edits",
            "temporal_monotonicity": "nondecreasing modulo temporal_group",
            "rendered_identity": "exclusive contiguous zero-based",
            "canonical_span": "independently reparsed tie-aware ScoreEventIndex",
            "replay": "identity occurrence, pass transition, source block, resume",
            "round_trip": "raw MIDI + generator lineage -> canonical events exact",
            "fail_closed": True,
        },
        "source_hashes": source_hashes,
        "locked_test_touched": False,
        "production_mutated": False,
    }
    if train_supervision is not None:
        document["train_supervision"] = str(train_supervision.resolve())
        document["train_supervision_sha256"] = _sha256(train_supervision)
    report = output / f"{split}-audit-report.json"
    _atomic_json(report, document)
    return report


def _lease(args: argparse.Namespace, split: str):
    lease = resource_lease(
        args.resource_status,
        "cpu_validation",
        track=f"renderer-provenance-v7-{split}",
        command=[str(value) for value in __import__("sys").argv],
        metadata={"split": split, "locked_test": False},
    )
    lease.__enter__()
    atexit.register(lease.__exit__, None, None, None)
    return lease


def build_train(args: argparse.Namespace) -> None:
    lease = _lease(args, "train")
    ready = verify_data_ready(args.ready_marker)
    manifest_path = Path(str(ready["paths"]["manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    corrected_rows = _iter_gzip(args.corrected_supervision)
    old_rows = _iter_gzip(args.old_supervision)
    destination = output / "train-renderer-provenance.jsonl.gz"
    totals: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    examples: dict[str, Any] = {}
    excluded: list[dict[str, str]] = []
    admitted = 0
    with tempfile.NamedTemporaryFile(
        dir=output, prefix=".train-renderer-provenance.", suffix=".tmp", delete=False
    ) as raw:
        temporary = Path(raw.name)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            for manifest_row in manifest["train"]:
                sample = str(manifest_row["sample"])
                try:
                    corrected = next(corrected_rows)
                    old = next(old_rows)
                    if corrected["sample"] != sample or old["sample"] != sample:
                        raise ValueError(f"{sample}: supervision order mismatch")
                    document, row_counts, row_categories = _audit_row(
                        manifest_row, corrected, old, examples
                    )
                    stream.write(
                        json.dumps(document, sort_keys=True, separators=(",", ":"))
                        + "\n"
                    )
                    totals.update(row_counts)
                    categories.update(row_categories)
                    admitted += 1
                except Exception as error:
                    excluded.append({"sample": sample, "reason": str(error)})
            try:
                next(corrected_rows)
                raise ValueError("Corrected supervision has trailing rows")
            except StopIteration:
                pass
            try:
                next(old_rows)
                raise ValueError("Old supervision has trailing rows")
            except StopIteration:
                pass
        if excluded:
            raise RuntimeError(
                f"Fail-closed train audit excluded {len(excluded)} row(s): "
                f"{excluded[:3]}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    source_hashes = {
        "manifest": _sha256(manifest_path),
        "corrected_supervision_v6": _sha256(args.corrected_supervision),
        "packed_derived_supervision_v2": _sha256(args.old_supervision),
        "audit_code": _sha256(Path(__file__)),
    }
    report = _write_report(
        output,
        split="train",
        admitted=admitted,
        excluded=excluded,
        totals=totals,
        categories=categories,
        examples=examples,
        source_hashes=source_hashes,
        train_supervision=destination,
    )
    freeze = {
        "schema_version": f"{SCHEMA}-frozen-rules",
        "rules_frozen": True,
        "train_rows": admitted,
        "train_report_sha256": _sha256(report),
        "train_supervision_sha256": _sha256(destination),
        "audit_code_sha256": source_hashes["audit_code"],
        "manifest_sha256": source_hashes["manifest"],
        "corrected_supervision_v6_sha256": source_hashes[
            "corrected_supervision_v6"
        ],
        "locked_test_touched": False,
    }
    _atomic_json(output / "FROZEN_RULES.json", freeze)
    atexit.unregister(lease.__exit__)
    lease.__exit__(None, None, None)


def audit_validation(args: argparse.Namespace) -> None:
    lease = _lease(args, "validation")
    ready = verify_data_ready(args.ready_marker)
    manifest_path = Path(str(ready["paths"]["manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    freeze_path = output / "FROZEN_RULES.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if not freeze.get("rules_frozen"):
        raise ValueError("Train audit rules are not frozen")
    if _sha256(Path(__file__)) != freeze["audit_code_sha256"]:
        raise ValueError("Audit code changed after train rule freeze")
    if _sha256(manifest_path) != freeze["manifest_sha256"]:
        raise ValueError("Manifest changed after train rule freeze")

    totals: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    examples: dict[str, Any] = {}
    excluded: list[dict[str, str]] = []
    admitted = 0
    selected = manifest["val"]
    for manifest_row in selected:
        sample = str(manifest_row["sample"])
        try:
            sample_dir = Path(manifest_row["sample_dir"])
            note_map = json.loads(
                (sample_dir / "note_map.json").read_text(encoding="utf-8")
            )
            metadata = json.loads(
                (sample_dir / "metadata.json").read_text(encoding="utf-8")
            )
            midi = audit._midi_events(sample_dir / "performance_audio.mid")
            if midi is None:
                raise ValueError(f"{sample}: raw performance MIDI is unreadable")
            shift = audit._inferred_midi_shift(note_map, midi, metadata)
            raw_rows = list(midi["notes"])
            performed = list(note_map["performed_notes"])
            if len(raw_rows) != len(performed) or any(
                int(actual["pitch"]) + shift != int(source["pitch_midi"])
                for actual, source in zip(raw_rows, performed)
            ):
                raise ValueError(f"{sample}: validation exact lineage failed")
            rebuilt = _reconstructed_lineage(note_map, raw_rows, shift)
            index = ScoreEventIndex.from_musicxml(
                sample_dir / "verified_score.musicxml", rebuilt
            )
            corrected = {
                "sample": sample,
                "events": [
                    {
                        "rendered_event_id": event.rendered_index,
                        "pitch_written": event.pitch,
                        "start_sec": event.start,
                        "end_sec": event.end,
                        "score_span": event.score_span,
                        "source_indices": event.source_indices,
                        "relationship": event.relationship,
                        "copy_pass": event.copy_pass,
                        "origin_relationship": event.origin_relationship,
                    }
                    for event in index.rendered_events
                ],
            }
            _document, row_counts, row_categories = _audit_row(
                manifest_row, corrected, None, examples
            )
            totals.update(row_counts)
            categories.update(row_categories)
            admitted += 1
        except Exception as error:
            excluded.append({"sample": sample, "reason": str(error)})
    if excluded:
        raise RuntimeError(
            f"Fail-closed validation audit excluded {len(excluded)} row(s): "
            f"{excluded[:3]}"
        )
    _write_report(
        output,
        split="validation",
        admitted=admitted,
        excluded=excluded,
        totals=totals,
        categories=categories,
        examples=examples,
        source_hashes={
            "manifest": _sha256(manifest_path),
            "frozen_rules": _sha256(freeze_path),
            "audit_code": _sha256(Path(__file__)),
        },
        train_supervision=None,
    )
    atexit.unregister(lease.__exit__)
    lease.__exit__(None, None, None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build-train", "audit-validation"))
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    parser.add_argument("--corrected-supervision", type=Path)
    parser.add_argument("--old-supervision", type=Path)
    args = parser.parse_args()
    if args.command == "build-train":
        if args.corrected_supervision is None or args.old_supervision is None:
            parser.error(
                "build-train requires --corrected-supervision and --old-supervision"
            )
        build_train(args)
    else:
        audit_validation(args)


if __name__ == "__main__":
    main()
