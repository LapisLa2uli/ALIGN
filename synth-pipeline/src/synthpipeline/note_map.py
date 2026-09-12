from __future__ import annotations

import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from music21 import converter, note, stream


NOTE_MAP_SCHEMA_VERSION = "1.0"
_CLEAN_ATTR = "_synthpipeline_clean_note_index"
_CLEAN_EDITORIAL = "synthpipeline_clean_note_index"
_EXTRA_ATTR = "_synthpipeline_extra_note_id"
_EXTRA_EDITORIAL = "synthpipeline_extra_note_id"


def sounding_notes(score_or_part: stream.Stream) -> list[note.Note]:
    """Return non-grace Note objects in deterministic score order."""
    return [
        item
        for item in score_or_part.recurse().getElementsByClass(note.Note)
        if not item.duration.isGrace
    ]


def tag_clean_notes(score: stream.Score) -> int:
    """Attach stable clean-note indices that survive music21 deepcopy operations."""
    notes = sounding_notes(score)
    for clean_index, item in enumerate(notes):
        setattr(item, _CLEAN_ATTR, clean_index)
        item.editorial[_CLEAN_EDITORIAL] = clean_index
    return len(notes)


def tag_extra_note(item: note.Note, score: stream.Score) -> str:
    """Tag a newly injected note so repeated copies of the extra remain identifiable."""
    used = {
        extra_id
        for existing in sounding_notes(score)
        if (extra_id := _identity_value(existing, _EXTRA_ATTR, _EXTRA_EDITORIAL))
        is not None
    }
    serial = 0
    while f"extra-{serial}" in used:
        serial += 1
    extra_id = f"extra-{serial}"
    setattr(item, _EXTRA_ATTR, extra_id)
    item.editorial[_EXTRA_EDITORIAL] = extra_id
    return extra_id


def clean_note_index(item: note.Note) -> int | None:
    value = _identity_value(item, _CLEAN_ATTR, _CLEAN_EDITORIAL)
    return int(value) if value is not None else None


def extra_note_id(item: note.Note) -> str | None:
    value = _identity_value(item, _EXTRA_ATTR, _EXTRA_EDITORIAL)
    return str(value) if value is not None else None


def note_signature(item: note.Note, score: stream.Score) -> dict[str, Any]:
    """Return the written-pitch signature used by replay validation and caches."""
    try:
        onset = float(item.getOffsetInHierarchy(score))
    except Exception:
        onset = float(item.offset)
    measure = item.getContextByClass(stream.Measure)
    number = getattr(measure, "number", None) if measure is not None else None
    return {
        "pitch_midi": int(item.pitch.midi),
        "pitch": item.pitch.nameWithOctave,
        "onset_ql": round(onset, 9),
        "duration_ql": round(float(item.duration.quarterLength), 9),
        "measure": int(number) if number is not None else None,
    }


def note_signatures(score: stream.Score) -> list[dict[str, Any]]:
    return [note_signature(item, score) for item in sounding_notes(score)]


def build_note_map(clean_score: stream.Score, performed_score: stream.Score) -> dict[str, Any]:
    """Build exact performed-to-clean lineage from generation-time note identities.

    ``performed_score`` must be the in-memory score returned by error injection.
    Saved performance MusicXML is deliberately not consulted.
    """
    clean = sounding_notes(clean_score)
    performed = sounding_notes(performed_score)
    clean_by_index: dict[int, note.Note] = {}
    for expected, item in enumerate(clean):
        actual = clean_note_index(item)
        if actual is None:
            raise ValueError("Clean score is not tagged; call tag_clean_notes before injection")
        if actual != expected:
            raise ValueError(
                f"Clean note identity mismatch at position {expected}: got {actual}"
            )
        clean_by_index[actual] = item

    seen: defaultdict[tuple[str, object], int] = defaultdict(int)
    mapped_clean: Counter[int] = Counter()
    performed_payload: list[dict[str, Any]] = []
    for performed_index, item in enumerate(performed):
        mapped_index = clean_note_index(item)
        injected_id = extra_note_id(item)
        if mapped_index is not None:
            if mapped_index not in clean_by_index:
                raise ValueError(
                    f"Performed note {performed_index} refers to unknown clean note {mapped_index}"
                )
            identity: tuple[str, object] = ("clean", mapped_index)
            mapped_clean[mapped_index] += 1
            clean_pitch = int(clean_by_index[mapped_index].pitch.midi)
            origin = (
                "match" if int(item.pitch.midi) == clean_pitch else "substitute"
            )
        elif injected_id is not None:
            identity = ("extra", injected_id)
            origin = "extra"
        else:
            # Unknown inserted notes remain valid extras, but cannot have a replay pass.
            identity = ("untagged", performed_index)
            origin = "extra"

        copy_pass = seen[identity]
        seen[identity] += 1
        relationship = "copy" if copy_pass else origin
        performed_payload.append(
            {
                "performed_index": performed_index,
                "clean_index": mapped_index,
                "relationship": relationship,
                "origin_relationship": origin,
                "copy_pass": copy_pass,
                **note_signature(item, performed_score),
            }
        )

    deleted = [index for index in range(len(clean)) if mapped_clean[index] == 0]
    clean_payload = [
        {
            "clean_index": index,
            "deleted": index in deleted,
            **note_signature(item, clean_score),
        }
        for index, item in enumerate(clean)
    ]
    relationship_counts = Counter(
        entry["relationship"] for entry in performed_payload
    )
    return {
        "schema_version": NOTE_MAP_SCHEMA_VERSION,
        "kind": "synth_note_lineage",
        "clean_note_count": len(clean_payload),
        "performed_note_count": len(performed_payload),
        "clean_notes": clean_payload,
        "performed_notes": performed_payload,
        "deleted_clean_notes": deleted,
        "relationship_counts": {
            key: relationship_counts.get(key, 0)
            for key in ("match", "substitute", "extra", "copy")
        },
    }


def attach_rendered_events(
    payload: dict[str, Any],
    midi_path: Path,
    *,
    sounding_transpose: int = -2,
    performed_score_path: Path | None = None,
    midi_to_written_shift: int | None = None,
) -> dict[str, Any]:
    """Attach the audible MIDI events, including many-notated-notes-to-one ties.

    MusicXML can contain tied or ornament-expanded notes that the MIDI writer
    renders as one event. Alignment must be supervised on audible events rather
    than assuming every notated performance note produces a separate onset.
    """
    from synthpipeline.timing import midi_note_times

    performed = list(payload.get("performed_notes") or [])
    midi = midi_note_times(Path(midi_path))
    written_shift = (
        -int(sounding_transpose)
        if midi_to_written_shift is None
        else int(midi_to_written_shift)
    )
    written_pitch = [int(pitch) + written_shift for pitch, _s, _e in midi]
    score_pitch = [int(row["pitch_midi"]) for row in performed]
    mapping, deleted = _sequence_alignment(written_pitch, score_pitch)
    by_midi: list[list[int]] = [[] for _ in midi]
    timed_rows = (
        _performed_note_seconds(performed_score_path)
        if performed_score_path is not None
        else []
    )
    if len(timed_rows) == len(performed):
        claimed: set[int] = set()
        for midi_i, (raw_pitch, start, end) in enumerate(midi):
            pitch = int(raw_pitch) + written_shift
            overlapping = [
                score_i
                for score_i, row in enumerate(timed_rows)
                if score_i not in claimed
                and int(row["pitch_midi"]) == pitch
                and float(row["start_sec"]) < float(end) + 0.04
                and float(start) < float(row["end_sec"]) + 0.04
            ]
            if not overlapping:
                nearby = [
                    score_i
                    for score_i, row in enumerate(timed_rows)
                    if score_i not in claimed
                    and int(row["pitch_midi"]) == pitch
                    and abs(float(row["start_sec"]) - float(start)) <= 0.12
                ]
                overlapping = sorted(
                    nearby,
                    key=lambda score_i: abs(
                        float(timed_rows[score_i]["start_sec"]) - float(start)
                    ),
                )[:1]
            if overlapping:
                by_midi[midi_i] = sorted(overlapping)
                claimed.update(overlapping)
        # Sequence fallback is only for audible events with no timed match.
        for midi_i, score_i in enumerate(mapping):
            if not by_midi[midi_i] and score_i is not None and score_i not in claimed:
                by_midi[midi_i] = [int(score_i)]
                claimed.add(int(score_i))
        deleted = [index for index in range(len(performed)) if index not in claimed]
    else:
        by_midi = [
            ([int(score_i)] if score_i is not None else []) for score_i in mapping
        ]

    # A tied chain commonly appears as one MIDI event followed by one or more
    # deleted same-pitch MusicXML notes. Fold those rows into the nearest event.
    for score_i in deleted:
        pitch = score_pitch[score_i]
        candidates = [
            midi_i
            for midi_i, score_rows in enumerate(by_midi)
            if score_rows
            and pitch == score_pitch[score_rows[-1]]
            and abs(score_i - score_rows[-1]) <= 4
        ]
        if not candidates:
            candidates = [
                midi_i
                for midi_i, score_rows in enumerate(by_midi)
                if score_rows
                and pitch == score_pitch[score_rows[0]]
                and abs(score_i - score_rows[0]) <= 4
            ]
        if candidates:
            midi_i = min(
                candidates,
                key=lambda index: min(
                    abs(score_i - existing) for existing in by_midi[index]
                ),
            )
            by_midi[midi_i].append(score_i)
            by_midi[midi_i].sort()

    rendered = []
    for midi_i, ((raw_pitch, start, end), score_rows) in enumerate(
        zip(midi, by_midi)
    ):
        clean_indices = []
        relationships = []
        for score_i in score_rows:
            row = performed[score_i]
            clean = row.get("clean_index")
            if clean is not None and int(clean) not in clean_indices:
                clean_indices.append(int(clean))
            relationships.append(str(row.get("relationship") or "extra"))
        rendered.append(
            {
                "rendered_index": midi_i,
                "pitch_midi_sounding": int(raw_pitch),
                "pitch_midi_written": int(raw_pitch) + written_shift,
                "start_sec": round(float(start), 9),
                "end_sec": round(max(float(end), float(start) + 0.001), 9),
                "performed_indices": score_rows,
                "clean_indices": clean_indices,
                "primary_clean_index": clean_indices[0] if clean_indices else None,
                "relationship": (
                    "copy"
                    if "copy" in relationships
                    else (relationships[0] if relationships else "extra")
                ),
            }
        )
    payload["rendered_note_count"] = len(rendered)
    payload["rendered_notes"] = rendered
    payload["render_validation"] = {
        "midi_events": len(midi),
        "performed_score_notes": len(performed),
        "directly_mapped": sum(value is not None for value in mapping),
        "folded_tied_notes": sum(max(0, len(rows) - 1) for rows in by_midi),
        "unmapped_performed_notes": sum(
            1
            for score_i in deleted
            if not any(score_i in rows for rows in by_midi)
        ),
    }
    return payload


def _performed_note_seconds(path: Path) -> list[dict[str, Any]]:
    parsed = converter.parse(str(path))
    rows = []
    for item in parsed.flatten().secondsMap:
        element = item.get("element")
        if not isinstance(element, note.Note) or element.duration.isGrace:
            continue
        start = float(item.get("offsetSeconds", 0.0))
        rows.append(
            {
                "pitch_midi": int(element.pitch.midi),
                "start_sec": start,
                "end_sec": max(float(item.get("endTimeSeconds", start)), start + 0.001),
            }
        )
    return rows


def _sequence_alignment(
    source: list[int], target: list[int]
) -> tuple[list[int | None], list[int]]:
    """Monotonic pitch edit path from audible events to notated performance."""
    n, m = len(source), len(target)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    bt = np.zeros((n + 1, m + 1), dtype=np.int8)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    bt[1:, 0] = 1
    bt[0, 1:] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = (
                (
                    int(dp[i - 1, j - 1])
                    + int(source[i - 1] != target[j - 1]),
                    0,
                ),
                (int(dp[i - 1, j]) + 1, 1),
                (int(dp[i, j - 1]) + 1, 2),
            )
            value, code = min(options)
            dp[i, j], bt[i, j] = value, code
    mapping: list[int | None] = [None] * n
    deleted: list[int] = []
    i, j = n, m
    while i or j:
        code = int(bt[i, j])
        if i and j and code == 0:
            mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif i and (not j or code == 1):
            i -= 1
        else:
            deleted.append(j - 1)
            j -= 1
    deleted.reverse()
    return mapping, deleted


def write_note_map(path: Path, payload: dict[str, Any]) -> Path:
    """Atomically write a note-map cache."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path


def _identity_value(item: note.Note, attr: str, editorial_key: str) -> object | None:
    value = getattr(item, attr, None)
    if value is not None:
        return value
    try:
        return item.editorial.get(editorial_key)
    except (AttributeError, TypeError):
        return None
