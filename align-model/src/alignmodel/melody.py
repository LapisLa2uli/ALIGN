from __future__ import annotations

from pathlib import Path
from typing import Any

from datacreate.melody import (
    MelodySpan,
    ScoreSoundingNote,
    WeakMelody,
    extra_neighbor_core,
    is_contiguous_part,
    is_repeated_pass,
    label_already_converted,
    lcs_length,
    match_melodies,
    match_melodies_detail,
    melodies_containment_match,
    melody_similarity,
    midi_from_comment,
    note_set_iou,
    notes_for_measure_pitch,
    notes_in_measures,
    notes_overlapping_time,
    padded_melody,
    parse_sounding_notes,
    score_bpm,
)

__all__ = [
    "MelodySpan",
    "ScoreSoundingNote",
    "WeakMelody",
    "extra_neighbor_core",
    "gold_melodies_from_labels",
    "is_contiguous_part",
    "is_repeated_pass",
    "label_already_converted",
    "lcs_length",
    "match_melodies",
    "match_melodies_detail",
    "melodies_containment_match",
    "melody_similarity",
    "midi_from_comment",
    "note_set_iou",
    "notes_for_measure_pitch",
    "notes_in_measures",
    "notes_overlapping_time",
    "padded_melody",
    "parse_sounding_notes",
    "pred_melodies_from_labels",
    "score_bpm",
]


def gold_melodies_from_labels(labels: list[dict[str, Any]]) -> list[WeakMelody]:
    out: list[WeakMelody] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for lab in labels:
        if is_repeated_pass(lab):
            continue
        pitches = lab.get("pitches")
        if not isinstance(pitches, list) or not pitches:
            continue
        ints = [int(p) for p in pitches]
        key = (str(lab.get("type") or ""), tuple(ints))
        if key in seen:
            continue
        seen.add(key)
        extra = lab.get("extra_copies")
        out.append(
            WeakMelody(
                pitches=ints,
                type=lab.get("type"),
                extra_copies=int(extra) if extra is not None else None,
                note_ids=[str(x) for x in (lab.get("note_ids") or [])],
            )
        )
    return out


def pred_melodies_from_labels(
    labels: list[dict[str, Any]],
    notes: list[ScoreSoundingNote],
    pad_notes: int = 2,
) -> list[WeakMelody]:
    out: list[WeakMelody] = []
    for lab in labels:
        if is_repeated_pass(lab):
            continue
        pitches = lab.get("pitches")
        if isinstance(pitches, list) and pitches:
            out.append(
                WeakMelody(
                    pitches=[int(p) for p in pitches],
                    type=lab.get("type"),
                    extra_copies=lab.get("extra_copies"),
                    note_ids=[str(x) for x in (lab.get("note_ids") or [])],
                )
            )
            continue
        if not notes:
            continue
        core = _core_from_timed_label(lab, notes)
        if core is None:
            continue
        if lab.get("type") == "extra_note":
            core = extra_neighbor_core(notes, core[0])
        span = padded_melody(notes, core[0], core[1], pad_notes)
        out.append(WeakMelody(pitches=span.pitches, type=lab.get("type"), note_ids=span.note_ids))
    return out


def _core_from_timed_label(
    lab: dict[str, Any], notes: list[ScoreSoundingNote]
) -> tuple[int, int] | None:
    kind = lab.get("type")
    if kind == "extra_note":
        t0 = lab.get("start_time")
        t1 = lab.get("end_time")
        if t0 is not None and t1 is not None:
            return notes_overlapping_time(notes, float(t0), float(t1))
    if kind == "repetition":
        src = lab.get("repeats_label_range") or {}
        t0 = src.get("start_time", lab.get("start_time"))
        t1 = src.get("end_time", lab.get("end_time"))
        if t0 is not None and t1 is not None:
            return notes_overlapping_time(notes, float(t0), float(t1))
    measure = lab.get("measure_number")
    midis = midi_from_comment(lab.get("comment"))
    pitch = midis[0] if midis else None
    if measure is not None and pitch is not None:
        hit = notes_for_measure_pitch(notes, int(measure), int(pitch))
        if hit is not None:
            return hit
    t0 = lab.get("start_time")
    t1 = lab.get("end_time")
    if t0 is None or t1 is None:
        return None
    return notes_overlapping_time(notes, float(t0), float(t1))


def load_bundle_notes(sample_dir: Path) -> list[ScoreSoundingNote]:
    score = Path(sample_dir) / "verified_score.musicxml"
    if not score.exists():
        return []
    return parse_sounding_notes(score)
