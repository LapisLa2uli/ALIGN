from __future__ import annotations

from pathlib import Path
from typing import Any

from datacreate.melody import (
    MATCH_LENGTH_RATIO,
    MATCH_SIMILARITY_THRESHOLD,
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
    melodies_set_match,
    melody_pair_score,
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

from alignmodel.stages.gold import SCORED_TYPES
from alignmodel.types import PipelineLabel, PipelineState, ScorePart

__all__ = [
    "MATCH_LENGTH_RATIO",
    "MATCH_SIMILARITY_THRESHOLD",
    "MelodySpan",
    "ScoreSoundingNote",
    "WeakMelody",
    "attach_schema12_fields",
    "extra_neighbor_core",
    "gold_melodies_from_labels",
    "is_contiguous_part",
    "is_repeated_pass",
    "label_already_converted",
    "lcs_length",
    "match_melodies",
    "match_melodies_detail",
    "melodies_containment_match",
    "melodies_set_match",
    "melody_pair_score",
    "melody_similarity",
    "melody_span_from_label",
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


def melody_span_from_label(
    lab: dict[str, Any],
    notes: list[ScoreSoundingNote],
    pad_notes: int = 2,
) -> MelodySpan | None:
    if not notes:
        return None
    if label_already_converted(lab):
        part = lab["score_part"]
        i0 = max(0, min(int(part["start_note_index"]), len(notes) - 1))
        i1 = max(i0, min(int(part["end_note_index"]), len(notes) - 1))
        span_notes = notes[i0 : i1 + 1]
        return MelodySpan(
            start_note_index=i0,
            end_note_index=i1,
            pad_notes=int(part.get("pad_notes") or pad_notes),
            pitches=[item.pitch for item in span_notes],
            note_ids=[item.note_id for item in span_notes],
            start_measure=span_notes[0].measure,
            end_measure=span_notes[-1].measure,
        )
    core = _core_from_timed_label(lab, notes)
    if core is None:
        return None
    if lab.get("type") == "extra_note":
        core = extra_neighbor_core(notes, core[0])
    return padded_melody(notes, core[0], core[1], pad_notes)


def attach_schema12_fields(state: PipelineState, pad_notes: int = 2) -> None:
    notes = load_bundle_notes(Path(state.sample_dir))
    if not notes:
        return
    for lab in state.labels:
        if lab.type not in SCORED_TYPES:
            continue
        span = melody_span_from_label(_label_as_mapping(lab), notes, pad_notes=pad_notes)
        if span is None:
            continue
        lab.score_part = ScorePart(
            start_note_index=span.start_note_index,
            end_note_index=span.end_note_index,
            pad_notes=span.pad_notes,
            start_measure=span.start_measure,
            end_measure=span.end_measure,
        )
        lab.pitches = list(span.pitches)
        lab.note_ids = list(span.note_ids)
        if lab.type == "repetition" and lab.extra_copies is None:
            lab.extra_copies = 1


def _label_as_mapping(lab: PipelineLabel) -> dict[str, Any]:
    src = None
    if lab.repeats_label_range is not None:
        src = {
            "start_time": lab.repeats_label_range.start_time,
            "end_time": lab.repeats_label_range.end_time,
        }
    return {
        "type": lab.type,
        "start_time": lab.start_time,
        "end_time": lab.end_time,
        "comment": lab.comment,
        "measure_number": lab.measure_number,
        "repeats_label_range": src,
        "score_part": (
            {
                "start_note_index": lab.score_part.start_note_index,
                "end_note_index": lab.score_part.end_note_index,
                "pad_notes": lab.score_part.pad_notes,
            }
            if lab.score_part is not None
            else None
        ),
        "pitches": lab.pitches,
    }
