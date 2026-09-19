from __future__ import annotations

from pathlib import Path
from collections import defaultdict, deque

import mido

from datacreate.melody import extra_neighbor_core, padded_melody
from synthpipeline.errors import PlannedLabel

MIN_DURATION = 0.05


def ql_to_seconds(ql: float, bpm: float) -> float:
    return float(ql) * 60.0 / float(bpm or 120.0)


def midi_note_times(midi_path: Path) -> list[tuple[int, float, float]]:
    """Read exact MIDI note events, ordered by onset, without score quantization.

    music21's default MIDI quantization can merge adjacent fast triplet notes
    into Chords, dropping events from a Note-only scan. Mido merges tracks and
    applies each tempo change when yielding message deltas in seconds.
    """
    active = defaultdict(deque)
    events = []
    seconds = 0.0
    serial = 0
    for message in mido.MidiFile(midi_path):
        seconds += message.time
        if message.type == "note_on" and message.velocity > 0:
            active[(message.channel, message.note)].append((seconds, serial))
            serial += 1
        elif message.type == "note_off" or (message.type == "note_on" and message.velocity == 0):
            pending = active[(message.channel, message.note)]
            if pending:
                start, order = pending.popleft()
                if seconds <= start:
                    raise ValueError(f"Nonpositive MIDI note duration in {midi_path}")
                events.append((order, int(message.note), start, seconds))
    if any(active.values()):
        raise ValueError(f"Unclosed MIDI note events in {midi_path}")
    return [(pitch, start, end) for _, pitch, start, end in sorted(events)]


def refine_labels(
    planned: list[PlannedLabel],
    bpm: float,
    midi_path: Path | None,
    clean_notes=None,
    pad_notes: int = 2,
) -> list[dict]:
    midi_notes = midi_note_times(midi_path) if midi_path and midi_path.exists() else []
    out: list[dict] = []
    for i, label in enumerate(planned, start=1):
        start = ql_to_seconds(label.ql_start, bpm)
        end = ql_to_seconds(label.ql_end, bpm)
        if label.type != "repetition" and label.type != "missed_note":
            start, end = _snap_to_midi(label, start, end, midi_notes)
        payload: dict = {
            "id": f"syn_{i:03d}",
            "source": "synthetic",
            "start_time": _fmt(start),
            "end_time": _fmt(max(end, start + MIN_DURATION)),
            "type": label.type,
            "severity": 5,
            "comment": label.comment,
        }
        if label.measure_number is not None:
            payload["measure_number"] = label.measure_number
        if label.deviation_cents is not None:
            payload["deviation_cents"] = round(float(label.deviation_cents), 2)
        if label.type == "repetition" and label.repeats_ql_start is not None:
            r_start = ql_to_seconds(label.repeats_ql_start, bpm)
            r_end = ql_to_seconds(label.repeats_ql_end or (label.repeats_ql_start + 0.25), bpm)
            payload["repeats_label_range"] = {
                "start_time": _fmt(r_start),
                "end_time": _fmt(max(r_end, r_start + MIN_DURATION)),
            }
        if label.extra_copies is not None:
            payload["extra_copies"] = int(label.extra_copies)
        if clean_notes and label.clean_note_index is not None and _emit_score_part(label):
            i0 = int(label.clean_note_index)
            if label.type == "extra_note":
                i0, i1 = extra_neighbor_core(clean_notes, i0)
            else:
                count = max(1, int(label.clean_note_count or 1))
                i1 = i0 + count
            span = padded_melody(clean_notes, i0, i1, pad_notes)
            payload.update(span.as_fields())
        out.append(payload)
    return out


def _emit_score_part(label: PlannedLabel) -> bool:
    comment = label.comment or ""
    if "repeated pass" in comment:
        return False
    if "first pass" in comment or label.type == "repetition":
        return True
    return "(pass " not in comment


def _snap_to_midi(
    label: PlannedLabel,
    start: float,
    end: float,
    midi_notes: list[tuple[int, float, float]],
) -> tuple[float, float]:
    if not midi_notes:
        return start, end
    if label.note_index is not None and 0 <= label.note_index < len(midi_notes):
        pitch_i, s, e = midi_notes[label.note_index]
        count = max(1, int(label.note_count or 1))
        last_i = min(label.note_index + count - 1, len(midi_notes) - 1)
        _, _, e_last = midi_notes[last_i]
        if count > 1:
            return s, e_last
        if label.midi_pitch is None or pitch_i == label.midi_pitch:
            return s, e
    matches = [n for n in midi_notes if label.midi_pitch is None or n[0] == label.midi_pitch]
    if not matches:
        return start, end
    pitch_i, s, e = min(matches, key=lambda n: abs(n[1] - start))
    return s, e


def _fmt(value: float) -> float:
    return round(float(value), 4)
