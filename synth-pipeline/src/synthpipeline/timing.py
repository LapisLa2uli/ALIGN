from __future__ import annotations

from pathlib import Path

from music21 import converter, note

from synthpipeline.errors import PlannedLabel

MIN_DURATION = 0.05


def ql_to_seconds(ql: float, bpm: float) -> float:
    return float(ql) * 60.0 / float(bpm or 120.0)


def midi_note_times(midi_path: Path) -> list[tuple[int, float, float]]:
    """Return (midi_pitch, start_sec, end_sec) in performance order."""
    parsed = converter.parse(str(midi_path))
    flat = parsed.flatten()
    times: list[tuple[int, float, float]] = []
    try:
        sec_map = flat.secondsMap
    except Exception:
        sec_map = []
    for item in sec_map:
        el = item.get("element")
        if not isinstance(el, note.Note):
            continue
        start = float(item.get("offsetSeconds", 0.0))
        end = float(item.get("endTimeSeconds", start))
        times.append((int(el.pitch.midi), start, max(end, start + MIN_DURATION)))
    if times:
        return times

    bpm = 120.0
    from music21 import tempo

    for mark in flat.getElementsByClass(tempo.MetronomeMark):
        if mark.number:
            bpm = float(mark.number)
            break
    for n in flat.getElementsByClass(note.Note):
        start = ql_to_seconds(float(n.offset), bpm)
        end = start + ql_to_seconds(float(n.duration.quarterLength), bpm)
        times.append((int(n.pitch.midi), start, max(end, start + MIN_DURATION)))
    return times


def refine_labels(
    planned: list[PlannedLabel],
    bpm: float,
    midi_path: Path | None,
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
        out.append(payload)
    return out


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
