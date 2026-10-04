"""Score identity helpers for full-score locations and rendered reference clips."""
from collections import Counter
import json
from pathlib import Path


def label_note_indices(label: dict, *, context: bool = False):
    part = label.get("score_part") or {}
    if part:
        prefix = "core_" if not context and "core_start_note_index" in part else ""
        return range(part[prefix + "start_note_index"], part[prefix + "end_note_index"] + 1)
    return label.get("score_event_indices") or [int(value[5:]) for value in label.get("note_ids", [])]


def whole_score_locations(parsed, notes, sample_directory: Path):
    """Keep note ordinals within bars, but enumerate bars against the full score."""
    from music21 import converter, meter, stream
    from datacreate.melody import parse_sounding_notes

    metadata_path = sample_directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    segment = metadata.get("score_segment") or {}
    selected = list(parsed.parts[0].getElementsByClass(stream.Measure))
    full_path = sample_directory / "full_score.musicxml"
    full = converter.parse(str(full_path), forceSource=True) if full_path.is_file() else None
    full_measures = list(full.parts[0].getElementsByClass(stream.Measure)) if full is not None else []
    full_numbers = [m.number for m in full_measures]
    start = segment.get("start_measure")
    bar_map = {m.number: m.number for m in selected}
    source_map = dict(bar_map)
    if start is not None and selected:
        if selected[0].number not in (1, start):
            raise ValueError("Selected score numbering disagrees with score_segment metadata.")
        if full is not None:
            if start not in full_numbers:
                raise ValueError("Selection start bar is absent from the full score.")
            offset = full_numbers.index(start)
            originals = full_measures[offset:offset + len(selected)]
            if len(originals) != len(selected):
                raise ValueError("Selected measures extend beyond the full score.")
            source_map = {m.number: original.number for m, original in zip(selected, originals)}
            bar_map = {m.number: offset + i + 1 for i, m in enumerate(selected)}
        else:
            bar_map = {m.number: start + i for i, m in enumerate(selected)}
            source_map = dict(bar_map)
    elif full is not None:
        if any(m.number not in full_numbers for m in selected):
            raise ValueError("Selected bars cannot be located in the full score; provide score_segment metadata.")
        bar_map = {m.number: full_numbers.index(m.number) + 1 for m in selected}

    counts = Counter()
    if selected and segment.get("start_beat", 1) > 1:
        first = selected[0].number
        if full is None:
            # A partial first bar has no trustworthy within-bar ordinal without the full score.
            unknown_first = first
        else:
            unknown_first = None
            original = full_measures[full_numbers.index(source_map[first])]
            signature = original.timeSignature or original.getContextByClass(meter.TimeSignature)
            if signature is None:
                raise ValueError("Cannot locate the selection's start beat in the full score.")
            cut = float(original.getOffsetInHierarchy(full)) + (segment["start_beat"] - 1) * float(signature.beatDuration.quarterLength)
            counts[first] = sum(n.measure == original.number and n.ql_start < cut - 1e-8 for n in parse_sounding_notes(full))
    else:
        unknown_first = None
    locations = []
    for note in notes:
        counts[note.measure] += 1
        locations.append((bar_map.get(note.measure, note.measure),
                          None if note.measure == unknown_first else counts[note.measure]))
    return locations, bar_map


def reference_note_times(score_path: Path, midi_path: Path):
    """Match sounding score notes to actual rendered MIDI time, including tempo changes."""
    import mido
    from datacreate.melody import parse_sounding_notes

    score_notes = parse_sounding_notes(score_path)
    active, events = {}, []
    seconds = 0.0
    for message in mido.MidiFile(str(midi_path)):
        seconds += message.time
        if message.type == "note_on" and message.velocity > 0:
            key = (message.channel, message.note)
            if key in active:
                raise ValueError("Overlapping MIDI notes cannot be matched reliably to score indices.")
            active[key] = seconds
        elif message.type in ("note_on", "note_off"):
            key = (message.channel, message.note)
            start = active.pop(key, None)
            if start is not None:
                events.append({"pitch": message.note, "start": start, "end": seconds})
    events.sort(key=lambda item: (item["start"], item["pitch"]))
    if active or len(events) != len(score_notes):
        raise ValueError("Reference MIDI does not match the sounding notes in the selected score. Re-render the reference.")
    # A constant written-to-sounding transposition is expected for clarinet.
    shifts = {event["pitch"] - note.pitch for event, note in zip(events, score_notes)}
    if len(shifts) > 1:
        raise ValueError("Reference MIDI pitches disagree with the selected score. Re-render the reference.")
    return events
