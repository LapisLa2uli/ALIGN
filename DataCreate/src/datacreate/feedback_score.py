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
    from music21 import converter
    from datacreate.melody import parse_sounding_notes

    parsed = converter.parse(str(score_path), forceSource=True)
    score_notes = parse_sounding_notes(parsed)
    active, events = {}, []
    seconds, ticks, tempo = 0.0, 0, 500000
    midi = mido.MidiFile(str(midi_path))
    for message in mido.merge_tracks(midi.tracks):
        ticks += message.time
        seconds += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == 'set_tempo':
            tempo = message.tempo
        if message.type == "note_on" and message.velocity > 0:
            key = (message.channel, message.note)
            if key in active:
                raise ValueError("Overlapping MIDI notes cannot be matched reliably to score indices.")
            active[key] = (seconds, ticks / midi.ticks_per_beat)
        elif message.type in ("note_on", "note_off"):
            key = (message.channel, message.note)
            start = active.pop(key, None)
            if start is not None:
                events.append({"pitch": message.note, "start": start[0], "end": seconds,
                               "ql_start": start[1], "ql_end": ticks / midi.ticks_per_beat})
    events.sort(key=lambda item: (item["start"], item["pitch"]))
    if active:
        raise ValueError("Reference MIDI does not match the sounding notes in the selected score. Re-render the reference.")
    if len(events) != len(score_notes):
        return _decorated_reference_times(parsed, score_notes, events, midi.ticks_per_beat)
    # A constant written-to-sounding transposition is expected for clarinet.
    shifts = {event["pitch"] - note.pitch for event, note in zip(events, score_notes)}
    if len(shifts) > 1:
        raise ValueError("Reference MIDI pitches disagree with the selected score. Re-render the reference.")
    return events


def _decorated_reference_times(parsed, notes, events, ticks_per_beat):
    """Project rendered ornaments onto canonical notes using MIDI score ticks.

    Playback seconds alone cannot locate notes after tempo changes. Canonical
    indices exclude grace notes and collapse ties, whereas MuseScore expands
    ornaments. Validate every rendered pitch against the notation before grouping.
    Repeats that unfold the score timeline are intentionally not guessed here.
    """
    from bisect import bisect_right
    from music21 import expressions, note
    from datacreate.score_notes import is_decorative_element

    def mismatch():
        raise ValueError("Reference MIDI does not match the score's note and ornament timeline. Re-render the reference.")

    if not notes or not events:
        mismatch()
    starts = [n.ql_start for n in notes]
    tolerance = 1 / ticks_per_beat + 1e-7
    if any(a.ql_end > b.ql_start + tolerance for a, b in zip(notes, notes[1:])):
        mismatch()  # This narration pipeline requires one monophonic part.
    allowed = [{n.pitch} for n in notes]
    decorated = set()
    for element in parsed.recurse().getElementsByClass(note.Note):
        ql = float(element.getOffsetInHierarchy(parsed))
        index = bisect_right(starts, ql + tolerance) - 1
        if is_decorative_element(element):
            # Grace notes can borrow time before or after their notated onset.
            # Restrict them to the two adjacent canonical notes at that boundary.
            for i in (index - 1, index):
                if i >= 0 and (abs(notes[i].ql_end - ql) <= tolerance
                               or abs(notes[i].ql_start - ql) <= tolerance):
                    allowed[i].add(int(element.pitch.midi))
                    decorated.add(i)
        elif index >= 0 and any(isinstance(e, expressions.Ornament) for e in element.expressions):
            allowed[index].update(int(n.pitch.midi) for n in expressions.realizeOrnaments(element)
                                  if isinstance(n, note.Note))
            decorated.add(index)

    grouped = [[] for _ in notes]
    for event in events:
        index = bisect_right(starts, event['ql_start'] + tolerance) - 1
        if index < 0 or event['ql_start'] >= notes[index].ql_end + tolerance:
            mismatch()
        grouped[index].append(event)
    if any(not group for group in grouped):
        mismatch()
    # Establish a single transposition from unornamented anchors, then validate
    # ornament groups too. An unrelated MIDI must not pass on note count alone.
    shifts = {group[0]['pitch'] - n.pitch for i, (n, group) in enumerate(zip(notes, grouped))
              if i not in decorated and len(group) == 1}
    if len(shifts) != 1:
        mismatch()
    shift = shifts.pop()
    result = []
    for i, (n, group) in enumerate(zip(notes, grouped)):
        pitches = {e['pitch'] - shift for e in group}
        if n.pitch not in pitches or not pitches <= allowed[i] or (len(group) > 1 and i not in decorated):
            mismatch()
        if any(e['ql_end'] > n.ql_end + tolerance for e in group):
            mismatch()
        result.append({'pitch': n.pitch + shift, 'start': min(e['start'] for e in group),
                       'end': max(e['end'] for e in group)})
    return result
