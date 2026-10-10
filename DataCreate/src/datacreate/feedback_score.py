"""Score identity helpers for full-score locations and rendered reference clips."""
from collections import Counter
import json
from pathlib import Path


class ReferenceMismatchError(ValueError):
    """The rendered reference cannot be validated against the selected score."""


def regenerate_reference(score_path, midi_path, *, config=None, audio_path=None):
    """Rebuild a validated MIDI/WAV pair before replacing the old reference."""
    import logging
    import shutil
    from tempfile import TemporaryDirectory
    from uuid import uuid4
    from datacreate.config import PipelineConfig
    from datacreate.tools.musescore import export_score_to_midi, render_midi_to_wav

    config = config or PipelineConfig.load()
    audio_path = Path(audio_path) if audio_path else midi_path.with_suffix('.wav')
    logger = logging.getLogger(__name__)
    with TemporaryDirectory(prefix='reference-rebuild-', dir=midi_path.parent) as temporary:
        fresh_midi = Path(temporary) / 'reference.mid'
        fresh_audio = Path(temporary) / 'reference.wav'
        export_score_to_midi(config, score_path, fresh_midi, logger)
        reference_note_times(score_path, fresh_midi)
        render_midi_to_wav(fresh_midi, fresh_audio, config, logger)
        # Preserve the original pair for diagnosis; failed validation/rendering
        # never overwrites either original artifact.
        backup = midi_path.parent / ('reference-before-rebuild-' + uuid4().hex)
        backup.mkdir()
        for path in (midi_path, audio_path):
            if path.exists():
                shutil.copy2(path, backup / path.name)
        fresh_audio.replace(audio_path)
        fresh_midi.replace(midi_path)
    logger.info('Automatically regenerated reference MIDI and audio for %s', score_path.name)


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
                raise ReferenceMismatchError("Overlapping MIDI notes cannot be matched reliably to score indices.")
            active[key] = (seconds, ticks / midi.ticks_per_beat)
        elif message.type in ("note_on", "note_off"):
            key = (message.channel, message.note)
            start = active.pop(key, None)
            if start is not None:
                events.append({"pitch": message.note, "start": start[0], "end": seconds,
                               "ql_start": start[1], "ql_end": ticks / midi.ticks_per_beat})
    events.sort(key=lambda item: (item["start"], item["pitch"]))
    if active:
        raise ReferenceMismatchError("Reference MIDI does not match the sounding notes in the selected score. Re-render the reference.")
    events = _validate_chord_events(parsed, events, midi.ticks_per_beat)
    if len(events) != len(score_notes):
        return _decorated_reference_times(parsed, score_notes, events, midi.ticks_per_beat)
    # A constant written-to-sounding transposition is expected for clarinet.
    shifts = {event["pitch"] - note.pitch for event, note in zip(events, score_notes)}
    if len(shifts) > 1:
        raise ReferenceMismatchError("Reference MIDI pitches disagree with the selected score. Re-render the reference.")
    return events


def _validate_chord_events(parsed, events, ticks_per_beat):
    """Validate chord playback without changing the detector's single-note indices.

    Octave alternatives in a wind part are exported as chords by MuseScore, but
    are absent from parse_sounding_notes. Keep them in the actual playback MIDI;
    exclude only validated chord events from the canonical note mapping.
    """
    from music21 import chord, expressions, note
    from datacreate.score_notes import collapse_tied_records

    chords = list(parsed.recurse().getElementsByClass(chord.Chord))
    if not chords:
        return events
    tolerance = 1 / ticks_per_beat + 1e-7
    # Use plain notes to establish sounding pitch independently of chord pitches.
    anchors = []
    for element in parsed.recurse().getElementsByClass(note.Note):
        if element.duration.isGrace or any(isinstance(e, expressions.Ornament) for e in element.expressions):
            continue
        onset = float(element.getOffsetInHierarchy(parsed))
        matches = [e for e in events if abs(e['ql_start'] - onset) <= tolerance]
        if len(matches) == 1:
            anchors.append(matches[0]['pitch'] - int(element.pitch.midi))
    if not anchors or len(set(anchors)) != 1:
        raise ReferenceMismatchError('Cannot validate reference chord transposition. Re-render the reference.')
    shift = anchors[0]
    # Merge each chord tone's ties separately (an octave chord has two voices).
    records = []
    for element in chords:
        for tone in element.notes:
            records.append({'voice': (0, int(tone.pitch.midi)), 'midi': int(tone.pitch.midi),
                            'offset_ql': float(element.getOffsetInHierarchy(parsed)),
                            'duration_ql': float(element.duration.quarterLength),
                            'tie_type': tone.tie.type if tone.tie else None})
    spans = collapse_tied_records(records)
    consumed = set()
    for span in spans:
        start, end = span['offset_ql'], span['offset_ql'] + span['duration_ql']
        matches = [i for i, event in enumerate(events)
                   if i not in consumed and event['pitch'] == span['midi'] + shift
                   and abs(event['ql_start'] - start) <= tolerance
                   and start < event['ql_end'] <= end + tolerance]
        if len(matches) != 1:
            raise ReferenceMismatchError('Reference MIDI chord notes disagree with the score. Re-render the reference.')
        consumed.add(matches[0])
    return [event for i, event in enumerate(events) if i not in consumed]


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
        raise ReferenceMismatchError("Reference MIDI does not match the score's note and ornament timeline. Re-render the reference.")

    if not notes or not events:
        mismatch()
    starts = [n.ql_start for n in notes]
    tolerance = 1 / ticks_per_beat + 1e-7
    if any(a.ql_end > b.ql_start + tolerance for a, b in zip(notes, notes[1:])):
        mismatch()  # This narration pipeline requires one monophonic part.
    allowed = [{n.pitch} for n in notes]
    decorated = set()
    early_graces = []
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
            # MuseScore can place a slashed grace before the preceding short
            # note, taking the second half of the note before that (e.g. a
            # slurred sixteenth-note run). Record only this adjacent pattern;
            # validate the actual grace event after establishing transposition.
            i = index - 2
            if (element.duration.isGrace and element.duration.slash and i >= 0
                    and abs(notes[index].ql_start - ql) <= tolerance
                    and notes[i].measure == notes[index].measure
                    and abs(notes[i].ql_end - notes[i + 1].ql_start) <= tolerance
                    and abs(notes[i + 1].ql_end - ql) <= tolerance
                    and ql - notes[i].ql_start <= 1 + tolerance):
                early_graces.append((i, int(element.pitch.midi)))
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
    for i, pitch in early_graces:
        n, group = notes[i], grouped[i]
        if len(group) != 2 or pitch == n.pitch:
            continue
        main, grace = group
        midpoint = (n.ql_start + n.ql_end) / 2
        if (main['pitch'] == n.pitch + shift and grace['pitch'] == pitch + shift
                and abs(main['ql_start'] - n.ql_start) <= tolerance
                and abs(grace['ql_start'] - midpoint) <= tolerance
                and main['ql_end'] <= grace['ql_start'] + tolerance
                and abs(grace['ql_end'] - n.ql_end) <= tolerance):
            allowed[i].add(pitch)
            decorated.add(i)
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
