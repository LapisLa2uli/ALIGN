"""Whole-bar, paired musical examples rendered from note events, never audio crops."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from pathlib import Path

import mido
import numpy as np
import soundfile as sf


def _notes(rows):
    result = []
    for row in rows:
        if row.get("ignored"):
            continue
        pitch, start, end = row.get("pitch"), row.get("start"), row.get("end")
        if (type(pitch) is not int or not 0 <= pitch <= 127 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, end)) or
                not 0 <= start < end):
            raise ValueError("Transcription must contain finite start/end seconds and MIDI pitch values.")
        result.append({"pitch": pitch, "start": float(start), "end": float(end)})
    return sorted(result, key=lambda n: (n["start"], n["pitch"]))


def _performance_pitch_shift(alignment, transcription, reference_shift):
    """Convert the transcription's pitch coordinate system to audible MIDI.

    ALIGN compares notes on written-score axes. An explicit concert-pitch
    transcription must bypass this conversion, even for a transposing score.
    """
    for document in (transcription, alignment):
        if not isinstance(document, dict):
            continue
        provenance = document.get("provenance") or {}
        space = document.get("pitch_space", provenance.get("pitch_space", "")).lower()
        if space in {"sounding", "sounding_midi", "concert", "concert_pitch"}:
            return 0, "explicit_sounding_pitch"
        convention = str(provenance.get("pitch_convention", document.get("pitch_convention", "")))
        declared = re.search(r"sounding\s*=\s*written\s*([+-])\s*(\d+)", convention, re.I)
        if declared:
            shift = int(declared[2]) * (-1 if declared[1] == "-" else 1)
            return shift, "declared_pitch_convention"
        if space in {"written", "written_midi"} or convention.lower().startswith("written"):
            return reference_shift, "written_score_to_sounding"
    # Legacy ALIGN note files use the same written pitch axes as score matching.
    return reference_shift, "written_score_to_sounding"


def slow_pair(reference, performance, ref_duration, perf_duration, minimum_seconds=3.0, note_seconds=.22):
    """Scale both time axes equally; do not quantize or correct performed rhythm."""
    factor = max(1.0, minimum_seconds / max(.1, min(ref_duration, perf_duration)))
    for notes in (reference, performance):
        intervals = [n["end"] - n["start"] for n in notes]
        intervals += [b["start"] - a["start"] for a, b in zip(notes, notes[1:]) if b["start"] > a["start"]]
        if intervals:
            factor = max(factor, note_seconds / max(.01, float(np.percentile(intervals, 10))))
    # Avoid a stray 10 ms transcription event making a teaching example minutes long.
    factor = min(4.0, factor)
    def scaled(notes):
        return [{**n, "start": n["start"] * factor, "end": n["end"] * factor} for n in notes]
    return scaled(reference), scaled(performance), factor


def _midi_clock(path):
    midi = mido.MidiFile(str(path))
    ticks, seconds, tempo = 0, 0., 500000
    changes = [(0., 0., tempo)]
    program = 71
    active, events = {}, []
    for message in mido.merge_tracks(midi.tracks):
        ticks += message.time
        seconds += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = message.tempo
            changes.append((ticks / midi.ticks_per_beat, seconds, tempo))
        elif message.type == "program_change":
            program = message.program
        elif message.type == "note_on" and message.velocity > 0:
            active[(message.channel, message.note)] = seconds
        elif message.type in ("note_on", "note_off"):
            start = active.pop((message.channel, message.note), None)
            if start is not None:
                events.append({"pitch": message.note, "start": start, "end": seconds})
    def seconds_at(ql):
        q0, s0, bpm = next((x for x in reversed(changes) if x[0] <= ql), changes[0])
        return s0 + (ql - q0) * bpm / 1_000_000
    return seconds_at, program, sorted(events, key=lambda n: (n["start"], n["pitch"]))


def prepare_examples(report, sample: Path, score_path: Path, midi_path: Path,
                     transcription_path: Path | None = None, alignment_path: Path | None = None):
    from music21 import converter, meter, stream
    from datacreate.melody import parse_sounding_notes
    from datacreate.feedback_score import label_note_indices, reference_note_times, whole_score_locations

    alignment_path = alignment_path or sample / "note_alignment_v2.json"
    if not alignment_path.is_file():
        raise ValueError("Synthesized examples need note_alignment_v2.json to locate complete performed bars.")
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    data = None
    if transcription_path is not None:
        data = json.loads(transcription_path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("transcribed_notes")
    else:
        # Use the exact transcription aligned by this result, not an unrelated older file.
        rows = alignment.get("transcribed_notes")
    if not isinstance(rows, list):
        raise ValueError("The alignment needs transcribed_notes, or supply --transcription with its matching note JSON.")
    performed = _notes(rows)
    parsed = converter.parse(str(score_path), forceSource=True)
    if len(parsed.parts) != 1:
        raise ValueError("Synthesized examples require a single-part score.")
    selected_notes = parse_sounding_notes(parsed)
    locations, bar_map = whole_score_locations(parsed, selected_notes, sample)
    midi_notes = reference_note_times(score_path, midi_path)
    shifts = {r["pitch"] - n.pitch for r, n in zip(midi_notes, selected_notes)}
    if len(shifts) != 1:
        raise ValueError("Cannot establish reference MIDI transposition.")
    shift = shifts.pop()
    performance_shift, pitch_conversion = _performance_pitch_shift(alignment, data, shift)
    for n in performed:
        n["pitch"] += performance_shift
        if not 0 <= n["pitch"] <= 127:
            raise ValueError("Transcription pitch is outside the MIDI range after conversion to sounding pitch.")
    clock, program, rendered_notes = _midi_clock(midi_path)
    meta_path = sample / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    segment = metadata.get("score_segment") or {}
    full_path = sample / "full_score.musicxml"
    selected_measures = list(parsed.parts[0].getElementsByClass(stream.Measure))
    full = converter.parse(str(full_path), forceSource=True) if full_path.is_file() else parsed
    measures = list(full.parts[0].getElementsByClass(stream.Measure))
    if full_path.is_file():
        # bar_map values are full-score enumeration, not MusicXML labels.
        by_bar = {i + 1: m for i, m in enumerate(measures)}
        first = by_bar[bar_map[selected_measures[0].number]]
        signature = first.timeSignature or first.getContextByClass(meter.TimeSignature)
        beat = segment.get("start_beat", 1)
        if beat > 1 and signature is None:
            raise ValueError("Cannot locate a partial first bar without its meter.")
        origin = float(first.getOffsetInHierarchy(full)) + (beat - 1) * (float(signature.beatDuration.quarterLength) if signature else 1)
    else:
        if segment.get("start_beat", 1) > 1 or segment.get("end_beat") is not None:
            raise ValueError("Whole-bar examples of a partial selection require full_score.musicxml.")
        by_bar = {bar_map[m.number]: m for m in measures}
        origin = 0.
    full_notes = parse_sounding_notes(full)
    aligned = alignment.get("events", [])
    examples = {}
    alignment_hash = hashlib.sha256(alignment_path.read_bytes()).hexdigest()
    for index, label in enumerate(report["labels"]):
        indices = list(label_note_indices(label))
        if not indices:
            indices = [int(e["sounding_index"]) for e in aligned
                       if type(e.get("sounding_index")) is int and isinstance(e.get("perf_start"), (int, float))
                       and isinstance(e.get("perf_end"), (int, float)) and "start_time" in label
                       and e["perf_end"] > label["start_time"] and e["perf_start"] < label["end_time"]]
        if not indices:
            indices = [i for i, n in enumerate(selected_notes) if n.measure == label.get("measure_number")]
        if not indices:
            raise ValueError("Cannot locate a labeled error in the score; supply aligned note indices or a bar number, or use --no-excerpts.")
        if min(indices) < 0 or max(indices) >= len(selected_notes):
            raise ValueError("Label score indices are outside the selected score.")
        first_bar = min(locations[i][0] for i in indices)
        last_bar = max(locations[i][0] for i in indices)
        bars = list(range(first_bar, last_bar + 1))
        if any(bar not in by_bar for bar in bars):
            raise ValueError("Cannot expand the example to its complete score bars.")
        first, last = by_bar[first_bar], by_bar[last_bar]
        q0 = float(first.getOffsetInHierarchy(full))
        q1 = (float(by_bar[last_bar+1].getOffsetInHierarchy(full)) if last_bar+1 in by_bar
              else float(last.getOffsetInHierarchy(full)) + float(last.highestTime))
        ref_start, ref_end = clock(q0-origin), clock(q1-origin)
        reference = [{"pitch": n.pitch + shift, "start": clock(max(q0,n.ql_start)-origin)-ref_start,
                      "end": clock(min(q1,n.ql_end)-origin)-ref_start}
                     for n in full_notes if n.ql_start < q1 and n.ql_end > q0]
        # Preserve validated MIDI ornaments and articulation inside the selection.
        # The full score supplies only missing parts of a partial selection.
        selected_end = clock(float(parsed.highestTime))
        outside = []
        for n in reference:
            for lo, hi in ((ref_start, min(0., ref_end)), (max(selected_end, ref_start), ref_end)):
                start, end = max(n["start"]+ref_start, lo), min(n["end"]+ref_start, hi)
                if end-start > 1e-8:
                    outside.append({"pitch": n["pitch"], "start": start-ref_start, "end": end-ref_start})
        reference = outside
        reference += [{"pitch": n["pitch"], "start": max(ref_start, n["start"])-ref_start,
                       "end": min(ref_end, n["end"])-ref_start}
                      for n in rendered_notes if n["start"] < ref_end and n["end"] > ref_start]
        reference.sort(key=lambda n: (n["start"], n["pitch"]))
        in_bars = {i for i, loc in enumerate(locations) if first_bar <= loc[0] <= last_bar}
        anchors = [e for e in aligned if e.get("sounding_index") in in_bars
                   and type(e.get("perf_start")) in (int,float) and type(e.get("perf_end")) in (int,float)
                   and math.isfinite(e["perf_start"]) and math.isfinite(e["perf_end"])
                   and 0 <= e["perf_start"] < e["perf_end"]]
        if not anchors:
            # Entirely missed bars can still be demonstrated as the labeled silence.
            if "start_time" not in label:
                raise ValueError("Cannot locate the performed time range for this bar; supply alignment or timed labels.")
            p0, p1 = label["start_time"], label["end_time"]
        else:
            p0 = min(e["perf_start"] for e in anchors)
            p1 = max(e["perf_end"] for e in anchors)
            # Extend over leading/trailing rests and missed notes in the bar.
            # Only the boundary estimate uses local tempo; internal note times
            # always come directly from the transcription.
            seconds_per_quarter = float(np.median([
                (e["perf_end"]-e["perf_start"]) /
                (selected_notes[e["sounding_index"]].ql_end-selected_notes[e["sounding_index"]].ql_start)
                for e in anchors]))
            first_q = min(selected_notes[e["sounding_index"]].ql_start for e in anchors)
            last_q = max(selected_notes[e["sounding_index"]].ql_end for e in anchors)
            # A selection may begin/end mid-bar: never invent unrecorded notes.
            available_start = max(0., q0-origin)
            available_end = min(float(parsed.highestTime), q1-origin)
            p0 = max(0., p0-max(0., first_q-available_start)*seconds_per_quarter)
            p1 += max(0., available_end-last_q)*seconds_per_quarter
            # Do not include a neighbor bar when its aligned boundary is known.
            before = [e["perf_end"] for e in aligned
                      if type(e.get("sounding_index")) is int and 0 <= e["sounding_index"] < len(locations)
                      and locations[e["sounding_index"]][0] < first_bar
                      and type(e.get("perf_end")) in (int, float) and e["perf_end"] <= min(a["perf_start"] for a in anchors)]
            after = [e["perf_start"] for e in aligned
                     if type(e.get("sounding_index")) is int and 0 <= e["sounding_index"] < len(locations)
                     and locations[e["sounding_index"]][0] > last_bar
                     and type(e.get("perf_start")) in (int, float) and e["perf_start"] >= max(a["perf_end"] for a in anchors)]
            if before:
                p0 = max(p0, max(before))
            if after:
                p1 = min(p1, min(after))
            # Include leading/trailing extra notes explicitly marked by the user.
            p0 = min(p0, label.get("start_time", p0))
            p1 = max(p1, label.get("end_time", p1))
        performance = [{"pitch": n["pitch"], "start": max(p0,n["start"])-p0,
                        "end": min(p1,n["end"])-p0}
                       for n in performed if n["start"] < p1 and n["end"] > p0]
        ref, perf, factor = slow_pair(reference, performance, ref_end-ref_start, p1-p0)
        phrase = f"bar {first_bar}" if first_bar == last_bar else f"bars {first_bar} to {last_bar}"
        common = {"synthesized": True, "bars": bars, "slowdown_factor": factor, "program": program,
                  "scope": "whole_bars", "alignment_sha256": alignment_hash}
        examples[index] = {
            "reference": {**common, "notes": ref, "source_start_time": ref_start, "source_end_time": ref_end,
                          "duration_seconds": (ref_end-ref_start)*factor, "pitch_source": "validated_reference_midi_and_score"},
            "performance": {**common, "notes": perf, "source_start_time": p0, "source_end_time": p1,
                            "duration_seconds": (p1-p0)*factor, "pitch_source": "transcription.pitch_to_sounding",
                            "pitch_shift_semitones": performance_shift, "pitch_conversion": pitch_conversion,
                            "timing_source": "transcription.start/end", "partial_recording":
                            bool((first_bar == min(bar_map.values()) and segment.get("start_beat",1)>1)
                                 or (last_bar == max(bar_map.values()) and segment.get("end_beat") is not None))}}
        label["playback_example"] = {"phrase": phrase, "whole_bars": True, "synthesized": True,
                                     "slowdown_factor": round(factor,4),
                                     "partial_recording": examples[index]["performance"]["partial_recording"]}
    return examples


def synthesis_soundfont() -> Path:
    """Check local prerequisites before requesting paid narration."""
    try:
        import tinysoundfont  # noqa: F401
    except ImportError:
        raise ValueError("Install tinysoundfont to synthesize the musical examples.") from None
    from datacreate.config import PipelineConfig
    from datacreate.tools.musescore import find_soundfont
    soundfont = find_soundfont(PipelineConfig.load())
    if soundfont is None or not soundfont.is_file():
        raise ValueError("A SoundFont is required to synthesize musical examples.")
    return soundfont


def render_example(example: dict, destination: Path, soundfont: Path | None = None) -> dict:
    """Schedule notes at sample-accurate times, with identical instrument for both examples."""
    import tinysoundfont
    from datacreate.tools.musescore import _cached_synth, _synth_lock

    soundfont = soundfont or synthesis_soundfont()
    rate = 44100
    schedule = []
    for n in example["notes"]:
        schedule.extend([(round(n["start"]*rate),1,n["pitch"]), (round(n["end"]*rate),0,n["pitch"])])
    schedule.sort()
    end = round((example["duration_seconds"]+.15)*rate)
    chunks, position, active = [], 0, {}
    with _synth_lock:
        synth = _cached_synth(tinysoundfont, soundfont, rate, -6, logging.getLogger(__name__))
        synth.sounds_off()
        synth.program_change(0, example["program"])
        try:
            for frame, on, pitch in schedule + [(end, -1, 0)]:
                if frame > position:
                    while position < frame:
                        count = min(4096, frame-position)
                        chunks.append(np.frombuffer(synth.generate(count),dtype=np.float32).reshape(-1,2).copy())
                        position += count
                if on == 1:
                    synth.noteon(0,pitch,85)
                    active[pitch] = active.get(pitch,0)+1
                elif on == 0:
                    active[pitch] = active.get(pitch,1)-1
                    if active[pitch] == 0:
                        synth.noteoff(0,pitch)
        finally:
            synth.sounds_off()
    audio=np.concatenate(chunks)
    if not np.isfinite(audio).all():
        raise ValueError("Synthesized example contains invalid audio.")
    sf.write(destination,audio,rate,subtype="FLOAT")
    destination.with_suffix('.notes.json').write_text(json.dumps(example,indent=2),encoding='utf-8')
    return {**{k:v for k,v in example.items() if k!='notes'}, "note_count":len(example['notes']),
            "file":destination.name,"sha256":hashlib.sha256(destination.read_bytes()).hexdigest(),
            "sample_rate":rate,"note_events_file":destination.with_suffix('.notes.json').name}
