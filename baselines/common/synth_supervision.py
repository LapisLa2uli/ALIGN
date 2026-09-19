"""Exact three-class targets for current synthetic bundles.

Replays the seeded generator, validates note lineage and both original MIDIs,
and tracks removed notes as tagged rests through later edits and repetitions.
No DTW, interpolation between played notes, or padded error-window times are
used to place missing notes. Ambiguous audible tie groups are rejected.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict, deque
from contextlib import contextmanager
import copy
import hashlib
import json
import logging
from pathlib import Path
import random
import tempfile

import mido
from music21 import note

from synthpipeline import errors
from synthpipeline.config import SynthConfig
from synthpipeline.note_map import build_note_map, clean_note_index, tag_clean_notes
from synthpipeline.render import export_score_to_midi_music21
from synthpipeline.scoregen import (
    generate_score, load_score, resolve_score_inputs, snippet_score,
    sounding_note_count, write_musicxml,
)

EXPORTER_VERSION = "seeded_rest_lineage_v1"
_MISSING = "_baseline_missing_clean_index"


class SupervisionError(ValueError):
    pass


class MidiTimeline:
    def __init__(self, path: Path):
        midi = mido.MidiFile(path)
        if midi.type == 2:
            raise SupervisionError("asynchronous_midi")
        self.tpq = midi.ticks_per_beat
        self.tempos = [(0, 0.0, 500000)]
        ticks, seconds, tempo, serial = 0, 0.0, 500000, 0
        active = defaultdict(deque)
        events = []
        for message in mido.merge_tracks(midi.tracks):
            ticks += message.time
            seconds += mido.tick2second(message.time, self.tpq, tempo)
            if message.type == "set_tempo":
                tempo = message.tempo
                self.tempos.append((ticks, seconds, tempo))
            elif message.type == "note_on" and message.velocity > 0:
                active[(message.channel, message.note)].append((ticks, seconds, serial))
                serial += 1
            elif message.type == "note_off" or (message.type == "note_on" and message.velocity == 0):
                pending = active[(message.channel, message.note)]
                if pending:
                    tick0, start, order = pending.popleft()
                    if ticks <= tick0:
                        raise SupervisionError("nonpositive_midi_duration")
                    events.append(dict(index=order, pitch=message.note, start=start, end=seconds,
                                       onset_ql=tick0 / self.tpq, end_ql=ticks / self.tpq))
        if any(active.values()):
            raise SupervisionError("unclosed_midi_events")
        self.events = sorted(events, key=lambda row: row["index"])
        self.tempo_ticks = [row[0] for row in self.tempos]

    def seconds_at(self, quarter_length: float) -> float:
        tick = quarter_length * self.tpq
        index = max(0, bisect_right(self.tempo_ticks, tick) - 1)
        start_tick, start_seconds, tempo = self.tempos[index]
        return start_seconds + mido.tick2second(tick - start_tick, self.tpq, tempo)


@contextmanager
def capture_missing_rests():
    original = errors._missed_note

    def tracked(score, *args, **kwargs):
        before = {id(n): n for n in score.recurse().getElementsByClass(note.Note)}
        result = original(score, *args, **kwargs)
        remaining = {id(n) for n in result.score.recurse().getElementsByClass(note.Note)}
        removed = [n for key, n in before.items() if key not in remaining]
        if len(removed) != 1 or clean_note_index(removed[0]) is None:
            raise SupervisionError("missing_note_without_exact_clean_identity")
        label = result.labels[0]
        rests = [r for r in result.score.recurse().getElementsByClass(note.Rest)
                 if abs(float(r.getOffsetInHierarchy(result.score)) - label.ql_start) < 1e-8
                 and abs(float(r.duration.quarterLength) - (label.ql_end - label.ql_start)) < 1e-8]
        if len(rests) != 1:
            raise SupervisionError("ambiguous_injected_rest")
        setattr(rests[0], _MISSING, clean_note_index(removed[0]))
        return result

    errors._missed_note = tracked
    try:
        yield
    finally:
        errors._missed_note = original


def replay(bundle: Path, config: SynthConfig, temp: Path):
    metadata = json.loads((bundle / "metadata.json").read_text())
    paths = resolve_score_inputs(None, config)
    clean = None
    for attempt in range(24):
        rng = random.Random(int(metadata["seed"]) + 1009 * attempt)
        try:
            if paths is None:
                source, clean = "gen", generate_score(rng, config)
            else:
                path = paths[(int(metadata["index"]) + attempt) % len(paths)]
                source, clean = path.stem, load_score(path, config)
                if config.generation.get("use_snippets"):
                    clean, _ = snippet_score(clean, rng, config)
            if sounding_note_count(clean) > 0:
                break
            clean = None
        except Exception:
            clean = None
    if clean is None or source != metadata["source"]:
        raise SupervisionError("score_preparation_replay_mismatch")
    # Mirror _build_sample exactly: write_musicxml strips ornaments in place,
    # but duration normalization happens on its own deep copy.
    write_musicxml(clean, temp / "clean.musicxml")
    tag_clean_notes(clean)
    with capture_missing_rests():
        result = errors.inject_error(copy.deepcopy(clean), rng, config)
    write_musicxml(result.score, temp / "performance.musicxml")
    lineage = build_note_map(clean, result.score)
    original = json.loads((bundle / "note_map.json").read_text())
    if any(lineage[key] != original[key] for key in lineage):
        raise SupervisionError("generation_lineage_replay_mismatch")
    transpose = int(metadata["sounding_transpose"])
    if metadata.get("midi_pitch_space") != "sounding" or transpose != -2:
        raise SupervisionError("unsupported_pitch_convention")
    clocks = []
    for name, score in (("reference", clean), ("performance", result.score)):
        target = temp / f"{name}.mid"
        export_score_to_midi_music21(score, temp / "unused.musicxml", target,
                                    logging.getLogger("baseline_replay"), sounding_transpose=transpose)
        actual = MidiTimeline(bundle / f"{name}_audio.mid")
        recreated = MidiTimeline(target)
        if len(actual.events) != len(recreated.events) or any(
            a["pitch"] != b["pitch"] or any(abs(a[k] - b[k]) > 1e-6 for k in ("start", "end", "onset_ql", "end_ql"))
            for a, b in zip(actual.events, recreated.events)
        ) or actual.tempos != recreated.tempos:
            raise SupervisionError(f"{name}_midi_replay_mismatch")
        clocks.append(actual)
    missing = defaultdict(list)
    for rest in result.score.recurse().getElementsByClass(note.Rest):
        index = getattr(rest, _MISSING, None)
        if index is not None:
            start = float(rest.getOffsetInHierarchy(result.score))
            missing[int(index)].append((start, start + float(rest.duration.quarterLength)))
    first_missing = {index: min(spans) for index, spans in missing.items()}
    if set(first_missing) != set(lineage["deleted_clean_notes"]):
        raise SupervisionError("deleted_note_rest_lineage_mismatch")
    return lineage, clocks[0], clocks[1], first_missing


def audible_groups(rows: list[dict], clock: MidiTimeline, transpose: int = -2):
    """Require an exact partition of notated notes into audible MIDI events."""
    tolerance = 2 / clock.tpq
    groups, claimed = [], set()
    for event in clock.events:
        members = [i for i, row in enumerate(rows)
                   if row["pitch_midi"] + transpose == event["pitch"]
                   and row["onset_ql"] >= event["onset_ql"] - tolerance
                   and row["onset_ql"] + row["duration_ql"] <= event["end_ql"] + tolerance]
        members.sort(key=lambda i: rows[i]["onset_ql"])
        if not members or claimed.intersection(members):
            raise SupervisionError("ambiguous_midi_note_group")
        cursor = event["onset_ql"]
        for index in members:
            if abs(rows[index]["onset_ql"] - cursor) > tolerance:
                raise SupervisionError("noncontiguous_midi_note_group")
            cursor = rows[index]["onset_ql"] + rows[index]["duration_ql"]
        if abs(cursor - event["end_ql"]) > tolerance:
            raise SupervisionError("midi_note_duration_mismatch")
        groups.append(members)
        claimed.update(members)
    if claimed != set(range(len(rows))):
        raise SupervisionError("unrendered_notated_notes")
    return groups


def make_labels(lineage, reference, performance, missing):
    clean, performed = lineage["clean_notes"], lineage["performed_notes"]
    ref_groups = audible_groups(clean, reference)
    perf_groups = audible_groups(performed, performance)
    clean_to_ref = {c: r for r, members in enumerate(ref_groups) for c in members}
    first_pass, perf_labels, missed_labels = {}, [], []
    for index, (event, members) in enumerate(zip(performance.events, perf_groups)):
        rows = [performed[i] for i in members]
        copies = {int(row["copy_pass"]) for row in rows}
        indices = {row["clean_index"] for row in rows}
        if len(copies) != 1 or (None in indices and len(indices) != 1):
            raise SupervisionError("mixed_origin_performance_tie")
        copy_pass = next(iter(copies))
        cls, sub = "extra", "repeat" if copy_pass else "inserted"
        if not copy_pass and indices != {None}:
            refs = {clean_to_ref[c] for c in indices}
            if len(refs) != 1:
                raise SupervisionError("performance_tie_crosses_reference_events")
            ref_index = next(iter(refs))
            if indices != set(ref_groups[ref_index]) or ref_index in first_pass:
                raise SupervisionError("partially_changed_reference_tie")
            first_pass[ref_index] = index
            cls, sub = ("correct", "correct") if event["pitch"] == reference.events[ref_index]["pitch"] else ("extra", "wrong")
        perf_labels.append(dict(index=index, sounding_pitch=event["pitch"], onset=event["start"],
                                offset=event["end"], cls=cls, sub=sub, copy=copy_pass, tie_prev=False))
    reference_labels = []
    for index, (event, members) in enumerate(zip(reference.events, ref_groups)):
        perf_index = first_pass.get(index)
        cls, sub = "missed", "rest"
        if perf_index is not None:
            perf = perf_labels[perf_index]
            cls, sub = ("correct", "correct") if perf["cls"] == "correct" else ("missed", "wrong")
            start, end = perf["onset"], perf["offset"]
        else:
            if any(c not in missing for c in members):
                raise SupervisionError("missing_reference_without_injected_rest")
            spans = sorted(missing[c] for c in members)
            if any(abs(left[1] - right[0]) > 2 / performance.tpq for left, right in zip(spans, spans[1:])):
                raise SupervisionError("noncontiguous_missing_tie")
            start, end = performance.seconds_at(spans[0][0]), performance.seconds_at(spans[-1][1])
        if cls == "missed":
            if not 0 <= start < end:
                raise SupervisionError("invalid_missing_note_interval")
            missed_labels.append(dict(sounding_pitch=event["pitch"], onset=start, offset=end,
                                      cls="missed", sub=sub, copy=0, tie_prev=False, reference_index=index))
        reference_labels.append(dict(index=index, sounding_pitch=event["pitch"], onset=event["start"],
                                     offset=event["end"], cls=cls, sub=sub, perf_index=perf_index if cls == "correct" else None,
                                     tie_prev=False, clean_indices=members))
    if sum(p["cls"] == "correct" for p in perf_labels) + len(missed_labels) != len(reference_labels):
        raise SupervisionError("reference_partition_mismatch")
    return dict(schema_version="1.0", performance_notes=perf_labels,
                missed_notes=missed_labels, reference_notes=reference_labels,
                check={"perf": {"ok": True}, "ref": {"ok": True}},
                exporter=EXPORTER_VERSION,
                missing_timing="tagged injected rest through final performance MIDI tempo map; substitutions use performed event time")


def export_bundle(bundle: Path, config_path: Path) -> dict:
    config = SynthConfig.load(config_path)
    with tempfile.TemporaryDirectory(prefix="baseline-supervision-") as directory:
        lineage, reference, performance, missing = replay(bundle, config, Path(directory))
        labels = make_labels(lineage, reference, performance, missing)
    labels["provenance"] = dict(
        source_bundle=str(bundle.resolve()), config=str(config_path.resolve()),
        source_sha256={name: hashlib.sha256((bundle / name).read_bytes()).hexdigest()
                       for name in ("note_map.json", "reference_audio.mid", "performance_audio.mid", "metadata.json")},
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
    )
    return labels
