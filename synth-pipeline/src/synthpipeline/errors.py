from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field, replace

from music21 import note, pitch, stream

from synthpipeline.config import SynthConfig

ERROR_TYPES = (
    "wrong_note",
    "missed_note",
    "extra_note",
    "rhythm_error",
    "intonation_error",
)


class InjectionError(RuntimeError):
    pass


@dataclass
class PlannedLabel:
    type: str
    ql_start: float
    ql_end: float
    comment: str
    midi_pitch: int | None = None
    note_index: int | None = None
    note_count: int | None = None
    measure_number: int | None = None
    deviation_cents: float | None = None
    repeats_ql_start: float | None = None
    repeats_ql_end: float | None = None


@dataclass
class ErrorResult:
    score: stream.Score
    labels: list[PlannedLabel]
    error_type: str
    repeated: bool
    bpm: float
    extra: dict = field(default_factory=dict)


def inject_error(score: stream.Score, rng: random.Random, config: SynthConfig) -> ErrorResult:
    weights_cfg = dict(config.errors.get("weights") or {})
    types = [t for t in ERROR_TYPES if float(weights_cfg.get(t, 0.0)) > 0]
    if not types:
        types = list(ERROR_TYPES)
    weight_vals = [float(weights_cfg.get(t, 1.0)) for t in types]
    chosen = rng.choices(types, weights=weight_vals, k=1)[0]
    to_try = [chosen] + [t for t in types if t != chosen]

    last_error: Exception | None = None
    for error_type in to_try:
        attempt = copy.deepcopy(score)
        try:
            result = _apply_error(attempt, error_type, rng, config)
            rep_prob = float(config.errors.get("repetition_prob", 0.35))
            if rng.random() < rep_prob:
                try:
                    result = _repeat_error_measures(result)
                except InjectionError:
                    pass
            return result
        except InjectionError as exc:
            last_error = exc
            continue
    raise InjectionError(f"Could not inject any error: {last_error}")


def _apply_error(
    score: stream.Score,
    error_type: str,
    rng: random.Random,
    config: SynthConfig,
) -> ErrorResult:
    bpm = _score_bpm(score)
    if error_type == "wrong_note":
        return _wrong_note(score, rng, bpm, config)
    if error_type == "missed_note":
        return _missed_note(score, rng, bpm)
    if error_type == "extra_note":
        return _extra_note(score, rng, bpm, config)
    if error_type == "rhythm_error":
        return _rhythm_error(score, rng, bpm)
    if error_type == "intonation_error":
        return _intonation_error(score, rng, bpm, config)
    raise InjectionError(f"Unknown error type {error_type}")


def _wrong_note(
    score: stream.Score, rng: random.Random, bpm: float, config: SynthConfig
) -> ErrorResult:
    target = _pick_note(score, rng, min_ql=0.0)
    orig_midi = target.pitch.midi
    lo, hi = _pitch_bounds(config)
    semis = rng.choice([-2, -1, 1, 2])
    new_midi = orig_midi + semis
    if new_midi < lo or new_midi > hi:
        semis = -semis
        new_midi = orig_midi + semis
    new_midi = max(lo, min(hi, new_midi))
    if new_midi == orig_midi:
        new_midi = orig_midi + 1 if orig_midi < hi else orig_midi - 1
    target.pitch = pitch.Pitch(midi=new_midi)
    ql_start, ql_end = _element_ql_span(target, score)
    label = PlannedLabel(
        type="wrong_note",
        ql_start=ql_start,
        ql_end=ql_end,
        midi_pitch=new_midi,
        note_index=_sounding_index(score, target),
        measure_number=_measure_number(target),
        comment=f"shifted {semis:+d} semitones ({orig_midi} -> {new_midi})",
    )
    return ErrorResult(
        score=score,
        labels=[label],
        error_type="wrong_note",
        repeated=False,
        bpm=bpm,
        extra={"target_offset_in_measure": float(target.offset), "target_midi": new_midi},
    )


def _missed_note(score: stream.Score, rng: random.Random, bpm: float) -> ErrorResult:
    target = _pick_note(score, rng, min_ql=0.0)
    parent = target.activeSite
    if parent is None:
        raise InjectionError("Note has no parent site")
    ql_start, ql_end = _element_ql_span(target, score)
    measure_number = _measure_number(target)
    orig_midi = target.pitch.midi
    off = float(target.offset)
    dur = float(target.duration.quarterLength)
    parent.remove(target)
    rest = note.Rest(quarterLength=dur)
    parent.insert(off, rest)
    label = PlannedLabel(
        type="missed_note",
        ql_start=ql_start,
        ql_end=ql_end,
        midi_pitch=None,
        note_index=None,
        measure_number=measure_number,
        comment=f"replaced MIDI {orig_midi} with rest",
    )
    return ErrorResult(
        score=score,
        labels=[label],
        error_type="missed_note",
        repeated=False,
        bpm=bpm,
    )


def _extra_note(
    score: stream.Score, rng: random.Random, bpm: float, config: SynthConfig
) -> ErrorResult:
    target = _pick_note(score, rng, min_ql=0.5)
    parent = target.activeSite
    if parent is None:
        raise InjectionError("Note has no parent site")
    lo, hi = _pitch_bounds(config)
    half = float(target.duration.quarterLength) / 2.0
    if half < 0.125:
        raise InjectionError("Note too short to split")
    neighbor = _neighbor_midi(target.pitch.midi, rng, lo, hi)
    target.duration.quarterLength = half
    extra = note.Note(pitch.Pitch(midi=neighbor), quarterLength=half)
    parent.insert(float(target.offset) + half, extra)
    ql_start, _ = _element_ql_span(target, score)
    _, extra_end = _element_ql_span(extra, score)
    label = PlannedLabel(
        type="extra_note",
        ql_start=ql_start + half,
        ql_end=extra_end,
        midi_pitch=neighbor,
        note_index=_sounding_index(score, extra),
        measure_number=_measure_number(extra) or _measure_number(target),
        comment=f"inserted neighbor MIDI {neighbor} by splitting a note",
    )
    return ErrorResult(
        score=score,
        labels=[label],
        error_type="extra_note",
        repeated=False,
        bpm=bpm,
    )


def _intonation_error(
    score: stream.Score, rng: random.Random, bpm: float, config: SynthConfig
) -> ErrorResult:
    """Keep written pitch class; detune audio via MIDI pitch bend (cents)."""
    cfg = dict(config.errors.get("intonation") or {})
    cents_min = float(cfg.get("cents_min", 40.0))
    cents_max = float(cfg.get("cents_max", 80.0))
    if cents_max < cents_min:
        cents_min, cents_max = cents_max, cents_min
    group_prob = float(cfg.get("group_prob", 0.45))
    group_max = max(1, int(cfg.get("group_max", 4)))
    n_notes = 1
    if group_max > 1 and rng.random() < group_prob:
        n_notes = rng.randint(2, group_max)
    chosen = _pick_note_span(score, rng, n_notes)
    sign = rng.choice((-1.0, 1.0))
    cents = round(sign * rng.uniform(cents_min, cents_max), 1)
    ql_start, _ = _element_ql_span(chosen[0], score)
    _, ql_end = _element_ql_span(chosen[-1], score)
    midis = [int(n.pitch.midi) for n in chosen]
    label = PlannedLabel(
        type="intonation_error",
        ql_start=ql_start,
        ql_end=ql_end,
        midi_pitch=midis[0],
        note_index=_sounding_index(score, chosen[0]),
        note_count=len(chosen),
        measure_number=_measure_number(chosen[0]),
        deviation_cents=cents,
        comment=(
            f"detuned {cents:+.1f} cents across {len(chosen)} note(s) "
            f"(MIDI {', '.join(str(m) for m in midis)})"
        ),
    )
    return ErrorResult(
        score=score,
        labels=[label],
        error_type="intonation_error",
        repeated=False,
        bpm=bpm,
        extra={
            "pitch_bends": [
                {"ql_start": ql_start, "ql_end": ql_end, "cents": cents}
            ]
        },
    )


def _rhythm_error(score: stream.Score, rng: random.Random, bpm: float) -> ErrorResult:
    notes = _candidate_notes(score)
    dotted = _try_dotted_pair(score, notes, rng)
    if dotted is not None:
        return dotted

    target = _pick_note(score, rng, min_ql=0.5)
    parent = target.activeSite
    if parent is None:
        raise InjectionError("Note has no parent site")
    orig = float(target.duration.quarterLength)
    new_dur = orig / 2.0
    if new_dur < 0.125:
        raise InjectionError("Cannot shorten note further")
    rest_dur = orig - new_dur
    target.duration.quarterLength = new_dur
    rest = note.Rest(quarterLength=rest_dur)
    parent.insert(float(target.offset) + new_dur, rest)
    ql_start, _ = _element_ql_span(target, score)
    rest_start, rest_end = _element_ql_span(rest, score)
    label = PlannedLabel(
        type="rhythm_error",
        ql_start=ql_start,
        ql_end=rest_end,
        midi_pitch=target.pitch.midi,
        note_index=_sounding_index(score, target),
        measure_number=_measure_number(target),
        comment=f"shortened {orig}ql to {new_dur}ql with compensatory rest",
    )
    return ErrorResult(
        score=score,
        labels=[label],
        error_type="rhythm_error",
        repeated=False,
        bpm=bpm,
    )


def _try_dotted_pair(
    score: stream.Score, notes: list[note.Note], rng: random.Random
) -> ErrorResult | None:
    pairs: list[tuple[note.Note, note.Note]] = []
    for a, b in zip(notes, notes[1:]):
        if a.activeSite is not b.activeSite:
            continue
        da = float(a.duration.quarterLength)
        db = float(b.duration.quarterLength)
        if abs(da - db) < 1e-9 and da in {0.5, 1.0}:
            pairs.append((a, b))
    if not pairs:
        return None
    a, b = rng.choice(pairs)
    unit = float(a.duration.quarterLength)
    a.duration.quarterLength = unit * 1.5
    b.duration.quarterLength = unit * 0.5
    # Keep b starting where a now ends (same measure offsets).
    parent = a.activeSite
    if parent is not None:
        old_b = float(b.offset)
        new_b = float(a.offset) + float(a.duration.quarterLength)
        if abs(old_b - new_b) > 1e-9:
            parent.remove(b)
            parent.insert(new_b, b)
    ql_start, _ = _element_ql_span(a, score)
    _, ql_end = _element_ql_span(b, score)
    bpm = _score_bpm(score)
    label = PlannedLabel(
        type="rhythm_error",
        ql_start=ql_start,
        ql_end=ql_end,
        midi_pitch=a.pitch.midi,
        note_index=_sounding_index(score, a),
        measure_number=_measure_number(a),
        comment=f"dotted pair {unit}+{unit} -> {unit * 1.5}+{unit * 0.5}",
    )
    return ErrorResult(
        score=score,
        labels=[label],
        error_type="rhythm_error",
        repeated=False,
        bpm=bpm,
    )


def _repeat_error_measures(result: ErrorResult) -> ErrorResult:
    """Replay the measure(s) that contain the injected error, then continue."""
    score = result.score
    error_labels = [lb for lb in result.labels if lb.type != "repetition"]
    if not error_labels:
        raise InjectionError("No error label to repeat")
    part = score.parts[0]
    measures = list(part.getElementsByClass(stream.Measure))
    if not measures:
        raise InjectionError("Score has no measures to repeat")

    indices = set()
    for label in error_labels:
        indices.add(_measure_index_for_ql(part, label.ql_start))
        indices.add(_measure_index_for_ql(part, max(label.ql_start, label.ql_end - 1e-6)))
    start_idx = min(indices)
    end_idx = max(indices)
    n_block = end_idx - start_idx + 1

    rebuilt: list[stream.Measure] = []
    for i, measure in enumerate(measures):
        rebuilt.append(copy.deepcopy(measure))
        if i == end_idx:
            for j in range(start_idx, end_idx + 1):
                rebuilt.append(copy.deepcopy(measures[j]))

    for existing in list(part.getElementsByClass(stream.Measure)):
        part.remove(existing)
    for i, measure in enumerate(rebuilt, start=1):
        measure.number = i
        part.append(measure)

    measures = list(part.getElementsByClass(stream.Measure))
    orig_first = measures[start_idx]
    orig_last = measures[end_idx]
    dup_first = measures[end_idx + 1]
    dup_last = measures[end_idx + n_block]
    orig_start, _ = _element_ql_span(orig_first, score)
    _, orig_end = _element_ql_span(orig_last, score)
    dup_start, _ = _element_ql_span(dup_first, score)
    _, dup_end = _element_ql_span(dup_last, score)
    shift = dup_start - orig_start
    if shift <= 1e-9:
        raise InjectionError("Repeated span has zero duration")
    n_in_block = _sounding_count_in_span(score, orig_start, orig_end)

    labels: list[PlannedLabel] = []
    for label in error_labels:
        labels.append(
            replace(
                label,
                comment=_with_pass_suffix(label.comment, "first pass"),
                measure_number=_measure_number_at_ql(part, label.ql_start),
            )
        )
        second_index = label.note_index
        if second_index is not None and second_index >= 0:
            second_index = label.note_index + n_in_block
        labels.append(
            replace(
                label,
                ql_start=label.ql_start + shift,
                ql_end=label.ql_end + shift,
                note_index=second_index,
                comment=_with_pass_suffix(label.comment, "repeated pass"),
                measure_number=_measure_number_at_ql(part, label.ql_start + shift),
            )
        )

    window = "measure" if n_block == 1 else "measures"
    labels.append(
        PlannedLabel(
            type="repetition",
            ql_start=dup_start,
            ql_end=max(dup_end, dup_start + 0.25),
            midi_pitch=None,
            note_index=None,
            measure_number=int(dup_first.number) if dup_first.number else None,
            comment=f"repeated {window} containing {result.error_type}",
            repeats_ql_start=orig_start,
            repeats_ql_end=max(orig_end, orig_start + 0.25),
        )
    )

    extra = dict(result.extra)
    bends = list(extra.get("pitch_bends") or [])
    if bends:
        extra["pitch_bends"] = list(bends) + [
            {
                "ql_start": float(bend["ql_start"]) + shift,
                "ql_end": float(bend["ql_end"]) + shift,
                "cents": bend["cents"],
            }
            for bend in bends
        ]
    return ErrorResult(
        score=score,
        labels=labels,
        error_type=result.error_type,
        repeated=True,
        bpm=result.bpm,
        extra=extra,
    )


def _candidate_notes(score: stream.Score) -> list[note.Note]:
    notes = [
        n
        for n in score.recurse().getElementsByClass(note.Note)
        if not n.duration.isGrace
    ]
    if not notes:
        raise InjectionError("Score has no notes")
    return notes


def _pick_note(score: stream.Score, rng: random.Random, min_ql: float) -> note.Note:
    notes = [n for n in _candidate_notes(score) if float(n.duration.quarterLength) >= min_ql]
    if not notes:
        raise InjectionError(f"No notes with duration >= {min_ql}")
    if len(notes) >= 3:
        interior = notes[1:-1]
        eligible = [n for n in interior if float(n.duration.quarterLength) >= min_ql]
        if eligible:
            notes = eligible
    return rng.choice(notes)


def _pick_note_span(score: stream.Score, rng: random.Random, n_notes: int) -> list[note.Note]:
    notes = _candidate_notes(score)
    n_notes = max(1, min(int(n_notes), len(notes)))
    max_start = len(notes) - n_notes
    starts = list(range(0, max_start + 1))
    if len(notes) >= 3:
        interior = [
            i for i in starts if i > 0 and (i + n_notes) < len(notes)
        ]
        if interior:
            starts = interior
    start = rng.choice(starts)
    return notes[start : start + n_notes]


def _sounding_count_in_span(score: stream.Score, ql_start: float, ql_end: float) -> int:
    count = 0
    for item in _candidate_notes(score):
        start, _ = _element_ql_span(item, score)
        if ql_start - 1e-6 <= start < ql_end - 1e-9:
            count += 1
    return count


def _measure_number_at_ql(part: stream.Part, ql: float) -> int | None:
    measures = list(part.getElementsByClass(stream.Measure))
    if not measures:
        return None
    idx = _measure_index_for_ql(part, ql)
    number = getattr(measures[idx], "number", None)
    return int(number) if number else None


def _with_pass_suffix(comment: str, suffix: str) -> str:
    text = comment or ""
    if f"({suffix})" in text:
        return text
    return f"{text} ({suffix})" if text else suffix


def _sounding_index(score: stream.Score, target: note.Note) -> int:
    notes = _candidate_notes(score)
    for i, n in enumerate(notes):
        if n is target:
            return i
    return -1


def _measure_number(el) -> int | None:
    measure = el.getContextByClass(stream.Measure)
    if measure is None:
        return None
    number = getattr(measure, "number", None)
    return int(number) if number else None


def _element_ql_span(el, score: stream.Score) -> tuple[float, float]:
    try:
        start = float(el.getOffsetInHierarchy(score))
    except Exception:
        start = float(el.offset)
    dur = float(getattr(el.duration, "quarterLength", 0.0) or 0.0)
    if dur <= 0 and isinstance(el, stream.Measure):
        bar = getattr(el, "barDuration", None)
        if bar is not None:
            dur = float(bar.quarterLength)
    return start, start + dur


def _measure_index_for_ql(part: stream.Part, ql: float) -> int:
    measures = list(part.getElementsByClass(stream.Measure))
    for i, measure in enumerate(measures):
        try:
            start = float(measure.getOffsetInHierarchy(part))
        except Exception:
            start = float(measure.offset)
        dur = float(measure.duration.quarterLength) or float(measure.barDuration.quarterLength)
        if start - 1e-6 <= ql < start + dur - 1e-9:
            return i
    if not measures:
        raise InjectionError("No measures")
    return min(len(measures) - 1, max(0, 1))


def _score_bpm(score: stream.Score) -> float:
    from music21 import tempo

    for mark in score.flatten().getElementsByClass(tempo.MetronomeMark):
        if mark.number:
            return float(mark.number)
    return 120.0


def _pitch_bounds(config: SynthConfig) -> tuple[int, int]:
    gen = config.generation
    lo = pitch.Pitch(str(gen.get("pitch_min", "E3"))).midi
    hi = pitch.Pitch(str(gen.get("pitch_max", "C6"))).midi
    return lo, hi


def _neighbor_midi(midi: int, rng: random.Random, lo: int, hi: int) -> int:
    delta = rng.choice([-2, -1, 1, 2])
    value = max(lo, min(hi, midi + delta))
    if value == midi:
        value = midi + 1 if midi < hi else midi - 1
    return value
