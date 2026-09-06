from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field, replace

from music21 import note, pitch, stream, tempo

from datacreate.melody import notes_in_measures, parse_sounding_notes
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
    clean_note_index: int | None = None
    clean_note_count: int | None = None
    extra_copies: int | None = None


@dataclass
class ErrorResult:
    score: stream.Score
    labels: list[PlannedLabel]
    error_type: str
    repeated: bool
    bpm: float
    extra: dict = field(default_factory=dict)


def inject_error(score: stream.Score, rng: random.Random, config: SynthConfig) -> ErrorResult:
    clean_notes = parse_sounding_notes(score)
    per_min = max(1, int(config.errors.get("per_clip_min", 1)))
    per_max = max(per_min, int(config.errors.get("per_clip_max", 1)))
    n_errors = rng.randint(per_min, per_max)
    working = copy.deepcopy(score)
    labels: list[PlannedLabel] = []
    error_types: list[str] = []
    extra: dict = {"pitch_bends": [], "error_types": [], "extra_copies": 0}
    used_spans: list[tuple[float, float]] = []
    last_error: Exception | None = None

    for _ in range(n_errors):
        planted = None
        for error_type in _error_order(rng, config):
            attempt = copy.deepcopy(working)
            try:
                planted = _apply_error(
                    attempt, error_type, rng, config, clean_notes, used_spans
                )
                break
            except InjectionError as exc:
                last_error = exc
                continue
        if planted is None:
            if labels:
                break
            raise InjectionError(f"Could not inject any error: {last_error}")
        working = planted.score
        labels.extend(planted.labels)
        error_types.append(planted.error_type)
        for lab in planted.labels:
            used_spans.append((lab.ql_start, lab.ql_end))
        extra["pitch_bends"].extend(list((planted.extra or {}).get("pitch_bends") or []))
        for key, value in (planted.extra or {}).items():
            if key != "pitch_bends":
                extra[key] = value

    extra["error_types"] = error_types
    result = ErrorResult(
        score=working,
        labels=labels,
        error_type=error_types[0] if error_types else "repetition",
        repeated=False,
        bpm=_score_bpm(working),
        extra=extra,
    )
    gap_seconds = _repeat_gap_seconds(rng, config)
    copies = _weighted_int(rng, config.errors.get("repeat_extra_copies_weights") or {1: 1.0})
    rep_prob = float(config.errors.get("repetition_prob", 0.35))
    solo_prob = float(config.errors.get("standalone_repetition_prob", 0.0))
    if labels and rng.random() < rep_prob:
        try:
            result = _repeat_error_measures(
                result,
                extra_copies=copies,
                clean_notes=clean_notes,
                gap_seconds=gap_seconds,
            )
            result.extra["extra_copies"] = copies
            result.extra["error_types"] = error_types
        except InjectionError:
            pass
    if not result.repeated and rng.random() < solo_prob:
        try:
            result = _standalone_repetition(
                result,
                rng,
                extra_copies=copies,
                clean_notes=clean_notes,
                gap_seconds=gap_seconds,
            )
            result.extra["extra_copies"] = copies
            result.extra["error_types"] = list(error_types) + ["repetition"]
            result.extra["standalone_repetition"] = True
        except InjectionError:
            pass
    return result


def _error_order(rng: random.Random, config: SynthConfig) -> list[str]:
    weights_cfg = dict(config.errors.get("weights") or {})
    types = [t for t in ERROR_TYPES if float(weights_cfg.get(t, 0.0)) > 0]
    if not types:
        types = list(ERROR_TYPES)
    weight_vals = [float(weights_cfg.get(t, 1.0)) for t in types]
    chosen = rng.choices(types, weights=weight_vals, k=1)[0]
    return [chosen] + [t for t in types if t != chosen]


def _weighted_int(rng: random.Random, weights: dict) -> int:
    keys = [int(k) for k in weights]
    vals = [max(0.0, float(weights[k])) for k in weights]
    if not keys or sum(vals) <= 0:
        return 1
    return int(rng.choices(keys, weights=vals, k=1)[0])


def _apply_error(
    score: stream.Score,
    error_type: str,
    rng: random.Random,
    config: SynthConfig,
    clean_notes=None,
    used_spans: list[tuple[float, float]] | None = None,
) -> ErrorResult:
    bpm = _score_bpm(score)
    used = used_spans or []
    if error_type == "wrong_note":
        return _wrong_note(score, rng, bpm, config, clean_notes, used)
    if error_type == "missed_note":
        return _missed_note(score, rng, bpm, clean_notes, used)
    if error_type == "extra_note":
        return _extra_note(score, rng, bpm, config, clean_notes, used)
    if error_type == "rhythm_error":
        return _rhythm_error(score, rng, bpm, config, clean_notes, used)
    if error_type == "intonation_error":
        return _intonation_error(score, rng, bpm, config, clean_notes, used)
    raise InjectionError(f"Unknown error type {error_type}")


def _wrong_note(
    score: stream.Score,
    rng: random.Random,
    bpm: float,
    config: SynthConfig,
    clean_notes=None,
    used_spans: list[tuple[float, float]] | None = None,
) -> ErrorResult:
    target = _pick_note(score, rng, min_ql=0.0, used_spans=used_spans)
    orig_midi = target.pitch.midi
    if _use_squeak(rng, config):
        new_midi = _squeak_midi(rng, config)
        comment = f"squeak MIDI {new_midi} (was {orig_midi})"
    else:
        lo, hi = _pitch_bounds(config)
        semis = rng.choice([-2, -1, 1, 2])
        new_midi = orig_midi + semis
        if new_midi < lo or new_midi > hi:
            semis = -semis
            new_midi = orig_midi + semis
        new_midi = max(lo, min(hi, new_midi))
        if new_midi == orig_midi:
            new_midi = orig_midi + 1 if orig_midi < hi else orig_midi - 1
        comment = f"shifted {semis:+d} semitones ({orig_midi} -> {new_midi})"
    target.pitch = pitch.Pitch(midi=new_midi)
    ql_start, ql_end = _element_ql_span(target, score)
    label = PlannedLabel(
        type="wrong_note",
        ql_start=ql_start,
        ql_end=ql_end,
        midi_pitch=new_midi,
        note_index=_sounding_index(score, target),
        measure_number=_measure_number(target),
        comment=comment,
    )
    _set_clean(label, clean_notes, ql_start, orig_midi)
    return ErrorResult(
        score=score,
        labels=[label],
        error_type="wrong_note",
        repeated=False,
        bpm=bpm,
        extra={"target_offset_in_measure": float(target.offset), "target_midi": new_midi},
    )


def _missed_note(
    score: stream.Score,
    rng: random.Random,
    bpm: float,
    clean_notes=None,
    used_spans: list[tuple[float, float]] | None = None,
) -> ErrorResult:
    target = _pick_note(score, rng, min_ql=0.0, used_spans=used_spans)
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
    _set_clean(label, clean_notes, ql_start, orig_midi)
    return ErrorResult(
        score=score,
        labels=[label],
        error_type="missed_note",
        repeated=False,
        bpm=bpm,
    )


def _extra_note(
    score: stream.Score,
    rng: random.Random,
    bpm: float,
    config: SynthConfig,
    clean_notes=None,
    used_spans: list[tuple[float, float]] | None = None,
) -> ErrorResult:
    target = _pick_note(score, rng, min_ql=0.5, used_spans=used_spans)
    parent = target.activeSite
    if parent is None:
        raise InjectionError("Note has no parent site")
    lo, hi = _pitch_bounds(config)
    half = float(target.duration.quarterLength) / 2.0
    if half < 0.125:
        raise InjectionError("Note too short to split")
    if _use_squeak(rng, config):
        inserted = _squeak_midi(rng, config)
        comment = f"inserted squeak MIDI {inserted} by splitting a note"
    else:
        inserted = _neighbor_midi(target.pitch.midi, rng, lo, hi)
        comment = f"inserted neighbor MIDI {inserted} by splitting a note"
    target.duration.quarterLength = half
    extra = note.Note(pitch.Pitch(midi=inserted), quarterLength=half)
    parent.insert(float(target.offset) + half, extra)
    ql_start, _ = _element_ql_span(target, score)
    _, extra_end = _element_ql_span(extra, score)
    label = PlannedLabel(
        type="extra_note",
        ql_start=ql_start + half,
        ql_end=extra_end,
        midi_pitch=inserted,
        note_index=_sounding_index(score, extra),
        measure_number=_measure_number(extra) or _measure_number(target),
        comment=comment,
    )
    extra_count = 1
    if clean_notes:
        hit = min(clean_notes, key=lambda n: abs(n.ql_start - ql_start))
        if hit.index + 1 < len(clean_notes):
            extra_count = 2
    _set_clean(label, clean_notes, ql_start, int(target.pitch.midi), count=extra_count)
    return ErrorResult(
        score=score,
        labels=[label],
        error_type="extra_note",
        repeated=False,
        bpm=bpm,
    )


def _intonation_error(
    score: stream.Score,
    rng: random.Random,
    bpm: float,
    config: SynthConfig,
    clean_notes=None,
    used_spans: list[tuple[float, float]] | None = None,
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
    chosen = _pick_note_span(score, rng, n_notes, used_spans=used_spans)
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
    _set_clean(label, clean_notes, ql_start, midis[0], count=len(chosen))
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


def _rhythm_error(
    score: stream.Score,
    rng: random.Random,
    bpm: float,
    config: SynthConfig,
    clean_notes=None,
    used_spans: list[tuple[float, float]] | None = None,
) -> ErrorResult:
    kinds = list(config.errors.get("rhythm_kinds") or [])
    if not kinds:
        kinds = [
            "late_start",
            "early_start",
            "late_end",
            "early_end",
            "tempo_change",
            "uneven",
        ]
    order = list(kinds)
    rng.shuffle(order)
    last_exc: Exception | None = None
    for kind in order:
        try:
            if kind == "late_start":
                return _rhythm_late_start(score, rng, bpm, clean_notes, used_spans)
            if kind == "early_start":
                return _rhythm_early_start(score, rng, bpm, clean_notes, used_spans)
            if kind == "late_end":
                return _rhythm_late_end(score, rng, bpm, clean_notes, used_spans)
            if kind == "early_end":
                return _rhythm_early_end(score, rng, bpm, clean_notes, used_spans)
            if kind == "tempo_change":
                return _rhythm_tempo_change(score, rng, bpm, clean_notes, used_spans)
            if kind == "uneven":
                return _rhythm_uneven(score, rng, bpm, clean_notes, used_spans)
            if kind == "dotted":
                dotted = _try_dotted_pair(
                    score, _candidate_notes(score), rng, clean_notes, used_spans
                )
                if dotted is not None:
                    return dotted
        except InjectionError as exc:
            last_exc = exc
            continue
    dotted = _try_dotted_pair(score, _candidate_notes(score), rng, clean_notes, used_spans)
    if dotted is not None:
        return dotted
    try:
        return _rhythm_early_end(score, rng, bpm, clean_notes, used_spans)
    except InjectionError as exc:
        raise InjectionError(f"Could not plant rhythm error: {last_exc or exc}") from exc


def _rhythm_shift_ql(rng: random.Random, orig: float) -> float:
    choices = [q for q in (0.125, 0.25, 0.5) if q <= orig * 0.5 and orig - q >= 0.125]
    if not choices:
        raise InjectionError("Note too short to shift")
    return rng.choice(choices)


def _rhythm_late_start(
    score: stream.Score,
    rng: random.Random,
    bpm: float,
    clean_notes=None,
    used_spans: list[tuple[float, float]] | None = None,
) -> ErrorResult:
    target = _pick_note(score, rng, min_ql=0.5, used_spans=used_spans)
    parent = target.activeSite
    if parent is None:
        raise InjectionError("Note has no parent site")
    orig = float(target.duration.quarterLength)
    shift = _rhythm_shift_ql(rng, orig)
    off = float(target.offset)
    parent.remove(target)
    target.duration.quarterLength = orig - shift
    parent.insert(off, note.Rest(quarterLength=shift))
    parent.insert(off + shift, target)
    return _rhythm_result(
        score,
        bpm,
        target,
        clean_notes,
        f"late start by {shift}ql (end unchanged)",
        extra_end_el=target,
    )


def _rhythm_early_end(
    score: stream.Score,
    rng: random.Random,
    bpm: float,
    clean_notes=None,
    used_spans: list[tuple[float, float]] | None = None,
) -> ErrorResult:
    target = _pick_note(score, rng, min_ql=0.5, used_spans=used_spans)
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
    return _rhythm_result(
        score,
        bpm,
        target,
        clean_notes,
        f"early end: shortened {orig}ql to {new_dur}ql",
        extra_end_el=rest,
    )


def _rhythm_late_end(
    score: stream.Score,
    rng: random.Random,
    bpm: float,
    clean_notes=None,
    used_spans: list[tuple[float, float]] | None = None,
) -> ErrorResult:
    target = _pick_note(score, rng, min_ql=0.25, used_spans=used_spans)
    parent = target.activeSite
    if parent is None:
        raise InjectionError("Note has no parent site")
    following = _next_rest(parent, target)
    if following is None:
        raise InjectionError("No following rest to extend into")
    steal = min(0.5, float(following.duration.quarterLength))
    if steal < 0.125:
        raise InjectionError("Following rest too short")
    target.duration.quarterLength = float(target.duration.quarterLength) + steal
    leftover = float(following.duration.quarterLength) - steal
    parent.remove(following)
    if leftover >= 0.0625:
        following.duration.quarterLength = leftover
        parent.insert(float(target.offset) + float(target.duration.quarterLength), following)
        end_el = following
    else:
        end_el = target
    return _rhythm_result(
        score, bpm, target, clean_notes, f"late end by {steal}ql", extra_end_el=end_el
    )


def _rhythm_early_start(
    score: stream.Score,
    rng: random.Random,
    bpm: float,
    clean_notes=None,
    used_spans: list[tuple[float, float]] | None = None,
) -> ErrorResult:
    target = _pick_note(score, rng, min_ql=0.25, used_spans=used_spans)
    parent = target.activeSite
    if parent is None:
        raise InjectionError("Note has no parent site")
    preceding = _prev_rest(parent, target)
    if preceding is None:
        raise InjectionError("No preceding rest to start earlier into")
    steal = min(0.5, float(preceding.duration.quarterLength))
    if steal < 0.125:
        raise InjectionError("Preceding rest too short")
    leftover = float(preceding.duration.quarterLength) - steal
    new_off = float(preceding.offset) + leftover
    parent.remove(target)
    parent.remove(preceding)
    if leftover >= 0.0625:
        preceding.duration.quarterLength = leftover
        parent.insert(float(preceding.offset), preceding)
    target.duration.quarterLength = float(target.duration.quarterLength) + steal
    parent.insert(new_off, target)
    return _rhythm_result(
        score, bpm, target, clean_notes, f"early start by {steal}ql", extra_end_el=target
    )


def _rhythm_tempo_change(
    score: stream.Score,
    rng: random.Random,
    bpm: float,
    clean_notes=None,
    used_spans: list[tuple[float, float]] | None = None,
) -> ErrorResult:
    if not score.parts:
        raise InjectionError("Score has no parts")
    part = score.parts[0]
    measures = list(part.getElementsByClass(stream.Measure))
    used = used_spans or []
    eligible = []
    for i, measure in enumerate(measures):
        notes = [n for n in measure.recurse().getElementsByClass(note.Note) if not n.duration.isGrace]
        if not notes:
            continue
        if used and _overlaps_used(score, measure, used):
            continue
        eligible.append(i)
    if not eligible:
        raise InjectionError("No measure for tempo change")
    start_idx = rng.choice(eligible)
    n_span = rng.choice((1, 2))
    end_idx = min(len(measures) - 1, start_idx + n_span)
    factor = rng.choice((rng.uniform(0.68, 0.82), rng.uniform(1.2, 1.4)))
    new_bpm = max(40.0, min(200.0, round(bpm * factor)))
    measures[start_idx].insert(0, tempo.MetronomeMark(number=new_bpm))
    if end_idx + 1 < len(measures):
        measures[end_idx + 1].insert(0, tempo.MetronomeMark(number=bpm))
    first = measures[start_idx]
    last = measures[end_idx]
    ql_start, _ = _element_ql_span(first, score)
    _, ql_end = _element_ql_span(last, score)
    first_note = next(
        n for n in first.recurse().getElementsByClass(note.Note) if not n.duration.isGrace
    )
    label = PlannedLabel(
        type="rhythm_error",
        ql_start=ql_start,
        ql_end=ql_end,
        midi_pitch=int(first_note.pitch.midi),
        note_index=_sounding_index(score, first_note),
        measure_number=_measure_number(first_note),
        comment=f"sudden tempo {bpm:.0f} -> {new_bpm:.0f} bpm for {end_idx - start_idx + 1} measure(s)",
    )
    _set_clean(label, clean_notes, ql_start, int(first_note.pitch.midi), count=max(1, end_idx - start_idx + 1))
    return ErrorResult(
        score=score,
        labels=[label],
        error_type="rhythm_error",
        repeated=False,
        bpm=bpm,
    )


def _rhythm_uneven(
    score: stream.Score,
    rng: random.Random,
    bpm: float,
    clean_notes=None,
    used_spans: list[tuple[float, float]] | None = None,
) -> ErrorResult:
    if not score.parts:
        raise InjectionError("Score has no parts")
    part = score.parts[0]
    measures = list(part.getElementsByClass(stream.Measure))
    used = used_spans or []
    cands = []
    for measure in measures:
        notes = [n for n in measure.notes if not n.duration.isGrace]
        if len(notes) < 3:
            continue
        if used and _overlaps_used(score, measure, used):
            continue
        cands.append((measure, notes))
    if not cands:
        dotted = _try_dotted_pair(score, _candidate_notes(score), rng, clean_notes, used_spans)
        if dotted is not None:
            return dotted
        raise InjectionError("No measure with 3+ notes for uneven rhythm")
    parent, notes = rng.choice(cands)
    n_take = min(len(notes), rng.randint(3, 4))
    start = rng.randint(0, len(notes) - n_take)
    chosen = notes[start : start + n_take]
    total = sum(float(n.duration.quarterLength) for n in chosen)
    new_durs = _uneven_expressible_durs(rng, total, len(chosen))
    if new_durs is None:
        dotted = _try_dotted_pair(score, _candidate_notes(score), rng, clean_notes, used_spans)
        if dotted is not None:
            return dotted
        raise InjectionError("Could not build expressible uneven durations")
    off = float(chosen[0].offset)
    for item, dur in zip(chosen, new_durs):
        parent.remove(item)
        item.duration.quarterLength = dur
        parent.insert(off, item)
        off += dur
    ql_start, _ = _element_ql_span(chosen[0], score)
    _, ql_end = _element_ql_span(chosen[-1], score)
    label = PlannedLabel(
        type="rhythm_error",
        ql_start=ql_start,
        ql_end=ql_end,
        midi_pitch=int(chosen[0].pitch.midi),
        note_index=_sounding_index(score, chosen[0]),
        note_count=len(chosen),
        measure_number=_measure_number(chosen[0]),
        comment=f"uneven rhythm across {len(chosen)} notes",
    )
    _set_clean(label, clean_notes, ql_start, int(chosen[0].pitch.midi), count=len(chosen))
    return ErrorResult(
        score=score,
        labels=[label],
        error_type="rhythm_error",
        repeated=False,
        bpm=bpm,
    )


def _rhythm_result(
    score: stream.Score,
    bpm: float,
    target: note.Note,
    clean_notes,
    comment: str,
    extra_end_el=None,
) -> ErrorResult:
    ql_start, ql_end = _element_ql_span(target, score)
    if extra_end_el is not None:
        _, ql_end = _element_ql_span(extra_end_el, score)
    label = PlannedLabel(
        type="rhythm_error",
        ql_start=ql_start,
        ql_end=ql_end,
        midi_pitch=target.pitch.midi,
        note_index=_sounding_index(score, target),
        measure_number=_measure_number(target),
        comment=comment,
    )
    _set_clean(label, clean_notes, ql_start, int(target.pitch.midi))
    return ErrorResult(
        score=score,
        labels=[label],
        error_type="rhythm_error",
        repeated=False,
        bpm=bpm,
    )


def _next_rest(parent, target) -> note.Rest | None:
    items = list(parent.notesAndRests)
    try:
        idx = items.index(target)
    except ValueError:
        return None
    if idx + 1 < len(items) and items[idx + 1].isRest:
        return items[idx + 1]
    return None


def _prev_rest(parent, target) -> note.Rest | None:
    items = list(parent.notesAndRests)
    try:
        idx = items.index(target)
    except ValueError:
        return None
    if idx > 0 and items[idx - 1].isRest:
        return items[idx - 1]
    return None


def _try_dotted_pair(
    score: stream.Score,
    notes: list[note.Note],
    rng: random.Random,
    clean_notes=None,
    used_spans: list[tuple[float, float]] | None = None,
) -> ErrorResult | None:
    pairs: list[tuple[note.Note, note.Note]] = []
    for a, b in zip(notes, notes[1:]):
        if a.activeSite is not b.activeSite:
            continue
        da = float(a.duration.quarterLength)
        db = float(b.duration.quarterLength)
        if abs(da - db) < 1e-9 and da in {0.5, 1.0}:
            pairs.append((a, b))
    used = used_spans or []
    if used:
        pairs = [
            (a, b)
            for a, b in pairs
            if not _overlaps_used(score, a, used) and not _overlaps_used(score, b, used)
        ]
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
    _set_clean(label, clean_notes, ql_start, int(a.pitch.midi), count=2)
    return ErrorResult(
        score=score,
        labels=[label],
        error_type="rhythm_error",
        repeated=False,
        bpm=bpm,
    )


def _repeat_error_measures(
    result: ErrorResult,
    extra_copies: int = 1,
    clean_notes=None,
    gap_seconds: float = 0.0,
) -> ErrorResult:
    """Replay the measure(s) that contain the injected error `extra_copies` times."""
    extra_copies = max(1, int(extra_copies))
    score = result.score
    error_labels = [lb for lb in result.labels if lb.type != "repetition"]
    if not error_labels:
        raise InjectionError("No error label to repeat")
    if not score.parts:
        raise InjectionError("Score has no parts")
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
    return _repeat_span(
        result,
        start_idx=start_idx,
        end_idx=end_idx,
        extra_copies=extra_copies,
        clean_notes=clean_notes,
        gap_seconds=gap_seconds,
        standalone=False,
    )


def _standalone_repetition(
    result: ErrorResult,
    rng: random.Random,
    extra_copies: int = 1,
    clean_notes=None,
    gap_seconds: float = 0.0,
) -> ErrorResult:
    if not result.score.parts:
        raise InjectionError("Score has no parts")
    part = result.score.parts[0]
    measures = list(part.getElementsByClass(stream.Measure))
    eligible = [
        i
        for i, measure in enumerate(measures)
        if any(not n.duration.isGrace for n in measure.recurse().getElementsByClass(note.Note))
    ]
    if not eligible:
        raise InjectionError("No measure to repeat on its own")
    start_idx = rng.choice(eligible)
    end_idx = start_idx
    if start_idx + 1 in eligible and rng.random() < 0.25:
        end_idx = start_idx + 1
    return _repeat_span(
        result,
        start_idx=start_idx,
        end_idx=end_idx,
        extra_copies=max(1, int(extra_copies)),
        clean_notes=clean_notes,
        gap_seconds=gap_seconds,
        standalone=True,
    )


def _repeat_span(
    result: ErrorResult,
    start_idx: int,
    end_idx: int,
    extra_copies: int,
    clean_notes=None,
    gap_seconds: float = 0.0,
    standalone: bool = False,
) -> ErrorResult:
    score = result.score
    part = score.parts[0]
    measures = list(part.getElementsByClass(stream.Measure))
    extra_copies = max(1, int(extra_copies))
    n_block = end_idx - start_idx + 1
    gap_ql = max(0.0, float(gap_seconds) * float(result.bpm or 120.0) / 60.0)
    error_labels = [] if standalone else [lb for lb in result.labels if lb.type != "repetition"]

    rebuilt: list[stream.Measure] = []
    for i, measure in enumerate(measures):
        rebuilt.append(copy.deepcopy(measure))
        if i == end_idx:
            if gap_ql >= 0.05:
                gap = stream.Measure()
                for rest_ql in _expressible_ql_parts(gap_ql):
                    gap.append(note.Rest(quarterLength=rest_ql))
                rebuilt.append(gap)
            for _ in range(extra_copies):
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
    orig_start, _ = _element_ql_span(orig_first, score)
    _, orig_end = _element_ql_span(orig_last, score)
    n_in_block = _sounding_count_in_span(score, orig_start, orig_end)
    if n_in_block <= 0:
        raise InjectionError("Repeated span has no sounding notes")

    labels: list[PlannedLabel] = []
    if not standalone:
        labels = [
            replace(
                label,
                comment=_with_pass_suffix(label.comment, "first pass"),
                measure_number=_measure_number_at_ql(part, label.ql_start),
            )
            for label in error_labels
        ]

    first_dup_start = None
    last_dup_end = None
    extra = dict(result.extra)
    original_bends = [] if standalone else list(extra.get("pitch_bends") or [])
    shifted_bends = list(original_bends)
    gap_offset = 1 if gap_ql >= 0.05 else 0

    for k in range(1, extra_copies + 1):
        copy_first = measures[end_idx + gap_offset + 1 + (k - 1) * n_block]
        copy_last = measures[end_idx + gap_offset + k * n_block]
        dup_start, _ = _element_ql_span(copy_first, score)
        _, dup_end = _element_ql_span(copy_last, score)
        shift = dup_start - orig_start
        if shift <= 1e-9:
            raise InjectionError("Repeated span has zero duration")
        if first_dup_start is None:
            first_dup_start = dup_start
        last_dup_end = dup_end
        suffix = "repeated pass" if extra_copies == 1 else f"pass {k + 1}"
        for label in error_labels:
            new_index = label.note_index
            if new_index is not None and new_index >= 0:
                new_index = label.note_index + k * n_in_block
            labels.append(
                replace(
                    label,
                    ql_start=label.ql_start + shift,
                    ql_end=label.ql_end + shift,
                    note_index=new_index,
                    comment=_with_pass_suffix(label.comment, suffix),
                    measure_number=_measure_number_at_ql(part, label.ql_start + shift),
                )
            )
        for bend in original_bends:
            shifted_bends.append(
                {
                    "ql_start": float(bend["ql_start"]) + shift,
                    "ql_end": float(bend["ql_end"]) + shift,
                    "cents": bend["cents"],
                }
            )

    if standalone:
        insert_shift = (last_dup_end or orig_end) - orig_end
        for lab in result.labels:
            if insert_shift > 1e-9 and lab.ql_start >= orig_end - 1e-9:
                labels.append(
                    replace(
                        lab,
                        ql_start=lab.ql_start + insert_shift,
                        ql_end=lab.ql_end + insert_shift,
                    )
                )
            else:
                labels.append(lab)
        for bend in extra.get("pitch_bends") or []:
            if insert_shift > 1e-9 and float(bend["ql_start"]) >= orig_end - 1e-9:
                bend["ql_start"] = float(bend["ql_start"]) + insert_shift
                bend["ql_end"] = float(bend["ql_end"]) + insert_shift

    window = "measure" if n_block == 1 else "measures"
    plays = extra_copies + 1
    if error_labels:
        clean_i0, clean_count = _clean_block_span(error_labels, clean_notes)
    else:
        clean_i0, clean_count = _clean_measure_span(
            clean_notes,
            int(orig_first.number) if orig_first.number else start_idx + 1,
            int(orig_last.number) if orig_last.number else end_idx + 1,
        )
    why = "on its own" if standalone else f"containing {result.error_type}"
    gap_note = f" after {gap_seconds:.2f}s rest" if gap_ql >= 0.05 else ""
    labels.append(
        PlannedLabel(
            type="repetition",
            ql_start=first_dup_start or orig_start,
            ql_end=max(last_dup_end or orig_end, (first_dup_start or orig_start) + 0.25),
            midi_pitch=None,
            note_index=None,
            measure_number=int(orig_first.number) if orig_first.number else None,
            comment=f"repeated {window} {why} ({plays} plays){gap_note}",
            repeats_ql_start=orig_start,
            repeats_ql_end=max(orig_end, orig_start + 0.25),
            clean_note_index=clean_i0,
            clean_note_count=clean_count,
            extra_copies=extra_copies,
        )
    )
    extra["pitch_bends"] = list(extra.get("pitch_bends") or []) if standalone else shifted_bends
    extra["extra_copies"] = extra_copies
    extra["repeat_gap_seconds"] = float(gap_seconds) if gap_ql >= 0.05 else 0.0
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


def _pick_note(
    score: stream.Score,
    rng: random.Random,
    min_ql: float,
    used_spans: list[tuple[float, float]] | None = None,
) -> note.Note:
    notes = [n for n in _candidate_notes(score) if float(n.duration.quarterLength) >= min_ql]
    if used_spans:
        notes = [n for n in notes if not _overlaps_used(score, n, used_spans)]
    if not notes:
        raise InjectionError(f"No notes with duration >= {min_ql}")
    if len(notes) >= 3:
        interior = notes[1:-1]
        eligible = [n for n in interior if float(n.duration.quarterLength) >= min_ql]
        if eligible:
            notes = eligible
    return rng.choice(notes)


def _pick_note_span(
    score: stream.Score,
    rng: random.Random,
    n_notes: int,
    used_spans: list[tuple[float, float]] | None = None,
) -> list[note.Note]:
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
    if used_spans:
        starts = [
            i
            for i in starts
            if not any(_overlaps_used(score, notes[j], used_spans) for j in range(i, i + n_notes))
        ]
    if not starts:
        raise InjectionError("No unused note span left")
    start = rng.choice(starts)
    return notes[start : start + n_notes]


def _overlaps_used(
    score: stream.Score, el, used_spans: list[tuple[float, float]]
) -> bool:
    start, end = _element_ql_span(el, score)
    return any(start < ue and us < end for us, ue in used_spans)


def _set_clean(
    label: PlannedLabel,
    clean_notes,
    ql: float,
    midi: int | None = None,
    count: int = 1,
) -> None:
    if not clean_notes:
        return
    hit = None
    if midi is not None:
        close = [
            n
            for n in clean_notes
            if abs(n.ql_start - ql) < 0.08 and n.pitch == int(midi)
        ]
        if close:
            hit = min(close, key=lambda n: abs(n.ql_start - ql))
    if hit is None:
        hit = min(clean_notes, key=lambda n: abs(n.ql_start - ql))
    label.clean_note_index = hit.index
    label.clean_note_count = max(1, int(count))


def _clean_block_span(error_labels: list[PlannedLabel], clean_notes) -> tuple[int | None, int | None]:
    if not clean_notes:
        return None, None
    measures = [lab.measure_number for lab in error_labels if lab.measure_number is not None]
    if measures:
        block = notes_in_measures(clean_notes, range(min(measures), max(measures) + 1))
        if block is not None:
            return block[0], block[1] - block[0]
    idxs = [lab.clean_note_index for lab in error_labels if lab.clean_note_index is not None]
    if not idxs:
        return None, None
    last = max(
        (lab.clean_note_index or 0) + (lab.clean_note_count or 1)
        for lab in error_labels
        if lab.clean_note_index is not None
    )
    return min(idxs), last - min(idxs)


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


def _use_squeak(rng: random.Random, config: SynthConfig) -> bool:
    cfg = dict(config.errors.get("squeak") or {})
    return rng.random() < float(cfg.get("prob", 0.0))


def _squeak_midi(rng: random.Random, config: SynthConfig) -> int:
    cfg = dict(config.errors.get("squeak") or {})
    lo = pitch.Pitch(str(cfg.get("pitch_min", "C6"))).midi
    hi = pitch.Pitch(str(cfg.get("pitch_max", "A7"))).midi
    if hi < lo:
        lo, hi = hi, lo
    return int(rng.randint(lo, hi))


def _repeat_gap_seconds(rng: random.Random, config: SynthConfig) -> float:
    raw = config.errors.get("repeat_gap_seconds")
    if raw is None:
        return 0.0
    if isinstance(raw, (int, float)):
        return max(0.0, float(raw))
    values = [float(x) for x in raw]
    if not values:
        return 0.0
    lo, hi = min(values), max(values)
    if hi <= 0:
        return 0.0
    return float(rng.uniform(lo, hi))


def _clean_measure_span(clean_notes, start_measure: int, end_measure: int) -> tuple[int | None, int | None]:
    if not clean_notes:
        return None, None
    block = notes_in_measures(clean_notes, range(int(start_measure), int(end_measure) + 1))
    if block is None:
        return None, None
    return block[0], block[1] - block[0]


MUSICXML_QL = (0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)


def snap_musicxml_ql(ql: float) -> float:
    return min(MUSICXML_QL, key=lambda q: (abs(q - float(ql)), q))


def _expressible_ql_parts(target_ql: float) -> list[float]:
    remain = max(0.125, float(target_ql))
    parts: list[float] = []
    for q in sorted(MUSICXML_QL, reverse=True):
        while remain + 1e-9 >= q:
            parts.append(q)
            remain -= q
            if remain < 0.125 - 1e-9:
                break
    if remain >= 0.08:
        parts.append(snap_musicxml_ql(remain))
    return parts or [0.25]


def _uneven_expressible_durs(rng: random.Random, total: float, k: int) -> list[float] | None:
    allowed = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
    base = max(k, int(round(float(total) / 0.125)))
    for delta in (0, 1, -1, 2, -2, 3, -3, 4, -4):
        units = base + delta
        if units < k:
            continue
        parts = _partition_units(rng, units, k, allowed)
        if parts:
            return [p * 0.125 for p in parts]
    return None


def _partition_units(
    rng: random.Random, total: int, k: int, allowed: tuple[int, ...]
) -> list[int] | None:
    allowed_desc = tuple(sorted(allowed, reverse=True))

    def rec(remain: int, left: int) -> list[int] | None:
        if left == 1:
            return [remain] if remain in allowed else None
        choices = [u for u in allowed_desc if 1 <= u <= remain - (left - 1)]
        rng.shuffle(choices)
        for unit in choices:
            rest = rec(remain - unit, left - 1)
            if rest is not None:
                return [unit] + rest
        return None

    return rec(int(total), int(k))


def ensure_expressible_durations(score: stream.Score) -> None:
    """Snap any MusicXML-inexpressible note/rest to the nearest legal type."""
    from music21 import duration as m21dur

    for el in score.recurse().getElementsByClass((note.Note, note.Rest)):
        ql = float(el.duration.quarterLength)
        try:
            typ = m21dur.Duration(quarterLength=ql).type
        except Exception:
            typ = "inexpressible"
        if typ in (None, "inexpressible"):
            el.duration.quarterLength = snap_musicxml_ql(ql)
