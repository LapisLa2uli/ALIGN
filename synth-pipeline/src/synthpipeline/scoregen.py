from __future__ import annotations

import random
from pathlib import Path

from music21 import (
    clef,
    converter,
    instrument,
    key as m21key,
    metadata,
    meter,
    note,
    pitch,
    stream,
    tempo,
)

from synthpipeline.config import SynthConfig

DEFAULT_KEYS = [
    {"tonic": "C", "mode": "major"},
    {"tonic": "G", "mode": "major"},
    {"tonic": "F", "mode": "major"},
    {"tonic": "D", "mode": "major"},
    {"tonic": "Bb", "mode": "major"},
    {"tonic": "Eb", "mode": "major"},
    {"tonic": "A", "mode": "minor"},
    {"tonic": "E", "mode": "minor"},
    {"tonic": "D", "mode": "minor"},
    {"tonic": "G", "mode": "minor"},
]

DEFAULT_METERS = [(4, 4), (3, 4), (2, 4), (6, 8)]

# Duration in sixteenths. Onsets are biased onto the beat; see _pick_duration.
DURATION_UNITS = [1, 2, 3, 4, 6, 8]


def clarinet_instrument() -> instrument.Instrument:
    """Concert-pitch clarinet timbre (GM program 71), no Bb transposition."""
    inst = instrument.Clarinet()
    inst.instrumentName = "Clarinet"
    inst.midiProgram = 71
    inst.transposition = None
    return inst


def write_musicxml(score: stream.Score, path: Path) -> Path:
    return _write_score(score, path, "musicxml")


def write_midi(score: stream.Score, path: Path) -> Path:
    return _write_score(score, path, "midi")


def _write_score(score: stream.Score, path: Path, fmt: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = Path(str(score.write(fmt, fp=str(path))))
    if written.resolve() != path.resolve():
        path.write_bytes(written.read_bytes())
        written.unlink(missing_ok=True)
    return path


def generate_score(rng: random.Random, config: SynthConfig) -> stream.Score:
    gen = config.generation
    n_measures = rng.randint(int(gen.get("measures_min", 8)), int(gen.get("measures_max", 16)))
    meters = gen.get("meters") or DEFAULT_METERS
    meter_choice = rng.choice(meters)
    beats, beat_type = int(meter_choice[0]), int(meter_choice[1])
    keys = gen.get("keys") or DEFAULT_KEYS
    key_choice = rng.choice(keys)
    tonic = str(key_choice["tonic"])
    mode = str(key_choice.get("mode", "major"))
    bpm = rng.randint(int(gen.get("tempo_min", 72)), int(gen.get("tempo_max", 112)))
    lo = pitch.Pitch(str(gen.get("pitch_min", "E3")))
    hi = pitch.Pitch(str(gen.get("pitch_max", "C6")))
    rest_prob = float(gen.get("rest_probability", 0.08))
    syncopation_prob = float(gen.get("syncopation_prob", 0.12))

    k = m21key.Key(tonic, mode)
    scale_pitches = _diatonic_pitches(k, lo, hi)
    if not scale_pitches:
        raise RuntimeError(f"No scale pitches between {lo} and {hi} for {tonic} {mode}")

    measure_ql = beats * (4.0 / beat_type)
    beat_units = 6 if beat_type == 8 else max(1, int(round(4 * 4 / beat_type)))
    current_idx = _start_index(scale_pitches, k)

    score = stream.Score()
    score.insert(0, metadata.Metadata())
    score.metadata.title = f"Synth Etude {tonic} {mode}"
    score.metadata.composer = "ALIGN synth-pipeline"

    part = stream.Part(id="clarinet")
    part.partName = "Clarinet"
    part.insert(0, clarinet_instrument())

    ts = meter.TimeSignature(f"{beats}/{beat_type}")
    for m_i in range(n_measures):
        measure = stream.Measure(number=m_i + 1)
        if m_i == 0:
            measure.insert(0, ts)
            measure.insert(0, k)
            measure.insert(0, clef.TrebleClef())
            measure.insert(0, tempo.MetronomeMark(number=bpm))
        phrase_end = (m_i + 1) % 4 == 0 or m_i == n_measures - 1
        current_idx = _fill_measure(
            measure,
            rng,
            scale_pitches,
            current_idx,
            measure_ql,
            rest_prob,
            phrase_end=phrase_end,
            beat_units=beat_units,
            syncopation_prob=syncopation_prob,
        )
        part.append(measure)

    score.insert(0, part)
    return score


def load_score(path: Path, config: SynthConfig) -> stream.Score:
    score = converter.parse(str(path))
    if not score.parts:
        part = stream.Part(id="clarinet")
        part.partName = "Clarinet"
        for el in score.flatten().notesAndRests:
            part.append(el)
        score = stream.Score()
        score.insert(0, part)
    _ensure_clarinet(score)
    _chords_to_top_notes(score)
    _ensure_measures(score)
    _ensure_tempo(score, config)
    if score.metadata is None:
        score.insert(0, metadata.Metadata())
    if not score.metadata.title:
        score.metadata.title = path.stem
    return score


def resolve_score_inputs(score_arg: Path | None) -> list[Path] | None:
    """None means generate procedurally. Otherwise a list of MusicXML paths."""
    if score_arg is None:
        return None
    if score_arg.is_dir():
        files = sorted(score_arg.glob("*.musicxml")) + sorted(score_arg.glob("*.xml"))
        files += sorted(score_arg.glob("*.mxl"))
        if not files:
            raise FileNotFoundError(f"No MusicXML files in {score_arg}")
        return files
    if not score_arg.exists():
        raise FileNotFoundError(score_arg)
    return [score_arg]


def _ensure_clarinet(score: stream.Score) -> None:
    part = score.parts[0]
    for inst in list(part.recurse().getElementsByClass(instrument.Instrument)):
        site = inst.activeSite
        if site is not None:
            site.remove(inst)
    part.insert(0, clarinet_instrument())


def _ensure_tempo(score: stream.Score, config: SynthConfig) -> None:
    marks = list(score.flatten().getElementsByClass(tempo.MetronomeMark))
    if any(m.number for m in marks):
        return
    gen = config.generation
    bpm = int((int(gen.get("tempo_min", 72)) + int(gen.get("tempo_max", 112))) / 2)
    part = score.parts[0]
    measures = list(part.getElementsByClass(stream.Measure))
    if measures:
        measures[0].insert(0, tempo.MetronomeMark(number=bpm))
    else:
        part.insert(0, tempo.MetronomeMark(number=bpm))


def _ensure_measures(score: stream.Score) -> None:
    part = score.parts[0]
    if list(part.getElementsByClass(stream.Measure)):
        return
    part.makeMeasures(inPlace=True)


def _chords_to_top_notes(score: stream.Score) -> None:
    from music21 import chord

    for ch in list(score.recurse().getElementsByClass(chord.Chord)):
        site = ch.activeSite
        if site is None:
            continue
        sorted_chord = ch.sortDiatonicAscending()
        top = note.Note(sorted_chord.pitches[-1])
        top.duration = ch.duration
        top.offset = ch.offset
        site.replace(ch, top)


def _diatonic_pitches(k: m21key.Key, lo: pitch.Pitch, hi: pitch.Pitch) -> list[pitch.Pitch]:
    out: list[pitch.Pitch] = []
    for midi in range(lo.midi, hi.midi + 1):
        p = pitch.Pitch(midi=midi)
        if k.getScaleDegreeFromPitch(p) is not None:
            out.append(p)
    return out


def _start_index(scale_pitches: list[pitch.Pitch], k: m21key.Key) -> int:
    tonic_name = k.tonic.name
    around = 67  # G4
    best = 0
    best_dist = 10**9
    for i, p in enumerate(scale_pitches):
        if p.name != tonic_name:
            continue
        dist = abs(p.midi - around)
        if dist < best_dist:
            best = i
            best_dist = dist
    if best_dist == 10**9:
        return min(range(len(scale_pitches)), key=lambda i: abs(scale_pitches[i].midi - around))
    return best


def _fill_measure(
    measure: stream.Measure,
    rng: random.Random,
    scale_pitches: list[pitch.Pitch],
    current_idx: int,
    measure_ql: float,
    rest_prob: float,
    phrase_end: bool,
    beat_units: int = 4,
    syncopation_prob: float = 0.12,
) -> int:
    units = int(round(measure_ql / 0.25))
    remaining = units
    offset_units = 0
    while remaining > 0:
        dur_units = _pick_duration(
            rng,
            remaining,
            offset_units,
            beat_units,
            phrase_end=phrase_end,
            syncopation_prob=syncopation_prob,
        )
        ql = dur_units * 0.25
        offset = offset_units * 0.25
        if rng.random() < rest_prob and remaining != units:
            measure.insert(offset, note.Rest(quarterLength=ql))
        else:
            step = rng.choices([-2, -1, 0, 1, 2, 3, -3], weights=[1, 5, 2, 5, 1, 1, 1], k=1)[0]
            current_idx = max(0, min(len(scale_pitches) - 1, current_idx + step))
            n = note.Note(scale_pitches[current_idx], quarterLength=ql)
            measure.insert(offset, n)
        remaining -= dur_units
        offset_units += dur_units
    return current_idx


def _pick_duration(
    rng: random.Random,
    remaining: int,
    offset_units: int,
    beat_units: int,
    phrase_end: bool,
    syncopation_prob: float,
) -> int:
    fits = [u for u in DURATION_UNITS if u <= remaining]
    if not fits:
        return remaining
    if phrase_end and remaining in fits and remaining >= beat_units:
        return remaining

    on_beat = beat_units > 0 and offset_units % beat_units == 0
    aligned = [u for u in fits if beat_units and (offset_units + u) % beat_units == 0]
    half = beat_units // 2 if beat_units >= 2 else 0
    # Eighths (or compound-beat halves) starting on a beat are normal, not syncopation.
    mild = list(aligned)
    if on_beat and half in fits and half not in mild:
        mild.append(half)

    use_sync = rng.random() < syncopation_prob
    pool = fits if (use_sync or not mild) else mild

    weights: list[int] = []
    for u in pool:
        lands_on_beat = beat_units and (offset_units + u) % beat_units == 0
        if lands_on_beat and u == beat_units:
            weights.append(8)
        elif lands_on_beat and u == 2 * beat_units:
            weights.append(4)
        elif lands_on_beat:
            weights.append(6)
        elif on_beat and u == half:
            weights.append(3)
        else:
            weights.append(1)
    return rng.choices(pool, weights=weights, k=1)[0]
