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

# Duration in sixteenths, weighted toward quarters and eighths.
DURATION_UNITS = [1, 2, 3, 4, 6, 8]
DURATION_WEIGHTS = [1, 4, 1, 5, 2, 1]


def clarinet_instrument() -> instrument.Instrument:
    """Concert-pitch clarinet timbre (GM program 71), no Bb transposition."""
    inst = instrument.Clarinet()
    inst.instrumentName = "Clarinet"
    inst.midiProgram = 71
    inst.transposition = None
    return inst


def write_musicxml(score: stream.Score, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = Path(str(score.write("musicxml", fp=str(path))))
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

    k = m21key.Key(tonic, mode)
    scale_pitches = _diatonic_pitches(k, lo, hi)
    if not scale_pitches:
        raise RuntimeError(f"No scale pitches between {lo} and {hi} for {tonic} {mode}")

    measure_ql = beats * (4.0 / beat_type)
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
) -> int:
    units = int(round(measure_ql / 0.25))
    remaining = units
    offset_units = 0
    while remaining > 0:
        fits = [u for u in DURATION_UNITS if u <= remaining]
        if phrase_end and remaining in DURATION_UNITS and remaining >= 4:
            dur_units = remaining
        else:
            weights = [DURATION_WEIGHTS[DURATION_UNITS.index(u)] for u in fits]
            dur_units = rng.choices(fits, weights=weights, k=1)[0]
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
