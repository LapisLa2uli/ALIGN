import random

from music21 import meter, note, stream, tempo

from datacreate.melody import extra_neighbor_core, padded_melody, parse_sounding_notes
from synthpipeline.config import SynthConfig
from synthpipeline.errors import inject_error
from synthpipeline.timing import refine_labels


def _etude(n_measures: int = 4, notes_per: int = 4) -> stream.Score:
    score = stream.Score()
    part = stream.Part()
    part.append(tempo.MetronomeMark(number=100))
    part.append(meter.TimeSignature("4/4"))
    for i in range(n_measures):
        measure = stream.Measure(number=i + 1)
        for j in range(notes_per):
            measure.append(note.Note("C4", quarterLength=1.0))
        part.append(measure)
    score.append(part)
    return score


def _cfg(**errors) -> SynthConfig:
    payload = {
        "weights": {
            "wrong_note": 1.0,
            "missed_note": 0.0,
            "extra_note": 0.0,
            "rhythm_error": 0.0,
            "intonation_error": 0.0,
        },
        "per_clip_min": 1,
        "per_clip_max": 1,
        "repetition_prob": 0.0,
        "repeat_extra_copies_weights": {1: 1.0},
        "melody_pad_notes": [2],
    }
    payload.update(errors)
    return SynthConfig(errors=payload)


def test_triple_repeat_emits_one_repetition_and_three_passes():
    rng = random.Random(7)
    result = inject_error(
        _etude(),
        rng,
        _cfg(repetition_prob=1.0, repeat_extra_copies_weights={2: 1.0}),
    )
    assert result.repeated
    reps = [lab for lab in result.labels if lab.type == "repetition"]
    wrongs = [lab for lab in result.labels if lab.type == "wrong_note"]
    assert len(reps) == 1
    assert reps[0].extra_copies == 2
    assert len(wrongs) == 3
    assert result.extra.get("extra_copies") == 2


def test_double_repeat_two_wrong_passes():
    rng = random.Random(3)
    result = inject_error(
        _etude(),
        rng,
        _cfg(repetition_prob=1.0, repeat_extra_copies_weights={1: 1.0}),
    )
    wrongs = [lab for lab in result.labels if lab.type == "wrong_note"]
    reps = [lab for lab in result.labels if lab.type == "repetition"]
    assert len(wrongs) == 2
    assert len(reps) == 1
    assert reps[0].extra_copies == 1


def test_multiple_errors_no_repeat():
    rng = random.Random(11)
    result = inject_error(
        _etude(n_measures=6),
        rng,
        _cfg(per_clip_min=2, per_clip_max=2, repetition_prob=0.0),
    )
    planted = [lab for lab in result.labels if lab.type != "repetition"]
    assert len(planted) >= 2
    assert result.extra.get("error_types")
    assert len(result.extra["error_types"]) >= 2
    assert not result.repeated


def _etude_with_rests(n_measures: int = 4) -> stream.Score:
    score = stream.Score()
    part = stream.Part()
    part.append(tempo.MetronomeMark(number=100))
    part.append(meter.TimeSignature("4/4"))
    for i in range(n_measures):
        measure = stream.Measure(number=i + 1)
        measure.append(note.Rest(quarterLength=0.5))
        measure.append(note.Note("C4", quarterLength=1.0))
        measure.append(note.Rest(quarterLength=0.5))
        measure.append(note.Note("E4", quarterLength=1.0))
        measure.append(note.Note("G4", quarterLength=1.0))
        part.append(measure)
    score.append(part)
    return score


def test_extra_note_core_covers_neighbors():
    score = _etude()
    clean = parse_sounding_notes(score)
    rng = random.Random(5)
    result = inject_error(
        score,
        rng,
        _cfg(
            weights={
                "wrong_note": 0.0,
                "missed_note": 0.0,
                "extra_note": 1.0,
                "rhythm_error": 0.0,
                "intonation_error": 0.0,
            }
        ),
    )
    extras = [lab for lab in result.labels if lab.type == "extra_note"]
    assert extras
    lab = extras[0]
    assert lab.clean_note_index is not None
    assert lab.clean_note_count == 2
    i0, i1 = extra_neighbor_core(clean, lab.clean_note_index)
    assert i1 - i0 == 2
    payloads = refine_labels(result.labels, result.bpm, None, clean_notes=clean, pad_notes=2)
    extra_payload = next(p for p in payloads if p["type"] == "extra_note")
    assert extra_payload["pitches"] == padded_melody(clean, i0, i1, 2).pitches


def test_wrong_note_squeak_is_high():
    rng = random.Random(9)
    result = inject_error(
        _etude(),
        rng,
        _cfg(
            weights={
                "wrong_note": 1.0,
                "missed_note": 0.0,
                "extra_note": 0.0,
                "rhythm_error": 0.0,
                "intonation_error": 0.0,
            },
            squeak={"prob": 1.0, "pitch_min": "C6", "pitch_max": "A7"},
        ),
    )
    wrongs = [lab for lab in result.labels if lab.type == "wrong_note"]
    assert wrongs
    assert "squeak" in (wrongs[0].comment or "")
    assert wrongs[0].midi_pitch is not None and wrongs[0].midi_pitch >= 84


def test_standalone_repetition_has_gap_rest():
    rng = random.Random(4)
    result = inject_error(
        _etude(),
        rng,
        _cfg(
            repetition_prob=0.0,
            standalone_repetition_prob=1.0,
            repeat_gap_seconds=[0.5, 0.5],
        ),
    )
    assert result.repeated
    assert result.extra.get("standalone_repetition")
    assert abs(float(result.extra.get("repeat_gap_seconds") or 0) - 0.5) < 1e-6
    reps = [lab for lab in result.labels if lab.type == "repetition"]
    assert reps
    assert "on its own" in (reps[0].comment or "")
    assert "rest" in (reps[0].comment or "")


def test_after_error_repetition_has_gap_rest():
    rng = random.Random(7)
    result = inject_error(
        _etude(),
        rng,
        _cfg(
            repetition_prob=1.0,
            standalone_repetition_prob=0.0,
            repeat_gap_seconds=[0.4, 0.4],
            repeat_extra_copies_weights={1: 1.0},
        ),
    )
    assert result.repeated
    assert abs(float(result.extra.get("repeat_gap_seconds") or 0) - 0.4) < 1e-6
    reps = [lab for lab in result.labels if lab.type == "repetition"]
    assert "rest" in (reps[0].comment or "")


def test_rhythm_kinds_plant():
    kinds = ("late_start", "early_end", "tempo_change", "uneven", "early_start", "late_end")
    for kind in kinds:
        planted = False
        for seed in range(20):
            score = _etude_with_rests() if kind in {"early_start", "late_end"} else _etude()
            result = inject_error(
                score,
                random.Random(seed),
                _cfg(
                    weights={
                        "wrong_note": 0.0,
                        "missed_note": 0.0,
                        "extra_note": 0.0,
                        "rhythm_error": 1.0,
                        "intonation_error": 0.0,
                    },
                    rhythm_kinds=[kind],
                ),
            )
            labs = [lab for lab in result.labels if lab.type == "rhythm_error"]
            if labs:
                planted = True
                break
        assert planted, kind


def test_gap_and_uneven_durations_are_musicxml_legal():
    from music21 import duration as m21dur
    from synthpipeline.errors import _expressible_ql_parts, _uneven_expressible_durs
    from synthpipeline.scoregen import write_musicxml

    for target in (0.24, 0.33, 0.5, 0.83, 1.2, 1.87):
        for ql in _expressible_ql_parts(target):
            assert m21dur.Duration(quarterLength=ql).type not in (None, "inexpressible")

    rng = random.Random(1)
    durs = _uneven_expressible_durs(rng, 4.0, 3)
    assert durs is not None
    assert abs(sum(durs) - 4.0) < 0.26
    for ql in durs:
        assert m21dur.Duration(quarterLength=ql).type not in (None, "inexpressible")

    cfg = _cfg(
        weights={
            "wrong_note": 1.0,
            "missed_note": 1.0,
            "extra_note": 1.0,
            "rhythm_error": 1.0,
            "intonation_error": 1.0,
        },
        per_clip_min=2,
        per_clip_max=4,
        repetition_prob=1.0,
        standalone_repetition_prob=0.0,
        repeat_gap_seconds=[0.2, 1.0],
        rhythm_kinds=["uneven", "late_start", "early_end", "tempo_change"],
        squeak={"prob": 0.5, "pitch_min": "C6", "pitch_max": "A7"},
    )
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "perf.musicxml"
        for seed in range(12):
            result = inject_error(_etude(n_measures=8), random.Random(seed), cfg)
            write_musicxml(result.score, out)


def test_zero_intonation_weight_never_plants_intonation_or_squeak():
    cfg = _cfg(
        weights={
            "wrong_note": 1.0,
            "missed_note": 1.0,
            "extra_note": 1.0,
            "rhythm_error": 1.0,
            "intonation_error": 0.0,
        },
        per_clip_min=4,
        per_clip_max=8,
        repetition_prob=0.0,
        squeak={"prob": 0.0, "pitch_min": "C6", "pitch_max": "A7"},
    )
    types = set()
    for seed in range(30):
        result = inject_error(_etude(n_measures=8), random.Random(seed), cfg)
        types.update(lab.type for lab in result.labels)
        assert all("squeak" not in (lab.comment or "") for lab in result.labels)
        assert not result.extra.get("pitch_bends")
    assert "intonation_error" not in types
    assert types <= {"wrong_note", "missed_note", "extra_note", "rhythm_error"}
