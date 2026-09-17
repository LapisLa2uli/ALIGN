import random

from pathlib import Path

from synthpipeline.config import SynthConfig
from synthpipeline.scoregen import generate_score, resolve_score_inputs, snippet_score


def test_snippet_is_contiguous_window():
    cfg = SynthConfig(
        generation={
            "measures_min": 16,
            "measures_max": 16,
            "snippet_measures_min": 8,
            "snippet_measures_max": 8,
            "tempo_min": 100,
            "tempo_max": 100,
        }
    )
    full = generate_score(random.Random(1), cfg)
    snippet, meta = snippet_score(full, random.Random(2), cfg)
    measures = list(snippet.parts[0].getElementsByClass("Measure"))
    assert len(measures) == 8
    assert [m.number for m in measures] == list(range(1, 9))
    assert meta["snippet_length"] == 8
    assert meta["snippet_end_measure"] - meta["snippet_start_measure"] + 1 == 8


def test_snippet_skips_rest_only_prefix():
    from music21 import meter, note, stream, tempo

    score = stream.Score()
    part = stream.Part()
    part.partName = "Clarinet"
    part.append(tempo.MetronomeMark(number=96))
    part.append(meter.TimeSignature("4/4"))
    for i in range(8):
        measure = stream.Measure(number=i + 1)
        measure.append(note.Rest(quarterLength=4.0))
        part.append(measure)
    for i in range(8, 16):
        measure = stream.Measure(number=i + 1)
        for _ in range(4):
            measure.append(note.Note("C4", quarterLength=1.0))
        part.append(measure)
    score.append(part)
    cfg = SynthConfig(
        generation={
            "snippet_measures_min": 8,
            "snippet_measures_max": 8,
            "snippet_min_notes": 12,
        }
    )
    snippet, meta = snippet_score(score, random.Random(0), cfg)
    assert meta["snippet_start_measure"] >= 8
    assert meta["snippet_notes"] >= 12


def test_resolve_score_inputs_excludes_stems(tmp_path: Path):
    (tmp_path / "001.musicxml").write_text("<score-partwise/>", encoding="utf-8")
    (tmp_path / "MozartClConcertoA.musicxml").write_text(
        "<score-partwise/>", encoding="utf-8"
    )
    (tmp_path / "WeberITAV.musicxml").write_text("<score-partwise/>", encoding="utf-8")
    cfg = SynthConfig(paths={"score_exclude": ["001"]})
    names = [path.stem for path in resolve_score_inputs(tmp_path, cfg)]
    assert names == ["MozartClConcertoA", "WeberITAV"]


def test_generated_notes_stay_in_clarinet_range():
    from music21 import note, pitch

    from synthpipeline.scoregen import clarinet_midi_bounds, generate_score

    cfg = SynthConfig(
        generation={
            "measures_min": 8,
            "measures_max": 8,
            "pitch_min": "E3",
            "pitch_max": "G6",
            "ornament_prob": 1.0,
            "keep_ornaments": True,
            "tempo_min": 96,
            "tempo_max": 96,
        }
    )
    score = generate_score(random.Random(9), cfg)
    lo, hi = clarinet_midi_bounds(cfg)
    midis = [int(n.pitch.midi) for n in score.recurse().getElementsByClass(note.Note)]
    assert midis
    assert min(midis) >= lo
    assert max(midis) <= hi
    graces = [n for n in score.recurse().getElementsByClass(note.Note) if n.duration.isGrace]
    mordents = [
        n
        for n in score.recurse().getElementsByClass(note.Note)
        if any(type(e).__name__ == "Mordent" for e in (n.expressions or []))
    ]
    assert graces or mordents
    assert lo == pitch.Pitch("E3").midi
    assert hi == pitch.Pitch("G6").midi


def test_snippet_rejects_out_of_range_window():
    from music21 import meter, note, stream, tempo

    from synthpipeline.scoregen import snippet_score

    score = stream.Score()
    part = stream.Part()
    part.partName = "Clarinet"
    part.append(tempo.MetronomeMark(number=96))
    part.append(meter.TimeSignature("4/4"))
    for i in range(8):
        measure = stream.Measure(number=i + 1)
        for _ in range(4):
            measure.append(note.Note("C8", quarterLength=1.0))
        part.append(measure)
    score.append(part)
    cfg = SynthConfig(
        generation={
            "snippet_measures_min": 8,
            "snippet_measures_max": 8,
            "snippet_min_notes": 12,
            "pitch_min": "E3",
            "pitch_max": "G6",
            "require_playable_range": True,
        }
    )
    try:
        snippet_score(score, random.Random(0), cfg)
    except ValueError as exc:
        assert "playable" in str(exc).lower() or "range" in str(exc).lower()
    else:
        raise AssertionError("expected out-of-range snippet to fail")


def test_midi_export_is_written_minus_two(tmp_path: Path) -> None:
    import logging

    from music21 import note
    from tinysoundfont.midi import NoteOn, load

    from synthpipeline.render import export_score_to_midi_music21
    from synthpipeline.scoregen import write_musicxml

    cfg = SynthConfig.load()
    score = generate_score(random.Random(42), cfg)
    written = [int(n.pitch.midi) for n in score.flatten().getElementsByClass(note.Note)]
    xml_path = tmp_path / "verified.musicxml"
    midi_path = tmp_path / "reference.mid"
    write_musicxml(score, xml_path)
    export_score_to_midi_music21(
        score, xml_path, midi_path, logging.getLogger("test"), sounding_transpose=-2
    )
    keys = [
        int(ev.action.key)
        for ev in load(str(midi_path), persistent=False)
        if isinstance(ev.action, NoteOn) and int(getattr(ev.action, "velocity", 0) or 0) > 0
    ]
    assert keys[:12] == [pitch - 2 for pitch in written[:12]]
