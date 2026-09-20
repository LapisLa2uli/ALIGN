from pathlib import Path

from music21 import meter, note, stream, tempo

from datacreate.melody import ScoreSoundingNote
from datacreate.score_locate import (
    LocatedScoreSpan,
    TranscribedNote,
    list_score_candidates,
    locate_best_among_scores,
    locate_score_span,
    should_apply_location,
)


def _written() -> list[ScoreSoundingNote]:
    # Two similar phrases: m1-2 C D E F, later m10-11 C D E G
    pitches = [60, 62, 64, 65, 67, 69, 71, 72, 60, 62, 64, 67]
    notes = []
    for index, pitch in enumerate(pitches):
        measure = 1 + index // 4 if index < 8 else 10 + (index - 8) // 4
        start = float(index)
        notes.append(
            ScoreSoundingNote(
                index=index,
                pitch=pitch,
                start=start * 0.5,
                end=start * 0.5 + 0.45,
                ql_start=start,
                ql_end=start + 1.0,
                measure=measure,
                note_id=f"note_{index:04d}",
            )
        )
    return notes


def test_locate_prefers_matching_phrase_not_the_opening():
    written = _written()
    transcribed = [
        TranscribedNote(60, 0.0, 0.4),
        TranscribedNote(62, 0.5, 0.9),
        TranscribedNote(64, 1.0, 1.4),
        TranscribedNote(67, 1.5, 1.9),
    ]
    located = locate_score_span(transcribed, written)
    assert located is not None
    assert located.start_measure == 10
    assert located.end_measure == 10
    assert located.score_i0 == 8
    assert located.confidence >= 0.5


def test_locate_accepts_bb_clarinet_sounding_shift():
    written = _written()[:4]
    transcribed = [
        TranscribedNote(pitch - 2, index * 0.5, index * 0.5 + 0.4)
        for index, pitch in enumerate([60, 62, 64, 65])
    ]
    located = locate_score_span(transcribed, written)
    assert located is not None
    assert located.pitch_shift == 2
    assert located.start_measure == 1
    assert located.end_measure == 1


def test_should_replace_invalid_current_segment(tmp_path: Path):
    located = LocatedScoreSpan(
        start_measure=20,
        end_measure=36,
        confidence=0.2,
        mapped_notes=8,
    )
    assert should_apply_location(
        located,
        current={"start_measure": 672, "end_measure": 688, "start_beat": 2},
        total_measures=241,
    )
    assert not should_apply_location(
        located,
        current={"start_measure": 20, "end_measure": 36, "start_beat": 1},
        total_measures=241,
    )


def test_locate_writes_musicxml_beats(tmp_path: Path):
    part = stream.Part()
    part.insert(0, tempo.MetronomeMark(number=60))
    part.insert(0, meter.TimeSignature("4/4"))
    first = stream.Measure(number=1)
    first.append(note.Note("C4", quarterLength=2.0))
    first.append(note.Note("E4", quarterLength=2.0))
    second = stream.Measure(number=2)
    second.append(note.Note("G4", quarterLength=4.0))
    part.append(first)
    part.append(second)
    score = stream.Score()
    score.insert(0, part)
    path = tmp_path / "score.musicxml"
    score.write("musicxml", fp=str(path))
    from datacreate.melody import parse_sounding_notes

    written = parse_sounding_notes(path)
    transcribed = [
        TranscribedNote(int(note.pitch), note.start, note.end)
        for note in written[1:]
    ]
    located = locate_score_span(transcribed, written, score_path=path)
    assert located is not None
    assert located.start_measure == 1
    assert located.start_beat >= 2
    assert located.end_measure == 2


def test_locate_best_among_scores_prefers_matching_piece(tmp_path: Path):
    matching = tmp_path / "match.musicxml"
    other = tmp_path / "other.musicxml"
    for path, pitches in (
        (matching, [60, 62, 64, 65] * 8),
        (other, [72, 71, 69, 67] * 8),
    ):
        part = stream.Part()
        part.insert(0, tempo.MetronomeMark(number=60))
        part.insert(0, meter.TimeSignature("4/4"))
        for measure_number, chunk in enumerate(
            [pitches[i : i + 4] for i in range(0, len(pitches), 4)], start=1
        ):
            measure = stream.Measure(number=measure_number)
            for pitch in chunk:
                measure.append(note.Note(pitch, quarterLength=1.0))
            part.append(measure)
        score = stream.Score()
        score.insert(0, part)
        score.write("musicxml", fp=str(path))
    transcribed = [
        TranscribedNote(pitch, index * 0.5, index * 0.5 + 0.4)
        for index, pitch in enumerate([60, 62, 64, 65, 60, 62, 64, 65])
    ]
    best = locate_best_among_scores(transcribed, [other, matching])
    assert best is not None
    located, chosen = best
    assert chosen == matching
    assert located.start_measure == 1
    candidates = list_score_candidates(
        raw_score_root=tmp_path, sample_full_score=matching
    )
    assert matching in candidates
    assert other in candidates


def test_locate_best_skips_tiny_excerpt_when_full_piece_exists(tmp_path: Path):
    full = tmp_path / "full.musicxml"
    excerpt = tmp_path / "001.musicxml"
    for path, n_measures, base in ((full, 24, 60), (excerpt, 2, 60)):
        part = stream.Part()
        part.insert(0, tempo.MetronomeMark(number=60))
        part.insert(0, meter.TimeSignature("4/4"))
        for measure_number in range(1, n_measures + 1):
            measure = stream.Measure(number=measure_number)
            for offset in range(4):
                measure.append(note.Note(base + offset, quarterLength=1.0))
            part.append(measure)
        score = stream.Score()
        score.insert(0, part)
        score.write("musicxml", fp=str(path))
    transcribed = [
        TranscribedNote(60 + (i % 4), i * 0.45, i * 0.45 + 0.4) for i in range(20)
    ]
    best = locate_best_among_scores(transcribed, [excerpt, full])
    assert best is not None
    assert best[1] == full
