from __future__ import annotations

from pathlib import Path

from music21 import duration, expressions, note, stream

from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.lattice import JointCandidate
from alignmodel.joint.ornament_mapper_v1 import (
    _expression_pattern,
    decode_hard_extra_mapper,
    decode_ornament_mapper,
    expand_ornament_hypothesis,
    ornament_mapper_features,
    score_ornament_patterns,
)
from alignmodel.joint.grammar_mapper_v2 import grammar_hypotheses


def _score(path: Path) -> ScoreEventIndex:
    score = stream.Score()
    part = stream.Part(id="clarinet")
    measure = stream.Measure(number=1)
    first = note.Note("C4", quarterLength=1.0)
    first.expressions.append(expressions.Mordent())
    measure.append(first)
    measure.append(note.Note("D4", quarterLength=1.0))
    part.append(measure)
    score.append(part)
    score.write("musicxml", fp=path)
    return ScoreEventIndex.from_musicxml(path)


def test_mordent_expands_to_linked_principal_and_two_extras(
    tmp_path: Path,
) -> None:
    path = tmp_path / "score.musicxml"
    index = _score(path)
    patterns = score_ornament_patterns(path, index.events)
    hypothesis = grammar_hypotheses(index.events, 4)[0]
    expanded = expand_ornament_hypothesis(
        index.events, patterns, hypothesis
    )
    assert [(row.pitch, row.kind) for row in expanded] == [
        (60, "linked"),
        (59, "ornament_extra"),
        (60, "ornament_extra"),
        (62, "linked"),
    ]


def test_oracle_ornament_sequence_maps_to_exact_extra_identities(
    tmp_path: Path,
) -> None:
    path = tmp_path / "score.musicxml"
    index = _score(path)
    candidates = tuple(
        JointCandidate(pitch, position * 0.1, position * 0.1 + 0.09, 1.0)
        for position, pitch in enumerate((60, 59, 60, 62))
    )
    mapped, deletions, diagnostics = decode_ornament_mapper(
        candidates, index.events, path
    )
    assert [event.relationship for event in mapped] == [
        "match",
        "extra",
        "extra",
        "match",
    ]
    assert [event.rendered_index for event in mapped] == [0, 1, 2, 3]
    assert mapped[0].score_span == (0, 1)
    assert mapped[3].score_span == (1, 2)
    assert deletions == frozenset()
    assert diagnostics["matched_template_ornaments"] == 2


def test_missing_linked_unit_emits_canonical_deletion(tmp_path: Path) -> None:
    path = tmp_path / "score.musicxml"
    index = _score(path)
    candidates = tuple(
        JointCandidate(pitch, position * 0.1, position * 0.1 + 0.09, 1.0)
        for position, pitch in enumerate((60, 59, 60))
    )
    _mapped, deletions, _diagnostics = decode_ornament_mapper(
        candidates, index.events, path
    )
    assert deletions == frozenset({1})


def test_extra_prior_features_are_target_free_and_cardinality_preserving(
    tmp_path: Path,
) -> None:
    path = tmp_path / "score.musicxml"
    index = _score(path)
    candidates = tuple(
        JointCandidate(pitch, position * 0.1, position * 0.1 + 0.09, 1.0)
        for position, pitch in enumerate((60, 59, 60, 62))
    )
    mapped, _deletions, _diagnostics = decode_ornament_mapper(
        candidates,
        index.events,
        path,
        extra_probabilities=(0.05, 0.95, 0.95, 0.05),
        extra_weight=1.0,
    )
    features = ornament_mapper_features(
        candidates, mapped, index.events, path
    )
    assert len(features) == len(candidates)
    assert {len(row) for row in features} == {25}
    assert [event.relationship for event in mapped] == [
        "match",
        "extra",
        "extra",
        "match",
    ]


def test_hard_extra_mapper_preserves_original_rendered_indices(
    tmp_path: Path,
) -> None:
    path = tmp_path / "score.musicxml"
    index = _score(path)
    candidates = tuple(
        JointCandidate(pitch, position * 0.1, position * 0.1 + 0.09, 1.0)
        for position, pitch in enumerate((60, 59, 60, 62))
    )
    mapped, deletions, diagnostics = decode_hard_extra_mapper(
        candidates,
        index.events,
        (False, True, True, False),
    )
    assert [event.relationship for event in mapped] == [
        "match",
        "extra",
        "extra",
        "match",
    ]
    assert [event.rendered_index for event in mapped] == [0, 1, 2, 3]
    assert deletions == frozenset()
    assert diagnostics["hard_extra_events"] == 2


def test_trill_and_turn_renderer_patterns_are_exact() -> None:
    trill = note.Note("C4", quarterLength=1.0)
    trill.expressions.append(expressions.Trill())
    trill_pattern = _expression_pattern(trill)
    assert len(trill_pattern) == 8
    assert [pitch for pitch, _kind in trill_pattern[:4]] == [60, 61, 60, 61]
    turn = note.Note("C4", quarterLength=1.0)
    turn.expressions.append(expressions.Turn())
    assert [pitch for pitch, _kind in _expression_pattern(turn)] == [
        61,
        60,
        59,
        60,
    ]


def test_grace_is_attached_as_explicit_template_prefix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "grace.musicxml"
    score = stream.Score()
    part = stream.Part(id="clarinet")
    measure = stream.Measure(number=1)
    grace = note.Note("D4")
    grace.duration = duration.GraceDuration(0.25)
    measure.append(grace)
    measure.append(note.Note("C4", quarterLength=1.0))
    part.append(measure)
    score.append(part)
    score.write("musicxml", fp=path)
    index = ScoreEventIndex.from_musicxml(path)
    patterns = score_ornament_patterns(path, index.events)
    assert patterns[0].prefix == ((62, "grace"),)
