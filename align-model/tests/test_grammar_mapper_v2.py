from __future__ import annotations

from alignmodel.joint.grammar_mapper_v2 import decode_grammar_mapper
from alignmodel.joint.index import ScoreEvent
from alignmodel.joint.lattice import JointCandidate


def _score(
    pitches: list[int], measures: list[int] | None = None
) -> tuple[ScoreEvent, ...]:
    measures = measures or [index + 1 for index in range(len(pitches))]
    return tuple(
        ScoreEvent(index, pitch, index, index + 1, (index,), measures[index])
        for index, pitch in enumerate(pitches)
    )


def _events(pitches: list[int], gap_at: int | None = None) -> tuple[JointCandidate, ...]:
    rows = []
    time = 0.0
    for index, pitch in enumerate(pitches):
        if index == gap_at:
            time += 0.4
        rows.append(JointCandidate(pitch, time, time + 0.4, 1.0))
        time += 0.5
    return tuple(rows)


def test_no_repeat_exact_is_identity() -> None:
    mapped, grammar = decode_grammar_mapper(
        _events([60, 62, 64]), _score([60, 62, 64])
    )
    assert [event.score_span for event in mapped] == [(0, 1), (1, 2), (2, 3)]
    assert all(event.relationship == "match" for event in mapped)
    assert grammar["copies"] == 0


def test_substitution_extra_and_delete_backtrace() -> None:
    substituted, _ = decode_grammar_mapper(
        _events([60, 63, 64]), _score([60, 62, 64])
    )
    assert substituted[1].relationship == "substitute"
    extra, _ = decode_grammar_mapper(
        _events([60, 61, 62]), _score([60, 62])
    )
    assert extra[1].relationship == "extra"
    assert extra[1].rendered_index == 1
    deleted, _ = decode_grammar_mapper(
        _events([60, 64]), _score([60, 62, 64])
    )
    assert deleted[1].score_span == (1, 3)


def test_one_and_two_immediate_measure_copies() -> None:
    score = _score([60, 62, 64, 65], [1, 1, 2, 2])
    once, grammar = decode_grammar_mapper(
        _events([60, 62, 60, 62, 64, 65], gap_at=2), score
    )
    assert grammar["source_span"] == (0, 2)
    assert grammar["copies"] == 1
    assert [event.copy_pass for event in once] == [0, 0, 1, 1, 0, 0]
    twice, grammar = decode_grammar_mapper(
        _events([60, 62, 60, 62, 60, 62, 64, 65], gap_at=2), score
    )
    assert grammar["copies"] == 2
    assert [event.copy_pass for event in twice] == [0, 0, 1, 1, 2, 2, 0, 0]


def test_recurring_motif_without_insertion_is_not_repeat() -> None:
    score = _score([60, 62, 60, 62], [1, 1, 2, 2])
    mapped, grammar = decode_grammar_mapper(_events([60, 62, 60, 62]), score)
    assert grammar["copies"] == 0
    assert all(not event.is_copy for event in mapped)


def test_end_of_score_repeat_resumes_cleanly() -> None:
    score = _score([60, 62, 64], [1, 1, 2])
    mapped, grammar = decode_grammar_mapper(
        _events([60, 62, 64, 64], gap_at=3), score
    )
    assert grammar["copies"] == 1
    assert mapped[-1].copy_pass == 1


def test_repeated_part_interleaves_with_continuing_part() -> None:
    score = (
        ScoreEvent(0, 60, 0, 1, (0,), 1, "solo", None),
        ScoreEvent(1, 70, 0, 1, (10,), 1, "other", None),
        ScoreEvent(2, 62, 1, 2, (1,), 1, "solo", None),
        ScoreEvent(3, 72, 1, 2, (11,), 1, "other", None),
        ScoreEvent(4, 64, 2, 3, (2,), 2, "solo", None),
        ScoreEvent(5, 74, 2, 3, (12,), 2, "other", None),
        ScoreEvent(6, 65, 3, 4, (3,), 2, "solo", None),
        ScoreEvent(7, 76, 3, 4, (13,), 2, "other", None),
    )
    performed = _events([60, 70, 62, 72, 60, 74, 62, 76, 64, 65], gap_at=4)
    mapped, grammar = decode_grammar_mapper(performed, score)
    assert grammar["copies"] == 1
    assert [event.copy_pass for event in mapped] == [
        0, 0, 0, 0, 1, 0, 1, 0, 0, 0
    ]
    assert mapped[4].score_span == (0, 1)
    assert mapped[6].score_span == (2, 3)
