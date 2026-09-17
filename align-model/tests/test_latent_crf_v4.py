from __future__ import annotations

import math

import torch

from alignmodel.joint.grammar_mapper_v2 import GrammarHypothesis
from alignmodel.joint.index import JointEvent, ScoreEvent
from alignmodel.joint.latent_crf_v4 import (
    LatentSpanCRF,
    latent_crf_loss,
    log_partition,
    transitions,
)
from alignmodel.joint.lattice import JointCandidate


def _score(pitches: list[int]) -> tuple[ScoreEvent, ...]:
    return tuple(
        ScoreEvent(index, pitch, index, index + 1, (index,), index + 1)
        for index, pitch in enumerate(pitches)
    )


def _candidates(pitches: list[int]) -> tuple[JointCandidate, ...]:
    return tuple(
        JointCandidate(pitch, index, index + 0.5, 1.0)
        for index, pitch in enumerate(pitches)
    )


def _plan(units: tuple[tuple[int, int], ...]) -> GrammarHypothesis:
    copies = max((copy for _index, copy in units), default=0)
    return GrammarHypothesis(None, copies, units, tuple(range(len(units))))


def _path_count(lattice, final):
    count = {(0, 0): 1}
    for i in range(final[0] + 1):
        for j in range(final[1] + 1):
            for edge in lattice:
                if edge.source == (i, j):
                    count[edge.destination] = (
                        count.get(edge.destination, 0) + count.get((i, j), 0)
                    )
    return count[final]


def test_forward_matches_bruteforce_path_count() -> None:
    events = _candidates([60, 62])
    score = _score([60, 62])
    plan = _plan(((0, 0), (1, 0)))
    lattice = transitions(events, score, plan, max_span_units=2)
    scores = torch.zeros(len(lattice))
    partition = log_partition(scores, lattice, (2, 2))
    assert torch.allclose(
        partition, torch.tensor(math.log(_path_count(lattice, (2, 2))))
    )


def test_latent_gold_marginal_has_gradients() -> None:
    events = _candidates([60, 62])
    score = _score([60, 62])
    gold = (
        JointEvent(60, 0, 0.5, (0, 1)),
        JointEvent(62, 1, 1.5, (1, 2)),
    )
    model = LatentSpanCRF()
    loss, report = latent_crf_loss(
        model, events, score, _plan(((0, 0), (1, 0))), gold
    )
    assert torch.isfinite(loss)
    assert report["gold_coverage"]
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_deletion_folded_span_and_rendered_extra_have_gold_paths() -> None:
    score = _score([60, 61, 62])
    model = LatentSpanCRF()
    folded = (JointEvent(62, 0, 0.5, (0, 3)),)
    loss, _ = latent_crf_loss(
        model,
        _candidates([62]),
        score,
        _plan(((0, 0), (1, 0), (2, 0))),
        folded,
    )
    assert torch.isfinite(loss)
    extra = (JointEvent(65, 0, 0.5, None, "extra", rendered_index=0),)
    loss, _ = latent_crf_loss(
        model, _candidates([65]), (), _plan(()), extra
    )
    assert torch.isfinite(loss)


def test_one_and_two_copy_plans_cover_exact_copy_passes() -> None:
    score = _score([60, 62])
    model = LatentSpanCRF()
    for copies in (1, 2):
        units = tuple(
            (index, copy_pass)
            for copy_pass in range(copies + 1)
            for index in range(2)
        )
        events = _candidates([60, 62] * (copies + 1))
        gold = tuple(
            JointEvent(
                event.pitch,
                event.start,
                event.end,
                (index, index + 1),
                "copy" if copy_pass else "match",
                copy_pass,
            )
            for event, (index, copy_pass) in zip(events, units)
        )
        loss, report = latent_crf_loss(
            model, events, score, _plan(units), gold
        )
        assert torch.isfinite(loss)
        assert report["gold_coverage"]
