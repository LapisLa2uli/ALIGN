from __future__ import annotations

import math

import torch

from alignmodel.joint.grammar_mapper_v2 import GrammarHypothesis
from alignmodel.joint.identity_crf_v1 import (
    CANDIDATE_EXTRA,
    LINK,
    IdentityCandidate,
    IdentityLattice,
    IdentityTarget,
    LatticeHypothesis,
    OrnamentIdentityCRF,
    decode_identity_crf,
    exact_identity_round_trip,
    hypothesis_log_partition,
    identity_crf_nll,
)
from alignmodel.joint.index import ScoreEvent
from alignmodel.joint.ornament_mapper_v1 import OrnamentTemplateUnit


def _zero_model() -> OrnamentIdentityCRF:
    model = OrnamentIdentityCRF(hidden=8)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    return model


def _score() -> tuple[ScoreEvent, ...]:
    return (ScoreEvent(0, 60, 0.0, 1.0, (0,)),)


def _hypothesis(
    template: tuple[OrnamentTemplateUnit, ...],
    *,
    gold: bool = True,
) -> LatticeHypothesis:
    return LatticeHypothesis(
        grammar=GrammarHypothesis(
            None, 0, ((0, 0),), (0.0,)
        ),
        template=template,
        is_gold_compatible=gold,
        hard_negative_rank=0.0,
    )


def test_tiny_partition_matches_brute_force_and_has_gradients() -> None:
    candidate = IdentityCandidate(60, 0.0, 0.5)
    target = IdentityTarget("match", (0, 1), 0, 0, 60)
    unit = OrnamentTemplateUnit(60, 0.0, "linked", 0, 0)
    lattice = IdentityLattice(
        (candidate,),
        _score(),
        (_hypothesis((unit,)),),
        (target,),
    )
    model = _zero_model()
    # One LINK path, plus two orderings each for EXTRA+DELETE and
    # EXTRA_COPY+DELETE.
    partition = hypothesis_log_partition(
        model, lattice, lattice.hypotheses[0]
    )
    assert torch.allclose(partition, torch.tensor(math.log(5.0)))
    loss, _parts = identity_crf_nll(model, lattice, normalize=False)
    assert torch.allclose(loss, torch.tensor(math.log(5.0)))
    loss.backward()
    assert model.transitions.grad is not None
    assert torch.isfinite(model.transitions.grad).all()


def test_gold_partition_marginalizes_ambiguous_ornament_placement() -> None:
    candidate = IdentityCandidate(59, 0.0, 0.1)
    target = IdentityTarget("extra", None, 0, 0, 59)
    ornament = OrnamentTemplateUnit(
        59, 0.0, "ornament_extra", None, 0, "mordent"
    )
    lattice = IdentityLattice(
        (candidate,),
        _score(),
        (_hypothesis((ornament,)),),
        (target,),
    )
    model = _zero_model()
    gold = hypothesis_log_partition(
        model, lattice, lattice.hypotheses[0], gold_only=True
    )
    # ORNAMENT_EXTRA diagonal, or CANDIDATE_EXTRA and SKIP_ORNAMENT
    # in either order.
    assert torch.allclose(gold, torch.tensor(math.log(3.0)))


def test_viterbi_and_exact_identity_round_trip() -> None:
    candidate = IdentityCandidate(60, 0.0, 0.5)
    target = IdentityTarget("match", (0, 1), 0, 0, 60)
    unit = OrnamentTemplateUnit(60, 0.0, "linked", 0, 0)
    lattice = IdentityLattice(
        (candidate,),
        _score(),
        (_hypothesis((unit,)),),
        (target,),
    )
    model = _zero_model()
    events, deletions, diagnostics = decode_identity_crf(model, lattice)
    assert events[0].score_span == (0, 1)
    assert events[0].relationship == "match"
    assert deletions == frozenset()
    assert diagnostics["actions"]["link"] == 1
    assert exact_identity_round_trip(lattice)["passed"]


def test_repeat_copy_identity_round_trip() -> None:
    score = _score()
    candidates = (
        IdentityCandidate(60, 0.0, 0.5),
        IdentityCandidate(60, 1.0, 1.5),
    )
    targets = (
        IdentityTarget("match", (0, 1), 0, 0, 60),
        IdentityTarget("copy", (0, 1), 1, 1, 60),
    )
    template = (
        OrnamentTemplateUnit(60, 0.0, "linked", 0, 0),
        OrnamentTemplateUnit(60, 1.0, "linked", 0, 1),
    )
    hypothesis = LatticeHypothesis(
        grammar=GrammarHypothesis(
            (0, 1), 1, ((0, 0), (0, 1)), (0.0, 1.0)
        ),
        template=template,
        is_gold_compatible=True,
        hard_negative_rank=0.0,
    )
    lattice = IdentityLattice(
        candidates, score, (hypothesis,), targets
    )
    assert exact_identity_round_trip(lattice)["passed"]
