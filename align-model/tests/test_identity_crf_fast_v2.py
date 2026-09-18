from __future__ import annotations

import copy

import torch

from alignmodel.joint.grammar_mapper_v2 import GrammarHypothesis
from alignmodel.joint.identity_crf_fast_v2 import (
    fast_decode_identity_crf,
    fast_hypothesis_log_partition,
    fast_identity_crf_nll,
)
from alignmodel.joint.identity_crf_v1 import (
    IdentityCandidate,
    IdentityLattice,
    IdentityTarget,
    LatticeHypothesis,
    OrnamentIdentityCRF,
    decode_identity_crf,
    hypothesis_log_partition,
    identity_crf_nll,
)
from alignmodel.joint.index import ScoreEvent
from alignmodel.joint.ornament_mapper_v1 import OrnamentTemplateUnit


def _lattice() -> IdentityLattice:
    score = (
        ScoreEvent(0, 60, 0.0, 1.0, (0,)),
        ScoreEvent(1, 62, 1.0, 2.0, (1,)),
    )
    candidates = (
        IdentityCandidate(60, 0.0, 0.5, 0.9),
        IdentityCandidate(61, 0.5, 0.7, 0.8),
        IdentityCandidate(62, 1.0, 1.5, 0.95),
    )
    targets = (
        IdentityTarget("match", (0, 1), 0, 0, 60),
        IdentityTarget("extra", None, 0, 1, 61),
        IdentityTarget("match", (1, 2), 0, 2, 62),
    )
    template = (
        OrnamentTemplateUnit(60, 0.0, "linked", 0, 0),
        OrnamentTemplateUnit(
            61, 0.5, "ornament_extra", None, 0, "mordent"
        ),
        OrnamentTemplateUnit(62, 1.0, "linked", 1, 0),
    )
    hypothesis = LatticeHypothesis(
        GrammarHypothesis(
            None, 0, ((0, 0), (1, 0)), (0.0, 1.0)
        ),
        template,
        True,
        0.0,
    )
    return IdentityLattice(candidates, score, (hypothesis,), targets)


def test_fast_partitions_match_reference_strictly() -> None:
    torch.manual_seed(17)
    model = OrnamentIdentityCRF(hidden=8)
    lattice = _lattice()
    hypothesis = lattice.hypotheses[0]
    for gold_only in (False, True):
        reference = hypothesis_log_partition(
            model, lattice, hypothesis, gold_only=gold_only
        )
        fast = fast_hypothesis_log_partition(
            model, lattice, hypothesis, gold_only=gold_only
        )
        assert torch.allclose(reference, fast, atol=2e-5, rtol=2e-6)


def test_fast_loss_and_parameter_gradients_match_reference() -> None:
    torch.manual_seed(23)
    reference_model = OrnamentIdentityCRF(hidden=8)
    fast_model = copy.deepcopy(reference_model)
    lattice = _lattice()
    reference_loss, _ = identity_crf_nll(
        reference_model, lattice, normalize=False
    )
    fast_loss, _ = fast_identity_crf_nll(
        fast_model, lattice, normalize=False
    )
    assert torch.allclose(reference_loss, fast_loss, atol=2e-5, rtol=2e-6)
    reference_loss.backward()
    fast_loss.backward()
    for (reference_name, reference_parameter), (
        fast_name,
        fast_parameter,
    ) in zip(reference_model.named_parameters(), fast_model.named_parameters()):
        assert reference_name == fast_name
        assert reference_parameter.grad is not None
        assert fast_parameter.grad is not None
        assert torch.allclose(
            reference_parameter.grad,
            fast_parameter.grad,
            atol=3e-5,
            rtol=3e-5,
        ), reference_name


def test_fast_viterbi_matches_reference_identity_path() -> None:
    torch.manual_seed(31)
    model = OrnamentIdentityCRF(hidden=8)
    lattice = _lattice()
    reference_events, reference_deletions, reference_diagnostics = (
        decode_identity_crf(model, lattice)
    )
    fast_events, fast_deletions, fast_diagnostics = (
        fast_decode_identity_crf(model, lattice)
    )
    assert fast_events == reference_events
    assert fast_deletions == reference_deletions
    assert fast_diagnostics["copies"] == reference_diagnostics["copies"]
    assert fast_diagnostics["source_span"] == reference_diagnostics[
        "source_span"
    ]
    assert fast_diagnostics["actions"] == reference_diagnostics["actions"]


def test_fast_semimarkov_span_matches_reference() -> None:
    score = (
        ScoreEvent(0, 60, 0.0, 1.0, (0,)),
        ScoreEvent(1, 60, 1.0, 2.0, (1,)),
    )
    candidate = (IdentityCandidate(60, 0.0, 2.0, 1.0),)
    target = (IdentityTarget("match", (0, 2), 0, 0, 60),)
    template = (
        OrnamentTemplateUnit(60, 0.0, "linked", 0, 0),
        OrnamentTemplateUnit(60, 1.0, "linked", 1, 0),
    )
    hypothesis = LatticeHypothesis(
        GrammarHypothesis(
            None, 0, ((0, 0), (1, 0)), (0.0, 1.0)
        ),
        template,
        True,
        0.0,
    )
    lattice = IdentityLattice(
        candidate, score, (hypothesis,), target
    )
    torch.manual_seed(37)
    reference_model = OrnamentIdentityCRF(hidden=8)
    fast_model = copy.deepcopy(reference_model)
    reference_loss, _ = identity_crf_nll(
        reference_model, lattice, normalize=False
    )
    fast_loss, _ = fast_identity_crf_nll(
        fast_model, lattice, normalize=False
    )
    assert torch.allclose(reference_loss, fast_loss, atol=2e-5, rtol=2e-6)
    reference_events, reference_deletions, _ = decode_identity_crf(
        reference_model, lattice
    )
    fast_events, fast_deletions, _ = fast_decode_identity_crf(
        fast_model, lattice
    )
    assert fast_events == reference_events
    assert fast_deletions == reference_deletions
