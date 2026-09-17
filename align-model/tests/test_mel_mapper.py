from __future__ import annotations

import torch

from alignmodel.joint.index import ScoreEvent
from alignmodel.joint.lattice import JointCandidate
from alignmodel.joint.mel_mapper import (
    FEATURE_DIM,
    ContextualRepeatMapper,
    MapperState,
    decode_anchor_mapper,
    edge_features,
)
from alignmodel.joint.sequence_mapper import FixedScoreSequenceMapper


def _score(pitches: list[int]) -> tuple[ScoreEvent, ...]:
    return tuple(
        ScoreEvent(index, pitch, index, index + 1, 1, ())
        for index, pitch in enumerate(pitches)
    )


def _events(pitches: list[int]) -> tuple[JointCandidate, ...]:
    return tuple(
        JointCandidate(pitch, index, index + 0.5, 1.0)
        for index, pitch in enumerate(pitches)
    )


def test_contextual_mapper_feature_shape() -> None:
    score = _score([60, 62, 64])
    events = _events([60, 62, 64])
    assert edge_features(events, score, 0, 0, MapperState()).shape == (
        FEATURE_DIM,
    )
    assert ContextualRepeatMapper()(torch.zeros(2, FEATURE_DIM)).shape == (2,)


def test_anchor_extra_has_exclusive_rendered_identity() -> None:
    mapped = decode_anchor_mapper(
        _events([60, 61, 62]),
        _score([60, 62]),
    )
    assert mapped[1].relationship == "extra"
    assert mapped[1].rendered_index == 1


def test_sequence_mapper_supports_variable_score_lengths() -> None:
    model = FixedScoreSequenceMapper(hidden_dim=32)
    location, operation, span = model(
        torch.ones(2, 4, dtype=torch.long),
        torch.zeros(2, 4, 4),
        torch.ones(2, 7, dtype=torch.long),
        torch.rand(2, 7),
        torch.tensor([[True] * 7, [True] * 5 + [False] * 2]),
    )
    assert location.shape == (2, 4, 8)
    assert operation.shape == (2, 4, 4)
    assert span.shape == (2, 4, 129)
