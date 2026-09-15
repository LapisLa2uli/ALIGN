from __future__ import annotations

import numpy as np
import pytest

from alignmodel.joint.error_heads import (
    FEATURE_DIM,
    FEATURE_NAMES,
    HeadRow,
    direct_rhythm_probabilities,
)
from train_error_heads_v5 import assert_protocol_disjoint


def test_protocol_refuses_validation_row_leakage() -> None:
    groups = {1: "a", 2: "b", 3: "c"}
    with pytest.raises(ValueError, match="row leakage"):
        assert_protocol_disjoint([1, 3], [2], [3], groups)


def test_protocol_refuses_group_leakage() -> None:
    groups = {1: "same", 2: "cal", 3: "same"}
    with pytest.raises(ValueError, match="group leakage"):
        assert_protocol_disjoint([1], [2], [3], groups)


def test_protocol_accepts_disjoint_partitions() -> None:
    groups = {1: "fit", 2: "cal", 3: "val"}
    assert_protocol_disjoint([1], [2], [3], groups)


def test_direct_rhythm_uses_tempo_evidence_and_excludes_replay() -> None:
    features = np.zeros(FEATURE_DIM, dtype=np.float32)
    for name, value in {
        "is_mapped": 1.0,
        "candidate_confidence": 0.95,
        "previous_confidence": 0.9,
        "next_confidence": 0.9,
        "upstream_path_margin": 3.0,
        "log_tempo_normalized_duration_ratio": 1.1,
        "previous_ioi_ratio": 1.0,
        "next_ioi_ratio": 1.0,
    }.items():
        features[FEATURE_NAMES.index(name)] = value
    row = HeadRow(features, "event", 0, (1, 2), 0.0, 1.0, False)
    probability = direct_rhythm_probabilities((row,))
    assert probability[0] > 0.5
    replay = features.copy()
    replay[FEATURE_NAMES.index("is_replay_state")] = 1.0
    replay_row = HeadRow(replay, "event", 0, (1, 2), 0.0, 1.0, True)
    assert direct_rhythm_probabilities((replay_row,))[0] == 0.0
