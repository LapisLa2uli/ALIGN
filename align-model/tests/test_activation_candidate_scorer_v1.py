from __future__ import annotations

import torch

from alignmodel.transcription.activation_candidate_scorer_v1 import (
    ActivationCandidateScorer,
    candidate_features,
    focal_candidate_loss,
)


def _candidate() -> dict:
    return {
        "confidence": 0.4,
        "note_peak": 0.5,
        "note_mean": 0.3,
        "onset_peak": 0.6,
        "contour_peak": 0.4,
        "onset_contrast": 0.3,
        "lower_harmonic": 0.1,
        "upper_harmonic": 0.2,
        "duration_frames": 4,
        "source_kind": "fixed_4_frames",
        "alternative_confidences": [0.5, 0.2],
    }


def test_candidate_features_are_fixed_width() -> None:
    assert len(candidate_features(_candidate())) == 16


def test_focal_candidate_loss_is_differentiable() -> None:
    model = ActivationCandidateScorer(hidden=16)
    features = torch.tensor(
        [candidate_features(_candidate()) for _ in range(4)]
    )
    logits = model(features)
    loss = focal_candidate_loss(
        logits,
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
        torch.tensor([1.0, 1.0, 2.0, 1.0]),
        positive_weight=2.0,
    )
    assert torch.isfinite(loss)
    loss.backward()
