"""Exact-supervision scorer for Basic Pitch activation interval candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


SCHEMA_VERSION = "align-activation-candidate-scorer-v1"


def candidate_features(value: Mapping[str, Any]) -> list[float]:
    alternatives = list(value.get("alternative_confidences") or ())
    top = alternatives[0] if alternatives else 0.0
    second = alternatives[1] if len(alternatives) > 1 else 0.0
    kind = str(value.get("source_kind") or "")
    return [
        float(value["confidence"]),
        float(value["note_peak"]),
        float(value["note_mean"]),
        float(value["onset_peak"]),
        float(value["contour_peak"]),
        float(value["onset_contrast"]),
        float(value["lower_harmonic"]),
        float(value["upper_harmonic"]),
        np.log1p(float(value["duration_frames"])) / 5.0,
        float(top),
        float(top - second),
        float(kind == "standard_decode"),
        float(kind.startswith("activation_run_")),
        float(kind == "fixed_2_frames"),
        float(kind == "fixed_4_frames"),
        float(kind == "fixed_8_frames"),
    ]


class ActivationCandidateScorer(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.network = nn.Sequential(
            nn.Linear(16, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features.float()).squeeze(-1)


def focal_candidate_loss(
    logits: Tensor,
    labels: Tensor,
    weights: Tensor,
    *,
    positive_weight: float,
    gamma: float = 1.0,
) -> Tensor:
    values = labels.float()
    base = F.binary_cross_entropy_with_logits(
        logits,
        values,
        reduction="none",
        pos_weight=logits.new_tensor(positive_weight),
    )
    probability = torch.sigmoid(logits)
    focal = torch.where(values > 0.5, 1.0 - probability, probability).pow(gamma)
    effective = weights.float()
    return (base * focal * effective).sum() / effective.sum().clamp_min(1.0)


@dataclass(frozen=True)
class ScoredCandidate:
    index: int
    probability: float


@torch.inference_mode()
def score_candidates(
    model: ActivationCandidateScorer,
    candidates: Sequence[Mapping[str, Any]],
    device: torch.device | str,
    *,
    batch_size: int = 8192,
) -> np.ndarray:
    features = np.asarray(
        [candidate_features(value) for value in candidates], np.float32
    )
    output = []
    model.eval()
    target = torch.device(device)
    for start in range(0, len(features), batch_size):
        logits = model(torch.from_numpy(features[start : start + batch_size]).to(target))
        output.append(torch.sigmoid(logits).float().cpu().numpy())
    return (
        np.concatenate(output).astype(np.float32)
        if output
        else np.zeros(0, np.float32)
    )
