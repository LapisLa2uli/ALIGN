"""Contextual sequence model for repetition-aware note-to-score alignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

NOTE_FEATURE_DIM = 9


@dataclass
class ContextualAlignerConfig:
    hidden_dim: int = 64
    layers: int = 2
    dropout: float = 0.10
    deletion_cost: float = 2.0

    def to_dict(self) -> dict:
        return asdict(self)


def sequence_features(notes: list[Any]) -> np.ndarray:
    if not notes:
        return np.zeros((0, NOTE_FEATURE_DIM), np.float32)

    def value(note, name, default=0.0):
        if isinstance(note, dict):
            return float(note.get(name, default))
        return float(getattr(note, name, default))

    pitch = np.asarray([value(note, "pitch") for note in notes])
    start = np.asarray([value(note, "start") for note in notes])
    end = np.asarray([value(note, "end") for note in notes])
    confidence = np.asarray([value(note, "confidence", 1.0) for note in notes])
    duration = np.maximum(end - start, 1e-3)
    previous_ioi = np.diff(start, prepend=start[0])
    if len(start) > 1:
        previous_ioi[0] = np.median(previous_ioi[1:])
    previous_interval = np.diff(pitch, prepend=pitch[0])
    next_interval = np.diff(pitch, append=pitch[-1])
    order = np.linspace(0.0, 1.0, len(notes))
    return np.stack(
        [
            pitch / 128.0,
            np.sin(2 * np.pi * pitch / 12.0),
            np.cos(2 * np.pi * pitch / 12.0),
            np.clip(np.log(duration + 1e-3), -5.0, 3.0) / 5.0,
            np.clip(
                np.log(np.maximum(previous_ioi, 1e-3)), -5.0, 3.0
            )
            / 5.0,
            np.clip(previous_interval, -24.0, 24.0) / 24.0,
            np.clip(next_interval, -24.0, 24.0) / 24.0,
            np.clip(confidence, 0.0, 1.0),
            order,
        ],
        axis=-1,
    ).astype(np.float32)


class ContextualNoteAligner(nn.Module):
    def __init__(self, config: ContextualAlignerConfig | None = None) -> None:
        super().__init__()
        self.config = config or ContextualAlignerConfig()
        hidden = self.config.hidden_dim
        self.input_projection = nn.Linear(NOTE_FEATURE_DIM, hidden)
        self.observed_encoder = nn.GRU(
            hidden,
            hidden,
            num_layers=self.config.layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.config.dropout if self.config.layers > 1 else 0.0,
        )
        self.score_encoder = nn.GRU(
            hidden,
            hidden,
            num_layers=self.config.layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.config.dropout if self.config.layers > 1 else 0.0,
        )
        context = 2 * hidden
        self.observed_projection = nn.Linear(context, context)
        self.score_projection = nn.Linear(context, context)
        self.pair_head = nn.Sequential(
            nn.Linear(5, 24),
            nn.SiLU(),
            nn.Linear(24, 1),
        )
        self.extra_head = nn.Linear(context, 1)
        self.scale = context ** -0.5

    def forward(
        self,
        observed: torch.Tensor,
        score: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
        score_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        obs, _ = self.observed_encoder(self.input_projection(observed))
        scr, _ = self.score_encoder(self.input_projection(score))
        similarity = torch.einsum(
            "bnd,bmd->bnm",
            self.observed_projection(obs),
            self.score_projection(scr),
        ) * self.scale
        obs_pitch = observed[..., 0] * 128.0
        score_pitch = score[..., 0] * 128.0
        delta = obs_pitch.unsqueeze(2) - score_pitch.unsqueeze(1)
        pair_features = torch.stack(
            [
                delta / 24.0,
                delta.abs() / 24.0,
                (delta.abs() < 0.5).float(),
                (
                    torch.remainder(delta.abs(), 12.0) < 0.5
                ).float(),
                (
                    observed[..., 8].unsqueeze(2)
                    - score[..., 8].unsqueeze(1)
                ).abs(),
            ],
            dim=-1,
        )
        pair = similarity + self.pair_head(pair_features).squeeze(-1)
        if score_mask is not None:
            pair = pair.masked_fill(~score_mask.unsqueeze(1), -1e4)
        extra = self.extra_head(obs)
        logits = torch.cat([pair, extra], dim=-1)
        if observed_mask is not None:
            logits = logits.masked_fill(
                ~observed_mask.unsqueeze(-1), -1e4
            )
        return logits

    @torch.no_grad()
    def align(self, observed_notes: list[Any], score_notes: list[Any]) -> list[int | None]:
        if not observed_notes:
            return []
        if not score_notes:
            return [None] * len(observed_notes)
        device = next(self.parameters()).device
        observed = torch.from_numpy(sequence_features(observed_notes))[None].to(device)
        score = torch.from_numpy(sequence_features(score_notes))[None].to(device)
        logits = self(observed, score)[0]
        probabilities = logits.log_softmax(dim=-1).cpu().numpy()
        n, m = len(observed_notes), len(score_notes)
        dp = np.full((n + 1, m + 1), np.inf, np.float64)
        back = np.zeros((n + 1, m + 1), np.int8)
        dp[0, 0] = 0.0
        for i in range(1, n + 1):
            dp[i, 0] = dp[i - 1, 0] - probabilities[i - 1, m]
            back[i, 0] = 1
        for j in range(1, m + 1):
            dp[0, j] = dp[0, j - 1] + self.config.deletion_cost
            back[0, j] = 2
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                choices = (
                    (dp[i - 1, j - 1] - probabilities[i - 1, j - 1], 0),
                    (dp[i - 1, j] - probabilities[i - 1, m], 1),
                    (dp[i, j - 1] + self.config.deletion_cost, 2),
                )
                dp[i, j], back[i, j] = min(choices, key=lambda item: item[0])
        mapping: list[int | None] = [None] * n
        i, j = n, m
        while i or j:
            code = int(back[i, j])
            if i and j and code == 0:
                mapping[i - 1] = j - 1
                i -= 1
                j -= 1
            elif i and (not j or code == 1):
                i -= 1
            else:
                j -= 1
        return mapping


def save_contextual_aligner(path: Path | str, model: ContextualNoteAligner, extra=None):
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model.config.to_dict(),
            "extra": dict(extra or {}),
        },
        Path(path),
    )


def load_contextual_aligner(path: Path | str, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = ContextualNoteAligner(
        ContextualAlignerConfig(**payload.get("model_config", {}))
    )
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, dict(payload.get("extra") or {})
