from __future__ import annotations

import torch
from torch import nn


TYPE_NAMES = ("match", "copy", "substitute", "extra")


class MelOperationTagger(nn.Module):
    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.pitch = nn.Embedding(128, 24)
        self.input = nn.Linear(28, hidden_dim)
        self.encoder = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.output = nn.Linear(hidden_dim * 2, len(TYPE_NAMES))

    def forward(
        self, pitch: torch.Tensor, continuous: torch.Tensor
    ) -> torch.Tensor:
        value = torch.cat(
            (self.pitch(pitch.clamp(0, 127)), continuous), dim=-1
        )
        encoded, _state = self.encoder(
            torch.nn.functional.gelu(self.input(value))
        )
        return self.output(encoded)
