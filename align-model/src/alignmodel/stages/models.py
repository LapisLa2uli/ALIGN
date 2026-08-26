from __future__ import annotations

import torch
from torch import nn

EDIT_CLASSES = (
    "match",
    "missed_note",
    "extra_note",
    "wrong_note",
    "intonation_error",
)


class RestartScorer(nn.Module):
    """Binary: this performance span restates an earlier span."""

    def __init__(self, n_mels: int = 128, hidden: int = 128):
        super().__init__()
        d_in = n_mels * 2 + 4
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class EditCropNet(nn.Module):
    """Classify a log-mel crop as match / miss / extra / wrong / intonation."""

    def __init__(self, n_mels: int = 128, n_classes: int = 5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, mel_crop):
        h = self.conv(mel_crop).squeeze(-1)
        return self.head(h)


class RhythmNet(nn.Module):
    """Binary rhythm error from duration-ratio features."""

    def __init__(self, d_in: int = 8, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
