from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

EDIT_CLASSES = (
    "match",
    "missed_note",
    "extra_note",
    "wrong_note",
    "intonation_error",
)


class MelEncoder(nn.Module):
    def __init__(self, n_mels: int = 128, d_out: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, d_out, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, mel_crop: torch.Tensor) -> torch.Tensor:
        return self.conv(mel_crop).squeeze(-1)


class RestartEncoder(nn.Module):
    """Mean+max pooled conv so a full restated measure stays distinctive."""

    def __init__(self, n_mels: int = 128, d_out: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, d_out, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )
        self.proj = nn.Linear(d_out * 2, d_out)

    def forward(self, mel_crop: torch.Tensor) -> torch.Tensor:
        h = self.conv(mel_crop)
        stats = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)
        return self.proj(stats)


class RestartScorer(nn.Module):
    """Siamese: is crop_a a restatement of crop_b?"""

    def __init__(self, n_mels: int = 128, hidden: int = 64):
        super().__init__()
        self.enc = RestartEncoder(n_mels, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, 1),
        )

    def encode(self, crop: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.enc(crop), dim=-1)

    def forward(self, crop_a: torch.Tensor, crop_b: torch.Tensor) -> torch.Tensor:
        ha = self.encode(crop_a)
        hb = self.encode(crop_b)
        x = torch.cat([ha, hb, (ha - hb).abs(), ha * hb], dim=-1)
        return self.head(x).squeeze(-1)


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
    """Binary rhythm error: time-warped mel crop plus duration/energy stats."""

    def __init__(self, n_mels: int = 128, hidden: int = 64, n_aux: int = 4):
        super().__init__()
        self.enc = MelEncoder(n_mels, hidden)
        self.aux = nn.Sequential(
            nn.Linear(n_aux, 16),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + 16, hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, 1),
        )

    def forward(self, mel_crop: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        h = self.enc(mel_crop)
        if aux is None:
            aux = torch.zeros(h.shape[0], 4, device=h.device, dtype=h.dtype)
        a = self.aux(aux)
        return self.head(torch.cat([h, a], dim=-1)).squeeze(-1)


def cosine_pair_loss(ha: torch.Tensor, hb: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pull copies together, push non-copies apart."""
    cos = F.cosine_similarity(ha, hb, dim=-1)
    pos = (1.0 - cos) * y
    neg = F.relu(cos - 0.15) * (1.0 - y)
    return (pos + neg).mean()
