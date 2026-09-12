from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class NoteFrameNetConfig:
    """Frame-synchronous transcriber. Defaults are v2; v1 checkpoints reconstruct via from_dict."""

    n_mels: int = 128
    channels: int = 64
    temporal_channels: int = 128
    spectral_blocks: int = 3
    temporal_blocks: int = 10
    dropout: float = 0.12
    midi_min: int = 36
    midi_max: int = 108
    use_multiscale: bool = True
    predict_cents: bool = True
    use_f0: bool = True

    @property
    def n_pitches(self) -> int:
        return self.midi_max - self.midi_min + 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict | None) -> "NoteFrameNetConfig":
        data = dict(payload or {})
        if "spectral_blocks" not in data:
            data["spectral_blocks"] = 4
            data.setdefault("temporal_channels", data.get("channels", 48))
            data.setdefault("use_multiscale", False)
            data.setdefault("predict_cents", False)
            data.setdefault("use_f0", False)
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in allowed})


class _ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: tuple[int, int]) -> None:
        super().__init__()
        groups = min(8, c_out)
        while c_out % groups:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, c_out),
            nn.SiLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _TemporalResidual(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else (4 if channels % 4 == 0 else 1)
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                5,
                padding=2 * dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


def _flatten_freq(x: Tensor) -> Tensor:
    return x.permute(0, 1, 3, 2).reshape(x.size(0), -1, x.size(3))


class NoteFrameNet(nn.Module):
    """Predict voice, written MIDI pitch, onset, offset, and intonation cents."""

    def __init__(self, config: NoteFrameNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or NoteFrameNetConfig()
        c = self.config.channels
        t_ch = self.config.temporal_channels
        blocks = []
        in_ch = 1
        out_ch = max(c // 2, 8)
        for index in range(self.config.spectral_blocks):
            next_ch = c if index else out_ch
            blocks.append(_ConvBlock(in_ch, next_ch, (2, 1)))
            in_ch = next_ch
        self.spectral = nn.ModuleList(blocks)
        reduced = self.config.n_mels
        skip_mels = None
        for index in range(self.config.spectral_blocks):
            reduced = (reduced + 1) // 2
            if self.config.use_multiscale and index == self.config.spectral_blocks - 2:
                skip_mels = reduced
        self.reduced_mels = max(1, reduced)
        if self.config.use_multiscale and skip_mels:
            self.skip_projection = nn.Conv1d(c * skip_mels, t_ch // 2, 1)
            self.spectral_projection = nn.Conv1d(c * self.reduced_mels, t_ch // 2, 1)
        else:
            self.skip_projection = None
            self.spectral_projection = nn.Conv1d(in_ch * self.reduced_mels, t_ch, 1)
        self.f0_projection = (
            nn.Conv1d(2, t_ch, 1) if self.config.use_f0 else None
        )
        dilations = [2 ** (i % 5) for i in range(self.config.temporal_blocks)]
        self.temporal = nn.Sequential(
            *[
                _TemporalResidual(t_ch, dilation, self.config.dropout)
                for dilation in dilations
            ]
        )
        self.voiced_head = nn.Conv1d(t_ch, 1, 1)
        self.pitch_head = nn.Conv1d(t_ch, self.config.n_pitches, 1)
        self.onset_head = nn.Conv1d(t_ch, 1, 1)
        self.offset_head = nn.Conv1d(t_ch, 1, 1)
        self.cents_head = nn.Conv1d(t_ch, 1, 1) if self.config.predict_cents else None

    def forward(self, mel: Tensor, f0: Tensor | None = None) -> dict[str, Tensor]:
        if mel.ndim != 3 or mel.size(1) != self.config.n_mels:
            raise ValueError(
                f"Expected mel [B,{self.config.n_mels},T], got {tuple(mel.shape)}"
            )
        x = ((mel.clamp(-100.0, 20.0) + 80.0) / 40.0).unsqueeze(1)
        skip = None
        last_index = len(self.spectral) - 1
        for index, block in enumerate(self.spectral):
            x = block(x)
            if self.skip_projection is not None and index == last_index - 1:
                skip = x
        projected = self.spectral_projection(_flatten_freq(x))
        if self.skip_projection is not None and skip is not None:
            skip_feat = self.skip_projection(_flatten_freq(skip))
            if skip_feat.size(-1) != projected.size(-1):
                skip_feat = F.interpolate(
                    skip_feat, size=projected.size(-1), mode="nearest"
                )
            projected = torch.cat([projected, skip_feat], dim=1)
        if self.f0_projection is not None:
            if f0 is None:
                f0 = mel.new_zeros(mel.size(0), 2, mel.size(-1))
            if f0.size(-1) != projected.size(-1):
                f0 = F.interpolate(f0, size=projected.size(-1), mode="nearest")
            projected = projected + self.f0_projection(f0)
        x = self.temporal(projected)
        outputs = {
            "voiced_logits": self.voiced_head(x).squeeze(1),
            "pitch_logits": self.pitch_head(x).transpose(1, 2),
            "onset_logits": self.onset_head(x).squeeze(1),
            "offset_logits": self.offset_head(x).squeeze(1),
        }
        if self.cents_head is not None:
            outputs["cents"] = 100.0 * torch.tanh(self.cents_head(x).squeeze(1))
        return outputs


def _masked_bce(
    logits: Tensor, target: Tensor, valid: Tensor, pos_weight: float
) -> Tensor:
    loss = F.binary_cross_entropy_with_logits(
        logits,
        target.float(),
        reduction="none",
        pos_weight=logits.new_tensor(pos_weight),
    )
    return (loss * valid.float()).sum() / valid.sum().clamp_min(1)


def note_frame_loss(
    outputs: dict[str, Tensor],
    batch: dict[str, Tensor],
    *,
    pitch_weight: float = 1.0,
    voiced_weight: float = 1.0,
    onset_weight: float = 2.0,
    offset_weight: float = 1.5,
    cents_weight: float = 0.5,
    onset_pos_weight: float = 12.0,
    offset_pos_weight: float = 10.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Multi-task frame loss; pitch CE and cents are evaluated on voiced frames."""

    valid = batch.get("frame_mask")
    if valid is None:
        valid = torch.ones_like(batch["voiced"], dtype=torch.bool)
    voiced_target = batch["voiced"].float()
    voiced = _masked_bce(outputs["voiced_logits"], voiced_target, valid, 2.0)
    onset = _masked_bce(
        outputs["onset_logits"], batch["onset"], valid, onset_pos_weight
    )
    offset = _masked_bce(
        outputs["offset_logits"], batch["offset"], valid, offset_pos_weight
    )

    pitch_mask = valid & (batch["pitch"] >= 0) & (batch["voiced"] > 0)
    pitch_target = batch["pitch"].long().masked_fill(~pitch_mask, -1)
    pitch = F.cross_entropy(
        outputs["pitch_logits"].transpose(1, 2),
        pitch_target,
        ignore_index=-1,
        reduction="sum",
        label_smoothing=0.05,
    ) / pitch_mask.sum().clamp_min(1)

    cents = outputs["voiced_logits"].sum() * 0.0
    if "cents" in outputs and "cents" in batch:
        cents_error = (outputs["cents"] - batch["cents"].float()).abs() / 100.0
        cents = (cents_error * pitch_mask.float()).sum() / pitch_mask.sum().clamp_min(1)
    total = (
        voiced_weight * voiced
        + pitch_weight * pitch
        + onset_weight * onset
        + offset_weight * offset
        + cents_weight * cents
    )
    return total, {
        "loss": total.detach(),
        "voiced": voiced.detach(),
        "pitch": pitch.detach(),
        "onset": onset.detach(),
        "offset": offset.detach(),
        "cents": cents.detach(),
    }
