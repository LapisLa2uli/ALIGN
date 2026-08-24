from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from alignmodel.config import ModelConfig


class SinusoidalTime(nn.Module):
    """Map a scalar time in seconds to a d_model vector."""

    def __init__(self, d_model: int, max_period: float = 10_000.0):
        super().__init__()
        self.d_model = d_model
        half = d_model // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / max(half, 1)
        )
        self.register_buffer("freqs", freqs)

    def forward(self, seconds: Tensor) -> Tensor:
        # seconds: [B, T]
        ang = seconds.unsqueeze(-1) * self.freqs
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)[..., : self.d_model]


class AudioEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.stride = cfg.audio_stride
        self.down = nn.Sequential(
            nn.Conv1d(cfg.n_mels, cfg.d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(
                cfg.d_model,
                cfg.d_model,
                kernel_size=cfg.audio_stride,
                stride=cfg.audio_stride,
            ),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=cfg.audio_layers, enable_nested_tensor=False
        )
        self.time_enc = SinusoidalTime(cfg.d_model)

    def forward(self, mel: Tensor, frame_mask: Tensor, hop_sec: float) -> Tensor:
        # mel: [B, M, T], frame_mask True = valid
        x = self.down(mel).transpose(1, 2)  # [B, T', D]
        t = x.size(1)
        stride = self.stride
        times = torch.arange(t, device=mel.device, dtype=mel.dtype) * hop_sec * stride
        x = x + self.time_enc(times.unsqueeze(0).expand(x.size(0), -1))
        # Downsample mask by requiring all frames in the stride window? use any-valid.
        if frame_mask.size(-1) >= t * stride:
            m = frame_mask[:, : t * stride].view(frame_mask.size(0), t, stride).any(dim=-1)
        else:
            m = torch.ones(x.size(0), t, dtype=torch.bool, device=mel.device)
        padded = ~m
        return self.blocks(x, src_key_padding_mask=padded)


class ScoreEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.pitch = nn.Embedding(128, cfg.d_model)
        self.time_enc = SinusoidalTime(cfg.d_model)
        self.dur_proj = nn.Linear(1, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=cfg.score_layers, enable_nested_tensor=False
        )

    def forward(
        self,
        pitch: Tensor,
        onset: Tensor,
        duration: Tensor,
        note_mask: Tensor,
    ) -> Tensor:
        x = self.pitch(pitch) + self.time_enc(onset) + self.dur_proj(duration.unsqueeze(-1))
        padded = ~note_mask
        return self.blocks(x, src_key_padding_mask=padded)


class HierarchicalFusion(nn.Module):
    """RUMAA-style: audio cross-attn, then score self-attn, stacked."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.layers = nn.ModuleList()
        for _ in range(cfg.fusion_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "audio_attn": nn.MultiheadAttention(
                            d, cfg.nhead, dropout=cfg.dropout, batch_first=True
                        ),
                        "audio_norm": nn.LayerNorm(d),
                        "self_attn": nn.MultiheadAttention(
                            d, cfg.nhead, dropout=cfg.dropout, batch_first=True
                        ),
                        "self_norm": nn.LayerNorm(d),
                        "ff": nn.Sequential(
                            nn.Linear(d, d * 4),
                            nn.GELU(),
                            nn.Dropout(cfg.dropout),
                            nn.Linear(d * 4, d),
                            nn.Dropout(cfg.dropout),
                        ),
                        "ff_norm": nn.LayerNorm(d),
                    }
                )
            )
        self.audio_to_score = nn.MultiheadAttention(
            d, cfg.nhead, dropout=cfg.dropout, batch_first=True
        )
        self.audio_out_norm = nn.LayerNorm(d)

    def forward(
        self,
        score: Tensor,
        audio: Tensor,
        note_mask: Tensor,
        audio_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        score_pad = ~note_mask
        audio_pad = ~audio_mask
        h = score
        for layer in self.layers:
            ctx, _ = layer["audio_attn"](
                h, audio, audio, key_padding_mask=audio_pad, need_weights=False
            )
            h = layer["audio_norm"](h + ctx)
            ctx, _ = layer["self_attn"](
                h, h, h, key_padding_mask=score_pad, need_weights=False
            )
            h = layer["self_norm"](h + ctx)
            h = layer["ff_norm"](h + layer["ff"](h))
            h = h.masked_fill(score_pad.unsqueeze(-1), 0.0)
        a_ctx, _ = self.audio_to_score(
            audio, h, h, key_padding_mask=score_pad, need_weights=False
        )
        audio_out = self.audio_out_norm(audio + a_ctx)
        audio_out = audio_out.masked_fill(audio_pad.unsqueeze(-1), 0.0)
        return h, audio_out


class RumaLite(nn.Module):
    """
    ALIGN-native RUMAA: MusicXML notes + performance mel in, Match/Miss/Wrong/
    Rhythm/Intonation on score notes, Insert on audio frames, Repeat at clip level.
    """

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.audio = AudioEncoder(self.cfg)
        self.score = ScoreEncoder(self.cfg)
        self.fusion = HierarchicalFusion(self.cfg)
        self.score_head = nn.Linear(self.cfg.d_model, self.cfg.num_score_classes)
        self.extra_head = nn.Linear(self.cfg.d_model, 1)
        self.repeat_head = nn.Linear(self.cfg.d_model, 1)

    def forward(
        self,
        mel: Tensor,
        mel_mask: Tensor,
        pitch: Tensor,
        onset: Tensor,
        duration: Tensor,
        note_mask: Tensor,
        hop_sec: float,
    ) -> dict[str, Tensor]:
        audio_h = self.audio(mel, mel_mask, hop_sec)
        t_audio = audio_h.size(1)
        stride = self.cfg.audio_stride
        if mel_mask.size(-1) >= t_audio * stride:
            audio_mask = mel_mask[:, : t_audio * stride].view(
                mel_mask.size(0), t_audio, stride
            ).any(dim=-1)
        else:
            audio_mask = torch.ones(
                audio_h.size(0), t_audio, dtype=torch.bool, device=mel.device
            )
        score_h = self.score(pitch, onset, duration, note_mask)
        fused_score, fused_audio = self.fusion(score_h, audio_h, note_mask, audio_mask)
        score_logits = self.score_head(fused_score)
        extra_logits = self.extra_head(fused_audio).squeeze(-1)
        pooled = (fused_score * note_mask.unsqueeze(-1)).sum(1) / note_mask.sum(
            1, keepdim=True
        ).clamp_min(1)
        repeat_logits = self.repeat_head(pooled).squeeze(-1)
        return {
            "score_logits": score_logits,
            "extra_logits": extra_logits,
            "repeat_logits": repeat_logits,
            "audio_mask": audio_mask,
        }
