"""V3: BIO span heads. Isolated from V1 MelodyFirst weights."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from alignmodel.bakeoff.common import (
    BIO_B,
    BIO_I,
    BIO_O,
    FAULT_CLASSES,
    copies_from_out,
    emit_schema12,
)
from alignmodel.config import ModelConfig
from alignmodel.melody import ScoreSoundingNote
from alignmodel.model import AudioEncoder, HierarchicalFusion, ScoreEncoder


class MelodyBIO(nn.Module):
    """BIO + per-span type. Same encoders as MelodyFirst; different heads."""

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.audio = AudioEncoder(self.cfg)
        self.score = ScoreEncoder(self.cfg)
        self.fusion = HierarchicalFusion(self.cfg)
        self.bio_head = nn.Linear(self.cfg.d_model, 3)
        self.type_head = nn.Linear(self.cfg.d_model, len(FAULT_CLASSES))
        self.copies_head = nn.Linear(self.cfg.d_model, 3)
        with torch.no_grad():
            self.bio_head.bias.zero_()
            self.bio_head.bias[BIO_O] = 0.4
            self.bio_head.bias[BIO_B] = -0.2
            self.bio_head.bias[BIO_I] = -0.2

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
            audio_mask = (
                mel_mask[:, : t_audio * stride]
                .view(mel_mask.size(0), t_audio, stride)
                .any(dim=-1)
            )
        else:
            audio_mask = torch.ones(audio_h.size(0), t_audio, dtype=torch.bool, device=mel.device)
        score_h = self.score(pitch, onset, duration, note_mask)
        fused_score, _ = self.fusion(score_h, audio_h, note_mask, audio_mask)
        pooled = (fused_score * note_mask.unsqueeze(-1)).sum(1) / note_mask.sum(
            1, keepdim=True
        ).clamp_min(1)
        return {
            "bio_logits": self.bio_head(fused_score),
            "type_logits": self.type_head(fused_score),
            "copies_logits": self.copies_head(pooled),
            "audio_mask": audio_mask,
        }


def compute_loss(outputs: dict[str, Tensor], batch: dict, cfg) -> tuple[Tensor, dict[str, float]]:
    note_mask = batch["note_mask"]
    bio_y = batch["bio_y"]
    type_y = batch["type_span_y"]
    bio_ce = nn.functional.cross_entropy(
        outputs["bio_logits"].transpose(1, 2), bio_y, reduction="none"
    )
    bio_loss = (bio_ce * note_mask).sum() / note_mask.sum().clamp_min(1)
    span_mask = note_mask & (bio_y != BIO_O)
    type_ce = nn.functional.cross_entropy(
        outputs["type_logits"].transpose(1, 2), type_y, reduction="none"
    )
    type_loss = (type_ce * span_mask).sum() / span_mask.sum().clamp_min(1)
    copies_loss = nn.functional.cross_entropy(outputs["copies_logits"], batch["copies_y"])
    total = bio_loss + type_loss + cfg.copies_loss_weight * copies_loss
    return total, {
        "loss": float(total.detach()),
        "type": float(type_loss.detach()),
        "err": float(bio_loss.detach()),
        "copies": float(copies_loss.detach()),
        "coverage": 0.0,
    }


def decode_bio_runs(
    bio: list[int],
    type_ids: list[int],
    notes: list[ScoreSoundingNote],
    extra_copies: int,
    pad_notes: int = 2,
) -> list[dict[str, Any]]:
    """Valid B + I* runs only. No 5-note tile split."""
    n = min(len(bio), len(notes), len(type_ids))
    labels: list[dict[str, Any]] = []
    i = 0
    while i < n:
        if bio[i] != BIO_B:
            i += 1
            continue
        j = i + 1
        while j < n and bio[j] == BIO_I:
            j += 1
        votes = type_ids[i:j]
        tid = max(set(votes), key=votes.count)
        tid = max(0, min(int(tid), len(FAULT_CLASSES) - 1))
        kind = FAULT_CLASSES[tid]
        item = emit_schema12(notes, i, j, kind, extra_copies, pad_notes=pad_notes)
        item["id"] = f"mel_{len(labels):03d}"
        labels.append(item)
        i = j
    return labels


def decode_sample(out: dict[str, Tensor], index: int, n: int, notes: list[ScoreSoundingNote]):
    bio = out["bio_logits"][index, :n].argmax(-1).detach().cpu().tolist()
    if isinstance(bio, int):
        bio = [bio]
    tids = out["type_logits"][index, :n].argmax(-1).detach().cpu().tolist()
    if isinstance(tids, int):
        tids = [tids]
    return decode_bio_runs(bio, [int(x) for x in tids], notes, copies_from_out(out, index))
