"""V5: type-agnostic gold-core mask with Dice+BCE. Isolated from V1."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from alignmodel.bakeoff.common import copies_from_out, emit_schema12, soft_dice
from alignmodel.config import MELODY_NOTE_CLASSES, ModelConfig
from alignmodel.melody import ScoreSoundingNote
from alignmodel.melody_model import class_index
from alignmodel.model import AudioEncoder, HierarchicalFusion, ScoreEncoder

MASK_THRESHOLD = 0.5
MIN_RUN_NOTES = 2
MAX_RUN_NOTES = 16
TYPE_AUX_WEIGHT = 0.25


class MelodyDice(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.audio = AudioEncoder(self.cfg)
        self.score = ScoreEncoder(self.cfg)
        self.fusion = HierarchicalFusion(self.cfg)
        self.mask_head = nn.Linear(self.cfg.d_model, 1)
        self.type_head = nn.Linear(self.cfg.d_model, len(MELODY_NOTE_CLASSES))
        self.copies_head = nn.Linear(self.cfg.d_model, 3)
        with torch.no_grad():
            self.mask_head.bias.fill_(-1.2)
            self.type_head.bias.zero_()
            self.type_head.bias[class_index("match")] = 0.2

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
            "mask_logits": self.mask_head(fused_score).squeeze(-1),
            "type_logits": self.type_head(fused_score),
            "copies_logits": self.copies_head(pooled),
            "audio_mask": audio_mask,
        }


def compute_loss(outputs: dict[str, Tensor], batch: dict, cfg) -> tuple[Tensor, dict[str, float]]:
    note_mask = batch["note_mask"]
    mask_f = note_mask.float()
    gold = batch["mask_y"].float()
    logits = outputs["mask_logits"]
    probs = torch.sigmoid(logits)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, gold, reduction="none")
    bce_loss = (bce * mask_f).sum() / mask_f.sum().clamp_min(1)
    dice_loss = soft_dice(probs, gold, mask=mask_f)
    core = note_mask & (gold > 0.5)
    type_ce = nn.functional.cross_entropy(
        outputs["type_logits"].transpose(1, 2), batch["score_y"], reduction="none"
    )
    type_aux = (type_ce * core).sum() / core.sum().clamp_min(1)
    copies_loss = nn.functional.cross_entropy(outputs["copies_logits"], batch["copies_y"])
    total = bce_loss + dice_loss + TYPE_AUX_WEIGHT * type_aux + cfg.copies_loss_weight * copies_loss
    return total, {
        "loss": float(total.detach()),
        "type": float(type_aux.detach()),
        "err": float(dice_loss.detach()),
        "copies": float(copies_loss.detach()),
        "coverage": float(bce_loss.detach()),
    }


def decode_mask_runs(
    mask: list[bool],
    type_ids: list[int],
    notes: list[ScoreSoundingNote],
    extra_copies: int,
    scores: list[float] | None = None,
    pad_notes: int = 2,
    min_notes: int = MIN_RUN_NOTES,
    max_notes: int = MAX_RUN_NOTES,
) -> list[dict[str, Any]]:
    """Contiguous mask runs. One span per run, min 2 / max 16, no tiles."""
    n = min(len(mask), len(notes))
    labels: list[dict[str, Any]] = []
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        lo, hi = i, j
        length = hi - lo
        if length < min_notes:
            i = j
            continue
        if length > max_notes:
            if scores is not None:
                best_s, best = lo, -1.0
                for s in range(lo, hi - max_notes + 1):
                    sc = sum(scores[s : s + max_notes]) / max_notes
                    if sc > best:
                        best, best_s = sc, s
                lo, hi = best_s, best_s + max_notes
            else:
                lo, hi = i, i + max_notes
        votes = type_ids[lo:hi]
        tid = max(set(votes), key=votes.count) if votes else class_index("wrong")
        kind = MELODY_NOTE_CLASSES[int(tid)]
        if kind == "match":
            kind = "wrong"
        item = emit_schema12(notes, lo, hi, kind, extra_copies, pad_notes=pad_notes)
        item["id"] = f"mel_{len(labels):03d}"
        labels.append(item)
        i = j
    return labels


def decode_sample(out: dict[str, Tensor], index: int, n: int, notes: list[ScoreSoundingNote]):
    probs = torch.sigmoid(out["mask_logits"][index, :n]).detach().cpu()
    mask = (probs >= MASK_THRESHOLD).tolist()
    if isinstance(mask, bool):
        mask = [mask]
    tids = out["type_logits"][index, :n].argmax(-1).detach().cpu().tolist()
    if isinstance(tids, int):
        tids = [tids]
    scores = probs.tolist()
    if isinstance(scores, float):
        scores = [scores]
    return decode_mask_runs(
        [bool(x) for x in mask],
        [int(x) for x in tids],
        notes,
        copies_from_out(out, index),
        scores=[float(x) for x in scores],
    )
