"""V4 binary fault mask + type-on-faults. No tiles."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from alignmodel.bakeoff.even_common import (
    MATCH_I,
    copies_and_coverage,
    decode_fault_runs,
    encode_fused,
)
from alignmodel.config import FRAME_HOP_SEC, MELODY_NOTE_CLASSES, ModelConfig
from alignmodel.melody import load_bundle_notes
from alignmodel.melody_model import class_index
from alignmodel.melody_train import MelodyBundleDataset, MelodyTrainConfig
from alignmodel.model import AudioEncoder, HierarchicalFusion, ScoreEncoder

VARIANT = "v4"
RUN_ID = "melody-b-v4-binary"
FAULT_THRESHOLD = 0.5


class MelodyBinary(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.audio = AudioEncoder(self.cfg)
        self.score = ScoreEncoder(self.cfg)
        self.fusion = HierarchicalFusion(self.cfg)
        self.fault_head = nn.Linear(self.cfg.d_model, 1)
        self.type_head = nn.Linear(self.cfg.d_model, len(MELODY_NOTE_CLASSES))
        self.copies_head = nn.Linear(self.cfg.d_model, 3)
        with torch.no_grad():
            self.fault_head.bias.fill_(-0.5)
            self.type_head.bias.zero_()
            self.type_head.bias[class_index("match")] = -0.2

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
        fused, audio_mask = encode_fused(
            self.audio,
            self.score,
            self.fusion,
            self.cfg,
            mel,
            mel_mask,
            pitch,
            onset,
            duration,
            note_mask,
            hop_sec,
        )
        fault_logits = self.fault_head(fused).squeeze(-1)
        type_logits = self.type_head(fused)
        pooled = (fused * note_mask.unsqueeze(-1)).sum(1) / note_mask.sum(1, keepdim=True).clamp_min(1)
        copies_logits = self.copies_head(pooled)
        return {
            "fault_logits": fault_logits,
            "type_logits": type_logits,
            "copies_logits": copies_logits,
            "audio_mask": audio_mask,
        }


def build_model(cfg: ModelConfig) -> nn.Module:
    return MelodyBinary(cfg)


def _dice_loss(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    probs = torch.sigmoid(logits)
    mask_f = mask.float()
    intersection = (probs * target * mask_f).sum()
    denom = (probs * mask_f).sum() + (target * mask_f).sum()
    return 1.0 - (2.0 * intersection + 1.0) / (denom + 1.0)


def compute_loss(
    outputs: dict[str, Tensor],
    batch: dict,
    cfg: MelodyTrainConfig,
) -> tuple[Tensor, dict[str, float]]:
    note_mask = batch["note_mask"]
    y = batch["score_y"]
    fault_y = (y != MATCH_I).float()
    fault_logits = outputs["fault_logits"]
    pos = (fault_y * note_mask.float()).sum()
    neg = ((1.0 - fault_y) * note_mask.float()).sum()
    pos_weight = (neg / pos.clamp_min(1.0)).clamp(0.5, 8.0)
    bce = nn.functional.binary_cross_entropy_with_logits(
        fault_logits,
        fault_y,
        reduction="none",
        pos_weight=pos_weight,
    )
    bce_loss = (bce * note_mask.float()).sum() / note_mask.float().sum().clamp_min(1)
    dice = _dice_loss(fault_logits, fault_y, note_mask)
    fault_loss = 0.5 * bce_loss + 0.5 * dice

    type_logits = outputs["type_logits"]
    fault_mask = note_mask & (y != MATCH_I)
    ce = nn.functional.cross_entropy(type_logits.transpose(1, 2), y, reduction="none")
    type_loss = (ce * fault_mask).sum() / fault_mask.sum().clamp_min(1)

    extra, copies_loss, coverage = copies_and_coverage(
        type_logits,
        outputs["copies_logits"],
        note_mask,
        batch["copies_y"],
        cfg.copies_loss_weight,
        cfg.coverage_loss_weight,
    )
    total = fault_loss + type_loss + extra
    err_loss = type_loss
    return total, {
        "loss": float(total.detach()),
        "type": float((fault_loss + type_loss).detach()),
        "err": float(err_loss.detach()),
        "copies": float(copies_loss.detach()),
        "coverage": float(coverage.detach()),
        "fault": float(fault_loss.detach()),
        "dice": float(dice.detach()),
    }


def decode_outputs(outputs: dict[str, Tensor], batch: dict, b: int) -> tuple[list[str], list[dict]]:
    from alignmodel.bakeoff.even_common import notes_from_tensors

    n = int(batch["n_notes"][b])
    fault_prob = torch.sigmoid(outputs["fault_logits"][b, :n])
    copies = int(outputs["copies_logits"][b].argmax(-1).cpu())
    notes = notes_from_tensors(batch["pitch"][b], batch["onset"][b], batch["duration"][b], n)
    types, labels = decode_fault_runs(
        fault_prob,
        outputs["type_logits"][b, :n],
        notes,
        copies,
        threshold=FAULT_THRESHOLD,
    )
    return types, labels


@torch.no_grad()
def infer_sample(model: nn.Module, sample_dir: Path, device: torch.device) -> dict:
    sample_dir = Path(sample_dir)
    ds = MelodyBundleDataset([sample_dir], model.cfg)
    item = ds[0]
    out = model(
        item["mel"].unsqueeze(0).to(device),
        item["mel_mask"].unsqueeze(0).to(device),
        item["pitch"].unsqueeze(0).to(device),
        item["onset"].unsqueeze(0).to(device),
        item["duration"].unsqueeze(0).to(device),
        item["note_mask"].unsqueeze(0).to(device),
        FRAME_HOP_SEC,
    )
    n = int(item["n_notes"])
    fault_prob = torch.sigmoid(out["fault_logits"][0, :n])
    extra_copies = int(out["copies_logits"][0].argmax(-1).cpu())
    notes = load_bundle_notes(sample_dir)[:n]
    types, labels = decode_fault_runs(
        fault_prob,
        out["type_logits"][0, :n],
        notes,
        extra_copies,
        threshold=FAULT_THRESHOLD,
    )
    return {
        "sample_id": sample_dir.name,
        "labels": labels,
        "extra_copies": extra_copies,
        "note_types": types,
    }


def pred_type_ids(outputs: dict[str, Tensor], n: int, b: int = 0) -> Tensor:
    fault = torch.sigmoid(outputs["fault_logits"][b, :n]) > FAULT_THRESHOLD
    logits = outputs["type_logits"][b, :n].clone()
    logits[..., MATCH_I] = -1e9
    ids = logits.argmax(-1)
    return torch.where(fault, ids, torch.full_like(ids, MATCH_I))
