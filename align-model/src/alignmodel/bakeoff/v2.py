"""V2 sparse match-default: balanced-ish CE, no tiles, confidence gate, match bias."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from alignmodel.bakeoff.even_common import copies_and_coverage, decode_runs_no_tile, notes_from_tensors
from alignmodel.config import FRAME_HOP_SEC, MELODY_NOTE_CLASSES, ModelConfig
from alignmodel.melody import load_bundle_notes
from alignmodel.melody_model import MelodyFirst, class_index, types_from_logits
from alignmodel.melody_train import MelodyBundleDataset, MelodyTrainConfig

VARIANT = "v2"
RUN_ID = "melody-b-v2-sparse"
MATCH_DECODE_BIAS = 0.8
CONF_MIN = 0.55
ERR_CE_WEIGHT = 1.25
MATCH_CE_WEIGHT = 1.0


class MelodySparse(MelodyFirst):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__(cfg)
        with torch.no_grad():
            self.type_head.bias[class_index("match")] = 1.0
            self.type_head.bias[class_index("repetition")] = -0.4


def build_model(cfg: ModelConfig) -> nn.Module:
    return MelodySparse(cfg)


def compute_loss(
    outputs: dict[str, Tensor],
    batch: dict,
    cfg: MelodyTrainConfig,
) -> tuple[Tensor, dict[str, float]]:
    type_logits = outputs["type_logits"]
    note_mask = batch["note_mask"]
    y = batch["score_y"]
    match_i = class_index("match")
    ce = nn.functional.cross_entropy(type_logits.transpose(1, 2), y, reduction="none")
    match_mask = note_mask & (y == match_i)
    err_mask = note_mask & (y != match_i)
    match_loss = (ce * match_mask).sum() / match_mask.sum().clamp_min(1)
    err_loss = (ce * err_mask).sum() / err_mask.sum().clamp_min(1)
    type_loss = MATCH_CE_WEIGHT * match_loss + ERR_CE_WEIGHT * err_loss
    extra, copies_loss, coverage = copies_and_coverage(
        type_logits,
        outputs["copies_logits"],
        note_mask,
        batch["copies_y"],
        cfg.copies_loss_weight,
        cfg.coverage_loss_weight,
    )
    total = type_loss + extra
    return total, {
        "loss": float(total.detach()),
        "type": float(type_loss.detach()),
        "err": float(err_loss.detach()),
        "copies": float(copies_loss.detach()),
        "coverage": float(coverage.detach()),
    }


def decode_outputs(outputs: dict[str, Tensor], batch: dict, b: int) -> tuple[list[str], list[dict]]:
    n = int(batch["n_notes"][b])
    logits = outputs["type_logits"][b, :n]
    types = types_from_logits(logits, match_bias=MATCH_DECODE_BIAS)
    probs = torch.softmax(logits, dim=-1)
    copies = int(outputs["copies_logits"][b].argmax(-1).cpu())
    notes = notes_from_tensors(batch["pitch"][b], batch["onset"][b], batch["duration"][b], n)
    labels = decode_runs_no_tile(types, notes, copies, type_probs=probs, conf_min=CONF_MIN)
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
    logits = out["type_logits"][0, :n]
    types = types_from_logits(logits, match_bias=MATCH_DECODE_BIAS)
    probs = torch.softmax(logits, dim=-1)
    extra_copies = int(out["copies_logits"][0].argmax(-1).cpu())
    notes = load_bundle_notes(sample_dir)[:n]
    labels = decode_runs_no_tile(types, notes, extra_copies, type_probs=probs, conf_min=CONF_MIN)
    return {
        "sample_id": sample_dir.name,
        "labels": labels,
        "extra_copies": extra_copies,
        "note_types": types,
    }


def pred_type_ids(outputs: dict[str, Tensor], n: int, b: int = 0) -> Tensor:
    logits = outputs["type_logits"][b, :n].clone()
    logits[..., class_index("match")] = logits[..., class_index("match")] + MATCH_DECODE_BIAS
    return logits.argmax(-1)
