from __future__ import annotations

import json
from pathlib import Path

import torch

from alignmodel.config import FRAME_HOP_SEC, ModelConfig
from alignmodel.melody import load_bundle_notes
from alignmodel.melody_model import MelodyFirst, decode_note_runs, types_from_logits
from alignmodel.melody_train import MelodyBundleDataset
from alignmodel.types import schema12_document


def load_melody_model(ckpt_path: Path, device: torch.device) -> MelodyFirst:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**blob["config"])
    model = MelodyFirst(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model


@torch.no_grad()
def infer_melody_sample(
    model: MelodyFirst, sample_dir: Path, device: torch.device
) -> dict:
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
    types = types_from_logits(out["type_logits"][0, :n])
    extra_copies = int(out["copies_logits"][0].argmax(-1).cpu())
    notes = load_bundle_notes(sample_dir)[:n]
    labels = decode_note_runs(types, notes, extra_copies)
    return {
        "sample_id": sample_dir.name,
        "labels": labels,
        "extra_copies": extra_copies,
        "note_types": types,
    }


def write_melody_prediction(result: dict, path: Path) -> None:
    doc = schema12_document(
        sample_id=result["sample_id"],
        labels=result["labels"],
        annotator_id="align_melody",
        extra={"melody": {"extra_copies": result.get("extra_copies")}},
    )
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
