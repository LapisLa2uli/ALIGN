from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from alignmodel.config import FRAME_HOP_SEC, SCORE_CLASSES, ModelConfig
from alignmodel.dataset import AlignBundleDataset
from alignmodel.model import RumaLite


def load_model(ckpt_path: Path, device: torch.device) -> RumaLite:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**blob["config"])
    model = RumaLite(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model


@torch.no_grad()
def infer_sample(model: RumaLite, sample_dir: Path, device: torch.device) -> dict:
    ds = AlignBundleDataset([sample_dir], model.cfg)
    item = ds[0]
    mel = item["mel"].unsqueeze(0).to(device)
    mel_mask = item["mel_mask"].unsqueeze(0).to(device)
    pitch = item["pitch"].unsqueeze(0).to(device)
    onset = item["onset"].unsqueeze(0).to(device)
    duration = item["duration"].unsqueeze(0).to(device)
    note_mask = item["note_mask"].unsqueeze(0).to(device)
    out = model(mel, mel_mask, pitch, onset, duration, note_mask, FRAME_HOP_SEC)
    score_pred = out["score_logits"][0].argmax(-1).cpu()
    extra_prob = torch.sigmoid(out["extra_logits"][0]).cpu().numpy()
    repeat_prob = float(torch.sigmoid(out["repeat_logits"][0]).cpu())
    notes = []
    n = int(item["n_notes"])
    for i in range(n):
        pred_i = int(score_pred[i])
        true_i = int(item["score_y"][i])
        notes.append(
            {
                "index": i,
                "pitch": int(item["pitch"][i]),
                "onset": float(item["onset"][i]),
                "duration": float(item["duration"][i]),
                "pred": SCORE_CLASSES[pred_i],
                "label": SCORE_CLASSES[true_i],
            }
        )
    stride = model.cfg.audio_stride
    extra_spans = _frames_to_spans(extra_prob > 0.5, FRAME_HOP_SEC * stride)
    return {
        "sample_id": item["sample_id"],
        "repeat_prob": repeat_prob,
        "repeat_pred": repeat_prob >= 0.5,
        "repeat_label": bool(item["repeat_y"].item()),
        "notes": notes,
        "extra_spans": extra_spans,
    }


def _frames_to_spans(flags: np.ndarray, hop: float) -> list[dict]:
    spans = []
    on = None
    for i, flag in enumerate(flags.tolist()):
        if flag and on is None:
            on = i
        if not flag and on is not None:
            spans.append({"start_time": round(on * hop, 4), "end_time": round(i * hop, 4)})
            on = None
    if on is not None:
        spans.append(
            {"start_time": round(on * hop, 4), "end_time": round(len(flags) * hop, 4)}
        )
    return spans


def write_prediction(result: dict, path: Path) -> None:
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
