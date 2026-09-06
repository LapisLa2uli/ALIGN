from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from alignmodel.melody import (
    gold_melodies_from_labels,
    load_bundle_notes,
    match_melodies_detail,
    pred_melodies_from_labels,
)
from alignmodel.pipeline import run_pipeline
from alignmodel.types import pipeline_label_to_dict


def _label_dicts_from_pipeline(state) -> list[dict[str, Any]]:
    return [pipeline_label_to_dict(lab) for lab in state.labels]


def eval_sample(
    sample_dir: Path,
    *,
    pred_labels: list[dict[str, Any]] | None = None,
    pad_notes: int = 2,
    run_infer: bool = False,
    device: str = "cuda",
) -> dict[str, Any]:
    sample_dir = Path(sample_dir)
    gold_doc = json.loads((sample_dir / "labels.json").read_text(encoding="utf-8"))
    gold = gold_melodies_from_labels(gold_doc.get("labels") or [])
    notes = load_bundle_notes(sample_dir)
    if not gold:
        gold = pred_melodies_from_labels(gold_doc.get("labels") or [], notes, pad_notes=pad_notes)

    if pred_labels is None and run_infer:
        state = run_pipeline(sample_dir, device=device)
        pred_labels = _label_dicts_from_pipeline(state)
    pred_labels = pred_labels or []
    pred = pred_melodies_from_labels(pred_labels, notes, pad_notes=pad_notes)
    detail = match_melodies_detail(gold, pred)
    f1 = round(detail["f1"], 4)
    precision = round(detail["precision"], 4)
    recall = round(detail["recall"], 4)
    return {
        "sample": sample_dir.name,
        "n_gold": len(gold),
        "n_pred": len(pred),
        "n_matched": int(detail.get("n_matched", detail["n_pred_correct"])),
        "melody_f1": f1,
        "melody_precision": precision,
        "melody_recall": recall,
        "melody_similarity": f1,
        "note_set_iou": precision,
        "gold_types": [g.type for g in gold],
        "pred_types": [p.type for p in pred],
    }


def eval_root(
    root: Path,
    *,
    pred_name: str = "pipeline_pred.json",
    run_infer: bool = False,
    max_samples: int = 0,
    pad_notes: int = 2,
    device: str = "cuda",
) -> dict[str, Any]:
    root = Path(root)
    dirs = sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and (p / "labels.json").exists() and (p / "verified_score.musicxml").exists()
    )
    if max_samples:
        dirs = dirs[: max_samples]
    rows = []
    for sample in dirs:
        pred_labels = None
        pred_path = sample / pred_name
        if (not run_infer) and pred_path.exists():
            pred_labels = json.loads(pred_path.read_text(encoding="utf-8")).get("labels") or []
        row = eval_sample(
            sample,
            pred_labels=pred_labels,
            pad_notes=pad_notes,
            run_infer=run_infer,
            device=device,
        )
        rows.append(row)
    n = max(len(rows), 1)
    mean_f1 = sum(r["melody_f1"] for r in rows) / n if rows else 0.0
    mean_prec = sum(r["melody_precision"] for r in rows) / n if rows else 0.0
    mean_rec = sum(r["melody_recall"] for r in rows) / n if rows else 0.0
    return {
        "root": str(root),
        "n_samples": len(rows),
        "mean_melody_f1": round(mean_f1, 4),
        "mean_melody_precision": round(mean_prec, 4),
        "mean_melody_recall": round(mean_rec, 4),
        "mean_melody_similarity": round(mean_f1, 4),
        "mean_note_set_iou": round(mean_prec, 4),
        "mean_n_gold": round(sum(r["n_gold"] for r in rows) / n, 3) if rows else 0.0,
        "mean_n_pred": round(sum(r["n_pred"] for r in rows) / n, 3) if rows else 0.0,
        "samples": rows,
    }
