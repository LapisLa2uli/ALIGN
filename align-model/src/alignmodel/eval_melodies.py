from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from alignmodel.melody import (
    match_note_wise_labels_detail,
    gold_melodies_from_labels,
    load_bundle_notes,
    match_melodies_detail,
    pred_melodies_from_labels,
)
from alignmodel.pipeline import run_pipeline
from alignmodel.types import PipelineConfig, pipeline_label_to_dict


def _label_dicts_from_pipeline(state) -> list[dict[str, Any]]:
    return [pipeline_label_to_dict(lab) for lab in state.labels]


def _note_per_type_rows(
    gold: list[dict[str, Any]],
    pred: list[dict[str, Any]],
    *,
    score_event_count: int,
) -> dict[str, dict[str, Any]]:
    output = {}
    for kind in sorted(
        {str(value.get("type")) for value in [*gold, *pred]}
    ):
        detail = match_note_wise_labels_detail(
            [value for value in gold if str(value.get("type")) == kind],
            [value for value in pred if str(value.get("type")) == kind],
            score_event_count=score_event_count,
        )
        output[kind] = detail
    return output


def _per_type_rows(gold, pred, *, soft: bool, ignore_type: bool) -> dict[str, dict[str, Any]]:
    types = sorted({item.type for item in gold} | {item.type for item in pred})
    out: dict[str, dict[str, Any]] = {}
    for kind in types:
        gold_k = [item for item in gold if item.type == kind]
        pred_k = [item for item in pred if item.type == kind]
        detail = match_melodies_detail(gold_k, pred_k, soft=soft, ignore_type=ignore_type)
        out[kind] = {
            "n_gold": len(gold_k),
            "n_pred": len(pred_k),
            "n_matched": round(float(detail.get("n_matched", 0.0)), 4),
            "melody_f1": round(detail["f1"], 4),
            "melody_precision": round(detail["precision"], 4),
            "melody_recall": round(detail["recall"], 4),
        }
    return out


def _summarize_per_type(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    kinds: set[str] = set()
    for row in rows:
        kinds.update((row.get("per_type") or {}).keys())
    summary: dict[str, dict[str, Any]] = {}
    for kind in sorted(kinds):
        n_gold = sum(int((r.get("per_type") or {}).get(kind, {}).get("n_gold", 0)) for r in rows)
        n_pred = sum(int((r.get("per_type") or {}).get(kind, {}).get("n_pred", 0)) for r in rows)
        n_matched = sum(
            float((r.get("per_type") or {}).get(kind, {}).get("n_matched", 0.0)) for r in rows
        )
        prec = n_matched / n_pred if n_pred else 0.0
        rec = n_matched / n_gold if n_gold else 0.0
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        summary[kind] = {
            "n_gold": n_gold,
            "n_pred": n_pred,
            "n_matched": round(n_matched, 4),
            "melody_f1": round(f1, 4),
            "melody_precision": round(prec, 4),
            "melody_recall": round(rec, 4),
        }
    return summary


def eval_sample(
    sample_dir: Path,
    *,
    pred_labels: list[dict[str, Any]] | None = None,
    pad_notes: int = 2,
    run_infer: bool = False,
    device: str = "cuda",
    soft: bool = False,
    ignore_type: bool = False,
    weights_dir: Path | str | None = None,
    types: set[str] | None = None,
) -> dict[str, Any]:
    sample_dir = Path(sample_dir)
    gold_doc = json.loads((sample_dir / "labels.json").read_text(encoding="utf-8"))
    gold_labels = [dict(value) for value in gold_doc.get("labels") or []]
    gold = gold_melodies_from_labels(gold_doc.get("labels") or [])
    notes = load_bundle_notes(sample_dir)
    if not gold:
        gold = pred_melodies_from_labels(gold_doc.get("labels") or [], notes, pad_notes=pad_notes)

    if pred_labels is None and run_infer:
        state = run_pipeline(sample_dir, device=device, weights_dir=weights_dir)
        pred_labels = _label_dicts_from_pipeline(state)
    pred_labels = pred_labels or []
    if types:
        gold_labels = [
            value for value in gold_labels if value.get("type") in types
        ]
        pred_labels = [
            value for value in pred_labels if value.get("type") in types
        ]
    official = match_note_wise_labels_detail(
        gold_labels,
        pred_labels,
        score_event_count=len(notes),
        type_mismatch_credit=1.0 if ignore_type else 0.5,
    )
    pred = pred_melodies_from_labels(pred_labels, notes, pad_notes=pad_notes)
    if types:
        gold = [item for item in gold if item.type in types]
        pred = [item for item in pred if item.type in types]
    detail = match_melodies_detail(gold, pred, soft=soft, ignore_type=ignore_type)
    available = official["status"] == "available"
    f1 = round(float(official["f1"]), 4) if available else None
    precision = round(float(official["precision"]), 4) if available else None
    recall = round(float(official["recall"]), 4) if available else None
    return {
        "sample": sample_dir.name,
        "metric_schema": "align-note-wise-score-event-metric-v1",
        "official_note_wise": official,
        "n_gold": len(gold_labels),
        "n_pred": len(pred_labels),
        "n_matched": round(float(official.get("credit", 0.0)), 4),
        "melody_f1": f1,
        "melody_precision": precision,
        "melody_recall": recall,
        "legacy_pitch_similarity_f1": round(detail["f1"], 4),
        "legacy_pitch_similarity_precision": round(detail["precision"], 4),
        "legacy_pitch_similarity_recall": round(detail["recall"], 4),
        "melody_similarity": round(detail["f1"], 4),
        "note_set_iou": round(detail["precision"], 4),
        "soft": soft,
        "ignore_type": ignore_type,
        "similarity_sum": round(float(detail.get("similarity_sum") or 0.0), 4),
        "gold_types": [g.type for g in gold],
        "pred_types": [p.type for p in pred],
        "per_type": (
            _note_per_type_rows(
                gold_labels, pred_labels, score_event_count=len(notes)
            )
            if available
            else {}
        ),
    }


def summarize_eval_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = [
        row for row in rows if row["official_note_wise"]["status"] == "available"
    ]
    n = max(len(available), 1)
    return {
        "n_samples": len(rows),
        "official_note_wise_available": len(available),
        "official_note_wise_unavailable": len(rows) - len(available),
        "mean_melody_f1": round(sum(r["melody_f1"] for r in available) / n, 4) if available else None,
        "mean_melody_precision": round(sum(r["melody_precision"] for r in available) / n, 4)
        if available
        else None,
        "mean_melody_recall": round(sum(r["melody_recall"] for r in available) / n, 4)
        if available
        else None,
        "mean_n_gold": round(sum(r["n_gold"] for r in available) / n, 3) if available else 0.0,
        "mean_n_pred": round(sum(r["n_pred"] for r in available) / n, 3) if available else 0.0,
        "per_type": _summarize_per_type(available),
        "samples": rows,
    }


def eval_dirs(
    sample_dirs: list[Path],
    *,
    stages: set[int] | None = None,
    weights_dir: Path | str | None = None,
    device: str = "cuda",
    soft: bool = False,
    ignore_type: bool = False,
    pad_notes: int = 2,
    config=None,
    types: set[str] | None = None,
) -> dict[str, Any]:
    """Run the pipeline on a fixed clip list and score official melody F1."""
    cfg = config or PipelineConfig()
    if weights_dir is not None:
        cfg.weights_dir = str(weights_dir)
    elif config is None:
        cfg.weights_dir = None
    rows = []
    for i, sample in enumerate(sample_dirs, start=1):
        print(f"eval {i}/{len(sample_dirs)} {sample.name}", flush=True)
        state = run_pipeline(
            sample,
            stages=stages,
            device=device,
            weights_dir=weights_dir,
            config=cfg,
        )
        row = eval_sample(
            sample,
            pred_labels=_label_dicts_from_pipeline(state),
            pad_notes=pad_notes,
            device=device,
            soft=soft,
            ignore_type=ignore_type,
            types=types,
        )
        counts: dict[str, int] = {}
        for lab in state.labels:
            counts[lab.type] = counts.get(lab.type, 0) + 1
        row["label_counts"] = counts
        row["pred_labels"] = _label_dicts_from_pipeline(state)
        rows.append(row)
    summary = summarize_eval_rows(rows)
    summary["soft"] = soft
    summary["ignore_type"] = ignore_type
    summary["stages"] = sorted(stages or {1, 2, 3})
    summary["weights_dir"] = str(weights_dir) if weights_dir is not None else None
    summary["types"] = sorted(types) if types else None
    return summary


def eval_root(
    root: Path,
    *,
    pred_name: str = "pipeline_pred.json",
    run_infer: bool = False,
    max_samples: int = 0,
    pad_notes: int = 2,
    device: str = "cuda",
    soft: bool = False,
    ignore_type: bool = False,
    weights_dir: Path | str | None = None,
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
            soft=soft,
            ignore_type=ignore_type,
            weights_dir=weights_dir,
        )
        rows.append(row)
    available = [
        row for row in rows if row["official_note_wise"]["status"] == "available"
    ]
    n = max(len(available), 1)
    mean_f1 = sum(r["melody_f1"] for r in available) / n if available else None
    mean_prec = sum(r["melody_precision"] for r in available) / n if available else None
    mean_rec = sum(r["melody_recall"] for r in available) / n if available else None
    return {
        "root": str(root),
        "n_samples": len(rows),
        "official_note_wise_available": len(available),
        "official_note_wise_unavailable": len(rows) - len(available),
        "mean_melody_f1": round(mean_f1, 4) if mean_f1 is not None else None,
        "mean_melody_precision": round(mean_prec, 4) if mean_prec is not None else None,
        "mean_melody_recall": round(mean_rec, 4) if mean_rec is not None else None,
        "mean_melody_similarity": None,
        "mean_note_set_iou": None,
        "mean_n_gold": round(sum(r["n_gold"] for r in available) / n, 3) if available else 0.0,
        "mean_n_pred": round(sum(r["n_pred"] for r in available) / n, 3) if available else 0.0,
        "soft": soft,
        "ignore_type": ignore_type,
        "samples": rows,
    }
