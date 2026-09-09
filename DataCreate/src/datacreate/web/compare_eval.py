"""Pair gold vs pipeline labels for the compare viewer."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from datacreate.melody import (
    MATCH_SIMILARITY_THRESHOLD,
    TYPE_MISMATCH_SCALE,
    WeakMelody,
    _exclusive_pairs,
    melody_label_score,
    melody_pair_score,
)
from datacreate.utils import read_json

TIME_IOU_HIT = 0.3


def default_eval_dir() -> Path:
    env = os.environ.get("ALIGN_COMPARE_EVAL")
    if env:
        return Path(env)
    return (
        Path(__file__).resolve().parents[4]
        / "align-model"
        / "runs"
        / "eval-datacreate-typed"
    )


def load_summary(eval_dir: Path) -> dict[str, Any]:
    path = eval_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Eval summary not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _time_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    try:
        a0, a1 = float(a["start_time"]), float(a["end_time"])
        b0, b1 = float(b["start_time"]), float(b["end_time"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    start = max(a0, b0)
    end = min(a1, b1)
    inter = max(0.0, end - start)
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def _pitches(lab: dict[str, Any]) -> list[int]:
    raw = lab.get("pitches")
    if not isinstance(raw, list) or not raw:
        return []
    try:
        return [int(p) for p in raw]
    except (TypeError, ValueError):
        return []


def _range_score(pred: dict[str, Any], gold: dict[str, Any]) -> float:
    pp, gp = _pitches(pred), _pitches(gold)
    if pp and gp:
        return melody_pair_score(pp, gp)
    return _time_iou(pred, gold)


def _credit(pred: dict[str, Any], gold: dict[str, Any], range_score: float) -> tuple[str, float]:
    pp, gp = _pitches(pred), _pitches(gold)
    if pp and gp:
        wm_p = WeakMelody(pitches=pp, type=pred.get("type"))
        wm_g = WeakMelody(pitches=gp, type=gold.get("type"))
        credit = melody_label_score(wm_p, wm_g, soft=False, ignore_type=False)
        if credit >= 1.0 - 1e-9:
            return "full", 1.0
        if credit >= TYPE_MISMATCH_SCALE - 1e-9:
            return "type_mismatch", TYPE_MISMATCH_SCALE
        return "unmatched", 0.0
    hit = range_score >= TIME_IOU_HIT
    if not hit:
        return "unmatched", 0.0
    if str(pred.get("type") or "") == str(gold.get("type") or ""):
        return "full", 1.0
    return "type_mismatch", TYPE_MISMATCH_SCALE


def pair_labels(
    gold: list[dict[str, Any]], pred: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Annotate copies of gold/pred with exclusive match status."""
    gold_out = [dict(lab) for lab in gold]
    pred_out = [dict(lab) for lab in pred]
    for i, lab in enumerate(gold_out):
        lab["compare"] = {
            "layer": "gold",
            "index": i,
            "match": "unmatched",
            "pair_id": None,
            "pair_index": None,
            "credit": 0.0,
            "range_score": 0.0,
        }
    for i, lab in enumerate(pred_out):
        lab["compare"] = {
            "layer": "pred",
            "index": i,
            "match": "unmatched",
            "pair_id": None,
            "pair_index": None,
            "credit": 0.0,
            "range_score": 0.0,
        }
    if not gold_out or not pred_out:
        return gold_out, pred_out
    scores = [
        [_range_score(p, g) for g in gold_out] for p in pred_out
    ]
    for i, j, raw in _exclusive_pairs(scores):
        match, credit = _credit(pred_out[i], gold_out[j], raw)
        if match == "unmatched":
            continue
        pred_id = pred_out[i].get("id")
        gold_id = gold_out[j].get("id")
        gold_out[j]["compare"].update(
            {
                "match": match,
                "pair_id": pred_id,
                "pair_index": i,
                "credit": credit,
                "range_score": round(float(raw), 4),
            }
        )
        pred_out[i]["compare"].update(
            {
                "match": match,
                "pair_id": gold_id,
                "pair_index": j,
                "credit": credit,
                "range_score": round(float(raw), 4),
            }
        )
    return gold_out, pred_out


def sample_payload(
    sample_dir: Path, eval_dir: Path, summary: dict[str, Any]
) -> dict[str, Any]:
    sample_id = sample_dir.name
    gold_doc = (
        read_json(sample_dir / "labels.json")
        if (sample_dir / "labels.json").exists()
        else {"labels": []}
    )
    pred_path = eval_dir / "preds" / f"{sample_id}.json"
    pred_doc = read_json(pred_path) if pred_path.exists() else {"labels": []}
    gold = gold_doc.get("labels") or []
    pred = pred_doc.get("labels") or []
    gold_out, pred_out = pair_labels(gold, pred)
    metrics = None
    for row in summary.get("samples") or []:
        if str(row.get("sample")) == sample_id:
            metrics = {
                "n_gold": row.get("n_gold"),
                "n_pred": row.get("n_pred"),
                "hard_type_sensitive": row.get("hard_type_sensitive"),
                "hard_type_insensitive": row.get("hard_type_insensitive"),
                "soft_type_sensitive": row.get("soft_type_sensitive"),
                "soft_type_insensitive": row.get("soft_type_insensitive"),
            }
            break
    return {
        "sample_id": sample_id,
        "gold": gold_out,
        "pred": pred_out,
        "metrics": metrics,
        "pred_path": str(pred_path) if pred_path.exists() else None,
        "match_threshold": MATCH_SIMILARITY_THRESHOLD,
    }
