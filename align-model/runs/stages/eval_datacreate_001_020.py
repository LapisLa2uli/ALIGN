"""Evaluate trained stage weights on DataCreate samples 001-020."""
from __future__ import annotations

import json
import traceback
from collections import Counter, defaultdict
from pathlib import Path

from alignmodel.pipeline import run_pipeline
from alignmodel.stages.gold import load_labels

ROOT = Path(r"D:\stuff\Audio Evaluation\ALIGN\DataCreate\samples")
WEIGHTS = Path(r"D:\stuff\Audio Evaluation\ALIGN\align-model\runs\stages")
OUT = Path(r"D:\stuff\Audio Evaluation\ALIGN\align-model\runs\stages\eval_datacreate_001_020.json")

SCORED = (
    "repetition",
    "extra_note",
    "wrong_note",
    "missed_note",
    "rhythm_error",
    "intonation_error",
)
IOU_PRIMARY = 0.3
IOU_STRICT = 0.5


def iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 1e-9 else 0.0


def match(gold: list[dict], pred: list[dict], thr: float) -> tuple[int, int, int]:
    used = [False] * len(pred)
    tp = 0
    for g in gold:
        g0, g1 = float(g["start_time"]), float(g["end_time"])
        best_i, best = -1, -1.0
        for i, p in enumerate(pred):
            if used[i]:
                continue
            score = iou(g0, g1, float(p["start_time"]), float(p["end_time"]))
            if score > best:
                best, best_i = score, i
        if best_i >= 0 and best >= thr:
            used[best_i] = True
            tp += 1
    return tp, len(pred) - tp, len(gold) - tp


def prf(tp: int, fp: int, fn: int) -> dict:
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


def main() -> None:
    rows = []
    gold_all: dict[str, list] = defaultdict(list)
    pred_all: dict[str, list] = defaultdict(list)
    gold_types = Counter()
    pred_types = Counter()
    errors = []

    for i in range(1, 21):
        sid = f"{i:03d}"
        sample = ROOT / sid
        gold_raw = [lab for lab in load_labels(sample) if lab.get("type") in SCORED]
        try:
            state = run_pipeline(sample, stages={1, 2, 3}, device="cuda", weights_dir=WEIGHTS)
            pred_raw = [
                {
                    "type": lab.type,
                    "start_time": lab.start_time,
                    "end_time": lab.end_time,
                    "comment": lab.comment,
                }
                for lab in state.labels
                if lab.type in SCORED
            ]
            status = "ok"
            n_seg = len(state.segments)
        except Exception as exc:
            pred_raw = []
            status = f"error: {exc}"
            n_seg = 0
            errors.append({"id": sid, "error": traceback.format_exc(limit=4)})
            print(f"{sid} FAIL {exc}")

        gold_c = Counter(lab["type"] for lab in gold_raw)
        pred_c = Counter(lab["type"] for lab in pred_raw)
        gold_types.update(gold_c)
        pred_types.update(pred_c)
        per_type = {}
        tp_s = fp_s = fn_s = 0
        for kind in SCORED:
            g = [lab for lab in gold_raw if lab["type"] == kind]
            p = [lab for lab in pred_raw if lab["type"] == kind]
            gold_all[kind].extend(g)
            pred_all[kind].extend(p)
            tp, fp, fn = match(g, p, IOU_PRIMARY)
            tp_s += tp
            fp_s += fp
            fn_s += fn
            per_type[kind] = prf(tp, fp, fn) | {"gold": len(g), "pred": len(p)}

        micro = prf(tp_s, fp_s, fn_s)
        row = {
            "id": sid,
            "status": status,
            "n_segments": n_seg,
            "gold": dict(gold_c),
            "pred": dict(pred_c),
            "micro_iou03": micro,
            "per_type": per_type,
        }
        rows.append(row)
        print(
            f"{sid} gold={dict(gold_c) or '-'} pred={dict(pred_c) or '-'} "
            f"P={micro['precision']:.2f} R={micro['recall']:.2f} F1={micro['f1']:.2f} "
            f"seg={n_seg} {status}"
        )

    by_type = {}
    for kind in SCORED:
        tp, fp, fn = match(gold_all[kind], pred_all[kind], IOU_PRIMARY)
        by_type[kind] = prf(tp, fp, fn) | {
            "gold": len(gold_all[kind]),
            "pred": len(pred_all[kind]),
        }
        tp5, fp5, fn5 = match(gold_all[kind], pred_all[kind], IOU_STRICT)
        by_type[kind]["f1_iou05"] = prf(tp5, fp5, fn5)["f1"]

    micro_tp = sum(by_type[k]["tp"] for k in SCORED)
    micro_fp = sum(by_type[k]["fp"] for k in SCORED)
    micro_fn = sum(by_type[k]["fn"] for k in SCORED)
    summary = {
        "n_clips": 20,
        "scored_types": list(SCORED),
        "iou": IOU_PRIMARY,
        "gold_counts": dict(gold_types),
        "pred_counts": dict(pred_types),
        "micro": prf(micro_tp, micro_fp, micro_fn),
        "by_type": by_type,
        "n_errors": len(errors),
    }
    payload = {"summary": summary, "clips": rows, "errors": errors}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summary, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
