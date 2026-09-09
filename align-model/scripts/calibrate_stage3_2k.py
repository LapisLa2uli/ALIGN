"""Calibrate stage-3 DataCreate DTW + rhythm detector on the raw 2k holdout."""

from __future__ import annotations

import copy
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

from alignmodel.dataset import list_sample_dirs
from alignmodel.eval_melodies import eval_sample
from alignmodel.melody import (
    gold_melodies_from_labels,
    load_bundle_notes,
    match_melodies_detail,
    pred_melodies_from_labels,
)
from alignmodel.melody import attach_schema12_fields
from alignmodel.pipeline import run_pipeline
from alignmodel.stage_train import _load_mel
from alignmodel.stages.dc_alignment import ensure_rhythm_pairs
from alignmodel.stages.gold import load_first_pass_labels
from alignmodel.stages.learned import load_stage_models
from alignmodel.stages.rhythm import run_stage3
from alignmodel.types import PipelineConfig, pipeline_label_to_dict, schema12_document

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("E:/output_2k_rawdata")
WEIGHTS = ROOT / "align-model" / "runs" / "stages-random12k-typed"
OUT = ROOT / "align-model" / "runs" / "eval-stage3-dc-2k"
SUMMARY = OUT / "summary.json"
HOLDOUT_N = 100
SEED = 365


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--max-samples", type=int, default=HOLDOUT_N)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def val_like_dirs(root: Path, n: int, seed: int = SEED) -> list[Path]:
    dirs = [
        p
        for p in list_sample_dirs(root)
        if (p / "verified_score.musicxml").exists()
    ]
    dirs = sorted(dirs, key=lambda p: p.name)
    rng = random.Random(seed)
    shuffled = list(dirs)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * 0.1))
    if len(shuffled) > 1:
        n_val = min(n_val, len(shuffled) - 1)
    return shuffled[:n_val][:n]


def _mean_block(rows: list[dict]) -> dict:
    n = max(len(rows), 1)
    keys = (
        "melody_f1",
        "melody_precision",
        "melody_recall",
        "n_pred",
        "n_gold",
        "rhythm_f1",
        "rhythm_precision",
        "rhythm_recall",
        "n_pred_rhythm",
        "n_gold_rhythm",
    )
    out = {"n_samples": len(rows)}
    for key in keys:
        out[f"mean_{key}" if not key.startswith("n_") else f"mean_{key}"] = round(
            sum(float(r.get(key) or 0.0) for r in rows) / n, 4
        )
    out["pred_type_counts"] = dict(
        sum((Counter(r.get("pred_types") or []) for r in rows), Counter())
    )
    out["gold_type_counts"] = dict(
        sum((Counter(r.get("gold_types") or []) for r in rows), Counter())
    )
    return out


def _rhythm_detail(sample: Path, labels: list[dict]) -> dict:
    gold_doc = json.loads((sample / "labels.json").read_text(encoding="utf-8"))
    gold = [
        m
        for m in gold_melodies_from_labels(gold_doc.get("labels") or [])
        if m.type == "rhythm_error"
    ]
    notes = load_bundle_notes(sample)
    pred = [
        m
        for m in pred_melodies_from_labels(labels, notes, pad_notes=2)
        if m.type == "rhythm_error"
    ]
    detail = match_melodies_detail(gold, pred, soft=False, ignore_type=False)
    return {
        "rhythm_f1": round(detail["f1"], 4),
        "rhythm_precision": round(detail["precision"], 4),
        "rhythm_recall": round(detail["recall"], 4),
        "n_pred_rhythm": len(pred),
        "n_gold_rhythm": len(gold),
    }


def _score_labels(sample: Path, labels: list[dict]) -> dict:
    hard = eval_sample(sample, pred_labels=labels, soft=False)
    hard.update(_rhythm_detail(sample, labels))
    return hard


def _variants(ckpt_thr: float) -> list[dict]:
    return [
        {"name": "heuristic", "rhythm_detector": "heuristic", "rhythm_threshold": None},
        {
            "name": "gated_net_ckpt",
            "rhythm_detector": "gated_net",
            "rhythm_threshold": ckpt_thr,
        },
        {
            "name": "gated_net_0.5",
            "rhythm_detector": "gated_net",
            "rhythm_threshold": 0.5,
        },
        {
            "name": "gated_net_1.0",
            "rhythm_detector": "gated_net",
            "rhythm_threshold": 1.0,
        },
        {"name": "net_ckpt", "rhythm_detector": "net", "rhythm_threshold": ckpt_thr},
        {"name": "net_0.5", "rhythm_detector": "net", "rhythm_threshold": 0.5},
        {
            "name": "heuristic_ewma_0.35",
            "rhythm_detector": "heuristic",
            "rhythm_threshold": None,
            "ewma_log_threshold": 0.35,
        },
    ]


def _apply_variant(state, learned, mel, variant: dict, ckpt_thr: float):
    if variant["rhythm_threshold"] is not None:
        state.config.rhythm_logit_override = float(variant["rhythm_threshold"])
        learned.rhythm_threshold = float(variant["rhythm_threshold"])
    else:
        state.config.rhythm_logit_override = None
        learned.rhythm_threshold = ckpt_thr
    state.config.rhythm_detector = variant["rhythm_detector"]
    if "ewma_log_threshold" in variant:
        state.config.ewma_log_threshold = float(variant["ewma_log_threshold"])
    else:
        state.config.ewma_log_threshold = 0.25
    run_stage3(state, learned=learned, mel=mel)
    attach_schema12_fields(state)


def main() -> None:
    args = _parse_args()
    data = args.data
    weights = args.weights
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "eval.log"
    log_f = open(log_path, "a", encoding="utf-8", buffering=1)
    class _Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, data):
            for handle in self.files:
                handle.write(data)
                handle.flush()

        def flush(self):
            for handle in self.files:
                handle.flush()

        def isatty(self):
            return False

    sys.stdout = _Tee(sys.__stdout__, log_f)
    sys.stderr = _Tee(sys.__stderr__, log_f)

    if not data.exists():
        raise SystemExit(f"missing 2k dataset {data}")
    holdout = val_like_dirs(data, args.max_samples, seed=args.seed)
    print(f"stage3-dc 2k holdout n={len(holdout)} data={data} weights={weights}", flush=True)
    learned = load_stage_models(weights, "cuda")
    ckpt_thr = float(learned.rhythm_threshold)
    print(f"stage3 logit_threshold={ckpt_thr:.3f}", flush=True)
    variants = _variants(ckpt_thr)
    by_name = {v["name"]: [] for v in variants}
    pred_by_name: dict[str, dict[str, list]] = {v["name"]: {} for v in variants}
    t0 = time.time()
    cfg = PipelineConfig()
    for i, sample in enumerate(holdout, start=1):
        state12 = run_pipeline(
            sample,
            stages={1, 2},
            device="cuda",
            weights_dir=weights,
            config=copy.deepcopy(cfg),
        )
        ensure_rhythm_pairs(state12)
        mel = _load_mel(sample) if (sample / "performance_mel.npy").exists() else None
        gold_types = [lab.get("type") for lab in load_first_pass_labels(sample)]
        for variant in variants:
            state = copy.deepcopy(state12)
            _apply_variant(state, learned, mel, variant, ckpt_thr)
            labels = [pipeline_label_to_dict(lab) for lab in state.labels]
            pred_by_name[variant["name"]][sample.name] = labels
            row = _score_labels(sample, labels)
            row["gold_first_pass_types"] = gold_types
            row["n_rhythm_pairs"] = len(state.rhythm_pairs)
            by_name[variant["name"]].append(row)
        if i == 1 or i % 10 == 0:
            gated = by_name["gated_net_ckpt"][-1]
            print(
                f"{i}/{len(holdout)} {sample.name} "
                f"gated_f1={gated['melody_f1']:.3f} "
                f"n_pred={gated['n_pred']} n_rhythm={gated['n_pred_rhythm']} "
                f"pairs={gated['n_rhythm_pairs']}",
                flush=True,
            )

    summary = {
        "data": str(data),
        "weights": str(weights),
        "holdout": f"raw2k seed={args.seed} first 10% then {args.max_samples}",
        "n_samples": len(holdout),
        "stage3_ckpt_threshold": ckpt_thr,
        "elapsed_min": round((time.time() - t0) / 60, 2),
        "variants": {},
    }
    best_name = None
    best_key = None
    for variant in variants:
        name = variant["name"]
        block = _mean_block(by_name[name])
        summary["variants"][name] = {
            **{k: v for k, v in block.items() if k != "samples"},
            "config": {k: variant[k] for k in variant if k != "name"},
            "samples": by_name[name],
        }
        key = (
            block["mean_melody_f1"],
            -block["mean_n_pred"],
            block["mean_rhythm_f1"],
        )
        print(
            f"VARIANT {name} hard_f1={block['mean_melody_f1']:.3f} "
            f"p={block['mean_melody_precision']:.3f} r={block['mean_melody_recall']:.3f} "
            f"n_pred={block['mean_n_pred']:.2f} rhythm_f1={block['mean_rhythm_f1']:.3f} "
            f"n_pred_rhythm={block['mean_n_pred_rhythm']:.2f} "
            f"n_gold_rhythm={block['mean_n_gold_rhythm']:.2f}",
            flush=True,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_name = name

    summary["winner"] = best_name
    pred_dir = out / f"preds_{best_name}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing winner preds detector={best_name}", flush=True)
    for sample in holdout:
        labels = pred_by_name[best_name][sample.name]
        doc = schema12_document(
            sample_id=sample.name,
            labels=labels,
            annotator_id="align_pipeline",
            extra={"pipeline": {"sample_id": sample.name, "stages_run": [1, 2, 3]}},
        )
        (pred_dir / f"{sample.name}.json").write_text(
            json.dumps(doc, indent=2), encoding="utf-8"
        )
    summary["pred_dir"] = str(pred_dir)
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"WINNER {best_name} wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
