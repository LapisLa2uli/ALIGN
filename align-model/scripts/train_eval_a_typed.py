"""Retrain Model A on random 12k; score with type-aware hard and soft set-F1."""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

from alignmodel.dataset import list_sample_dirs
from alignmodel.eval_melodies import eval_sample
from alignmodel.pipeline import run_pipeline, write_prediction
from alignmodel.stage_train import StageTrainConfig, train_stages
from alignmodel.types import pipeline_label_to_dict

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("E:/output")
RAW_DATA = Path("E:/output_2k_rawdata")
OUT = ROOT / "align-model" / "runs" / "stages-random12k-typed"
PRED = ROOT / "align-model" / "runs" / "eval-dual" / "A-random12k-typed_on_random12k-holdout"
PRED_RAW = ROOT / "align-model" / "runs" / "eval-dual" / "A-random12k-typed_on_raw2k-holdout"
SUMMARY = ROOT / "align-model" / "runs" / "eval-dual" / "summary_model_a_typed.json"


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


def val_like_dirs(root: Path, n: int, seed: int = 365) -> list[Path]:
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
    return {
        "n_samples": len(rows),
        "mean_melody_f1": round(sum(r["melody_f1"] for r in rows) / n, 4) if rows else 0.0,
        "mean_melody_precision": round(sum(r["melody_precision"] for r in rows) / n, 4)
        if rows
        else 0.0,
        "mean_melody_recall": round(sum(r["melody_recall"] for r in rows) / n, 4)
        if rows
        else 0.0,
        "mean_n_pred": round(sum(r["n_pred"] for r in rows) / n, 3) if rows else 0.0,
        "mean_n_gold": round(sum(r["n_gold"] for r in rows) / n, 3) if rows else 0.0,
        "samples": rows,
    }


def _eval_holdout(holdout: list[Path], pred_dir: Path, tag: str) -> tuple[list[dict], list[dict]]:
    pred_dir.mkdir(parents=True, exist_ok=True)
    hard_rows: list[dict] = []
    soft_rows: list[dict] = []
    for i, sample in enumerate(holdout, start=1):
        pred_path = pred_dir / f"{sample.name}.json"
        if pred_path.exists():
            labels = json.loads(pred_path.read_text(encoding="utf-8")).get("labels") or []
        else:
            state = run_pipeline(sample, device="cuda", weights_dir=OUT)
            write_prediction(state, pred_path)
            labels = [pipeline_label_to_dict(lab) for lab in state.labels]
        hard = eval_sample(sample, pred_labels=labels, soft=False)
        soft = eval_sample(sample, pred_labels=labels, soft=True)
        hard_rows.append(hard)
        soft_rows.append(soft)
        if i == 1 or i % 20 == 0:
            print(
                f"eval {tag} {i}/{len(holdout)} "
                f"hard_f1={hard['melody_f1']:.3f} soft_f1={soft['melody_f1']:.3f}",
                flush=True,
            )
    return hard_rows, soft_rows


def _collect_train_metrics(out: Path) -> dict:
    metrics: dict = {}
    for stage in ("stage1", "stage2", "stage3"):
        hist_path = out / f"{stage}_history.json"
        if hist_path.exists():
            rows = json.loads(hist_path.read_text(encoding="utf-8"))
            metrics[stage] = rows[-1] if rows else {}
            metrics[f"{stage}_history"] = rows
    ckpt = out / "stage2.pt"
    if ckpt.exists():
        try:
            import torch

            blob = torch.load(ckpt, map_location="cpu", weights_only=False)
            metrics["stage2_softmax_threshold"] = blob.get("softmax_threshold")
            metrics["stage2_logit_threshold"] = blob.get("logit_threshold")
            calib = blob.get("calibration") or {}
            chosen = calib.get("chosen") or {}
            metrics["stage2_calibration"] = {
                k: chosen.get(k) for k in ("t", "f1", "prec", "rec", "mean_n_pred", "mean_n_gold")
            }
        except Exception as exc:
            metrics["stage2_ckpt_read_error"] = str(exc)
    s2 = metrics.get("stage2") or {}
    metrics["stage2_collapsed_all_match"] = float(s2.get("error_acc") or 0.0) == 0.0
    return metrics


def _write_summary(summary: dict) -> None:
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {SUMMARY}", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "train.log"
    log_f = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_f)
    sys.stderr = _Tee(sys.__stderr__, log_f)

    t0 = time.time()
    print(f"train A typed data={DATA} out={OUT}", flush=True)
    train_stages(
        StageTrainConfig(
            data_root=DATA,
            output_dir=OUT,
            epochs=8,
            batch_size=32,
            device="cuda",
            stages=(1, 2, 3),
        )
    )
    train_elapsed_min = round((time.time() - t0) / 60, 2)
    stage_paths = {
        "stage1": str(OUT / "stage1.pt"),
        "stage2": str(OUT / "stage2.pt"),
        "stage3": str(OUT / "stage3.pt"),
    }
    print(f"train finished elapsed_min={train_elapsed_min} paths={stage_paths}", flush=True)

    holdout = val_like_dirs(DATA, 100)
    print(f"eval 12k holdout n={len(holdout)} seed=365", flush=True)
    hard_rows, soft_rows = _eval_holdout(holdout, PRED, "12k")
    hard = _mean_block(hard_rows)
    soft = _mean_block(soft_rows)
    summary = {
        "weights": str(OUT),
        "metric": "type_aware_set_f1",
        "holdout": "random12k seed=365 first 10% then 100",
        "n_samples": hard["n_samples"],
        "stage_paths": stage_paths,
        "train_elapsed_min": train_elapsed_min,
        "hard_type_aware": {k: v for k, v in hard.items() if k != "samples"},
        "soft_type_aware": {k: v for k, v in soft.items() if k != "samples"},
        "samples_hard": hard["samples"],
        "samples_soft": soft["samples"],
        "train_metrics": _collect_train_metrics(OUT),
        "elapsed_min": round((time.time() - t0) / 60, 2),
        "raw2k_holdout": None,
        "raw2k_skipped": None,
    }
    _write_summary(summary)
    print(
        f"DONE 12k hard_f1={hard['mean_melody_f1']:.3f} "
        f"p={hard['mean_melody_precision']:.3f} r={hard['mean_melody_recall']:.3f} "
        f"soft_f1={soft['mean_melody_f1']:.3f} "
        f"p={soft['mean_melody_precision']:.3f} r={soft['mean_melody_recall']:.3f} "
        f"n_pred={hard['mean_n_pred']:.2f} n_gold={hard['mean_n_gold']:.2f}",
        flush=True,
    )

    if not RAW_DATA.exists():
        summary["raw2k_skipped"] = f"missing {RAW_DATA}"
        _write_summary(summary)
        return
    try:
        raw_holdout = val_like_dirs(RAW_DATA, 100)
        print(f"eval raw2k holdout n={len(raw_holdout)} seed=365", flush=True)
        raw_hard_rows, raw_soft_rows = _eval_holdout(raw_holdout, PRED_RAW, "raw2k")
        raw_hard = _mean_block(raw_hard_rows)
        raw_soft = _mean_block(raw_soft_rows)
        summary["raw2k_holdout"] = {
            "n_samples": raw_hard["n_samples"],
            "hard_type_aware": {k: v for k, v in raw_hard.items() if k != "samples"},
            "soft_type_aware": {k: v for k, v in raw_soft.items() if k != "samples"},
            "samples_hard": raw_hard["samples"],
            "samples_soft": raw_soft["samples"],
        }
        summary["elapsed_min"] = round((time.time() - t0) / 60, 2)
        _write_summary(summary)
        print(
            f"DONE raw2k hard_f1={raw_hard['mean_melody_f1']:.3f} "
            f"soft_f1={raw_soft['mean_melody_f1']:.3f} "
            f"n_pred={raw_hard['mean_n_pred']:.2f} n_gold={raw_hard['mean_n_gold']:.2f}",
            flush=True,
        )
    except Exception as exc:
        summary["raw2k_skipped"] = str(exc)
        summary["elapsed_min"] = round((time.time() - t0) / 60, 2)
        _write_summary(summary)
        print(f"SKIP raw2k eval: {exc}", flush=True)


if __name__ == "__main__":
    main()
