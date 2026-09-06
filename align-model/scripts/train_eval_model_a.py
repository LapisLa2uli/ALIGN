"""Train and evaluate ALIGN Model A only (two datasets, separate weight folders)."""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import torch

from alignmodel.device import resolve_device
from alignmodel.eval_melodies import eval_sample
from alignmodel.pipeline import run_pipeline, write_prediction
from alignmodel.stage_train import StageTrainConfig, train_stages
from alignmodel.types import pipeline_label_to_dict

from train_eval_dual import (
    EVAL_SETS,
    JOBS,
    RANDOM_ROOT,
    RAW_ROOT,
    RUNS,
    collect_train_metrics,
    model_a_ready,
    val_like_dirs,
)

A_JOBS = [j for j in JOBS if j["model"] == "A"]
LOG_PATH = RUNS / "eval-dual" / "train_eval_model_a.log"
SUMMARY_PATH = RUNS / "eval-dual" / "summary_model_a.json"


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def stage3_incomplete(out: Path) -> bool:
    if not (out / "stage1.pt").exists() or not (out / "stage2.pt").exists():
        return False
    hist = out / "stage3_history.json"
    if not hist.exists():
        return True
    try:
        rows = json.loads(hist.read_text(encoding="utf-8"))
    except Exception:
        return True
    return not rows or not (out / "stage3.pt").exists()


def train_stages_oom_retry(
    data: Path,
    out: Path,
    stages: tuple[int, ...],
    epochs: int,
    device: str,
    batch_size: int = 32,
) -> None:
    batches = [batch_size]
    for b in (16, 8, 4):
        if b < batch_size:
            batches.append(b)
    last_exc: Exception | None = None
    for batch in batches:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log(f"train-stages data={data} out={out} stages={stages} batch={batch}")
            train_stages(
                StageTrainConfig(
                    data_root=data,
                    output_dir=out,
                    epochs=epochs,
                    batch_size=batch,
                    lr=1e-3,
                    device=device,
                    stages=stages,
                )
            )
            return
        except RuntimeError as exc:
            last_exc = exc
            if "out of memory" not in str(exc).lower():
                raise
            log(f"OOM at batch={batch} stages={stages} out={out}; retrying smaller batch")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise last_exc if last_exc else RuntimeError("train-stages failed")


def train_model_a(epochs: int, device: str) -> list[dict]:
    notes: list[dict] = []
    random_out = RUNS / "stages-random12k"
    raw_out = RUNS / "stages-raw2k"

    if model_a_ready(random_out) and not stage3_incomplete(random_out):
        log(f"SKIP train A-random12k (complete weights in {random_out})")
        notes.append({"id": "A-random12k", "action": "skip_complete"})
    elif (random_out / "stage1.pt").exists() and (random_out / "stage2.pt").exists():
        log("=== START train A-random12k stage 3 only ===")
        t0 = time.time()
        train_stages_oom_retry(RANDOM_ROOT, random_out, (3,), epochs, device)
        notes.append(
            {
                "id": "A-random12k",
                "action": "stage3_only",
                "elapsed_min": round((time.time() - t0) / 60, 2),
            }
        )
        log(f"=== DONE train A-random12k stage3 elapsed_min={(time.time() - t0) / 60:.1f} ===")
    else:
        log("=== START train A-random12k stages 1,2,3 ===")
        t0 = time.time()
        train_stages_oom_retry(RANDOM_ROOT, random_out, (1, 2, 3), epochs, device)
        notes.append(
            {
                "id": "A-random12k",
                "action": "all_stages",
                "elapsed_min": round((time.time() - t0) / 60, 2),
            }
        )
        log(f"=== DONE train A-random12k elapsed_min={(time.time() - t0) / 60:.1f} ===")

    if model_a_ready(raw_out):
        log(f"SKIP train A-raw2k (complete weights in {raw_out})")
        notes.append({"id": "A-raw2k", "action": "skip_complete"})
    else:
        log("=== START train A-raw2k stages 1,2,3 ===")
        t0 = time.time()
        train_stages_oom_retry(RAW_ROOT, raw_out, (1, 2, 3), epochs, device)
        notes.append(
            {
                "id": "A-raw2k",
                "action": "all_stages",
                "elapsed_min": round((time.time() - t0) / 60, 2),
            }
        )
        log(f"=== DONE train A-raw2k elapsed_min={(time.time() - t0) / 60:.1f} ===")
    return notes


def evaluate_model_a(n_eval: int, device: str, pred_dir: Path) -> dict:
    pred_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"n_eval": n_eval, "device": device, "cells": []}
    eval_dirs = {ev["id"]: val_like_dirs(ev["root"], n_eval) for ev in EVAL_SETS}
    for ev in EVAL_SETS:
        log(f"eval set {ev['id']}: {len(eval_dirs[ev['id']])} clips")

    for job in A_JOBS:
        out = Path(job["out"])
        if not model_a_ready(out):
            log(f"SKIP eval {job['id']} (missing weights)")
            continue
        for ev in EVAL_SETS:
            cell_id = f"{job['id']}_on_{ev['id']}"
            cell_dir = pred_dir / cell_id
            cell_dir.mkdir(parents=True, exist_ok=True)
            log(f"=== START eval {cell_id} ===")
            t0 = time.time()
            rows = []
            for i, sample in enumerate(eval_dirs[ev["id"]], start=1):
                pred_path = cell_dir / f"{sample.name}.json"
                if pred_path.exists():
                    labels = json.loads(pred_path.read_text(encoding="utf-8")).get("labels") or []
                else:
                    try:
                        state = run_pipeline(sample, device=device, weights_dir=out)
                    except RuntimeError as exc:
                        if "out of memory" not in str(exc).lower():
                            raise
                        log(f"OOM infer {cell_id} {sample.name}; empty_cache retry")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        state = run_pipeline(sample, device=device, weights_dir=out)
                    write_prediction(state, pred_path)
                    labels = [pipeline_label_to_dict(lab) for lab in state.labels]
                row = eval_sample(sample, pred_labels=labels)
                rows.append(row)
                if i == 1 or i % 20 == 0:
                    log(f"  {cell_id} {i}/{len(eval_dirs[ev['id']])} f1={row['melody_f1']:.3f}")
            n = max(len(rows), 1)
            cell = {
                "id": cell_id,
                "model": "A",
                "train": job["id"],
                "eval": ev["id"],
                "weights": str(out),
                "n_samples": len(rows),
                "mean_melody_f1": round(sum(r["melody_f1"] for r in rows) / n, 4),
                "mean_melody_precision": round(sum(r["melody_precision"] for r in rows) / n, 4),
                "mean_melody_recall": round(sum(r["melody_recall"] for r in rows) / n, 4),
                "mean_n_gold": round(sum(r["n_gold"] for r in rows) / n, 3),
                "mean_n_pred": round(sum(r["n_pred"] for r in rows) / n, 3),
                "elapsed_min": round((time.time() - t0) / 60, 2),
                "samples": rows,
            }
            report["cells"].append(cell)
            log(
                f"=== DONE eval {cell_id} f1={cell['mean_melody_f1']:.3f} "
                f"p={cell['mean_melody_precision']:.3f} r={cell['mean_melody_recall']:.3f} "
                f"min={cell['elapsed_min']} ==="
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return report


def collect_histories() -> dict:
    histories = {}
    last = {}
    for job in A_JOBS:
        out = Path(job["out"])
        blob_hist = {}
        blob_last = {}
        for stage in ("stage1", "stage2", "stage3"):
            hist = out / f"{stage}_history.json"
            if hist.exists():
                rows = json.loads(hist.read_text(encoding="utf-8"))
                blob_hist[stage] = rows
                blob_last[stage] = rows[-1] if rows else {}
        histories[job["id"]] = blob_hist
        last[job["id"]] = blob_last
    return histories, last


def weight_inventory() -> dict:
    inv = {}
    for job in A_JOBS:
        out = Path(job["out"])
        inv[job["id"]] = {
            "dir": str(out),
            "stage1.pt": (out / "stage1.pt").exists(),
            "stage2.pt": (out / "stage2.pt").exists(),
            "stage3.pt": (out / "stage3.pt").exists(),
            "stage1_history.json": (out / "stage1_history.json").exists(),
            "stage2_history.json": (out / "stage2_history.json").exists(),
            "stage3_history.json": (out / "stage3_history.json").exists(),
        }
    return inv


def main() -> None:
    epochs = 8
    n_eval = 100
    device = "cuda"
    t0 = time.time()
    summary = {
        "model": "A",
        "random_root": str(RANDOM_ROOT),
        "raw_root": str(RAW_ROOT),
        "epochs": epochs,
        "batch_size": 32,
        "n_eval": n_eval,
        "device": device,
        "jobs": [
            {"id": j["id"], "model": j["model"], "data": str(j["data"]), "out": str(j["out"])}
            for j in A_JOBS
        ],
        "notes": [],
        "failures": [],
        "oom": [],
    }
    log(f"model-A train/eval epochs={epochs} n_eval={n_eval} device={device}")
    try:
        summary["train_actions"] = train_model_a(epochs=epochs, device=device)
    except Exception as exc:
        summary["failures"].append({"phase": "train", "error": repr(exc)})
        log(f"FAILED train: {exc}")
        traceback.print_exc()
        raise
    histories, last = collect_histories()
    summary["train_histories"] = histories
    summary["train_metrics"] = last
    summary["train_metrics_dual_format"] = {
        k: v for k, v in collect_train_metrics().items() if k.startswith("A-")
    }
    try:
        pred_dir = RUNS / "eval-dual"
        summary["eval"] = evaluate_model_a(n_eval, device, pred_dir)
    except Exception as exc:
        summary["failures"].append({"phase": "eval", "error": repr(exc)})
        log(f"FAILED eval: {exc}")
        traceback.print_exc()
        raise
    finally:
        summary["weights"] = weight_inventory()
        summary["elapsed_min"] = round((time.time() - t0) / 60, 2)
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
