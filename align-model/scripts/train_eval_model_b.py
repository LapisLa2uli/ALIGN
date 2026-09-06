"""Train Model B (melody-first) on random 12k then raw 2k, then holdout F1."""

from __future__ import annotations

import json
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

from alignmodel.dataset import list_sample_dirs
from alignmodel.device import resolve_device
from alignmodel.eval_melodies import eval_sample
from alignmodel.melody_infer import infer_melody_sample, load_melody_model, write_melody_prediction

ROOT = Path(__file__).resolve().parents[2]
RANDOM_ROOT = ROOT / "synth-pipeline" / "output"
RAW_ROOT = ROOT / "synth-pipeline" / "output_2k_rawdata"
RUNS = ROOT / "align-model" / "runs"
PRED_DIR = RUNS / "eval-dual"
LOG_PATH = PRED_DIR / "train_model_b.log"
SUMMARY_PATH = PRED_DIR / "summary_model_b.json"

JOBS = [
    {
        "id": "B-random12k",
        "data": RANDOM_ROOT,
        "out": RUNS / "melody-random12k",
    },
    {
        "id": "B-raw2k",
        "data": RAW_ROOT,
        "out": RUNS / "melody-raw2k",
    },
]

EVAL_SETS = [
    {"id": "random12k-holdout", "root": RANDOM_ROOT},
    {"id": "raw2k-holdout", "root": RAW_ROOT},
]


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def history_complete(out: Path, epochs: int = 8) -> bool:
    hist = out / "history.json"
    best = out / "best.pt"
    if not best.exists() or not hist.exists():
        return False
    try:
        rows = json.loads(hist.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if not isinstance(rows, list) or not rows:
        return False
    if len(rows) >= epochs:
        return True
    last = rows[-1]
    if isinstance(last, dict) and last.get("early_complete"):
        return True
    return False


def train_job(job: dict, epochs: int, device: str) -> dict:
    out = Path(job["out"])
    data = Path(job["data"])
    info = {"id": job["id"], "out": str(out), "data": str(data), "skipped": False, "oom": False}
    if history_complete(out, epochs):
        log(f"SKIP train {job['id']} (completed weights already in {out})")
        info["skipped"] = True
        info["elapsed_min"] = 0.0
        return info
    log(f"=== START train {job['id']} data={data} out={out} ===")
    t0 = time.time()
    py = sys.executable
    batches = [2, 1]
    last_err = None
    for batch in batches:
        cmd = [
            py,
            "-u",
            "-m",
            "alignmodel.cli",
            "train-melody",
            "--data",
            str(data),
            "--out",
            str(out),
            "--epochs",
            str(epochs),
            "--device",
            device,
            "--batch-size",
            str(batch),
        ]
        log(f"{job['id']} CLI batch={batch}: {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=str(ROOT))
        if proc.returncode == 0:
            info["batch_size"] = batch
            info["elapsed_min"] = round((time.time() - t0) / 60, 2)
            log(f"=== DONE train {job['id']} batch={batch} elapsed_min={info['elapsed_min']} ===")
            return info
        last_err = proc.returncode
        if batch == 2:
            log(f"{job['id']} CLI failed rc={proc.returncode} at batch 2; empty_cache and retry batch=1")
            info["oom"] = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        break
    info["elapsed_min"] = round((time.time() - t0) / 60, 2)
    info["failed"] = True
    info["returncode"] = last_err
    log(f"FAILED train {job['id']} rc={last_err} elapsed_min={info['elapsed_min']}")
    raise RuntimeError(f"train {job['id']} failed returncode={last_err}")


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


def last_train_metrics(out: Path) -> dict:
    hist = out / "history.json"
    if not hist.exists():
        return {}
    rows = json.loads(hist.read_text(encoding="utf-8"))
    return rows[-1] if rows else {}


def evaluate_b(n_eval: int, device: str) -> dict:
    pred_dir = PRED_DIR
    pred_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"n_eval": n_eval, "device": device, "cells": []}
    eval_dirs = {ev["id"]: val_like_dirs(ev["root"], n_eval) for ev in EVAL_SETS}
    for ev in EVAL_SETS:
        log(f"eval set {ev['id']}: {len(eval_dirs[ev['id']])} clips")

    for job in JOBS:
        out = Path(job["out"])
        if not (out / "best.pt").exists():
            log(f"SKIP eval {job['id']} (missing best.pt)")
            continue
        melody_model = load_melody_model(out / "best.pt", resolve_device(device))
        for ev in EVAL_SETS:
            cell_id = f"{job['id']}_on_{ev['id']}"
            cell_dir = pred_dir / cell_id
            cell_dir.mkdir(parents=True, exist_ok=True)
            log(f"=== START eval {cell_id} ===")
            t0 = time.time()
            rows = []
            samples = eval_dirs[ev["id"]]
            for i, sample in enumerate(samples, start=1):
                pred_path = cell_dir / f"{sample.name}.json"
                if pred_path.exists():
                    labels = json.loads(pred_path.read_text(encoding="utf-8")).get("labels") or []
                else:
                    result = infer_melody_sample(melody_model, sample, resolve_device(device))
                    write_melody_prediction(result, pred_path)
                    labels = result["labels"]
                row = eval_sample(sample, pred_labels=labels)
                rows.append(row)
                if i == 1 or i % 20 == 0:
                    log(f"  {cell_id} {i}/{len(samples)} f1={row['melody_f1']:.3f}")
            n = max(len(rows), 1)
            cell = {
                "id": cell_id,
                "model": "B",
                "train": job["id"],
                "eval": ev["id"],
                "weights": str(out),
                "best_pt": str(out / "best.pt"),
                "last_pt": str(out / "last.pt"),
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
        del melody_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return report


def main() -> None:
    epochs = 8
    n_eval = 100
    device = "cuda"
    t_all = time.time()
    summary = {
        "model": "B",
        "random_root": str(RANDOM_ROOT),
        "raw_root": str(RAW_ROOT),
        "n_random": len(list_sample_dirs(RANDOM_ROOT)),
        "n_raw": len(list_sample_dirs(RAW_ROOT)),
        "epochs": epochs,
        "n_eval": n_eval,
        "jobs": [
            {"id": j["id"], "data": str(j["data"]), "out": str(j["out"])} for j in JOBS
        ],
        "train_runs": [],
        "failures": [],
    }
    log(
        f"model-B train/eval random={summary['n_random']} raw={summary['n_raw']} "
        f"epochs={epochs} n_eval={n_eval}"
    )
    for job in JOBS:
        try:
            summary["train_runs"].append(train_job(job, epochs=epochs, device=device))
        except Exception as exc:
            summary["failures"].append({"id": job["id"], "error": str(exc)})
            traceback.print_exc()
            log(f"FAILED train {job['id']}: {exc}")

    summary["train_metrics"] = {job["id"]: last_train_metrics(Path(job["out"])) for job in JOBS}
    summary["checkpoints"] = {
        job["id"]: {
            "best.pt": str(Path(job["out"]) / "best.pt"),
            "last.pt": str(Path(job["out"]) / "last.pt"),
            "best_exists": (Path(job["out"]) / "best.pt").exists(),
            "last_exists": (Path(job["out"]) / "last.pt").exists(),
        }
        for job in JOBS
    }

    ready = all((Path(job["out"]) / "best.pt").exists() for job in JOBS)
    if ready:
        try:
            summary["eval"] = evaluate_b(n_eval, device)
        except Exception as exc:
            summary["failures"].append({"id": "eval", "error": str(exc)})
            traceback.print_exc()
            log(f"FAILED eval: {exc}")
    else:
        log("SKIP eval (not both B checkpoints present)")

    summary["elapsed_min"] = round((time.time() - t_all) / 60, 2)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Wrote {SUMMARY_PATH} elapsed_min={summary['elapsed_min']}")


if __name__ == "__main__":
    main()
