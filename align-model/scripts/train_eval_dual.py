"""Train Model A and Model B on random 12k then raw 2k, then score containment F1."""

from __future__ import annotations

import argparse
import json
import random
import time
import traceback
from pathlib import Path

import torch

from alignmodel.dataset import list_sample_dirs
from alignmodel.device import resolve_device
from alignmodel.eval_melodies import eval_sample
from alignmodel.melody_infer import infer_melody_sample, load_melody_model, write_melody_prediction
from alignmodel.melody_train import MelodyTrainConfig, train_melody
from alignmodel.pipeline import run_pipeline, write_prediction
from alignmodel.stage_train import StageTrainConfig, train_stages
from alignmodel.types import pipeline_label_to_dict

ROOT = Path(__file__).resolve().parents[2]
RANDOM_ROOT = ROOT / "synth-pipeline" / "output"
RAW_ROOT = ROOT / "synth-pipeline" / "output_2k_rawdata"
RUNS = ROOT / "align-model" / "runs"

JOBS = [
    {
        "id": "A-random12k",
        "model": "A",
        "data": RANDOM_ROOT,
        "out": RUNS / "stages-random12k",
    },
    {
        "id": "B-random12k",
        "model": "B",
        "data": RANDOM_ROOT,
        "out": RUNS / "melody-random12k",
    },
    {
        "id": "A-raw2k",
        "model": "A",
        "data": RAW_ROOT,
        "out": RUNS / "stages-raw2k",
    },
    {
        "id": "B-raw2k",
        "model": "B",
        "data": RAW_ROOT,
        "out": RUNS / "melody-raw2k",
    },
]

EVAL_SETS = [
    {"id": "random12k-holdout", "root": RANDOM_ROOT},
    {"id": "raw2k-holdout", "root": RAW_ROOT},
]

LOG_PATH = RUNS / "eval-dual" / "train_eval.log"


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def model_a_ready(out: Path) -> bool:
    return all((out / name).exists() for name in ("stage1.pt", "stage2.pt", "stage3.pt"))


def model_b_ready(out: Path) -> bool:
    return (out / "best.pt").exists()


def train_job(job: dict, epochs: int, device: str) -> None:
    out = Path(job["out"])
    data = Path(job["data"])
    if job["model"] == "A" and model_a_ready(out):
        log(f"SKIP train {job['id']} (weights already in {out})")
        return
    if job["model"] == "B" and model_b_ready(out):
        log(f"SKIP train {job['id']} (weights already in {out})")
        return
    log(f"=== START train {job['id']} data={data} out={out} ===")
    t0 = time.time()
    if job["model"] == "A":
        train_stages(
            StageTrainConfig(
                data_root=data,
                output_dir=out,
                epochs=epochs,
                batch_size=32,
                lr=1e-3,
                device=device,
                stages=(1, 2, 3),
            )
        )
    else:
        batch = 2
        try:
            train_melody(
                MelodyTrainConfig(
                    data_root=data,
                    output_dir=out,
                    epochs=epochs,
                    batch_size=batch,
                    lr=2e-4,
                    device=device,
                )
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            log(f"{job['id']} OOM at batch {batch}; retrying batch=1")
            torch.cuda.empty_cache()
            train_melody(
                MelodyTrainConfig(
                    data_root=data,
                    output_dir=out,
                    epochs=epochs,
                    batch_size=1,
                    lr=2e-4,
                    device=device,
                )
            )
    log(f"=== DONE train {job['id']} elapsed_min={(time.time() - t0) / 60:.1f} ===")


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


def evaluate_all(n_eval: int, device: str, pred_dir: Path) -> dict:
    pred_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"n_eval": n_eval, "device": device, "cells": []}
    eval_dirs = {ev["id"]: val_like_dirs(ev["root"], n_eval) for ev in EVAL_SETS}
    for ev in EVAL_SETS:
        log(f"eval set {ev['id']}: {len(eval_dirs[ev['id']])} clips")

    for job in JOBS:
        out = Path(job["out"])
        ready = model_a_ready(out) if job["model"] == "A" else model_b_ready(out)
        if not ready:
            log(f"SKIP eval {job['id']} (missing weights)")
            continue
        melody_model = None
        if job["model"] == "B":
            melody_model = load_melody_model(out / "best.pt", resolve_device(device))
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
                elif job["model"] == "A":
                    state = run_pipeline(sample, device=device, weights_dir=out)
                    write_prediction(state, pred_path)
                    labels = [pipeline_label_to_dict(lab) for lab in state.labels]
                else:
                    result = infer_melody_sample(melody_model, sample, resolve_device(device))
                    write_melody_prediction(result, pred_path)
                    labels = result["labels"]
                row = eval_sample(sample, pred_labels=labels)
                rows.append(row)
                if i == 1 or i % 20 == 0:
                    log(f"  {cell_id} {i}/{len(eval_dirs[ev['id']])} f1={row['melody_f1']:.3f}")
            n = max(len(rows), 1)
            cell = {
                "id": cell_id,
                "model": job["model"],
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
        if melody_model is not None:
            del melody_model
            torch.cuda.empty_cache()
    return report


def collect_train_metrics() -> dict:
    metrics = {}
    for job in JOBS:
        out = Path(job["out"])
        if job["model"] == "A":
            blob = {}
            for stage in ("stage1", "stage2", "stage3"):
                hist = out / f"{stage}_history.json"
                if hist.exists():
                    rows = json.loads(hist.read_text(encoding="utf-8"))
                    blob[stage] = rows[-1] if rows else {}
            metrics[job["id"]] = blob
        else:
            hist = out / "history.json"
            if hist.exists():
                rows = json.loads(hist.read_text(encoding="utf-8"))
                metrics[job["id"]] = rows[-1] if rows else {}
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    summary = {
        "random_root": str(RANDOM_ROOT),
        "raw_root": str(RAW_ROOT),
        "n_random": len(list_sample_dirs(RANDOM_ROOT)),
        "n_raw": len(list_sample_dirs(RAW_ROOT)),
        "epochs": args.epochs,
        "n_eval": args.n_eval,
        "jobs": [
            {"id": j["id"], "model": j["model"], "data": str(j["data"]), "out": str(j["out"])}
            for j in JOBS
        ],
    }
    log(
        f"dual-track train/eval random={summary['n_random']} raw={summary['n_raw']} "
        f"epochs={args.epochs} n_eval={args.n_eval}"
    )

    if not args.skip_train:
        for job in JOBS:
            try:
                train_job(job, epochs=args.epochs, device=args.device)
            except Exception:
                log(f"FAILED train {job['id']}")
                traceback.print_exc()
                raise

    summary["train_metrics"] = collect_train_metrics()
    if not args.skip_eval:
        pred_dir = RUNS / "eval-dual"
        summary["eval"] = evaluate_all(args.n_eval, args.device, pred_dir)

    out_path = RUNS / "eval-dual" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
