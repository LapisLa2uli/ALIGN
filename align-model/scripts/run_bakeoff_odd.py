"""Train or eval odd-team Model B bakeoff versions (V1/V3/V5/V7)."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
RANDOM_ROOT = ROOT / "synth-pipeline" / "output"
RUNS = ROOT / "align-model" / "runs"
BAKEOFF = RUNS / "melody-bakeoff"
PRED_ROOT = RUNS / "eval-dual"
FORBIDDEN = {
    (RUNS / "melody-random12k-set").resolve(),
    (RUNS / "melody-raw2k-set").resolve(),
}

VERSIONS = {
    "v1": {"id": "melody-b-v1-es", "variant": "v1"},
    "v3": {"id": "melody-b-v3-bio", "variant": "v3"},
    "v5": {"id": "melody-b-v5-dice", "variant": "v5"},
    "v7": {"id": "melody-b-v7-combo", "variant": "v7"},
}


def val_like_dirs(root: Path, n: int, seed: int = 365) -> list[Path]:
    from alignmodel.dataset import list_sample_dirs

    dirs = [p for p in list_sample_dirs(root) if (p / "verified_score.musicxml").exists()]
    dirs = sorted(dirs, key=lambda p: p.name)
    rng = random.Random(seed)
    shuffled = list(dirs)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * 0.1))
    if len(shuffled) > 1:
        n_val = min(n_val, len(shuffled) - 1)
    return shuffled[:n_val][:n]


def train_one(key: str, epochs: int = 8, device: str = "cuda") -> dict:
    from alignmodel.melody_train import MelodyTrainConfig, train_melody

    spec = VERSIONS[key]
    out = (BAKEOFF / spec["id"]).resolve()
    if out in FORBIDDEN:
        raise RuntimeError(f"refusing to overwrite {out}")
    info = {"id": spec["id"], "variant": spec["variant"], "out": str(out), "oom": False}
    last_err = None
    for batch in (2, 1):
        print(f"=== TRAIN {spec['id']} variant={spec['variant']} batch={batch} ===", flush=True)
        t0 = time.time()
        try:
            train_melody(
                MelodyTrainConfig(
                    data_root=RANDOM_ROOT,
                    output_dir=out,
                    epochs=epochs,
                    batch_size=batch,
                    lr=2e-4,
                    device=device,
                    seed=365,
                    variant=spec["variant"],
                )
            )
            info["batch_size"] = batch
            info["elapsed_min"] = round((time.time() - t0) / 60, 2)
            print(f"=== DONE TRAIN {spec['id']} batch={batch} min={info['elapsed_min']} ===", flush=True)
            return info
        except RuntimeError as exc:
            last_err = exc
            if "out of memory" not in str(exc).lower() or batch == 1:
                raise
            info["oom"] = True
            print(f"{spec['id']} OOM at batch 2; retrying batch=1", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise last_err  # pragma: no cover


def _history_blob(out: Path) -> dict:
    hist = out / "history.json"
    if not hist.exists():
        return {}
    blob = json.loads(hist.read_text(encoding="utf-8"))
    if isinstance(blob, list):
        last = blob[-1] if blob else {}
        return {
            "stopped_reason": last.get("stopped_reason"),
            "epoch": last.get("epoch"),
            "step": last.get("step"),
            "history": blob,
            "last": last,
        }
    rows = blob.get("history") or []
    return {
        "stopped_reason": blob.get("stopped_reason"),
        "epoch": blob.get("epoch"),
        "step": blob.get("step"),
        "history": rows,
        "last": rows[-1] if rows else {},
    }


def eval_one(key: str, n_eval: int = 100, device: str = "cuda") -> dict:
    from alignmodel.device import resolve_device
    from alignmodel.eval_melodies import eval_sample
    from alignmodel.melody_infer import infer_melody_sample, load_melody_model, write_melody_prediction

    spec = VERSIONS[key]
    out = BAKEOFF / spec["id"]
    best = out / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"missing {best}")
    cell_id = f"{spec['id']}_on_random12k-holdout"
    cell_dir = PRED_ROOT / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    samples = val_like_dirs(RANDOM_ROOT, n_eval, seed=365)
    print(f"=== EVAL {cell_id} n={len(samples)} ===", flush=True)
    t0 = time.time()
    model = load_melody_model(best, resolve_device(device))
    rows = []
    for i, sample in enumerate(samples, start=1):
        pred_path = cell_dir / f"{sample.name}.json"
        if pred_path.exists():
            labels = json.loads(pred_path.read_text(encoding="utf-8")).get("labels") or []
        else:
            result = infer_melody_sample(model, sample, resolve_device(device))
            write_melody_prediction(result, pred_path)
            labels = result["labels"]
        row = eval_sample(sample, pred_labels=labels)
        rows.append(row)
        if i == 1 or i % 20 == 0:
            print(
                f"  {cell_id} {i}/{len(samples)} f1={row['melody_f1']:.3f} "
                f"n_pred={row['n_pred']} n_gold={row['n_gold']}",
                flush=True,
            )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    n = max(len(rows), 1)
    hist = _history_blob(out)
    last = hist.get("last") or {}
    metrics = {
        "id": spec["id"],
        "variant": spec["variant"],
        "set_f1": round(sum(r["melody_f1"] for r in rows) / n, 4),
        "precision": round(sum(r["melody_precision"] for r in rows) / n, 4),
        "recall": round(sum(r["melody_recall"] for r in rows) / n, 4),
        "P": round(sum(r["melody_precision"] for r in rows) / n, 4),
        "R": round(sum(r["melody_recall"] for r in rows) / n, 4),
        "mean_n_pred": round(sum(r["n_pred"] for r in rows) / n, 3),
        "mean_n_gold": round(sum(r["n_gold"] for r in rows) / n, 3),
        "n_samples": len(rows),
        "val": {
            "set_f1": last.get("val_set_f1"),
            "set_precision": last.get("val_set_precision"),
            "set_recall": last.get("val_set_recall"),
            "mean_n_pred": last.get("val_mean_n_pred"),
            "mean_n_gold": last.get("val_mean_n_gold"),
            "loss": last.get("val_loss"),
            "note_acc": last.get("val_note_acc"),
        },
        "epochs": hist.get("epoch") or last.get("epoch"),
        "steps": hist.get("step") or last.get("step"),
        "stopped_reason": hist.get("stopped_reason") or last.get("stopped_reason"),
        "weight_path": str(best.resolve()),
        "pred_dir": str(cell_dir),
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"=== DONE EVAL {cell_id} set_f1={metrics['set_f1']:.3f} "
        f"P={metrics['P']:.3f} R={metrics['R']:.3f} "
        f"n_pred={metrics['mean_n_pred']:.2f} stop={metrics['stopped_reason']} ===",
        flush=True,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train", "eval"))
    parser.add_argument("version", choices=sorted(VERSIONS))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.action == "train":
        info = train_one(args.version, epochs=args.epochs, device=args.device)
        print(json.dumps(info, indent=2), flush=True)
    else:
        metrics = eval_one(args.version, n_eval=args.n_eval, device=args.device)
        print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    sys.exit(main())
