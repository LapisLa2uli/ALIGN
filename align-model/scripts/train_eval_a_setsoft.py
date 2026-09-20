"""Retrain Model A on random 12k; score with soft exclusive set-F1."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from alignmodel.dataset import list_sample_dirs
from alignmodel.eval_melodies import eval_sample
from alignmodel.pipeline import run_pipeline, write_prediction
from alignmodel.stage_train import StageTrainConfig, train_stages
from alignmodel.types import pipeline_label_to_dict

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("E:/output")
OUT = ROOT / "align-model" / "runs" / "stages-random12k-setsoft"
PRED = ROOT / "align-model" / "runs" / "eval-dual" / "A-random12k-setsoft_on_random12k-holdout"
SUMMARY = ROOT / "align-model" / "runs" / "eval-dual" / "summary_model_a_setsoft.json"


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


def main() -> None:
    t0 = time.time()
    print(f"train A setsoft data={DATA} out={OUT}", flush=True)
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
    holdout = val_like_dirs(DATA, 100)
    PRED.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, sample in enumerate(holdout, start=1):
        pred_path = PRED / f"{sample.name}.json"
        if pred_path.exists():
            labels = json.loads(pred_path.read_text(encoding="utf-8")).get("labels") or []
        else:
            state = run_pipeline(sample, device="cuda", weights_dir=OUT)
            write_prediction(state, pred_path)
            labels = [pipeline_label_to_dict(lab) for lab in state.labels]
        row = eval_sample(sample, pred_labels=labels, soft=False)
        rows.append(row)
        if i == 1 or i % 20 == 0:
            print(f"eval {i}/{len(holdout)} note_wise_f1={row['melody_f1']:.3f}", flush=True)
    n = max(len(rows), 1)
    summary = {
        "weights": str(OUT),
        "metric": "official_note_wise",
        "legacy_soft_set_f1": "diagnostic_only",
        "n_samples": len(rows),
        "mean_melody_f1": round(sum(r["melody_f1"] for r in rows) / n, 4),
        "mean_melody_precision": round(sum(r["melody_precision"] for r in rows) / n, 4),
        "mean_melody_recall": round(sum(r["melody_recall"] for r in rows) / n, 4),
        "mean_n_pred": round(sum(r["n_pred"] for r in rows) / n, 3),
        "mean_n_gold": round(sum(r["n_gold"] for r in rows) / n, 3),
        "elapsed_min": round((time.time() - t0) / 60, 2),
        "samples": rows,
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"DONE note_wise_f1={summary['mean_melody_f1']:.3f} "
        f"p={summary['mean_melody_precision']:.3f} "
        f"r={summary['mean_melody_recall']:.3f} "
        f"n_pred={summary['mean_n_pred']:.2f} "
        f"wrote {SUMMARY}",
        flush=True,
    )


if __name__ == "__main__":
    main()
