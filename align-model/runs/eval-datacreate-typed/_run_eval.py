"""Evaluate Model A on labeled DataCreate bundles. Do not train; write a new eval dir."""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from alignmodel.eval_melodies import eval_sample
from alignmodel.pipeline import run_pipeline, write_prediction
from alignmodel.types import pipeline_label_to_dict

ROOT = Path(r"D:\stuff\Audio Evaluation\ALIGN")
DATA = ROOT / "DataCreate" / "samples"
WEIGHTS = ROOT / "align-model" / "runs" / "stages-random12k-setsoft"
OUT = ROOT / "align-model" / "runs" / "eval-datacreate-typed"
DEVICE = "cuda"


def labeled_sample_dirs(root: Path) -> tuple[list[Path], list[dict]]:
    included: list[Path] = []
    skipped: list[dict] = []
    for path in sorted(p for p in root.iterdir() if p.is_dir()):
        labels_path = path / "labels.json"
        score_path = path / "verified_score.musicxml"
        if not labels_path.exists():
            skipped.append({"sample": path.name, "reason": "no labels.json"})
            continue
        if not score_path.exists():
            skipped.append({"sample": path.name, "reason": "no verified_score.musicxml"})
            continue
        try:
            doc = json.loads(labels_path.read_text(encoding="utf-8"))
        except Exception as exc:
            skipped.append({"sample": path.name, "reason": f"labels.json unreadable: {exc}"})
            continue
        labels = doc.get("labels") or []
        if not labels:
            skipped.append({"sample": path.name, "reason": "empty labels"})
            continue
        included.append(path)
    return included, skipped


def means(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {
            "n_samples": 0,
            "mean_melody_f1": 0.0,
            "mean_melody_precision": 0.0,
            "mean_melody_recall": 0.0,
            "mean_n_gold": 0.0,
            "mean_n_pred": 0.0,
            "mean_similarity_sum": 0.0,
        }
    return {
        "n_samples": n,
        "mean_melody_f1": round(sum(r["melody_f1"] for r in rows) / n, 4),
        "mean_melody_precision": round(sum(r["melody_precision"] for r in rows) / n, 4),
        "mean_melody_recall": round(sum(r["melody_recall"] for r in rows) / n, 4),
        "mean_n_gold": round(sum(r["n_gold"] for r in rows) / n, 3),
        "mean_n_pred": round(sum(r["n_pred"] for r in rows) / n, 3),
        "mean_similarity_sum": round(sum(r["similarity_sum"] for r in rows) / n, 4),
    }


def score_all(sample: Path, pred_labels: list[dict]) -> dict:
    criteria = {
        "hard_type_sensitive": {"soft": False, "ignore_type": False},
        "hard_type_insensitive": {"soft": False, "ignore_type": True},
        "soft_type_sensitive": {"soft": True, "ignore_type": False},
        "soft_type_insensitive": {"soft": True, "ignore_type": True},
    }
    out = {}
    for name, kwargs in criteria.items():
        row = eval_sample(sample, pred_labels=pred_labels, **kwargs)
        out[name] = row
    return out


def print_table(summary: dict) -> None:
    print("\n=== Model A DataCreate melody set-F1 ===")
    print(f"checkpoint: {summary['checkpoint']}")
    print(f"data_root:  {summary['data_root']}")
    print(f"n_included: {summary['n_samples']}  n_skipped: {summary['n_skipped']}")
    print(f"infer:      {summary['infer_device']}  (training left running: {summary['training_left_running']})")
    print()
    print(f"{'criterion':<26} {'n':>3} {'F1':>7} {'P':>7} {'R':>7} {'n_pred':>8} {'n_gold':>8}")
    for key in (
        "hard_type_sensitive",
        "hard_type_insensitive",
        "soft_type_sensitive",
        "soft_type_insensitive",
    ):
        m = summary["criteria"][key]
        print(
            f"{key:<26} {m['n_samples']:>3} {m['mean_melody_f1']:>7.4f} "
            f"{m['mean_melody_precision']:>7.4f} {m['mean_melody_recall']:>7.4f} "
            f"{m['mean_n_pred']:>8.3f} {m['mean_n_gold']:>8.3f}"
        )
    if summary["failures"]:
        print("\nfailures:")
        for row in summary["failures"]:
            print(f"  {row['sample']}: {row['error']}")


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    preds_dir = OUT / "preds"
    preds_dir.mkdir(exist_ok=True)
    included, skipped = labeled_sample_dirs(DATA)
    print(
        f"data={DATA} included={len(included)} skipped={len(skipped)} "
        f"weights={WEIGHTS} device={DEVICE}",
        flush=True,
    )
    per_sample = []
    failures = []
    infer_devices = []
    for i, sample in enumerate(included, start=1):
        pred_path = preds_dir / f"{sample.name}.json"
        t_sample = time.time()
        try:
            if pred_path.exists():
                doc = json.loads(pred_path.read_text(encoding="utf-8"))
                labels = doc.get("labels") or []
                used_device = str(doc.get("device") or "cached")
            else:
                state = run_pipeline(
                    sample,
                    stages={1, 2, 3},
                    device=DEVICE,
                    weights_dir=WEIGHTS,
                )
                write_prediction(state, pred_path)
                labels = [pipeline_label_to_dict(lab) for lab in state.labels]
                used_device = state.device
            scored = score_all(sample, labels)
            row = {
                "sample": sample.name,
                "sample_dir": str(sample),
                "pred_path": str(pred_path),
                "device": used_device,
                "n_gold": scored["hard_type_sensitive"]["n_gold"],
                "n_pred": scored["hard_type_sensitive"]["n_pred"],
                "gold_types": scored["hard_type_sensitive"]["gold_types"],
                "pred_types": scored["hard_type_sensitive"]["pred_types"],
                "hard_type_sensitive": {
                    k: scored["hard_type_sensitive"][k]
                    for k in (
                        "melody_f1",
                        "melody_precision",
                        "melody_recall",
                        "similarity_sum",
                        "n_matched",
                    )
                },
                "hard_type_insensitive": {
                    k: scored["hard_type_insensitive"][k]
                    for k in (
                        "melody_f1",
                        "melody_precision",
                        "melody_recall",
                        "similarity_sum",
                        "n_matched",
                    )
                },
                "soft_type_sensitive": {
                    k: scored["soft_type_sensitive"][k]
                    for k in (
                        "melody_f1",
                        "melody_precision",
                        "melody_recall",
                        "similarity_sum",
                        "n_matched",
                    )
                },
                "soft_type_insensitive": {
                    k: scored["soft_type_insensitive"][k]
                    for k in (
                        "melody_f1",
                        "melody_precision",
                        "melody_recall",
                        "similarity_sum",
                        "n_matched",
                    )
                },
            }
            per_sample.append(row)
            infer_devices.append(used_device)
            hs = row["hard_type_sensitive"]
            hi = row["hard_type_insensitive"]
            print(
                f"[{i}/{len(included)}] {sample.name} "
                f"sens_f1={hs['melody_f1']:.3f} ins_f1={hi['melody_f1']:.3f} "
                f"n_gold={row['n_gold']} n_pred={row['n_pred']} "
                f"device={used_device} {time.time() - t_sample:.1f}s",
                flush=True,
            )
        except Exception as exc:
            failures.append(
                {
                    "sample": sample.name,
                    "sample_dir": str(sample),
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
            )
            print(f"[{i}/{len(included)}] {sample.name} FAIL {exc}", flush=True)

    def rows_for(key: str) -> list[dict]:
        out = []
        for row in per_sample:
            item = dict(row[key])
            item["n_gold"] = row["n_gold"]
            item["n_pred"] = row["n_pred"]
            item["melody_f1"] = item["melody_f1"]
            item["melody_precision"] = item["melody_precision"]
            item["melody_recall"] = item["melody_recall"]
            item["similarity_sum"] = item["similarity_sum"]
            out.append(item)
        return out

    unique_devices = sorted(set(infer_devices))
    summary = {
        "checkpoint": str(WEIGHTS),
        "checkpoint_files": {
            "stage1": str(WEIGHTS / "stage1.pt"),
            "stage2": str(WEIGHTS / "stage2.pt"),
            "stage3": str(WEIGHTS / "stage3.pt"),
        },
        "data_root": str(DATA),
        "out_dir": str(OUT),
        "n_samples": len(per_sample),
        "n_skipped": len(skipped),
        "n_failed": len(failures),
        "sample_dirs": [str(p) for p in included],
        "skipped": skipped,
        "failures": failures,
        "infer_device": unique_devices[0] if len(unique_devices) == 1 else unique_devices,
        "gpu_training_observed_before_infer": False,
        "training_left_running": True,
        "elapsed_min": round((time.time() - t0) / 60.0, 2),
        "criteria": {
            "hard_type_sensitive": means(rows_for("hard_type_sensitive"))
            | {"soft": False, "ignore_type": False, "headline": True},
            "hard_type_insensitive": means(rows_for("hard_type_insensitive"))
            | {"soft": False, "ignore_type": True, "headline": True},
            "soft_type_sensitive": means(rows_for("soft_type_sensitive"))
            | {"soft": True, "ignore_type": False, "headline": False},
            "soft_type_insensitive": means(rows_for("soft_type_insensitive"))
            | {"soft": True, "ignore_type": True, "headline": False},
        },
        "samples": per_sample,
    }
    out_path = OUT / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_table(summary)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
