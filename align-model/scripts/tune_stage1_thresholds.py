"""Sweep Stage 1 copy/boundary thresholds on the non-holdout calib pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alignmodel.eval_melodies import eval_dirs
from alignmodel.types import PipelineConfig
from eval_stage_dev import calib_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("E:/output"))
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=365)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("align-model/runs/model-a-improve/stage1-threshold-sweep.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dirs = calib_dirs(args.data, args.seed, args.n)
    grid = [
        {"copy_sim_threshold": 0.64, "min_window_sec": 0.45},
        {"copy_sim_threshold": 0.72, "min_window_sec": 0.55},
        {"copy_sim_threshold": 0.80, "min_window_sec": 0.55},
        {"copy_sim_threshold": 0.76, "min_window_sec": 0.70},
    ]
    rows = []
    for params in grid:
        config = PipelineConfig(
            weights_dir=str(args.weights) if args.weights else None,
            rhythm_logit_override=None,
            **params,
        )
        report = eval_dirs(
            dirs,
            stages={1},
            weights_dir=args.weights,
            device=args.device,
            config=config,
            types={"repetition"},
        )
        row = {
            **params,
            "mean_melody_f1": report["mean_melody_f1"],
            "mean_melody_precision": report["mean_melody_precision"],
            "mean_melody_recall": report["mean_melody_recall"],
            "mean_n_gold": report["mean_n_gold"],
            "mean_n_pred": report["mean_n_pred"],
            "per_type": report.get("per_type"),
        }
        rows.append(row)
        print(json.dumps(row, indent=2))
    best = max(rows, key=lambda r: (r["mean_melody_f1"], -abs(r["mean_n_pred"] - r["mean_n_gold"])))
    payload = {"n": args.n, "seed": args.seed, "rows": rows, "best": best}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"best={json.dumps(best)}")


if __name__ == "__main__":
    main()
