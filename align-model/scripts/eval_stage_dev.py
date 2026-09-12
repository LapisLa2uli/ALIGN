"""Leakage-safe Stage 1/2/3 melody-F1 evaluation on calib or holdout clips."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from alignmodel.dataset import list_sample_dirs
from alignmodel.eval_melodies import eval_dirs
from alignmodel.stage_train import _holdout_sample_dirs
from alignmodel.types import PipelineConfig


def calib_dirs(root: Path, seed: int = 365, n: int = 20) -> list[Path]:
    holdout = _holdout_sample_dirs(root, seed)
    pool = [
        sample
        for sample in list_sample_dirs(root)
        if sample not in holdout and (sample / "verified_score.musicxml").exists()
    ]
    rng = random.Random(seed + 17)
    rng.shuffle(pool)
    return pool[:n]


def holdout_dirs(root: Path, seed: int = 365, n: int = 100) -> list[Path]:
    selected = sorted(_holdout_sample_dirs(root, seed), key=lambda p: p.name)
    rng = random.Random(seed)
    shuffled = list(selected)
    rng.shuffle(shuffled)
    return shuffled[:n]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("E:/output"))
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--stages", default="1,2")
    parser.add_argument("--pool", choices=("calib", "holdout"), default="calib")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=365)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--soft", action="store_true")
    parser.add_argument(
        "--types",
        default="",
        help="Comma-separated gold/pred types to score; empty scores every type",
    )
    parser.add_argument("--rhythm-logit-override", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stages = {int(x.strip()) for x in str(args.stages).split(",") if x.strip()}
    dirs = (
        calib_dirs(args.data, args.seed, args.n)
        if args.pool == "calib"
        else holdout_dirs(args.data, args.seed, args.n)
    )
    types = {item.strip() for item in str(args.types).split(",") if item.strip()}
    config = PipelineConfig(weights_dir=str(args.weights) if args.weights else None)
    if args.rhythm_logit_override is not None:
        config.rhythm_logit_override = args.rhythm_logit_override
    report = eval_dirs(
        dirs,
        stages=stages,
        weights_dir=args.weights,
        device=args.device,
        soft=bool(args.soft),
        config=config,
        types=types or None,
    )
    report["pool"] = args.pool
    report["n_requested"] = args.n
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
