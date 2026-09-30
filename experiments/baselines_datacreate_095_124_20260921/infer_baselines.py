"""Infer Polytune and LadderSym on the prepared 095-124 DataCreate split."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RETRAIN = ROOT / "experiments" / "baselines_retrain_20260918"
DATA = ROOT / "baselines" / "data" / "datacreate_095_124_20260921"
IDS = [f"{index:03d}" for index in range(95, 125)]


def python_for_infer() -> str:
    candidate = ROOT / "baselines" / "envs" / "polytune" / "Scripts" / "python.exe"
    return str(candidate if candidate.is_file() else sys.executable)


def checkpoint_for(model: str) -> Path:
    paths = [
        RETRAIN / "training" / model / "best.pt",
        HERE / "checkpoints" / model / "best.pt",
        ROOT / "baselines" / "runs" / "retrain_20260918" / model / "best.pt",
    ]
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No {model} checkpoint. Expected one of: " + ", ".join(str(path) for path in paths)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("polytune", "laddersym"), required=True)
    args = parser.parse_args()
    ckpt = checkpoint_for(args.model)
    dest = HERE / "predictions" / args.model
    cmd = [
        python_for_infer(),
        str(RETRAIN / "infer_datacreate.py"),
        "--model",
        args.model,
        "--data",
        str(DATA),
        "--ckpt",
        str(ckpt),
        "--no-restore",
        "--out",
        str(dest),
        "--ids",
        *IDS,
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
