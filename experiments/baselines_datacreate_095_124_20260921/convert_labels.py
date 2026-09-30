"""Convert frozen baseline MIDI into schema-linked DataCreate label documents."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SAMPLES = ROOT / "DataCreate" / "samples"


def python_for_convert() -> str:
    candidate = ROOT / "align-model" / ".venv-amt-bench" / "Scripts" / "python.exe"
    return str(candidate if candidate.is_file() else sys.executable)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("polytune", "laddersym"), required=True)
    args = parser.parse_args()
    pred = HERE / "predictions" / args.model
    cmd = [
        python_for_convert(),
        str(ROOT / "baselines" / "scripts" / "label_datacreate_baselines.py"),
        "--model",
        args.model,
        "--pred-dir",
        str(pred),
        "--samples",
        str(SAMPLES),
        "--output",
        str(HERE),
        "--inference-manifest",
        str(pred / "inference_manifest.json"),
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
