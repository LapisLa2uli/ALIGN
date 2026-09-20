"""Restore epoch-30 weights, decode DataCreate, write GUI labels, then score 001-030."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RETRAIN = ROOT / "experiments" / "baselines_retrain_20260918"
DATA = ROOT / "baselines" / "data" / "datacreate_all_20260920"
SAMPLES = ROOT / "DataCreate" / "samples"
MODELS = ("polytune", "laddersym")
PROTOCOL = (
    "bf16 autocast, batch 1, 1024 tokens, prompted deterministic LadderSym, "
    "epoch-30 selected checkpoints"
)


def run(cmd: list[str], **kwargs) -> None:
    print("+", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def python_for(stage: str) -> str:
    infer = ROOT / "baselines" / "envs" / "polytune" / "Scripts" / "python.exe"
    convert = ROOT / "align-model" / ".venv-amt-bench" / "Scripts" / "python.exe"
    if stage == "infer" and infer.is_file():
        return str(infer)
    if convert.is_file():
        return str(convert)
    return sys.executable


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_freeze_manifest() -> Path:
    ids = []
    split_path = DATA / "split.json"
    if split_path.is_file():
        split = _read(split_path)
        ids = sorted(
            Path(path).name.replace(".midi", "")
            for key, path in split["midi_filename"].items()
            if split["split"].get(key) == "test"
        )
    models = {}
    failures = []
    for model in MODELS:
        inference = HERE / "predictions" / model / "inference_manifest.json"
        labels = HERE / f"{model}_label_manifest.json"
        inference_payload = _read(inference) if inference.is_file() else {}
        label_payload = _read(labels) if labels.is_file() else {}
        pred_ids = [row["sample"] for row in inference_payload.get("rows") or []]
        label_ids = [row["sample"] for row in label_payload.get("rows") or []]
        missing_pred = [sample_id for sample_id in ids if sample_id not in pred_ids]
        missing_labels = [sample_id for sample_id in ids if sample_id not in label_ids]
        if missing_pred:
            failures.append({"model": model, "stage": "inference", "ids": missing_pred})
        if missing_labels:
            failures.append({"model": model, "stage": "labels", "ids": missing_labels})
        models[model] = {
            "n_predictions": len(pred_ids),
            "n_labels": len(label_ids),
            "checkpoint": inference_payload.get("checkpoint"),
            "checkpoint_sha256": inference_payload.get("checkpoint_sha256"),
            "selected": inference_payload.get("selected"),
            "hydra_config_sha256": inference_payload.get("hydra_config_sha256"),
            "precision": inference_payload.get("precision"),
            "batch_size": inference_payload.get("batch_size"),
            "max_length": inference_payload.get("max_length"),
            "prompted": inference_payload.get("prompted"),
            "label_counts_by_type": label_payload.get("counts_by_type"),
            "total_labels": label_payload.get("total_labels"),
            "unlocated_events": int(
                sum(int(row.get("unlocated_events") or 0) for row in label_payload.get("rows") or [])
            ),
            "inference_manifest_sha256": sha256(inference),
            "label_manifest_sha256": sha256(labels),
            "rows": label_payload.get("rows") or [],
        }
    freeze = {
        "experiment": "baselines_datacreate_all_20260920",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data": str(DATA),
        "data_manifest_sha256": sha256(DATA / "manifest.json"),
        "data_split_sha256": sha256(DATA / "split.json"),
        "samples": str(SAMPLES),
        "ids": ids,
        "n_ids": len(ids),
        "predictions": {
            "polytune": str(HERE / "predictions" / "polytune"),
            "laddersym": str(HERE / "predictions" / "laddersym"),
        },
        "protocol": PROTOCOL,
        "source_hashes": sha256(RETRAIN / "source_hashes.json"),
        "protocol_sha256": sha256(RETRAIN / "protocol.json"),
        "models": models,
        "failures": failures,
        "human_labels_json_touched": False,
        "promotion": "experimental",
    }
    path = HERE / "freeze_manifest.json"
    path.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore-host", default="i-2.gpushare.com")
    parser.add_argument("--restore-attempts", type=int, default=180)
    parser.add_argument("--restore-delay", type=int, default=20)
    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument("--skip-labels", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    if not args.skip_infer:
        for model in MODELS:
            run(
                [
                    python_for("infer"),
                    str(RETRAIN / "infer_datacreate.py"),
                    "--model",
                    model,
                    "--data",
                    str(DATA),
                    "--restore-host",
                    args.restore_host,
                    "--restore-attempts",
                    str(args.restore_attempts),
                    "--restore-delay",
                    str(args.restore_delay),
                    "--out",
                    str(HERE / "predictions" / model),
                ]
            )
    if not args.skip_labels:
        for model in MODELS:
            run(
                [
                    python_for("convert"),
                    str(ROOT / "baselines" / "scripts" / "label_datacreate_baselines.py"),
                    "--model",
                    model,
                    "--pred-dir",
                    str(HERE / "predictions" / model),
                    "--samples",
                    str(SAMPLES),
                    "--output",
                    str(HERE),
                    "--inference-manifest",
                    str(HERE / "predictions" / model / "inference_manifest.json"),
                ]
            )
    if not args.skip_eval:
        run(
            [
                python_for("convert"),
                str(HERE / "evaluate.py"),
                "--samples",
                str(SAMPLES),
                "--output",
                str(HERE),
            ]
        )
    print(write_freeze_manifest(), flush=True)


if __name__ == "__main__":
    main()
