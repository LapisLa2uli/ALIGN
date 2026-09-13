"""Resume-safe procedural-only Basic Pitch refiner training and promotion."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _run(arguments: list[str], *, python: Path | None = None) -> None:
    command = [str(python or Path(sys.executable)), *arguments]
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT
        / "align-model"
        / "runs"
        / "note-align-full-e-s365"
        / "split.json",
    )
    parser.add_argument(
        "--note-map-root",
        type=Path,
        default=ROOT
        / "align-model"
        / "runs"
        / "note-align-full-e-s365"
        / "targets",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "align-model"
        / "runs"
        / "basic-pitch-refiner-procedural-s365",
    )
    parser.add_argument(
        "--external-python",
        type=Path,
        default=ROOT
        / "align-model"
        / ".venv-amt-bench"
        / "Scripts"
        / "python.exe",
    )
    parser.add_argument(
        "--baseline-aligner",
        type=Path,
        default=ROOT
        / "align-model"
        / "runs"
        / "note-align-full-e-s365"
        / "aligner-v2"
        / "note_aligner.pt",
        help="Frozen aligner used until all refiner promotion gates pass",
    )
    parser.add_argument("--effective-audio-transpose", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=80)
    parser.add_argument("--max-test-samples", type=int, default=80)
    parser.add_argument("--cache-workers", type=int, default=4)
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=365)
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "split.json"
    basic_cache = out / "basic-pitch-cache"
    pesto_cache = out / "pesto-cache"
    refiner_dir = out / "refiner"
    evaluation = out / "evaluation-refiner.json"
    weights = out / "weights"
    weights.mkdir(exist_ok=True)

    if not manifest.exists():
        _run(
            [
                "align-model/scripts/prepare_basic_pitch_refiner_split.py",
                "--source-manifest",
                str(args.source_manifest),
                "--note-map-root",
                str(args.note_map_root),
                "--out",
                str(manifest),
                "--effective-audio-transpose",
                str(args.effective_audio_transpose),
            ]
        )

    cache_limits = (
        ("train", args.max_train_samples),
        ("val", args.max_val_samples),
        ("test_id", args.max_test_samples),
    )
    for split, limit in (() if args.skip_cache else cache_limits):
        basic_args = [
            "align-model/scripts/cache_basic_pitch_features.py",
            "--manifest",
            str(manifest),
            "--split",
            split,
            "--cache-root",
            str(basic_cache),
            "--workers",
            str(args.cache_workers),
            "--procedural-only",
        ]
        if limit:
            basic_args.extend(["--max-samples", str(limit)])
        _run(basic_args, python=args.external_python)

        pesto_args = [
            "align-model/scripts/cache_pesto_features.py",
            "--manifest",
            str(manifest),
            "--basic-cache-root",
            str(basic_cache),
            "--cache-root",
            str(pesto_cache),
            "--split",
            split,
            "--device",
            args.device,
            "--procedural-only",
        ]
        if limit:
            pesto_args.extend(["--max-samples", str(limit)])
        _run(pesto_args, python=args.external_python)

    checkpoint = refiner_dir / "best.pt"
    if not checkpoint.exists():
        training_args = [
            "align-model/scripts/train_note_refiner.py",
            "--manifest",
            str(manifest),
            "--basic-cache-root",
            str(basic_cache),
            "--pesto-cache-root",
            str(pesto_cache),
            "--out",
            str(refiner_dir),
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--max-val-samples",
            str(args.max_val_samples),
        ]
        if args.max_train_samples:
            training_args.extend(
                ["--max-train-samples", str(args.max_train_samples)]
            )
        _run(training_args)

    if not evaluation.exists():
        eval_args = [
            "align-model/scripts/eval_note_refiner.py",
            "--manifest",
            str(manifest),
            "--checkpoint",
            str(checkpoint),
            "--basic-cache-root",
            str(basic_cache),
            "--pesto-cache-root",
            str(pesto_cache),
            "--split",
            "val",
            "--split",
            "test_id",
            "--device",
            args.device,
            "--out",
            str(evaluation),
        ]
        if args.max_test_samples:
            eval_args.extend(["--max-samples", str(args.max_test_samples)])
        _run(eval_args, python=args.external_python)

    report = json.loads(evaluation.read_text(encoding="utf-8"))
    val = report["splits"]["val"]["metrics"]
    test_id = report["splits"]["test_id"]["metrics"]
    baseline = report["splits"]["test_id"]["frozen_basic_pitch"]
    promoted = (
        float(test_id["f1"]) >= 0.85
        and float(test_id["f1"]) > float(baseline["f1"])
        and 0.85 <= float(test_id["pred_target_ratio"]) <= 1.15
    )
    shutil.copy2(checkpoint, out / "candidate-note-refiner.pt")
    decoder_json = weights / "note_decoder.json"
    decoder_pt = weights / "note_decoder.pt"
    if promoted:
        shutil.copy2(checkpoint, decoder_pt)
        decoder_json.unlink(missing_ok=True)
    else:
        decoder_pt.unlink(missing_ok=True)
        decoder_json.write_text(
            json.dumps(
                {
                    "format_version": 3,
                    "kind": "basic-pitch",
                    "frontend": "basic-pitch-0.4.0",
                    "fine_pitch": "pesto-2.0.1",
                    "runtime": "tensorflow",
                    "corpus": "procedural12k",
                    "effective_audio_transpose": args.effective_audio_transpose,
                    "basic_cache_root": str(basic_cache),
                    "pesto_cache_root": str(pesto_cache),
                    "decode": {
                        "onset_threshold": 0.5,
                        "frame_threshold": 0.4,
                        "minimum_note_length_ms": 55.0,
                    },
                    "promotion_reason": "refiner_gate_failed",
                    "candidate_metrics": test_id,
                    "baseline_metrics": baseline,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if args.baseline_aligner.is_file():
        shutil.copy2(args.baseline_aligner, weights / "note_aligner.pt")
    summary = {
        "procedural_only": True,
        "raw_derived_data_used": False,
        "promoted_refiner": promoted,
        "validation": val,
        "test_id": test_id,
        "frozen_basic_pitch_test_id": baseline,
        "published_decoder": str(decoder_pt if promoted else decoder_json),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
