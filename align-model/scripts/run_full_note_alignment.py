"""Resume-safe full E:\\ note-alignment target, training, and evaluation run."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _run(args: list[str]) -> None:
    print("+", subprocess.list2cmdline(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def _training_finished(output_dir: Path, checkpoint: Path) -> bool:
    history_path = output_dir / "history.json"
    if not checkpoint.exists() or not history_path.exists():
        return False
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return history.get("stopped_reason") is not None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--procedural-root", type=Path, default=Path("E:/output"))
    parser.add_argument("--raw-root", type=Path, default=Path("E:/output_2k_rawdata"))
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "align-model" / "runs" / "note-align-full-e-s365",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=365)
    parser.add_argument("--transcriber-epochs", type=int, default=24)
    parser.add_argument("--transcriber-workers", type=int, default=16)
    parser.add_argument("--transcriber-batch-size", type=int, default=64)
    parser.add_argument("--transcriber-prefetch-factor", type=int, default=3)
    parser.add_argument("--transcriber-max-val-clips", type=int, default=80)
    parser.add_argument("--aligner-epochs", type=int, default=12)
    parser.add_argument("--force-targets", action="store_true")
    args = parser.parse_args()

    procedural = args.procedural_root.resolve()
    raw = args.raw_root.resolve()
    missing = [str(path) for path in (procedural, raw) if not path.exists()]
    if missing:
        raise SystemExit(
            "Cannot start full training because these dataset roots are not mounted: "
            + ", ".join(missing)
        )

    out = args.out.resolve()
    targets = out / "targets"
    targets_complete = targets / "backfill-complete.json"
    split = out / "split.json"
    local_transcriber_split = out / "transcriber-split-local.json"
    transcriber_dir = out / "transcriber-v2"
    aligner_dir = out / "aligner-v2"
    weights = out / "weights-v2"
    evaluation = out / "evaluation-v2.json"
    out.mkdir(parents=True, exist_ok=True)

    if not split.exists():
        _run(
            [
                sys.executable,
                "align-model/scripts/prepare_note_alignment_split.py",
                "--procedural-root",
                str(procedural),
                "--raw-root",
                str(raw),
                "--out",
                str(split),
                "--seed",
                str(args.seed),
            ]
        )

    if args.force_targets or not targets_complete.exists():
        target_force = ["--force"] if args.force_targets else []
        for corpus, data_root, config in (
            (
                "procedural12k",
                procedural,
                ROOT / "synth-pipeline" / "config" / "multi_error_10k.yaml",
            ),
            (
                "raw2k",
                raw,
                ROOT / "synth-pipeline" / "config" / "rawdata_snippets_2k.yaml",
            ),
        ):
            _run(
                [
                    sys.executable,
                    "align-model/scripts/backfill_synth_note_maps.py",
                    "--root",
                    str(data_root),
                    "--output-root",
                    str(targets / corpus),
                    "--config",
                    str(config),
                    "--workers",
                    str(args.workers),
                    "--allow-failures",
                    *target_force,
                ]
            )
        targets_complete.write_text(
            json.dumps(
                {
                    "procedural_root": str(procedural),
                    "raw_root": str(raw),
                    "allow_failures": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    transcriber_ckpt = transcriber_dir / "best.pt"
    if not _training_finished(transcriber_dir, transcriber_ckpt):
        transcriber_manifest = (
            local_transcriber_split
            if local_transcriber_split.exists()
            else split
        )
        transcriber_args = [
            sys.executable,
            "align-model/scripts/train_note_transcriber.py",
            "--root",
            str(procedural),
            "--root",
            str(raw),
            "--manifest",
            str(transcriber_manifest),
            "--out",
            str(transcriber_dir),
            "--epochs",
            str(args.transcriber_epochs),
            "--batch-size",
            str(args.transcriber_batch_size),
            "--workers",
            str(args.transcriber_workers),
            "--prefetch-factor",
            str(args.transcriber_prefetch_factor),
            "--infer-batch-size",
            "8",
            "--infer-window-frames",
            "2048",
            "--infer-overlap-frames",
            "512",
            "--crop-frames",
            "1024",
            "--crops-per-clip",
            "2",
            "--max-val-clips",
            str(args.transcriber_max_val_clips),
            "--calibrate-val-clips",
            "0",
            "--channels",
            "64",
            "--temporal-channels",
            "128",
            "--spectral-blocks",
            "3",
            "--temporal-blocks",
            "10",
            "--midi-max",
            "108",
            "--device",
            args.device,
            "--seed",
            str(args.seed),
        ]
        transcriber_last = transcriber_dir / "last.pt"
        if transcriber_last.exists():
            transcriber_args.extend(["--resume", str(transcriber_last)])
        _run(transcriber_args)

    aligner_ckpt = aligner_dir / "note_aligner.pt"
    if not aligner_ckpt.exists():
        _run(
            [
                sys.executable,
                "align-model/scripts/train_note_aligner.py",
                str(procedural),
                "--manifest",
                str(split),
                "--cache-dir",
                str(targets),
                "--output-dir",
                str(aligner_dir),
                "--epochs",
                str(args.aligner_epochs),
                "--batch-size",
                "2048",
                "--augmentations-per-map",
                "1",
                "--transcriber-ckpt",
                str(transcriber_ckpt),
                "--device",
                args.device,
                "--seed",
                str(args.seed),
            ]
        )

    weights.mkdir(parents=True, exist_ok=True)
    shutil.copy2(transcriber_ckpt, weights / "note_transcriber.pt")
    shutil.copy2(aligner_ckpt, weights / "note_aligner.pt")

    if not evaluation.exists():
        _run(
            [
                sys.executable,
                "align-model/scripts/eval_full_note_alignment.py",
                "--manifest",
                str(split),
                "--cache-root",
                str(targets),
                "--weights",
                str(weights),
                "--device",
                args.device,
                "--decode",
                "neural",
                "--onset-tolerance-ms",
                "50",
                "--out",
                str(evaluation),
            ]
        )
    print(f"Full note-alignment run complete: {evaluation}", flush=True)


if __name__ == "__main__":
    main()
