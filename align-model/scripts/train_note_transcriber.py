"""Train the score-free audio-to-written-note recognizer.

Example:
  python align-model/scripts/train_note_transcriber.py \
    --root E:/output_2k_rawdata --root E:/other_takes \
    --manifest align-model/splits/note_transcription.json \
    --out align-model/runs/note-transcriber
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alignmodel.transcription import (
    NoteFrameNetConfig,
    NoteTrainConfig,
    train_note_transcriber,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train framewise audio-to-note transcription from cached ALIGN mels"
    )
    parser.add_argument(
        "--root",
        action="append",
        required=True,
        help="Bundle root; repeat for multiple E: roots",
    )
    parser.add_argument("--manifest", type=Path, required=True, help="JSON/JSONL train/val split")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-frames", type=int, default=1024)
    parser.add_argument("--crops-per-clip", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--infer-batch-size", type=int, default=4)
    parser.add_argument("--infer-window-frames", type=int, default=2048)
    parser.add_argument("--infer-overlap-frames", type=int, default=512)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from a last.pt checkpoint; --epochs remains the total target",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=365)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-val-clips", type=int, default=80)
    parser.add_argument("--calibrate-val-clips", type=int, default=0)
    parser.add_argument("--midi-min", type=int, default=36)
    parser.add_argument("--midi-max", type=int, default=108)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--temporal-channels", type=int, default=128)
    parser.add_argument("--spectral-blocks", type=int, default=3)
    parser.add_argument("--temporal-blocks", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_cfg = NoteFrameNetConfig(
        channels=args.channels,
        temporal_channels=args.temporal_channels,
        spectral_blocks=args.spectral_blocks,
        temporal_blocks=args.temporal_blocks,
        midi_min=args.midi_min,
        midi_max=args.midi_max,
    )
    cfg = NoteTrainConfig(
        output_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        crop_frames=args.crop_frames,
        crops_per_clip=args.crops_per_clip,
        lr=args.lr,
        num_workers=args.workers,
        prefetch_factor=args.prefetch_factor,
        infer_batch_size=args.infer_batch_size,
        infer_window_frames=args.infer_window_frames,
        infer_overlap_frames=args.infer_overlap_frames,
        device=args.device,
        seed=args.seed,
        resume_from=args.resume,
        early_stop_patience=args.patience,
        max_val_clips=args.max_val_clips,
        calibrate_val_clips=args.calibrate_val_clips,
        model=model_cfg,
    )
    checkpoint = train_note_transcriber(args.root, args.manifest, cfg)
    history = json.loads((args.out / "history.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "history": str(args.out / "history.json"),
                "stopped_reason": history["stopped_reason"],
                "calibrated_metrics": history["calibrated_metrics"],
                "calibration": history["calibration"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
