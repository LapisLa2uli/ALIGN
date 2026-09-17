"""Train Track B's score-free high-resolution mel transcriber from scratch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from alignmodel.training_resources import resource_lease
from alignmodel.transcription.mel_v1 import MelTranscriberConfig
from alignmodel.transcription.mel_v1_train import (
    MelTrainConfig,
    train_mel_transcriber,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-frames", type=int, default=1024)
    parser.add_argument("--crops-per-clip", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--checkpoint-every-steps", type=int, default=100)
    parser.add_argument("--calibration-max-clips", type=int, default=64)
    parser.add_argument("--calibration-fraction", type=float, default=0.05)
    parser.add_argument("--calibration-every-epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--amp", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--augmentation-probability", type=float, default=0.55)
    parser.add_argument("--temporal-kind", choices=("tcn", "bigru"), default="tcn")
    parser.add_argument("--temporal-dim", type=int, default=128)
    parser.add_argument("--temporal-blocks", type=int, default=8)
    parser.add_argument("--conv-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    model = MelTranscriberConfig(
        temporal_kind=args.temporal_kind,
        temporal_dim=args.temporal_dim,
        temporal_blocks=args.temporal_blocks,
        conv_channels=args.conv_channels,
    )
    config = MelTrainConfig(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        crop_frames=args.crop_frames,
        crops_per_clip=args.crops_per_clip,
        learning_rate=args.learning_rate,
        accumulation_steps=args.accumulation_steps,
        workers=args.workers,
        prefetch_factor=args.prefetch_factor,
        checkpoint_every_steps=args.checkpoint_every_steps,
        calibration_fraction=args.calibration_fraction,
        calibration_max_clips=args.calibration_max_clips,
        calibration_every_epochs=args.calibration_every_epochs,
        early_stop_patience=args.patience,
        amp=args.amp,
        compile_model=args.compile,
        max_train_rows=args.max_train_rows,
        augmentation_probability=args.augmentation_probability,
        model=model,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "schema_version": "align-mel-transcriber-train-config-v1",
                "cache": str(args.cache.resolve()),
                "device": args.device,
                "from_scratch": args.resume is None,
                "basic_pitch_dependency": False,
                "score_input_to_acoustic_model": False,
                "locked_test_materialized": False,
                "config": {
                    **vars(args),
                    "cache": str(args.cache),
                    "output_dir": str(args.output_dir),
                    "resource_status": str(args.resource_status),
                    "resume": str(args.resume) if args.resume else None,
                },
                "model": model.to_dict(),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    with resource_lease(
        args.resource_status,
        "gpu",
        track="mel-transcriber-v1-full-training",
        command=[sys.executable, *sys.argv],
        metadata={
            "cache": str(args.cache.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "architecture": args.temporal_kind,
            "basic_pitch_dependency": False,
        },
    ):
        checkpoint = train_mel_transcriber(
            args.cache,
            config,
            device=args.device,
            resume=args.resume,
        )
    print(json.dumps({"checkpoint": str(checkpoint)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
