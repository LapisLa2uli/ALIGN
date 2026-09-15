"""Train the procedural-only Basic Pitch + PESTO clarinet note refiner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alignmodel.transcription.refiner import NoteRefinerConfig
from alignmodel.transcription.refiner_data import RefinerAugmentConfig
from alignmodel.transcription.refiner_train import (
    RefinerTrainConfig,
    train_note_refiner,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--basic-cache-root", type=Path, required=True)
    parser.add_argument("--pesto-cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-frames", type=int, default=384)
    parser.add_argument("--crops-per-clip", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=80)
    parser.add_argument("--interval-weight", type=float, default=0.20)
    parser.add_argument("--augment-probability", type=float, default=0.75)
    parser.add_argument("--short-note-weight", type=float, default=2.0)
    parser.add_argument("--hard-negative-ratio", type=float, default=1.0)
    parser.add_argument(
        "--same-pitch-split-probability", type=float, default=0.45
    )
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--temporal-blocks", type=int, default=6)
    parser.add_argument("--midi-min", type=int, default=36)
    parser.add_argument("--midi-max", type=int, default=108)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=365)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    model = NoteRefinerConfig(
        midi_min=args.midi_min,
        midi_max=args.midi_max,
        channels=args.channels,
        temporal_blocks=args.temporal_blocks,
        pesto_pitch_unit="midi",
        pesto_written_shift=0.0,
    )
    checkpoint = train_note_refiner(
        RefinerTrainConfig(
            manifest=args.manifest,
            basic_cache_root=args.basic_cache_root,
            pesto_cache_root=args.pesto_cache_root,
            output_dir=args.out,
            epochs=max(1, args.epochs),
            batch_size=max(1, args.batch_size),
            crop_frames=max(64, args.crop_frames),
            crops_per_clip=max(1, args.crops_per_clip),
            workers=max(0, args.workers),
            max_train_samples=max(0, args.max_train_samples),
            max_val_samples=max(0, args.max_val_samples),
            interval_weight=max(0.0, args.interval_weight),
            augment=RefinerAugmentConfig(
                probability=max(0.0, min(1.0, args.augment_probability)),
                short_note_positive_weight=max(
                    1.0, args.short_note_weight
                ),
                hard_negative_ratio=max(0.0, args.hard_negative_ratio),
                same_pitch_split_probability=max(
                    0.0,
                    min(1.0, args.same_pitch_split_probability),
                ),
            ),
            device=args.device,
            seed=args.seed,
            resume_from=args.resume,
            model=model,
        )
    )
    history = json.loads(
        (args.out / "history.json").read_text(encoding="utf-8")
    )
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "best_f1": history["best_f1"],
                "stopped_reason": history["stopped_reason"],
                "calibration": history["calibration"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
