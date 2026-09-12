"""Train the learned note-list to written-score aligner."""

from __future__ import annotations

import argparse
from pathlib import Path

from alignmodel.note_align_train import NoteAlignTrainConfig, train_note_aligner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, help="14k bundle root or exact-cache root")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/note-aligner"),
        help="Checkpoint/history destination",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional separate exact note-map cache directory",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional frozen split manifest; uses its train/val rows",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--learned-weight", type=float, default=0.65)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=365)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--augmentations-per-map", type=int, default=1)
    parser.add_argument("--drop-probability", type=float, default=0.08)
    parser.add_argument("--pitch-error-probability", type=float, default=0.10)
    parser.add_argument("--spurious-probability", type=float, default=0.08)
    parser.add_argument("--timing-jitter-sec", type=float, default=0.035)
    parser.add_argument("--calibration-maps", type=int, default=128)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument(
        "--transcriber-ckpt",
        type=Path,
        default=None,
        help="Also train on decoded audio notes from this checkpoint",
    )
    parser.add_argument(
        "--build-missing-cache",
        action="store_true",
        help="Build exact caches from synthetic clean/performance scores",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_note_aligner(
        NoteAlignTrainConfig(
            data_root=args.data_root,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            manifest=args.manifest,
            epochs=max(1, args.epochs),
            batch_size=max(32, args.batch_size),
            lr=args.lr,
            weight_decay=max(0.0, args.weight_decay),
            hidden_dim=max(4, args.hidden_dim),
            learned_weight=min(1.0, max(0.0, args.learned_weight)),
            device=args.device,
            seed=args.seed,
            max_samples=max(0, args.max_samples),
            augmentations_per_map=max(0, args.augmentations_per_map),
            drop_probability=min(1.0, max(0.0, args.drop_probability)),
            pitch_error_probability=min(
                1.0, max(0.0, args.pitch_error_probability)
            ),
            spurious_probability=min(1.0, max(0.0, args.spurious_probability)),
            timing_jitter_sec=max(0.0, args.timing_jitter_sec),
            calibration_maps=max(1, args.calibration_maps),
            early_stop_patience=max(1, args.early_stop_patience),
            build_missing_cache=args.build_missing_cache,
            transcriber_checkpoint=args.transcriber_ckpt,
        )
    )


if __name__ == "__main__":
    main()
