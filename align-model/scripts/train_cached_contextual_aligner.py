"""Fine-tune contextual alignment on cached Basic Pitch + Layer 1 output."""

from __future__ import annotations

import argparse
from pathlib import Path

from alignmodel.cached_alignment_train import train_cached_contextual_aligner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--basic-cache-root", type=Path, required=True)
    parser.add_argument("--repetition-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--val-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = train_cached_contextual_aligner(
        args.manifest,
        args.basic_cache_root,
        args.repetition_checkpoint,
        args.initial_checkpoint,
        args.out,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
