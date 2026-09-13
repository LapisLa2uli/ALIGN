"""Train the Layer-1-aware contextual note aligner on procedural maps."""

from __future__ import annotations

import argparse
from pathlib import Path

from alignmodel.contextual_align_train import train_contextual_aligner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--val-samples", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=365)
    args = parser.parse_args()
    path = train_contextual_aligner(
        args.manifest,
        args.out,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
