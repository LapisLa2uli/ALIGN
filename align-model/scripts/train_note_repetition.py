"""Train Layer 1 note-sequence repetition scoring on procedural bundles."""

from __future__ import annotations

import argparse
from pathlib import Path

from alignmodel.note_repetition_train import train_note_repetition_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--val-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=365)
    args = parser.parse_args()
    checkpoint = train_note_repetition_model(
        args.manifest,
        args.out,
        train_samples=max(1, args.train_samples),
        val_samples=max(1, args.val_samples),
        epochs=max(1, args.epochs),
        batch_size=max(1, args.batch_size),
        device=args.device,
        seed=args.seed,
    )
    print(f"Wrote {checkpoint}")


if __name__ == "__main__":
    main()
