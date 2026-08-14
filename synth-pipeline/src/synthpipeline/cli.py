from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from synthpipeline.config import SynthConfig
from synthpipeline.pipeline import generate_samples


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate clarinet MusicXML scores and ALIGN bundles with known errors"
    )
    parser.add_argument("--config", type=str, help="Path to synth-pipeline YAML config")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate synthetic ALIGN sample bundles")
    gen.add_argument("--count", type=int, default=1, help="Number of samples to generate")
    gen.add_argument(
        "--score",
        type=Path,
        help="Existing MusicXML file or directory (omit to generate original scores)",
    )
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--output", type=Path, help="Output root (default: config paths.output_root)")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    config = SynthConfig.load(Path(args.config) if args.config else None)

    if args.command == "generate":
        dirs = generate_samples(
            config=config,
            count=args.count,
            seed=args.seed,
            output_root=args.output,
            score_arg=args.score,
        )
        root = args.output or config.output_root()
        print(f"Created {len(dirs)} samples under {root}")
        return
    parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
