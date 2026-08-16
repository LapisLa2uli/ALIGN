from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import time

from synthpipeline.config import SynthConfig
from synthpipeline.pipeline import generate_samples, generate_samples_parallel
from synthpipeline.soundfonts import SOUNDFONT_IDS, fetch_soundfonts, list_soundfonts


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
    gen.add_argument(
        "--midi-backend",
        choices=("music21", "musescore"),
        default="music21",
        help="MIDI export backend (default: music21)",
    )
    gen.add_argument(
        "--soundfont",
        choices=SOUNDFONT_IDS,
        help="Clarinet SoundFont: freepats, u220, mcb, or msbasic (default: config render.soundfont)",
    )
    gen.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel processes (default: 1)",
    )

    fonts = sub.add_parser("list-soundfonts", help="Show available clarinet SoundFonts")
    fetch = sub.add_parser("fetch-soundfonts", help="Download bundled clarinet SoundFonts")
    fetch.add_argument("--soundfont", choices=SOUNDFONT_IDS, action="append")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    config = SynthConfig.load(Path(args.config) if args.config else None)

    if args.command == "generate":
        workers = max(1, int(args.workers))
        started = time.perf_counter()
        if workers > 1:
            results = generate_samples_parallel(
                config=config,
                count=args.count,
                seed=args.seed,
                workers=workers,
                output_root=args.output,
                score_arg=args.score,
                midi_backend=args.midi_backend or "music21",
                soundfont=args.soundfont,
            )
        else:
            results = generate_samples(
                config=config,
                count=args.count,
                seed=args.seed,
                output_root=args.output,
                score_arg=args.score,
                midi_backend=args.midi_backend or "music21",
                soundfont=args.soundfont,
            )
        wall = time.perf_counter() - started
        root = args.output or config.output_root()
        print(f"Created {len(results)} samples under {root} with {workers} worker(s)")
        for item in results:
            print(
                f"  {item.sample_dir.name}: {item.elapsed_sec:.2f}s "
                f"({item.error_type}{' +repeat' if item.repeated else ''})"
            )
        if results:
            cpu = sum(item.elapsed_sec for item in results)
            print(
                f"Wall {wall:.2f}s  sum {cpu:.2f}s  mean {cpu / len(results):.2f}s  "
                f"workers {workers}"
            )
        return
    if args.command == "list-soundfonts":
        for row in list_soundfonts(config):
            status = "ready" if row["installed"] else "MISSING"
            print(f"{row['id']:10} {status:8} program={row['program']}  {row['label']}")
            if row["path"]:
                print(f"           {row['path']}")
        return
    if args.command == "fetch-soundfonts":
        logger = logging.getLogger("synthpipeline")
        paths = fetch_soundfonts(args.soundfont, logger)
        print(f"Fetched {len(paths)} soundfont(s)")
        for path in paths:
            print(f"  {path}")
        return
    parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
