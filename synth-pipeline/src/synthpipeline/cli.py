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
    gen.add_argument(
        "--config",
        type=str,
        dest="generate_config",
        help="Path to synth-pipeline YAML config",
    )
    gen.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip sample IDs that already have a complete bundle on disk",
    )

    conv = sub.add_parser(
        "convert-labels",
        help="Add score-part melody fields to existing synth labels.json files",
    )
    conv.add_argument(
        "--root",
        type=Path,
        action="append",
        default=None,
        help="Bundle root (repeatable; default: output_root)",
    )
    conv.add_argument("--pad", type=int, default=2)
    conv.add_argument("--pad-random", action="store_true")
    conv.add_argument("--force", action="store_true")
    conv.add_argument("--seed", type=int, default=365)
    conv.add_argument("--workers", type=int, default=8)

    trans = sub.add_parser(
        "transpose-audio",
        help="Shift existing synth WAVs to sounding pitch (Bb clarinet: -2 semitones)",
    )
    trans.add_argument("--root", type=Path, action="append", default=None)
    trans.add_argument("--semitones", type=int, default=-2)
    trans.add_argument("--force", action="store_true")
    trans.add_argument("--workers", type=int, default=8)

    regen = sub.add_parser(
        "regenerate-audio",
        help="Re-render bundle WAVs from MIDI at sounding pitch (keeps pitch-bends)",
    )
    regen.add_argument("--root", type=Path, action="append", default=None)
    regen.add_argument("--semitones", type=int, default=-2)
    regen.add_argument("--force", action="store_true")
    regen.add_argument("--workers", type=int, default=8)

    fonts = sub.add_parser("list-soundfonts", help="Show available clarinet SoundFonts")
    fetch = sub.add_parser("fetch-soundfonts", help="Download bundled clarinet SoundFonts")
    fetch.add_argument("--soundfont", choices=SOUNDFONT_IDS, action="append")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    config_path = getattr(args, "generate_config", None) or args.config
    config = SynthConfig.load(Path(config_path) if config_path else None)

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
                skip_existing=bool(args.skip_existing),
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
                skip_existing=bool(args.skip_existing),
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
    if args.command == "convert-labels":
        from synthpipeline.convert_labels import convert_root

        roots = args.root or [config.output_root()]
        failed = 0
        for root in roots:
            counts = convert_root(
                Path(root),
                pad=args.pad,
                force=bool(args.force),
                pad_random=bool(args.pad_random),
                seed=args.seed,
                workers=args.workers,
            )
            print(
                f"root={root} bundles={counts.get('n_bundles', 0)} "
                f"converted={counts['converted']} skip_done={counts['skip_done']} "
                f"skip_missing={counts['skip_missing']} empty={counts['empty']} "
                f"failed={counts['failed']}"
            )
            failed += counts["failed"]
        return 0 if failed == 0 else 2
    if args.command == "transpose-audio":
        from synthpipeline.transpose_audio import transpose_root

        roots = args.root or [config.output_root()]
        failed = 0
        for root in roots:
            counts = transpose_root(
                Path(root),
                semitones=int(args.semitones),
                force=bool(args.force),
                workers=int(args.workers),
            )
            print(
                f"root={root} bundles={counts['n_bundles']} converted={counts['converted']} "
                f"skip_done={counts['skip_done']} skip_missing={counts['skip_missing']} "
                f"failed={counts['failed']} failed_mel={counts.get('failed_mel', 0)}"
            )
            failed += counts["failed"] + counts.get("failed_mel", 0)
        return 0 if failed == 0 else 2
    if args.command == "regenerate-audio":
        from synthpipeline.regenerate_audio import regenerate_root

        roots = args.root or [config.output_root()]
        failed = 0
        for root in roots:
            counts = regenerate_root(
                Path(root),
                semitones=int(args.semitones),
                force=bool(args.force),
                workers=int(args.workers),
            )
            print(
                f"root={root} bundles={counts['n_bundles']} converted={counts['converted']} "
                f"skip_done={counts['skip_done']} skip_missing={counts['skip_missing']} "
                f"failed={counts['failed']}"
            )
            failed += counts["failed"]
        return 0 if failed == 0 else 2
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
