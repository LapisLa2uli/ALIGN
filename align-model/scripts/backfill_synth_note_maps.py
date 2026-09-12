"""Replay synth generation and write validated note-map caches.

This script never derives lineage from performance_score.musicxml.  It replays
the seeded in-memory mutation, then uses both saved score files only to reject
a replay whose sounding-note signatures differ from the original bundle.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from music21 import converter

from synthpipeline.config import SynthConfig
from synthpipeline.errors import ensure_expressible_durations, inject_error
from synthpipeline.note_map import (
    attach_rendered_events,
    build_note_map,
    note_signatures,
    tag_clean_notes,
    write_note_map,
)
from synthpipeline.scoregen import (
    generate_score,
    load_score,
    resolve_score_inputs,
    snippet_score,
    sounding_note_count,
    write_musicxml,
)


def replay_scores(
    metadata: dict,
    config: SynthConfig,
) -> tuple[object, object, dict]:
    """Recreate the exact clean/performed score pair for one bundle."""
    sample_seed = int(metadata["seed"])
    sample_index = int(metadata["index"])
    expected_source = str(metadata.get("source") or "")
    score_paths = resolve_score_inputs(None, config)
    last_error: Exception | None = None

    for attempt in range(24):
        rng_seed = sample_seed if attempt == 0 else sample_seed + 1009 * attempt
        rng = random.Random(rng_seed)
        try:
            if score_paths is None:
                source = "gen"
                clean = generate_score(rng, config)
                snippet_meta: dict = {}
            else:
                source_path = score_paths[(sample_index + attempt) % len(score_paths)]
                source = source_path.stem
                clean = load_score(source_path, config)
                snippet_meta = {"source_score": str(source_path)}
                if bool(config.generation.get("use_snippets", False)):
                    clean, picked = snippet_score(clean, rng, config)
                    snippet_meta.update(picked)
            if sounding_note_count(clean) <= 0:
                raise ValueError("Score has no sounding notes")
            if expected_source and source != expected_source:
                raise ValueError(
                    f"source mismatch: replay selected {source!r}, metadata has "
                    f"{expected_source!r}"
                )

            # _build_sample writes each score before the next stage.  The writer
            # normalizes inexpressible durations in-place; mirror that without I/O.
            ensure_expressible_durations(clean)
            tag_clean_notes(clean)
            result = inject_error(copy.deepcopy(clean), rng, config)
            ensure_expressible_durations(result.score)
            return clean, result.score, {
                "source": source,
                "rng_seed": rng_seed,
                "prep_attempt": attempt,
                **snippet_meta,
            }
        except Exception as exc:
            last_error = exc
            if expected_source and "source mismatch:" in str(exc):
                raise
            continue
    raise RuntimeError(f"could not replay score preparation: {last_error}")


def validate_replay(bundle: Path, clean_score, performed_score) -> None:
    expected = {
        "verified_score.musicxml": _serialized_signatures(
            clean_score, "verified_score.musicxml"
        ),
        "performance_score.musicxml": _serialized_signatures(
            performed_score, "performance_score.musicxml"
        ),
    }
    for filename, replayed in expected.items():
        path = bundle / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        saved = converter.parse(str(path))
        on_disk = note_signatures(saved)
        if replayed != on_disk:
            raise ValueError(_signature_mismatch(filename, replayed, on_disk))


def _serialized_signatures(score, filename: str) -> list[dict]:
    """Mirror MusicXML's measure layout before comparing with a saved score."""
    with tempfile.TemporaryDirectory(prefix="synth-note-map-validate-") as tmp:
        path = Path(tmp) / filename
        write_musicxml(copy.deepcopy(score), path)
        return note_signatures(converter.parse(str(path)))


def backfill_bundle(
    bundle: Path,
    output_root: Path,
    config: SynthConfig,
    config_path: Path,
    force: bool = False,
) -> str:
    destination = output_root / bundle.name / "note_map.json"
    if destination.exists() and not force:
        return "skipped"
    metadata_path = bundle / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    clean, performed, replay_meta = replay_scores(metadata, config)
    validate_replay(bundle, clean, performed)
    payload = build_note_map(clean, performed)
    midi_path = bundle / "performance_audio.mid"
    if not midi_path.exists():
        raise FileNotFoundError(midi_path)
    from synthpipeline.pitch_convention import midi_to_written_shift, sounding_transpose

    attach_rendered_events(
        payload,
        midi_path,
        sounding_transpose=sounding_transpose(metadata),
        performed_score_path=bundle / "performance_score.musicxml",
        midi_to_written_shift=midi_to_written_shift(metadata),
    )
    payload["replay_validation"] = {
        "verified_score": "matched",
        "performance_score": "matched",
        "metadata_seed": int(metadata["seed"]),
        "metadata_index": int(metadata["index"]),
        "metadata_source": metadata.get("source"),
        "config": str(config_path.resolve()),
        **replay_meta,
    }
    write_note_map(destination, payload)
    return "written"


def bundle_directories(root: Path) -> list[Path]:
    return sorted(
        {
            metadata.parent
            for metadata in root.rglob("metadata.json")
            if (metadata.parent / "verified_score.musicxml").is_file()
            and (metadata.parent / "performance_score.musicxml").is_file()
        }
    )


def _backfill_job(args: tuple[str, str, str, bool]) -> tuple[str, str, str | None]:
    bundle_text, output_text, config_text, force = args
    bundle = Path(bundle_text)
    try:
        config_path = Path(config_text)
        config = SynthConfig.load(config_path)
        status = backfill_bundle(
            bundle,
            Path(output_text),
            config,
            config_path,
            force=force,
        )
        return bundle.name, status, None
    except Exception as exc:  # noqa: BLE001
        return bundle.name, "failed", str(exc)


def _signature_mismatch(
    filename: str,
    replayed: list[dict],
    on_disk: list[dict],
) -> str:
    limit = min(len(replayed), len(on_disk))
    for index in range(limit):
        if replayed[index] != on_disk[index]:
            return (
                f"{filename} signature mismatch at sounding note {index}: "
                f"replay={replayed[index]!r}, disk={on_disk[index]!r}"
            )
    return (
        f"{filename} sounding-note count mismatch: "
        f"replay={len(replayed)}, disk={len(on_disk)}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Existing bundle root")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Cache root; writes <output-root>/<bundle>/note_map.json only",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Generation YAML (multi_error_10k or rawdata_snippets_2k)",
    )
    parser.add_argument("--force", action="store_true", help="Replace note-map caches only")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Report rejected replays but return success so training can use validated maps",
    )
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    bundles = bundle_directories(args.root.resolve())
    if args.max_samples:
        bundles = bundles[: max(0, args.max_samples)]
    counts = {"written": 0, "skipped": 0, "failed": 0}
    jobs = [
        (
            str(bundle),
            str(args.output_root.resolve()),
            str(config_path),
            bool(args.force),
        )
        for bundle in bundles
    ]
    workers = max(1, int(args.workers))
    if workers == 1:
        results = map(_backfill_job, jobs)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(_backfill_job, jobs, chunksize=4)
    try:
        for i, (sample, status, error) in enumerate(results, start=1):
            counts[status] += 1
            if error:
                print(f"failed: {sample}: {error}", file=sys.stderr)
            if i == 1 or i % 100 == 0 or i == len(jobs):
                print(
                    f"{i}/{len(jobs)} written={counts['written']} "
                    f"skipped={counts['skipped']} failed={counts['failed']}",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    print(
        f"bundles={len(bundles)} written={counts['written']} "
        f"skipped={counts['skipped']} failed={counts['failed']}"
    )
    return 0 if counts["failed"] == 0 or args.allow_failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
