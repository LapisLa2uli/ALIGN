from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from datacreate.melody import (
    ScoreSoundingNote,
    extra_neighbor_core,
    is_repeated_pass,
    label_already_converted,
    midi_from_comment,
    notes_for_measure_pitch,
    notes_in_measures,
    notes_overlapping_time,
    padded_melody,
    parse_sounding_notes,
)
from datacreate.utils import write_json

SCHEMA_VERSION = "1.2"


def convert_bundle(
    sample_dir: Path,
    pad: int = 2,
    *,
    force: bool = False,
) -> str:
    labels_path = sample_dir / "labels.json"
    score_path = sample_dir / "verified_score.musicxml"
    if not labels_path.exists() or not score_path.exists():
        return "skip_missing"
    doc = json.loads(labels_path.read_text(encoding="utf-8"))
    labels = doc.get("labels") or []
    if not labels:
        doc["schema_version"] = SCHEMA_VERSION
        write_json(labels_path, doc)
        return "empty"
    if not force and all(label_already_converted(lab) or is_repeated_pass(lab) for lab in labels):
        if any(label_already_converted(lab) for lab in labels):
            return "skip_done"
    notes = parse_sounding_notes(score_path)
    if not notes:
        return "failed"
    use_pad = _existing_pad(labels)
    if use_pad is None:
        use_pad = pad
    changed = False
    mapped = 0
    failed_labels = 0
    for lab in labels:
        if is_repeated_pass(lab):
            continue
        if not force and label_already_converted(lab):
            continue
        core = _core_indices(lab, notes)
        if core is None:
            failed_labels += 1
            continue
        if lab.get("type") == "extra_note":
            core = extra_neighbor_core(notes, core[0])
        span = padded_melody(notes, core[0], core[1], use_pad)
        lab.update(span.as_fields())
        if lab.get("type") == "repetition":
            lab["extra_copies"] = int(lab.get("extra_copies") or 1)
        changed = True
        mapped += 1
    if changed or doc.get("schema_version") != SCHEMA_VERSION:
        doc["schema_version"] = SCHEMA_VERSION
        write_json(labels_path, doc)
        if failed_labels and not mapped:
            return "failed"
        return "converted"
    if failed_labels:
        return "failed"
    return "skip_done"


def _existing_pad(labels: list[dict[str, Any]]) -> int | None:
    for lab in labels:
        part = lab.get("score_part")
        if isinstance(part, dict) and part.get("pad_notes") is not None:
            return max(0, int(part["pad_notes"]))
    return None


def _core_indices(
    lab: dict[str, Any], notes: list[ScoreSoundingNote]
) -> tuple[int, int] | None:
    kind = lab.get("type")
    if kind == "extra_note":
        t0 = lab.get("start_time")
        t1 = lab.get("end_time")
        if t0 is not None and t1 is not None:
            return notes_overlapping_time(notes, float(t0), float(t1))
        measure = lab.get("measure_number")
        if measure is not None:
            return notes_in_measures(notes, [int(measure)])
        return None
    if kind == "repetition":
        src = lab.get("repeats_label_range") or {}
        t0 = src.get("start_time")
        t1 = src.get("end_time")
        if t0 is not None and t1 is not None:
            hit = notes_overlapping_time(notes, float(t0), float(t1))
            if hit is not None:
                return hit
        measure = lab.get("measure_number")
        if measure is not None:
            block = notes_in_measures(notes, [int(measure)])
            if block is not None:
                return block
    measure = lab.get("measure_number")
    midis = midi_from_comment(lab.get("comment"))
    pitch = midis[0] if midis else None
    if measure is not None:
        hit = notes_for_measure_pitch(notes, int(measure), pitch)
        if hit is not None:
            if midis and len(midis) > 1:
                last = notes_for_measure_pitch(notes, int(measure), midis[-1])
                if last is not None:
                    return hit[0], last[1]
            return hit
    t0 = lab.get("start_time")
    t1 = lab.get("end_time")
    if t0 is None or t1 is None:
        return None
    return notes_overlapping_time(notes, float(t0), float(t1))


def discover_bundles(root: Path) -> list[Path]:
    out: list[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "labels.json").exists() and (path / "verified_score.musicxml").exists():
            out.append(path)
            continue
        for child in sorted(path.iterdir()):
            if (
                child.is_dir()
                and (child / "labels.json").exists()
                and (child / "verified_score.musicxml").exists()
            ):
                out.append(child)
    return out


def _convert_one(payload: tuple[str, int, bool]) -> str:
    sample, pad, force = payload
    try:
        return convert_bundle(Path(sample), pad=pad, force=force)
    except Exception:
        return "failed"


def convert_root(
    root: Path,
    pad: int = 2,
    *,
    force: bool = False,
    pad_random: bool = False,
    seed: int = 365,
    workers: int = 1,
) -> dict[str, int]:
    import random

    rng = random.Random(seed)
    counts = {
        "converted": 0,
        "skip_done": 0,
        "skip_missing": 0,
        "empty": 0,
        "failed": 0,
        "n_bundles": 0,
    }
    dirs = discover_bundles(root)
    counts["n_bundles"] = len(dirs)
    jobs = [
        (str(sample), rng.choice((1, 2)) if pad_random else pad, force) for sample in dirs
    ]
    workers = max(1, int(workers))
    if workers == 1:
        statuses = [_convert_one(job) for job in jobs]
    else:
        statuses = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_convert_one, job) for job in jobs]
            for i, fut in enumerate(as_completed(futs), start=1):
                statuses.append(fut.result())
                if i % 250 == 0 or i == len(futs):
                    print(f"  {root}: {i}/{len(futs)}", flush=True)
    for status in statuses:
        counts[status] = counts.get(status, 0) + 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Add score-part melody fields to existing synth labels.json files"
    )
    parser.add_argument(
        "--root",
        type=Path,
        action="append",
        default=None,
        help="Root of synth sample bundles (repeatable)",
    )
    parser.add_argument("--pad", type=int, default=2, help="Notes of padding on each side")
    parser.add_argument(
        "--pad-random",
        action="store_true",
        help="Pick pad in {1, 2} per bundle instead of --pad",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing score_part fields")
    parser.add_argument("--seed", type=int, default=365)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    roots = [p.resolve() for p in (args.root or [Path("synth-pipeline/output")])]
    total = {
        "converted": 0,
        "skip_done": 0,
        "skip_missing": 0,
        "empty": 0,
        "failed": 0,
        "n_bundles": 0,
    }
    for root in roots:
        if not root.exists():
            print(f"error: {root} does not exist")
            return 1
        counts = convert_root(
            root,
            pad=args.pad,
            force=bool(args.force),
            pad_random=bool(args.pad_random),
            seed=args.seed,
            workers=args.workers,
        )
        print(
            f"root={root} bundles={counts['n_bundles']} converted={counts['converted']} "
            f"skip_done={counts['skip_done']} skip_missing={counts['skip_missing']} "
            f"empty={counts['empty']} failed={counts['failed']}"
        )
        for key in total:
            total[key] += counts.get(key, 0)
    if len(roots) > 1:
        print(
            f"total bundles={total['n_bundles']} converted={total['converted']} "
            f"skip_done={total['skip_done']} skip_missing={total['skip_missing']} "
            f"empty={total['empty']} failed={total['failed']}"
        )
    return 0 if total["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
