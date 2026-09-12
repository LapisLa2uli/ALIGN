"""Stage only transcriber training inputs from removable storage to local disk."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REQUIRED_FILES = (
    "performance_mel.npy",
    "performance_audio.mid",
    "performance_audio.wav",
    "metadata.json",
    "labels.json",
    "note_map.json",
    "performance_f0.npy",
)


def _stage_row(row: dict, cache_root: Path, *, compute_f0: bool = False) -> dict:
    source = Path(row["sample_dir"])
    corpus = str(row.get("corpus", row.get("root", source.parent.name)))
    destination = cache_root / corpus / source.name
    destination.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES:
        source_file = source / name
        if not source_file.exists():
            if name in {
                "performance_audio.mid",
                "metadata.json",
                "labels.json",
                "performance_f0.npy",
            }:
                continue
            raise FileNotFoundError(source_file)
        destination_file = destination / name
        if (
            destination_file.exists()
            and destination_file.stat().st_size == source_file.stat().st_size
        ):
            continue
        temporary = destination_file.with_name(
            f"{destination_file.name}.{os.getpid()}.tmp"
        )
        shutil.copy2(source_file, temporary)
        temporary.replace(destination_file)
    wav = source / "performance_audio.wav"
    if (
        compute_f0
        and wav.exists()
        and not (destination / "performance_f0.npy").exists()
    ):
        try:
            from alignmodel.transcription.data import write_fine_pitch_feature

            write_fine_pitch_feature(source)
            f0_src = source / "performance_f0.npy"
            if f0_src.exists():
                shutil.copy2(f0_src, destination / "performance_f0.npy")
        except Exception:
            pass
    staged = dict(row)
    staged["sample_dir"] = str(destination.resolve())
    staged.setdefault("recording_kind", "synthetic")
    return staged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--compute-f0",
        action="store_true",
        help="Derive performance_f0.npy from WAVs when the cache is missing",
    )
    args = parser.parse_args()

    document = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.cache_root.mkdir(parents=True, exist_ok=True)
    output = {
        "version": document.get("version", 1),
        "source_manifest": str(args.manifest.resolve()),
        "roots": document.get("roots", {}),
    }
    for split in ("train", "val"):
        rows = list(document.get(split, []))
        staged_rows: list[dict] = []
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for index, staged in enumerate(
                pool.map(
                    lambda row: _stage_row(
                        row, args.cache_root, compute_f0=args.compute_f0
                    ),
                    rows,
                ),
                1,
            ):
                staged_rows.append(staged)
                if index % 250 == 0 or index == len(rows):
                    print(f"{split} {index}/{len(rows)}", flush=True)
        output[split] = staged_rows

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary_out = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary_out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    temporary_out.replace(args.out)
    print(f"staged manifest: {args.out}", flush=True)


if __name__ == "__main__":
    main()
