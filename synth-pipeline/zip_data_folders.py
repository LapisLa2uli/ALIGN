"""Zip each sample folder in 1000dataexport into a sibling output directory."""

from __future__ import annotations

import argparse
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = SCRIPT_DIR / "1000dataexport"
DEFAULT_DEST = SCRIPT_DIR / "1000dataexport_zips"


def zip_folder(folder: Path, zip_path: Path, overwrite: bool) -> str:
    if zip_path.exists() and not overwrite:
        return f"skip {folder.name} (already exists)"

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = zip_path.with_suffix(zip_path.suffix + ".partial")
    try:
        with zipfile.ZipFile(
            tmp_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(folder.parent).as_posix())
        tmp_path.replace(zip_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return f"ok   {folder.name} -> {zip_path.name}"


def _zip_folder_job(args: tuple[str, str, bool]) -> str:
    folder, zip_path, overwrite = args
    return zip_folder(Path(folder), Path(zip_path), overwrite)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compress each data folder in 1000dataexport into individual zip files."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Folder containing sample directories (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Folder to write zip files into (default: {DEFAULT_DEST})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace zip files that already exist in the destination folder.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of folders to zip in parallel (default: 1).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    dest = args.dest.resolve()

    if not source.is_dir():
        print(f"Source folder not found: {source}", file=sys.stderr)
        return 1

    dest.mkdir(parents=True, exist_ok=True)
    folders = sorted(
        path
        for path in source.iterdir()
        if path.is_dir() and path.resolve() != dest
    )
    if not folders:
        print(f"No data folders found in {source}")
        return 0

    jobs = [
        (str(folder), str(dest / f"{folder.name}.zip"), args.overwrite)
        for folder in folders
    ]
    print(f"Zipping {len(jobs)} folders from {source}")
    print(f"Output: {dest}")

    completed = 0
    if args.workers <= 1:
        for folder, zip_path, overwrite in jobs:
            print(zip_folder(Path(folder), Path(zip_path), overwrite), flush=True)
            completed += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_zip_folder_job, job) for job in jobs]
            for future in as_completed(futures):
                print(future.result(), flush=True)
                completed += 1
                if completed % 50 == 0 or completed == len(jobs):
                    print(f"Progress: {completed}/{len(jobs)}", flush=True)

    print(f"Done. {completed} zip files in {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
