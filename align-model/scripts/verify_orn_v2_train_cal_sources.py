"""Verify immutable ORN v2 train/calibration source files without target reads."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from alignmodel.joint.packed_data import sha256_file
from alignmodel.training_resources import resource_lease


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--expected-release-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    actual_manifest_sha = sha256_file(args.release_manifest)
    if actual_manifest_sha != args.expected_release_sha256:
        raise ValueError("Frozen ORN v2 release manifest mismatch")
    release = json.loads(args.release_manifest.read_text(encoding="utf-8"))
    if (
        args.release_manifest.parent / "lockbox" / "LOCKBOX_OPENED.json"
    ).exists():
        raise ValueError("ORN v2 lockbox opening sentinel exists")
    rows = [
        row
        for split in ("train", "calibration")
        for row in release["splits"]["development"][split]
    ]
    if len(rows) != 965:
        raise ValueError(f"Expected 965 train/calibration rows, got {len(rows)}")
    jobs = []
    for row in rows:
        sample_dir = Path(row["sample_dir"])
        for name, expected in row["source_hashes"].items():
            jobs.append(
                {
                    "sample": row["sample"],
                    "split": row["split"],
                    "name": name,
                    "path": sample_dir / name,
                    "expected": expected,
                }
            )

    def verify(job: Mapping[str, Any]) -> dict[str, Any]:
        path = Path(job["path"])
        actual = sha256_file(path) if path.is_file() else None
        return {
            "sample": job["sample"],
            "split": job["split"],
            "name": job["name"],
            "path": str(path),
            "expected_sha256": job["expected"],
            "actual_sha256": actual,
            "passed": actual == job["expected"],
        }

    with resource_lease(
        args.resource_status,
        "cpu_support",
        track="orn-v2-train-cal-source-integrity",
        command=[str(Path(__file__).resolve()), *map(str, vars(args).values())],
        metadata={
            "rows": len(rows),
            "files": len(jobs),
            "locked_test": False,
        },
    ):
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            checks = list(executor.map(verify, jobs))
    failures = [row for row in checks if not row["passed"]]
    report = {
        "schema_version": "align-orn-v2-train-cal-source-integrity-v1",
        "release_manifest": str(args.release_manifest.resolve()),
        "release_manifest_sha256": actual_manifest_sha,
        "splits_read": ["train", "calibration"],
        "rows": len(rows),
        "split_counts": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "calibration")
        },
        "files_checked": len(checks),
        "failures": failures,
        "passed": not failures,
        "development_target_archive_read": False,
        "open_validation_source_or_targets_read": False,
        "v1_or_v2_lockbox_read": False,
    }
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "files_checked": len(checks),
                "failures": len(failures),
                "passed": not failures,
                "output": str(args.output.resolve()),
                "output_sha256": sha256_file(args.output),
            },
            indent=2,
        ),
        flush=True,
    )
    if failures:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
