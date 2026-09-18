"""Cache score-free Basic Pitch maps for ORN train and calibration only."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from alignmodel.joint.packed_data import sha256_file
from alignmodel.training_resources import resource_lease
from alignmodel.transcription.basic_pitch import (
    basic_pitch_cache_path,
    extract_sample_basic_pitch_features,
)


EXPECTED_RELEASE_SHA256 = (
    "d4baeb90389136fbdd4549cdfb40fb2aceb2a97d899ae903bebffa2df6aefbc0"
)


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
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    args = parser.parse_args(argv)
    if sha256_file(args.release_manifest) != EXPECTED_RELEASE_SHA256:
        raise ValueError("Frozen ORN release mismatch")
    release = json.loads(args.release_manifest.read_text(encoding="utf-8"))
    if (
        args.release_manifest.parent / "lockbox" / "LOCKBOX_OPENED.json"
    ).exists():
        raise ValueError("Replacement lockbox is not sealed")
    rows = [
        {**row, "release_split": split}
        for split in ("train", "calibration")
        for row in release["splits"]["development"][split]
    ]
    artifacts = []
    with resource_lease(
        args.resource_status,
        "gpu",
        track="orn-basic-pitch-v1-cache",
        command=[str(Path(__file__).resolve()), *map(str, vars(args).values())],
        metadata={
            "train_rows": 379,
            "calibration_rows": 53,
            "locked_test": False,
        },
    ):
        for position, row in enumerate(rows, 1):
            path = basic_pitch_cache_path(
                args.cache_root,
                row["sample_dir"],
                f"orn-{row['release_split']}",
            )
            features = extract_sample_basic_pitch_features(
                row["sample_dir"], cache_path=path
            )
            artifacts.append(
                {
                    "sample": row["sample"],
                    "split": row["release_split"],
                    "audio_sha256": row["source_hashes"][
                        "performance_audio.wav"
                    ],
                    "cache": str(path.resolve()),
                    "cache_sha256": sha256_file(path),
                    "frames": len(features.frame_times),
                }
            )
            if position == 1 or position % 20 == 0 or position == len(rows):
                print(f"cache={position}/{len(rows)}", flush=True)
    _atomic_json(
        args.output,
        {
            "schema_version": "align-orn-basic-pitch-cache-v1",
            "release_manifest_sha256": sha256_file(args.release_manifest),
            "rows": len(artifacts),
            "split_counts": {
                split: sum(row["split"] == split for row in artifacts)
                for split in ("train", "calibration")
            },
            "artifacts": artifacts,
            "score_input": False,
            "open_validation_read": False,
            "lockbox_targets_read": False,
        },
    )
    print(
        json.dumps(
            {
                "rows": len(artifacts),
                "output": str(args.output.resolve()),
                "output_sha256": sha256_file(args.output),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
