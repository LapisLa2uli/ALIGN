"""Build a resumable train/calibration-only ORN mel cache."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from alignmodel.joint.packed_data import sha256_file
from alignmodel.training_resources import resource_lease
from alignmodel.transcription.mel_v1 import (
    CACHE_SCHEMA_VERSION,
    MelFrontendConfig,
    extract_log_mel,
    load_audio_mono,
)
from alignmodel.transcription.mel_v1_data import MelPackedCache


EXPECTED_RELEASE_SHA256 = (
    "d4baeb90389136fbdd4549cdfb40fb2aceb2a97d899ae903bebffa2df6aefbc0"
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


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


def _new_index(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS records("
        "ordinal INTEGER PRIMARY KEY, sample TEXT UNIQUE NOT NULL, "
        "split TEXT NOT NULL CHECK(split IN ('train','val','test')), "
        "source TEXT NOT NULL, shard INTEGER NOT NULL, "
        "frame_offset INTEGER NOT NULL, frame_count INTEGER NOT NULL, "
        "duration_sec REAL NOT NULL, audio_render TEXT NOT NULL, "
        "effective_audio_transpose INTEGER NOT NULL, "
        "audio_sha256 TEXT NOT NULL, mel_sha256 TEXT NOT NULL, "
        "normalization TEXT NOT NULL, target BLOB NOT NULL)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS records_split ON records(split,ordinal)"
    )
    connection.commit()
    return connection


def _targets(release: Mapping[str, Any]) -> dict[str, tuple[dict[str, Any], ...]]:
    artifact = release["artifacts"]["development_targets"]
    path = Path(artifact["path"])
    if sha256_file(path) != artifact["sha256"]:
        raise ValueError("Development target archive hash mismatch")
    output = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row["split"] not in {"train", "calibration"}:
                continue
            output[row["sample"]] = tuple(
                {
                    "pitch": int(event["pitch_midi_written"]),
                    "start_sec": float(event["start_sec"]),
                    "end_sec": float(event["end_sec"]),
                    "relationship": str(event.get("relationship") or "extra"),
                    "rendered_index": int(event["rendered_index"]),
                }
                for event in row["lineage"]["rendered_notes"]
            )
    return output


def build(args: argparse.Namespace) -> Path:
    release_path = args.release_manifest.resolve()
    if sha256_file(release_path) != EXPECTED_RELEASE_SHA256:
        raise ValueError("Frozen ORN release manifest mismatch")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if (release_path.parent / "lockbox" / "LOCKBOX_OPENED.json").exists():
        raise ValueError("Replacement lockbox is not sealed")
    destination = args.output.resolve()
    frontend = MelFrontendConfig(hop_length=256)
    if destination.exists():
        cache = MelPackedCache(destination, deep=False)
        if cache.frontend != frontend:
            raise ValueError("Existing ORN cache frontend mismatch")
        cache.close()
        return destination
    targets = _targets(release)
    selected = []
    for release_split, cache_split in (
        ("train", "train"),
        ("calibration", "val"),
    ):
        for row in release["splits"]["development"][release_split]:
            selected.append(
                {
                    **row,
                    "cache_split": cache_split,
                    "release_split": release_split,
                }
            )
    if set(targets) != {row["sample"] for row in selected}:
        raise ValueError("ORN cache target population mismatch")
    staging = destination.with_name(f".{destination.name}.building")
    staging.mkdir(parents=True, exist_ok=True)
    for path in staging.glob("*.tmp"):
        path.unlink(missing_ok=True)
    index_path = staging / "index.sqlite"
    connection = _new_index(index_path)
    committed = int(
        connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    )
    if committed % args.shard_rows:
        raise ValueError("Interrupted ORN cache is not on a shard boundary")
    started = time.perf_counter()
    try:
        for chunk_start in range(committed, len(selected), args.shard_rows):
            chunk = selected[chunk_start : chunk_start + args.shard_rows]
            shard = chunk_start // args.shard_rows
            arrays = []
            records = []
            frame_offset = 0
            for row in chunk:
                sample = row["sample"]
                audio_path = Path(row["sample_dir"]) / "performance_audio.wav"
                if (
                    sha256_file(audio_path)
                    != row["source_hashes"]["performance_audio.wav"]
                ):
                    raise ValueError(f"ORN audio hash mismatch: {sample}")
                audio = load_audio_mono(audio_path, frontend.sample_rate)
                mel, normalization = extract_log_mel(
                    audio, frontend, device=args.device
                )
                contiguous = np.ascontiguousarray(mel.T, dtype="<f2")
                raw = contiguous.tobytes(order="C")
                arrays.append(contiguous)
                records.append(
                    (
                        sample,
                        row["cache_split"],
                        row["provenance"],
                        shard,
                        frame_offset,
                        int(contiguous.shape[0]),
                        float(row["duration_sec"]),
                        "soundfont_v1_ornaments",
                        2,
                        row["source_hashes"]["performance_audio.wav"],
                        hashlib.sha256(raw).hexdigest(),
                        json.dumps(normalization, sort_keys=True),
                        sqlite3.Binary(
                            zlib.compress(_canonical(targets[sample]), level=1)
                        ),
                    )
                )
                frame_offset += int(contiguous.shape[0])
                done = chunk_start + len(records)
                if done == 1 or done % 20 == 0 or done == len(selected):
                    rate = done / max(time.perf_counter() - started, 1e-9)
                    print(
                        f"cache={done}/{len(selected)} "
                        f"rows_per_sec={rate:.2f}",
                        flush=True,
                    )
            shard_path = staging / f"shard-{shard:05d}.mel.float16.bin"
            temporary = shard_path.with_suffix(shard_path.suffix + ".tmp")
            with temporary.open("wb") as stream:
                for array in arrays:
                    stream.write(array.tobytes(order="C"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, shard_path)
            connection.executemany(
                "INSERT INTO records(sample,split,source,shard,frame_offset,"
                "frame_count,duration_sec,audio_render,effective_audio_transpose,"
                "audio_sha256,mel_sha256,normalization,target) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                records,
            )
            connection.commit()
    finally:
        connection.close()
    shards = [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(staging.glob("shard-*.bin"))
    ]
    metadata: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "frontend_config": frontend.to_dict(),
        "dtype": "float16",
        "layout": "time_major_contiguous",
        "shard_rows": args.shard_rows,
        "split_counts": {
            "train": sum(row["cache_split"] == "train" for row in selected),
            "val": sum(row["cache_split"] == "val" for row in selected),
        },
        "record_count": len(selected),
        "source": {
            "release": "orn-generalization-v1-development",
            "release_manifest": str(release_path),
            "release_manifest_sha256": sha256_file(release_path),
            "pack_id": sha256_file(release_path),
            "locked_test_materialized": False,
            "targets_included": True,
            "open_validation_materialized": False,
        },
        "index": {
            "name": index_path.name,
            "bytes": index_path.stat().st_size,
            "sha256": sha256_file(index_path),
        },
        "shards": shards,
    }
    metadata["pack_id"] = hashlib.sha256(_canonical(metadata)).hexdigest()
    _atomic_json(staging / "metadata.json", metadata)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, destination)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-manifest",
        type=Path,
        default=Path(
            "runs/joint-outputraw-full-v1/orn-generalization-v1/"
            "release_manifest.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/joint-outputraw-full-v1/orn-generalization-v1/"
            "development/ornament-multipitch-v1/mel-cache"
        ),
    )
    parser.add_argument(
        "--resource-status",
        type=Path,
        default=Path("runs/TRAINING_RESOURCE_STATUS.json"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-rows", type=int, default=32)
    args = parser.parse_args(argv)
    with resource_lease(
        args.resource_status,
        "gpu",
        track="ornament-multipitch-v1-cache",
        command=[str(Path(__file__).resolve()), *sys.argv[1:]],
        metadata={
            "split": "train+calibration",
            "locked_test": False,
        },
    ):
        output = build(args)
    cache = MelPackedCache(output, deep=True)
    print(
        json.dumps(
            {
                "cache": str(output),
                "pack_id": cache.pack_id,
                "split_counts": cache.metadata["split_counts"],
                "deep_verification": "passed",
            },
            indent=2,
        ),
        flush=True,
    )
    cache.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
