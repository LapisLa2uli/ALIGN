"""Immutable sharded storage and deterministic loading for joint training.

The packed format keeps variable-length Basic Pitch maps in read-only binary
shards and small sparse targets in a SQLite index.  It deliberately has no
API for materializing a protected test split.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import sqlite3
import tempfile
import zlib
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from alignmodel.transcription.basic_pitch import BasicPitchFeatures

from .data import JointTrainingExample
from .index import JointEvent, ScoreEvent
from .lattice import JointCandidate


PACK_SCHEMA_VERSION = "align-outputraw-packed-v1"
TARGET_SCHEMA_VERSION = "align-full-joint-target-v1"
CURSOR_SCHEMA_VERSION = "align-packed-cursor-v1"
_ARRAYS = {
    "note": ("float32", 88),
    "onset": ("float32", 88),
    "contour": ("float32", 264),
    "frame_times": ("float64", 1),
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def packed_record_key(
    row: Mapping[str, Any],
    feature_metadata: Mapping[str, Any],
    *,
    candidate_version: str,
    target_version: str = TARGET_SCHEMA_VERSION,
) -> str:
    """Content address a record by source, frontend, and target policy."""

    payload = {
        "pack_schema": PACK_SCHEMA_VERSION,
        "target_schema": target_version,
        "candidate_version": candidate_version,
        "source_hashes": row.get("source_hashes"),
        "clean_fingerprint": row.get("clean_fingerprint"),
        "leakage_group": row.get("leakage_group"),
        "feature_metadata": dict(feature_metadata),
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def validate_split_manifest(document: Mapping[str, Any]) -> None:
    """Fail closed on protected-test materialization or fingerprint leakage."""

    split_rows = {
        name: list(document.get(name) or [])
        for name in ("train", "val", "test")
    }
    memberships: dict[str, set[str]] = {}
    for split, rows in split_rows.items():
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"{split} contains a non-object row")
            for field in (
                "leakage_group",
                "clean_fingerprint",
                "near_lineage_hash",
                "audio_lineage_hash",
            ):
                value = row.get(field)
                if value:
                    memberships.setdefault(f"{field}:{value}", set()).add(split)
            if split == "test":
                forbidden = {
                    "packed_record_key",
                    "target_record",
                    "target_db",
                    "target_cache",
                    "feature_cache",
                }
                present = sorted(field for field in forbidden if row.get(field) is not None)
                if present:
                    raise ValueError(
                        "Protected test row is materialized: "
                        f"{row.get('sample')} has {present}"
                    )
    leaked = {
        key: sorted(splits)
        for key, splits in memberships.items()
        if len(splits) > 1
    }
    if leaked:
        examples = dict(list(sorted(leaked.items()))[:10])
        raise ValueError(f"Split fingerprint leakage detected: {examples}")


def _event_dict(event: JointEvent) -> dict[str, Any]:
    return {
        "pitch": int(event.pitch),
        "start": float(event.start),
        "end": float(event.end),
        "score_span": list(event.score_span) if event.score_span else None,
        "relationship": event.relationship,
        "copy_pass": int(event.copy_pass),
        "origin_relationship": event.origin_relationship,
        "rendered_index": event.rendered_index,
        "source_indices": list(event.source_indices),
    }


def _score_dict(event: ScoreEvent) -> dict[str, Any]:
    return {
        "index": int(event.index),
        "pitch": int(event.pitch),
        "ql_start": float(event.ql_start),
        "ql_end": float(event.ql_end),
        "source_indices": list(event.source_indices),
        "measure": event.measure,
        "tie_chain": len(event.source_indices) > 1,
    }


def _candidate_dict(candidate: JointCandidate) -> dict[str, Any]:
    return {
        "pitch": int(candidate.pitch),
        "start": float(candidate.start),
        "end": float(candidate.end),
        "confidence": float(candidate.confidence),
        "score_hints": list(candidate.score_hints),
        "acoustic_features": list(candidate.acoustic_features),
    }


def canonical_target(
    example: JointTrainingExample,
    *,
    valid_labels: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Project exact lineage into the complete four-part supervision schema."""

    events = list(example.target_events)
    score = list(example.score)
    rhythm_labels = [
        row for row in valid_labels if str(row.get("type")) == "rhythm_error"
    ]
    repeats: list[dict[str, Any]] = []
    position = 0
    while position < len(events):
        if not events[position].is_copy:
            position += 1
            continue
        start = position
        copy_pass = int(events[position].copy_pass)
        while (
            position < len(events)
            and events[position].is_copy
            and int(events[position].copy_pass) == copy_pass
        ):
            position += 1
        copied = events[start:position]
        spans = [value.score_span for value in copied if value.score_span is not None]
        source_span = (
            [min(value[0] for value in spans), max(value[1] for value in spans)]
            if spans
            else None
        )
        following = next(
            (
                value.score_span[0]
                for value in events[position:]
                if not value.is_copy and value.score_span is not None
            ),
            source_span[1] if source_span is not None else -1,
        )
        repeats.append(
            {
                "source_span": source_span,
                "rendered_event_span": [start, position],
                "resume_event": int(following),
                "copy_pass": copy_pass,
                "copy_count": 0,
            }
        )
    by_source: dict[tuple[int, int], int] = {}
    for row in repeats:
        key = tuple(row["source_span"] or (-1, -1))
        by_source[key] = max(by_source.get(key, 0), int(row["copy_pass"]))
    for row in repeats:
        row["copy_count"] = by_source[tuple(row["source_span"] or (-1, -1))]

    operations: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        operation = (
            "extra"
            if event.score_span is None or event.relationship == "extra"
            else {
                "match": "match",
                "substitute": "wrong",
                "copy": "match",
            }[event.relationship]
        )
        operations.append(
            {
                "operation": operation,
                "rendered_event": index,
                "score_span": list(event.score_span) if event.score_span else None,
                "copy": event.is_copy,
            }
        )
    operations.extend(
        {
            "operation": "missed",
            "rendered_event": None,
            "score_span": [int(index), int(index) + 1],
            "copy": False,
        }
        for index in sorted(example.target_deletions)
    )

    rhythm: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        overlaps_error = any(
            float(label["start_time"]) < event.end
            and event.start < float(label["end_time"])
            for label in rhythm_labels
        )
        expected_ql = None
        if event.score_span is not None:
            first, last = event.score_span[0], event.score_span[1] - 1
            expected_ql = float(score[last].ql_end - score[first].ql_start)
        rhythm.append(
            {
                "rendered_event": index,
                "start_sec": float(event.start),
                "end_sec": float(event.end),
                "duration_sec": float(event.end - event.start),
                "score_duration_ql": expected_ql,
                "rhythm_error": bool(overlaps_error),
                "supervision_mask": event.score_span is not None,
            }
        )

    return {
        "schema_version": TARGET_SCHEMA_VERSION,
        "sample": example.sample,
        "score": [_score_dict(value) for value in score],
        "transcription": [
            {
                "pitch": int(value.pitch),
                "start_sec": float(value.start),
                "end_sec": float(value.end),
            }
            for value in events
        ],
        "layer1_repeats": repeats,
        "layer2_operations": operations,
        "layer3_rhythm": rhythm,
        "target_events": [_event_dict(value) for value in events],
        "target_deletions": sorted(int(value) for value in example.target_deletions),
        "masks": {
            "intonation": False,
            "intonation_reason": "historical_outputRaw_intonation_unreliable",
        },
    }


@dataclass(frozen=True)
class PackedCursor:
    pack_id: str
    split: str
    epoch: int
    seed: int
    position: int

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": CURSOR_SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PackedCursor":
        if value.get("schema_version") != CURSOR_SCHEMA_VERSION:
            raise ValueError("Unsupported packed cursor schema")
        return cls(
            pack_id=str(value["pack_id"]),
            split=str(value["split"]),
            epoch=int(value["epoch"]),
            seed=int(value["seed"]),
            position=int(value["position"]),
        )


@dataclass(frozen=True)
class PackedSample:
    ordinal: int
    record_key: str
    split: str
    sample: str
    source: str
    features: BasicPitchFeatures | None
    candidates: tuple[JointCandidate, ...]
    target: Mapping[str, Any]

    def training_example(self) -> JointTrainingExample:
        score = tuple(
            ScoreEvent(
                index=int(row["index"]),
                pitch=int(row["pitch"]),
                ql_start=float(row["ql_start"]),
                ql_end=float(row["ql_end"]),
                source_indices=tuple(int(value) for value in row["source_indices"]),
                measure=row.get("measure"),
            )
            for row in self.target["score"]
        )
        target_events = tuple(
            JointEvent(
                pitch=int(row["pitch"]),
                start=float(row["start"]),
                end=float(row["end"]),
                score_span=(
                    tuple(int(value) for value in row["score_span"])
                    if row.get("score_span")
                    else None
                ),
                relationship=str(row["relationship"]),
                copy_pass=int(row.get("copy_pass") or 0),
                origin_relationship=row.get("origin_relationship"),
                rendered_index=row.get("rendered_index"),
                source_indices=tuple(int(value) for value in row.get("source_indices") or []),
            )
            for row in self.target["target_events"]
        )
        gold_spans = tuple(
            (
                tuple(int(value) for value in value)
                if value is not None
                else None
            )
            for value in self.target["sparse_lattice"]["gold_spans"]
        )
        return JointTrainingExample(
            sample=self.sample,
            source=self.source,
            candidates=self.candidates,
            score=score,
            gold_spans=gold_spans,
            gold_keep_unlinked=tuple(
                bool(value)
                for value in self.target["sparse_lattice"]["gold_keep_unlinked"]
            ),
            target_events=target_events,
            target_deletions=frozenset(
                int(value) for value in self.target["target_deletions"]
            ),
        )


class ExistingShardMismatch(ValueError):
    """Raised when crash-staged bytes do not match the deterministic source."""


class PackedJointWriter:
    """Streaming writer that publishes only after all hashes are frozen."""

    def __init__(
        self,
        destination: Path,
        *,
        manifest_sha256: str,
        candidate_version: str,
        shard_rows: int = 128,
        staging: Path | None = None,
        checkpoint_rows: int = 8,
    ) -> None:
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.destination = destination.resolve()
        self.staging = (
            Path(staging).resolve()
            if staging is not None
            else Path(
                tempfile.mkdtemp(
                    prefix=f".{destination.name}.",
                    dir=destination.parent,
                )
            )
        )
        if not self.staging.is_dir():
            raise FileNotFoundError(self.staging)
        self.manifest_sha256 = manifest_sha256
        self.candidate_version = candidate_version
        self.shard_rows = max(1, int(shard_rows))
        self.checkpoint_rows = max(1, int(checkpoint_rows))
        self.index_path = self.staging / "index.sqlite"
        self.connection = sqlite3.connect(self.index_path)
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS records("
            "ordinal INTEGER PRIMARY KEY, record_key TEXT UNIQUE NOT NULL, "
            "split TEXT NOT NULL CHECK(split IN ('train','val')), "
            "sample TEXT NOT NULL, source TEXT NOT NULL, leakage_group TEXT NOT NULL, "
            "shard INTEGER NOT NULL, frame_offset INTEGER NOT NULL, "
            "frame_count INTEGER NOT NULL, source_hashes TEXT NOT NULL, "
            "feature_metadata TEXT NOT NULL, candidates BLOB NOT NULL, "
            "target BLOB NOT NULL, record_sha256 TEXT NOT NULL)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS records_split_ordinal "
            "ON records(split, ordinal)"
        )
        self.connection.commit()
        saved = self.connection.execute(
            "SELECT COUNT(*), MIN(ordinal), MAX(ordinal) FROM records"
        ).fetchone()
        self._ordinal = int(saved[0])
        if self._ordinal and (
            int(saved[1]) != 0 or int(saved[2]) != self._ordinal - 1
        ):
            raise ValueError("Crash-staged packed index is not contiguous")
        self._shard = (
            (self._ordinal - 1) // self.shard_rows
            if self._ordinal
            else -1
        )
        self._shard_count = (
            self._ordinal - self._shard * self.shard_rows
            if self._ordinal
            else 0
        )
        self._frame_offset = (
            int(
                self.connection.execute(
                    "SELECT COALESCE(SUM(frame_count),0) FROM records "
                    "WHERE shard=?",
                    (self._shard,),
                ).fetchone()[0]
            )
            if self._ordinal
            else 0
        )
        self._handles: dict[str, Any] = {}
        self._closed = False

    def _open_shard(self) -> None:
        self._close_shard()
        self._shard += 1
        self._shard_count = 0
        self._frame_offset = 0
        for name, (dtype, _width) in _ARRAYS.items():
            path = self.staging / f"shard-{self._shard:05d}.{name}.{dtype}.bin"
            self._handles[name] = path.open("wb")

    def _close_shard(self) -> None:
        if not self._handles:
            return
        for stream in self._handles.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        self._handles.clear()

    @property
    def committed_records(self) -> int:
        return self._ordinal

    def validate_committed_prefix(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        if self._ordinal > len(rows):
            raise ValueError("Crash-staged index is longer than the manifest")
        saved = self.connection.execute(
            "SELECT ordinal,sample,split,leakage_group FROM records "
            "ORDER BY ordinal"
        )
        for ordinal, sample, split, leakage_group in saved:
            expected = rows[int(ordinal)]
            if (
                sample != expected["sample"]
                or split != expected["split"]
                or leakage_group != expected["leakage_group"]
            ):
                raise ValueError(
                    f"Crash-staged record {ordinal} does not match manifest"
                )

    @staticmethod
    def _record_arrays(features: BasicPitchFeatures) -> dict[str, np.ndarray]:
        frame_count = int(features.note.shape[0])
        arrays = {
            "note": np.asarray(features.note, dtype="<f4", order="C"),
            "onset": np.asarray(features.onset, dtype="<f4", order="C"),
            "contour": np.asarray(features.contour, dtype="<f4", order="C"),
            "frame_times": np.asarray(features.frame_times, dtype="<f8", order="C"),
        }
        if (
            arrays["note"].shape != (frame_count, 88)
            or arrays["onset"].shape != (frame_count, 88)
            or arrays["contour"].shape != (frame_count, 264)
            or arrays["frame_times"].shape != (frame_count,)
        ):
            raise ValueError("Invalid Basic Pitch array shapes")
        return arrays

    def _insert_record(
        self,
        row: Mapping[str, Any],
        features: BasicPitchFeatures,
        example: JointTrainingExample,
        target: Mapping[str, Any],
        arrays: Mapping[str, np.ndarray],
    ) -> str:
        frame_count = int(features.note.shape[0])
        target_value = dict(target)
        target_value["sparse_lattice"] = {
            "gold_spans": [
                list(value) if value is not None else None
                for value in example.gold_spans
            ],
            "gold_keep_unlinked": list(example.gold_keep_unlinked),
        }
        target_raw = _canonical_json(target_value)
        candidates_raw = _canonical_json(
            [_candidate_dict(value) for value in example.candidates]
        )
        key = packed_record_key(
            row,
            features.metadata,
            candidate_version=self.candidate_version,
        )
        digest = hashlib.sha256()
        digest.update(key.encode("ascii"))
        for name in _ARRAYS:
            digest.update(arrays[name].tobytes(order="C"))
        digest.update(candidates_raw)
        digest.update(target_raw)
        self.connection.execute(
            "INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                self._ordinal,
                key,
                row["split"],
                row["sample"],
                row["source"],
                row["leakage_group"],
                self._shard,
                self._frame_offset,
                frame_count,
                json.dumps(row["source_hashes"], sort_keys=True),
                json.dumps(dict(features.metadata), sort_keys=True),
                sqlite3.Binary(zlib.compress(candidates_raw, level=1)),
                sqlite3.Binary(zlib.compress(target_raw, level=1)),
                digest.hexdigest(),
            ),
        )
        self._ordinal += 1
        self._shard_count += 1
        self._frame_offset += frame_count
        if self._ordinal % self.checkpoint_rows == 0:
            for stream in self._handles.values():
                stream.flush()
                os.fsync(stream.fileno())
            self.connection.commit()
        return key

    def reuse(
        self,
        row: Mapping[str, Any],
        features: BasicPitchFeatures,
        example: JointTrainingExample,
        target: Mapping[str, Any],
    ) -> str:
        """Validate and index one deterministic record from crash-staged bytes."""

        if self._handles:
            raise RuntimeError("Cannot reuse bytes after append mode starts")
        expected_shard = self._ordinal // self.shard_rows
        expected_count = self._ordinal % self.shard_rows
        if expected_shard != self._shard:
            self._shard = expected_shard
            self._shard_count = 0
            self._frame_offset = 0
        if self._shard_count != expected_count:
            raise ValueError("Crash recovery shard cursor is inconsistent")
        arrays = self._record_arrays(features)
        for name, (dtype, width) in _ARRAYS.items():
            path = self.staging / f"shard-{self._shard:05d}.{name}.{dtype}.bin"
            expected = arrays[name].tobytes(order="C")
            byte_offset = self._frame_offset * width * np.dtype(dtype).itemsize
            try:
                with path.open("rb") as stream:
                    stream.seek(byte_offset)
                    actual = stream.read(len(expected))
            except FileNotFoundError as exc:
                raise ExistingShardMismatch(path.name) from exc
            if actual != expected:
                raise ExistingShardMismatch(
                    f"Staged bytes diverge at record {self._ordinal}: {path.name}"
                )
        return self._insert_record(row, features, example, target, arrays)

    def begin_append(self) -> None:
        """Discard only unverified tail bytes, then resume durable appends."""

        if self._handles:
            return
        expected_shard = self._ordinal // self.shard_rows
        expected_count = self._ordinal % self.shard_rows
        if expected_shard != self._shard:
            self._shard = expected_shard
            self._shard_count = 0
            self._frame_offset = 0
        self._shard_count = expected_count
        for path in self.staging.glob("shard-*.bin"):
            try:
                shard = int(path.name.split(".", 1)[0].split("-")[1])
            except (IndexError, ValueError):
                continue
            if shard > self._shard:
                path.unlink()
        for name, (dtype, width) in _ARRAYS.items():
            path = self.staging / f"shard-{self._shard:05d}.{name}.{dtype}.bin"
            expected_bytes = self._frame_offset * width * np.dtype(dtype).itemsize
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a+b") as stream:
                stream.truncate(expected_bytes)
            self._handles[name] = path.open("ab")

    def add(
        self,
        row: Mapping[str, Any],
        features: BasicPitchFeatures,
        example: JointTrainingExample,
        target: Mapping[str, Any],
    ) -> str:
        if row.get("split") not in {"train", "val"}:
            raise ValueError("Packed writer refuses protected or unknown splits")
        if self._shard_count >= self.shard_rows or not self._handles:
            self._open_shard()
        arrays = self._record_arrays(features)
        if int(features.note.shape[0]) != int(arrays["note"].shape[0]):
            raise ValueError(f"Invalid Basic Pitch shapes for {row.get('sample')}")
        for name in _ARRAYS:
            arrays[name].tofile(self._handles[name])
        return self._insert_record(row, features, example, target, arrays)

    def finalize(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Packed writer is already closed")
        self._close_shard()
        self.connection.commit()
        self.connection.execute("PRAGMA optimize")
        self.connection.close()
        shard_files = []
        for path in sorted(self.staging.glob("shard-*.bin")):
            shard_files.append(
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        counts = {}
        connection = sqlite3.connect(
            f"file:{self.index_path.resolve().as_posix()}?mode=ro", uri=True
        )
        try:
            for split, count in connection.execute(
                "SELECT split, COUNT(*) FROM records GROUP BY split"
            ):
                counts[str(split)] = int(count)
        finally:
            connection.close()
        metadata = {
            "schema_version": PACK_SCHEMA_VERSION,
            "manifest_sha256": self.manifest_sha256,
            "candidate_version": self.candidate_version,
            "target_version": TARGET_SCHEMA_VERSION,
            "shard_rows": self.shard_rows,
            "record_count": self._ordinal,
            "split_counts": counts,
            "index": {
                "name": self.index_path.name,
                "size": self.index_path.stat().st_size,
                "sha256": sha256_file(self.index_path),
            },
            "shards": shard_files,
            "arrays": {
                name: {"dtype": dtype, "width": width}
                for name, (dtype, width) in _ARRAYS.items()
            },
        }
        metadata["pack_id"] = hashlib.sha256(_canonical_json(metadata)).hexdigest()
        (self.staging / "metadata.json").write_bytes(
            json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )
        os.replace(self.staging, self.destination)
        self._closed = True
        return metadata

    def abort(self) -> None:
        if self._closed:
            return
        self._close_shard()
        self.connection.close()
        shutil.rmtree(self.staging, ignore_errors=True)
        self._closed = True

    def preserve(self) -> None:
        """Durably close a staging directory for a later deterministic resume."""

        if self._closed:
            return
        self._close_shard()
        self.connection.commit()
        self.connection.close()
        self._closed = True

    def __enter__(self) -> "PackedJointWriter":
        return self

    def __exit__(self, exc_type: Any, _exc: Any, _tb: Any) -> None:
        if exc_type is not None:
            self.preserve()


class PackedJointDataset(Sequence[PackedSample]):
    """Read-only, lazy mmap dataset with resumable deterministic epochs.

    Local/path training consumes the already-canonical candidates and targets,
    not the large frontend activation maps.  ``load_feature_arrays=False``
    therefore provides a metadata-only training path after the release has
    passed deep verification, avoiding roughly 32 GB of irrelevant reads per
    epoch.
    """

    def __init__(
        self,
        root: Path,
        *,
        manifest_sha256: str | None = None,
        verify_records: bool = True,
        max_open_shards: int = 4,
        load_feature_arrays: bool = True,
    ) -> None:
        self.root = Path(root)
        metadata_path = self.root / "metadata.json"
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("schema_version") != PACK_SCHEMA_VERSION:
            raise ValueError("Unsupported packed dataset schema")
        calculated_pack_id = hashlib.sha256(
            _canonical_json(
                {
                    key: value
                    for key, value in self.metadata.items()
                    if key != "pack_id"
                }
            )
        ).hexdigest()
        if calculated_pack_id != self.metadata.get("pack_id"):
            raise ValueError("Packed metadata hash mismatch")
        if (
            manifest_sha256 is not None
            and manifest_sha256 != self.metadata.get("manifest_sha256")
        ):
            raise ValueError("Packed dataset belongs to a different manifest")
        self.verify_records = bool(verify_records)
        self.load_feature_arrays = bool(load_feature_arrays)
        if self.verify_records and not self.load_feature_arrays:
            raise ValueError(
                "Per-record verification requires frontend feature arrays; "
                "use metadata-only loading only after deep release verification"
            )
        self.max_open_shards = max(1, int(max_open_shards))
        index = self.metadata["index"]
        index_path = self.root / index["name"]
        if index_path.stat().st_size != int(index["size"]):
            raise ValueError("Packed SQLite index size mismatch")
        if sha256_file(index_path) != index["sha256"]:
            raise ValueError("Packed SQLite index hash mismatch")
        for row in self.metadata["shards"]:
            path = self.root / row["name"]
            if not path.is_file() or path.stat().st_size != int(row["size"]):
                raise ValueError(f"Packed shard size mismatch: {path.name}")
        self.connection = sqlite3.connect(
            f"file:{index_path.resolve().as_posix()}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        self.connection.execute("PRAGMA query_only=ON")
        self._maps: OrderedDict[tuple[int, str], np.memmap] = OrderedDict()

    def close(self) -> None:
        self._maps.clear()
        self.connection.close()

    def __enter__(self) -> "PackedJointDataset":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __len__(self) -> int:
        return int(self.metadata["record_count"])

    def _map(self, shard: int, name: str) -> np.memmap:
        key = (shard, name)
        saved = self._maps.pop(key, None)
        if saved is not None:
            self._maps[key] = saved
            return saved
        dtype, width = _ARRAYS[name]
        path = self.root / f"shard-{shard:05d}.{name}.{dtype}.bin"
        size = path.stat().st_size
        itemsize = np.dtype(dtype).itemsize
        if size % (itemsize * width):
            raise ValueError(f"Packed shard has invalid byte length: {path.name}")
        shape = (size // (itemsize * width), width) if width > 1 else (size // itemsize,)
        saved = np.memmap(path, dtype=dtype, mode="r", shape=shape)
        self._maps[key] = saved
        while len(self._maps) > self.max_open_shards * len(_ARRAYS):
            self._maps.popitem(last=False)
        return saved

    def _saved_row(self, ordinal: int) -> tuple[Any, ...]:
        row = self.connection.execute(
            "SELECT ordinal,record_key,split,sample,source,shard,"
            "frame_offset,frame_count,feature_metadata,candidates,target,"
            "record_sha256 FROM records WHERE ordinal=?",
            (int(ordinal),),
        ).fetchone()
        if row is None:
            raise IndexError(ordinal)
        return row

    def __getitem__(self, index: int | slice) -> PackedSample | list[PackedSample]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        ordinal = int(index)
        if ordinal < 0:
            ordinal += len(self)
        row = self._saved_row(ordinal)
        arrays = None
        if self.load_feature_arrays:
            shard, start, count = int(row[5]), int(row[6]), int(row[7])
            arrays = {
                name: np.asarray(self._map(shard, name)[start : start + count])
                for name in _ARRAYS
            }
        candidates_raw = zlib.decompress(row[9])
        target_raw = zlib.decompress(row[10])
        if self.verify_records:
            assert arrays is not None
            digest = hashlib.sha256()
            digest.update(str(row[1]).encode("ascii"))
            for name in _ARRAYS:
                digest.update(arrays[name].tobytes(order="C"))
            digest.update(candidates_raw)
            digest.update(target_raw)
            if digest.hexdigest() != row[11]:
                raise ValueError(f"Packed record checksum mismatch: {row[3]}")
        candidate_rows = json.loads(candidates_raw)
        candidates = tuple(
            JointCandidate(
                pitch=int(value["pitch"]),
                start=float(value["start"]),
                end=float(value["end"]),
                confidence=float(value["confidence"]),
                score_hints=tuple(int(item) for item in value["score_hints"]),
                acoustic_features=tuple(
                    float(item) for item in value["acoustic_features"]
                ),
            )
            for value in candidate_rows
        )
        return PackedSample(
            ordinal=int(row[0]),
            record_key=str(row[1]),
            split=str(row[2]),
            sample=str(row[3]),
            source=str(row[4]),
            features=(
                BasicPitchFeatures(
                    note=arrays["note"],
                    onset=arrays["onset"],
                    contour=arrays["contour"],
                    frame_times=arrays["frame_times"],
                    metadata=json.loads(row[8]),
                )
                if arrays is not None
                else None
            ),
            candidates=candidates,
            target=json.loads(target_raw),
        )

    def ordinals(self, split: str) -> list[int]:
        if split not in {"train", "val"}:
            raise ValueError("Packed loader exposes train/val only")
        return [
            int(row[0])
            for row in self.connection.execute(
                "SELECT ordinal FROM records WHERE split=? ORDER BY ordinal",
                (split,),
            )
        ]

    def deterministic_order(self, split: str, *, epoch: int, seed: int) -> list[int]:
        values = self.ordinals(split)
        random.Random(f"{self.metadata['pack_id']}:{seed}:{epoch}:{split}").shuffle(values)
        return values

    def cursor(
        self,
        split: str,
        *,
        epoch: int,
        seed: int,
        position: int = 0,
    ) -> PackedCursor:
        return PackedCursor(
            pack_id=str(self.metadata["pack_id"]),
            split=split,
            epoch=int(epoch),
            seed=int(seed),
            position=int(position),
        )

    def iter_from_cursor(
        self,
        cursor: PackedCursor,
        *,
        prefetch: int = 0,
        workers: int = 0,
    ) -> Iterator[tuple[PackedCursor, PackedSample]]:
        if cursor.pack_id != self.metadata["pack_id"]:
            raise ValueError("Resume cursor belongs to a different packed dataset")
        order = self.deterministic_order(
            cursor.split, epoch=cursor.epoch, seed=cursor.seed
        )
        if not 0 <= cursor.position <= len(order):
            raise ValueError("Resume cursor position is out of range")
        selected = order[cursor.position :]
        if workers <= 0 or prefetch <= 0:
            for offset, ordinal in enumerate(selected, cursor.position):
                yield (
                    PackedCursor(
                        cursor.pack_id,
                        cursor.split,
                        cursor.epoch,
                        cursor.seed,
                        offset + 1,
                    ),
                    self[ordinal],
                )
            return
        limit = max(1, int(prefetch))
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            pending: list[Future[PackedSample]] = []
            iterator = iter(selected)
            for _ in range(limit):
                try:
                    pending.append(pool.submit(self.__getitem__, next(iterator)))
                except StopIteration:
                    break
            offset = cursor.position
            while pending:
                future = pending.pop(0)
                sample = future.result()
                offset += 1
                yield (
                    PackedCursor(
                        cursor.pack_id,
                        cursor.split,
                        cursor.epoch,
                        cursor.seed,
                        offset,
                    ),
                    sample,
                )
                try:
                    pending.append(pool.submit(self.__getitem__, next(iterator)))
                except StopIteration:
                    pass

    def validate(self, *, deep: bool = False) -> dict[str, Any]:
        checked = 0
        if deep:
            for row in self.metadata["shards"]:
                path = self.root / row["name"]
                if sha256_file(path) != row["sha256"]:
                    raise ValueError(f"Packed shard hash mismatch: {path.name}")
            for ordinal in range(len(self)):
                self[ordinal]
                checked += 1
        counts = {
            split: len(self.ordinals(split)) for split in ("train", "val")
        }
        if sum(counts.values()) != len(self):
            raise ValueError("Packed split counts do not equal record count")
        return {
            "status": "ok",
            "pack_id": self.metadata["pack_id"],
            "records": len(self),
            "record_checks": checked,
            "split_counts": counts,
            "deep": bool(deep),
        }

