from __future__ import annotations

import hashlib
import json
import os
import pickle
import sqlite3
import tempfile
import time
import zlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence, overload

from .candidates import CANDIDATE_GENERATION_VERSION, HIGH_RECALL_DECODE_CONFIGS
from .data import JointTrainingExample, build_training_example


CACHE_SCHEMA_VERSION = "align-joint-example-cache-v2"


def example_cache_key(
    row: Mapping[str, Any],
    *,
    pairing_tolerance_sec: float,
    minimum_candidate_confidence: float,
) -> str:
    identity = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "candidate_generation": CANDIDATE_GENERATION_VERSION,
        "decode_configs": [asdict(value) for value in HIGH_RECALL_DECODE_CONFIGS],
        "sample_dir": os.path.normcase(os.path.abspath(str(row["sample_dir"]))),
        "corpus": row.get("corpus") or row.get("root"),
        "source_hashes": row.get("source_hashes"),
        "target_db": os.path.normcase(
            os.path.abspath(str(row.get("target_db")))
        ),
        "target_record": row.get("target_record"),
        "pairing_tolerance_sec": float(pairing_tolerance_sec),
        "minimum_candidate_confidence": float(minimum_candidate_confidence),
    }
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _serialize(example: JointTrainingExample) -> bytes:
    return zlib.compress(
        pickle.dumps(example, protocol=pickle.HIGHEST_PROTOCOL),
        level=1,
    )


def _deserialize(payload: bytes) -> JointTrainingExample:
    value = pickle.loads(zlib.decompress(payload))
    if not isinstance(value, JointTrainingExample):
        raise TypeError("Cached payload is not a JointTrainingExample")
    return value


def _build_task(
    task: tuple[
        str,
        Mapping[str, Any],
        str,
        float,
        float,
    ],
) -> tuple[str, bytes, int, int, float]:
    key, row, cache_root, tolerance, confidence = task
    started = time.perf_counter()
    example = build_training_example(
        row,
        Path(cache_root),
        pairing_tolerance_sec=tolerance,
        minimum_candidate_confidence=confidence,
    )
    return (
        key,
        _serialize(example),
        len(example.candidates),
        len(example.score),
        time.perf_counter() - started,
    )


class JointExampleCache:
    """Consolidated immutable cache for deterministic training examples."""

    def __init__(
        self,
        path: Path,
        *,
        pairing_tolerance_sec: float,
        minimum_candidate_confidence: float,
    ) -> None:
        self.path = path
        self.pairing_tolerance_sec = float(pairing_tolerance_sec)
        self.minimum_candidate_confidence = float(
            minimum_candidate_confidence
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path), timeout=60.0)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS examples ("
            "cache_key TEXT PRIMARY KEY, payload BLOB NOT NULL, "
            "candidate_count INTEGER NOT NULL, score_count INTEGER NOT NULL, "
            "build_seconds REAL NOT NULL)"
        )
        saved = self.connection.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()
        if saved is not None and saved[0] != CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported example cache schema {saved[0]!r}"
            )
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
            ("schema_version", CACHE_SCHEMA_VERSION),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> JointExampleCache:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def key(self, row: Mapping[str, Any]) -> str:
        return example_cache_key(
            row,
            pairing_tolerance_sec=self.pairing_tolerance_sec,
            minimum_candidate_confidence=self.minimum_candidate_confidence,
        )

    def contains(self, row: Mapping[str, Any]) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM examples WHERE cache_key=?",
                (self.key(row),),
            ).fetchone()
            is not None
        )

    def get(self, row: Mapping[str, Any]) -> JointTrainingExample:
        key = self.key(row)
        saved = self.connection.execute(
            "SELECT payload FROM examples WHERE cache_key=?",
            (key,),
        ).fetchone()
        if saved is None:
            raise KeyError(f"Training example is not cached: {key}")
        return _deserialize(saved[0])

    def _put_result(
        self, result: tuple[str, bytes, int, int, float]
    ) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO examples("
            "cache_key,payload,candidate_count,score_count,build_seconds"
            ") VALUES(?,?,?,?,?)",
            result,
        )

    def prepare(
        self,
        rows: Sequence[Mapping[str, Any]],
        cache_root: Path,
        *,
        workers: int,
        prefetch: int,
        progress: Callable[[int, int, int], None] | None = None,
    ) -> dict[str, float | int]:
        keyed = [(self.key(row), row) for row in rows]
        existing = {
            value[0]
            for value in self.connection.execute(
                "SELECT cache_key FROM examples"
            )
        }
        missing = [
            (key, row) for key, row in keyed if key not in existing
        ]
        started = time.perf_counter()
        built = 0
        if workers <= 1:
            for key, row in missing:
                self._put_result(
                    _build_task(
                        (
                            key,
                            row,
                            str(cache_root),
                            self.pairing_tolerance_sec,
                            self.minimum_candidate_confidence,
                        )
                    )
                )
                built += 1
                if built % 32 == 0:
                    self.connection.commit()
                if progress is not None:
                    progress(len(existing) + built, len(keyed), built)
        elif missing:
            limit = max(int(prefetch), int(workers))
            with ProcessPoolExecutor(max_workers=workers) as executor:
                iterator = iter(missing)
                pending = {}

                def submit_one() -> bool:
                    try:
                        key, row = next(iterator)
                    except StopIteration:
                        return False
                    future = executor.submit(
                        _build_task,
                        (
                            key,
                            row,
                            str(cache_root),
                            self.pairing_tolerance_sec,
                            self.minimum_candidate_confidence,
                        ),
                    )
                    pending[future] = key
                    return True

                while len(pending) < limit and submit_one():
                    pass
                while pending:
                    completed, _ = wait(
                        pending, return_when=FIRST_COMPLETED
                    )
                    for future in completed:
                        pending.pop(future)
                        self._put_result(future.result())
                        built += 1
                        if built % 32 == 0:
                            self.connection.commit()
                        if progress is not None:
                            progress(
                                len(existing) + built,
                                len(keyed),
                                built,
                            )
                        submit_one()
        self.connection.commit()
        elapsed = time.perf_counter() - started
        return {
            "requested": len(keyed),
            "cache_hits": len(keyed) - len(missing),
            "built": built,
            "seconds": elapsed,
            "rows_per_second": (
                built / elapsed if built and elapsed > 0.0 else 0.0
            ),
        }

    def iter_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        order: Sequence[int] | None = None,
    ) -> Iterator[JointTrainingExample]:
        indices = order if order is not None else range(len(rows))
        for index in indices:
            yield self.get(rows[index])

    def export_metadata(self) -> dict[str, Any]:
        count, payload_bytes, build_seconds = self.connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(payload)),0), "
            "COALESCE(SUM(build_seconds),0) FROM examples"
        ).fetchone()
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "path": str(self.path),
            "examples": int(count),
            "payload_bytes": int(payload_bytes),
            "source_build_seconds": float(build_seconds),
        }


class CachedExampleSequence(Sequence[JointTrainingExample]):
    def __init__(
        self,
        cache: JointExampleCache,
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        self.cache = cache
        self.rows = rows

    @overload
    def __getitem__(self, index: int) -> JointTrainingExample: ...

    @overload
    def __getitem__(self, index: slice) -> list[JointTrainingExample]: ...

    def __getitem__(
        self, index: int | slice
    ) -> JointTrainingExample | list[JointTrainingExample]:
        if isinstance(index, slice):
            return [self.cache.get(row) for row in self.rows[index]]
        return self.cache.get(self.rows[index])

    def __len__(self) -> int:
        return len(self.rows)


def atomic_cache_copy(source: Path, destination: Path) -> None:
    """Copy a closed cache atomically when a portable snapshot is required."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, tempfile.NamedTemporaryFile(
        dir=destination.parent, delete=False
    ) as writer:
        while chunk := reader.read(1024 * 1024):
            writer.write(chunk)
        temporary = Path(writer.name)
    os.replace(temporary, destination)
