"""Memory-mapped precomputed local lattices for outputRaw training.

The cache is derived only from the already-audited train split.  It stores the
exact tensors produced by ``prepare_local_sample`` in sequential shards, keyed
by the packed release id and complete lattice configuration.  Interrupted
builds retain a committed prefix and truncate only uncommitted tail bytes on
resume.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .lattice import FEATURE_DIM, LatticeConfig


PREPARED_LOCAL_SCHEMA_VERSION = "align-prepared-local-v1"
_ARRAYS = {
    "edges": ("float32", FEATURE_DIM),
    "groups": ("int32", 3),
    "classes": ("int16", 7),
    "regression": ("float32", 3),
}
_CLASS_FIELDS = (
    "keep",
    "boundary",
    "split",
    "emission",
    "structure",
    "layer2",
    "rhythm",
)
_REGRESSION_FIELDS = (
    "duration_target",
    "duration_weight",
    "rearticulation_weight",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def prepared_local_fingerprint(
    pack_id: str,
    lattice_config: LatticeConfig,
) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "schema_version": PREPARED_LOCAL_SCHEMA_VERSION,
                "pack_id": str(pack_id),
                "lattice": asdict(lattice_config),
                "feature_dim": FEATURE_DIM,
            }
        )
    ).hexdigest()


@dataclass(frozen=True)
class PreparedLocalRecord:
    edge_features: torch.Tensor
    groups: tuple[tuple[int, int, int], ...]
    keep: torch.Tensor
    boundary: torch.Tensor
    split: torch.Tensor
    emission: torch.Tensor
    structure: torch.Tensor
    layer2: torch.Tensor
    rhythm: torch.Tensor
    duration_target: torch.Tensor
    duration_weight: torch.Tensor
    rearticulation_weight: torch.Tensor
    copy_count_target: int
    sample: str

    @property
    def edge_count(self) -> int:
        return int(self.edge_features.shape[0])


def _record_arrays(prepared: Any) -> dict[str, np.ndarray]:
    edge_features = np.asarray(
        prepared.edge_features.detach().cpu().numpy(),
        dtype="<f4",
        order="C",
    )
    groups = np.asarray(prepared.groups, dtype="<i4", order="C").reshape(-1, 3)
    classes = np.column_stack(
        [
            np.asarray(
                getattr(prepared, name).detach().cpu().numpy(),
                dtype="<i2",
            )
            for name in _CLASS_FIELDS
        ]
    )
    regression = np.column_stack(
        [
            np.asarray(
                getattr(prepared, name).detach().cpu().numpy(),
                dtype="<f4",
            )
            for name in _REGRESSION_FIELDS
        ]
    )
    group_count = len(groups)
    if edge_features.ndim != 2 or edge_features.shape[1] != FEATURE_DIM:
        raise ValueError("Prepared edge feature shape is invalid")
    if classes.shape != (group_count, len(_CLASS_FIELDS)):
        raise ValueError("Prepared class target shape is invalid")
    if regression.shape != (group_count, len(_REGRESSION_FIELDS)):
        raise ValueError("Prepared regression target shape is invalid")
    return {
        "edges": edge_features,
        "groups": groups,
        "classes": classes,
        "regression": regression,
    }


class PreparedLocalWriter:
    """Crash-resumable writer for a deterministic source-ordinal prefix."""

    def __init__(
        self,
        destination: Path,
        *,
        pack_id: str,
        lattice_config: LatticeConfig,
        shard_rows: int = 128,
        checkpoint_rows: int = 16,
        staging: Path | None = None,
    ) -> None:
        self.destination = Path(destination).resolve()
        if self.destination.exists():
            raise FileExistsError(self.destination)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.staging = (
            Path(staging).resolve()
            if staging is not None
            else Path(
                tempfile.mkdtemp(
                    prefix=f".{self.destination.name}.",
                    dir=self.destination.parent,
                )
            )
        )
        self.pack_id = str(pack_id)
        self.lattice_config = lattice_config
        self.cache_fingerprint = prepared_local_fingerprint(
            self.pack_id, self.lattice_config
        )
        self.shard_rows = max(1, int(shard_rows))
        self.checkpoint_rows = max(1, int(checkpoint_rows))
        self.index_path = self.staging / "index.sqlite"
        self.connection = sqlite3.connect(self.index_path)
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS records("
            "ordinal INTEGER PRIMARY KEY, source_ordinal INTEGER UNIQUE NOT NULL, "
            "sample TEXT UNIQUE NOT NULL, shard INTEGER NOT NULL, "
            "edge_offset INTEGER NOT NULL, edge_count INTEGER NOT NULL, "
            "group_offset INTEGER NOT NULL, group_count INTEGER NOT NULL, "
            "copy_count INTEGER NOT NULL, record_sha256 TEXT NOT NULL)"
        )
        self.connection.commit()
        saved = self.connection.execute(
            "SELECT COUNT(*),MIN(ordinal),MAX(ordinal) FROM records"
        ).fetchone()
        self._ordinal = int(saved[0])
        if self._ordinal and (
            int(saved[1]) != 0 or int(saved[2]) != self._ordinal - 1
        ):
            raise ValueError("Prepared cache committed prefix is not contiguous")
        self._shard = (
            int(
                self.connection.execute(
                    "SELECT shard FROM records ORDER BY ordinal DESC LIMIT 1"
                ).fetchone()[0]
            )
            if self._ordinal
            else 0
        )
        self._shard_count = (
            int(
                self.connection.execute(
                    "SELECT COUNT(*) FROM records WHERE shard=?",
                    (self._shard,),
                ).fetchone()[0]
            )
            if self._ordinal
            else 0
        )
        offsets = (
            self.connection.execute(
                "SELECT COALESCE(SUM(edge_count),0),"
                "COALESCE(SUM(group_count),0) FROM records WHERE shard=?",
                (self._shard,),
            ).fetchone()
            if self._ordinal
            else (0, 0)
        )
        self._edge_offset = int(offsets[0])
        self._group_offset = int(offsets[1])
        self._handles: dict[str, Any] = {}
        self._closed = False

    @property
    def committed_records(self) -> int:
        return self._ordinal

    def validate_prefix(
        self,
        expected: Sequence[tuple[int, str]],
    ) -> None:
        if self._ordinal > len(expected):
            raise ValueError("Prepared cache is longer than the source order")
        for ordinal, source_ordinal, sample in self.connection.execute(
            "SELECT ordinal,source_ordinal,sample FROM records ORDER BY ordinal"
        ):
            wanted_ordinal, wanted_sample = expected[int(ordinal)]
            if (
                int(source_ordinal) != int(wanted_ordinal)
                or str(sample) != str(wanted_sample)
            ):
                raise ValueError(
                    f"Prepared cache prefix diverges at ordinal {ordinal}"
                )

    def _path(self, shard: int, name: str) -> Path:
        dtype, _width = _ARRAYS[name]
        return self.staging / f"shard-{shard:05d}.{name}.{dtype}.bin"

    def _close_handles(self) -> None:
        for stream in self._handles.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        self._handles.clear()

    def _open_append(self) -> None:
        self._close_handles()
        if self._shard_count >= self.shard_rows:
            self._shard += 1
            self._shard_count = 0
            self._edge_offset = 0
            self._group_offset = 0
        for path in self.staging.glob("shard-*.bin"):
            try:
                shard = int(path.name.split(".", 1)[0].split("-")[1])
            except (IndexError, ValueError):
                continue
            if shard > self._shard:
                path.unlink()
        lengths = {
            "edges": self._edge_offset,
            "groups": self._group_offset,
            "classes": self._group_offset,
            "regression": self._group_offset,
        }
        for name, (dtype, width) in _ARRAYS.items():
            path = self._path(self._shard, name)
            expected_bytes = lengths[name] * width * np.dtype(dtype).itemsize
            with path.open("a+b") as stream:
                stream.truncate(expected_bytes)
            self._handles[name] = path.open("ab")

    def add(
        self,
        *,
        source_ordinal: int,
        sample: str,
        prepared: Any,
    ) -> str:
        if not self._handles or self._shard_count >= self.shard_rows:
            self._open_append()
        arrays = _record_arrays(prepared)
        digest = hashlib.sha256()
        digest.update(self.cache_fingerprint.encode("ascii"))
        digest.update(str(int(source_ordinal)).encode("ascii"))
        for name in _ARRAYS:
            raw = arrays[name].tobytes(order="C")
            digest.update(raw)
            self._handles[name].write(raw)
        edge_count = int(arrays["edges"].shape[0])
        group_count = int(arrays["groups"].shape[0])
        record_hash = digest.hexdigest()
        self.connection.execute(
            "INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                self._ordinal,
                int(source_ordinal),
                str(sample),
                self._shard,
                self._edge_offset,
                edge_count,
                self._group_offset,
                group_count,
                int(prepared.copy_count_target),
                record_hash,
            ),
        )
        self._ordinal += 1
        self._shard_count += 1
        self._edge_offset += edge_count
        self._group_offset += group_count
        if self._ordinal % self.checkpoint_rows == 0:
            for stream in self._handles.values():
                stream.flush()
                os.fsync(stream.fileno())
            self.connection.commit()
        return record_hash

    def preserve(self) -> None:
        if self._closed:
            return
        self._close_handles()
        self.connection.commit()
        self.connection.close()
        self._closed = True

    def finalize(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Prepared cache writer is closed")
        self._close_handles()
        self.connection.commit()
        self.connection.execute("PRAGMA optimize")
        self.connection.close()
        files = [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in sorted(self.staging.glob("shard-*.bin"))
        ]
        index = {
            "name": self.index_path.name,
            "size": self.index_path.stat().st_size,
            "sha256": _sha256_file(self.index_path),
        }
        metadata = {
            "schema_version": PREPARED_LOCAL_SCHEMA_VERSION,
            "pack_id": self.pack_id,
            "cache_fingerprint": self.cache_fingerprint,
            "lattice": asdict(self.lattice_config),
            "feature_dim": FEATURE_DIM,
            "shard_rows": self.shard_rows,
            "record_count": self._ordinal,
            "index": index,
            "files": files,
            "arrays": {
                name: {"dtype": dtype, "width": width}
                for name, (dtype, width) in _ARRAYS.items()
            },
        }
        metadata["metadata_sha256"] = hashlib.sha256(
            _canonical_json(metadata)
        ).hexdigest()
        (self.staging / "metadata.json").write_bytes(
            json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )
        os.replace(self.staging, self.destination)
        self._closed = True
        return metadata

    def __enter__(self) -> "PreparedLocalWriter":
        return self

    def __exit__(self, exc_type: Any, _exc: Any, _tb: Any) -> None:
        if exc_type is not None:
            self.preserve()


class PreparedLocalDataset(Sequence[PreparedLocalRecord]):
    def __init__(
        self,
        root: Path,
        *,
        pack_id: str,
        lattice_config: LatticeConfig,
        max_open_shards: int = 4,
    ) -> None:
        self.root = Path(root)
        self.metadata = json.loads(
            (self.root / "metadata.json").read_text(encoding="utf-8")
        )
        if self.metadata.get("schema_version") != PREPARED_LOCAL_SCHEMA_VERSION:
            raise ValueError("Unsupported prepared local cache")
        metadata_hash = self.metadata.get("metadata_sha256")
        calculated = hashlib.sha256(
            _canonical_json(
                {
                    key: value
                    for key, value in self.metadata.items()
                    if key != "metadata_sha256"
                }
            )
        ).hexdigest()
        if calculated != metadata_hash:
            raise ValueError("Prepared local metadata hash mismatch")
        expected = prepared_local_fingerprint(pack_id, lattice_config)
        if self.metadata.get("cache_fingerprint") != expected:
            raise ValueError("Prepared local cache configuration mismatch")
        index = self.metadata["index"]
        index_path = self.root / index["name"]
        if (
            index_path.stat().st_size != int(index["size"])
            or _sha256_file(index_path) != index["sha256"]
        ):
            raise ValueError("Prepared local index hash mismatch")
        for row in self.metadata["files"]:
            path = self.root / row["name"]
            if not path.is_file() or path.stat().st_size != int(row["size"]):
                raise ValueError(f"Prepared local shard mismatch: {path.name}")
        self.connection = sqlite3.connect(
            f"file:{index_path.resolve().as_posix()}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        self.connection.execute("PRAGMA query_only=ON")
        self.max_open_shards = max(1, int(max_open_shards))
        self._maps: OrderedDict[tuple[int, str], np.memmap] = OrderedDict()

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
        items = path.stat().st_size // np.dtype(dtype).itemsize
        shape = (items // width, width)
        saved = np.memmap(path, dtype=dtype, mode="r", shape=shape)
        self._maps[key] = saved
        while len(self._maps) > self.max_open_shards * len(_ARRAYS):
            self._maps.popitem(last=False)
        return saved

    def __getitem__(
        self, index: int | slice
    ) -> PreparedLocalRecord | list[PreparedLocalRecord]:
        if isinstance(index, slice):
            source_ordinals = [
                int(row[0])
                for row in self.connection.execute(
                    "SELECT source_ordinal FROM records ORDER BY ordinal"
                )
            ]
            return [
                self[source_ordinal]
                for source_ordinal in source_ordinals[index]
            ]
        row = self.connection.execute(
            "SELECT source_ordinal,sample,shard,edge_offset,edge_count,"
            "group_offset,group_count,copy_count FROM records "
            "WHERE source_ordinal=?",
            (int(index),),
        ).fetchone()
        if row is None:
            raise IndexError(index)
        shard = int(row[2])
        edge_start, edge_count = int(row[3]), int(row[4])
        group_start, group_count = int(row[5]), int(row[6])
        edges = np.array(
            self._map(shard, "edges")[
                edge_start : edge_start + edge_count
            ],
            copy=True,
        )
        groups_array = np.array(
            self._map(shard, "groups")[
                group_start : group_start + group_count
            ],
            copy=True,
        )
        classes = np.array(
            self._map(shard, "classes")[
                group_start : group_start + group_count
            ],
            copy=True,
        )
        regression = np.array(
            self._map(shard, "regression")[
                group_start : group_start + group_count
            ],
            copy=True,
        )
        class_tensors = {
            name: torch.from_numpy(classes[:, column].astype(np.int64))
            for column, name in enumerate(_CLASS_FIELDS)
        }
        regression_tensors = {
            name: torch.from_numpy(regression[:, column])
            for column, name in enumerate(_REGRESSION_FIELDS)
        }
        return PreparedLocalRecord(
            edge_features=torch.from_numpy(edges),
            groups=tuple(
                tuple(int(value) for value in group)
                for group in groups_array
            ),
            **class_tensors,
            **regression_tensors,
            copy_count_target=int(row[7]),
            sample=str(row[1]),
        )

    def validate(self, *, deep: bool = False) -> dict[str, Any]:
        if deep:
            for row in self.metadata["files"]:
                path = self.root / row["name"]
                if _sha256_file(path) != row["sha256"]:
                    raise ValueError(
                        f"Prepared local shard hash mismatch: {path.name}"
                    )
        saved = self.connection.execute(
            "SELECT COUNT(*),COUNT(DISTINCT source_ordinal),"
            "MIN(ordinal),MAX(ordinal) FROM records"
        ).fetchone()
        count = int(saved[0])
        if count != int(saved[1]) or (
            count and (int(saved[2]) != 0 or int(saved[3]) != count - 1)
        ):
            raise ValueError("Prepared local index is not contiguous")
        return {
            "status": "ok",
            "records": count,
            "deep": bool(deep),
            "cache_fingerprint": self.metadata["cache_fingerprint"],
        }

    def close(self) -> None:
        self._maps.clear()
        self.connection.close()

    def __enter__(self) -> "PreparedLocalDataset":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def find_staging(destination: Path) -> Path | None:
    paths = sorted(
        destination.parent.glob(f".{destination.name}.*"),
        key=lambda path: path.stat().st_mtime,
    )
    if len(paths) > 1:
        raise ValueError(f"Multiple prepared-cache staging directories: {paths}")
    return paths[0] if paths else None


def remove_prepared_cache(path: Path) -> None:
    """Remove only a derived, explicitly selected prepared cache."""

    shutil.rmtree(path)
