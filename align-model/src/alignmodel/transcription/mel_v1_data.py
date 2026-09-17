"""Audited packed mel cache and deterministic crop loading for Track B."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from alignmodel.joint.outputraw_full import verify_data_ready
from alignmodel.joint.packed_data import PackedJointDataset, sha256_file

from .mel_v1 import (
    CACHE_SCHEMA_VERSION,
    MelFrontendConfig,
    extract_log_mel,
    load_audio_mono,
    make_mel_targets,
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True).encode("utf-8"))
        stream.write(b"\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class MelCacheRecord:
    ordinal: int
    sample: str
    split: str
    source: str
    shard: int
    frame_offset: int
    frame_count: int
    duration_sec: float
    audio_render: str
    effective_audio_transpose: int
    mel_sha256: str
    target: tuple[dict[str, float | int], ...] = ()


class MelPackedCache:
    """Checksum-verified, read-only float16 packed mel shards."""

    def __init__(self, root: Path | str, *, deep: bool = False) -> None:
        self.root = Path(root)
        self.metadata = json.loads(
            (self.root / "metadata.json").read_text(encoding="utf-8")
        )
        if self.metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError("Unsupported mel cache schema")
        expected_id = hashlib.sha256(_canonical_json({
            key: value for key, value in self.metadata.items() if key != "pack_id"
        })).hexdigest()
        if expected_id != self.metadata.get("pack_id"):
            raise ValueError("Mel cache metadata checksum mismatch")
        index = self.metadata["index"]
        index_path = self.root / index["name"]
        if (
            not index_path.is_file()
            or index_path.stat().st_size != int(index["bytes"])
            or sha256_file(index_path) != index["sha256"]
        ):
            raise ValueError("Mel cache index checksum mismatch")
        for shard in self.metadata["shards"]:
            path = self.root / shard["name"]
            if not path.is_file() or path.stat().st_size != int(shard["bytes"]):
                raise ValueError(f"Missing or truncated mel shard: {path.name}")
            if deep and sha256_file(path) != shard["sha256"]:
                raise ValueError(f"Mel shard checksum mismatch: {path.name}")
        self.frontend = MelFrontendConfig.from_dict(
            self.metadata["frontend_config"]
        )
        self._index_path = index_path
        self._maps: dict[int, np.memmap] = {}

    @property
    def pack_id(self) -> str:
        return str(self.metadata["pack_id"])

    def records(
        self, split: str, *, include_targets: bool = True
    ) -> list[MelCacheRecord]:
        if split not in {"train", "val"}:
            authorized_test = (
                split == "test"
                and include_targets is False
                and self.metadata.get("source", {}).get(
                    "locked_test_materialized"
                )
                is True
            )
            if not authorized_test:
                raise ValueError("Mel cache exposes train/val only")
        connection = sqlite3.connect(
            f"file:{self._index_path.resolve().as_posix()}?mode=ro", uri=True
        )
        try:
            columns = (
                "ordinal,sample,split,source,shard,frame_offset,frame_count,"
                "duration_sec,audio_render,effective_audio_transpose,mel_sha256"
            )
            if include_targets:
                columns += ",target"
            rows = connection.execute(
                f"SELECT {columns} FROM records WHERE split=? ORDER BY ordinal",
                (split,),
            ).fetchall()
        finally:
            connection.close()
        return [
            MelCacheRecord(
                ordinal=int(row[0]),
                sample=str(row[1]),
                split=str(row[2]),
                source=str(row[3]),
                shard=int(row[4]),
                frame_offset=int(row[5]),
                frame_count=int(row[6]),
                duration_sec=float(row[7]),
                audio_render=str(row[8]),
                effective_audio_transpose=int(row[9]),
                mel_sha256=str(row[10]),
                target=(
                    tuple(json.loads(zlib.decompress(row[11])))
                    if include_targets
                    else ()
                ),
            )
            for row in rows
        ]

    def mel(self, record: MelCacheRecord, *, verify: bool = False) -> np.ndarray:
        mapping = self._maps.get(record.shard)
        if mapping is None:
            path = self.root / f"shard-{record.shard:05d}.mel.float16.bin"
            count = path.stat().st_size // np.dtype("<f2").itemsize
            if count % self.frontend.n_mels:
                raise ValueError(f"Invalid packed mel size: {path.name}")
            mapping = np.memmap(
                path,
                dtype="<f2",
                mode="r",
                shape=(count // self.frontend.n_mels, self.frontend.n_mels),
            )
            self._maps[record.shard] = mapping
        start = record.frame_offset
        end = start + record.frame_count
        # On disk is [T,M] so every variable-length record is contiguous.
        value = np.asarray(mapping[start:end]).T
        if verify and _sha256_bytes(value.T.tobytes(order="C")) != record.mel_sha256:
            raise ValueError(f"Mel record checksum mismatch: {record.sample}")
        return value

    def close(self) -> None:
        self._maps.clear()


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


def build_mel_cache(
    ready_marker: Path | str,
    destination: Path | str,
    frontend: MelFrontendConfig,
    *,
    device: torch.device | str = "cuda",
    shard_rows: int = 64,
    progress_callback=None,
    splits: tuple[str, ...] = ("train", "val"),
    include_targets: bool = True,
    allow_locked_test: bool = False,
) -> Path:
    """Build audited mel features, requiring explicit lockbox authorization."""

    ready_path = Path(ready_marker).resolve()
    ready = verify_data_ready(ready_path)
    destination = Path(destination).resolve()
    if destination.exists():
        cache = MelPackedCache(destination, deep=False)
        if cache.frontend != frontend:
            raise ValueError("Existing cache has a different frontend")
        if cache.metadata["source"]["pack_id"] != ready["hashes"]["pack_id"]:
            raise ValueError("Existing cache belongs to another audited release")
        return destination
    manifest_path = Path(ready["paths"]["manifest"]).resolve()
    if sha256_file(manifest_path) != ready["hashes"]["manifest_sha256"]:
        raise ValueError("Audited manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "test" in splits and not allow_locked_test:
        raise ValueError("Locked-test mel materialization requires explicit authorization")
    if "test" in splits and include_targets:
        raise ValueError("Locked-test feature cache must not contain targets")
    selected = [
        dict(row)
        for split in splits
        for row in manifest.get(split, ())
    ]
    expected = sum(len(manifest.get(split, ())) for split in splits)
    if len(selected) != expected:
        raise ValueError("Audited train/val row count mismatch")
    if any(row.get("split") not in set(splits) for row in selected):
        raise ValueError("Cache request contains an unknown split")

    target_by_sample = {}
    if include_targets:
        packed = PackedJointDataset(
            Path(ready["paths"]["packed_root"]),
            manifest_sha256=ready["hashes"]["manifest_sha256"],
            verify_records=False,
            load_feature_arrays=False,
        )
        try:
            target_by_sample = {
                item.sample: tuple(item.target["transcription"])
                for split in splits
                for item in (packed[index] for index in packed.ordinals(split))
            }
        finally:
            packed.close()
        if set(target_by_sample) != {str(row["sample"]) for row in selected}:
            raise ValueError("Packed targets do not match requested rows")

    staging = destination.with_name(f".{destination.name}.building")
    if staging.exists():
        # A shard is committed to SQLite only after its atomic rename.  Any
        # interrupted unindexed temporary is therefore safe to remove.
        for temporary in staging.glob("*.tmp"):
            temporary.unlink(missing_ok=True)
    else:
        staging.mkdir(parents=True)
    index_path = staging / "index.sqlite"
    connection = _new_index(index_path)
    committed = int(connection.execute(
        "SELECT COUNT(*) FROM records"
    ).fetchone()[0])
    if committed and committed % max(1, shard_rows):
        raise ValueError("Interrupted cache does not end on a shard boundary")

    try:
        for chunk_start in range(committed, len(selected), max(1, shard_rows)):
            chunk = selected[chunk_start:chunk_start + max(1, shard_rows)]
            shard = chunk_start // max(1, shard_rows)
            arrays: list[np.ndarray] = []
            records = []
            frame_offset = 0
            for row in chunk:
                sample = str(row["sample"])
                source_dir = Path(str(row["sample_dir"]))
                audio_path = source_dir / "performance_audio.wav"
                metadata_path = source_dir / "metadata.json"
                hashes = row.get("source_hashes") or {}
                if sha256_file(audio_path) != hashes.get("performance_audio.wav"):
                    raise ValueError(f"Audited audio checksum mismatch: {sample}")
                if sha256_file(metadata_path) != hashes.get("metadata.json"):
                    raise ValueError(f"Immutable metadata checksum mismatch: {sample}")
                transpose = int(row["effective_audio_transpose"])
                if str(row.get("audio_pitch_space")) != "sounding":
                    raise ValueError(f"Unexpected audio pitch space: {sample}")
                audio = load_audio_mono(audio_path, frontend.sample_rate)
                mel, normalization = extract_log_mel(
                    audio, frontend, device=device
                )
                contiguous = np.ascontiguousarray(mel.T, dtype="<f2")
                raw = contiguous.tobytes(order="C")
                arrays.append(contiguous)
                records.append((
                    sample,
                    str(row["split"]),
                    str(row["source"]),
                    shard,
                    frame_offset,
                    int(contiguous.shape[0]),
                    float(row["duration_sec"]),
                    str(row["audio_render"]),
                    transpose,
                    str(hashes["performance_audio.wav"]),
                    _sha256_bytes(raw),
                    json.dumps(normalization, sort_keys=True),
                    sqlite3.Binary(zlib.compress(
                        _canonical_json(target_by_sample.get(sample, ())), level=1
                    )),
                ))
                frame_offset += int(contiguous.shape[0])
                if progress_callback is not None:
                    progress_callback(chunk_start + len(records), len(selected), sample)
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
        connection.execute("PRAGMA optimize")
        connection.commit()
    except BaseException:
        connection.close()
        raise
    connection.close()

    shards = [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(staging.glob("shard-*.bin"))
    ]
    index = {
        "name": index_path.name,
        "bytes": index_path.stat().st_size,
        "sha256": sha256_file(index_path),
    }
    metadata: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "frontend_config": frontend.to_dict(),
        "dtype": "float16",
        "layout": "time_major_contiguous",
        "shard_rows": max(1, int(shard_rows)),
        "split_counts": {
            split: sum(
                1 for row in selected if row.get("split") == split
            )
            for split in splits
        },
        "record_count": len(selected),
        "source": {
            "release": ready["release"],
            "ready_marker": str(ready_path),
            "ready_sha256": sha256_file(ready_path),
            "manifest_sha256": ready["hashes"]["manifest_sha256"],
            "pack_id": ready["hashes"]["pack_id"],
            "locked_test_materialized": "test" in splits,
            "targets_included": include_targets,
        },
        "index": index,
        "shards": shards,
    }
    metadata["pack_id"] = hashlib.sha256(_canonical_json(metadata)).hexdigest()
    _atomic_json(staging / "metadata.json", metadata)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, destination)
    return destination


class MelCropDataset(Dataset):
    """Deterministic packed crops; augmentation stays on the training device."""

    def __init__(
        self,
        cache_root: Path | str,
        records: Sequence[MelCacheRecord],
        *,
        crop_frames: int,
        epoch: int,
        seed: int,
        crops_per_clip: int = 1,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.records = tuple(records)
        self.crop_frames = int(crop_frames)
        self.epoch = int(epoch)
        self.seed = int(seed)
        self.crops_per_clip = max(1, int(crops_per_clip))
        metadata = json.loads(
            (self.cache_root / "metadata.json").read_text(encoding="utf-8")
        )
        self.frontend = MelFrontendConfig.from_dict(metadata["frontend_config"])
        self._maps: dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return len(self.records) * self.crops_per_clip

    def _mel(self, record: MelCacheRecord) -> np.ndarray:
        mapping = self._maps.get(record.shard)
        if mapping is None:
            path = self.cache_root / (
                f"shard-{record.shard:05d}.mel.float16.bin"
            )
            values = path.stat().st_size // np.dtype("<f2").itemsize
            mapping = np.memmap(
                path, dtype="<f2", mode="r",
                shape=(values // self.frontend.n_mels, self.frontend.n_mels),
            )
            self._maps[record.shard] = mapping
        start = record.frame_offset
        return np.asarray(
            mapping[start:start + record.frame_count], dtype=np.float32
        ).T

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index % len(self.records)]
        mel = self._mel(record)
        maximum = max(0, record.frame_count - self.crop_frames)
        digest = hashlib.sha256(
            f"{self.seed}:{self.epoch}:{record.ordinal}:{index}".encode("ascii")
        ).digest()
        crop_start = (
            int.from_bytes(digest[:8], "little") % (maximum + 1)
            if maximum else 0
        )
        valid_frames = min(self.crop_frames, record.frame_count - crop_start)
        crop = np.zeros(
            (self.frontend.n_mels, self.crop_frames), dtype=np.float32
        )
        crop[:, :valid_frames] = mel[
            :, crop_start:crop_start + valid_frames
        ]
        targets = make_mel_targets(
            record.target,
            frames=self.crop_frames,
            hop_sec=self.frontend.hop_sec,
            midi_min=52,
            midi_max=100,
            crop_start=crop_start,
        )
        frame_mask = np.zeros(self.crop_frames, dtype=np.bool_)
        frame_mask[:valid_frames] = True
        return {
            "mel": torch.from_numpy(crop),
            **{key: torch.from_numpy(value) for key, value in targets.items()},
            "frame_mask": torch.from_numpy(frame_mask),
            "sample": record.sample,
            "source": record.source,
            "audio_render": record.audio_render,
            "crop_start": crop_start,
        }


def augment_mel_batch(
    mel: torch.Tensor, *, probability: float = 0.90
) -> torch.Tensor:
    """Timbre/noise/band/EQ/reverb perturbations with unchanged targets."""

    batch, bands, frames = mel.shape
    active = (
        torch.rand(batch, 1, 1, device=mel.device) < probability
    )
    gain = torch.empty(batch, 1, 1, device=mel.device).uniform_(-0.35, 0.25)
    knots = torch.randn(batch, 1, 8, device=mel.device) * 0.14
    equalization = F.interpolate(
        knots, size=bands, mode="linear", align_corners=True
    ).transpose(1, 2)
    colored = torch.randn(batch, 12, frames, device=mel.device)
    colored = F.interpolate(
        colored.transpose(1, 2), size=bands, mode="linear",
        align_corners=False,
    ).transpose(1, 2)
    noise = (
        torch.randn_like(mel)
        * torch.empty(batch, 1, 1, device=mel.device).uniform_(0.0, 0.10)
        + colored
        * torch.empty(batch, 1, 1, device=mel.device).uniform_(0.0, 0.06)
    )
    result = mel + gain + equalization + noise
    if frames > 4:
        reverberant = result.clone()
        reverberant[:, :, 1:] += 0.16 * result[:, :, :-1]
        reverberant[:, :, 2:] += 0.08 * result[:, :, :-2]
        use_reverb = (
            torch.rand(batch, 1, 1, device=mel.device) < 0.45
        )
        result = torch.where(use_reverb, reverberant, result)
    positions = torch.arange(bands, device=mel.device)[None, :, None]
    width = torch.randint(2, max(3, bands // 10), (batch, 1, 1), device=mel.device)
    start = (
        torch.rand(batch, 1, 1, device=mel.device) * (bands - width)
    ).long()
    attenuation = torch.empty(
        batch, 1, 1, device=mel.device
    ).uniform_(0.25, 0.85)
    band_mask = (
        (positions >= start) & (positions < start + width)
        & (torch.rand(batch, 1, 1, device=mel.device) < 0.55)
    )
    result = torch.where(band_mask, result - attenuation, result)
    return torch.where(active, result.clamp(-4.5, 3.5), mel)

