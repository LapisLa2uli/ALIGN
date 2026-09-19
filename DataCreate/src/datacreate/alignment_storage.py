"""Alignment archives with optional storage of the dense diagnostic DTW cost."""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import stat
import tempfile
import zipfile

import numpy as np


REQUIRED_ARRAYS = frozenset({
    "ref_features", "perf_features", "warping_path", "frame_residuals",
    "hop_length", "sample_rate", "silence_frames",
})


def alignment_storage_kind(archive) -> str:
    missing = REQUIRED_ARRAYS - set(archive.files)
    if missing:
        raise ValueError(f"Missing alignment arrays: {sorted(missing)}")
    omitted = archive["dtw_cost_omitted"] if "dtw_cost_omitted" in archive.files else None
    if "dtw_cost" in archive.files:
        if omitted is not None and bool(np.any(omitted)):
            raise ValueError("Archive both contains and declares omission of DTW cost")
        return "full"
    if omitted is None or omitted.shape != () or omitted.dtype != np.bool_ or not bool(omitted):
        raise ValueError("Missing DTW cost without an explicit omission marker")
    return "compact"


def save_alignment(path: Path, *, save_dtw_cost: bool = True, **arrays) -> None:
    if not save_dtw_cost:
        arrays.pop("dtw_cost")
        arrays["dtw_cost_omitted"] = np.asarray(True)
    np.savez(path, **arrays)


def _digest_member(archive: zipfile.ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with archive.open(name) as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def omit_dtw_cost(path: Path) -> dict:
    """Atomically omit only dtw_cost.npy, checking retained bytes with SHA-256.

    The dense matrix is never loaded. Existing compact archives are accepted
    only with the explicit marker. A failed copy or verification leaves the
    original file in place.
    """
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"Refusing to replace a symlink: {path}")
    before = path.stat()
    with np.load(path, allow_pickle=False) as arrays:
        kind = alignment_storage_kind(arrays)
    if kind == "compact":
        return {"path": str(path), "changed": False, "bytes_saved": 0}
    temp = None
    try:
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temp = Path(name)
        retained = {}
        with os.fdopen(fd, "w+b") as output, zipfile.ZipFile(path) as source:
            names = source.namelist()
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate archive members: {path}")
            with zipfile.ZipFile(output, "w", allowZip64=True) as dest:
                dest.comment = source.comment
                for member in source.infolist():
                    if member.filename in {"dtw_cost.npy", "dtw_cost_omitted.npy"}:
                        continue
                    digest = hashlib.sha256()
                    with source.open(member) as src, dest.open(member, "w", force_zip64=True) as dst:
                        while block := src.read(1024 * 1024):
                            digest.update(block)
                            dst.write(block)
                    retained[member.filename] = digest.hexdigest()
                marker = io.BytesIO()
                np.save(marker, np.asarray(True), allow_pickle=False)
                dest.writestr("dtw_cost_omitted.npy", marker.getvalue())
            output.flush()
            os.fsync(output.fileno())
        with zipfile.ZipFile(temp) as check:
            if set(check.namelist()) != set(retained) | {"dtw_cost_omitted.npy"}:
                raise ValueError(f"Archive member mismatch: {path}")
            if any(_digest_member(check, name) != digest for name, digest in retained.items()):
                raise ValueError(f"Retained array changed: {path}")
        with np.load(temp, allow_pickle=False) as arrays:
            if alignment_storage_kind(arrays) != "compact":
                raise ValueError(f"Invalid compact archive: {path}")
        current = path.stat()
        if (current.st_ino, current.st_size, current.st_mtime_ns) != (before.st_ino, before.st_size, before.st_mtime_ns):
            raise RuntimeError(f"Source changed during conversion: {path}")
        os.chmod(temp, stat.S_IMODE(before.st_mode))
        after_bytes = temp.stat().st_size
        os.replace(temp, path)
        return {
            "path": str(path), "changed": True,
            "before_bytes": before.st_size, "after_bytes": after_bytes,
            "bytes_saved": before.st_size - after_bytes,
            "retained_sha256": retained,
        }
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
