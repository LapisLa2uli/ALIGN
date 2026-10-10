"""Durable acceptance records for the single-process, localhost Studio server."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import re
import sqlite3

from fastapi import HTTPException


def key_digest(key: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", key):
        raise HTTPException(422, "Idempotency-Key must be 1–128 ASCII letters, digits, '.', '_', ':', or '-'.")
    return hashlib.sha256(key.encode("ascii")).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def request_store(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(root / "requests.sqlite3", timeout=10)
    try:
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("""CREATE TABLE IF NOT EXISTS requests (
            key_hash TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, job_id TEXT NOT NULL UNIQUE
        )""")
        # Serializes duplicate acceptance, including requests on different threads.
        connection.execute("BEGIN IMMEDIATE")
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def receipt(root: Path, job_id: str, *, replayed: bool):
    if not (root / "jobs" / job_id / "status.json").is_file():
        # Never forget an accepted key just because somebody removed its job.
        raise HTTPException(410, "Accepted take artifacts are unavailable; this key will not start another take.")
    return {"id": job_id, "status_url": f"/api/studio/takes/{job_id}", "replayed": replayed}


def lookup(root: Path, key: str):
    digest = key_digest(key)
    with request_store(root) as connection:
        row = connection.execute("SELECT job_id FROM requests WHERE key_hash = ?", (digest,)).fetchone()
        if row is None:
            raise HTTPException(404, "Idempotency key has not been accepted.")
        return receipt(root, row[0], replayed=True)
