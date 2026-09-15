"""Random-access loader for audited, corrected training targets."""

from __future__ import annotations

import json
import os
import sqlite3
import zlib
from pathlib import Path
from threading import Lock
from typing import Any, Mapping


_CONNECTIONS: dict[tuple[int, str], sqlite3.Connection] = {}
_LOCK = Lock()


def _norm(path: Path | str) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _connection(path: Path) -> sqlite3.Connection:
    key = (os.getpid(), _norm(path))
    with _LOCK:
        connection = _CONNECTIONS.get(key)
        if connection is None:
            connection = sqlite3.connect(
                f"file:{path.resolve().as_posix()}?mode=ro",
                uri=True,
                check_same_thread=False,
            )
            _CONNECTIONS[key] = connection
        return connection


def load_validated_target(row: Mapping[str, Any]) -> dict[str, Any]:
    """Load and verify the corrected target referenced by a manifest row."""

    database = row.get("target_db")
    ordinal = row.get("target_record")
    if database is None or ordinal is None:
        note_map = row.get("note_map")
        if note_map is None:
            sample = Path(str(row["sample_dir"]))
            note_map = sample / "note_map.json"
        document = json.loads(Path(str(note_map)).read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(f"Target document is not an object: {note_map}")
        return document

    path = Path(str(database))
    if not path.is_file():
        raise FileNotFoundError(path)
    saved = _connection(path).execute(
        "SELECT split, corpus, sample_dir, source_hashes, payload "
        "FROM targets WHERE ordinal=?",
        (int(ordinal),),
    ).fetchone()
    if saved is None:
        raise KeyError(f"No validated target record {ordinal} in {path}")
    payload = json.loads(zlib.decompress(saved[4]).decode("utf-8"))
    expected_dir = _norm(str(row["sample_dir"]))
    if saved[2] != expected_dir or _norm(payload["sample_dir"]) != expected_dir:
        raise ValueError(
            f"Target record {ordinal} belongs to {payload.get('sample_dir')}, "
            f"not {row['sample_dir']}"
        )
    if str(saved[1]) != str(row.get("corpus") or row.get("root")):
        raise ValueError(f"Target record {ordinal} corpus does not match manifest")
    expected_hashes = row.get("source_hashes")
    if expected_hashes is not None:
        database_hashes = json.loads(saved[3])
        if database_hashes != expected_hashes or payload.get("source_hashes") != expected_hashes:
            raise ValueError(f"Target record {ordinal} source hashes do not match manifest")
    return payload


def target_note_map(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a note-map-shaped document from a validated target row."""

    payload = load_validated_target(row)
    if payload.get("kind") == "synth_note_lineage":
        return payload
    clean = list(payload.get("clean_notes") or [])
    performed = list(payload.get("performed_notes") or [])
    rendered = list(payload.get("rendered_notes") or [])
    return {
        "schema_version": "1.0",
        "kind": "synth_note_lineage",
        "clean_note_count": len(clean),
        "performed_note_count": len(performed),
        "rendered_note_count": len(rendered),
        "clean_notes": clean,
        "performed_notes": performed,
        "rendered_notes": rendered,
        "deleted_clean_notes": list(payload.get("deleted_clean_notes") or []),
        "source_hashes": payload.get("source_hashes") or {},
        "pitch_policy": payload.get("pitch_policy") or {},
    }
