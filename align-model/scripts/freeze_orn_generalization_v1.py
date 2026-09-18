"""Freeze an outcome-blind ORN development/lockbox release.

This tool deliberately consumes only the frozen ORN audit manifest and its
complete inventory.  It never reads ORN predictions or reports.  Every row
that can enter the replacement release is independently reopened and checked
against the source bundle before split assignment.

The public lockbox contains copied inference inputs only.  Canonical lockbox
lineage is AES-256-GCM encrypted with a key kept outside the repository.
"""

from __future__ import annotations

import argparse
import atexit
import gzip
import hashlib
import importlib.util
import json
import os
import secrets
import shutil
import sqlite3
import sys
import tempfile
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.packed_data import sha256_file
from alignmodel.training_resources import resource_lease


SCHEMA_VERSION = "align-orn-generalization-release-v1"
REPRESENTATION_VERSION = "orn-rendered-lineage-v1"
DEFAULT_SEED = 20260918
DEFAULT_UPSTREAM = Path(
    "runs/joint-outputraw-full-v1/orn-eval-500-v1/manifest.json"
)
DEFAULT_OUTPUT = Path(
    "runs/joint-outputraw-full-v1/orn-generalization-v1"
)
DEFAULT_OUTPUTRAW_TARGETS = Path(
    "data-audit/joint-outputraw-full-v1/canonical_dev_targets.sqlite"
)
DEFAULT_OUTPUTRAW_SEAL = Path(
    "data-audit/joint-outputraw-full-v1/LOCKBOX_SEALED.json"
)
DEFAULT_RESOURCE_STATUS = Path("runs/TRAINING_RESOURCE_STATUS.json")
REQUIRED_FILES = (
    "metadata.json",
    "labels.json",
    "note_map.json",
    "verified_score.musicxml",
    "performance_score.musicxml",
    "performance_audio.wav",
    "performance_audio.mid",
    "reference_audio.wav",
    "reference_audio.mid",
)
RANDOM_SOURCE_NAMES = frozenset(
    {"", "gen", "generated", "procedural", "random"}
)
SPLITS = ("train", "calibration", "open_validation", "lockbox")
SPLIT_FRACTIONS = {
    "lockbox": (0.00, 0.10),
    "open_validation": (0.10, 0.25),
    "calibration": (0.25, 0.35),
    "train": (0.35, 1.00),
}
MIN_RAW_LOCKBOX_GROUPS = 20
LOCKBOX_AAD = f"{SCHEMA_VERSION}:lockbox-targets".encode("ascii")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _stable(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _fraction(seed: int, value: str) -> float:
    return int(_stable(seed, value)[:16], 16) / float(16**16)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_jsonl_gz(
    path: Path, rows: Iterable[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "wb") as raw:
            with gzip.GzipFile(
                filename="", fileobj=raw, mode="wb", mtime=0
            ) as compressed:
                for row in rows:
                    compressed.write(_canonical_bytes(row))
                    compressed.write(b"\n")
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    try:
        shutil.copyfile(source, temporary)
        if sha256_file(source) != sha256_file(Path(temporary)):
            raise ValueError(f"Copied file hash mismatch: {source}")
        os.replace(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class RecordingIssueLog:
    """Minimal IssueLog implementation used by the source audit."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        code: str,
        severity: str,
        sample: str,
        detail: str,
        *,
        path: str | None = None,
    ) -> None:
        row = {
            "code": str(code),
            "severity": str(severity),
            "sample": str(sample),
            "detail": str(detail),
        }
        if path is not None:
            row["path"] = str(path)
        self.rows.append(row)


def _rounded(value: Any) -> float:
    return round(float(value), 6)


def _lineage_fingerprints(
    lineage: Mapping[str, Any], audio_sha256: str
) -> dict[str, str]:
    clean = sorted(
        lineage.get("clean_notes") or [],
        key=lambda row: int(row["clean_index"]),
    )
    performed = sorted(
        lineage.get("performed_notes") or [],
        key=lambda row: int(row["performed_index"]),
    )
    clean_payload = [
        [
            int(row["pitch_midi"]),
            _rounded(row["onset_ql"]),
            _rounded(row["duration_ql"]),
        ]
        for row in clean
    ]
    first_pitch = clean_payload[0][0] if clean_payload else 0
    near_payload = [[row[0] - first_pitch, row[2]] for row in clean_payload]
    lineage_payload = [
        [
            row.get("clean_index"),
            str(row.get("relationship") or ""),
            str(row.get("origin_relationship") or ""),
            int(row.get("copy_pass") or 0),
            int(row["pitch_midi"]),
            _rounded(row["onset_ql"]),
            _rounded(row["duration_ql"]),
        ]
        for row in performed
    ]
    clean_hash = hashlib.sha256(_canonical_bytes(clean_payload)).hexdigest()
    near_hash = hashlib.sha256(_canonical_bytes(near_payload)).hexdigest()
    lineage_hash = hashlib.sha256(
        _canonical_bytes(lineage_payload)
    ).hexdigest()
    return {
        "clean_fingerprint": clean_hash,
        "near_lineage_hash": near_hash,
        "lineage_hash": lineage_hash,
        "audio_lineage_hash": hashlib.sha256(
            f"{audio_sha256}:{lineage_hash}".encode("ascii")
        ).hexdigest(),
    }


def _max_polyphony(rows: Sequence[Mapping[str, Any]]) -> int:
    points: list[tuple[float, int]] = []
    for row in rows:
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        if end <= start:
            continue
        points.extend(((start, 1), (end, -1)))
    active = maximum = 0
    for _time, change in sorted(points, key=lambda item: (item[0], item[1])):
        active += change
        maximum = max(maximum, active)
    return maximum


def _source_identity(
    metadata: Mapping[str, Any],
    verified_sha256: str,
    audit: Any,
    hash_cache: dict[str, dict[str, Any]],
) -> tuple[str, str, str | None]:
    declared = str(metadata.get("source") or "").strip()
    if declared.casefold() in RANDOM_SOURCE_NAMES:
        return "generated", f"generated:{verified_sha256}", None
    source_score = Path(str(metadata.get("source_score") or ""))
    if not source_score.is_file():
        raise ValueError("raw row lacks an existing authoritative source score")
    source_sha256 = audit._sha256(source_score, hash_cache)
    return "raw", f"raw-score-sha256:{source_sha256}", source_sha256


def _lineage_semantics(
    lineage: Mapping[str, Any],
    score_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reasons: list[dict[str, Any]] = []

    def reject(code: str, detail: str) -> None:
        reasons.append({"code": code, "detail": detail})

    if str(lineage.get("schema_version")) != "1.0":
        reject("note_map_schema_invalid", repr(lineage.get("schema_version")))
    if str(lineage.get("kind")) != "synth_note_lineage":
        reject("note_map_kind_invalid", repr(lineage.get("kind")))
    performed = list(lineage.get("performed_notes") or [])
    rendered = list(lineage.get("rendered_notes") or [])
    owners: Counter[int] = Counter()
    for row in rendered:
        for value in row.get("performed_indices") or []:
            owners[int(value)] += 1
    missing = sorted(set(range(len(performed))) - set(owners))
    duplicate = sorted(index for index, count in owners.items() if count != 1)
    if missing:
        reject(
            "unmapped_performed_notes",
            f"{len(missing)} performed events have no audible rendered owner",
        )
    if duplicate:
        reject(
            "nonexclusive_performed_lineage",
            f"{len(duplicate)} performed events have multiple rendered owners",
        )
    repair = lineage.get("render_repair")
    representation = (
        "in_bundle_rendered_lineage_v1"
        if repair is None
        else str(repair.get("policy") or "")
    )
    if repair is not None and representation != "raw_midi_authoritative_v1":
        reject("render_repair_policy_invalid", representation)
    if repair is not None and int(repair.get("unmapped_performed_notes", -1)):
        reject(
            "render_repair_incomplete",
            f"unmapped={repair.get('unmapped_performed_notes')}",
        )
    try:
        index = ScoreEventIndex.from_musicxml(score_path, lineage)
    except Exception as exc:
        reject("canonical_projection_invalid", f"{type(exc).__name__}: {exc}")
        index = None
    if index is not None:
        if not index.events or not index.rendered_events:
            reject("canonical_projection_empty", "no score or rendered events")
        if [event.rendered_index for event in index.rendered_events] != list(
            range(len(index.rendered_events))
        ):
            reject(
                "rendered_identity_invalid",
                "rendered_index is not contiguous and exclusive",
            )
        for event in index.rendered_events:
            if event.score_span is None and not event.is_extra:
                reject(
                    "unprojectable_nonextra_event",
                    f"rendered_index={event.rendered_index}",
                )
    stats = {
        "representation_version": REPRESENTATION_VERSION,
        "rendered_lineage_policy": representation,
        "score_events": 0 if index is None else len(index.events),
        "rendered_events": len(rendered),
        "performed_events": len(performed),
        "rendered_extra_events": sum(
            not (row.get("performed_indices") or []) for row in rendered
        ),
        "max_rendered_polyphony": _max_polyphony(rendered),
        "short_rendered_events_lt_120ms": sum(
            float(row["end_sec"]) - float(row["start_sec"]) < 0.120
            for row in rendered
        ),
    }
    return reasons, stats


def _independent_audit(
    sample_dir: Path,
    source_root: Path,
    audit: Any,
    hash_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    sample = sample_dir.name
    issues = RecordingIssueLog()
    result = audit._audit_bundle(
        "orn10k",
        source_root,
        0,
        sample_dir,
        {},
        {},
        hash_cache,
        issues,
    )
    reasons: list[dict[str, Any]] = []

    def reject(code: str, detail: str) -> None:
        reasons.append({"code": code, "detail": detail})

    missing = [name for name in REQUIRED_FILES if not (sample_dir / name).is_file()]
    if missing:
        reject("required_file_missing", ", ".join(missing))
    try:
        metadata = _load_json(sample_dir / "metadata.json")
        labels = _load_json(sample_dir / "labels.json")
    except Exception as exc:
        return {
            "sample": sample,
            "sample_dir": str(sample_dir),
            "eligible": False,
            "exclusion_reasons": [
                {"code": "required_json_invalid", "detail": str(exc)}
            ],
        }
    if str(metadata.get("schema_version")) != "1.2":
        reject("metadata_schema_invalid", repr(metadata.get("schema_version")))
    if str(labels.get("schema_version")) != "1.2":
        reject("labels_schema_invalid", repr(labels.get("schema_version")))
    if not bool(result.get("note_map_valid")):
        reject("note_map_invalid", "independent source audit rejected note_map")
    if int(result.get("invalid_label_count") or 0):
        reject(
            "invalid_label_spans",
            str(result.get("invalid_label_count")),
        )
    if not bool(result.get("repeat_consistent", True)):
        reject(
            "repeat_representation_inconsistent",
            "metadata repetition flag and rendered COPY lineage disagree",
        )
    expected_pitch = {
        "audio_pitch_space": "sounding",
        "midi_pitch_space": "sounding",
        "effective_audio_transpose": 2,
        "sounding_transpose": -2,
    }
    actual_pitch = {
        key: result.get(key)
        for key in (
            "audio_pitch_space",
            "midi_pitch_space",
            "effective_audio_transpose",
            "sounding_transpose",
        )
    }
    if actual_pitch != expected_pitch:
        reject(
            "bb_written_pitch_policy_invalid",
            f"expected={expected_pitch} actual={actual_pitch}",
        )
    wav = audit._read_wave_info(sample_dir / "performance_audio.wav")
    midi = audit._midi_events(sample_dir / "performance_audio.mid")
    if wav is None or float(wav.get("duration_sec") or 0) <= 0:
        reject("performance_audio_invalid", "WAV is unreadable or empty")
    if midi is None or not (midi.get("notes") or []):
        reject("performance_midi_invalid", "MIDI is unreadable or has no notes")
    if wav is not None and midi is not None:
        latest_event = max(
            (float(row["end"]) for row in midi.get("notes") or []),
            default=0.0,
        )
        if latest_event > float(wav["duration_sec"]) + 0.15:
            reject(
                "audible_event_outside_audio",
                f"event_end={latest_event:.6f} wav={wav['duration_sec']:.6f}",
            )
    hashes: dict[str, str] = {}
    for name in REQUIRED_FILES:
        path = sample_dir / name
        if path.is_file():
            try:
                hashes[name] = audit._sha256(path, hash_cache)
            except OSError as exc:
                reject("source_hash_failed", f"{name}: {exc}")
    lineage = result.get("_note_map_document")
    semantics: dict[str, Any] = {}
    if not isinstance(lineage, Mapping):
        reject("canonical_lineage_missing", "audited note-map is unavailable")
        lineage = {}
    else:
        semantic_reasons, semantics = _lineage_semantics(
            lineage, sample_dir / "verified_score.musicxml"
        )
        reasons.extend(semantic_reasons)
    verified_sha256 = hashes.get("verified_score.musicxml", "")
    try:
        provenance, source_song_id, source_score_sha256 = _source_identity(
            metadata, verified_sha256, audit, hash_cache
        )
    except Exception as exc:
        provenance, source_song_id, source_score_sha256 = (
            "invalid",
            f"invalid:{sample}",
            None,
        )
        reject("source_identity_invalid", str(exc))
    fingerprints = (
        _lineage_fingerprints(
            lineage, hashes.get("performance_audio.wav", "")
        )
        if lineage
        else {}
    )
    return {
        "sample": sample,
        "sample_dir": str(sample_dir.resolve()),
        "eligible": not reasons,
        "exclusion_reasons": reasons,
        "provenance": provenance,
        "source_song_id": source_song_id,
        "source_score_sha256": source_score_sha256,
        "duration_sec": result.get("duration_sec"),
        "source_hashes": hashes,
        **fingerprints,
        "target_lineage_sha256": (
            hashlib.sha256(_canonical_bytes(lineage)).hexdigest()
            if lineage
            else None
        ),
        "target_stats": semantics,
        "_lineage": dict(lineage),
    }


def _read_upstream_inventory(
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    artifact = manifest.get("artifacts", {}).get("inventory", {})
    path = Path(str(artifact.get("path") or ""))
    if not path.is_file():
        raise FileNotFoundError(f"Upstream inventory is missing: {path}")
    if sha256_file(path) != str(artifact.get("sha256")):
        raise ValueError("Upstream inventory SHA-256 mismatch")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != int(manifest.get("counts", {}).get("bundles_discovered", -1)):
        raise ValueError("Upstream inventory row count mismatch")
    return rows


def _exposure_identities(
    manifest: Mapping[str, Any],
) -> dict[str, set[str]]:
    output: defaultdict[str, set[str]] = defaultdict(set)
    for row in manifest.get("selected_rows") or []:
        output["sample"].add(str(row["sample"]))
        output["source_song_id"].add(str(row["source_song_id"]))
        for name, value in (row.get("hashes") or {}).items():
            if value:
                output[f"hash:{name}"].add(str(value))
        for name in (
            "content_fingerprint",
            "source_score_sha256",
            "target_lineage_sha256",
        ):
            if row.get(name):
                output[name].add(str(row[name]))
    return dict(output)


def _is_exactly_exposed(
    row: Mapping[str, Any],
    upstream_row: Mapping[str, Any],
    exposure: Mapping[str, set[str]],
) -> list[str]:
    matches = []
    if str(row["sample"]) in exposure.get("sample", set()):
        matches.append("sample")
    for name, value in (row.get("source_hashes") or {}).items():
        if value and str(value) in exposure.get(f"hash:{name}", set()):
            matches.append(f"hash:{name}")
    if (
        upstream_row.get("content_fingerprint")
        and str(upstream_row["content_fingerprint"])
        in exposure.get("content_fingerprint", set())
    ):
        matches.append("content_fingerprint")
    return sorted(set(matches))


def _source_is_exposed(
    row: Mapping[str, Any], exposure: Mapping[str, set[str]]
) -> bool:
    if str(row["source_song_id"]) in exposure.get("source_song_id", set()):
        return True
    source_sha256 = row.get("source_score_sha256")
    return bool(
        source_sha256
        and str(source_sha256) in exposure.get("source_score_sha256", set())
    )


def _outputraw_development_identities(
    target_db: Path, seal_path: Path
) -> dict[str, set[str]]:
    seal = _load_json(seal_path)
    attested = (
        seal.get("artifacts", {})
        .get("canonical_dev_targets.sqlite", {})
        .get("sha256")
    )
    if attested != sha256_file(target_db):
        raise ValueError("Current OutputRaw development target DB is not attested")
    output: defaultdict[str, set[str]] = defaultdict(set)
    connection = sqlite3.connect(f"file:{target_db}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT split,sample_dir,source_hashes,payload FROM targets "
            "WHERE split IN ('train','val')"
        )
        for split, sample_dir, raw_hashes, payload in rows:
            output["sample_dir"].add(os.path.normcase(str(sample_dir)))
            hashes = json.loads(str(raw_hashes))
            for name, value in hashes.items():
                if value:
                    output[f"hash:{name}"].add(str(value))
            lineage = json.loads(zlib.decompress(bytes(payload)))
            fingerprints = _lineage_fingerprints(
                lineage, str(hashes.get("performance_audio.wav") or "")
            )
            for name, value in fingerprints.items():
                output[name].add(value)
            output["split"].add(str(split))
    finally:
        connection.close()
    return dict(output)


def _outputraw_matches(
    row: Mapping[str, Any], identities: Mapping[str, set[str]]
) -> list[str]:
    matches = []
    for name, value in (row.get("source_hashes") or {}).items():
        if value and str(value) in identities.get(f"hash:{name}", set()):
            matches.append(f"hash:{name}")
    for name in (
        "clean_fingerprint",
        "near_lineage_hash",
        "lineage_hash",
        "audio_lineage_hash",
    ):
        if row.get(name) in identities.get(name, set()):
            matches.append(name)
    return sorted(matches)


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _group_records(
    records: Sequence[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Union source, score, content-lineage, and audio duplicate identities."""

    disjoint = _DisjointSet(len(records))
    seen: dict[tuple[str, str], int] = {}
    for index, row in enumerate(records):
        identities = [
            ("verified", row["source_hashes"]["verified_score.musicxml"]),
            ("clean", row["clean_fingerprint"]),
            ("near", row["near_lineage_hash"]),
            ("lineage", row["lineage_hash"]),
            ("audio", row["source_hashes"]["performance_audio.wav"]),
        ]
        if row["provenance"] == "raw":
            identities.append(("raw_source", row["source_score_sha256"]))
        for kind, raw_value in identities:
            value = str(raw_value or "")
            if not value:
                continue
            previous = seen.setdefault((kind, value), index)
            disjoint.union(index, previous)
    by_root: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(records):
        by_root[disjoint.find(index)].append(row)
    output = {}
    for members in by_root.values():
        group_id = hashlib.sha256(
            _canonical_bytes(
                sorted(
                    (
                        row["provenance"],
                        row["source_song_id"],
                        row["source_hashes"]["verified_score.musicxml"],
                        row["clean_fingerprint"],
                        row["lineage_hash"],
                        row["source_hashes"]["performance_audio.wav"],
                    )
                    for row in members
                )
            )
        ).hexdigest()
        for row in members:
            row["leakage_group"] = group_id
        output[group_id] = members
    return output


def _base_split(seed: int, group_id: str) -> str:
    value = _fraction(seed, group_id)
    for split in ("lockbox", "open_validation", "calibration", "train"):
        lower, upper = SPLIT_FRACTIONS[split]
        if lower <= value < upper:
            return split
    raise AssertionError(value)


def _assign_groups(
    groups: Mapping[str, Sequence[dict[str, Any]]],
    *,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    raw_unexposed = [
        group_id
        for group_id, members in groups.items()
        if any(row["provenance"] == "raw" for row in members)
        and not any(row.get("exposed_source") for row in members)
    ]
    raw_lockbox_allowed = len(raw_unexposed) >= MIN_RAW_LOCKBOX_GROUPS
    assigned = {split: [] for split in SPLITS}
    forced_counts: Counter[str] = Counter()
    for group_id, members in sorted(groups.items()):
        reasons = set()
        if any(row.get("exposed_source") for row in members):
            reasons.add("exposed_source")
        if any(row.get("outputraw_development_matches") for row in members):
            reasons.add("current_outputraw_development_identity")
        has_raw = any(row["provenance"] == "raw" for row in members)
        proposed = _base_split(seed, group_id)
        if has_raw and not raw_lockbox_allowed and proposed != "train":
            reasons.add("too_few_independent_raw_sources")
        split = "train" if reasons else proposed
        for row in members:
            row["split"] = split
            row["forced_train_reasons"] = sorted(reasons)
            assigned[split].append(row)
        forced_counts.update(reasons)
    for split, rows in assigned.items():
        rows.sort(key=lambda row: (_stable(seed + len(split), row["sample"]), row["sample"]))
    if any(not assigned[split] for split in SPLITS):
        raise ValueError(
            f"Outcome-blind split produced an empty partition: "
            f"{ {key: len(value) for key, value in assigned.items()} }"
        )
    return assigned, {
        "raw_unexposed_independent_groups": len(raw_unexposed),
        "minimum_raw_groups_for_lockbox": MIN_RAW_LOCKBOX_GROUPS,
        "raw_lockbox_allowed": raw_lockbox_allowed,
        "raw_lockbox_limitation": (
            None
            if raw_lockbox_allowed
            else (
                "Too few outcome-unseen independent raw source scores; every raw "
                "row was forced to training and the lockbox contains generated "
                "source groups only."
            )
        ),
        "forced_train_rows_by_reason": dict(sorted(forced_counts.items())),
    }


def _split_overlap(
    assigned: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    fields = (
        "leakage_group",
        "source_song_id",
        "clean_fingerprint",
        "near_lineage_hash",
        "lineage_hash",
        "audio_lineage_hash",
    )
    fields_with_hash = (
        ("verified_score_sha256", "verified_score.musicxml"),
        ("performance_audio_sha256", "performance_audio.wav"),
    )
    report: dict[str, Any] = {}
    passed = True
    for field in fields:
        membership: defaultdict[str, set[str]] = defaultdict(set)
        for split, rows in assigned.items():
            for row in rows:
                membership[str(row[field])].add(split)
        overlap = {
            value: sorted(splits)
            for value, splits in membership.items()
            if len(splits) > 1
        }
        report[field] = {"overlap_count": len(overlap), "examples": dict(list(overlap.items())[:10])}
        passed &= not overlap
    for field, hash_name in fields_with_hash:
        membership = defaultdict(set)
        for split, rows in assigned.items():
            for row in rows:
                membership[str(row["source_hashes"][hash_name])].add(split)
        overlap = {
            value: sorted(splits)
            for value, splits in membership.items()
            if len(splits) > 1
        }
        report[field] = {"overlap_count": len(overlap), "examples": dict(list(overlap.items())[:10])}
        passed &= not overlap
    report["passed"] = passed
    return report


def _public_development_row(
    row: Mapping[str, Any], target_record: int
) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if not key.startswith("_")
        and key
        not in {
            "exclusion_reasons",
            "target_stats",
        }
    } | {
        "target_archive": "development_targets.jsonl.gz",
        "target_record": target_record,
        "target_stats": row["target_stats"],
    }


def _create_or_load_key(path: Path) -> bytes:
    if path.is_file():
        key = path.read_bytes()
        if len(key) != 32:
            raise ValueError(f"Lockbox key must contain exactly 32 bytes: {path}")
        return key
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    handle = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(handle, key)
        os.fsync(handle)
    finally:
        os.close(handle)
    return key


def _encrypt_targets(
    rows: Sequence[Mapping[str, Any]], key: bytes
) -> bytes:
    plaintext = gzip.compress(
        b"".join(_canonical_bytes(row) + b"\n" for row in rows),
        compresslevel=9,
        mtime=0,
    )
    nonce = secrets.token_bytes(12)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, LOCKBOX_AAD)
    return b"ORNLBX1\0" + nonce + ciphertext


def _decrypt_targets(value: bytes, key: bytes) -> list[dict[str, Any]]:
    if not value.startswith(b"ORNLBX1\0") or len(value) < 20:
        raise ValueError("Unsupported lockbox target envelope")
    plaintext = AESGCM(key).decrypt(value[8:20], value[20:], LOCKBOX_AAD)
    return [
        json.loads(line)
        for line in gzip.decompress(plaintext).splitlines()
        if line.strip()
    ]


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _tree_hash(paths: Sequence[Path], root: Path) -> str:
    return hashlib.sha256(
        _canonical_bytes(
            [
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in sorted(paths)
            ]
        )
    ).hexdigest()


@dataclass(frozen=True)
class FreezePaths:
    output: Path

    @property
    def manifest(self) -> Path:
        return self.output / "release_manifest.json"

    @property
    def status(self) -> Path:
        return self.output / "STATUS.json"

    @property
    def freeze(self) -> Path:
        return self.output / "FREEZE.json"

    @property
    def partial(self) -> Path:
        return self.output / "audit_partial.jsonl"

    @property
    def hash_cache(self) -> Path:
        return self.output / "audit_hash_cache.json"


def _load_partial(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    output = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        output[str(row["sample"])] = row
    return output


def _append_partial(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_freeze(args: argparse.Namespace) -> None:
    paths = FreezePaths(args.output_dir.resolve())
    if paths.freeze.exists() or paths.manifest.exists():
        raise FileExistsError(f"Release is already frozen: {paths.output}")
    paths.output.mkdir(parents=True, exist_ok=True)
    upstream_path = args.upstream_manifest.resolve()
    upstream = _load_json(upstream_path)
    if upstream.get("schema_version") != "align-orn-external-eval-v1":
        raise ValueError("Unsupported upstream ORN audit manifest")
    if not bool(upstream.get("selection_policy", {}).get("outcome_blind")):
        raise ValueError("Upstream ORN inventory was not outcome-blind")
    inventory = _read_upstream_inventory(upstream)
    exposure = _exposure_identities(upstream)
    inventory_by_sample = {str(row["sample"]): row for row in inventory}
    candidate_rows = [
        row
        for row in inventory
        if bool(row.get("eligible"))
        and str(row["sample"]) not in exposure.get("sample", set())
    ]
    if not candidate_rows:
        raise ValueError("No upstream-audited unexposed ORN candidates")
    audit = _load_module(
        args.repo / "scripts" / "audit_training_data.py",
        "orn_generalization_source_audit",
    )
    bootstrap_cache_path = Path(
        str(
            upstream.get("artifacts", {})
            .get("audit_hash_cache", {})
            .get("path")
            or ""
        )
    )
    if paths.hash_cache.is_file():
        hash_cache = _load_json(paths.hash_cache)
    elif (
        bootstrap_cache_path.is_file()
        and sha256_file(bootstrap_cache_path)
        == upstream["artifacts"]["audit_hash_cache"]["sha256"]
    ):
        hash_cache = _load_json(bootstrap_cache_path)
    else:
        hash_cache = {}
    completed = _load_partial(paths.partial)
    source_root = Path(str(upstream["source_root"])).resolve()
    command = [sys.executable, *sys.argv]
    lease = resource_lease(
        args.resource_status,
        "cpu_validation",
        track="orn-generalization-v1-freeze",
        command=command,
        metadata={
            "candidate_rows": len(candidate_rows),
            "locked_test": False,
            "ORN_outcomes_read": False,
        },
    )
    lease.__enter__()
    atexit.register(lease.__exit__, None, None, None)
    try:
        for position, upstream_row in enumerate(candidate_rows, 1):
            sample = str(upstream_row["sample"])
            if sample in completed:
                continue
            row = _independent_audit(
                Path(str(upstream_row["sample_dir"])),
                source_root,
                audit,
                hash_cache,
            )
            exact_matches = _is_exactly_exposed(
                row, upstream_row, exposure
            )
            row["exact_exposure_matches"] = exact_matches
            if exact_matches:
                row["eligible"] = False
                row["exclusion_reasons"].append(
                    {
                        "code": "exposed_content_or_audio",
                        "detail": ", ".join(exact_matches),
                    }
                )
            row["exposed_source"] = (
                _source_is_exposed(row, exposure)
                if row.get("eligible")
                else False
            )
            completed[sample] = row
            _append_partial(paths.partial, row)
            if position % args.checkpoint_every == 0:
                _atomic_json(paths.hash_cache, hash_cache)
                print(
                    f"audit={position}/{len(candidate_rows)} "
                    f"eligible={sum(bool(value.get('eligible')) for value in completed.values())}",
                    flush=True,
                )
        _atomic_json(paths.hash_cache, hash_cache)

        independently_eligible = [
            row for row in completed.values() if bool(row.get("eligible"))
        ]
        outputraw = _outputraw_development_identities(
            args.outputraw_targets.resolve(),
            args.outputraw_seal.resolve(),
        )
        for row in independently_eligible:
            row["outputraw_development_matches"] = _outputraw_matches(
                row, outputraw
            )
        groups = _group_records(independently_eligible)
        assigned, assignment_summary = _assign_groups(
            groups, seed=args.seed
        )
        leakage = _split_overlap(assigned)
        if not leakage["passed"]:
            raise RuntimeError("New ORN split identity leakage detected")
        exposed_lockbox = [
            row["sample"]
            for row in assigned["lockbox"]
            if row.get("exposed_source")
            or row.get("exact_exposure_matches")
        ]
        outputraw_lockbox = [
            row["sample"]
            for row in assigned["lockbox"]
            if row.get("outputraw_development_matches")
        ]
        if exposed_lockbox or outputraw_lockbox:
            raise RuntimeError("Lockbox leakage quarantine failed")

        development_targets_path = paths.output / "development_targets.jsonl.gz"
        development_rows: list[dict[str, Any]] = []
        development_manifest_rows = {
            split: [] for split in SPLITS if split != "lockbox"
        }
        ordinal = 0
        for split in ("train", "calibration", "open_validation"):
            for row in assigned[split]:
                target_row = {
                    "ordinal": ordinal,
                    "split": split,
                    "sample": row["sample"],
                    "sample_dir": row["sample_dir"],
                    "source_hashes": row["source_hashes"],
                    "target_lineage_sha256": row["target_lineage_sha256"],
                    "representation_version": REPRESENTATION_VERSION,
                    "lineage": row["_lineage"],
                }
                development_targets_path_value = ordinal
                development_rows.append(target_row)
                development_manifest_rows[split].append(
                    _public_development_row(
                        row, development_targets_path_value
                    )
                )
                ordinal += 1
        _atomic_jsonl_gz(development_targets_path, development_rows)

        key = _create_or_load_key(args.lockbox_key.resolve())
        lockbox_dir = paths.output / "lockbox"
        lockbox_inputs = lockbox_dir / "inputs"
        lockbox_manifest_rows = []
        encrypted_target_rows = []
        for position, row in enumerate(assigned["lockbox"]):
            row_id = hashlib.sha256(
                f"{SCHEMA_VERSION}:{args.seed}:{row['leakage_group']}:{row['sample']}".encode(
                    "utf-8"
                )
            ).hexdigest()[:24]
            audio_source = Path(row["sample_dir"]) / "performance_audio.wav"
            score_source = Path(row["sample_dir"]) / "verified_score.musicxml"
            audio_copy = lockbox_inputs / f"{row_id}.wav"
            score_copy = lockbox_inputs / f"{row_id}.musicxml"
            if not audio_copy.is_file():
                _atomic_copy(audio_source, audio_copy)
            if not score_copy.is_file():
                _atomic_copy(score_source, score_copy)
            if (
                sha256_file(audio_copy)
                != row["source_hashes"]["performance_audio.wav"]
                or sha256_file(score_copy)
                != row["source_hashes"]["verified_score.musicxml"]
            ):
                raise ValueError(f"Lockbox copied input mismatch: {row_id}")
            lockbox_manifest_rows.append(
                {
                    "ordinal": position,
                    "row_id": row_id,
                    "provenance": row["provenance"],
                    "leakage_group": row["leakage_group"],
                    "duration_sec": row["duration_sec"],
                    "audio": _artifact(audio_copy),
                    "score": _artifact(score_copy),
                }
            )
            encrypted_target_rows.append(
                {
                    "ordinal": position,
                    "row_id": row_id,
                    "sample": row["sample"],
                    "sample_dir": row["sample_dir"],
                    "source_hashes": row["source_hashes"],
                    "target_lineage_sha256": row[
                        "target_lineage_sha256"
                    ],
                    "representation_version": REPRESENTATION_VERSION,
                    "lineage": row["_lineage"],
                }
            )
        encrypted_path = lockbox_dir / "targets.aes256gcm"
        encrypted = _encrypt_targets(encrypted_target_rows, key)
        _atomic_bytes(encrypted_path, encrypted)
        decrypted_check = _decrypt_targets(encrypted, key)
        if decrypted_check != encrypted_target_rows:
            raise RuntimeError("Lockbox target encryption round-trip failed")
        lockbox_manifest_path = lockbox_dir / "manifest.json"
        lockbox_manifest = {
            "schema_version": f"{SCHEMA_VERSION}-lockbox-manifest",
            "created_utc": _utc(),
            "rows": len(lockbox_manifest_rows),
            "inputs": lockbox_manifest_rows,
            "target_envelope": {
                **_artifact(encrypted_path),
                "algorithm": "AES-256-GCM",
                "aad": LOCKBOX_AAD.decode("ascii"),
                "key_sha256": hashlib.sha256(key).hexdigest(),
                "key_stored_outside_repository": True,
            },
            "target_content_public": False,
            "source_bundle_paths_public": False,
        }
        _atomic_json(lockbox_manifest_path, lockbox_manifest)

        exclusion_rows = []
        for upstream_row in inventory:
            sample = str(upstream_row["sample"])
            checked = completed.get(sample)
            if checked is not None and not checked.get("eligible"):
                exclusion_rows.append(
                    {
                        "sample": sample,
                        "stage": "independent_semantic_audit",
                        "reasons": checked.get("exclusion_reasons") or [],
                    }
                )
            elif not bool(upstream_row.get("eligible")):
                exclusion_rows.append(
                    {
                        "sample": sample,
                        "stage": "upstream_outcome_blind_audit",
                        "reasons": upstream_row.get("exclusion_reasons")
                        or [{"code": "upstream_ineligible"}],
                    }
                )
            elif sample in exposure.get("sample", set()):
                exclusion_rows.append(
                    {
                        "sample": sample,
                        "stage": "exposed_orn500_quarantine",
                        "reasons": [{"code": "exposed_orn500_row"}],
                    }
                )
        exclusions_path = paths.output / "exclusions.jsonl.gz"
        _atomic_jsonl_gz(exclusions_path, exclusion_rows)

        public_inventory_path = paths.output / "inventory.jsonl.gz"
        public_inventory = []
        lockbox_samples = {row["sample"] for row in assigned["lockbox"]}
        for row in completed.values():
            if row["sample"] in lockbox_samples:
                public_inventory.append(
                    {
                        "sample": "[sealed]",
                        "eligible": True,
                        "split": "lockbox",
                        "provenance": row["provenance"],
                        "leakage_group": row["leakage_group"],
                        "target_content_public": False,
                    }
                )
            else:
                public_inventory.append(
                    {
                        key: value
                        for key, value in row.items()
                        if not key.startswith("_")
                    }
                )
        _atomic_jsonl_gz(public_inventory_path, public_inventory)

        leakage_path = paths.output / "leakage_report.json"
        leakage_report = {
            "schema_version": f"{SCHEMA_VERSION}-leakage-report",
            "created_utc": _utc(),
            "new_split_identity_overlap": leakage,
            "lockbox_exposed_orn500_rows_or_sources": len(
                exposed_lockbox
            ),
            "lockbox_current_outputraw_development_matches": len(
                outputraw_lockbox
            ),
            "outputraw_identity_source": {
                "path": str(args.outputraw_targets.resolve()),
                "splits_read": ["train", "val"],
                "original_mapper_v6_test_read": False,
            },
            "exposed_orn_identity_source": {
                "path": str(upstream_path),
                "fields_read": [
                    "selected sample IDs",
                    "source-song IDs",
                    "content hashes",
                    "audio hashes",
                ],
                "prediction_or_outcome_fields_read": False,
            },
            "passed": (
                leakage["passed"]
                and not exposed_lockbox
                and not outputraw_lockbox
            ),
        }
        _atomic_json(leakage_path, leakage_report)

        split_counts = {
            split: len(rows) for split, rows in assigned.items()
        }
        source_counts = {
            split: len({row["leakage_group"] for row in rows})
            for split, rows in assigned.items()
        }
        provenance_counts = {
            split: dict(
                sorted(Counter(row["provenance"] for row in rows).items())
            )
            for split, rows in assigned.items()
        }
        protocol_path = paths.output / "protocol.json"
        protocol = {
            "schema_version": f"{SCHEMA_VERSION}-protocol",
            "created_utc": _utc(),
            "official_metric": {
                "unit": "canonical score-note/event identity",
                "matching": "exclusive one-to-one",
                "credit": {
                    "exact_location_and_type": 1.0,
                    "exact_location_wrong_type": 0.5,
                    "wrong_location": 0.0,
                },
                "reported": [
                    "micro precision/recall/F1",
                    "95% source bootstrap CI",
                    "per-type F1 and support",
                    "source macro F1",
                ],
                "timestamps": "diagnostic only",
            },
            "development_gate": {
                "combined_open_validation_f1_minimum": 0.85,
                "oracle_mapper_and_transcriber_diagnostics_required": True,
                "candidate_artifact_config_and_hash_frozen": True,
            },
            "sealed_evaluation": {
                "openings_allowed": 1,
                "preconditions": [
                    "development gate attained",
                    "candidate artifact/config/hash frozen",
                    "lockbox predictions frozen from public inference inputs",
                    "all release and input hashes verified",
                ],
                "post_test_tuning_allowed": False,
                "if_gate_not_attained": "leave lockbox sealed",
                "required_open_sentinel": "lockbox/LOCKBOX_OPENED.json",
                "required_result": "lockbox/LOCKBOX_RESULT.json",
            },
            "population": {
                "membership_selected_without_predictions_or outcomes": True,
                "raw_lockbox_policy": assignment_summary[
                    "raw_lockbox_limitation"
                ],
                "split_fractions_by_group": SPLIT_FRACTIONS,
                "seed": args.seed,
            },
        }
        _atomic_json(protocol_path, protocol)

        input_paths = [
            path
            for path in lockbox_inputs.iterdir()
            if path.is_file()
        ]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "release": "orn-generalization-v1",
            "phase": "frozen_before_model_iteration",
            "created_utc": _utc(),
            "command": command,
            "seed": args.seed,
            "source_root": str(source_root),
            "outcome_isolation": {
                "upstream_manifest_only": str(upstream_path),
                "upstream_manifest_sha256": sha256_file(upstream_path),
                "upstream_inventory_sha256": upstream["artifacts"][
                    "inventory"
                ]["sha256"],
                "ORN_prediction_files_read": False,
                "ORN_report_or_status_outcomes_read": False,
                "original_mapper_v6_test_read": False,
            },
            "eligibility": {
                "policy": "deterministic fail-closed",
                "upstream_bundles": len(inventory),
                "upstream_eligible_candidates": sum(
                    bool(row.get("eligible")) for row in inventory
                ),
                "exposed_exact_rows_removed_before_reaudit": len(
                    exposure.get("sample", set())
                ),
                "independently_reaudited": len(completed),
                "independently_eligible": len(independently_eligible),
                "independently_excluded": len(completed)
                - len(independently_eligible),
                "checks": [
                    "required files and schema 1.2",
                    "readable nonempty WAV and MIDI",
                    "all audible MIDI events inside WAV",
                    "Bb sounding-to-written +2 convention",
                    "valid label spans",
                    "exact clean/performed MusicXML lineage",
                    "one-to-one performed-to-rendered ownership",
                    "versioned raw-MIDI rendered ornament representation",
                    "canonical ScoreEventIndex projection",
                    "consistent repeat/COPY representation",
                ],
            },
            "grouping": {
                "raw": "authoritative source-score SHA-256",
                "generated": (
                    "connected components over verified-score, clean content, "
                    "transposition-invariant content, performed lineage, and "
                    "performance-audio SHA-256"
                ),
                "groups": len(groups),
            },
            "splits": {
                "counts": split_counts,
                "source_group_counts": source_counts,
                "provenance_counts": provenance_counts,
                "assignment": assignment_summary,
                "development": development_manifest_rows,
                "lockbox_manifest": _artifact(lockbox_manifest_path),
            },
            "artifacts": {
                "development_targets": _artifact(
                    development_targets_path
                ),
                "exclusions": _artifact(exclusions_path),
                "inventory": _artifact(public_inventory_path),
                "leakage_report": _artifact(leakage_path),
                "protocol": _artifact(protocol_path),
                "audit_hash_cache": _artifact(paths.hash_cache),
                "lockbox_targets_encrypted": _artifact(encrypted_path),
                "lockbox_input_tree": {
                    "path": str(lockbox_inputs.resolve()),
                    "files": len(input_paths),
                    "sha256": _tree_hash(input_paths, paths.output),
                    "bytes": sum(path.stat().st_size for path in input_paths),
                },
            },
            "lockbox": {
                "sealed": True,
                "target_content_public": False,
                "target_key_stored_outside_repository": True,
                "target_key_sha256": hashlib.sha256(key).hexdigest(),
                "openings_allowed": 1,
                "opened": False,
            },
            "metric_gate": {
                "combined_canonical_note_wise_f1": ">=0.85",
                "test_if_gate_fails": "remain sealed",
                "post_test_tuning": "forbidden",
            },
            "production_weights_mutated": False,
            "paper_updated": False,
        }
        _atomic_json(paths.manifest, manifest)
        freeze = {
            "schema_version": f"{SCHEMA_VERSION}-freeze",
            "frozen_utc": _utc(),
            "release_manifest": _artifact(paths.manifest),
            "protocol": _artifact(protocol_path),
            "leakage_report": _artifact(leakage_path),
            "lockbox_manifest": _artifact(lockbox_manifest_path),
            "lockbox_targets_encrypted": _artifact(encrypted_path),
            "lockbox_input_tree_sha256": manifest["artifacts"][
                "lockbox_input_tree"
            ]["sha256"],
            "source_membership_frozen_before_training": True,
            "ORN_outcomes_used_for_membership": False,
            "original_mapper_v6_test_read": False,
        }
        _atomic_json(paths.freeze, freeze)
        status = {
            "schema_version": f"{SCHEMA_VERSION}-status",
            "status": "frozen_ready_for_development",
            "completed_utc": _utc(),
            "release_manifest": _artifact(paths.manifest),
            "freeze": _artifact(paths.freeze),
            "split_counts": split_counts,
            "source_group_counts": source_counts,
            "provenance_counts": provenance_counts,
            "raw_lockbox_limitation": assignment_summary[
                "raw_lockbox_limitation"
            ],
            "leakage_checks_passed": leakage_report["passed"],
            "development_target_f1": 0.85,
            "lockbox_opened": False,
            "next_phase": (
                "Train and tune on train/calibration/open_validation only; "
                "leave lockbox targets and key unopened."
            ),
            "commands": {
                "freeze": command,
                "resume_if_interrupted": command,
                "verify": [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "verify",
                    "--output-dir",
                    str(paths.output),
                ],
            },
            "production_weights_mutated": False,
            "paper_updated": False,
        }
        _atomic_json(paths.status, status)
        try:
            paths.partial.unlink()
        except FileNotFoundError:
            pass
        print(
            json.dumps(
                {
                    "status": status["status"],
                    "split_counts": split_counts,
                    "source_group_counts": source_counts,
                    "raw_lockbox_limitation": status[
                        "raw_lockbox_limitation"
                    ],
                    "manifest": str(paths.manifest),
                },
                indent=2,
            ),
            flush=True,
        )
    finally:
        atexit.unregister(lease.__exit__)
        lease.__exit__(None, None, None)


def run_verify(args: argparse.Namespace) -> None:
    paths = FreezePaths(args.output_dir.resolve())
    freeze = _load_json(paths.freeze)
    manifest = _load_json(paths.manifest)
    status = _load_json(paths.status)
    failures = []
    if (
        freeze.get("release_manifest", {}).get("sha256")
        != sha256_file(paths.manifest)
    ):
        failures.append("release_manifest_sha256")
    for name, artifact in manifest.get("artifacts", {}).items():
        if name == "lockbox_input_tree":
            root = Path(str(artifact["path"]))
            files = [path for path in root.iterdir() if path.is_file()]
            if _tree_hash(files, paths.output) != artifact["sha256"]:
                failures.append(name)
            continue
        path = Path(str(artifact.get("path") or ""))
        if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
            failures.append(name)
    lockbox_manifest_artifact = manifest["splits"]["lockbox_manifest"]
    lockbox_manifest_path = Path(lockbox_manifest_artifact["path"])
    if (
        not lockbox_manifest_path.is_file()
        or sha256_file(lockbox_manifest_path)
        != lockbox_manifest_artifact["sha256"]
    ):
        failures.append("lockbox_manifest")
    result = {
        "schema_version": f"{SCHEMA_VERSION}-verification",
        "verified_utc": _utc(),
        "release_manifest_sha256": sha256_file(paths.manifest),
        "status_sha256": sha256_file(paths.status),
        "failures": failures,
        "passed": not failures,
        "lockbox_opened": (paths.output / "lockbox" / "LOCKBOX_OPENED.json").exists(),
        "original_mapper_v6_test_read": False,
    }
    print(json.dumps(result, indent=2), flush=True)
    if failures:
        raise SystemExit(1)
    if status.get("status") != "frozen_ready_for_development":
        raise ValueError("Release status is not development-ready")


def _resolve(repo: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo / value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "verify"))
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--upstream-manifest", type=Path, default=DEFAULT_UPSTREAM
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--outputraw-targets", type=Path, default=DEFAULT_OUTPUTRAW_TARGETS
    )
    parser.add_argument(
        "--outputraw-seal", type=Path, default=DEFAULT_OUTPUTRAW_SEAL
    )
    parser.add_argument(
        "--resource-status", type=Path, default=DEFAULT_RESOURCE_STATUS
    )
    parser.add_argument("--lockbox-key", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.repo = args.repo.resolve()
    args.upstream_manifest = _resolve(args.repo, args.upstream_manifest)
    args.output_dir = _resolve(args.repo, args.output_dir)
    args.outputraw_targets = _resolve(args.repo, args.outputraw_targets)
    args.outputraw_seal = _resolve(args.repo, args.outputraw_seal)
    args.resource_status = _resolve(args.repo, args.resource_status)
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")
    if args.phase == "freeze":
        if args.lockbox_key is None:
            raise ValueError("--lockbox-key outside the repository is required")
        args.lockbox_key = args.lockbox_key.resolve()
        try:
            args.lockbox_key.relative_to(args.repo)
        except ValueError:
            pass
        else:
            raise ValueError("Lockbox key must be outside the repository")
        run_freeze(args)
    else:
        run_verify(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
