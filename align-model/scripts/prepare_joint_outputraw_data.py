"""Build the audited, packed outputRaw-only foundation for full-joint training.

The protected test split is sealed before feature packing.  Every operation
after sealing is restricted to train/validation rows, and the test manifest
contains metadata only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
import time
import zlib
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from alignmodel.joint.candidates import (
    CANDIDATE_GENERATION_VERSION,
    add_score_repeat_hints,
    basic_pitch_candidate_union,
)
from alignmodel.joint.data import JointTrainingExample
from alignmodel.joint.index import JointEvent, ScoreEventIndex
from alignmodel.joint.metrics import pair_exact_pitch_onset
from alignmodel.joint.packed_data import (
    ExistingShardMismatch,
    PACK_SCHEMA_VERSION,
    PackedJointDataset,
    PackedJointWriter,
    canonical_target,
    sha256_file,
    validate_split_manifest,
)
from alignmodel.transcription.basic_pitch import (
    BasicPitchFeatures,
    basic_pitch_cache_path,
    load_audio_metadata,
    load_basic_pitch_cache,
)


RELEASE = "joint-outputraw-full-v1"
AUDIT_SCHEMA = "align-outputraw-joint-audit-v1"
DEFAULT_SEED = 20260915
SOURCE_ROOT = Path("E:/outputRaw_sf_10k")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _stable(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _audit_module(repo: Path) -> Any:
    script = repo / "align-model" / "scripts" / "audit_training_data.py"
    spec = importlib.util.spec_from_file_location("align_repaired_audit", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import repaired audit tool: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _norm(path: Path | str) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _history_inventory(repo: Path) -> dict[str, list[str]]:
    """Conservatively inventory every historically evaluated raw-score id."""

    runs = repo / "align-model" / "runs"
    evidence: dict[str, set[str]] = defaultdict(set)
    markers = (
        "eval",
        "calibr",
        "benchmark",
        "oracle",
        "diagnos",
        "sweep",
    )
    if not runs.is_dir():
        return {}
    for path in runs.rglob("synth_*.json"):
        relative = str(path.relative_to(runs)).casefold()
        if any(marker in relative for marker in markers):
            evidence[path.stem].add(str(path))
    for manifest_path in runs.rglob("split.json"):
        try:
            document = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(document, Mapping):
            continue
        for split in ("val", "test", "test_id", "test_ood", "calibration", "eval"):
            rows = document.get(split)
            if not isinstance(rows, list):
                continue
            for raw in rows[:200]:
                if not isinstance(raw, Mapping):
                    continue
                name = Path(
                    str(
                        raw.get("sample")
                        or raw.get("sample_dir")
                        or raw.get("path")
                        or ""
                    )
                ).name
                if name.startswith(("synth_MozartClConcertoA_", "synth_WeberITAV_")):
                    evidence[name].add(f"{manifest_path}#{split}:first200")
    return {
        sample: sorted(paths)
        for sample, paths in sorted(evidence.items())
    }


def _rounded(value: Any) -> float:
    return round(float(value), 6)


def _fingerprints(record: Mapping[str, Any]) -> dict[str, str]:
    lineage = record.get("_note_map_document") or {}
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
    near_payload = [
        [row[0] - first_pitch, row[2]]
        for row in clean_payload
    ]
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
    clean_hash = hashlib.sha256(_canonical_json(clean_payload)).hexdigest()
    near_hash = hashlib.sha256(_canonical_json(near_payload)).hexdigest()
    lineage_hash = hashlib.sha256(_canonical_json(lineage_payload)).hexdigest()
    audio_lineage_hash = hashlib.sha256(
        (
            str((record.get("hashes") or {}).get("performance_audio.wav") or "")
            + ":"
            + lineage_hash
        ).encode("ascii")
    ).hexdigest()
    return {
        "clean_fingerprint": clean_hash,
        "near_lineage_hash": near_hash,
        "lineage_hash": lineage_hash,
        "audio_lineage_hash": audio_lineage_hash,
    }


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _group_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Union all exact and near-duplicate symbolic/audio identities."""

    dsu = _DisjointSet(len(records))
    seen: dict[tuple[str, str, str], int] = {}
    for index, record in enumerate(records):
        policy = "|".join(
            (
                str(record["source"]).casefold(),
                str(record["audio_render"]),
                str(record["audio_pitch_space"]),
                str(record["midi_pitch_space"]),
                str(record["effective_audio_transpose"]),
            )
        )
        snippet = record.get("snippet_measures")
        identities = [
            ("clean", record["clean_fingerprint"]),
            ("near", record["near_lineage_hash"]),
            ("lineage", record["lineage_hash"]),
            ("audio_lineage", record["audio_lineage_hash"]),
        ]
        if snippet:
            identities.append(("snippet", f"{snippet[0]}:{snippet[1]}"))
        for kind, value in identities:
            key = (policy, kind, str(value))
            previous = seen.setdefault(key, index)
            dsu.union(index, previous)
    groups_by_root: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        groups_by_root[dsu.find(index)].append(record)
    output: dict[str, list[dict[str, Any]]] = {}
    for members in groups_by_root.values():
        identity = hashlib.sha256(
            _canonical_json(
                sorted(
                    (
                        row["source"],
                        row["clean_fingerprint"],
                        row["near_lineage_hash"],
                        row["lineage_hash"],
                        row["audio_lineage_hash"],
                        row["sample"],
                    )
                    for row in members
                )
            )
        ).hexdigest()
        for row in members:
            row["leakage_group"] = identity
        output[identity] = members
    return output


def _manifest_row(
    record: Mapping[str, Any],
    split: str,
    *,
    target_db: Path | None = None,
    target_record: int | None = None,
) -> dict[str, Any]:
    row = {
        "sample": record["sample"],
        "sample_dir": record["sample_dir"],
        "corpus": "rawsf10k",
        "source": record["source"],
        "split": split,
        "snippet_measures": record["snippet_measures"],
        "audio_render": record["audio_render"],
        "audio_pitch_space": record["audio_pitch_space"],
        "midi_pitch_space": record["midi_pitch_space"],
        "effective_audio_transpose": record["effective_audio_transpose"],
        "duration_sec": record["duration_sec"],
        "error_types": record["error_types"],
        "clean_fingerprint": record["clean_fingerprint"],
        "near_lineage_hash": record["near_lineage_hash"],
        "lineage_hash": record["lineage_hash"],
        "audio_lineage_hash": record["audio_lineage_hash"],
        "leakage_group": record["leakage_group"],
        "source_hashes": record["hashes"],
        "intonation_supervision": False,
    }
    if target_db is not None and target_record is not None:
        row["target_db"] = str(target_db)
        row["target_record"] = int(target_record)
    return row


def _write_dev_targets(
    path: Path,
    assigned: Mapping[str, Sequence[dict[str, Any]]],
) -> dict[tuple[str, str], int]:
    connection = sqlite3.connect(path)
    positions: dict[tuple[str, str], int] = {}
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(
            "CREATE TABLE targets("
            "ordinal INTEGER PRIMARY KEY, split TEXT NOT NULL, corpus TEXT NOT NULL, "
            "sample_dir TEXT UNIQUE NOT NULL, source_hashes TEXT NOT NULL, "
            "payload BLOB NOT NULL)"
        )
        ordinal = 0
        for split in ("train", "val"):
            for record in assigned[split]:
                lineage = dict(record["_note_map_document"])
                lineage.update(
                    {
                        "ordinal": ordinal,
                        "split": split,
                        "corpus": "rawsf10k",
                        "sample_dir": _norm(record["sample_dir"]),
                        "source_hashes": record["hashes"],
                        "valid_labels": record.get("_valid_labels") or [],
                    }
                )
                connection.execute(
                    "INSERT INTO targets VALUES(?,?,?,?,?,?)",
                    (
                        ordinal,
                        split,
                        "rawsf10k",
                        _norm(record["sample_dir"]),
                        json.dumps(record["hashes"], sort_keys=True),
                        sqlite3.Binary(zlib.compress(_canonical_json(lineage), level=1)),
                    ),
                )
                positions[(split, record["sample"])] = ordinal
                ordinal += 1
        connection.commit()
    finally:
        connection.close()
    return positions


def _public_record(record: Mapping[str, Any]) -> dict[str, Any]:
    omitted = {"_labels", "_valid_labels", "_note_map_document", "intonation_checks"}
    return {
        key: value
        for key, value in record.items()
        if key not in omitted
    }


def build_audit_release(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    root = args.source_root.resolve()
    destination = args.audit_dir.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    bundles = sorted(
        {
            path.parent
            for name in ("metadata.json", "labels.json")
            for path in root.rglob(name)
        },
        key=_norm,
    )
    if len(bundles) != 10_000:
        raise ValueError(f"Expected exactly 10,000 outputRaw bundles, found {len(bundles)}")
    repaired = _audit_module(repo)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        hash_cache = {}
        prior_hashes = (
            repo
            / "align-model"
            / "data-audit"
            / "2026-09-14-v2"
            / "hash_cache.json"
        )
        if prior_hashes.is_file():
            hash_cache = json.loads(prior_hashes.read_text(encoding="utf-8"))
        issues = repaired.IssueLog()
        records: list[dict[str, Any]] = []
        for index, sample_dir in enumerate(bundles, 1):
            record = repaired._audit_bundle(
                "rawsf10k",
                root,
                0,
                sample_dir,
                {},
                {},
                hash_cache,
                issues,
            )
            if record["eligible"]:
                record.update(_fingerprints(record))
            records.append(record)
            if index == 1 or index % 250 == 0 or index == len(bundles):
                print(f"audit {index}/{len(bundles)}", flush=True)

        eligible = [row for row in records if row["eligible"]]
        groups = _group_records(eligible)
        historical = _history_inventory(repo)
        assigned: dict[str, list[dict[str, Any]]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        excluded_groups: list[dict[str, Any]] = []
        for group_id, members in sorted(groups.items()):
            history_members = sorted(
                row["sample"] for row in members if row["sample"] in historical
            )
            source = str(members[0]["source"]).casefold()
            if source == "weberitav":
                if history_members:
                    excluded_groups.append(
                        {
                            "leakage_group": group_id,
                            "source": members[0]["source"],
                            "group_rows": len(members),
                            "historical_samples": history_members,
                            "excluded_samples": sorted(row["sample"] for row in members),
                            "evidence": {
                                sample: historical[sample] for sample in history_members
                            },
                        }
                    )
                    continue
                split = "test"
            else:
                # Prior evaluations cannot contaminate validation selection.
                fraction = int(_stable(args.seed, group_id)[:16], 16) / float(16**16)
                split = "train" if history_members or fraction >= 0.10 else "val"
            assigned[split].extend(members)
        for split in assigned:
            assigned[split].sort(
                key=lambda row: _stable(args.seed + len(split), row["sample"])
            )
        if not all(assigned.values()):
            raise ValueError(
                f"Generated an empty split: { {key: len(value) for key, value in assigned.items()} }"
            )

        final_target_db = destination / "canonical_dev_targets.sqlite"
        target_db = staging / final_target_db.name
        positions = _write_dev_targets(target_db, assigned)
        manifest = {
            "schema_version": AUDIT_SCHEMA,
            "release": RELEASE,
            "created_utc": _utc(),
            "seed": args.seed,
            "source_root": str(root),
            "policy": {
                "test": (
                    "WeberITAV work-level holdout, minus every component touching "
                    "a historically evaluated sample id"
                ),
                "validation": (
                    "10% deterministic Mozart fingerprint components; historical "
                    "evaluation components are forced to train"
                ),
                "grouping": [
                    "clean-score note fingerprint",
                    "source work and snippet measure range",
                    "render and pitch policy",
                    "transposition-invariant near-lineage hash",
                    "exact performed-lineage hash",
                    "audio-plus-lineage hash",
                ],
                "two_work_constraint": (
                    "Three source-work-disjoint splits are impossible with two "
                    "works; validation is snippet/fingerprint-disjoint while test "
                    "is source-work-disjoint."
                ),
                "intonation": "masked for every row",
                "locked_test_materialization": "metadata only; no targets or features",
            },
            "train": [
                _manifest_row(
                    row,
                    "train",
                    target_db=final_target_db,
                    target_record=positions[("train", row["sample"])],
                )
                for row in assigned["train"]
            ],
            "val": [
                _manifest_row(
                    row,
                    "val",
                    target_db=final_target_db,
                    target_record=positions[("val", row["sample"])],
                )
                for row in assigned["val"]
            ],
            "test": [_manifest_row(row, "test") for row in assigned["test"]],
            "excluded": {
                "audit_ineligible": len(records) - len(eligible),
                "historical_lockbox_groups": len(excluded_groups),
                "historical_lockbox_rows": sum(
                    row["group_rows"] for row in excluded_groups
                ),
            },
        }
        validate_split_manifest(manifest)
        manifest_path = staging / "split.json"
        _atomic_json(manifest_path, manifest)
        exclusions_path = staging / "historical_lockbox_exclusions.json"
        _atomic_json(
            exclusions_path,
            {
                "policy": (
                    "All matching evaluated ids and their complete near-duplicate "
                    "components are absent from the lockbox."
                ),
                "historical_ids_discovered": len(historical),
                "groups": excluded_groups,
            },
        )
        bundle_index = staging / "bundle_index.jsonl"
        with bundle_index.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(
                    json.dumps(
                        _public_record(record),
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        _atomic_json(staging / "hash_cache.json", hash_cache)
        issue_rows = issues.report()
        report = {
            "schema_version": AUDIT_SCHEMA,
            "release": RELEASE,
            "created_utc": _utc(),
            "source_root": str(root),
            "bundles_discovered": len(records),
            "eligible": len(eligible),
            "ineligible": len(records) - len(eligible),
            "split_counts": {key: len(value) for key, value in assigned.items()},
            "source_counts": {
                split: dict(Counter(row["source"] for row in rows))
                for split, rows in assigned.items()
            },
            "fingerprint_groups": len(groups),
            "historical_ids_discovered": len(historical),
            "historical_lockbox_groups_excluded": len(excluded_groups),
            "historical_lockbox_rows_excluded": sum(
                row["group_rows"] for row in excluded_groups
            ),
            "intonation_targets_usable": 0,
            "issues": issue_rows,
            "issue_counts": {
                row["code"]: row["count"] for row in issue_rows
            },
            "tie_projection": (
                "MusicXML tie elements are projected later by ScoreEventIndex; "
                "slur spanners are never treated as ties."
            ),
        }
        _atomic_json(staging / "audit_report.json", report)
        artifacts = {
            name: {
                "path": str(destination / name),
                "sha256": sha256_file(staging / name),
                "size": (staging / name).stat().st_size,
            }
            for name in (
                "split.json",
                "canonical_dev_targets.sqlite",
                "bundle_index.jsonl",
                "historical_lockbox_exclusions.json",
                "audit_report.json",
                "hash_cache.json",
            )
        }
        protocol = {
            "schema_version": "align-outputraw-lockbox-v1",
            "release": RELEASE,
            "sealed_utc": _utc(),
            "locked_split": "test",
            "rules": [
                "Never materialize test features or targets before model freeze.",
                "Never inspect test audio-derived features after this seal.",
                "Use train/val only for thresholds, calibration, and model selection.",
                "A final test evaluation requires verifying every artifact hash.",
            ],
            "artifacts": artifacts,
            "split_counts": report["split_counts"],
            "success_target": {
                "metric": "combined_full_joint_micro_f1",
                "threshold": 0.80,
                "status": "target_not_demonstrated",
            },
        }
        _atomic_json(staging / "LOCKBOX_SEALED.json", protocol)
        os.replace(staging, destination)
        print(json.dumps(report, indent=2), flush=True)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _verify_audit_release(audit_dir: Path) -> tuple[dict[str, Any], str]:
    protocol_path = audit_dir / "LOCKBOX_SEALED.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "align-outputraw-lockbox-v1":
        raise ValueError("Unsupported lockbox protocol")
    for name, expected in protocol["artifacts"].items():
        path = audit_dir / name
        if path.stat().st_size != int(expected["size"]):
            raise ValueError(f"Audit artifact size mismatch: {name}")
        if sha256_file(path) != expected["sha256"]:
            raise ValueError(f"Audit artifact hash mismatch: {name}")
    manifest_path = audit_dir / "split.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_split_manifest(manifest)
    return manifest, sha256_file(manifest_path)


def _load_target(row: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(row["target_db"]))
    connection = sqlite3.connect(
        f"file:{path.resolve().as_posix()}?mode=ro", uri=True
    )
    try:
        saved = connection.execute(
            "SELECT split,corpus,sample_dir,source_hashes,payload "
            "FROM targets WHERE ordinal=?",
            (int(row["target_record"]),),
        ).fetchone()
    finally:
        connection.close()
    if saved is None:
        raise KeyError(row["target_record"])
    if saved[0] != row["split"] or saved[2] != _norm(row["sample_dir"]):
        raise ValueError("Canonical target row identity mismatch")
    hashes = json.loads(saved[3])
    if hashes != row["source_hashes"]:
        raise ValueError("Canonical target source hash mismatch")
    payload = json.loads(zlib.decompress(saved[4]))
    if payload["source_hashes"] != row["source_hashes"]:
        raise ValueError("Canonical target payload hash mismatch")
    return payload


def _load_features(
    row: Mapping[str, Any],
    cache_root: Path,
) -> BasicPitchFeatures:
    sample_dir = Path(str(row["sample_dir"]))
    metadata = load_audio_metadata(sample_dir)
    path = basic_pitch_cache_path(cache_root, sample_dir, "rawsf10k")
    features = load_basic_pitch_cache(
        path,
        sample_dir / "performance_audio.wav",
        metadata,
        wav_sha256=str(row["source_hashes"]["performance_audio.wav"]),
    )
    if features is None:
        raise ValueError(f"Missing/stale immutable Basic Pitch cache: {path}")
    return features


def _candidate_event(candidate: Any) -> JointEvent:
    return JointEvent(
        pitch=candidate.pitch,
        start=candidate.start,
        end=candidate.end,
        score_span=None,
        relationship="extra",
        confidence=candidate.confidence,
    )


def _build_example(
    row: Mapping[str, Any],
    cache_root: Path,
) -> tuple[BasicPitchFeatures, JointTrainingExample, dict[str, Any]]:
    lineage = _load_target(row)
    sample_dir = Path(str(row["sample_dir"]))
    index = ScoreEventIndex.from_musicxml(
        sample_dir / "verified_score.musicxml",
        lineage,
    )
    features = _load_features(row, cache_root)
    candidates = tuple(
        add_score_repeat_hints(
            basic_pitch_candidate_union(features, minimum_confidence=0.0),
            index.events,
        )
    )
    pairs = pair_exact_pitch_onset(
        tuple(_candidate_event(value) for value in candidates),
        index.rendered_events,
        tolerance_sec=0.050,
    )
    gold_spans: list[tuple[int, int] | None] = [None] * len(candidates)
    gold_keep = [False] * len(candidates)
    for candidate_index, target_index in pairs:
        gold_spans[candidate_index] = index.rendered_events[target_index].score_span
        gold_keep[candidate_index] = True
    example = JointTrainingExample(
        sample=str(row["sample"]),
        source=str(row["source"]),
        candidates=candidates,
        score=index.events,
        gold_spans=tuple(gold_spans),
        gold_keep_unlinked=tuple(gold_keep),
        target_events=index.rendered_events,
        target_deletions=index.deleted_event_indices,
    )
    return features, example, lineage


def pack_release(args: argparse.Namespace) -> dict[str, Any]:
    audit_dir = args.audit_dir.resolve()
    destination = args.pack_root.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    manifest, manifest_hash = _verify_audit_release(audit_dir)
    rows = [*manifest["train"], *manifest["val"]]
    staged = sorted(
        destination.parent.glob(f".{destination.name}.*"),
        key=lambda path: path.stat().st_mtime,
    )
    if len(staged) > 1:
        raise ValueError(
            "Multiple crash-staged packs require manual identity review: "
            f"{[str(path) for path in staged]}"
        )
    staging = staged[0] if staged else None
    recovery: dict[str, Any] = {
        "resumed_staging": str(staging) if staging is not None else None,
        "committed_prefix": 0,
        "reused_records": 0,
        "append_cursor": 0,
    }
    if staging is not None and (staging / "index.sqlite-journal").is_file():
        recovery_dir = audit_dir / "crash-recovery"
        recovery_dir.mkdir(parents=True, exist_ok=True)
        snapshot = recovery_dir / "initial-hot-index"
        if not snapshot.exists():
            snapshot.mkdir()
            for name in ("index.sqlite", "index.sqlite-journal"):
                source = staging / name
                if source.is_file():
                    shutil.copy2(source, snapshot / name)
            _atomic_json(
                snapshot / "inventory.json",
                {
                    "captured_utc": _utc(),
                    "reason": "hot rollback journal after OS crash",
                    "staging": str(staging),
                    "files": [
                        {
                            "name": path.name,
                            "size": path.stat().st_size,
                            "sha256": (
                                sha256_file(path)
                                if path.name.startswith("index.sqlite")
                                else None
                            ),
                        }
                        for path in sorted(staging.iterdir())
                        if path.is_file()
                    ],
                },
            )
    with PackedJointWriter(
        destination,
        manifest_sha256=manifest_hash,
        candidate_version=CANDIDATE_GENERATION_VERSION,
        shard_rows=args.shard_rows,
        staging=staging,
        checkpoint_rows=8,
    ) as writer:
        writer.validate_committed_prefix(rows)
        recovery["committed_prefix"] = writer.committed_records
        reuse_existing = staging is not None
        for index, row in enumerate(rows, 1):
            if index <= writer.committed_records:
                continue
            features, example, lineage = _build_example(
                row, args.source_cache_root.resolve()
            )
            target = canonical_target(
                example,
                valid_labels=lineage.get("valid_labels") or [],
            )
            if reuse_existing:
                try:
                    writer.reuse(row, features, example, target)
                    recovery["reused_records"] += 1
                except ExistingShardMismatch:
                    recovery["append_cursor"] = index - 1
                    writer.begin_append()
                    reuse_existing = False
                    writer.add(row, features, example, target)
            else:
                writer.add(row, features, example, target)
            if index == 1 or index % 25 == 0 or index == len(rows):
                print(
                    f"pack {index}/{len(rows)} "
                    f"reused={recovery['reused_records']}",
                    flush=True,
                )
        metadata = writer.finalize()
    report = {
        "schema_version": PACK_SCHEMA_VERSION,
        "created_utc": _utc(),
        "pack_root": str(destination),
        "pack_id": metadata["pack_id"],
        "record_count": metadata["record_count"],
        "split_counts": metadata["split_counts"],
        "shard_rows": metadata["shard_rows"],
        "shard_files": len(metadata["shards"]),
        "bytes": sum(int(row["size"]) for row in metadata["shards"]),
        "crash_recovery": recovery,
    }
    _atomic_json(audit_dir / "pack_report.json", report)
    return report


def repack_release(args: argparse.Namespace) -> dict[str, Any]:
    """Re-layout an existing verified pack without touching source/test data."""

    source_root = args.pack_root.resolve()
    destination = args.repack_destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    manifest, manifest_hash = _verify_audit_release(args.audit_dir.resolve())
    rows = [*manifest["train"], *manifest["val"]]
    staged = sorted(
        destination.parent.glob(f".{destination.name}.*"),
        key=lambda path: path.stat().st_mtime,
    )
    if len(staged) > 1:
        raise ValueError(f"Multiple repack staging directories: {staged}")
    staging = staged[0] if staged else None
    with PackedJointDataset(
        source_root,
        manifest_sha256=manifest_hash,
        verify_records=True,
        max_open_shards=4,
    ) as source:
        source_ordinals = {
            str(sample): int(ordinal)
            for sample, ordinal in source.connection.execute(
                "SELECT sample,ordinal FROM records"
            )
        }
        with PackedJointWriter(
            destination,
            manifest_sha256=manifest_hash,
            candidate_version=CANDIDATE_GENERATION_VERSION,
            shard_rows=args.shard_rows,
            staging=staging,
            checkpoint_rows=8,
        ) as writer:
            writer.validate_committed_prefix(rows)
            reuse_existing = staging is not None
            reused = 0
            initial_prefix = writer.committed_records
            for index, row in enumerate(rows, 1):
                if index <= initial_prefix:
                    continue
                packed = source[source_ordinals[row["sample"]]]
                example = packed.training_example()
                if reuse_existing:
                    try:
                        writer.reuse(
                            row,
                            packed.features,
                            example,
                            packed.target,
                        )
                        reused += 1
                    except ExistingShardMismatch:
                        writer.begin_append()
                        reuse_existing = False
                        writer.add(
                            row,
                            packed.features,
                            example,
                            packed.target,
                        )
                else:
                    if staging is not None and not writer._handles:
                        writer.begin_append()
                    writer.add(
                        row,
                        packed.features,
                        example,
                        packed.target,
                    )
                if index == 1 or index % 100 == 0 or index == len(rows):
                    print(
                        f"repack {index}/{len(rows)} reused={reused}",
                        flush=True,
                    )
            metadata = writer.finalize()
    with PackedJointDataset(
        destination,
        manifest_sha256=manifest_hash,
        verify_records=True,
    ) as rebuilt:
        validation = rebuilt.validate(deep=False)
    report = {
        "schema_version": PACK_SCHEMA_VERSION,
        "created_utc": _utc(),
        "source_pack": str(source_root),
        "destination": str(destination),
        "record_count": metadata["record_count"],
        "shard_rows": metadata["shard_rows"],
        "pack_id": metadata["pack_id"],
        "bytes": sum(int(row["size"]) for row in metadata["shards"]),
        "initial_committed_prefix": initial_prefix,
        "reused_crash_rows": reused,
        "validation": validation,
        "protected_test_accessed": False,
    }
    _atomic_json(args.audit_dir.resolve() / "repack_report.json", report)
    return report


def _process_metrics(start_wall: float, start_cpu: float, start_io: int | None) -> dict[str, Any]:
    wall = max(time.perf_counter() - start_wall, 1e-9)
    cpu = max(time.process_time() - start_cpu, 0.0)
    read_bytes = None
    rss = None
    try:
        import psutil

        process = psutil.Process()
        rss = int(process.memory_info().rss)
        if start_io is not None:
            read_bytes = max(0, int(process.io_counters().read_bytes) - start_io)
    except ImportError:
        pass
    return {
        "seconds": wall,
        "process_cpu_seconds": cpu,
        "cpu_percent_one_core_equivalent": 100.0 * cpu / wall,
        "cpu_percent_machine": 100.0 * cpu / wall / max(os.cpu_count() or 1, 1),
        "rss_bytes": rss,
        "process_disk_read_bytes": read_bytes,
        "process_disk_mib_per_sec": (
            read_bytes / 1024**2 / wall if read_bytes is not None else None
        ),
    }


def _measure(function: Any, rows: int) -> dict[str, Any]:
    start_io = None
    process = None
    peak_rss = None
    stop_monitor = threading.Event()
    monitor = None
    try:
        import psutil

        process = psutil.Process()
        start_io = int(process.io_counters().read_bytes)
        peak = [int(process.memory_info().rss)]

        def sample_memory() -> None:
            while not stop_monitor.wait(0.02):
                peak[0] = max(peak[0], int(process.memory_info().rss))

        monitor = threading.Thread(target=sample_memory, daemon=True)
        monitor.start()
    except ImportError:
        pass
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    try:
        function()
    finally:
        stop_monitor.set()
        if monitor is not None:
            monitor.join()
            peak_rss = max(peak[0], int(process.memory_info().rss))
    result = _process_metrics(start_wall, start_cpu, start_io)
    result["rows"] = rows
    result["rows_per_sec"] = rows / result["seconds"]
    result["peak_rss_bytes"] = peak_rss
    return result


def _consume_packed(
    dataset: PackedJointDataset,
    ordinals: Sequence[int],
    *,
    workers: int,
    prefetch: int,
) -> None:
    if workers <= 0:
        for ordinal in ordinals:
            dataset[ordinal]
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        iterator = iter(ordinals)
        pending: list[Future[Any]] = []
        for _ in range(max(prefetch, workers)):
            try:
                pending.append(pool.submit(dataset.__getitem__, next(iterator)))
            except StopIteration:
                break
        while pending:
            pending.pop(0).result()
            try:
                pending.append(pool.submit(dataset.__getitem__, next(iterator)))
            except StopIteration:
                pass


def _consume_baseline(
    rows: Sequence[Mapping[str, Any]],
    cache_root: Path,
) -> None:
    for row in rows:
        _build_example(row, cache_root)


def _hardware_profile(path: Path) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
    }
    try:
        import psutil

        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(path.anchor)
        profile.update(
            {
                "physical_cpus": psutil.cpu_count(logical=False),
                "ram_bytes": int(memory.total),
                "ram_available_bytes": int(memory.available),
                "artifact_volume_bytes": int(disk.total),
                "artifact_volume_free_bytes": int(disk.free),
            }
        )
    except ImportError:
        profile["psutil"] = "unavailable"
    if os.name == "nt":
        command = (
            "$cpu=Get-CimInstance Win32_Processor | "
            "Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,"
            "MaxClockSpeed;"
            "$computer=Get-CimInstance Win32_ComputerSystem | "
            "Select-Object Manufacturer,Model,TotalPhysicalMemory;"
            "$disks=Get-PhysicalDisk | "
            "Select-Object FriendlyName,MediaType,BusType,Size,HealthStatus;"
            "[PSCustomObject]@{cpu=$cpu;computer=$computer;disks=$disks} | "
            "ConvertTo-Json -Depth 4 -Compress"
        )
        try:
            saved = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            profile["windows_hardware"] = json.loads(saved.stdout)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return profile


def _tune_shard_rows(
    source: PackedJointDataset,
    rows: Sequence[Mapping[str, Any]],
    sample_to_ordinal: Mapping[str, int],
    pack_parent: Path,
    manifest_hash: str,
) -> dict[str, Any]:
    """Empirically compare compact shard layouts on a bounded 128-row pilot."""

    pilot_rows = list(rows[:128])
    root = Path(
        tempfile.mkdtemp(prefix=".joint-shard-tune.", dir=pack_parent)
    )
    results = []
    try:
        for shard_rows in (64, 128):
            destination = root / f"rows-{shard_rows}"
            build_start = time.perf_counter()
            with PackedJointWriter(
                destination,
                manifest_sha256=manifest_hash,
                candidate_version=CANDIDATE_GENERATION_VERSION,
                shard_rows=shard_rows,
                checkpoint_rows=16,
            ) as writer:
                for row in pilot_rows:
                    packed = source[sample_to_ordinal[row["sample"]]]
                    writer.add(
                        row,
                        packed.features,
                        packed.training_example(),
                        packed.target,
                    )
                writer.finalize()
            build_seconds = time.perf_counter() - build_start
            startup = time.perf_counter()
            pilot = PackedJointDataset(
                destination,
                manifest_sha256=manifest_hash,
                verify_records=True,
            )
            startup_seconds = time.perf_counter() - startup
            try:
                ordinals = list(range(len(pilot)))
                shuffled = list(ordinals)
                np.random.default_rng(365).shuffle(shuffled)
                sequential_runs = [
                    _measure(
                        lambda: _consume_packed(
                            pilot, ordinals, workers=0, prefetch=0
                        ),
                        len(ordinals),
                    )
                    for _ in range(3)
                ]
                random_runs = [
                    _measure(
                        lambda: _consume_packed(
                            pilot, shuffled, workers=0, prefetch=0
                        ),
                        len(shuffled),
                    )
                    for _ in range(3)
                ]
            finally:
                pilot.close()
            sequential_rate = float(
                np.median([row["rows_per_sec"] for row in sequential_runs])
            )
            random_rate = float(
                np.median([row["rows_per_sec"] for row in random_runs])
            )
            results.append(
                {
                    "shard_rows": shard_rows,
                    "pilot_rows": len(pilot_rows),
                    "build_seconds": build_seconds,
                    "startup_seconds": startup_seconds,
                    "sequential_rows_per_sec": sequential_rate,
                    "random_rows_per_sec": random_rate,
                    "sequential_repeats": sequential_runs,
                    "random_repeats": random_runs,
                    "peak_rss_bytes": max(
                        *(
                            row.get("peak_rss_bytes") or 0
                            for row in sequential_runs
                        ),
                        *(
                            row.get("peak_rss_bytes") or 0
                            for row in random_runs
                        ),
                    ),
                }
            )
        best = max(
            results,
            key=lambda row: (
                (
                    float(row["sequential_rows_per_sec"])
                    * float(row["random_rows_per_sec"])
                    * (float(row["pilot_rows"]) / float(row["build_seconds"]))
                )
                ** (1.0 / 3.0)
                / (1.0 + float(row["startup_seconds"]))
            ),
        )
        return {
            "candidates": results,
            "selected_shard_rows": int(best["shard_rows"]),
            "selection_metric": (
                "geometric mean build/sequential/random rows/sec divided by "
                "(1 + startup seconds), with three measured load repeats"
            ),
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _compare_full_layouts(
    roots: Sequence[Path],
    rows: Sequence[Mapping[str, Any]],
    manifest_hash: str,
) -> dict[str, Any]:
    """Use repeated 1,000-row reads to resolve noisy small-pilot results."""

    selected_rows = list(rows[: min(1000, len(rows))])
    layouts = []
    for root in roots:
        startup = time.perf_counter()
        dataset = PackedJointDataset(
            root,
            manifest_sha256=manifest_hash,
            verify_records=True,
            max_open_shards=4,
        )
        startup_seconds = time.perf_counter() - startup
        try:
            by_sample = {
                str(sample): int(ordinal)
                for sample, ordinal in dataset.connection.execute(
                    "SELECT sample,ordinal FROM records WHERE split='train'"
                )
            }
            ordinals = [by_sample[row["sample"]] for row in selected_rows]
            shuffled = list(ordinals)
            np.random.default_rng(20260915).shuffle(shuffled)
            sequential = [
                _measure(
                    lambda: _consume_packed(
                        dataset, ordinals, workers=0, prefetch=0
                    ),
                    len(ordinals),
                )
                for _ in range(3)
            ]
            random_access = [
                _measure(
                    lambda: _consume_packed(
                        dataset, shuffled, workers=0, prefetch=0
                    ),
                    len(shuffled),
                )
                for _ in range(3)
            ]
            sequential_rate = float(
                np.median([row["rows_per_sec"] for row in sequential])
            )
            random_rate = float(
                np.median([row["rows_per_sec"] for row in random_access])
            )
            layouts.append(
                {
                    "path": str(root),
                    "shard_rows": int(dataset.metadata["shard_rows"]),
                    "startup_seconds": startup_seconds,
                    "sequential_repeats": sequential,
                    "random_repeats": random_access,
                    "sequential_rows_per_sec_median": sequential_rate,
                    "random_rows_per_sec_median": random_rate,
                    "score": (sequential_rate * random_rate) ** 0.5,
                }
            )
        finally:
            dataset.close()
    best = max(layouts, key=lambda row: float(row["score"]))
    return {
        "layouts": layouts,
        "selected_path": best["path"],
        "selected_shard_rows": best["shard_rows"],
        "selection_metric": (
            "geometric mean of median sequential and random rows/sec; "
            "three 1,000-row repetitions each"
        ),
    }


def benchmark_release(args: argparse.Namespace) -> dict[str, Any]:
    manifest, manifest_hash = _verify_audit_release(args.audit_dir.resolve())
    train_rows = list(manifest["train"])
    maximum = min(1000, len(train_rows))
    if maximum < 100:
        raise ValueError("Benchmark requires at least 100 train rows")
    benchmark_path = args.audit_dir.resolve() / "loader_benchmark.json"
    prior = (
        json.loads(benchmark_path.read_text(encoding="utf-8"))
        if benchmark_path.is_file()
        else None
    )
    sibling_128 = args.pack_root.resolve().parent / RELEASE
    sibling_64 = args.pack_root.resolve().parent / f"{RELEASE}-shard64"
    layout_roots = [
        path for path in (sibling_128, sibling_64) if path.is_dir()
    ]
    full_layout_tuning = _compare_full_layouts(
        layout_roots, train_rows, manifest_hash
    )
    _atomic_json(
        args.audit_dir.resolve() / "full_layout_comparison.json",
        full_layout_tuning,
    )
    if Path(full_layout_tuning["selected_path"]).resolve() != args.pack_root.resolve():
        raise ValueError(
            "Requested pack is not the empirically selected full layout: "
            f"{full_layout_tuning['selected_path']}"
        )
    results: dict[str, Any] = {
        "schema_version": "align-packed-loader-benchmark-v1",
        "created_utc": _utc(),
        "hardware": _hardware_profile(args.pack_root.resolve()),
        "methodology": {
            "baseline": (
                "validated NPZ map + SQLite target + MusicXML parse + candidate decode"
            ),
            "packed": (
                "mmap map + SQLite sparse target/candidate + per-record checksum"
            ),
            "cache_state": "natural OS cache; baseline measured before packed",
            "protected_test_accessed": False,
            "baseline_reused": bool(prior),
        },
        "sizes": {},
        "worker_tuning": [],
        "full_layout_tuning": full_layout_tuning,
    }
    baseline_start = time.perf_counter()
    first_db = Path(str(train_rows[0]["target_db"]))
    baseline_connection = sqlite3.connect(
        f"file:{first_db.resolve().as_posix()}?mode=ro", uri=True
    )
    baseline_connection.execute(
        "SELECT ordinal FROM targets WHERE ordinal=?",
        (int(train_rows[0]["target_record"]),),
    ).fetchone()
    baseline_connection.close()
    results["baseline_startup_seconds"] = time.perf_counter() - baseline_start
    startup = time.perf_counter()
    dataset = PackedJointDataset(
        args.pack_root.resolve(),
        manifest_sha256=manifest_hash,
        verify_records=True,
    )
    results["packed_startup_seconds"] = time.perf_counter() - startup
    sample_to_ordinal = {
        str(sample): int(ordinal)
        for sample, ordinal in dataset.connection.execute(
            "SELECT sample,ordinal FROM records WHERE split='train'"
        )
    }
    try:
        results["shard_tuning"] = _tune_shard_rows(
            dataset,
            train_rows,
            sample_to_ordinal,
            args.pack_root.resolve().parent,
            manifest_hash,
        )
        actual_shard_rows = int(dataset.metadata["shard_rows"])
        results["shard_tuning"]["small_pilot_selected_shard_rows"] = int(
            results["shard_tuning"]["selected_shard_rows"]
        )
        results["shard_tuning"]["selected_shard_rows"] = int(
            full_layout_tuning["selected_shard_rows"]
        )
        results["shard_tuning"]["selection_note"] = (
            "The repeated full 1,000-row comparison overrides noisy 128-row "
            "pilot ordering."
        )
        for count in (100, maximum):
            selected = train_rows[:count]
            saved_size = (
                (prior.get("sizes") or {}).get(str(count))
                if isinstance(prior, Mapping)
                else None
            )
            baseline = (
                saved_size["baseline_npz_sqlite"]
                if isinstance(saved_size, Mapping)
                and "baseline_npz_sqlite" in saved_size
                else _measure(
                    lambda selected=selected: _consume_baseline(
                        selected, args.source_cache_root.resolve()
                    ),
                    count,
                )
            )
            ordinals = [sample_to_ordinal[row["sample"]] for row in selected]
            packed = _measure(
                lambda ordinals=ordinals: _consume_packed(
                    dataset, ordinals, workers=0, prefetch=0
                ),
                count,
            )
            results["sizes"][str(count)] = {
                "baseline_npz_sqlite": baseline,
                "packed_mmap": packed,
                "speedup": packed["rows_per_sec"] / baseline["rows_per_sec"],
            }
        tuning_ordinals = [
            sample_to_ordinal[row["sample"]]
            for row in train_rows[:maximum]
        ]
        for workers, prefetch in ((0, 0), (1, 4), (2, 8), (4, 16)):
            measured = _measure(
                lambda workers=workers, prefetch=prefetch: _consume_packed(
                    dataset,
                    tuning_ordinals,
                    workers=workers,
                    prefetch=prefetch,
                ),
                maximum,
            )
            measured.update({"workers": workers, "prefetch": prefetch})
            results["worker_tuning"].append(measured)
        best = max(
            results["worker_tuning"],
            key=lambda row: float(row["rows_per_sec"]),
        )
        results["recommended"] = {
            "workers": int(best["workers"]),
            "prefetch": int(best["prefetch"]),
            "shard_rows": actual_shard_rows,
            "max_open_shards": 4,
            "verify_records": True,
            "pin_memory": False,
            "pin_memory_reason": (
                "The protected structured trainer is CPU-only; pinning consumes "
                "scarce locked RAM and provides no transfer benefit."
            ),
            "deterministic_seed": args.seed,
        }
    finally:
        dataset.close()
    _atomic_json(benchmark_path, results)
    return results


def verify_and_mark_ready(args: argparse.Namespace) -> dict[str, Any]:
    audit_dir = args.audit_dir.resolve()
    manifest, manifest_hash = _verify_audit_release(audit_dir)
    validate_split_manifest(manifest)
    benchmark_path = audit_dir / "loader_benchmark.json"
    if not benchmark_path.is_file():
        raise FileNotFoundError("Benchmark must complete before readiness")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    with PackedJointDataset(
        args.pack_root.resolve(),
        manifest_sha256=manifest_hash,
        verify_records=True,
    ) as dataset:
        validation = dataset.validate(deep=True)
        expected = len(manifest["train"]) + len(manifest["val"])
        if len(dataset) != expected:
            raise ValueError(f"Packed rows={len(dataset)}, expected={expected}")
        # Compare exact targets and all frontend arrays against original storage.
        samples = [
            *manifest["train"][:5],
            *manifest["val"][:5],
            *manifest["train"][-5:],
            *manifest["val"][-5:],
        ]
        packed_ordinals = {
            str(sample): int(ordinal)
            for sample, ordinal in dataset.connection.execute(
                "SELECT sample,ordinal FROM records"
            )
        }
        equivalence = []
        for row in samples:
            features, example, lineage = _build_example(
                row, args.source_cache_root.resolve()
            )
            packed = dataset[packed_ordinals[row["sample"]]]
            expected_target = canonical_target(
                example,
                valid_labels=lineage.get("valid_labels") or [],
            )
            expected_target["sparse_lattice"] = {
                "gold_spans": [
                    list(value) if value is not None else None
                    for value in example.gold_spans
                ],
                "gold_keep_unlinked": list(example.gold_keep_unlinked),
            }
            if packed.target != expected_target:
                raise ValueError(f"Packed target differs for {row['sample']}")
            for name in ("note", "onset", "contour", "frame_times"):
                if not np.array_equal(
                    getattr(features, name),
                    getattr(packed.features, name),
                ):
                    raise ValueError(
                        f"Packed numerical mismatch: {row['sample']} {name}"
                    )
            if packed.training_example() != example:
                raise ValueError(f"Packed joint example differs: {row['sample']}")
            equivalence.append(row["sample"])
        pack_metadata = json.loads(
            (args.pack_root.resolve() / "metadata.json").read_text(encoding="utf-8")
        )
    readiness = {
        "schema_version": "align-joint-data-ready-v1",
        "release": RELEASE,
        "created_utc": _utc(),
        "status": "ready",
        "success_target": {
            "combined_f1": ">0.80",
            "status": "training_goal_not_a_data-result",
        },
        "paths": {
            "manifest": str(audit_dir / "split.json"),
            "lockbox_protocol": str(audit_dir / "LOCKBOX_SEALED.json"),
            "audit_report": str(audit_dir / "audit_report.json"),
            "historical_exclusions": str(
                audit_dir / "historical_lockbox_exclusions.json"
            ),
            "packed_root": str(args.pack_root.resolve()),
            "packed_index": str(args.pack_root.resolve() / "index.sqlite"),
            "benchmark": str(benchmark_path),
        },
        "counts": {
            "audited": 10_000,
            "train": len(manifest["train"]),
            "val": len(manifest["val"]),
            "locked_test_metadata_only": len(manifest["test"]),
            "packed": pack_metadata["record_count"],
            **manifest["excluded"],
        },
        "hashes": {
            "manifest_sha256": manifest_hash,
            "lockbox_protocol_sha256": sha256_file(
                audit_dir / "LOCKBOX_SEALED.json"
            ),
            "packed_metadata_sha256": sha256_file(
                args.pack_root.resolve() / "metadata.json"
            ),
            "packed_index_sha256": pack_metadata["index"]["sha256"],
            "pack_id": pack_metadata["pack_id"],
            "benchmark_sha256": sha256_file(benchmark_path),
        },
        "verification": {
            **validation,
            "packed_vs_original_exact_samples": equivalence,
            "split_leakage": "none across all recorded fingerprints",
            "test_features_materialized": False,
            "test_targets_materialized": False,
        },
        "loader": {
            "python": "alignmodel.joint.packed_data.PackedJointDataset",
            "root": str(args.pack_root.resolve()),
            **benchmark["recommended"],
            "shuffle": "deterministic_order(split, epoch, seed)",
            "resume": "PackedCursor.to_dict()/from_dict()",
        },
    }
    marker = args.ready_marker.resolve()
    if marker.exists():
        raise FileExistsError(marker)
    _atomic_json(marker, readiness)
    # Freeze all packed payloads after deep verification.
    for path in args.pack_root.resolve().rglob("*"):
        if path.is_file():
            path.chmod(stat.S_IREAD)
    return readiness


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_default = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo", type=Path, default=repo_default)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=repo_default
        / "align-model"
        / "data-audit"
        / RELEASE,
    )
    parser.add_argument(
        "--pack-root",
        type=Path,
        default=repo_default
        / "align-model"
        / "data-packed"
        / RELEASE,
    )
    parser.add_argument(
        "--source-cache-root",
        type=Path,
        default=repo_default
        / "align-model"
        / "runs"
        / "joint-audit-v2"
        / "basic-pitch-cache",
    )
    parser.add_argument(
        "--ready-marker",
        type=Path,
        default=repo_default
        / "align-model"
        / "runs"
        / RELEASE
        / "DATA_READY.json",
    )
    parser.add_argument(
        "--repack-destination",
        type=Path,
        default=repo_default
        / "align-model"
        / "data-packed"
        / f"{RELEASE}-shard64",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--shard-rows", type=int, default=128)
    parser.add_argument(
        "--phase",
        choices=("all", "audit", "pack", "repack", "benchmark", "verify"),
        default="all",
    )
    args = parser.parse_args(argv)
    args.audit_dir.parent.mkdir(parents=True, exist_ok=True)
    args.pack_root.parent.mkdir(parents=True, exist_ok=True)
    if args.phase in {"all", "audit"}:
        build_audit_release(args)
    if args.phase in {"all", "pack"}:
        pack_release(args)
    if args.phase == "repack":
        repack_release(args)
    if args.phase in {"all", "benchmark"}:
        benchmark_release(args)
    if args.phase in {"all", "verify"}:
        result = verify_and_mark_ready(args)
        print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
